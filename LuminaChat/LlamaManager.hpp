// LlamaManager.hpp - header-only implementation for core llama.cpp functionality
// 
// RESPONSIBILITY: Core management and infrastructure
// - Model loading, management, and lifecycle
// - Context creation, switching, and management  
// - Batch operations and token processing
// - Template application and conversation state
// - Text-to-token conversion (input processing)
// - Message history and summarization coordination
//
// DELEGATION: Response generation is delegated to LlamaResponse class
// 
// Handles core classs pertaining to llama.cpp backend usage.
//
// BATCHING APPROACH:
// Batching in llama.cpp is designed for processing multiple separate inputs/sequences 
// simultaneously, NOT for splitting a single input into chunks. For large single inputs
// during full context rebuilds, we use incremental batch processing to achieve better
// performance than sequential token-by-token processing while maintaining proper
// context state management.
//
// File Specific Directives:
// Only keep a maximum of 90% maximum token usage in the context.
// Use a maximum of 90% context usage, prune older messages to bring us down to 60% usage.
//
// Project Settings:
// C++ Language Standard: ISO C++20 Standard (/std:c++20)
// C Language Standard: ISO C17 (2018) Standard (/std:c17)
// Optimization: Maximum Optimization (Favor Speed) (/O2)
// Favor Size or Speed: Favor fast code (/Ot)
// Runtime Library: Multi-threaded DLL (/MD)
// Enable Run-Time Type Information (RTTI) YES (/GR)
//
// CRITICAL CODING DIRECTIVES:
// 1.  Minimalism & Performance: Deliver lean, efficient solutions; do not create or preserve unused helpers, wrappers, trivial accessors (setters/getters), or scaffolding.
// 2.  Redundancy Elimination: Remove unused, obsolete, and legacy code—including unneeded interfaces, includes, helper or accessor methods.
// 3.  Consistent Style: Adopt a uniform coding style and structure for clarity and maintainability.
// 4.  Documentation: Write concise comments that explain complex logic and key design decisions.
// 5.  Zero Magic & Strong Typing: Replace magic literals with named constants, enums, or constexpr; prefer scoped enums over raw ints.
// 6.  Function Boundaries: Define clear responsibilities; reduce overlap and avoid unnecessary layers of indirection.
// 7.  Core Preservation: Streamline code while safeguarding essential features; favor direct variable or object access/passing over extra abstractions (e.g., setters/getters).
// 8.  Const-Correctness & Immutability: Mark variables, parameters, and methods as const wherever possible.
// 9.  RAII & Resource Safety: Encapsulate resource acquisition/release in constructors/destructors or smart pointers.
// 10. Standard Library Preference: Favor STL algorithms and containers over custom loops and buffers for clarity and safety.
// 11. Cross-Platform Portability: Use fixed-width types and proper initialization to guarantee identical behavior everywhere.
// 12. Thread Safety: Define and document thread-safety contracts; protect shared state with mutexes, atomics, or thread-safe containers.
// 13. Smart Caching: Cache frequently used values to minimize allocations and improve performance.
// 14. Logical Consistency: Verify code flow to ensure coherent, error-free execution paths.
// 15. Continuous Refinement: Regularly refactor and confirm that updates preserve stable functionality.

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <filesystem>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <functional>
#include "llama-cpp.h"
#include "LogHandler.hpp"
#include "TokenCache.hpp"

// Forward declarations
class LlamaSummarizer;
bool model_loading_progress_callback(float progress, void *user_data);

// Constants for configuration and performance (Directive #13: Zero Magic & Strong Typing)
namespace LlamaConstants {
    // Model defaults
    constexpr int32_t DEFAULT_CONTEXT_SIZE = 2048;
    constexpr int32_t DEFAULT_GPU_LAYERS = 0;
    constexpr int32_t DEFAULT_PREDICT_TOKENS = 256;
    constexpr int32_t DEFAULT_TOKEN_CACHE_SIZE = 1024;
      // Batch processing - for multiple separate inputs, not splitting single inputs
    constexpr int32_t MAX_BATCH_SIZE = 8192;
    constexpr int32_t BATCH_DIVISOR = 4;
    constexpr int32_t MAX_SEQ_IDS = 8;
      // Safety margins and limits
    constexpr int32_t TOKEN_SAFETY_MARGIN = 32;
    constexpr int32_t BATCH_SAFETY_MARGIN = 128;
    constexpr int32_t MAX_TOKEN_BUFFER_SIZE = 1024;
    constexpr int32_t MAX_SUMMARY_LENGTH = 512;
    constexpr int32_t INITIAL_TOKEN_BUFFER_SIZE = 32;
    constexpr size_t MAX_MESSAGE_HISTORY_SIZE = 1000;    // Sampler defaults
    constexpr float DEFAULT_TEMPERATURE = 0.8f;
    constexpr float DEFAULT_MIN_P = 0.05f;
    constexpr float DEFAULT_TOP_P = 0.9f;
    constexpr int32_t DEFAULT_TOP_K = 40;
    
    // String processing constants
    constexpr size_t MAX_TEXT_PREVIEW_LENGTH = 50;
    constexpr size_t STRING_RESERVE_MULTIPLIER = 4;
    constexpr int32_t MAX_RETRY_ATTEMPTS = 2;
    
    // Timing and sleep constants
    constexpr int32_t RETRY_BACKOFF_MS = 50;
    constexpr float MS_TO_MICROSECONDS = 1000.0f;
}

// Forward declaration of SummarizerConstants (fully defined in LlamaSummarizer.hpp)
namespace SummarizerConstants {
    extern const size_t MAX_SUMMARY_SLOTS;
    extern const float TARGET_CONTEXT_USAGE;
    extern const float AGGRESSIVE_PRUNING_RATIO;
}

// Model information container
struct ModelInfo {
    llama_model* model;
    const llama_vocab* vocab;
    llama_sampler* sampler;
    int32_t n_ctx;
    int32_t n_gpu_layers;
    int32_t n_predict;
    std::string model_path;
    std::string custom_chat_template;
    bool model_loaded;
    
    ModelInfo() : model(nullptr), vocab(nullptr), sampler(nullptr), 
                 n_ctx(LlamaConstants::DEFAULT_CONTEXT_SIZE), n_gpu_layers(LlamaConstants::DEFAULT_GPU_LAYERS), 
                 n_predict(LlamaConstants::DEFAULT_PREDICT_TOKENS), model_loaded(false) {}
    
    ~ModelInfo() {
        if (sampler) {
            llama_sampler_free(sampler);
            sampler = nullptr;
        }
        if (model) {
            llama_model_free(model);
            model = nullptr;
        }
        vocab = nullptr;
        model_loaded = false;
    }
    
    // Get chat template (custom or model default)
    const char* get_chat_template() const {
        if (!custom_chat_template.empty()) {
            return custom_chat_template.c_str();
        }
        if (model) {
            return llama_model_chat_template(model, nullptr);
        }
        return nullptr;
    }
};

// Multi-context support
struct ContextInfo {
    llama_context* context;
    llama_batch batch;
    bool batch_initialized;
    int32_t n_past;
    int32_t prev_len;
    std::vector<std::pair<std::string, std::string>> message_history;
    std::string system_message;
    
    // Performance tracking
    int64_t total_generation_tokens = 0;
    int64_t last_decode_time_us = 0;
      // Cache state
    mutable bool message_cache_dirty = true;
    mutable std::vector<llama_chat_message> message_cache;    // Token count tracking for efficiency - maintains accurate count of tokens in message history
    // Always >= 0, updated during context rebuilds and incremental updates
    // NOTE: This field was previously sometimes set to -1 to indicate invalidation,
    // but that pattern is no longer used. The count is now always maintained accurately.
    int32_t message_history_token_count = 0;
    
    // Reference to associated model
    ModelInfo* model_info;
      // Special flag for contexts that should reset before each generation
    // Primarily used for summary models that need a clean slate for each task
    bool reset_after_generation = false;
      // Summarizer for handling conversation summarization per context
    std::unique_ptr<LlamaSummarizer> summarizer;
      ContextInfo() : context(nullptr), batch{}, batch_initialized(false), 
                   n_past(0), prev_len(0), message_history_token_count(0), model_info(nullptr) {
    }
    
    // Get context size directly from associated model
    int32_t get_context_size() const noexcept {
        return model_info ? model_info->n_ctx : LlamaConstants::DEFAULT_CONTEXT_SIZE;
    }
};

#include "LlamaResponse.hpp"

// Thread Safety Contract (Directive #12):
// This class is NOT thread-safe. External synchronization is required for concurrent access.
// - All public methods must be called from a single thread or protected by external mutexes
// - The llama.cpp backend itself has thread-safety limitations that require careful handling
// - Model loading/unloading operations are particularly sensitive to race conditions
// - Context switching operations modify shared state and must be serialized
//
// INTEGRATION WITH LlamaResponse:
// - LlamaManager handles setup, context management, and provides callback functions
// - LlamaResponse handles pure generation logic, receiving callbacks for batch operations
// - This separation allows LlamaManager to focus on infrastructure while LlamaResponse focuses on generation
class LlamaManager {
private:
    
    std::unordered_map<std::string, std::unique_ptr<ModelInfo>> models;
    std::unordered_map<std::string, std::unique_ptr<ContextInfo>> contexts;
    std::string active_context_id;    ContextInfo* current_context;
    
    // Template and cache management
    mutable std::string template_buffer;
    mutable TokenCache token_cache;    // Working buffers
    mutable std::string temp_string_buffer;

    // Response generation handler
    mutable LlamaResponse response_generator;    // Unified batch management - for single sequence processing
    void clear_batch() {
        if (!current_context || !current_context->batch_initialized) return;
        current_context->batch.n_tokens = 0;
    }

    // Unified batch token addition - adds tokens from a single input to batch
    // NOTE: This is for single-sequence processing, not multi-sequence batching
    bool add_tokens_to_batch(const std::vector<llama_token>& tokens, int32_t start_pos, 
                            const std::vector<llama_seq_id>& seq_ids, bool output_logits = false) {
        if (!current_context || !current_context->batch_initialized || tokens.empty()) return false;
        
        if (!current_context->model_info) {
            LLAMA_LOG("Error: No model info available for batch operations");
            return false;
        }
        
        // Validate start_pos is reasonable
        if (start_pos < 0 || start_pos >= current_context->model_info->n_ctx) {
            LLAMA_LOG("Error: Invalid start position " + std::to_string(start_pos) + " for context size " + std::to_string(current_context->model_info->n_ctx));
            return false;
        }        
        const int32_t n_batch = llama_n_batch(current_context->context);
        if (n_batch <= 0) {
            LLAMA_LOG("Error: Invalid batch size: " + std::to_string(n_batch));
            return false;
        }
        
        // Validate batch size against our maximum
        if (n_batch > LlamaConstants::MAX_BATCH_SIZE) {
            LLAMA_LOG("Warning: Context batch size (" + std::to_string(n_batch) + 
                      ") exceeds MAX_BATCH_SIZE (" + std::to_string(LlamaConstants::MAX_BATCH_SIZE) + ")");
        }
        
        clear_batch(); // Clear batch
        
        // Validate sequence IDs
        if (seq_ids.empty()) {
            LLAMA_LOG("Error: Empty sequence ID vector");
            return false;
        }
        
        // Use n_batch as the capacity limit
        for (size_t i = 0; i < tokens.size() && current_context->batch.n_tokens < n_batch; ++i) {
            // Check position bounds carefully
            int32_t pos = start_pos + static_cast<int32_t>(i);
            if (pos >= current_context->model_info->n_ctx || pos < 0) {
                LLAMA_LOG("Warning: Token position " + std::to_string(pos) + " exceeds context bounds [0, " + std::to_string(current_context->model_info->n_ctx) + ")");
                break;
            }
            
            // Validate token value
            if (tokens[i] < 0) {
                LLAMA_LOG("Error: Invalid token value " + std::to_string(tokens[i]) + " at position " + std::to_string(i));
                return false;
            }
            
            // Ensure we don't exceed batch array bounds
            if (current_context->batch.n_tokens >= n_batch) {
                LLAMA_LOG("Warning: Batch capacity exceeded, stopping token addition");
                break;
            }
            
            current_context->batch.token[current_context->batch.n_tokens] = tokens[i];
            current_context->batch.pos[current_context->batch.n_tokens] = pos;
            current_context->batch.n_seq_id[current_context->batch.n_tokens] = static_cast<int32_t>(std::min(seq_ids.size(), size_t(LlamaConstants::MAX_SEQ_IDS)));
            
            // Safe sequence ID copying with bounds check
            for (size_t j = 0; j < std::min(seq_ids.size(), size_t(LlamaConstants::MAX_SEQ_IDS)); ++j) {
                current_context->batch.seq_id[current_context->batch.n_tokens][j] = seq_ids[j];
            }
            
            current_context->batch.logits[current_context->batch.n_tokens] = (i == tokens.size() - 1) ? output_logits : false;
            current_context->batch.n_tokens++;
        }
        
        return current_context->batch.n_tokens > 0;
    }

    // Single token addition for generation loop
    bool add_single_token_to_batch(llama_token token, int32_t pos, 
                                  const std::vector<llama_seq_id>& seq_ids, bool output_logits = true) {
        return add_tokens_to_batch({token}, pos, seq_ids, output_logits);
    }

    // Unified tokenization with caching - Updated to use context's model
    std::vector<llama_token> process_text_to_tokens(const std::string& text, bool add_special = true) const {
        if (text.empty()) return {};
        
        // Get vocab from current context's model
        if (!current_context || !current_context->model_info || !current_context->model_info->vocab) {
            LLAMA_LOG("Error: No vocabulary available from current context's model");
            return {};
        }
        
        std::string cache_key = text + (add_special ? ":s" : ":n");
        
        // Check cache first
        std::vector<llama_token> cached_tokens = token_cache.get(cache_key);
        if (!cached_tokens.empty()) {
            return cached_tokens;
        }
        
        // Get required buffer size for tokenization
        const int32_t n_tokens_required = -llama_tokenize(current_context->model_info->vocab, text.c_str(), text.size(), nullptr, 0, add_special, true);
        if (n_tokens_required <= 0) {
            // Don't treat empty tokenization as warning for whitespace-only text
            if (std::all_of(text.begin(), text.end(), [](char c) { return std::isspace(c); })) {
                // Cache empty result for whitespace-only strings
                token_cache.put(cache_key, {});
                return {};
            }            LLAMA_LOG("Warning: Text tokenization failed or resulted in 0 tokens: '" + 
                       text.substr(0, LlamaConstants::MAX_TEXT_PREVIEW_LENGTH) + (text.size() > LlamaConstants::MAX_TEXT_PREVIEW_LENGTH ? "..." : "") + "'");
            return {};
        }
        
        // Add bounds checking for extremely large token counts
        if (n_tokens_required > current_context->model_info->n_ctx) {
            LLAMA_LOG("Error: Text would produce " + std::to_string(n_tokens_required) + 
                       " tokens, exceeding context limit of " + std::to_string(current_context->model_info->n_ctx));
            return {};
        }
        
        // Allocate buffer and tokenize
        std::vector<llama_token> tokens(n_tokens_required);
        const int32_t n_tokens_actual = llama_tokenize(current_context->model_info->vocab, text.c_str(), text.size(), 
                                                       tokens.data(), tokens.size(), add_special, true);
        
        if (n_tokens_actual < 0) {
            LLAMA_LOG("Error: Tokenization failed with error code: " + std::to_string(n_tokens_actual));
            return {};
        }
        
        // Handle case where actual tokens is 0 but expected was > 0
        if (n_tokens_actual == 0 && n_tokens_required > 0) {
            LLAMA_LOG("Warning: Expected " + std::to_string(n_tokens_required) + " tokens but got 0");
            tokens.clear();
        } else if (n_tokens_actual != n_tokens_required) {
            LLAMA_LOG("Warning: Token count mismatch - expected " + std::to_string(n_tokens_required) + 
                       ", got " + std::to_string(n_tokens_actual));
            tokens.resize(std::max(0, n_tokens_actual)); // Ensure non-negative size
        }
        
        // Cache the result
        token_cache.put(cache_key, tokens);
        
        return tokens;
    }

    // REFACTOR: Enhanced context processing - Updated for proper single-input batching
    // NOTE: Batching in llama.cpp is designed for processing multiple separate inputs/sequences 
    // simultaneously, NOT for splitting a single input into chunks. Each batch operation should
    // contain tokens from potentially multiple different inputs, each with their own sequence IDs.
    // For single inputs that exceed batch capacity, we use incremental batch processing for optimal performance.
    bool process_context_tokens(const std::vector<llama_token>& tokens, bool is_incremental = true) {
        if (!current_context || !current_context->batch_initialized) {
            LLAMA_LOG("Error: No active context or batch not initialized");
            return false;
        }
        
        if (tokens.empty()) {
            return true; // Empty tokens are valid
        }
        
        if (!current_context->model_info) {
            LLAMA_LOG("Error: No model info available for current context");
            return false;
        }
        
        // Additional validation before processing
        if (!current_context->context) {
            LLAMA_LOG("Error: Context is null during token processing");
            return false;
        }
        
        const int32_t n_batch = calculate_optimal_batch_size();
        if (n_batch <= 0) {
            LLAMA_LOG("Error: Invalid batch size calculated: " + std::to_string(n_batch));
            return false;
        }
        
        std::vector<llama_seq_id> seq_ids = {0};
          // Check context capacity using model's n_ctx
        int32_t max_threshold = static_cast<int32_t>(current_context->model_info->n_ctx * 0.9f);
        if (!is_incremental) {
            LLAMA_LOG("Starting context rebuild: FULL (non-incremental) - processing " + std::to_string(tokens.size()) + " tokens");
            current_context->n_past = 0; // Reset for full context rebuild
        }
        
        // Validate n_past bounds before processing
        if (current_context->n_past < 0) {
            LLAMA_LOG("Error: Invalid n_past value: " + std::to_string(current_context->n_past));
            current_context->n_past = 0;
        }
        
        if (current_context->n_past >= current_context->model_info->n_ctx) {
            LLAMA_LOG("Error: n_past exceeds context size, resetting");
            current_context->n_past = 0;
            if (current_context->context) {
                llama_memory_clear(llama_get_memory(current_context->context), true); // Ensure kv memory/cache is cleared
            }
        }
        
        // Safer overflow check
        const size_t max_safe_add = static_cast<size_t>(std::numeric_limits<int32_t>::max() - current_context->n_past);
        if (tokens.size() > max_safe_add) {
            LLAMA_LOG("Error: Token addition would cause overflow");
            return false;
        }
          if (current_context->n_past + static_cast<int32_t>(tokens.size()) > max_threshold) {
            if (is_incremental) {
                LLAMA_LOG("Starting context rebuild: PARTIAL (triggering pruning) - context would exceed 90% (" + std::to_string(current_context->n_past + tokens.size()) + 
                           "/" + std::to_string(current_context->model_info->n_ctx) + " tokens)");
                if (current_context->context) {
                    llama_memory_clear(llama_get_memory(current_context->context), true); // Ensure kv memory/cache is cleared
                }
                current_context->n_past = 0;
                prune_conversation_with_summary(SummarizerConstants::TARGET_CONTEXT_USAGE);
                return false;
            } else {
                LLAMA_LOG("Error: Full context rebuild would exceed context limit");
                return false;
            }
        }
        
        // Process large token sets using incremental batch processing
        // When tokens exceed batch capacity, use incremental rebuilds instead of sequential processing
        if (static_cast<int32_t>(tokens.size()) > n_batch) {
            LLAMA_LOG("Warning: Input tokens (" + std::to_string(tokens.size()) + 
                     ") exceed batch size (" + std::to_string(n_batch) + "), using incremental batch processing");
              // For full rebuilds with large token counts, process in optimal batch-sized chunks
            // This leverages batch efficiency while avoiding sequential token-by-token processing
            if (!is_incremental) {
                return process_large_context_incrementally(tokens, n_batch);
            } else {
                // For incremental updates, also use incremental batch processing
                // This provides better performance than sequential processing
                LLAMA_LOG("Note: Large incremental update, using incremental batch processing");
                return process_large_context_incrementally(tokens, n_batch);
            }
        }else {
            // Process all tokens in a single batch (normal case)
            bool output_logits = is_incremental;
            
            if (!add_tokens_to_batch(tokens, current_context->n_past, seq_ids, output_logits)) {
                LLAMA_LOG("Error: Failed to add tokens to batch");
                return false;
            }
            
            // Validate batch state before decode
            if (current_context->batch.n_tokens <= 0) {
                LLAMA_LOG("Warning: Empty batch after token addition");
                return true;
            }
            
            // Additional validation before decode
            if (!current_context->context) {
                LLAMA_LOG("Error: Context became null before decode");
                return false;
            }
            
            if (!current_context->batch_initialized) {
                LLAMA_LOG("Error: Batch became uninitialized before decode");
                return false;
            }
            
            // Validate batch arrays are not null
            if (!current_context->batch.token || !current_context->batch.pos || 
                !current_context->batch.logits || !current_context->batch.seq_id) {
                LLAMA_LOG("Error: Batch arrays are null before decode");
                return false;
            }
            
            // Use n_batch for validation
            const int32_t context_n_batch = llama_n_batch(current_context->context);
            if (current_context->batch.n_tokens > context_n_batch) {
                LLAMA_LOG("Error: Batch token count exceeds batch size limit");
                return false;
            }
            
            // Add comprehensive error checking for decode operation with try-catch
            try {
                int decode_result = llama_decode(current_context->context, current_context->batch);
                if (decode_result != 0) {
                    LLAMA_LOG("Error: Failed to decode batch at position " + std::to_string(current_context->n_past) + 
                               " (error code: " + std::to_string(decode_result) + ")");
                    
                    // Don't return false immediately, try to recover
                    if (decode_result == -1) {
                        LLAMA_LOG("Decode error -1: Attempting context reset...");
                        if (current_context->context) {
                            llama_memory_clear(llama_get_memory(current_context->context), true); // Ensure kv memory/cache is cleared
                            current_context->n_past = 0;
                        }
                        return false; // Let caller handle retry
                    } else {
                        LLAMA_LOG("Severe decode error, aborting token processing");
                        return false;
                    }
                }
            } catch (const std::exception& e) {
                LLAMA_LOG("Exception during decode: " + std::string(e.what()));
                return false;
            } catch (...) {
                LLAMA_LOG("Unknown exception during decode");
                return false;
            }
            
            current_context->n_past += static_cast<int32_t>(tokens.size());
        }
        
        return true;
    }

    // Helper function to process large context using incremental batch rebuilds
    // This allows us to process large token sets efficiently instead of sequentially
    bool process_large_context_incrementally(const std::vector<llama_token>& tokens, int32_t n_batch) {
        if (!current_context || tokens.empty()) return false;
        
        LLAMA_LOG("Processing " + std::to_string(tokens.size()) + " tokens using incremental batch method (batch size: " + std::to_string(n_batch) + ")");
        
        // Clear context for full rebuild
        if (current_context->context) {
            llama_memory_clear(llama_get_memory(current_context->context), true); // Ensure kv memory/cache is cleared
        }
        current_context->n_past = 0;
        
        // Process tokens in batch-sized chunks
        size_t total_tokens = tokens.size();
        size_t processed = 0;
        std::vector<llama_seq_id> seq_ids = {0};
        
        while (processed < total_tokens) {
            size_t chunk_size = std::min(static_cast<size_t>(n_batch), total_tokens - processed);
            std::vector<llama_token> chunk(tokens.begin() + processed, tokens.begin() + processed + chunk_size);
            
            // Only output logits on the final chunk
            bool output_logits = (processed + chunk_size >= total_tokens);
            
            LLAMA_LOG("Processing incremental batch " + std::to_string(processed / n_batch + 1) + 
                     " (" + std::to_string(chunk_size) + " tokens, pos: " + std::to_string(current_context->n_past) + ")");
            
            if (!add_tokens_to_batch(chunk, current_context->n_past, seq_ids, output_logits)) {
                LLAMA_LOG("Error: Failed to add batch chunk starting at token " + std::to_string(processed));
                return false;
            }
            
            // Validate and decode the batch
            if (current_context->batch.n_tokens <= 0) {
                LLAMA_LOG("Warning: Empty batch in incremental processing");
                processed += chunk_size;
                continue;
            }
            
            try {
                int decode_result = llama_decode(current_context->context, current_context->batch);
                if (decode_result != 0) {
                    LLAMA_LOG("Error: Failed to decode incremental batch at position " + 
                             std::to_string(current_context->n_past) + " (error code: " + std::to_string(decode_result) + ")");
                    return false;
                }
            } catch (const std::exception& e) {
                LLAMA_LOG("Exception during incremental decode: " + std::string(e.what()));
                return false;
            } catch (...) {
                LLAMA_LOG("Unknown exception during incremental decode");
                return false;
            }
            
            current_context->n_past += static_cast<int32_t>(chunk_size);
            processed += chunk_size;
        }
        
        LLAMA_LOG("Successfully processed " + std::to_string(total_tokens) + " tokens using " + 
                 std::to_string((total_tokens + n_batch - 1) / n_batch) + " incremental batches");        
        return true;
    }

    // Unified template application - Updated to use context's model
    bool apply_template_optimized(bool add_generation_prompt, std::string& result) const {
        if (!current_context) return false;
        
        if (!current_context->model_info) return false;
        
        const char* tmpl = current_context->model_info->get_chat_template();
        if (!tmpl) return false;
        
        // Update message cache if needed
        if (current_context->message_cache_dirty) {
            current_context->message_cache = convert_to_llama_messages();
            current_context->message_cache_dirty = false;
        }
          // Apply template with auto-resize
        template_buffer.resize(current_context->model_info->n_ctx * LlamaConstants::STRING_RESERVE_MULTIPLIER);
        int32_t result_len = llama_chat_apply_template(
            tmpl, current_context->message_cache.data(), current_context->message_cache.size(),
            add_generation_prompt, template_buffer.data(), template_buffer.size()
        );
        
        if (result_len < 0) {
            template_buffer.resize(-result_len);
            result_len = llama_chat_apply_template(
                tmpl, current_context->message_cache.data(), current_context->message_cache.size(),
                add_generation_prompt, template_buffer.data(), template_buffer.size()
            );
        }
        
        if (result_len > 0) {
            result = std::string(template_buffer.data(), result_len);            return true;
        }
        return false;
    }

    // Validate conversation state and attempt recovery if needed
    bool validate_conversation_state() {
        if (!current_context) {
            LLAMA_LOG("Error: No active context");
            return false;
        }
        
        if (!current_context->model_info || !current_context->model_info->model || !current_context->context) {
            LLAMA_LOG("Error: Invalid model or context state");
            return false;
        }
          // Check if context position is reasonable
        if (current_context->n_past < 0 || current_context->n_past >= current_context->model_info->n_ctx) {
            LLAMA_LOG("Starting context rebuild: FULL (position validation) - context position out of bounds (" + std::to_string(current_context->n_past) + 
                      "/" + std::to_string(current_context->model_info->n_ctx) + "), resetting");
            
            // Reset context state
            if (current_context->context) {
                llama_memory_clear(llama_get_memory(current_context->context), true); // Ensure kv memory/cache is cleared
            }
            current_context->n_past = 0;
            current_context->prev_len = 0;
            current_context->message_cache_dirty = true;
            
            return false; // Indicate that recovery was needed
        }
        
        // Check if message history is reasonable
        if (current_context->message_history.empty()) {
            LLAMA_LOG("Warning: Empty message history");
            return true; // This is actually okay
        }        // Check for extremely long message history that might cause issues
        if (current_context->message_history.size() > LlamaConstants::MAX_MESSAGE_HISTORY_SIZE) {
            LLAMA_LOG("Starting context rebuild: PARTIAL (history validation) - very large message history (" + 
                      std::to_string(current_context->message_history.size()) + " messages), triggering aggressive pruning");
              // Only trigger aggressive pruning if we're not already in a summary context
            if (active_context_id != "summary_context") {
                // Clear context state before aggressive pruning
                if (current_context->context) {
                    llama_memory_clear(llama_get_memory(current_context->context), true); // Ensure kv memory/cache is cleared
                }
                current_context->n_past = 0;
                current_context->prev_len = 0;
                current_context->message_cache_dirty = true;
                
                // Trigger aggressive pruning
                prune_conversation_with_summary(SummarizerConstants::AGGRESSIVE_PRUNING_RATIO); // Keep only 30%
                return false; // Indicate that recovery was needed
            } else {
                LLAMA_LOG("Skipping pruning for summary context");
            }
        }
        
        return true; // State is valid
    }
    
public:
    LlamaManager() : current_context(nullptr), token_cache(LlamaConstants::DEFAULT_TOKEN_CACHE_SIZE) {
    }

    ~LlamaManager() noexcept {
        cleanup();
    }
    
    // Initialize llama.cpp backend
    bool initialize() {
        ggml_backend_load_all();
        return true;    }

    // Load .gguf model file and create ModelInfo with specific parameters
    bool load_model(const std::string& model_path, const std::string& model_id = "", 
                   int32_t context_size = LlamaConstants::DEFAULT_CONTEXT_SIZE, int32_t gpu_layers = LlamaConstants::DEFAULT_GPU_LAYERS, int32_t predict_tokens = LlamaConstants::DEFAULT_PREDICT_TOKENS,
                   void* progress_callback_user_data = nullptr, const std::string& chat_template = "") {
        if (!std::filesystem::exists(model_path)) {
            LLAMA_LOG("Error: Model file does not exist: " + model_path);
            return false;
        }

        std::string actual_model_id = model_id.empty() ? std::filesystem::path(model_path).stem().string() : model_id;
        
        if (models.find(actual_model_id) != models.end()) {
            LLAMA_LOG("Error: Model '" + actual_model_id + "' already loaded");
            return false;
        }

        auto model_info = std::make_unique<ModelInfo>();
        
        // Set up model parameters with provided values
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = gpu_layers;
        
        // Set progress callback if user data is provided
        if (progress_callback_user_data) {
            model_params.progress_callback = model_loading_progress_callback;
            model_params.progress_callback_user_data = progress_callback_user_data;
        }

        // Load the model
        model_info->model = llama_model_load_from_file(model_path.c_str(), model_params);
        if (!model_info->model) {
            LLAMA_LOG("Error: Failed to load model from " + model_path);
            return false;
        }

        model_info->vocab = llama_model_get_vocab(model_info->model);
        model_info->model_path = model_path;
        model_info->n_ctx = context_size;
        model_info->n_gpu_layers = gpu_layers;
        model_info->n_predict = predict_tokens; // Store predict tokens in model info
        model_info->model_loaded = true;
        
        // Store custom chat template if provided
        if (!chat_template.empty()) {
            model_info->custom_chat_template = chat_template;
            LLAMA_LOG("Custom chat template stored for model '" + actual_model_id + "'");
        }
        
        // Initialize default sampler for the model
        // TODO: I don't think we need to initialize a sampler here since the response class will handle it
        /*auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        model_info->sampler = llama_sampler_chain_init(sparams);
        
        if (!model_info->sampler) {
            LLAMA_LOG("Error: Failed to create sampler for model '" + actual_model_id + "'");
            return false;
        }
        
        llama_sampler_chain_add(model_info->sampler, llama_sampler_init_greedy());
          models[actual_model_id] = std::move(model_info);*/
        
        // Clear caches when new model is loaded
        clear_caches();
        
        LLAMA_LOG("Model loaded successfully: " + model_path + " as '" + actual_model_id + 
                  "' (ctx:" + std::to_string(context_size) + ", gpu:" + std::to_string(gpu_layers) + ")");
        return true;
    }
    
    // Context creation with consistent system prompt usage
    bool create_context(const std::string& context_id, const std::string& model_id, const std::string& system_prompt = "", bool reset_after_generation = false) {
        auto model_it = models.find(model_id);
        if (model_it == models.end()) {
            LLAMA_LOG("Error: Model '" + model_id + "' not found");
            return false;
        }
        
        ModelInfo* model_info = model_it->second.get();
        if (!model_info->model_loaded || !model_info->model) {
            LLAMA_LOG("Error: Model '" + model_id + "' not properly loaded");
            return false;
        }
        
        if (contexts.find(context_id) != contexts.end()) {
            LLAMA_LOG("Error: Context '" + context_id + "' already exists");
            return false;
        }
        
        auto context_info = std::make_unique<ContextInfo>();
        
        // Associate with model
        context_info->model_info = model_info;
        
        // Set up context parameters using model's settings
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = model_info->n_ctx;
        ctx_params.n_batch = std::min(LlamaConstants::MAX_BATCH_SIZE, model_info->n_ctx / LlamaConstants::BATCH_DIVISOR);
        ctx_params.n_threads = std::thread::hardware_concurrency();
        ctx_params.no_perf = false;
        
        // Create context
        context_info->context = llama_init_from_model(model_info->model, ctx_params);
        if (!context_info->context) {
            LLAMA_LOG("Error: Failed to create context '" + context_id + "'");
            return false;
        }
        
        // Initialize batch
        int32_t batch_size = std::min(LlamaConstants::MAX_BATCH_SIZE, model_info->n_ctx / LlamaConstants::BATCH_DIVISOR);
        context_info->batch = llama_batch_init(batch_size, 0, 1);
        if (context_info->batch.token == nullptr) {
            LLAMA_LOG("Error: Failed to initialize batch for context '" + context_id + "'");
            llama_free(context_info->context);
            return false;
        }
        context_info->batch_initialized = true;
        
        // Always use provided system prompt, or copy from main context if empty
        std::string prompt_to_use = system_prompt;
        if (prompt_to_use.empty() && current_context && !current_context->system_message.empty()) {
            prompt_to_use = current_context->system_message;
        }        // Set system message if we have one
        if (!prompt_to_use.empty()) {
            context_info->system_message = prompt_to_use;
            context_info->message_history.emplace_back("system", prompt_to_use);
            context_info->message_cache_dirty = true;
            context_info->message_history_token_count = 0; // Initialize to 0 for new context
        }
        
        // Ensure token count is properly initialized (safety check)
        if (context_info->message_history_token_count < 0) {
            context_info->message_history_token_count = 0;
        }
          // Set the reset after generation flag
        context_info->reset_after_generation = reset_after_generation;
        
        // Initialize summarizer for this context
        context_info->summarizer = std::make_unique<LlamaSummarizer>(this);
        
        contexts[context_id] = std::move(context_info);
        LLAMA_LOG("Created context '" + context_id + "' with model '" + model_id + "' successfully");
        
        // If this is the first context, make it active
        if (active_context_id.empty()) {
            switch_to_context(context_id);
        }
        
        return true;
    }

    bool switch_to_context(const std::string& context_id) {
        auto it = contexts.find(context_id);
        if (it == contexts.end()) {
            LLAMA_LOG("Error: Context '" + context_id + "' not found");
            return false;
        }
          // Validate the context before switching
        if (!it->second || !it->second->context || !it->second->model_info || !it->second->model_info->model) {
            LLAMA_LOG("Error: Context '" + context_id + "' has invalid state");
            return false;
        }
        
        // Validate the sampler for this context's model
        // TODO: I really don't think this is necessary here since the response class will validate the sampler and create one if needed
        /*if (!it->second->model_info->sampler) {
            LLAMA_LOG("Warning: Context '" + context_id + "' has no sampler, attempting to create one");
            // Try to create a sampler for this model
            it->second->model_info->sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
            if (it->second->model_info->sampler) {
                llama_sampler_chain_add(it->second->model_info->sampler, 
                                       llama_sampler_init_temp(LlamaConstants::DEFAULT_TEMPERATURE));
                llama_sampler_chain_add(it->second->model_info->sampler, 
                                       llama_sampler_init_min_p(LlamaConstants::DEFAULT_MIN_P, 1));
                llama_sampler_chain_add(it->second->model_info->sampler, 
                                       llama_sampler_init_top_k(LlamaConstants::DEFAULT_TOP_K));
                llama_sampler_chain_add(it->second->model_info->sampler, 
                                       llama_sampler_init_top_p(LlamaConstants::DEFAULT_TOP_P, 1));
                llama_sampler_chain_add(it->second->model_info->sampler, 
                                       llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
                LLAMA_LOG("Created new sampler for context '" + context_id + "'");
            } else {
                LLAMA_LOG("Error: Failed to create sampler for context '" + context_id + "'");
                return false;
            }
        }*/
        
        active_context_id = context_id;
        current_context = it->second.get();
        LLAMA_LOG("Switched to context '" + context_id + "' (model: " + current_context->model_info->model_path + ")");
        return true;
    }
    
    bool remove_context(const std::string& context_id) {
        auto it = contexts.find(context_id);
        if (it == contexts.end()) {
            LLAMA_LOG("Error: Context '" + context_id + "' not found");
            return false;
        }
        
        // Clean up the context
        if (it->second->batch_initialized) {
            llama_batch_free(it->second->batch);
        }
        if (it->second->context) {
            llama_free(it->second->context);
        }
        
        // If this was the active context, clear it
        if (active_context_id == context_id) {
            active_context_id.clear();
            current_context = nullptr;
            
            // Switch to another context if available
            if (!contexts.empty()) {
                auto first_context = contexts.begin();
                switch_to_context(first_context->first);
            }
        }
        
        contexts.erase(it);
        LLAMA_LOG("Removed context '" + context_id + "'");
        return true;
    }

    std::string get_active_context() const noexcept {
        return active_context_id;
    }

    // Check if a context exists by ID
    // Critical Needed for context validation in various operations
    bool has_context(const std::string& context_id) const noexcept {
        return contexts.find(context_id) != contexts.end();
    }
    
    // Retrieve a ContextInfo object by its ID
    // Critical Needed for context access in various operations
    ContextInfo* get_context_info(const std::string& context_id) const {
        auto it = contexts.find(context_id);
        if (it == contexts.end()) {
            return nullptr;
        }
        return it->second.get();
    }

    // Retrieve a ModelInfo object by its ID
    // Critical Needed for model access in various operations
    ModelInfo* get_model_info(const std::string& model_id) const {
        auto it = models.find(model_id);
        if (it == models.end()) {
            return nullptr;
        }
        return it->second.get();
    }

public:
    // Clear conversation history
    void clear_conversation();// Prune message history with summarization and context update
    bool prune_conversation_with_summary(float keep_ratio = 0.6f);
    
    // Convert message history to llama_chat_message format
    std::vector<llama_chat_message> convert_to_llama_messages() const {
        if (!current_context) return {};
        
        std::vector<llama_chat_message> messages;
        for (const auto& msg : current_context->message_history) {
            messages.push_back({ msg.first.c_str(), msg.second.c_str() });
        }
        return messages;    }
      // Forward declaration for summary slot info - implementation after LlamaSummarizer include
    struct SummarySlotInfo {
        size_t total_slots;
        size_t used_slots;
        std::vector<std::string> summaries;
    };
    
    SummarySlotInfo get_summary_slot_info() const;
  
    // Enhanced context update with better tokenization handling
    bool update_context_with_pruning() {
        if (!current_context || !current_context->context || !current_context->model_info || !current_context->model_info->model || !current_context->model_info->vocab) {
            LLAMA_LOG("Error: Model components not initialized");
            return false;
        }

        // Skip pruning for summary contexts - they manage their own state
        bool is_summary_context = (active_context_id == "summary_context");
        
        // Calculate projected context usage before making pruning decisions
        // This prevents the issue where we're at exactly the threshold when generation starts
        std::string formatted_content;
        if (!apply_template_optimized(false, formatted_content)) {
            LLAMA_LOG("Error: Failed to apply chat template for context calculation");
            return false;
        }
        
        // Calculate how many tokens the current conversation would use
        std::vector<llama_token> projected_tokens = process_text_to_tokens(formatted_content, true);
        int32_t projected_usage = static_cast<int32_t>(projected_tokens.size());
        int32_t max_threshold = static_cast<int32_t>(current_context->model_info->n_ctx * 0.9f);
        
        bool context_pruned = false;
        
        // Check if we need pruning based on projected usage, not current n_past
        if (!is_summary_context && projected_usage > max_threshold) {
            LLAMA_LOG("Starting context rebuild: PARTIAL (with pruning) - context usage " + std::to_string(static_cast<float>(projected_usage) / current_context->model_info->n_ctx * 100.0f) + 
                       "% (" + std::to_string(projected_usage) + "/" + std::to_string(current_context->model_info->n_ctx) + "), pruning to 60%");
            
            llama_memory_clear(llama_get_memory(current_context->context), true); // Ensure kv memory/cache is cleared
            current_context->n_past = 0;
            current_context->prev_len = 0;
            prune_conversation_with_summary(SummarizerConstants::TARGET_CONTEXT_USAGE);
            context_pruned = true;
            
            // Mark message cache as dirty since message history was modified
            current_context->message_cache_dirty = true;
            
            // Recalculate formatted content after pruning
            if (!apply_template_optimized(false, formatted_content)) {
                LLAMA_LOG("Error: Failed to apply chat template after pruning");
                return false;
            }
        }
        
        // Apply template and process
        int32_t new_len = static_cast<int32_t>(formatted_content.length());
        
        // CRITICAL: After pruning, the message structure has fundamentally changed.
        // The context state (n_past, prev_len) no longer matches the new message history.
        // We MUST do a full rebuild, never an incremental update, to avoid garbage output.
        if (context_pruned || current_context->prev_len > new_len || 
            (current_context->prev_len == 0 && !current_context->message_history.empty() && !is_summary_context)) {
            
            LLAMA_LOG("Starting context rebuild: FULL - rebuilding complete context from " + std::to_string(current_context->message_history.size()) + " messages");
              // Use incremental batch processing for large message histories
            std::vector<llama_token> rebuild_tokens = process_text_to_tokens(formatted_content, true);
            if (!rebuild_tokens.empty() && process_context_tokens(rebuild_tokens, false)) {
                current_context->prev_len = new_len;
                
                // After rebuilding context, ensure we have valid logits for generation
                // This is critical when pruning occurs right before generation
                if (current_context->n_past > 0) {
                    float* logits = llama_get_logits(current_context->context);
                    if (!logits) {
                        LLAMA_LOG("Warning: No logits available after context rebuild, will need manual decode");
                        // Context was rebuilt but we need to ensure logits are available
                        // This can happen if the rebuild ended with no logit generation
                    }
                }
                
                LLAMA_LOG("Context rebuilt successfully with " + 
                           std::to_string(rebuild_tokens.size()) + " tokens (pruned=" + 
                           std::string(context_pruned ? "true" : "false") + ")");
                return true;
            }
            LLAMA_LOG("Error: Failed to rebuild context");
            return false;
        }
        
        // Process new content incrementally (only if no pruning occurred)
        if (new_len > current_context->prev_len) {
            LLAMA_LOG("Starting context rebuild: INCREMENTAL - adding " + std::to_string(new_len - current_context->prev_len) + " new characters");
            
            std::string new_content = formatted_content.substr(current_context->prev_len);
            if (!new_content.empty()) {
                std::vector<llama_token> new_tokens = process_text_to_tokens(new_content, false);
                if (!new_tokens.empty() && !process_context_tokens(new_tokens, true)) {
                    LLAMA_LOG("Error: Failed to process incremental tokens");
                    return false;
                }                // For incremental updates, we can optimize by adding to the existing token count
                // instead of invalidating the cache, since we know exactly how many tokens were added
                current_context->message_history_token_count += static_cast<int32_t>(new_tokens.size());
                LLAMA_LOG("Updated token count incrementally: +" + std::to_string(new_tokens.size()) + 
                          " = " + std::to_string(current_context->message_history_token_count));
            }
        }

        current_context->prev_len = new_len;
        return true;
    }
    
    // Enhanced generation with response delegation to LlamaResponse
    std::string generate_response(const std::string& input, const std::string& username = "Schwi") {
        if (!current_context || !current_context->context || !current_context->model_info || !current_context->model_info->model_loaded || !current_context->model_info->model || !current_context->model_info->vocab || !current_context->batch_initialized) {
            return "Error: Model components not properly initialized or no active context";
        }

        if (input.empty()) return "Error: Empty input";

        // Validate conversation state before proceeding
        if (!validate_conversation_state()) {
            LLAMA_LOG("Conversation state required recovery, retrying...");
        }
        
        // Setup conversation - ensure system message is in history if context is empty
        if (current_context->message_history.empty() && !current_context->system_message.empty()) {
            current_context->message_history.emplace_back("system", current_context->system_message);
            current_context->message_cache_dirty = true;
            // Token count will be updated after context processing
        }

        current_context->message_history.emplace_back(username, input);
        current_context->message_cache_dirty = true;
        // Token count will be updated after context processing

        // Update context using unified function - with retry logic
        int32_t retry_count = 0;
        const int32_t max_retries = LlamaConstants::MAX_RETRY_ATTEMPTS;
        
        while (retry_count < max_retries) {
            if (update_context_with_pruning()) {
                break; // Success
            }
            retry_count++;
            if (retry_count < max_retries) {
                LLAMA_LOG("Starting context rebuild: FULL (retry " + std::to_string(retry_count) + "/" + std::to_string(max_retries) + ") - retrying context update with aggressive cleanup");
                
                // On retry, try a more aggressive cleanup
                if (current_context && current_context->context) {
                    llama_memory_clear(llama_get_memory(current_context->context), true); // Ensure kv memory/cache is cleared
                    current_context->n_past = 0;
                    current_context->prev_len = 0;
                    current_context->message_cache_dirty = true;
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(LlamaConstants::MAX_RETRY_ATTEMPTS * LlamaConstants::RETRY_BACKOFF_MS)); // Simple backoff
            } else {
                // If we still can't update, try to continue with a minimal context
                LLAMA_LOG("Starting context rebuild: FULL (minimal recovery) - failed to update context, attempting minimal recovery");
                
                if (current_context && current_context->context) {
                    llama_memory_clear(llama_get_memory(current_context->context), true); // Ensure kv memory/cache is cleared
                    current_context->n_past = 0;
                    current_context->prev_len = 0;
                    current_context->message_cache_dirty = true;
                    
                    // Keep only the last few messages for minimal context
                    if (current_context->message_history.size() > 3) {
                        auto minimal_history = std::vector<std::pair<std::string, std::string>>();
                        
                        // Keep system message if present
                        if (!current_context->message_history.empty() && 
                            current_context->message_history[0].first == "system") {
                            minimal_history.emplace_back(current_context->message_history[0]);
                        }
                        
                        // Keep last 2 messages
                        size_t start_idx = std::max(size_t(1), current_context->message_history.size() - 2);
                        for (size_t i = start_idx; i < current_context->message_history.size(); ++i) {
                            if (current_context->message_history[i].first != "system") {
                                minimal_history.emplace_back(current_context->message_history[i]);
                            }
                        }
                        
                        current_context->message_history = std::move(minimal_history);
                        current_context->message_cache_dirty = true;
                        // Token count will be updated during context processing
                        
                        // Try one more time with minimal context
                        if (update_context_with_pruning()) {
                            LLAMA_LOG("Recovered with minimal context");
                            break;
                        }
                    }
                }
                
                return "Error: Context recovery failed. The conversation history may have become too complex. Try starting a new conversation.";
            }
        }

        // Prepare for generation using unified template function
        std::string generation_content;
        if (!apply_template_optimized(true, generation_content)) {
            return "Error: Failed to apply generation template";
        }

        // Handle generation prompt correctly for reset contexts
        std::string generation_prompt;
        if (current_context->prev_len < static_cast<int32_t>(generation_content.length())) {
            generation_prompt = generation_content.substr(current_context->prev_len);
        }
        
        if (!generation_prompt.empty()) {
            // For generation prompts, typically don't add special tokens
            std::vector<llama_token> prompt_tokens = process_text_to_tokens(generation_prompt, false);
            if (!prompt_tokens.empty()) {
                if (!process_context_tokens(prompt_tokens, true)) {
                    return "Error: Failed to process generation prompt";
                }
            } else {
                LLAMA_LOG("Warning: Generation prompt produced no tokens");
            }
        } else {
            LLAMA_LOG("No new generation prompt content to process (prev_len=" + 
                      std::to_string(current_context->prev_len) + ", total_len=" + 
                      std::to_string(generation_content.length()) + ")");
        }

        // Delegate response generation to LlamaResponse class
        auto token_adder = [this](llama_token token, int32_t pos, const std::vector<llama_seq_id>& seq_ids, bool output_logits) -> bool {
            return add_single_token_to_batch(token, pos, seq_ids, output_logits);
        };
        
        auto context_updater = [this]() -> void {
            std::string updated_content;
            if (apply_template_optimized(false, updated_content)) {
                current_context->prev_len = static_cast<int32_t>(updated_content.length());
            }
        };

        return response_generator.generate_response(input, username, current_context, token_adder, context_updater);
    }

    // Get performance statistics
    struct PerformanceStats {
        int64_t total_generation_tokens;
        int64_t last_decode_time_us;
        float average_tokens_per_second;
    };
    
    PerformanceStats get_performance_stats() const {
        if (!current_context) {
            return {0, 0, 0.0f};
        }
        
        float avg_tps = 0.0f;
        if (current_context->last_decode_time_us > 0 && current_context->total_generation_tokens > 0) {
            avg_tps = static_cast<float>(current_context->total_generation_tokens) / (static_cast<float>(current_context->last_decode_time_us) / 1000000.0f);
        }
        
        return {
            current_context->total_generation_tokens,
            current_context->last_decode_time_us,
            avg_tps
        };
    }

    // Timing API methods for compatibility with UI
    struct Timings {
        int32_t n_eval = 0;
        float t_eval_ms = 0.0f;
    };
    
    void reset_timings() {
        if (!current_context) return;
        
        current_context->total_generation_tokens = 0;
        current_context->last_decode_time_us = 0;
    }
    
    Timings get_timings() const {
        if (!current_context) {
            return {0, 0.0f};
        }
        
        Timings timings;
        timings.n_eval = static_cast<int32_t>(current_context->total_generation_tokens);
        timings.t_eval_ms = static_cast<float>(current_context->last_decode_time_us) / LlamaConstants::MS_TO_MICROSECONDS;
        return timings;
    }

    // Simplified batch processing using unified functions only
    bool process_full_context(const std::string& full_content) {
        if (!current_context || full_content.empty()) return true;

        if (!current_context->model_info) {
            LLAMA_LOG("Error: No model info available for full context processing");
            return false;
        }

        // Use unified tokenization
        std::vector<llama_token> tokens = process_text_to_tokens(full_content, true);
        if (tokens.empty()) return false;

        // Check context capacity with safety margin
        const int32_t safety_margin = LlamaConstants::BATCH_SAFETY_MARGIN;
        if (static_cast<int32_t>(tokens.size()) > current_context->model_info->n_ctx - safety_margin) {
            LLAMA_LOG("Warning: Context too large (" + std::to_string(tokens.size()) + 
                       " tokens > " + std::to_string(current_context->model_info->n_ctx - safety_margin) + " limit)");
            return false;
        }

        // Use unified context processing
        current_context->n_past = 0;
        if (!process_context_tokens(tokens, false)) {
            return false;
        }
        
        current_context->prev_len = static_cast<int32_t>(full_content.length());
        
        LLAMA_LOG("Successfully processed " + std::to_string(tokens.size()) + 
                   " tokens in full context rebuild");
        return true;
    }

    // Optimized batch initialization with calculated size
    bool initialize_batch() {
        if (!current_context || current_context->batch_initialized) {
            return true;
        }

        int32_t batch_size = calculate_optimal_batch_size();
        if (batch_size <= 0) {
            LLAMA_LOG("Error: Invalid batch size for initialization: " + std::to_string(batch_size));
            return false;
        }
        
        // Use safer batch initialization
        current_context->batch = llama_batch_init(batch_size, 0, 1);
        if (current_context->batch.token == nullptr || 
            current_context->batch.pos == nullptr || 
            current_context->batch.n_seq_id == nullptr ||
            current_context->batch.seq_id == nullptr ||
            current_context->batch.logits == nullptr) {
            LLAMA_LOG("Error: Failed to initialize batch arrays with size " + std::to_string(batch_size));
            return false;
        }

        current_context->batch_initialized = true;
        LLAMA_LOG("Initialized batch with optimal size: " + std::to_string(batch_size));
        return true;
    }
    
    // Enhanced cleanup with memory optimization
    void cleanup() {
        LLAMA_LOG("Cleanup called - cleaning up " + std::to_string(contexts.size()) + " contexts");
        
        clear_caches();
        
        // Clean up all contexts using STL algorithms (Directive #14: Standard Library Preference)
        std::for_each(contexts.begin(), contexts.end(), [](auto& pair) {
            auto& context_info = pair.second;
            if (context_info->batch_initialized) {
                llama_batch_free(context_info->batch);
                context_info->batch_initialized = false;
            }
            if (context_info->context) {
                llama_free(context_info->context);
                context_info->context = nullptr;
            }
        });
        
        contexts.clear();
        active_context_id.clear();
        current_context = nullptr;
          // Clean up all models - ModelInfo destructor handles model cleanup
        models.clear();
        
        // Efficient memory cleanup
        std::string().swap(template_buffer);
        
        LLAMA_LOG("Cleanup completed");
    }
    
    // Helper method to add messages to history without immediate context update
    void add_message_to_history(const std::string& role, const std::string& content) {
        if (!current_context) return;

        current_context->message_history.emplace_back(role, content);
        current_context->message_cache_dirty = true;
        
        // Token count will be updated during the next context update for accuracy
    }
    
    // Get context size for capacity calculations - uses current context's model
    int32_t get_context_size() const noexcept {
        return current_context ? current_context->get_context_size() : LlamaConstants::DEFAULT_CONTEXT_SIZE;
    }
    
    // Enhanced batch update with proper token tracking
    bool update_context_from_history() {
        if (!current_context) return false;

        LLAMA_LOG("Starting context rebuild: FULL (from message history) - rebuilding from " + std::to_string(current_context->message_history.size()) + " messages");

        // Store original state for rollback
        int32_t original_n_past = current_context->n_past;
        int32_t original_prev_len = current_context->prev_len;

        // Clear current context state for rebuild
        if (current_context->context) {
            llama_memory_clear(llama_get_memory(current_context->context), true); // Ensure kv memory/cache is cleared
        }
        current_context->n_past = 0;
        current_context->prev_len = 0;

        // Rebuild context from message history with enhanced error handling
        bool success = update_context_with_pruning();
        
        if (!success) {
            // Rollback on failure
            current_context->n_past = original_n_past;
            current_context->prev_len = original_prev_len;
            LLAMA_LOG("Failed to update context from history, rolled back to previous state");
            return false;
        }
        
        // Validate the rebuild was successful
        if (current_context->n_past <= 0 && !current_context->message_history.empty()) {
            LLAMA_LOG("Warning: Context rebuild resulted in zero tokens despite having message history");
        }
          LLAMA_LOG("Successfully rebuilt context from " + std::to_string(current_context->message_history.size()) + 
                   " messages, using " + std::to_string(current_context->n_past) + " tokens");
          // Update the cached message history token count since we just rebuilt the context
        // This gives us an accurate count based on the actual tokenization that just occurred
        current_context->message_history_token_count = current_context->n_past;
        LLAMA_LOG("Updated cached message history token count: " + std::to_string(current_context->message_history_token_count));
        
        return true;
    }

public:
    // Public tokenization method for external use
    std::vector<llama_token> tokenize_text(const std::string& text, bool add_special = false) const {
        return process_text_to_tokens(text, add_special);
    }

private:
    int32_t calculate_optimal_batch_size() const {
        if (!current_context || !current_context->context) {
            return LlamaConstants::MAX_BATCH_SIZE;
        }
        
        if (!current_context->model_info) {
            return LlamaConstants::MAX_BATCH_SIZE;
        }
        
        int32_t n_batch = llama_n_batch(current_context->context);
        int32_t available_ctx = current_context->model_info->n_ctx - current_context->n_past;
        
        return std::max(1, std::min({n_batch, available_ctx, LlamaConstants::MAX_BATCH_SIZE}));
    }
      void clear_caches() const {
        token_cache.clear();    }
};

// Include LlamaSummarizer implementation after class declaration to avoid circular dependency
#include "LlamaSummarizer.hpp"

// Implementation of methods that depend on LlamaSummarizer
inline LlamaManager::SummarySlotInfo LlamaManager::get_summary_slot_info() const {
    if (!current_context || !current_context->summarizer) {
        return {SummarizerConstants::MAX_SUMMARY_SLOTS, 0, {}};
    }
    auto summarizer_info = current_context->summarizer->get_summary_slot_info();
    return {summarizer_info.total_slots, summarizer_info.used_slots, summarizer_info.summaries};
}

inline bool LlamaManager::prune_conversation_with_summary(float keep_ratio) {
    if (!current_context || current_context->message_history.empty()) {
        LLAMA_LOG("Warning: No active context or empty message history for pruning");
        return false;
    }
    
    if (!current_context->summarizer) {
        LLAMA_LOG("Warning: No summarizer available for pruning");
        return false;
    }
    
    // Don't prune summary contexts
    if (active_context_id == "summary_context") {
        LLAMA_LOG("Skipping pruning for summary context");
        return true;
    }
    
    size_t original_message_count = current_context->message_history.size();
    LLAMA_LOG("Starting pruning with " + std::to_string(original_message_count) + " messages, keep_ratio=" + std::to_string(keep_ratio));    // Perform the pruning with summarization
    current_context->summarizer->prune_message_history(current_context->message_history, keep_ratio);    
    // Always mark message cache as dirty after pruning since message structure changed
    current_context->message_cache_dirty = true;
    // Token count will be updated during context processing
    
    // Update context after pruning
    bool success = update_context_from_history();
    
    if (success) {
        LLAMA_LOG("Manual pruning completed successfully. Messages: " + 
                  std::to_string(original_message_count) + " -> " + 
                  std::to_string(current_context->message_history.size()));
    } else {
        LLAMA_LOG("Warning: Context update failed after manual pruning");
    }
      return success;
}

inline void LlamaManager::clear_conversation() {
    if (!current_context) return;
    
    LLAMA_LOG("Starting context rebuild: FULL (conversation cleared) - clearing all " + std::to_string(current_context->message_history.size()) + " messages");
    
    if (current_context->context) {
        llama_memory_clear(llama_get_memory(current_context->context), true); // Ensure kv memory/cache is cleared
    }
    current_context->message_history.clear();
    if (current_context->summarizer) {
        current_context->summarizer->summary_slots.clear(); // Clear summary slots when conversation is cleared (Directive #7: direct access)
    }
    current_context->message_cache_dirty = true;
    current_context->n_past = 0;
    current_context->prev_len = 0;
    
    // Reset token count since conversation is cleared
    current_context->message_history_token_count = 0;
    
    LLAMA_LOG("Cleared conversation history and summary slots");
}

// Progress callback function declaration (needs to be outside class for C compatibility)
extern bool model_loading_progress_callback(float progress, void *user_data);

//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODING DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//