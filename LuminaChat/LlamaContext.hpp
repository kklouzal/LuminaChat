// LlamaContext.hpp - header-only implementation for LLaMA context management
// 
// RESPONSIBILITY: Context state management and operations
// - Context state tracking and validation
// - Message history management
// - Batch operations and token processing
// - Template application and conversation state
// - Conversation summarization coordination
//
// DELEGATION: Model management is handled by LlamaManager
// 
// Handles context-specific operations for llama.cpp backend usage.
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
#include <memory>
#include <utility>
#include <limits>
#include "llama-cpp.h"
#include "LogHandler.hpp"

// Forward declarations
class LlamaSummarizer;

// Forward declaration of SummarizerConstants (defined in LlamaSummarizer.hpp)
namespace SummarizerConstants {
    extern const size_t MAX_SUMMARY_SLOTS;
    extern const float MAX_CONTEXT_USAGE;
    extern const float TARGET_CONTEXT_USAGE;
    extern const float AGGRESSIVE_PRUNING_RATIO;
}

// Constants for configuration and performance
namespace LlamaConstants {
    constexpr int32_t DEFAULT_CONTEXT_SIZE = 2048;
    constexpr int32_t DEFAULT_GPU_LAYERS = 0;
    constexpr int32_t DEFAULT_PREDICT_TOKENS = 256;
    constexpr int32_t MAX_SEQ_IDS = 8;
    constexpr size_t MAX_MESSAGE_HISTORY_SIZE = 1000;
    constexpr size_t STRING_RESERVE_MULTIPLIER = 4;
    // Additional constants needed for batch processing
    constexpr int32_t MAX_BATCH_SIZE = 8192;
    constexpr int32_t BATCH_DIVISOR = 4;
}

// ModelInfo definition - moved from LlamaManager.hpp
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

// Context information and state management
struct ContextInfo {
    llama_context* context;
    llama_batch batch;
    bool batch_initialized;
    int32_t n_past;
    int32_t prev_len;
    std::vector<std::pair<std::string, std::string>> message_history;
    std::string system_message;
    
    // Template buffer - context-specific to avoid conflicts
    mutable std::string template_buffer;
    
    // Performance tracking
    int64_t total_generation_tokens = 0;
    int64_t last_decode_time_us = 0;
    
    // Cache state - enhanced with conversation state tracking
    mutable bool message_cache_dirty = true;
    mutable std::vector<llama_chat_message> message_cache;
    
    // Conversation state for improved logic flow - single source of truth
    struct ConversationState {
        std::string formatted_content;     // Template-applied content ready for tokenization
        std::vector<llama_token> tokens;   // Tokenized representation of formatted_content
        int32_t total_token_count = 0;     // Exact count of tokens (for efficiency)
        bool needs_rebuild = true;         // Flag indicating if context rebuild is required
        
        // Mark state as invalid, requiring full rebuild
        void invalidate() {
            needs_rebuild = true;
            formatted_content.clear();
            tokens.clear();
            total_token_count = 0;
        }
        
        // Update state with new processed content (marks as valid)
        void update(const std::string& content, const std::vector<llama_token>& new_tokens) {
            formatted_content = content;
            tokens = new_tokens;
            total_token_count = static_cast<int32_t>(new_tokens.size());
            needs_rebuild = false;  // State is now up-to-date
        }
    } conversation_state;
    
    // Token count tracking for efficiency - maintains accurate count of tokens in message history
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
    
    // Template application specific to this context
    bool apply_template(bool add_generation_prompt, std::string& result) const {
        if (!model_info) return false;
        
        const char* tmpl = model_info->get_chat_template();
        if (!tmpl) return false;
        
        // Update message cache if needed
        if (message_cache_dirty) {
            message_cache = convert_to_llama_messages();
            message_cache_dirty = false;
        }
        
        // Apply template with auto-resize
        template_buffer.resize(model_info->n_ctx * LlamaConstants::STRING_RESERVE_MULTIPLIER);
        int32_t result_len = llama_chat_apply_template(
            tmpl, message_cache.data(), message_cache.size(),
            add_generation_prompt, template_buffer.data(), template_buffer.size()
        );
        
        if (result_len < 0) {
            template_buffer.resize(-result_len);
            result_len = llama_chat_apply_template(
                tmpl, message_cache.data(), message_cache.size(),
                add_generation_prompt, template_buffer.data(), template_buffer.size()
            );
        }
        
        if (result_len > 0) {
            result = std::string(template_buffer.data(), result_len);
            return true;
        }
        return false;
    }
    
    // Validate context state
    bool validate_state() const {
        if (!model_info || !model_info->model || !context) {
            return false;
        }
        
        // Check if context position is reasonable
        if (n_past < 0 || n_past >= model_info->n_ctx) {
            return false;
        }
        
        // Check for extremely long message history that might cause issues
        if (message_history.size() > LlamaConstants::MAX_MESSAGE_HISTORY_SIZE) {
            return false;
        }
        
        return true;
    }
    
    // Reset context state
    void reset_context_state() {
        if (context) {
            llama_memory_clear(llama_get_memory(context), true);
        }
        n_past = 0;
        prev_len = 0;
        conversation_state.invalidate();
        message_cache_dirty = true;
    }
    
    // Check if context needs pruning based on token count
    bool needs_pruning(int32_t token_count) const {
        if (!model_info) return false;
        int32_t max_threshold = static_cast<int32_t>(model_info->n_ctx * 0.9f);
        return token_count > max_threshold;
    }
    
    // Batch management methods - context-specific
    void clear_batch() {
        if (!batch_initialized) return;
        batch.n_tokens = 0;
    }

    // Add tokens from a single input to batch
    // NOTE: This is for single-sequence processing, not multi-sequence batching
    bool add_tokens_to_batch(const std::vector<llama_token>& tokens, int32_t start_pos, 
                            const std::vector<llama_seq_id>& seq_ids, bool output_logits = false) {
        if (!batch_initialized || tokens.empty() || !model_info) return false;
        
        // Validate start_pos is reasonable
        if (start_pos < 0 || start_pos >= model_info->n_ctx) {
            return false;
        }        
        
        const int32_t n_batch = llama_n_batch(context);
        if (n_batch <= 0) {
            return false;
        }
        
        clear_batch(); // Clear batch
        
        // Validate sequence IDs
        if (seq_ids.empty()) {
            return false;
        }
        
        // Use n_batch as the capacity limit
        for (size_t i = 0; i < tokens.size() && batch.n_tokens < n_batch; ++i) {
            // Check position bounds carefully
            int32_t pos = start_pos + static_cast<int32_t>(i);
            if (pos >= model_info->n_ctx || pos < 0) {
                break;
            }
            
            // Validate token value
            if (tokens[i] < 0) {
                return false;
            }
            
            // Ensure we don't exceed batch array bounds
            if (batch.n_tokens >= n_batch) {
                break;
            }
            
            batch.token[batch.n_tokens] = tokens[i];
            batch.pos[batch.n_tokens] = pos;
            batch.n_seq_id[batch.n_tokens] = static_cast<int32_t>(std::min(seq_ids.size(), size_t(LlamaConstants::MAX_SEQ_IDS)));
            
            // Safe sequence ID copying with bounds check
            for (size_t j = 0; j < std::min(seq_ids.size(), size_t(LlamaConstants::MAX_SEQ_IDS)); ++j) {
                batch.seq_id[batch.n_tokens][j] = seq_ids[j];
            }
            
            batch.logits[batch.n_tokens] = (i == tokens.size() - 1) ? output_logits : false;
            batch.n_tokens++;
        }
        
        return batch.n_tokens > 0;
    }
    
    // Add message to this context's history
    void add_message(const std::string& role, const std::string& content) {
        message_history.emplace_back(role, content);
        message_cache_dirty = true;
        conversation_state.needs_rebuild = true;
        
        // Token count will be updated during the next context update for accuracy
    }
    
    // Clear conversation for this context (implementation after LlamaSummarizer include)
    void clear_conversation();

    // Prune conversation with summarization for this context (implementation after LlamaSummarizer include)
    bool prune_with_summarization(float keep_ratio = 0.6f);
    
    // Calculate optimal batch size for this context
    int32_t calculate_optimal_batch_size() const {
        if (!context) {
            return LlamaConstants::MAX_BATCH_SIZE;
        }
        
        if (!model_info) {
            return LlamaConstants::MAX_BATCH_SIZE;
        }
        
        int32_t n_batch = llama_n_batch(context);
        int32_t available_ctx = model_info->n_ctx - n_past;
        
        return std::max(1, std::min({n_batch, available_ctx, LlamaConstants::MAX_BATCH_SIZE}));
    }
    
    // Enhanced context processing - Updated for proper single-input batching
    // NOTE: Batching in llama.cpp is designed for processing multiple separate inputs/sequences 
    // simultaneously, NOT for splitting a single input into chunks. Each batch operation should
    // contain tokens from potentially multiple different inputs, each with their own sequence IDs.
    // For single inputs that exceed batch capacity, we use incremental batch processing for optimal performance.
    template<typename SummarizerCallback>
    bool process_context_tokens(const std::vector<llama_token>& tokens, bool is_incremental, SummarizerCallback&& pruning_callback) {
        if (!batch_initialized) {
            LLAMA_LOG("Error: No active context or batch not initialized");
            return false;
        }
        
        if (tokens.empty()) {
            return true; // Empty tokens are valid
        }
        
        if (!model_info) {
            LLAMA_LOG("Error: No model info available for current context");
            return false;
        }
        
        // Additional validation before processing
        if (!context) {
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
        int32_t max_threshold = static_cast<int32_t>(model_info->n_ctx * 0.9f);
        if (!is_incremental) {
            LLAMA_LOG("Starting context rebuild: FULL (non-incremental) - processing " + std::to_string(tokens.size()) + " tokens");
            n_past = 0; // Reset for full context rebuild
        }
        
        // Validate n_past bounds before processing
        if (n_past < 0) {
            LLAMA_LOG("Error: Invalid n_past value: " + std::to_string(n_past));
            n_past = 0;
        }
        
        if (n_past >= model_info->n_ctx) {
            LLAMA_LOG("Error: n_past exceeds context size, resetting");
            n_past = 0;
            if (context) {
                llama_memory_clear(llama_get_memory(context), true); // Ensure kv memory/cache is cleared
            }
        }
        
        // Safer overflow check
        const size_t max_safe_add = static_cast<size_t>(std::numeric_limits<int32_t>::max() - n_past);
        if (tokens.size() > max_safe_add) {
            LLAMA_LOG("Error: Token addition would cause overflow");
            return false;
        }
        
        if (n_past + static_cast<int32_t>(tokens.size()) > max_threshold) {
            if (is_incremental) {
                LLAMA_LOG("Starting context rebuild: PARTIAL (triggering pruning) - context would exceed 90% (" + std::to_string(n_past + tokens.size()) + 
                           "/" + std::to_string(model_info->n_ctx) + " tokens)");
                
                if (pruning_callback()) {
                    // After pruning, the calling method will handle context rebuild
                    return false;
                } else {
                    LLAMA_LOG("Error: Pruning failed during token processing");
                    return false;
                }
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
        } else {
            // Process all tokens in a single batch (normal case)
            bool output_logits = is_incremental;
            
            if (!add_tokens_to_batch(tokens, n_past, seq_ids, output_logits)) {
                LLAMA_LOG("Error: Failed to add tokens to batch");
                return false;
            }
            
            // Validate batch state before decode
            if (batch.n_tokens <= 0) {
                LLAMA_LOG("Warning: Empty batch after token addition");
                return true;
            }
            
            // Additional validation before decode
            if (!context) {
                LLAMA_LOG("Error: Context became null before decode");
                return false;
            }
            
            if (!batch_initialized) {
                LLAMA_LOG("Error: Batch became uninitialized before decode");
                return false;
            }
            
            // Validate batch arrays are not null
            if (!batch.token || !batch.pos || 
                !batch.logits || !batch.seq_id) {
                LLAMA_LOG("Error: Batch arrays are null before decode");
                return false;
            }
            
            // Use n_batch for validation
            const int32_t context_n_batch = llama_n_batch(context);
            if (batch.n_tokens > context_n_batch) {
                LLAMA_LOG("Error: Batch token count exceeds batch size limit");
                return false;
            }
            
            // Add comprehensive error checking for decode operation with try-catch
            try {
                int decode_result = llama_decode(context, batch);
                if (decode_result != 0) {
                    LLAMA_LOG("Error: Failed to decode batch at position " + std::to_string(n_past) + 
                               " (error code: " + std::to_string(decode_result) + ")");
                    
                    // Don't return false immediately, try to recover
                    if (decode_result == -1) {
                        LLAMA_LOG("Decode error -1: Attempting context reset...");
                        if (context) {
                            llama_memory_clear(llama_get_memory(context), true); // Ensure kv memory/cache is cleared
                            n_past = 0;
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
            
            n_past += static_cast<int32_t>(tokens.size());
        }
        
        return true;
    }
    
    // Helper function to process large context using incremental batch rebuilds
    // This allows us to process large token sets efficiently instead of sequentially
    bool process_large_context_incrementally(const std::vector<llama_token>& tokens, int32_t n_batch) {
        if (tokens.empty()) return false;
        
        LLAMA_LOG("Processing " + std::to_string(tokens.size()) + " tokens using incremental batch method (batch size: " + std::to_string(n_batch) + ")");        
        // Clear context for full rebuild using context's reset method
        reset_context_state();
        
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
                     " (" + std::to_string(chunk_size) + " tokens, pos: " + std::to_string(n_past) + ")");
            
            if (!add_tokens_to_batch(chunk, n_past, seq_ids, output_logits)) {
                LLAMA_LOG("Error: Failed to add batch chunk starting at token " + std::to_string(processed));
                return false;
            }
            
            // Validate and decode the batch
            if (batch.n_tokens <= 0) {
                LLAMA_LOG("Warning: Empty batch in incremental processing");
                processed += chunk_size;
                continue;
            }
            
            try {
                int decode_result = llama_decode(context, batch);
                if (decode_result != 0) {
                    LLAMA_LOG("Error: Failed to decode incremental batch at position " + 
                             std::to_string(n_past) + " (error code: " + std::to_string(decode_result) + ")");
                    return false;
                }
            } catch (const std::exception& e) {
                LLAMA_LOG("Exception during incremental decode: " + std::string(e.what()));
                return false;
            } catch (...) {
                LLAMA_LOG("Unknown exception during incremental decode");
                return false;
            }
            
            n_past += static_cast<int32_t>(chunk_size);
            processed += chunk_size;
        }
        
        LLAMA_LOG("Successfully processed " + std::to_string(total_tokens) + " tokens using " + 
                 std::to_string((total_tokens + n_batch - 1) / n_batch) + " incremental batches");        
        return true;
    }
    
private:
    // Convert message history to llama_chat_message format
    std::vector<llama_chat_message> convert_to_llama_messages() const {
        std::vector<llama_chat_message> messages;
        messages.reserve(message_history.size());
        for (const auto& msg : message_history) {
            messages.push_back({ msg.first.c_str(), msg.second.c_str() });
        }
        return messages;
    }
};

// Include LlamaSummarizer implementation after class declaration to avoid circular dependency
#include "LlamaSummarizer.hpp"

// Implementation of ContextInfo methods that depend on LlamaSummarizer
inline void ContextInfo::clear_conversation() {
    if (context) {
        llama_memory_clear(llama_get_memory(context), true); // Ensure kv memory/cache is cleared
    }
    message_history.clear();
    if (summarizer) {
        summarizer->summary_slots.clear(); // Clear summary slots when conversation is cleared
    }
    message_cache_dirty = true;
    n_past = 0;
    prev_len = 0;
    message_history_token_count = 0;
    conversation_state.needs_rebuild = true;
}

inline bool ContextInfo::prune_with_summarization(float keep_ratio) {
    if (!summarizer) {
        return false; // Cannot prune without summarizer
    }
    
    size_t original_message_count = message_history.size();
    if (original_message_count < 3) {
        return false; // Need at least a few messages to prune
    }
    
    // Reset context state before pruning since we'll need to rebuild
    reset_context_state();
    
    // Perform the pruning with summarization
    summarizer->prune_message_history(message_history, keep_ratio);
    
    // Always mark message cache as dirty after pruning since message structure changed
    message_cache_dirty = true;
    conversation_state.needs_rebuild = true;
    
    return true;
}

//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODING DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//
