// LlamaManager.hpp - header-only implementation for core llama.cpp functionality
// Handles core classs pertaining to llama.cpp backend usage.
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
// 1.  Minimalism & Performance: Deliver lean, efficient solutions; do not create or preserve unused helpers or wrappers.
// 2.  Redundancy Elimination: Remove unused, obsolete, and legacy code—including unneeded interfaces and includes.
// 3.  Consistent Style: Adopt a uniform coding style and structure for clarity and maintainability.
// 4.  Documentation: Write concise comments that explain complex logic and key design decisions.
// 5.  Zero Magic & Strong Typing: Replace magic literals with named constants, enums, or constexpr; prefer scoped enums.
// 6.  Function Boundaries: Define clear responsibilities; reduce overlap and avoid unnecessary layers.
// 7.  Core Preservation: Streamline code while safeguarding essential features; favor direct access over extra abstractions.
// 8.  Const-Correctness & Immutability: Mark variables, parameters, and methods as const wherever possible.
// 9.  RAII & Resource Safety: Encapsulate resource acquisition/release in constructors/destructors or smart pointers.
// 10. Standard Library Preference: Favor STL algorithms and containers over custom loops and buffers.
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
#include <string_view>
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
    
    // Batch processing
    constexpr int32_t MAX_BATCH_SIZE = 512;
    constexpr int32_t BATCH_DIVISOR = 4;
    constexpr int32_t MAX_SEQ_IDS = 8;
      // Safety margins and limits
    constexpr int32_t TOKEN_SAFETY_MARGIN = 32;
    constexpr int32_t BATCH_SAFETY_MARGIN = 128;
    constexpr int32_t MAX_TOKEN_BUFFER_SIZE = 1024;
    constexpr int32_t MAX_SUMMARY_LENGTH = 512;
    constexpr int32_t INITIAL_TOKEN_BUFFER_SIZE = 32;
    constexpr size_t MAX_MESSAGE_HISTORY_SIZE = 1000;
      // Sampler defaults
    constexpr float DEFAULT_TEMPERATURE = 0.8f;
    constexpr float DEFAULT_MIN_P = 0.05f;
    constexpr float DEFAULT_TOP_P = 0.9f;
    constexpr int32_t DEFAULT_TOP_K = 40;
    
    // String processing constants
    constexpr size_t MAX_TEXT_PREVIEW_LENGTH = 50;
    constexpr size_t STRING_RESERVE_MULTIPLIER = 4;
    constexpr int32_t MAX_RETRY_ATTEMPTS = 2;
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
    mutable std::vector<llama_chat_message> message_cache;
    
    // Reference to associated model
    ModelInfo* model_info;
    
    // Special flag for contexts that should reset before each generation
    // Primarily used for summary models that need a clean slate for each task
    bool reset_after_generation = false;
    
    // 5-slot summary system: maintains chronological order of conversation summaries
    // When the 6th summary is generated, slot 0 is dropped, slots shift left, and new summary goes to slot 4
    std::vector<std::string> summary_slots;
    
    ContextInfo() : context(nullptr), batch{}, batch_initialized(false), 
                   n_past(0), prev_len(0), model_info(nullptr) {
        summary_slots.reserve(SummarizerConstants::MAX_SUMMARY_SLOTS);
    }
};

#include "LlamaResponse.hpp"

// Thread Safety Contract (Directive #12):
// This class is NOT thread-safe. External synchronization is required for concurrent access.
// - All public methods must be called from a single thread or protected by external mutexes
// - The llama.cpp backend itself has thread-safety limitations that require careful handling
// - Model loading/unloading operations are particularly sensitive to race conditions
// - Context switching operations modify shared state and must be serialized
class LlamaManager {
private:
    
    std::unordered_map<std::string, std::unique_ptr<ModelInfo>> models;
    std::unordered_map<std::string, std::unique_ptr<ContextInfo>> contexts;
    std::string active_context_id;    ContextInfo* current_context;
    
    // Template and cache management
    mutable std::string template_buffer;
    mutable TokenCache token_cache;
      // Working buffers
    mutable std::string temp_string_buffer;

    // Response generation handler
    mutable LlamaResponse response_generator;

    // Summarizer for handling conversation summarization
    std::unique_ptr<LlamaSummarizer> summarizer;

    // Unified batch management
    void manage_batch(bool clear_only = true) {
        if (!current_context || !current_context->batch_initialized) return;
        
        if (clear_only) {
            current_context->batch.n_tokens = 0;
        }
    }

    // Unified batch token addition with direct position control - FIXED API usage
    bool add_tokens_to_batch(const std::vector<llama_token>& tokens, int32_t start_pos, 
                            const std::vector<llama_seq_id>& seq_ids, bool output_logits = false) {
        if (!current_context || !current_context->batch_initialized || tokens.empty()) return false;
        
        ModelInfo* model_info = get_current_model_info();
        if (!model_info) {
            LLAMA_LOG("Error: No model info available for batch operations");
            return false;
        }
        
        // FIXED: Validate start_pos is reasonable
        if (start_pos < 0 || start_pos >= model_info->n_ctx) {
            LLAMA_LOG("Error: Invalid start position " + std::to_string(start_pos) + " for context size " + std::to_string(model_info->n_ctx));
            return false;
        }
        
        const int32_t n_batch = llama_n_batch(current_context->context);
        if (n_batch <= 0) {
            LLAMA_LOG("Error: Invalid batch size: " + std::to_string(n_batch));
            return false;
        }
        
        manage_batch(true); // Clear batch
        
        // FIXED: Validate sequence IDs
        if (seq_ids.empty()) {
            LLAMA_LOG("Error: Empty sequence ID vector");
            return false;
        }
        
        // FIXED: Use n_batch as the capacity limit instead of n_tokens_alloc
        for (size_t i = 0; i < tokens.size() && current_context->batch.n_tokens < n_batch; ++i) {
            // FIXED: Check position bounds more carefully
            int32_t pos = start_pos + static_cast<int32_t>(i);
            if (pos >= model_info->n_ctx || pos < 0) {
                LLAMA_LOG("Warning: Token position " + std::to_string(pos) + " exceeds context bounds [0, " + std::to_string(model_info->n_ctx) + ")");
                break;
            }
            
            // FIXED: Validate token value
            if (tokens[i] < 0) {
                LLAMA_LOG("Error: Invalid token value " + std::to_string(tokens[i]) + " at position " + std::to_string(i));
                return false;
            }
            
            // FIXED: Ensure we don't exceed batch array bounds using n_batch
            if (current_context->batch.n_tokens >= n_batch) {
                LLAMA_LOG("Warning: Batch capacity exceeded, stopping token addition");
                break;
            }
            
            current_context->batch.token[current_context->batch.n_tokens] = tokens[i];
            current_context->batch.pos[current_context->batch.n_tokens] = pos;            current_context->batch.n_seq_id[current_context->batch.n_tokens] = static_cast<int32_t>(std::min(seq_ids.size(), size_t(LlamaConstants::MAX_SEQ_IDS)));
            
            // FIXED: Safe sequence ID copying with bounds check
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
        ModelInfo* model_info = get_current_model_info();
        if (!model_info || !model_info->vocab) {
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
        const int32_t n_tokens_required = -llama_tokenize(model_info->vocab, text.c_str(), text.size(), nullptr, 0, add_special, true);
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
        if (n_tokens_required > model_info->n_ctx) {
            LLAMA_LOG("Error: Text would produce " + std::to_string(n_tokens_required) + 
                       " tokens, exceeding context limit of " + std::to_string(model_info->n_ctx));
            return {};
        }
        
        // Allocate buffer and tokenize
        std::vector<llama_token> tokens(n_tokens_required);
        const int32_t n_tokens_actual = llama_tokenize(model_info->vocab, text.c_str(), text.size(), 
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

    // REFACTOR: Enhanced context processing - Updated to use context's model
    bool process_context_tokens(const std::vector<llama_token>& tokens, bool is_incremental = true) {
        if (!current_context || !current_context->batch_initialized) {
            LLAMA_LOG("Error: No active context or batch not initialized");
            return false;
        }
        
        if (tokens.empty()) {
            return true; // Empty tokens are valid
        }
        
        ModelInfo* model_info = get_current_model_info();
        if (!model_info) {
            LLAMA_LOG("Error: No model info available for current context");
            return false;
        }
        
        // FIXED: Additional validation before processing
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
        int32_t max_threshold = static_cast<int32_t>(model_info->n_ctx * 0.9f);
        if (!is_incremental) current_context->n_past = 0; // Reset for full context rebuild
        
        // FIXED: Validate n_past bounds before processing
        if (current_context->n_past < 0) {
            LLAMA_LOG("Error: Invalid n_past value: " + std::to_string(current_context->n_past));
            current_context->n_past = 0;
        }
        
        if (current_context->n_past >= model_info->n_ctx) {
            LLAMA_LOG("Error: n_past exceeds context size, resetting");
            current_context->n_past = 0;
            if (current_context->context) {
                llama_kv_self_clear(current_context->context);
            }
        }
        
        // FIXED: Safer overflow check
        const size_t max_safe_add = static_cast<size_t>(std::numeric_limits<int32_t>::max() - current_context->n_past);
        if (tokens.size() > max_safe_add) {
            LLAMA_LOG("Error: Token addition would cause overflow");
            return false;
        }
        
        if (current_context->n_past + static_cast<int32_t>(tokens.size()) > max_threshold) {
            if (is_incremental) {
                LLAMA_LOG("Context would exceed 90% (" + std::to_string(current_context->n_past + tokens.size()) + 
                           "/" + std::to_string(model_info->n_ctx) + " tokens), triggering pruning...");                if (current_context->context) {
                    llama_kv_self_clear(current_context->context);
                }
                current_context->n_past = 0;
                prune_message_history(SummarizerConstants::TARGET_CONTEXT_USAGE);
                return false;
            } else {
                LLAMA_LOG("Error: Full context rebuild would exceed context limit");
                return false;
            }
        }
        
        // Process in optimal chunks with better error handling
        for (size_t start = 0; start < tokens.size(); start += n_batch) {
            size_t end = std::min(start + static_cast<size_t>(n_batch), tokens.size());
            std::vector<llama_token> chunk(tokens.begin() + start, tokens.begin() + end);
            
            bool output_logits = is_incremental && (end == tokens.size());
            
            if (add_tokens_to_batch(chunk, current_context->n_past, seq_ids, output_logits)) {
                // FIXED: Validate batch state before decode
                if (current_context->batch.n_tokens <= 0) {
                    LLAMA_LOG("Warning: Empty batch after token addition");
                    continue;
                }
                
                // FIXED: Additional validation before decode
                if (!current_context->context) {
                    LLAMA_LOG("Error: Context became null before decode");
                    return false;
                }
                
                if (!current_context->batch_initialized) {
                    LLAMA_LOG("Error: Batch became uninitialized before decode");
                    return false;
                }
                
                // FIXED: Validate batch arrays are not null
                if (!current_context->batch.token || !current_context->batch.pos || 
                    !current_context->batch.logits || !current_context->batch.seq_id) {
                    LLAMA_LOG("Error: Batch arrays are null before decode");
                    return false;
                }
                
                // FIXED: Use n_batch for validation instead of n_tokens_alloc
                const int32_t context_n_batch = llama_n_batch(current_context->context);
                if (current_context->batch.n_tokens > context_n_batch) {
                    LLAMA_LOG("Error: Batch token count exceeds batch size limit");
                    return false;
                }
                
                // FIXED: Add comprehensive error checking for decode operation with try-catch
                try {
                    int decode_result = llama_decode(current_context->context, current_context->batch);
                    if (decode_result != 0) {
                        LLAMA_LOG("Error: Failed to decode batch at position " + std::to_string(current_context->n_past) + 
                                   " (error code: " + std::to_string(decode_result) + ")");
                        
                        // FIXED: Don't return false immediately, try to recover
                        if (decode_result == -1) {
                            LLAMA_LOG("Decode error -1: Attempting context reset...");
                            if (current_context->context) {
                                llama_kv_self_clear(current_context->context);
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
                
                current_context->n_past += static_cast<int32_t>(chunk.size());
            } else {
                LLAMA_LOG("Error: Failed to add tokens to batch at position " + std::to_string(current_context->n_past));
                return false;
            }
        }
        
        return true;
    }

    // Unified template application - Updated to use context's model
    bool apply_template_optimized(bool add_generation_prompt, std::string& result) const {
        if (!current_context) return false;
        
        ModelInfo* model_info = get_current_model_info();
        if (!model_info) return false;
        
        const char* tmpl = model_info->get_chat_template();
        if (!tmpl) return false;
        
        // Update message cache if needed
        if (current_context->message_cache_dirty) {
            current_context->message_cache = convert_to_llama_messages();
            current_context->message_cache_dirty = false;
        }
          // Apply template with auto-resize
        template_buffer.resize(model_info->n_ctx * LlamaConstants::STRING_RESERVE_MULTIPLIER);
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
        
        ModelInfo* model_info = get_current_model_info();
        if (!model_info || !model_info->model || !current_context->context) {
            LLAMA_LOG("Error: Invalid model or context state");
            return false;
        }
        
        // Check if context position is reasonable
        if (current_context->n_past < 0 || current_context->n_past >= model_info->n_ctx) {
            LLAMA_LOG("Warning: Context position out of bounds (" + std::to_string(current_context->n_past) + 
                      "/" + std::to_string(model_info->n_ctx) + "), attempting reset");
            
            // Reset context state
            if (current_context->context) {
                llama_kv_self_clear(current_context->context);
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
            LLAMA_LOG("Warning: Very large message history (" + 
                      std::to_string(current_context->message_history.size()) + " messages)");
              // Only trigger aggressive pruning if we're not already in a summary context
            if (active_context_id != "summary_context") {
                // Clear context state before aggressive pruning
                if (current_context->context) {
                    llama_kv_self_clear(current_context->context);
                }
                current_context->n_past = 0;
                current_context->prev_len = 0;
                current_context->message_cache_dirty = true;
                
                // Trigger aggressive pruning
                prune_message_history(SummarizerConstants::AGGRESSIVE_PRUNING_RATIO); // Keep only 30%
                return false; // Indicate that recovery was needed
            } else {
                LLAMA_LOG("Skipping pruning for summary context");
            }
        }
        
        return true; // State is valid
    }    // Helper to get current model info
    ModelInfo* get_current_model_info() const noexcept {
        return current_context ? current_context->model_info : nullptr;
    }

public:
    LlamaManager() : current_context(nullptr), token_cache(LlamaConstants::DEFAULT_TOKEN_CACHE_SIZE) {
        summarizer = std::make_unique<LlamaSummarizer>(this);
    }

    ~LlamaManager() noexcept {
        cleanup();
    }    // Initialize llama.cpp backend
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
        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        model_info->sampler = llama_sampler_chain_init(sparams);
        
        if (!model_info->sampler) {
            LLAMA_LOG("Error: Failed to create sampler for model '" + actual_model_id + "'");
            return false;
        }
        
        llama_sampler_chain_add(model_info->sampler, llama_sampler_init_greedy());
          models[actual_model_id] = std::move(model_info);
        
        // Clear caches when new model is loaded
        clear_caches();
        
        LLAMA_LOG("Model loaded successfully: " + model_path + " as '" + actual_model_id + 
                  "' (ctx:" + std::to_string(context_size) + ", gpu:" + std::to_string(gpu_layers) + ")");
        return true;
    }    // SIMPLIFIED: Context creation with consistent system prompt usage
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
        
        // Don't initialize sampler here - it's now part of the model
        
        // SIMPLIFIED: Always use provided system prompt, or copy from main context if empty
        std::string prompt_to_use = system_prompt;
        if (prompt_to_use.empty() && current_context && !current_context->system_message.empty()) {
            prompt_to_use = current_context->system_message;
        }
          // Set system message if we have one
        if (!prompt_to_use.empty()) {
            context_info->system_message = prompt_to_use;
            context_info->message_history.emplace_back("system", prompt_to_use);
            context_info->message_cache_dirty = true;
        }
        
        // Set the reset after generation flag
        context_info->reset_after_generation = reset_after_generation;
        
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
        if (!it->second->model_info->sampler) {
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
        }
        
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
      std::vector<std::string> list_contexts() const {
        std::vector<std::string> context_list;
        context_list.reserve(contexts.size()); // Directive #8: Smart Caching
        std::transform(contexts.begin(), contexts.end(), std::back_inserter(context_list),
                      [](const auto& pair) { return pair.first; }); // Directive #14: STL algorithms preferred
        return context_list;
    }
      std::string get_active_context() const noexcept {
        return active_context_id;
    }
      bool has_context(const std::string& context_id) const noexcept {
        return contexts.find(context_id) != contexts.end();
    }
    
    // Get context size for a specific context without switching
    int32_t get_context_size_for(const std::string& context_id) const {
        auto it = contexts.find(context_id);
        if (it == contexts.end() || !it->second || !it->second->model_info) {
            return 0;
        }
        return it->second->model_info->n_ctx;
    }
      // Get context usage for a specific context without switching
    int32_t get_context_usage_for(const std::string& context_id) const {
        auto it = contexts.find(context_id);
        if (it == contexts.end() || !it->second) {
            return 0;
        }
        return it->second->n_past;
    }
      // Get the system message from the current context
    const std::string& get_current_system_message() const noexcept {
        static const std::string empty_string;
        if (!current_context) return empty_string;
        return current_context->system_message;    }
    
    // Set the reset before generation flag for a specific context
    bool set_context_reset_flag(const std::string& context_id, bool reset_after_generation) {
        auto it = contexts.find(context_id);
        if (it == contexts.end()) {
            LLAMA_LOG("Error: Context '" + context_id + "' not found");
            return false;
        }
        
        it->second->reset_after_generation = reset_after_generation;
        LLAMA_LOG("Set reset_before_generation flag to " + std::string(reset_after_generation ? "true" : "false") + 
                  " for context '" + context_id + "'");
        return true;
    }    // Clear conversation history
    void clear_conversation() {
        if (!current_context) return;
        
        if (current_context->context) {
            llama_kv_self_clear(current_context->context);
        }
        current_context->message_history.clear();
        current_context->summary_slots.clear(); // Clear summary slots when conversation is cleared
        current_context->message_cache_dirty = true;
        current_context->n_past = 0;
        current_context->prev_len = 0;
        
        LLAMA_LOG("Cleared conversation history and summary slots");    }    // Bridge method for message history pruning with summarization
    void prune_message_history(float keep_ratio);

    // Manually trigger message history pruning with summarization
    bool prune_conversation_with_summary(float keep_ratio = 0.6f);

    // Check if summarization is available (summary context exists)
    bool is_summarization_available() const;

    // Get current message count (useful for determining when pruning might be needed)
    size_t get_message_count() const {
        return current_context ? current_context->message_history.size() : 0;
    }    // Convert message history to llama_chat_message format
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

    // Get the model's default chat template
    std::string get_model_chat_template() const {
        ModelInfo* model_info = get_current_model_info();
        if (!model_info || !model_info->model) {
            return "";
        }
        
        const char* tmpl = llama_model_chat_template(model_info->model, nullptr);
        return tmpl ? std::string(tmpl) : "";
    }
    
    // Get current chat template from current context's model only
    const char* get_current_chat_template() const {
        ModelInfo* model_info = get_current_model_info();
        return model_info ? model_info->get_chat_template() : nullptr;
    }    // Enhanced context update with better tokenization handling - FIXED recursion issue
    bool update_context_with_pruning() {
        ModelInfo* model_info = get_current_model_info();
        if (!model_info || !model_info->model || !current_context || !current_context->context || !model_info->vocab) {
            LLAMA_LOG("Error: Model components not initialized");
            return false;
        }

        // Skip pruning for summary contexts - they manage their own state
        bool is_summary_context = (active_context_id == "summary_context");
        
        // FIXED: Calculate projected context usage before making pruning decisions
        // This prevents the issue where we're at exactly the threshold when generation starts
        std::string formatted_content;
        if (!apply_template_optimized(false, formatted_content)) {
            LLAMA_LOG("Error: Failed to apply chat template for context calculation");
            return false;
        }
        
        // Calculate how many tokens the current conversation would use
        std::vector<llama_token> projected_tokens = process_text_to_tokens(formatted_content, true);
        int32_t projected_usage = static_cast<int32_t>(projected_tokens.size());
        int32_t max_threshold = static_cast<int32_t>(model_info->n_ctx * 0.9f);
        
        bool context_pruned = false;
        
        // Check if we need pruning based on projected usage, not current n_past
        if (!is_summary_context && projected_usage > max_threshold) {
            LLAMA_LOG("Projected context usage at " + std::to_string((float)projected_usage / model_info->n_ctx * 100.0f) + 
                       "% (" + std::to_string(projected_usage) + "/" + std::to_string(model_info->n_ctx) + "), pruning to 60%");
            
            llama_kv_self_clear(current_context->context);
            current_context->n_past = 0;
            current_context->prev_len = 0;
            prune_message_history(SummarizerConstants::TARGET_CONTEXT_USAGE);
            context_pruned = true;
            
            // Mark message cache as dirty since message history was modified
            current_context->message_cache_dirty = true;
            
            // Recalculate formatted content after pruning
            if (!apply_template_optimized(false, formatted_content)) {
                LLAMA_LOG("Error: Failed to apply chat template after pruning");
                return false;
            }
        }        // Apply template and process
        int32_t new_len = static_cast<int32_t>(formatted_content.length());        // CRITICAL: After pruning, the message structure has fundamentally changed.
        // The context state (n_past, prev_len) no longer matches the new message history.
        // We MUST do a full rebuild, never an incremental update, to avoid garbage output.
        if (context_pruned || current_context->prev_len > new_len || 
            (current_context->prev_len == 0 && !current_context->message_history.empty() && !is_summary_context)) {
            // Rebuild context - use add_special=true for full context
            std::vector<llama_token> tokens = process_text_to_tokens(formatted_content, true);
            if (tokens.empty()) {
                LLAMA_LOG("Warning: Template produced no tokens for full rebuild");
                current_context->prev_len = new_len;
                return true;
            }
            
            if (process_context_tokens(tokens, false)) {
                current_context->prev_len = new_len;
                
                // FIXED: After rebuilding context, ensure we have valid logits for generation
                // This is critical when pruning occurs right before generation
                if (current_context->n_past > 0) {
                    float* logits = llama_get_logits(current_context->context);
                    if (!logits) {
                        LLAMA_LOG("Warning: No logits available after context rebuild, will need manual decode");
                        // Context was rebuilt but we need to ensure logits are available
                        // This can happen if the rebuild ended without proper logit generation
                    }
                }
                
                LLAMA_LOG("Context rebuilt successfully with " + 
                           std::to_string(tokens.size()) + " tokens (pruned=" + 
                           std::string(context_pruned ? "true" : "false") + ")");
                return true;
            }
            LLAMA_LOG("Error: Failed to rebuild context");
            return false;
        }

        // Process new content incrementally (only if no pruning occurred)
        if (new_len > current_context->prev_len) {
            std::string new_content = formatted_content.substr(current_context->prev_len);
            if (!new_content.empty()) {
                std::vector<llama_token> new_tokens = process_text_to_tokens(new_content, false);
                if (!new_tokens.empty() && !process_context_tokens(new_tokens, true)) {
                    LLAMA_LOG("Error: Failed to process incremental tokens");
                    return false;
                }
            }
        }

        current_context->prev_len = new_len;
        return true;
    }

    // FIXED: Enhanced sampler configuration with runtime validation
    void configure_sampler(float temperature = LlamaConstants::DEFAULT_TEMPERATURE, float min_p = LlamaConstants::DEFAULT_MIN_P, float top_p = LlamaConstants::DEFAULT_TOP_P, int32_t top_k = LlamaConstants::DEFAULT_TOP_K) {
        ModelInfo* model_info = get_current_model_info();
        if (!model_info) {
            LLAMA_LOG("Error: No model info available for sampler configuration");
            return;
        }
        
        LLAMA_LOG("Configuring sampler for model '" + model_info->model_path + "'");
        
        if (model_info->sampler) {
            llama_sampler_free(model_info->sampler);
            model_info->sampler = nullptr;
        }
        
        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        model_info->sampler = llama_sampler_chain_init(sparams);
        
        if (!model_info->sampler) {
            LLAMA_LOG("Error: Failed to create sampler chain");
            return;
        }
        
        // FIXED: Add sampling strategies without checking return values incorrectly
        if (top_k > 0) {
            llama_sampler_chain_add(model_info->sampler, llama_sampler_init_top_k(top_k));
        }
        
        if (top_p < 1.0f) {
            llama_sampler_chain_add(model_info->sampler, llama_sampler_init_top_p(top_p, 1));
        }
        
        if (min_p > 0.0f) {
            llama_sampler_chain_add(model_info->sampler, llama_sampler_init_min_p(min_p, 1));
        }
        
        if (temperature > 0.0f) {
            llama_sampler_chain_add(model_info->sampler, llama_sampler_init_temp(temperature));
            llama_sampler_chain_add(model_info->sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
        } else {
            llama_sampler_chain_add(model_info->sampler, llama_sampler_init_greedy());
        }
        
        // FIXED: Only validate final sampler state
        if (!model_info->sampler) {
            LLAMA_LOG("Error: Sampler became null during configuration");
        } else {
            LLAMA_LOG("Sampler reconfigured successfully for model");
        }
    }    // Enhanced generation with response delegation to LlamaResponse
    std::string generate_response(const std::string& input, const std::string& username = "Schwi") {
        ModelInfo* model_info = get_current_model_info();
        if (!model_info || !model_info->model_loaded || !model_info->model || !current_context || !current_context->context || !model_info->vocab || !current_context->batch_initialized) {
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
        }

        current_context->message_history.emplace_back(username, input);
        current_context->message_cache_dirty = true;

        // Update context using unified function - with retry logic
        int32_t retry_count = 0;
        const int32_t max_retries = LlamaConstants::MAX_RETRY_ATTEMPTS;
        
        while (retry_count < max_retries) {
            if (update_context_with_pruning()) {
                break; // Success
            }
            
            retry_count++;
            if (retry_count < max_retries) {
                LLAMA_LOG("Retrying context update (" + std::to_string(retry_count + 1) + "/" + std::to_string(max_retries) + ")");
                
                // On retry, try a more aggressive cleanup
                if (current_context && current_context->context) {
                    llama_kv_self_clear(current_context->context);
                    current_context->n_past = 0;
                    current_context->prev_len = 0;
                    current_context->message_cache_dirty = true;
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(ResponseConstants::RECOVERY_SLEEP_MS));
            } else {
                // If we still can't update, try to continue with a minimal context
                LLAMA_LOG("Failed to update context, attempting minimal recovery");
                
                if (current_context && current_context->context) {
                    llama_kv_self_clear(current_context->context);
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

        return response_generator.generate_response(input, username, model_info, current_context, token_adder, context_updater);
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
            avg_tps = (float)current_context->total_generation_tokens / ((float)current_context->last_decode_time_us / 1000000.0f);
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
        timings.t_eval_ms = static_cast<float>(current_context->last_decode_time_us) / 1000.0f;
        return timings;
    }

    // Simplified batch processing using unified functions only
    bool process_full_context(const std::string& full_content) {
        if (!current_context || full_content.empty()) return true;

        ModelInfo* model_info = get_current_model_info();
        if (!model_info) {
            LLAMA_LOG("Error: No model info available for full context processing");
            return false;
        }

        // Use unified tokenization
        std::vector<llama_token> tokens = process_text_to_tokens(full_content, true);
        if (tokens.empty()) return false;

        // Check context capacity with safety margin
        const int32_t safety_margin = LlamaConstants::BATCH_SAFETY_MARGIN;
        if (static_cast<int32_t>(tokens.size()) > model_info->n_ctx - safety_margin) {
            LLAMA_LOG("Warning: Context too large (" + std::to_string(tokens.size()) + 
                       " tokens > " + std::to_string(model_info->n_ctx - safety_margin) + " limit)");
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
        
        // FIXED: Use safer batch initialization without checking non-existent members
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
    }    // Enhanced cleanup with memory optimization
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

    // MOVED: Helper method to add messages to history without immediate context update
    void add_message_to_history(const std::string& role, const std::string& content) {
        if (!current_context) return;

        current_context->message_history.emplace_back(role, content);
        current_context->message_cache_dirty = true;
    }    // MOVED: Get context size for capacity calculations - now uses model-specific value
    int32_t get_context_size() const noexcept {
        ModelInfo* model_info = get_current_model_info();
        return model_info ? model_info->n_ctx : LlamaConstants::DEFAULT_CONTEXT_SIZE; // Default fallback
    }

    // MOVED: Get current context token usage
    int32_t get_context_usage() const noexcept {
        if (!current_context) return 0;
        return current_context->n_past;
    }

    // FIXED: Enhanced batch update with proper token tracking
    bool update_context_from_history() {
        if (!current_context) return false;

        // Store original state for rollback
        int32_t original_n_past = current_context->n_past;
        int32_t original_prev_len = current_context->prev_len;

        // Clear current context state for rebuild
        if (current_context->context) {
            llama_kv_self_clear(current_context->context);
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
        return true;
    }

    // ADDED: Get actual tokenized length of current message history
    int32_t get_message_history_token_count() const {
        ModelInfo* model_info = get_current_model_info();
        if (!current_context || !model_info || !model_info->model || !model_info->vocab) return 0;
        
        // Apply template to get formatted content
        std::string formatted_content;
        if (!apply_template_optimized(false, formatted_content)) {
            return 0;
        }

        // Tokenize and return count
        std::vector<llama_token> tokens = process_text_to_tokens(formatted_content, true);
        return static_cast<int32_t>(tokens.size());
    }

    // Direct access to token cache - no pass-through methods needed
    const TokenCache& get_token_cache() const {
        return token_cache;
    }
    
    TokenCache& get_token_cache() {
        return token_cache;
    }

public:
    // ADDED: Accessor for model to enable external tokenization
    const llama_model* get_model() const {
        ModelInfo* model_info = get_current_model_info();
        return model_info ? model_info->model : nullptr;
    }
    
    // ADDED: Public tokenization method for external use
    std::vector<llama_token> tokenize_text(const std::string& text, bool add_special = false) const {
        return process_text_to_tokens(text, add_special);
    }

private:
    int32_t calculate_optimal_batch_size() const {
        if (!current_context || !current_context->context) {
            return 512;
        }
        
        ModelInfo* model_info = get_current_model_info();
        if (!model_info) {
            return 512;
        }
        
        int32_t n_batch = llama_n_batch(current_context->context);
        int32_t available_ctx = model_info->n_ctx - current_context->n_past;
        
        return std::max(1, std::min({n_batch, available_ctx, 512}));
    }
      void clear_caches() const {
        token_cache.clear();    }
};

// Include LlamaSummarizer implementation after class declaration to avoid circular dependency
#include "LlamaSummarizer.hpp"

// Implementation of methods that depend on LlamaSummarizer
inline LlamaManager::SummarySlotInfo LlamaManager::get_summary_slot_info() const {
    if (!current_context || !summarizer) {
        return {SummarizerConstants::MAX_SUMMARY_SLOTS, 0, {}};
    }
    auto summarizer_info = summarizer->get_summary_slot_info(current_context->summary_slots);
    return {summarizer_info.total_slots, summarizer_info.used_slots, summarizer_info.summaries};
}

inline void LlamaManager::prune_message_history(float keep_ratio) {
    if (!current_context || !summarizer) return;
    
    // Store original message count for logging
    size_t original_count = current_context->message_history.size();
    
    // Perform the pruning with summarization
    summarizer->prune_message_history(current_context->message_history, current_context->summary_slots, keep_ratio);
    
    // Always mark message cache as dirty after pruning since message structure changed
    current_context->message_cache_dirty = true;
    
    LLAMA_LOG("Pruned message history: " + std::to_string(original_count) + " -> " + 
              std::to_string(current_context->message_history.size()) + " messages");
}

inline bool LlamaManager::prune_conversation_with_summary(float keep_ratio) {
    if (!current_context || current_context->message_history.empty()) {
        LLAMA_LOG("Warning: No active context or empty message history for pruning");
        return false;
    }
    
    // Don't prune summary contexts
    if (active_context_id == "summary_context") {
        LLAMA_LOG("Skipping pruning for summary context");
        return true;
    }
    
    size_t original_message_count = current_context->message_history.size();
    LLAMA_LOG("Starting pruning with " + std::to_string(original_message_count) + " messages, keep_ratio=" + std::to_string(keep_ratio));
    
    // Perform pruning with summarization
    prune_message_history(keep_ratio);
    
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

inline bool LlamaManager::is_summarization_available() const {
    bool available = summarizer ? summarizer->is_summarization_available() : false;
    LLAMA_LOG("Summarization available: " + std::string(available ? "true" : "false"));
    return available;
}

// Progress callback function declaration (needs to be outside class for C compatibility)
extern bool model_loading_progress_callback(float progress, void *user_data);

//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODING DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//