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
// 1. Minimalism & Performance: Deliver lean, efficient solutions that avoid unnecessary bloat.
// 2. Consistent Coding Style: Maintain uniform style and structure for clear, maintainable code.
// 3. Clear Documentation: Provide concise comments explaining complex logic and key decisions.
// 4. Eliminate Redundancy: Remove unused, obsolete, and legacy code along with excess includes.
// 5. Optimize Function Structure: Adjust function boundaries to reduce overlap and clarify responsibilities.
// 6. Preserve Core Functionality: Streamline code while safeguarding essential features.
// 7. Cross-Platform Standards: Use fixed-width types and proper initialization to ensure portability.
// 8. Smart Caching: Cache frequently used variables to reduce repeated allocations.
// 9. Ensure Logical Consistency: Review code flow to maintain coherent, error-free execution.
// 10. Continuous Refinement: Regularly refactor and verify that updates preserve stable functionality.
#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <filesystem>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <list>
#include <string_view>
#include "llama-cpp.h"
#include "LogHandler.hpp"
#include "TokenCache.hpp"

// Forward declare the progress callback function
bool model_loading_progress_callback(float progress, void *user_data);

class LlamaManager {
private:
    llama_model* model;
    const llama_vocab* vocab;
    
    // Multi-context support
    struct ContextInfo {
        llama_context* context;
        llama_sampler* sampler;
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
        
        ContextInfo() : context(nullptr), sampler(nullptr), batch{}, batch_initialized(false), 
                       n_past(0), prev_len(0) {}
    };
    
    std::unordered_map<std::string, std::unique_ptr<ContextInfo>> contexts;
    std::string active_context_id;
    ContextInfo* current_context;
    
    // Model settings
    int32_t n_ctx;
    int32_t n_predict;
    int32_t n_gpu_layers;
    bool model_loaded;
    
    // Template and cache management
    std::string custom_chat_template;
    mutable std::string template_buffer;
    mutable TokenCache token_cache;
    
    // Working buffers
    mutable std::string temp_string_buffer;

    // Unified batch management
    void manage_batch(bool clear_only = true) {
        if (!current_context || !current_context->batch_initialized) return;
        
        if (clear_only) {
            current_context->batch.n_tokens = 0;
        }
    }

    // Helper function for thread-safe logging
    void log_message(const std::string& message) const {
        LLAMA_LOG(message);
    }

    // Unified batch token addition with direct position control - FIXED API usage
    bool add_tokens_to_batch(const std::vector<llama_token>& tokens, int32_t start_pos, 
                            const std::vector<llama_seq_id>& seq_ids, bool output_logits = false) {
        if (!current_context || !current_context->batch_initialized || tokens.empty()) return false;
        
        // FIXED: Validate start_pos is reasonable
        if (start_pos < 0 || start_pos >= n_ctx) {
            log_message("Error: Invalid start position " + std::to_string(start_pos) + " for context size " + std::to_string(n_ctx));
            return false;
        }
        
        const int32_t n_batch = llama_n_batch(current_context->context);
        if (n_batch <= 0) {
            log_message("Error: Invalid batch size: " + std::to_string(n_batch));
            return false;
        }
        
        manage_batch(true); // Clear batch
        
        // FIXED: Validate sequence IDs
        if (seq_ids.empty()) {
            log_message("Error: Empty sequence ID vector");
            return false;
        }
        
        // FIXED: Use n_batch as the capacity limit instead of n_tokens_alloc
        for (size_t i = 0; i < tokens.size() && current_context->batch.n_tokens < n_batch; ++i) {
            // FIXED: Check position bounds more carefully
            int32_t pos = start_pos + static_cast<int32_t>(i);
            if (pos >= n_ctx || pos < 0) {
                log_message("Warning: Token position " + std::to_string(pos) + " exceeds context bounds [0, " + std::to_string(n_ctx) + ")");
                break;
            }
            
            // FIXED: Validate token value
            if (tokens[i] < 0) {
                log_message("Error: Invalid token value " + std::to_string(tokens[i]) + " at position " + std::to_string(i));
                return false;
            }
            
            // FIXED: Ensure we don't exceed batch array bounds using n_batch
            if (current_context->batch.n_tokens >= n_batch) {
                log_message("Warning: Batch capacity exceeded, stopping token addition");
                break;
            }
            
            current_context->batch.token[current_context->batch.n_tokens] = tokens[i];
            current_context->batch.pos[current_context->batch.n_tokens] = pos;
            current_context->batch.n_seq_id[current_context->batch.n_tokens] = static_cast<int32_t>(std::min(seq_ids.size(), size_t(8)));
            
            // FIXED: Safe sequence ID copying with bounds check
            for (size_t j = 0; j < std::min(seq_ids.size(), size_t(8)); ++j) {
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

    // Unified tokenization with caching - Updated to use TokenCache
    std::vector<llama_token> process_text_to_tokens(const std::string& text, bool add_special = true) const {
        if (text.empty()) return {};
        
        // Ensure vocab is available
        if (!vocab) {
            log_message("Error: Vocabulary not initialized");
            return {};
        }
        
        std::string cache_key = text + (add_special ? ":s" : ":n");
        
        // Check cache first
        std::vector<llama_token> cached_tokens = token_cache.get(cache_key);
        if (!cached_tokens.empty()) {
            return cached_tokens;
        }
        
        // Get required buffer size for tokenization
        const int32_t n_tokens_required = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, add_special, true);
        if (n_tokens_required <= 0) {
            // Don't treat empty tokenization as warning for whitespace-only text
            if (std::all_of(text.begin(), text.end(), [](char c) { return std::isspace(c); })) {
                // Cache empty result for whitespace-only strings
                token_cache.put(cache_key, {});
                return {};
            }
            log_message("Warning: Text tokenization failed or resulted in 0 tokens: '" + 
                       text.substr(0, 50) + (text.size() > 50 ? "..." : "") + "'");
            return {};
        }
        
        // Add bounds checking for extremely large token counts
        if (n_tokens_required > n_ctx) {
            log_message("Error: Text would produce " + std::to_string(n_tokens_required) + 
                       " tokens, exceeding context limit of " + std::to_string(n_ctx));
            return {};
        }
        
        // Allocate buffer and tokenize
        std::vector<llama_token> tokens(n_tokens_required);
        const int32_t n_tokens_actual = llama_tokenize(vocab, text.c_str(), text.size(), 
                                                       tokens.data(), tokens.size(), add_special, true);
        
        if (n_tokens_actual < 0) {
            log_message("Error: Tokenization failed with error code: " + std::to_string(n_tokens_actual));
            return {};
        }
        
        // Handle case where actual tokens is 0 but expected was > 0
        if (n_tokens_actual == 0 && n_tokens_required > 0) {
            log_message("Warning: Expected " + std::to_string(n_tokens_required) + " tokens but got 0");
            tokens.clear();
        } else if (n_tokens_actual != n_tokens_required) {
            log_message("Warning: Token count mismatch - expected " + std::to_string(n_tokens_required) + 
                       ", got " + std::to_string(n_tokens_actual));
            tokens.resize(std::max(0, n_tokens_actual)); // Ensure non-negative size
        }
        
        // Cache the result
        token_cache.put(cache_key, tokens);
        
        return tokens;
    }

    // REFACTOR: Enhanced context processing with corrected API usage
    bool process_context_tokens(const std::vector<llama_token>& tokens, bool is_incremental = true) {
        if (!current_context || !current_context->batch_initialized) {
            log_message("Error: No active context or batch not initialized");
            return false;
        }
        
        if (tokens.empty()) {
            return true; // Empty tokens are valid
        }
        
        // FIXED: Additional validation before processing
        if (!current_context->context) {
            log_message("Error: Context is null during token processing");
            return false;
        }
        
        const int32_t n_batch = calculate_optimal_batch_size();
        if (n_batch <= 0) {
            log_message("Error: Invalid batch size calculated: " + std::to_string(n_batch));
            return false;
        }
        
        std::vector<llama_seq_id> seq_ids = {0};
        
        // Check context capacity
        int32_t max_threshold = static_cast<int32_t>(n_ctx * 0.9f);
        if (!is_incremental) current_context->n_past = 0; // Reset for full context rebuild
        
        // FIXED: Validate n_past bounds before processing
        if (current_context->n_past < 0) {
            log_message("Error: Invalid n_past value: " + std::to_string(current_context->n_past));
            current_context->n_past = 0;
        }
        
        if (current_context->n_past >= n_ctx) {
            log_message("Error: n_past exceeds context size, resetting");
            current_context->n_past = 0;
            if (current_context->context) {
                llama_kv_self_clear(current_context->context);
            }
        }
        
        // FIXED: Safer overflow check
        const size_t max_safe_add = static_cast<size_t>(std::numeric_limits<int32_t>::max() - current_context->n_past);
        if (tokens.size() > max_safe_add) {
            log_message("Error: Token addition would cause overflow");
            return false;
        }
        
        if (current_context->n_past + static_cast<int32_t>(tokens.size()) > max_threshold) {
            if (is_incremental) {
                log_message("Context would exceed 90% (" + std::to_string(current_context->n_past + tokens.size()) + 
                           "/" + std::to_string(n_ctx) + " tokens), triggering pruning...");
                if (current_context->context) {
                    llama_kv_self_clear(current_context->context);
                }
                current_context->n_past = 0;
                prune_message_history(0.6f);
                return false;
            } else {
                log_message("Error: Full context rebuild would exceed context limit");
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
                    log_message("Warning: Empty batch after token addition");
                    continue;
                }
                
                // FIXED: Additional validation before decode
                if (!current_context->context) {
                    log_message("Error: Context became null before decode");
                    return false;
                }
                
                if (!current_context->batch_initialized) {
                    log_message("Error: Batch became uninitialized before decode");
                    return false;
                }
                
                // FIXED: Validate batch arrays are not null
                if (!current_context->batch.token || !current_context->batch.pos || 
                    !current_context->batch.logits || !current_context->batch.seq_id) {
                    log_message("Error: Batch arrays are null before decode");
                    return false;
                }
                
                // FIXED: Use n_batch for validation instead of n_tokens_alloc
                const int32_t context_n_batch = llama_n_batch(current_context->context);
                if (current_context->batch.n_tokens > context_n_batch) {
                    log_message("Error: Batch token count exceeds batch size limit");
                    return false;
                }

                if (!current_context->batch.embd) {
                    log_message("Error: embd null");
                }
                
                // FIXED: Add comprehensive error checking for decode operation with try-catch
                try {
                    int decode_result = llama_decode(current_context->context, current_context->batch);
                    if (decode_result != 0) {
                        log_message("Error: Failed to decode batch at position " + std::to_string(current_context->n_past) + 
                                   " (error code: " + std::to_string(decode_result) + ")");
                        
                        // FIXED: Don't return false immediately, try to recover
                        if (decode_result == -1) {
                            log_message("Decode error -1: Attempting context reset...");
                            if (current_context->context) {
                                llama_kv_self_clear(current_context->context);
                                current_context->n_past = 0;
                            }
                            return false; // Let caller handle retry
                        } else {
                            log_message("Severe decode error, aborting token processing");
                            return false;
                        }
                    }
                } catch (const std::exception& e) {
                    log_message("Exception during decode: " + std::string(e.what()));
                    return false;
                } catch (...) {
                    log_message("Unknown exception during decode");
                    return false;
                }
                
                current_context->n_past += static_cast<int32_t>(chunk.size());
            } else {
                log_message("Error: Failed to add tokens to batch at position " + std::to_string(current_context->n_past));
                return false;
            }
        }
        
        return true;
    }

    // Unified template application
    bool apply_template_optimized(bool add_generation_prompt, std::string& result) const {
        if (!current_context) return false;
        
        const char* tmpl = get_current_chat_template();
        if (!tmpl) return false;
        
        // Update message cache if needed
        if (current_context->message_cache_dirty) {
            current_context->message_cache = convert_to_llama_messages();
            current_context->message_cache_dirty = false;
        }
        
        // Apply template with auto-resize
        template_buffer.resize(n_ctx * 4);
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
            result = std::string(template_buffer.data(), result_len);
            return true;
        }
        return false;
    }

    // FIXED: Enhanced token-to-text conversion with simplified validation
    std::string convert_token_to_text(llama_token token) const {
        if (!vocab) {
            log_message("Error: Vocabulary not available for token conversion");
            return "";
        }
        
        // FIXED: Simplified token validation - just check for negative values
        if (token < 0) {
            log_message("Warning: Token " + std::to_string(token) + " is negative");
            return "";
        }
        
        temp_string_buffer.resize(32);
        int32_t result = llama_token_to_piece(vocab, token, temp_string_buffer.data(), temp_string_buffer.size(), 0, true);
        if (result < 0) {
            size_t required_size = static_cast<size_t>(-result);
            if (required_size > 1024) {
                log_message("Error: Token conversion requires excessive buffer size: " + std::to_string(required_size));
                return "";
            }
            temp_string_buffer.resize(required_size);
            result = llama_token_to_piece(vocab, token, temp_string_buffer.data(), temp_string_buffer.size(), 0, true);
        }
        
        return (result > 0) ? std::string(temp_string_buffer.data(), result) : "";
    }

    // Prune message history to keep specified ratio of messages
    void prune_message_history(float keep_ratio) {
        if (!current_context || current_context->message_history.empty()) return;
        
        // Always keep system message if present
        bool has_system = !current_context->message_history.empty() && current_context->message_history[0].first == "system";
        size_t system_offset = has_system ? 1 : 0;
        
        // Calculate how many non-system messages to keep
        size_t total_messages = current_context->message_history.size() - system_offset;
        size_t messages_to_keep = std::max(size_t(2), static_cast<size_t>(total_messages * keep_ratio));
        
        if (messages_to_keep >= total_messages) {
            return;
        }
        
        // Use move semantics for better performance
        std::vector<std::pair<std::string, std::string>> pruned;
        pruned.reserve(messages_to_keep + system_offset);
        
        if (has_system) {
            pruned.emplace_back(std::move(current_context->message_history[0]));
        }
        
        size_t start_idx = current_context->message_history.size() - messages_to_keep;
        for (size_t i = start_idx; i < current_context->message_history.size(); ++i) {
            pruned.emplace_back(std::move(current_context->message_history[i]));
        }
        
        current_context->message_history = std::move(pruned);
        current_context->message_cache_dirty = true;
        log_message("Pruned message history to " + std::to_string(current_context->message_history.size()) + " messages");
    }

    // REFACTOR: Update context validation
    bool validate_and_recover_sampler() {
        if (!current_context) {
            log_message("Error: No active context for sampler validation");
            return false;
        }
        
        if (current_context->sampler) {
            return true; // Sampler is valid
        }
        
        log_message("WARNING: Sampler is NULL, attempting recovery...");
        
        // Attempt to recreate a basic greedy sampler
        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        current_context->sampler = llama_sampler_chain_init(sparams);
        
        if (!current_context->sampler) {
            log_message("CRITICAL: Failed to recover sampler!");
            return false;
        }
        
        // Add basic greedy sampling
        llama_sampler_chain_add(current_context->sampler, llama_sampler_init_greedy());
        
        if (!current_context->sampler) {
            log_message("CRITICAL: Sampler became NULL after adding greedy sampler during recovery!");
            return false;
        }
        
        log_message("Sampler recovered successfully with greedy sampling");
        return true;
    }

public:
    LlamaManager() : model(nullptr), vocab(nullptr), current_context(nullptr),
                     n_ctx(2048), n_predict(256), n_gpu_layers(0), model_loaded(false),
                     token_cache(1024) {}

    ~LlamaManager() {
        cleanup();
    }

    // Initialize llama.cpp backend
    bool initialize() {
        ggml_backend_load_all();
        return true;
    }

    // Set configuration parameters
    void set_context_size(int32_t context_size) {
        n_ctx = context_size;
    }
    
    void set_gpu_layers(int32_t gpu_layers) {
        n_gpu_layers = gpu_layers;
    }
    
    void set_predict_tokens(int32_t predict_tokens) {
        n_predict = predict_tokens;
    }

    // Load .gguf model file with optional progress callback
    bool load_model(const std::string& model_path, void* progress_callback_user_data = nullptr) {
        if (!std::filesystem::exists(model_path)) {
            log_message("Error: Model file does not exist: " + model_path);
            return false;
        }

        // Set up model parameters
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = n_gpu_layers;
        
        // Set progress callback if user data is provided
        if (progress_callback_user_data) {
            model_params.progress_callback = model_loading_progress_callback;
            model_params.progress_callback_user_data = progress_callback_user_data;
        }

        // Load the model
        model = llama_model_load_from_file(model_path.c_str(), model_params);
        if (!model) {
            log_message("Error: Failed to load model from " + model_path);
            return false;
        }

        vocab = llama_model_get_vocab(model);

        model_loaded = true;
        
        // Clear caches when new model is loaded
        clear_caches();
        
        log_message("Model loaded successfully: " + model_path);
        return true;
    }

    // SIMPLIFIED: Context creation with consistent system prompt usage
    bool create_context(const std::string& context_id, const std::string& system_prompt = "") {
        if (!model_loaded || !model) {
            log_message("Error: Model must be loaded before creating contexts");
            return false;
        }
        
        if (contexts.find(context_id) != contexts.end()) {
            log_message("Error: Context '" + context_id + "' already exists");
            return false;
        }
        
        auto context_info = std::make_unique<ContextInfo>();
        
        // Set up context parameters
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = std::min(512, n_ctx / 4);
        ctx_params.n_threads = std::thread::hardware_concurrency();
        ctx_params.no_perf = false;
        
        // Create context
        context_info->context = llama_init_from_model(model, ctx_params);
        if (!context_info->context) {
            log_message("Error: Failed to create context '" + context_id + "'");
            return false;
        }
        
        // Initialize batch
        int32_t batch_size = std::min(512, n_ctx / 4);
        context_info->batch = llama_batch_init(batch_size, 0, 1);
        if (context_info->batch.token == nullptr) {
            log_message("Error: Failed to initialize batch for context '" + context_id + "'");
            llama_free(context_info->context);
            return false;
        }
        context_info->batch_initialized = true;
        
        // Initialize sampler
        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        context_info->sampler = llama_sampler_chain_init(sparams);
        
        if (!context_info->sampler) {
            log_message("Error: Failed to create sampler for context '" + context_id + "'");
            llama_batch_free(context_info->batch);
            llama_free(context_info->context);
            return false;
        }
        
        llama_sampler_chain_add(context_info->sampler, llama_sampler_init_greedy());
        
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
        
        contexts[context_id] = std::move(context_info);
        log_message("Created context '" + context_id + "' successfully");
        
        // If this is the first context, make it active
        if (active_context_id.empty()) {
            switch_to_context(context_id);
        }
        
        return true;
    }
    
    bool switch_to_context(const std::string& context_id) {
        auto it = contexts.find(context_id);
        if (it == contexts.end()) {
            log_message("Error: Context '" + context_id + "' not found");
            return false;
        }
        
        active_context_id = context_id;
        current_context = it->second.get();
        log_message("Switched to context '" + context_id + "'");
        return true;
    }
    
    bool remove_context(const std::string& context_id) {
        auto it = contexts.find(context_id);
        if (it == contexts.end()) {
            log_message("Error: Context '" + context_id + "' not found");
            return false;
        }
        
        // Clean up the context
        if (it->second->sampler) {
            llama_sampler_free(it->second->sampler);
        }
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
        log_message("Removed context '" + context_id + "'");
        return true;
    }
    
    std::vector<std::string> list_contexts() const {
        std::vector<std::string> context_list;
        for (const auto& [id, _] : contexts) {
            context_list.push_back(id);
        }
        return context_list;
    }
    
    std::string get_active_context() const {
        return active_context_id;
    }
    
    bool has_context(const std::string& context_id) const {
        return contexts.find(context_id) != contexts.end();
    }

    // Clear conversation history
    void clear_conversation() {
        if (!current_context) return;
        
        if (current_context->context) {
            llama_kv_self_clear(current_context->context);
        }
        current_context->message_history.clear();
        current_context->message_cache_dirty = true;
        current_context->n_past = 0;
        current_context->prev_len = 0;
    }

    // Set system message
    bool set_system_prompt(const std::string& system_prompt) {
        if (!model_loaded || !current_context) {
            return false;
        }

        current_context->system_message = system_prompt;
        current_context->message_cache_dirty = true;
        
        clear_conversation();
        
        if (!current_context->system_message.empty()) {
            current_context->message_history.emplace_back("system", current_context->system_message);
            return update_context_with_pruning();
        }
        
        return true;
    }

    // Convert message history to llama_chat_message format
    std::vector<llama_chat_message> convert_to_llama_messages() const {
        if (!current_context) return {};
        
        std::vector<llama_chat_message> messages;
        for (const auto& msg : current_context->message_history) {
            messages.push_back({ msg.first.c_str(), msg.second.c_str() });
        }
        return messages;
    }

    // Get the model's default chat template
    std::string get_model_chat_template() const {
        if (!model) {
            return "";
        }
        
        const char* tmpl = llama_model_chat_template(model, nullptr);
        return tmpl ? std::string(tmpl) : "";
    }
    
    // Set a custom chat template
    void set_custom_chat_template(const std::string& template_str) {
        custom_chat_template = template_str;
    }
    
    // Get the current chat template (custom or model default)
    const char* get_current_chat_template() const {
        if (!custom_chat_template.empty()) {
            return custom_chat_template.c_str();
        }
        
        if (model) {
            return llama_model_chat_template(model, nullptr);
        }
        
        return nullptr;
    }

    // Enhanced context update with better tokenization handling - FIXED recursion issue
    bool update_context_with_pruning() {
        if (!model_loaded || !model || !current_context || !current_context->context || !vocab) {
            log_message("Error: Model components not initialized");
            return false;
        }

        // Check and handle pruning first
        int32_t n_ctx_used = current_context->n_past;
        int32_t max_threshold = static_cast<int32_t>(n_ctx * 0.9f);
        
        bool context_pruned = false;
        if (n_ctx_used > max_threshold) {
            log_message("Context usage at " + std::to_string((float)n_ctx_used / n_ctx * 100.0f) + 
                       "%, pruning to 60%");
            
            llama_kv_self_clear(current_context->context);
            current_context->n_past = 0;
            current_context->prev_len = 0;
            prune_message_history(0.6f);
            context_pruned = true;
        }

        // Apply template and process
        std::string formatted_content;
        if (!apply_template_optimized(false, formatted_content)) {
            log_message("Error: Failed to apply chat template");
            return false;
        }

        int32_t new_len = static_cast<int32_t>(formatted_content.length());

        // FIXED: If context was pruned, always rebuild from scratch
        if (context_pruned || current_context->prev_len > new_len) {
            // Rebuild context - use add_special=true for full context
            std::vector<llama_token> tokens = process_text_to_tokens(formatted_content, true);
            if (tokens.empty()) {
                log_message("Warning: Template produced no tokens for full rebuild");
                current_context->prev_len = new_len;
                return true;
            }
            
            if (process_context_tokens(tokens, false)) {
                current_context->prev_len = new_len;
                log_message("Context rebuilt successfully after pruning with " + 
                           std::to_string(tokens.size()) + " tokens");
                return true;
            }
            log_message("Error: Failed to rebuild context after pruning");
            return false;
        }

        // Process new content incrementally (only if no pruning occurred)
        if (new_len > current_context->prev_len) {
            std::string new_content = formatted_content.substr(current_context->prev_len);
            if (!new_content.empty()) {
                std::vector<llama_token> new_tokens = process_text_to_tokens(new_content, false);
                if (!new_tokens.empty() && !process_context_tokens(new_tokens, true)) {
                    log_message("Error: Failed to process incremental tokens");
                    return false;
                }
            }
        }

        current_context->prev_len = new_len;
        return true;
    }

    // FIXED: Enhanced sampler configuration with runtime validation
    void configure_sampler(float temperature = 0.8f, float min_p = 0.05f, float top_p = 0.9f, int32_t top_k = 40) {
        if (!current_context) {
            log_message("Error: No active context for sampler configuration");
            return;
        }
        
        log_message("Configuring sampler for context '" + active_context_id + "'");
        
        if (current_context->sampler) {
            llama_sampler_free(current_context->sampler);
            current_context->sampler = nullptr;
        }
        
        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        current_context->sampler = llama_sampler_chain_init(sparams);
        
        if (!current_context->sampler) {
            log_message("Error: Failed to create sampler chain");
            return;
        }
        
        // FIXED: Add sampling strategies without checking return values incorrectly
        if (top_k > 0) {
            llama_sampler_chain_add(current_context->sampler, llama_sampler_init_top_k(top_k));
        }
        
        if (top_p < 1.0f) {
            llama_sampler_chain_add(current_context->sampler, llama_sampler_init_top_p(top_p, 1));
        }
        
        if (min_p > 0.0f) {
            llama_sampler_chain_add(current_context->sampler, llama_sampler_init_min_p(min_p, 1));
        }
        
        if (temperature > 0.0f) {
            llama_sampler_chain_add(current_context->sampler, llama_sampler_init_temp(temperature));
            llama_sampler_chain_add(current_context->sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
        } else {
            llama_sampler_chain_add(current_context->sampler, llama_sampler_init_greedy());
        }
        
        // FIXED: Only validate final sampler state
        if (!current_context->sampler) {
            log_message("Error: Sampler became null during configuration");
        } else {
            log_message("Sampler reconfigured successfully for context '" + active_context_id + "'");
        }
    }

    // FIXED: Enhanced generation with sampler validation and recovery
    std::string generate_response(const std::string& input, const std::string& username = "Schwi") {
        if (!model_loaded || !model || !current_context || !current_context->context || !vocab || !current_context->batch_initialized) {
            return "Error: Model components not properly initialized or no active context";
        }

        if (input.empty()) return "Error: Empty input";

        // FIXED: Validate sampler before proceeding and attempt recovery if needed
        if (!validate_and_recover_sampler()) {
            return "Error: Sampler validation/recovery failed";
        }

        // Setup conversation
        if (current_context->message_history.empty() && !current_context->system_message.empty()) {
            current_context->message_history.emplace_back("system", current_context->system_message);
            current_context->message_cache_dirty = true;
        }

        current_context->message_history.emplace_back(username, input);
        current_context->message_cache_dirty = true;

        // Update context using unified function - with retry logic
        int32_t retry_count = 0;
        const int32_t max_retries = 2;
        
        while (retry_count < max_retries) {
            if (update_context_with_pruning()) {
                break; // Success
            }
            
            retry_count++;
            if (retry_count < max_retries) {
                log_message("Retrying context update (" + std::to_string(retry_count + 1) + "/" + std::to_string(max_retries) + ")");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            } else {
                return "Error: Failed to update context after " + std::to_string(max_retries) + " attempts";
            }
        }

        // Prepare for generation using unified template function
        std::string generation_content;
        if (!apply_template_optimized(true, generation_content)) {
            return "Error: Failed to apply generation template";
        }

        // Process generation prompt using unified tokenization
        std::string generation_prompt = generation_content.substr(current_context->prev_len);
        if (!generation_prompt.empty()) {
            // For generation prompts, typically don't add special tokens
            std::vector<llama_token> prompt_tokens = process_text_to_tokens(generation_prompt, false);
            if (!prompt_tokens.empty()) {
                if (!process_context_tokens(prompt_tokens, true)) {
                    return "Error: Failed to process generation prompt";
                }
            } else {
                log_message("Warning: Generation prompt produced no tokens");
            }
        }

        // Generate response using unified functions
        std::string response;
        const int32_t safety_margin = 32; // Reserve space for potential special tokens
        const int32_t max_new_tokens = std::min(n_predict, n_ctx - current_context->n_past - safety_margin);
        response.reserve(max_new_tokens * 4);
        
        if (max_new_tokens <= 0) {
            return "Error: No space left in context for generation (context: " + 
                   std::to_string(current_context->n_past) + "/" + std::to_string(n_ctx) + ")";
        }

        // FIXED: Enhanced validation before generation loop
        if (!current_context->sampler) {
            log_message("CRITICAL: Sampler is null before generation loop after validation!");
            return "Error: Sampler validation failed";
        }
        
        if (!current_context->context) {
            log_message("Error: Context is null before generation");
            return "Error: Context not properly initialized";
        }
        
        // FIXED: Validate that we have logits available for sampling
        if (current_context->n_past == 0) {
            log_message("Error: No tokens processed yet, cannot generate");
            return "Error: Context is empty, cannot generate response";
        }

        log_message("Starting generation with " + std::to_string(max_new_tokens) + " max tokens, n_past=" + std::to_string(current_context->n_past));

        auto decode_start = std::chrono::high_resolution_clock::now();
        int32_t n_generated = 0;
        std::vector<llama_seq_id> seq_ids = {0};

        // FIXED: Enhanced generation loop with comprehensive validation
        while (n_generated < max_new_tokens) {
            // FIXED: Validate sampler on every iteration to catch when it becomes null
            if (!current_context->sampler) {
                log_message("CRITICAL: Sampler became null during generation at token " + std::to_string(n_generated));
                return "Error: Sampler failed during generation";
            }
            
            if (!current_context->context) {
                log_message("CRITICAL: Context became null during generation");
                return "Error: Context lost during generation";
            }
            
            // FIXED: Validate that n_past is within reasonable bounds
            if (current_context->n_past <= 0 || current_context->n_past >= n_ctx) {
                log_message("Error: Invalid context position during generation: " + std::to_string(current_context->n_past));
                return "Error: Context position invalid";
            }
            
            llama_token new_token;
            try {
                new_token = llama_sampler_sample(current_context->sampler, current_context->context, -1);
            } catch (const std::exception& e) {
                log_message("Exception during token sampling: " + std::string(e.what()));
                return "Error: Exception during token generation";
            } catch (...) {
                log_message("Unknown exception during token sampling");
                return "Error: Unknown exception during generation";
            }
            
            if (new_token < 0) {
                log_message("Error: Invalid token generated: " + std::to_string(new_token));
                break;
            }
            
            if (llama_vocab_is_eog(vocab, new_token)) {
                log_message("End of generation token encountered");
                break;
            }

            // Convert token to text using unified function
            std::string token_text = convert_token_to_text(new_token);
            if (token_text.empty()) {
                log_message("Warning: Empty token text for token " + std::to_string(new_token));
                continue;
            }
            
            response += token_text;

            // FIXED: Validate batch state before adding token
            if (!current_context->batch_initialized) {
                log_message("Error: Batch not initialized during generation");
                return "Error: Batch system failed";
            }

            // Process token using unified batch function
            if (!add_single_token_to_batch(new_token, current_context->n_past, seq_ids, true)) {
                log_message("Error: Failed to add token to batch during generation");
                return "Error: Token processing failed";
            }
            
            if (current_context->batch.n_tokens > 0 && llama_decode(current_context->context, current_context->batch) != 0) {
                log_message("Error: Failed to decode during generation at token " + std::to_string(n_generated));
                return "Error: Token decode failed";
            }
            
            current_context->n_past++;
            n_generated++;
            
            // FIXED: Periodic sampler validation during long generation
            if (n_generated % 10 == 0 && !current_context->sampler) {
                log_message("CRITICAL: Sampler became null during long generation at token " + std::to_string(n_generated));
                return "Error: Sampler failed during long generation";
            }
        }

        // Update timing and history
        auto decode_end = std::chrono::high_resolution_clock::now();
        current_context->last_decode_time_us = std::chrono::duration_cast<std::chrono::microseconds>(decode_end - decode_start).count();
        current_context->total_generation_tokens += n_generated;

        if (!response.empty()) {
            current_context->message_history.emplace_back("assistant", response);
            current_context->message_cache_dirty = true;
            
            std::string updated_content;
            if (apply_template_optimized(false, updated_content)) {
                current_context->prev_len = static_cast<int32_t>(updated_content.length());
            }
        }

        if (n_generated > 0) {
            float tokens_per_second = (float)n_generated / ((float)current_context->last_decode_time_us / 1000000.0f);
            log_message("Generated " + std::to_string(n_generated) + " tokens in " + 
                       std::to_string(current_context->last_decode_time_us / 1000.0f) + "ms (" + 
                       std::to_string(tokens_per_second) + " t/s)");
        }

        // FIXED: Final sampler validation
        if (!current_context->sampler) {
            log_message("WARNING: Sampler is NULL at end of generation!");
        }

        return response;
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

        // Use unified tokenization
        std::vector<llama_token> tokens = process_text_to_tokens(full_content, true);
        if (tokens.empty()) return false;

        // Check context capacity with safety margin
        const int32_t safety_margin = 128;
        if (static_cast<int32_t>(tokens.size()) > n_ctx - safety_margin) {
            log_message("Warning: Context too large (" + std::to_string(tokens.size()) + 
                       " tokens > " + std::to_string(n_ctx - safety_margin) + " limit)");
            return false;
        }

        // Use unified context processing
        current_context->n_past = 0;
        if (!process_context_tokens(tokens, false)) {
            return false;
        }
        
        current_context->prev_len = static_cast<int32_t>(full_content.length());
        
        log_message("Successfully processed " + std::to_string(tokens.size()) + 
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
            log_message("Error: Invalid batch size for initialization: " + std::to_string(batch_size));
            return false;
        }
        
        // FIXED: Use safer batch initialization without checking non-existent members
        current_context->batch = llama_batch_init(batch_size, 0, 1);
        if (current_context->batch.token == nullptr || 
            current_context->batch.pos == nullptr || 
            current_context->batch.n_seq_id == nullptr ||
            current_context->batch.seq_id == nullptr ||
            current_context->batch.logits == nullptr) {
            log_message("Error: Failed to initialize batch arrays with size " + std::to_string(batch_size));
            return false;
        }

        current_context->batch_initialized = true;
        log_message("Initialized batch with optimal size: " + std::to_string(batch_size));
        return true;
    }

    // Enhanced cleanup with memory optimization
    void cleanup() {
        log_message("Cleanup called - cleaning up " + std::to_string(contexts.size()) + " contexts");
        
        clear_caches();
        
        // Clean up all contexts
        for (auto& [id, context_info] : contexts) {
            if (context_info->sampler) {
                llama_sampler_free(context_info->sampler);
                context_info->sampler = nullptr;
            }
            if (context_info->batch_initialized) {
                llama_batch_free(context_info->batch);
                context_info->batch_initialized = false;
            }
            if (context_info->context) {
                llama_free(context_info->context);
                context_info->context = nullptr;
            }
        }
        
        contexts.clear();
        active_context_id.clear();
        current_context = nullptr;
        
        if (model) {
            llama_model_free(model);
            model = nullptr;
        }
        
        vocab = nullptr;
        model_loaded = false;
        
        // Efficient memory cleanup
        std::string().swap(template_buffer);
        std::string().swap(custom_chat_template);
        
        log_message("Cleanup completed");
    }

    // MOVED: Helper method to add messages to history without immediate context update
    void add_message_to_history(const std::string& role, const std::string& content) {
        if (!current_context) return;

        current_context->message_history.emplace_back(role, content);
        current_context->message_cache_dirty = true;
    }

    // MOVED: Get context size for capacity calculations
    int32_t get_context_size() const {
        return n_ctx;
    }

    // MOVED: Get current context token usage
    int32_t get_context_usage() const {
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
            log_message("Failed to update context from history, rolled back to previous state");
            return false;
        }
        
        // Validate the rebuild was successful
        if (current_context->n_past <= 0 && !current_context->message_history.empty()) {
            log_message("Warning: Context rebuild resulted in zero tokens despite having message history");
        }
        
        log_message("Successfully rebuilt context from " + std::to_string(current_context->message_history.size()) + 
                   " messages, using " + std::to_string(current_context->n_past) + " tokens");
        return true;
    }

    // ADDED: Get actual tokenized length of current message history
    int32_t get_message_history_token_count() const {
        if (!current_context || !model || !vocab) return 0;
        
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
        return model;
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
        
        int32_t n_batch = llama_n_batch(current_context->context);
        int32_t available_ctx = n_ctx - current_context->n_past;
        
        return std::max(1, std::min({n_batch, available_ctx, 512}));
    }
    
    void clear_caches() const {
        token_cache.clear();
    }
};

// Progress callback function declaration (needs to be outside class for C compatibility)
extern bool model_loading_progress_callback(float progress, void *user_data);

//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODING DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//