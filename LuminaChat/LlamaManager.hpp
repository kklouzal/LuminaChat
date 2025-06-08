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
// Character Set: Use Unicode Character Set
// Whole Program Optimization: Use Link Time Code Generation
// Optimization: Maximum Optimization (Favor Speed) (/O2)
// Enable Intrinsic Functions: Yes (/Oi)
// Favor Size or Speed: Favor fast code (/Ot)
// Whole Program Optimization: Yes (/GL)
// Enable String Pooling: Yes (/GF)
// Runtime Library: Multi-threaded DLL (/MD)
// Enable Run-Time Type Information (RTTI) YES (/GR)
// Link Time Code Generation: Use Link Time Code Generation (/LTCG)
//
// CODING DIRECTIVES:
// 1. Keep the codebase minimalistic, focused on functionality and efficiency.
// 2. Stay consistent with similar coding styles and patterns throughout the codebase.
// 3. Comment code thoroughly, where necessary, to explain complex logic or decisions.
// 4. Always eliminate unused code, dead code, legacy code, and cleanup includes.
// 5. Use consistent _t fixed-width variable types to ensure portability across platforms.
// 6. Ensure there are no logical errors and the execution paths flow as expected.
// 7. Refactor where necessary to maintain clean code, efficient code, and to conform to the above settings and directives.
// 8. NEVER BREAK FUNCTIONALITY THAT IS ALREADY WORKING.
#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <filesystem>
#include <thread>
#include <chrono>
#include "llama-cpp.h"

// Forward declare the progress callback function
bool model_loading_progress_callback(float progress, void *user_data);

class LlamaManager {
private:
    llama_model* model;
    const llama_vocab* vocab;
    llama_context* context;
    llama_sampler* sampler;
    llama_batch batch;
    int32_t n_ctx;
    int32_t n_predict;
    int32_t n_gpu_layers;
    bool model_loaded;
    bool batch_initialized;
    int32_t n_past;
    std::vector<std::pair<std::string, std::string>> message_history;
    std::vector<char> formatted_buffer;
    int32_t prev_len;
    std::string system_message;
    std::string custom_chat_template;
    
    // Performance tracking
    int64_t total_prompt_tokens = 0;
    int64_t total_generation_tokens = 0;
    int64_t last_decode_time_us = 0;
    
    // Optimize buffer management - use string for better memory handling
    mutable std::string template_buffer;
    mutable std::vector<llama_chat_message> message_cache;
    mutable bool message_cache_dirty = true;

    // Helper function to clear batch (equivalent to common_batch_clear from batched.cpp)
    void common_batch_clear(llama_batch & batch) {
        batch.n_tokens = 0;
    }

    // Helper function to add tokens to batch (equivalent to common_batch_add from batched.cpp)
    void common_batch_add(llama_batch & batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool logits) {
        batch.token[batch.n_tokens] = id;
        batch.pos[batch.n_tokens] = pos;
        batch.n_seq_id[batch.n_tokens] = seq_ids.size();
        for (size_t i = 0; i < seq_ids.size(); ++i) {
            batch.seq_id[batch.n_tokens][i] = seq_ids[i];
        }
        batch.logits[batch.n_tokens] = logits;
        batch.n_tokens++;
    }

    // Helper function to clear and prepare batch (equivalent to common_batch_clear)
    void clear_batch() {
        if (batch_initialized) {
            common_batch_clear(batch);
        }
    }

    // Helper function to add tokens to batch (equivalent to common_batch_add from batched.cpp)
    void add_to_batch(llama_token token, llama_pos pos, const std::vector<llama_seq_id>& seq_ids, bool output_logits = false) {
        if (!batch_initialized) {
            return;
        }

        // Check if we have space in the batch
        const int32_t n_batch = llama_n_batch(context);
        if (batch.n_tokens >= n_batch) {
            return;
        }

        common_batch_add(batch, token, pos, seq_ids, output_logits);
    }

    // Process tokens using batched.cpp pattern with proper overflow handling
    bool process_tokens_batch(const std::vector<llama_token>& tokens, bool is_prompt = true) {
        if (!batch_initialized || tokens.empty()) {
            return false;
        }

        const int32_t n_batch = llama_n_batch(context);
        std::vector<llama_seq_id> seq_ids = {0};
        
        // Process tokens in chunks to avoid batch overflow (like batched.cpp)
        for (size_t start = 0; start < tokens.size(); start += n_batch) {
            clear_batch();
            
            size_t end = std::min(start + n_batch, tokens.size());
            
            // Add tokens to current batch
            for (size_t i = start; i < end; ++i) {
                bool output_logits = is_prompt && (i == tokens.size() - 1);
                add_to_batch(tokens[i], n_past + static_cast<int32_t>(i - start), seq_ids, output_logits);
            }
            
            // Decode the batch
            if (batch.n_tokens > 0) {
                if (llama_decode(context, batch) != 0) {
                    std::cerr << "Error: Failed to decode batch at tokens " << start << "-" << (end-1) << std::endl;
                    return false;
                }
                
                n_past += batch.n_tokens;
            }
        }
        
        return true;
    }

public:
    LlamaManager() : model(nullptr), vocab(nullptr), context(nullptr), sampler(nullptr),
                     n_ctx(2048), n_predict(256), n_gpu_layers(0), model_loaded(false), 
                     batch_initialized(false), n_past(0), prev_len(0) {}

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
            std::cerr << "Error: Model file does not exist: " << model_path << std::endl;
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
            std::cerr << "Error: Failed to load model from " << model_path << std::endl;
            return false;
        }

        vocab = llama_model_get_vocab(model);

        // Set up context parameters with proper batch sizing
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = std::min(512, n_ctx / 4); // Optimal batch size
        ctx_params.n_threads = std::thread::hardware_concurrency();
        ctx_params.no_perf = false;

        // Create context
        context = llama_init_from_model(model, ctx_params);
        if (!context) {
            std::cerr << "Error: Failed to create context" << std::endl;
            llama_model_free(model);
            model = nullptr;
            return false;
        }

        // Initialize batch after context creation
        if (!initialize_batch()) {
            llama_free(context);
            llama_model_free(model);
            context = nullptr;
            model = nullptr;
            return false;
        }

        // Initialize sampler with proper error checking
        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        sampler = llama_sampler_chain_init(sparams);
        
        if (!sampler) {
            std::cerr << "Error: Failed to create sampler chain" << std::endl;
            llama_free(context);
            llama_model_free(model);
            context = nullptr;
            model = nullptr;
            return false;
        }
        
        // Use greedy sampling for stability instead of temperature
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
        
        // Verify sampler was created successfully after adding components
        if (!sampler) {
            std::cerr << "Error: Failed to initialize sampler components" << std::endl;
            llama_free(context);
            llama_model_free(model);
            context = nullptr;
            model = nullptr;
            return false;
        }

        model_loaded = true;
        n_past = 0;
        message_history.clear();
        std::cout << "Model loaded successfully: " << model_path << std::endl;
        return true;
    }

    // Clear conversation history
    void clear_conversation() {
        if (context) {
            llama_kv_self_clear(context);
        }
        message_history.clear();
        message_cache_dirty = true;
        n_past = 0;
        prev_len = 0;
    }

    // Set system message
    bool set_system_prompt(const std::string& system_prompt) {
        if (!model_loaded) {
            return false;
        }

        system_message = system_prompt;
        message_cache_dirty = true;
        
        clear_conversation();
        
        if (!system_message.empty()) {
            message_history.emplace_back("system", system_message);
            return update_context();
        }
        
        return true;
    }

    // Convert message history to llama_chat_message format
    std::vector<llama_chat_message> convert_to_llama_messages() const {
        std::vector<llama_chat_message> messages;
        for (const auto& msg : message_history) {
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

    // Enhanced context management with proper pruning (replaces memory sequence management)
    bool prune_context_if_needed() {
        if (!context) return false;
        
        // Check current context usage
        int32_t n_ctx_used = n_past;
        int32_t max_threshold = static_cast<int32_t>(n_ctx * 0.9f); // 90% maximum
        int32_t prune_target = static_cast<int32_t>(n_ctx * 0.6f);  // Prune down to 60%
        
        if (n_ctx_used > max_threshold) {
            std::cout << "Context usage at " << ((float)n_ctx_used / n_ctx * 100.0f) 
                      << "%, pruning to 60%" << std::endl;
            
            llama_kv_self_clear(context);
            n_past = 0;
            prev_len = 0;
            
            prune_message_history(0.6f);
            
            return update_context();
        }
        
        return true;
    }
    
    // Prune message history to keep specified ratio of messages
    void prune_message_history(float keep_ratio) {
        if (message_history.empty()) return;
        
        // Always keep system message if present
        bool has_system = !message_history.empty() && message_history[0].first == "system";
        size_t system_offset = has_system ? 1 : 0;
        
        // Calculate how many non-system messages to keep
        size_t total_messages = message_history.size() - system_offset;
        size_t messages_to_keep = std::max(size_t(2), static_cast<size_t>(total_messages * keep_ratio));
        
        if (messages_to_keep >= total_messages) {
            return; // No pruning needed
        }
        
        // Use move semantics for better performance
        std::vector<std::pair<std::string, std::string>> pruned;
        pruned.reserve(messages_to_keep + system_offset);
        
        if (has_system) {
            pruned.emplace_back(std::move(message_history[0]));
        }
        
        size_t start_idx = message_history.size() - messages_to_keep;
        for (size_t i = start_idx; i < message_history.size(); ++i) {
            pruned.emplace_back(std::move(message_history[i]));
        }
        
        message_history = std::move(pruned);
        message_cache_dirty = true;
        std::cout << "Pruned message history to " << message_history.size() << " messages" << std::endl;
    }

    // Update context with current message history
    bool update_context() {
        if (!model_loaded || !model || !context || !vocab) {
            std::cerr << "Error: Model components not initialized" << std::endl;
            return false;
        }

        // Check if we need to prune context first
        if (!prune_context_if_needed()) {
            std::cerr << "Error: Failed to prune context" << std::endl;
            return false;
        }

        // Get chat template (custom or model default)
        const char* tmpl = get_current_chat_template();
        if (!tmpl) {
            std::cerr << "Error: No chat template available" << std::endl;
            return false;
        }

        // Use optimized template application
        std::string formatted_content;
        if (!apply_chat_template(tmpl, false, formatted_content)) {
            std::cerr << "Error: Failed to apply chat template" << std::endl;
            return false;
        }

        int32_t new_len = static_cast<int32_t>(formatted_content.length());

        // Validate state consistency
        if (prev_len > new_len) {
            std::cerr << "Warning: prev_len (" << prev_len << ") > new_len (" << new_len << "), rebuilding context" << std::endl;
            llama_kv_self_clear(context);
            n_past = 0;
            prev_len = 0;
            
            return process_full_context(formatted_content);
        }

        // Process new tokens if any
        if (new_len > prev_len) {
            std::string new_content = formatted_content.substr(prev_len);
            
            if (!new_content.empty()) {
                // Tokenize new content
                bool is_first = (prev_len == 0);
                const int32_t n_tokens = -llama_tokenize(vocab, new_content.c_str(), new_content.size(), NULL, 0, is_first, true);
                
                if (n_tokens > 0) {
                    std::vector<llama_token> tokens(n_tokens);
                    if (llama_tokenize(vocab, new_content.c_str(), new_content.size(), tokens.data(), tokens.size(), is_first, true) >= 0) {
                        
                        // Check context capacity before processing
                        int32_t max_threshold = static_cast<int32_t>(n_ctx * 0.9f);
                        if (n_past + n_tokens > max_threshold) {
                            std::cout << "Context would exceed 90% with new tokens, pruning..." << std::endl;
                            llama_kv_self_clear(context);
                            n_past = 0;
                            prev_len = 0;
                            
                            prune_message_history(0.6f);
                            
                            return process_full_context(formatted_content);
                        }

                        // Use enhanced batch processing
                        if (process_tokens_batch(tokens, true)) {
                            prev_len = new_len;
                            return true;
                        } else {
                            std::cerr << "Error: Failed to process new tokens" << std::endl;
                            return false;
                        }
                    } else {
                        std::cerr << "Error: Failed to tokenize new content" << std::endl;
                        return false;
                    }
                } else if (n_tokens < 0) {
                    std::cerr << "Error: Tokenization failed with code " << n_tokens << std::endl;
                    return false;
                }
            }
        }

        prev_len = new_len;
        return true;
    }

    // Enhanced sampler configuration with multiple strategies
    void configure_sampler(float temperature = 0.8f, float min_p = 0.05f, float top_p = 0.9f, int32_t top_k = 40) {
        if (sampler) {
            llama_sampler_free(sampler);
            sampler = nullptr;
        }
        
        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        sampler = llama_sampler_chain_init(sparams);
        
        if (!sampler) {
            std::cerr << "Error: Failed to create sampler chain" << std::endl;
            return;
        }
        
        // Add sampling strategies in optimal order (similar to generate.cpp)
        if (top_k > 0) {
            llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
        }
        
        if (top_p < 1.0f) {
            llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
        }
        
        if (min_p > 0.0f) {
            llama_sampler_chain_add(sampler, llama_sampler_init_min_p(min_p, 1));
        }
        
        if (temperature > 0.0f) {
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
        } else {
            // Use greedy sampling for temperature = 0
            llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
        }
    }

    // Generate response for given input - Updated to use batched.cpp patterns
    std::string generate_response(const std::string& input) {
        if (!model_loaded || !model || !context || !sampler || !vocab || !batch_initialized) {
            return "Error: Model components not properly initialized";
        }

        if (input.empty()) {
            return "Error: Empty input";
        }

        auto start_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        // Add system message if this is the first message
        if (message_history.empty() && !system_message.empty()) {
            message_history.emplace_back("system", system_message);
            message_cache_dirty = true;
        }

        // Add user message
        message_history.emplace_back("user", input);
        message_cache_dirty = true;

        // Update context with new message (includes pruning check)
        if (!update_context()) {
            return "Error: Failed to update context";
        }

        // Get chat template for generation (custom or model default)
        const char* tmpl = get_current_chat_template();
        if (!tmpl) {
            return "Error: No chat template available";
        }

        // Use optimized template application for generation
        std::string generation_content;
        if (!apply_chat_template(tmpl, true, generation_content)) {
            return "Error: Failed to apply generation template";
        }

        int32_t gen_len = static_cast<int32_t>(generation_content.length());
        std::string generation_prompt = generation_content.substr(prev_len);
        
        if (!generation_prompt.empty()) {
            // Tokenize generation prompt
            const int32_t n_prompt_tokens = -llama_tokenize(vocab, generation_prompt.c_str(), generation_prompt.size(), NULL, 0, false, true);
            
            if (n_prompt_tokens > 0) {
                std::vector<llama_token> prompt_tokens(n_prompt_tokens);
                if (llama_tokenize(vocab, generation_prompt.c_str(), generation_prompt.size(), prompt_tokens.data(), prompt_tokens.size(), false, true) >= 0) {
                    
                    // Process prompt tokens using batch processing (like batched.cpp initial eval)
                    if (!process_tokens_batch(prompt_tokens, true)) {
                        return "Error: Failed to process generation prompt";
                    }
                    
                    total_prompt_tokens += n_prompt_tokens;
                }
            }
        }

        // Pre-allocate response string for better performance
        std::string response;
        const int32_t max_new_tokens = std::min(n_predict, n_ctx - n_past - 16);
        response.reserve(max_new_tokens * 4); // Pre-allocate assuming average 4 chars per token
        
        int32_t n_generated = 0;

        if (max_new_tokens <= 0) {
            return "Error: No space left in context for generation";
        }

        auto decode_start = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        // Main generation loop (following batched.cpp pattern)
        while (n_generated < max_new_tokens) {
            // Sample next token (like batched.cpp)
            llama_token new_token = llama_sampler_sample(sampler, context, -1);
            
            if (new_token < 0) {
                break;
            }
            
            // Check for end of generation
            if (llama_vocab_is_eog(vocab, new_token)) {
                break;
            }

            // Convert token to text
            char buf[256];
            int32_t n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf) - 1, 0, true);
            if (n <= 0) {
                break;
            }
            
            buf[n] = '\0';
            response.append(buf, n);

            // Process the single token (like batched.cpp single token processing)
            clear_batch();
            std::vector<llama_seq_id> seq_ids = {0};
            add_to_batch(new_token, n_past, seq_ids, true); // Always output logits for next sampling
            
            if (batch.n_tokens > 0 && llama_decode(context, batch) != 0) {
                break;
            }
            
            n_past++;
            n_generated++;
        }

        last_decode_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count() - decode_start;
        total_generation_tokens += n_generated;

        // Add assistant response to message history
        if (!response.empty()) {
            message_history.emplace_back("assistant", response);
            message_cache_dirty = true;
            
            // Update prev_len efficiently
            std::string updated_content;
            if (apply_chat_template(tmpl, false, updated_content)) {
                prev_len = static_cast<int32_t>(updated_content.length());
            }
        }

        if (n_generated > 0) {
            float tokens_per_second = (float)n_generated / ((float)last_decode_time_us / 1000000.0f);
            std::cout << "Generated " << n_generated << " tokens in " 
                      << (last_decode_time_us / 1000.0f) << "ms (" 
                      << tokens_per_second << " t/s)" << std::endl;
        }

        return response;
    }

    // Get performance statistics
    struct PerformanceStats {
        int64_t total_prompt_tokens;
        int64_t total_generation_tokens;
        int64_t last_decode_time_us;
        float average_tokens_per_second;
    };
    
    PerformanceStats get_performance_stats() const {
        float avg_tps = 0.0f;
        if (last_decode_time_us > 0 && total_generation_tokens > 0) {
            avg_tps = (float)total_generation_tokens / ((float)last_decode_time_us / 1000000.0f);
        }
        
        return {
            total_prompt_tokens,
            total_generation_tokens,
            last_decode_time_us,
            avg_tps
        };
    }

    // Timing API methods for compatibility with UI
    struct Timings {
        int32_t n_eval = 0;
        float t_eval_ms = 0.0f;
    };
    
    void reset_timings() {
        total_prompt_tokens = 0;
        total_generation_tokens = 0;
        last_decode_time_us = 0;
    }
    
    Timings get_timings() const {
        Timings timings;
        timings.n_eval = static_cast<int32_t>(total_generation_tokens);
        timings.t_eval_ms = static_cast<float>(last_decode_time_us) / 1000.0f;
        return timings;
    }

    // Enhanced batch processing for context updates (following batched.cpp patterns)
    bool process_full_context(const std::string& full_content) {
        if (full_content.empty()) {
            return true;
        }

        // Tokenize full content
        const int32_t n_tokens = -llama_tokenize(vocab, full_content.c_str(), full_content.size(), NULL, 0, true, true);
        
        if (n_tokens <= 0) {
            return false;
        }

        // Check if it fits in context
        if (n_tokens > n_ctx - 64) {
            std::cerr << "Warning: Context too large (" << n_tokens << " tokens), truncating" << std::endl;
            return false;
        }

        std::vector<llama_token> tokens(n_tokens);
        if (llama_tokenize(vocab, full_content.c_str(), full_content.size(), tokens.data(), tokens.size(), true, true) < 0) {
            return false;
        }

        // Reset context state for full rebuild
        n_past = 0;
        
        // Process tokens using proper batching (like batched.cpp initial evaluation)
        std::vector<llama_seq_id> seq_ids = {0};
        const int32_t n_batch = llama_n_batch(context);
        
        // Process in chunks like batched.cpp
        for (size_t start = 0; start < tokens.size(); start += n_batch) {
            clear_batch();
            
            size_t end = std::min(start + n_batch, tokens.size());
            
            // Add tokens to batch following batched.cpp pattern
            for (size_t i = start; i < end; ++i) {
                bool output_logits = (i == tokens.size() - 1); // Only last token needs logits
                common_batch_add(batch, tokens[i], static_cast<int32_t>(i), seq_ids, output_logits);
            }
            
            // Decode the batch (like batched.cpp)
            if (batch.n_tokens > 0) {
                if (llama_decode(context, batch) != 0) {
                    std::cerr << "Error: Failed to decode context batch at position " << start << std::endl;
                    return false;
                }
            }
        }
        
        // Update state after successful processing
        n_past = n_tokens;
        prev_len = static_cast<int32_t>(full_content.length());
        
        std::cout << "Successfully processed " << n_tokens << " tokens in full context rebuild" << std::endl;
        return true;
    }

    // Clean up resources
    void cleanup() {
        cleanup_batch();
        if (sampler) {
            llama_sampler_free(sampler);
            sampler = nullptr;
        }
        if (context) {
            llama_free(context);
            context = nullptr;
        }
        if (model) {
            llama_model_free(model);
            model = nullptr;
        }
        vocab = nullptr;
        model_loaded = false;
        n_past = 0;
        prev_len = 0;
        message_history.clear();
        message_history.shrink_to_fit(); // Free memory
        formatted_buffer.clear();
        formatted_buffer.shrink_to_fit(); // Free memory
        template_buffer.clear();
        template_buffer.shrink_to_fit(); // Free memory
        message_cache.clear();
        message_cache.shrink_to_fit(); // Free memory
        message_cache_dirty = true;
        custom_chat_template.clear();
    }

    // Free batch resources
    void cleanup_batch() {
        if (batch_initialized) {
            llama_batch_free(batch);
            batch_initialized = false;
        }
    }

    // Initialize batch object exactly like batched.cpp
    bool initialize_batch() {
        if (batch_initialized) {
            return true;
        }

        // Use the same pattern as batched.cpp for batch size calculation
        int32_t batch_size = std::max(static_cast<int32_t>(512), n_ctx / 4);
        
        // Initialize batch with single sequence support (like batched.cpp)
        batch = llama_batch_init(batch_size, 0, 1);
        if (batch.token == nullptr) {
            std::cerr << "Error: Failed to initialize batch" << std::endl;
            return false;
        }

        batch_initialized = true;
        return true;
    }

    // Optimize message conversion with caching
    const std::vector<llama_chat_message>& get_cached_messages() const {
        if (message_cache_dirty) {
            message_cache.clear();
            message_cache.reserve(message_history.size());
            for (const auto& msg : message_history) {
                message_cache.push_back({ msg.first.c_str(), msg.second.c_str() });
            }
            message_cache_dirty = false;
        }
        return message_cache;
    }

    // Optimize template application with better buffer management
    bool apply_chat_template(const char* tmpl, bool add_generation_prompt, std::string& output) const {
        if (!tmpl) return false;
        
        const auto& messages = get_cached_messages();
        
        // Get required size
        int32_t required_size = llama_chat_apply_template(tmpl, messages.data(), messages.size(), 
                                                         add_generation_prompt, nullptr, 0);
        if (required_size <= 0) {
            return false;
        }
        
        // Resize output string directly - more efficient than vector<char>
        output.resize(required_size);
        
        // Apply template
        int32_t actual_size = llama_chat_apply_template(tmpl, messages.data(), messages.size(), 
                                                       add_generation_prompt, output.data(), output.size());
        if (actual_size < 0) {
            return false;
        }
        
        output.resize(actual_size);
        return true;
    }
};

// Progress callback function declaration (needs to be outside class for C compatibility)
extern bool model_loading_progress_callback(float progress, void *user_data);
