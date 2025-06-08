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
// 6. Cache frequently used variables to avoid repeated allocations.
// 7. Ensure there are no logical errors and the execution paths flow as expected.
// 8. Refactor where necessary to maintain clean code, efficient code, and to conform to the above settings and directives.
// 9. NEVER BREAK FUNCTIONALITY THAT IS ALREADY WORKING.
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

    // Cache for tokenization results with hit ratio tracking
    mutable std::unordered_map<std::string, std::vector<llama_token>> token_cache;
    mutable std::list<std::string> token_cache_lru;  // Track access order for LRU eviction
    mutable std::unordered_map<std::string, std::list<std::string>::iterator> token_cache_lru_map;  // Fast lookup in LRU list
    mutable size_t max_cache_size = 1024; // Limit cache size
    mutable size_t cache_hits = 0;
    mutable size_t cache_requests = 0;
    
    // Pre-allocated working buffers to avoid repeated allocations
    mutable std::vector<char> token_buffer;
    mutable std::string temp_string_buffer;
    mutable std::vector<llama_token> temp_token_buffer;
    
    // Cache for formatted templates to avoid repeated template application
    mutable std::string last_template_key;
    mutable std::string last_template_result;

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
        
        // Clear caches when new model is loaded
        clear_caches();
        
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
        
        // Use cached tokenization for generation prompt
        if (!generation_prompt.empty()) {
            std::vector<llama_token> prompt_tokens = tokenize_cached(generation_prompt, false);
            
            if (!prompt_tokens.empty()) {
                if (!process_tokens_batch(prompt_tokens, true)) {
                    return "Error: Failed to process generation prompt";
                }
                total_prompt_tokens += prompt_tokens.size();
            }
        }

        // Pre-allocate response string for better performance
        std::string response;
        const int32_t max_new_tokens = std::min(n_predict, n_ctx - n_past - 16);
        response.reserve(max_new_tokens * 4);
        
        int32_t n_generated = 0;

        if (max_new_tokens <= 0) {
            return "Error: No space left in context for generation";
        }

        auto decode_start = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        // Main generation loop with cached token conversion
        while (n_generated < max_new_tokens) {
            // Sample next token (like batched.cpp)
            llama_token new_token = llama_sampler_sample(sampler, context, -1);
            
            if (new_token < 0 || llama_vocab_is_eog(vocab, new_token)) {
                break;
            }

            // Use cached token-to-text conversion
            std::string token_text = token_to_text_cached(new_token);
            if (token_text.empty()) {
                break;
            }
            
            response += token_text;

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

        std::vector<llama_token> tokens = tokenize_cached(full_content, true);
        
        if (tokens.empty()) {
            return false;
        }

        // Check context capacity with safety margin
        const int32_t safety_margin = 128;
        if (static_cast<int32_t>(tokens.size()) > n_ctx - safety_margin) {
            std::cerr << "Warning: Context too large (" << tokens.size() 
                     << " tokens > " << (n_ctx - safety_margin) << " limit)" << std::endl;
            return false;
        }

        n_past = 0;
        
        // Use optimal batch size
        const int32_t optimal_batch = calculate_optimal_batch_size();
        std::vector<llama_seq_id> seq_ids = {0};
        
        // Process in optimally-sized chunks
        for (size_t start = 0; start < tokens.size(); start += optimal_batch) {
            clear_batch();
            
            size_t end = std::min(start + optimal_batch, tokens.size());
            
            // Batch process tokens
            for (size_t i = start; i < end; ++i) {
                bool output_logits = (i == tokens.size() - 1);
                common_batch_add(batch, tokens[i], static_cast<int32_t>(i), seq_ids, output_logits);
            }
            
            if (batch.n_tokens > 0) {
                if (llama_decode(context, batch) != 0) {
                    std::cerr << "Error: Failed to decode context batch at position " << start << std::endl;
                    return false;
                }
            }
        }
        
        n_past = static_cast<int32_t>(tokens.size());
        prev_len = static_cast<int32_t>(full_content.length());
        
        std::cout << "Successfully processed " << tokens.size() 
                 << " tokens in full context rebuild (batch size: " << optimal_batch << ")" << std::endl;
        return true;
    }

    // Optimized batch initialization with calculated size
    bool initialize_batch() {
        if (batch_initialized) {
            return true;
        }

        int32_t batch_size = calculate_optimal_batch_size();
        
        batch = llama_batch_init(batch_size, 0, 1);
        if (batch.token == nullptr) {
            std::cerr << "Error: Failed to initialize batch with size " << batch_size << std::endl;
            return false;
        }

        batch_initialized = true;
        std::cout << "Initialized batch with optimal size: " << batch_size << std::endl;
        return true;
    }

    // Enhanced cleanup with memory optimization
    void cleanup() {
        clear_caches();
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
        
        // Efficient memory cleanup
        std::vector<std::pair<std::string, std::string>>().swap(message_history);
        std::vector<char>().swap(formatted_buffer);
        std::string().swap(template_buffer);
        std::vector<llama_chat_message>().swap(message_cache);
        std::string().swap(custom_chat_template);
        
        message_cache_dirty = true;
        reset_cache_stats();
    }

    // Memory usage reporting
    struct MemoryStats {
        size_t token_cache_bytes;
        size_t message_history_bytes;
        size_t buffer_bytes;
        size_t total_estimated_bytes;
    };
    
    MemoryStats get_memory_stats() const {
        size_t token_cache_bytes = 0;
        for (const auto& [key, tokens] : token_cache) {
            token_cache_bytes += key.size() + (tokens.size() * sizeof(llama_token));
        }
        
        size_t message_history_bytes = 0;
        for (const auto& [role, content] : message_history) {
            message_history_bytes += role.size() + content.size();
        }
        
        size_t buffer_bytes = token_buffer.capacity() + temp_string_buffer.capacity() + 
                             template_buffer.capacity() + formatted_buffer.capacity();
        
        return {
            token_cache_bytes,
            message_history_bytes,
            buffer_bytes,
            token_cache_bytes + message_history_bytes + buffer_bytes
        };
    }

private:
    // Optimized LRU cache management
    void add_to_token_cache(const std::string& key, const std::vector<llama_token>& tokens) const {
        // Pre-emptive cleanup if approaching limit
        if (token_cache.size() >= max_cache_size * 0.9f) {
            trim_token_cache();
        }
        
        // Check if key already exists (update case)
        auto existing = token_cache.find(key);
        if (existing != token_cache.end()) {
            existing->second = tokens;
            update_lru_access(key);
            return;
        }
        
        // Add new entry
        token_cache[key] = tokens;
        token_cache_lru.push_front(key);
        token_cache_lru_map[key] = token_cache_lru.begin();
    }
    
    // More aggressive cache trimming for better memory management
    void trim_token_cache() const {
        // Remove 25% of entries when trimming to reduce frequency
        size_t target_size = static_cast<size_t>(max_cache_size * 0.75f);
        
        while (token_cache.size() > target_size && !token_cache_lru.empty()) {
            std::string lru_key = token_cache_lru.back();
            token_cache_lru.pop_back();
            token_cache_lru_map.erase(lru_key);
            token_cache.erase(lru_key);
        }
    }

    // Add missing helper methods
    std::vector<llama_token> tokenize_cached(const std::string& text, bool add_special) const {
        cache_requests++;
        
        std::string cache_key = text + (add_special ? ":special" : ":normal");
        
        auto it = token_cache.find(cache_key);
        if (it != token_cache.end()) {
            cache_hits++;
            update_lru_access(cache_key);
            return it->second;
        }
        
        // Tokenize new text
        const int32_t n_tokens = -llama_tokenize(vocab, text.c_str(), text.size(), NULL, 0, add_special, true);
        if (n_tokens <= 0) {
            return {};
        }
        
        std::vector<llama_token> tokens(n_tokens);
        if (llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_special, true) < 0) {
            return {};
        }
        
        add_to_token_cache(cache_key, tokens);
        return tokens;
    }
    
    std::string token_to_text_cached(llama_token token) const {
        // Simple caching for token-to-text conversion
        static std::unordered_map<llama_token, std::string> text_cache;
        
        auto it = text_cache.find(token);
        if (it != text_cache.end()) {
            return it->second;
        }
        
        // Convert token to text
        temp_string_buffer.clear();
        temp_string_buffer.resize(32);  // Most tokens are small
        
        int32_t result = llama_token_to_piece(vocab, token, temp_string_buffer.data(), temp_string_buffer.size(), 0, true);
        if (result < 0) {
            temp_string_buffer.resize(-result);
            result = llama_token_to_piece(vocab, token, temp_string_buffer.data(), temp_string_buffer.size(), 0, true);
        }
        
        if (result > 0) {
            std::string text(temp_string_buffer.data(), result);
            text_cache[token] = text;
            return text;
        }
        
        return "";
    }
    
    bool apply_chat_template(const char* tmpl, bool add_generation_prompt, std::string& result) const {
        if (!tmpl) return false;
        
        // Create cache key for template application
        std::string cache_key = std::string(tmpl) + ":" + (add_generation_prompt ? "gen" : "chat");
        for (const auto& msg : message_history) {
            cache_key += "|" + msg.first + ":" + msg.second.substr(0, 50); // First 50 chars for cache key
        }
        
        // Check if we have cached result
        if (cache_key == last_template_key && !last_template_result.empty()) {
            result = last_template_result;
            return true;
        }
        
        // Convert message history to llama format
        if (message_cache_dirty) {
            message_cache = convert_to_llama_messages();
            message_cache_dirty = false;
        }
        
        // Apply template
        template_buffer.clear();
        template_buffer.resize(n_ctx * 4); // Estimate buffer size
        
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
            last_template_key = cache_key;
            last_template_result = result;
            return true;
        }
        
        return false;
    }
    
    int32_t calculate_optimal_batch_size() const {
        if (!context) return 512;
        
        int32_t n_batch = llama_n_batch(context);
        int32_t available_ctx = n_ctx - n_past;
        
        // Use smaller of configured batch size or available context
        return std::min({n_batch, available_ctx, 512});
    }
    
    void cleanup_batch() {
        if (batch_initialized) {
            llama_batch_free(batch);
            batch_initialized = false;
        }
    }
    
    void clear_caches() const {
        token_cache.clear();
        token_cache_lru.clear();
        token_cache_lru_map.clear();
        last_template_key.clear();
        last_template_result.clear();
        cache_hits = 0;
        cache_requests = 0;
    }
    
    void reset_cache_stats() const {
        cache_hits = 0;
        cache_requests = 0;
    }

    // LRU cache management helpers
    void update_lru_access(const std::string& key) const {
        auto lru_it = token_cache_lru_map.find(key);
        if (lru_it != token_cache_lru_map.end()) {
            token_cache_lru.splice(token_cache_lru.begin(), token_cache_lru, lru_it->second);
        }
    }
};

// Progress callback function declaration (needs to be outside class for C compatibility)
extern bool model_loading_progress_callback(float progress, void *user_data);
