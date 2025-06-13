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

// Forward declare the progress callback function
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
    
    // Sampler defaults
    constexpr float DEFAULT_TEMPERATURE = 0.8f;
    constexpr float DEFAULT_MIN_P = 0.05f;
    constexpr float DEFAULT_TOP_P = 0.9f;
    constexpr int32_t DEFAULT_TOP_K = 40;
      // Context management
    constexpr float MAX_CONTEXT_USAGE = 0.90f;
    constexpr float TARGET_CONTEXT_USAGE = 0.60f;
      // String processing constants
    constexpr size_t MAX_TEXT_PREVIEW_LENGTH = 50;
    constexpr size_t STRING_RESERVE_MULTIPLIER = 4;
    constexpr int32_t MAX_RETRY_ATTEMPTS = 2;
    constexpr size_t MAX_MESSAGE_HISTORY_SIZE = 1000;
    constexpr float AGGRESSIVE_PRUNING_RATIO = 0.3f;
    constexpr size_t SUMMARY_CONTENT_RESERVE_SIZE = 4096;
}

// Thread Safety Contract (Directive #12):
// This class is NOT thread-safe. External synchronization is required for concurrent access.
// - All public methods must be called from a single thread or protected by external mutexes
// - The llama.cpp backend itself has thread-safety limitations that require careful handling
// - Model loading/unloading operations are particularly sensitive to race conditions
// - Context switching operations modify shared state and must be serialized
class LlamaManager {
private:
    // Model information container
    struct ModelInfo {
        llama_model* model;
        const llama_vocab* vocab;
        llama_sampler* sampler;
        int32_t n_ctx;
        int32_t n_gpu_layers;
        int32_t n_predict;
        std::string model_path;
        std::string custom_chat_template;        bool model_loaded;
        
        ModelInfo() : model(nullptr), vocab(nullptr), sampler(nullptr), n_ctx(LlamaConstants::DEFAULT_CONTEXT_SIZE), n_gpu_layers(LlamaConstants::DEFAULT_GPU_LAYERS), 
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
            return nullptr;        }
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
        static constexpr size_t MAX_SUMMARY_SLOTS = 5;
        
        ContextInfo() : context(nullptr), batch{}, batch_initialized(false), 
                       n_past(0), prev_len(0), model_info(nullptr) {
            summary_slots.reserve(MAX_SUMMARY_SLOTS);
        }
    };
    
    std::unordered_map<std::string, std::unique_ptr<ModelInfo>> models;
    std::unordered_map<std::string, std::unique_ptr<ContextInfo>> contexts;
    std::string active_context_id;
    ContextInfo* current_context;
      // Legacy compatibility - only keep model_loaded flag
    bool model_loaded;
    
    // Template and cache management
    mutable std::string template_buffer;
    mutable TokenCache token_cache;
    
    // Working buffers
    mutable std::string temp_string_buffer;

    // Summary logging callbacks
    std::function<void(const std::string&)> summary_input_callback;
    std::function<void(const std::string&)> summary_output_callback;

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
                prune_message_history(LlamaConstants::TARGET_CONTEXT_USAGE);
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
            result = std::string(template_buffer.data(), result_len);
            return true;
        }
        return false;
    }

    // FIXED: Enhanced token-to-text conversion - Updated to use context's model
    std::string convert_token_to_text(llama_token token) const {
        ModelInfo* model_info = get_current_model_info();
        if (!model_info || !model_info->vocab) {
            LLAMA_LOG("Error: Vocabulary not available from current context's model");
            return "";
        }
        
        // FIXED: Simplified token validation - just check for negative values
        if (token < 0) {
            LLAMA_LOG("Warning: Token " + std::to_string(token) + " is negative");
            return "";
        }
          temp_string_buffer.resize(LlamaConstants::INITIAL_TOKEN_BUFFER_SIZE);
        int32_t result = llama_token_to_piece(model_info->vocab, token, temp_string_buffer.data(), temp_string_buffer.size(), 0, true);
        if (result < 0) {
            size_t required_size = static_cast<size_t>(-result);
            if (required_size > LlamaConstants::MAX_TOKEN_BUFFER_SIZE) {
                LLAMA_LOG("Error: Token conversion requires excessive buffer size: " + std::to_string(required_size));
                return "";
            }
            temp_string_buffer.resize(required_size);
            result = llama_token_to_piece(model_info->vocab, token, temp_string_buffer.data(), temp_string_buffer.size(), 0, true);
        }
          return (result > 0) ? std::string(temp_string_buffer.data(), result) : "";
    }
      // Add summary to the 5-slot chronological summary system
    void add_summary_to_slots(const std::string& new_summary) {
        if (!current_context || new_summary.empty()) return;
        
        // If we're at capacity (5 slots), implement rollover summarization
        if (current_context->summary_slots.size() >= ContextInfo::MAX_SUMMARY_SLOTS) {
            LLAMA_LOG("Summary slots at capacity, performing rollover summarization");
            
            // Get the two oldest summaries (slots 0 and 1)
            std::string oldest_summary = current_context->summary_slots[0];
            std::string second_oldest_summary = current_context->summary_slots[1];
            
            // Create a combined summary from the two oldest
            std::vector<std::pair<std::string, std::string>> rollover_messages;
            rollover_messages.emplace_back("system", "Previous summary 1: " + oldest_summary);
            rollover_messages.emplace_back("system", "Previous summary 2: " + second_oldest_summary);
            
            std::string combined_summary = summarize_messages(rollover_messages);
            
            if (!combined_summary.empty()) {
                // Remove the two oldest summaries and replace with the combined one
                current_context->summary_slots.erase(current_context->summary_slots.begin(), current_context->summary_slots.begin() + 2);
                current_context->summary_slots.insert(current_context->summary_slots.begin(), combined_summary);
                LLAMA_LOG("Rollover summarization successful - combined 2 oldest summaries into 1");
            } else {
                // Fallback: just remove the oldest if rollover summarization fails
                current_context->summary_slots.erase(current_context->summary_slots.begin());
                LLAMA_LOG("Rollover summarization failed, removed oldest summary");
            }
        }
        
        // Add new summary to the end (newest position)
        current_context->summary_slots.push_back(new_summary);
        
        LLAMA_LOG("Added new summary to slot " + std::to_string(current_context->summary_slots.size()) + 
                  " of " + std::to_string(ContextInfo::MAX_SUMMARY_SLOTS));
    }
    
    // Enhanced prune message history using summary model to condense pruned messages
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
        
        // Extract messages to be pruned (everything except system message and messages to keep)
        size_t prune_start_idx = system_offset;
        size_t prune_end_idx = current_context->message_history.size() - messages_to_keep;
        
        if (prune_end_idx <= prune_start_idx) {
            return; // Nothing to prune
        }
        
        // Collect messages to be summarized
        std::vector<std::pair<std::string, std::string>> messages_to_summarize;
        for (size_t i = prune_start_idx; i < prune_end_idx; ++i) {
            messages_to_summarize.emplace_back(current_context->message_history[i]);
        }        // Attempt to summarize using summary context
        std::string summary = summarize_messages(messages_to_summarize);
        
        // Calculate how many messages were summarized for logging
        size_t summarized_count = prune_end_idx - prune_start_idx;
          // Build new message history
        std::vector<std::pair<std::string, std::string>> new_history;
        new_history.reserve(system_offset + ContextInfo::MAX_SUMMARY_SLOTS + messages_to_keep);
        
        // Add system message if present
        if (has_system) {
            new_history.emplace_back(std::move(current_context->message_history[0]));
        }        // Handle 5-slot summary system
        if (!summary.empty()) {
            // Add new summary to the slot system
            add_summary_to_slots(summary);
            LLAMA_LOG("Successfully created summary for " + std::to_string(summarized_count) + " pruned messages");
            LLAMA_LOG("Summary slots now contain " + std::to_string(current_context->summary_slots.size()) + " summaries");
        } else {
            // Fallback: keep more messages if summarization failed
            LLAMA_LOG("Warning: Summarization failed or unavailable, keeping more messages instead");
            
            // Keep up to half of the messages that would have been pruned
            size_t fallback_keep = std::min(summarized_count / 2, size_t(3)); // Keep at least 1, max 3
            size_t fallback_start = prune_end_idx - fallback_keep;
            
            for (size_t i = fallback_start; i < prune_end_idx; ++i) {
                if (i < current_context->message_history.size()) {
                    new_history.emplace_back(std::move(current_context->message_history[i]));
                }
            }
            
            // Add a system message explaining what happened
            new_history.emplace_back("system", "[Note: " + std::to_string(summarized_count - fallback_keep) + 
                                    " older messages removed due to context limits]");
        }
        
        // FIXED: Do NOT add summary slots to message history during regular pruning
        // Summary slots are maintained separately and only used for rollover summarization
        // The actual summaries are not part of the conversation context
        
        // Add the recent messages to keep
        for (size_t i = prune_end_idx; i < current_context->message_history.size(); ++i) {
            new_history.emplace_back(std::move(current_context->message_history[i]));
        }
          current_context->message_history = std::move(new_history);
        current_context->message_cache_dirty = true;
        
        LLAMA_LOG("Pruned " + std::to_string(summarized_count) + " messages into summary. " +
                  "New history has " + std::to_string(current_context->message_history.size()) + " messages");
    }

    // Summarize a collection of messages using the summary context
    std::string summarize_messages(const std::vector<std::pair<std::string, std::string>>& messages_to_summarize) {
        if (messages_to_summarize.empty()) {
            return "";
        }        // Check if we have a summary context available
        auto summary_context_it = contexts.find("summary_context");
        if (summary_context_it == contexts.end()) {
            LLAMA_LOG("Warning: No summary context available for message summarization");
            
            // Log to summaries tab that no context is available
            if (summary_output_callback) {
                summary_output_callback("ERROR: No summary context available");
            }
            
            return "";
        }
        
        // Store current context to restore later
        std::string original_context_id = active_context_id;
        ContextInfo* original_context = current_context;        // Switch to summary context temporarily
        if (!switch_to_context("summary_context")) {
            LLAMA_LOG("Error: Failed to switch to summary context");
            
            // Log to summaries tab that context switch failed
            if (summary_output_callback) {
                summary_output_callback("ERROR: Failed to switch to summary context");
            }
            
            return "";
        }
          // Build the content to summarize
        std::string content_to_summarize;
        content_to_summarize.reserve(LlamaConstants::SUMMARY_CONTENT_RESERVE_SIZE); // Reserve reasonable space
        
        for (const auto& [role, content] : messages_to_summarize) {
            content_to_summarize += role + ": " + content + "\n\n";
        }
          // Create summarization request
        std::string summarization_request = "Please provide a concise summary of the following conversation:\n\n" + 
                                          content_to_summarize + 
                                          "\nSummary:";        // Log the input to the summaries tab
        if (summary_input_callback) {
            // Show full content for better debugging and visibility
            std::string ui_input = "Summarizing " + std::to_string(messages_to_summarize.size()) + " messages:\n";
            ui_input += std::string(50, '=') + "\n";
            
            for (const auto& [role, content] : messages_to_summarize) {
                ui_input += role + ": " + content + "\n\n";
            }
            ui_input += std::string(50, '=') + "\n";
            summary_input_callback(ui_input);
        }// Generate summary using the summary context
        std::string summary;
        try {
            summary = generate_response(summarization_request, "user");
            
            // Check if the summary is actually an error message
            if (!summary.empty() && summary.starts_with("Error:")) {
                LLAMA_LOG("Summary generation returned error: " + summary);
                
                // Log the error to the summaries tab
                if (summary_output_callback) {
                    summary_output_callback(summary);
                }
                
                summary = ""; // Treat as failed summarization
            } else if (!summary.empty()) {                // Clean up the summary (remove any extra whitespace, newlines)
                size_t start = summary.find_first_not_of(" \t\n\r");
                size_t end = summary.find_last_not_of(" \t\n\r");
                if (start != std::string::npos && end != std::string::npos) {
                    summary = summary.substr(start, end - start + 1);
                }
                
                // Note: No artificial length limit - let the AI determine appropriate summary length
                  // Log the output to the summaries tab
                if (summary_output_callback) {
                    summary_output_callback(summary);
                }
            }} catch (const std::exception& e) {
            LLAMA_LOG("Exception during message summarization: " + std::string(e.what()));
            summary = "";
            
            // Log the error to the summaries tab
            if (summary_output_callback) {
                summary_output_callback("ERROR: " + std::string(e.what()));
            }
        } catch (...) {
            LLAMA_LOG("Unknown exception during message summarization");
            summary = "";
            
            // Log the error to the summaries tab
            if (summary_output_callback) {
                summary_output_callback("ERROR: Unknown exception during summarization");
            }
        }
        
        // Restore original context
        if (!original_context_id.empty() && original_context) {
            switch_to_context(original_context_id);
        }
        
        if (summary.empty()) {
            LLAMA_LOG("Warning: Message summarization produced empty result");
        } else {
            LLAMA_LOG("Successfully summarized " + std::to_string(messages_to_summarize.size()) + 
                      " messages into " + std::to_string(summary.length()) + " character summary");
        }
        
        return summary;
    }

    // REFACTOR: Update context validation
    bool validate_and_recover_sampler() {
        ModelInfo* model_info = get_current_model_info();
        if (!model_info) {
            LLAMA_LOG("Error: No model info available for sampler validation");
            return false;
        }
        
        if (model_info->sampler) {
            return true; // Sampler is valid
        }
        
        LLAMA_LOG("WARNING: Sampler is NULL, attempting recovery...");
        
        // Attempt to recreate a basic greedy sampler
        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        model_info->sampler = llama_sampler_chain_init(sparams);
        
        if (!model_info->sampler) {
            LLAMA_LOG("CRITICAL: Failed to recover sampler!");
            return false;
        }
        
        // Add basic greedy sampling
        llama_sampler_chain_add(model_info->sampler, llama_sampler_init_greedy());
        
        if (!model_info->sampler) {
            LLAMA_LOG("CRITICAL: Sampler became NULL after adding greedy sampler during recovery!");
            return false;
        }
        
        LLAMA_LOG("Sampler recovered successfully with greedy sampling");
        return true;
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
        }
          // Check for extremely long message history that might cause issues
        if (current_context->message_history.size() > LlamaConstants::MAX_MESSAGE_HISTORY_SIZE) {
            LLAMA_LOG("Warning: Very large message history (" + 
                      std::to_string(current_context->message_history.size()) + " messages)");
            
            // Trigger aggressive pruning
            prune_message_history(LlamaConstants::AGGRESSIVE_PRUNING_RATIO); // Keep only 30%
            return false; // Indicate that recovery was needed
        }
        
        return true; // State is valid
    }    // Helper to get current model info
    ModelInfo* get_current_model_info() const noexcept {
        return current_context ? current_context->model_info : nullptr;
    }

public:    LlamaManager() : current_context(nullptr), model_loaded(false), token_cache(LlamaConstants::DEFAULT_TOKEN_CACHE_SIZE) {}

    ~LlamaManager() noexcept {
        cleanup();
    }

    // Initialize llama.cpp backend
    bool initialize() {
        ggml_backend_load_all();
        return true;
    }

    // Set callbacks for summary logging
    void set_summary_input_callback(std::function<void(const std::string&)> callback) {
        summary_input_callback = std::move(callback);
    }
    
    void set_summary_output_callback(std::function<void(const std::string&)> callback) {
        summary_output_callback = std::move(callback);
    }

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
        
        // Update legacy flags for compatibility
        model_loaded = true;
        
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
        
        active_context_id = context_id;
        current_context = it->second.get();
        LLAMA_LOG("Switched to context '" + context_id + "'");
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
        
        LLAMA_LOG("Cleared conversation history and summary slots");
    }

    // Manually trigger message history pruning with summarization
    bool prune_conversation_with_summary(float keep_ratio = 0.6f) {
        if (!current_context || current_context->message_history.empty()) {
            LLAMA_LOG("Warning: No active context or empty message history for pruning");
            return false;
        }
        
        size_t original_message_count = current_context->message_history.size();
        
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

    // Check if summarization is available (summary context exists)
    bool is_summarization_available() const {
        return contexts.find("summary_context") != contexts.end();
    }

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
        return messages;
    }
    
    // Get information about the current summary slots
    struct SummarySlotInfo {
        size_t total_slots;
        size_t used_slots;
        std::vector<std::string> summaries; // In chronological order (oldest to newest)
    };
    
    SummarySlotInfo get_summary_slot_info() const {
        SummarySlotInfo info;
        info.total_slots = ContextInfo::MAX_SUMMARY_SLOTS;
        
        if (current_context) {
            info.used_slots = current_context->summary_slots.size();
            info.summaries = current_context->summary_slots;
        } else {
            info.used_slots = 0;
        }
        
        return info;
    }

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
    }

    // Enhanced context update with better tokenization handling - FIXED recursion issue
    bool update_context_with_pruning() {
        ModelInfo* model_info = get_current_model_info();
        if (!model_loaded || !model_info || !model_info->model || !current_context || !current_context->context || !model_info->vocab) {
            LLAMA_LOG("Error: Model components not initialized");
            return false;
        }

        // Check and handle pruning first
        int32_t n_ctx_used = current_context->n_past;
        int32_t max_threshold = static_cast<int32_t>(model_info->n_ctx * 0.9f);
        
        bool context_pruned = false;
        if (n_ctx_used > max_threshold) {            LLAMA_LOG("Context usage at " + std::to_string((float)n_ctx_used / model_info->n_ctx * 100.0f) + 
                       "%, pruning to 60%");
            
            llama_kv_self_clear(current_context->context);
            current_context->n_past = 0;
            current_context->prev_len = 0;
            prune_message_history(LlamaConstants::TARGET_CONTEXT_USAGE);
            context_pruned = true;
        }

        // Apply template and process
        std::string formatted_content;
        if (!apply_template_optimized(false, formatted_content)) {
            LLAMA_LOG("Error: Failed to apply chat template");
            return false;
        }

        int32_t new_len = static_cast<int32_t>(formatted_content.length());        // FIXED: If context was pruned, prev_len is 0 with messages (reset context), or prev_len > new_len, rebuild from scratch
        if (context_pruned || current_context->prev_len > new_len || 
            (current_context->prev_len == 0 && !current_context->message_history.empty())) {
            // Rebuild context - use add_special=true for full context
            std::vector<llama_token> tokens = process_text_to_tokens(formatted_content, true);
            if (tokens.empty()) {
                LLAMA_LOG("Warning: Template produced no tokens for full rebuild");
                current_context->prev_len = new_len;
                return true;
            }
            
            if (process_context_tokens(tokens, false)) {
                current_context->prev_len = new_len;
                LLAMA_LOG("Context rebuilt successfully with " + 
                           std::to_string(tokens.size()) + " tokens (reset_context=" + 
                           std::string((current_context->prev_len == 0) ? "true" : "false") + ")");
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
    }    // FIXED: Enhanced generation with sampler validation and recovery
    std::string generate_response(const std::string& input, const std::string& username = "Schwi") {
        ModelInfo* model_info = get_current_model_info();
        if (!model_loaded || !model_info || !model_info->model || !current_context || !current_context->context || !model_info->vocab || !current_context->batch_initialized) {
            return "Error: Model components not properly initialized or no active context";
        }

        if (input.empty()) return "Error: Empty input";        // Validate conversation state before proceeding
        if (!validate_conversation_state()) {
            LLAMA_LOG("Conversation state required recovery, retrying...");
        }

        // FIXED: Validate sampler before proceeding and attempt recovery if needed
        if (!validate_and_recover_sampler()) {
            return "Error: Sampler validation/recovery failed";
        }        // Setup conversation - ensure system message is in history if context is empty
        if (current_context->message_history.empty() && !current_context->system_message.empty()) {
            current_context->message_history.emplace_back("system", current_context->system_message);
            current_context->message_cache_dirty = true;
        }

        current_context->message_history.emplace_back(username, input);
        current_context->message_cache_dirty = true;        // Update context using unified function - with retry logic
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
                
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
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
        }        // Prepare for generation using unified template function
        std::string generation_content;
        if (!apply_template_optimized(true, generation_content)) {
            return "Error: Failed to apply generation template";
        }

        // FIXED: Handle generation prompt correctly for reset contexts
        // For contexts with reset_after_generation (like summary contexts), prev_len starts at 0
        // so we need to process the incremental content only if there's actually new content
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

        // Generate response using unified functions
        std::string response;
        const int32_t safety_margin = LlamaConstants::TOKEN_SAFETY_MARGIN; // Reserve space for potential special tokens
        const int32_t max_new_tokens = std::min(model_info->n_predict, model_info->n_ctx - current_context->n_past - safety_margin);
        response.reserve(max_new_tokens * LlamaConstants::STRING_RESERVE_MULTIPLIER);
        
        if (max_new_tokens <= 0) {
            return "Error: No space left in context for generation (context: " + 
                   std::to_string(current_context->n_past) + "/" + std::to_string(model_info->n_ctx) + ")";
        }

        // FIXED: Enhanced validation before generation loop
        if (!model_info->sampler) {
            LLAMA_LOG("CRITICAL: Sampler is null before generation loop after validation!");
            return "Error: Sampler validation failed";
        }
        
        if (!current_context->context) {
            LLAMA_LOG("Error: Context is null before generation");
            return "Error: Context not properly initialized";
        }
          // FIXED: Validate that we have logits available for sampling
        // For reset contexts, we might start with n_past=0 if the context was just created/reset
        if (current_context->n_past == 0 && current_context->message_history.empty()) {
            LLAMA_LOG("Error: Context is completely empty, cannot generate");
            return "Error: Context is empty, cannot generate response";
        }

        LLAMA_LOG("Starting generation with " + std::to_string(max_new_tokens) + " max tokens, n_past=" + std::to_string(current_context->n_past));

        auto decode_start = std::chrono::high_resolution_clock::now();
        int32_t n_generated = 0;
        std::vector<llama_seq_id> seq_ids = {0};

        // Enhanced generation loop with model-specific vocab checking
        while (n_generated < max_new_tokens) {
            // FIXED: Validate sampler on every iteration to catch when it becomes null
            if (!model_info->sampler) {
                LLAMA_LOG("CRITICAL: Sampler became null during generation at token " + std::to_string(n_generated));
                return "Error: Sampler failed during generation";
            }
            
            if (!current_context->context) {
                LLAMA_LOG("CRITICAL: Context became null during generation");
                return "Error: Context lost during generation";
            }
            
            // FIXED: Validate that n_past is within reasonable bounds
            if (current_context->n_past <= 0 || current_context->n_past >= model_info->n_ctx) {
                LLAMA_LOG("Error: Invalid context position during generation: " + std::to_string(current_context->n_past));
                return "Error: Context position invalid";
            }
            
            llama_token new_token;
            try {
                new_token = llama_sampler_sample(model_info->sampler, current_context->context, -1);
            } catch (const std::exception& e) {
                LLAMA_LOG("Exception during token sampling: " + std::string(e.what()));
                return "Error: Exception during token generation";
            } catch (...) {
                LLAMA_LOG("Unknown exception during token sampling");
                return "Error: Unknown exception during generation";
            }
            
            if (new_token < 0) {
                LLAMA_LOG("Error: Invalid token generated: " + std::to_string(new_token));
                break;
            }
            
            if (llama_vocab_is_eog(model_info->vocab, new_token)) {
                LLAMA_LOG("End of generation token encountered");
                break;
            }

            // Convert token to text using unified function
            std::string token_text = convert_token_to_text(new_token);
            if (token_text.empty()) {
                LLAMA_LOG("Warning: Empty token text for token " + std::to_string(new_token));
                continue;
            }
            
            response += token_text;

            // FIXED: Validate batch state before adding token
            if (!current_context->batch_initialized) {
                LLAMA_LOG("Error: Batch not initialized during generation");
                return "Error: Batch system failed";
            }

            // Process token using unified batch function
            if (!add_single_token_to_batch(new_token, current_context->n_past, seq_ids, true)) {
                LLAMA_LOG("Error: Failed to add token to batch during generation");
                return "Error: Token processing failed";
            }
            
            if (current_context->batch.n_tokens > 0 && llama_decode(current_context->context, current_context->batch) != 0) {
                LLAMA_LOG("Error: Failed to decode during generation at token " + std::to_string(n_generated));
                return "Error: Token decode failed";
            }
            
            current_context->n_past++;
            n_generated++;
            
            // FIXED: Periodic sampler validation during long generation
            if (n_generated % 10 == 0 && !model_info->sampler) {
                LLAMA_LOG("CRITICAL: Sampler became null during long generation at token " + std::to_string(n_generated));
                return "Error: Sampler failed during long generation";
            }
        }

        // Update timing and history
        auto decode_end = std::chrono::high_resolution_clock::now();
        current_context->last_decode_time_us = std::chrono::duration_cast<std::chrono::microseconds>(decode_end - decode_start).count();
        current_context->total_generation_tokens += n_generated;        if (!response.empty()) {
            current_context->message_history.emplace_back("assistant", response);
            current_context->message_cache_dirty = true;
            
            std::string updated_content;
            if (apply_template_optimized(false, updated_content)) {
                current_context->prev_len = static_cast<int32_t>(updated_content.length());
            }
        }        if (n_generated > 0) {
            float tokens_per_second = (float)n_generated / ((float)current_context->last_decode_time_us / 1000000.0f);
            LLAMA_LOG("Generated " + std::to_string(n_generated) + " tokens in " + 
                       std::to_string(current_context->last_decode_time_us / 1000.0f) + "ms (" + 
                       std::to_string(tokens_per_second) + " t/s)");
        }        
        
        // FIXED: Store the response before clearing context if reset_after_generation is set
        std::string final_response = response;
          // FIXED: Reset context after generation if flag is set (for summary contexts)
        if (current_context->reset_after_generation) {
            LLAMA_LOG("Resetting context '" + active_context_id + "' after generation (reset_after_generation flag is set)");
            
            // For summary contexts, preserve the system message but clear everything else
            std::string saved_system_message = current_context->system_message;
            clear_conversation();
            
            // Restore the system message for the next summarization task
            if (!saved_system_message.empty()) {
                current_context->system_message = saved_system_message;
                current_context->message_history.emplace_back("system", saved_system_message);
                current_context->message_cache_dirty = true;
            }
        }
        // FIXED: Final sampler validation
        if (!model_info->sampler) {
            LLAMA_LOG("WARNING: Sampler is NULL at end of generation!");
        }

        return final_response;
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
        
        model_loaded = false;
        
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
        token_cache.clear();
    }
};

// Progress callback function declaration (needs to be outside class for C compatibility)
extern bool model_loading_progress_callback(float progress, void *user_data);

//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODING DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//