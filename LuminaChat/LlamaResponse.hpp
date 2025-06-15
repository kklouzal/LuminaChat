// LlamaResponse.hpp - header-only implementation for AI response generation
// 
// RESPONSIBILITY: Pure response generation functionality
// - Token-to-text conversion for generated output
// - Response generation loop and token sampling
// - Generation state validation and recovery
// - Sampler validation and recovery
// 
// DEPENDENCIES: Requires ModelInfo and ContextInfo from LlamaManager
// INTEGRATION: Called by LlamaManager through template callbacks for batch operations
//
// Provides response generation functionality extracted from LlamaManager for better modularity.
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
#include <chrono>
#include <thread>
#include <algorithm>
#include "llama-cpp.h"
#include "LogHandler.hpp"

// Forward declaration of shared constants from LlamaManager
// Note: These are defined in LlamaManager.hpp and should not be redefined here
// This ensures single source of truth for configuration constants

// Constants for response generation (Directive #5: Zero Magic & Strong Typing)
namespace ResponseConstants {
    constexpr int32_t SAMPLER_VALIDATION_INTERVAL = 10;
    constexpr int32_t INITIAL_BUFFER_SIZE = 32;
    constexpr int32_t MAX_BUFFER_SIZE = 1024;
    constexpr int32_t DUMMY_TOKEN_DECODE_POS_OFFSET = 1;
}

// Forward declarations and type definitions for LlamaManager integration
// These structures are defined in LlamaManager.hpp to maintain proper ownership
// while enabling tight integration with LlamaResponse functionality
struct ModelInfo;
struct ContextInfo;

// Forward declaration for summary constants
namespace SummarizerConstants {
    extern const size_t MAX_SUMMARY_SLOTS;
}

// Thread Safety Contract (Directive #12):
// This class is NOT thread-safe and should only be accessed from within LlamaManager's context.
// All methods assume valid ModelInfo and ContextInfo structures are provided.
//
// USAGE PATTERN:
// 1. LlamaManager prepares context and provides callback functions for batch operations
// 2. LlamaResponse handles the generation loop, token sampling, and text conversion
// 3. LlamaManager receives tokens through callbacks to manage batch operations and context updates
class LlamaResponse {
private:
    // Working buffers for token conversion
    mutable std::vector<char> temp_string_buffer;

    // Enhanced sampler configuration with runtime validation
    static void configure_sampler(ModelInfo* model_info, float temperature = LlamaConstants::DEFAULT_TEMPERATURE, float min_p = LlamaConstants::DEFAULT_MIN_P, float top_p = LlamaConstants::DEFAULT_TOP_P, int32_t top_k = LlamaConstants::DEFAULT_TOP_K) {
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
        
        // Add sampling strategies
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
        
        // Validate final sampler state
        if (!model_info->sampler) {
            LLAMA_LOG("Error: Sampler became null during configuration");
        } else {
            LLAMA_LOG("Sampler configured successfully for model");
        }
    }

    // Validate and recover sampler if needed
    static bool validate_and_recover_sampler(ModelInfo* model_info) {
        if (!model_info) {
            LLAMA_LOG("Error: No model info available for sampler validation");
            return false;
        }
        
        if (model_info->sampler) {
            return true; // Sampler is valid
        }
        
        LLAMA_LOG("WARNING: Sampler is NULL, attempting recovery...");
        
        // Attempt to recreate the sampler
        configure_sampler(model_info, 
                          LlamaConstants::DEFAULT_TEMPERATURE, 
                          LlamaConstants::DEFAULT_MIN_P, 
                          LlamaConstants::DEFAULT_TOP_P, 
                          LlamaConstants::DEFAULT_TOP_K);
        
        if (!model_info->sampler) {
            LLAMA_LOG("CRITICAL: Failed to recover sampler!");
            return false;
        }
        
        LLAMA_LOG("Sampler recovered successfully");
        return true;
    }
    
    // Convert token to text using the context's model vocabulary
    std::string convert_token_to_text(llama_token token, ModelInfo* model_info) const {
        if (!model_info || !model_info->vocab) {
            LLAMA_LOG("Error: Vocabulary not available from current context's model");
            return "";
        }
        
        // Check for negative token values
        if (token < 0) {
            LLAMA_LOG("Warning: Token " + std::to_string(token) + " is negative");
            return "";
        }
        
        temp_string_buffer.resize(ResponseConstants::INITIAL_BUFFER_SIZE);
        int32_t result = llama_token_to_piece(model_info->vocab, token, temp_string_buffer.data(), 
                                             temp_string_buffer.size(), 0, true);
        
        if (result < 0) {
            size_t required_size = static_cast<size_t>(-result);
            if (required_size > ResponseConstants::MAX_BUFFER_SIZE) {
                LLAMA_LOG("Error: Token conversion requires excessive buffer size: " + std::to_string(required_size));
                return "";
            }
            temp_string_buffer.resize(required_size);
            result = llama_token_to_piece(model_info->vocab, token, temp_string_buffer.data(), 
                                         temp_string_buffer.size(), 0, true);
        }
        
        return (result > 0) ? std::string(temp_string_buffer.data(), result) : "";
    }

    // Attempt to recover logits by performing a minimal decode operation
    static bool recover_logits(ContextInfo* context_info, ModelInfo* model_info) {
        if (!context_info || !model_info || !context_info->batch_initialized || context_info->n_past <= 0) {
            return false;
        }
        
        LLAMA_LOG("Warning: No logits available, attempting recovery decode");
        
        // Clear batch and set up for logit generation
        context_info->batch.n_tokens = 0;
        
        // Add a dummy entry to get logits at current position
        if (context_info->batch.token && context_info->batch.pos && 
            context_info->batch.logits && context_info->batch.seq_id) {
              // Use EOS token as dummy for logit generation
            llama_token dummy_token = llama_vocab_eos(model_info->vocab);
            context_info->batch.token[0] = dummy_token;
            context_info->batch.pos[0] = context_info->n_past - ResponseConstants::DUMMY_TOKEN_DECODE_POS_OFFSET;
            context_info->batch.logits[0] = 1; // Request logits
            context_info->batch.seq_id[0] = 0;
            context_info->batch.n_tokens = 1;
            
            int decode_result = llama_decode(context_info->context, context_info->batch);
            if (decode_result == 0) {
                float* logits = llama_get_logits(context_info->context);
                if (logits) {
                    LLAMA_LOG("Successfully recovered logits with dummy decode");
                    return true;
                }
            }
        }
        
        LLAMA_LOG("Error: Could not recover logits for generation");
        return false;
    }

    // Validate context state before generation
    static bool validate_generation_state(ContextInfo* context_info, ModelInfo* model_info) {
        if (!context_info || !model_info || !context_info->context) {
            LLAMA_LOG("Error: Invalid context or model state");
            return false;
        }
        
        // Check context position bounds
        if (context_info->n_past <= 0 || context_info->n_past >= model_info->n_ctx) {
            LLAMA_LOG("Error: Invalid context position during generation: " + std::to_string(context_info->n_past));
            return false;
        }
        
        // Check for completely empty context
        if (context_info->n_past == 0 && context_info->message_history.empty()) {
            LLAMA_LOG("Error: Context is completely empty, cannot generate");
            return false;
        }
        
        // Validate batch initialization
        if (!context_info->batch_initialized) {
            LLAMA_LOG("Error: Batch not initialized for generation");
            return false;
        }
        
        return true;
    }

public:
    LlamaResponse() {
        temp_string_buffer.reserve(64); // Pre-allocate reasonable buffer size
    }    // Public utility method for token-to-text conversion
    std::string token_to_text(llama_token token, ModelInfo* model_info) const {
        return convert_token_to_text(token, model_info);
    }

    // Public utility methods for validation (can be used by LlamaManager for state checking)
    static bool validate_sampler(ModelInfo* model_info) {
        return validate_and_recover_sampler(model_info);
    }    // Main response generation function
    // Dependencies: Requires valid ModelInfo, ContextInfo, and proper LlamaManager integration
    template<typename BatchTokenAdder, typename ContextUpdater>
    std::string generate_response(const std::string& input, const std::string& username,
                                ModelInfo* model_info, ContextInfo* context_info,
                                BatchTokenAdder add_token_to_batch, ContextUpdater update_context) const {
        
        if (!model_info || !context_info || input.empty()) {
            return "Error: Invalid generation parameters";
        }

        if (!model_info->model || !context_info->context || !model_info->vocab || !context_info->batch_initialized) {
            return "Error: Model components not properly initialized or no active context";
        }

        // Validate and recover sampler if needed
        if (!validate_and_recover_sampler(model_info)) {
            return "Error: Sampler validation/recovery failed";
        }

        // Validate generation state
        if (!validate_generation_state(context_info, model_info)) {
            return "Error: Context state invalid for generation";
        }

        // Validate or attempt to recover logits
        float* logits = llama_get_logits(context_info->context);
        if (!logits && context_info->n_past > 0) {
            if (!recover_logits(context_info, model_info)) {
                return "Error: Context state invalid - no logits available and recovery failed";
            }
        } else if (!logits) {
            return "Error: Context state invalid - no logits available";
        }        // Calculate available space for generation
        const int32_t max_new_tokens = std::min(model_info->n_predict, 
                                               model_info->n_ctx - context_info->n_past - LlamaConstants::TOKEN_SAFETY_MARGIN);
        
        if (max_new_tokens <= 0) {
            return "Error: No space left in context for generation (context: " + 
                   std::to_string(context_info->n_past) + "/" + std::to_string(model_info->n_ctx) + ")";
        }

        LLAMA_LOG("Starting generation with " + std::to_string(max_new_tokens) + " max tokens, n_past=" + std::to_string(context_info->n_past));        // Initialize generation state
        std::string response;
        response.reserve(max_new_tokens * LlamaConstants::STRING_RESERVE_MULTIPLIER);
        
        auto generation_start = std::chrono::high_resolution_clock::now();
        int32_t n_generated = 0;
        std::vector<llama_seq_id> seq_ids = {0};        // Main generation loop
        while (n_generated < max_new_tokens) {
            // Periodic sampler validation during long generation
            if (n_generated % ResponseConstants::SAMPLER_VALIDATION_INTERVAL == 0 && !model_info->sampler) {
                LLAMA_LOG("CRITICAL: Sampler became null during generation at token " + std::to_string(n_generated));
                return "Error: Sampler failed during generation";
            }
            
            // Sample next token
            llama_token new_token;
            try {
                new_token = llama_sampler_sample(model_info->sampler, context_info->context, -1);
            } catch (const std::exception& e) {
                LLAMA_LOG("Exception during token sampling: " + std::string(e.what()));
                return "Error: Exception during token generation";
            } catch (...) {
                LLAMA_LOG("Unknown exception during token sampling");
                return "Error: Unknown exception during generation";
            }
            
            // Validate generated token
            if (new_token < 0) {
                LLAMA_LOG("Error: Invalid token generated: " + std::to_string(new_token));
                break;
            }
            
            // Check for end-of-generation token
            if (llama_vocab_is_eog(model_info->vocab, new_token)) {
                LLAMA_LOG("End of generation token encountered");
                break;
            }

            // Convert token to text
            std::string token_text = convert_token_to_text(new_token, model_info);
            if (token_text.empty()) {
                LLAMA_LOG("Warning: Empty token text for token " + std::to_string(new_token));
                continue;
            }
            
            response += token_text;

            // Add token to batch and decode
            if (!add_token_to_batch(new_token, context_info->n_past, seq_ids, true)) {
                LLAMA_LOG("Error: Failed to add token to batch during generation");
                return "Error: Token processing failed";
            }
            
            if (context_info->batch.n_tokens > 0 && llama_decode(context_info->context, context_info->batch) != 0) {
                LLAMA_LOG("Error: Failed to decode during generation at token " + std::to_string(n_generated));
                return "Error: Token decode failed";
            }
            
            context_info->n_past++;
            n_generated++;
        }

        // Update performance statistics
        auto generation_end = std::chrono::high_resolution_clock::now();
        context_info->last_decode_time_us = std::chrono::duration_cast<std::chrono::microseconds>(generation_end - generation_start).count();
        context_info->total_generation_tokens += n_generated;

        // Log generation performance
        if (n_generated > 0) {
            float tokens_per_second = static_cast<float>(n_generated) / (static_cast<float>(context_info->last_decode_time_us) / 1000000.0f);
            LLAMA_LOG("Generated " + std::to_string(n_generated) + " tokens in " + 
                     std::to_string(context_info->last_decode_time_us / 1000.0f) + "ms (" + 
                     std::to_string(tokens_per_second) + " t/s)");
        }        // Update conversation history and context
        if (!response.empty()) {
            context_info->message_history.emplace_back("assistant", response);
            context_info->message_cache_dirty = true;
            // Token count will be updated during context processing
            
            // Update context length using provided updater
            update_context();
        }

        // Handle reset-after-generation contexts (e.g., summary contexts)
        if (context_info->reset_after_generation) {
            LLAMA_LOG("Context marked for reset after generation (reset_after_generation flag is set)");
            
            // Preserve system message but clear everything else
            std::string saved_system_message = context_info->system_message;
            
            // Clear context state
            if (context_info->context) {
                llama_memory_clear(llama_get_memory(context_info->context), true); // Ensure kv memory/cache is cleared
            }
            context_info->n_past = 0;
            context_info->prev_len = 0;
            context_info->message_history.clear();
            context_info->message_cache_dirty = true;
            context_info->message_history_token_count = 0; // Reset token count since we cleared history
              // Restore system message for next task
            if (!saved_system_message.empty()) {
                context_info->system_message = saved_system_message;
                context_info->message_history.emplace_back("system", saved_system_message);
                context_info->message_cache_dirty = true;
                // Token count will be updated during next context processing
                LLAMA_LOG("Restored system message for next task");
            }
        }

        // Final sampler validation
        if (!model_info->sampler) {
            LLAMA_LOG("WARNING: Sampler is NULL at end of generation!");
        }

        return response;
    }
};

//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODING DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//
