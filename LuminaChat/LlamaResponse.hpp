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
#include <string_view>
#include <vector>
#include <cstdint>
#include <chrono>
#include <thread>
#include <algorithm>
#include <functional>
#include <optional>
#include <sstream>
#include <type_traits>
#include "llama-cpp.h"
#include "LogHandler.hpp"
#include "TokenCache.hpp"

// Forward declarations to avoid circular dependencies
struct ModelInfo;
struct ContextInfo;

// Type aliases for callback functions used in batch processing integration
using BatchTokenAdder = std::function<bool(llama_token, int32_t, const std::vector<llama_seq_id>&, bool)>;
using ContextUpdater = std::function<void()>;

// Constants for response generation (Directive #5: Zero Magic & Strong Typing)
namespace ResponseConstants {
    constexpr int32_t SAMPLER_VALIDATION_INTERVAL = 10;
    constexpr int32_t INITIAL_BUFFER_SIZE = 32;
    constexpr int32_t MAX_BUFFER_SIZE = 1024;
    constexpr int32_t DUMMY_TOKEN_DECODE_POS_OFFSET = 1;
    
    // C++17 Performance optimizations
    constexpr size_t STRING_BUFFER_GROWTH_FACTOR = 2;
    constexpr size_t MIN_STRING_RESERVE = 64;
    constexpr size_t LOG_MESSAGE_RESERVE = 256;
}

// Local constants to avoid dependency issues (values must match LlamaManager/LlamaContext)
namespace LlamaResponseConstants {
    // These values must be kept in sync with LlamaManager.hpp and LlamaContext.hpp
    constexpr float DEFAULT_TEMPERATURE = 0.8f;
    constexpr float DEFAULT_MIN_P = 0.05f;
    constexpr float DEFAULT_TOP_P = 0.9f;
    constexpr int32_t DEFAULT_TOP_K = 40;
    constexpr int32_t TOKEN_SAFETY_MARGIN = 32;
    constexpr size_t STRING_RESERVE_MULTIPLIER = 4;
    constexpr float TARGET_CONTEXT_USAGE = 0.60f;
}

// Thread Safety Contract (Directive #12):
// This class is NOT thread-safe and should only be accessed from within LlamaManager's context.
// All methods assume valid ModelInfo and ContextInfo structures are provided.
//
// RESPONSIBILITY SEPARATION:
// LlamaResponse is solely responsible for:
// 1. Pure response generation - token sampling, text conversion, generation loop
// 2. Sampler lifecycle management - configuration, validation, recovery
// 3. Generation state validation - ensuring context is ready for generation
// 4. Performance tracking - generation timing and token count statistics
//
// LlamaResponse does NOT handle:
// - Model loading/management (LlamaManager responsibility)
// - Context creation/switching (LlamaManager responsibility)  
// - Text-to-token conversion for inputs (LlamaManager responsibility)
// - Template application (LlamaManager responsibility)
// - Conversation history management (LlamaManager responsibility)
// - Batch operations setup (LlamaManager provides callbacks)
//
// INTEGRATION PATTERN:
// 1. LlamaManager prepares context and provides callback functions for batch operations
// 2. LlamaResponse handles the generation loop, token sampling, and text conversion
// 3. LlamaManager receives tokens through callbacks to manage batch operations and context updates
// 4. This separation allows LlamaManager to focus on infrastructure while LlamaResponse focuses on generation
class LlamaResponse {
private:
    // Working buffers for token conversion - optimized with better allocation strategy
    mutable std::vector<char> temp_string_buffer;
    mutable std::string log_buffer; // Pre-allocated buffer for log messages
    
    // Reference to TokenCache for cached token-to-text conversion
    const TokenCache* token_cache_ref = nullptr;    // C++17 optimization: Fast string building with pre-allocated buffer
    template<typename... Args>
    std::string build_log_message(Args&&... args) const {
        std::ostringstream oss;
        (oss << ... << std::forward<Args>(args)); // C++17 fold expression
        return oss.str();
    }
      // Enhanced sampler configuration with runtime validation - C++17 optimized
    static void configure_sampler(ModelInfo* model_info, float temperature, float min_p, float top_p, int32_t top_k) noexcept {
        // C++17 optimization: Use string_view for efficient logging
        constexpr std::string_view config_prefix = "Configuring sampler for model '";
        std::string log_msg;
        log_msg.reserve(config_prefix.size() + model_info->model_path.size() + 2);
        log_msg.append(config_prefix).append(model_info->model_path).append("'");
        LLAMA_RESPONSE_LOG_DEBUG(std::move(log_msg));
        
        LLAMA_RESPONSE_LOG_DEBUG("Sampler params: temp=" + std::to_string(temperature) + 
                                ", min_p=" + std::to_string(min_p) + 
                                ", top_p=" + std::to_string(top_p) + 
                                ", top_k=" + std::to_string(top_k));
        
        if (model_info->sampler) {
            LLAMA_RESPONSE_LOG_DEBUG("Freeing existing sampler");
            llama_sampler_free(model_info->sampler);
            model_info->sampler = nullptr;
        }
        
        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        model_info->sampler = llama_sampler_chain_init(sparams);
        
        if (!model_info->sampler) [[unlikely]] {
            LLAMA_RESPONSE_LOG_ERROR("Failed to create sampler chain");
            return;
        }
        
        // Add sampling strategies with likely/unlikely hints
        if (top_k > 0) [[likely]] {
            LLAMA_RESPONSE_LOG_DEBUG("Adding top_k sampler with k=" + std::to_string(top_k));
            llama_sampler_chain_add(model_info->sampler, llama_sampler_init_top_k(top_k));
        }
        
        if (top_p < 1.0f) [[likely]] {
            LLAMA_RESPONSE_LOG_DEBUG("Adding top_p sampler with p=" + std::to_string(top_p));
            llama_sampler_chain_add(model_info->sampler, llama_sampler_init_top_p(top_p, 1));
        }
        
        if (min_p > 0.0f) [[likely]] {
            LLAMA_RESPONSE_LOG_DEBUG("Adding min_p sampler with p=" + std::to_string(min_p));
            llama_sampler_chain_add(model_info->sampler, llama_sampler_init_min_p(min_p, 1));
        }
        
        if (temperature > 0.0f) [[likely]] {
            LLAMA_RESPONSE_LOG_DEBUG("Adding temperature sampler with temp=" + std::to_string(temperature));
            llama_sampler_chain_add(model_info->sampler, llama_sampler_init_temp(temperature));
            llama_sampler_chain_add(model_info->sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
        } else {
            LLAMA_RESPONSE_LOG_DEBUG("Using greedy sampler (temperature = 0)");
            llama_sampler_chain_add(model_info->sampler, llama_sampler_init_greedy());
        }
        
        // Validate final sampler state
        if (!model_info->sampler) [[unlikely]] {
            LLAMA_RESPONSE_LOG_ERROR("Sampler became null during configuration");
        } else {
            LLAMA_RESPONSE_LOG("Sampler configured successfully for model");
        }
    }    // Validate and recover sampler if needed - C++17 optimized
    static bool validate_and_recover_sampler(ModelInfo* model_info) noexcept {
        if (model_info->sampler) [[likely]] {
            return true; // Sampler is valid
        }
        
        LLAMA_LOG("WARNING: Sampler is NULL, attempting recovery...");
        
        // Attempt to recreate the sampler with default parameters
        configure_sampler(model_info, 
                          LlamaResponseConstants::DEFAULT_TEMPERATURE, 
                          LlamaResponseConstants::DEFAULT_MIN_P, 
                          LlamaResponseConstants::DEFAULT_TOP_P, 
                          LlamaResponseConstants::DEFAULT_TOP_K);
        
        if (!model_info->sampler) [[unlikely]] {
            LLAMA_LOG("CRITICAL: Failed to recover sampler!");
            return false;
        }
        
        LLAMA_LOG("Sampler recovered successfully");
        return true;
    }      // Convert token to text using the context's model vocabulary - C++17 optimized
    std::string convert_token_to_text(llama_token token, ModelInfo* model_info) const {
        if (!model_info->vocab) [[unlikely]] {
            LLAMA_LOG("Error: Vocabulary not available from current context's model");
            return {};
        }
        
        // Check for negative token values
        if (token < 0) [[unlikely]] {
            // C++17 optimization: Use structured binding and efficient string building
            std::string warning_msg;
            warning_msg.reserve(32);
            warning_msg.append("Warning: Token ").append(std::to_string(token)).append(" is negative");
            LLAMA_LOG(std::move(warning_msg));
            return {};
        }
        
        // Try cached lookup first if TokenCache is available
        if (token_cache_ref) [[likely]] {
            std::vector<llama_token> single_token = {token};
            if (auto cached_text = token_cache_ref->get_text(single_token)) {
                return *cached_text;
            }
        }
        
        // Fallback to direct llama.cpp conversion
        // Optimize buffer allocation strategy
        if (temp_string_buffer.size() < ResponseConstants::INITIAL_BUFFER_SIZE) {
            temp_string_buffer.resize(ResponseConstants::INITIAL_BUFFER_SIZE);
        }
        
        int32_t result = llama_token_to_piece(model_info->vocab, token, temp_string_buffer.data(), 
                                             temp_string_buffer.size(), 0, true);
        
        if (result < 0) [[unlikely]] {
            const auto required_size = static_cast<size_t>(-result);
            if (required_size > ResponseConstants::MAX_BUFFER_SIZE) [[unlikely]] {
                // C++17 optimization: Efficient error message construction
                std::string error_msg;
                error_msg.reserve(64);
                error_msg.append("Error: Token conversion requires excessive buffer size: ")
                         .append(std::to_string(required_size));
                LLAMA_LOG(std::move(error_msg));
                return {};
            }
            
            temp_string_buffer.resize(required_size);
            result = llama_token_to_piece(model_info->vocab, token, temp_string_buffer.data(), 
                                         temp_string_buffer.size(), 0, true);
        }
        
        std::string token_text = (result > 0) ? std::string(temp_string_buffer.data(), static_cast<size_t>(result)) : std::string{};
        
        // Cache the result if TokenCache is available and conversion was successful
        if (token_cache_ref && !token_text.empty()) [[likely]] {
            std::vector<llama_token> single_token = {token};
            std::string cache_key = "tok:" + std::to_string(token);
            token_cache_ref->put(cache_key, token_text, single_token);
        }
        
        return token_text;
    }

    // Attempt to recover logits by performing a minimal decode operation - C++17 optimized
    static bool recover_logits(ContextInfo* context_info, ModelInfo* model_info) noexcept {
        if (!context_info->batch_initialized || context_info->n_past <= 0) [[unlikely]] {
            LLAMA_RESPONSE_LOG_DEBUG("Cannot recover logits: batch not initialized or n_past <= 0");
            return false;
        }
        
        LLAMA_RESPONSE_LOG_DEBUG("No logits available, attempting recovery decode");
        
        // Clear batch and set up for logit generation
        context_info->batch.n_tokens = 0;
        
        // C++17 optimization: Use structured validation for batch components
        const auto validate_batch_pointers = [&]() noexcept -> bool {
            return context_info->batch.token && context_info->batch.pos && 
                   context_info->batch.logits && context_info->batch.seq_id;
        };
        
        if (validate_batch_pointers()) [[likely]] {
            // Use EOS token as dummy for logit generation
            const llama_token dummy_token = llama_vocab_eos(model_info->vocab);
            context_info->batch.token[0] = dummy_token;
            
            // CRITICAL FIX: Ensure position is valid (non-negative and within context)
            int32_t decode_pos = std::max(0, context_info->n_past - ResponseConstants::DUMMY_TOKEN_DECODE_POS_OFFSET);
            decode_pos = std::min(decode_pos, model_info->n_ctx - 1);
            
            LLAMA_RESPONSE_LOG_DEBUG("Attempting recovery decode at position " + std::to_string(decode_pos) + 
                                    " with dummy token " + std::to_string(dummy_token));
            
            context_info->batch.pos[0] = decode_pos;
            context_info->batch.logits[0] = 1; // Request logits
            context_info->batch.seq_id[0] = 0;
            context_info->batch.n_tokens = 1;
            
            const int decode_result = llama_decode(context_info->context, context_info->batch);
            if (decode_result == 0) [[likely]] {
                const float* logits = llama_get_logits(context_info->context);
                if (logits) [[likely]] {
                    LLAMA_RESPONSE_LOG("Successfully recovered logits with dummy decode at position " + std::to_string(decode_pos));
                    return true;
                } else {
                    LLAMA_RESPONSE_LOG_ERROR("Decode succeeded but logits still null");
                }
            } else {
                LLAMA_RESPONSE_LOG_ERROR("Dummy decode failed with error code: " + std::to_string(decode_result));
            }        } else {
            LLAMA_RESPONSE_LOG_ERROR("Batch pointer validation failed during logit recovery");
        }
        
        LLAMA_RESPONSE_LOG_ERROR("Could not recover logits for generation");
        return false;
    }
      // Comprehensive validation and recovery for generation components - C++17 optimized
    static bool validate_generation_components(ContextInfo* context_info, ModelInfo* model_info) noexcept {
        // Basic component validation with early returns
        if (!model_info) {
            LLAMA_RESPONSE_LOG_ERROR("model_info is NULL");
            return false;
        }

        if (!model_info->model) {
            LLAMA_RESPONSE_LOG_ERROR("model is NULL");
            return false;
        }
        
        if (!context_info->context) {
            LLAMA_RESPONSE_LOG_ERROR("context is NULL");
            return false;
        }
        
        if (!model_info->vocab) {
            LLAMA_RESPONSE_LOG_ERROR("vocab is NULL");
            return false;
        }
        
        if (!context_info->batch_initialized) {
            LLAMA_RESPONSE_LOG_ERROR("batch not initialized");
            return false;
        }
        
        // Context position validation with efficient range checking
        const auto [n_past, n_ctx] = std::make_pair(context_info->n_past, model_info->n_ctx);
        if (n_past < 0 || n_past >= n_ctx) [[unlikely]] {
            // C++17 optimization: Efficient error message with structured binding
            std::string error_msg;
            error_msg.reserve(128);
            error_msg.append("Error: Invalid context position during generation: ")
                     .append(std::to_string(n_past))
                     .append(" (valid range: 0 to ")
                     .append(std::to_string(n_ctx - 1))                     .append(")");
            LLAMA_RESPONSE_LOG_ERROR(std::move(error_msg));
            return false;
        }
        
        // Check for completely empty context
        if (n_past == 0 && context_info->message_history.empty()) [[unlikely]] {
            LLAMA_RESPONSE_LOG_ERROR("Context is completely empty, cannot generate");
            return false;
        }
        
        return true;
    }

public:
    LlamaResponse() {
        temp_string_buffer.reserve(ResponseConstants::MIN_STRING_RESERVE); // Pre-allocate reasonable buffer size
        log_buffer.reserve(ResponseConstants::LOG_MESSAGE_RESERVE); // Pre-allocate log buffer
    }
      // Constructor with TokenCache reference for cached token-to-text conversion
    explicit LlamaResponse(const TokenCache* cache) : token_cache_ref(cache) {
        temp_string_buffer.reserve(ResponseConstants::MIN_STRING_RESERVE); // Pre-allocate reasonable buffer size
        log_buffer.reserve(ResponseConstants::LOG_MESSAGE_RESERVE); // Pre-allocate log buffer
    }

    // Main response generation function - C++17 optimized
    // Dependencies: Requires valid ModelInfo, ContextInfo, and proper LlamaManager integration
    template<typename BatchTokenAdder, typename ContextUpdater>
    std::string generate_response(std::string_view input, std::string_view username,
                                ContextInfo* context_info,
                                BatchTokenAdder add_token_to_batch, ContextUpdater update_context) const {
        return generate_response_impl(input, username, context_info, add_token_to_batch, update_context, nullptr);
    }

    // Streaming response generation function - C++20 optimized
    // Dependencies: Requires valid ModelInfo, ContextInfo, and proper LlamaManager integration
    // StreamCallback signature: void(std::string_view token_text)
    template<typename BatchTokenAdder, typename ContextUpdater, typename StreamCallback>
    std::string generate_response(std::string_view input, std::string_view username,
                                ContextInfo* context_info,
                                BatchTokenAdder add_token_to_batch, ContextUpdater update_context,
                                StreamCallback stream_callback) const {
        return generate_response_impl(input, username, context_info, add_token_to_batch, update_context, stream_callback);
    }

private:    // Implementation method that handles both streaming and non-streaming cases
    template<typename BatchTokenAdder, typename ContextUpdater, typename StreamCallback = std::nullptr_t>
    std::string generate_response_impl(std::string_view input, std::string_view username,
                                     ContextInfo* context_info,
                                     BatchTokenAdder add_token_to_batch, ContextUpdater update_context,
                                     StreamCallback stream_callback) const {
        if (input.empty()) [[unlikely]] {
            LLAMA_RESPONSE_LOG_ERROR("Input is empty");
            return "Error: Invalid generation parameters";
        }

        LLAMA_RESPONSE_LOG_DEBUG("Generation started for user '" + std::string(username) + 
                                "', input length: " + std::to_string(input.length()));
        
        // Comprehensive validation of all generation components
        if (!validate_generation_components(context_info, context_info->model_info)) [[unlikely]] {
            LLAMA_RESPONSE_LOG_ERROR("Model components validation failed");
            return "Error: Model components not properly initialized or invalid context state";
        }

        // Validate and recover sampler if needed
        if (!validate_and_recover_sampler(context_info->model_info)) [[unlikely]] {
            LLAMA_RESPONSE_LOG_ERROR("Sampler validation/recovery failed");
            return "Error: Sampler validation/recovery failed";
        }
        
        PERF_TRACE(LLAMA_RESPONSE, "Starting response generation");
        
        // Validate or attempt to recover logits
        const float* logits = llama_get_logits(context_info->context);
        if (!logits) [[unlikely]] {
            if (context_info->n_past > 0) {
                LLAMA_RESPONSE_LOG_DEBUG("No logits available but n_past > 0, attempting recovery");
                // Context has been processed but no logits - attempt recovery
                if (!recover_logits(context_info, context_info->model_info)) [[unlikely]] {
                    LLAMA_RESPONSE_LOG_ERROR("Context state invalid - no logits available and recovery failed");
                    return "Error: Context state invalid - no logits available and recovery failed";
                }
            } else {
                // n_past == 0, this might be normal for initial generation
                // We'll try to proceed and let the generation loop handle the decode
                LLAMA_RESPONSE_LOG_DEBUG("No logits available at n_past=0, will attempt initial decode during generation");
            }
        } else {
            LLAMA_RESPONSE_LOG_DEBUG("Logits available, proceeding with generation");
        }

        // C++17 optimization: Use structured binding for cleaner code
        const auto [n_past, n_ctx, n_predict] = std::make_tuple(
            context_info->n_past, 
            context_info->model_info->n_ctx, 
            context_info->model_info->n_predict
        );
        
        // Calculate available space for generation
        const int32_t max_new_tokens = std::min(n_predict, 
                                               n_ctx - n_past - LlamaResponseConstants::TOKEN_SAFETY_MARGIN);
        
        if (max_new_tokens <= 0) [[unlikely]] {
            // C++17 optimization: Efficient error message construction
            std::string error_msg;
            error_msg.reserve(128);
            error_msg.append("Error: No space left in context for generation (context: ")
                     .append(std::to_string(n_past))
                     .append("/")
                     .append(std::to_string(n_ctx))
                     .append(")");
            return error_msg;
        }

        // C++17 optimization: Efficient log message construction
        std::string log_msg;
        log_msg.reserve(96);
        log_msg.append("Starting generation with ")
               .append(std::to_string(max_new_tokens))
               .append(" max tokens, n_past=")
               .append(std::to_string(n_past));
        LLAMA_LOG(std::move(log_msg));

        // Initialize generation state with optimized allocation
        std::string response;
        response.reserve(static_cast<size_t>(max_new_tokens) * LlamaResponseConstants::STRING_RESERVE_MULTIPLIER);
        
        const auto generation_start = std::chrono::high_resolution_clock::now();
        int32_t n_generated = 0;
        const std::vector<llama_seq_id> seq_ids = {0}; // C++17: const for immutable data        // Main generation loop - C++17 optimized with fluid streaming
        while (true) { // Remove hard token limit, rely on EOS detection
            // Periodic sampler validation during long generation
            if (n_generated % ResponseConstants::SAMPLER_VALIDATION_INTERVAL == 0 && 
                !context_info->model_info->sampler) [[unlikely]] {
                std::string error_msg;
                error_msg.reserve(80);
                error_msg.append("CRITICAL: Sampler became null during generation at token ")
                         .append(std::to_string(n_generated));
                LLAMA_LOG(std::move(error_msg));
                return "Error: Sampler failed during generation";
            }

            // Check context space before sampling (soft limit)
            if (context_info->n_past >= (n_ctx - LlamaResponseConstants::TOKEN_SAFETY_MARGIN)) [[unlikely]] {
                LLAMA_LOG("Warning: Approaching context limit, attempting graceful termination");
                break;
            }

            // Sample next token with exception safety
            llama_token new_token;
            try {
                new_token = llama_sampler_sample(context_info->model_info->sampler, context_info->context, -1);
            } catch (const std::exception& e) {
                std::string error_msg;
                error_msg.reserve(64);
                error_msg.append("Exception during token sampling: ").append(e.what());
                LLAMA_LOG(std::move(error_msg));
                return "Error: Exception during token generation";
            } catch (...) {
                LLAMA_LOG("Unknown exception during token sampling");
                return "Error: Unknown exception during generation";
            }

            // Validate generated token
            if (new_token < 0) [[unlikely]] {
                std::string error_msg;
                error_msg.reserve(48);
                error_msg.append("Error: Invalid token generated: ").append(std::to_string(new_token));
                LLAMA_LOG(std::move(error_msg));
                break;
            }            // PRIMARY: Check for end-of-generation token (this takes priority)
            if (llama_vocab_is_eog(context_info->model_info->vocab, new_token)) [[likely]] {
                LLAMA_LOG("End of generation token encountered - natural completion");
                break;
            }
            
            // Convert token to text and add to response
            auto token_text = convert_token_to_text(new_token, context_info->model_info);
            if (!token_text.empty()) [[likely]] {
                // Stream the token immediately if callback is provided
                if constexpr (!std::is_same_v<StreamCallback, std::nullptr_t>) {
                    stream_callback(std::string_view{token_text});
                }
                response += std::move(token_text);
            } else [[unlikely]] {
                std::string warning_msg;
                warning_msg.reserve(48);
                warning_msg.append("Warning: Empty token text for token ").append(std::to_string(new_token));
                LLAMA_LOG(std::move(warning_msg));
                continue;
            }

            // Add token to batch and decode
            if (!add_token_to_batch(new_token, context_info->n_past, seq_ids, true)) [[unlikely]] {
                LLAMA_LOG("Error: Failed to add token to batch during generation");
                return "Error: Token processing failed";
            }
            
            if (context_info->batch.n_tokens > 0 && 
                llama_decode(context_info->context, context_info->batch) != 0) [[unlikely]] {
                std::string error_msg;
                error_msg.reserve(64);
                error_msg.append("Error: Failed to decode during generation at token ")
                         .append(std::to_string(n_generated));
                LLAMA_LOG(std::move(error_msg));
                return "Error: Token decode failed";
            }
            
            context_info->n_past++;
            n_generated++;            // Optional: Soft limit warning (but don't break)
            if (n_generated >= max_new_tokens) [[unlikely]] {
                LLAMA_LOG("Warning: Exceeded predicted token count, continuing until EOS or context limit");
            }
        }
        
        // Update performance statistics - C++17 optimized
        const auto generation_end = std::chrono::high_resolution_clock::now();
        context_info->last_decode_time_us = std::chrono::duration_cast<std::chrono::microseconds>(generation_end - generation_start).count();
        context_info->total_generation_tokens += n_generated;

        // Log generation performance with efficient string construction
        if (n_generated > 0) [[likely]] {
            const auto time_seconds = static_cast<float>(context_info->last_decode_time_us) / 1000000.0f;
            const auto tokens_per_second = static_cast<float>(n_generated) / time_seconds;
            const auto time_ms = context_info->last_decode_time_us / 1000.0f;
            
            // C++17 optimization: Efficient performance log message
            std::string perf_msg;
            perf_msg.reserve(128);
            perf_msg.append("Generated ").append(std::to_string(n_generated))
                    .append(" tokens in ").append(std::to_string(time_ms))
                    .append("ms (").append(std::to_string(tokens_per_second))
                    .append(" t/s)");
            LLAMA_LOG(std::move(perf_msg));
        }
          // Do NOT automatically update conversation history - let LlamaManager handle this
        // The linear flow in LlamaManager will add the response to conversation history
        if (!response.empty()) [[likely]] {
            // Update context length using provided updater
            update_context();
        }

        // Handle reset-after-generation contexts (e.g., summary contexts) - C++17 optimized
        if (context_info->reset_after_generation) [[unlikely]] {
            LLAMA_LOG("Context marked for reset after generation (reset_after_generation flag is set)");
            
            // C++17 optimization: Use move semantics to preserve system message
            std::string saved_system_message = std::move(context_info->system_message);

            // Clear context state
            if (context_info->context) [[likely]] {
                llama_memory_clear(llama_get_memory(context_info->context), true); // Ensure kv memory/cache is cleared
            }
            context_info->n_past = 0;
            context_info->prev_len = 0;
            context_info->clear_conversation();
            
            // Restore system message for next task
            if (!saved_system_message.empty()) [[likely]] {
                context_info->system_message = std::move(saved_system_message);
                context_info->add_message("system", context_info->system_message);
                // Token count will be updated during next context processing
                LLAMA_LOG("Restored system message for next task");
            }
        }

        // Final sampler validation
        if (!context_info->model_info->sampler) [[unlikely]] {
            LLAMA_LOG("WARNING: Sampler is NULL at end of generation!");
        }

        return response;
    }
};

//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODING DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//
