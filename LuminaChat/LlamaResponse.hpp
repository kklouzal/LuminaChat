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
#include <functional>
#include "llama-cpp.h"
#include "LogHandler.hpp"

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
    
    // Lookahead decoding constants
    constexpr int32_t DEFAULT_LOOKAHEAD_WINDOW = 15;    // W - lookahead window size
    constexpr int32_t DEFAULT_NGRAM_SIZE = 5;           // N - n-gram size
    constexpr int32_t DEFAULT_MAX_VERIFICATION = 15;    // G - max verification n-grams
    constexpr int32_t MIN_LOOKAHEAD_WINDOW = 3;
    constexpr int32_t MAX_LOOKAHEAD_WINDOW = 32;
    constexpr int32_t MIN_NGRAM_SIZE = 2;
    constexpr int32_t MAX_NGRAM_SIZE = 8;
    constexpr int32_t MIN_MAX_VERIFICATION = 1;
    constexpr int32_t MAX_MAX_VERIFICATION = 32;
}

// Configuration structure for lookahead decoding
struct LookaheadConfig {
    bool enabled = false;                               // Enable/disable lookahead decoding
    int32_t window_size = ResponseConstants::DEFAULT_LOOKAHEAD_WINDOW;     // W - lookahead window size  
    int32_t ngram_size = ResponseConstants::DEFAULT_NGRAM_SIZE;            // N - n-gram size
    int32_t max_verification = ResponseConstants::DEFAULT_MAX_VERIFICATION; // G - max verification n-grams
    
    // Validate configuration parameters
    bool validate() const noexcept {
        return window_size >= ResponseConstants::MIN_LOOKAHEAD_WINDOW && 
               window_size <= ResponseConstants::MAX_LOOKAHEAD_WINDOW &&
               ngram_size >= ResponseConstants::MIN_NGRAM_SIZE &&
               ngram_size <= ResponseConstants::MAX_NGRAM_SIZE &&
               max_verification >= ResponseConstants::MIN_MAX_VERIFICATION &&
               max_verification <= ResponseConstants::MAX_MAX_VERIFICATION;
    }
};

// N-gram data structure for tracking verification sequences
struct NgramData {
    bool active = false;
    llama_seq_id seq_id = -1;
    std::vector<int32_t> i_batch;
    std::vector<llama_token> tokens;
    
    void reset() noexcept {
        active = false;
        seq_id = -1;
        i_batch.clear();
        tokens.clear();
    }
};

// N-gram container for storing observed patterns
struct NgramContainer {
    int32_t n_total = 0;
    std::vector<int32_t> cnt;           // Count of n-grams for each token
    std::vector<int32_t> head;          // Ring buffer head position for each token
    std::vector<llama_token> tokens;    // [n_vocab][G][N-1] storage
    
    NgramContainer(int32_t n_vocab, int32_t N, int32_t G) {
        cnt.resize(n_vocab, 0);
        head.resize(n_vocab, 0);
        tokens.resize(n_vocab * G * (N - 1), 0);
    }
    
    // Get n-gram tokens for a specific first token and index
    std::vector<llama_token> get_ngram_tokens(llama_token first_token, int32_t index, int32_t N, int32_t G) const {
        std::vector<llama_token> result;
        if (first_token < 0 || first_token >= static_cast<llama_token>(cnt.size()) || 
            index < 0 || index >= cnt[first_token]) {
            return result;
        }
        
        result.reserve(N - 1);
        const int32_t base_idx = first_token * G * (N - 1) + index * (N - 1);
        for (int32_t i = 0; i < N - 1; ++i) {
            result.push_back(tokens[base_idx + i]);
        }
        return result;
    }
    
    // Add new n-gram for a specific first token
    void add_ngram(llama_token first_token, const std::vector<llama_token>& ngram_tokens, int32_t N, int32_t G) {
        if (first_token < 0 || first_token >= static_cast<llama_token>(cnt.size()) || 
            ngram_tokens.size() != static_cast<size_t>(N - 1)) {
            return;
        }
        
        // Check for duplicates
        for (int32_t i = 0; i < cnt[first_token]; ++i) {
            bool is_duplicate = true;
            const int32_t base_idx = first_token * G * (N - 1) + i * (N - 1);
            for (int32_t j = 0; j < N - 1; ++j) {
                if (tokens[base_idx + j] != ngram_tokens[j]) {
                    is_duplicate = false;
                    break;
                }
            }
            if (is_duplicate) return; // Skip duplicate
        }
        
        // Add new n-gram using ring buffer
        const int32_t current_head = head[first_token];
        const int32_t base_idx = first_token * G * (N - 1) + current_head * (N - 1);
          for (int32_t i = 0; i < N - 1; ++i) {
            tokens[base_idx + i] = ngram_tokens[i];
        }
        
        cnt[first_token] = std::min(G, cnt[first_token] + 1);
        head[first_token] = (current_head + 1) % G;
        n_total++;    }
};

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
    // Working buffers for token conversion
    mutable std::vector<char> temp_string_buffer;
    
    // Lookahead state (mutable to allow const methods to modify for generation)
    mutable LookaheadConfig lookahead_config;
    mutable std::unique_ptr<NgramContainer> ngrams_observed;
    mutable std::vector<NgramData> ngrams_cur;
    mutable std::vector<llama_token> tokens_j_prev;
    mutable std::vector<std::vector<llama_token>> tokens_j;// Enhanced sampler configuration with runtime validation
    static void configure_sampler(ModelInfo* model_info, float temperature, float min_p, float top_p, int32_t top_k) {
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
        if (model_info->sampler) {
            return true; // Sampler is valid
        }
        
        LLAMA_LOG("WARNING: Sampler is NULL, attempting recovery...");        // Attempt to recreate the sampler with default parameters
        configure_sampler(model_info, 
                          LlamaResponseConstants::DEFAULT_TEMPERATURE, 
                          LlamaResponseConstants::DEFAULT_MIN_P, 
                          LlamaResponseConstants::DEFAULT_TOP_P, 
                          LlamaResponseConstants::DEFAULT_TOP_K);
        
        if (!model_info->sampler) {
            LLAMA_LOG("CRITICAL: Failed to recover sampler!");
            return false;
        }
        
        LLAMA_LOG("Sampler recovered successfully");
        return true;
    }
    
    // Convert token to text using the context's model vocabulary
    std::string convert_token_to_text(llama_token token, ModelInfo* model_info) const {
        if (!model_info->vocab) {
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
        if (!context_info->batch_initialized || context_info->n_past <= 0) {
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
    }    // Comprehensive validation and recovery for generation components
    static bool validate_generation_components(ContextInfo* context_info, ModelInfo* model_info) {
        // Basic component validation
        if (!model_info) {
            LLAMA_LOG("Error: model_info is NULL");
            return false;
        }

        if (!model_info->model) {
            LLAMA_LOG("Error: model is NULL");
            return false;
        }
        
        if (!context_info->context) {
            LLAMA_LOG("Error: context is NULL");
            return false;
        }
        
        if (!model_info->vocab) {
            LLAMA_LOG("Error: vocab is NULL");
            return false;
        }
        
        if (!context_info->batch_initialized) {
            LLAMA_LOG("Error: batch not initialized");
            return false;
        }
        
        // Context position validation
        if (context_info->n_past < 0 || context_info->n_past >= model_info->n_ctx) {
            LLAMA_LOG("Error: Invalid context position during generation: " + std::to_string(context_info->n_past) + 
                      " (valid range: 0 to " + std::to_string(model_info->n_ctx - 1) + ")");
            return false;
        }
          // Check for completely empty context
        if (context_info->n_past == 0 && context_info->message_history.empty()) {
            LLAMA_LOG("Error: Context is completely empty, cannot generate");
            return false;
        }
        
        return true;    }    // Initialize lookahead structures
    void initialize_lookahead(ModelInfo* model_info, const LookaheadConfig& config) const {
        if (!config.enabled || !config.validate()) {
            LLAMA_LOG("Lookahead initialization skipped: disabled or invalid config");
            ngrams_observed.reset();
            return;
        }
        
        lookahead_config = config;
        
        // Initialize n-gram container
        const int32_t n_vocab = static_cast<int32_t>(llama_vocab_n_tokens(model_info->vocab));
        ngrams_observed = std::make_unique<NgramContainer>(n_vocab, config.ngram_size, config.max_verification);
        
        // Initialize verification n-gram data
        ngrams_cur.clear();
        ngrams_cur.reserve(config.max_verification);
          // Initialize lookahead token history following the working example
        tokens_j_prev.clear();
        tokens_j_prev.resize(config.window_size, 0);
        
        tokens_j.clear();
        tokens_j.resize(config.ngram_size - 1);
        for (int32_t j = 0; j < config.ngram_size - 1; ++j) {
            tokens_j[j].resize(config.window_size);
            
            // Initialize with simple increasing sequence like the working example
            for (int32_t i = 0; i < config.window_size; ++i) {
                tokens_j[j][i] = 100 + i;  // Simple initialization like working example
            }
        }
        
        LLAMA_LOG("Lookahead decoding initialized: W=" + std::to_string(config.window_size) + 
                  ", N=" + std::to_string(config.ngram_size) + 
                  ", G=" + std::to_string(config.max_verification));
    }    // Build lookahead batch directly (bypassing single-sequence BatchTokenAdder)
    template<typename BatchTokenAdder>
    bool build_lookahead_batch(llama_token current_token, int32_t n_past, ModelInfo* model_info,
                              BatchTokenAdder add_token_to_batch) const {
        const int32_t W = lookahead_config.window_size;
        const int32_t N = lookahead_config.ngram_size;
        const int32_t G = lookahead_config.max_verification;

        // The issue: BatchTokenAdder uses single-sequence batch operations that clear the batch
        // Solution: Return false to indicate we need to handle this specially
        // The caller will create its own multi-sequence batch for lookahead
        
        LLAMA_LOG("Lookahead batch building requires multi-sequence support - delegating to caller");
        return false; // Signal that we need special multi-sequence batch handling
    }    // Build lookahead batch directly like the working example
    bool build_lookahead_batch_direct(llama_token current_token, int32_t n_past, ContextInfo* context_info) const {
        const int32_t W = lookahead_config.window_size;
        const int32_t N = lookahead_config.ngram_size;
        const int32_t G = lookahead_config.max_verification;

        // Create a new batch for lookahead with proper sequence support (like working example)
        const int32_t max_batch_size = 1 + G * (N - 1) + (W - 1) + (N - 1) * W;
        llama_batch lookahead_batch = llama_batch_init(max_batch_size, 0, W + G + 1);
        
        if (!lookahead_batch.token) {
            LLAMA_LOG("Error: Failed to create lookahead batch");
            return false;
        }

        // Clear the batch
        lookahead_batch.n_tokens = 0;

        // Step 1: Current token goes to ALL sequences (working example approach)
        std::vector<llama_seq_id> seq_id_all;
        seq_id_all.reserve(W + G + 1);
        for (int32_t i = 0; i < W + G + 1; ++i) {
            seq_id_all.push_back(i);
        }
        
        // Add current token like working example: common_batch_add(batch, id, n_past, seq_id_all, true);
        lookahead_batch.token[lookahead_batch.n_tokens] = current_token;
        lookahead_batch.pos[lookahead_batch.n_tokens] = n_past;
        lookahead_batch.logits[lookahead_batch.n_tokens] = 1; // Request logits
        
        // Set all sequence IDs for current token
        for (size_t i = 0; i < seq_id_all.size(); ++i) {
            lookahead_batch.seq_id[lookahead_batch.n_tokens * lookahead_batch.n_seq_max + i] = seq_id_all[i];
        }
        lookahead_batch.n_tokens++;

        // Step 2: Add verification n-grams (working example order)
        const int32_t g_cur = (ngrams_observed && current_token < static_cast<llama_token>(ngrams_observed->cnt.size())) 
                             ? ngrams_observed->cnt[current_token] : 0;
        
        ngrams_cur.clear();
        ngrams_cur.resize(g_cur);
        
        for (int32_t g = 0; g < g_cur; ++g) {
            ngrams_cur[g].active = true;
            ngrams_cur[g].tokens.resize(N);
            ngrams_cur[g].i_batch.resize(N);
            ngrams_cur[g].seq_id = W + 1 + g;
            ngrams_cur[g].i_batch[0] = 0; // Current token position
            ngrams_cur[g].tokens[0] = current_token;
        }
        
        // Add verification tokens
        for (int32_t j = 0; j < N - 1; ++j) {
            for (int32_t g = 0; g < g_cur; ++g) {
                auto ngram_tokens = ngrams_observed->get_ngram_tokens(current_token, g, N, G);
                if (static_cast<size_t>(j) < ngram_tokens.size()) {
                    const llama_token t = ngram_tokens[j];
                    ngrams_cur[g].tokens[j + 1] = t;
                    ngrams_cur[g].i_batch[j + 1] = lookahead_batch.n_tokens;
                    
                    // Add like: common_batch_add(batch, t, n_past + j + 1, { W + 1 + g }, true);
                    lookahead_batch.token[lookahead_batch.n_tokens] = t;
                    lookahead_batch.pos[lookahead_batch.n_tokens] = n_past + j + 1;
                    lookahead_batch.logits[lookahead_batch.n_tokens] = 1; // Request logits
                    lookahead_batch.seq_id[lookahead_batch.n_tokens * lookahead_batch.n_seq_max] = W + 1 + g;
                    lookahead_batch.n_tokens++;
                } else {
                    ngrams_cur[g].active = false;
                }
            }
        }
        
        // Step 3: Add lookahead tokens (working example structure)
        // Fill the remaining W - 1 tokens for the first level
        for (int32_t i = 1; i < W; ++i) {
            std::vector<llama_seq_id> seq_id_look;
            seq_id_look.reserve(W - i);
            for (int32_t j = 0; j < W - i; ++j) {
                seq_id_look.push_back(i + j + 1);
            }
            
            llama_token token_to_add = (tokens_j.empty() || tokens_j[0].empty() || i >= static_cast<int32_t>(tokens_j[0].size()))
                                     ? static_cast<llama_token>(100 + i)
                                     : tokens_j[0][i];
            
            // Add like: common_batch_add(batch, tokens_j[0][i], n_past + i, seq_id_look, false);
            lookahead_batch.token[lookahead_batch.n_tokens] = token_to_add;
            lookahead_batch.pos[lookahead_batch.n_tokens] = n_past + i;
            lookahead_batch.logits[lookahead_batch.n_tokens] = 0; // No logits
            
            for (size_t k = 0; k < seq_id_look.size(); ++k) {
                lookahead_batch.seq_id[lookahead_batch.n_tokens * lookahead_batch.n_seq_max + k] = seq_id_look[k];
            }
            lookahead_batch.n_tokens++;
        }
        
        // Fill the rest of the lookahead levels
        for (int32_t j = 1; j < N - 1; ++j) {
            for (int32_t i = 0; i < W; ++i) {
                llama_token token_to_add = (tokens_j.size() <= static_cast<size_t>(j) || 
                                          tokens_j[j].size() <= static_cast<size_t>(i))
                                         ? static_cast<llama_token>(100 + i + j * W)
                                         : tokens_j[j][i];
                
                bool request_logits = (j == N - 2); // Last level requests logits
                
                // Add like: common_batch_add(batch, tokens_j[j][i], n_past + j + i, { i + 1 }, j == N - 2);
                lookahead_batch.token[lookahead_batch.n_tokens] = token_to_add;
                lookahead_batch.pos[lookahead_batch.n_tokens] = n_past + j + i;
                lookahead_batch.logits[lookahead_batch.n_tokens] = request_logits ? 1 : 0;
                lookahead_batch.seq_id[lookahead_batch.n_tokens * lookahead_batch.n_seq_max] = i + 1;
                lookahead_batch.n_tokens++;
            }
        }
        
        // Replace the context's batch with our lookahead batch
        if (context_info->batch_initialized) {
            llama_batch_free(context_info->batch);
        }
        context_info->batch = lookahead_batch;
        context_info->batch_initialized = true;
        
        LLAMA_LOG("Built direct lookahead batch: " + std::to_string(lookahead_batch.n_tokens) + 
                  " tokens, W=" + std::to_string(W) + ", N=" + std::to_string(N) + ", G=" + std::to_string(g_cur));
        return true;
    }// Update lookahead tokens following the working example approach
    template<typename Sampler>
    void update_lookahead_tokens(ContextInfo* context_info, Sampler sample_func, 
                                int32_t verification_count, int32_t level, bool is_first_token) const {
        const int32_t W = lookahead_config.window_size;
        const int32_t N = lookahead_config.ngram_size;
        
        // Bounds checking for tokens_j
        if (tokens_j.empty() || tokens_j[0].empty()) {
            LLAMA_LOG("Warning: Lookahead tokens not properly initialized");
            return;
        }
        
        // Step 1: Save previous tokens from first level (exactly like working example)
        for (int32_t i = 0; i < W && i < static_cast<int32_t>(tokens_j_prev.size()) && 
             i < static_cast<int32_t>(tokens_j[0].size()); ++i) {
            tokens_j_prev[i] = tokens_j[0][i];
        }
        
        // Step 2: Shift token levels (exactly like working example)
        for (int32_t j = 0; j < N - 2 && j + 1 < static_cast<int32_t>(tokens_j.size()); ++j) {
            if (tokens_j[j].size() == tokens_j[j + 1].size()) {
                tokens_j[j] = tokens_j[j + 1];
            }
        }
        
        // Step 3: Generate new tokens for the last level (following working example logic)
        if (N >= 2 && static_cast<size_t>(N - 2) < tokens_j.size()) {
            auto& last_level = tokens_j[N - 2];
            
            if (level == 0) {
                // Sample from the last level using the working example batch position calculation
                for (int32_t i = 0; i < W && i < static_cast<int32_t>(last_level.size()); ++i) {
                    // Working example formula: ngrams_cur.size() * (N - 1) + W * (N - 2) + i
                    int32_t batch_idx = verification_count * (N - 1) + W * (N - 2) + i;
                    
                    try {
                        last_level[i] = sample_func(batch_idx);
                    } catch (...) {
                        // Fallback: use token from first level like working example
                        if (i < static_cast<int32_t>(tokens_j[0].size())) {
                            last_level[i] = tokens_j[0][i];
                        } else {
                            // Final fallback - simple initialization
                            last_level[i] = static_cast<llama_token>(100 + i);
                        }
                        LLAMA_LOG("Warning: Failed to sample lookahead token " + std::to_string(i) + 
                                 ", using fallback");
                    }
                }
            } else {
                // For verification levels > 0, reinitialize like working example
                for (int32_t i = 0; i < W && i < static_cast<int32_t>(last_level.size()); ++i) {
                    // Working example uses tokens from first level for reinitialization
                    if (i < static_cast<int32_t>(tokens_j[0].size())) {
                        last_level[i] = tokens_j[0][i];
                    } else {
                        last_level[i] = static_cast<llama_token>(100 + i);
                    }
                }
            }
        }
    }// Update observed n-grams with new patterns
    void update_observed_ngrams(llama_token first_token) const {
        const int32_t W = lookahead_config.window_size;
        const int32_t N = lookahead_config.ngram_size;
        
        if (!ngrams_observed || tokens_j_prev.empty()) {
            return;
        }
        
        // Generate n-grams from lookahead window using the previous tokens
        for (int32_t f = 0; f < W && f < static_cast<int32_t>(tokens_j_prev.size()); ++f) {
            const llama_token ft = tokens_j_prev[f];
            
            // Skip invalid tokens
            if (ft <= 0) continue;
            
            std::vector<llama_token> ngram(N - 1);
            bool valid_ngram = true;
            
            // Build n-gram from consecutive positions  
            for (int32_t j = 0; j < N - 1; ++j) {
                int32_t pos = f + j + 1;
                if (pos < W && pos < static_cast<int32_t>(tokens_j_prev.size())) {
                    ngram[j] = tokens_j_prev[pos];
                    if (ngram[j] <= 0) {
                        valid_ngram = false;
                        break;
                    }
                } else {
                    valid_ngram = false;
                    break;
                }
            }
            
            // Only add valid n-grams to the database
            if (valid_ngram) {
                ngrams_observed->add_ngram(ft, ngram, N, lookahead_config.max_verification);
            }
        }
        
        LLAMA_LOG("Updated n-grams database, total: " + std::to_string(ngrams_observed->n_total));
    }

public:
    LlamaResponse() {
        temp_string_buffer.reserve(64); // Pre-allocate reasonable buffer size
    }
    
    // Configure lookahead decoding parameters
    void configure_lookahead(const LookaheadConfig& config) {
        lookahead_config = config;
        if (!config.enabled) {
            ngrams_observed.reset();
            ngrams_cur.clear();
            tokens_j_prev.clear();
            tokens_j.clear();
            LLAMA_LOG("Lookahead decoding disabled");
        }
    }
    
    // Get current lookahead configuration
    const LookaheadConfig& get_lookahead_config() const noexcept {
        return lookahead_config;
    }
      
    // Lookahead-enabled response generation function
    template<typename BatchTokenAdder, typename ContextUpdater>
    std::string generate_response_with_lookahead(const std::string& input, const std::string& username,
                                                ContextInfo* context_info, const LookaheadConfig& config,
                                                BatchTokenAdder add_token_to_batch, ContextUpdater update_context) const {
        if (input.empty()) {
            LLAMA_LOG("Error: input is empty");
            return "Error: Invalid generation parameters";
        }

        // Comprehensive validation of all generation components
        if (!validate_generation_components(context_info, context_info->model_info)) {
            return "Error: Model components not properly initialized or invalid context state";
        }

        // Validate and recover sampler if needed
        if (!validate_and_recover_sampler(context_info->model_info)) {
            return "Error: Sampler validation/recovery failed";
        }
        
        // Initialize lookahead structures
        initialize_lookahead(context_info->model_info, config);
          if (!config.enabled || !ngrams_observed) {
            LLAMA_LOG("Falling back to standard generation (lookahead disabled or failed)");
            return generate_response(input, username, context_info, add_token_to_batch, update_context);
        }

        // Calculate available space for generation
        const int32_t max_new_tokens = std::min(context_info->model_info->n_predict, 
                                               context_info->model_info->n_ctx - context_info->n_past - LlamaResponseConstants::TOKEN_SAFETY_MARGIN);
        
        if (max_new_tokens <= 0) {
            return "Error: No space left in context for lookahead generation (context: " + 
                   std::to_string(context_info->n_past) + "/" + std::to_string(context_info->model_info->n_ctx) + ")";
        }

        LLAMA_LOG("Starting lookahead generation with " + std::to_string(max_new_tokens) + " max tokens, n_past=" + std::to_string(context_info->n_past));

        // Initialize generation state
        std::string response;
        response.reserve(max_new_tokens * LlamaResponseConstants::STRING_RESERVE_MULTIPLIER);
          auto generation_start = std::chrono::high_resolution_clock::now();
        int32_t n_generated = 0;
        int32_t n_accept = 0;
        
        // Sample first token using standard method
        llama_token current_id = llama_sampler_sample(context_info->model_info->sampler, context_info->context, -1);
        llama_sampler_accept(context_info->model_info->sampler, current_id);
        
        // Convert and output first token
        std::string token_text = convert_token_to_text(current_id, context_info->model_info);
        if (!token_text.empty()) {
            response += token_text;
            LLAMA_LOG("First token generated: " + token_text);
        }
          // Count the first token
        n_generated++;
        context_info->n_past++;
          // Initialize all sequences following the working example approach
        auto* memory = llama_get_memory(context_info->context);
        if (memory) {
            // Copy sequence 0 to all lookahead and verification sequences like working example
            const int32_t W = config.window_size;
            const int32_t G = config.max_verification;
            for (int32_t s = 1; s < W + G + 1; ++s) {
                llama_memory_seq_cp(memory, 0, s, -1, -1);
            }
            LLAMA_LOG("Initialized " + std::to_string(W + G) + " sequences from main sequence");
        }
          // Main lookahead generation loop
        while (n_generated < max_new_tokens) {            LLAMA_LOG("Loop iteration: n_generated=" + std::to_string(n_generated) + 
                     "/" + std::to_string(max_new_tokens) + ", current_token=" + std::to_string(current_id));
            
            // Check for end-of-generation token
            if (llama_vocab_is_eog(context_info->model_info->vocab, current_id)) {
                LLAMA_LOG("End of generation token encountered in lookahead");
                break;
            }
              // Build lookahead batch
            if (!build_lookahead_batch(current_id, context_info->n_past, 
                                     context_info->model_info, add_token_to_batch)) {
                LLAMA_LOG("BatchTokenAdder approach failed, using direct lookahead batch building");
                
                // Use direct batch building like the working example
                if (!build_lookahead_batch_direct(current_id, context_info->n_past, context_info)) {
                    LLAMA_LOG("Error: Direct lookahead batch building failed, falling back to standard generation");
                    // Fall back to standard token generation
                    current_id = llama_sampler_sample(context_info->model_info->sampler, context_info->context, -1);
                    llama_sampler_accept(context_info->model_info->sampler, current_id);
                    
                    token_text = convert_token_to_text(current_id, context_info->model_info);
                    if (!token_text.empty()) {
                        response += token_text;
                    }
                    
                    n_generated++;
                    context_info->n_past++;
                    continue;
                }
            }
              // Decode the batch with error handling
            int decode_result = llama_decode(context_info->context, context_info->batch);
            if (decode_result != 0) {
                LLAMA_LOG("Warning: Failed to decode lookahead batch (error " + std::to_string(decode_result) + "), falling back to standard generation");
                
                // Clear KV cache state and fall back to standard generation
                auto* memory = llama_get_memory(context_info->context);
                if (memory) {
                    // Remove problematic sequences
                    for (int32_t s = 1; s < 32; ++s) {
                        llama_memory_seq_rm(memory, s, -1, -1);
                    }
                }
                
                // Generate one token normally
                current_id = llama_sampler_sample(context_info->model_info->sampler, context_info->context, -1);
                llama_sampler_accept(context_info->model_info->sampler, current_id);
                
                token_text = convert_token_to_text(current_id, context_info->model_info);
                if (!token_text.empty()) {
                    response += token_text;
                }
                
                n_generated++;
                context_info->n_past++;
                
                // Update n-grams and continue
                update_observed_ngrams(current_id);
                continue;
            }            // Verification loop following the working example exactly
            llama_seq_id best_seq_id = 0;
            int32_t total_accepted = 0;
            
            // Iterate through verification levels like the working example (v = 0 to N)
            for (int32_t v = 0; v < config.ngram_size; ++v) {
                int32_t i_batch = 0;
                
                // If v > 0, check for active verification n-grams
                if (v > 0) {
                    bool found_active = false;
                    for (size_t g = 0; g < ngrams_cur.size(); ++g) {
                        if (ngrams_cur[g].active) {
                            if (static_cast<size_t>(v) < ngrams_cur[g].i_batch.size()) {
                                i_batch = ngrams_cur[g].i_batch[v];
                                best_seq_id = ngrams_cur[g].seq_id;
                                found_active = true;
                                total_accepted++;
                                break;
                            }
                        }
                    }
                    
                    // No more active matches -> stop verification
                    if (!found_active || i_batch == 0) {
                        break;
                    }
                }
                
                // Sample the next token (working example approach)
                current_id = llama_sampler_sample(context_info->model_info->sampler, context_info->context, i_batch);
                llama_sampler_accept(context_info->model_info->sampler, current_id);
                
                // Convert and output token
                token_text = convert_token_to_text(current_id, context_info->model_info);
                if (!token_text.empty()) {
                    if (v == 0) {
                        response += token_text;
                        LLAMA_LOG("Generated: " + token_text);
                    } else {
                        response += token_text;
                        LLAMA_LOG("Verified: " + token_text);
                    }
                }
                
                // Check for EOS
                if (llama_vocab_is_eog(context_info->model_info->vocab, current_id)) {
                    LLAMA_LOG("End of generation token encountered");
                    break;
                }
                
                n_generated++;
                context_info->n_past++;
                
                if (n_generated >= max_new_tokens) {
                    break;
                }
                
                // Verify against active n-grams (working example logic)
                for (size_t g = 0; g < ngrams_cur.size(); ++g) {
                    if (ngrams_cur[g].active) {
                        if (v == config.ngram_size - 1) {
                            // Reached end of n-gram
                            ngrams_cur[g].active = false;
                        } else {
                            // Check if current token matches expected token
                            if (static_cast<size_t>(v + 1) < ngrams_cur[g].tokens.size()) {
                                if (current_id != ngrams_cur[g].tokens[v + 1]) {
                                    ngrams_cur[g].active = false;
                                }
                            } else {
                                ngrams_cur[g].active = false;
                            }
                        }
                    }
                }
                
                // Update lookahead tokens (working example approach)
                auto sampler_func = [context_info](int32_t batch_idx) -> llama_token {
                    return llama_sampler_sample(context_info->model_info->sampler, context_info->context, batch_idx);
                };
                
                update_lookahead_tokens(context_info, sampler_func, static_cast<int32_t>(ngrams_cur.size()), v, v == 0);
                
                // Update observed n-grams on first token only (working example)
                if (v == 0) {
                    update_observed_ngrams(current_id);
                }
            }
            
            if (total_accepted > 0) {
                n_accept += total_accepted;
                LLAMA_LOG("Verified and accepted " + std::to_string(total_accepted) + " tokens");
            }            // KV cache management following the working example exactly
            if (n_generated >= max_new_tokens || llama_vocab_is_eog(context_info->model_info->vocab, current_id)) {
                break;
            }
            
            // KV cache management - exactly like working example
            memory = llama_get_memory(context_info->context);
            if (memory) {
                // Remove all positions after n_past (working example approach)
                llama_memory_seq_rm(memory, -1, context_info->n_past, -1);
                
                if (best_seq_id != 0) {
                    // Keep the best verification sequence and copy it back (working example)
                    llama_memory_seq_keep(memory, best_seq_id);
                    llama_memory_seq_cp(memory, best_seq_id, 0, -1, -1);
                    llama_memory_seq_rm(memory, best_seq_id, -1, -1);
                    
                    // Copy main sequence to all lookahead and verification sequences
                    const int32_t W = config.window_size;
                    const int32_t G = config.max_verification;
                    for (int32_t s = 1; s < W + G + 1; ++s) {
                        llama_memory_seq_cp(memory, 0, s, -1, -1);
                    }
                } else {
                    // No verification match - still copy to all sequences for next iteration
                    const int32_t W = config.window_size;
                    const int32_t G = config.max_verification;
                    for (int32_t s = 1; s < W + G + 1; ++s) {
                        llama_memory_seq_cp(memory, 0, s, -1, -1);
                    }
                }
            }
        }
        
        // Update performance statistics
        auto generation_end = std::chrono::high_resolution_clock::now();
        context_info->last_decode_time_us = std::chrono::duration_cast<std::chrono::microseconds>(generation_end - generation_start).count();
        context_info->total_generation_tokens += n_generated;

        // Log performance
        if (n_generated > 0) {
            float tokens_per_second = static_cast<float>(n_generated) / (static_cast<float>(context_info->last_decode_time_us) / 1000000.0f);
            float acceptance_rate = n_accept > 0 ? static_cast<float>(n_accept) / static_cast<float>(n_generated) * 100.0f : 0.0f;
            LLAMA_LOG("Lookahead generated " + std::to_string(n_generated) + " tokens in " + 
                     std::to_string(context_info->last_decode_time_us / 1000.0f) + "ms (" + 
                     std::to_string(tokens_per_second) + " t/s, " + std::to_string(acceptance_rate) + "% acceptance)");
            LLAMA_LOG("Lookahead stats: W=" + std::to_string(config.window_size) + 
                     ", N=" + std::to_string(config.ngram_size) + 
                     ", G=" + std::to_string(config.max_verification) + 
                     ", n_generated=" + std::to_string(n_generated) + 
                     ", n_accept=" + std::to_string(n_accept));
        }
        
        // Update context using provided updater
        if (!response.empty()) {
            update_context();
        }

        return response;
    }

    // Main response generation function
    // Dependencies: Requires valid ModelInfo, ContextInfo, and proper LlamaManager integration
    template<typename BatchTokenAdder, typename ContextUpdater>
    std::string generate_response(const std::string& input, const std::string& username,
                                ContextInfo* context_info,
                                BatchTokenAdder add_token_to_batch, ContextUpdater update_context) const {
        // Check if lookahead is enabled and delegate to lookahead method
        if (lookahead_config.enabled && lookahead_config.validate()) {
            return generate_response_with_lookahead(input, username, context_info, lookahead_config, add_token_to_batch, update_context);
        }
        
        // Original generation method (preserved for backward compatibility)
        if (input.empty()) {
            LLAMA_LOG("Error: input is empty");
            return "Error: Invalid generation parameters";
        }

        // Comprehensive validation of all generation components
        if (!validate_generation_components(context_info, context_info->model_info)) {
            return "Error: Model components not properly initialized or invalid context state";
        }// Validate and recover sampler if needed
        if (!validate_and_recover_sampler(context_info->model_info)) {
            return "Error: Sampler validation/recovery failed";
        }
        
        // Validate or attempt to recover logits
        float* logits = llama_get_logits(context_info->context);
        if (!logits) {
            if (context_info->n_past > 0) {
                // Context has been processed but no logits - attempt recovery
                if (!recover_logits(context_info, context_info->model_info)) {
                    return "Error: Context state invalid - no logits available and recovery failed";
                }
            } else {
                // n_past == 0, this might be normal for initial generation
                // We'll try to proceed and let the generation loop handle the decode
                LLAMA_LOG("Warning: No logits available at n_past=0, will attempt initial decode during generation");
            }
        }
          // Calculate available space for generation
        const int32_t max_new_tokens = std::min(context_info->model_info->n_predict, 
                                               context_info->model_info->n_ctx - context_info->n_past - LlamaResponseConstants::TOKEN_SAFETY_MARGIN);
        
        if (max_new_tokens <= 0) {
            return "Error: No space left in context for generation (context: " + 
                   std::to_string(context_info->n_past) + "/" + std::to_string(context_info->model_info->n_ctx) + ")";
        }

        LLAMA_LOG("Starting generation with " + std::to_string(max_new_tokens) + " max tokens, n_past=" + std::to_string(context_info->n_past));        // Initialize generation state
        std::string response;
        response.reserve(max_new_tokens * LlamaResponseConstants::STRING_RESERVE_MULTIPLIER);
        
        auto generation_start = std::chrono::high_resolution_clock::now();
        int32_t n_generated = 0;
        std::vector<llama_seq_id> seq_ids = {0};        // Main generation loop
        while (n_generated < max_new_tokens) {
            // Periodic sampler validation during long generation
            if (n_generated % ResponseConstants::SAMPLER_VALIDATION_INTERVAL == 0 && !context_info->model_info->sampler) {
                LLAMA_LOG("CRITICAL: Sampler became null during generation at token " + std::to_string(n_generated));
                return "Error: Sampler failed during generation";
            }
            
            // Sample next token
            llama_token new_token;
            try {
                new_token = llama_sampler_sample(context_info->model_info->sampler, context_info->context, -1);
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
            if (llama_vocab_is_eog(context_info->model_info->vocab, new_token)) {
                LLAMA_LOG("End of generation token encountered");
                break;
            }

            // Convert token to text
            std::string token_text = convert_token_to_text(new_token, context_info->model_info);
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
        }
        
        // Do NOT automatically update conversation history - let LlamaManager handle this
        // The linear flow in LlamaManager will add the response to conversation history
        if (!response.empty()) {
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
            context_info->clear_conversation();
            
            // Restore system message for next task
            if (!saved_system_message.empty()) {
                context_info->system_message = saved_system_message;
                context_info->add_message("system", saved_system_message);
                // Token count will be updated during next context processing
                LLAMA_LOG("Restored system message for next task");
            }
        }

        // Final sampler validation
        if (!context_info->model_info->sampler) {
            LLAMA_LOG("WARNING: Sampler is NULL at end of generation!");
        }

        return response;
    }
};

//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODING DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//
