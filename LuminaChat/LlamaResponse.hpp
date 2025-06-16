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
        
        // Initialize lookahead token history
        tokens_j_prev.clear();
        tokens_j_prev.resize(config.window_size, 0);
        
        tokens_j.clear();
        tokens_j.resize(config.ngram_size - 1);
        for (int32_t j = 0; j < config.ngram_size - 1; ++j) {
            tokens_j[j].resize(config.window_size);
            
            // Initialize with sequence of tokens (can be randomized if needed)
            for (int32_t i = 0; i < config.window_size; ++i) {
                tokens_j[j][i] = 100 + i; // Simple initialization
            }
        }
        
        LLAMA_LOG("Lookahead decoding initialized: W=" + std::to_string(config.window_size) + 
                  ", N=" + std::to_string(config.ngram_size) + 
                  ", G=" + std::to_string(config.max_verification));
    }    // Build lookahead batch with current token + verification n-grams + lookahead tokens
    template<typename BatchTokenAdder>
    bool build_lookahead_batch(llama_token current_token, int32_t n_past, ModelInfo* model_info,
                              BatchTokenAdder add_token_to_batch) const {
        const int32_t W = lookahead_config.window_size;
        const int32_t N = lookahead_config.ngram_size;
        const int32_t G = lookahead_config.max_verification;
        
        // Limit sequence count to avoid memory issues
        const int32_t max_sequences = std::min(W + G + 1, 32); // Cap at 32 sequences
        
        // Clear any existing batch state (callback should handle this)
        std::vector<llama_seq_id> seq_id_all;
        seq_id_all.reserve(max_sequences);
        for (int32_t i = 0; i < max_sequences; ++i) {
            seq_id_all.push_back(i);
        }
        
        // Add current token - belongs to all sequences
        if (!add_token_to_batch(current_token, n_past, seq_id_all, true)) {
            LLAMA_LOG("Error: Failed to add current token to batch");
            return false;
        }
        
        // Add verification n-grams (limited to available sequences)
        const int32_t g_cur = std::min(G, static_cast<int32_t>(ngrams_observed->cnt[current_token]));
        const int32_t available_verification_seqs = std::min(g_cur, max_sequences - W - 1);
        
        ngrams_cur.resize(available_verification_seqs);
        
        for (int32_t g = 0; g < available_verification_seqs; ++g) {
            ngrams_cur[g].active = true;
            ngrams_cur[g].seq_id = W + 1 + g; // Start verification sequences after lookahead
            ngrams_cur[g].tokens.resize(N);
            ngrams_cur[g].i_batch.resize(N);
            ngrams_cur[g].tokens[0] = current_token;
            ngrams_cur[g].i_batch[0] = 0; // Current token batch index
            
            // Add verification tokens for this n-gram
            auto ngram_tokens = ngrams_observed->get_ngram_tokens(current_token, g, N, G);
            for (int32_t j = 0; j < std::min(N - 1, static_cast<int32_t>(ngram_tokens.size())); ++j) {
                ngrams_cur[g].tokens[j + 1] = ngram_tokens[j];
                ngrams_cur[g].i_batch[j + 1] = 1 + g * (N - 1) + j; // Track batch position
                
                std::vector<llama_seq_id> verification_seq = {W + 1 + g};
                if (!add_token_to_batch(ngram_tokens[j], n_past + j + 1, verification_seq, true)) {
                    LLAMA_LOG("Warning: Failed to add verification token, continuing...");
                    ngrams_cur[g].active = false;
                    continue;
                }
            }
        }
        
        // Add lookahead tokens for the first level (fill remaining W-1 tokens)
        for (int32_t i = 1; i < W && i < max_sequences - 1; ++i) {
            std::vector<llama_seq_id> seq_id_look;
            seq_id_look.reserve(W - i);
            for (int32_t j = 0; j < W - i && (i + j) < max_sequences; ++j) {
                seq_id_look.push_back(i + j);
            }
            
            if (!seq_id_look.empty() && i < static_cast<int32_t>(tokens_j[0].size())) {
                if (!add_token_to_batch(tokens_j[0][i], n_past + i, seq_id_look, false)) {
                    LLAMA_LOG("Warning: Failed to add lookahead token at position " + std::to_string(i));
                }
            }
        }
        
        // Add tokens for remaining levels (simplified to avoid complexity)
        for (int32_t j = 1; j < N - 1 && j < static_cast<int32_t>(tokens_j.size()); ++j) {
            for (int32_t i = 0; i < W && i < static_cast<int32_t>(tokens_j[j].size()) && (i + 1) < max_sequences; ++i) {
                std::vector<llama_seq_id> level_seq = {i + 1};
                bool request_logits = (j == N - 2); // Request logits on final level
                
                if (!add_token_to_batch(tokens_j[j][i], n_past + j + i + 1, level_seq, request_logits)) {
                    // Log warning but continue
                    LLAMA_LOG("Warning: Failed to add level " + std::to_string(j) + " token at position " + std::to_string(i));
                }
            }
        }
        
        LLAMA_LOG("Built lookahead batch with " + std::to_string(available_verification_seqs) + " verification n-grams");
        return true;
    }    // Update lookahead tokens after sampling
    template<typename Sampler>
    void update_lookahead_tokens(ContextInfo* context_info, Sampler sample_func, 
                                int32_t verification_count, int32_t level) const {
        const int32_t W = lookahead_config.window_size;
        const int32_t N = lookahead_config.ngram_size;
        
        // Bounds checking for tokens_j
        if (tokens_j.empty() || tokens_j[0].empty()) {
            LLAMA_LOG("Warning: Lookahead tokens not properly initialized");
            return;
        }
        
        // Save previous tokens (with bounds checking)
        for (int32_t i = 0; i < W && i < static_cast<int32_t>(tokens_j_prev.size()) && 
             i < static_cast<int32_t>(tokens_j[0].size()); ++i) {
            tokens_j_prev[i] = tokens_j[0][i];
        }
        
        // Shift token levels
        for (int32_t j = 0; j < N - 2 && j + 1 < static_cast<int32_t>(tokens_j.size()); ++j) {
            if (tokens_j[j].size() == tokens_j[j + 1].size()) {
                tokens_j[j] = tokens_j[j + 1];
            }
        }
        
        // Generate new tokens for the last level
        if (N >= 2 && static_cast<size_t>(N - 2) < tokens_j.size()) {
            auto& last_level = tokens_j[N - 2];
            
            if (level == 0) {
                // Sample from the last level using available batch positions
                for (int32_t i = 0; i < W && i < static_cast<int32_t>(last_level.size()); ++i) {
                    // Calculate batch index more conservatively
                    int32_t batch_idx = verification_count * (N - 1) + i;
                    
                    try {
                        last_level[i] = sample_func(batch_idx);
                    } catch (...) {
                        // If sampling fails, use a simple fallback
                        last_level[i] = 100 + i; // Simple sequence
                        LLAMA_LOG("Warning: Failed to sample lookahead token, using fallback");
                    }
                }
            } else {
                // Initialize from previous level or with simple sequence
                if (!tokens_j.empty() && tokens_j[0].size() == last_level.size()) {
                    for (int32_t i = 0; i < W && i < static_cast<int32_t>(last_level.size()) && 
                         i < static_cast<int32_t>(tokens_j[0].size()); ++i) {
                        last_level[i] = tokens_j[0][i];
                    }
                } else {
                    // Fallback initialization
                    for (int32_t i = 0; i < W && i < static_cast<int32_t>(last_level.size()); ++i) {
                        last_level[i] = 100 + i;
                    }
                }
            }
        }
    }
      // Update observed n-grams with new patterns
    void update_observed_ngrams(llama_token first_token) const {
        const int32_t W = lookahead_config.window_size;
        const int32_t N = lookahead_config.ngram_size;
        
        std::vector<llama_token> ngram(N - 1);
        
        // Generate n-grams from lookahead window
        for (int32_t f = 0; f < W; ++f) {
            const llama_token ft = tokens_j_prev[f];
            
            for (int32_t j = 0; j < N - 1; ++j) {
                ngram[j] = tokens_j[j][f];
            }
            
            ngrams_observed->add_ngram(ft, ngram, N, lookahead_config.max_verification);
        }
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
        }        // Main lookahead generation loop
        while (n_generated < max_new_tokens) {
            // Check for end-of-generation token
            if (llama_vocab_is_eog(context_info->model_info->vocab, current_id)) {
                LLAMA_LOG("End of generation token encountered in lookahead");
                break;
            }
            
            // Build lookahead batch
            if (!build_lookahead_batch(current_id, context_info->n_past, 
                                     context_info->model_info, add_token_to_batch)) {
                LLAMA_LOG("Error: Failed to build lookahead batch");
                break;
            }
            
            // Decode the batch
            if (llama_decode(context_info->context, context_info->batch) != 0) {
                LLAMA_LOG("Error: Failed to decode lookahead batch");
                break;
            }
            
            // Process verification n-grams to find best accepted sequence
            llama_seq_id best_seq_id = 0;
            int32_t accepted_count = 0;
            int32_t best_verification_length = 0;
            
            // Check verification n-grams for matches
            for (size_t g = 0; g < ngrams_cur.size(); ++g) {
                if (!ngrams_cur[g].active) continue;
                
                int32_t match_length = 0;
                bool sequence_valid = true;
                
                // Check how many tokens in this n-gram match
                for (int32_t v = 1; v < config.ngram_size && sequence_valid; ++v) {
                    if (static_cast<size_t>(v) < ngrams_cur[g].i_batch.size()) {
                        int32_t i_batch = ngrams_cur[g].i_batch[v];
                        llama_token expected_token = ngrams_cur[g].tokens[v];
                        llama_token sampled_token = llama_sampler_sample(context_info->model_info->sampler, 
                                                                        context_info->context, i_batch);
                        
                        if (sampled_token == expected_token) {
                            match_length++;
                        } else {
                            sequence_valid = false;
                        }
                    } else {
                        sequence_valid = false;
                    }
                }
                
                // Track best matching sequence
                if (sequence_valid && match_length > best_verification_length) {
                    best_verification_length = match_length;
                    best_seq_id = ngrams_cur[g].seq_id;
                    accepted_count = match_length;
                }
            }
            
            // Accept verified tokens or generate new token
            if (accepted_count > 0 && best_seq_id != 0) {
                // Accept the verified sequence
                for (int32_t v = 0; v < accepted_count && n_generated < max_new_tokens; ++v) {
                    // Find the correct n-gram and batch index
                    for (size_t g = 0; g < ngrams_cur.size(); ++g) {
                        if (ngrams_cur[g].active && ngrams_cur[g].seq_id == best_seq_id) {
                            if (static_cast<size_t>(v + 1) < ngrams_cur[g].tokens.size()) {
                                current_id = ngrams_cur[g].tokens[v + 1];
                                llama_sampler_accept(context_info->model_info->sampler, current_id);
                                
                                // Convert and add to response
                                token_text = convert_token_to_text(current_id, context_info->model_info);
                                if (!token_text.empty()) {
                                    response += token_text;
                                }
                                
                                n_generated++;
                                context_info->n_past++;
                                n_accept++;
                                break;
                            }
                        }
                    }
                }
                
                LLAMA_LOG("Accepted " + std::to_string(accepted_count) + " tokens via verification");
            } else {
                // No verification match, sample one new token normally
                current_id = llama_sampler_sample(context_info->model_info->sampler, context_info->context, -1);
                llama_sampler_accept(context_info->model_info->sampler, current_id);
                
                // Convert and add to response
                token_text = convert_token_to_text(current_id, context_info->model_info);
                if (!token_text.empty()) {
                    response += token_text;
                }
                
                n_generated++;
                context_info->n_past++;
            }
            
            // Update lookahead tokens for next iteration
            auto sampler_func = [context_info](int32_t batch_idx) -> llama_token {
                return llama_sampler_sample(context_info->model_info->sampler, context_info->context, batch_idx);
            };
            
            update_lookahead_tokens(context_info, sampler_func, static_cast<int32_t>(ngrams_cur.size()), 0);
            
            // Update observed n-grams
            update_observed_ngrams(current_id);
            
            // Reset verification state for next iteration
            for (auto& ngram : ngrams_cur) {
                ngram.reset();
            }
            
            // Handle KV cache management - only keep main sequence
            auto* memory = llama_get_memory(context_info->context);
            if (memory) {
                // Remove all auxiliary sequences, keep only sequence 0
                for (int32_t s = 1; s < config.window_size + config.max_verification + 1; ++s) {
                    llama_memory_seq_rm(memory, s, -1, -1);
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
