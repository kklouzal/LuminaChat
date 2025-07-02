#pragma once

#include "BatchManager.hpp"
#include "../Logger.hpp"
#include "../ModelInfo.hpp"
#include "../TokenCache.hpp"
#include "../Utilities.hpp"
#include "llama-cpp.h"
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

// Forward declarations
struct llama_context;

// Performance optimization constants - aligned with BatchManager optimizations
namespace ContextIOConstants {
    static constexpr bool ENABLE_DEBUG_LOGGING = false;  // Compile-time debug control
    static constexpr size_t CACHE_LINE_SIZE = 64;        // For alignment optimizations
    static constexpr int32_t MIN_GENERATION_TOKENS = 128; // Minimum tokens to generate
    static constexpr int32_t MAX_GENERATION_TOKENS = 2048; // Maximum tokens per generation
    static constexpr int32_t USER_MESSAGE_RESERVE = 128;   // Reserve space for user's next message
    static constexpr int32_t VOCAB_SEARCH_UNROLL = 8;     // Loop unrolling for vocab search
}

// Callback types for streaming generation
using TokenCallback = std::function<void(const std::string& token_text)>;
using GenerationCompleteCallback = std::function<void(const std::string& full_response, bool success)>;
using GenerationErrorCallback = std::function<void(const std::string& error_message)>;

struct GenerationCallbacks {
    TokenCallback on_token;
    GenerationCompleteCallback on_complete;
    GenerationErrorCallback on_error;
    
    GenerationCallbacks() = default;
    GenerationCallbacks(TokenCallback token_cb, GenerationCompleteCallback complete_cb, GenerationErrorCallback error_cb = nullptr)
        : on_token(std::move(token_cb)), on_complete(std::move(complete_cb)), on_error(std::move(error_cb)) {}
};

/**
 * ContextInputOutput: Orchestrates the complete generation pipeline
 * 
 * Responsibilities:
 * - Coordinate the entire generation workflow from input to output
 * - Manage prompt processing (input phase) through inherited BatchManager functionality
 * - Handle response generation (output phase) through inherited BatchManager functionality
 * - Provide streaming and async generation capabilities
 * - Manage generation threads and callbacks
 * 
 * Generation Pipeline:
 * 1. Input Phase:    Tokenize and process prompt tokens
 * 2. Processing:     Context rebuilding and validation
 * 3. Output Phase:   Generate response tokens with streaming
 * 4. Finalization:   Detokenize and clean response
 * 
 * The complete "Generation Output" encompasses both input processing
 * and response generation, with inherited BatchManager handling all token operations.
 */
class ContextInputOutput : public BatchManager {
private:
    // Core components - cache-aligned for performance
    alignas(ContextIOConstants::CACHE_LINE_SIZE) ModelInfo* parent_model = nullptr;
    TokenCache* token_cache = nullptr;
    llama_context* llama_ctx = nullptr;
    
    // Context state (references to parent ContextInfo state)
    int32_t& n_past_ref;
    size_t max_context_tokens = 0;
    
    // Generation state management - optimized atomic operations
    alignas(16) std::atomic<bool> is_generating{false};
    std::atomic<bool> should_stop_generation{false};
    std::unique_ptr<std::thread> generation_thread;
    std::condition_variable generation_cv;
    std::mutex generation_mutex;
    
    // Thread safety for generation operations (reduced scope)
    mutable std::mutex io_mutex;
    
    // Cached generation parameters for performance
    mutable std::atomic<int32_t> cached_vocab_size{0};
    mutable std::atomic<int32_t> cached_eos_token{0};
    
    // Internal generation helpers - optimized versions
    [[nodiscard]] bool ProcessPromptPhaseOptimized(const std::string& prompt) noexcept;
    [[nodiscard]] std::string ExecuteGenerationPhaseOptimized(const GenerationCallbacks& callbacks) noexcept;
    [[nodiscard]] [[msvc::forceinline]] bool ValidateGenerationStateOptimized() const noexcept;
    
    // Ultra-fast token processing helpers
    [[nodiscard]] [[msvc::forceinline]] bool ProcessPromptTokensOptimized(std::vector<int32_t>&& tokens) noexcept;
    [[nodiscard]] [[msvc::forceinline]] int32_t FindBestTokenOptimized(const float* logits) const noexcept;
    
    // Internal token processing (caller must hold io_mutex)
    bool ProcessPromptTokensInternal(const std::vector<int32_t>& tokens);
    bool ProcessPromptTokensInternal(std::vector<int32_t>&& tokens); // Move semantics overload
    
public:
    // Constructor/Destructor
    ContextInputOutput(ModelInfo* model, TokenCache* cache, llama_context* ctx, 
                      int32_t& n_past_reference, size_t max_tokens);
    ~ContextInputOutput();
    
    // Generation pipeline orchestration - optimized versions
    std::string GenerateResponse(const std::string& prompt);
    bool GenerateResponseAsync(const std::string& prompt, const GenerationCallbacks& callbacks);
    
    // Generation control
    void StopGeneration();
    [[nodiscard]] [[msvc::forceinline]] bool IsGenerating() const noexcept { return is_generating.load(std::memory_order_relaxed); }
    
    // Context management - optimized
    [[nodiscard]] bool InitializeLlamaContext();
    [[nodiscard]] bool ProcessPromptTokens(const std::vector<int32_t>& tokens);
    [[msvc::forceinline]] void UpdateContextBounds(size_t new_max_tokens) noexcept;
    
    // State access - optimized getters
    [[nodiscard]] [[msvc::forceinline]] constexpr size_t GetMaxContextTokens() const noexcept { return max_context_tokens; }
    [[nodiscard]] [[msvc::forceinline]] int32_t GetCurrentPosition() const noexcept { return n_past_ref; }
    
    // Performance helper - initialize cached values
    void InitializeCachedValues() noexcept;
    
    // Disable copy/move operations (manages complex state and threads)
    ContextInputOutput(const ContextInputOutput&) = delete;
    ContextInputOutput& operator=(const ContextInputOutput&) = delete;
    ContextInputOutput(ContextInputOutput&&) = delete;
    ContextInputOutput& operator=(ContextInputOutput&&) = delete;
};

// Inline implementation
inline ContextInputOutput::ContextInputOutput(ModelInfo* model, TokenCache* cache, llama_context* ctx,
                                             int32_t& n_past_reference, size_t max_tokens)
    : BatchManager(ctx, max_tokens, n_past_reference)
    , parent_model(model)
    , token_cache(cache)
    , llama_ctx(ctx)
    , n_past_ref(n_past_reference)
    , max_context_tokens(max_tokens)
{
    if (!parent_model) [[unlikely]] {
        if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
            LOG_ERROR("ContextInputOutput", "Created with null ModelInfo");
        }
        return;
    }
    
    if (!token_cache) [[unlikely]] {
        if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
            LOG_ERROR("ContextInputOutput", "Created with null TokenCache");
        }
        return;
    }
    
    if (!llama_ctx) [[unlikely]] {
        if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
            LOG_ERROR("ContextInputOutput", "Created with null llama context");
        }
        return;
    }
    
    // Initialize cached values for performance
    InitializeCachedValues();
    
    if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
        LOG_DEBUG("ContextInputOutput", "Created with max_tokens: " + std::to_string(max_tokens));
    }
}

inline ContextInputOutput::~ContextInputOutput() {
    // Stop any ongoing generation
    should_stop_generation = true;
    
    // Wait for generation thread to complete
    if (generation_thread && generation_thread->joinable()) {
        generation_cv.notify_all();
        generation_thread->join();
    }
    
    std::lock_guard<std::mutex> lock(io_mutex);
    
    if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
        LOG_DEBUG("ContextInputOutput", "Destroyed");
    }
}

inline void ContextInputOutput::InitializeCachedValues() noexcept {
    if (parent_model) [[likely]] {
        const llama_vocab* vocab = parent_model->GetVocab();
        if (vocab) [[likely]] {
            cached_vocab_size.store(llama_vocab_n_tokens(vocab), std::memory_order_relaxed);
            cached_eos_token.store(llama_vocab_eos(vocab), std::memory_order_relaxed);
        }
    }
}

inline bool ContextInputOutput::InitializeLlamaContext() {
    std::lock_guard<std::mutex> lock(io_mutex);
    
    if (!llama_ctx) [[unlikely]] {
        if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
            LOG_ERROR("ContextInputOutput", "Llama context is null");
        }
        return false;
    }
    
    // Ensure batch is properly initialized
    if (!IsBatchInitialized()) [[unlikely]] {
        if (!InitializeBatch()) [[unlikely]] {
            if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
                LOG_ERROR("ContextInputOutput", "Failed to initialize batch");
            }
            return false;
        }
    }
    
    // Initialize cached values if not already done
    if (cached_vocab_size.load(std::memory_order_relaxed) == 0) [[unlikely]] {
        InitializeCachedValues();
    }
    
    return true;
}

inline bool ContextInputOutput::ProcessPromptTokens(const std::vector<int32_t>& tokens) {
    if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
        LOG_DEBUG("ContextInputOutput", "ProcessPromptTokens - acquiring io_mutex");
    }
    std::lock_guard<std::mutex> lock(io_mutex);
    if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
        LOG_DEBUG("ContextInputOutput", "ProcessPromptTokens - acquired io_mutex");
        LOG_DEBUG("ContextInputOutput", "Processing " + std::to_string(tokens.size()) + " prompt tokens");
    }
    
    // Use optimized processing with move semantics
    std::vector<int32_t> tokens_copy = tokens; // Copy for move
    bool result = ProcessPromptTokensOptimized(std::move(tokens_copy));
    
    if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
        LOG_DEBUG("ContextInputOutput", "ProcessPromptTokens - batch processing result: " + std::to_string(result));
    }
    
    return result;
}

// Optimized internal token processing leveraging BatchManager optimizations
[[msvc::forceinline]] inline bool ContextInputOutput::ProcessPromptTokensOptimized(std::vector<int32_t>&& tokens) noexcept {
    if (tokens.empty()) [[unlikely]] {
        return true;
    }
    
    if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
        LOG_DEBUG("ContextInputOutput", "Processing " + std::to_string(tokens.size()) + " prompt tokens (optimized)");
    }
    
    // Use BatchManager's optimized move semantics processing
    bool result = ProcessTokensBatch(std::move(tokens));
    
    if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
        LOG_DEBUG("ContextInputOutput", "ProcessPromptTokensOptimized - batch processing result: " + std::to_string(result));
    }
    
    return result;
}

// Optimized prompt processing phase leveraging BatchManager template validation
[[nodiscard]] inline bool ContextInputOutput::ProcessPromptPhaseOptimized(const std::string& prompt) noexcept {
    if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
        LOG_DEBUG("ContextInputOutput", "ProcessPromptPhaseOptimized - starting with prompt length: " + std::to_string(prompt.length()));
        LOG_DEBUG("ContextInputOutput", "ProcessPromptPhaseOptimized - initial n_past_ref: " + std::to_string(n_past_ref));
        LOG_DEBUG("ContextInputOutput", "ProcessPromptPhaseOptimized - prompt content: \"" + prompt.substr(0, 200) + 
                  (prompt.length() > 200 ? "..." : "") + "\"");
    }
    
    if (!token_cache) [[unlikely]] {
        if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
            LOG_ERROR("ContextInputOutput", "TokenCache not available for prompt processing");
        }
        return false;
    }
    
    // Ensure batch is initialized before processing tokens (caller must hold io_mutex)
    if (!IsBatchInitialized()) [[unlikely]] {
        if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
            LOG_DEBUG("ContextInputOutput", "ProcessPromptPhaseOptimized - initializing batch");
        }
        if (!InitializeBatch()) [[unlikely]] {
            if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
                LOG_ERROR("ContextInputOutput", "ProcessPromptPhaseOptimized - failed to initialize batch");
            }
            return false;
        }
    }
    
    if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
        LOG_DEBUG("ContextInputOutput", "ProcessPromptPhaseOptimized - tokenizing prompt");
    }
    
    // Tokenize the prompt
    std::vector<int32_t> prompt_tokens = token_cache->TokenizeText(prompt, true);
    
    if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
        LOG_DEBUG("ContextInputOutput", "Prompt tokenized to " + std::to_string(prompt_tokens.size()) + " tokens");
        LOG_DEBUG("ContextInputOutput", "Current n_past_ref before processing: " + std::to_string(n_past_ref));
    }
    
    // Process prompt tokens through optimized batch manager using move semantics (caller must hold io_mutex)
    if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
        LOG_DEBUG("ContextInputOutput", "ProcessPromptPhaseOptimized - calling ProcessPromptTokensOptimized with move semantics");
    }
    bool result = ProcessPromptTokensOptimized(std::move(prompt_tokens));
    if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
        LOG_DEBUG("ContextInputOutput", "ProcessPromptPhaseOptimized - ProcessPromptTokensOptimized returned: " + std::to_string(result));
        LOG_DEBUG("ContextInputOutput", "Current n_past_ref after processing: " + std::to_string(n_past_ref));
    }
    
    return result;
}

// Ultra-optimized token selection leveraging cached atomic values
[[nodiscard]] [[msvc::forceinline]] inline int32_t ContextInputOutput::FindBestTokenOptimized(const float* logits) const noexcept {
    // Use cached atomic value for performance
    const int32_t vocab_size = cached_vocab_size.load(std::memory_order_relaxed);
    if (vocab_size <= 0) [[unlikely]] {
        return 0;
    }
    
    // Optimized greedy sampling with loop unrolling hint
    int32_t best_token = 0;
    float max_logit = logits[0];
    
    // Unroll loop for better performance on common vocab sizes
    constexpr int32_t unroll_count = ContextIOConstants::VOCAB_SEARCH_UNROLL;
    const int32_t unrolled_end = (vocab_size / unroll_count) * unroll_count;
    
    // Unrolled loop for bulk processing
    for (int32_t i = 1; i < unrolled_end; i += unroll_count) {
        #pragma unroll 8
        for (int32_t j = 0; j < unroll_count; ++j) {
            const int32_t idx = i + j;
            if (logits[idx] > max_logit) [[unlikely]] {
                max_logit = logits[idx];
                best_token = idx;
            }
        }
    }
    
    // Handle remaining tokens
    for (int32_t i = unrolled_end; i < vocab_size; ++i) {
        if (logits[i] > max_logit) [[unlikely]] {
            max_logit = logits[i];
            best_token = i;
        }
    }
    
    return best_token;
}

// Ultra-optimized generation phase leveraging all BatchManager optimizations
[[nodiscard]] inline std::string ContextInputOutput::ExecuteGenerationPhaseOptimized(const GenerationCallbacks& callbacks) noexcept {
    if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
        LOG_DEBUG("ContextInputOutput", "ExecuteGenerationPhaseOptimized - starting generation phase");
    }
    
    if (!token_cache || !parent_model) [[unlikely]] {
        if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
            LOG_ERROR("ContextInputOutput", "Required components not available for generation");
        }
        return "";
    }
    
    std::vector<int32_t> response_tokens;
    
    try {
        // Calculate generation limits with cached atomic safety limits
        const int generation_buffer = ContextConstants::GENERATION_BUFFER_TOKENS;
        const int total_reserve = generation_buffer + ContextIOConstants::USER_MESSAGE_RESERVE;
        
        const int available_space = static_cast<int>(max_context_tokens) - n_past_ref - total_reserve;
        const int max_new_tokens = std::max(ContextIOConstants::MIN_GENERATION_TOKENS, 
                                          std::min(ContextIOConstants::MAX_GENERATION_TOKENS, available_space));
        
        if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
            LOG_DEBUG("ContextInputOutput", "Generation limits: max_new_tokens=" + std::to_string(max_new_tokens) + 
                     ", available_space=" + std::to_string(available_space) + 
                     ", current_position=" + std::to_string(n_past_ref));
        }
        
        // Get cached EOS token for fast comparison
        const int32_t eos_token = cached_eos_token.load(std::memory_order_relaxed);
        
        // Ultra-optimized generation loop
        for (int i = 0; i < max_new_tokens && !should_stop_generation.load(std::memory_order_relaxed); ++i) {
            
            // Get logits for next token prediction
            float* logits = llama_get_logits_ith(llama_ctx, GetBatchSize() - 1);
            if (!logits) [[unlikely]] {
                if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
                    LOG_ERROR("ContextInputOutput", "Failed to get logits for token generation");
                }
                break;
            }
            
            // Ultra-fast token selection using optimized search
            const int32_t next_token = FindBestTokenOptimized(logits);
            
            // Fast EOS check using cached value
            if (next_token == eos_token) [[unlikely]] {
                if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
                    LOG_DEBUG("ContextInputOutput", "EOS token generated, stopping generation");
                }
                break;
            }
            
            // Use BatchManager's cached safety limit for ultra-fast bounds checking
            if (HasErrorState()) [[unlikely]] {
                if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
                    LOG_DEBUG("ContextInputOutput", "Approaching context limit during generation, stopping early");
                }
                break;
            }
            
            response_tokens.push_back(next_token);
            
            // Stream only the new token text to prevent exponential duplication
            if (callbacks.on_token) [[likely]] {
                std::vector<int32_t> single_token = {next_token};
                std::string new_token_text = token_cache->DetokenizeTokens(single_token);
                
                if (!new_token_text.empty()) [[likely]] {
                    callbacks.on_token(new_token_text);
                }
            }
            
            // Process the generated token using ultra-fast BatchManager method
            {
                std::lock_guard<std::mutex> lock(io_mutex);
                
                // Use BatchManager's ProcessSingleTokenFast - ultra-optimized hot path
                // Preconditions are guaranteed: batch initialized, context valid, bounds checked above
                if (!ProcessSingleTokenFast(next_token, true)) [[unlikely]] {
                    if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
                        LOG_ERROR("ContextInputOutput", "Failed to process generated token");
                    }
                    break;
                }
            }
        }
        
        // Detokenize the complete token sequence for the final response
        // This ensures proper character encoding and prevents truncation issues
        std::string full_response;
        if (!response_tokens.empty()) [[likely]] {
            full_response = token_cache->DetokenizeTokens(response_tokens);
        }
        
        // Extract clean response content (remove any template artifacts)
        std::string clean_response = LuminaChat::Utilities::ExtractCleanResponse(full_response);
        
        if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
            LOG_DEBUG("ContextInputOutput", "Generated " + std::to_string(response_tokens.size()) + 
                     " tokens, final response length: " + std::to_string(clean_response.length()));
        }
        
        return clean_response;
        
    } catch (const std::exception& e) {
        if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
            LOG_ERROR("ContextInputOutput", "Exception during generation phase: " + std::string(e.what()));
        }
        if (callbacks.on_error) {
            callbacks.on_error("Generation failed: " + std::string(e.what()));
        }
        return "";
    }
}

// Optimized validation leveraging BatchManager's template-based validation
[[nodiscard]] [[msvc::forceinline]] inline bool ContextInputOutput::ValidateGenerationStateOptimized() const noexcept {
    if (!token_cache || !parent_model || !llama_ctx) [[unlikely]] {
        if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
            LOG_ERROR("ContextInputOutput", "Required components not available");
        }
        return false;
    }
    
    // Use BatchManager's optimized context state validation
    if (!ValidateContextState()) [[unlikely]] {
        if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
            LOG_ERROR("ContextInputOutput", "BatchManager context state validation failed");
        }
        return false;
    }
    
    // Check if batch is initialized for generation (lock-free check)
    if (!IsBatchInitialized()) [[unlikely]] {
        if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
            LOG_DEBUG("ContextInputOutput", "Batch not initialized during validation - will be initialized during processing");
        }
        // This is not an error - batch will be initialized when needed
    }
    
    return true;
}

inline std::string ContextInputOutput::GenerateResponse(const std::string& prompt) {
    // DEPRECATED FOR UI CONTEXTS: This method blocks and should only be used for non-UI contexts
    // like Discord bot responses or batch processing. For UI contexts, use GenerateResponseAsync instead.
    
    if (is_generating.load(std::memory_order_relaxed)) [[unlikely]] {
        if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
            LOG_WARNING("ContextInputOutput", "Generation already in progress");
        }
        return "Error: Generation already in progress";
    }
    
    if (!ValidateGenerationStateOptimized()) [[unlikely]] {
        if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
            LOG_ERROR("ContextInputOutput", "Generation state validation failed");
        }
        return "Error: Invalid generation state";
    }
    
    is_generating.store(true, std::memory_order_relaxed);
    should_stop_generation.store(false, std::memory_order_relaxed);
    
    try {
        // Phase 1: Process prompt tokens (Input Phase) - acquire mutex only for this phase
        {
            std::lock_guard<std::mutex> lock(io_mutex);
            if (!ProcessPromptPhaseOptimized(prompt)) [[unlikely]] {
                is_generating.store(false, std::memory_order_relaxed);
                return "Error: Failed to process prompt";
            }
        }
        
        // Phase 2: Generate response tokens (Output Phase) - ExecuteGenerationPhaseOptimized manages its own mutex
        GenerationCallbacks empty_callbacks; // No streaming for synchronous generation
        std::string response = ExecuteGenerationPhaseOptimized(empty_callbacks);
        
        is_generating.store(false, std::memory_order_relaxed);
        return response;
        
    } catch (const std::exception& e) {
        is_generating.store(false, std::memory_order_relaxed);
        if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
            LOG_ERROR("ContextInputOutput", "Exception in GenerateResponse: " + std::string(e.what()));
        }
        return "Error: " + std::string(e.what());
    }
}

inline bool ContextInputOutput::GenerateResponseAsync(const std::string& prompt, const GenerationCallbacks& callbacks) {
    // Check if already generating using optimized atomic operation
    if (is_generating.load(std::memory_order_relaxed)) [[unlikely]] {
        if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
            LOG_WARNING("ContextInputOutput", "Generation already in progress");
        }
        if (callbacks.on_error) {
            callbacks.on_error("Generation already in progress");
        }
        return false;
    }
    
    // Validate state before starting async operation using optimized validation
    if (!ValidateGenerationStateOptimized()) [[unlikely]] {
        if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
            LOG_ERROR("ContextInputOutput", "Generation state validation failed");
        }
        if (callbacks.on_error) {
            callbacks.on_error("Invalid generation state");
        }
        return false;
    }
    
    // Stop any existing generation thread
    if (generation_thread && generation_thread->joinable()) {
        should_stop_generation.store(true, std::memory_order_relaxed);
        generation_cv.notify_all();
        generation_thread->join();
    }
    
    // Reset generation state using optimized atomic operations
    should_stop_generation.store(false, std::memory_order_relaxed);
    is_generating.store(true, std::memory_order_relaxed);
    
    // Start new generation thread with optimized processing
    generation_thread = std::make_unique<std::thread>([this, prompt, callbacks]() {
        std::string full_response;
        bool success = false;
        
        try {
            // Phase 1: Process prompt tokens (Input Phase) using optimized processing
            {
                std::lock_guard<std::mutex> lock(io_mutex);
                
                if (!ProcessPromptPhaseOptimized(prompt)) [[unlikely]] {
                    if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
                        LOG_ERROR("ContextInputOutput", "Failed to process prompt in async generation");
                    }
                    if (callbacks.on_error) {
                        callbacks.on_error("Failed to process prompt");
                    }
                    is_generating.store(false, std::memory_order_relaxed);
                    return;
                }
            }
            
            // Phase 2: Generate response tokens (Output Phase) using ultra-optimized generation
            full_response = ExecuteGenerationPhaseOptimized(callbacks);
            success = !should_stop_generation.load(std::memory_order_relaxed) && !full_response.empty();
            
        } catch (const std::exception& e) {
            if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
                LOG_ERROR("ContextInputOutput", "Exception in async generation: " + std::string(e.what()));
            }
            if (callbacks.on_error) {
                callbacks.on_error("Generation failed: " + std::string(e.what()));
            }
        }
        
        // Update state using optimized atomic operation
        is_generating.store(false, std::memory_order_relaxed);
        
        // Call completion callback with the final, clean response
        if (callbacks.on_complete) {
            callbacks.on_complete(full_response, success);
        }
    });
    
    return true;
}

inline void ContextInputOutput::StopGeneration() {
    should_stop_generation.store(true, std::memory_order_relaxed);
    
    // Notify generation thread to wake up and check the stop flag
    {
        std::lock_guard<std::mutex> lock(generation_mutex);
        generation_cv.notify_all();
    }
    
    // Wait for generation thread to complete
    if (generation_thread && generation_thread->joinable()) {
        generation_thread->join();
        generation_thread.reset();
    }
    
    is_generating.store(false, std::memory_order_relaxed);
    if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
        LOG_DEBUG("ContextInputOutput", "Generation stopped");
    }
}

[[msvc::forceinline]] inline void ContextInputOutput::UpdateContextBounds(size_t new_max_tokens) noexcept {
    std::lock_guard<std::mutex> lock(io_mutex);
    
    max_context_tokens = new_max_tokens;
    
    // Update inherited BatchManager bounds using optimized method
    BatchManager::UpdateContextBounds(new_max_tokens);
    
    if constexpr (ContextIOConstants::ENABLE_DEBUG_LOGGING) {
        LOG_DEBUG("ContextInputOutput", "Updated context bounds to: " + std::to_string(new_max_tokens));
    }
}
