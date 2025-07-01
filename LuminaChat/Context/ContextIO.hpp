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
 * - Manage prompt processing (input phase) through BatchManager
 * - Handle response generation (output phase) through BatchManager
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
 * and response generation, with BatchManager handling all token operations.
 */
class ContextInputOutput {
private:
    // Core components
    std::unique_ptr<BatchManager> batch_manager;
    ModelInfo* parent_model = nullptr;
    TokenCache* token_cache = nullptr;
    llama_context* llama_ctx = nullptr;
    
    // Context state (references to parent ContextInfo state)
    int32_t& n_past_ref;
    size_t max_context_tokens = 0;
    
    // Generation state management
    std::atomic<bool> is_generating{false};
    std::atomic<bool> should_stop_generation{false};
    std::unique_ptr<std::thread> generation_thread;
    std::condition_variable generation_cv;
    std::mutex generation_mutex;
    
    // Thread safety for generation operations
    mutable std::mutex io_mutex;
    
    // Internal generation helpers
    bool ProcessPromptPhase(const std::string& prompt);
    std::string ExecuteGenerationPhase(const GenerationCallbacks& callbacks);
    bool ValidateGenerationState() const;
    
public:
    // Constructor/Destructor
    ContextInputOutput(ModelInfo* model, TokenCache* cache, llama_context* ctx, 
                      int32_t& n_past_reference, size_t max_tokens);
    ~ContextInputOutput();
    
    // Generation pipeline orchestration
    std::string GenerateResponse(const std::string& prompt);
    bool GenerateResponseAsync(const std::string& prompt, const GenerationCallbacks& callbacks);
    
    // Generation control
    void StopGeneration();
    bool IsGenerating() const { return is_generating.load(); }
    
    // Context management
    bool InitializeLlamaContext();
    bool ProcessPromptTokens(const std::vector<int32_t>& tokens);
    void UpdateContextBounds(size_t new_max_tokens);
    
    // State access
    size_t GetMaxContextTokens() const { return max_context_tokens; }
    int32_t GetCurrentPosition() const { return n_past_ref; }
    BatchManager* GetBatchManager() { return batch_manager.get(); }
    
    // Disable copy/move operations (manages complex state and threads)
    ContextInputOutput(const ContextInputOutput&) = delete;
    ContextInputOutput& operator=(const ContextInputOutput&) = delete;
    ContextInputOutput(ContextInputOutput&&) = delete;
    ContextInputOutput& operator=(ContextInputOutput&&) = delete;
};

// Inline implementation
inline ContextInputOutput::ContextInputOutput(ModelInfo* model, TokenCache* cache, llama_context* ctx,
                                             int32_t& n_past_reference, size_t max_tokens)
    : parent_model(model)
    , token_cache(cache)
    , llama_ctx(ctx)
    , n_past_ref(n_past_reference)
    , max_context_tokens(max_tokens)
{
    if (!parent_model) {
        LOG_ERROR("ContextInputOutput", "Created with null ModelInfo");
        return;
    }
    
    if (!token_cache) {
        LOG_ERROR("ContextInputOutput", "Created with null TokenCache");
        return;
    }
    
    if (!llama_ctx) {
        LOG_ERROR("ContextInputOutput", "Created with null llama context");
        return;
    }
    
    // Create BatchManager for token operations
    batch_manager = std::make_unique<BatchManager>(llama_ctx, max_tokens, n_past_ref);
    
    if (!batch_manager->InitializeBatch()) {
        LOG_ERROR("ContextInputOutput", "Failed to initialize BatchManager");
        batch_manager.reset();
        return;
    }
    
    LOG_DEBUG("ContextInputOutput", "Created with max_tokens: " + std::to_string(max_tokens));
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
    
    // BatchManager destructor will handle cleanup automatically
    batch_manager.reset();
    
    LOG_DEBUG("ContextInputOutput", "Destroyed");
}

inline bool ContextInputOutput::InitializeLlamaContext() {
    std::lock_guard<std::mutex> lock(io_mutex);
    
    if (!llama_ctx) {
        LOG_ERROR("ContextInputOutput", "Llama context is null");
        return false;
    }
    
    if (!batch_manager) {
        LOG_ERROR("ContextInputOutput", "BatchManager is null");
        return false;
    }
    
    // Ensure batch is properly initialized
    if (!batch_manager->IsBatchInitialized()) {
        if (!batch_manager->InitializeBatch()) {
            LOG_ERROR("ContextInputOutput", "Failed to initialize batch");
            return false;
        }
    }
    
    return true;
}

inline bool ContextInputOutput::ProcessPromptTokens(const std::vector<int32_t>& tokens) {
    std::lock_guard<std::mutex> lock(io_mutex);
    
    if (!batch_manager) {
        LOG_ERROR("ContextInputOutput", "BatchManager not available for prompt processing");
        return false;
    }
    
    LOG_DEBUG("ContextInputOutput", "Processing " + std::to_string(tokens.size()) + " prompt tokens");
    
    return batch_manager->ProcessTokensBatch(tokens);
}

inline bool ContextInputOutput::ProcessPromptPhase(const std::string& prompt) {
    if (!token_cache) {
        LOG_ERROR("ContextInputOutput", "TokenCache not available for prompt processing");
        return false;
    }
    
    // Tokenize the prompt
    std::vector<int32_t> prompt_tokens = token_cache->TokenizeText(prompt, true);
    
    LOG_DEBUG("ContextInputOutput", "Prompt tokenized to " + std::to_string(prompt_tokens.size()) + " tokens");
    
    // Process prompt tokens through batch manager
    return ProcessPromptTokens(prompt_tokens);
}

inline std::string ContextInputOutput::ExecuteGenerationPhase(const GenerationCallbacks& callbacks) {
    if (!batch_manager || !token_cache || !parent_model) {
        LOG_ERROR("ContextInputOutput", "Required components not available for generation");
        return "";
    }
    
    std::vector<int32_t> response_tokens;
    std::string full_response;
    
    try {
        // Calculate generation limits with proper buffer management
        const int generation_buffer = ContextConstants::GENERATION_BUFFER_TOKENS;
        const int user_message_reserve = 128; // Reserve space for user's next message
        const int total_reserve = generation_buffer + user_message_reserve;
        
        const int available_space = static_cast<int>(max_context_tokens) - n_past_ref - total_reserve;
        const int max_new_tokens = std::max(128, std::min(2048, available_space));
        
        LOG_DEBUG("ContextInputOutput", "Generation limits: max_new_tokens=" + std::to_string(max_new_tokens) + 
                 ", available_space=" + std::to_string(available_space) + 
                 ", current_position=" + std::to_string(n_past_ref));
        
        // Generate tokens one by one with streaming
        for (int i = 0; i < max_new_tokens && !should_stop_generation.load(); ++i) {
            // Get logits for next token prediction
            float* logits = llama_get_logits_ith(llama_ctx, batch_manager->GetBatchSize() - 1);
            if (!logits) {
                LOG_ERROR("ContextInputOutput", "Failed to get logits for token generation");
                break;
            }
            
            // Simple greedy sampling (take highest probability token)
            const llama_vocab* vocab = parent_model->GetVocab();
            int32_t vocab_size = llama_vocab_n_tokens(vocab);
            llama_token next_token = 0;
            float max_logit = logits[0];
            
            for (int32_t j = 1; j < vocab_size; ++j) {
                if (logits[j] > max_logit) {
                    max_logit = logits[j];
                    next_token = j;
                }
            }
            
            // Check for end-of-sequence token
            if (next_token == llama_vocab_eos(vocab)) {
                LOG_DEBUG("ContextInputOutput", "EOS token generated, stopping generation");
                break;
            }
            
            // Check context limits during generation
            const int32_t current_usage = n_past_ref + static_cast<int32_t>(response_tokens.size());
            if (current_usage >= static_cast<int32_t>(max_context_tokens) - generation_buffer) {
                LOG_DEBUG("ContextInputOutput", "Approaching context limit during generation, stopping early");
                break;
            }
            
            response_tokens.push_back(next_token);
            
            // Convert token to text for streaming
            std::vector<int32_t> single_token = {next_token};
            std::string token_text = token_cache->DetokenizeTokens(single_token);
            
            // Stream the token if callback is provided
            if (callbacks.on_token && !token_text.empty()) {
                callbacks.on_token(token_text);
            }
            
            full_response += token_text;
            
            // Process the generated token for next iteration
            {
                std::lock_guard<std::mutex> lock(io_mutex);
                if (!batch_manager->ProcessSingleToken(next_token, true)) {
                    LOG_ERROR("ContextInputOutput", "Failed to process generated token");
                    break;
                }
            }
        }
        
        // Extract clean response content (remove any template artifacts)
        std::string clean_response = LuminaChat::Utilities::ExtractCleanResponse(full_response);
        
        LOG_DEBUG("ContextInputOutput", "Generated " + std::to_string(response_tokens.size()) + 
                 " tokens -> " + clean_response.substr(0, 100) + 
                 (clean_response.length() > 100 ? "..." : ""));
        
        return clean_response;
        
    } catch (const std::exception& e) {
        LOG_ERROR("ContextInputOutput", "Exception during generation phase: " + std::string(e.what()));
        if (callbacks.on_error) {
            callbacks.on_error("Generation failed: " + std::string(e.what()));
        }
        return "";
    }
}

inline bool ContextInputOutput::ValidateGenerationState() const {
    if (!batch_manager || !token_cache || !parent_model || !llama_ctx) {
        LOG_ERROR("ContextInputOutput", "Required components not available");
        return false;
    }
    
    if (!batch_manager->ValidateContextState()) {
        LOG_ERROR("ContextInputOutput", "BatchManager context state validation failed");
        return false;
    }
    
    return true;
}

inline std::string ContextInputOutput::GenerateResponse(const std::string& prompt) {
    // DEPRECATED FOR UI CONTEXTS: This method blocks and should only be used for non-UI contexts
    // like Discord bot responses or batch processing. For UI contexts, use GenerateResponseAsync instead.
    
    std::lock_guard<std::mutex> lock(io_mutex);
    
    if (is_generating.load()) {
        LOG_WARNING("ContextInputOutput", "Generation already in progress");
        return "Error: Generation already in progress";
    }
    
    if (!ValidateGenerationState()) {
        LOG_ERROR("ContextInputOutput", "Generation state validation failed");
        return "Error: Invalid generation state";
    }
    
    is_generating = true;
    should_stop_generation = false;
    
    try {
        // Phase 1: Process prompt tokens (Input Phase)
        if (!ProcessPromptPhase(prompt)) {
            is_generating = false;
            return "Error: Failed to process prompt";
        }
        
        // Phase 2: Generate response tokens (Output Phase)
        GenerationCallbacks empty_callbacks; // No streaming for synchronous generation
        std::string response = ExecuteGenerationPhase(empty_callbacks);
        
        is_generating = false;
        return response;
        
    } catch (const std::exception& e) {
        is_generating = false;
        LOG_ERROR("ContextInputOutput", "Exception in GenerateResponse: " + std::string(e.what()));
        return "Error: " + std::string(e.what());
    }
}

inline bool ContextInputOutput::GenerateResponseAsync(const std::string& prompt, const GenerationCallbacks& callbacks) {
    // Check if already generating
    if (is_generating.load()) {
        if (callbacks.on_error) {
            callbacks.on_error("Generation already in progress");
        }
        return false;
    }
    
    // Validate state before starting async operation
    if (!ValidateGenerationState()) {
        if (callbacks.on_error) {
            callbacks.on_error("Invalid generation state");
        }
        return false;
    }
    
    // Stop any existing generation thread
    if (generation_thread && generation_thread->joinable()) {
        should_stop_generation = true;
        generation_cv.notify_all();
        generation_thread->join();
    }
    
    // Reset generation state
    should_stop_generation = false;
    is_generating = true;
    
    // Start new generation thread
    generation_thread = std::make_unique<std::thread>([this, prompt, callbacks]() {
        std::string full_response;
        bool success = false;
        
        try {
            // Phase 1: Process prompt tokens (Input Phase)
            {
                std::lock_guard<std::mutex> lock(io_mutex);
                if (!ProcessPromptPhase(prompt)) {
                    if (callbacks.on_error) {
                        callbacks.on_error("Failed to process prompt");
                    }
                    is_generating = false;
                    return;
                }
            }
            
            // Phase 2: Generate response tokens (Output Phase)
            full_response = ExecuteGenerationPhase(callbacks);
            success = !should_stop_generation.load() && !full_response.empty();
            
        } catch (const std::exception& e) {
            LOG_ERROR("ContextInputOutput", "Exception in async generation: " + std::string(e.what()));
            if (callbacks.on_error) {
                callbacks.on_error("Generation failed: " + std::string(e.what()));
            }
        }
        
        // Update state
        is_generating = false;
        
        // Call completion callback
        if (callbacks.on_complete) {
            callbacks.on_complete(full_response, success);
        }
    });
    
    return true;
}

inline void ContextInputOutput::StopGeneration() {
    should_stop_generation = true;
    
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
    
    is_generating = false;
    LOG_DEBUG("ContextInputOutput", "Generation stopped");
}

inline void ContextInputOutput::UpdateContextBounds(size_t new_max_tokens) {
    std::lock_guard<std::mutex> lock(io_mutex);
    
    max_context_tokens = new_max_tokens;
    
    if (batch_manager) {
        batch_manager->UpdateContextBounds(new_max_tokens);
    }
    
    LOG_DEBUG("ContextInputOutput", "Updated context bounds to: " + std::to_string(new_max_tokens));
}
