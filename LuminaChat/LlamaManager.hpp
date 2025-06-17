// LlamaManager.hpp - header-only implementation for core llama.cpp functionality
// 
// RESPONSIBILITY: Core management and infrastructure
// - Model loading, management, and lifecycle
// - Context creation, switching, and management  
// - Batch operations and token processing
// - Template application and conversation state
// - Text-to-token conversion (input processing)
// - Message history and summarization coordination
//
// DELEGATION: Response generation is delegated to LlamaResponse class
// 
// Handles core classs pertaining to llama.cpp backend usage.
//
// BATCHING APPROACH:
// Batching in llama.cpp is designed for processing multiple separate inputs/sequences 
// simultaneously, NOT for splitting a single input into chunks. For large single inputs
// during full context rebuilds, we use incremental batch processing to achieve better
// performance than sequential token-by-token processing while maintaining proper
// context state management.
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
#include <filesystem>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <functional>
#include "llama-cpp.h"
#include "LogHandler.hpp"
#include "TokenCache.hpp"
#include "LlamaContext.hpp"

// Forward declarations
class LlamaSummarizer;
bool model_loading_progress_callback(float progress, void *user_data);

// Constants for configuration and performance (Directive #13: Zero Magic & Strong Typing)
namespace LlamaConstants {
    // Additional constants specific to LlamaManager (core constants are in LlamaContext.hpp)
    constexpr int32_t DEFAULT_TOKEN_CACHE_SIZE = 1024;
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
    // String processing constants
    constexpr size_t MAX_TEXT_PREVIEW_LENGTH = 50;
    constexpr int32_t MAX_RETRY_ATTEMPTS = 2;
    // Timing and sleep constants
    constexpr int32_t RETRY_BACKOFF_MS = 50;
    constexpr float MS_TO_MICROSECONDS = 1000.0f;
}

#include "LlamaResponse.hpp"

// Thread Safety Contract (Directive #12):
// This class is NOT thread-safe. External synchronization is required for concurrent access.
// - All public methods must be called from a single thread or protected by external mutexes
// - The llama.cpp backend itself has thread-safety limitations that require careful handling
// - Model loading/unloading operations are particularly sensitive to race conditions
// - Context switching operations modify shared state and must be serialized
//
// INTEGRATION WITH LlamaResponse:
// - LlamaManager handles setup, context management, and provides callback functions
// - LlamaResponse handles pure generation logic, receiving callbacks for batch operations
// - This separation allows LlamaManager to focus on infrastructure while LlamaResponse focuses on generation
class LlamaManager {
private:
    
    std::unordered_map<std::string, std::unique_ptr<ModelInfo>> models;
    std::unordered_map<std::string, std::unique_ptr<ContextInfo>> contexts;
    //std::string active_context_id;    ContextInfo* current_context;
      // Working buffers
    mutable TokenCache token_cache;
    mutable std::string temp_string_buffer;// Working buffers

    // Response generation handler
    mutable LlamaResponse response_generator;
    
public:
    LlamaManager() : token_cache(LlamaConstants::DEFAULT_TOKEN_CACHE_SIZE), response_generator(&token_cache) {
    }

    ~LlamaManager() noexcept {
        cleanup();
    }
    
    // Initialize llama.cpp backend
    bool initialize() {
        ggml_backend_load_all();
        return true;    }

    // Load .gguf model file and create ModelInfo with specific parameters
    bool load_model(const std::string& model_path, const std::string& model_id = "", 
                   int32_t context_size = LlamaConstants::DEFAULT_CONTEXT_SIZE, int32_t gpu_layers = LlamaConstants::DEFAULT_GPU_LAYERS, int32_t predict_tokens = LlamaConstants::DEFAULT_PREDICT_TOKENS,
                   void* progress_callback_user_data = nullptr, const std::string& chat_template = "") {
        if (!std::filesystem::exists(model_path)) {
            LLAMA_LOG("Error: Model file does not exist: " + model_path);
            return false;
        }        std::string actual_model_id = model_id.empty() ? std::filesystem::path(model_path).stem().string() : model_id;
        
        if (models.find(actual_model_id) != models.end()) {
            LLAMA_LOG("Model '" + actual_model_id + "' already loaded, using existing model");
            return true;
        }

        auto model_info = std::make_unique<ModelInfo>();
        
        // Set up model parameters with provided values
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = gpu_layers;
        model_params.use_mmap = true; // Enable memory-mapped file support
        
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
        
        models[actual_model_id] = std::move(model_info);
        
        // Clear caches when new model is loaded
        clear_caches();
        
        LLAMA_LOG("Model loaded successfully: " + model_path + " as '" + actual_model_id + 
                  "' (ctx:" + std::to_string(context_size) + ", gpu:" + std::to_string(gpu_layers) + ")");
        return true;
    }
    
    // Context creation with consistent system prompt usage
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
        ctx_params.flash_attn = true;
        ctx_params.op_offload = true;
        ctx_params.offload_kqv = true;
        
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
        
        // Always use provided system prompt, or copy from main context if empty
        std::string prompt_to_use = system_prompt;
		auto main_context = get_context_info("main_context");
        if (prompt_to_use.empty() && main_context && !main_context->system_message.empty()) {
            prompt_to_use = main_context->system_message;
        }
        
        // Set system message if we have one
        if (!prompt_to_use.empty()) {
            context_info->system_message = prompt_to_use;
            context_info->message_history.emplace_back("system", prompt_to_use);
            context_info->message_cache_dirty = true;
            context_info->message_history_token_count = 0; // Initialize to 0 for new context
        }
        
        // Ensure token count is properly initialized (safety check)
        if (context_info->message_history_token_count < 0) {
            context_info->message_history_token_count = 0;
        }
        
        // Set the reset after generation flag
        context_info->reset_after_generation = reset_after_generation;
          // Initialize summarizer for this context with parent context only (summary resources will be set later)
        context_info->summarizer = std::make_unique<LlamaSummarizer>(context_info.get());
        
        // If summary resources are already available, set them up immediately
        ContextInfo* summary_ctx = get_context_info("summary_context");
        ModelInfo* summary_mdl = get_model_info("summary_model");
        if (summary_ctx && summary_mdl) {
            auto response_callback = [this](const std::string& input, const std::string& username, ContextInfo* target_context) -> std::string {
                return generate_response_on_context(input, username, target_context);
            };
            context_info->summarizer->set_summary_resources(summary_mdl, summary_ctx, response_callback);
            LLAMA_LOG("Set summarizer resources for new context '" + context_id + "'");
        }
        
        contexts[context_id] = std::move(context_info);
        LLAMA_LOG("Created context '" + context_id + "' with model '" + model_id + "' successfully");
        
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
        
        contexts.erase(it);
        LLAMA_LOG("Removed context '" + context_id + "'");
        return true;
    }

    // Check if a context exists by ID
    // Critical Needed for context validation in various operations
    bool has_context(const std::string& context_id) const noexcept {
        return contexts.find(context_id) != contexts.end();
    }
    
    // Retrieve a ContextInfo object by its ID
    // Critical Needed for context access in various operations
    ContextInfo* get_context_info(const std::string& context_id) const {
        auto it = contexts.find(context_id);
        if (it == contexts.end()) {
            return nullptr;
        }
        return it->second.get();
    }

    // Retrieve a ModelInfo object by its ID
    // Critical Needed for model access in various operations
    ModelInfo* get_model_info(const std::string& model_id) const {
        auto it = models.find(model_id);
        if (it == models.end()) {
            return nullptr;
        }
        return it->second.get();
    }
      // Context-specific tokenization with bidirectional caching
    std::vector<llama_token> process_text_to_tokens(const std::string& text, ContextInfo* target_context, bool add_special = true) const {
        if (text.empty()) return {};
        
        // Get vocab from specified context's model
        if (!target_context->model_info || !target_context->model_info->vocab) {
            LLAMA_LOG("Error: No vocabulary available from specified context's model");
            return {};
        }
        
        std::string cache_key = text + (add_special ? ":s" : ":n");
        
        // Check cache first - bidirectional lookup
        if (auto cached_tokens = token_cache.get_tokens(cache_key)) {
            return *cached_tokens;
        }
        
        // Get required buffer size for tokenization
        const int32_t n_tokens_required = -llama_tokenize(target_context->model_info->vocab, text.c_str(), text.size(), nullptr, 0, add_special, true);
        if (n_tokens_required <= 0) {
            // Don't treat empty tokenization as warning for whitespace-only text
            if (std::all_of(text.begin(), text.end(), [](char c) { return std::isspace(c); })) {
                // Cache empty result for whitespace-only strings
                token_cache.put(cache_key, text, {});
                return {};
            }
            
            LLAMA_LOG("Warning: Text tokenization failed or resulted in 0 tokens: '" + 
                        text.substr(0, LlamaConstants::MAX_TEXT_PREVIEW_LENGTH) + (text.size() > LlamaConstants::MAX_TEXT_PREVIEW_LENGTH ? "..." : "") + "'");
            return {};
        }
        
        // Add bounds checking for extremely large token counts
        if (n_tokens_required > target_context->model_info->n_ctx) {
            LLAMA_LOG("Error: Text would produce " + std::to_string(n_tokens_required) + 
                       " tokens, exceeding context limit of " + std::to_string(target_context->model_info->n_ctx));
            return {};
        }
        
        // Allocate buffer and tokenize
        std::vector<llama_token> tokens(n_tokens_required);
        const int32_t n_tokens_actual = llama_tokenize(target_context->model_info->vocab, text.c_str(), text.size(), 
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
        
        // Cache the result with bidirectional mapping for detokenization
        token_cache.put(cache_key, text, tokens);
        
        return tokens;
    }
      // Context-specific detokenization with bidirectional caching (reverse lookup)
    std::string process_tokens_to_text(const std::vector<llama_token>& tokens, ContextInfo* target_context) const {
        if (tokens.empty()) return {};
        
        // Get vocab from specified context's model
        if (!target_context->model_info || !target_context->model_info->vocab) {
            LLAMA_LOG("Error: No vocabulary available from specified context's model for detokenization");
            return {};
        }
        
        // Check reverse cache first - bidirectional lookup
        if (auto cached_text = token_cache.get_text(tokens)) {
            return *cached_text;
        }
        
        // Fallback to llama.cpp detokenization
        std::string result;
        result.reserve(tokens.size() * 4); // Rough estimate for token-to-text expansion
        
        for (const auto& token : tokens) {
            std::vector<char> buffer(32); // Start with reasonable buffer size
            
            int32_t result_length = llama_token_to_piece(
                target_context->model_info->vocab, 
                token, 
                buffer.data(), 
                buffer.size(), 
                0, 
                true
            );
            
            if (result_length < 0) {
                // Buffer too small, resize and retry
                buffer.resize(-result_length);
                result_length = llama_token_to_piece(
                    target_context->model_info->vocab, 
                    token, 
                    buffer.data(), 
                    buffer.size(), 
                    0, 
                    true
                );
            }
            
            if (result_length > 0) {
                result.append(buffer.data(), result_length);
            }
        }
        
        // Cache the result bidirectionally for future lookups
        std::string cache_key = "detok:" + std::to_string(std::hash<std::string>{}(result));
        token_cache.put(cache_key, result, tokens);
        
        return result;
    }    // Cache-aware token-to-text conversion with bidirectional caching
    std::string convert_token_to_text_cached(llama_token token, ContextInfo* target_context) const {
        if (!target_context->model_info || !target_context->model_info->vocab) {
            LLAMA_LOG("Error: No vocabulary available for token conversion");
            return {};
        }
        
        // Try reverse cache lookup first for single token
        std::vector<llama_token> single_token = {token};
        if (auto cached_text = token_cache.get_text(single_token)) {
            return *cached_text;
        }
        
        // Fallback to direct llama.cpp conversion
        std::vector<char> buffer(32); // Start with reasonable buffer size
        
        int32_t result_length = llama_token_to_piece(
            target_context->model_info->vocab, 
            token, 
            buffer.data(), 
            buffer.size(), 
            0, 
            true
        );
        
        if (result_length < 0) {
            // Buffer too small, resize and retry
            buffer.resize(-result_length);
            result_length = llama_token_to_piece(
                target_context->model_info->vocab, 
                token, 
                buffer.data(), 
                buffer.size(), 
                0, 
                true
            );
        }
        
        std::string result;
        if (result_length > 0) {
            result.assign(buffer.data(), result_length);
            
            // Cache the result bidirectionally
            std::string cache_key = "tok:" + std::to_string(token);
            token_cache.put(cache_key, result, single_token);
        }
        
        return result;
    }

    // Forward declaration for summary slot info - implementation after LlamaSummarizer include
    // TODO: Move this into LlamaSummarizer and access it through the parent ContextInfo
    struct SummarySlotInfo {
        size_t total_slots;
        size_t used_slots;
        std::vector<std::string> summaries;
    };

    // Context-specific summary slot information
    // TODO: Move this into LlamaSummarizer and access it through the parent ContextInfo
    SummarySlotInfo get_summary_slot_info(ContextInfo* target_context) const;
    
    // Initialize summarizer resources for all contexts when summary context becomes available
    void initialize_summarizer_resources() {
        // Find the summary context and model
        ContextInfo* summary_ctx = get_context_info("summary_context");
        ModelInfo* summary_mdl = get_model_info("summary_model");
        
        if (!summary_ctx || !summary_mdl) {
            LLAMA_LOG("Warning: Summary context or model not available for summarizer initialization");
            return;
        }
        
        // Create callback function for response generation
        auto response_callback = [this](const std::string& input, const std::string& username, ContextInfo* target_context) -> std::string {
            return generate_response_on_context(input, username, target_context);
        };
        
        // Initialize summarizer resources for ALL contexts that have summarizers
        int32_t initialized_count = 0;
        for (auto& [context_id, context_info] : contexts) {
            if (context_info->summarizer) {
                context_info->summarizer->set_summary_resources(summary_mdl, summary_ctx, response_callback);
                initialized_count++;
                LLAMA_LOG("Initialized summarizer resources for context: " + context_id);
            }
        }
        
        LLAMA_LOG("Successfully initialized summarizer resources for " + std::to_string(initialized_count) + " contexts");
    }
  
public:

/* 
     * IMPROVED LINEAR LOGIC FLOW FOR RESPONSE GENERATION
     * ================================================
     * 
     * This method implements a clean, predictable, and easy-to-follow logic flow:
     *
     * 1. PRE-FLIGHT VALIDATION: Ensure all required components are ready
     * 2. ADD MESSAGE & INVALIDATE: Add user input to conversation, mark state as needing rebuild
     * 3. PREPARE CONTEXT: Centralized preparation handling:
     *    - Template application (converts message history to model format)
     *    - Tokenization (gets exact token counts)
     *    - Pruning decision (checks if context exceeds 90% threshold)
     *    - Context rebuild (full rebuild with or without pruning)
     *    - Generation prompt preparation
     * 4. GENERATE RESPONSE: Pure token generation using LlamaResponse
     * 5. UPDATE CONVERSATION: Add response to conversation state
     *
     * KEY IMPROVEMENTS:
     * - Single source of truth for conversation state (ConversationState)
     * - Clear decision points (needs_pruning())
     * - Lazy evaluation (only rebuild when needed)
     * - Separated concerns (preparation vs. generation)
     * - Linear, predictable flow (no jumping between methods)
     * - Comprehensive logging for debugging
     */
    
     // Context-specific response generation
     std::string generate_response(const std::string& input, ContextInfo* target_context, const std::string& username = "Schwi") {
        // Pre-flight validation
        if (!target_context->context || !target_context->model_info || 
            !target_context->model_info->model_loaded || !target_context->model_info->model || 
            !target_context->model_info->vocab || !target_context->batch_initialized) {
            return "Error: Model components not properly initialized for generation";
        }

        if (input.empty()) {
            return "Error: Empty input";
        }

        LLAMA_LOG("Starting improved linear generation flow for input: " + 
                  (input.length() > 50 ? input.substr(0, 50) + "..." : input));

        // STEP 1: Update conversation with new input
        target_context->add_message(username, input);
        
        // STEP 2: Prepare context for generation (handles template, tokenization, pruning, rebuild)
        auto token_processor = [this, target_context](const std::string& text, bool add_special) {
            return process_text_to_tokens(text, target_context, add_special);
        };
        auto pruning_callback = [this, target_context](float keep_ratio) {
            return target_context->prune_with_summarization(keep_ratio);
        };
        
        if (!target_context->prepare_context_for_generation(token_processor, pruning_callback)) {
            LLAMA_LOG("Context preparation failed, attempting one recovery");
            
            // Single recovery attempt - clear context state and try again
            if (target_context->context) {
                llama_memory_clear(llama_get_memory(target_context->context), true);
                target_context->n_past = 0;
                target_context->prev_len = 0;
                target_context->message_cache_dirty = true;
                target_context->conversation_state.invalidate();
                  if (!target_context->prepare_context_for_generation(token_processor, pruning_callback)) {
                    return "Error: Failed to prepare context for generation after recovery attempt";
                }
                LLAMA_LOG("Context preparation recovered successfully");
            } else {
                return "Error: Failed to prepare context for generation";
            }
        }
          // STEP 3: Generate response tokens
        // Setup callback functions for LlamaResponse
        auto token_adder = [this, target_context](llama_token token, int32_t pos, const std::vector<llama_seq_id>& seq_ids, bool output_logits) -> bool {
            return target_context->add_tokens_to_batch({token}, pos, seq_ids, output_logits);
        };
        auto context_updater = [target_context]() -> void {
            // Update context length tracking after generation
            std::string updated_content;
            if (target_context->apply_template(false, updated_content)) {
                target_context->prev_len = static_cast<int32_t>(updated_content.length());
            }
        };
        
        LLAMA_LOG("Delegating to LlamaResponse for token generation");
        std::string response = response_generator.generate_response("generate", "assistant", target_context, token_adder, context_updater);
        
        // STEP 4: Update conversation with response (if successful)
        if (!response.empty() && !response.starts_with("Error:")) {
            target_context->add_message("assistant", response);
            LLAMA_LOG("Linear generation flow completed successfully");
        } else {
            LLAMA_LOG("Linear generation flow failed: " + response);
        }
        
        return response;
    }
    
    // Get performance statistics - overloaded for specific context
    // TODO: PerformanceStats should probably integrate Timings and be moved into ContextInfo
    struct PerformanceStats {
        int64_t total_generation_tokens;
        int64_t last_decode_time_us;
        float average_tokens_per_second;
    };
    
    PerformanceStats get_performance_stats(ContextInfo* target_context) const {
        float avg_tps = 0.0f;
        if (target_context->last_decode_time_us > 0 && target_context->total_generation_tokens > 0) {
            avg_tps = static_cast<float>(target_context->total_generation_tokens) / (static_cast<float>(target_context->last_decode_time_us) / 1000000.0f);
        }
        
        return {
            target_context->total_generation_tokens,
            target_context->last_decode_time_us,
            avg_tps
        };
    }
    
    // Timing API methods for compatibility with UI - overloaded for specific context
    // TODO: Should probably integrate into PerformanceStats and be moved into ContextInfo
    struct Timings {
        int32_t n_eval = 0;
        float t_eval_ms = 0.0f;
    };
    
    void reset_timings(ContextInfo* target_context) {
        target_context->total_generation_tokens = 0;
        target_context->last_decode_time_us = 0;
    }
    
    Timings get_timings(ContextInfo* target_context) const {
        Timings timings;
        timings.n_eval = static_cast<int32_t>(target_context->total_generation_tokens);
        timings.t_eval_ms = static_cast<float>(target_context->last_decode_time_us) / LlamaConstants::MS_TO_MICROSECONDS;
        return timings;
    }
    
    // Enhanced cleanup with memory optimization
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
        
        // Clean up all models - ModelInfo destructor handles model cleanup
        models.clear();        
        LLAMA_LOG("Cleanup completed");
    }
    
    // Get context size for capacity calculations
    // TODO: This function should be removed and callers directly access ContextInfo::get_context_size()
    int32_t get_context_size(ContextInfo* target_context) const noexcept {
        return target_context->get_context_size();
    }
    
    bool update_context_from_history(ContextInfo* target_context) {
        // Delegate to the context's rebuild method
        auto token_processor = [this, target_context](const std::string& text, bool add_special) {
            return process_text_to_tokens(text, target_context, add_special);
        };
        auto pruning_callback = [this, target_context](float keep_ratio) {
            return target_context->prune_with_summarization(keep_ratio);        };
          return target_context->update_context_from_history(token_processor, pruning_callback);
    }
    
    // TokenCache management and statistics
    // Get token cache performance statistics for monitoring and optimization
    TokenCache::CacheStats get_token_cache_stats() const {
        return token_cache.get_stats();
    }
    
    // Configure token cache settings
    void configure_token_cache(TokenCache::EvictionPolicy policy = TokenCache::EvictionPolicy::LRU, 
                              bool enable_thread_safety = false) {
        token_cache.configure(policy, enable_thread_safety);
        LLAMA_LOG("Token cache configured - Policy: " + std::to_string(static_cast<int>(policy)) + 
                  ", Thread Safety: " + (enable_thread_safety ? "enabled" : "disabled"));
    }

private:
    void clear_caches() const {
        token_cache.clear();
    }
    
    // Helper method for summarizer callback - generates response on a specific context without context switching
    std::string generate_response_on_context(const std::string& input, const std::string& username, ContextInfo* target_context) {
        // Generate response directly on the target context without switching
        LLAMA_LOG("Generating response directly on target context without switching");
        
        try {
            // Use the overloaded method that accepts a specific context
            return generate_response(input, target_context, username);
        } catch (...) {
            LLAMA_LOG("Exception occurred during context-specific response generation");
            return "Error: Exception during response generation";
        }
    }
};

// Include LlamaSummarizer implementation after class declaration to avoid circular dependency
#include "LlamaSummarizer.hpp"

// Implementation of LlamaManager methods that depend on LlamaSummarizer
inline LlamaManager::SummarySlotInfo LlamaManager::get_summary_slot_info(ContextInfo* target_context) const {
    if (!target_context->summarizer) {
        return {SummarizerConstants::MAX_SUMMARY_SLOTS, 0, {}};
    }
    auto summarizer_info = target_context->summarizer->get_summary_slot_info();
    return {summarizer_info.total_slots, summarizer_info.used_slots, summarizer_info.summaries};
}

//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODING DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//