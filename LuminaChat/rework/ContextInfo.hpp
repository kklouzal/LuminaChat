#pragma once

#include "ChatTemplateManager.hpp"
#include "ModelInfo.hpp"
#include "TokenCache.hpp"
#include "Logger.hpp"
#include "Utilities.hpp"
// llama.cpp includes
#include "llama-cpp.h"
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <mutex>
#include <chrono>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <thread>
#include <atomic>
#include <condition_variable>

// ContextInfo Configuration Constants
namespace ContextConstants {
    constexpr int32_t SAFETY_BUFFER_TOKENS = 64;      // Safety margin for all context operations
    constexpr int32_t GENERATION_BUFFER_TOKENS = 128; // Buffer for response generation (larger to account for responses)
}

// ContextInfo: Individual conversation context management
// 
// Key Features:
// - Dynamic Template Management: Owns a ChatTemplateManager instance
// - Performance statistics tracking
// - Reference to parent ModelInfo and its TokenCache
// - Message history storage: vector<pair<string, string>> (role, content) - pure conversation only
// - Batch management for generation
// - Smart context rebuilding
// - Plugin Integration: Template-based summarization and context management

// Forward declarations
struct llama_context;

// Helper function declarations
std::string GenerateContextId();

// Forward declaration for ModelInfo
class ModelInfo;

enum class RebuildStrategy {
    FULL,           // Complete rebuild including template re-rendering
    PARTIAL,        // Efficient append-only rebuild for new messages
    TEMPLATE_ONLY   // Only re-render template, keep existing tokens
};

enum class ContextState {
    READY,
    PROCESSING,
    GENERATING,        // New state for async generation
    ERROR_STATE
};

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
 * Individual conversation context management with dynamic template system
 * 
 * Core Responsibilities:
 * - Message history storage and management
 * - Dynamic template rendering via ChatTemplateManager
 * - LLM token processing and response generation
 * - Context overflow detection and logging
 * - Template section updates from plugins
 * 
 * Revolutionary Processing Flow (Plugin-Based Architecture):
 * 1. Receive input from Orchestrator (pre-sanitized if from Discord)
 * 2. Add pure conversation pair to message history (no system contamination)
 * 3. Render dynamic template with current message history via ChatTemplateManager
 * 4. Process rendered template to tokens via TokenCache
 * 5. Context overflow check - log warnings if needed (plugin handles pruning)
 * 6. Generate LLM response using rendered template
 * 7. Detokenize response via TokenCache
 * 8. Add response pair to message history
 * 9. Return human-readable response
 * 
 * Plugin Architecture Integration:
 * - ContextPruningPlugin: Monitors contexts via Orchestrator, triggers pruning decisions
 * - SummarizationPlugin: Manages pruning buffer, processes summaries, applies results
 * - EmoTagPlugin: Manages emotional analysis buffer, processes emotions, applies results
 * - Orchestrator: Coordinates all plugin interactions and scheduled monitoring
 * 
 * Division of Responsibilities:
 * - ContextInfo: Core LLM operations, message storage, context safety monitoring, template management
 * - Plugins: Specialized processing, buffer management, decision algorithms
 * - Orchestrator: Workflow coordination, plugin scheduling, pipeline management
 * 
 * Template Integration Benefits:
 * - Clean Message History: Only actual conversation, no system pollution
 * - Dynamic Context: Summaries, system messages embedded in template sections
 * - Plugin-Driven Updates: Plugins update template directly via public methods
 * - Per-Context Flexibility: Each context can have completely different template evolution
 * - Non-Blocking: Critical path stays fast, heavy work happens in background plugins
 */
class ContextInfo {
private:    // Core components
    std::unique_ptr<ChatTemplateManager> template_manager;
    ModelInfo* parent_model;
    TokenCache* token_cache;
    
    // Context configuration
    int32_t context_size;  // Individual context size (not tied to model)
    
    // Context state
    std::string context_id;
    ContextState state;
    size_t max_context_tokens = 0;  // Maximum context size for this instance
    mutable std::mutex context_mutex;
    
    // Message history - pure conversation only
    std::vector<std::pair<std::string, std::string>> message_history; // (role, content)
    
    // Summary storage - maintains up to 5 summaries in chronological order (oldest to newest)
    std::vector<std::string> summaries;
    static constexpr size_t MAX_SUMMARIES = 5;
    
    // Llama context management
    llama_context* llama_ctx = nullptr;
    llama_batch batch;
    bool batch_initialized = false;
    std::vector<int32_t> current_tokens;
    int32_t n_past = 0;  // Number of tokens processed in context
    bool context_needs_rebuild = true;
    
    // Generation state
    std::atomic<bool> is_generating{false};
    std::atomic<bool> should_stop_generation{false};
    std::unique_ptr<std::thread> generation_thread;
    std::condition_variable generation_cv;
    std::mutex generation_mutex;
    GenerationCallbacks current_callbacks;
    
    // Context size management
    void AddSummaryToList(const std::string& summary);
    void UpdateTemplateWithAllSummaries();
    
public:    // Constructor overloads
    ContextInfo(const std::string& context_id, ModelInfo* model, int32_t context_size);
    ContextInfo(ModelInfo* model, int32_t context_size); // For testing with auto-generated context_id
    
    // Factory method for default template
    static std::unique_ptr<ContextInfo> Create(const std::string& context_id, ModelInfo* model, int32_t context_size);
    static std::unique_ptr<ContextInfo> Create(ModelInfo* model, int32_t context_size); // Auto-generated ID
    
    // Core pruning method (immediate, critical path)
    void PruneContextImmediate(size_t keep_recent_messages = 5);
    
    // Enhanced pruning method that returns pruned messages for summarization coordination
    std::vector<std::pair<std::string, std::string>> PruneContextImmediateWithExtraction(size_t keep_recent_messages = 5);
    
private:
    // Internal pruning method (assumes context_mutex is already held)
    void PruneContextImmediate_Internal(size_t keep_recent_messages = 5);
    
    // State transition validation
    bool IsValidStateTransition(ContextState from, ContextState to) const {
        // Define valid state transitions
        switch (from) {
            case ContextState::READY:
                return true; // Can transition to any state from READY
                
            case ContextState::PROCESSING:
                return to == ContextState::READY || to == ContextState::ERROR_STATE;
                
            case ContextState::GENERATING:
                return to == ContextState::READY || to == ContextState::ERROR_STATE;
                
            case ContextState::ERROR_STATE:
                return to == ContextState::READY; // Can recover from error
                
            default:
                return false;
        }
    }
    
public:
    
    // Plugin callback to apply completed summaries
    void ApplyCompletedSummary(const std::string& summary);
    
    // Destructor
    ~ContextInfo();
    
    // Core processing interface
    std::string HandleInput(const std::string& input, const std::string& username = "user"); // DEPRECATED for UI: Use HandleInputAsync instead
    bool HandleInputAsync(const std::string& input, const GenerationCallbacks& callbacks, const std::string& username = "user");
    void AddHistoricalMessage(const std::string& role, const std::string& content);
    
    // Generation control
    void StopGeneration();
    bool IsGenerating() const { return is_generating.load(); }
    
    // Llama.cpp integration methods
    bool InitializeLlamaContext();
    bool ProcessTokensBatch(const std::vector<int32_t>& tokens);
    std::string GenerateResponse(const std::string& prompt);
    void StartGenerationAsync(const std::string& prompt, const GenerationCallbacks& callbacks);
    
    // Template section management - direct access to ChatTemplateManager
    void UpdateEnvironment(const std::string& env);
    void UpdateIdentity(const std::string& identity);
    void UpdateSystemPrompt(const std::string& system_msg);
    void ApplySummary(const std::string& summary);  // Updates template's summary section
    void UpdateOldChatSummary(const std::string& old_summary); // For context pruning
    void UpdateMotifContext(const std::string& motif);
    void UpdateInternalReflection(const std::string& reflection);
    void UpdateEmotionalState(const std::string& emotional_state);
    void AddPastSessionMemory(const std::string& memory);
    
    // Context management
    bool RebuildContext(RebuildStrategy strategy = RebuildStrategy::FULL);
    void ClearContext();
    void ClearMessageHistory();
    
    // State and statistics
    ContextState GetState() const { 
        std::lock_guard<std::mutex> lock(context_mutex); 
        return state; 
    }
    
    bool SetState(ContextState new_state) {
        std::lock_guard<std::mutex> lock(context_mutex);
        
        // Validate state transitions
        if (!IsValidStateTransition(state, new_state)) {
            return false;
        }
        
        state = new_state;
        return true;
    }
    
    bool IsAvailableForGeneration() const {
        std::lock_guard<std::mutex> lock(context_mutex);
        return state == ContextState::READY;
    }
    
    size_t GetMaxContextTokens() const { return max_context_tokens; }
    const std::string& GetContextId() const { return context_id; }
    
    // Get actual context usage based on n_past (single source of truth)
    size_t GetActualContextTokens() const { return static_cast<size_t>(std::max(0, n_past)); }
    float GetActualContextUsageRatio() const {
        return max_context_tokens > 0 ? 
            static_cast<float>(GetActualContextTokens()) / max_context_tokens : 0.0f;
    }
    
    // Message history access
    const std::vector<std::pair<std::string, std::string>>& GetMessageHistory() const { return message_history; }
    size_t GetMessageCount() const { return message_history.size(); }
    
    // Template access
    ChatTemplateManager& GetTemplateManager() { return *template_manager; }
    const ChatTemplateManager& GetTemplateManager() const { return *template_manager; }
    
    // Advanced features
    void SetMaxContextTokens(size_t max_tokens);
    int32_t GetContextSize() const { return static_cast<int32_t>(max_context_tokens); }
    std::string GetCurrentPrompt() const;
    
    // Template rendering (made public for testing)
    std::string BuildFullPrompt();
    
    // Batch management methods - context-specific
    void clear_batch();
    
    // Context rebuilding helpers (made public for testing)
    void RebuildContext_Full();
    void RebuildContext_Partial();
    void RebuildContext_TemplateOnly();
    
    // Validate and synchronize context state before batch processing
    bool ValidateAndSyncContextState();
    
    // Synchronize stats with actual context state (n_past) - call after all context operations
    void SyncStatsWithContextState();
    
    // Query actual context usage from llama.cpp and sync our internal state
    void QueryAndSyncActualContextUsage();
    
    // Context pruning for summarization
    void PruneMessageHistoryWithSummary(const std::string& summary, size_t keep_recent_messages = 5);
};

// Inline implementation of ContextInfo methods
inline ContextInfo::ContextInfo(const std::string& context_id, ModelInfo* model, int32_t context_size)
    : context_id(context_id)
    , parent_model(model)
    , token_cache(model ? &model->GetTokenCache() : nullptr)
    , context_size(context_size)
    , state(ContextState::READY)
    , template_manager(std::make_unique<ChatTemplateManager>())
    , llama_ctx(nullptr)
    , context_needs_rebuild(true)
{
    if (!parent_model) {
        LOG_ERROR_ContextInfo("ContextInfo created with null ModelInfo");
        state = ContextState::ERROR_STATE;
        return;
    }
    
    if (!token_cache) {
        LOG_ERROR_ContextInfo("ContextInfo created with null TokenCache");
        state = ContextState::ERROR_STATE;
        return;
    }
    
    if (context_size <= 0) {
        LOG_ERROR_ContextInfo("ContextInfo created with invalid context size: " + std::to_string(context_size));
        state = ContextState::ERROR_STATE;
        return;
    }
    
    // Set max context tokens to match our individual context size
    max_context_tokens = static_cast<size_t>(context_size);
    
    LOG_DEBUG_ContextInfo("ContextInfo created: " + context_id + " with context size: " + std::to_string(context_size));
}

inline ContextInfo::ContextInfo(ModelInfo* model, int32_t context_size)
    : ContextInfo(GenerateContextId(), model, context_size) // Auto-generate context_id
{
    LOG_DEBUG_ContextInfo("ContextInfo created with auto-generated ID: " + context_id + " and context size: " + std::to_string(context_size));
}

inline ContextInfo::~ContextInfo() {
    // Stop any ongoing generation first
    should_stop_generation = true;
    
    // Wait for generation thread to complete
    if (generation_thread && generation_thread->joinable()) {
        generation_cv.notify_all();  // Wake up the generation thread
        generation_thread->join();
    }
    
    std::lock_guard<std::mutex> lock(context_mutex);
    
    LOG_DEBUG_ContextInfo("ContextInfo destructor called for: " + context_id);
    
    // Clean up batch
    if (batch_initialized) {
        llama_batch_free(batch);
        batch_initialized = false;
        LOG_DEBUG_ContextInfo("Batch freed for context: " + context_id);
    }
    
    // Clean up llama context
    if (llama_ctx) {
        llama_free(llama_ctx);
        llama_ctx = nullptr;
        LOG_DEBUG_ContextInfo("Llama context freed for: " + context_id);
    }
    
    LOG_DEBUG_ContextInfo("ContextInfo destroyed: " + context_id);
}

inline bool ContextInfo::InitializeLlamaContext() {
    // Note: context_mutex should already be held by caller
    
    if (!parent_model || !parent_model->IsLoaded()) {
        LOG_ERROR_ContextInfo("Parent model not loaded for context initialization");
        return false;
    }
    
    if (llama_ctx) {
        LOG_DEBUG_ContextInfo("Llama context already initialized");
        return true;
    }
    
    try {
        // Set up context parameters using model's settings
        llama_context_params ctx_params = llama_context_default_params();
        // Initialize context parameters with this context's specific size
        ctx_params.n_ctx = context_size;
        ctx_params.n_batch = std::min(512, context_size / 8);
        ctx_params.n_threads = parent_model->GetConfig().threads > 0 ? 
                               parent_model->GetConfig().threads : 
                               std::max(1u, std::thread::hardware_concurrency());
        ctx_params.n_threads_batch = ctx_params.n_threads;
        
        // Performance optimizations
        ctx_params.flash_attn = true;
        ctx_params.offload_kqv = true;
        
        LOG_DEBUG_ContextInfo("Creating llama context with params: n_ctx=" + 
                             std::to_string(ctx_params.n_ctx) + 
                             ", n_batch=" + std::to_string(ctx_params.n_batch) + 
                             ", n_threads=" + std::to_string(ctx_params.n_threads));
        
        // Create context
        llama_ctx = llama_init_from_model(parent_model->GetModel(), ctx_params);
        if (!llama_ctx) {
            LOG_ERROR_ContextInfo("Failed to create llama context");
            return false;
        }
        
        // Initialize batch
        batch = llama_batch_init(ctx_params.n_batch, 0, 1);
        if (!batch.token) {
            LOG_ERROR_ContextInfo("Failed to initialize batch");
            llama_free(llama_ctx);
            llama_ctx = nullptr;
            return false;
        }
        
        batch_initialized = true;
        
        // Verify that the actual allocated context size matches what we requested
        if (ctx_params.n_ctx != context_size) {
            LOG_WARNING_ContextInfo("Allocated context size (" + std::to_string(ctx_params.n_ctx) + 
                                   ") differs from requested size (" + std::to_string(context_size) + ")");
        }
        
        // Keep max_context_tokens consistent with the original context_size for accounting
        // This ensures UI and calculations remain consistent with the requested size
        max_context_tokens = static_cast<size_t>(context_size);
        
        LOG_ContextInfo("Llama context initialized successfully for: " + context_id);
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR_ContextInfo("Exception during llama context initialization: " + std::string(e.what()));
        if (llama_ctx) {
            llama_free(llama_ctx);
            llama_ctx = nullptr;
        }
        return false;
    }
}

inline std::string ContextInfo::HandleInput(const std::string& input, const std::string& username) {
    // DEPRECATED FOR UI CONTEXTS: This method blocks and should only be used for non-UI contexts
    // like Discord bot responses or batch processing. For UI contexts, use HandleInputAsync instead.
    
    std::lock_guard<std::mutex> lock(context_mutex);
    
    if (state == ContextState::ERROR_STATE) {
        LOG_ERROR_ContextInfo("Cannot handle input - context in error state");
        return "Error: Context unavailable";
    }
    
    state = ContextState::PROCESSING;
    
    try {
        // CRITICAL: This MUST be sent as username, input NOT "user", input; Chat template handles roles dynamically and will allow the AI to interpret properly here.
        // 1. Add pure conversation pair to message history
        message_history.emplace_back(username, input);
        LOG_DEBUG_ContextInfo("Added user message to history: " + username + " -> " + 
                             input.substr(0, 50) + (input.length() > 50 ? "..." : ""));
        
        // 2. Check if context needs rebuilding after adding message
        if (context_needs_rebuild || template_manager->IsTemplateDirty()) {
            if (!RebuildContext(RebuildStrategy::FULL)) {
                state = ContextState::ERROR_STATE;
                return "Error: Failed to rebuild context";
            }
            // Sync stats after successful rebuild
            SyncStatsWithContextState();
        }
        
        // 3. Render dynamic template with current message history
        std::string full_prompt = BuildFullPrompt();
        
        // 4. Check context size - delegate to ContextPruningPlugin for intelligent pruning decisions
        // Note: Removed inline threshold checking - ContextPruningPlugin handles this through scheduled monitoring
        std::vector<int32_t> prompt_tokens = token_cache->TokenizeText(full_prompt, true);
        
        // Only check for emergency situations that require immediate action
        size_t estimated_tokens = static_cast<size_t>(prompt_tokens.size());
        // Use estimated_tokens directly since full_prompt already includes everything
        // n_past would be double counting since it's from previous context state
        size_t total_estimated = estimated_tokens;
        if (max_context_tokens > 0 && total_estimated >= max_context_tokens - ContextConstants::SAFETY_BUFFER_TOKENS) {
            LOG_WARNING_ContextInfo("Context approaching hard limit - plugin intervention needed");
            LOG_WARNING_ContextInfo("Estimated tokens: " + std::to_string(total_estimated) + "/" + std::to_string(max_context_tokens));
            
            // Trust the plugin to handle this through its monitoring
            // No direct emergency pruning - maintain clean architecture
        }

        
        // 5. Generate LLM response using actual AI model
        std::string response = GenerateResponse(full_prompt);
        
        // 6. Add assistant response to message history
        message_history.emplace_back("assistant", response);
        
        state = ContextState::READY;
        
        LOG_DEBUG_ContextInfo("Generated response: " + response.substr(0, 100) + 
                             (response.length() > 100 ? "..." : ""));
        
        return response;
        
    } catch (const std::exception& e) {
        LOG_ERROR_ContextInfo("Exception in HandleInput: " + std::string(e.what()));
        state = ContextState::ERROR_STATE;
        return "Error: " + std::string(e.what());
    }
}

inline void ContextInfo::AddHistoricalMessage(const std::string& role, const std::string& content) {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    message_history.emplace_back(role, content);
    context_needs_rebuild = true;
    
    LOG_DEBUG_ContextInfo("Added historical message: " + role + " -> " + 
                         content.substr(0, 50) + (content.length() > 50 ? "..." : ""));
    context_needs_rebuild = true;
}

// Template section management methods
inline void ContextInfo::UpdateEnvironment(const std::string& env) {
    template_manager->UpdateEnvironment(env);
    context_needs_rebuild = true;
    LOG_DEBUG_ContextInfo("Updated environment section");
}

inline void ContextInfo::UpdateIdentity(const std::string& identity) {
    template_manager->UpdateIdentity(identity);
    context_needs_rebuild = true;
    LOG_DEBUG_ContextInfo("Updated identity section");
}

inline void ContextInfo::UpdateSystemPrompt(const std::string& system_msg) {
    template_manager->UpdateSystemPrompt(system_msg);
    context_needs_rebuild = true;
    LOG_DEBUG_ContextInfo("Updated system prompt section");
}

inline void ContextInfo::ApplySummary(const std::string& summary) {
    template_manager->UpdateSummary(summary);
    context_needs_rebuild = true;
    LOG_ContextInfo("Applied summary to template: " + summary.substr(0, 100) + 
                   (summary.length() > 100 ? "..." : ""));
    
    // Automatically prune message history when a summary is applied
    // This prevents the context from growing indefinitely
    if (message_history.size() > 10) { // Only prune if we have substantial history
        LOG_ContextInfo("Auto-pruning message history due to summary application");
        
        // Keep only the most recent 5 message pairs (10 messages total)
        size_t keep_messages = 10;
        std::vector<std::pair<std::string, std::string>> recent_messages;
        size_t start_idx = message_history.size() - keep_messages;
        recent_messages.assign(message_history.begin() + start_idx, message_history.end());
        
        LOG_ContextInfo("Pruning message history: keeping " + std::to_string(keep_messages) + 
                       " recent messages out of " + std::to_string(message_history.size()) + " total");
        
        message_history = std::move(recent_messages);
    }
}

inline void ContextInfo::UpdateOldChatSummary(const std::string& old_summary) {
    template_manager->UpdateOldChatSummary(old_summary);
    context_needs_rebuild = true;
    LOG_DEBUG_ContextInfo("Updated old chat summary");
}

inline void ContextInfo::UpdateMotifContext(const std::string& motif) {
    template_manager->UpdateMotifContext(motif);
    context_needs_rebuild = true;
    LOG_DEBUG_ContextInfo("Updated motif context");
}

inline void ContextInfo::UpdateInternalReflection(const std::string& reflection) {
    template_manager->UpdateInternalReflection(reflection);
    context_needs_rebuild = true;
    LOG_DEBUG_ContextInfo("Updated internal reflection");
}

inline void ContextInfo::UpdateEmotionalState(const std::string& emotional_state) {
    template_manager->UpdateEmotionalState(emotional_state);
    context_needs_rebuild = true;
    LOG_DEBUG_ContextInfo("Updated emotional state");
}

inline void ContextInfo::AddPastSessionMemory(const std::string& memory) {
    template_manager->AddPastSession(memory);
    context_needs_rebuild = true;
    LOG_DEBUG_ContextInfo("Added past session memory");
}

inline std::string ContextInfo::BuildFullPrompt() {
    if (!template_manager) {
        LOG_ERROR_ContextInfo("Template manager not available");
        return "";
    }
    
    // Render template with current message history
    std::string rendered_prompt = template_manager->RenderTemplate(message_history);
    
    LOG_DEBUG_ContextInfo("Built full prompt: " + std::to_string(rendered_prompt.length()) + " characters");
    return rendered_prompt;
}

// Batch management methods - context-specific
inline void ContextInfo::clear_batch() {
    if (!batch_initialized) return;
    batch.n_tokens = 0;
}

inline bool ContextInfo::RebuildContext(RebuildStrategy strategy) {
    if (!InitializeLlamaContext()) {
        LOG_ERROR_ContextInfo("Failed to initialize llama context for rebuild");
        return false;
    }
    
    try {
        switch (strategy) {
            case RebuildStrategy::FULL:
                RebuildContext_Full();
                break;
            case RebuildStrategy::PARTIAL:
                RebuildContext_Partial();
                break;
            case RebuildStrategy::TEMPLATE_ONLY:
                RebuildContext_TemplateOnly();
                break;
        }
        
        context_needs_rebuild = false;
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR_ContextInfo("Exception during context rebuild: " + std::string(e.what()));
        return false;
    }
}

inline void ContextInfo::RebuildContext_Full() {
    // Note: context_mutex should already be held by caller
    
    LOG_DEBUG_ContextInfo("Performing full context rebuild");
    
    // Clear current context and reset state
    if (llama_ctx) {
        llama_memory_clear(llama_get_memory(llama_ctx), true);
        n_past = 0;
    }
    
    // Ensure all rebuild flags are cleared
    context_needs_rebuild = false;
    
    // Build and tokenize full prompt
    std::string full_prompt = BuildFullPrompt();
    std::vector<int32_t> tokens = token_cache->TokenizeText(full_prompt, true);
    
    current_tokens = tokens;
    // CRITICAL FIX: Do NOT update stats here - let n_past be the single source of truth
    
    if (!tokens.empty() && llama_ctx) {
        // Process tokens in batch
        ProcessTokensBatch(tokens);
    }
    
    LOG_DEBUG_ContextInfo("Full rebuild completed: " + std::to_string(n_past) + " tokens processed");
    
    // CRITICAL FIX: Query and sync actual context usage after rebuild
    QueryAndSyncActualContextUsage();
}

inline void ContextInfo::RebuildContext_Partial() {
    // Note: context_mutex should already be held by caller
    
    LOG_DEBUG_ContextInfo("Performing partial context rebuild");
    
    // This is a simplified partial rebuild - in practice, you'd implement
    // incremental token processing based on what changed
    // For now, fall back to full rebuild
    RebuildContext_Full(); // Fallback to full rebuild for now
    
    // Note: QueryAndSyncActualContextUsage() already called by RebuildContext_Full()
}

inline void ContextInfo::RebuildContext_TemplateOnly() {
    LOG_DEBUG_ContextInfo("Performing template-only rebuild");
    
    // Just re-render template without reprocessing tokens
    BuildFullPrompt();
}

inline bool ContextInfo::ProcessTokensBatch(const std::vector<int32_t>& tokens) {
    if (!llama_ctx || !batch_initialized) {
        LOG_ERROR_ContextInfo("Llama context or batch not initialized");
        return false;
    }
    
    if (tokens.empty()) {
        return true;
    }
    
    // CRITICAL FIX: Validate context size and bounds before any processing
    if (max_context_tokens == 0) {
        LOG_ERROR_ContextInfo("Context size not set - cannot validate bounds");
        return false;
    }
    
    // CRITICAL FIX: Strict bounds validation - context must have space for tokens plus safety buffer
    const int32_t context_size = static_cast<int32_t>(max_context_tokens);
    const int32_t safety_buffer = ContextConstants::SAFETY_BUFFER_TOKENS;
    
    if (n_past < 0) {
        LOG_ERROR_ContextInfo("Invalid n_past value: " + std::to_string(n_past) + " - cannot continue");
        return false;
    }
    
    // CRITICAL: n_past must never reach context_size - 1 (llama.cpp uses 0-based indexing)
    if (n_past >= (context_size - safety_buffer)) {
        LOG_ERROR_ContextInfo("Context position (" + std::to_string(n_past) + 
                             ") too close to context limit (" + std::to_string(context_size) + 
                             ") - batch processing rejected");
        return false;
    }
    
    // CRITICAL FIX: Check available space with safety margin
    const int32_t available_space = context_size - n_past - safety_buffer;
    if (static_cast<int32_t>(tokens.size()) > available_space) {
        LOG_ERROR_ContextInfo("Token batch size (" + std::to_string(tokens.size()) + 
                             ") exceeds available context space (" + std::to_string(available_space) + 
                             ") - batch processing rejected to prevent overflow");
        return false;
    }
    
    // CRITICAL FIX: Remove recovery logic completely - failures indicate bugs
    // If bounds checking is correct, decode failures should not occur
    // Recovery logic masks the real problems and creates infinite loops
    
    // Validate context state before processing
    if (!ValidateAndSyncContextState()) {
        LOG_ERROR_ContextInfo("Context state validation failed before batch processing");
        return false;
    }
    
    try {
        // Process tokens in batches
        const int32_t n_batch = llama_n_batch(llama_ctx);
        if (n_batch <= 0) {
            LOG_ERROR_ContextInfo("Invalid batch size from llama context: " + std::to_string(n_batch));
            return false;
        }
        
        const size_t max_batch_size = static_cast<size_t>(n_batch);
        
        for (size_t i = 0; i < tokens.size(); i += max_batch_size) {
            size_t end = std::min(i + max_batch_size, tokens.size());
            size_t chunk_size = end - i;
            
            // CRITICAL FIX: Validate that this chunk will fit with safety margin
            const int32_t remaining_space = (context_size - safety_buffer) - n_past;
            if (static_cast<int32_t>(chunk_size) > remaining_space) {
                LOG_ERROR_ContextInfo("Chunk size (" + std::to_string(chunk_size) + 
                                     ") exceeds remaining safe context space (" + std::to_string(remaining_space) + 
                                     ") - stopping batch processing");
                break;
            }
            
            // Clear batch properly (don't free, just reset)
            clear_batch();
            
            // Add tokens to batch manually (proper API)
            for (size_t j = i; j < end && batch.n_tokens < max_batch_size; ++j) {
                size_t batch_idx = j - i;
                
                // CRITICAL FIX: Validate position bounds with safety margin
                const int32_t token_position = n_past + static_cast<int32_t>(batch_idx);
                if (token_position >= (context_size - safety_buffer)) {
                    LOG_ERROR_ContextInfo("Token position (" + std::to_string(token_position) + 
                                         ") would exceed safe context limit (" + std::to_string(context_size - safety_buffer) + 
                                         ") - stopping batch at token " + std::to_string(j));
                    break;
                }
                
                batch.token[batch.n_tokens] = static_cast<llama_token>(tokens[j]);
                batch.pos[batch.n_tokens] = token_position;
                batch.n_seq_id[batch.n_tokens] = 1;  // Number of sequences this token belongs to
                batch.seq_id[batch.n_tokens][0] = 0; // Sequence ID 0
                batch.logits[batch.n_tokens] = (j == end - 1); // Only last token gets logits
                batch.n_tokens++;
            }
            
            // CRITICAL FIX: Skip empty batches
            if (batch.n_tokens == 0) {
                LOG_DEBUG_ContextInfo("Skipping empty batch at position " + std::to_string(n_past));
                break;
            }
            
            LOG_DEBUG_ContextInfo("Processing batch: " + std::to_string(batch.n_tokens) + 
                                 " tokens, positions " + std::to_string(n_past) + 
                                 " to " + std::to_string(n_past + batch.n_tokens - 1) + 
                                 " (context size: " + std::to_string(context_size) + ")");
            
            // Decode batch - if this fails with proper bounds checking, it indicates a bug
            int result = llama_decode(llama_ctx, batch);
            if (result != 0) {
                LOG_ERROR_ContextInfo("CRITICAL BUG: Batch decode failed with result: " + std::to_string(result) + 
                                     " despite proper bounds checking - this should not happen");
                LOG_ERROR_ContextInfo("Debug info: batch_size=" + std::to_string(batch.n_tokens) + 
                                     ", n_past=" + std::to_string(n_past) + 
                                     ", positions=" + std::to_string(n_past) + "-" + 
                                     std::to_string(n_past + batch.n_tokens - 1) + 
                                     ", safe_limit=" + std::to_string(context_size - safety_buffer));
                return false;
            }
            
            n_past += static_cast<int32_t>(batch.n_tokens);
            
            // Track tokens actually processed in this batch (for debugging)
            // NOTE: Removed stats tracking for simplicity
        }
        
        LOG_DEBUG_ContextInfo("Processed " + std::to_string(tokens.size()) + " tokens, n_past=" + std::to_string(n_past));
        
        // CRITICAL FIX: Query and sync actual context usage after batch processing
        QueryAndSyncActualContextUsage();
        
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR_ContextInfo("Exception during token processing: " + std::string(e.what()));
        return false;
    }
}

inline std::string ContextInfo::GenerateResponse(const std::string& prompt) {
    // Note: context_mutex should already be held by caller
    
    if (!InitializeLlamaContext()) {
        LOG_ERROR_ContextInfo("Failed to initialize context for generation");
        return "Error: Context initialization failed";
    }
    
    is_generating = true;
    should_stop_generation = false;
    
    try {
        // Tokenize the prompt
        std::vector<int32_t> prompt_tokens = token_cache->TokenizeText(prompt, true);
        current_tokens = prompt_tokens;
        
        // Process prompt tokens
        if (!ProcessTokensBatch(prompt_tokens)) {
            is_generating = false;
            return "Error: Failed to process prompt";
        }
        
        // Generate response tokens (simplified - real implementation would use sampling)
        std::vector<int32_t> response_tokens;
        
        // Calculate max tokens based on available context space
        // Reserve some space for context management and leave room for user's next message
        const int context_reserve = 256; // Reserve space for context management
        const int available_space = static_cast<int>(max_context_tokens) - n_past - context_reserve;
        const int max_new_tokens = std::max(512, std::min(4096, available_space)); // At least 512, up to 4096 tokens
        
        for (int i = 0; i < max_new_tokens && !should_stop_generation; ++i) {
            // Get logits for next token
            float* logits = llama_get_logits_ith(llama_ctx, batch.n_tokens - 1);
            if (!logits) {
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
            
            // Check for EOS token
            if (next_token == llama_vocab_eos(vocab)) {
                break;
            }
            
            // CRITICAL FIX: Check if we're approaching context limit during generation
            // Use consistent generation buffer to prevent decode failures
            const int32_t generation_buffer = ContextConstants::GENERATION_BUFFER_TOKENS;
            const int32_t current_usage = n_past + static_cast<int32_t>(response_tokens.size());
            if (current_usage >= static_cast<int32_t>(max_context_tokens) - generation_buffer) {
                LOG_DEBUG_ContextInfo("Approaching context limit during generation (usage: " + 
                                     std::to_string(current_usage) + "/" + 
                                     std::to_string(max_context_tokens) + 
                                     "), stopping early with buffer: " + std::to_string(generation_buffer));
                break;
            }
            
            response_tokens.push_back(next_token);
            
            // Process the generated token
            clear_batch();
            
            // Add token to batch manually (proper API)
            batch.token[0] = next_token;
            batch.pos[0] = n_past;
            batch.n_seq_id[0] = 1;  // Single sequence
            batch.seq_id[0][0] = 0; // Sequence ID 0
            batch.logits[0] = true; // Generate logits for next token
            batch.n_tokens = 1;
            
            // CRITICAL FIX: Validate position before decode to prevent overflow
            if (n_past >= static_cast<int32_t>(max_context_tokens)) {
                LOG_ERROR_ContextInfo("Cannot decode token - position (" + std::to_string(n_past) + 
                                     ") exceeds context size (" + std::to_string(max_context_tokens) + ")");
                break;
            }
            
            if (llama_decode(llama_ctx, batch) != 0) {
                LOG_ERROR_ContextInfo("Failed to decode generated token at position " + 
                                     std::to_string(n_past) + " (context size: " + 
                                     std::to_string(max_context_tokens) + ")");
                break;
            }
            
            n_past++;
        }
        
        is_generating = false;
        
        // Query and sync actual context usage after successful generation
        QueryAndSyncActualContextUsage();
        
        // Convert response tokens back to text
        std::string raw_response = token_cache->DetokenizeTokens(response_tokens);
        
        // Extract clean response content (remove template tokens)
        std::string response = LuminaChat::Utilities::ExtractCleanResponse(raw_response);
        
        LOG_DEBUG_ContextInfo("Generated response: " + std::to_string(response_tokens.size()) + 
                             " tokens -> " + response.substr(0, 100) + 
                             (response.length() > 100 ? "..." : ""));
        
        return response;
        
    } catch (const std::exception& e) {
        is_generating = false;
        LOG_ERROR_ContextInfo("Exception during generation: " + std::string(e.what()));
        return "Error: " + std::string(e.what());
    }
}

inline void ContextInfo::ClearContext() {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    if (llama_ctx) {
        llama_memory_clear(llama_get_memory(llama_ctx), true);
        n_past = 0;
    }
    
    current_tokens.clear();
    context_needs_rebuild = false;  // Clear rebuild flag
    
    // CRITICAL FIX: Query and sync actual context usage after clearing
    QueryAndSyncActualContextUsage();
    
    LOG_DEBUG_ContextInfo("Context cleared and state reset");
}

inline void ContextInfo::ClearMessageHistory() {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    message_history.clear();
    context_needs_rebuild = true;
    
    LOG_DEBUG_ContextInfo("Message history cleared");
}



// Helper function to generate context IDs
inline std::string GenerateContextId() {
    static std::atomic<size_t> counter{0};
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    return "ctx_" + std::to_string(timestamp) + "_" + std::to_string(counter++);
}

// Factory methods for creating contexts with default template
inline std::unique_ptr<ContextInfo> ContextInfo::Create(const std::string& context_id, ModelInfo* model, int32_t context_size) {
    return std::make_unique<ContextInfo>(context_id, model, context_size);
}

inline std::unique_ptr<ContextInfo> ContextInfo::Create(ModelInfo* model, int32_t context_size) {
    return std::make_unique<ContextInfo>(model, context_size);
}

inline bool ContextInfo::HandleInputAsync(const std::string& input, const GenerationCallbacks& callbacks, const std::string& username) {
    // Check if already generating
    if (is_generating.load()) {
        if (callbacks.on_error) {
            callbacks.on_error("Generation already in progress");
        }
        return false;
    }
    
    std::string full_prompt;
    bool preparation_success = false;
    
    // Critical section: prepare context for generation
    {
        std::lock_guard<std::mutex> lock(context_mutex);
        
        try {
            state = ContextState::PROCESSING;

            // CRITICAL: This MUST be sent as username, input NOT "user", input; Chat template handles roles dynamically and will allow the AI to interpret properly here.
            // 1. Add user message to history
            message_history.emplace_back(username, input);
            context_needs_rebuild = true;
            
            LOG_DEBUG_ContextInfo("Processing async input: " + username + " -> " + 
                                 input.substr(0, 50) + (input.length() > 50 ? "..." : ""));
            
            // 2. Check context size BEFORE rebuilding - this allows us to trigger summarization
            // before we try to process tokens that might exceed the limit
            std::string estimated_prompt = BuildFullPrompt();
            size_t estimated_tokens = LuminaChat::Utilities::EstimateTokenCount(estimated_prompt);
                  LOG_DEBUG_ContextInfo("Estimated prompt tokens: " + std::to_string(estimated_tokens) + 
                             " (current context: " + std::to_string(static_cast<size_t>(std::max(0, n_past))) + 
                             "/" + std::to_string(max_context_tokens) + ")");
        
        // Check if we're approaching the limit with the new message
        if (max_context_tokens > 0) {
            // Use estimated_tokens directly since BuildFullPrompt() already includes everything
            // n_past would be double counting since it's from previous context state
            size_t total_estimated = estimated_tokens;
            
            // Only check for absolute hard limit overflow - plugin should prevent this from happening
            if (total_estimated >= max_context_tokens - ContextConstants::SAFETY_BUFFER_TOKENS) {
                // This should rarely happen if ContextPruningPlugin is monitoring properly
                LOG_WARNING_ContextInfo("Context approaching hard limit during message addition - plugin monitoring may need adjustment");
                LOG_WARNING_ContextInfo("Current tokens: " + std::to_string(total_estimated) + "/" + std::to_string(max_context_tokens));
                
                // Trust that the plugin will handle this through normal monitoring
                // No emergency pruning here - let the plugin algorithms handle it
            }
            }
            
            // 3. Rebuild context with the new message
            if (context_needs_rebuild || template_manager->IsTemplateDirty()) {
                if (!RebuildContext(RebuildStrategy::FULL)) {
                    state = ContextState::ERROR_STATE;
                    if (callbacks.on_error) {
                        callbacks.on_error("Failed to rebuild context");
                    }
                    return false;
                }
                // Sync stats after successful rebuild
                SyncStatsWithContextState();
            }
            
            // 4. Final safety check after rebuild - trust plugin to handle all pruning
            // Note: Normal threshold monitoring is handled by ContextPruningPlugin
            size_t actual_tokens = static_cast<size_t>(std::max(0, n_past));
            if (max_context_tokens > 0 && actual_tokens >= max_context_tokens - ContextConstants::SAFETY_BUFFER_TOKENS) {
                LOG_WARNING_ContextInfo("Context still near hard limit after rebuild - plugin intervention needed");
                LOG_WARNING_ContextInfo("Actual tokens: " + std::to_string(actual_tokens) + "/" + std::to_string(max_context_tokens));
                
                // Log the problem but don't attempt emergency pruning - that's the plugin's responsibility
                // The plugin should monitor and handle this through proper algorithms
            }
            
            // 5. Prepare for generation
            state = ContextState::GENERATING;
            current_callbacks = callbacks;
            full_prompt = BuildFullPrompt(); // Get the final prompt for generation
            preparation_success = true;
            
        } catch (const std::exception& e) {
            LOG_ERROR_ContextInfo("Exception in HandleInputAsync preparation: " + std::string(e.what()));
            state = ContextState::ERROR_STATE;
            if (callbacks.on_error) {
                callbacks.on_error("Exception: " + std::string(e.what()));
            }
            return false;
        }
    } // Release context_mutex here
    
    // Start async generation OUTSIDE the critical section to avoid deadlock
    if (preparation_success) {
        StartGenerationAsync(full_prompt, callbacks);
        return true;
    }
    
    return false;
}

inline void ContextInfo::StopGeneration() {
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
    
    std::lock_guard<std::mutex> lock(context_mutex);
    if (state == ContextState::GENERATING) {
        state = ContextState::READY;
        is_generating = false;
        LOG_DEBUG_ContextInfo("Generation stopped for context: " + context_id);
    }
}

inline void ContextInfo::StartGenerationAsync(const std::string& prompt, const GenerationCallbacks& callbacks) {
    // Stop any existing generation first
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
            // PERFORMANCE FIX: Minimize critical section duration to prevent UI hitches
            // Pre-validate context state before acquiring lock
            if (state == ContextState::ERROR_STATE) {
                LOG_ERROR_ContextInfo("Context in error state, cannot generate");
                if (callbacks.on_error) {
                    callbacks.on_error("Context in error state");
                }
                is_generating = false;
                return;
            }
            
            {
                // Use try_lock with timeout to prevent indefinite blocking
                std::unique_lock<std::mutex> lock(context_mutex, std::defer_lock);
                if (!lock.try_lock()) {
                    LOG_WARNING_ContextInfo("Context lock timeout - another operation may be blocking");
                    if (callbacks.on_error) {
                        callbacks.on_error("Context busy - try again");
                    }
                    is_generating = false;
                    return;
                }
                
                if (!InitializeLlamaContext()) {
                    LOG_ERROR_ContextInfo("Failed to initialize context for async generation");
                    if (callbacks.on_error) {
                        callbacks.on_error("Context initialization failed");
                    }
                    is_generating = false;
                    state = ContextState::ERROR_STATE;
                    return;
                }
                
                // OPTIMIZATION: Skip tokenization and processing if context was just rebuilt
                // The context should already contain all the prompt tokens from the rebuild
                if (context_needs_rebuild) {
                    LOG_DEBUG_ContextInfo("Context needs rebuild before generation");
                    
                    // Tokenize the prompt
                    std::vector<int32_t> prompt_tokens = token_cache->TokenizeText(prompt, true);
                    current_tokens = prompt_tokens;
                    
                    // Process prompt tokens
                    if (!ProcessTokensBatch(prompt_tokens)) {
                        if (callbacks.on_error) {
                            callbacks.on_error("Failed to process prompt");
                        }
                        is_generating = false;
                        state = ContextState::ERROR_STATE;
                        return;
                    }
                } else {
                    LOG_DEBUG_ContextInfo("Using already-processed context for generation (n_past=" + 
                                         std::to_string(n_past) + ")");
                }
            }
            
            // Generate response tokens with streaming
            std::vector<int32_t> response_tokens;
            
            // PERFORMANCE FIX: Calculate max tokens based on actual available space with better buffer management
            // Reserve space for context management and leave room for user's next message
            const int generation_buffer = 384; // Larger buffer to prevent frequent limit hits
            const int user_message_reserve = 128; // Reserve space for user's next message
            const int total_reserve = generation_buffer + user_message_reserve;
            
            const int available_space = static_cast<int>(max_context_tokens) - n_past - total_reserve;
            const int max_new_tokens = std::max(128, std::min(2048, available_space)); // Reduced max to prevent overruns
            
            LOG_DEBUG_ContextInfo("Generation limits: max_new_tokens=" + std::to_string(max_new_tokens) + 
                                 ", available_space=" + std::to_string(available_space) + 
                                 ", n_past=" + std::to_string(n_past));
            
            for (int i = 0; i < max_new_tokens && !should_stop_generation.load(); ++i) {
                std::lock_guard<std::mutex> lock(context_mutex);
                
                // Verify we have a valid context state for generation
                if (n_past <= 0) {
                    LOG_ERROR_ContextInfo("Invalid context state for generation: n_past=" + std::to_string(n_past));
                    break;
                }
                
                // Get logits for next token
                float* logits = llama_get_logits_ith(llama_ctx, batch.n_tokens - 1);
                if (!logits) {
                    LOG_ERROR_ContextInfo("Failed to get logits for token generation");
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
                
                // Check for EOS token
                if (next_token == llama_vocab_eos(vocab)) {
                    LOG_DEBUG_ContextInfo("EOS token encountered, ending generation");
                    break;
                }
                
                // CRITICAL FIX: Check if we're approaching context limit during generation
                // Use consistent generation buffer to prevent decode failures
                const int32_t generation_buffer = ContextConstants::GENERATION_BUFFER_TOKENS;
                const int32_t current_usage = n_past + static_cast<int32_t>(response_tokens.size());
                if (current_usage >= static_cast<int32_t>(max_context_tokens) - generation_buffer) {
                    LOG_DEBUG_ContextInfo("Approaching context limit during generation (usage: " + 
                                         std::to_string(current_usage) + "/" + 
                                         std::to_string(max_context_tokens) + 
                                         "), stopping early with buffer: " + std::to_string(generation_buffer));
                    break;
                }
                
                response_tokens.push_back(next_token);
                
                // Convert this token to text and stream it
                std::vector<int32_t> single_token = {next_token};
                std::string token_text = token_cache->DetokenizeTokens(single_token);
                
                // Call the streaming callback with the new token
                if (callbacks.on_token && !token_text.empty()) {
                    callbacks.on_token(token_text);
                }
                
                full_response += token_text;
                
                // Process the generated token for next iteration
                clear_batch();
                
                // Add token to batch manually (proper API)
                batch.token[0] = next_token;
                batch.pos[0] = n_past;
                batch.n_seq_id[0] = 1;  // Single sequence
                batch.seq_id[0][0] = 0; // Sequence ID 0
                batch.logits[0] = true; // Generate logits for next token
                batch.n_tokens = 1;
                
                // CRITICAL FIX: Validate position before decode to prevent overflow
                if (n_past >= static_cast<int32_t>(max_context_tokens)) {
                    LOG_ERROR_ContextInfo("Cannot decode token - position (" + std::to_string(n_past) + 
                                         ") exceeds context size (" + std::to_string(max_context_tokens) + ")");
                    break;
                }
                
                if (llama_decode(llama_ctx, batch) != 0) {
                    LOG_ERROR_ContextInfo("Failed to decode generated token at position " + 
                                         std::to_string(n_past) + " (context size: " + 
                                         std::to_string(max_context_tokens) + ")");
                    break;
                }
                
                n_past++;
                
                // Small delay to prevent overwhelming the UI
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            
            success = !should_stop_generation.load();
            
            // Add assistant response to message history
            {
                std::lock_guard<std::mutex> lock(context_mutex);
                if (success && !full_response.empty()) {
                    message_history.emplace_back("assistant", full_response);
                }
                
                // Query and sync actual context usage after successful generation
                if (success) {
                    QueryAndSyncActualContextUsage();
                }
                
                state = ContextState::READY;
                is_generating = false;
            
            // Background context maintenance is now handled by ContextPruningPlugin monitoring
            // No need for direct pruning calls - plugin will handle this through its regular monitoring
            }
            
            LOG_DEBUG_ContextInfo("Async generation completed: " + std::to_string(response_tokens.size()) + 
                                 " tokens -> " + full_response.substr(0, 100) + 
                                 (full_response.length() > 100 ? "..." : ""));
            
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(context_mutex);
            is_generating = false;
            state = ContextState::ERROR_STATE;
            LOG_ERROR_ContextInfo("Exception during async generation: " + std::string(e.what()));
            
            if (callbacks.on_error) {
                callbacks.on_error("Exception: " + std::string(e.what()));
            }
            return;
        }
        
        // Call completion callback
        if (callbacks.on_complete) {
            callbacks.on_complete(full_response, success);
        }
    });
}

inline bool ContextInfo::ValidateAndSyncContextState() {
    // NOTE: context_mutex should already be held by caller
    
    if (!llama_ctx) {
        return false;
    }
    
    // CRITICAL FIX: Check context size bounds first
    if (max_context_tokens == 0) {
        LOG_ERROR_ContextInfo("Context size not initialized - cannot validate state");
        return false;
    }
    
    const int32_t max_context = static_cast<int32_t>(max_context_tokens);
    
    // CRITICAL FIX: Strict validation with safety margin
    const int32_t safety_buffer = ContextConstants::SAFETY_BUFFER_TOKENS;
    
    if (n_past < 0) {
        LOG_ERROR_ContextInfo("n_past is negative (" + std::to_string(n_past) + ") - invalid state");
        return false;
    }
    
    // CRITICAL: n_past must never reach the context limit
    if (n_past >= (max_context - safety_buffer)) {
        LOG_ERROR_ContextInfo("CRITICAL: n_past (" + std::to_string(n_past) + 
                             ") is at or beyond safe context limit (" + std::to_string(max_context - safety_buffer) + 
                             ") - context requires rebuild");
        return false;
    }
    
    // Check if context was flagged for rebuild
    if (context_needs_rebuild) {
        LOG_DEBUG_ContextInfo("Context rebuild flag detected - n_past will be reset during rebuild");
        context_needs_rebuild = false;
    }
    
    // Validate consistency between n_past and available tokens (more lenient check)
    if (n_past > static_cast<int32_t>(current_tokens.size() + 500)) {
        LOG_WARNING_ContextInfo("n_past (" + std::to_string(n_past) + 
                               ") appears inconsistent with token count (" + std::to_string(current_tokens.size()) + 
                               ") - this may indicate context state issues");
        return false;
    }
    
    // Sync stats with validated state
    SyncStatsWithContextState();
    return true;
}

inline void ContextInfo::SyncStatsWithContextState() {
    // NOTE: context_mutex should already be held by caller
    // CRITICAL FIX: This method is now simplified since we removed the ContextStats struct
    // n_past is the single source of truth for context size
    
    LOG_DEBUG_ContextInfo("Context state synchronized: n_past=" + 
                         std::to_string(n_past) + ", max_context_tokens=" + 
                         std::to_string(max_context_tokens));
}

inline void ContextInfo::QueryAndSyncActualContextUsage() {
    // NOTE: context_mutex should already be held by caller
    
    if (!llama_ctx) {
        LOG_WARNING_ContextInfo("Cannot query actual context usage - llama context not initialized");
        return;
    }
    
    // Query the actual context position from llama.cpp
    // Note: llama.cpp tracks the current position internally, and we should sync with it
    // In the current llama.cpp API, the context position is managed by the decode operations
    // and our n_past should already be accurate, but we can add additional validation
    
    int32_t actual_n_past = n_past; // In current API, we manage this ourselves
    
    // Additional validation: ensure the position makes sense
    if (actual_n_past < 0) {
        LOG_ERROR_ContextInfo("Invalid actual context position: " + std::to_string(actual_n_past) + " - resetting to 0");
        actual_n_past = 0;
        n_past = 0;
    }
    
    if (max_context_tokens > 0 && actual_n_past > static_cast<int32_t>(max_context_tokens)) {
        LOG_ERROR_ContextInfo("Actual context position (" + std::to_string(actual_n_past) + 
                             ") exceeds context size (" + std::to_string(max_context_tokens) + 
                             ") - this indicates a serious bug");
        // Don't automatically fix this as it indicates a bug in our logic
    }
    
    // Update our internal tracking if there's a discrepancy
    if (n_past != actual_n_past) {
        LOG_WARNING_ContextInfo("Context position mismatch detected - syncing internal state");
        LOG_WARNING_ContextInfo("Internal n_past: " + std::to_string(n_past) + 
                               ", Actual n_past: " + std::to_string(actual_n_past));
        n_past = actual_n_past;
    }
    
    // Log the current actual usage for monitoring
    size_t actual_usage = static_cast<size_t>(std::max(0, actual_n_past));
    float usage_ratio = max_context_tokens > 0 ? 
        static_cast<float>(actual_usage) / max_context_tokens : 0.0f;
    
    LOG_DEBUG_ContextInfo("Actual context usage after operation: " + 
                         std::to_string(actual_usage) + "/" + 
                         std::to_string(max_context_tokens) + 
                         " (" + std::to_string(static_cast<int>(usage_ratio * 100)) + "%)");
}

inline void ContextInfo::PruneMessageHistoryWithSummary(const std::string& summary, size_t keep_recent_messages) {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    if (message_history.size() <= keep_recent_messages) {
        LOG_DEBUG_ContextInfo("No pruning needed: only " + std::to_string(message_history.size()) + " messages");
        return;
    }
    
    // Extract messages to be pruned
    size_t prune_count = message_history.size() - keep_recent_messages;
    std::vector<std::pair<std::string, std::string>> pruned_messages;
    pruned_messages.assign(message_history.begin(), message_history.begin() + prune_count);
    
    // REMOVED: Direct buffer management - plugins now handle this through orchestrator coordination
    // Orchestrator monitors contexts and triggers plugins as needed
    
    // Immediately prune from active history
    message_history.erase(message_history.begin(), message_history.begin() + prune_count);
    context_needs_rebuild = true;
    
    LOG_ContextInfo("Pruned " + std::to_string(prune_count) + " messages from context (plugins handle summarization through orchestrator). " +
                   "Keeping " + std::to_string(keep_recent_messages) + " recent messages.");
}

inline void ContextInfo::ApplyCompletedSummary(const std::string& summary) {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    if (!summary.empty()) {
        // Add summary to chronological list and update template with all summaries
        AddSummaryToList(summary);
        UpdateTemplateWithAllSummaries();
        context_needs_rebuild = true;
        
        LOG_ContextInfo("Added new summary to chronological list (total: " + 
                       std::to_string(summaries.size()) + "/" + std::to_string(MAX_SUMMARIES) + "): " + 
                       summary.substr(0, 100) + (summary.length() > 100 ? "..." : ""));
    }
}

// Advanced features
inline void ContextInfo::SetMaxContextTokens(size_t max_tokens) {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    // Update max context tokens
    max_context_tokens = max_tokens;
    
    LOG_DEBUG_ContextInfo("Max context tokens set to: " + std::to_string(max_tokens));
}

inline std::string ContextInfo::GetCurrentPrompt() const {
    // For const method, return a simple representation without modifying state
    std::ostringstream prompt_stream;
    prompt_stream << "Current context with " << message_history.size() << " messages";
    return prompt_stream.str();
}

inline void ContextInfo::PruneContextImmediate(size_t keep_recent_messages) {
    std::lock_guard<std::mutex> lock(context_mutex);
    PruneContextImmediate_Internal(keep_recent_messages);
}

inline std::vector<std::pair<std::string, std::string>> ContextInfo::PruneContextImmediateWithExtraction(size_t keep_recent_messages) {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    if (message_history.size() <= keep_recent_messages) {
        LOG_DEBUG_ContextInfo("No pruning needed: only " + std::to_string(message_history.size()) + " messages");
        return {}; // Return empty vector
    }
    
    // Extract messages to be pruned for summarization
    size_t prune_count = message_history.size() - keep_recent_messages;
    std::vector<std::pair<std::string, std::string>> pruned_messages;
    pruned_messages.assign(message_history.begin(), message_history.begin() + prune_count);
    
    // Immediately prune from active history
    message_history.erase(message_history.begin(), message_history.begin() + prune_count);
    context_needs_rebuild = true;
    
    LOG_ContextInfo("Pruned " + std::to_string(prune_count) + " messages from context - returning for orchestrator coordination");
    
    return pruned_messages; // Return extracted messages for summarization
}

inline void ContextInfo::PruneContextImmediate_Internal(size_t keep_recent_messages) {
    // Note: context_mutex should already be held by caller
    
    if (message_history.size() <= keep_recent_messages) {
        LOG_DEBUG_ContextInfo("No pruning needed: only " + std::to_string(message_history.size()) + " messages");
        return;
    }
    
    // Extract messages to be pruned for potential summarization
    size_t prune_count = message_history.size() - keep_recent_messages;
    std::vector<std::pair<std::string, std::string>> pruned_messages;
    pruned_messages.assign(message_history.begin(), message_history.begin() + prune_count);
    
    // Immediately prune from active history
    message_history.erase(message_history.begin(), message_history.begin() + prune_count);
    context_needs_rebuild = true;
    
    LOG_ContextInfo("Pruned " + std::to_string(prune_count) + " messages from context - orchestrator will handle summarization");
    
    // Plugin coordination handles all summarization requests through the extraction method
}

// REMOVED: Static member definitions moved to respective plugins
// - pruning_buffer_mutex, global_pruning_buffer moved to SummarizationPlugin
// - emotional_analysis_buffer_mutex, global_emotional_analysis_buffer moved to EmoTagPlugin

// Implementation of new summary management methods
inline void ContextInfo::AddSummaryToList(const std::string& summary) {
    if (summary.empty()) return;
    
    // Add new summary to the end (newest)
    summaries.push_back(summary);
    
    // Enforce maximum limit by removing oldest summary if needed
    if (summaries.size() > MAX_SUMMARIES) {
        summaries.erase(summaries.begin()); // Remove oldest (first) summary
        LOG_ContextInfo("Removed oldest summary to maintain maximum of " + 
                       std::to_string(MAX_SUMMARIES) + " summaries");
    }
    
    LOG_ContextInfo("Summary list updated: " + std::to_string(summaries.size()) + 
                   " summaries in chronological order");
}

inline void ContextInfo::UpdateTemplateWithAllSummaries() {
    if (template_manager) {
        template_manager->UpdateMultipleSummaries(summaries);
        LOG_ContextInfo("Updated template with " + std::to_string(summaries.size()) + 
                       " summaries in chronological order");
    }
}
