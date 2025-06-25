#pragma once

#include "ChatTemplateManager.hpp"
#include "ModelInfo.hpp"
#include "TokenCache.hpp"
#include "Logger.hpp"
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
#include <future>
#include <condition_variable>

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
    AWAITING_SUMMARIZATION,
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

struct ContextStats {
    size_t total_tokens_processed = 0;
    size_t current_context_tokens = 0;
    size_t max_context_tokens = 0;
    size_t message_pairs = 0;
    size_t template_renders = 0;
    size_t full_rebuilds = 0;
    size_t partial_rebuilds = 0;
    std::chrono::steady_clock::time_point created_time;
    std::chrono::steady_clock::time_point last_activity;
    
    ContextStats() : created_time(std::chrono::steady_clock::now()), 
                    last_activity(std::chrono::steady_clock::now()) {}
                    
    float GetContextUsageRatio() const {
        return max_context_tokens > 0 ? 
            static_cast<float>(current_context_tokens) / max_context_tokens : 0.0f;
    }
    
    std::chrono::seconds GetAge() const {
        return std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - created_time);
    }
    
    std::chrono::seconds GetTimeSinceActivity() const {
        return std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - last_activity);
    }
};

/**
 * Individual conversation context management with dynamic template system
 * 
 * Revolutionary Processing Flow:
 * 1. Receive input from Orchestrator (pre-sanitized if from Discord)
 * 2. Add pure conversation pair to message history (no system contamination)
 * 3. Render dynamic template with current message history via ChatTemplateManager
 * 4. Process rendered template to tokens via TokenCache
 * 5. Check context size - trigger summarization plugin if needed (updates template sections)
 * 6. Generate LLM response using rendered template
 * 7. Detokenize response via TokenCache
 * 8. Add response pair to message history
 * 9. Return human-readable response
 * 
 * Template Integration Benefits:
 * - Clean Message History: Only actual conversation, no system pollution
 * - Dynamic Context: Summaries, system messages embedded in template sections
 * - Plugin-Driven Updates: Summarization updates template directly, not message history
 * - Per-Context Flexibility: Each context can have completely different template evolution
 */
class ContextInfo {
private:    // Core components
    std::unique_ptr<ChatTemplateManager> template_manager;
    ModelInfo* parent_model;
    TokenCache* token_cache;
    
    // Context state
    std::string context_id;
    ContextState state;
    ContextStats stats;
    mutable std::mutex context_mutex;
    
    // Message history - pure conversation only
    std::vector<std::pair<std::string, std::string>> message_history; // (role, content)
    
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
    std::atomic<bool> needs_background_summarization{false}; // Flag for post-generation summarization
    std::unique_ptr<std::thread> generation_thread;
    std::condition_variable generation_cv;
    std::mutex generation_mutex;
    GenerationCallbacks current_callbacks;
    
    // Callback for summarization requests
    std::function<void(std::string, std::string)> summarization_callback;
    
    // Helper methods (moved to public for testing)
    
    // Context size management
    bool IsNearContextLimit(float threshold = 0.8f) const;
    void RequestSummarization(const std::string& content);
    void UpdateStats();
    
public:    // Constructor overloads
    ContextInfo(const std::string& context_id, ModelInfo* model, const std::string& base_template = "");
    ContextInfo(ModelInfo* model, const std::string& base_template = ""); // For testing with auto-generated context_id
    
    // Factory method for default template
    static std::unique_ptr<ContextInfo> CreateWithDefaultTemplate(const std::string& context_id, ModelInfo* model);
    static std::unique_ptr<ContextInfo> CreateWithDefaultTemplate(ModelInfo* model); // Auto-generated ID
    
    // Destructor
    ~ContextInfo();
    
    // Core processing interface
    std::string HandleInput(const std::string& input, const std::string& username = "user"); // DEPRECATED for UI: Use HandleInputAsync instead
    bool HandleInputAsync(const std::string& input, const GenerationCallbacks& callbacks, const std::string& username = "user");
    void AddHistoricalMessage(const std::string& role, const std::string& content);
    
    // Generation control
    void StopGeneration();
    bool IsGenerating() const { return is_generating.load(); }
    bool NeedsBackgroundSummarization() const { return needs_background_summarization.load(); }
    
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
    void AddPastSessionMemory(const std::string& memory);
    
    // Context management
    bool RebuildContext(RebuildStrategy strategy = RebuildStrategy::FULL);
    void ClearContext();
    void ClearMessageHistory();
    
    // State and statistics
    ContextState GetState() const { return state; }
    const ContextStats& GetStats() const { return stats; }
    const std::string& GetContextId() const { return context_id; }
    
    // Get actual context usage based on n_past (single source of truth)
    size_t GetActualContextTokens() const { return static_cast<size_t>(std::max(0, n_past)); }
    float GetActualContextUsageRatio() const {
        return stats.max_context_tokens > 0 ? 
            static_cast<float>(GetActualContextTokens()) / stats.max_context_tokens : 0.0f;
    }
    
    // Message history access
    const std::vector<std::pair<std::string, std::string>>& GetMessageHistory() const { return message_history; }
    size_t GetMessageCount() const { return message_history.size(); }
    std::string GetConversationHistory() const;  // String representation of conversation
    
    // Template access
    ChatTemplateManager& GetTemplateManager() { return *template_manager; }
    const ChatTemplateManager& GetTemplateManager() const { return *template_manager; }
    
    // Callback registration for plugin integration
    void RegisterSummarizationCallback(std::function<void(std::string, std::string)> callback);
    
    // Context validation
    bool ValidateContext() const;
    
    // Advanced features
    void SetMaxContextTokens(size_t max_tokens);
    std::string GetCurrentPrompt() const;
    
    // Debug and testing helpers
    void DumpContextInfo() const;
    std::string GetContextSummary() const;
    void SetContextSizeForTesting(size_t test_size); // For testing with smaller context sizes
    
    // Template rendering and tokenization (made public for testing)
    std::string BuildFullPrompt();
    std::vector<int32_t> TokenizePrompt(const std::string& prompt);
    std::string DetokenizeResponse(const std::vector<int32_t>& tokens);
    
    // Batch management methods - context-specific
    void clear_batch();
    
    // Context rebuilding helpers (made public for testing)
    void RebuildContext_Full();
    void RebuildContext_Partial();
    void RebuildContext_TemplateOnly();
    
    // Additional method implementations for testing compatibility
    void AddMessage(const std::string& role, const std::string& content);
    std::vector<std::pair<std::string, std::string>> GetMessages() const;
    int32_t GetCurrentTokenCount() const;
    void UpdateMotif(const std::string& motif);
    
    // Validate and synchronize context state before batch processing
    bool ValidateAndSyncContextState();
    
    // Synchronize stats with actual context state (n_past) - call after all context operations
    void SyncStatsWithContextState();
    
    // Context pruning for summarization
    void PruneMessageHistoryWithSummary(const std::string& summary, size_t keep_recent_messages = 5);
};

// Helper functions for context management
namespace ContextUtils {
    // Estimate token count for text without full tokenization
    inline size_t EstimateTokenCount(const std::string& text) {
        // Rough estimation: ~4 characters per token on average
        // This is a simplification - actual tokenization varies by model
        return text.length() / 4;
    }
    
    // Calculate optimal batch size for context rebuilding
    inline size_t CalculateOptimalBatchSize(size_t context_size, size_t available_tokens) {
        // Calculate batch size as a percentage of available space
        size_t batch_size = available_tokens / 10; // 10% of available space
        return std::max(batch_size, static_cast<size_t>(512)); // Minimum 512 tokens
    }
    
    // Validate message history integrity
    inline bool ValidateMessageHistory(const std::vector<std::pair<std::string, std::string>>& history) {
        // Check for basic consistency
        for (const auto& msg : history) {
            if (msg.first.empty() || msg.second.empty()) {
                return false;
            }
            
            // Valid roles should be user, assistant, system, etc.
            if (msg.first != "user" && msg.first != "assistant" && msg.first != "system") {
                // Allow other roles but warn they might not be optimal
            }
        }
        
        return true;
    }
    
    // Extract conversation summary for debugging
    inline std::string SummarizeConversation(const std::vector<std::pair<std::string, std::string>>& history, 
                                           size_t max_lines = 10) {
        std::ostringstream oss;
        
        if (history.empty()) {
            return "No conversation history";
        }
        
        oss << "Conversation summary (" << history.size() << " messages):" << std::endl;
        
        size_t start_idx = history.size() > max_lines ? history.size() - max_lines : 0;
        
        for (size_t i = start_idx; i < history.size(); ++i) {
            const auto& msg = history[i];
            std::string preview = msg.second.substr(0, 60);
            if (msg.second.length() > 60) {
                preview += "...";
            }
            
            oss << "[" << i << "] " << msg.first << ": " << preview << std::endl;
        }
        
        if (start_idx > 0) {
            oss << "... (" << start_idx << " earlier messages)" << std::endl;
        }
        
        return oss.str();
    }
}

// Inline implementation of ContextInfo methods
inline ContextInfo::ContextInfo(const std::string& context_id, ModelInfo* model, const std::string& base_template)
    : context_id(context_id)
    , parent_model(model)
    , token_cache(model ? &model->GetTokenCache() : nullptr)
    , state(ContextState::READY)
    , template_manager(std::make_unique<ChatTemplateManager>(
        base_template.empty() ? ChatTemplateManager::GetDefaultTemplate() : base_template))
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
    
    // Initialize llama context from parent model
    // Note: This would need actual llama.cpp integration
    // For testing, we'll skip the actual llama context creation
    
    if (!template_manager->ValidateTemplate()) {
        LOG_ERROR_ContextInfo("Invalid base template provided");
        state = ContextState::ERROR_STATE;
        return;
    }
    
    LOG_DEBUG_ContextInfo("ContextInfo created: " + context_id);
    UpdateStats();
}

inline ContextInfo::ContextInfo(ModelInfo* model, const std::string& base_template)
    : ContextInfo(GenerateContextId(), model, base_template) // Auto-generate context_id
{
    LOG_DEBUG_ContextInfo("ContextInfo created with auto-generated ID: " + context_id);
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
        ctx_params.n_ctx = parent_model->GetConfig().context_size;
        ctx_params.n_batch = std::min(512, parent_model->GetConfig().context_size / 8);
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
        stats.max_context_tokens = ctx_params.n_ctx;
        
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
    
    if (state == ContextState::AWAITING_SUMMARIZATION) {
        LOG_ContextInfo("Input queued - context awaiting summarization");
        return "Processing... please wait";
    }
    
    state = ContextState::PROCESSING;
    UpdateStats();
    
    try {
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
        
        // 4. Check context size - trigger summarization if needed
        std::vector<int32_t> prompt_tokens = TokenizePrompt(full_prompt);
        if (IsNearContextLimit(0.8f)) {
            LOG_ContextInfo("Context approaching limit, requesting summarization");
            
            // Extract content for summarization (recent conversation)
            std::ostringstream content_stream;
            size_t start_idx = message_history.size() > 20 ? message_history.size() - 20 : 0;
            for (size_t i = start_idx; i < message_history.size(); ++i) {
                content_stream << message_history[i].first << ": " << message_history[i].second << "\n";
            }
            
            RequestSummarization(content_stream.str());
            state = ContextState::AWAITING_SUMMARIZATION;
            return "Context full - summarizing recent conversation...";
        }
        
        // 5. Generate LLM response using actual AI model
        std::string response = GenerateResponse(full_prompt);
        
        // 6. Add assistant response to message history
        message_history.emplace_back("assistant", response);
        
        state = ContextState::READY;
        UpdateStats();
        
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
    UpdateStats();
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
        UpdateStats();
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
    stats.template_renders++;
    
    LOG_DEBUG_ContextInfo("Built full prompt: " + std::to_string(rendered_prompt.length()) + " characters");
    return rendered_prompt;
}

inline std::vector<int32_t> ContextInfo::TokenizePrompt(const std::string& prompt) {
    if (!parent_model || !parent_model->IsLoaded()) {
        LOG_ERROR_ContextInfo("Parent model not available for tokenization");
        return {};
    }
    
    // Use ModelInfo's tokenization
    std::vector<llama_token> llama_tokens = parent_model->TokenizeText(prompt, true);
    
    // Convert to int32_t
    std::vector<int32_t> tokens;
    tokens.reserve(llama_tokens.size());
    for (llama_token token : llama_tokens) {
        tokens.push_back(static_cast<int32_t>(token));
    }
    
    current_tokens = tokens;
    // CRITICAL FIX: Do NOT update stats here - let n_past be the single source of truth
    // stats.current_context_tokens = tokens.size(); // REMOVED - causes desynchronization
    
    LOG_DEBUG_ContextInfo("Tokenized prompt: " + std::to_string(tokens.size()) + " tokens");
    return tokens;
}

inline std::string ContextInfo::DetokenizeResponse(const std::vector<int32_t>& tokens) {
    if (!parent_model || !parent_model->IsLoaded()) {
        LOG_ERROR_ContextInfo("Parent model not available for detokenization");
        return "";
    }
    
    // Convert to llama_token
    std::vector<llama_token> llama_tokens;
    llama_tokens.reserve(tokens.size());
    for (int32_t token : tokens) {
        llama_tokens.push_back(static_cast<llama_token>(token));
    }
    
    // Use ModelInfo's detokenization
    std::string result = parent_model->DetokenizeTokens(llama_tokens);
    
    LOG_DEBUG_ContextInfo("Detokenized response: " + std::to_string(result.length()) + " characters");
    return result;
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
    std::vector<int32_t> tokens = TokenizePrompt(full_prompt);
    
    if (!tokens.empty() && llama_ctx) {
        // Process tokens in batch
        ProcessTokensBatch(tokens);
    }
    
    stats.full_rebuilds++;
    LOG_DEBUG_ContextInfo("Full rebuild completed: " + std::to_string(n_past) + " tokens processed");
    
    // CRITICAL FIX: Synchronize stats with actual context state after rebuild
    SyncStatsWithContextState();
}

inline void ContextInfo::RebuildContext_Partial() {
    // Note: context_mutex should already be held by caller
    
    LOG_DEBUG_ContextInfo("Performing partial context rebuild");
    
    // This is a simplified partial rebuild - in practice, you'd implement
    // incremental token processing based on what changed
    // For now, fall back to full rebuild
    RebuildContext_Full(); // Fallback to full rebuild for now
    
    stats.partial_rebuilds++;
    
    // CRITICAL FIX: Synchronize stats with actual context state after partial rebuild
    SyncStatsWithContextState();
}

inline void ContextInfo::RebuildContext_TemplateOnly() {
    LOG_DEBUG_ContextInfo("Performing template-only rebuild");
    
    // Just re-render template without reprocessing tokens
    BuildFullPrompt();
    stats.template_renders++;
}

inline bool ContextInfo::ProcessTokensBatch(const std::vector<int32_t>& tokens) {
    if (!llama_ctx || !batch_initialized) {
        LOG_ERROR_ContextInfo("Llama context or batch not initialized");
        return false;
    }
    
    if (tokens.empty()) {
        return true;
    }
    
    // CRITICAL FIX: Validate and synchronize context state before batch processing
    if (!ValidateAndSyncContextState()) {
        LOG_ERROR_ContextInfo("Context state validation failed - resetting context");
        n_past = 0;
        if (llama_ctx) {
            llama_memory_clear(llama_get_memory(llama_ctx), true);
        }
        // Sync stats after emergency reset
        SyncStatsWithContextState();
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
            
            // Clear batch properly (don't free, just reset)
            clear_batch();
            
            // Add tokens to batch manually (proper API)
            for (size_t j = i; j < end && batch.n_tokens < max_batch_size; ++j) {
                size_t batch_idx = j - i;
                
                batch.token[batch.n_tokens] = static_cast<llama_token>(tokens[j]);
                batch.pos[batch.n_tokens] = n_past + static_cast<int32_t>(batch_idx);
                batch.n_seq_id[batch.n_tokens] = 1;  // Number of sequences this token belongs to
                batch.seq_id[batch.n_tokens][0] = 0; // Sequence ID 0
                batch.logits[batch.n_tokens] = (j == end - 1); // Only last token gets logits
                batch.n_tokens++;
            }
            
            LOG_DEBUG_ContextInfo("Processing batch: " + std::to_string(batch.n_tokens) + 
                                 " tokens, positions " + std::to_string(n_past) + 
                                 " to " + std::to_string(n_past + chunk_size - 1));
            
            // Decode batch
            int result = llama_decode(llama_ctx, batch);
            if (result != 0) {
                LOG_ERROR_ContextInfo("Batch decode failed with result: " + std::to_string(result) + 
                                     " (batch size: " + std::to_string(batch.n_tokens) + 
                                     ", n_past: " + std::to_string(n_past) + ")");
                
                // Try to recover by clearing context and starting fresh
                if (result == 1) { // Common error code for context overflow
                    LOG_ContextInfo("Attempting recovery from batch decode failure - clearing context");
                    if (llama_ctx) {
                        llama_memory_clear(llama_get_memory(llama_ctx), true);
                    }
                    n_past = 0;
                    SyncStatsWithContextState();
                    
                    // Try processing the batch again from clean state
                    clear_batch();
                    for (size_t j = i; j < end && batch.n_tokens < max_batch_size; ++j) {
                        size_t batch_idx = j - i;
                        batch.token[batch.n_tokens] = static_cast<llama_token>(tokens[j]);
                        batch.pos[batch.n_tokens] = static_cast<int32_t>(batch_idx);
                        batch.n_seq_id[batch.n_tokens] = 1;
                        batch.seq_id[batch.n_tokens][0] = 0;
                        batch.logits[batch.n_tokens] = (j == end - 1);
                        batch.n_tokens++;
                    }
                    
                    result = llama_decode(llama_ctx, batch);
                    if (result != 0) {
                        LOG_ERROR_ContextInfo("Recovery attempt failed with result: " + std::to_string(result));
                        return false;
                    } else {
                        LOG_ContextInfo("Successfully recovered from batch decode failure");
                    }
                } else {
                    return false;
                }
            }
            
            n_past += static_cast<int32_t>(chunk_size);
        }
        
        LOG_DEBUG_ContextInfo("Processed " + std::to_string(tokens.size()) + " tokens, n_past=" + std::to_string(n_past));
        
        // CRITICAL FIX: Synchronize stats with actual context state after batch processing
        SyncStatsWithContextState();
        
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
        std::vector<int32_t> prompt_tokens = TokenizePrompt(prompt);
        
        // Process prompt tokens
        if (!ProcessTokensBatch(prompt_tokens)) {
            is_generating = false;
            return "Error: Failed to process prompt";
        }
        
        // Generate response tokens (simplified - real implementation would use sampling)
        std::vector<int32_t> response_tokens;
        const int max_new_tokens = 512;
        
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
            
            if (llama_decode(llama_ctx, batch) != 0) {
                LOG_ERROR_ContextInfo("Failed to decode generated token");
                break;
            }
            
            n_past++;
        }
        
        is_generating = false;
        
        // Convert response tokens back to text
        std::string response = DetokenizeResponse(response_tokens);
        
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
    
    // CRITICAL FIX: Synchronize stats with actual context state after clearing
    SyncStatsWithContextState();
    
    LOG_DEBUG_ContextInfo("Context cleared and state reset");
}

inline void ContextInfo::ClearMessageHistory() {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    message_history.clear();
    context_needs_rebuild = true;
    stats.message_pairs = 0;
    
    LOG_DEBUG_ContextInfo("Message history cleared");
}

inline bool ContextInfo::IsNearContextLimit(float threshold) const {
    if (stats.max_context_tokens == 0) {
        LOG_DEBUG_ContextInfo("max_context_tokens not set, cannot check context limit");
        return false;
    }
    
    // CRITICAL FIX: Use n_past as the single source of truth for context size
    // This ensures accurate context limit detection based on actual llama.cpp state
    size_t actual_context_tokens = static_cast<size_t>(std::max(0, n_past));
    float usage = static_cast<float>(actual_context_tokens) / stats.max_context_tokens;
    
    LOG_DEBUG_ContextInfo("Context usage check: " + std::to_string(actual_context_tokens) + 
                         "/" + std::to_string(stats.max_context_tokens) + 
                         " (" + std::to_string(usage * 100.0f) + "%) vs threshold " + 
                         std::to_string(threshold * 100.0f) + "%");
    
    bool near_limit = usage >= threshold;
    if (near_limit) {
        LOG_ContextInfo("Context is near limit: " + std::to_string(usage * 100.0f) + "% >= " + 
                       std::to_string(threshold * 100.0f) + "%");
    }
    
    return near_limit;
}

inline void ContextInfo::RequestSummarization(const std::string& content) {
    if (summarization_callback) {
        // Determine if this is background or immediate summarization based on current state
        bool is_background = (state == ContextState::READY);
        std::string type = is_background ? "background" : "immediate";
        
        LOG_ContextInfo("Requesting " + type + " summarization for context: " + context_id + 
                       " (content length: " + std::to_string(content.length()) + " chars)");
        LOG_DEBUG_ContextInfo("Summarization content preview: " + 
                             content.substr(0, 200) + (content.length() > 200 ? "..." : ""));
        summarization_callback(context_id, content);
    } else {
        LOG_ERROR_ContextInfo("No summarization callback available for context: " + context_id + 
                             " - context will continue to grow without pruning!");
        
        // As a fallback, try to prune without summarization if context is critically full
        if (IsNearContextLimit(0.95f)) {
            LOG_ContextInfo("Emergency pruning without summarization - context critically full");
            if (message_history.size() > 6) {
                // Keep only the last 3 message pairs
                std::vector<std::pair<std::string, std::string>> recent_messages;
                recent_messages.assign(message_history.end() - 6, message_history.end());
                message_history = std::move(recent_messages);
                context_needs_rebuild = true;
                UpdateStats();
                LOG_ContextInfo("Emergency pruned to " + std::to_string(message_history.size()) + " messages");
            }
        }
    }
}

inline void ContextInfo::UpdateStats() {
    stats.message_pairs = message_history.size();
    // CRITICAL FIX: Use n_past as the single source of truth for context size
    // This ensures UI and internal logic always reflect the actual llama.cpp context state
    stats.current_context_tokens = static_cast<size_t>(std::max(0, n_past));
    stats.total_tokens_processed += static_cast<size_t>(std::max(0, n_past));
    stats.last_activity = std::chrono::steady_clock::now();
}

inline void ContextInfo::RegisterSummarizationCallback(std::function<void(std::string, std::string)> callback) {
    summarization_callback = std::move(callback);
    LOG_ContextInfo("Summarization callback registered for context: " + context_id);
}

// Helper function to generate context IDs
inline std::string GenerateContextId() {
    static std::atomic<size_t> counter{0};
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    return "ctx_" + std::to_string(timestamp) + "_" + std::to_string(counter++);
}

// Factory methods for creating contexts with default template
inline std::unique_ptr<ContextInfo> ContextInfo::CreateWithDefaultTemplate(const std::string& context_id, ModelInfo* model) {
    return std::make_unique<ContextInfo>(context_id, model, ""); // Empty string triggers default template
}

inline std::unique_ptr<ContextInfo> ContextInfo::CreateWithDefaultTemplate(ModelInfo* model) {
    return std::make_unique<ContextInfo>(model, ""); // Empty string triggers default template
}

inline bool ContextInfo::HandleInputAsync(const std::string& input, const GenerationCallbacks& callbacks, const std::string& username) {
    // Check if already generating
    if (is_generating.load()) {
        if (callbacks.on_error) {
            callbacks.on_error("Generation already in progress");
        }
        return false;
    }
    
    std::lock_guard<std::mutex> lock(context_mutex);
    
    try {
        state = ContextState::PROCESSING;
        
        // 1. Add user message to history
        message_history.emplace_back(username, input);
        context_needs_rebuild = true;
        
        LOG_DEBUG_ContextInfo("Processing async input: " + username + " -> " + 
                             input.substr(0, 50) + (input.length() > 50 ? "..." : ""));
        
        // 2. Check context size BEFORE rebuilding - this allows us to trigger summarization
        // before we try to process tokens that might exceed the limit
        std::string estimated_prompt = BuildFullPrompt();
        size_t estimated_tokens = ContextUtils::EstimateTokenCount(estimated_prompt);
        
        LOG_DEBUG_ContextInfo("Estimated prompt tokens: " + std::to_string(estimated_tokens) + 
                             " (current context: " + std::to_string(static_cast<size_t>(std::max(0, n_past))) + 
                             "/" + std::to_string(stats.max_context_tokens) + ")");
        
        // Check if we're approaching the limit with the new message
        if (stats.max_context_tokens > 0) {
            size_t total_estimated = static_cast<size_t>(std::max(0, n_past)) + estimated_tokens;
            float estimated_usage = static_cast<float>(total_estimated) / stats.max_context_tokens;
            
            if (estimated_usage >= 0.9f) {
                // Emergency situation: context usage > 90% - must summarize before responding
                LOG_ContextInfo("Estimated context usage after rebuild would be " + 
                               std::to_string(estimated_usage * 100.0f) + "% - emergency summarization required");
                
                // Extract content for summarization (recent conversation)
                std::ostringstream content_stream;
                size_t start_idx = message_history.size() > 20 ? message_history.size() - 20 : 0;
                for (size_t i = start_idx; i < message_history.size(); ++i) {
                    content_stream << message_history[i].first << ": " << message_history[i].second << "\n";
                }
                
                RequestSummarization(content_stream.str());
                state = ContextState::AWAITING_SUMMARIZATION;
                if (callbacks.on_complete) {
                    callbacks.on_complete("Context critically full - summarizing before response...", true);
                }
                return true;
            }
            else if (estimated_usage >= 0.8f) {
                // Moderate usage: 80-90% - generate response first, then summarize in background
                LOG_ContextInfo("Estimated context usage after rebuild would be " + 
                               std::to_string(estimated_usage * 100.0f) + "% - will summarize after response");
                
                // Set flag to trigger background summarization after generation completes
                needs_background_summarization = true;
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
        
        // 4. Double-check context size after rebuild (safety check)
        if (IsNearContextLimit(0.9f)) { // Higher threshold for final check
            LOG_ContextInfo("Context still at limit after rebuild, requesting emergency summarization");
            
            // Extract content for summarization (recent conversation)
            std::ostringstream content_stream;
            size_t start_idx = message_history.size() > 10 ? message_history.size() - 10 : 0;
            for (size_t i = start_idx; i < message_history.size(); ++i) {
                content_stream << message_history[i].first << ": " << message_history[i].second << "\n";
            }
            
            RequestSummarization(content_stream.str());
            state = ContextState::AWAITING_SUMMARIZATION;
            if (callbacks.on_complete) {
                callbacks.on_complete("Context requires emergency summarization...", true);
            }
            return true;
        }
        
        // 5. Start async generation (context is already rebuilt, no need to rebuild again)
        state = ContextState::GENERATING;
        current_callbacks = callbacks;
        std::string full_prompt = BuildFullPrompt(); // Get the final prompt for generation
        StartGenerationAsync(full_prompt, callbacks);
        
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR_ContextInfo("Exception in HandleInputAsync: " + std::string(e.what()));
        state = ContextState::ERROR_STATE;
        if (callbacks.on_error) {
            callbacks.on_error("Exception: " + std::string(e.what()));
        }
        return false;
    }
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
        needs_background_summarization = false; // Reset flag when stopping generation
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
            // Note: We cannot hold context_mutex for the entire duration as it would block other operations
            // We need to carefully manage locking for thread safety
            
            {
                std::lock_guard<std::mutex> lock(context_mutex);
                
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
                    std::vector<int32_t> prompt_tokens = TokenizePrompt(prompt);
                    
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
            const int max_new_tokens = 512;
            
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
                
                response_tokens.push_back(next_token);
                
                // Convert this token to text and stream it
                std::vector<int32_t> single_token = {next_token};
                std::string token_text = DetokenizeResponse(single_token);
                
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
                
                if (llama_decode(llama_ctx, batch) != 0) {
                    LOG_ERROR_ContextInfo("Failed to decode generated token");
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
                    UpdateStats();
                }
                
                state = ContextState::READY;
                is_generating = false;
                
                // Check if background summarization is needed
                if (needs_background_summarization.load()) {
                    LOG_ContextInfo("Triggering background summarization after response completion");
                    needs_background_summarization = false;
                    
                    // Extract content for summarization (recent conversation)
                    std::ostringstream content_stream;
                    size_t start_idx = message_history.size() > 20 ? message_history.size() - 20 : 0;
                    for (size_t i = start_idx; i < message_history.size(); ++i) {
                        content_stream << message_history[i].first << ": " << message_history[i].second << "\n";
                    }
                    
                    // Trigger background summarization (non-blocking)
                    RequestSummarization(content_stream.str());
                    // Note: We don't change state to AWAITING_SUMMARIZATION here since the user
                    // can continue chatting while summarization happens in the background
                }
            }
            
            LOG_DEBUG_ContextInfo("Async generation completed: " + std::to_string(response_tokens.size()) + 
                                 " tokens -> " + full_response.substr(0, 100) + 
                                 (full_response.length() > 100 ? "..." : ""));
            
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(context_mutex);
            is_generating = false;
            needs_background_summarization = false; // Reset flag on error
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
    
    // Check if context was recently rebuilt or cleared
    if (context_needs_rebuild) {
        LOG_DEBUG_ContextInfo("Context rebuild flag detected - resetting n_past");
        n_past = 0;
        context_needs_rebuild = false;
        SyncStatsWithContextState(); // Sync stats after reset
        return true;
    }
    
    // Validate that n_past doesn't exceed reasonable bounds
    if (n_past < 0 || n_past > static_cast<int32_t>(current_tokens.size() + 1000)) {
        LOG_WARNING_ContextInfo("n_past value appears invalid (" + std::to_string(n_past) + 
                               "), resetting to 0");
        n_past = 0;
        SyncStatsWithContextState(); // Sync stats after reset
        return false;
    }
    
    // Additional validation: ensure n_past is consistent with context state
    // This helps catch cases where context was modified without proper n_past update
    if (stats.current_context_tokens > 0 && n_past == 0 && !current_tokens.empty()) {
        LOG_DEBUG_ContextInfo("Detected inconsistent state: tokens exist but n_past=0, keeping n_past=0");
        // Keep n_past = 0 for safety
        SyncStatsWithContextState(); // Sync stats to reflect the actual state
    }
    
    return true;
}

inline void ContextInfo::SyncStatsWithContextState() {
    // NOTE: context_mutex should already be held by caller
    // CRITICAL FIX: Synchronize stats with actual llama.cpp context state (n_past)
    // This ensures UI and internal logic always reflect the true context size
    stats.current_context_tokens = static_cast<size_t>(std::max(0, n_past));
    stats.last_activity = std::chrono::steady_clock::now();
    
    LOG_DEBUG_ContextInfo("Stats synchronized with context state: n_past=" + 
                         std::to_string(n_past) + ", current_context_tokens=" + 
                         std::to_string(stats.current_context_tokens));
}

inline void ContextInfo::PruneMessageHistoryWithSummary(const std::string& summary, size_t keep_recent_messages) {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    if (message_history.size() <= keep_recent_messages) {
        LOG_DEBUG_ContextInfo("Message history too short to prune (size: " + 
                             std::to_string(message_history.size()) + ")");
        return;
    }
    
    // Keep only the most recent messages
    std::vector<std::pair<std::string, std::string>> recent_messages;
    size_t start_idx = message_history.size() - keep_recent_messages;
    recent_messages.assign(message_history.begin() + start_idx, message_history.end());
    
    LOG_ContextInfo("Pruning message history: keeping " + std::to_string(keep_recent_messages) + 
                   " recent messages out of " + std::to_string(message_history.size()) + " total");
    
    // Update message history with recent messages only
    message_history = std::move(recent_messages);
    
    // Apply the summary to the template
    ApplySummary(summary);
    
    // Force full context rebuild after pruning
    context_needs_rebuild = true;
    
    LOG_ContextInfo("Message history pruned and summary applied: \"" + 
                   summary.substr(0, 100) + (summary.length() > 100 ? "..." : "") + "\"");
    
    UpdateStats();
}

inline void ContextInfo::SetContextSizeForTesting(size_t test_size) {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    LOG_ContextInfo("Setting context size for testing: " + std::to_string(test_size) + 
                   " (was: " + std::to_string(stats.max_context_tokens) + ")");
    
    stats.max_context_tokens = test_size;
    
    // Check if current usage exceeds new limit
    size_t current_usage = static_cast<size_t>(std::max(0, n_past));
    if (current_usage > test_size) {
        LOG_ContextInfo("Current usage (" + std::to_string(current_usage) + 
                       ") exceeds new test limit (" + std::to_string(test_size) + ")");
    }
    
    // Force a context limit check with the new size
    float usage = test_size > 0 ? static_cast<float>(current_usage) / test_size : 0.0f;
    LOG_ContextInfo("New context usage ratio: " + std::to_string(usage * 100.0f) + "%");
}
