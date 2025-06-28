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

// Pruning buffer for plugin consumption
struct PrunedMessageBatch {
    std::string context_id;
    std::vector<std::pair<std::string, std::string>> pruned_messages;
    std::chrono::steady_clock::time_point pruned_at;
    bool needs_summarization = true;
    
    PrunedMessageBatch(const std::string& id, 
                       std::vector<std::pair<std::string, std::string>> messages)
        : context_id(id), pruned_messages(std::move(messages)), 
          pruned_at(std::chrono::steady_clock::now()) {}
};

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
 * Revolutionary Processing Flow (Plugin-Based Summarization):
 * 1. Receive input from Orchestrator (pre-sanitized if from Discord)
 * 2. Add pure conversation pair to message history (no system contamination)
 * 3. Render dynamic template with current message history via ChatTemplateManager
 * 4. Process rendered template to tokens via TokenCache
 * 5. Check context size - trigger immediate pruning if needed (buffers messages for plugin)
 * 6. Generate LLM response using rendered template
 * 7. Detokenize response via TokenCache
 * 8. Add response pair to message history
 * 9. Return human-readable response
 * 
 * Background Plugin Processing:
 * - SummarizationPlugin polls pruning buffer periodically
 * - Plugin creates summary contexts on-demand via Orchestrator
 * - Heavy summarization work happens async through ProcessingPipeline
 * - Completed summaries applied back via ApplyCompletedSummary()
 * 
 * Template Integration Benefits:
 * - Clean Message History: Only actual conversation, no system pollution
 * - Dynamic Context: Summaries, system messages embedded in template sections
 * - Plugin-Driven Updates: Summarization updates template directly, not message history
 * - Per-Context Flexibility: Each context can have completely different template evolution
 * - Non-Blocking: Critical path stays fast, heavy work happens in background
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
    ContextStats stats;
    mutable std::mutex context_mutex;
    
    // Message history - pure conversation only
    std::vector<std::pair<std::string, std::string>> message_history; // (role, content)
    
    // Summary storage - maintains up to 5 summaries in chronological order (oldest to newest)
    std::vector<std::string> summaries;
    static constexpr size_t MAX_SUMMARIES = 5;
    
    // Static pruning buffer for plugin consumption
    static std::mutex pruning_buffer_mutex;
    static std::vector<PrunedMessageBatch> global_pruning_buffer;
    
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
    
    // Helper methods (moved to public for testing)
    
    // Context size management
    bool IsNearContextLimit(float threshold = 0.8f) const;
    void EmergencyPrune(); // Emergency fallback when plugin system unavailable
    void UpdateStats();
    
    // Summary management methods
    void AddSummaryToList(const std::string& summary);
    void UpdateTemplateWithAllSummaries();
    
public:    // Constructor overloads
    ContextInfo(const std::string& context_id, ModelInfo* model, int32_t context_size);
    ContextInfo(ModelInfo* model, int32_t context_size); // For testing with auto-generated context_id
    
    // Factory method for default template
    static std::unique_ptr<ContextInfo> Create(const std::string& context_id, ModelInfo* model, int32_t context_size);
    static std::unique_ptr<ContextInfo> Create(ModelInfo* model, int32_t context_size); // Auto-generated ID
    
    // Plugin interface for accessing pruning buffer
    static std::vector<PrunedMessageBatch> GetAndClearPruningBuffer();
    static bool HasPendingSummarization();
    
    // Core pruning method (immediate, critical path)
    void PruneContextImmediate(size_t keep_recent_messages = 5);
    
private:
    // Internal pruning method (assumes context_mutex is already held)
    void PruneContextImmediate_Internal(size_t keep_recent_messages = 5);
    
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
    
    // Context validation
    bool ValidateContext() const;
    
    // Advanced features
    void SetMaxContextTokens(size_t max_tokens);
    int32_t GetContextSize() const { return context_size; }
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
    
    // Debug helper to dump all context size values
    void DumpContextSizeValues() const {
        LOG_DEBUG_ContextInfo("=== Context Size Debug Dump ===");
        LOG_DEBUG_ContextInfo("n_past: " + std::to_string(n_past));
        LOG_DEBUG_ContextInfo("stats.current_context_tokens: " + std::to_string(stats.current_context_tokens));
        LOG_DEBUG_ContextInfo("stats.max_context_tokens: " + std::to_string(stats.max_context_tokens));
        LOG_DEBUG_ContextInfo("stats.total_tokens_processed: " + std::to_string(stats.total_tokens_processed));
        LOG_DEBUG_ContextInfo("current_tokens.size(): " + std::to_string(current_tokens.size()));
        LOG_DEBUG_ContextInfo("GetActualContextTokens(): " + std::to_string(GetActualContextTokens()));
        LOG_DEBUG_ContextInfo("GetActualContextUsageRatio(): " + std::to_string(GetActualContextUsageRatio()));
        LOG_DEBUG_ContextInfo("stats.GetContextUsageRatio(): " + std::to_string(stats.GetContextUsageRatio()));
        LOG_DEBUG_ContextInfo("=== End Context Size Debug ===");
    }
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
    
    // Set max context tokens in stats to match our individual context size
    stats.max_context_tokens = static_cast<size_t>(context_size);
    
    LOG_DEBUG_ContextInfo("ContextInfo created: " + context_id + " with context size: " + std::to_string(context_size));
    UpdateStats();
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
            
            // Debug: Dump context size values after rebuild
            DumpContextSizeValues();
        }
        
        // 3. Render dynamic template with current message history
        std::string full_prompt = BuildFullPrompt();
        
        // 4. Check context size - trigger summarization if needed
        std::vector<int32_t> prompt_tokens = TokenizePrompt(full_prompt);
        if (IsNearContextLimit(0.8f)) {
            LOG_ContextInfo("Context approaching limit, performing immediate pruning");
            
            // Prune messages immediately to pruning buffer - plugin will handle summarization
            PruneContextImmediate_Internal(5); // Keep only last 5 messages
            
            // Mark flag for background summarization
            needs_background_summarization = true;
            
            // Continue processing after pruning
            LOG_ContextInfo("Immediate pruning complete, continuing with generation");
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
    
    // CRITICAL FIX: Validate context size and bounds before any processing
    if (stats.max_context_tokens == 0) {
        LOG_ERROR_ContextInfo("Context size not set - cannot validate bounds");
        return false;
    }
    
    // CRITICAL FIX: Strict bounds validation - context must have space for tokens plus safety buffer
    const int32_t context_size = static_cast<int32_t>(stats.max_context_tokens);
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
            
            // Track tokens actually processed in this batch
            stats.total_tokens_processed += batch.n_tokens;
        }
        
        LOG_DEBUG_ContextInfo("Processed " + std::to_string(tokens.size()) + " tokens, n_past=" + std::to_string(n_past));
        
        // CRITICAL FIX: Synchronize stats with actual context state after batch processing
        SyncStatsWithContextState();
        
        // Debug: Dump context size values after processing
        DumpContextSizeValues();
        
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
        
        // Calculate max tokens based on available context space
        // Reserve some space for context management and leave room for user's next message
        const int context_reserve = 256; // Reserve space for context management
        const int available_space = static_cast<int>(stats.max_context_tokens) - n_past - context_reserve;
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
            if (current_usage >= static_cast<int32_t>(stats.max_context_tokens) - generation_buffer) {
                LOG_DEBUG_ContextInfo("Approaching context limit during generation (usage: " + 
                                     std::to_string(current_usage) + "/" + 
                                     std::to_string(stats.max_context_tokens) + 
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
            if (n_past >= static_cast<int32_t>(stats.max_context_tokens)) {
                LOG_ERROR_ContextInfo("Cannot decode token - position (" + std::to_string(n_past) + 
                                     ") exceeds context size (" + std::to_string(stats.max_context_tokens) + ")");
                break;
            }
            
            if (llama_decode(llama_ctx, batch) != 0) {
                LOG_ERROR_ContextInfo("Failed to decode generated token at position " + 
                                     std::to_string(n_past) + " (context size: " + 
                                     std::to_string(stats.max_context_tokens) + ")");
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
    
    // CRITICAL FIX: Also check if we're within a dangerous margin of the limit
    // Use consistent safety buffer
    const size_t safety_buffer = ContextConstants::SAFETY_BUFFER_TOKENS;
    if (actual_context_tokens + safety_buffer >= stats.max_context_tokens) {
        LOG_ContextInfo("Context within safety buffer of limit - treating as near limit");
        near_limit = true;
    }
    
    if (near_limit) {
        LOG_ContextInfo("Context is near limit: " + std::to_string(usage * 100.0f) + "% >= " + 
                       std::to_string(threshold * 100.0f) + "% (or within safety buffer)");
    }
    
    return near_limit;
}

// Emergency fallback pruning for extreme situations (no plugin dependency)
inline void ContextInfo::EmergencyPrune() {
    LOG_ERROR_ContextInfo("EMERGENCY: Plugin system unavailable - performing emergency pruning for context: " + context_id);
    
    if (IsNearContextLimit(0.95f)) {
        LOG_ContextInfo("Emergency pruning without summarization - context critically full");
        PruneContextImmediate(3); // Very aggressive - keep only 3 most recent messages
        LOG_ContextInfo("Emergency pruning completed");
    }
}

inline void ContextInfo::UpdateStats() {
    stats.message_pairs = message_history.size();
    // CRITICAL FIX: Use n_past as the single source of truth for context size
    // This ensures UI and internal logic always reflect the actual llama.cpp context state
    stats.current_context_tokens = static_cast<size_t>(std::max(0, n_past));
    // NOTE: total_tokens_processed should NOT be updated here - it tracks cumulative processing
    // and should only be incremented when actually processing new tokens, not on every stats update
    stats.last_activity = std::chrono::steady_clock::now();
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
                    // Emergency situation: context usage > 90% - must prune immediately
                    LOG_ContextInfo("Estimated context usage after rebuild would be " + 
                                   std::to_string(estimated_usage * 100.0f) + "% - emergency pruning required");
                    
                    // Immediate pruning to keep conversation flowing
                    PruneContextImmediate_Internal(5); // Keep only last 5 messages
                    
                    LOG_ContextInfo("Emergency pruning completed, continuing with generation");
                }
                else if (estimated_usage >= 0.8f) {
                    // Moderate usage: 80-90% - prune less aggressively
                    LOG_ContextInfo("Estimated context usage after rebuild would be " + 
                                   std::to_string(estimated_usage * 100.0f) + "% - performing moderate pruning");
                    
                    // Moderate pruning - keep more messages for better context
                    PruneContextImmediate_Internal(8); // Keep more messages for better conversation flow
                    
                    LOG_ContextInfo("Moderate pruning completed");
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
                LOG_ContextInfo("Context still at limit after rebuild, performing final emergency pruning");
                
                // Final emergency pruning if context is still too full
                PruneContextImmediate_Internal(3); // Very aggressive - keep only 3 most recent messages
                
                LOG_ContextInfo("Final emergency pruning completed");
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
        bool trigger_background_pruning = false;
        
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
            
            // Calculate max tokens based on available context space
            // Reserve some space for context management and leave room for user's next message
            const int context_reserve = 256; // Reserve space for context management  
            const int available_space = static_cast<int>(stats.max_context_tokens) - n_past - context_reserve;
            const int max_new_tokens = std::max(512, std::min(4096, available_space)); // At least 512, up to 4096 tokens
            
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
                if (current_usage >= static_cast<int32_t>(stats.max_context_tokens) - generation_buffer) {
                    LOG_DEBUG_ContextInfo("Approaching context limit during generation (usage: " + 
                                         std::to_string(current_usage) + "/" + 
                                         std::to_string(stats.max_context_tokens) + 
                                         "), stopping early with buffer: " + std::to_string(generation_buffer));
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
                
                // CRITICAL FIX: Validate position before decode to prevent overflow
                if (n_past >= static_cast<int32_t>(stats.max_context_tokens)) {
                    LOG_ERROR_ContextInfo("Cannot decode token - position (" + std::to_string(n_past) + 
                                         ") exceeds context size (" + std::to_string(stats.max_context_tokens) + ")");
                    break;
                }
                
                if (llama_decode(llama_ctx, batch) != 0) {
                    LOG_ERROR_ContextInfo("Failed to decode generated token at position " + 
                                         std::to_string(n_past) + " (context size: " + 
                                         std::to_string(stats.max_context_tokens) + ")");
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
            trigger_background_pruning = needs_background_summarization.load();
            if (trigger_background_pruning) {
                LOG_ContextInfo("Triggering background summarization after response completion");
                needs_background_summarization = false;
            }
        }
        
        // Perform background pruning OUTSIDE the context lock to avoid deadlock
        if (trigger_background_pruning) {
            PruneContextImmediate(8); // This will acquire its own lock safely
            LOG_ContextInfo("Background pruning complete - messages buffered for plugin summarization");
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
    
    // CRITICAL FIX: Check context size bounds first
    if (stats.max_context_tokens == 0) {
        LOG_ERROR_ContextInfo("Context size not initialized - cannot validate state");
        return false;
    }
    
    const int32_t max_context = static_cast<int32_t>(stats.max_context_tokens);
    
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
        LOG_DEBUG_ContextInfo("No pruning needed: only " + std::to_string(message_history.size()) + " messages");
        return;
    }
    
    // Extract messages to be pruned
    size_t prune_count = message_history.size() - keep_recent_messages;
    std::vector<std::pair<std::string, std::string>> pruned_messages;
    pruned_messages.assign(message_history.begin(), message_history.begin() + prune_count);
    
    // Add to global pruning buffer for plugin processing
    {
        std::lock_guard<std::mutex> buffer_lock(pruning_buffer_mutex);
        global_pruning_buffer.emplace_back(context_id, std::move(pruned_messages));
    }
    
    // Immediately prune from active history
    message_history.erase(message_history.begin(), message_history.begin() + prune_count);
    context_needs_rebuild = true;
    
    LOG_ContextInfo("Pruned " + std::to_string(prune_count) + " messages to buffer for summarization. " +
                   "Keeping " + std::to_string(keep_recent_messages) + " recent messages.");
    UpdateStats();
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

// Context threshold checking with simplified logic

// Static method implementations for pruning buffer plugin interface
inline std::vector<PrunedMessageBatch> ContextInfo::GetAndClearPruningBuffer() {
    std::lock_guard<std::mutex> lock(pruning_buffer_mutex);
    
    std::vector<PrunedMessageBatch> result;
    result.swap(global_pruning_buffer);
    
    return result;
}

inline bool ContextInfo::HasPendingSummarization() {
    std::lock_guard<std::mutex> lock(pruning_buffer_mutex);
    
    return !global_pruning_buffer.empty();
}

inline void ContextInfo::PruneContextImmediate(size_t keep_recent_messages) {
    std::lock_guard<std::mutex> lock(context_mutex);
    PruneContextImmediate_Internal(keep_recent_messages);
}

inline void ContextInfo::PruneContextImmediate_Internal(size_t keep_recent_messages) {
    // Note: context_mutex should already be held by caller
    
    if (message_history.size() <= keep_recent_messages) {
        LOG_DEBUG_ContextInfo("No pruning needed: only " + std::to_string(message_history.size()) + " messages");
        return;
    }
    
    // Extract messages to be pruned
    size_t prune_count = message_history.size() - keep_recent_messages;
    std::vector<std::pair<std::string, std::string>> pruned_messages;
    pruned_messages.assign(message_history.begin(), message_history.begin() + prune_count);
    
    // Add to global pruning buffer for plugin processing
    {
        std::lock_guard<std::mutex> buffer_lock(pruning_buffer_mutex);
        global_pruning_buffer.emplace_back(context_id, std::move(pruned_messages));
    }
    
    // Immediately prune from active history
    message_history.erase(message_history.begin(), message_history.begin() + prune_count);
    context_needs_rebuild = true;
    
    LOG_ContextInfo("Pruned " + std::to_string(prune_count) + " messages to buffer for summarization");
    UpdateStats();
}

// Static member definitions for pruning buffer
inline std::mutex ContextInfo::pruning_buffer_mutex;
inline std::vector<PrunedMessageBatch> ContextInfo::global_pruning_buffer;

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
