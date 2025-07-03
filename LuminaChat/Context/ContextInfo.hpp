#pragma once

#include "ContextState.hpp"
#include "ContextIO.hpp"
#include "../ChatTemplateManager.hpp"
#include "../ModelInfo.hpp"
#include "../TokenCache.hpp"
#include "../Logger.hpp"
#include "../Utilities.hpp"
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

// Import ContextState enum into current namespace for easier use
using LuminaChat::ContextState;

// ContextInfo Configuration Constants
namespace ContextConstants {
    constinit const int32_t SAFETY_BUFFER_TOKENS = 64;      // Safety margin for all context operations
    constinit const int32_t GENERATION_BUFFER_TOKENS = 128; // Buffer for response generation (larger to account for responses)
    
    // Additional compile-time constants for performance optimization
    [[nodiscard]] constexpr size_t DefaultMessageHistoryCapacity() noexcept { return 32; }
    [[nodiscard]] constexpr size_t DefaultContextThreads() noexcept { return 1; }
    [[nodiscard]] constexpr int32_t DefaultBatchSize() noexcept { return 512; }
    [[nodiscard]] constexpr double CharToTokenEstimateRatio() noexcept { return 4.0; }
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

// Compile-time helper functions for optimization
namespace ContextHelpers {
    [[nodiscard]] constexpr size_t CalculateReserveSize(size_t base_size, size_t growth_factor = 2) noexcept {
        return base_size * growth_factor;
    }
    
    [[nodiscard]] constexpr bool IsValidContextSize(int32_t size) noexcept {
        return size > 0 && size <= (1 << 20); // Max 1M tokens
    }
    
    [[nodiscard]] constexpr size_t AlignToCache(size_t size) noexcept {
        return (size + 63) & ~63; // Align to 64-byte cache line
    }
}

// Forward declaration for ModelInfo
class ModelInfo;

enum class RebuildStrategy {
    FULL,           // Complete rebuild including template re-rendering
    PARTIAL,        // Efficient append-only rebuild for new messages
    TEMPLATE_ONLY   // Only re-render template, keep existing tokens
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
class ContextInfo : public LuminaChat::ContextStateManager {
private:    // Core components - optimized memory layout for cache efficiency
    // Hot path data - frequently accessed together (first cache line)
    alignas(64) ModelInfo* parent_model;
    TokenCache* token_cache;
    llama_context* llama_ctx = nullptr;
    int32_t n_past = 0;  // Number of tokens processed in context
    int32_t context_size;  // Individual context size (not tied to model)
    size_t max_context_tokens = 0;  // Maximum context size for this instance
    
    // State flags - cache-aligned atomic for lock-free access
    alignas(16) mutable std::atomic<bool> context_needs_rebuild{true};
    
    // Lock-free counters for performance monitoring (cache-aligned)
    alignas(16) mutable std::atomic<size_t> message_count{0};
    mutable std::atomic<size_t> generation_count{0};
    mutable std::atomic<size_t> rebuild_count{0};
    
    // Lock-free flags for hot path optimization
    mutable std::atomic<bool> template_dirty{false};
    mutable std::atomic<bool> io_manager_available{false};
    
    // Managers - second cache line
    alignas(64) std::unique_ptr<ChatTemplateManager> template_manager;
    std::unique_ptr<ContextInputOutput> io_manager;
    
    // Context identification - rarely accessed, separate cache line
    alignas(64) std::string context_id;
    mutable std::mutex context_mutex;
    
    // Message history - pure conversation only (optimized with reserved capacity)
    std::vector<std::pair<std::string, std::string>> message_history; // (role, content)
    
    // Summary storage - maintains up to 5 summaries in chronological order (oldest to newest)
    std::vector<std::string> summaries;
    static constexpr size_t MAX_SUMMARIES = 5;
    
    // Compile-time helper for summary management
    [[nodiscard]] static constexpr size_t GetMaxSummaries() noexcept { return MAX_SUMMARIES; }
    
    // Context size management
    void AddSummaryToList(const std::string& summary) noexcept;
    void AddSummaryToList(std::string&& summary) noexcept;
    void UpdateTemplateWithAllSummaries() noexcept;
    
public:    // Constructor overloads
    ContextInfo(const std::string& context_id, ModelInfo* model, int32_t context_size);
    ContextInfo(ModelInfo* model, int32_t context_size); // For testing with auto-generated context_id
    
    // Factory method for default template
    static std::unique_ptr<ContextInfo> Create(const std::string& context_id, ModelInfo* model, int32_t context_size);
    static std::unique_ptr<ContextInfo> Create(ModelInfo* model, int32_t context_size); // Auto-generated ID
    
    // Enhanced pruning method that returns pruned messages for summarization coordination
    [[nodiscard]] std::vector<std::pair<std::string, std::string>> PruneContextImmediateWithExtraction(size_t keep_recent_messages = 5) noexcept;
    
    // Plugin callback to apply completed summaries
    void ApplyCompletedSummary(const std::string& summary) noexcept;
    void ApplyCompletedSummary(std::string&& summary) noexcept;
    
    // Destructor
    ~ContextInfo();
    
    // Core processing interface
    std::string HandleInput(const std::string& input, const std::string& username = "user"); // DEPRECATED for UI: Use HandleInputAsync instead
    bool HandleInputAsync(const std::string& input, const GenerationCallbacks& callbacks, const std::string& username = "user");
    void AddHistoricalMessage(const std::string& role, const std::string& content);
    
    // Llama.cpp integration methods
    bool InitializeLlamaContext();
    
    // Direct access to IO manager for generation operations
    [[nodiscard]] [[msvc::forceinline]] ContextInputOutput* GetIOManager() noexcept { return io_manager.get(); }
    [[nodiscard]] [[msvc::forceinline]] const ContextInputOutput* GetIOManager() const noexcept { return io_manager.get(); }
    
    // Minimal accessors needed by plugins and UI (delegates to specialized classes)
    [[nodiscard]] [[msvc::forceinline]] bool IsGenerating() const noexcept { 
        return io_manager_available.load(std::memory_order_relaxed) && io_manager && io_manager->IsGenerating(); 
    }
    [[msvc::forceinline]] void StopGeneration() noexcept { 
        if (io_manager_available.load(std::memory_order_relaxed) && io_manager) [[likely]] 
            io_manager->StopGeneration(); 
    }
    
    // Context usage monitoring (delegates to BatchManager via IOManager)
    [[nodiscard]] [[msvc::forceinline]] size_t GetActualContextTokens() const noexcept { 
        return io_manager_available.load(std::memory_order_relaxed) && io_manager ? 
               static_cast<size_t>(io_manager->GetCurrentPosition()) : 0; 
    }
    
    // Template section management - direct access to ChatTemplateManager
    void UpdateEnvironment(const std::string& env) noexcept;
    void UpdateEnvironment(std::string&& env) noexcept;
    void UpdateIdentity(const std::string& identity) noexcept;
    void UpdateIdentity(std::string&& identity) noexcept;
    void UpdateSystemPrompt(const std::string& system_msg) noexcept;
    void UpdateSystemPrompt(std::string&& system_msg) noexcept;
    void ApplySummary(const std::string& summary) noexcept;  // Updates template's summary section
    void ApplySummary(std::string&& summary) noexcept;
    void UpdateOldChatSummary(const std::string& old_summary) noexcept; // For context pruning
    void UpdateOldChatSummary(std::string&& old_summary) noexcept;
    void UpdateMotifContext(const std::string& motif) noexcept;
    void UpdateMotifContext(std::string&& motif) noexcept;
    void UpdateInternalReflection(const std::string& reflection) noexcept;
    void UpdateInternalReflection(std::string&& reflection) noexcept;
    void UpdateEmotionalState(const std::string& emotional_state) noexcept;
    void UpdateEmotionalState(std::string&& emotional_state) noexcept;
    void AddPastSessionMemory(const std::string& memory) noexcept;
    void AddPastSessionMemory(std::string&& memory) noexcept;
    
    // Context management
    bool RebuildContext(RebuildStrategy strategy = RebuildStrategy::FULL);
    void ClearContext() noexcept;
    void ClearMessageHistory() noexcept;
    
    // Override base class methods to include generation state
    [[nodiscard]] [[msvc::forceinline]] bool IsAvailableForGeneration() const noexcept override {
        return ContextStateManager::IsAvailableForGeneration() && 
               (!io_manager_available.load(std::memory_order_relaxed) || !io_manager || !io_manager->IsGenerating());
    }
    [[nodiscard]] [[msvc::forceinline]] bool IsAvailableForPluginProcessing() const noexcept override {
        return ContextStateManager::IsAvailableForPluginProcessing() && 
               (!io_manager_available.load(std::memory_order_relaxed) || !io_manager || !io_manager->IsGenerating());
    }
    
    [[nodiscard]] [[msvc::forceinline]] size_t GetMaxContextTokens() const noexcept { return max_context_tokens; }
    [[nodiscard]] [[msvc::forceinline]] const std::string& GetContextId() const noexcept { return context_id; }
    
    // Message history access - ultra-fast inline accessors
    [[nodiscard]] [[msvc::forceinline]] const std::vector<std::pair<std::string, std::string>>& GetMessageHistory() const noexcept { return message_history; }
    [[nodiscard]] [[msvc::forceinline]] size_t GetMessageCount() const noexcept { return message_count.load(std::memory_order_relaxed); }
    
    // Lock-free performance counters for ultimate performance monitoring
    [[nodiscard]] [[msvc::forceinline]] size_t GetGenerationCount() const noexcept { return generation_count.load(std::memory_order_relaxed); }
    [[nodiscard]] [[msvc::forceinline]] size_t GetRebuildCount() const noexcept { return rebuild_count.load(std::memory_order_relaxed); }
    [[nodiscard]] [[msvc::forceinline]] bool IsTemplateDirty() const noexcept { return template_dirty.load(std::memory_order_relaxed); }
    [[nodiscard]] [[msvc::forceinline]] bool IsIOManagerAvailable() const noexcept { return io_manager_available.load(std::memory_order_relaxed); }
    
    // Compile-time helper accessors
    [[nodiscard]] [[msvc::forceinline]] static constexpr size_t GetMaxSummariesCount() noexcept { return MAX_SUMMARIES; }
    [[nodiscard]] [[msvc::forceinline]] bool IsMessageHistoryEmpty() const noexcept { return message_count.load(std::memory_order_relaxed) == 0; }
    [[nodiscard]] [[msvc::forceinline]] bool IsSummaryListFull() const noexcept { return summaries.size() >= MAX_SUMMARIES; }
    [[nodiscard]] [[msvc::forceinline]] bool NeedsRebuild() const noexcept { return context_needs_rebuild.load(std::memory_order_relaxed); }
    
    // Template access - force inlined for performance
    [[nodiscard]] [[msvc::forceinline]] ChatTemplateManager& GetTemplateManager() noexcept { return *template_manager; }
    [[nodiscard]] [[msvc::forceinline]] const ChatTemplateManager& GetTemplateManager() const noexcept { return *template_manager; }
    
    // Advanced features - optimized setters
    void SetMaxContextTokens(size_t max_tokens) noexcept;
    [[nodiscard]] [[msvc::forceinline]] int32_t GetContextSize() const noexcept { return context_size; }
    [[nodiscard]] std::string GetCurrentPrompt() const;
    
    // Template rendering (made public for testing)
    [[msvc::forceinline]] std::string BuildFullPrompt() noexcept;
    
    // Context rebuilding helpers (made public for testing)
    void RebuildContext_Full();
    void RebuildContext_Partial();
    void RebuildContext_TemplateOnly();
};

// Inline implementation of ContextInfo methods
inline ContextInfo::ContextInfo(const std::string& context_id, ModelInfo* model, int32_t context_size)
    : ContextStateManager()  // Initialize base class first
    , parent_model(model)
    , token_cache(model ? &model->GetTokenCache() : nullptr)
    , llama_ctx(nullptr)
    , n_past(0)
    , context_size(context_size)
    , max_context_tokens(static_cast<size_t>(context_size))
    , context_needs_rebuild(true)
    , message_count(0)
    , generation_count(0)
    , rebuild_count(0)
    , template_dirty(false)
    , io_manager_available(false)
    , template_manager(std::make_unique<ChatTemplateManager>())
    , io_manager(nullptr)  // Will be initialized later
    , context_id(context_id)
{
    if (!parent_model) [[unlikely]] {
        LOG_ERROR("ContextInfo", "ContextInfo created with null ModelInfo");
        (void)TrySetState(ContextState::ERROR_STATE);
        return;
    }
    
    if (!token_cache) [[unlikely]] {
        LOG_ERROR("ContextInfo", "ContextInfo created with null TokenCache");
        (void)TrySetState(ContextState::ERROR_STATE);
        return;
    }
    
    if (context_size <= 0) [[unlikely]] {
        LOG_ERROR("ContextInfo", "ContextInfo created with invalid context size: " + std::to_string(context_size));
        (void)TrySetState(ContextState::ERROR_STATE);
        return;
    }
    
    // Pre-reserve capacity for message history to reduce allocations
    message_history.reserve(ContextConstants::DefaultMessageHistoryCapacity());  // Reserve space for typical conversation length
    summaries.reserve(MAX_SUMMARIES);  // Reserve space for all summaries
    
    LOG_DEBUG("ContextInfo", "ContextInfo created: " + context_id + " with context size: " + std::to_string(context_size));
}

inline ContextInfo::ContextInfo(ModelInfo* model, int32_t context_size)
    : ContextInfo(GenerateContextId(), model, context_size) // Auto-generate context_id
{
    LOG_DEBUG("ContextInfo", "ContextInfo created with auto-generated ID: " + context_id + " and context size: " + std::to_string(context_size));
}

inline ContextInfo::~ContextInfo() {
    // Stop any ongoing generation first
    if (io_manager) {
        io_manager->StopGeneration();
    }
    
    // Clear atomic flags
    io_manager_available.store(false, std::memory_order_relaxed);
    template_dirty.store(false, std::memory_order_relaxed);
    
    std::lock_guard<std::mutex> lock(context_mutex);
    
    LOG_DEBUG_ContextInfo("ContextInfo destructor called for: " + context_id);
    
    // Clean up IO manager (handles generation threads and batch cleanup automatically)
    io_manager.reset();
    
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
    
    if (!parent_model || !parent_model->IsLoaded()) [[unlikely]] {
        LOG_ERROR_ContextInfo("Parent model not loaded for context initialization");
        return false;
    }
    
    if (llama_ctx) [[likely]] {
        LOG_DEBUG_ContextInfo("Llama context already initialized");
        return true;
    }
    
    try {
        // Set up context parameters using model's settings
        llama_context_params ctx_params = llama_context_default_params();
        // Initialize context parameters with this context's specific size
        ctx_params.n_ctx = context_size;
        ctx_params.n_batch = std::min(ContextConstants::DefaultBatchSize(), context_size / 8);
        ctx_params.n_threads = parent_model->GetConfig().threads > 0 ? 
                               static_cast<unsigned int>(parent_model->GetConfig().threads) : 
                               std::max(static_cast<unsigned int>(ContextConstants::DefaultContextThreads()), 
                                       std::thread::hardware_concurrency());
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
        if (!llama_ctx) [[unlikely]] {
            LOG_ERROR_ContextInfo("Failed to create llama context");
            return false;
        }
        
        // Verify that the actual allocated context size matches what we requested
        if (ctx_params.n_ctx != context_size) [[unlikely]] {
            LOG_WARNING_ContextInfo("Allocated context size (" + std::to_string(ctx_params.n_ctx) + 
                                   ") differs from requested size (" + std::to_string(context_size) + ")");
        }
        
        // Keep max_context_tokens consistent with the original context_size for accounting
        // This ensures UI and calculations remain consistent with the requested size
        max_context_tokens = static_cast<size_t>(context_size);
        
        // Create IO manager for generation operations
        io_manager = std::make_unique<ContextInputOutput>(parent_model, token_cache, llama_ctx, n_past, max_context_tokens);
        if (!io_manager) [[unlikely]] {
            LOG_ERROR_ContextInfo("Failed to create ContextInputOutput manager");
            llama_free(llama_ctx);
            llama_ctx = nullptr;
            io_manager_available.store(false, std::memory_order_relaxed);
            return false;
        }
        
        io_manager_available.store(true, std::memory_order_relaxed);
        
        LOG_ContextInfo("Llama context initialized successfully for: " + context_id);
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR_ContextInfo("Exception during llama context initialization: " + std::string(e.what()));
        if (llama_ctx) [[unlikely]] {
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
    
    if (state.load() == ContextState::ERROR_STATE) {
        LOG_ERROR_ContextInfo("Cannot handle input - context in error state");
        return "Error: Context unavailable";
    }
    // Try to transition to generating state
    if (!TrySetState(ContextState::CONTEXT_GENERATING)) {
        LOG_WARNING_ContextInfo("Cannot handle input - context not available");
        return "Error: Context busy";
    }
    
    try {
        // CRITICAL: This MUST be sent as username, input NOT "user", input; Chat template handles roles dynamically and will allow the AI to interpret properly here.
        // 1. Add pure conversation pair to message history
        message_history.emplace_back(username, input);
        message_count.fetch_add(1, std::memory_order_relaxed);
        LOG_DEBUG_ContextInfo("Added user message to history: " + username + " -> " + 
                             input.substr(0, 50) + (input.length() > 50 ? "..." : ""));
        
        // 2. Check if context needs rebuilding after adding message
        if (context_needs_rebuild || template_dirty.load(std::memory_order_relaxed)) {
            if (!RebuildContext(RebuildStrategy::FULL)) {
                (void)TrySetState(ContextState::ERROR_STATE);
                return "Error: Failed to rebuild context";
            }
        }
        
        // 3. Render dynamic template with current message history
        std::string full_prompt = BuildFullPrompt();
        
        // 4. Context size checking is handled by ContextInputOutput and plugins
        // No need for inline threshold checking here - maintain clean architecture

        
        // 5. Generate LLM response using ContextInputOutput
        std::string response;
        if (io_manager) {
            response = io_manager->GenerateResponse(full_prompt);
        } else {
            LOG_ERROR_ContextInfo("IO manager not available for generation");
            (void)TrySetState(ContextState::ERROR_STATE);
            return "Error: Generation system not available";
        }
        
        // 6. Add assistant response to message history
        message_history.emplace_back("assistant", response);
        message_count.fetch_add(1, std::memory_order_relaxed);
        
        (void)TrySetState(ContextState::CONTEXT_IDLE);
          LOG_DEBUG("ContextInfo", "Generated response: " + response.substr(0, 100) + 
                 (response.length() > 100 ? "..." : ""));
        
        return response;
        
    } catch (const std::exception& e) {
        LOG_ERROR("ContextInfo", "Exception in HandleInput: " + std::string(e.what()));
        (void)TrySetState(ContextState::ERROR_STATE);
        return "Error: " + std::string(e.what());
    }
}

[[msvc::forceinline]] inline void ContextInfo::AddHistoricalMessage(const std::string& role, const std::string& content) {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    message_history.emplace_back(role, content);
    message_count.fetch_add(1, std::memory_order_relaxed);
    context_needs_rebuild.store(true, std::memory_order_relaxed);
    
    LOG_DEBUG_ContextInfo("Added historical message: " + role + " -> " + 
                         content.substr(0, 50) + (content.length() > 50 ? "..." : ""));
}

// Template section management methods - optimized for minimal lock contention
[[msvc::forceinline]] inline void ContextInfo::UpdateEnvironment(const std::string& env) noexcept {
    template_manager->UpdateEnvironment(env);
    context_needs_rebuild.store(true, std::memory_order_relaxed);
    template_dirty.store(true, std::memory_order_relaxed);
    LOG_DEBUG_ContextInfo("Updated environment section");
}

[[msvc::forceinline]] inline void ContextInfo::UpdateEnvironment(std::string&& env) noexcept {
    template_manager->UpdateEnvironment(std::move(env));
    context_needs_rebuild.store(true, std::memory_order_relaxed);
    template_dirty.store(true, std::memory_order_relaxed);
    LOG_DEBUG_ContextInfo("Updated environment section (moved)");
}

[[msvc::forceinline]] inline void ContextInfo::UpdateIdentity(const std::string& identity) noexcept {
    template_manager->UpdateIdentity(identity);
    context_needs_rebuild.store(true, std::memory_order_relaxed);
    template_dirty.store(true, std::memory_order_relaxed);
    LOG_DEBUG_ContextInfo("Updated identity section");
}

[[msvc::forceinline]] inline void ContextInfo::UpdateIdentity(std::string&& identity) noexcept {
    template_manager->UpdateIdentity(std::move(identity));
    context_needs_rebuild.store(true, std::memory_order_relaxed);
    template_dirty.store(true, std::memory_order_relaxed);
    LOG_DEBUG_ContextInfo("Updated identity section (moved)");
}

[[msvc::forceinline]] inline void ContextInfo::UpdateSystemPrompt(const std::string& system_msg) noexcept {
    template_manager->UpdateSystemPrompt(system_msg);
    context_needs_rebuild.store(true, std::memory_order_relaxed);
    template_dirty.store(true, std::memory_order_relaxed);
    LOG_DEBUG_ContextInfo("Updated system prompt section");
}

[[msvc::forceinline]] inline void ContextInfo::UpdateSystemPrompt(std::string&& system_msg) noexcept {
    template_manager->UpdateSystemPrompt(std::move(system_msg));
    context_needs_rebuild.store(true, std::memory_order_relaxed);
    template_dirty.store(true, std::memory_order_relaxed);
    LOG_DEBUG_ContextInfo("Updated system prompt section (moved)");
}

[[msvc::forceinline]] inline void ContextInfo::ApplySummary(const std::string& summary) noexcept {
    template_manager->UpdateSummary(summary);
    context_needs_rebuild.store(true, std::memory_order_relaxed);
    template_dirty.store(true, std::memory_order_relaxed);
    LOG_ContextInfo("Applied summary to template: " + summary.substr(0, 100) + 
                   (summary.length() > 100 ? "..." : ""));
    
    // Automatically prune message history when a summary is applied
    // This prevents the context from growing indefinitely
    if (message_history.size() > 10) [[likely]] { // Only prune if we have substantial history
        LOG_ContextInfo("Auto-pruning message history due to summary application");
        
        // Keep only the most recent 5 message pairs (10 messages total)
        constexpr size_t keep_messages = 10;
        const size_t start_idx = message_history.size() - keep_messages;
        std::vector<std::pair<std::string, std::string>> recent_messages;
        recent_messages.assign(message_history.begin() + start_idx, message_history.end());
        
        LOG_ContextInfo("Pruning message history: keeping " + std::to_string(keep_messages) + 
                       " recent messages out of " + std::to_string(message_history.size()) + " total");
        
        message_history = std::move(recent_messages);
        message_count.store(keep_messages, std::memory_order_relaxed);
    }
}

[[msvc::forceinline]] inline void ContextInfo::ApplySummary(std::string&& summary) noexcept {
    const std::string summary_ref = summary; // Create reference for logging before move
    template_manager->UpdateSummary(std::move(summary));
    context_needs_rebuild.store(true, std::memory_order_relaxed);
    template_dirty.store(true, std::memory_order_relaxed);
    LOG_ContextInfo("Applied summary to template (moved): " + summary_ref.substr(0, 100) + 
                   (summary_ref.length() > 100 ? "..." : ""));
    
    // Automatically prune message history when a summary is applied
    // This prevents the context from growing indefinitely
    if (message_history.size() > 10) [[likely]] { // Only prune if we have substantial history
        LOG_ContextInfo("Auto-pruning message history due to summary application");
        
        // Keep only the most recent 5 message pairs (10 messages total)
        constexpr size_t keep_messages = 10;
        const size_t start_idx = message_history.size() - keep_messages;
        std::vector<std::pair<std::string, std::string>> recent_messages;
        recent_messages.assign(message_history.begin() + start_idx, message_history.end());
        
        LOG_ContextInfo("Pruning message history: keeping " + std::to_string(keep_messages) + 
                       " recent messages out of " + std::to_string(message_history.size()) + " total");
        
        message_history = std::move(recent_messages);
        message_count.store(keep_messages, std::memory_order_relaxed);
    }
}

[[msvc::forceinline]] inline void ContextInfo::UpdateOldChatSummary(const std::string& old_summary) noexcept {
    template_manager->UpdateOldChatSummary(old_summary);
    context_needs_rebuild.store(true, std::memory_order_relaxed);
    template_dirty.store(true, std::memory_order_relaxed);
    LOG_DEBUG_ContextInfo("Updated old chat summary");
}

[[msvc::forceinline]] inline void ContextInfo::UpdateOldChatSummary(std::string&& old_summary) noexcept {
    template_manager->UpdateOldChatSummary(std::move(old_summary));
    context_needs_rebuild.store(true, std::memory_order_relaxed);
    template_dirty.store(true, std::memory_order_relaxed);
    LOG_DEBUG_ContextInfo("Updated old chat summary (moved)");
}

[[msvc::forceinline]] inline void ContextInfo::UpdateMotifContext(const std::string& motif) noexcept {
    template_manager->UpdateMotifContext(motif);
    context_needs_rebuild.store(true, std::memory_order_relaxed);
    template_dirty.store(true, std::memory_order_relaxed);
    LOG_DEBUG_ContextInfo("Updated motif context");
}

[[msvc::forceinline]] inline void ContextInfo::UpdateMotifContext(std::string&& motif) noexcept {
    template_manager->UpdateMotifContext(std::move(motif));
    context_needs_rebuild.store(true, std::memory_order_relaxed);
    template_dirty.store(true, std::memory_order_relaxed);
    LOG_DEBUG_ContextInfo("Updated motif context (moved)");
}

[[msvc::forceinline]] inline void ContextInfo::UpdateInternalReflection(const std::string& reflection) noexcept {
    template_manager->UpdateInternalReflection(reflection);
    context_needs_rebuild.store(true, std::memory_order_relaxed);
    template_dirty.store(true, std::memory_order_relaxed);
    LOG_DEBUG_ContextInfo("Updated internal reflection");
}

[[msvc::forceinline]] inline void ContextInfo::UpdateInternalReflection(std::string&& reflection) noexcept {
    template_manager->UpdateInternalReflection(std::move(reflection));
    context_needs_rebuild.store(true, std::memory_order_relaxed);
    template_dirty.store(true, std::memory_order_relaxed);
    LOG_DEBUG_ContextInfo("Updated internal reflection (moved)");
}

[[msvc::forceinline]] inline void ContextInfo::UpdateEmotionalState(const std::string& emotional_state) noexcept {
    template_manager->UpdateEmotionalState(emotional_state);
    context_needs_rebuild.store(true, std::memory_order_relaxed);
    template_dirty.store(true, std::memory_order_relaxed);
    LOG_DEBUG_ContextInfo("Updated emotional state");
}

[[msvc::forceinline]] inline void ContextInfo::UpdateEmotionalState(std::string&& emotional_state) noexcept {
    template_manager->UpdateEmotionalState(std::move(emotional_state));
    context_needs_rebuild.store(true, std::memory_order_relaxed);
    template_dirty.store(true, std::memory_order_relaxed);
    LOG_DEBUG_ContextInfo("Updated emotional state (moved)");
}

[[msvc::forceinline]] inline void ContextInfo::AddPastSessionMemory(const std::string& memory) noexcept {
    template_manager->AddPastSession(memory);
    context_needs_rebuild.store(true, std::memory_order_relaxed);
    template_dirty.store(true, std::memory_order_relaxed);
    LOG_DEBUG_ContextInfo("Added past session memory");
}

[[msvc::forceinline]] inline void ContextInfo::AddPastSessionMemory(std::string&& memory) noexcept {
    template_manager->AddPastSession(std::move(memory));
    context_needs_rebuild.store(true, std::memory_order_relaxed);
    template_dirty.store(true, std::memory_order_relaxed);
    LOG_DEBUG_ContextInfo("Added past session memory (moved)");
}

[[msvc::forceinline]] inline std::string ContextInfo::BuildFullPrompt() noexcept {
    if (!template_manager) [[unlikely]] {
        LOG_ERROR_ContextInfo("Template manager not available");
        return "";
    }
    
    // Render template with current message history
    std::string rendered_prompt = template_manager->RenderTemplate(message_history);
    
    LOG_DEBUG_ContextInfo("Built full prompt: " + std::to_string(rendered_prompt.length()) + " characters");
    return rendered_prompt;
}

// Batch management is now handled by ContextInputOutput

inline bool ContextInfo::RebuildContext(RebuildStrategy strategy) {
    if (!InitializeLlamaContext()) [[unlikely]] {
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
        
        context_needs_rebuild.store(false, std::memory_order_relaxed);
        template_dirty.store(false, std::memory_order_relaxed);
        rebuild_count.fetch_add(1, std::memory_order_relaxed);
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
    if (llama_ctx) [[likely]] {
        llama_memory_clear(llama_get_memory(llama_ctx), true);
        n_past = 0;
    }
    
    // Ensure all rebuild flags are cleared
    context_needs_rebuild.store(false, std::memory_order_relaxed);
    template_dirty.store(false, std::memory_order_relaxed);
    
    // Build full prompt - actual token processing will be handled by ContextInputOutput during generation
    std::string full_prompt = BuildFullPrompt();
    
    LOG_DEBUG_ContextInfo("Full rebuild completed: context cleared, prompt built (" + 
                         std::to_string(full_prompt.length()) + " characters)");
}

inline void ContextInfo::RebuildContext_Partial() {
    // Note: context_mutex should already be held by caller
    
    LOG_DEBUG_ContextInfo("Performing partial context rebuild");
    
    // This is a simplified partial rebuild - in practice, you'd implement
    // incremental token processing based on what changed
    // For now, fall back to full rebuild
    RebuildContext_Full(); // Fallback to full rebuild for now
}

inline void ContextInfo::RebuildContext_TemplateOnly() {
    LOG_DEBUG_ContextInfo("Performing template-only rebuild");
    
    // Just re-render template without reprocessing tokens
    BuildFullPrompt();
}

// Token processing is now handled by ContextInputOutput

[[msvc::forceinline]] inline void ContextInfo::ClearContext() noexcept {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    if (llama_ctx) [[likely]] {
        llama_memory_clear(llama_get_memory(llama_ctx), true);
        n_past = 0;
    }
    
    context_needs_rebuild.store(false, std::memory_order_relaxed);  // Clear rebuild flag
    
    LOG_DEBUG_ContextInfo("Context cleared and state reset");
}

[[msvc::forceinline]] inline void ContextInfo::ClearMessageHistory() noexcept {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    message_history.clear();
    message_count.store(0, std::memory_order_relaxed);
    context_needs_rebuild.store(true, std::memory_order_relaxed);
    
    LOG_DEBUG_ContextInfo("Message history cleared");
}

// Helper function to generate context IDs
inline std::string GenerateContextId() {
    static std::atomic<size_t> counter{0};
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    return "ctx_" + std::to_string(timestamp) + "_" + std::to_string(counter++);
}

// Compile-time string generation helpers
namespace ContextStringHelpers {
    [[nodiscard]] constexpr const char* GetContextPrefix() noexcept { return "ctx_"; }
    [[nodiscard]] constexpr size_t GetContextPrefixLength() noexcept { return 4; } // strlen("ctx_")
}

// Factory methods for creating contexts with default template
inline std::unique_ptr<ContextInfo> ContextInfo::Create(const std::string& context_id, ModelInfo* model, int32_t context_size) {
    return std::make_unique<ContextInfo>(context_id, model, context_size);
}

inline std::unique_ptr<ContextInfo> ContextInfo::Create(ModelInfo* model, int32_t context_size) {
    return std::make_unique<ContextInfo>(model, context_size);
}

inline bool ContextInfo::HandleInputAsync(const std::string& input, const GenerationCallbacks& callbacks, const std::string& username) {
    LOG_DEBUG_ContextInfo("HandleInputAsync called for input: " + input.substr(0, 30) + "...");
    
    std::string full_prompt;
    bool preparation_success = false;
    
    // Critical section: prepare context for generation
    {
        LOG_DEBUG_ContextInfo("HandleInputAsync acquiring context_mutex");
        std::lock_guard<std::mutex> lock(context_mutex);
        LOG_DEBUG_ContextInfo("HandleInputAsync acquired context_mutex");
        
        try {
            // Ensure context is initialized first
            if (!InitializeLlamaContext()) [[unlikely]] {
                LOG_ERROR_ContextInfo("HandleInputAsync - Failed to initialize context");
                if (callbacks.on_error) [[likely]] {
                    callbacks.on_error("Failed to initialize context");
                }
                return false;
            }
            LOG_DEBUG_ContextInfo("HandleInputAsync - context initialized");
            
            // Check if generation system is available
            if (!io_manager) [[unlikely]] {
                LOG_ERROR_ContextInfo("HandleInputAsync - Generation system not available");
                if (callbacks.on_error) [[likely]] {
                    callbacks.on_error("Generation system not available");
                }
                return false;
            }
            LOG_DEBUG_ContextInfo("HandleInputAsync - io_manager available");
            
            // Check if already generating
            if (io_manager && io_manager->IsGenerating()) [[unlikely]] {
                LOG_WARNING_ContextInfo("HandleInputAsync - Generation already in progress");
                if (callbacks.on_error) [[likely]] {
                    callbacks.on_error("Generation already in progress");
                }
                return false;
            }
            LOG_DEBUG_ContextInfo("HandleInputAsync - not currently generating");
            
            // Try to transition to generating state for processing
            if (!TrySetState(ContextState::CONTEXT_GENERATING)) [[unlikely]] {
                ContextState current_state = GetState();
                LOG_WARNING_ContextInfo("HandleInputAsync - Context not available for generation. Current state: " + 
                                       std::to_string(static_cast<int>(current_state)) + " (" + GetStateString() + ")");
                
                // If we're stuck in GENERATING state but not actually generating, force reset
                if (current_state == ContextState::CONTEXT_GENERATING && (!io_manager || !io_manager->IsGenerating())) [[unlikely]] {
                    LOG_WARNING_ContextInfo("HandleInputAsync - Context stuck in GENERATING state but not actually generating, forcing reset");
                    ContextState previous_state = ForceResetToIdle("stuck in GENERATING state");
                    LOG_WARNING_ContextInfo("HandleInputAsync - Forced reset from state " + std::to_string(static_cast<int>(previous_state)) + " to IDLE");
                    
                    // Try the transition again after reset
                    if (!TrySetState(ContextState::CONTEXT_GENERATING)) [[unlikely]] {
                        LOG_ERROR_ContextInfo("HandleInputAsync - Still cannot transition to GENERATING after force reset");
                        if (callbacks.on_error) [[likely]] {
                            callbacks.on_error("Context not available for generation after recovery attempt");
                        }
                        return false;
                    }
                    LOG_DEBUG_ContextInfo("HandleInputAsync - Successfully transitioned to GENERATING after force reset");
                } else {
                    if (callbacks.on_error) [[likely]] {
                        callbacks.on_error("Context not available for generation");
                    }
                    return false;
                }
            }
            LOG_DEBUG_ContextInfo("HandleInputAsync - state set to GENERATING");

            // 1. Add user message to history
            message_history.emplace_back(username, input);
            message_count.fetch_add(1, std::memory_order_relaxed);
            context_needs_rebuild.store(true, std::memory_order_relaxed);
            
            LOG_DEBUG_ContextInfo("Processing async input: " + username + " -> " + 
                                 input.substr(0, 50) + (input.length() > 50 ? "..." : ""));
            
            // 2. Check context size - delegate to plugins for monitoring
            // Basic size estimation for logging purposes only
            std::string estimated_prompt = BuildFullPrompt();
            size_t estimated_length = estimated_prompt.length();
            LOG_DEBUG_ContextInfo("Estimated prompt length: " + std::to_string(estimated_length) + " characters");
        
            // Check if we're approaching the limit with the new message
            if (max_context_tokens > 0 && estimated_length > max_context_tokens * ContextConstants::CharToTokenEstimateRatio()) [[unlikely]] { // Compile-time char-to-token estimate
                LOG_WARNING_ContextInfo("Context approaching length limit - plugin monitoring should handle this");
            }
            
            // 3. Rebuild context with the new message
            if (context_needs_rebuild.load(std::memory_order_relaxed) || template_dirty.load(std::memory_order_relaxed)) [[likely]] {
                LOG_DEBUG_ContextInfo("HandleInputAsync - rebuilding context");
                if (!RebuildContext(RebuildStrategy::FULL)) [[unlikely]] {
                    LOG_ERROR_ContextInfo("HandleInputAsync - Failed to rebuild context");
                    
                    // CRITICAL: Reset state on rebuild failure to prevent stuck states
                    if (!TrySetState(ContextState::CONTEXT_IDLE)) [[unlikely]] {
                        LOG_WARNING_ContextInfo("HandleInputAsync rebuild failure - failed to reset to IDLE, forcing reset");
                        ForceResetToIdle("rebuild failure recovery");
                    }
                    
                    if (callbacks.on_error) [[likely]] {
                        callbacks.on_error("Failed to rebuild context");
                    }
                    return false;
                }
                LOG_DEBUG_ContextInfo("HandleInputAsync - context rebuilt successfully");
            }
            
            // 4. Final safety check after rebuild - delegate to plugins
            // ContextInputOutput will handle actual token validation during generation
            
            // 5. Prepare for generation
            full_prompt = BuildFullPrompt();
            preparation_success = true;
            LOG_DEBUG_ContextInfo("HandleInputAsync - preparation completed successfully");
            
        } catch (const std::exception& e) {
            LOG_ERROR("ContextInfo", "Exception in HandleInputAsync preparation: " + std::string(e.what()));
            
            // CRITICAL: Always reset state on exceptions to prevent stuck states
            if (!TrySetState(ContextState::CONTEXT_IDLE)) [[unlikely]] {
                LOG_WARNING_ContextInfo("HandleInputAsync exception handler - failed to reset to IDLE, forcing reset");
                ForceResetToIdle("HandleInputAsync exception recovery");
            }
            
            if (callbacks.on_error) [[likely]] {
                callbacks.on_error("Exception: " + std::string(e.what()));
            }
            return false;
        }
    } // Release context_mutex here
    LOG_DEBUG_ContextInfo("HandleInputAsync released context_mutex");
    
    // Start async generation OUTSIDE the critical section to avoid deadlock
    if (preparation_success) [[likely]] {
        LOG_DEBUG_ContextInfo("HandleInputAsync - starting async generation outside critical section");
        
        // Create properly structured callbacks for ContextInputOutput
        GenerationCallbacks io_callbacks;
        io_callbacks.on_token = callbacks.on_token;
        io_callbacks.on_error = callbacks.on_error;
        io_callbacks.on_complete = [this, callbacks](const std::string& response, bool success) {
            LOG_DEBUG_ContextInfo("Generation completion callback called - success: " + std::to_string(success));
            
            // Add assistant response to message history upon completion
            if (success && !response.empty()) [[likely]] {
                LOG_DEBUG_ContextInfo("Completion callback attempting to acquire context_mutex for success handling");
                auto start_time = std::chrono::steady_clock::now();
                {
                    std::lock_guard<std::mutex> lock(context_mutex);
                    auto end_time = std::chrono::steady_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                    LOG_DEBUG_ContextInfo("Completion callback acquired context_mutex for success handling (took " + std::to_string(duration.count()) + "ms)");
                    
                    message_history.emplace_back("assistant", response);
                    message_count.fetch_add(1, std::memory_order_relaxed);
                    generation_count.fetch_add(1, std::memory_order_relaxed);
                    (void)TrySetState(ContextState::CONTEXT_IDLE);
                    LOG_DEBUG_ContextInfo("Completion callback - success handling complete, state set to IDLE");
                }
                LOG_DEBUG_ContextInfo("Completion callback released context_mutex after success handling");
            } else {
                // Also reset state on failure - CRITICAL: Always reset to IDLE on failure
                LOG_DEBUG_ContextInfo("Completion callback attempting to acquire context_mutex for error handling");
                auto start_time = std::chrono::steady_clock::now();
                {
                    std::lock_guard<std::mutex> lock(context_mutex);
                    auto end_time = std::chrono::steady_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                    LOG_DEBUG_ContextInfo("Completion callback acquired context_mutex for error handling (took " + std::to_string(duration.count()) + "ms)");
                    
                    // On failure, try to go to IDLE first, then ERROR_STATE if needed
                    if (!TrySetState(ContextState::CONTEXT_IDLE)) [[unlikely]] {
                        LOG_WARNING_ContextInfo("Completion callback - failed to set IDLE state, forcing reset");
                        ForceResetToIdle("generation failure recovery");
                    }
                    LOG_DEBUG_ContextInfo("Completion callback - error handling complete, state reset to IDLE");
                }
                LOG_DEBUG_ContextInfo("Completion callback released context_mutex after error handling");
            }
            
            // Forward to original callback
            LOG_DEBUG_ContextInfo("Completion callback forwarding to original callback");
            if (callbacks.on_complete) [[likely]] {
                callbacks.on_complete(response, success);
            }
            LOG_DEBUG_ContextInfo("Completion callback finished");
        };
        
        LOG_DEBUG_ContextInfo("HandleInputAsync calling io_manager->GenerateResponseAsync");
        return io_manager->GenerateResponseAsync(full_prompt, io_callbacks);
    }
    
    LOG_ERROR_ContextInfo("HandleInputAsync - preparation failed, returning false");
    return false;
}

// ContextInputOutput handles all generation operations

[[msvc::forceinline]] inline void ContextInfo::ApplyCompletedSummary(const std::string& summary) noexcept {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    if (!summary.empty()) {
        // Add summary to chronological list and update template with all summaries
        AddSummaryToList(summary);
        UpdateTemplateWithAllSummaries();
        context_needs_rebuild.store(true, std::memory_order_relaxed);
        
        LOG_ContextInfo("Added new summary to chronological list (total: " + 
                       std::to_string(summaries.size()) + "/" + std::to_string(MAX_SUMMARIES) + "): " + 
                       summary.substr(0, 100) + (summary.length() > 100 ? "..." : ""));
    }
}

[[msvc::forceinline]] inline void ContextInfo::ApplyCompletedSummary(std::string&& summary) noexcept {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    if (!summary.empty()) {
        // Add summary to chronological list and update template with all summaries
        AddSummaryToList(std::move(summary));
        UpdateTemplateWithAllSummaries();
        context_needs_rebuild.store(true, std::memory_order_relaxed);
        
        LOG_ContextInfo("Added new summary to chronological list (total: " + 
                       std::to_string(summaries.size()) + "/" + std::to_string(MAX_SUMMARIES) + ") (moved)");
    }
}

// Advanced features
inline void ContextInfo::SetMaxContextTokens(size_t max_tokens) noexcept {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    // Update max context tokens
    max_context_tokens = max_tokens;
    
    // Update IO manager bounds if it exists
    if (io_manager) [[likely]] {
        io_manager->UpdateContextBounds(max_tokens);
    }
    
    LOG_DEBUG_ContextInfo("Max context tokens set to: " + std::to_string(max_tokens));
}

inline std::string ContextInfo::GetCurrentPrompt() const {
    // For const method, return a simple representation without modifying state
    std::ostringstream prompt_stream;
    prompt_stream << "Current context with " << message_count.load(std::memory_order_relaxed) << " messages";
    return prompt_stream.str();
}

[[msvc::forceinline]] inline std::vector<std::pair<std::string, std::string>> ContextInfo::PruneContextImmediateWithExtraction(size_t keep_recent_messages) noexcept {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    if (message_history.size() <= keep_recent_messages) [[likely]] {
        LOG_DEBUG_ContextInfo("No pruning needed: only " + std::to_string(message_history.size()) + " messages");
        return {}; // Return empty vector
    }
    
    // Extract messages to be pruned for potential summarization
    size_t prune_count = message_history.size() - keep_recent_messages;
    std::vector<std::pair<std::string, std::string>> pruned_messages;
    pruned_messages.reserve(prune_count); // Pre-allocate for performance
    pruned_messages.assign(message_history.begin(), message_history.begin() + prune_count);
    
    // Immediately prune from active history
    message_history.erase(message_history.begin(), message_history.begin() + prune_count);
    message_count.store(keep_recent_messages, std::memory_order_relaxed);
    context_needs_rebuild.store(true, std::memory_order_relaxed);
    
    LOG_ContextInfo("Pruned " + std::to_string(prune_count) + " messages from context - returning for orchestrator coordination");
    
    return pruned_messages; // Return extracted messages for summarization
}

// REMOVED: Static member definitions moved to respective plugins
// - pruning_buffer_mutex, global_pruning_buffer moved to SummarizationPlugin
// - emotional_analysis_buffer_mutex, global_emotional_analysis_buffer moved to EmoTagPlugin

// Implementation of new summary management methods
[[msvc::forceinline]] inline void ContextInfo::AddSummaryToList(const std::string& summary) noexcept {
    if (summary.empty()) [[unlikely]] return;
    
    // Add new summary to the end (newest)
    summaries.push_back(summary);
    
    // Enforce maximum limit by removing oldest summary if needed
    if (summaries.size() > MAX_SUMMARIES) [[unlikely]] {
        summaries.erase(summaries.begin()); // Remove oldest (first) summary
        LOG_ContextInfo("Removed oldest summary to maintain maximum of " + 
                       std::to_string(MAX_SUMMARIES) + " summaries");
    }
    
    LOG_ContextInfo("Summary list updated: " + std::to_string(summaries.size()) + 
                   " summaries in chronological order");
}

[[msvc::forceinline]] inline void ContextInfo::AddSummaryToList(std::string&& summary) noexcept {
    if (summary.empty()) [[unlikely]] return;
    
    // Add new summary to the end (newest)
    summaries.push_back(std::move(summary));
    
    // Enforce maximum limit by removing oldest summary if needed
    if (summaries.size() > MAX_SUMMARIES) [[unlikely]] {
        summaries.erase(summaries.begin()); // Remove oldest (first) summary
        LOG_ContextInfo("Removed oldest summary to maintain maximum of " + 
                       std::to_string(MAX_SUMMARIES) + " summaries");
    }
    
    LOG_ContextInfo("Summary list updated (moved): " + std::to_string(summaries.size()) + 
                   " summaries in chronological order");
}

[[msvc::forceinline]] inline void ContextInfo::UpdateTemplateWithAllSummaries() noexcept {
    if (template_manager) [[likely]] {
        template_manager->UpdateMultipleSummaries(summaries);
        LOG_ContextInfo("Updated template with " + std::to_string(summaries.size()) + 
                       " summaries in chronological order");
    }
}
