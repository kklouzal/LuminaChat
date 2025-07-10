#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <functional>
#include <unordered_map>
#include <chrono>
#include <queue>
#include <mutex>
#include <shared_mutex>
#include <atomic>

#include "ProcessingPipeline.hpp"
#include "Sanitizer.hpp"
#include "LlamaManager.hpp"
#include "Context/ContextInfo.hpp"
#include "Logger.hpp"
#include "ErrorHandling.hpp"

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

// Emotional analysis buffer for plugin consumption
struct EmotionalAnalysisBatch {
    std::string context_id;
    std::vector<std::string> ai_responses; // Recent AI responses for analysis
    std::chrono::steady_clock::time_point requested_at;
    
    EmotionalAnalysisBatch(const std::string& id, 
                          std::vector<std::string> responses)
        : context_id(id), ai_responses(std::move(responses)), 
          requested_at(std::chrono::steady_clock::now()) {}
};

// Performance constants and compile-time optimizations
namespace OrchestratorConstants {
    constexpr size_t CACHE_LINE_SIZE = 64;
    constexpr size_t CONTEXT_STATE_INITIAL_BUCKETS = 32;
    constexpr size_t SCHEDULED_TASK_RESERVE = 16;
    constexpr size_t STRING_POOL_INITIAL_SIZE = 256;
    
    // Performance thresholds
    constexpr size_t SMALL_STRING_OPTIMIZATION_THRESHOLD = 23; // SSO threshold
    constexpr size_t UNORDERED_MAP_LOAD_FACTOR_THRESHOLD = 1;   // Keep load factor reasonable
    
    // Compile-time validation functions
    [[nodiscard]] consteval bool ValidateConstants() {
        static_assert(CACHE_LINE_SIZE == 64, "Cache line size must be 64 bytes for optimal alignment");
        static_assert(CONTEXT_STATE_INITIAL_BUCKETS > 0, "Initial buckets must be positive");
        static_assert(SCHEDULED_TASK_RESERVE > 0, "Task reserve must be positive");
        static_assert(SMALL_STRING_OPTIMIZATION_THRESHOLD >= 15, "SSO threshold too small for typical implementations");
        return true;
    }
    
    // Force compile-time validation
    constexpr bool constants_validated = ValidateConstants();
    
    // Compile-time string optimization helpers
    [[nodiscard]] constexpr bool UsesSSOOptimization(size_t str_size) noexcept {
        return str_size <= SMALL_STRING_OPTIMIZATION_THRESHOLD;
    }
    
    [[nodiscard]] constexpr bool ShouldPreallocate(size_t expected_size) noexcept {
        return expected_size > SMALL_STRING_OPTIMIZATION_THRESHOLD;
    }
    
    // Consteval functions for compile-time configuration
    [[nodiscard]] consteval size_t CalculateOptimalReserveSize(size_t expected_load) {
        return expected_load + (expected_load / 4); // 25% overhead for growth
    }
    
    [[nodiscard]] consteval bool IsValidCacheLineAlignment(size_t alignment) {
        return alignment == 64 || alignment == 128; // Common cache line sizes
    }
}

// Forward declarations
namespace LuminaChat {
    class SummarizationPlugin;
    class EmoTagPlugin;
    class ContextPruningPlugin;
    enum class RequestPriority : uint8_t;
}

// Use plugin structures directly instead of duplicating them
namespace LuminaChat {
    struct SummarizationRequest;
    struct SummarizationResponse;
    struct EmotionAnalysisRequest;
    struct EmotionAnalysisResponse;
    struct ContextUsageStats;  // Forward declaration for ContextPruning
}

// Forward declarations for ContextInfo types
struct PrunedMessageBatch;
struct EmotionalAnalysisBatch;
enum class ContextState : uint8_t;

// Optimized Discord request/response structures with cache alignment
struct alignas(OrchestratorConstants::CACHE_LINE_SIZE) DiscordChannelRequest {
    enum class Type : uint8_t { INCOMING_MESSAGE, OUTGOING_MESSAGE, CHANNEL_SETUP };
    Type request_type;
    std::string channel_id;
    std::string username;
    std::string content;
    std::string timestamp;
    
    // Move constructor for performance
    DiscordChannelRequest(DiscordChannelRequest&& other) noexcept = default;
    DiscordChannelRequest& operator=(DiscordChannelRequest&& other) noexcept = default;
    
    // Copy constructor
    DiscordChannelRequest(const DiscordChannelRequest& other) = default;
    DiscordChannelRequest& operator=(const DiscordChannelRequest& other) = default;
    
    DiscordChannelRequest() = default;
};

struct alignas(OrchestratorConstants::CACHE_LINE_SIZE) DiscordChannelResponse {
    bool should_respond = true;
    std::string response_content;
    std::string internal_reasoning = "";  // Inner voice reasoning/thoughts
    std::string target_channel;
    std::string error_message;
    size_t context_current = 0;  // Current context usage
    size_t context_maximum = 0;  // Maximum context size
    std::string model_name = "";  // Which model was used
    std::string processing_time = "";  // How long processing took
    
    // Move constructor for performance
    DiscordChannelResponse(DiscordChannelResponse&& other) noexcept = default;
    DiscordChannelResponse& operator=(DiscordChannelResponse&& other) noexcept = default;
    
    // Copy constructor
    DiscordChannelResponse(const DiscordChannelResponse& other) = default;
    DiscordChannelResponse& operator=(const DiscordChannelResponse& other) = default;
    
    DiscordChannelResponse() = default;
};

// Input source enumeration
enum class InputSource {
    UI,
    DISCORD,
    SYSTEM,
    SCHEDULED_TASK
};

// Compile-time utility functions for InputSource
[[nodiscard]] constexpr bool IsUserSource(InputSource source) noexcept {
    return source == InputSource::UI || source == InputSource::DISCORD;
}

[[nodiscard]] constexpr bool IsSystemSource(InputSource source) noexcept {
    return source == InputSource::SYSTEM || source == InputSource::SCHEDULED_TASK;
}

[[nodiscard]] constexpr bool RequiresSanitization(InputSource source) noexcept {
    return source == InputSource::DISCORD || source == InputSource::UI;
}

// Processing state tracking
enum class ProcessingState {
    NORMAL_PROCESSING,
    AWAITING_SUMMARIZATION,
    BACKFILL_IN_PROGRESS,
    ERROR_STATE
};

// Compile-time utility functions for ProcessingState
[[nodiscard]] constexpr bool IsAvailableForProcessing(ProcessingState state) noexcept {
    return state == ProcessingState::NORMAL_PROCESSING;
}

[[nodiscard]] constexpr bool IsErrorState(ProcessingState state) noexcept {
    return state == ProcessingState::ERROR_STATE;
}

[[nodiscard]] constexpr bool IsBusyState(ProcessingState state) noexcept {
    return state != ProcessingState::NORMAL_PROCESSING;
}

// Scheduled task types
enum class ScheduledTaskType {
    CONTEXT_MAINTENANCE,
    CACHE_CLEANUP,
    HEALTH_CHECK,
    DISCORD_PRESENCE_UPDATE,
    PRUNING_BUFFER_PROCESSING,  // Processes messages pruned by context size management
    EMOTION_ANALYSIS_PROCESSING,  // Processes queued emotional analysis requests
    CONTEXT_SIZE_MONITORING,      // Monitors context sizes and triggers pruning when needed
    INNER_VOICE_ANALYSIS,         // Analyzes inner voice pairs and creates summarization requests
    SUMMARIZATION_PROCESSING      // Monitors summarization pipeline health (not processing requests)
};

// Compile-time utility functions for ScheduledTaskType
[[nodiscard]] constexpr bool IsMaintenanceTask(ScheduledTaskType type) noexcept {
    return type == ScheduledTaskType::CONTEXT_MAINTENANCE || 
           type == ScheduledTaskType::CACHE_CLEANUP;
}

[[nodiscard]] constexpr bool IsProcessingTask(ScheduledTaskType type) noexcept {
    return type == ScheduledTaskType::PRUNING_BUFFER_PROCESSING || 
           type == ScheduledTaskType::EMOTION_ANALYSIS_PROCESSING ||
           type == ScheduledTaskType::CONTEXT_SIZE_MONITORING ||
           type == ScheduledTaskType::SUMMARIZATION_PROCESSING;
}

[[nodiscard]] constexpr bool IsHighFrequencyTask(ScheduledTaskType type) noexcept {
    return type == ScheduledTaskType::EMOTION_ANALYSIS_PROCESSING ||
           type == ScheduledTaskType::DISCORD_PRESENCE_UPDATE ||
           type == ScheduledTaskType::CONTEXT_SIZE_MONITORING ||
           type == ScheduledTaskType::INNER_VOICE_ANALYSIS ||
           type == ScheduledTaskType::SUMMARIZATION_PROCESSING;
}

struct ScheduledTask {
    ScheduledTaskType type;
    std::chrono::steady_clock::time_point next_run;
    std::chrono::milliseconds interval;
    std::function<void()> task_function;
    bool active = true;
};

/**
 * @class Orchestrator
 * @brief Central message routing and AI workflow orchestration hub
 * 
 * The Orchestrator serves as the central coordinator for all AI processing workflows,
 * implementing a plugin architecture that handles complex multi-step operations
 * through a consistent Request -> Process -> Callback pattern.
 * 
 * Key responsibilities:
 * - Message routing between UI/Discord and ContextInfo
 * - Plugin management for summarization, sanitization, Discord channel routing
 * - State management to track active workflows and prevent conflicts
 * - Scheduled task foundation for timed operations
 * - Integration with existing Sanitizer for content filtering
 */
class Orchestrator {
private:
    // Core component references
    std::unique_ptr<Sanitizer> sanitizer;
    LlamaManager* llama_manager;
    
    // Plugin references for delegation with atomic availability flags
    LuminaChat::SummarizationPlugin* summarization_plugin = nullptr;
    LuminaChat::EmoTagPlugin* emotag_plugin = nullptr;
    LuminaChat::ContextPruningPlugin* context_pruning_plugin = nullptr;
    std::atomic<bool> summarization_plugin_available{false};
    std::atomic<bool> emotag_plugin_available{false};
    std::atomic<bool> context_pruning_plugin_available{false};
    
    // Processing pipelines for plugin architecture
    LuminaChat::ProcessingPipeline<LuminaChat::SummarizationRequest, LuminaChat::SummarizationResponse> summarization_pipeline;
    LuminaChat::ProcessingPipeline<LuminaChat::EmotionAnalysisRequest, LuminaChat::EmotionAnalysisResponse> emotion_analysis_pipeline;
    LuminaChat::ProcessingPipeline<DiscordChannelRequest, DiscordChannelResponse> discord_channel_pipeline;
    
    // State tracking for contexts - optimized with unordered_map for O(1) lookup with good bucket management
    std::unordered_map<std::string, ProcessingState> context_states;
    mutable std::shared_mutex state_mutex;  // Use shared_mutex for read-mostly access patterns
    
    // Scheduled task management - pre-allocated for cache efficiency
    std::vector<ScheduledTask> scheduled_tasks;
    std::chrono::steady_clock::time_point last_scheduled_run;
    mutable std::shared_mutex task_mutex;  // Use shared_mutex for read-mostly access patterns
    std::atomic<size_t> active_task_count{0};  // Lock-free task count for quick checks
    
    // Callback registrations
    std::function<void(std::string_view, InputSource)> output_callback;
    std::function<void(const DiscordChannelResponse&)> discord_response_callback;  // Enhanced Discord response
    std::atomic<bool> output_callback_registered{false};  // Lock-free callback availability check
    
    // Plugin processing functions
    //
    // CRITICAL: This function is defined at the END of SummarizationPlugin.hpp to combat circular dependencies
    void ProcessSummarizationRequest(const LuminaChat::SummarizationRequest& request, 
                                   std::function<void(LuminaChat::SummarizationResponse)> callback);
    // CRITICAL: This function is defined at the END of EmoTagPlugin.hpp to combat circular dependencies
    void ProcessEmotionAnalysisRequest(const LuminaChat::EmotionAnalysisRequest& request,
                                     std::function<void(LuminaChat::EmotionAnalysisResponse)> callback);
    void ProcessDiscordChannelRequest(const DiscordChannelRequest& request,
                                    std::function<void(DiscordChannelResponse)> callback);
    
    // Discord two-stage reasoning implementation
    void ExecuteDiscordTwoStageReasoning(ContextInfo* inner_context, ContextInfo* outer_context,
                                       const std::string& input, const DiscordChannelRequest& request,
                                       std::function<void(DiscordChannelResponse)> callback);
    
    // Internal helpers - optimized with string_view for hot paths
    [[nodiscard]] std::string GetSanitizedInput(std::string_view input, InputSource source);
    [[nodiscard]] bool IsContextBusy(std::string_view context_id) const noexcept;
    void SetContextState(std::string_view context_id, ProcessingState state);
    [[nodiscard]] ProcessingState GetContextState(std::string_view context_id) const noexcept;
    
public:
    /**
     * @brief Constructor
     * @param llama_manager Pointer to the LlamaManager instance
     */
    explicit Orchestrator(LlamaManager* llama_manager);
    
    /**
     * @brief Destructor - ensures clean shutdown
     */
    ~Orchestrator();
    
    // Core initialization and shutdown
    [[nodiscard]] bool Initialize();
    void Shutdown();
    
    // Callback registration (called by higher-level components)
    void RegisterOutputCallback(std::function<void(std::string_view, InputSource)>&& callback);
    void RegisterDiscordResponseCallback(std::function<void(const DiscordChannelResponse&)>&& callback);
    
    // Core message routing - optimized with string_view for reduced copying
    void InputReceived(std::string_view input, std::string_view context_id, 
                      InputSource source, std::string_view username = "User");
    
    // Discord integration - optimized with string_view for reduced copying
    void OnRawDiscordMessage(std::string_view content, std::string_view channel_id, 
                           std::string_view username);
    
    // Plugin workflow coordination
    // CRITICAL: This function is defined at the END of SummarizationPlugin.hpp to combat circular dependencies
    void RequestSummarization(const PrunedMessageBatch& batch);
    // CRITICAL: This function is defined at the END of SummarizationPlugin.hpp to combat circular dependencies
    void OnSummarizationComplete(const std::string& context_id, const LuminaChat::SummarizationResponse& response);

    // CRITICAL: This function is defined at the END of EmoTagPlugin.hpp to combat circular dependencies
    void RequestEmotionAnalysis(const EmotionalAnalysisBatch& batch, LuminaChat::RequestPriority priority = LuminaChat::RequestPriority::NORMAL);
    // CRITICAL: This function is defined at the END of EmoTagPlugin.hpp to combat circular dependencies
    void OnEmotionAnalysisComplete(const std::string& context_id, const LuminaChat::EmotionAnalysisResponse& response);
    
    // Scheduled task management
    void AddScheduledTask(ScheduledTaskType type, std::chrono::milliseconds interval,
                         std::function<void()>&& task_function);
    void RemoveScheduledTask(ScheduledTaskType type);
    void ProcessScheduledTasks();
    
    // Manual processing triggers (public for immediate processing)
    // CRITICAL: This function is defined at the END of EmoTagPlugin.hpp to combat circular dependencies
    void ProcessEmotionAnalysisBuffer();  // Process emotion analysis buffer
    void ProcessInnerVoiceAnalysis();     // Process inner voice user input/AI output pairs
    // CRITICAL: This function is defined at the END of ContextPruningPlugin.hpp to combat circular dependencies
    void MonitorAllContextSizes();        // Monitor all context sizes for pruning
    // CRITICAL: This function is defined at the END of SummarizationPlugin.hpp to combat circular dependencies
    void ProcessSummarizationPipeline();  // Process pending summarization requests
    
    // Plugin coordination helpers
    // CRITICAL: This function is defined at the END of SummarizationPlugin.hpp to combat circular dependencies
    void RequestSummarizationForPrunedMessages(const std::string& context_id, const std::vector<std::pair<std::string, std::string>>& pruned_messages);
    
    // Core component access for plugins
    [[nodiscard]] LlamaManager* GetLlamaManager() noexcept { return llama_manager; }
    [[nodiscard]] const LlamaManager* GetLlamaManager() const noexcept { return llama_manager; }
    [[nodiscard]] LuminaChat::EmoTagPlugin* GetEmoTagPlugin() noexcept { return emotag_plugin; }
    [[nodiscard]] const LuminaChat::EmoTagPlugin* GetEmoTagPlugin() const noexcept { return emotag_plugin; }
    [[nodiscard]] LuminaChat::SummarizationPlugin* GetSummarizationPlugin() noexcept { return summarization_plugin; }
    [[nodiscard]] const LuminaChat::SummarizationPlugin* GetSummarizationPlugin() const noexcept { return summarization_plugin; }
    
    // Compile-time configuration queries
    [[nodiscard]] static constexpr size_t GetCacheLineSize() noexcept { return OrchestratorConstants::CACHE_LINE_SIZE; }
    [[nodiscard]] static constexpr size_t GetInitialContextBuckets() noexcept { return OrchestratorConstants::CONTEXT_STATE_INITIAL_BUCKETS; }
    [[nodiscard]] static constexpr size_t GetScheduledTaskReserve() noexcept { return OrchestratorConstants::SCHEDULED_TASK_RESERVE; }
    [[nodiscard]] static constexpr size_t GetSSOThreshold() noexcept { return OrchestratorConstants::SMALL_STRING_OPTIMIZATION_THRESHOLD; }
    
    // Compile-time utility for template parameters and static assertions
    template<typename T>
    [[nodiscard]] static constexpr bool IsAlignedToCache() noexcept {
        return alignof(T) >= OrchestratorConstants::CACHE_LINE_SIZE;
    }
    
    // Lock-free plugin availability checks
    [[nodiscard]] bool IsSummarizationPluginAvailable() const noexcept {
        return summarization_plugin_available.load(std::memory_order_relaxed);
    }
    [[nodiscard]] bool IsEmoTagPluginAvailable() const noexcept {
        return emotag_plugin_available.load(std::memory_order_relaxed);
    }
    [[nodiscard]] bool IsContextPruningPluginAvailable() const noexcept {
        return context_pruning_plugin_available.load(std::memory_order_relaxed);
    }
    [[nodiscard]] bool IsOutputCallbackAvailable() const noexcept {
        return output_callback_registered.load(std::memory_order_relaxed);
    }
    
    // Settings access
    [[nodiscard]] SettingsManager* GetSettingsManager() { 
        if (llama_manager) [[likely]] {
            return llama_manager->GetSettingsManager();
        } else [[unlikely]] {
            return nullptr;
        }
    }
    [[nodiscard]] const SettingsManager* GetSettingsManager() const { 
        if (llama_manager) [[likely]] {
            return llama_manager->GetSettingsManager();
        } else [[unlikely]] {
            return nullptr;
        }
    }
    
    // Plugin registration
    void RegisterSummarizationPlugin(LuminaChat::SummarizationPlugin* plugin) noexcept { 
        summarization_plugin = plugin; 
        summarization_plugin_available.store(plugin != nullptr, std::memory_order_relaxed);
    }
    void RegisterEmoTagPlugin(LuminaChat::EmoTagPlugin* plugin) noexcept { 
        emotag_plugin = plugin; 
        emotag_plugin_available.store(plugin != nullptr, std::memory_order_relaxed);
    }
    void RegisterContextPruningPlugin(LuminaChat::ContextPruningPlugin* plugin) noexcept { 
        context_pruning_plugin = plugin; 
        context_pruning_plugin_available.store(plugin != nullptr, std::memory_order_relaxed);
    }
    
    // Statistics and monitoring - cache-aligned for optimal performance
    struct alignas(OrchestratorConstants::CACHE_LINE_SIZE) OrchestrationStats {
        // Use atomic counters for lock-free statistics in hot paths
        std::atomic<size_t> messages_processed{0};
        std::atomic<size_t> summarizations_completed{0};
        std::atomic<size_t> emotion_analyses_completed{0};
        std::atomic<size_t> discord_messages_handled{0};
        std::atomic<size_t> scheduled_tasks_executed{0};
        std::atomic<size_t> sanitization_blocks{0};
        std::chrono::steady_clock::time_point start_time;
        
        // Copy constructor for atomic variables
        OrchestrationStats(const OrchestrationStats& other) noexcept
            : messages_processed(other.messages_processed.load(std::memory_order_relaxed))
            , summarizations_completed(other.summarizations_completed.load(std::memory_order_relaxed))
            , emotion_analyses_completed(other.emotion_analyses_completed.load(std::memory_order_relaxed))
            , discord_messages_handled(other.discord_messages_handled.load(std::memory_order_relaxed))
            , scheduled_tasks_executed(other.scheduled_tasks_executed.load(std::memory_order_relaxed))
            , sanitization_blocks(other.sanitization_blocks.load(std::memory_order_relaxed))
            , start_time(other.start_time) {}
        
        // Move constructor for atomic variables
        OrchestrationStats(OrchestrationStats&& other) noexcept
            : messages_processed(other.messages_processed.load(std::memory_order_relaxed))
            , summarizations_completed(other.summarizations_completed.load(std::memory_order_relaxed))
            , emotion_analyses_completed(other.emotion_analyses_completed.load(std::memory_order_relaxed))
            , discord_messages_handled(other.discord_messages_handled.load(std::memory_order_relaxed))
            , scheduled_tasks_executed(other.scheduled_tasks_executed.load(std::memory_order_relaxed))
            , sanitization_blocks(other.sanitization_blocks.load(std::memory_order_relaxed))
            , start_time(std::move(other.start_time)) {}
        
        // Assignment operator for atomic variables
        OrchestrationStats& operator=(const OrchestrationStats& other) noexcept {
            if (this != &other) {
                messages_processed.store(other.messages_processed.load(std::memory_order_relaxed), std::memory_order_relaxed);
                summarizations_completed.store(other.summarizations_completed.load(std::memory_order_relaxed), std::memory_order_relaxed);
                emotion_analyses_completed.store(other.emotion_analyses_completed.load(std::memory_order_relaxed), std::memory_order_relaxed);
                discord_messages_handled.store(other.discord_messages_handled.load(std::memory_order_relaxed), std::memory_order_relaxed);
                scheduled_tasks_executed.store(other.scheduled_tasks_executed.load(std::memory_order_relaxed), std::memory_order_relaxed);
                sanitization_blocks.store(other.sanitization_blocks.load(std::memory_order_relaxed), std::memory_order_relaxed);
                start_time = other.start_time;
            }
            return *this;
        }
        
        // Move assignment operator for atomic variables
        OrchestrationStats& operator=(OrchestrationStats&& other) noexcept {
            if (this != &other) {
                messages_processed.store(other.messages_processed.load(std::memory_order_relaxed), std::memory_order_relaxed);
                summarizations_completed.store(other.summarizations_completed.load(std::memory_order_relaxed), std::memory_order_relaxed);
                emotion_analyses_completed.store(other.emotion_analyses_completed.load(std::memory_order_relaxed), std::memory_order_relaxed);
                discord_messages_handled.store(other.discord_messages_handled.load(std::memory_order_relaxed), std::memory_order_relaxed);
                scheduled_tasks_executed.store(other.scheduled_tasks_executed.load(std::memory_order_relaxed), std::memory_order_relaxed);
                sanitization_blocks.store(other.sanitization_blocks.load(std::memory_order_relaxed), std::memory_order_relaxed);
                start_time = std::move(other.start_time);
            }
            return *this;
        }
        
        // Default constructor
        OrchestrationStats() noexcept : start_time(std::chrono::steady_clock::now()) {}
    };
    
    [[nodiscard]] OrchestrationStats GetStats() const;
    void ResetStats() noexcept;

private:
    OrchestrationStats stats;
    std::atomic<bool> is_initialized{false};  // Lock-free initialization check
    std::atomic<bool> is_shutting_down{false};  // Lock-free shutdown state
    
    // Internal pipeline processors
    void InitializePipelines();
    void SetupDefaultScheduledTasks();
    
    // Context maintenance tasks
    void PerformContextMaintenance();
    void PerformCacheCleanup();
    void PerformHealthCheck();
    void UpdateDiscordPresence();
    // CRITICAL: This function is defined at the END of SummarizationPlugin.hpp to combat circular dependencies
    void ProcessPruningBuffer();
};

// Implementation

inline Orchestrator::Orchestrator(LlamaManager* llama_manager)
    : llama_manager(llama_manager)
    , sanitizer(std::make_unique<Sanitizer>())
    , summarization_pipeline("SummarizationPipeline")
    , emotion_analysis_pipeline("EmotionAnalysisPipeline")
    , discord_channel_pipeline("DiscordChannelPipeline")
    , last_scheduled_run(std::chrono::steady_clock::now())
    , stats{} {
    
    stats.start_time = std::chrono::steady_clock::now();
    
    // Pre-allocate collections for optimal performance with perfect sizing
    scheduled_tasks.reserve(OrchestratorConstants::SCHEDULED_TASK_RESERVE);
    
    // Pre-allocate context_states for better performance with fewer rehashes
    context_states.reserve(OrchestratorConstants::CONTEXT_STATE_INITIAL_BUCKETS);
}

inline Orchestrator::~Orchestrator() {
    Shutdown();
}

inline [[nodiscard]] bool Orchestrator::Initialize() {
    // Early exit if already initialized - lock-free check
    if (is_initialized.load(std::memory_order_relaxed)) [[unlikely]] {
        LOG_Orchestrator("Orchestrator already initialized");
        return true;
    }
    
    LOG_Orchestrator("Initializing Orchestrator...");
    
    // Sanitizer is initialized in constructor - no additional initialization needed
    LOG_Orchestrator("Sanitizer ready");
    
    // Initialize processing pipelines
    InitializePipelines();
    
    // Setup default scheduled tasks
    SetupDefaultScheduledTasks();
    
    // Mark as initialized atomically
    is_initialized.store(true, std::memory_order_relaxed);
    
    LOG_Orchestrator("Orchestrator initialized successfully");
    return true;
}

inline void Orchestrator::Shutdown() {
    // Early exit if already shutting down - lock-free check
    if (is_shutting_down.exchange(true, std::memory_order_relaxed)) [[unlikely]] {
        LOG_Orchestrator("Shutdown already in progress");
        return;
    }
    
    LOG_Orchestrator("Shutting down Orchestrator...");
    
    // Shutdown processing pipelines
    summarization_pipeline.Shutdown();
    emotion_analysis_pipeline.Shutdown();
    discord_channel_pipeline.Shutdown();
    
    // Clear scheduled tasks
    {
        std::unique_lock<std::shared_mutex> lock(task_mutex);
        active_task_count.store(0, std::memory_order_relaxed);
        scheduled_tasks.clear();
    }
    
    // Clear state tracking
    {
        std::unique_lock<std::shared_mutex> lock(state_mutex);
        context_states.clear();
    }
    
    // Reset atomic flags
    output_callback_registered.store(false, std::memory_order_relaxed);
    summarization_plugin_available.store(false, std::memory_order_relaxed);
    emotag_plugin_available.store(false, std::memory_order_relaxed);
    is_initialized.store(false, std::memory_order_relaxed);
    
    // Sanitizer cleanup handled by unique_ptr
    LOG_Orchestrator("Orchestrator shutdown complete");
}

inline void Orchestrator::RegisterOutputCallback(std::function<void(std::string_view, InputSource)>&& callback) {
    output_callback = std::move(callback);
    output_callback_registered.store(true, std::memory_order_relaxed);
    LOG_Orchestrator("Output callback registered");
}

inline void Orchestrator::RegisterDiscordResponseCallback(std::function<void(const DiscordChannelResponse&)>&& callback) {
    discord_response_callback = std::move(callback);
    LOG_Orchestrator("Discord response callback registered");
}

inline void Orchestrator::InputReceived(std::string_view input, std::string_view context_id, 
                                       InputSource source, std::string_view username) {
    
    // Early exit if shutting down - lock-free check
    if (is_shutting_down.load(std::memory_order_relaxed)) [[unlikely]] {
        LOG_Orchestrator("Cannot process input - orchestrator is shutting down");
        return;
    }
    
    // Check if context is available for processing (read-only operation)
    if (IsContextBusy(context_id)) [[unlikely]] {
        LOG_Orchestrator("Context " + std::string(context_id) + " is busy, queuing request");
        // In a full implementation, we'd queue the request
        return;
    }
    
    // Sanitize input based on source
    std::string sanitized_input = GetSanitizedInput(input, source);
    if (sanitized_input.empty()) [[unlikely]] {
        LOG_Orchestrator("Input blocked by sanitization");
        stats.sanitization_blocks.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    
#if __cpp_assume >= 202207L  // C++23 assume support
    // Optimization hint: sanitized_input is non-empty at this point
    [[assume(sanitized_input.size() > 0)]];
#endif
    
    // Set context to processing state
    SetContextState(context_id, ProcessingState::NORMAL_PROCESSING);
    
    SafeExecute([this, context_id, sanitized_input, source, username]() {
        // Get context size for main model - settings manager is required
        auto* settings = GetSettingsManager();
        if (!settings) [[unlikely]] {
            LOG_ERROR_Orchestrator("SettingsManager not available for context creation: " + std::string(context_id));
            SetContextState(context_id, ProcessingState::ERROR_STATE);
            return;
        }
        
#if __cpp_assume >= 202207L  // C++23 assume support
        // Optimization hint: settings is non-null at this point
        [[assume(settings != nullptr)]];
#endif
        
        // No fallback values - settings must be properly configured
        int32_t context_size = settings->GetInt("Models", "outer_context_size", 0);
        if (context_size <= 0) [[unlikely]] {
            LOG_ERROR_Orchestrator("Invalid outer_context_size configuration for context: " + std::string(context_id));
            SetContextState(context_id, ProcessingState::ERROR_STATE);
            return;
        }
        
        // Route to appropriate context
        auto* context = llama_manager->GetOrCreateContextInfo(std::string(context_id), "outer_model", context_size);
        if (!context) [[unlikely]] {
            LOG_ERROR_Orchestrator("Failed to get/create context: " + std::string(context_id));
            SetContextState(context_id, ProcessingState::ERROR_STATE);
            return;
        }
        
        // Check if context is in error state (template validation failed)
        if (context->GetState() == ContextState::ERROR_STATE) [[unlikely]] {
            LOG_ERROR_Orchestrator("Context is in error state: " + std::string(context_id));
            SetContextState(context_id, ProcessingState::ERROR_STATE);
            return;
        }
        
        // Process through context using async interface (for consistency with new architecture)
        // Note: For Discord/non-UI contexts, we can still use blocking behavior by providing simple callbacks
        GenerationCallbacks callbacks;
        callbacks.on_complete = [this, source](const std::string& response, bool success) {
            if (success && IsOutputCallbackAvailable()) {
                output_callback(response, source);
            }
        };
        callbacks.on_error = [this, context_id](const std::string& error) {
            LOG_ERROR_Orchestrator("Generation error for context " + std::string(context_id) + ": " + error);
            SetContextState(context_id, ProcessingState::ERROR_STATE);
        };
        
        // Use async interface for consistency with new architecture
        bool started = context->HandleInputAsync(sanitized_input, callbacks, std::string(username));
        if (!started) [[unlikely]] {
            LOG_ERROR_Orchestrator("Failed to start async processing for context: " + std::string(context_id));
            SetContextState(context_id, ProcessingState::ERROR_STATE);
            return;
        }
        
        // Update statistics - lock-free atomic increment
        stats.messages_processed.fetch_add(1, std::memory_order_relaxed);
        
    }, "input processing", "Orchestrator");
}

inline void Orchestrator::OnRawDiscordMessage(std::string_view content, 
                                             std::string_view channel_id, 
                                             std::string_view username) {
    
    LOG_Orchestrator("Processing Discord message from " + std::string(username) + " in " + std::string(channel_id));
    
    // Create Discord channel request with efficient string construction
    DiscordChannelRequest request;
    request.request_type = DiscordChannelRequest::Type::INCOMING_MESSAGE;
    request.channel_id = std::string(channel_id);
    request.username = std::string(username);
    request.content = std::string(content);
    request.timestamp = std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    
    // Process request with response callback
    ProcessDiscordChannelRequest(request, [this](const DiscordChannelResponse& response) {
        // Send response back to Discord if we should respond and have a callback
        if (response.should_respond && !response.response_content.empty() && discord_response_callback) {
            LOG_Orchestrator("Sending Discord response to channel " + response.target_channel + ": " + 
                           response.response_content.substr(0, 50) + "...");
            discord_response_callback(response);
        } else if (!response.should_respond) {
            LOG_Orchestrator("Discord response suppressed for channel " + response.target_channel);
        } else if (response.response_content.empty()) {
            LOG_Orchestrator("Discord response empty for channel " + response.target_channel);
        } else if (!discord_response_callback) {
            LOG_ERROR_Orchestrator("Discord response callback not registered - cannot send response to channel " + response.target_channel);
        }
        
        // Log any errors
        if (!response.error_message.empty()) {
            LOG_ERROR_Orchestrator("Discord processing error for channel " + response.target_channel + ": " + response.error_message);
        }
    });
    
    // Update statistics - lock-free atomic increment
    stats.discord_messages_handled.fetch_add(1, std::memory_order_relaxed);
}

inline void Orchestrator::ProcessScheduledTasks() {
    auto now = std::chrono::steady_clock::now();
    
    // Early exit optimization - avoid lock if no tasks have been scheduled (lock-free check)
    if (active_task_count.load(std::memory_order_relaxed) == 0) [[unlikely]] {
        return; // No tasks scheduled - unlikely after initialization
    }
    
    std::shared_lock<std::shared_mutex> lock(task_mutex);
    
    // Process only tasks that are due, with minimal work in the loop
    for (auto& task : scheduled_tasks) [[likely]] {
        if (task.active && now >= task.next_run) [[likely]] {
            try {                
                task.task_function();
                task.next_run = now + task.interval;
                
                // Update statistics using lock-free atomic increment
                stats.scheduled_tasks_executed.fetch_add(1, std::memory_order_relaxed);
                
            } catch (const std::exception& e) {
                // Exception handling is unlikely in normal operation
                LOG_ERROR_Orchestrator("Scheduled task error: " + std::string(e.what()));
            }
        }
    }
    
    last_scheduled_run = now;
}

inline void Orchestrator::AddScheduledTask(ScheduledTaskType type, std::chrono::milliseconds interval,
                                          std::function<void()>&& task_function) {
    std::unique_lock<std::shared_mutex> lock(task_mutex);
    
    ScheduledTask task{
        .type = type,
        .next_run = std::chrono::steady_clock::now() + interval,
        .interval = interval,
        .task_function = std::move(task_function),
        .active = true
    };
    
    scheduled_tasks.push_back(std::move(task));
    active_task_count.fetch_add(1, std::memory_order_relaxed);
    LOG_Orchestrator("Scheduled task added");
}

inline void Orchestrator::RemoveScheduledTask(ScheduledTaskType type) {
    std::unique_lock<std::shared_mutex> lock(task_mutex);
    
    auto initial_size = scheduled_tasks.size();
    scheduled_tasks.erase(
        std::remove_if(scheduled_tasks.begin(), scheduled_tasks.end(),
                      [type](const ScheduledTask& task) { return task.type == type; }),
        scheduled_tasks.end());
    
    // Update atomic counter if tasks were removed
    auto removed_count = initial_size - scheduled_tasks.size();
    if (removed_count > 0) [[likely]] {
        active_task_count.fetch_sub(removed_count, std::memory_order_relaxed);
    }
    
    LOG_Orchestrator("Scheduled task removed");
}

// Private implementation methods

inline void Orchestrator::InitializePipelines() {
    LOG_Orchestrator("Setting up pipeline processors...");
    
    // Setup summarization pipeline processor
    summarization_pipeline.SetProcessor(
        [this](const LuminaChat::SummarizationRequest& request, 
               std::function<void(LuminaChat::SummarizationResponse)> success_callback,
               std::function<void(const std::string&)> error_callback) {
            try {
                ProcessSummarizationRequest(request, success_callback);
            } catch (const std::exception& e) {
                // Exception handling is unlikely in normal operation
                error_callback(e.what());
            }
        });
    
    // Setup emotion analysis pipeline processor
    emotion_analysis_pipeline.SetProcessor(
        [this](const LuminaChat::EmotionAnalysisRequest& request,
               std::function<void(LuminaChat::EmotionAnalysisResponse)> success_callback,
               std::function<void(const std::string&)> error_callback) {
            LOG_Orchestrator("Pipeline processor called for emotion analysis request");
            try {
                ProcessEmotionAnalysisRequest(request, success_callback);
            } catch (const std::exception& e) {
                // Exception handling is unlikely in normal operation
                LOG_ERROR_Orchestrator("Exception in emotion analysis pipeline processor: " + std::string(e.what()));
                error_callback(e.what());
            }
        });
    
    // Setup Discord channel pipeline processor
    discord_channel_pipeline.SetProcessor(
        [this](const DiscordChannelRequest& request,
               std::function<void(DiscordChannelResponse)> success_callback,
               std::function<void(const std::string&)> error_callback) {
            try {
                ProcessDiscordChannelRequest(request, success_callback);
            } catch (const std::exception& e) {
                // Exception handling is unlikely in normal operation
                error_callback(e.what());
            }
        });
    
    LOG_Orchestrator("Pipeline processors configured, starting pipelines...");
    
    // Start all pipelines AFTER setting processors
    summarization_pipeline.Start();
    emotion_analysis_pipeline.Start();
    discord_channel_pipeline.Start();
    
    LOG_Orchestrator("Processing pipelines initialized and started");
}

inline void Orchestrator::ProcessDiscordChannelRequest(const DiscordChannelRequest& request,
                                                      std::function<void(DiscordChannelResponse)> callback) {
    LOG_Orchestrator("Processing Discord channel request for: " + request.channel_id);
    
    try {
    switch (request.request_type) {
        case DiscordChannelRequest::Type::INCOMING_MESSAGE: [[likely]] {
                // Create inner and outer context pairs for this channel
                std::string inner_context_id = "inner_context_" + request.channel_id;
                std::string outer_context_id = "outer_context_" + request.channel_id;
                
                // Get settings for context sizes
                auto* settings = GetSettingsManager();
                if (!settings) [[unlikely]] {
                    LOG_ERROR_Orchestrator("SettingsManager not available for Discord message processing: " + request.channel_id);
                    DiscordChannelResponse response;
                    response.should_respond = false;
                    response.response_content = "";
                    response.target_channel = request.channel_id;
                    response.error_message = "SettingsManager not available";
                    callback(response);
                    break;
                }
                
                // Get context sizes for inner and outer voices
                int32_t inner_context_size = settings->GetInt("Models", "inner_context_size", 4096);
                int32_t outer_context_size = settings->GetInt("Models", "outer_context_size", 8192);
                
                // Create or get the context pairs
                auto* inner_context = llama_manager->GetOrCreateContextInfo(inner_context_id, "inner_model", inner_context_size);
                auto* outer_context = llama_manager->GetOrCreateContextInfo(outer_context_id, "outer_model", outer_context_size);
                
                if (!inner_context || !outer_context) [[unlikely]] {
                    LOG_ERROR_Orchestrator("Failed to create context pairs for Discord channel: " + request.channel_id);
                    DiscordChannelResponse response;
                    response.should_respond = false;
                    response.response_content = "";
                    response.target_channel = request.channel_id;
                    response.error_message = "Failed to create context pairs";
                    callback(response);
                    break;
                }
                
                // Load and apply all template settings for Discord contexts (same as UI contexts)
                
                // 1. Environment Description - use the same rich environment as UI contexts
                std::string environment_description = settings->GetString("Templates", "environment_description", "");
                if (!environment_description.empty()) {
                    inner_context->UpdateEnvironment(environment_description);
                    outer_context->UpdateEnvironment(environment_description);
                    LOG_DEBUG_Orchestrator("Applied environment description to Discord contexts for channel: " + request.channel_id);
                } else {
                    // Fallback to basic Discord channel info if no environment description set
                    std::string fallback_environment = "Discord channel: " + request.channel_id;
                    inner_context->UpdateEnvironment(fallback_environment);
                    outer_context->UpdateEnvironment(fallback_environment);
                    LOG_WARNING_Orchestrator("No environment description configured, using fallback for Discord channel: " + request.channel_id);
                }
                
                // 2. Identity Directive - load from settings
                std::string identity = settings->GetString("Templates", "identity_directive", "");
                if (!identity.empty()) {
                    inner_context->UpdateIdentity(identity);
                    outer_context->UpdateIdentity(identity);
                    LOG_DEBUG_Orchestrator("Applied identity directive to Discord contexts for channel: " + request.channel_id);
                } else {
                    LOG_WARNING_Orchestrator("No identity directive configured for Discord channel: " + request.channel_id);
                }
                
                // 3. System Prompts - load context-specific prompts
                std::string inner_system_prompt = settings->GetString("Models", "inner_system_prompt", "");
                std::string outer_system_prompt = settings->GetString("Models", "outer_system_prompt", "");
                
                if (!inner_system_prompt.empty()) {
                    inner_context->UpdateSystemPrompt(inner_system_prompt);
                    LOG_DEBUG_Orchestrator("Applied inner system prompt to Discord context for channel: " + request.channel_id);
                } else {
                    LOG_WARNING_Orchestrator("No inner system prompt configured for Discord channel: " + request.channel_id);
                }
                
                if (!outer_system_prompt.empty()) {
                    outer_context->UpdateSystemPrompt(outer_system_prompt);
                    LOG_DEBUG_Orchestrator("Applied outer system prompt to Discord context for channel: " + request.channel_id);
                } else {
                    LOG_WARNING_Orchestrator("No outer system prompt configured for Discord channel: " + request.channel_id);
                }
                
                // Process message through two-stage reasoning
                std::string sanitized_input = GetSanitizedInput(request.content, InputSource::DISCORD);
                if (sanitized_input.empty()) [[unlikely]] {
                    LOG_Orchestrator("Discord message blocked by sanitization for channel: " + request.channel_id);
                    DiscordChannelResponse response;
                    response.should_respond = false;
                    response.response_content = "";
                    response.target_channel = request.channel_id;
                    response.error_message = "";
                    callback(response);
                    break;
                }
                
                // Execute two-stage reasoning for Discord messages
                ExecuteDiscordTwoStageReasoning(inner_context, outer_context, sanitized_input, request, callback);
                
                break;
            }
            
        case DiscordChannelRequest::Type::CHANNEL_SETUP: [[unlikely]] {
                // Get context size for main model - settings manager is required
                auto* settings = GetSettingsManager();
                if (!settings) [[unlikely]] {
                    LOG_ERROR_Orchestrator("SettingsManager not available for Discord channel setup: " + request.channel_id);
                    DiscordChannelResponse response;
                    response.should_respond = false;
                    response.response_content = "";
                    response.target_channel = request.channel_id;
                    response.error_message = "SettingsManager not available";
                    callback(response);
                    break;
                }
                
                // No fallback values - settings must be properly configured
                int32_t context_size = settings->GetInt("Models", "outer_context_size", 0);
                if (context_size <= 0) [[unlikely]] {
                    LOG_ERROR_Orchestrator("Invalid outer_context_size configuration for Discord channel: " + request.channel_id);
                    DiscordChannelResponse response;
                    response.should_respond = false;
                    response.response_content = "";
                    response.target_channel = request.channel_id;
                    response.error_message = "Invalid outer_context_size configuration";
                    callback(response);
                    break;
                }
                
                // Setup new channel context
                std::string context_id = "discord_" + request.channel_id;
                auto* context = llama_manager->GetOrCreateContextInfo(context_id, "outer_model", context_size);
                
                if (context) [[likely]] {
                    context->UpdateEnvironment("Discord channel: " + request.channel_id);
                }
                
                DiscordChannelResponse response;
                response.should_respond = false;
                response.response_content = "";
                response.target_channel = request.channel_id;
                response.error_message = "";
                callback(response);
                break;
            }
            
            case DiscordChannelRequest::Type::OUTGOING_MESSAGE: [[unlikely]] {
                // Handle outgoing message (for future implementation)
                DiscordChannelResponse response;
                response.should_respond = false;
                response.response_content = "";
                response.target_channel = request.channel_id;
                response.error_message = "";
                callback(response);
                break;
            }
            
            default: {
                // Unknown request type - should be very rare
                DiscordChannelResponse response;
                response.should_respond = false;
                response.response_content = "";
                response.target_channel = request.channel_id;
                response.error_message = "Unknown request type";
                callback(response);
                break;
            }
        }
        
    } catch (const std::exception& e) {
        // Exception handling is unlikely in normal operation
        DiscordChannelResponse response;
        response.should_respond = false;
        response.response_content = "";
        response.target_channel = request.channel_id;
        response.error_message = e.what();
        callback(response);
    }
}

inline void Orchestrator::SetupDefaultScheduledTasks() {
    // Context maintenance every 5 minutes
    AddScheduledTask(ScheduledTaskType::CONTEXT_MAINTENANCE, 
                    std::chrono::minutes(5),
                    [this]() { PerformContextMaintenance(); });
    
    // Cache cleanup every 10 minutes
    AddScheduledTask(ScheduledTaskType::CACHE_CLEANUP,
                    std::chrono::minutes(10),
                    [this]() { PerformCacheCleanup(); });
    
    // Health check every minute
    AddScheduledTask(ScheduledTaskType::HEALTH_CHECK,
                    std::chrono::minutes(1),
                    [this]() { PerformHealthCheck(); });
    
    // Discord presence update every 30 seconds
    AddScheduledTask(ScheduledTaskType::DISCORD_PRESENCE_UPDATE,
                    std::chrono::seconds(30),
                    [this]() { UpdateDiscordPresence(); });
    
    // Pruning buffer processing every 2 minutes (processes pruned messages from context size management)
    AddScheduledTask(ScheduledTaskType::PRUNING_BUFFER_PROCESSING,
                    std::chrono::minutes(2),
                    [this]() { ProcessPruningBuffer(); });
    
    // Emotion analysis processing every 2 seconds
    AddScheduledTask(ScheduledTaskType::EMOTION_ANALYSIS_PROCESSING,
                    std::chrono::seconds(2),
                    [this]() { ProcessEmotionAnalysisBuffer(); });
    
    // Context size monitoring every 10 seconds
    AddScheduledTask(ScheduledTaskType::CONTEXT_SIZE_MONITORING,
                    std::chrono::seconds(10),
                    [this]() { MonitorAllContextSizes(); });
    
    // Inner voice analysis every 5 seconds (creates direct summarization requests)
    AddScheduledTask(ScheduledTaskType::INNER_VOICE_ANALYSIS,
                    std::chrono::seconds(5),
                    [this]() { ProcessInnerVoiceAnalysis(); });
    
    // Summarization pipeline health monitoring every 10 seconds (monitors pipeline health only)
    AddScheduledTask(ScheduledTaskType::SUMMARIZATION_PROCESSING,
                    std::chrono::seconds(10),
                    [this]() { ProcessSummarizationPipeline(); });
    
    LOG_Orchestrator("Default scheduled tasks configured");
}

inline [[nodiscard]] std::string Orchestrator::GetSanitizedInput(std::string_view input, InputSource source) {
    // Apply sanitization based on source - optimize for most common paths using constexpr utilities
    if constexpr (true) {  // Enable compile-time branch optimization
        if (IsSystemSource(source)) [[unlikely]] {
            // System inputs are always trusted - most predictable path
            return std::string{input};
        }
    }
    
    switch (source) {
        case InputSource::DISCORD: [[likely]] {
            // Always sanitize Discord input - create string only when sanitization needed
            std::string sanitized{input};  // Efficient construction from string_view
            if (!sanitizer->SanitizeInput(sanitized)) [[unlikely]] {
                return ""; // Input was blocked
            }
            return sanitized;
        }
        case InputSource::UI: [[likely]] {
            // UI input is often safe - direct conversion is common case
            if (input.size() <= OrchestratorConstants::SMALL_STRING_OPTIMIZATION_THRESHOLD) [[likely]] {
                // Take advantage of SSO for small strings
                return std::string{input};
            }
            return std::string{input}; // Direct conversion - no sanitization needed
        }
        case InputSource::SYSTEM: [[unlikely]]
        case InputSource::SCHEDULED_TASK: [[unlikely]] {
            // System inputs are always trusted - most predictable path
            return std::string{input};
        }
    }
    
    // Default case - should never reach here, but handle gracefully
    return std::string{input};
}

inline [[nodiscard]] bool Orchestrator::IsContextBusy(std::string_view context_id) const noexcept {
    std::shared_lock<std::shared_mutex> lock(state_mutex);
    
    // Optimize: avoid string allocation for lookup when using heterogeneous lookup
    // For unordered_map, we need the string conversion but can optimize with string_view key type
    auto it = context_states.find(std::string(context_id));
    if (it == context_states.end()) [[likely]] {
        return false; // Context doesn't exist yet, not busy - most common case
    }
    
    // Use constexpr utility for better optimization
    return IsBusyState(it->second);
}

inline void Orchestrator::SetContextState(std::string_view context_id, ProcessingState state) {
    std::lock_guard<std::shared_mutex> lock(state_mutex);
    
    // Optimize: use emplace_or_assign for potentially better performance
    context_states[std::string(context_id)] = state;
}

inline [[nodiscard]] ProcessingState Orchestrator::GetContextState(std::string_view context_id) const noexcept {
    std::shared_lock<std::shared_mutex> lock(state_mutex);
    
    // Optimize: avoid string allocation for lookup when possible
    auto it = context_states.find(std::string(context_id));
    if (it != context_states.end()) [[likely]] {
        return it->second;
    } else [[unlikely]] {
        return ProcessingState::NORMAL_PROCESSING;
    }
}

inline [[nodiscard]] Orchestrator::OrchestrationStats Orchestrator::GetStats() const {
    // Use lock-free atomic loads for maximum performance
    OrchestrationStats snapshot;
    snapshot.messages_processed.store(stats.messages_processed.load(std::memory_order_relaxed), std::memory_order_relaxed);
    snapshot.summarizations_completed.store(stats.summarizations_completed.load(std::memory_order_relaxed), std::memory_order_relaxed);
    snapshot.emotion_analyses_completed.store(stats.emotion_analyses_completed.load(std::memory_order_relaxed), std::memory_order_relaxed);
    snapshot.discord_messages_handled.store(stats.discord_messages_handled.load(std::memory_order_relaxed), std::memory_order_relaxed);
    snapshot.scheduled_tasks_executed.store(stats.scheduled_tasks_executed.load(std::memory_order_relaxed), std::memory_order_relaxed);
    snapshot.sanitization_blocks.store(stats.sanitization_blocks.load(std::memory_order_relaxed), std::memory_order_relaxed);
    snapshot.start_time = stats.start_time;
    return snapshot;
}

inline void Orchestrator::ResetStats() noexcept {
    // Lock-free atomic reset for maximum performance
    stats.messages_processed.store(0, std::memory_order_relaxed);
    stats.summarizations_completed.store(0, std::memory_order_relaxed);
    stats.emotion_analyses_completed.store(0, std::memory_order_relaxed);
    stats.discord_messages_handled.store(0, std::memory_order_relaxed);
    stats.scheduled_tasks_executed.store(0, std::memory_order_relaxed);
    stats.sanitization_blocks.store(0, std::memory_order_relaxed);
    stats.start_time = std::chrono::steady_clock::now();
}

// Scheduled task implementations
inline void Orchestrator::PerformContextMaintenance() {
    LOG_Orchestrator("Performing context maintenance...");
    
    // Check for contexts that need pruning
    // This would integrate with ContextPruningPlugin in a full implementation
    
    // For now, just log that maintenance was performed
    LOG_Orchestrator("Context maintenance complete");
}

inline void Orchestrator::PerformCacheCleanup() {
    LOG_Orchestrator("Performing cache cleanup...");
    
    // This would clean up old cache entries
    // Integration with TokenCache cleanup methods
    
    LOG_Orchestrator("Cache cleanup complete");
}

inline void Orchestrator::PerformHealthCheck() {
    
    // Check system health, memory usage, etc.
    //auto stats_snapshot = GetStats();
    //LOG_Orchestrator("Health check - Messages processed: " + std::to_string(stats_snapshot.messages_processed.load(std::memory_order_relaxed)) +
    //                ", Summarizations: " + std::to_string(stats_snapshot.summarizations_completed.load(std::memory_order_relaxed)));
}

inline void Orchestrator::UpdateDiscordPresence() {
    // Update Discord bot presence/status
    // This would integrate with Discord API
}

// CRITICAL: Some method definitions are located at the BOTTOM of individual plugin files to combat circular dependencies.

inline void Orchestrator::ExecuteDiscordTwoStageReasoning(ContextInfo* inner_context, ContextInfo* outer_context,
                                                         const std::string& input, const DiscordChannelRequest& request,
                                                         std::function<void(DiscordChannelResponse)> callback) {
    LOG_Orchestrator("Starting two-stage reasoning for Discord channel: " + request.channel_id);
    
    // Start timing for performance tracking
    auto start_time = std::chrono::steady_clock::now();
    
    try {
        // CRITICAL: Only synchronize contexts if this is the very first message in the conversation
        // After that, let contexts diverge naturally (inner context will summarize, outer context keeps full history)
        LOG_Orchestrator("Checking if chat history synchronization is needed for channel: " + request.channel_id);
        const auto& outer_history = outer_context->GetMessageHistory();
        const auto& inner_history = inner_context->GetMessageHistory();
        
        // Only sync if BOTH contexts are empty (first message in conversation)
        // This prevents undoing summarization work done by the inner context
        if (outer_history.empty() && inner_history.empty()) {
            LOG_Orchestrator("First message in conversation for channel " + request.channel_id + " - contexts are already synchronized (both empty)");
        } else if (inner_history.empty() && !outer_history.empty()) {
            // Inner context is empty but outer has history - this means inner context was reset or is new
            // Copy outer history to inner context for initial sync
            for (const auto& [role, content] : outer_history) {
                inner_context->AddHistoricalMessage(role, content);
            }
            LOG_Orchestrator("Initial sync: Copied " + std::to_string(outer_history.size()) + 
                            " messages from outer to inner context for channel " + request.channel_id);
        } else {
            // Both contexts have history - let them remain independent
            // Inner context may have fewer messages due to summarization, and that's intentional
            LOG_Orchestrator("Contexts diverged naturally for channel " + request.channel_id + 
                            " (outer: " + std::to_string(outer_history.size()) + 
                            " messages, inner: " + std::to_string(inner_history.size()) + 
                            " messages) - preserving independent histories");
        }
        
        // Stage 1: Generate inner voice reasoning
        LOG_Orchestrator("Stage 1: Inner voice reasoning for Discord channel: " + request.channel_id);
        
        // Create callbacks for inner voice (Stage 1)
        GenerationCallbacks inner_callbacks;
        
        // Completion callback for inner voice
        inner_callbacks.on_complete = [this, outer_context, input, request, callback, start_time](const std::string& inner_response, bool success) {
            if (success && !inner_response.empty()) {
                LOG_Orchestrator("Stage 1 Complete: Inner voice generated reasoning for channel " + request.channel_id +
                               " (length: " + std::to_string(inner_response.length()) + " characters)");
                
                // Stage 2: Update outer voice with reasoning thoughts
                LOG_Orchestrator("Stage 2: Updating outer voice with internal reflection for channel: " + request.channel_id);
                try {
                    outer_context->UpdateInternalReflection(inner_response);
                    LOG_Orchestrator("Stage 2 Complete: Internal reflection updated in outer voice for channel: " + request.channel_id);
                    
                    // Stage 3: Generate final response with outer voice
                    LOG_Orchestrator("Stage 3: Generating final response with outer voice for channel: " + request.channel_id);
                    
                    // Create callbacks for outer voice final response
                    GenerationCallbacks outer_callbacks;
                    outer_callbacks.on_complete = [this, request, callback, inner_response, outer_context, start_time](const std::string& response, bool success) {
                        // Calculate processing time
                        auto end_time = std::chrono::steady_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                        std::string processing_time = std::to_string(duration.count()) + "ms";
                        
                        DiscordChannelResponse discord_response;
                        discord_response.should_respond = success;
                        discord_response.response_content = response;
                        discord_response.internal_reasoning = inner_response;  // Capture inner voice reasoning
                        discord_response.target_channel = request.channel_id;
                        discord_response.error_message = success ? "" : "Generation failed";
                        
                        // Add context usage information
                        if (outer_context) {
                            discord_response.context_current = outer_context->GetActualContextTokens();
                            discord_response.context_maximum = outer_context->GetMaxContextTokens();
                        }
                        
                        // Add model and timing information
                        discord_response.model_name = "LuminaChat AI";  // Could be enhanced to show actual model names
                        discord_response.processing_time = processing_time;
                        
                        LOG_Orchestrator("Two-stage reasoning complete for Discord channel " + request.channel_id + 
                                       " (success: " + (success ? "true" : "false") + ", time: " + processing_time + ")");
                        callback(discord_response);
                    };
                    
                    outer_callbacks.on_error = [this, request, callback](const std::string& error) {
                        LOG_ERROR_Orchestrator("Stage 3 Error for Discord channel " + request.channel_id + ": " + error);
                        DiscordChannelResponse discord_response;
                        discord_response.should_respond = false;
                        discord_response.response_content = "";
                        discord_response.target_channel = request.channel_id;
                        discord_response.error_message = "Stage 3 error: " + error;
                        callback(discord_response);
                    };
                    
                    // Generate final response with outer voice
                    bool started = outer_context->HandleInputAsync(input, outer_callbacks, request.username);
                    if (!started) [[unlikely]] {
                        LOG_ERROR_Orchestrator("Failed to start outer voice generation for Discord channel: " + request.channel_id);
                        DiscordChannelResponse response;
                        response.should_respond = false;
                        response.response_content = "";
                        response.target_channel = request.channel_id;
                        response.error_message = "Failed to start Stage 3 processing";
                        callback(response);
                    }
                    
                } catch (const std::exception& e) {
                    LOG_ERROR_Orchestrator("Stage 2 Error for Discord channel " + request.channel_id + ": " + e.what());
                    DiscordChannelResponse response;
                    response.should_respond = false;
                    response.response_content = "";
                    response.target_channel = request.channel_id;
                    response.error_message = "Stage 2 error: " + std::string(e.what());
                    callback(response);
                }
                
            } else {
                // Inner voice failed
                std::string error_msg = success ? "Inner voice generated empty response" : "Inner voice generation failed";
                LOG_ERROR_Orchestrator("Stage 1 Failed for Discord channel " + request.channel_id + ": " + error_msg);
                DiscordChannelResponse response;
                response.should_respond = false;
                response.response_content = "";
                response.target_channel = request.channel_id;
                response.error_message = "Stage 1 failed: " + error_msg;
                callback(response);
            }
        };
        
        // Error callback for inner voice
        inner_callbacks.on_error = [this, request, callback](const std::string& error_message) {
            LOG_ERROR_Orchestrator("Stage 1 Error for Discord channel " + request.channel_id + ": " + error_message);
            DiscordChannelResponse response;
            response.should_respond = false;
            response.response_content = "";
            response.target_channel = request.channel_id;
            response.error_message = "Stage 1 error: " + error_message;
            callback(response);
        };
        
        // Start inner voice reasoning (Stage 1)
        bool started = inner_context->HandleInputAsync(input, inner_callbacks, request.username);
        if (!started) [[unlikely]] {
            LOG_ERROR_Orchestrator("Failed to start inner voice reasoning for Discord channel: " + request.channel_id);
            DiscordChannelResponse response;
            response.should_respond = false;
            response.response_content = "";
            response.target_channel = request.channel_id;
            response.error_message = "Failed to start two-stage reasoning";
            callback(response);
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR_Orchestrator("Exception in two-stage reasoning for Discord channel " + request.channel_id + ": " + e.what());
        DiscordChannelResponse response;
        response.should_respond = false;
        response.response_content = "";
        response.target_channel = request.channel_id;
        response.error_message = "Two-stage reasoning error: " + std::string(e.what());
        callback(response);
    }
}
