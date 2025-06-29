#pragma once

#include <memory>
#include <string>
#include <functional>
#include <unordered_map>
#include <chrono>
#include <queue>
#include <mutex>
#include <atomic>

#include "ProcessingPipeline.hpp"
#include "Sanitizer.hpp"
#include "LlamaManager.hpp"
#include "ContextInfo.hpp"
#include "Logger.hpp"

// Forward declarations
namespace LuminaChat {
    class SummarizationPlugin;
    class EmoTagPlugin;
    enum class RequestPriority : uint8_t;
}

// Use plugin structures directly instead of duplicating them
namespace LuminaChat {
    struct SummarizationRequest;
    struct SummarizationResponse;
    struct EmotionAnalysisRequest;
    struct EmotionAnalysisResponse;
}

// Forward declarations for ContextInfo types
struct PrunedMessageBatch;
struct EmotionalAnalysisBatch;
enum class ContextState;

struct DiscordChannelRequest {
    enum class Type { INCOMING_MESSAGE, OUTGOING_MESSAGE, CHANNEL_SETUP };
    Type request_type;
    std::string channel_id;
    std::string username;
    std::string content;
    std::string timestamp;
};

struct DiscordChannelResponse {
    bool should_respond = true;
    std::string response_content;
    std::string target_channel;
    std::string error_message;
};

// Input source enumeration
enum class InputSource {
    UI,
    DISCORD,
    SYSTEM,
    SCHEDULED_TASK
};

// Processing state tracking
enum class ProcessingState {
    NORMAL_PROCESSING,
    AWAITING_SUMMARIZATION,
    BACKFILL_IN_PROGRESS,
    ERROR_STATE
};

// Scheduled task types
enum class ScheduledTaskType {
    CONTEXT_MAINTENANCE,
    CACHE_CLEANUP,
    HEALTH_CHECK,
    DISCORD_PRESENCE_UPDATE,
    PRUNING_BUFFER_PROCESSING,  // New scheduled task for processing pruning buffer
    EMOTION_ANALYSIS_PROCESSING  // New scheduled task for processing emotion analysis buffer
};

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
 * through a consistent Request → Process → Callback pattern.
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
    
    // Plugin references for delegation
    LuminaChat::SummarizationPlugin* summarization_plugin = nullptr;
    LuminaChat::EmoTagPlugin* emotag_plugin = nullptr;
    
    // Processing pipelines for plugin architecture
    LuminaChat::ProcessingPipeline<LuminaChat::SummarizationRequest, LuminaChat::SummarizationResponse> summarization_pipeline;
    LuminaChat::ProcessingPipeline<LuminaChat::EmotionAnalysisRequest, LuminaChat::EmotionAnalysisResponse> emotion_analysis_pipeline;
    LuminaChat::ProcessingPipeline<DiscordChannelRequest, DiscordChannelResponse> discord_channel_pipeline;
    
    // State tracking for contexts
    std::unordered_map<std::string, ProcessingState> context_states;
    mutable std::mutex state_mutex;
    
    // Scheduled task management
    std::vector<ScheduledTask> scheduled_tasks;
    std::chrono::steady_clock::time_point last_scheduled_run;
    mutable std::mutex task_mutex;
    
    // Callback registrations
    std::function<void(std::string_view, InputSource)> output_callback;
    
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
    
    // Internal helpers
    std::string GetSanitizedInput(const std::string& input, InputSource source);
    bool IsContextBusy(const std::string& context_id) const;
    void SetContextState(const std::string& context_id, ProcessingState state);
    ProcessingState GetContextState(const std::string& context_id) const;
    
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
    bool Initialize();
    void Shutdown();
    
    // Callback registration (called by higher-level components)
    void RegisterOutputCallback(std::function<void(std::string_view, InputSource)> callback);
    
    // Core message routing
    void InputReceived(const std::string& input, const std::string& context_id, 
                      InputSource source, const std::string& username = "User");
    
    // Discord integration
    void OnRawDiscordMessage(const std::string& content, const std::string& channel_id, 
                           const std::string& username);
    
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
                         std::function<void()> task_function);
    void RemoveScheduledTask(ScheduledTaskType type);
    void ProcessScheduledTasks();
    
    // Manual processing triggers (public for immediate processing)
    void ProcessPruningBuffer();  // New method for processing pruning buffer
    void ProcessEmotionAnalysisBuffer();  // New method for processing emotion analysis buffer
    
    // State queries
    bool IsContextAvailable(const std::string& context_id) const;
    ProcessingState GetCurrentState(const std::string& context_id) const;
    
    // Sanitizer integration
    Sanitizer* GetSanitizer() { return sanitizer.get(); }
    const Sanitizer* GetSanitizer() const { return sanitizer.get(); }
    
    // Core component access for plugins
    LlamaManager* GetLlamaManager() { return llama_manager; }
    const LlamaManager* GetLlamaManager() const { return llama_manager; }
    
    // Plugin registration
    void RegisterSummarizationPlugin(LuminaChat::SummarizationPlugin* plugin) { summarization_plugin = plugin; }
    void RegisterEmoTagPlugin(LuminaChat::EmoTagPlugin* plugin) { emotag_plugin = plugin; }
    
    // Get SettingsManager from LlamaManager (convenience method)
    SettingsManager* GetSettingsManager() { 
        return llama_manager ? llama_manager->GetSettingsManager() : nullptr; 
    }
    const SettingsManager* GetSettingsManager() const { 
        return llama_manager ? llama_manager->GetSettingsManager() : nullptr; 
    }
    
    // Statistics and monitoring
    struct OrchestrationStats {
        size_t messages_processed = 0;
        size_t summarizations_completed = 0;
        size_t emotion_analyses_completed = 0;
        size_t discord_messages_handled = 0;
        size_t scheduled_tasks_executed = 0;
        size_t sanitization_blocks = 0;
        std::chrono::steady_clock::time_point start_time;
    };
    
    OrchestrationStats GetStats() const;
    void ResetStats();

private:
    OrchestrationStats stats;
    mutable std::mutex stats_mutex;
    
    // Internal pipeline processors
    void InitializePipelines();
    void SetupDefaultScheduledTasks();
    
    // Context maintenance tasks
    void PerformContextMaintenance();
    void PerformCacheCleanup();
    void PerformHealthCheck();
    void UpdateDiscordPresence();
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
}

inline Orchestrator::~Orchestrator() {
    Shutdown();
}

inline bool Orchestrator::Initialize() {
    LOG_Orchestrator("Initializing Orchestrator...");
    
    // Sanitizer is initialized in constructor - no additional initialization needed
    LOG_Orchestrator("Sanitizer ready");
    
    // Initialize processing pipelines
    InitializePipelines();
    
    // Setup default scheduled tasks
    SetupDefaultScheduledTasks();
    
    LOG_Orchestrator("Orchestrator initialized successfully");
    return true;
}

inline void Orchestrator::Shutdown() {
    LOG_Orchestrator("Shutting down Orchestrator...");
    
    // Shutdown processing pipelines
    summarization_pipeline.Shutdown();
    emotion_analysis_pipeline.Shutdown();
    discord_channel_pipeline.Shutdown();
    
    // Clear scheduled tasks
    {
        std::lock_guard<std::mutex> lock(task_mutex);
        scheduled_tasks.clear();
    }
    
    // Clear state tracking
    {
        std::lock_guard<std::mutex> lock(state_mutex);
        context_states.clear();
    }
    
    // Sanitizer cleanup handled by unique_ptr
    LOG_Orchestrator("Orchestrator shutdown complete");
}

inline void Orchestrator::RegisterOutputCallback(std::function<void(std::string_view, InputSource)> callback) {
    output_callback = std::move(callback);
    LOG_Orchestrator("Output callback registered");
}

inline void Orchestrator::InputReceived(const std::string& input, const std::string& context_id, 
                                       InputSource source, const std::string& username) {
    
    // Check if context is available for processing
    if (IsContextBusy(context_id)) {
        LOG_Orchestrator("Context " + context_id + " is busy, queuing request");
        // In a full implementation, we'd queue the request
        return;
    }
    
    // Sanitize input based on source
    std::string sanitized_input = GetSanitizedInput(input, source);
    if (sanitized_input.empty()) {
        LOG_Orchestrator("Input blocked by sanitization");
        std::lock_guard<std::mutex> lock(stats_mutex);
        stats.sanitization_blocks++;
        return;
    }
    
    // Set context to processing state
    SetContextState(context_id, ProcessingState::NORMAL_PROCESSING);
    
    try {
        // Get context size for main model - settings manager is required
        auto* settings = GetSettingsManager();
        if (!settings) {
            LOG_ERROR_Orchestrator("SettingsManager not available for context creation: " + context_id);
            SetContextState(context_id, ProcessingState::ERROR_STATE);
            return;
        }
        
        // No fallback values - settings must be properly configured
        int32_t context_size = settings->GetInt("Models", "main_context_size", 0);
        if (context_size <= 0) {
            LOG_ERROR_Orchestrator("Invalid main_context_size configuration for context: " + context_id);
            SetContextState(context_id, ProcessingState::ERROR_STATE);
            return;
        }
        
        // Route to appropriate context
        auto* context = llama_manager->GetOrCreateContextInfo(context_id, "main_model", context_size);
        if (!context) {
            LOG_ERROR_Orchestrator("Failed to get/create context: " + context_id);
            SetContextState(context_id, ProcessingState::ERROR_STATE);
            return;
        }
        
        // Check if context is in error state (template validation failed)
        if (context->GetState() == ContextState::ERROR_STATE) {
            LOG_ERROR_Orchestrator("Context is in error state: " + context_id);
            SetContextState(context_id, ProcessingState::ERROR_STATE);
            return;
        }
        
        // Process through context
        std::string response = context->HandleInput(sanitized_input, username);
        
        // Send response back through callback
        if (output_callback) {
            output_callback(response, source);
        }
        
        // Update statistics
        {
            std::lock_guard<std::mutex> lock(stats_mutex);
            stats.messages_processed++;
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR_Orchestrator("Error processing input: " + std::string(e.what()));
        SetContextState(context_id, ProcessingState::ERROR_STATE);
    }
}

inline void Orchestrator::OnRawDiscordMessage(const std::string& content, 
                                             const std::string& channel_id, 
                                             const std::string& username) {
    
    LOG_Orchestrator("Processing Discord message from " + username + " in " + channel_id);
    
    // Create Discord channel request
    DiscordChannelRequest request{
        .request_type = DiscordChannelRequest::Type::INCOMING_MESSAGE,
        .channel_id = channel_id,
        .username = username,
        .content = content,
        .timestamp = std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count())
    };
    
    // Queue for processing through Discord channel pipeline
    discord_channel_pipeline.QueueRequest(request);
    
    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex);
        stats.discord_messages_handled++;
    }
}

inline void Orchestrator::ProcessScheduledTasks() {
    auto now = std::chrono::steady_clock::now();
    
    std::lock_guard<std::mutex> lock(task_mutex);
    
    for (auto& task : scheduled_tasks) {
        if (task.active && now >= task.next_run) {
            try {
                // Add debug logging to identify which task is running
                if (task.type == ScheduledTaskType::EMOTION_ANALYSIS_PROCESSING) {
                    LOG_Orchestrator("Executing emotion analysis processing scheduled task");
                }
                
                task.task_function();
                task.next_run = now + task.interval;
                
                // Update statistics
                {
                    std::lock_guard<std::mutex> stats_lock(stats_mutex);
                    stats.scheduled_tasks_executed++;
                }
                
            } catch (const std::exception& e) {
                LOG_ERROR_Orchestrator("Scheduled task error: " + std::string(e.what()));
            }
        }
    }
    
    last_scheduled_run = now;
}

inline void Orchestrator::AddScheduledTask(ScheduledTaskType type, std::chrono::milliseconds interval,
                                          std::function<void()> task_function) {
    std::lock_guard<std::mutex> lock(task_mutex);
    
    ScheduledTask task{
        .type = type,
        .next_run = std::chrono::steady_clock::now() + interval,
        .interval = interval,
        .task_function = std::move(task_function),
        .active = true
    };
    
    scheduled_tasks.push_back(std::move(task));
    LOG_Orchestrator("Scheduled task added");
}

inline void Orchestrator::RemoveScheduledTask(ScheduledTaskType type) {
    std::lock_guard<std::mutex> lock(task_mutex);
    
    scheduled_tasks.erase(
        std::remove_if(scheduled_tasks.begin(), scheduled_tasks.end(),
                      [type](const ScheduledTask& task) { return task.type == type; }),
        scheduled_tasks.end());
    
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
        case DiscordChannelRequest::Type::INCOMING_MESSAGE: {
                // Map channel to context (simple 1:1 mapping for now)
                std::string context_id = "discord_" + request.channel_id;
                
                // Process message through main workflow
                InputReceived(request.content, context_id, InputSource::DISCORD, request.username);
                
                // For now, assume response should be sent
                callback(DiscordChannelResponse{
                    .should_respond = true,
                    .response_content = "", // Response handled by output callback
                    .target_channel = request.channel_id,
                    .error_message = ""
                });
                break;
            }
            
            case DiscordChannelRequest::Type::CHANNEL_SETUP: {
                // Get context size for main model - settings manager is required
                auto* settings = GetSettingsManager();
                if (!settings) {
                    LOG_ERROR_Orchestrator("SettingsManager not available for Discord channel setup: " + request.channel_id);
                    callback(DiscordChannelResponse{
                        .should_respond = false,
                        .response_content = "",
                        .target_channel = request.channel_id,
                        .error_message = "SettingsManager not available"
                    });
                    break;
                }
                
                // No fallback values - settings must be properly configured
                int32_t context_size = settings->GetInt("Models", "main_context_size", 0);
                if (context_size <= 0) {
                    LOG_ERROR_Orchestrator("Invalid main_context_size configuration for Discord channel: " + request.channel_id);
                    callback(DiscordChannelResponse{
                        .should_respond = false,
                        .response_content = "",
                        .target_channel = request.channel_id,
                        .error_message = "Invalid main_context_size configuration"
                    });
                    break;
                }
                
                // Setup new channel context
                std::string context_id = "discord_" + request.channel_id;
                auto* context = llama_manager->GetOrCreateContextInfo(context_id, "main_model", context_size);
                
                if (context) {
                    context->UpdateEnvironment("Discord channel: " + request.channel_id);
                }
                
                callback(DiscordChannelResponse{
                    .should_respond = false,
                    .response_content = "",
                    .target_channel = request.channel_id,
                    .error_message = ""
                });
                break;
            }
            
            case DiscordChannelRequest::Type::OUTGOING_MESSAGE: {
                // Handle outgoing message (for future implementation)
                callback(DiscordChannelResponse{
                    .should_respond = false,
                    .response_content = "",
                    .target_channel = request.channel_id,
                    .error_message = ""
                });
                break;
            }
            
            default:
                callback(DiscordChannelResponse{
                    .should_respond = false,
                    .response_content = "",
                    .target_channel = request.channel_id,
                    .error_message = "Unknown request type"
                });
                break;
        }
        
    } catch (const std::exception& e) {
        callback(DiscordChannelResponse{
            .should_respond = false,
            .response_content = "",
            .target_channel = request.channel_id,
            .error_message = e.what()
        });
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
    
    // Pruning buffer processing every 2 minutes
    AddScheduledTask(ScheduledTaskType::PRUNING_BUFFER_PROCESSING,
                    std::chrono::minutes(2),
                    [this]() { ProcessPruningBuffer(); });
    
    // Emotion analysis processing every 2 seconds
    AddScheduledTask(ScheduledTaskType::EMOTION_ANALYSIS_PROCESSING,
                    std::chrono::seconds(2),
                    [this]() { ProcessEmotionAnalysisBuffer(); });
    
    LOG_Orchestrator("Default scheduled tasks configured");
}

inline std::string Orchestrator::GetSanitizedInput(const std::string& input, InputSource source) {
    std::string sanitized = input;
    
    // Apply sanitization based on source
    switch (source) {
        case InputSource::DISCORD:
            // Always sanitize Discord input
            if (!sanitizer->SanitizeInput(sanitized)) {
                return ""; // Input was blocked
            }
            break;
            
        case InputSource::UI:
            // Optional sanitization for UI input (could be configurable)
            break;
            
        case InputSource::SYSTEM:
        case InputSource::SCHEDULED_TASK:
            // No sanitization for system/scheduled inputs
            break;
    }
    
    return sanitized;
}

inline bool Orchestrator::IsContextBusy(const std::string& context_id) const {
    std::lock_guard<std::mutex> lock(state_mutex);
    auto it = context_states.find(context_id);
    if (it == context_states.end()) {
        return false; // Context doesn't exist yet, not busy
    }
    
    return it->second != ProcessingState::NORMAL_PROCESSING;
}

inline void Orchestrator::SetContextState(const std::string& context_id, ProcessingState state) {
    std::lock_guard<std::mutex> lock(state_mutex);
    context_states[context_id] = state;
}

inline ProcessingState Orchestrator::GetContextState(const std::string& context_id) const {
    std::lock_guard<std::mutex> lock(state_mutex);
    auto it = context_states.find(context_id);
    return (it != context_states.end()) ? it->second : ProcessingState::NORMAL_PROCESSING;
}

inline bool Orchestrator::IsContextAvailable(const std::string& context_id) const {
    return !IsContextBusy(context_id);
}

inline ProcessingState Orchestrator::GetCurrentState(const std::string& context_id) const {
    return GetContextState(context_id);
}

inline Orchestrator::OrchestrationStats Orchestrator::GetStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex);
    return stats;
}

inline void Orchestrator::ResetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex);
    stats = OrchestrationStats{};
    stats.start_time = std::chrono::steady_clock::now();
}

// Scheduled task implementations
inline void Orchestrator::PerformContextMaintenance() {
    LOG_Orchestrator("Performing context maintenance...");
    
    // Check for contexts that need pruning
    // This would integrate with ContextSizeManager in a full implementation
    
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
    LOG_Orchestrator("Performing health check...");
    
    // Check system health, memory usage, etc.
    auto stats = GetStats();
    LOG_Orchestrator("Health check - Messages processed: " + std::to_string(stats.messages_processed) +
                    ", Summarizations: " + std::to_string(stats.summarizations_completed));
}

inline void Orchestrator::UpdateDiscordPresence() {
    // Update Discord bot presence/status
    // This would integrate with Discord API
    LOG_Orchestrator("Discord presence updated");
}

inline void Orchestrator::ProcessPruningBuffer() {
    // Check if there are any pruned messages waiting for summarization
    if (!ContextInfo::HasPendingSummarization()) {
        return; // No work to do
    }
    
    LOG_Orchestrator("Processing pruning buffer...");
    
    // Get all pending pruning batches
    /*auto pruning_batches = ContextInfo::GetAndClearPruningBuffer();
    
    LOG_Orchestrator("Found " + std::to_string(pruning_batches.size()) + " pruning batches to process");
    
    // Process each batch through the summarization pipeline
    for (const auto& batch : pruning_batches) {
        if (!batch.needs_summarization) {
            continue; // Skip batches that don't need summarization
        }
        
        // Create summarization request using plugin structure
        LuminaChat::SummarizationRequest request(batch, batch.context_id);
        
        // Queue for pipeline processing
        bool queued = summarization_pipeline.QueueRequest(request, LuminaChat::RequestPriority::NORMAL, 
                                                         "pruning_buffer_" + batch.context_id);
        
        if (queued) {
            LOG_Orchestrator("Queued summarization for context: " + batch.context_id + 
                           " (messages: " + std::to_string(batch.pruned_messages.size()) + ")");
        } else {
            LOG_ERROR_Orchestrator("Failed to queue summarization for context: " + batch.context_id);
        }
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex);
            stats.summarizations_completed++;
        }
    }*/
    
    //LOG_Orchestrator("Pruning buffer processing complete - processed " + 
    //                std::to_string(pruning_batches.size()) + " batches");
}

inline void Orchestrator::ProcessEmotionAnalysisBuffer() {
    LOG_Orchestrator("ProcessEmotionAnalysisBuffer() called - checking for pending analysis...");
    
    // Check if there are any AI responses waiting for emotional analysis
    if (!ContextInfo::HasPendingEmotionalAnalysis()) {
        LOG_Orchestrator("No pending emotional analysis found");
        return; // No work to do
    }
    
    LOG_Orchestrator("Processing emotion analysis buffer...");
    
    // Get all pending emotional analysis batches
    auto analysis_batches = ContextInfo::GetAndClearEmotionalAnalysisBuffer();
    
    if (!analysis_batches.empty()) {
        LOG_Orchestrator("Found " + std::to_string(analysis_batches.size()) + " emotion analysis batches to process");
        
        // Process each batch through the emotion analysis pipeline
        for (const auto& batch : analysis_batches) {
            RequestEmotionAnalysis(batch, LuminaChat::RequestPriority::NORMAL);
        }
        
        LOG_Orchestrator("Emotion analysis buffer processing complete");
    }
}
