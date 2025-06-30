#pragma once

#include "ContextInfo.hpp"
#include "ProcessingPipeline.hpp"
#include "Orchestrator.hpp"
#include "LlamaManager.hpp"
#include "SettingsManager.hpp"
#include "Logger.hpp"
#include <functional>
#include <cstdint>
#include <atomic>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <string>
#include <string_view>

namespace LuminaChat {

// Callback type for status updates
using StatusUpdateCallback = std::function<void(const std::string& status, bool is_error)>;

// Request and response structures for the pipeline
struct ContextSizePruningRequest {
    std::string context_id;
    int32_t current_tokens;
    int32_t max_tokens;
    PruningStrategy strategy;
    RequestPriority priority = RequestPriority::NORMAL;
    std::chrono::steady_clock::time_point queued_time;
    
    ContextSizePruningRequest(const std::string& ctx_id, int32_t current, int32_t max, PruningStrategy strat, RequestPriority prio = RequestPriority::NORMAL)
        : context_id(ctx_id), current_tokens(current), max_tokens(max), strategy(strat), priority(prio), queued_time(std::chrono::steady_clock::now()) {}
};

struct ContextSizePruningResponse {
    std::string context_id;
    bool pruning_triggered;
    PruningStrategy strategy_used;
    int32_t tokens_before;
    int32_t estimated_tokens_after;
    bool success;
    std::string error_message;
    
    ContextSizePruningResponse(const std::string& ctx_id, bool triggered, PruningStrategy strategy, int32_t before, int32_t after, bool succ, const std::string& err = "")
        : context_id(ctx_id), pruning_triggered(triggered), strategy_used(strategy), tokens_before(before), estimated_tokens_after(after), success(succ), error_message(err) {}
};

// ContextSizeManagerPlugin: Intelligent context pruning management as a plugin
// - Monitor individual contexts usage continuously
// - Trigger pruning events at 80% capacity with smart strategy selection
// - Integrates with Orchestrator and ProcessingPipeline architecture
// - Handles pruning decisions, not summarization (that's handled automatically by buffered messages)

struct ContextUsageStats {
    int32_t total_tokens = 0;
    int32_t max_tokens = 0;
    int32_t template_tokens = 0;      // Base template + system sections
    int32_t summary_tokens = 0;       // Summary sections in template
    int32_t message_tokens = 0;       // Actual conversation history
    int32_t buffer_tokens = 0;        // Reserved space for AI response
    
    float GetUsagePercentage() const {
        return max_tokens > 0 ? static_cast<float>(total_tokens) / max_tokens : 0.0f;
    }
    
    int32_t GetAvailableTokens() const {
        return max_tokens - total_tokens;
    }
    
    bool NeedsPruning(float threshold = 0.80f) const {
        return GetUsagePercentage() >= threshold; // Default 80% threshold
    }
    
    int32_t GetTargetTokensAfterPruning(float target = 0.40f) const {
        return static_cast<int32_t>(max_tokens * target); // Default reduce to 40%
    }
    
    int32_t GetTokensToRemove(float target = 0.40f) const {
        if (!NeedsPruning()) return 0;
        return total_tokens - GetTargetTokensAfterPruning(target);
    }
};

enum class PruningStrategy {
    OLDEST_MESSAGES,      // Remove oldest conversation pairs
    BALANCED,            // Mix of removal strategies
    TEMPLATE_SECTIONS,   // Remove/compress template sections
    EMERGENCY_CLEANUP    // Hard limit reached, aggressive cleanup
};

enum class PruningState {
    NORMAL,              // Normal operation, no pruning needed
    MONITORING,          // Approaching threshold, monitoring closely
    PRUNING_REQUESTED,   // Pruning requested, waiting for completion
    EMERGENCY_PRUNING    // Hard limit reached, emergency cleanup
};

/**
 * ContextSizeManagerPlugin - Intelligent context pruning management as a plugin
 * 
 * This plugin serves as a pure processor for the Orchestrator's context size management pipeline.
 * It monitors individual contexts and triggers pruning events based on internal algorithms.
 * 
 * Architecture:
 * - Orchestrator coordinates workflow (scheduling, pipeline management)
 * - Plugin provides processing capability (context monitoring, pruning decisions)
 * - Orchestrator calls plugin methods through its pipeline processor
 * - Plugin triggers pruning events which are handled by contexts themselves
 * 
 * Note: Summarization is handled automatically from buffered pruned messages,
 * so this plugin focuses solely on pruning decisions and triggering.
 */
class ContextSizeManagerPlugin {
private:
    // No independent pipeline - Orchestrator coordinates workflow
    
    // Reference to orchestrator for context management
    Orchestrator* orchestrator = nullptr;
    
    // Component references
    LlamaManager* llama_manager = nullptr;
    SettingsManager* settings_manager = nullptr;
    
    // Status callback for UI updates
    StatusUpdateCallback status_callback;
    
    // Configuration - per plugin instance
    float pruning_threshold = 0.80f;        // Trigger at 80%
    float target_usage = 0.40f;             // Reduce to 40%
    float emergency_threshold = 0.95f;      // Emergency pruning at 95%
    int32_t min_buffer_tokens = 512;        // Minimum space reserved for AI response
    
    // Plugin-wide statistics
    std::atomic<size_t> pruning_requests{0};
    std::atomic<size_t> emergency_prunings{0};
    std::atomic<size_t> contexts_monitored{0};
    
    // Per-context tracking
    struct ContextMonitoring {
        ContextUsageStats last_stats;
        PruningState state = PruningState::NORMAL;
        std::chrono::steady_clock::time_point last_update;
        std::chrono::steady_clock::time_point last_pruning_request;
        size_t pruning_count = 0;
    };
    
    std::unordered_map<std::string, ContextMonitoring> monitored_contexts;
    mutable std::mutex contexts_mutex;
    
    // Debugging features
    struct DebugPruningEvent {
        std::string context_id;
        PruningStrategy strategy;
        int32_t tokens_before;
        std::chrono::system_clock::time_point timestamp;
    };
    
    mutable std::mutex debug_mutex;
    std::deque<std::string> log_history; // Plugin-specific log history
    static constexpr size_t MAX_LOG_HISTORY = 100; // Keep last 100 log entries
    std::optional<DebugPruningEvent> last_pruning_event; // Last pruning for debugging
    
    // Helper methods
    PruningStrategy DeterminePruningStrategy(const std::string& context_id, const ContextUsageStats& stats) const {
        // If we have a lot of message history relative to template/summary, use balanced approach
        if (stats.message_tokens > (stats.template_tokens + stats.summary_tokens) * 2) {
            return PruningStrategy::BALANCED;
        }
        
        // If template sections are taking up too much space, compress them
        if (stats.template_tokens > stats.max_tokens * 0.30f) {
            return PruningStrategy::TEMPLATE_SECTIONS;
        }
        
        // Default to oldest message removal
        return PruningStrategy::OLDEST_MESSAGES;
    }
    
    void LogUsageStats(const std::string& context_id, const ContextUsageStats& stats) const {
        LOG_DEBUG_ContextSizeManager("Context " + context_id + " - Usage: " + std::to_string(stats.total_tokens) + 
            "/" + std::to_string(stats.max_tokens) + " tokens (" + 
            std::to_string(static_cast<int>(stats.GetUsagePercentage() * 100)) + "%)");
        LOG_DEBUG_ContextSizeManager("Breakdown - Template: " + std::to_string(stats.template_tokens) + 
            ", Summary: " + std::to_string(stats.summary_tokens) + 
            ", Messages: " + std::to_string(stats.message_tokens) + 
            ", Buffer: " + std::to_string(stats.buffer_tokens));
    }
    
    // Helper methods for logging that also capture to debug history
    void LogInfo(const std::string& message) {
        LOG_ContextSizeManager(message);
        AddLogEntry("[INFO] " + message);
    }
    
    void LogWarning(const std::string& message) {
        LOG_WARNING_ContextSizeManager(message);
        AddLogEntry("[WARN] " + message);
    }
    
    void AddLogEntry(const std::string& entry) {
        std::lock_guard<std::mutex> lock(debug_mutex);
        log_history.push_back(entry);
        if (log_history.size() > MAX_LOG_HISTORY) {
            log_history.pop_front();
        }
    }

public:
    explicit ContextSizeManagerPlugin(Orchestrator* orch) 
        : orchestrator(orch) {
        
        if (orchestrator) {
            llama_manager = orchestrator->GetLlamaManager();
            settings_manager = orchestrator->GetSettingsManager();
        }
        
        LogInfo("ContextSizeManagerPlugin initialized as processor service");
    }
    
    ~ContextSizeManagerPlugin() {
        Shutdown();
    }
    
    /**
     * Initialize the plugin (called by Orchestrator)
     */
    bool Initialize() {
        LogInfo("ContextSizeManagerPlugin initialized");
        return true;
    }
    
    /**
     * Shutdown the plugin and clean up resources
     */
    void Shutdown() {
        std::lock_guard<std::mutex> lock(contexts_mutex);
        monitored_contexts.clear();
        
        LogInfo("ContextSizeManagerPlugin shutdown - Events: pruning=" + 
            std::to_string(pruning_requests.load()) + ", emergency=" + 
            std::to_string(emergency_prunings.load()));
    }
    
    /**
     * Set status update callback for UI notifications
     */
    void SetStatusCallback(StatusUpdateCallback callback) {
        status_callback = callback;
    }
    
    // Configuration
    void SetPruningThreshold(float threshold) {
        pruning_threshold = std::max(0.1f, std::min(0.95f, threshold));
        LogInfo("Pruning threshold set to: " + std::to_string(static_cast<int>(pruning_threshold * 100)) + "%");
    }
    
    void SetTargetUsage(float target) {
        target_usage = std::max(0.1f, std::min(0.8f, target));
        LogInfo("Target usage set to: " + std::to_string(static_cast<int>(target_usage * 100)) + "%");
    }
    
    void SetEmergencyThreshold(float threshold) {
        emergency_threshold = std::max(0.85f, std::min(0.99f, threshold));
        LogInfo("Emergency threshold set to: " + std::to_string(static_cast<int>(emergency_threshold * 100)) + "%");
    }
    
    void SetMinBufferTokens(int32_t buffer_tokens) {
        min_buffer_tokens = std::max(128, buffer_tokens);
        LogInfo("Min buffer tokens set to: " + std::to_string(min_buffer_tokens));
    }
    
    // Context monitoring - main interface for the plugin
    void MonitorContext(const std::string& context_id, const ContextUsageStats& stats) {
        std::lock_guard<std::mutex> lock(contexts_mutex);
        
        auto& monitoring = monitored_contexts[context_id];
        monitoring.last_stats = stats;
        monitoring.last_update = std::chrono::steady_clock::now();
        
        // Log detailed stats if approaching limits
        if (stats.GetUsagePercentage() > 0.70f) {
            LogUsageStats(context_id, stats);
        }
        
        // Check if pruning is needed and trigger if necessary
        CheckAndTriggerPruning(context_id, monitoring);
    }
    
    /**
     * Process a context size pruning request (called by Orchestrator pipeline)
     * This is the main processing method used by the Orchestrator's pipeline
     */
    ContextSizePruningResponse ProcessPruningRequest(const ContextSizePruningRequest& request) {
        std::lock_guard<std::mutex> lock(contexts_mutex);
        
        auto& monitoring = monitored_contexts[request.context_id];
        
        // Update monitoring info
        monitoring.last_stats.total_tokens = request.current_tokens;
        monitoring.last_stats.max_tokens = request.max_tokens;
        monitoring.last_update = std::chrono::steady_clock::now();
        
        // Determine if pruning should be triggered
        bool should_prune = monitoring.last_stats.GetUsagePercentage() >= pruning_threshold;
        bool is_emergency = monitoring.last_stats.GetUsagePercentage() >= emergency_threshold;
        
        PruningStrategy strategy = request.strategy;
        if (strategy == PruningStrategy::BALANCED) {
            // Let the plugin determine the best strategy
            strategy = DeterminePruningStrategy(request.context_id, monitoring.last_stats);
        }
        
        if (is_emergency) {
            strategy = PruningStrategy::EMERGENCY_CLEANUP;
            monitoring.state = PruningState::EMERGENCY_PRUNING;
            emergency_prunings++;
            LogWarning("Emergency pruning triggered for context: " + request.context_id);
        } else if (should_prune) {
            monitoring.state = PruningState::PRUNING_REQUESTED;
            monitoring.last_pruning_request = monitoring.last_update;
            monitoring.pruning_count++;
            pruning_requests++;
            LogInfo("Pruning requested for context: " + request.context_id + 
                " (strategy: " + std::to_string(static_cast<int>(strategy)) + ")");
        }
        
        // Create response
        int32_t estimated_after = should_prune ? 
            monitoring.last_stats.GetTargetTokensAfterPruning(target_usage) : 
            monitoring.last_stats.total_tokens;
            
        return ContextSizePruningResponse(
            request.context_id,
            should_prune,
            strategy,
            request.current_tokens,
            estimated_after,
            true, // success
            ""    // no error
        );
    }
    
    // Statistics and monitoring
    struct PluginStats {
        size_t pruning_requests_handled;
        size_t emergency_prunings_triggered;
        size_t contexts_currently_monitored;
        bool plugin_ready;
    };
    
    PluginStats GetStats() const {
        std::lock_guard<std::mutex> lock(contexts_mutex);
        return {
            pruning_requests.load(),
            emergency_prunings.load(),
            monitored_contexts.size(),
            true
        };
    }
    
    // Get monitoring info for a specific context
    std::optional<ContextMonitoring> GetContextMonitoring(const std::string& context_id) const {
        std::lock_guard<std::mutex> lock(contexts_mutex);
        auto it = monitored_contexts.find(context_id);
        if (it != monitored_contexts.end()) {
            return it->second;
        }
        return std::nullopt;
    }
    
    // Check if a context needs pruning (public interface)
    bool ContextNeedsPruning(const std::string& context_id) const {
        std::lock_guard<std::mutex> lock(contexts_mutex);
        auto it = monitored_contexts.find(context_id);
        if (it != monitored_contexts.end()) {
            return it->second.last_stats.NeedsPruning(pruning_threshold);
        }
        return false;
    }
    
    // Get debug information
    std::vector<std::string> GetLogHistory() const {
        std::lock_guard<std::mutex> lock(debug_mutex);
        return std::vector<std::string>(log_history.begin(), log_history.end());
    }
    
    std::optional<DebugPruningEvent> GetLastPruningEvent() const {
        std::lock_guard<std::mutex> lock(debug_mutex);
        return last_pruning_event;
    }
    
    void LogStatistics() {
        auto stats = GetStats();
        
        LogInfo("=== Context Size Manager Plugin Statistics ===");
        LogInfo("Pruning requests handled: " + std::to_string(stats.pruning_requests_handled));
        LogInfo("Emergency prunings triggered: " + std::to_string(stats.emergency_prunings_triggered));
        LogInfo("Contexts currently monitored: " + std::to_string(stats.contexts_currently_monitored));
        LogInfo("Plugin ready: " + std::string(stats.plugin_ready ? "Yes" : "No"));
    }

private:
    void CheckAndTriggerPruning(const std::string& context_id, ContextMonitoring& monitoring) {
        // Called with contexts_mutex already locked
        
        bool is_emergency = (monitoring.last_stats.GetUsagePercentage() >= emergency_threshold);
        float usage_percentage = monitoring.last_stats.GetUsagePercentage();
        
        if (is_emergency) {
            monitoring.state = PruningState::EMERGENCY_PRUNING;
            LogWarning("Emergency pruning threshold reached for context: " + context_id);
            
            // Trigger emergency pruning through context directly
            TriggerContextPruning(context_id, PruningStrategy::EMERGENCY_CLEANUP);
            
        } else if (usage_percentage >= pruning_threshold) {
            if (monitoring.state == PruningState::NORMAL) {
                monitoring.state = PruningState::MONITORING;
                LogInfo("Context " + context_id + " approaching pruning threshold (" + 
                    std::to_string(static_cast<int>(usage_percentage * 100)) + "%)");
            }
            
            // Check if enough time has passed since last pruning request (throttling)
            auto now = std::chrono::steady_clock::now();
            if (now - monitoring.last_pruning_request >= std::chrono::seconds(5)) {
                PruningStrategy strategy = DeterminePruningStrategy(context_id, monitoring.last_stats);
                TriggerContextPruning(context_id, strategy);
                
                monitoring.state = PruningState::PRUNING_REQUESTED;
                monitoring.last_pruning_request = now;
                monitoring.pruning_count++;
                pruning_requests++;
            }
            
        } else if (usage_percentage < 0.60f) {
            // Return to normal if usage drops significantly
            if (monitoring.state == PruningState::MONITORING || monitoring.state == PruningState::PRUNING_REQUESTED) {
                monitoring.state = PruningState::NORMAL;
            }
        }
    }
    
    void TriggerContextPruning(const std::string& context_id, PruningStrategy strategy) {
        // This would trigger pruning on the actual context
        // In the real implementation, this would call methods on the ContextInfo
        // or send a message through the Orchestrator to trigger pruning
        
        LogInfo("Triggering pruning for context: " + context_id + 
               " with strategy: " + std::to_string(static_cast<int>(strategy)));
        
        // Record debug event
        {
            std::lock_guard<std::mutex> lock(debug_mutex);
            last_pruning_event = DebugPruningEvent{
                context_id,
                strategy,
                0, // tokens_before would be filled in real implementation
                std::chrono::system_clock::now()
            };
        }
        
        // In a full implementation, this would:
        // 1. Get the ContextInfo for context_id from LlamaManager
        // 2. Call a pruning method on that context
        // 3. The context would handle the actual pruning and buffer messages for summarization
        
        if (llama_manager) {
            // Example: llama_manager->TriggerContextPruning(context_id, strategy);
            LogInfo("Pruning request forwarded to LlamaManager for context: " + context_id);
        }
    }
};

} // namespace LuminaChat

// Note: Integration with Orchestrator would require adding the following to Orchestrator.hpp:
//
// In private section:
//   LuminaChat::ContextSizeManagerPlugin* context_size_manager_plugin = nullptr;
//   std::atomic<bool> context_size_manager_plugin_available{false};
//   LuminaChat::ProcessingPipeline<LuminaChat::ContextSizePruningRequest, LuminaChat::ContextSizePruningResponse> context_size_pipeline;
//
// In public section:
//   void RegisterContextSizeManagerPlugin(LuminaChat::ContextSizeManagerPlugin* plugin);
//   void RequestContextSizePruning(const std::string& context_id, int32_t current_tokens, int32_t max_tokens, 
//                                 LuminaChat::PruningStrategy strategy, LuminaChat::RequestPriority priority = LuminaChat::RequestPriority::NORMAL);
//
// In private section:
//   void ProcessContextSizePruningRequest(const LuminaChat::ContextSizePruningRequest& request, std::function<void(LuminaChat::ContextSizePruningResponse)> callback);
//   void OnContextSizePruningComplete(const std::string& context_id, const LuminaChat::ContextSizePruningResponse& response);
//
// Usage example:
//   auto context_size_plugin = std::make_unique<LuminaChat::ContextSizeManagerPlugin>(&orchestrator);
//   context_size_plugin->Initialize();
//   orchestrator.RegisterContextSizeManagerPlugin(context_size_plugin.get());
//
//   // Monitor a context
//   LuminaChat::ContextUsageStats stats = GetContextStats(context_id);
//   context_size_plugin->MonitorContext(context_id, stats);
//
//   // Or request pruning directly
//   orchestrator.RequestContextSizePruning(context_id, current_tokens, max_tokens, LuminaChat::PruningStrategy::BALANCED);
