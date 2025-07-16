#pragma once

#include "../Context/ContextInfo.hpp"
#include "../ContextStats.hpp"
#include "../ProcessingPipeline.hpp"
#include "../Orchestrator.hpp"
#include "../LlamaManager.hpp"
#include "../SettingsManager.hpp"
#include "../Logger.hpp"
#include "../ErrorHandling.hpp"
#include <functional>
#include <cstdint>
#include <atomic>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <string>
#include <string_view>
#include <deque>
#include <optional>
#include <vector>

namespace LuminaChat {

// Callback type for status updates
using StatusUpdateCallback = std::function<void(const std::string& status, bool is_error)>;

// Enums defined before they are used
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

// Request and response structures for the pipeline
struct ContextPruningRequest {
    std::string context_id;
    int32_t current_tokens;
    int32_t max_tokens;
    PruningStrategy strategy;
    RequestPriority priority = RequestPriority::NORMAL;
    std::chrono::steady_clock::time_point queued_time;
    
    ContextPruningRequest(const std::string& ctx_id, int32_t current, int32_t max, PruningStrategy strat, RequestPriority prio = RequestPriority::NORMAL)
        : context_id(ctx_id), current_tokens(current), max_tokens(max), strategy(strat), priority(prio), queued_time(std::chrono::steady_clock::now()) {}
};

struct ContextPruningResponse {
    std::string context_id;
    bool pruning_triggered;
    PruningStrategy strategy_used;
    int32_t tokens_before;
    int32_t estimated_tokens_after;
    bool success;
    std::string error_message;
    
    ContextPruningResponse(const std::string& ctx_id, bool triggered, PruningStrategy strategy, int32_t before, int32_t after, bool succ, const std::string& err = "")
        : context_id(ctx_id), pruning_triggered(triggered), strategy_used(strategy), tokens_before(before), estimated_tokens_after(after), success(succ), error_message(err) {}
};

// ContextPruningPlugin: Intelligent context pruning management as a plugin
// - Monitor individual contexts usage continuously
// - Trigger pruning events at 80% capacity with smart strategy selection
// - Integrates with Orchestrator and ProcessingPipeline architecture
// - Handles pruning decisions, not summarization (that's handled automatically by buffered messages)

/**
 * ContextPruningPlugin - Intelligent context pruning management as a plugin
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
 * Division of Responsibilities:
 * - ContextPruningPlugin: Threshold monitoring, pruning strategy decisions, state coordination
 * - ContextInfo: Emergency overflow prevention, actual pruning execution, template management
 * - Orchestrator: Scheduled monitoring, plugin coordination, workflow management
 * 
 * Note: Summarization is handled automatically from buffered pruned messages,
 * so this plugin focuses solely on pruning decisions and triggering.
 * 
 * Deprecated ContextInfo Methods:
 * - IsNearContextLimit(): Replaced by this plugin's MonitorContext() with configurable thresholds
 * - Hardcoded 0.8f/0.9f thresholds: Replaced by configurable pruning_threshold/emergency_threshold
 */
class ContextPruningPlugin {
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
    float pruning_threshold = 0.75f;        // Trigger at 75% (more aggressive)
    float target_usage = 0.40f;             // Reduce to 40%
    float emergency_threshold = 0.90f;      // Emergency pruning at 90% (earlier intervention)
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
        
        // Deferred pruning support
        bool has_deferred_pruning = false;
        PruningStrategy deferred_pruning_strategy = PruningStrategy::BALANCED;
        std::chrono::steady_clock::time_point deferred_pruning_time;
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
        LOG_DEBUG_ContextPruning("Context " + context_id + " - Usage: " + std::to_string(stats.total_tokens) + 
            "/" + std::to_string(stats.max_tokens) + " tokens (" + 
            std::to_string(static_cast<int>(stats.GetUsagePercentage() * 100)) + "%)");
        LOG_DEBUG_ContextPruning("Breakdown - Template: " + std::to_string(stats.template_tokens) + 
            ", Summary: " + std::to_string(stats.summary_tokens) + 
            ", Messages: " + std::to_string(stats.message_tokens) + 
            ", Buffer: " + std::to_string(stats.buffer_tokens));
    }
    
    // Helper methods for logging that also capture to debug history
    void LogInfo(const std::string& message) {
        LOG_ContextPruning(message);
        AddLogEntry("[INFO] " + message);
    }
    
    void LogWarning(const std::string& message) {
        LOG_WARNING_ContextPruning(message);
        AddLogEntry("[WARN] " + message);
        // Also use unified error handling for consistency
        HandleWarning(message, "ContextPruning", false);
    }
    
    void LogError(const std::string& message) {
        LOG_ERROR_ContextPruning(message);
        AddLogEntry("[ERROR] " + message);
        // Also use unified error handling for consistency and UI notification
        HandleError(message, "ContextPruning", false);
    }
    
    void LogDebug(const std::string& message) {
        LOG_DEBUG_ContextPruning(message);
        AddLogEntry("[DEBUG] " + message);
    }
    
    void AddLogEntry(const std::string& entry) {
        std::lock_guard<std::mutex> lock(debug_mutex);
        log_history.push_back(entry);
        if (log_history.size() > MAX_LOG_HISTORY) {
            log_history.pop_front();
        }
    }
    
    // Helper methods for context access and state management
    ContextInfo* GetContextInfo(const std::string& context_id) {
        if (!llama_manager) return nullptr;
        return llama_manager->GetContextInfo(context_id);
    }
    
    bool IsContextAvailableForPruning(ContextInfo* context) {
        if (!context) return false;
        
        // Check if context is currently generating or processing
        if (context->IsGenerating()) {
            return false;
        }
        
        // Check context state
        ContextState state = context->GetState();
        return state == ContextState::CONTEXT_IDLE;
    }
    
    ContextUsageStats BuildStatsFromContext(ContextInfo* context) {
        if (!context) return {};
        
        ContextUsageStats stats;
        stats.total_tokens = static_cast<int32_t>(context->GetActualContextTokens());
        stats.max_tokens = static_cast<int32_t>(context->GetMaxContextTokens());
        stats.message_tokens = static_cast<int32_t>(context->GetMessageCount() * 75); // Rough estimate
        // Calculate buffer tokens to leave space for AI response
        stats.buffer_tokens = min_buffer_tokens;
        // Note: template_tokens, summary_tokens would need additional methods in ContextInfo
        
        return stats;
    }
    
    void DeferPruningRequest(const std::string& context_id, PruningStrategy strategy) {
        // Note: This method assumes contexts_mutex is already locked
        auto& monitoring = monitored_contexts[context_id];
        monitoring.deferred_pruning_strategy = strategy;
        monitoring.has_deferred_pruning = true;
        monitoring.deferred_pruning_time = std::chrono::steady_clock::now();
        
        LogInfo("Deferring pruning request for busy context: " + context_id);
    }

public:
    explicit ContextPruningPlugin(Orchestrator* orch) 
        : orchestrator(orch) {
        LogInfo("ContextPruningPlugin initialized as processor service");
    }
    
    ~ContextPruningPlugin() {
        Shutdown();
    }
    
    /**
     * Initialize the plugin (called by Orchestrator)
     */
    bool Initialize() {
        if (orchestrator) {
            llama_manager = orchestrator->GetLlamaManager();
            settings_manager = orchestrator->GetSettingsManager();
        }
        
        LogInfo("ContextPruningPlugin initialized");
        return true;
    }
    
    /**
     * Shutdown the plugin and clean up resources
     */
    void Shutdown() {
        std::lock_guard<std::mutex> lock(contexts_mutex);
        monitored_contexts.clear();
        
        LogInfo("ContextPruningPlugin shutdown - Events: pruning=" + 
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
        // Early exit check for pruning eligibility
        if (!llama_manager) {
            return;
        }
        
        auto* context = GetContextInfo(context_id);
        if (!context) {
            return;
        }
        
        // Check if this context is eligible for pruning
        if (!context->IsPruningEligible()) {
            LogDebug("Skipping pruning monitoring for context: " + context_id + " (not eligible for pruning)");
            return;
        }
        
        std::lock_guard<std::mutex> lock(contexts_mutex);
        
        auto& monitoring = monitored_contexts[context_id];
        monitoring.last_stats = stats;
        monitoring.last_update = std::chrono::steady_clock::now();
        
        // Log detailed stats if approaching limits (more proactive monitoring)
        if (stats.GetUsagePercentage() > 0.60f) {
            LogUsageStats(context_id, stats);
        }
        
        // Check for deferred pruning requests first
        if (monitoring.has_deferred_pruning) {
            // Check if context is now available and enough time has passed
            if (IsContextAvailableForPruning(context) && 
                (monitoring.last_update - monitoring.deferred_pruning_time >= std::chrono::seconds(2))) {
                
                LogInfo("Executing deferred pruning for context: " + context_id);
                TriggerContextPruning(context_id, monitoring.deferred_pruning_strategy);
                monitoring.has_deferred_pruning = false;
                return; // Don't check for new pruning in the same call
            }
        }
        
        // Check if pruning is needed and trigger if necessary
        CheckAndTriggerPruning(context_id, monitoring);
    }
    
    /**
     * Process a context pruning request (called by Orchestrator pipeline)
     * This is the main processing method used by the Orchestrator's pipeline
     */
    ContextPruningResponse ProcessPruningRequest(const ContextPruningRequest& request) {
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
            
        return ContextPruningResponse(
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
    
    /**
     * Emergency pruning interface for ContextInfo to request immediate pruning
     * This ensures all pruning goes through the plugin for proper coordination
     */
    std::vector<std::pair<std::string, std::string>> RequestEmergencyPruning(const std::string& context_id, size_t keep_messages = 3) {
        LogWarning("Emergency pruning requested for context: " + context_id + " (keep " + std::to_string(keep_messages) + " messages)");
        
        if (!llama_manager) {
            LogWarning("LlamaManager not available for emergency pruning: " + context_id);
            return {};
        }
        
        // Get the context directly
        auto* context = GetContextInfo(context_id);
        if (!context) {
            LogWarning("Context not found for emergency pruning: " + context_id);
            return {};
        }
        
        // For emergency pruning, we bypass the availability check since this is critical
        // Record tokens before pruning
        int32_t tokens_before = static_cast<int32_t>(context->GetActualContextTokens());
        
        LogInfo("Executing emergency pruning for context: " + context_id + 
               " (" + std::to_string(tokens_before) + " tokens)");
        
        // Try to acquire plugin processing state for this context
        if (!context->TryAcquirePluginProcessing("ContextPruningPlugin")) {
            LogWarning("Failed to acquire plugin processing state for emergency pruning: " + context_id);
            return {};
        }
        
        // Execute emergency pruning and get the pruned messages
        std::vector<std::pair<std::string, std::string>> pruned_messages = 
            context->PruneContextImmediateWithExtraction(keep_messages);
        
        // Release plugin processing state
        [[maybe_unused]] bool released = context->ReleasePluginProcessing("ContextPruningPlugin");
        
        // Coordinate with Orchestrator for summarization (plugin responsibility)
        if (!pruned_messages.empty() && orchestrator) {
            orchestrator->RequestSummarizationForPrunedMessages(context_id, pruned_messages);
            LogInfo("Requested summarization for " + std::to_string(pruned_messages.size()) + 
                   " emergency pruned messages from context: " + context_id);
        }
        
        // Update emergency statistics
        emergency_prunings++;
        
        // Update monitoring state
        {
            std::lock_guard<std::mutex> lock(contexts_mutex);
            auto& monitoring = monitored_contexts[context_id];
            monitoring.state = PruningState::NORMAL; // Reset to normal after emergency pruning
            monitoring.last_stats.total_tokens = static_cast<int32_t>(context->GetActualContextTokens());
            monitoring.last_update = std::chrono::steady_clock::now();
        }
        
        int32_t tokens_after = static_cast<int32_t>(context->GetActualContextTokens());
        LogInfo("Emergency pruning completed for context: " + context_id + 
               " (" + std::to_string(tokens_before) + " -> " + std::to_string(tokens_after) + " tokens)");
        
        return pruned_messages;
    }
    
    void LogStatistics() {
        auto stats = GetStats();
        
        LogInfo("=== Context Pruning Plugin Statistics ===");
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
            
            // Trigger emergency pruning - this will handle deferred requests if context is busy
            TriggerContextPruning(context_id, PruningStrategy::EMERGENCY_CLEANUP);
            
        } else if (usage_percentage >= pruning_threshold) {
            if (monitoring.state == PruningState::NORMAL) {
                monitoring.state = PruningState::MONITORING;
                LogInfo("Context " + context_id + " approaching pruning threshold (" + 
                    std::to_string(static_cast<int>(usage_percentage * 100)) + "%)");
            }
            
            // Check if enough time has passed since last pruning request (reduced throttling for faster response)
            auto now = std::chrono::steady_clock::now();
            if (now - monitoring.last_pruning_request >= std::chrono::seconds(2)) {
                PruningStrategy strategy = DeterminePruningStrategy(context_id, monitoring.last_stats);
                TriggerContextPruning(context_id, strategy);
                
                monitoring.state = PruningState::PRUNING_REQUESTED;
                monitoring.last_pruning_request = now;
                monitoring.pruning_count++;
                pruning_requests++;
            }
            
        } else if (usage_percentage < 0.50f) {
            // Return to normal if usage drops significantly (more conservative threshold)
            if (monitoring.state == PruningState::MONITORING || monitoring.state == PruningState::PRUNING_REQUESTED) {
                monitoring.state = PruningState::NORMAL;
            }
        }
    }
    
    void TriggerContextPruning(const std::string& context_id, PruningStrategy strategy) {
        LogInfo("Triggering pruning for context: " + context_id + 
               " with strategy: " + std::to_string(static_cast<int>(strategy)));
        
        if (!llama_manager) {
            LogWarning("LlamaManager not available for pruning context: " + context_id);
            return;
        }
        
        // Get the context and check if it's available for pruning
        auto* context = GetContextInfo(context_id);
        if (!context) {
            LogWarning("Context not found for pruning: " + context_id);
            return;
        }
        
        // Check if context is currently busy with generation
        if (!IsContextAvailableForPruning(context)) {
            LogWarning("Context " + context_id + " is busy, deferring pruning");
            // Defer the pruning request for later (contexts_mutex is already locked by caller)
            DeferPruningRequest(context_id, strategy);
            return;
        }
        
        // Record tokens before pruning
        int32_t tokens_before = static_cast<int32_t>(context->GetActualContextTokens());
        
        try {
            // Execute the actual pruning based on strategy
            bool success = ExecutePruningStrategy(context, strategy);
            
            if (success) {
                int32_t tokens_after = static_cast<int32_t>(context->GetActualContextTokens());
                LogInfo("Pruning completed for context: " + context_id + 
                       " (" + std::to_string(tokens_before) + " -> " + std::to_string(tokens_after) + " tokens)");
                
                // Record debug event with actual data
                {
                    std::lock_guard<std::mutex> lock(debug_mutex);
                    last_pruning_event = DebugPruningEvent{
                        context_id,
                        strategy,
                        tokens_before,
                        std::chrono::system_clock::now()
                    };
                }
                
                // Update monitoring state - contexts_mutex is already locked by caller
                auto it = monitored_contexts.find(context_id);
                if (it != monitored_contexts.end()) {
                    it->second.state = PruningState::NORMAL;
                    it->second.last_stats.total_tokens = tokens_after;
                }
            } else {
                LogWarning("Pruning failed for context: " + context_id);
            }
            
        } catch (const std::exception& e) {
            LogWarning("Exception during pruning: " + std::string(e.what()));
        }
    }
    
    bool ExecutePruningStrategy(ContextInfo* context, PruningStrategy strategy) {
        if (!context) return false;
        
        try {
            switch (strategy) {
                case PruningStrategy::OLDEST_MESSAGES:
                    return PruneOldestMessages(context);
                    
                case PruningStrategy::BALANCED:
                    return PruneBalanced(context);
                    
                case PruningStrategy::TEMPLATE_SECTIONS:
                    // For now, fall back to oldest messages since template compression isn't implemented
                    LogInfo("Template section pruning not fully implemented, using oldest messages strategy");
                    return PruneOldestMessages(context);
                    
                case PruningStrategy::EMERGENCY_CLEANUP:
                    return PruneEmergencyCleanup(context);
                    
                default:
                    LogWarning("Unknown pruning strategy: " + std::to_string(static_cast<int>(strategy)));
                    return PruneOldestMessages(context); // Fallback
            }
        } catch (const std::exception& e) {
            LogWarning("Exception during pruning strategy execution: " + std::string(e.what()));
            return false;
        }
    }
    
    bool PruneOldestMessages(ContextInfo* context) {
        // Determine how many messages to keep based on target usage
        size_t current_messages = context->GetMessageCount();
        if (current_messages <= 6) return true; // Keep at least 3 conversation pairs
        
        // Calculate target number of messages to keep
        size_t keep_messages = static_cast<size_t>(current_messages * target_usage);
        keep_messages = std::max(keep_messages, size_t(6)); // Minimum 3 pairs (user + assistant)
        keep_messages = std::min(keep_messages, current_messages); // Don't exceed current count
        
        LogInfo("Pruning oldest messages: keeping " + std::to_string(keep_messages) + 
               " out of " + std::to_string(current_messages) + " messages");
        
        // Perform the actual pruning operation and get the pruned messages
        std::vector<std::pair<std::string, std::string>> pruned_messages = 
            context->PruneContextImmediateWithExtraction(keep_messages);
        
        // Coordinate with Orchestrator for summarization (plugin responsibility)
        if (!pruned_messages.empty() && orchestrator) {
            orchestrator->RequestSummarizationForPrunedMessages(context->GetContextId(), pruned_messages);
            LogInfo("Requested summarization for " + std::to_string(pruned_messages.size()) + 
                   " pruned messages from context: " + context->GetContextId());
        }
        
        return true;
    }
    
    bool PruneBalanced(ContextInfo* context) {
        // For balanced strategy, be more conservative - keep more messages
        size_t current_messages = context->GetMessageCount();
        if (current_messages <= 8) return true; // Keep at least 4 conversation pairs
        
        // Keep 60% of messages for balanced approach (less aggressive than oldest-only)
        size_t keep_messages = static_cast<size_t>(current_messages * 0.6f);
        keep_messages = std::max(keep_messages, size_t(8)); // Minimum 4 pairs
        keep_messages = std::min(keep_messages, current_messages);
        
        LogInfo("Pruning with balanced strategy: keeping " + std::to_string(keep_messages) + 
               " out of " + std::to_string(current_messages) + " messages");
        
        // Perform the actual pruning operation and get the pruned messages
        std::vector<std::pair<std::string, std::string>> pruned_messages = 
            context->PruneContextImmediateWithExtraction(keep_messages);
        
        // Coordinate with Orchestrator for summarization (plugin responsibility)
        if (!pruned_messages.empty() && orchestrator) {
            orchestrator->RequestSummarizationForPrunedMessages(context->GetContextId(), pruned_messages);
            LogInfo("Requested summarization for " + std::to_string(pruned_messages.size()) + 
                   " pruned messages from context: " + context->GetContextId());
        }
        
        return true;
    }
    
    bool PruneEmergencyCleanup(ContextInfo* context) {
        // Aggressive pruning - keep only last 2-3 conversation pairs
        size_t current_messages = context->GetMessageCount();
        if (current_messages <= 4) return true; // Already minimal
        
        size_t keep_messages = 4; // Keep only 2 conversation pairs
        
        LogInfo("Emergency pruning: keeping only " + std::to_string(keep_messages) + 
               " out of " + std::to_string(current_messages) + " messages");
        
        // Perform the actual pruning operation and get the pruned messages
        std::vector<std::pair<std::string, std::string>> pruned_messages = 
            context->PruneContextImmediateWithExtraction(keep_messages);
        
        // Coordinate with Orchestrator for summarization (plugin responsibility)
        if (!pruned_messages.empty() && orchestrator) {
            orchestrator->RequestSummarizationForPrunedMessages(context->GetContextId(), pruned_messages);
            LogInfo("Requested summarization for " + std::to_string(pruned_messages.size()) + 
                   " pruned messages from context: " + context->GetContextId());
        }
        
        // Update emergency statistics
        emergency_prunings++;
        
        return true;
    }
};

} // namespace LuminaChat

// Implementation of Orchestrator::MonitorAllContextSizes to avoid circular dependencies
// This method is declared in Orchestrator.hpp but defined here to access ContextPruningPlugin
inline void Orchestrator::MonitorAllContextSizes() {
    // Early exit if context pruning plugin is not available
    if (!context_pruning_plugin_available.load(std::memory_order_relaxed)) [[unlikely]] {
        LOG_DEBUG_Orchestrator("Context pruning plugin not available, skipping monitoring");
        return;
    }
    
    if (!llama_manager) [[unlikely]] {
        LOG_WARNING_Orchestrator("LlamaManager not available for context size monitoring");
        return;
    }
    
    // Get all active contexts
    auto contexts = llama_manager->GetAllActiveContexts();
    if (contexts.empty()) [[likely]] {
        LOG_DEBUG_Orchestrator("No active contexts to monitor");
        return; // No contexts to monitor
    }
    
    LOG_DEBUG_Orchestrator("Monitoring " + std::to_string(contexts.size()) + " contexts for size management");
    
    // Monitor each context through the plugin
    for (auto* context : contexts) [[likely]] {
        if (!context) continue;
        
        try {
            // Build usage stats from context
            LuminaChat::ContextUsageStats stats;
            stats.total_tokens = static_cast<int32_t>(context->GetActualContextTokens());
            stats.max_tokens = static_cast<int32_t>(context->GetMaxContextTokens());
            stats.message_tokens = static_cast<int32_t>(context->GetMessageCount() * 75); // Rough estimate
            stats.buffer_tokens = 512; // Default buffer
            // Note: More detailed stats would require additional methods in ContextInfo
            
            // Let the plugin monitor and make decisions
            context_pruning_plugin->MonitorContext(context->GetContextId(), stats);
            
            // Log context usage for debugging - use INFO level for high usage
            float usage_percentage = static_cast<float>(stats.total_tokens) / static_cast<float>(stats.max_tokens);
            if (usage_percentage > 0.50f) {
                LOG_Orchestrator("Context " + context->GetContextId() + " usage: " + 
                                      std::to_string(stats.total_tokens) + "/" + std::to_string(stats.max_tokens) + 
                                      " (" + std::to_string(static_cast<int>(usage_percentage * 100)) + "%)");
            } else {
                LOG_DEBUG_Orchestrator("Context " + context->GetContextId() + " usage: " + 
                                      std::to_string(stats.total_tokens) + "/" + std::to_string(stats.max_tokens) + 
                                      " (" + std::to_string(static_cast<int>(usage_percentage * 100)) + "%)");
            }
            
        } catch (const std::exception& e) {
            LOG_WARNING_Orchestrator("Exception during context size monitoring for " + 
                                    context->GetContextId() + ": " + e.what());
        }
    }
}
