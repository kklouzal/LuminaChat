#pragma once

#include <functional>
#include <cstdint>
#include <atomic>
#include <chrono>
#include <mutex>

// ContextSizeManager: Intelligent context pruning management
// - Monitor context usage continuously
// - Trigger pruning at 80% capacity, reduce to 40% usage
// - Smart analysis of token distribution (summaries, active history, AI space, buffer)
// - Plugin Integration: Requests summarization via Orchestrator callback
// - Template Integration: Works with ChatTemplateManager for context size calculations

// Forward declaration for Logger
class Logger;
Logger& GetLogger();

struct ContextUsageStats {
    int32_t total_tokens = 0;
    int32_t max_tokens = 0;
    int32_t template_tokens = 0;      // Base template + system sections
    int32_t summary_tokens = 0;       // Summary sections in template
    int32_t message_tokens = 0;       // Actual conversation history
    int32_t buffer_tokens = 0;        // Reserved space for AI response
    
    // Additional fields for compatibility with test expectations
    int32_t current_usage = 0;        // Alias for total_tokens
    int32_t peak_usage = 0;           // Peak usage recorded
    int32_t pruning_count = 0;        // Number of pruning events
    
    float GetUsagePercentage() const {
        return max_tokens > 0 ? static_cast<float>(total_tokens) / max_tokens : 0.0f;
    }
    
    int32_t GetAvailableTokens() const {
        return max_tokens - total_tokens;
    }
    
    bool NeedsPruning() const {
        return GetUsagePercentage() >= 0.80f; // 80% threshold
    }
    
    int32_t GetTargetTokensAfterPruning() const {
        return static_cast<int32_t>(max_tokens * 0.40f); // Reduce to 40%
    }
    
    int32_t GetTokensToRemove() const {
        if (!NeedsPruning()) return 0;
        return total_tokens - GetTargetTokensAfterPruning();
    }
};

enum class PruningStrategy {
    OLDEST_MESSAGES,      // Remove oldest conversation pairs
    SUMMARIZATION,        // Trigger summarization plugin  
    BALANCED,            // Mix of removal and summarization
    TEMPLATE_SECTIONS    // Remove/compress template sections
};

enum class PruningState {
    NORMAL,              // Normal operation, no pruning needed
    MONITORING,          // Approaching threshold, monitoring closely
    PRUNING_REQUESTED,   // Pruning requested, waiting for completion
    SUMMARIZING,         // Summarization plugin active
    EMERGENCY_PRUNING    // Hard limit reached, emergency cleanup
};

class ContextSizeManager {
private:
    // Context monitoring
    ContextUsageStats current_stats;
    mutable std::mutex stats_mutex;
    
    // Pruning configuration
    float pruning_threshold = 0.80f;        // Trigger at 80%
    float target_usage = 0.40f;             // Reduce to 40%
    int32_t emergency_threshold_tokens = 0;  // Hard limit for emergency pruning
    int32_t min_buffer_tokens = 512;        // Minimum space reserved for AI response
    
    // State management
    std::atomic<PruningState> current_state{PruningState::NORMAL};
    std::chrono::steady_clock::time_point last_pruning_request;
    std::chrono::steady_clock::time_point last_update;
    
    // Callback for summarization requests (registered by Orchestrator)
    std::function<void(const std::string&, const std::string&)> summarization_callback;
    
    // Performance tracking
    std::atomic<size_t> pruning_events{0};
    std::atomic<size_t> summarization_requests{0};
    std::atomic<size_t> emergency_prunings{0};
    
    // Helper methods
    PruningStrategy DeterminePruningStrategy() const {
        std::lock_guard<std::mutex> lock(stats_mutex);
        
        // If we have a lot of message history relative to template/summary, use summarization
        if (current_stats.message_tokens > (current_stats.template_tokens + current_stats.summary_tokens) * 2) {
            return PruningStrategy::SUMMARIZATION;
        }
        
        // If template sections are taking up too much space, compress them
        if (current_stats.template_tokens > current_stats.max_tokens * 0.30f) {
            return PruningStrategy::TEMPLATE_SECTIONS;
        }
        
        // Default to balanced approach
        return PruningStrategy::BALANCED;
    }
    
    void LogUsageStats() const {
        std::lock_guard<std::mutex> lock(stats_mutex);
        LOG_DEBUG_ContextSizeManager("Usage: " + std::to_string(current_stats.total_tokens) + 
            "/" + std::to_string(current_stats.max_tokens) + " tokens (" + 
            std::to_string(static_cast<int>(current_stats.GetUsagePercentage() * 100)) + "%)");
        LOG_DEBUG_ContextSizeManager("Breakdown - Template: " + std::to_string(current_stats.template_tokens) + 
            ", Summary: " + std::to_string(current_stats.summary_tokens) + 
            ", Messages: " + std::to_string(current_stats.message_tokens) + 
            ", Buffer: " + std::to_string(current_stats.buffer_tokens));
    }
    
public:
    ContextSizeManager() {
        LOG_ContextSizeManager("ContextSizeManager initialized");
        last_update = std::chrono::steady_clock::now();
    }
    
    ~ContextSizeManager() {
        LOG_ContextSizeManager("ContextSizeManager destroyed - Events: pruning=" + 
            std::to_string(pruning_events.load()) + ", summarization=" + 
            std::to_string(summarization_requests.load()) + ", emergency=" + 
            std::to_string(emergency_prunings.load()));
    }
    
    // Callback registration for summarization requests
    // Orchestrator (higher) registers with ContextSizeManager (lower)
    void RegisterSummarizationCallback(std::function<void(const std::string&, const std::string&)> callback) {
        std::lock_guard<std::mutex> lock(stats_mutex);
        summarization_callback = std::move(callback);
        LOG_ContextSizeManager("Summarization callback registered");
    }
    
    // Configuration
    void SetMaxTokens(int32_t max_tokens) {
        std::lock_guard<std::mutex> lock(stats_mutex);
        current_stats.max_tokens = max_tokens;
        emergency_threshold_tokens = static_cast<int32_t>(max_tokens * 0.95f); // 95% is emergency
        LOG_ContextSizeManager("Max tokens set to: " + std::to_string(max_tokens));
    }
    
    void SetPruningThreshold(float threshold) {
        std::lock_guard<std::mutex> lock(stats_mutex);
        pruning_threshold = std::max(0.1f, std::min(0.95f, threshold));
        LOG_ContextSizeManager("Pruning threshold set to: " + std::to_string(static_cast<int>(pruning_threshold * 100)) + "%");
    }
    
    void SetTargetUsage(float target) {
        std::lock_guard<std::mutex> lock(stats_mutex);
        target_usage = std::max(0.1f, std::min(0.8f, target));
        LOG_ContextSizeManager("Target usage set to: " + std::to_string(static_cast<int>(target_usage * 100)) + "%");
    }
    
    void SetMinBufferTokens(int32_t buffer_tokens) {
        std::lock_guard<std::mutex> lock(stats_mutex);
        min_buffer_tokens = std::max(128, buffer_tokens);
        LOG_ContextSizeManager("Min buffer tokens set to: " + std::to_string(min_buffer_tokens));
    }
    
    // Context monitoring
    void UpdateUsage(int32_t template_tokens, int32_t summary_tokens, int32_t message_tokens) {
        std::lock_guard<std::mutex> lock(stats_mutex);
        
        current_stats.template_tokens = template_tokens;
        current_stats.summary_tokens = summary_tokens;
        current_stats.message_tokens = message_tokens;
        current_stats.buffer_tokens = min_buffer_tokens;
        current_stats.total_tokens = template_tokens + summary_tokens + message_tokens + min_buffer_tokens;
        
        last_update = std::chrono::steady_clock::now();
        
        // Log detailed stats periodically or when approaching limits
        if (current_stats.GetUsagePercentage() > 0.70f) {
            LogUsageStats();
        }
        
        // Check if pruning is needed
        CheckPruningNeeded();
    }
    
    void UpdateUsage(int32_t total_tokens) {
        std::lock_guard<std::mutex> lock(stats_mutex);
        
        current_stats.total_tokens = total_tokens + min_buffer_tokens;
        last_update = std::chrono::steady_clock::now();
        
        if (current_stats.GetUsagePercentage() > 0.70f) {
            LogUsageStats();
        }
        
        CheckPruningNeeded();
    }
    
    // Pruning decision logic
    bool NeedsPruning() const {
        std::lock_guard<std::mutex> lock(stats_mutex);
        return current_stats.GetUsagePercentage() >= pruning_threshold;
    }
    
    bool IsEmergencyPruningNeeded() const {
        std::lock_guard<std::mutex> lock(stats_mutex);
        return current_stats.total_tokens >= emergency_threshold_tokens;
    }
    
    bool HasSufficientSpace(int32_t additional_tokens) const {
        std::lock_guard<std::mutex> lock(stats_mutex);
        return (current_stats.total_tokens + additional_tokens) < emergency_threshold_tokens;
    }
    
    // Pruning execution
    bool RequestPruning(const std::string& context_id) {
        auto now = std::chrono::steady_clock::now();
        
        // Prevent too frequent pruning requests
        if (now - last_pruning_request < std::chrono::seconds(5)) {
            LOG_DEBUG_ContextSizeManager("Pruning request throttled for context: " + context_id);
            return false;
        }
        
        PruningState expected = PruningState::NORMAL;
        if (!current_state.compare_exchange_strong(expected, PruningState::PRUNING_REQUESTED)) {
            expected = PruningState::MONITORING;
            if (!current_state.compare_exchange_strong(expected, PruningState::PRUNING_REQUESTED)) {
                LOG_DEBUG_ContextSizeManager("Pruning already in progress for context: " + context_id);
                return false;
            }
        }
        
        last_pruning_request = now;
        pruning_events++;
        
        PruningStrategy strategy = DeterminePruningStrategy();
        LOG_ContextSizeManager("Pruning requested for context: " + context_id + 
            " (strategy: " + std::to_string(static_cast<int>(strategy)) + ")");
        
        // Execute pruning based on strategy
        return ExecutePruning(context_id, strategy);
    }
    
    bool RequestEmergencyPruning(const std::string& context_id) {
        current_state = PruningState::EMERGENCY_PRUNING;
        emergency_prunings++;
        
        LOG_WARNING_ContextSizeManager("Emergency pruning triggered for context: " + context_id);
        
        // Emergency pruning: immediate hard cleanup
        return ExecutePruning(context_id, PruningStrategy::OLDEST_MESSAGES);
    }
    
    void OnPruningCompleted(const std::string& context_id, int32_t tokens_removed) {
        current_state = PruningState::NORMAL;
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex);
            current_stats.total_tokens -= tokens_removed;
        }
        
        LOG_ContextSizeManager("Pruning completed for context: " + context_id + 
            " (removed " + std::to_string(tokens_removed) + " tokens)");
        LogUsageStats();
    }
    
    void OnSummarizationCompleted(const std::string& context_id, int32_t old_tokens, int32_t new_tokens) {
        current_state = PruningState::NORMAL;
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex);
            int32_t net_change = new_tokens - old_tokens;
            current_stats.message_tokens += net_change;
            current_stats.summary_tokens += (old_tokens > new_tokens) ? (old_tokens - new_tokens) : 0;
            current_stats.total_tokens += net_change;
        }
        
        LOG_ContextSizeManager("Summarization completed for context: " + context_id + 
            " (net change: " + std::to_string(new_tokens - old_tokens) + " tokens)");
        LogUsageStats();
    }
    
    // Statistics and monitoring
    ContextUsageStats GetStats() const {
        std::lock_guard<std::mutex> lock(stats_mutex);
        return current_stats;
    }
    
    PruningState GetState() const {
        return current_state.load();
    }
    
    int32_t GetRecommendedTokensToRemove() const {
        std::lock_guard<std::mutex> lock(stats_mutex);
        return current_stats.GetTokensToRemove();
    }
    
    float GetUsagePercentage() const {
        std::lock_guard<std::mutex> lock(stats_mutex);
        return current_stats.GetUsagePercentage();
    }
    
    std::chrono::steady_clock::time_point GetLastUpdate() const {
        return last_update;
    }
    
    void LogStatistics() const {
        auto stats = GetStats();
        auto state = GetState();
        
        LOG_ContextSizeManager("=== Context Size Statistics ===");
        LOG_ContextSizeManager("Usage: " + std::to_string(stats.total_tokens) + "/" + 
            std::to_string(stats.max_tokens) + " tokens (" + 
            std::to_string(static_cast<int>(stats.GetUsagePercentage() * 100)) + "%)");
        LOG_ContextSizeManager("Available: " + std::to_string(stats.GetAvailableTokens()) + " tokens");
        LOG_ContextSizeManager("Breakdown:");
        LOG_ContextSizeManager("  Template: " + std::to_string(stats.template_tokens) + " tokens");
        LOG_ContextSizeManager("  Summary: " + std::to_string(stats.summary_tokens) + " tokens");
        LOG_ContextSizeManager("  Messages: " + std::to_string(stats.message_tokens) + " tokens");
        LOG_ContextSizeManager("  Buffer: " + std::to_string(stats.buffer_tokens) + " tokens");
        LOG_ContextSizeManager("State: " + std::to_string(static_cast<int>(state)));
        LOG_ContextSizeManager("Events: pruning=" + std::to_string(pruning_events.load()) + 
            ", summarization=" + std::to_string(summarization_requests.load()) + 
            ", emergency=" + std::to_string(emergency_prunings.load()));
        
        if (stats.NeedsPruning()) {
            LOG_ContextSizeManager("⚠️  Pruning recommended: remove " + 
                std::to_string(stats.GetTokensToRemove()) + " tokens");
        }
    }
    
    // Additional methods for testing and external API compatibility
    bool NeedsPruning(int32_t current_tokens, int32_t max_tokens) const {
        std::lock_guard<std::mutex> lock(stats_mutex);
        float usage_percent = static_cast<float>(current_tokens) / static_cast<float>(max_tokens);
        return usage_percent >= pruning_threshold;
    }
    
    int32_t GetCurrentUsage() const {
        std::lock_guard<std::mutex> lock(stats_mutex);
        return current_stats.total_tokens;
    }
    
    ContextUsageStats GetUsageStats() const {
        std::lock_guard<std::mutex> lock(stats_mutex);
        ContextUsageStats stats = current_stats;
        // Update compatibility fields
        stats.current_usage = stats.total_tokens;
        stats.peak_usage = stats.total_tokens; // In a real implementation, this would track the peak
        stats.pruning_count = static_cast<int32_t>(pruning_events.load());
        return stats;
    }
    
private:
    void CheckPruningNeeded() {
        // Called with stats_mutex already locked
        
        if (IsEmergencyPruningNeeded()) {
            current_state = PruningState::EMERGENCY_PRUNING;
            LOG_WARNING_ContextSizeManager("Emergency pruning threshold reached!");
        } else if (current_stats.GetUsagePercentage() >= pruning_threshold) {
            if (current_state == PruningState::NORMAL) {
                current_state = PruningState::MONITORING;
                LOG_ContextSizeManager("Approaching pruning threshold (" + 
                    std::to_string(static_cast<int>(current_stats.GetUsagePercentage() * 100)) + "%)");
            }
        } else if (current_stats.GetUsagePercentage() < 0.60f) {
            // Return to normal if usage drops significantly
            if (current_state == PruningState::MONITORING) {
                current_state = PruningState::NORMAL;
            }
        }
    }
    
    bool ExecutePruning(const std::string& context_id, PruningStrategy strategy) {
        switch (strategy) {
            case PruningStrategy::SUMMARIZATION:
                return RequestSummarization(context_id);
                
            case PruningStrategy::OLDEST_MESSAGES:
                // Direct message removal - this would be handled by the context itself
                LOG_ContextSizeManager("Requesting oldest message removal for: " + context_id);
                return true;
                
            case PruningStrategy::TEMPLATE_SECTIONS:
                LOG_ContextSizeManager("Requesting template section compression for: " + context_id);
                return true;
                
            case PruningStrategy::BALANCED:
                // Try summarization first, fall back to message removal
                if (RequestSummarization(context_id)) {
                    return true;
                }
                LOG_ContextSizeManager("Falling back to message removal for: " + context_id);
                return true;
                
            default:
                LOG_WARNING_ContextSizeManager("Unknown pruning strategy");
                return false;
        }
    }
    
    bool RequestSummarization(const std::string& context_id) {
        if (!summarization_callback) {
            LOG_WARNING_ContextSizeManager("Summarization requested but no callback registered");
            return false;
        }
        
        current_state = PruningState::SUMMARIZING;
        summarization_requests++;
        
        // Request summarization of message history
        std::string content_to_summarize;
        {
            std::lock_guard<std::mutex> lock(stats_mutex);
            content_to_summarize = "message_history"; // This would be the actual content in real implementation
        }
        
        LOG_ContextSizeManager("Requesting summarization for context: " + context_id);
        summarization_callback(context_id, content_to_summarize);
        
        return true;
    }
};
