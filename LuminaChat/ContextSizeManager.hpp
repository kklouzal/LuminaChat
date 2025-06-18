// ContextSizeManager.hpp - Production-ready adaptive context size management for LLaMA contexts
//
// COMPREHENSIVE STRATEGY:
// - Dynamic tracking of AI response sizes with 1.5x buffer
// - Dynamic tracking of summary sizes with 1.5x buffer  
// - Hard 30% cap for summary allocation with dynamic threshold management
// - 10% global emergency buffer for prediction overruns
// - Minimum 3 summary slots with intelligent merging
// - Truly adaptive space allocation based on rolling averages
// - Mathematical validation of all allocation constraints

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <memory>
#include <cmath>
#include <mutex>
#include "LogHandler.hpp"

// Forward declarations - actual definitions in LlamaContext.hpp
struct ModelInfo;
struct ContextInfo;

// Context management strategies (auto-adaptive only)
enum class ContextStrategy : uint8_t {
    BALANCED = 0,      // Balanced allocation based on actual usage patterns
    AI_HEAVY = 1,      // Prioritize AI response space for large responses
    SUMMARY_HEAVY = 2  // Prioritize summary space for detailed summarization
};

// Mathematically validated context size management configuration
namespace ContextSizeConstants {
    // Core allocation constraints (mathematically validated)
    constexpr float MAX_TOTAL_SUMMARY_ALLOCATION = 0.30f;     // Hard 30% cap for all summaries
    constexpr float GLOBAL_EMERGENCY_BUFFER = 0.10f;          // 10% global safety buffer  
    constexpr float MIN_AI_ALLOCATION = 0.15f;                // Minimum 15% for AI responses
    constexpr float MIN_ACTIVE_CONTENT = 0.25f;               // Minimum 25% for active conversation
    constexpr float DYNAMIC_BUFFER_MULTIPLIER = 1.5f;         // 1.5x multiplier for predictions
    
    // Summary management
    constexpr size_t MIN_SUMMARY_SLOTS = 3;                   // Minimum slots before merging
    constexpr float MAX_HISTORY_PER_SUMMARY = 0.10f;          // Max 10% of context per summary creation
      // Dynamic threshold calculation (mathematically bounded)
    constexpr float BASE_SUMMARY_MERGE_THRESHOLD = 0.25f;     // Base threshold (25%)
    constexpr float MIN_SUMMARY_MERGE_THRESHOLD = 0.22f;      // Minimum threshold (22%)
    constexpr float MAX_SUMMARY_MERGE_THRESHOLD = 0.30f;      // Maximum threshold (30% - full utilization)
    constexpr float THRESHOLD_ADJUSTMENT_SENSITIVITY = 1.5f;  // Sensitivity to usage patterns
    
    // Rolling average tracking
    constexpr size_t MIN_SAMPLES_FOR_RELIABILITY = 5;         
    constexpr size_t MAX_SAMPLES_TO_TRACK = 50;               
    constexpr int32_t DEFAULT_AI_RESPONSE_SIZE = 150;         
    constexpr int32_t DEFAULT_SUMMARY_SIZE = 200;             
    
    // Usage pattern analysis
    constexpr size_t USAGE_PATTERN_WINDOW = 15;               // Track last 15 interactions
    constexpr float HIGH_USAGE_THRESHOLD = 0.85f;             // High context usage threshold
    constexpr float RAPID_GROWTH_THRESHOLD = 0.025f;          // Rapid growth threshold (2.5% per interaction)
    constexpr float HIGH_VOLATILITY_THRESHOLD = 0.15f;        // High volatility threshold
    
    // Strategy adaptation thresholds
    constexpr float LARGE_RESPONSE_THRESHOLD = 500.0f;        
    constexpr float LARGE_SUMMARY_THRESHOLD = 400.0f;         
    constexpr float VARIABILITY_THRESHOLD = 0.6f;             
    
    // Mathematical validation: Ensure allocations don't exceed safe limits
    static_assert(GLOBAL_EMERGENCY_BUFFER + MAX_TOTAL_SUMMARY_ALLOCATION + 
                  MIN_AI_ALLOCATION + MIN_ACTIVE_CONTENT <= 0.85f, 
                  "Allocation constraints exceed safe limits - total must be <= 85%");
}

// Thread-safe adaptive size tracker with prediction accuracy monitoring
template<typename T>
class AdaptiveSizeTracker {
private:
    mutable std::mutex mutex_;
    std::vector<T> size_samples_;
    T estimated_size_;
    T default_size_;
    std::string tracker_name_;
    
    // Enhanced statistics
    float average_size_ = 0.0f;
    float standard_deviation_ = 0.0f;
    T max_size_ = 0;
    T min_size_ = std::numeric_limits<T>::max();
    
    // Prediction accuracy tracking
    std::vector<float> prediction_errors_;
    float average_prediction_error_ = 0.0f;
    
public:
    explicit AdaptiveSizeTracker(T default_size, const std::string& name) 
        : estimated_size_(default_size), default_size_(default_size), tracker_name_(name) {}
    
    // Thread-safe sample addition with prediction accuracy tracking
    void add_sample(T actual_size, T predicted_size = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (actual_size <= 0) return;
        
        size_samples_.push_back(actual_size);
        
        // Track prediction accuracy if we had a prediction
        if (predicted_size > 0) {
            float error = std::abs(static_cast<float>(actual_size - predicted_size)) / actual_size;
            prediction_errors_.push_back(error);
            
            if (prediction_errors_.size() > ContextSizeConstants::MAX_SAMPLES_TO_TRACK) {
                prediction_errors_.erase(prediction_errors_.begin());
            }
            
            // Update average prediction error
            average_prediction_error_ = std::accumulate(prediction_errors_.begin(), prediction_errors_.end(), 0.0f) / 
                                       prediction_errors_.size();
        }
        
        // Maintain rolling window
        if (size_samples_.size() > ContextSizeConstants::MAX_SAMPLES_TO_TRACK) {
            size_samples_.erase(size_samples_.begin());
        }
        
        update_statistics();
        
        LLAMA_LOG(tracker_name_ + " tracked: " + std::to_string(actual_size) + 
                  " tokens (avg: " + std::to_string(average_size_) + 
                  ", estimated: " + std::to_string(estimated_size_) + ")");
    }
    
    // Get estimated size with dynamic buffer
    T get_estimated_size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return estimated_size_;
    }
    
    // Get required space with safety margin and prediction error compensation
    T get_required_space() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (size_samples_.size() < ContextSizeConstants::MIN_SAMPLES_FOR_RELIABILITY) {
            return static_cast<T>(default_size_ * ContextSizeConstants::DYNAMIC_BUFFER_MULTIPLIER);
        }
        
        // Use average + std deviation for base estimate
        float base_estimate = average_size_ + standard_deviation_;
        
        // Apply dynamic buffer multiplier
        float safe_estimate = base_estimate * ContextSizeConstants::DYNAMIC_BUFFER_MULTIPLIER;
        
        // Compensate for prediction errors
        if (!prediction_errors_.empty() && average_prediction_error_ > 0.1f) {
            safe_estimate *= (1.0f + average_prediction_error_);
        }
        
        return static_cast<T>(safe_estimate);
    }
    
    // Get comprehensive statistics
    struct SizeStats {
        float average;
        float std_deviation;
        T min_size;
        T max_size;
        size_t sample_count;
        bool has_reliable_data;
        float prediction_accuracy;
        T estimated_size;
        T required_space;
    };
    
    SizeStats get_statistics() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return {
            average_size_,
            standard_deviation_,
            min_size_,
            max_size_,
            size_samples_.size(),
            size_samples_.size() >= ContextSizeConstants::MIN_SAMPLES_FOR_RELIABILITY,
            1.0f - average_prediction_error_,
            estimated_size_,
            get_required_space_unsafe()
        };
    }
    
    // Check if patterns suggest heavy allocation
    bool suggests_heavy_allocation() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (size_samples_.size() < ContextSizeConstants::MIN_SAMPLES_FOR_RELIABILITY) {
            return false;
        }
        
        bool has_large_sizes = average_size_ > (tracker_name_.find("AI") != std::string::npos ? 
                                               ContextSizeConstants::LARGE_RESPONSE_THRESHOLD : 
                                               ContextSizeConstants::LARGE_SUMMARY_THRESHOLD);
        
        bool has_high_variability = standard_deviation_ > average_size_ * ContextSizeConstants::VARIABILITY_THRESHOLD;
        
        return has_large_sizes || has_high_variability;
    }
    
private:
    // Internal helper for get_required_space (assumes lock is held)
    T get_required_space_unsafe() const {
        if (size_samples_.size() < ContextSizeConstants::MIN_SAMPLES_FOR_RELIABILITY) {
            return static_cast<T>(default_size_ * ContextSizeConstants::DYNAMIC_BUFFER_MULTIPLIER);
        }
        
        float base_estimate = average_size_ + standard_deviation_;
        float safe_estimate = base_estimate * ContextSizeConstants::DYNAMIC_BUFFER_MULTIPLIER;
        
        if (!prediction_errors_.empty() && average_prediction_error_ > 0.1f) {
            safe_estimate *= (1.0f + average_prediction_error_);
        }
        
        return static_cast<T>(safe_estimate);
    }
    
    void update_statistics() {
        if (size_samples_.empty()) return;
        
        // Calculate average
        average_size_ = static_cast<float>(std::accumulate(size_samples_.begin(), size_samples_.end(), T(0))) / 
                       size_samples_.size();
        
        // Calculate standard deviation
        float variance = 0.0f;
        for (T size : size_samples_) {
            float diff = static_cast<float>(size) - average_size_;
            variance += diff * diff;
        }
        variance /= size_samples_.size();
        standard_deviation_ = std::sqrt(variance);
        
        // Update min/max
        auto [min_it, max_it] = std::minmax_element(size_samples_.begin(), size_samples_.end());
        min_size_ = *min_it;
        max_size_ = *max_it;
        
        // Update estimated size (conservative estimate)
        estimated_size_ = static_cast<T>(average_size_ + standard_deviation_ * 0.5f);
    }
};

// Usage pattern tracker for dynamic threshold calculation
class UsagePatternTracker {
private:
    mutable std::mutex mutex_;
    std::vector<float> recent_context_usage_;
    std::vector<int32_t> recent_ai_responses_;
    std::vector<int32_t> recent_user_inputs_;
    
    // Pattern analysis metrics
    float average_context_growth_per_interaction_ = 0.0f;
    float peak_usage_in_window_ = 0.0f;
    float usage_volatility_ = 0.0f;
    
public:
    void track_interaction(float context_usage, int32_t ai_tokens = 0, int32_t user_tokens = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Add to rolling windows
        recent_context_usage_.push_back(context_usage);
        if (ai_tokens > 0) recent_ai_responses_.push_back(ai_tokens);
        if (user_tokens > 0) recent_user_inputs_.push_back(user_tokens);
        
        // Maintain window size
        if (recent_context_usage_.size() > ContextSizeConstants::USAGE_PATTERN_WINDOW) {
            recent_context_usage_.erase(recent_context_usage_.begin());
        }
        if (recent_ai_responses_.size() > ContextSizeConstants::USAGE_PATTERN_WINDOW) {
            recent_ai_responses_.erase(recent_ai_responses_.begin());
        }
        if (recent_user_inputs_.size() > ContextSizeConstants::USAGE_PATTERN_WINDOW) {
            recent_user_inputs_.erase(recent_user_inputs_.begin());
        }
        
        update_pattern_analysis();
    }
    
    // Calculate dynamic threshold based on usage patterns
    float calculate_dynamic_threshold() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (recent_context_usage_.size() < ContextSizeConstants::MIN_SAMPLES_FOR_RELIABILITY) {
            return ContextSizeConstants::BASE_SUMMARY_MERGE_THRESHOLD;
        }
        
        float base_threshold = ContextSizeConstants::BASE_SUMMARY_MERGE_THRESHOLD;
        float adjustment = 0.0f;
        
        // Factor 1: Context growth rate
        if (average_context_growth_per_interaction_ > ContextSizeConstants::RAPID_GROWTH_THRESHOLD) {
            adjustment -= 0.02f;  // Lower threshold (trigger earlier)
        } else if (average_context_growth_per_interaction_ < 0.01f) {
            adjustment += 0.01f;  // Higher threshold (trigger later)
        }
        
        // Factor 2: Usage volatility
        if (usage_volatility_ > ContextSizeConstants::HIGH_VOLATILITY_THRESHOLD) {
            adjustment -= 0.015f;  // Lower threshold for safety
        } else if (usage_volatility_ < 0.05f) {
            adjustment += 0.01f;   // Higher threshold, more predictable
        }
        
        // Factor 3: Peak usage in recent window
        if (peak_usage_in_window_ > ContextSizeConstants::HIGH_USAGE_THRESHOLD) {
            adjustment -= 0.025f;  // Much lower threshold
        } else if (peak_usage_in_window_ < 0.65f) {
            adjustment += 0.015f;  // Higher threshold
        }
        
        // Apply sensitivity multiplier
        adjustment *= ContextSizeConstants::THRESHOLD_ADJUSTMENT_SENSITIVITY;
        
        // Calculate final threshold with bounds
        float dynamic_threshold = base_threshold + adjustment;
        return std::clamp(dynamic_threshold, 
                         ContextSizeConstants::MIN_SUMMARY_MERGE_THRESHOLD,
                         ContextSizeConstants::MAX_SUMMARY_MERGE_THRESHOLD);
    }
    
    // Get usage pattern statistics
    struct UsagePatternStats {
        float average_context_growth_per_interaction;
        float peak_usage_in_window;
        float usage_volatility;
        float current_dynamic_threshold;
        size_t sample_count;
        bool has_reliable_patterns;
    };
    
    UsagePatternStats get_statistics() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return {
            average_context_growth_per_interaction_,
            peak_usage_in_window_,
            usage_volatility_,
            calculate_dynamic_threshold_unsafe(),
            recent_context_usage_.size(),
            recent_context_usage_.size() >= ContextSizeConstants::MIN_SAMPLES_FOR_RELIABILITY
        };
    }
    
private:
    float calculate_dynamic_threshold_unsafe() const {
        if (recent_context_usage_.size() < ContextSizeConstants::MIN_SAMPLES_FOR_RELIABILITY) {
            return ContextSizeConstants::BASE_SUMMARY_MERGE_THRESHOLD;
        }
        
        float base_threshold = ContextSizeConstants::BASE_SUMMARY_MERGE_THRESHOLD;
        float adjustment = 0.0f;
        
        if (average_context_growth_per_interaction_ > ContextSizeConstants::RAPID_GROWTH_THRESHOLD) {
            adjustment -= 0.02f;
        } else if (average_context_growth_per_interaction_ < 0.01f) {
            adjustment += 0.01f;
        }
        
        if (usage_volatility_ > ContextSizeConstants::HIGH_VOLATILITY_THRESHOLD) {
            adjustment -= 0.015f;
        } else if (usage_volatility_ < 0.05f) {
            adjustment += 0.01f;
        }
        
        if (peak_usage_in_window_ > ContextSizeConstants::HIGH_USAGE_THRESHOLD) {
            adjustment -= 0.025f;
        } else if (peak_usage_in_window_ < 0.65f) {
            adjustment += 0.015f;
        }
        
        adjustment *= ContextSizeConstants::THRESHOLD_ADJUSTMENT_SENSITIVITY;
        
        float dynamic_threshold = base_threshold + adjustment;
        return std::clamp(dynamic_threshold, 
                         ContextSizeConstants::MIN_SUMMARY_MERGE_THRESHOLD,
                         ContextSizeConstants::MAX_SUMMARY_MERGE_THRESHOLD);
    }
    
    void update_pattern_analysis() {
        if (recent_context_usage_.size() < 2) return;
        
        // Calculate average context growth per interaction
        std::vector<float> growth_rates;
        for (size_t i = 1; i < recent_context_usage_.size(); ++i) {
            float growth = recent_context_usage_[i] - recent_context_usage_[i-1];
            growth_rates.push_back(std::max(0.0f, growth));  // Only positive growth
        }
        
        if (!growth_rates.empty()) {
            average_context_growth_per_interaction_ = 
                std::accumulate(growth_rates.begin(), growth_rates.end(), 0.0f) / growth_rates.size();
        }
        
        // Calculate peak usage in current window
        peak_usage_in_window_ = *std::max_element(recent_context_usage_.begin(), recent_context_usage_.end());
        
        // Calculate usage volatility (standard deviation)
        if (recent_context_usage_.size() >= 3) {
            float mean = std::accumulate(recent_context_usage_.begin(), recent_context_usage_.end(), 0.0f) / 
                        recent_context_usage_.size();
            
            float variance = 0.0f;
            for (float usage : recent_context_usage_) {
                float diff = usage - mean;
                variance += diff * diff;
            }
            variance /= recent_context_usage_.size();
            usage_volatility_ = std::sqrt(variance);
        }
    }
};

// Dynamic summary slot manager with 30% hard cap enforcement
class DynamicSummarySlotManager {
private:
    mutable std::mutex mutex_;
    std::vector<int32_t> slot_sizes_;
    int32_t total_summary_tokens_ = 0;
    int32_t context_size_ = 0;
    size_t merge_operations_count_ = 0;
    mutable UsagePatternTracker usage_tracker_;
    
public:    void set_context_size(int32_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        context_size_ = size;
        LLAMA_LOG("DynamicSummarySlotManager: Set context size to " + std::to_string(size));
    }
    
    void track_usage_pattern(float context_usage, int32_t ai_tokens = 0, int32_t user_tokens = 0) const {
        usage_tracker_.track_interaction(context_usage, ai_tokens, user_tokens);
    }
    
    // Check if we need to merge before adding a new summary
    bool needs_merge_before_adding(int32_t new_summary_size) const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (slot_sizes_.size() < ContextSizeConstants::MIN_SUMMARY_SLOTS) {
            return false;
        }
        
        // Check against hard 30% cap first (critical safety check)
        int32_t projected_total = total_summary_tokens_ + new_summary_size;
        float projected_usage = static_cast<float>(projected_total) / context_size_;
        if (projected_usage > ContextSizeConstants::MAX_TOTAL_SUMMARY_ALLOCATION) {
            return true;
        }
        
        // Check against dynamic threshold
        float dynamic_threshold = usage_tracker_.calculate_dynamic_threshold();
        float current_usage = get_usage_percentage_unsafe();
        return current_usage > dynamic_threshold;
    }
      void add_summary_slot(int32_t tokens) {
        std::lock_guard<std::mutex> lock(mutex_);
        slot_sizes_.push_back(tokens);
        total_summary_tokens_ += tokens;
        
        LLAMA_LOG("Added summary slot: " + std::to_string(tokens) + " tokens (total: " + 
                  std::to_string(total_summary_tokens_) + ", usage: " + 
                  std::to_string(get_usage_percentage_unsafe() * 100) + "%)");
    }
    
    bool merge_oldest_slots(int32_t merged_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (slot_sizes_.size() < 2) {
            LLAMA_LOG("Warning: Cannot merge - insufficient slots");
            return false;
        }
        
        int32_t removed_tokens = slot_sizes_[0] + slot_sizes_[1];
        slot_sizes_.erase(slot_sizes_.begin(), slot_sizes_.begin() + 2);
        slot_sizes_.insert(slot_sizes_.begin(), merged_size);
        
        total_summary_tokens_ = total_summary_tokens_ - removed_tokens + merged_size;
        merge_operations_count_++;
        
        LLAMA_LOG("Merged oldest slots: " + std::to_string(removed_tokens) + 
                  " -> " + std::to_string(merged_size) + " tokens (usage: " + 
                  std::to_string(get_usage_percentage_unsafe() * 100) + "%)");
        return true;
    }
    
    // Get current statistics
    struct SummarySlotStats {
        size_t slot_count;
        int32_t total_tokens;
        float usage_percentage;
        float current_dynamic_threshold;
        bool exceeds_hard_cap;
        bool at_minimum_slots;
        bool needs_merge;
        UsagePatternTracker::UsagePatternStats usage_patterns;
    };
    
    SummarySlotStats get_statistics() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        float usage = get_usage_percentage_unsafe();
        float dynamic_threshold = usage_tracker_.calculate_dynamic_threshold();
        bool exceeds_cap = usage > ContextSizeConstants::MAX_TOTAL_SUMMARY_ALLOCATION;
        bool at_minimum = slot_sizes_.size() <= ContextSizeConstants::MIN_SUMMARY_SLOTS;
        bool needs_merge = usage > dynamic_threshold || exceeds_cap;
        
        return {
            slot_sizes_.size(),
            total_summary_tokens_,
            usage,
            dynamic_threshold,
            exceeds_cap,
            at_minimum,
            needs_merge,
            usage_tracker_.get_statistics()
        };
    }
    
    void clear_all_slots() {
        std::lock_guard<std::mutex> lock(mutex_);
        slot_sizes_.clear();
        total_summary_tokens_ = 0;
        merge_operations_count_ = 0;
        LLAMA_LOG("All summary slots cleared");
    }
    
    float get_usage_percentage() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return get_usage_percentage_unsafe();
    }
    
    size_t get_slot_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return slot_sizes_.size();
    }
    
    int32_t get_total_tokens() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return total_summary_tokens_;
    }
    
private:
    float get_usage_percentage_unsafe() const {
        return context_size_ > 0 ? static_cast<float>(total_summary_tokens_) / context_size_ : 0.0f;
    }
};

// Enhanced conversation pattern analyzer
class EnhancedConversationAnalyzer {
private:
    mutable std::mutex mutex_;
    size_t total_messages_ = 0;
    size_t pruning_events_ = 0;
    size_t summary_events_ = 0;
    std::chrono::steady_clock::time_point conversation_start_;
    std::chrono::steady_clock::time_point last_activity_;
    
    // Context pressure tracking
    mutable std::vector<float> context_usage_history_;
    mutable float average_context_pressure_ = 0.0f;
    
public:
    EnhancedConversationAnalyzer() : conversation_start_(std::chrono::steady_clock::now()),
                                    last_activity_(std::chrono::steady_clock::now()) {}
    
    void track_message() {
        std::lock_guard<std::mutex> lock(mutex_);
        total_messages_++;
        last_activity_ = std::chrono::steady_clock::now();
    }
    
    void track_pruning_event() {
        std::lock_guard<std::mutex> lock(mutex_);
        pruning_events_++;
        last_activity_ = std::chrono::steady_clock::now();
    }
    
    void track_summary_event() {
        std::lock_guard<std::mutex> lock(mutex_);
        summary_events_++;
        last_activity_ = std::chrono::steady_clock::now();
    }
    
    void track_context_usage(float usage_percentage) const {
        std::lock_guard<std::mutex> lock(mutex_);
        context_usage_history_.push_back(usage_percentage);
        if (context_usage_history_.size() > 100) {  // Keep last 100 measurements
            context_usage_history_.erase(context_usage_history_.begin());
        }
        
        // Update average context pressure
        average_context_pressure_ = std::accumulate(context_usage_history_.begin(), context_usage_history_.end(), 0.0f) / 
                                   context_usage_history_.size();
    }
    
    ContextStrategy suggest_optimal_strategy(const AdaptiveSizeTracker<int32_t>& ai_tracker,
                                           const AdaptiveSizeTracker<int32_t>& summary_tracker) const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (total_messages_ < 20) {
            return ContextStrategy::BALANCED;  // Default for new conversations
        }
        
        // Analyze patterns
        bool ai_needs_heavy_allocation = ai_tracker.suggests_heavy_allocation();
        bool summary_needs_heavy_allocation = summary_tracker.suggests_heavy_allocation();
        bool high_context_pressure = average_context_pressure_ > 0.8f;
        
        if (ai_needs_heavy_allocation && !summary_needs_heavy_allocation) {
            return ContextStrategy::AI_HEAVY;
        } else if (summary_needs_heavy_allocation && !ai_needs_heavy_allocation) {
            return ContextStrategy::SUMMARY_HEAVY;
        } else if (high_context_pressure) {
            return ContextStrategy::SUMMARY_HEAVY;  // Prioritize summarization under pressure
        } else {
            return ContextStrategy::BALANCED;
        }
    }
    
    struct ConversationMetrics {
        size_t total_messages;
        size_t pruning_events;
        size_t summary_events;
        float average_context_pressure;
        std::chrono::minutes conversation_duration;
        float pruning_frequency;
    };
    
    ConversationMetrics get_metrics() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto duration = std::chrono::duration_cast<std::chrono::minutes>(
            last_activity_ - conversation_start_);
        
        float pruning_frequency = total_messages_ > 0 ? 
            static_cast<float>(pruning_events_) / total_messages_ : 0.0f;
        
        return {
            total_messages_,
            pruning_events_,
            summary_events_,
            average_context_pressure_,
            duration,
            pruning_frequency
        };
    }
};

// Enhanced context analysis structure
struct EnhancedContextAnalysis {
    // Current state
    int32_t context_size = 0;
    int32_t total_used_tokens = 0;
    int32_t summary_tokens = 0;
    int32_t active_history_tokens = 0;
    int32_t available_tokens = 0;
    
    // Dynamic space requirements
    int32_t required_ai_space = 0;
    int32_t required_summary_space = 0;
    int32_t emergency_buffer_space = 0;
    
    // Dynamic allocations (percentages)
    float ai_allocation_percentage = 0.0f;
    float summary_allocation_percentage = 0.0f;
    float active_content_percentage = 0.0f;
    float emergency_buffer_percentage = 0.0f;
    
    // Summary management
    DynamicSummarySlotManager::SummarySlotStats summary_slot_stats;
    
    // Action flags
    bool needs_pruning = false;
    bool needs_summary_merge = false;
    bool emergency_buffer_violated = false;
    bool summary_hard_cap_exceeded = false;
    
    // Strategy recommendations
    bool strategy_change_recommended = false;
    ContextStrategy recommended_strategy = ContextStrategy::BALANCED;
    
    // Performance metrics
    AdaptiveSizeTracker<int32_t>::SizeStats ai_stats;
    AdaptiveSizeTracker<int32_t>::SizeStats summary_stats;
    EnhancedConversationAnalyzer::ConversationMetrics conversation_metrics;
    
    // Efficiency metrics
    float context_utilization_efficiency = 0.0f;
    float prediction_accuracy_score = 0.0f;
};

// Production-ready enhanced context size manager
class EnhancedContextSizeManager {
private:
    mutable std::mutex mutex_;
    ContextStrategy current_strategy_;
    
    // Dual adaptive trackers
    AdaptiveSizeTracker<int32_t> ai_response_tracker_;
    AdaptiveSizeTracker<int32_t> summary_tracker_;
    
    // Enhanced summary slot management with dynamic thresholds
    DynamicSummarySlotManager summary_slot_manager_;
    
    EnhancedConversationAnalyzer conversation_analyzer_;
    std::chrono::steady_clock::time_point last_strategy_review_;
    
public:
    explicit EnhancedContextSizeManager()
        : current_strategy_(ContextStrategy::BALANCED),
          ai_response_tracker_(ContextSizeConstants::DEFAULT_AI_RESPONSE_SIZE, "AI Response"),
          summary_tracker_(ContextSizeConstants::DEFAULT_SUMMARY_SIZE, "Summary"),
          last_strategy_review_(std::chrono::steady_clock::now()) {}
    
    // Set context size for all calculations
    void set_context_size(int32_t context_size) {
        summary_slot_manager_.set_context_size(context_size);
    }
    
    // Track AI response with optional prediction for accuracy monitoring
    void track_ai_response(int32_t actual_tokens, int32_t predicted_tokens = 0) {
        ai_response_tracker_.add_sample(actual_tokens, predicted_tokens);
        conversation_analyzer_.track_message();
        review_strategy();
    }
    
    // Track summary creation with optional prediction
    void track_summary_creation(int32_t actual_tokens, int32_t predicted_tokens = 0) {
        summary_tracker_.add_sample(actual_tokens, predicted_tokens);
        conversation_analyzer_.track_summary_event();
        review_strategy();
    }
    
    // Track user message activity
    void track_user_message() {
        conversation_analyzer_.track_message();
    }
      // Track pruning events
    void track_pruning_event() {
        conversation_analyzer_.track_pruning_event();
    }
    
    // Get estimated AI response size
    int32_t get_estimated_ai_response_size() const {
        return ai_response_tracker_.get_estimated_size();
    }
    
    // Get estimated summary size  
    int32_t get_estimated_summary_size() const {
        return summary_tracker_.get_estimated_size();
    }
    
    // Comprehensive context analysis with all dynamic features
    EnhancedContextAnalysis analyze_context(const ContextInfo& context) const;
    
    // Summary management workflow
    struct SummaryManagementPlan {
        bool can_add_directly;
        bool needs_merge_first;
        size_t merge_cycles_needed;
        std::string action_plan;
        float estimated_final_usage_percentage;
    };
    
    SummaryManagementPlan plan_summary_addition(int32_t estimated_summary_size) const {
        SummaryManagementPlan plan{};
        
        plan.needs_merge_first = summary_slot_manager_.needs_merge_before_adding(estimated_summary_size);
        plan.can_add_directly = !plan.needs_merge_first;
        
        if (plan.can_add_directly) {
            plan.action_plan = "Add summary directly - within limits";
            plan.estimated_final_usage_percentage = 
                (summary_slot_manager_.get_total_tokens() + estimated_summary_size) / 
                static_cast<float>(4096);  // Estimate
        } else {
            plan.merge_cycles_needed = 1;  // Conservative estimate
            plan.action_plan = "Merge required before adding summary";
            
            // Estimate usage after merge (assume 30% reduction)
            int32_t estimated_after_merge = static_cast<int32_t>(summary_slot_manager_.get_total_tokens() * 0.7f);
            plan.estimated_final_usage_percentage = 
                (estimated_after_merge + estimated_summary_size) / static_cast<float>(4096);
        }
        
        return plan;
    }
    
    // Execute summary addition with automatic merge management
    template<typename MergeCallback>
    bool execute_summary_addition(int32_t actual_summary_size, MergeCallback&& merge_callback) {
        auto plan = plan_summary_addition(actual_summary_size);
        
        LLAMA_LOG("Executing summary addition: " + plan.action_plan);
        
        // Perform merge if needed
        if (plan.needs_merge_first) {
            if (summary_slot_manager_.get_slot_count() <= ContextSizeConstants::MIN_SUMMARY_SLOTS) {
                LLAMA_LOG("Warning: Cannot merge - at minimum slot count");
                return false;
            }
            
            // Call the merge callback to get merged summary size
            int32_t merged_size = merge_callback();
            if (merged_size <= 0) {
                LLAMA_LOG("Error: Merge operation failed");
                return false;
            }
            
            if (!summary_slot_manager_.merge_oldest_slots(merged_size)) {
                LLAMA_LOG("Error: Failed to merge summary slots");
                return false;
            }
        }
        
        // Add the new summary
        summary_slot_manager_.add_summary_slot(actual_summary_size);
        
        // Track for learning
        track_summary_creation(actual_summary_size, summary_tracker_.get_estimated_size());
        
        // Verify we're still within hard cap
        auto stats = summary_slot_manager_.get_statistics();
        if (stats.exceeds_hard_cap) {
            LLAMA_LOG("Critical: Still exceeding hard cap after operations - " +
                      std::to_string(stats.usage_percentage * 100) + "% usage");
            return false;
        }
        
        LLAMA_LOG("Summary addition successful - " +
                  std::to_string(stats.usage_percentage * 100) + "% usage (" +
                  std::to_string(stats.slot_count) + " slots)");
        
        return true;
    }
    
    // Get optimization recommendations
    std::vector<std::string> get_optimization_recommendations(const EnhancedContextAnalysis& analysis) const {
        std::vector<std::string> recommendations;
        
        if (analysis.summary_hard_cap_exceeded) {
            recommendations.push_back("CRITICAL: Summary usage exceeds 30% hard cap - immediate merge required");
        }
        
        if (analysis.emergency_buffer_violated) {
            recommendations.push_back("CRITICAL: Emergency buffer violated - immediate pruning required");
        }
        
        if (analysis.summary_slot_stats.needs_merge) {
            recommendations.push_back("Summary merge recommended based on dynamic threshold (" + 
                std::to_string(analysis.summary_slot_stats.current_dynamic_threshold * 100) + "%)");
        }
        
        if (analysis.needs_pruning) {
            recommendations.push_back("Context pruning recommended - insufficient space for AI responses");
        }
        
        if (analysis.strategy_change_recommended) {
            recommendations.push_back("Strategy change recommended: " + strategy_to_string(analysis.recommended_strategy));
        }
        
        // Performance insights
        if (analysis.prediction_accuracy_score < 0.7f) {
            recommendations.push_back("Low prediction accuracy - consider larger safety margins");
        }
        
        if (analysis.context_utilization_efficiency > 0.9f) {
            recommendations.push_back("Excellent context utilization - " + 
                std::to_string(analysis.context_utilization_efficiency * 100) + "% efficiency");
        }
        
        return recommendations;
    }
    
    // Get current strategy
    ContextStrategy get_current_strategy() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_strategy_;
    }
    
    // Get summary statistics
    DynamicSummarySlotManager::SummarySlotStats get_summary_statistics() const {
        return summary_slot_manager_.get_statistics();
    }
    
    // Reset all summary slots
    void reset_summary_slots() {
        summary_slot_manager_.clear_all_slots();
    }
    
private:
    void review_strategy() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto now = std::chrono::steady_clock::now();
        if (now - last_strategy_review_ < std::chrono::minutes(3)) {
            return;  // Review every 3 minutes max
        }
        
        ContextStrategy suggested = conversation_analyzer_.suggest_optimal_strategy(ai_response_tracker_, summary_tracker_);
        if (suggested != current_strategy_) {
            LLAMA_LOG("Auto-strategy adaptation: " + strategy_to_string(current_strategy_) + 
                      " -> " + strategy_to_string(suggested));
            current_strategy_ = suggested;
        }
        
        last_strategy_review_ = now;
    }
    
    static std::string strategy_to_string(ContextStrategy strategy) {
        switch (strategy) {
            case ContextStrategy::BALANCED: return "BALANCED";            case ContextStrategy::AI_HEAVY: return "AI_HEAVY";
            case ContextStrategy::SUMMARY_HEAVY: return "SUMMARY_HEAVY";
            default: return "UNKNOWN";
        }
    }
};

// Implementation note: EnhancedContextSizeManager::analyze_context method 
// implementation is moved to LlamaContext.hpp to avoid circular dependencies

//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODING DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//


