// ContextSizeManager.hpp - Production-ready adaptive context size management for LLaMA contexts
//
// ISOLATED CONTEXT STRATEGY:
// - Each ContextInfo has its own EnhancedContextSizeManager instance
// - Dynamic tracking of AI response sizes with 1.5x buffer per context
// - Dynamic tracking of summary sizes with 1.5x buffer per context
// - Hard 30% cap for summary allocation with dynamic threshold management per context
// - 10% emergency buffer for prediction overruns per context
// - Minimum 3 summary slots with intelligent merging per context
// - Truly adaptive space allocation based on rolling averages per context
// - Mathematical validation of all allocation constraints per context
//
// INTEGRATION: Functions implemented in LlamaContext.hpp to avoid circular dependencies
// Usage: initialize_context_size_manager(), analyze_context_usage(), track_user_message(), track_ai_response()

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
#include <optional>
#include <unordered_set>
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

// Mathematically validated context size management configuration for isolated contexts
namespace ContextSizeConstants {
    // Core allocation constraints per context (mathematically validated)
    static constexpr float MAX_TOTAL_SUMMARY_ALLOCATION = 0.30f;     // Hard 30% cap for all summaries per context
    static constexpr float EMERGENCY_BUFFER = 0.10f;                 // 10% emergency buffer per context
    static constexpr float MIN_AI_ALLOCATION = 0.15f;                // Minimum 15% for AI responses per context
    static constexpr float MIN_ACTIVE_CONTENT = 0.25f;               // Minimum 25% for active conversation per context
    static constexpr float DYNAMIC_BUFFER_MULTIPLIER = 1.5f;         // 1.5x multiplier for predictions per context
    
    // Summary management
    static constexpr size_t MIN_SUMMARY_SLOTS = 3;                   // Minimum slots before merging
    static constexpr float MAX_HISTORY_PER_SUMMARY = 0.10f;          // Max 10% of context per summary creation
      // Dynamic threshold calculation (mathematically bounded)
    static constexpr float BASE_SUMMARY_MERGE_THRESHOLD = 0.25f;     // Base threshold (25%)
    static constexpr float MIN_SUMMARY_MERGE_THRESHOLD = 0.22f;      // Minimum threshold (22%)
    static constexpr float MAX_SUMMARY_MERGE_THRESHOLD = 0.30f;      // Maximum threshold (30% - full utilization)
    static constexpr float THRESHOLD_ADJUSTMENT_SENSITIVITY = 1.5f;  // Sensitivity to usage patterns
      // Rolling average tracking
    static constexpr size_t MIN_SAMPLES_FOR_RELIABILITY = 5;         
    static constexpr size_t MAX_SAMPLES_TO_TRACK = 50;               
    static constexpr int32_t DEFAULT_AI_RESPONSE_SIZE = 150;         
    static constexpr int32_t DEFAULT_SUMMARY_SIZE = 200;             
    
    // Usage pattern analysis
    static constexpr size_t USAGE_PATTERN_WINDOW = 15;               // Track last 15 interactions
    static constexpr float HIGH_USAGE_THRESHOLD = 0.85f;             // High context usage threshold
    static constexpr float RAPID_GROWTH_THRESHOLD = 0.025f;          // Rapid growth threshold (2.5% per interaction)
    static constexpr float HIGH_VOLATILITY_THRESHOLD = 0.15f;        // High volatility threshold
      // Strategy adaptation thresholds
    static constexpr float LARGE_RESPONSE_THRESHOLD = 500.0f;        
    static constexpr float LARGE_SUMMARY_THRESHOLD = 400.0f;         
    static constexpr float VARIABILITY_THRESHOLD = 0.6f;             
    
    // Context management ratios (moved from SummarizerConstants to avoid circular dependency)
    static constexpr float MAX_CONTEXT_USAGE = 0.90f;
    static constexpr float TARGET_CONTEXT_USAGE = 0.60f;
    static constexpr float AGGRESSIVE_PRUNING_RATIO = 0.30f;
      // Mathematical validation: Ensure allocations don't exceed safe limits per context
    static_assert(EMERGENCY_BUFFER + MAX_TOTAL_SUMMARY_ALLOCATION + 
                  MIN_AI_ALLOCATION + MIN_ACTIVE_CONTENT <= 0.85f, 
                  "Allocation constraints exceed safe limits - total must be <= 85% per context");
}

// Thread-safe adaptive size tracker with prediction accuracy monitoring for isolated contexts
template<typename T>
class AdaptiveSizeTracker {
private:
    mutable std::mutex mutex_;
    std::vector<T> size_samples_;
    T estimated_size_;
    T default_size_;
    std::string tracker_name_;
    
    // Enhanced statistics for this context's tracker
    float average_size_ = 0.0f;
    float standard_deviation_ = 0.0f;
    T max_size_ = 0;
    T min_size_ = std::numeric_limits<T>::max();
    
    // Prediction accuracy tracking for this context
    std::vector<float> prediction_errors_;
    float average_prediction_error_ = 0.0f;
public:
    explicit AdaptiveSizeTracker(T default_size, const std::string& name) noexcept
        : estimated_size_(default_size), default_size_(default_size), tracker_name_(name) {
        size_samples_.reserve(ContextSizeConstants::MAX_SAMPLES_TO_TRACK);
        prediction_errors_.reserve(ContextSizeConstants::MAX_SAMPLES_TO_TRACK);
    }
      // Thread-safe sample addition with prediction accuracy tracking
    void add_sample(T actual_size, T predicted_size = 0) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (actual_size <= 0) [[unlikely]] return;
        
        size_samples_.push_back(actual_size);
        
        // Track prediction accuracy if we had a prediction
        if (predicted_size > 0) [[likely]] {
            const float error = std::abs(static_cast<float>(actual_size - predicted_size)) / static_cast<float>(actual_size);
            prediction_errors_.push_back(error);
            
            if (prediction_errors_.size() > ContextSizeConstants::MAX_SAMPLES_TO_TRACK) [[unlikely]] {
                prediction_errors_.erase(prediction_errors_.begin());
            }
            
            // Update average prediction error using STL algorithm (optimized)
            const float sum = std::accumulate(prediction_errors_.cbegin(), prediction_errors_.cend(), 0.0f);
            average_prediction_error_ = sum / static_cast<float>(prediction_errors_.size());
        }
        
        // Maintain rolling window
        if (size_samples_.size() > ContextSizeConstants::MAX_SAMPLES_TO_TRACK) [[unlikely]] {
            size_samples_.erase(size_samples_.begin());        }
        
        update_statistics();
        
        LLAMA_LOG(tracker_name_ + " tracked: " + std::to_string(actual_size) + 
                  " tokens (avg: " + std::to_string(average_size_) + 
                  ", estimated: " + std::to_string(estimated_size_) + ")");
    }
    
    // Get estimated size with dynamic buffer
    [[nodiscard]] inline T get_estimated_size() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return estimated_size_;    }
    
    // Get required space with safety margin and prediction error compensation
    [[nodiscard]] inline T get_required_space() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return get_required_space_unsafe();
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
        T required_space;    };
    
    [[nodiscard]] SizeStats get_statistics() const noexcept {
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
    [[nodiscard]] bool suggests_heavy_allocation() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (size_samples_.size() < ContextSizeConstants::MIN_SAMPLES_FOR_RELIABILITY) [[unlikely]] {
            return false;
        }
        
        const bool has_large_sizes = average_size_ > (tracker_name_.find("AI") != std::string::npos ? 
                                               ContextSizeConstants::LARGE_RESPONSE_THRESHOLD : 
                                               ContextSizeConstants::LARGE_SUMMARY_THRESHOLD);
        
        const bool has_high_variability = standard_deviation_ > (average_size_ * ContextSizeConstants::VARIABILITY_THRESHOLD);
          return has_large_sizes || has_high_variability;
    }
    
private:
    // Internal helper for get_required_space (assumes lock is held)
    [[nodiscard]] inline T get_required_space_unsafe() const noexcept {
        if (size_samples_.size() < ContextSizeConstants::MIN_SAMPLES_FOR_RELIABILITY) [[unlikely]] {
            return static_cast<T>(default_size_ * ContextSizeConstants::DYNAMIC_BUFFER_MULTIPLIER);
        }
        
        const float base_estimate = average_size_ + standard_deviation_;
        float safe_estimate = base_estimate * ContextSizeConstants::DYNAMIC_BUFFER_MULTIPLIER;
        
        if (!prediction_errors_.empty() && average_prediction_error_ > 0.1f) [[unlikely]] {
            safe_estimate *= (1.0f + average_prediction_error_);
        }
          return static_cast<T>(safe_estimate);
    }
    
    void update_statistics() noexcept {
        if (size_samples_.empty()) [[unlikely]] return;
        
        // Calculate average using STL algorithm (optimized)
        const auto sum = std::accumulate(size_samples_.cbegin(), size_samples_.cend(), T(0));
        average_size_ = static_cast<float>(sum) / static_cast<float>(size_samples_.size());
        
        // Calculate standard deviation using STL algorithm (optimized)
        const float variance = std::transform_reduce(
            size_samples_.cbegin(), size_samples_.cend(),
            0.0f,
            std::plus<float>{},
            [this](const T size) noexcept {
                const float diff = static_cast<float>(size) - average_size_;
                return diff * diff;
            }
        ) / static_cast<float>(size_samples_.size());
        standard_deviation_ = std::sqrt(variance);
        
        // Update min/max using STL algorithm
        const auto [min_it, max_it] = std::minmax_element(size_samples_.cbegin(), size_samples_.cend());
        min_size_ = *min_it;
        max_size_ = *max_it;
        
        // Update estimated size (conservative estimate)
        estimated_size_ = static_cast<T>(average_size_ + (standard_deviation_ * 0.5f));
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
    void track_interaction(float context_usage, int32_t ai_tokens = 0, int32_t user_tokens = 0) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Add to rolling windows
        recent_context_usage_.push_back(context_usage);
        if (ai_tokens > 0) [[likely]] recent_ai_responses_.push_back(ai_tokens);
        if (user_tokens > 0) [[likely]] recent_user_inputs_.push_back(user_tokens);
      // Maintain window size using efficient erase
        if (recent_context_usage_.size() > ContextSizeConstants::USAGE_PATTERN_WINDOW) [[unlikely]] {
            recent_context_usage_.erase(recent_context_usage_.begin());
        }
        if (recent_ai_responses_.size() > ContextSizeConstants::USAGE_PATTERN_WINDOW) [[unlikely]] {
            recent_ai_responses_.erase(recent_ai_responses_.begin());
        }
        if (recent_user_inputs_.size() > ContextSizeConstants::USAGE_PATTERN_WINDOW) [[unlikely]] {
            recent_user_inputs_.erase(recent_user_inputs_.begin());
        }
        
        update_pattern_analysis();
    }
    
    // Calculate dynamic threshold based on usage patterns
    [[nodiscard]] float calculate_dynamic_threshold() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (recent_context_usage_.size() < ContextSizeConstants::MIN_SAMPLES_FOR_RELIABILITY) [[unlikely]] {
            return ContextSizeConstants::BASE_SUMMARY_MERGE_THRESHOLD;
        }
        
        constexpr float base_threshold = ContextSizeConstants::BASE_SUMMARY_MERGE_THRESHOLD;
        float adjustment = 0.0f;
        
        // Factor 1: Context growth rate
        if (average_context_growth_per_interaction_ > ContextSizeConstants::RAPID_GROWTH_THRESHOLD) [[unlikely]] {
            adjustment -= 0.02f;  // Lower threshold (trigger earlier)
        } else if (average_context_growth_per_interaction_ < 0.01f) [[likely]] {
            adjustment += 0.01f;  // Higher threshold (trigger later)
        }
        
        // Factor 2: Usage volatility
        if (usage_volatility_ > ContextSizeConstants::HIGH_VOLATILITY_THRESHOLD) [[unlikely]] {
            adjustment -= 0.015f;  // Lower threshold for safety
        } else if (usage_volatility_ < 0.05f) [[likely]] {
            adjustment += 0.01f;   // Higher threshold, more predictable
        }
        
        // Factor 3: Peak usage in recent window
        if (peak_usage_in_window_ > ContextSizeConstants::HIGH_USAGE_THRESHOLD) [[unlikely]] {
            adjustment -= 0.025f;  // Much lower threshold
        } else if (peak_usage_in_window_ < 0.65f) [[likely]] {
            adjustment += 0.015f;  // Higher threshold
        }
        
        // Apply sensitivity multiplier
        adjustment *= ContextSizeConstants::THRESHOLD_ADJUSTMENT_SENSITIVITY;
        
        // Calculate final threshold with bounds
        const float dynamic_threshold = base_threshold + adjustment;
        return std::clamp(dynamic_threshold, 
                         ContextSizeConstants::MIN_SUMMARY_MERGE_THRESHOLD,
                         ContextSizeConstants::MAX_SUMMARY_MERGE_THRESHOLD);    }
    
    // Get usage pattern statistics
    struct UsagePatternStats {
        float average_context_growth_per_interaction;
        float peak_usage_in_window;
        float usage_volatility;
        float current_dynamic_threshold;
        size_t sample_count;
        bool has_reliable_patterns;
    };
    
    [[nodiscard]] UsagePatternStats get_statistics() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return {
            average_context_growth_per_interaction_,
            peak_usage_in_window_,
            usage_volatility_,
            calculate_dynamic_threshold_unsafe(),
            recent_context_usage_.size(),
            recent_context_usage_.size() >= ContextSizeConstants::MIN_SAMPLES_FOR_RELIABILITY        };
    }
    
private:
    [[nodiscard]] inline float calculate_dynamic_threshold_unsafe() const noexcept {
        if (recent_context_usage_.size() < ContextSizeConstants::MIN_SAMPLES_FOR_RELIABILITY) [[unlikely]] {
            return ContextSizeConstants::BASE_SUMMARY_MERGE_THRESHOLD;
        }
        
        constexpr float base_threshold = ContextSizeConstants::BASE_SUMMARY_MERGE_THRESHOLD;
        float adjustment = 0.0f;
        
        if (average_context_growth_per_interaction_ > ContextSizeConstants::RAPID_GROWTH_THRESHOLD) [[unlikely]] {
            adjustment -= 0.02f;
        } else if (average_context_growth_per_interaction_ < 0.01f) [[likely]] {
            adjustment += 0.01f;
        }
        
        if (usage_volatility_ > ContextSizeConstants::HIGH_VOLATILITY_THRESHOLD) [[unlikely]] {
            adjustment -= 0.015f;
        } else if (usage_volatility_ < 0.05f) [[likely]] {
            adjustment += 0.01f;
        }
        
        if (peak_usage_in_window_ > ContextSizeConstants::HIGH_USAGE_THRESHOLD) [[unlikely]] {
            adjustment -= 0.025f;
        } else if (peak_usage_in_window_ < 0.65f) [[likely]] {
            adjustment += 0.015f;
        }
        
        adjustment *= ContextSizeConstants::THRESHOLD_ADJUSTMENT_SENSITIVITY;
        
        const float dynamic_threshold = base_threshold + adjustment;
        return std::clamp(dynamic_threshold, 
                         ContextSizeConstants::MIN_SUMMARY_MERGE_THRESHOLD,
                         ContextSizeConstants::MAX_SUMMARY_MERGE_THRESHOLD);    }
    
    void update_pattern_analysis() noexcept {
        if (recent_context_usage_.size() < 2) [[unlikely]] return;
        
        // Calculate average context growth per interaction using STL algorithms (optimized)
        std::vector<float> growth_rates;
        growth_rates.reserve(recent_context_usage_.size() - 1);
        
        std::transform(
            recent_context_usage_.cbegin() + 1, recent_context_usage_.cend(),
            recent_context_usage_.cbegin(),
            std::back_inserter(growth_rates),
            [](const float current, const float previous) noexcept {
                return std::max(0.0f, current - previous);  // Only positive growth
            }
        );
        
        if (!growth_rates.empty()) [[likely]] {
            const float sum = std::accumulate(growth_rates.cbegin(), growth_rates.cend(), 0.0f);
            average_context_growth_per_interaction_ = sum / static_cast<float>(growth_rates.size());
        }
        
        // Calculate peak usage in current window using STL algorithm
        peak_usage_in_window_ = *std::max_element(recent_context_usage_.cbegin(), recent_context_usage_.cend());
        
        // Calculate usage volatility (standard deviation) - optimized
        if (recent_context_usage_.size() >= 3) [[likely]] {
            const float mean = std::accumulate(recent_context_usage_.cbegin(), recent_context_usage_.cend(), 0.0f) / 
                              static_cast<float>(recent_context_usage_.size());
            
            const float variance = std::transform_reduce(
                recent_context_usage_.cbegin(), recent_context_usage_.cend(),
                0.0f,
                std::plus<float>{},
                [mean](const float usage) noexcept {
                    const float diff = usage - mean;
                    return diff * diff;
                }
            ) / static_cast<float>(recent_context_usage_.size());
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
    
public:
    void set_context_size(int32_t size) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        context_size_ = size;
        LLAMA_LOG("DynamicSummarySlotManager: Set context size to " + std::to_string(size));
    }
    
    void track_usage_pattern(float context_usage, int32_t ai_tokens = 0, int32_t user_tokens = 0) const noexcept {
        usage_tracker_.track_interaction(context_usage, ai_tokens, user_tokens);
    }
    
    // Check if we need to merge before adding a new summary
    [[nodiscard]] bool needs_merge_before_adding(int32_t new_summary_size) const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (slot_sizes_.size() < ContextSizeConstants::MIN_SUMMARY_SLOTS) [[likely]] {
            return false;
        }
        
        // Check against hard 30% cap first (critical safety check)
        const int32_t projected_total = total_summary_tokens_ + new_summary_size;
        const float projected_usage = static_cast<float>(projected_total) / context_size_;
        if (projected_usage > ContextSizeConstants::MAX_TOTAL_SUMMARY_ALLOCATION) [[unlikely]] {
            return true;
        }
        
        // Check against dynamic threshold
        const float dynamic_threshold = usage_tracker_.calculate_dynamic_threshold();
        const float current_usage = get_usage_percentage_unsafe();
        return current_usage > dynamic_threshold;
    }
    
    void add_summary_slot(int32_t tokens) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        slot_sizes_.push_back(tokens);
        total_summary_tokens_ += tokens;
        
        LLAMA_LOG("Added summary slot: " + std::to_string(tokens) + " tokens (total: " + 
                  std::to_string(total_summary_tokens_) + ", usage: " + 
                  std::to_string(get_usage_percentage_unsafe() * 100) + "%)");
    }
    
    [[nodiscard]] bool merge_oldest_slots(int32_t merged_size) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (slot_sizes_.size() < 2) [[unlikely]] {
            LLAMA_LOG("Warning: Cannot merge - insufficient slots");
            return false;
        }
        
        const int32_t removed_tokens = slot_sizes_[0] + slot_sizes_[1];
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
        UsagePatternTracker::UsagePatternStats usage_patterns;    };
    
    [[nodiscard]] SummarySlotStats get_statistics() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        
        const float usage = get_usage_percentage_unsafe();
        const float dynamic_threshold = usage_tracker_.calculate_dynamic_threshold();
        const bool exceeds_cap = usage > ContextSizeConstants::MAX_TOTAL_SUMMARY_ALLOCATION;
        const bool at_minimum = slot_sizes_.size() <= ContextSizeConstants::MIN_SUMMARY_SLOTS;
        const bool needs_merge = usage > dynamic_threshold || exceeds_cap;
        
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
    
    void clear_all_slots() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        slot_sizes_.clear();
        total_summary_tokens_ = 0;
        merge_operations_count_ = 0;
        LLAMA_LOG("All summary slots cleared");
    }
    
    [[nodiscard]] float get_usage_percentage() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return get_usage_percentage_unsafe();
    }
    
    [[nodiscard]] size_t get_slot_count() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return slot_sizes_.size();
    }
    
    [[nodiscard]] int32_t get_total_tokens() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);        return total_summary_tokens_;
    }
    
private:
    [[nodiscard]] inline float get_usage_percentage_unsafe() const noexcept {
        return context_size_ > 0 ? static_cast<float>(total_summary_tokens_) / static_cast<float>(context_size_) : 0.0f;
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
    EnhancedConversationAnalyzer() noexcept : conversation_start_(std::chrono::steady_clock::now()),
                                    last_activity_(std::chrono::steady_clock::now()) {
        context_usage_history_.reserve(100);  // Pre-allocate for performance
    }
    
    void track_message() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        total_messages_++;
        last_activity_ = std::chrono::steady_clock::now();
    }
    
    void track_pruning_event() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        pruning_events_++;
        last_activity_ = std::chrono::steady_clock::now();
    }
    
    void track_summary_event() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        summary_events_++;
        last_activity_ = std::chrono::steady_clock::now();
    }
    
    void track_context_usage(float usage_percentage) const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        context_usage_history_.push_back(usage_percentage);
        if (context_usage_history_.size() > 100) [[unlikely]] {  // Keep last 100 measurements
            context_usage_history_.erase(context_usage_history_.begin());
        }
      // Update average context pressure using STL algorithm (optimized)
        const float sum = std::accumulate(context_usage_history_.cbegin(), context_usage_history_.cend(), 0.0f);
        average_context_pressure_ = sum / static_cast<float>(context_usage_history_.size());
    }
    
    [[nodiscard]] ContextStrategy suggest_optimal_strategy(const AdaptiveSizeTracker<int32_t>& ai_tracker,
                                           const AdaptiveSizeTracker<int32_t>& summary_tracker) const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (total_messages_ < 20) [[unlikely]] {
            return ContextStrategy::BALANCED;  // Default for new conversations
        }
        
        // Analyze patterns
        const bool ai_needs_heavy_allocation = ai_tracker.suggests_heavy_allocation();
        const bool summary_needs_heavy_allocation = summary_tracker.suggests_heavy_allocation();
        const bool high_context_pressure = average_context_pressure_ > 0.8f;
        
        if (ai_needs_heavy_allocation && !summary_needs_heavy_allocation) [[unlikely]] {
            return ContextStrategy::AI_HEAVY;
        } else if (summary_needs_heavy_allocation && !ai_needs_heavy_allocation) [[unlikely]] {
            return ContextStrategy::SUMMARY_HEAVY;
        } else if (high_context_pressure) [[unlikely]] {
            return ContextStrategy::SUMMARY_HEAVY;  // Prioritize summarization under pressure
        } else [[likely]] {
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
    
    [[nodiscard]] ConversationMetrics get_metrics() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        
        const auto duration = std::chrono::duration_cast<std::chrono::minutes>(
            last_activity_ - conversation_start_);
        
        const float pruning_frequency = total_messages_ > 0 ? 
            static_cast<float>(pruning_events_) / static_cast<float>(total_messages_) : 0.0f;
        
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
    explicit EnhancedContextSizeManager() noexcept
        : current_strategy_(ContextStrategy::BALANCED),
          ai_response_tracker_(ContextSizeConstants::DEFAULT_AI_RESPONSE_SIZE, "AI Response"),
          summary_tracker_(ContextSizeConstants::DEFAULT_SUMMARY_SIZE, "Summary"), last_strategy_review_(std::chrono::steady_clock::now()) {}
    
    // Set context size for all calculations
    void set_context_size(const int32_t context_size) noexcept {
        summary_slot_manager_.set_context_size(context_size);
    }
    
    // Track AI response with optional prediction for accuracy monitoring
    void track_ai_response(const int32_t actual_tokens, const int32_t predicted_tokens = 0) noexcept {
        ai_response_tracker_.add_sample(actual_tokens, predicted_tokens);
        conversation_analyzer_.track_message();
        review_strategy();
    }
    
    // Track summary creation with optional prediction
    void track_summary_creation(const int32_t actual_tokens, const int32_t predicted_tokens = 0) noexcept {
        summary_tracker_.add_sample(actual_tokens, predicted_tokens);
        conversation_analyzer_.track_summary_event();
        review_strategy();
    }
    
    // Track user message activity
    void track_user_message() noexcept {
        conversation_analyzer_.track_message();
    }
    
    // Track pruning events
    void track_pruning_event() noexcept {
        conversation_analyzer_.track_pruning_event();
    }
    
    // Get estimated AI response size
    [[nodiscard]] int32_t get_estimated_ai_response_size() const noexcept {
        return ai_response_tracker_.get_estimated_size();
    }
    
    // Get estimated summary size  
    [[nodiscard]] int32_t get_estimated_summary_size() const noexcept {
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
        float estimated_final_usage_percentage;    };
    
    [[nodiscard]] SummaryManagementPlan plan_summary_addition(const int32_t estimated_summary_size) const noexcept {
        SummaryManagementPlan plan{};
        
        plan.needs_merge_first = summary_slot_manager_.needs_merge_before_adding(estimated_summary_size);
        plan.can_add_directly = !plan.needs_merge_first;
        
        if (plan.can_add_directly) [[likely]] {
            plan.action_plan = "Add summary directly - within limits";
            plan.estimated_final_usage_percentage = 
                static_cast<float>(summary_slot_manager_.get_total_tokens() + estimated_summary_size) / 
                static_cast<float>(4096);  // Estimate
        } else [[unlikely]] {
            plan.merge_cycles_needed = 1;  // Conservative estimate
            plan.action_plan = "Merge required before adding summary";
            
            // Estimate usage after merge (assume 30% reduction)
            const int32_t estimated_after_merge = static_cast<int32_t>(static_cast<float>(summary_slot_manager_.get_total_tokens()) * 0.7f);
            plan.estimated_final_usage_percentage = 
                static_cast<float>(estimated_after_merge + estimated_summary_size) / static_cast<float>(4096);
        }
        
        return plan;    }
    
    // Execute summary addition with automatic merge management
    template<typename MergeCallback>
    [[nodiscard]] bool execute_summary_addition(const int32_t actual_summary_size, MergeCallback&& merge_callback) noexcept {
        const auto plan = plan_summary_addition(actual_summary_size);
        
        LLAMA_LOG("Executing summary addition: " + plan.action_plan);
        
        // Perform merge if needed
        if (plan.needs_merge_first) [[unlikely]] {
            if (summary_slot_manager_.get_slot_count() <= ContextSizeConstants::MIN_SUMMARY_SLOTS) [[unlikely]] {
                LLAMA_LOG("Warning: Cannot merge - at minimum slot count");
                return false;
            }
            
            // Call the merge callback to get merged summary size
            const int32_t merged_size = merge_callback();
            if (merged_size <= 0) [[unlikely]] {
                LLAMA_LOG("Error: Merge operation failed");
                return false;
            }
            
            if (!summary_slot_manager_.merge_oldest_slots(merged_size)) [[unlikely]] {
                LLAMA_LOG("Error: Failed to merge summary slots");
                return false;
            }
        }
        
        // Add the new summary
        summary_slot_manager_.add_summary_slot(actual_summary_size);
        
        // Track for learning
        track_summary_creation(actual_summary_size, summary_tracker_.get_estimated_size());
        
        // Verify we're still within hard cap
        const auto stats = summary_slot_manager_.get_statistics();
        if (stats.exceeds_hard_cap) [[unlikely]] {
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
    [[nodiscard]] std::vector<std::string> get_optimization_recommendations(const EnhancedContextAnalysis& analysis) const {
        std::vector<std::string> recommendations;
        recommendations.reserve(8);  // Pre-allocate for common case
        
        if (analysis.summary_hard_cap_exceeded) [[unlikely]] {
            recommendations.emplace_back("CRITICAL: Summary usage exceeds 30% hard cap - immediate merge required");
        }
        
        if (analysis.emergency_buffer_violated) [[unlikely]] {
            recommendations.emplace_back("CRITICAL: Emergency buffer violated - immediate pruning required");        }
        
        if (analysis.summary_slot_stats.needs_merge) [[likely]] {
            const std::string merge_message = "Summary merge recommended based on dynamic threshold (" + 
                std::to_string(analysis.summary_slot_stats.current_dynamic_threshold * 100) + "%)";
            recommendations.emplace_back(merge_message);
        }
        
        if (analysis.needs_pruning) [[unlikely]] {
            recommendations.emplace_back("Context pruning recommended - insufficient space for AI responses");
        }
        
        if (analysis.strategy_change_recommended) [[unlikely]] {
            const std::string strategy_message = std::string("Strategy change recommended: ") + 
                                                strategy_to_string(analysis.recommended_strategy);
            recommendations.emplace_back(strategy_message);
        }
        
        // Performance insights
        if (analysis.prediction_accuracy_score < 0.7f) [[unlikely]] {
            recommendations.emplace_back("Low prediction accuracy - consider larger safety margins");        }
        
        if (analysis.context_utilization_efficiency > 0.9f) [[likely]] {
            const std::string efficiency_message = "Excellent context utilization - " + 
                std::to_string(analysis.context_utilization_efficiency * 100) + "% efficiency";
            recommendations.emplace_back(efficiency_message);
        }
        
        return recommendations;
    }
    
    // Get current strategy
    [[nodiscard]] ContextStrategy get_current_strategy() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_strategy_;
    }
    
    // Get summary statistics
    [[nodiscard]] DynamicSummarySlotManager::SummarySlotStats get_summary_statistics() const noexcept {
        return summary_slot_manager_.get_statistics();
    }
    
    // Reset all summary slots
    void reset_summary_slots() noexcept {
        summary_slot_manager_.clear_all_slots();    }
    
private:
    void review_strategy() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        
        const auto now = std::chrono::steady_clock::now();
        if (now - last_strategy_review_ < std::chrono::minutes(3)) [[likely]] {
            return;  // Review every 3 minutes max
        }
        
        const ContextStrategy suggested = conversation_analyzer_.suggest_optimal_strategy(ai_response_tracker_, summary_tracker_);
        
        if (suggested != current_strategy_) [[unlikely]] {
            const std::string log_message = std::string("Auto-strategy adaptation: ") + 
                                          strategy_to_string(current_strategy_) + 
                                          " -> " + strategy_to_string(suggested);
            LLAMA_LOG(log_message);
            current_strategy_ = suggested;
        }
        
        last_strategy_review_ = now;    }
    
    [[nodiscard]] static constexpr const char* strategy_to_string(const ContextStrategy strategy) noexcept {
        switch (strategy) {
            case ContextStrategy::BALANCED: return "BALANCED";
            case ContextStrategy::AI_HEAVY: return "AI_HEAVY";
            case ContextStrategy::SUMMARY_HEAVY: return "SUMMARY_HEAVY";
            default: return "UNKNOWN";
        }
    }
};

// INTEGRATION NOTES:
// - Each ContextInfo instance has its own EnhancedContextSizeManager via std::unique_ptr
// - No shared/global context management - all operations are isolated per context
// - EnhancedContextSizeManager::analyze_context() implementation is in LlamaContext.hpp to avoid circular dependencies
// - Integration functions (initialize_context_size_manager, analyze_context_usage, track_user_message, track_ai_response) 
//   are implemented as inline functions in LlamaContext.hpp and called by other components
// - Thread safety: Each context's ContextSizeManager is protected by its own mutexes
// - Memory management: RAII through std::unique_ptr ownership in ContextInfo

//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODING DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//


