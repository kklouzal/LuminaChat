// ContextSizeManager.hpp - Production-ready adaptive context size management for LLaMA contexts
//
// OPTIMIZED CONTEXT STRATEGY (v2.1 - Stability & Consistency Focus):
// - Consistent thresholds aligned with LlamaManager: 90% max usage, 60% target after pruning
// - Conservative pruning: 60-90% retention ratios to prevent excessive message loss
// - Enhanced prediction accuracy: Increased buffer multipliers (2.0x dynamic, 1.5x conservative)
// - Improved decision logic consistency: Analysis recommendations align with actual decisions
// - Balanced allocation: 20% summaries, 35% active content, 15% AI responses, 3% emergency buffer
// - Each ContextInfo has its own EnhancedContextSizeManager instance for isolation
// - Mathematical validation ensures total allocations ≤ 73% with remaining space for messages
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
    // Core allocation constraints per context (optimized for maximum utilization)
    static constexpr float MAX_TOTAL_SUMMARY_ALLOCATION = 0.20f;     // Hard 20% cap for all summaries per context (reduced for better balance)
    static constexpr float EMERGENCY_BUFFER = 0.03f;                 // 3% emergency buffer per context (reduced for higher utilization)    
    static constexpr float MIN_AI_ALLOCATION = 0.15f;                // Minimum 15% for AI responses per context
    static constexpr float MIN_ACTIVE_CONTENT = 0.35f;               // Minimum 35% for active conversation per context (increased for more messages)    // Enhanced buffer multipliers for more accurate predictions
    static constexpr float DYNAMIC_BUFFER_MULTIPLIER = 2.0f;         // Increased from 1.5x to 2.0x for better safety margins 
    static constexpr float CONSERVATIVE_BUFFER_MULTIPLIER = 1.5f;    // Increased from 1.2x to 1.5x for better prediction accuracy
    // Summary management
    static constexpr size_t MIN_SUMMARY_SLOTS = 3;                   // Minimum slots before merging
    static constexpr float MAX_HISTORY_PER_SUMMARY = 0.08f;          // Max 8% of context per summary creation (reduced for more frequent summaries)
    static constexpr size_t MIN_MESSAGES_BEFORE_SUMMARY = 10;        // Minimum messages that should fit before next summary
    static constexpr size_t TARGET_MESSAGES_BETWEEN_SUMMARIES = 12;  // Target number of messages between summarizations
    
    // Dynamic threshold calculation (optimized for balanced summarization)
    static constexpr float BASE_SUMMARY_MERGE_THRESHOLD = 0.18f;     // Base threshold (18% - reduced for less frequent merging)
    static constexpr float MIN_SUMMARY_MERGE_THRESHOLD = 0.15f;      // Minimum threshold (15%)
    static constexpr float MAX_SUMMARY_MERGE_THRESHOLD = 0.20f;      // Maximum threshold (20% - aligned with hard cap)
    static constexpr float THRESHOLD_ADJUSTMENT_SENSITIVITY = 1.2f;  // Sensitivity to usage patterns (reduced for stability)
    
    // Rolling average tracking
    static constexpr size_t MIN_SAMPLES_FOR_RELIABILITY = 5;         
    static constexpr size_t MAX_SAMPLES_TO_TRACK = 50;                 // Default sizes with more conservative estimates to improve prediction accuracy
    static constexpr int32_t DEFAULT_AI_RESPONSE_SIZE = 200;         // Increased from 150 for better prediction accuracy
    static constexpr int32_t DEFAULT_SUMMARY_SIZE = 250;             // Increased from 200 for better prediction accuracy
    
    // Usage pattern analysis
    static constexpr size_t USAGE_PATTERN_WINDOW = 15;               // Track last 15 interactions
    static constexpr float HIGH_USAGE_THRESHOLD = 0.85f;             // High context usage threshold
    static constexpr float RAPID_GROWTH_THRESHOLD = 0.025f;          // Rapid growth threshold (2.5% per interaction)
    static constexpr float HIGH_VOLATILITY_THRESHOLD = 0.15f;        // High volatility threshold
    
    // Strategy adaptation thresholds
    static constexpr float LARGE_RESPONSE_THRESHOLD = 500.0f;        
    static constexpr float LARGE_SUMMARY_THRESHOLD = 400.0f;         
    static constexpr float VARIABILITY_THRESHOLD = 0.6f;       // Context management ratios (aligned with LlamaManager directives: 90% max, 60% target)
    static constexpr float MAX_CONTEXT_USAGE = 0.90f;              // Maximum 90% usage as specified in LlamaManager directives
    static constexpr float TARGET_CONTEXT_USAGE = 0.60f;           // Target 60% usage after pruning as specified in LlamaManager directives  
    static constexpr float AGGRESSIVE_PRUNING_RATIO = 0.30f;       // Emergency pruning ratio (keep 30% for emergency scenarios)
    
    // Message-based thresholds for balanced summarization
    static constexpr float ESTIMATED_TOKENS_PER_MESSAGE = 75.0f;    // Estimated tokens per user+AI message exchange
    static constexpr float MESSAGE_BUFFER_MULTIPLIER = 1.3f;        // Safety multiplier for message estimation
    
    // Mathematical validation: Ensure allocations don't exceed safe limits per context (updated for higher utilization)
    static_assert(EMERGENCY_BUFFER + MAX_TOTAL_SUMMARY_ALLOCATION + 
                  MIN_AI_ALLOCATION + MIN_ACTIVE_CONTENT <= 0.73f, 
                  "Allocation constraints exceed safe limits - total must be <= 73% per context");
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
        // Reserve capacity to prevent reallocations and potential exceptions
        size_samples_.reserve(ContextSizeConstants::MAX_SAMPLES_TO_TRACK + 5); // Small buffer
        prediction_errors_.reserve(ContextSizeConstants::MAX_SAMPLES_TO_TRACK + 5);
    }
    
    // Thread-safe sample addition with prediction accuracy tracking and enhanced logging
    void add_sample(T actual_size, T predicted_size = 0) noexcept {
        // Fast path: early validation without exceptions
        if (actual_size <= 0) [[unlikely]] {
            DEBUG_IF_ENABLED(CONTEXT_SIZE_MANAGER, {
                CONTEXT_SIZE_LOG_DEBUG(tracker_name_ + ": Invalid sample size ignored: " + std::to_string(actual_size));
            });
            return; // Skip invalid samples silently
        }
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Pre-check capacity to avoid potential reallocation exceptions
        if (size_samples_.size() < size_samples_.capacity()) [[likely]] {
            size_samples_.push_back(actual_size);
        } else {
            // Manual rotation to avoid erase() exceptions
            if (!size_samples_.empty()) {
                std::rotate(size_samples_.begin(), size_samples_.begin() + 1, size_samples_.end());
                size_samples_.back() = actual_size;
            }
        }
        
        // Track prediction accuracy with capacity check
        if (predicted_size > 0 && actual_size > 0) [[likely]] {
            const float error = std::abs(static_cast<float>(actual_size - predicted_size)) / static_cast<float>(actual_size);
            
            if (prediction_errors_.size() < prediction_errors_.capacity()) [[likely]] {
                prediction_errors_.push_back(error);
            } else if (!prediction_errors_.empty()) {
                // Manual rotation instead of erase()
                std::rotate(prediction_errors_.begin(), prediction_errors_.begin() + 1, prediction_errors_.end());
                prediction_errors_.back() = error;
            }
            
            // Update average prediction error safely
            if (!prediction_errors_.empty()) {
                const float sum = std::accumulate(prediction_errors_.cbegin(), prediction_errors_.cend(), 0.0f);
                average_prediction_error_ = sum / static_cast<float>(prediction_errors_.size());
            }
            
            // Log significant prediction errors for model tuning
            if (error > 0.5f) [[unlikely]] {
                DEBUG_IF_ENABLED(CONTEXT_SIZE_MANAGER, {
                    CONTEXT_SIZE_LOG_DEBUG(tracker_name_ + ": Large prediction error: " + 
                             std::to_string(static_cast<int>(error * 100.0f)) + "% (predicted: " + 
                             std::to_string(predicted_size) + ", actual: " + std::to_string(actual_size) + ")");
                });
            }
        }
        
        const T old_estimate = estimated_size_;
        update_statistics();
        
        // Log significant estimate changes for debugging
        if (size_samples_.size() >= ContextSizeConstants::MIN_SAMPLES_FOR_RELIABILITY && old_estimate > 0) [[likely]] {
            const float estimate_change = std::abs(static_cast<float>(estimated_size_ - old_estimate)) / static_cast<float>(old_estimate);
            if (estimate_change > 0.2f) [[unlikely]] {
                DEBUG_IF_ENABLED(CONTEXT_SIZE_MANAGER, {
                    CONTEXT_SIZE_LOG_DEBUG(tracker_name_ + ": Estimate adjusted by " + 
                             std::to_string(static_cast<int>(estimate_change * 100.0f)) + "% (" + 
                             std::to_string(old_estimate) + " -> " + std::to_string(estimated_size_) + ")");
                });
            }
        }
        
        // Periodic detailed logging for debugging sessions
        DEBUG_IF_ENABLED(CONTEXT_SIZE_MANAGER, {
            if (size_samples_.size() % 10 == 0) { // Log every 10th sample to avoid spam
                CONTEXT_SIZE_LOG_DEBUG(tracker_name_ + " sample #" + std::to_string(size_samples_.size()) + 
                          ": " + std::to_string(actual_size) + " tokens (avg: " + 
                          std::to_string(static_cast<int>(average_size_)) + ", est: " + 
                          std::to_string(estimated_size_) + ")");
            }
        });
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
        
        // All operations here are noexcept with proper containers
        const auto sum = std::accumulate(size_samples_.cbegin(), size_samples_.cend(), T(0));
        average_size_ = static_cast<float>(sum) / static_cast<float>(size_samples_.size());
        
        // Calculate standard deviation - this is noexcept since we're using floats
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
        
        // Update min/max - noexcept with iterators
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
    UsagePatternTracker() noexcept {
        // Pre-reserve capacity to prevent reallocations
        recent_context_usage_.reserve(ContextSizeConstants::USAGE_PATTERN_WINDOW + 2);
        recent_ai_responses_.reserve(ContextSizeConstants::USAGE_PATTERN_WINDOW + 2);
        recent_user_inputs_.reserve(ContextSizeConstants::USAGE_PATTERN_WINDOW + 2);
    }
    
    void track_interaction(float context_usage, int32_t ai_tokens = 0, int32_t user_tokens = 0) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Pre-check capacity to avoid reallocation exceptions
        if (recent_context_usage_.size() < ContextSizeConstants::USAGE_PATTERN_WINDOW) [[likely]] {
            recent_context_usage_.push_back(context_usage);
        } else if (!recent_context_usage_.empty()) {
            // Manual rotation instead of erase() to avoid exceptions
            std::rotate(recent_context_usage_.begin(), recent_context_usage_.begin() + 1, recent_context_usage_.end());
            recent_context_usage_.back() = context_usage;
        }
        
        // Same approach for other vectors
        if (ai_tokens > 0) [[likely]] {
            if (recent_ai_responses_.size() < ContextSizeConstants::USAGE_PATTERN_WINDOW) {
                recent_ai_responses_.push_back(ai_tokens);
            } else if (!recent_ai_responses_.empty()) {
                std::rotate(recent_ai_responses_.begin(), recent_ai_responses_.begin() + 1, recent_ai_responses_.end());
                recent_ai_responses_.back() = ai_tokens;
            }
        }
        
        if (user_tokens > 0) [[likely]] {
            if (recent_user_inputs_.size() < ContextSizeConstants::USAGE_PATTERN_WINDOW) {
                recent_user_inputs_.push_back(user_tokens);
            } else if (!recent_user_inputs_.empty()) {
                std::rotate(recent_user_inputs_.begin(), recent_user_inputs_.begin() + 1, recent_user_inputs_.end());
                recent_user_inputs_.back() = user_tokens;
            }
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
        
        // Factor 1: Context growth rate (adjusted for less frequent summarization)
        if (average_context_growth_per_interaction_ > ContextSizeConstants::RAPID_GROWTH_THRESHOLD) [[unlikely]] {
            adjustment -= 0.015f;  // Lower threshold (trigger earlier) - reduced from 0.02f
        } else if (average_context_growth_per_interaction_ < 0.01f) [[likely]] {
            adjustment += 0.015f;  // Higher threshold (trigger later) - increased from 0.01f
        }
        
        // Factor 2: Usage volatility (adjusted for stability)
        if (usage_volatility_ > ContextSizeConstants::HIGH_VOLATILITY_THRESHOLD) [[unlikely]] {
            adjustment -= 0.01f;  // Lower threshold for safety - reduced from 0.015f
        } else if (usage_volatility_ < 0.05f) [[likely]] {
            adjustment += 0.015f;   // Higher threshold, more predictable - increased from 0.01f
        }
        
        // Factor 3: Peak usage in recent window (adjusted for higher utilization targets)
        if (peak_usage_in_window_ > ContextSizeConstants::HIGH_USAGE_THRESHOLD) [[unlikely]] {
            adjustment -= 0.02f;  // Much lower threshold - reduced from 0.025f
        } else if (peak_usage_in_window_ < 0.70f) [[likely]] {  // Increased from 0.65f
            adjustment += 0.02f;  // Higher threshold - increased from 0.015f
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
            adjustment -= 0.015f;  // Lower threshold (trigger earlier) - reduced from 0.02f
        } else if (average_context_growth_per_interaction_ < 0.01f) [[likely]] {
            adjustment += 0.015f;  // Higher threshold (trigger later) - increased from 0.01f
        }
        
        if (usage_volatility_ > ContextSizeConstants::HIGH_VOLATILITY_THRESHOLD) [[unlikely]] {
            adjustment -= 0.01f;  // Lower threshold for safety - reduced from 0.015f
        } else if (usage_volatility_ < 0.05f) [[likely]] {
            adjustment += 0.015f;   // Higher threshold, more predictable - increased from 0.01f
        }
        
        if (peak_usage_in_window_ > ContextSizeConstants::HIGH_USAGE_THRESHOLD) [[unlikely]] {
            adjustment -= 0.02f;  // Much lower threshold - reduced from 0.025f
        } else if (peak_usage_in_window_ < 0.70f) [[likely]] {  // Increased from 0.65f
            adjustment += 0.02f;  // Higher threshold - increased from 0.015f
        }
        
        adjustment *= ContextSizeConstants::THRESHOLD_ADJUSTMENT_SENSITIVITY;
        
        const float dynamic_threshold = base_threshold + adjustment;
        return std::clamp(dynamic_threshold, 
                         ContextSizeConstants::MIN_SUMMARY_MERGE_THRESHOLD,
                         ContextSizeConstants::MAX_SUMMARY_MERGE_THRESHOLD);    
    }
    
    void update_pattern_analysis() noexcept {
        if (recent_context_usage_.size() < 2) [[unlikely]] return;
        
        // Reserve capacity for growth_rates to prevent allocations
        std::vector<float> growth_rates;
        growth_rates.reserve(recent_context_usage_.size());
        
        // Calculate growth rates without exceptions
        for (size_t i = 1; i < recent_context_usage_.size(); ++i) {
            const float growth = std::max(0.0f, recent_context_usage_[i] - recent_context_usage_[i-1]);
            if (growth_rates.size() < growth_rates.capacity()) {
                growth_rates.push_back(growth);
            }
        }
        
        if (!growth_rates.empty()) [[likely]] {
            const float sum = std::accumulate(growth_rates.cbegin(), growth_rates.cend(), 0.0f);
            average_context_growth_per_interaction_ = sum / static_cast<float>(growth_rates.size());
        }
        
        // Calculate peak usage - noexcept operation
        if (!recent_context_usage_.empty()) {
            peak_usage_in_window_ = *std::max_element(recent_context_usage_.cbegin(), recent_context_usage_.cend());
        }
        
        // Calculate usage volatility (standard deviation) - all noexcept operations
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
    DynamicSummarySlotManager() noexcept {
        // Pre-reserve capacity to prevent reallocations and exceptions
        slot_sizes_.reserve(10); // Reasonable initial capacity
    }
    
    void set_context_size(int32_t size) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        const int32_t old_size = context_size_;
        context_size_ = size;
        
        // Always log context size changes as they're critical for debugging
        if (old_size != size) {
            CONTEXT_SIZE_LOG("DynamicSummarySlotManager: Context size changed from " + 
                           std::to_string(old_size) + " to " + std::to_string(size) + " tokens");
        }
        
        // Log current state for debugging
        DEBUG_IF_ENABLED(CONTEXT_SIZE_MANAGER, {
            const float usage = get_usage_percentage_unsafe();
            CONTEXT_SIZE_LOG_DEBUG("Context size set: " + std::to_string(size) + 
                                 " tokens, current usage: " + std::to_string(static_cast<int>(usage * 100)) + "%");
        });
    }
    
    void track_usage_pattern(float context_usage, int32_t ai_tokens = 0, int32_t user_tokens = 0) const noexcept {
        usage_tracker_.track_interaction(context_usage, ai_tokens, user_tokens);
    }
    
    // Check if we need to merge before adding a new summary (enhanced with message-based logic)
    [[nodiscard]] bool needs_merge_before_adding(int32_t new_summary_size) const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (slot_sizes_.size() < ContextSizeConstants::MIN_SUMMARY_SLOTS) [[likely]] {
            return false;
        }
        
        // Check against hard 20% cap first (critical safety check)
        const int32_t projected_total = total_summary_tokens_ + new_summary_size;
        const float projected_usage = static_cast<float>(projected_total) / context_size_;
        if (projected_usage > ContextSizeConstants::MAX_TOTAL_SUMMARY_ALLOCATION) [[unlikely]] {
            CONTEXT_SIZE_LOG("Merge required: projected usage " + 
                           std::to_string(static_cast<int>(projected_usage * 100)) + 
                           "% exceeds hard cap " + 
                           std::to_string(static_cast<int>(ContextSizeConstants::MAX_TOTAL_SUMMARY_ALLOCATION * 100)) + "%");
            return true;
        }
        
        // Enhanced check: Ensure we have room for target number of messages before next summary
        const float remaining_context = 1.0f - projected_usage;
        const float estimated_space_for_messages = remaining_context - ContextSizeConstants::MIN_AI_ALLOCATION - ContextSizeConstants::EMERGENCY_BUFFER;
        const float messages_that_fit = (estimated_space_for_messages * context_size_) / ContextSizeConstants::ESTIMATED_TOKENS_PER_MESSAGE;
        
        if (messages_that_fit < ContextSizeConstants::MIN_MESSAGES_BEFORE_SUMMARY) [[unlikely]] {
            DEBUG_IF_ENABLED(CONTEXT_SIZE_MANAGER, {
                CONTEXT_SIZE_LOG_DEBUG("Merge required: insufficient message space - " + 
                                      std::to_string(static_cast<int>(messages_that_fit)) + 
                                      " messages fit (min: " + 
                                      std::to_string(ContextSizeConstants::MIN_MESSAGES_BEFORE_SUMMARY) + ")");
            });
            return true; // Need to merge to make room for adequate message exchanges
        }
        
        // Check against dynamic threshold (secondary check)
        const float dynamic_threshold = usage_tracker_.calculate_dynamic_threshold();
        const float current_usage = get_usage_percentage_unsafe();
        const bool threshold_exceeded = current_usage > dynamic_threshold;
        
        if (threshold_exceeded) {
            DEBUG_IF_ENABLED(CONTEXT_SIZE_MANAGER, {
                CONTEXT_SIZE_LOG_DEBUG("Merge required: usage " + 
                                      std::to_string(static_cast<int>(current_usage * 100)) + 
                                      "% exceeds dynamic threshold " + 
                                      std::to_string(static_cast<int>(dynamic_threshold * 100)) + "%");
            });
        }
        
        return threshold_exceeded;
    }
    
    void add_summary_slot(int32_t tokens) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Pre-check capacity to avoid reallocation
        if (slot_sizes_.size() < slot_sizes_.capacity()) [[likely]] {
            slot_sizes_.push_back(tokens);
            total_summary_tokens_ += tokens;
        } else {
            // Fallback: still add but may allocate
            slot_sizes_.push_back(tokens);
            total_summary_tokens_ += tokens;
        }
        
        // Always log summary slot additions as they're important for debugging
        const float usage = get_usage_percentage_unsafe();
        CONTEXT_SIZE_LOG_DEBUG("Added summary slot: " + std::to_string(tokens) + 
                              " tokens (total: " + std::to_string(total_summary_tokens_) + 
                              ", usage: " + std::to_string(static_cast<int>(usage * 100)) + "%, " +
                              std::to_string(slot_sizes_.size()) + " slots)");
        
        // Warn if approaching hard cap
        if (usage > ContextSizeConstants::MAX_TOTAL_SUMMARY_ALLOCATION * 0.85f) [[unlikely]] {
            CONTEXT_SIZE_LOG("Warning: Summary usage approaching hard cap: " + 
                           std::to_string(static_cast<int>(usage * 100)) + "% of context");
        }
    }
    
    [[nodiscard]] bool merge_oldest_slots(int32_t merged_size) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (slot_sizes_.size() < 2) [[unlikely]] {
            CONTEXT_SIZE_LOG_ERROR("Cannot merge summary slots - insufficient slots (count: " + 
                                  std::to_string(slot_sizes_.size()) + ")");
            return false;
        }
        
        const int32_t removed_tokens = slot_sizes_[0] + slot_sizes_[1];
        const float old_usage = get_usage_percentage_unsafe();
        
        // More efficient: modify in place rather than erase+insert
        slot_sizes_[0] = merged_size;
        if (slot_sizes_.size() > 1) {
            slot_sizes_.erase(slot_sizes_.begin() + 1);
        }
        
        total_summary_tokens_ = total_summary_tokens_ - removed_tokens + merged_size;
        merge_operations_count_++;
        
        const float new_usage = get_usage_percentage_unsafe();
        
        // Always log merge operations as they're critical for debugging
        CONTEXT_SIZE_LOG("Merged oldest summary slots: " + std::to_string(removed_tokens) + 
                        " -> " + std::to_string(merged_size) + " tokens " +
                        "(usage: " + std::to_string(static_cast<int>(old_usage * 100)) + "% -> " +
                        std::to_string(static_cast<int>(new_usage * 100)) + "%, " +
                        std::to_string(slot_sizes_.size()) + " slots remaining)");
        
        // Log performance metrics for merge operations
        DEBUG_IF_ENABLED(CONTEXT_SIZE_MANAGER, {
            const int compression_ratio = removed_tokens > 0 ? 
                static_cast<int>((static_cast<float>(merged_size) / removed_tokens) * 100) : 100;
            CONTEXT_SIZE_LOG_DEBUG("Merge efficiency: " + std::to_string(compression_ratio) + 
                                  "% compression, operation #" + std::to_string(merge_operations_count_));
        });
        
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
        const size_t cleared_slots = slot_sizes_.size();
        const int32_t cleared_tokens = total_summary_tokens_;
        
        slot_sizes_.clear();
        total_summary_tokens_ = 0;
        merge_operations_count_ = 0;
        
        // Always log clearing operations as they're significant events
        CONTEXT_SIZE_LOG("Cleared all summary slots: " + std::to_string(cleared_slots) + 
                        " slots, " + std::to_string(cleared_tokens) + " tokens freed");
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
        std::lock_guard<std::mutex> lock(mutex_);
        return total_summary_tokens_;
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
        context_usage_history_.reserve(100);  // Pre-allocate to prevent exceptions
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
        
        // Use capacity check to avoid reallocation
        if (context_usage_history_.size() < context_usage_history_.capacity()) [[likely]] {
            context_usage_history_.push_back(usage_percentage);
        } else {
            // Manual rotation instead of erase() to avoid exceptions
            std::rotate(context_usage_history_.begin(), context_usage_history_.begin() + 1, context_usage_history_.end());
            context_usage_history_.back() = usage_percentage;
        }
        
        // Update average context pressure - noexcept operation
        if (!context_usage_history_.empty()) {
            const float sum = std::accumulate(context_usage_history_.cbegin(), context_usage_history_.cend(), 0.0f);
            average_context_pressure_ = sum / static_cast<float>(context_usage_history_.size());
        }
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
        
        // Log significant AI responses for debugging
        DEBUG_IF_ENABLED(CONTEXT_SIZE_MANAGER, {
            if (actual_tokens > ContextSizeConstants::LARGE_RESPONSE_THRESHOLD) {
                CONTEXT_SIZE_LOG_DEBUG("Large AI response tracked: " + std::to_string(actual_tokens) + 
                                      " tokens (threshold: " + std::to_string(static_cast<int>(ContextSizeConstants::LARGE_RESPONSE_THRESHOLD)) + ")");
            }
        });
        
        review_strategy();
    }
    
    // Track summary creation with optional prediction
    void track_summary_creation(const int32_t actual_tokens, const int32_t predicted_tokens = 0) noexcept {
        summary_tracker_.add_sample(actual_tokens, predicted_tokens);
        conversation_analyzer_.track_summary_event();
        
        // Log summary creation for debugging
        DEBUG_IF_ENABLED(CONTEXT_SIZE_MANAGER, {
            CONTEXT_SIZE_LOG_DEBUG("Summary creation tracked: " + std::to_string(actual_tokens) + " tokens" +
                                  (predicted_tokens > 0 ? " (predicted: " + std::to_string(predicted_tokens) + ")" : ""));
        });
        
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
        float estimated_final_usage_percentage;
    };
    
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
        
        return plan;
    }
    
    // Execute summary addition with automatic merge management
    template<typename MergeCallback>
    [[nodiscard]] bool execute_summary_addition(const int32_t actual_summary_size, MergeCallback&& merge_callback) noexcept {
        const auto plan = plan_summary_addition(actual_summary_size);
        
        // Log the execution plan for debugging
        CONTEXT_SIZE_LOG("Executing summary addition: " + plan.action_plan + 
                        " (size: " + std::to_string(actual_summary_size) + " tokens)");
        
        // Perform merge if needed
        if (plan.needs_merge_first) [[unlikely]] {
            if (summary_slot_manager_.get_slot_count() <= ContextSizeConstants::MIN_SUMMARY_SLOTS) [[unlikely]] {
                CONTEXT_SIZE_LOG_ERROR("Cannot merge - at minimum slot count (" + 
                                      std::to_string(ContextSizeConstants::MIN_SUMMARY_SLOTS) + ")");
                return false;
            }
            
            // Call the merge callback to get merged summary size
            const int32_t merged_size = merge_callback();
            if (merged_size <= 0) [[unlikely]] {
                CONTEXT_SIZE_LOG_ERROR("Merge operation failed - invalid merged size: " + 
                                      std::to_string(merged_size));
                return false;
            }
            
            if (!summary_slot_manager_.merge_oldest_slots(merged_size)) [[unlikely]] {
                CONTEXT_SIZE_LOG_ERROR("Failed to merge summary slots");
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
            CONTEXT_SIZE_LOG_ERROR("CRITICAL: Still exceeding hard cap after operations - " +
                      std::to_string(static_cast<int>(stats.usage_percentage * 100)) + "% usage " +
                      "(" + std::to_string(stats.total_tokens) + "/" + 
                      std::to_string(static_cast<int>(stats.total_tokens / stats.usage_percentage)) + " tokens)");
            return false;
        }
        
        // Log successful completion with metrics
        CONTEXT_SIZE_LOG("Summary addition successful - " +
                  std::to_string(static_cast<int>(stats.usage_percentage * 100)) + "% usage (" +
                  std::to_string(stats.slot_count) + " slots, " + 
                  std::to_string(stats.total_tokens) + " tokens)");
        
        return true;
    }
    
    // Get optimization recommendations
    [[nodiscard]] std::vector<std::string> get_optimization_recommendations(const EnhancedContextAnalysis& analysis) const {
        std::vector<std::string> recommendations;
        recommendations.reserve(8);  // Pre-allocate for common case
        
        if (analysis.summary_hard_cap_exceeded) [[unlikely]] {
            const std::string critical_msg = "CRITICAL: Summary usage exceeds hard cap - immediate merge required";
            recommendations.emplace_back(critical_msg);
            CONTEXT_SIZE_LOG_ERROR(critical_msg);
        }
        
        if (analysis.emergency_buffer_violated) [[unlikely]] {
            const std::string critical_msg = "CRITICAL: Emergency buffer violated - immediate pruning required";
            recommendations.emplace_back(critical_msg);
            CONTEXT_SIZE_LOG_ERROR(critical_msg);
        }
        
        if (analysis.summary_slot_stats.needs_merge) [[likely]] {
            const std::string merge_message = "Summary merge recommended based on dynamic threshold (" + 
                std::to_string(static_cast<int>(analysis.summary_slot_stats.current_dynamic_threshold * 100)) + "%)";
            recommendations.emplace_back(merge_message);
            DEBUG_IF_ENABLED(CONTEXT_SIZE_MANAGER, {
                CONTEXT_SIZE_LOG_DEBUG(merge_message);
            });
        }
        
        if (analysis.needs_pruning) [[unlikely]] {
            const std::string pruning_msg = "Context pruning recommended - insufficient space for AI responses";
            recommendations.emplace_back(pruning_msg);
            CONTEXT_SIZE_LOG(pruning_msg);
        }
        
        if (analysis.strategy_change_recommended) [[unlikely]] {
            const std::string strategy_message = std::string("Strategy change recommended: ") + 
                                                strategy_to_string(analysis.recommended_strategy);
            recommendations.emplace_back(strategy_message);
            CONTEXT_SIZE_LOG(strategy_message);
        }
        
        // Performance insights
        if (analysis.prediction_accuracy_score < 0.7f) [[unlikely]] {
            const std::string accuracy_msg = "Low prediction accuracy (" + 
                std::to_string(static_cast<int>(analysis.prediction_accuracy_score * 100)) + 
                "%) - consider larger safety margins";
            recommendations.emplace_back(accuracy_msg);
            CONTEXT_SIZE_LOG(accuracy_msg);
        }
        
        if (analysis.context_utilization_efficiency > 0.9f) [[likely]] {
            const std::string efficiency_message = "Excellent context utilization - " + 
                std::to_string(static_cast<int>(analysis.context_utilization_efficiency * 100)) + "% efficiency";
            recommendations.emplace_back(efficiency_message);
            DEBUG_IF_ENABLED(CONTEXT_SIZE_MANAGER, {
                CONTEXT_SIZE_LOG_DEBUG(efficiency_message);
            });
        }
        
        // Log summary of recommendations for debugging
        if (!recommendations.empty()) {
            DEBUG_IF_ENABLED(CONTEXT_SIZE_MANAGER, {
                CONTEXT_SIZE_LOG_DEBUG("Generated " + std::to_string(recommendations.size()) + 
                                      " optimization recommendations");
            });
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
        summary_slot_manager_.clear_all_slots();
    }
    
    // Clear all summary slots (alias for reset_summary_slots)
    void clear_summary_slots() noexcept {
        summary_slot_manager_.clear_all_slots();
    }
    
    // Add a summary slot directly (for synchronization)
    void add_summary_slot_direct(int32_t tokens) noexcept {
        summary_slot_manager_.add_summary_slot(tokens);
    }
    
    // Periodic health check and diagnostics logging
    void log_health_check() const noexcept {
        const auto stats = summary_slot_manager_.get_statistics();
        const auto ai_stats = ai_response_tracker_.get_statistics();
        const auto summary_stats = summary_tracker_.get_statistics();
        const auto conv_metrics = conversation_analyzer_.get_metrics();
        
        CONTEXT_SIZE_LOG("=== Context Size Manager Health Check ===");
        CONTEXT_SIZE_LOG("Strategy: " + std::string(strategy_to_string(current_strategy_)));
        CONTEXT_SIZE_LOG("Summary usage: " + std::to_string(static_cast<int>(stats.usage_percentage * 100)) + 
                        "% (" + std::to_string(stats.slot_count) + " slots, " + 
                        std::to_string(stats.total_tokens) + " tokens)");
        CONTEXT_SIZE_LOG("AI responses: avg=" + std::to_string(static_cast<int>(ai_stats.average)) + 
                        " tokens, samples=" + std::to_string(ai_stats.sample_count));
        CONTEXT_SIZE_LOG("Summaries: avg=" + std::to_string(static_cast<int>(summary_stats.average)) + 
                        " tokens, samples=" + std::to_string(summary_stats.sample_count));
        CONTEXT_SIZE_LOG("Conversation: " + std::to_string(conv_metrics.total_messages) + 
                        " messages, " + std::to_string(conv_metrics.pruning_events) + 
                        " pruning events, " + std::to_string(conv_metrics.summary_events) + " summaries");
        
        // Warn about potential issues
        if (stats.usage_percentage > 0.15f) {
            CONTEXT_SIZE_LOG("Info: Summary usage above 15% - monitoring for merge needs");
        }
        if (ai_stats.prediction_accuracy < 0.8f && ai_stats.sample_count > 5) {
            CONTEXT_SIZE_LOG("Warning: AI response prediction accuracy low (" + 
                           std::to_string(static_cast<int>(ai_stats.prediction_accuracy * 100)) + "%)");
        }
        if (conv_metrics.pruning_frequency > 0.1f) {
            CONTEXT_SIZE_LOG("Info: High pruning frequency (" + 
                           std::to_string(static_cast<int>(conv_metrics.pruning_frequency * 100)) + "%)");
        }
        
        CONTEXT_SIZE_LOG("========================================");
    }
    
    // Atomic synchronization with LlamaSummarizer - rebuilds tracking from authoritative source
    template<typename SlotContainer>
    void sync_with_authoritative_slots(const SlotContainer& authoritative_slots) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Store current state for comparison
        auto old_stats = summary_slot_manager_.get_statistics();
        
        // Clear and rebuild from authoritative source atomically
        summary_slot_manager_.clear_all_slots();
        
        size_t synced_count = 0;
        int32_t total_estimated_tokens = 0;
        
        for (const auto& slot : authoritative_slots) {
            if (!slot.empty()) {
                // Estimate tokens for this slot (using same logic as LlamaSummarizer)
                constexpr float CHARS_PER_TOKEN = 4.0f;
                const size_t slot_length = slot.length();
                int32_t estimated = static_cast<int32_t>(slot_length / CHARS_PER_TOKEN);
                estimated = static_cast<int32_t>(estimated * 1.1f); // 10% buffer
                estimated = std::clamp(estimated, 10, 1000); // Same bounds as LlamaSummarizer
                
                summary_slot_manager_.add_summary_slot(estimated);
                total_estimated_tokens += estimated;
                synced_count++;
            }
        }
        
        auto new_stats = summary_slot_manager_.get_statistics();
        
        // Always log synchronization operations as they're critical for state consistency
        CONTEXT_SIZE_LOG("Atomic slot synchronization completed:");
        CONTEXT_SIZE_LOG("  Previous: " + std::to_string(old_stats.slot_count) + " slots, " + 
                  std::to_string(old_stats.total_tokens) + " tokens");
        CONTEXT_SIZE_LOG("  Current: " + std::to_string(new_stats.slot_count) + " slots, " + 
                  std::to_string(new_stats.total_tokens) + " tokens");
        CONTEXT_SIZE_LOG("  Synced: " + std::to_string(synced_count) + " non-empty slots");
        
        // Warn about significant changes
        const int32_t token_diff = new_stats.total_tokens - old_stats.total_tokens;
        if (std::abs(token_diff) > 100) [[unlikely]] {
            const std::string sign = (token_diff > 0) ? "+" : "";
            CONTEXT_SIZE_LOG("Significant token count change during sync: " + 
                           sign + std::to_string(token_diff) + " tokens");
        }
        
        // Debug information about sync accuracy
        DEBUG_IF_ENABLED(CONTEXT_SIZE_MANAGER, {
            if (synced_count > 0) {
                const int avg_tokens_per_slot = total_estimated_tokens / static_cast<int>(synced_count);
                CONTEXT_SIZE_LOG_DEBUG("Sync details: avg " + std::to_string(avg_tokens_per_slot) + 
                                      " tokens/slot, usage: " + 
                                      std::to_string(static_cast<int>(new_stats.usage_percentage * 100)) + "%");
            }
        });
    }
    
private:
void review_strategy() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        
        const auto now = std::chrono::steady_clock::now();
        if (now - last_strategy_review_ < std::chrono::minutes(3)) [[likely]] {
            return;  // Review every 3 minutes max
        }
        
        const ContextStrategy old_strategy = current_strategy_;
        const ContextStrategy suggested = conversation_analyzer_.suggest_optimal_strategy(ai_response_tracker_, summary_tracker_);
        
        if (suggested != current_strategy_) [[unlikely]] {
            current_strategy_ = suggested;
            
            // Always log strategy changes as they're important for debugging behavior
            CONTEXT_SIZE_LOG("Auto-strategy adaptation: " + 
                           std::string(strategy_to_string(old_strategy)) + " -> " + 
                           std::string(strategy_to_string(suggested)));
            
            // Log the reasoning behind the strategy change
            DEBUG_IF_ENABLED(CONTEXT_SIZE_MANAGER, {
                const auto ai_stats = ai_response_tracker_.get_statistics();
                const auto summary_stats = summary_tracker_.get_statistics();
                const auto conv_metrics = conversation_analyzer_.get_metrics();
                
                CONTEXT_SIZE_LOG_DEBUG("Strategy change reasoning:");
                CONTEXT_SIZE_LOG_DEBUG("  AI responses: avg=" + std::to_string(static_cast<int>(ai_stats.average)) + 
                                      ", heavy_alloc=" + (ai_stats.average > ContextSizeConstants::LARGE_RESPONSE_THRESHOLD ? "yes" : "no"));
                CONTEXT_SIZE_LOG_DEBUG("  Summaries: avg=" + std::to_string(static_cast<int>(summary_stats.average)) + 
                                      ", heavy_alloc=" + (summary_stats.average > ContextSizeConstants::LARGE_SUMMARY_THRESHOLD ? "yes" : "no"));
                CONTEXT_SIZE_LOG_DEBUG("  Context pressure: " + std::to_string(static_cast<int>(conv_metrics.average_context_pressure * 100)) + "%");
            });
        }
        
        last_strategy_review_ = now;
    }
    
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