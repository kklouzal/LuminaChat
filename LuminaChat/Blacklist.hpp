#pragma once

#include <string>
#include <vector>
#include <unordered_set>
#include <mutex>
#include <functional>
#include <algorithm>
#include <cctype>
#include "LogHandler.hpp"

/**
 * @class Blacklist
 * @brief Thread-safe blacklist management for AI responses with retroactive cleanup capabilities
 * 
 * This header-only class provides a comprehensive blacklist system for filtering AI responses.
 * Key features:
 * - Thread-safe operations using std::mutex
 * - Case-insensitive pattern matching (both exact and partial)
 * - Retroactive cleanup via callback mechanism
 * - Batch operations for efficiency
 * - Manual cleanup control for advanced scenarios
 * 
 * The class uses callbacks to integrate with external systems (like LlamaManager) for
 * retroactive cleanup without creating tight coupling.
 */
class Blacklist {
public:
    // Callback type for retroactive cleanup operations
    // Parameters: pattern (the blacklist pattern that was added)
    using RetroactiveCleanupCallback = std::function<void(const std::string&)>;

private:
    // Thread-safe container for blacklisted patterns
    std::unordered_set<std::string> blacklisted_patterns;
    mutable std::mutex blacklist_mutex;
    
    // Callback for performing retroactive cleanup when new patterns are added
    RetroactiveCleanupCallback cleanup_callback;

    /**
     * @brief Helper method to check if a response matches a specific blacklist pattern
     * @param response The response text to check
     * @param pattern The blacklist pattern to match against
     * @return true if the response matches the pattern (case-insensitive)
     */
    bool response_matches_pattern(const std::string& response, const std::string& pattern) const {
        if (response.empty() || pattern.empty()) return false;
        
        // Check for exact match
        if (response == pattern) {
            return true;
        }
        
        // Check for partial match (case-insensitive)
        std::string lower_response = response;
        std::string lower_pattern = pattern;
        std::transform(lower_response.begin(), lower_response.end(), lower_response.begin(), ::tolower);
        std::transform(lower_pattern.begin(), lower_pattern.end(), lower_pattern.begin(), ::tolower);
        
        return lower_response.find(lower_pattern) != std::string::npos;
    }

public:
    /**
     * @brief Constructor
     * @param cleanup_callback Optional callback for retroactive cleanup operations
     */
    explicit Blacklist(RetroactiveCleanupCallback cleanup_callback = nullptr)
        : cleanup_callback(std::move(cleanup_callback)) {}

    /**
     * @brief Destructor - ensures proper cleanup
     */
    ~Blacklist() = default;

    // Copy constructor and assignment operator (thread-safe)
    Blacklist(const Blacklist& other) {
        std::lock_guard<std::mutex> lock(other.blacklist_mutex);
        blacklisted_patterns = other.blacklisted_patterns;
        cleanup_callback = other.cleanup_callback;
    }

    Blacklist& operator=(const Blacklist& other) {
        if (this != &other) {
            std::lock(blacklist_mutex, other.blacklist_mutex);
            std::lock_guard<std::mutex> lock1(blacklist_mutex, std::adopt_lock);
            std::lock_guard<std::mutex> lock2(other.blacklist_mutex, std::adopt_lock);
            
            blacklisted_patterns = other.blacklisted_patterns;
            cleanup_callback = other.cleanup_callback;
        }
        return *this;
    }

    // Move constructor and assignment operator
    Blacklist(Blacklist&& other) noexcept {
        std::lock_guard<std::mutex> lock(other.blacklist_mutex);
        blacklisted_patterns = std::move(other.blacklisted_patterns);
        cleanup_callback = std::move(other.cleanup_callback);
    }

    Blacklist& operator=(Blacklist&& other) noexcept {
        if (this != &other) {
            std::lock(blacklist_mutex, other.blacklist_mutex);
            std::lock_guard<std::mutex> lock1(blacklist_mutex, std::adopt_lock);
            std::lock_guard<std::mutex> lock2(other.blacklist_mutex, std::adopt_lock);
            
            blacklisted_patterns = std::move(other.blacklisted_patterns);
            cleanup_callback = std::move(other.cleanup_callback);
        }
        return *this;
    }

    /**
     * @brief Set or update the retroactive cleanup callback
     * @param callback The callback function to invoke for retroactive cleanup
     */
    void set_cleanup_callback(RetroactiveCleanupCallback callback) {
        std::lock_guard<std::mutex> lock(blacklist_mutex);
        cleanup_callback = std::move(callback);
    }

    /**
     * @brief Check if a response is blacklisted
     * @param response The response text to check
     * @return true if the response matches any blacklisted pattern
     */
    bool is_blacklisted(const std::string& response) const {
        if (response.empty()) return false;
        
        std::lock_guard<std::mutex> lock(blacklist_mutex);
        
        // Check for exact match
        if (blacklisted_patterns.find(response) != blacklisted_patterns.end()) {
            return true;
        }
        
        // Check for partial matches (case-insensitive)
        std::string lower_response = response;
        std::transform(lower_response.begin(), lower_response.end(), lower_response.begin(), ::tolower);
        
        for (const auto& pattern : blacklisted_patterns) {
            std::string lower_pattern = pattern;
            std::transform(lower_pattern.begin(), lower_pattern.end(), lower_pattern.begin(), ::tolower);
            
            // Check if blacklisted phrase is contained in the response
            if (lower_response.find(lower_pattern) != std::string::npos) {
                return true;
            }
        }
        
        return false;
    }

    /**
     * @brief Check if a response matches a specific pattern (for external use)
     * @param response The response text to check
     * @param pattern The pattern to match against
     * @return true if the response matches the pattern
     */
    bool matches_pattern(const std::string& response, const std::string& pattern) const {
        return response_matches_pattern(response, pattern);
    }

    /**
     * @brief Add a pattern to the blacklist with automatic retroactive cleanup
     * @param pattern The pattern to add to the blacklist
     * @return true if the pattern was newly added, false if it already existed
     */
    bool add_pattern(const std::string& pattern) {
        if (pattern.empty()) return false;
        
        // Check if this pattern is already blacklisted to avoid redundant cleanup
        bool pattern_already_exists = false;
        {
            std::lock_guard<std::mutex> lock(blacklist_mutex);
            auto result = blacklisted_patterns.insert(pattern);
            pattern_already_exists = !result.second;
            
            if (!pattern_already_exists) {
                LLAMA_LOG("Added pattern to blacklist: '" + 
                          (pattern.length() > 50 ? pattern.substr(0, 50) + "..." : pattern) + "'");
            } else {
                LLAMA_LOG("Pattern already blacklisted, skipping: '" + 
                          (pattern.length() > 50 ? pattern.substr(0, 50) + "..." : pattern) + "'");
                return false;
            }
        }
        
        // Perform retroactive cleanup if callback is available
        if (cleanup_callback) {
            LLAMA_LOG("Performing retroactive cleanup for newly blacklisted pattern");
            cleanup_callback(pattern);
        }
        
        return true;
    }

    /**
     * @brief Add a pattern to the blacklist without retroactive cleanup
     * @param pattern The pattern to add to the blacklist
     * @return true if the pattern was newly added, false if it already existed
     */
    bool add_pattern_no_cleanup(const std::string& pattern) {
        if (pattern.empty()) return false;
        
        std::lock_guard<std::mutex> lock(blacklist_mutex);
        auto result = blacklisted_patterns.insert(pattern);
        
        if (result.second) {  // Insertion actually happened
            LLAMA_LOG("Added pattern to blacklist (no cleanup): '" + 
                      (pattern.length() > 50 ? pattern.substr(0, 50) + "..." : pattern) + "'");
            return true;
        } else {
            LLAMA_LOG("Pattern already blacklisted: '" + 
                      (pattern.length() > 50 ? pattern.substr(0, 50) + "..." : pattern) + "'");
            return false;
        }
    }

    /**
     * @brief Add multiple patterns to the blacklist efficiently
     * @param patterns Vector of patterns to add
     * @return Vector of newly added patterns (excludes duplicates)
     */
    std::vector<std::string> add_multiple_patterns(const std::vector<std::string>& patterns) {
        if (patterns.empty()) return {};
        
        std::vector<std::string> new_patterns;
        
        // Add all patterns to blacklist and collect new ones
        {
            std::lock_guard<std::mutex> lock(blacklist_mutex);
            for (const auto& pattern : patterns) {
                if (!pattern.empty() && blacklisted_patterns.find(pattern) == blacklisted_patterns.end()) {
                    blacklisted_patterns.insert(pattern);
                    new_patterns.push_back(pattern);
                    LLAMA_LOG("Added pattern to blacklist: '" + 
                              (pattern.length() > 50 ? pattern.substr(0, 50) + "..." : pattern) + "'");
                }
            }
        }
        
        // Perform retroactive cleanup for new patterns
        if (!new_patterns.empty() && cleanup_callback) {
            LLAMA_LOG("Performing retroactive cleanup for " + std::to_string(new_patterns.size()) + " new blacklist patterns");
            
            for (const auto& pattern : new_patterns) {
                cleanup_callback(pattern);
            }
        } else if (new_patterns.empty()) {
            LLAMA_LOG("All provided patterns were already blacklisted, no cleanup needed");
        }
        
        return new_patterns;
    }

    /**
     * @brief Remove a pattern from the blacklist
     * @param pattern The pattern to remove
     * @return true if the pattern was found and removed, false otherwise
     */
    bool remove_pattern(const std::string& pattern) {
        if (pattern.empty()) return false;
        
        std::lock_guard<std::mutex> lock(blacklist_mutex);
        auto it = blacklisted_patterns.find(pattern);
        if (it != blacklisted_patterns.end()) {
            blacklisted_patterns.erase(it);
            LLAMA_LOG("Removed pattern from blacklist: '" + 
                      (pattern.length() > 50 ? pattern.substr(0, 50) + "..." : pattern) + "'");
            return true;
        }
        return false;
    }

    /**
     * @brief Clear all blacklisted patterns
     * @return Number of patterns that were cleared
     */
    size_t clear_all() {
        std::lock_guard<std::mutex> lock(blacklist_mutex);
        size_t count = blacklisted_patterns.size();
        blacklisted_patterns.clear();
        LLAMA_LOG("Cleared " + std::to_string(count) + " blacklisted patterns");
        return count;
    }

    /**
     * @brief Get all blacklisted patterns (for UI display or export)
     * @return Vector containing all current blacklist patterns
     */
    std::vector<std::string> get_all_patterns() const {
        std::lock_guard<std::mutex> lock(blacklist_mutex);
        std::vector<std::string> result;
        result.reserve(blacklisted_patterns.size());
        for (const auto& pattern : blacklisted_patterns) {
            result.push_back(pattern);
        }
        return result;
    }

    /**
     * @brief Get the number of blacklisted patterns
     * @return Number of patterns currently in the blacklist
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(blacklist_mutex);
        return blacklisted_patterns.size();
    }

    /**
     * @brief Check if the blacklist is empty
     * @return true if no patterns are blacklisted, false otherwise
     */
    bool empty() const {
        std::lock_guard<std::mutex> lock(blacklist_mutex);
        return blacklisted_patterns.empty();
    }

    /**
     * @brief Manually trigger retroactive cleanup for all current patterns
     * Useful when cleanup callback is set after patterns are added, or for maintenance
     */
    void trigger_full_cleanup() {
        std::vector<std::string> all_patterns;
        
        // Get all current blacklist patterns
        {
            std::lock_guard<std::mutex> lock(blacklist_mutex);
            all_patterns.reserve(blacklisted_patterns.size());
            for (const auto& pattern : blacklisted_patterns) {
                all_patterns.push_back(pattern);
            }
        }
        
        if (all_patterns.empty()) {
            LLAMA_LOG("No blacklist patterns to clean up");
            return;
        }
        
        if (!cleanup_callback) {
            LLAMA_LOG("No cleanup callback available for retroactive cleanup");
            return;
        }
        
        LLAMA_LOG("Triggering full retroactive cleanup for " + std::to_string(all_patterns.size()) + " blacklist patterns");
        
        // Process each pattern
        for (const auto& pattern : all_patterns) {
            cleanup_callback(pattern);
        }
        
        LLAMA_LOG("Full retroactive cleanup completed");
    }
};
