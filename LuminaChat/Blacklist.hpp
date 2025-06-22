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

public:    /**
     * @brief Case-insensitive pattern matching using STL algorithms for performance
     * @param response The response text to check
     * @param pattern The blacklist pattern to match against
     * @return true if the response matches the pattern (case-insensitive)
     */
    [[nodiscard]] static bool matches_pattern(const std::string& response, const std::string& pattern) noexcept {
        if (response.empty() || pattern.empty()) [[unlikely]] return false;
        
        // Fast exact match check first
        if (response == pattern) [[unlikely]] return true;
        
        // Use STL search with case-insensitive comparison (Directive #10: Standard Library Preference)
        const auto it = std::search(response.begin(), response.end(), pattern.begin(), pattern.end(),
            [](const char a, const char b) noexcept { 
                return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b)); 
            });
        
        return it != response.end();
    }
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
    }    /**
     * @brief Set or update the retroactive cleanup callback
     * @param callback The callback function to invoke for retroactive cleanup
     */
    void set_cleanup_callback(RetroactiveCleanupCallback callback) noexcept {
        std::lock_guard<std::mutex> lock(blacklist_mutex);
        cleanup_callback = std::move(callback);
    }/**
     * @brief Check if a response is blacklisted using optimized STL algorithms
     * @param response The response text to check
     * @return true if the response matches any blacklisted pattern
     */
    [[nodiscard]] bool is_blacklisted(const std::string& response) const noexcept {
        if (response.empty()) [[unlikely]] return false;
        
        std::lock_guard<std::mutex> lock(blacklist_mutex);
        
        // Fast exact match check first using STL find (Directive #10: Standard Library Preference)
        if (blacklisted_patterns.find(response) != blacklisted_patterns.end()) [[unlikely]] {
            return true;
        }
          // Use STL any_of algorithm for pattern matching (Directive #10: Standard Library Preference)
        return std::any_of(blacklisted_patterns.begin(), blacklisted_patterns.end(),
            [&response](const std::string& pattern) noexcept {
                return matches_pattern(response, pattern);
            });
    }    /**
     * @brief Add a pattern to the blacklist with automatic retroactive cleanup
     * @param pattern The pattern to add to the blacklist
     * @return true if the pattern was newly added, false if it already existed
     */
    bool add_pattern(const std::string& pattern) noexcept {
        if (pattern.empty()) [[unlikely]] return false;
        
        // Check if this pattern is already blacklisted to avoid redundant cleanup
        bool pattern_already_exists = false;
        {
            std::lock_guard<std::mutex> lock(blacklist_mutex);
            auto result = blacklisted_patterns.insert(pattern);
            pattern_already_exists = !result.second;
            
            if (!pattern_already_exists) [[likely]] {
                LLAMA_LOG("Added pattern to blacklist: '" + 
                          (pattern.length() > 50 ? pattern.substr(0, 50) + "..." : pattern) + "'");
            } else {
                LLAMA_LOG("Pattern already blacklisted, skipping: '" + 
                          (pattern.length() > 50 ? pattern.substr(0, 50) + "..." : pattern) + "'");
                return false;
            }
        }
        
        // Perform retroactive cleanup if callback is available
        if (cleanup_callback) [[likely]] {
            LLAMA_LOG("Performing retroactive cleanup for newly blacklisted pattern");
            try {
                cleanup_callback(pattern);
            } catch (const std::exception& e) {
                LLAMA_LOG("Error during retroactive cleanup: " + std::string(e.what()));
            } catch (...) {
                LLAMA_LOG("Unknown error during retroactive cleanup");
            }
        }
        
        return true;
    }    /**
     * @brief Add a pattern to the blacklist without retroactive cleanup
     * @param pattern The pattern to add to the blacklist
     * @return true if the pattern was newly added, false if it already existed
     */
    bool add_pattern_no_cleanup(const std::string& pattern) noexcept {
        if (pattern.empty()) [[unlikely]] return false;
        
        std::lock_guard<std::mutex> lock(blacklist_mutex);
        auto result = blacklisted_patterns.insert(pattern);
        
        if (result.second) [[likely]] {  // Insertion actually happened
            LLAMA_LOG("Added pattern to blacklist (no cleanup): '" + 
                      (pattern.length() > 50 ? pattern.substr(0, 50) + "..." : pattern) + "'");
            return true;
        } else {
            LLAMA_LOG("Pattern already blacklisted: '" + 
                      (pattern.length() > 50 ? pattern.substr(0, 50) + "..." : pattern) + "'");
            return false;
        }
    }    /**
     * @brief Add multiple patterns to the blacklist efficiently using STL algorithms
     * @param patterns Vector of patterns to add
     * @return Vector of newly added patterns (excludes duplicates)
     */
    std::vector<std::string> add_multiple_patterns(const std::vector<std::string>& patterns) noexcept {
        if (patterns.empty()) [[unlikely]] return {};
        
        std::vector<std::string> new_patterns;
        new_patterns.reserve(patterns.size()); // Reserve space for efficiency
        
        // Filter out empty patterns and add new ones efficiently
        {
            std::lock_guard<std::mutex> lock(blacklist_mutex);
            
            // Use STL copy_if for filtering and processing (Directive #10: Standard Library Preference)
            std::copy_if(patterns.begin(), patterns.end(), std::back_inserter(new_patterns),
                [this](const std::string& pattern) noexcept {
                    if (pattern.empty()) return false;
                    
                    auto [_, was_inserted] = blacklisted_patterns.insert(pattern);
                    if (was_inserted) {
                        LLAMA_LOG("Added pattern to blacklist: '" + 
                                  (pattern.length() > 50 ? pattern.substr(0, 50) + "..." : pattern) + "'");
                        return true;
                    }
                    return false;
                });
        }
        
        // Perform retroactive cleanup for new patterns
        if (!new_patterns.empty() && cleanup_callback) [[likely]] {
            LLAMA_LOG("Performing retroactive cleanup for " + std::to_string(new_patterns.size()) + " new blacklist patterns");
            
            try {
                // Use STL for_each for callback invocation (Directive #10: Standard Library Preference)
                std::for_each(new_patterns.begin(), new_patterns.end(), cleanup_callback);
            } catch (const std::exception& e) {
                LLAMA_LOG("Error during batch retroactive cleanup: " + std::string(e.what()));
            } catch (...) {
                LLAMA_LOG("Unknown error during batch retroactive cleanup");
            }
        } else if (new_patterns.empty()) {
            LLAMA_LOG("All provided patterns were already blacklisted, no cleanup needed");
        }
        
        return new_patterns;
    }    /**
     * @brief Remove a pattern from the blacklist
     * @param pattern The pattern to remove
     * @return true if the pattern was found and removed, false otherwise
     */
    bool remove_pattern(const std::string& pattern) noexcept {
        if (pattern.empty()) [[unlikely]] return false;
        
        std::lock_guard<std::mutex> lock(blacklist_mutex);
        auto it = blacklisted_patterns.find(pattern);
        if (it != blacklisted_patterns.end()) [[likely]] {
            blacklisted_patterns.erase(it);
            LLAMA_LOG("Removed pattern from blacklist: '" + 
                      (pattern.length() > 50 ? pattern.substr(0, 50) + "..." : pattern) + "'");
            return true;
        }
        return false;
    }    /**
     * @brief Clear all blacklisted patterns
     * @return Number of patterns that were cleared
     */
    size_t clear_all() noexcept {
        std::lock_guard<std::mutex> lock(blacklist_mutex);
        size_t count = blacklisted_patterns.size();
        if (count > 0) [[likely]] {
            blacklisted_patterns.clear();
            LLAMA_LOG("Cleared " + std::to_string(count) + " blacklisted patterns");
        }
        return count;
    }    /**
     * @brief Get all blacklisted patterns using optimized STL algorithms (for UI display or export)
     * @return Vector containing all current blacklist patterns
     */
    [[nodiscard]] std::vector<std::string> get_all_patterns() const noexcept {
        std::lock_guard<std::mutex> lock(blacklist_mutex);
        
        // Use STL constructor for efficient copy (Directive #10: Standard Library Preference)
        return std::vector<std::string>(blacklisted_patterns.begin(), blacklisted_patterns.end());
    }

    /**
     * @brief Get the number of blacklisted patterns
     * @return Number of patterns currently in the blacklist
     */
    [[nodiscard]] size_t size() const noexcept {
        std::lock_guard<std::mutex> lock(blacklist_mutex);
        return blacklisted_patterns.size();
    }

    /**
     * @brief Check if the blacklist is empty
     * @return true if no patterns are blacklisted, false otherwise
     */
    [[nodiscard]] bool empty() const noexcept {
        std::lock_guard<std::mutex> lock(blacklist_mutex);
        return blacklisted_patterns.empty();
    }    /**
     * @brief Manually trigger retroactive cleanup for all current patterns using STL algorithms
     * Useful when cleanup callback is set after patterns are added, or for maintenance
     */
    void trigger_full_cleanup() noexcept {
        if (!cleanup_callback) [[unlikely]] {
            LLAMA_LOG("No cleanup callback available for retroactive cleanup");
            return;
        }
        
        std::vector<std::string> all_patterns;
        
        // Get all current blacklist patterns efficiently
        {
            std::lock_guard<std::mutex> lock(blacklist_mutex);
            if (blacklisted_patterns.empty()) {
                LLAMA_LOG("No blacklist patterns to clean up");
                return;
            }
            
            // Use STL constructor for efficient copy (Directive #10: Standard Library Preference)
            all_patterns = std::vector<std::string>(blacklisted_patterns.begin(), blacklisted_patterns.end());
        }
        
        LLAMA_LOG("Triggering full retroactive cleanup for " + std::to_string(all_patterns.size()) + " blacklist patterns");
        
        try {
            // Use STL for_each for callback processing (Directive #10: Standard Library Preference)
            std::for_each(all_patterns.begin(), all_patterns.end(), cleanup_callback);
        } catch (const std::exception& e) {
            LLAMA_LOG("Error during full retroactive cleanup: " + std::string(e.what()));
        } catch (...) {
            LLAMA_LOG("Unknown error during full retroactive cleanup");
        }
        
        LLAMA_LOG("Full retroactive cleanup completed");
    }
};
