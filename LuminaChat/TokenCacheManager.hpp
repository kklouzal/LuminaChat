// TokenCacheManager.hpp - header-only LRU token cache implementation
// Provides efficient caching for tokenization results with LRU eviction
//
// Project Settings:
// C++ Language Standard: ISO C++20 Standard (/std:c++20)
// Character Set: Use Unicode Character Set
// Optimization: Maximum Optimization (Favor Speed) (/O2)
//
// CRITICAL CODING DIRECTIVES:
// 1. Minimalism & Performance: Deliver lean, efficient solutions that avoid unnecessary bloat.
// 2. Consistent Coding Style: Maintain uniform style and structure for clear, maintainable code.
// 3. Clear Documentation: Provide concise comments explaining complex logic and key decisions.
// 4. Eliminate Redundancy: Remove unused, obsolete, and legacy code along with excess includes.
// 5. Optimize Function Structure: Adjust function boundaries to reduce overlap and clarify responsibilities.
// 6. Preserve Core Functionality: Streamline code while safeguarding essential features.
// 7. Cross-Platform Standards: Use fixed-width types and proper initialization to ensure portability.
// 8. Smart Caching: Cache frequently used variables to reduce repeated allocations.
// 9. Ensure Logical Consistency: Review code flow to maintain coherent, error-free execution.
// 10. Continuous Refinement: Regularly refactor and verify that updates preserve stable functionality.
#pragma once

#include <unordered_map>
#include <list>
#include <vector>
#include <string>
#include <cstdint>
#include "llama-cpp.h"

class TokenCacheManager {
private:
    // Cache storage and LRU tracking - using move semantics for better performance
    mutable std::unordered_map<std::string, std::vector<llama_token>> token_cache;
    mutable std::list<std::string> lru_list;
    mutable std::unordered_map<std::string, std::list<std::string>::iterator> lru_map;
    
    // Cache configuration and statistics with better defaults
    static constexpr size_t DEFAULT_MAX_SIZE = 1024;
    static constexpr float TRIM_THRESHOLD = 0.9f;
    static constexpr float TRIM_TARGET = 0.75f;
    
    mutable size_t max_cache_size = DEFAULT_MAX_SIZE;
    mutable size_t cache_hits = 0;
    mutable size_t cache_requests = 0;
    
    // Optimized LRU access with bounds checking
    void update_lru_access(const std::string& key) const noexcept {
        if (auto lru_it = lru_map.find(key); lru_it != lru_map.end()) {
            lru_list.splice(lru_list.begin(), lru_list, lru_it->second);
        }
    }
    
    // Streamlined cache addition with move semantics
    void add_to_cache(const std::string& key, std::vector<llama_token> tokens) const {
        // Pre-emptive cleanup if approaching limit
        if (token_cache.size() >= static_cast<size_t>(max_cache_size * TRIM_THRESHOLD)) {
            trim_cache();
        }
        
        // Handle existing key case efficiently
        if (auto existing = token_cache.find(key); existing != token_cache.end()) {
            existing->second = std::move(tokens);
            update_lru_access(key);
            return;
        }
        
        // Add new entry with move semantics
        auto [cache_it, inserted] = token_cache.emplace(key, std::move(tokens));
        if (inserted) {
            lru_list.push_front(key);
            lru_map.emplace(key, lru_list.begin());
        }
    }
    
    // Optimized cache trimming with batch removal
    void trim_cache() const noexcept {
        const size_t target_size = static_cast<size_t>(max_cache_size * TRIM_TARGET);
        
        while (token_cache.size() > target_size && !lru_list.empty()) {
            const std::string& lru_key = lru_list.back();
            
            // Remove from all data structures
            token_cache.erase(lru_key);
            lru_map.erase(lru_key);
            lru_list.pop_back();
        }
    }

public:
    TokenCacheManager() = default;
    ~TokenCacheManager() = default;
    
    // Non-copyable but movable for resource management
    TokenCacheManager(const TokenCacheManager&) = delete;
    TokenCacheManager& operator=(const TokenCacheManager&) = delete;
    TokenCacheManager(TokenCacheManager&&) = default;
    TokenCacheManager& operator=(TokenCacheManager&&) = default;
    
    // Optimized cache retrieval with move semantics
    std::vector<llama_token> get_cached_tokens(const std::string& cache_key) const {
        ++cache_requests;
        
        if (auto it = token_cache.find(cache_key); it != token_cache.end()) {
            ++cache_hits;
            update_lru_access(cache_key);
            return it->second; // Return copy for safety
        }
        
        return {}; // Empty vector for cache miss
    }
    
    // Cache tokens with move semantics for performance
    void cache_tokens(const std::string& cache_key, std::vector<llama_token> tokens) const {
        add_to_cache(cache_key, std::move(tokens));
    }
    
    // Inline simple checks for better performance
    bool has_cached_tokens(const std::string& cache_key) const noexcept {
        return token_cache.contains(cache_key);
    }
    
    // Configuration with validation
    void set_max_cache_size(size_t size) noexcept {
        max_cache_size = std::max(size, size_t{64}); // Minimum reasonable size
        if (token_cache.size() > max_cache_size) {
            trim_cache();
        }
    }
    
    // Inline getters for performance
    size_t get_max_cache_size() const noexcept { return max_cache_size; }
    size_t get_cache_size() const noexcept { return token_cache.size(); }
    size_t get_cache_hits() const noexcept { return cache_hits; }
    size_t get_cache_requests() const noexcept { return cache_requests; }
    
    float get_cache_hit_ratio() const noexcept {
        return cache_requests > 0 ? static_cast<float>(cache_hits) / cache_requests : 0.0f;
    }
    
    // Efficient cache management
    void clear_cache() const noexcept {
        token_cache.clear();
        lru_list.clear();
        lru_map.clear();
        reset_stats();
    }
    
    void reset_stats() const noexcept {
        cache_hits = 0;
        cache_requests = 0;
    }
    
    // Memory usage estimation for monitoring
    size_t estimate_memory_usage() const noexcept {
        size_t total = 0;
        for (const auto& [key, tokens] : token_cache) {
            total += key.size() + tokens.size() * sizeof(llama_token);
        }
        return total + lru_list.size() * sizeof(std::string) + 
               lru_map.size() * sizeof(std::pair<std::string, std::list<std::string>::iterator>);
    }
};
