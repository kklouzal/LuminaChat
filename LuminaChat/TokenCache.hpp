// TokenCache.hpp - header-only implementation for token caching with LRU eviction
// Provides efficient caching of tokenization results to avoid repeated expensive tokenization operations.
//
// CRITICAL CODING DIRECTIVES:
// 1. Minimalism & Performance: Deliver lean, efficient solutions that avoid unnecessary bloat.
// 2. Consistent Coding Style: Maintain uniform style and structure for clear, maintainable code.
// 3. Clear Documentation: Provide concise comments explaining complex logic and key decisions.
// 4. Smart Caching: Cache frequently used variables to reduce repeated allocations.
// 5. Cross-Platform Standards: Use fixed-width types and proper initialization to ensure portability.
#pragma once

#include <unordered_map>
#include <list>
#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>
#include "llama-cpp.h"

class TokenCache {
private:
    // Cache storage
    mutable std::unordered_map<std::string, std::vector<llama_token>> token_cache;
    mutable std::list<std::string> token_cache_lru;  // Track access order for LRU eviction
    mutable std::unordered_map<std::string, std::list<std::string>::iterator> token_cache_lru_map;  // Fast lookup in LRU list
    
    // Configuration
    size_t max_cache_size;
    
    // Performance tracking
    mutable size_t cache_hits = 0;
    mutable size_t cache_requests = 0;

    // Update LRU access order
    void update_lru_access(const std::string& key) const {
        auto lru_it = token_cache_lru_map.find(key);
        if (lru_it != token_cache_lru_map.end()) {
            token_cache_lru.splice(token_cache_lru.begin(), token_cache_lru, lru_it->second);
        }
    }

    // Optimized LRU cache management
    void add_to_cache_internal(const std::string& key, const std::vector<llama_token>& tokens) const {
        // Pre-emptive cleanup if approaching limit
        if (token_cache.size() >= max_cache_size * 0.9f) {
            trim_cache();
        }
        
        // Check if key already exists (update case)
        auto existing = token_cache.find(key);
        if (existing != token_cache.end()) {
            existing->second = tokens;
            update_lru_access(key);
            return;
        }
        
        // Add new entry
        token_cache[key] = tokens;
        token_cache_lru.push_front(key);
        token_cache_lru_map[key] = token_cache_lru.begin();
    }
    
    // More aggressive cache trimming for better memory management
    void trim_cache() const {
        // Remove 25% of entries when trimming to reduce frequency
        size_t target_size = static_cast<size_t>(max_cache_size * 0.75f);
        
        while (token_cache.size() > target_size && !token_cache_lru.empty()) {
            std::string lru_key = token_cache_lru.back();
            token_cache_lru.pop_back();
            token_cache_lru_map.erase(lru_key);
            token_cache.erase(lru_key);
        }
    }

public:
    explicit TokenCache(size_t max_size = 1024) : max_cache_size(max_size) {}
    
    // Get cached tokens or return empty vector if not found
    std::vector<llama_token> get(const std::string& key) const {
        cache_requests++;
        
        auto it = token_cache.find(key);
        if (it != token_cache.end()) {
            cache_hits++;
            update_lru_access(key);
            return it->second;
        }
        
        return {};
    }
    
    // Add tokens to cache
    void put(const std::string& key, const std::vector<llama_token>& tokens) const {
        add_to_cache_internal(key, tokens);
    }
    
    // Check if key exists in cache
    bool contains(const std::string& key) const {
        return token_cache.find(key) != token_cache.end();
    }
    
    // Clear all cached data
    void clear() const {
        token_cache.clear();
        token_cache_lru.clear();
        token_cache_lru_map.clear();
        cache_hits = 0;
        cache_requests = 0;
    }
    
    // Get cache statistics - Enhanced with more detailed information
    struct CacheStats {
        size_t hits;
        size_t requests;
        size_t size;
        size_t max_size;
        float hit_ratio;
        float fill_ratio;
    };
    
    CacheStats get_stats() const {
        float hit_ratio = (cache_requests > 0) ? static_cast<float>(cache_hits) / cache_requests : 0.0f;
        float fill_ratio = static_cast<float>(token_cache.size()) / max_cache_size;
        return {
            cache_hits, 
            cache_requests, 
            token_cache.size(), 
            max_cache_size,
            hit_ratio,
            fill_ratio
        };
    }
    
    // Reset statistics
    void reset_stats() const {
        cache_hits = 0;
        cache_requests = 0;
    }
    
    // Configure cache size
    void set_max_size(size_t max_size) {
        max_cache_size = max_size;
        if (token_cache.size() > max_cache_size) {
            trim_cache();
        }
    }
    
    size_t get_max_size() const {
        return max_cache_size;
    }
    
    size_t get_current_size() const {
        return token_cache.size();
    }
    
    // Get cache efficiency metrics
    float get_hit_ratio() const {
        return (cache_requests > 0) ? static_cast<float>(cache_hits) / cache_requests : 0.0f;
    }
    
    float get_fill_ratio() const {
        return static_cast<float>(token_cache.size()) / max_cache_size;
    }
    
    // Get performance information
    struct CachePerformance {
        size_t total_hits;
        size_t total_requests;
        size_t current_entries;
        size_t max_entries;
        float efficiency_percent;
        float memory_usage_percent;
    };
    
    CachePerformance get_performance() const {
        float efficiency = get_hit_ratio() * 100.0f;
        float memory_usage = get_fill_ratio() * 100.0f;
        return {
            cache_hits,
            cache_requests,
            token_cache.size(),
            max_cache_size,
            efficiency,
            memory_usage
        };
    }
};
