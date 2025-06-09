// TokenCacheManager.hpp - header-only LRU token cache implementation
// Provides efficient caching for tokenization results with LRU eviction
//
// Project Settings:
// C++ Language Standard: ISO C++20 Standard (/std:c++20)
// Character Set: Use Unicode Character Set
// Optimization: Maximum Optimization (Favor Speed) (/O2)
//
// CODING DIRECTIVES:
// 1. Keep the codebase minimalistic, focused on functionality and efficiency.
// 2. Use consistent _t fixed-width variable types to ensure portability across platforms.
// 3. Cache frequently used variables to avoid repeated allocations.
// 4. Focus on overall codebase reduction without sacrificing functionality.
#pragma once

#include <unordered_map>
#include <list>
#include <vector>
#include <string>
#include <cstdint>
#include "llama-cpp.h"

class TokenCacheManager {
private:
    // Cache storage and LRU tracking
    mutable std::unordered_map<std::string, std::vector<llama_token>> token_cache;
    mutable std::list<std::string> lru_list;  // Track access order for LRU eviction
    mutable std::unordered_map<std::string, std::list<std::string>::iterator> lru_map;  // Fast lookup in LRU list
    
    // Cache configuration and statistics
    mutable size_t max_cache_size = 1024;
    mutable size_t cache_hits = 0;
    mutable size_t cache_requests = 0;
    
    // Update LRU access order
    void update_lru_access(const std::string& key) const {
        auto lru_it = lru_map.find(key);
        if (lru_it != lru_map.end()) {
            lru_list.splice(lru_list.begin(), lru_list, lru_it->second);
        }
    }
    
    // Add entry to cache with LRU tracking
    void add_to_cache(const std::string& key, const std::vector<llama_token>& tokens) const {
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
        lru_list.push_front(key);
        lru_map[key] = lru_list.begin();
    }
    
    // Trim cache using LRU eviction - remove 25% of entries to reduce frequency
    void trim_cache() const {
        size_t target_size = static_cast<size_t>(max_cache_size * 0.75f);
        
        while (token_cache.size() > target_size && !lru_list.empty()) {
            std::string lru_key = lru_list.back();
            lru_list.pop_back();
            lru_map.erase(lru_key);
            token_cache.erase(lru_key);
        }
    }

public:
    TokenCacheManager() = default;
    ~TokenCacheManager() = default;
    
    // Non-copyable but movable
    TokenCacheManager(const TokenCacheManager&) = delete;
    TokenCacheManager& operator=(const TokenCacheManager&) = delete;
    TokenCacheManager(TokenCacheManager&&) = default;
    TokenCacheManager& operator=(TokenCacheManager&&) = default;
    
    // Get cached tokens or return empty vector if not found
    std::vector<llama_token> get_cached_tokens(const std::string& cache_key) const {
        cache_requests++;
        
        auto it = token_cache.find(cache_key);
        if (it != token_cache.end()) {
            cache_hits++;
            update_lru_access(cache_key);
            return it->second;
        }
        
        return {};
    }
    
    // Cache tokenization result
    void cache_tokens(const std::string& cache_key, const std::vector<llama_token>& tokens) const {
        add_to_cache(cache_key, tokens);
    }
    
    // Check if key exists in cache
    bool has_cached_tokens(const std::string& cache_key) const {
        return token_cache.find(cache_key) != token_cache.end();
    }
    
    // Configuration methods
    void set_max_cache_size(size_t size) {
        max_cache_size = size;
        if (token_cache.size() > max_cache_size) {
            trim_cache();
        }
    }
    
    size_t get_max_cache_size() const {
        return max_cache_size;
    }
    
    // Statistics
    size_t get_cache_size() const {
        return token_cache.size();
    }
    
    size_t get_cache_hits() const {
        return cache_hits;
    }
    
    size_t get_cache_requests() const {
        return cache_requests;
    }
    
    float get_cache_hit_ratio() const {
        return cache_requests > 0 ? static_cast<float>(cache_hits) / cache_requests : 0.0f;
    }
    
    // Cache management
    void clear_cache() const {
        token_cache.clear();
        lru_list.clear();
        lru_map.clear();
        cache_hits = 0;
        cache_requests = 0;
    }
    
    void reset_stats() const {
        cache_hits = 0;
        cache_requests = 0;
    }
};
