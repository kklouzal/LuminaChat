// TokenCache.hpp - header-only implementation for token caching with LRU eviction
// Provides efficient caching of tokenization results to avoid repeated expensive tokenization operations.
//
// Project Settings:
// C++ Language Standard: ISO C++20 Standard (/std:c++20)
// C Language Standard: ISO C17 (2018) Standard (/std:c17)
// Optimization: Maximum Optimization (Favor Speed) (/O2)
// Favor Size or Speed: Favor fast code (/Ot)
// Runtime Library: Multi-threaded DLL (/MD)
// Enable Run-Time Type Information (RTTI) YES (/GR)
//
// CRITICAL CODING DIRECTIVES:
// 1.  Minimalism & Performance: Deliver lean, efficient solutions; do not create or preserve unused helpers, wrappers, trivial accessors (setters/getters), or scaffolding.
// 2.  Redundancy Elimination: Remove unused, obsolete, and legacy code—including unneeded interfaces, includes, helper or accessor methods.
// 3.  Consistent Style: Adopt a uniform coding style and structure for clarity and maintainability.
// 4.  Documentation: Write concise comments that explain complex logic and key design decisions.
// 5.  Zero Magic & Strong Typing: Replace magic literals with named constants, enums, or constexpr; prefer scoped enums over raw ints.
// 6.  Function Boundaries: Define clear responsibilities; reduce overlap and avoid unnecessary layers of indirection.
// 7.  Core Preservation: Streamline code while safeguarding essential features; favor direct variable or object access/passing over extra abstractions (e.g., setters/getters).
// 8.  Const-Correctness & Immutability: Mark variables, parameters, and methods as const wherever possible.
// 9.  RAII & Resource Safety: Encapsulate resource acquisition/release in constructors/destructors or smart pointers.
// 10. Standard Library Preference: Favor STL algorithms and containers over custom loops and buffers for clarity and safety.
// 11. Cross-Platform Portability: Use fixed-width types and proper initialization to guarantee identical behavior everywhere.
// 12. Thread Safety: Define and document thread-safety contracts; protect shared state with mutexes, atomics, or thread-safe containers.
// 13. Smart Caching: Cache frequently used values to minimize allocations and improve performance.
// 14. Logical Consistency: Verify code flow to ensure coherent, error-free execution paths.
// 15. Continuous Refinement: Regularly refactor and confirm that updates preserve stable functionality.

#pragma once

#include <unordered_map>
#include <list>
#include <vector>
#include <string>
#include <string_view>
#include <cstdint>
#include <algorithm>
#include <optional>
#include <memory>
#include <shared_mutex>
#include <atomic>
#include <chrono>
#include "llama-cpp.h"

// Enhanced high-performance token cache with expanded capabilities
class TokenCache {
public:
    // Cache statistics with direct access (Directive #7)
    mutable std::atomic<size_t> cache_hits{0};
    mutable std::atomic<size_t> cache_requests{0};
    mutable std::atomic<size_t> memory_usage_bytes{0};
    size_t max_cache_size;

    // Eviction policy options for expanded functionality
    enum class EvictionPolicy : uint8_t {
        LRU = 0,        // Least Recently Used (default)
        LFU = 1,        // Least Frequently Used
        FIFO = 2        // First In, First Out
    };

private:
    // Named constants to replace magic literals (Directive #5)
    static constexpr float CACHE_PREEMPTIVE_THRESHOLD = 0.9f;
    static constexpr float CACHE_TRIM_TARGET_RATIO = 0.7f;
    static constexpr size_t DEFAULT_CACHE_SIZE = 2048;  // Increased default
    static constexpr float PERCENTAGE_MULTIPLIER = 100.0f;
    static constexpr size_t SMALL_TOKEN_THRESHOLD = 64;  // For small buffer optimization
    static constexpr size_t INITIAL_RESERVE_SIZE = 512;
      // Cache entry with enhanced metadata for better eviction decisions
    struct CacheEntry {
        std::string text;  // Store both text and tokens for bidirectional lookup
        std::vector<llama_token> tokens;
        mutable std::atomic<uint32_t> access_count{1};
        mutable std::chrono::steady_clock::time_point last_access;
        size_t memory_size;
        
        CacheEntry(std::string text_, std::vector<llama_token> tokens_) 
            : text(std::move(text_)),
              tokens(std::move(tokens_)), 
              last_access(std::chrono::steady_clock::now()),
              memory_size(text.size() + tokens.size() * sizeof(llama_token) + sizeof(CacheEntry)) {}
              
        // Move constructor for better performance
        CacheEntry(CacheEntry&& other) noexcept 
            : text(std::move(other.text)),
              tokens(std::move(other.tokens)),
              access_count(other.access_count.load()),
              last_access(other.last_access),
              memory_size(other.memory_size) {}
    };    // Optimized cache storage with better memory layout for bidirectional lookup
    mutable std::unordered_map<std::string, std::unique_ptr<CacheEntry>> cache_entries;  // text -> entry
    mutable std::unordered_map<std::string, std::unique_ptr<CacheEntry>*> token_hash_map; // token_hash -> entry pointer
    mutable std::list<std::string> access_order;  // For LRU/FIFO tracking (uses text keys)
    mutable std::unordered_map<std::string, std::list<std::string>::iterator> access_iterators;
    
    // Prefix cache for common token patterns (expanded capability)
    mutable std::unordered_map<std::string, std::vector<llama_token>> prefix_cache;
    
    // Configuration
    EvictionPolicy eviction_policy = EvictionPolicy::LRU;
    bool enable_prefix_caching = true;
    bool thread_safe = false;
    
    // Thread safety (optional, zero overhead when disabled)
    mutable std::shared_mutex cache_mutex;
    
    // Memory management helpers
    void update_memory_usage(const int64_t delta) const {
        memory_usage_bytes.fetch_add(delta, std::memory_order_relaxed);
    }
    
    // Optimized LRU access update using C++17 features
    void update_access_order(const std::string& key, CacheEntry& entry) const {
        entry.last_access = std::chrono::steady_clock::now();
        entry.access_count.fetch_add(1, std::memory_order_relaxed);
        
        if (eviction_policy == EvictionPolicy::LRU) {
            if (const auto it = access_iterators.find(key); it != access_iterators.end()) {
                access_order.splice(access_order.begin(), access_order, it->second);
            }
        }
    }    // Enhanced cache management with configurable eviction policies
    void add_to_cache_internal(std::string key, std::string text, std::vector<llama_token> tokens) const {
        // Pre-emptive cleanup if approaching limit
        if (cache_entries.size() >= static_cast<size_t>(max_cache_size * CACHE_PREEMPTIVE_THRESHOLD)) {
            trim_cache();
        }
        
        add_bidirectional_entry(std::move(key), std::move(text), std::move(tokens));
    }    // Improved cache trimming with configurable eviction policies
    void trim_cache() const {
        const size_t target_size = static_cast<size_t>(max_cache_size * CACHE_TRIM_TARGET_RATIO);
        
        while (cache_entries.size() > target_size && !cache_entries.empty()) {
            std::string victim_key;
            
            switch (eviction_policy) {
                case EvictionPolicy::LRU:
                case EvictionPolicy::FIFO:
                    if (!access_order.empty()) {
                        victim_key = access_order.back();
                        access_order.pop_back();
                    }
                    break;
                    
                case EvictionPolicy::LFU: {
                    // Find least frequently used entry
                    auto min_it = std::min_element(cache_entries.begin(), cache_entries.end(),
                        [](const auto& a, const auto& b) {
                            return a.second->access_count.load() < b.second->access_count.load();
                        });
                    if (min_it != cache_entries.end()) {
                        victim_key = min_it->first;
                    }
                    break;
                }
            }
            
            if (!victim_key.empty()) {
                remove_bidirectional_entry(victim_key);
            } else {
                break;  // Safety exit
            }
        }
    }
      // Prefix matching for enhanced caching capabilities (C++17 compatible)
    [[nodiscard]] std::optional<std::vector<llama_token>> find_prefix_match(std::string_view text) const {
        if (!enable_prefix_caching || prefix_cache.empty()) {
            return std::nullopt;
        }
        
        // Find longest matching prefix
        std::string_view best_match;
        for (const auto& entry : prefix_cache) {
            const std::string& prefix = entry.first;
            const auto& tokens = entry.second;
            
            // C++17 compatible prefix check
            if (text.size() >= prefix.size() && 
                text.substr(0, prefix.size()) == prefix && 
                prefix.length() > best_match.length()) {
                best_match = prefix;
            }
        }
        
        if (!best_match.empty()) {
            return prefix_cache.at(std::string(best_match));
        }
        
        return std::nullopt;
    }
    
    // Thread-safe wrapper helper
    template<typename Func>
    auto with_lock(Func&& func) const {
        if constexpr (std::is_void_v<std::invoke_result_t<Func>>) {
            if (thread_safe) {
                std::shared_lock lock(cache_mutex);
                func();
            } else {
                func();
            }
        } else {
            if (thread_safe) {
                std::shared_lock lock(cache_mutex);
                return func();
            } else {
                return func();
            }
        }
    }
    
    template<typename Func>
    auto with_write_lock(Func&& func) const {
        if constexpr (std::is_void_v<std::invoke_result_t<Func>>) {
            if (thread_safe) {
                std::unique_lock lock(cache_mutex);
                func();
            } else {
                func();
            }
        } else {
            if (thread_safe) {
                std::unique_lock lock(cache_mutex);
                return func();
            } else {
                return func();
            }
        }
    }

public:
    explicit TokenCache(const size_t max_size = DEFAULT_CACHE_SIZE, 
                       const EvictionPolicy policy = EvictionPolicy::LRU,
                       const bool enable_thread_safety = false) 
        : max_cache_size(max_size), eviction_policy(policy), thread_safe(enable_thread_safety) {
        // Reserve memory for better performance
        cache_entries.reserve(INITIAL_RESERVE_SIZE);
        access_iterators.reserve(INITIAL_RESERVE_SIZE);
        if (enable_prefix_caching) {
            prefix_cache.reserve(INITIAL_RESERVE_SIZE / 4);
        }
    }
    
    // Configure cache settings
    void configure(const EvictionPolicy policy, const bool enable_prefixes = true, 
                  const bool enable_thread_safety = false) {
        eviction_policy = policy;
        enable_prefix_caching = enable_prefixes;
        thread_safe = enable_thread_safety;
    }
    
    // Resize cache with automatic trimming if needed
    void resize_cache(const size_t new_max_size) {
        return with_write_lock([&]() {
            max_cache_size = new_max_size;
            if (cache_entries.size() > max_cache_size) {
                trim_cache();
            }
        });
    }
      // Enhanced get with optimal return semantics for bidirectional lookup
    [[nodiscard]] std::optional<std::vector<llama_token>> get_tokens(std::string_view key) const {
        return with_lock([&]() -> std::optional<std::vector<llama_token>> {
            cache_requests.fetch_add(1, std::memory_order_relaxed);
            
            // Try exact match first
            const std::string key_str(key);
            if (auto it = cache_entries.find(key_str); it != cache_entries.end()) {
                cache_hits.fetch_add(1, std::memory_order_relaxed);
                update_access_order(it->first, *it->second);
                return it->second->tokens;
            }
            
            // Try prefix match as fallback
            return find_prefix_match(key);
        });
    }    // Enhanced put with move semantics - bidirectional caching
    void put(std::string key, std::string text, std::vector<llama_token> tokens) const {
        return with_write_lock([&]() {
            add_to_cache_internal(std::move(key), std::move(text), std::move(tokens));
        });
    }
      // Optimized bidirectional reverse lookup - get text from tokens
    [[nodiscard]] std::optional<std::string> get_text(const std::vector<llama_token>& tokens) const {
        return with_lock([&]() -> std::optional<std::string> {
            cache_requests.fetch_add(1, std::memory_order_relaxed);
            
            const std::string token_hash = hash_tokens(tokens);
            if (auto it = token_hash_map.find(token_hash); it != token_hash_map.end()) {
                auto& entry = **(it->second);
                
                // Verify tokens match exactly (hash collision protection)
                if (entry.tokens == tokens) {
                    cache_hits.fetch_add(1, std::memory_order_relaxed);
                    
                    // Find the text key for LRU updates
                    for (const auto& [text_key, cached_entry] : cache_entries) {
                        if (cached_entry.get() == &entry) {
                            update_access_order(text_key, entry);
                            return entry.text;
                        }
                    }
                }
            }
            
            return std::nullopt;
        });
    }    // High-performance bulk operations for bidirectional caching
    void put_batch(std::vector<std::tuple<std::string, std::string, std::vector<llama_token>>> entries) const {
        return with_write_lock([&]() {
            for (auto& [key, text, tokens] : entries) {
                add_to_cache_internal(std::move(key), std::move(text), std::move(tokens));
            }
        });
    }
    
    // Add prefix to prefix cache
    void add_prefix(std::string prefix, std::vector<llama_token> tokens) const {
        if (!enable_prefix_caching) return;
        
        return with_write_lock([&]() {
            prefix_cache[std::move(prefix)] = std::move(tokens);
        });
    }
    
    // Check if key exists in cache
    [[nodiscard]] bool contains(std::string_view key) const {
        return with_lock([&]() {
            return cache_entries.find(std::string(key)) != cache_entries.end();
        });
    }
      // Clear all cached data
    void clear() const {
        return with_write_lock([&]() {
            cache_entries.clear();
            token_hash_map.clear();  // Clear reverse lookup map
            access_order.clear();
            access_iterators.clear();
            prefix_cache.clear();
            cache_hits.store(0, std::memory_order_relaxed);
            cache_requests.store(0, std::memory_order_relaxed);
            memory_usage_bytes.store(0, std::memory_order_relaxed);
        });
    }
      // Enhanced statistics with performance metrics - now includes bidirectional stats
    struct CacheStats {
        size_t hits;
        size_t requests;
        size_t entries;
        size_t max_size;
        size_t memory_bytes;
        size_t prefix_entries;
        size_t bidirectional_entries;  // NEW: entries supporting reverse lookup
        float hit_ratio;
        float fill_ratio;
        float memory_efficiency;  // tokens per byte
        float bidirectional_coverage;  // percentage of entries with bidirectional support
    };
      [[nodiscard]] CacheStats get_stats() const {
        return with_lock([&]() {
            const auto hits = cache_hits.load(std::memory_order_relaxed);
            const auto requests = cache_requests.load(std::memory_order_relaxed);
            const auto memory_bytes = memory_usage_bytes.load(std::memory_order_relaxed);
            const auto entries = cache_entries.size();
            const auto bidirectional_entries = token_hash_map.size();  // Count reverse mappings
            
            // Calculate total tokens for efficiency metric
            size_t total_tokens = 0;
            for (const auto& [key, entry] : cache_entries) {
                total_tokens += entry->tokens.size();
            }
            
            return CacheStats{
                .hits = hits,
                .requests = requests,
                .entries = entries,
                .max_size = max_cache_size,
                .memory_bytes = memory_bytes,
                .prefix_entries = prefix_cache.size(),
                .bidirectional_entries = bidirectional_entries,
                .hit_ratio = (requests > 0) ? static_cast<float>(hits) / static_cast<float>(requests) : 0.0f,
                .fill_ratio = static_cast<float>(entries) / static_cast<float>(max_cache_size),
                .memory_efficiency = (memory_bytes > 0) ? static_cast<float>(total_tokens) / static_cast<float>(memory_bytes) : 0.0f,
                .bidirectional_coverage = (entries > 0) ? static_cast<float>(bidirectional_entries) / static_cast<float>(entries) : 0.0f
            };
        });
    }
      // Memory optimization: compact cache by removing fragmentation
    void compact() const {
        return with_write_lock([&]() {
            // Rebuild cache with better memory layout
            auto old_entries = std::move(cache_entries);
            cache_entries.clear();
            cache_entries.reserve(old_entries.size());
            
            // Clear and rebuild reverse mappings
            token_hash_map.clear();
            token_hash_map.reserve(old_entries.size());
            
            for (auto& [key, entry] : old_entries) {
                entry->tokens.shrink_to_fit();  // Remove excess capacity
                entry->text.shrink_to_fit();    // Remove excess capacity from text too
                
                // Rebuild reverse mapping
                const std::string token_hash = hash_tokens(entry->tokens);
                
                // Move back to main cache
                cache_entries[key] = std::move(entry);
                token_hash_map[token_hash] = &(cache_entries[key]);
            }
        });
    }
    
    // Performance: warm up cache with expected size to reduce allocations
    void reserve(const size_t expected_size) {
        return with_write_lock([&]() {
            const size_t reserve_size = std::min(expected_size, max_cache_size);
            cache_entries.reserve(reserve_size);
            access_iterators.reserve(reserve_size);
        });
    }
    
    // Advanced: Get cache entry with metadata for debugging/monitoring
    struct EntryInfo {
        std::vector<llama_token> tokens;
        uint32_t access_count;
        std::chrono::steady_clock::time_point last_access;
        size_t memory_size;
    };
    
    [[nodiscard]] std::optional<EntryInfo> get_with_info(std::string_view key) const {
        return with_lock([&]() -> std::optional<EntryInfo> {
            cache_requests.fetch_add(1, std::memory_order_relaxed);
            
            const std::string key_str(key);
            if (auto it = cache_entries.find(key_str); it != cache_entries.end()) {
                cache_hits.fetch_add(1, std::memory_order_relaxed);
                auto& entry = *it->second;
                update_access_order(it->first, entry);
                
                return EntryInfo{
                    .tokens = entry.tokens,
                    .access_count = entry.access_count.load(),
                    .last_access = entry.last_access,
                    .memory_size = entry.memory_size
                };
            }
            
            return std::nullopt;
        });
    }
      // Optimized batch operations for bidirectional lookups
    [[nodiscard]] std::vector<std::pair<std::string, std::vector<llama_token>>> 
    get_tokens_batch(const std::vector<std::string>& keys) const {
        return with_lock([&]() {
            std::vector<std::pair<std::string, std::vector<llama_token>>> results;
            results.reserve(keys.size());
            
            for (const auto& key : keys) {
                if (auto tokens = get_tokens(key)) {
                    results.emplace_back(key, std::move(*tokens));
                }
            }
            
            return results;
        });
    }
    
    // Memory-efficient iteration over cache entries
    template<typename Func>
    void for_each_entry(Func&& func) const {
        return with_lock([&]() {
            for (const auto& [key, entry] : cache_entries) {
                func(key, entry->tokens, entry->access_count.load());
            }
        });
    }
  
    // Hash function for token vectors to enable reverse lookup
    [[nodiscard]] std::string hash_tokens(const std::vector<llama_token>& tokens) const {
        if (tokens.empty()) {
            return "empty_tokens";
        }
        
        // Use a simple but effective hash combining all token values
        std::hash<llama_token> hasher;
        size_t hash_value = tokens.size();  // Start with size for better distribution
        
        for (size_t i = 0; i < tokens.size(); ++i) {
            // Combine hash values using a variation of boost::hash_combine
            hash_value ^= hasher(tokens[i]) + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);
        }
        
        // Convert to string for use as map key
        return "tokens_" + std::to_string(hash_value);
    }
    
    // NEW: Advanced bidirectional operations
    
    // Check if tokens exist in cache (reverse lookup)
    [[nodiscard]] bool contains_tokens(const std::vector<llama_token>& tokens) const {
        return with_lock([&]() {
            const std::string token_hash = hash_tokens(tokens);
            if (auto it = token_hash_map.find(token_hash); it != token_hash_map.end()) {
                // Verify exact match to handle hash collisions
                return (*it->second)->tokens == tokens;
            }
            return false;
        });
    }
    
    // Get both text and tokens by text key (useful for validation)
    [[nodiscard]] std::optional<std::pair<std::string, std::vector<llama_token>>> 
    get_bidirectional(std::string_view key) const {
        return with_lock([&]() -> std::optional<std::pair<std::string, std::vector<llama_token>>> {
            cache_requests.fetch_add(1, std::memory_order_relaxed);
            
            const std::string key_str(key);
            if (auto it = cache_entries.find(key_str); it != cache_entries.end()) {
                cache_hits.fetch_add(1, std::memory_order_relaxed);
                update_access_order(it->first, *it->second);
                return std::make_pair(it->second->text, it->second->tokens);
            }
            
            return std::nullopt;
        });
    }
    
    // Batch reverse lookup for multiple token vectors
    [[nodiscard]] std::vector<std::pair<std::vector<llama_token>, std::string>> 
    get_text_batch(const std::vector<std::vector<llama_token>>& token_vectors) const {
        return with_lock([&]() {
            std::vector<std::pair<std::vector<llama_token>, std::string>> results;
            results.reserve(token_vectors.size());
            
            for (const auto& tokens : token_vectors) {
                if (auto text = get_text(tokens)) {
                    results.emplace_back(tokens, std::move(*text));
                }
            }
            
            return results;
        });
    }
    
    // Add entry to both forward and reverse lookup maps
    void add_bidirectional_entry(std::string text_key, std::string text, std::vector<llama_token> tokens) const {
        auto entry = std::make_unique<CacheEntry>(std::move(text), std::move(tokens));
        const size_t entry_memory = entry->memory_size;
        const std::string token_hash = hash_tokens(entry->tokens);
        
        // Check if text key already exists (update case)
        auto existing_it = cache_entries.find(text_key);
        if (existing_it != cache_entries.end()) {
            // Remove old reverse mapping
            const std::string old_token_hash = hash_tokens(existing_it->second->tokens);
            token_hash_map.erase(old_token_hash);
            
            // Update existing entry
            const int64_t memory_delta = static_cast<int64_t>(entry_memory) - 
                                       static_cast<int64_t>(existing_it->second->memory_size);
            existing_it->second = std::move(entry);
            update_memory_usage(memory_delta);
            update_access_order(existing_it->first, *existing_it->second);
            
            // Add new reverse mapping
            token_hash_map[token_hash] = &(existing_it->second);
        } else {
            // Add new entry
            auto [inserted_it, was_inserted] = cache_entries.emplace(std::move(text_key), std::move(entry));
            if (was_inserted) {
                update_memory_usage(entry_memory);
                access_order.push_front(inserted_it->first);
                access_iterators[inserted_it->first] = access_order.begin();
                
                // Add reverse mapping
                token_hash_map[token_hash] = &(inserted_it->second);
            }
        }
    }
    
    // Remove entry from both forward and reverse lookup maps
    void remove_bidirectional_entry(const std::string& text_key) const {
        auto it = cache_entries.find(text_key);
        if (it != cache_entries.end()) {
            // Remove reverse mapping
            const std::string token_hash = hash_tokens(it->second->tokens);
            token_hash_map.erase(token_hash);
            
            // Remove forward mapping and update memory
            update_memory_usage(-static_cast<int64_t>(it->second->memory_size));
            cache_entries.erase(it);
            access_iterators.erase(text_key);
        }
    }
};

//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODING DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//
