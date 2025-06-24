// TokenCache.hpp - Streamlined bidirectional token cache: Text↔Tokens with single cache reservoir
// Provides efficient caching of tokenization results in both directions without redundancy.
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
// 1.  Minimalism & Performance: Deliver lean, efficient solutions; no unused helpers or scaffolding.
// 2.  Redundancy Elimination: Remove unused, obsolete, and legacy code.
// 3.  Consistent Style: Adopt a uniform coding style and structure for clarity and maintainability.
// 4.  Documentation: Write concise comments that explain complex logic and key design decisions.
// 5.  Zero Magic & Strong Typing: Replace magic literals with named constants, enums, or constexpr.
// 6.  Function Boundaries: Define clear responsibilities; reduce overlap and avoid unnecessary layers.
// 7.  Core Preservation: Streamline code while safeguarding essential features.
// 8.  Const-Correctness & Immutability: Mark variables, parameters, and methods as const wherever possible.
// 9.  RAII & Resource Safety: Encapsulate resource acquisition/release in constructors/destructors.
// 10. Standard Library Preference: Favor STL algorithms and containers over custom loops.
// 11. Cross-Platform Portability: Use fixed-width types and proper initialization.
// 12. Thread Safety: Define and document thread-safety contracts; protect shared state with mutexes.
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
#include <numeric>
#include "llama-cpp.h"
#include "LogHandler.hpp"  // Add logging support

// Enhanced bidirectional token cache - Text↔Tokens with single cache reservoir
class TokenCache {
public:
    // Cache statistics with direct access
    mutable std::atomic<size_t> cache_hits{0};
    mutable std::atomic<size_t> cache_requests{0};
    mutable std::atomic<size_t> memory_usage_bytes{0};
    size_t max_cache_size;

    // Performance tracking for debugging
    mutable std::atomic<size_t> text_lookups{0};
    mutable std::atomic<size_t> token_lookups{0};
    mutable std::atomic<size_t> evictions{0};
    mutable std::atomic<size_t> memory_reclaimed_bytes{0};

    // Eviction policy options
    enum class EvictionPolicy : uint8_t {
        LRU = 0,        // Least Recently Used (default)
        LFU = 1,        // Least Frequently Used
        FIFO = 2        // First In, First Out
    };

private:
    // Named constants
    static constexpr float CACHE_PREEMPTIVE_THRESHOLD = 0.9f;
    static constexpr float CACHE_TRIM_TARGET_RATIO = 0.7f;
    static constexpr size_t DEFAULT_CACHE_SIZE = 2048;
    static constexpr size_t INITIAL_RESERVE_SIZE = 512;
    
    // Unified cache entry for bidirectional lookup
    struct CacheEntry {
        std::string text;
        std::vector<llama_token> tokens;
        mutable std::atomic<uint32_t> access_count{1};
        mutable std::chrono::steady_clock::time_point last_access;
        size_t memory_size;
          CacheEntry(std::string text_, std::vector<llama_token> tokens_) noexcept
            : text(std::move(text_)),
              tokens(std::move(tokens_)), 
              last_access(std::chrono::steady_clock::now()),
              memory_size(text.size() + tokens.size() * sizeof(llama_token) + sizeof(CacheEntry)) {}
    };

    // Unified cache storage with bidirectional access
    mutable std::unordered_map<std::string, std::unique_ptr<CacheEntry>> text_to_entry;  // text key -> entry
    mutable std::unordered_map<std::string, CacheEntry*> token_hash_to_entry;           // token hash -> entry
    mutable std::list<std::string> access_order;  // LRU tracking using text keys
    mutable std::unordered_map<std::string, std::list<std::string>::iterator> access_iterators;
      // Configuration - const for better optimization
    const EvictionPolicy eviction_policy = EvictionPolicy::LRU;
    const bool thread_safe = false;
    mutable std::shared_mutex cache_mutex;// Memory tracking - marked static for better optimization
    static void update_memory_usage_impl(std::atomic<size_t>& memory_bytes, const int64_t delta) noexcept {
        memory_bytes.fetch_add(delta, std::memory_order_relaxed);
    }
    
    void update_memory_usage(const int64_t delta) const noexcept {
        update_memory_usage_impl(memory_usage_bytes, delta);
    }    // Hash function for token vectors
    [[nodiscard]] std::string hash_tokens(const std::vector<llama_token>& tokens) const noexcept {
        if (tokens.empty()) [[unlikely]] {
            return "empty_tokens";
        }
        
        static const std::hash<llama_token> hasher{};
        static constexpr size_t HASH_CONSTANT = 0x9e3779b9;
        
        size_t hash_value = tokens.size();
        
        // Use STL algorithm for better optimization
        hash_value = std::accumulate(tokens.begin(), tokens.end(), hash_value,
            [](const size_t acc, const llama_token token) noexcept {
                return acc ^ (hasher(token) + HASH_CONSTANT + (acc << 6) + (acc >> 2));
            });
        
        return "tokens_" + std::to_string(hash_value);
    }
      // Update access order for LRU
    void update_access_order(const std::string& text_key, CacheEntry& entry) const noexcept {
        entry.last_access = std::chrono::steady_clock::now();
        entry.access_count.fetch_add(1, std::memory_order_relaxed);
        
        if (eviction_policy == EvictionPolicy::LRU) [[likely]] {
            if (const auto it = access_iterators.find(text_key); it != access_iterators.end()) [[likely]] {
                access_order.splice(access_order.begin(), access_order, it->second);
            }
        }
    }    // Add bidirectional entry to cache
    void add_entry_internal(std::string text_key, std::string text, std::vector<llama_token> tokens) const {
        if (text_to_entry.size() >= static_cast<size_t>(max_cache_size * CACHE_PREEMPTIVE_THRESHOLD)) [[unlikely]] {
            trim_cache();
        }
        
        auto entry = std::make_unique<CacheEntry>(std::move(text), std::move(tokens));
        const size_t entry_memory = entry->memory_size;
        const std::string token_hash = hash_tokens(entry->tokens);
        
        // Check if text key already exists
        if (const auto existing_it = text_to_entry.find(text_key); existing_it != text_to_entry.end()) [[unlikely]] {
            // Remove old reverse mapping
            const auto old_hash = hash_tokens(existing_it->second->tokens);
            token_hash_to_entry.erase(old_hash);
            
            // Update existing entry
            const int64_t memory_delta = static_cast<int64_t>(entry_memory) - 
                                       static_cast<int64_t>(existing_it->second->memory_size);
            existing_it->second = std::move(entry);
            token_hash_to_entry[token_hash] = existing_it->second.get();
            update_memory_usage(memory_delta);
            update_access_order(existing_it->first, *existing_it->second);
        } else [[likely]] {
            // Add new entry - this is the common path
            if (auto [inserted_it, was_inserted] = text_to_entry.emplace(std::move(text_key), std::move(entry)); was_inserted) [[likely]] {
                token_hash_to_entry[token_hash] = inserted_it->second.get();
                update_memory_usage(entry_memory);
                access_order.push_front(inserted_it->first);
                access_iterators[inserted_it->first] = access_order.begin();
            }
        }
    }
      // Enhanced cache trimming with debug logging
    void trim_cache() const {
        const size_t target_size = static_cast<size_t>(max_cache_size * CACHE_TRIM_TARGET_RATIO);
        size_t initial_size = text_to_entry.size();
        size_t total_reclaimed = 0;        // Debug logging for cache pressure
        if (initial_size > max_cache_size * 0.8f) {
            TOKEN_CACHE_LOG_DEBUG("Cache trim triggered - pressure at " + 
                     std::to_string(static_cast<float>(initial_size) / max_cache_size * 100.0f) + 
                     "% (" + std::to_string(initial_size) + "/" + std::to_string(max_cache_size) + ")");
        }
        
        while (text_to_entry.size() > target_size && !text_to_entry.empty()) {
            std::string victim_key;
            
            switch (eviction_policy) {
                case EvictionPolicy::LRU:
                case EvictionPolicy::FIFO:
                    if (!access_order.empty()) [[likely]] {
                        victim_key = access_order.back();
                        access_order.pop_back();
                    }
                    break;
                      case EvictionPolicy::LFU: {
                    const auto min_it = std::min_element(text_to_entry.begin(), text_to_entry.end(),
                        [](const auto& a, const auto& b) noexcept {
                            return a.second->access_count.load(std::memory_order_relaxed) < 
                                   b.second->access_count.load(std::memory_order_relaxed);
                        });
                    if (min_it != text_to_entry.end()) [[likely]] {
                        victim_key = min_it->first;
                    }
                    break;
                }
            }
              if (!victim_key.empty()) [[likely]] {
                if (const auto it = text_to_entry.find(victim_key); it != text_to_entry.end()) [[likely]] {
                    // Remove reverse mapping
                    const auto token_hash = hash_tokens(it->second->tokens);
                    token_hash_to_entry.erase(token_hash);
                      const size_t reclaimed_memory = it->second->memory_size;
                    total_reclaimed += reclaimed_memory;
                    update_memory_usage(-static_cast<int64_t>(reclaimed_memory));
                    text_to_entry.erase(it);
                    access_iterators.erase(victim_key);
                    
                    // Track eviction metrics
                    evictions.fetch_add(1, std::memory_order_relaxed);
                    memory_reclaimed_bytes.fetch_add(reclaimed_memory, std::memory_order_relaxed);
                }            } else [[unlikely]] {
                TOKEN_CACHE_LOG_DEBUG("Cache trim: No victim found, breaking");
                break;
            }
        }        // Debug logging for trim results
        if (initial_size != text_to_entry.size()) {
            TOKEN_CACHE_LOG_DEBUG("Cache trim completed: " + 
                     std::to_string(initial_size - text_to_entry.size()) + " entries removed, " +
                     std::to_string(total_reclaimed) + " bytes reclaimed");
        }
    }
      // Thread-safe wrappers
    template<typename Func>
    auto with_lock(Func&& func) const noexcept(noexcept(func())) {
        if constexpr (std::is_void_v<std::invoke_result_t<Func>>) {
            if (thread_safe) [[unlikely]] {
                const std::shared_lock lock(cache_mutex);
                func();
            } else [[likely]] {
                func();
            }
        } else {
            if (thread_safe) [[unlikely]] {
                const std::shared_lock lock(cache_mutex);
                return func();
            } else [[likely]] {
                return func();
            }
        }
    }
    
    template<typename Func>
    auto with_write_lock(Func&& func) const noexcept(noexcept(func())) {
        if constexpr (std::is_void_v<std::invoke_result_t<Func>>) {
            if (thread_safe) [[unlikely]] {
                const std::unique_lock lock(cache_mutex);
                func();
            } else [[likely]] {
                func();
            }
        } else {
            if (thread_safe) [[unlikely]] {
                const std::unique_lock lock(cache_mutex);
                return func();
            } else [[likely]] {
                return func();
            }
        }
    }

public:    explicit TokenCache(const size_t max_size = DEFAULT_CACHE_SIZE, 
                       const EvictionPolicy policy = EvictionPolicy::LRU,
                       const bool enable_thread_safety = false) noexcept
        : max_cache_size(max_size), eviction_policy(policy), thread_safe(enable_thread_safety) {
        text_to_entry.reserve(INITIAL_RESERVE_SIZE);
        token_hash_to_entry.reserve(INITIAL_RESERVE_SIZE);
        access_iterators.reserve(INITIAL_RESERVE_SIZE);
        
        TOKEN_CACHE_LOG_DEBUG("TokenCache initialized: max_size=" + std::to_string(max_size) + 
                             ", policy=" + std::to_string(static_cast<int>(policy)) + 
                             ", thread_safe=" + (enable_thread_safety ? "true" : "false"));
    }// Configure cache settings
    void configure(const EvictionPolicy policy, const bool enable_thread_safety = false) noexcept {
        const_cast<EvictionPolicy&>(eviction_policy) = policy;
        const_cast<bool&>(thread_safe) = enable_thread_safety;
    }    // Resize cache
    void resize_cache(const size_t new_max_size) {
        return with_write_lock([&]() {
            TOKEN_CACHE_LOG_DEBUG("Resizing cache from " + std::to_string(max_cache_size) + 
                                 " to " + std::to_string(new_max_size) + " entries");
            max_cache_size = new_max_size;
            if (text_to_entry.size() > max_cache_size) [[unlikely]] {
                trim_cache();
            }
        });
    }
      // Text -> Tokens lookup
    [[nodiscard]] std::optional<std::vector<llama_token>> get_tokens(const std::string_view text_key) const noexcept {
        return with_lock([&]() noexcept -> std::optional<std::vector<llama_token>> {
            cache_requests.fetch_add(1, std::memory_order_relaxed);
            text_lookups.fetch_add(1, std::memory_order_relaxed);
            
            const std::string key_str(text_key);
            if (const auto it = text_to_entry.find(key_str); it != text_to_entry.end()) [[likely]] {
                cache_hits.fetch_add(1, std::memory_order_relaxed);
                update_access_order(it->first, *it->second);
                return it->second->tokens;
            }            // Debug logging for frequent cache misses (potential performance issue)
            if (cache_requests.load() % 100 == 0) {
                float hit_ratio = static_cast<float>(cache_hits.load()) / cache_requests.load();
                if (hit_ratio < 0.5f) {
                    TOKEN_CACHE_LOG_DEBUG("Low hit ratio detected: " + 
                             std::to_string(hit_ratio * 100.0f) + "% (requests: " + 
                             std::to_string(cache_requests.load()) + ")");
                }
            }
            
            return std::nullopt;
        });
    }
      // Tokens -> Text lookup (reverse) with debug logging
    [[nodiscard]] std::optional<std::string> get_text(const std::vector<llama_token>& tokens) const noexcept {
        return with_lock([&]() noexcept -> std::optional<std::string> {
            cache_requests.fetch_add(1, std::memory_order_relaxed);
            token_lookups.fetch_add(1, std::memory_order_relaxed);
            
            const std::string token_hash = hash_tokens(tokens);
            if (const auto it = token_hash_to_entry.find(token_hash); it != token_hash_to_entry.end()) [[likely]] {
                auto& entry = *(it->second);
                
                // Verify exact match (hash collision protection)
                if (entry.tokens == tokens) [[likely]] {
                    cache_hits.fetch_add(1, std::memory_order_relaxed);
                    
                    // Find text key for LRU update using optimized search
                    const auto text_it = std::find_if(text_to_entry.begin(), text_to_entry.end(),
                        [&entry](const auto& pair) noexcept { return pair.second.get() == &entry; });
                    
                    if (text_it != text_to_entry.end()) [[likely]] {
                        update_access_order(text_it->first, entry);
                    }
                    return entry.text;                } else {
                    // Hash collision detected - this is rare but important to log
                    TOKEN_CACHE_LOG_DEBUG("Hash collision detected for token sequence");
                }
            }
            
            return std::nullopt;
        });
    }
      // Store bidirectional mapping: Text ↔ Tokens
    void put(std::string text_key, std::string text, std::vector<llama_token> tokens) const {
        return with_write_lock([&]() {
            add_entry_internal(std::move(text_key), std::move(text), std::move(tokens));
        });
    }
      // Bulk operations
    void put_batch(std::vector<std::tuple<std::string, std::string, std::vector<llama_token>>> entries) const {
        if (entries.empty()) [[unlikely]] return;  // Early exit for empty batch
        
        return with_write_lock([&]() {
            // Reserve space to minimize rehashing during batch insertion
            const size_t new_capacity = text_to_entry.size() + entries.size();
            if (new_capacity > text_to_entry.bucket_count()) [[unlikely]] {
                text_to_entry.reserve(new_capacity);
                token_hash_to_entry.reserve(new_capacity);
                access_iterators.reserve(new_capacity);
            }
              // Use STL for_each for better optimization than range-based for
            std::for_each(entries.begin(), entries.end(), [this](auto& entry) {
                auto& [key, text, tokens] = entry;
                add_entry_internal(std::move(key), std::move(text), std::move(tokens));
            });
        });
    }
      // Check existence
    [[nodiscard]] bool contains_text(const std::string_view text_key) const noexcept {
        return with_lock([&]() noexcept {
            return text_to_entry.find(std::string(text_key)) != text_to_entry.end();
        });
    }
    
    [[nodiscard]] bool contains_tokens(const std::vector<llama_token>& tokens) const noexcept {
        return with_lock([&]() noexcept {
            const std::string token_hash = hash_tokens(tokens);
            if (const auto it = token_hash_to_entry.find(token_hash); it != token_hash_to_entry.end()) [[likely]] {
                return it->second->tokens == tokens;
            }
            return false;
        });
    }
    
    // Clear cache
    void clear() const {
        return with_write_lock([&]() {
            size_t entries_cleared = text_to_entry.size();
            size_t memory_cleared = memory_usage_bytes.load();
            
            text_to_entry.clear();
            token_hash_to_entry.clear();
            access_order.clear();
            access_iterators.clear();            cache_hits.store(0, std::memory_order_relaxed);
            cache_requests.store(0, std::memory_order_relaxed);
            memory_usage_bytes.store(0, std::memory_order_relaxed);
            
            TOKEN_CACHE_LOG_DEBUG("Cache cleared: " + 
                     std::to_string(entries_cleared) + " entries, " + 
                     std::to_string(memory_cleared) + " bytes freed");
        });
    }
      // Enhanced statistics
    struct CacheStats {
        size_t hits;
        size_t requests;
        size_t entries;
        size_t max_size;
        size_t memory_bytes;
        float hit_ratio;
        float fill_ratio;
        float memory_efficiency;  // tokens per byte
    };
      [[nodiscard]] CacheStats get_stats() const noexcept {
        return with_lock([&]() noexcept {
            const auto hits = cache_hits.load(std::memory_order_relaxed);
            const auto requests = cache_requests.load(std::memory_order_relaxed);
            const auto memory_bytes = memory_usage_bytes.load(std::memory_order_relaxed);
            const auto entries = text_to_entry.size();
            
            // Use STL algorithm for better optimization
            const size_t total_tokens = std::accumulate(text_to_entry.begin(), text_to_entry.end(), size_t{0},
                [](size_t sum, const auto& pair) noexcept {
                    return sum + pair.second->tokens.size();
                });
            
            return CacheStats{
                .hits = hits,
                .requests = requests,
                .entries = entries,
                .max_size = max_cache_size,
                .memory_bytes = memory_bytes,
                .hit_ratio = (requests > 0) ? static_cast<float>(hits) / static_cast<float>(requests) : 0.0f,
                .fill_ratio = static_cast<float>(entries) / static_cast<float>(max_cache_size),
                .memory_efficiency = (memory_bytes > 0) ? static_cast<float>(total_tokens) / static_cast<float>(memory_bytes) : 0.0f
            };
        });
    }
    
    // Memory optimization
    void compact() const {
        return with_write_lock([&]() {
            auto old_entries = std::move(text_to_entry);
            text_to_entry.clear();
            text_to_entry.reserve(old_entries.size());
            
            for (auto& [key, entry] : old_entries) {
                entry->tokens.shrink_to_fit();
                text_to_entry[key] = std::move(entry);
            }
        });
    }
};

//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODING DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//
