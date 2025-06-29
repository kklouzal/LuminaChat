#pragma once

#include "Logger.hpp"
// llama.cpp includes
#include "llama-cpp.h"
#include <unordered_map>
#include <list>
#include <vector>
#include <string>
#include <string_view>
#include <shared_mutex>
#include <mutex>
#include <functional>
#include <chrono>
#include <atomic>
#include <optional>
#include <memory>
#include <algorithm>
#include <numeric>

// TokenCache: High-performance bidirectional token caching system
// - Unified storage for Text↔Tokens with single cache reservoir (no data duplication)
// - GDSF (GreedyDual-Size-Frequency) eviction policy for optimal cache management
// - Performance statistics tracking with memory usage monitoring
// - Single instance per ModelInfo, shared across contexts
// - Thread-safe operations with shared_mutex for better read performance
// - Batch operations for bulk cache updates
// - Memory compaction and sophisticated cleanup strategies
//
// ENHANCED DESIGN (December 2024):
// - Vocab reference stored during construction/initialization for cleaner API
// - TokenizeText() and DetokenizeTokens() no longer require vocab parameter
// - ModelInfo sets vocab on TokenCache after model loading via SetVocab()
// - Eliminates redundant vocab parameter passing throughout the codebase
// - Unified bidirectional storage eliminates memory duplication
// - Hash collision protection for robust token sequence handling

struct CacheStats {
    std::atomic<size_t> text_to_token_hits{0};
    std::atomic<size_t> text_to_token_misses{0};
    std::atomic<size_t> token_to_text_hits{0};
    std::atomic<size_t> token_to_text_misses{0};
    std::atomic<size_t> memory_usage_bytes{0};
    std::atomic<size_t> evictions{0};
    std::atomic<size_t> memory_reclaimed_bytes{0};
    
    // Default constructor
    CacheStats() = default;
    
    // Copy constructor
    CacheStats(const CacheStats& other) 
        : text_to_token_hits{other.text_to_token_hits.load()}
        , text_to_token_misses{other.text_to_token_misses.load()}
        , token_to_text_hits{other.token_to_text_hits.load()}
        , token_to_text_misses{other.token_to_text_misses.load()}
        , memory_usage_bytes{other.memory_usage_bytes.load()}
        , evictions{other.evictions.load()}
        , memory_reclaimed_bytes{other.memory_reclaimed_bytes.load()} {}
    
    // Copy assignment operator
    CacheStats& operator=(const CacheStats& other) {
        if (this != &other) {
            text_to_token_hits = other.text_to_token_hits.load();
            text_to_token_misses = other.text_to_token_misses.load();
            token_to_text_hits = other.token_to_text_hits.load();
            token_to_text_misses = other.token_to_text_misses.load();
            memory_usage_bytes = other.memory_usage_bytes.load();
            evictions = other.evictions.load();
            memory_reclaimed_bytes = other.memory_reclaimed_bytes.load();
        }
        return *this;
    }
    
    float GetTextToTokenHitRatio() const {
        size_t total = text_to_token_hits + text_to_token_misses;
        return total > 0 ? static_cast<float>(text_to_token_hits) / total : 0.0f;
    }
    
    float GetTokenToTextHitRatio() const {
        size_t total = token_to_text_hits + token_to_text_misses;
        return total > 0 ? static_cast<float>(token_to_text_hits) / total : 0.0f;
    }
    
    float GetOverallHitRatio() const {
        size_t total_hits = GetTotalHits();
        size_t total_requests = GetTotalRequests();
        return total_requests > 0 ? static_cast<float>(total_hits) / total_requests : 0.0f;
    }
    
    size_t GetTotalHits() const {
        return text_to_token_hits + token_to_text_hits;
    }
    
    size_t GetTotalRequests() const {
        return text_to_token_hits + text_to_token_misses + 
               token_to_text_hits + token_to_text_misses;
    }
    
    float GetMemoryEfficiency() const {
        size_t memory = memory_usage_bytes.load();
        return memory > 0 ? static_cast<float>(GetTotalRequests()) / memory : 0.0f;
    }
};

// Unified cache entry for bidirectional lookup - eliminates data duplication
struct UnifiedCacheEntry {
    std::string text;
    std::vector<int32_t> tokens;
    mutable std::atomic<uint32_t> access_count{1};
    mutable std::chrono::steady_clock::time_point last_accessed;
    size_t memory_size;
    mutable double gdsf_priority{0.0}; // GDSF priority value (H = L + F/S)
    
    UnifiedCacheEntry(std::string text_data, std::vector<int32_t> token_data) noexcept
        : text(std::move(text_data))
        , tokens(std::move(token_data))
        , last_accessed(std::chrono::steady_clock::now())
        , memory_size(text.capacity() + tokens.capacity() * sizeof(int32_t) + sizeof(UnifiedCacheEntry))
        , gdsf_priority(1.0 / memory_size) {} // Initial priority F/S where F=1, S=memory_size
    
    void UpdateAccess() const noexcept {
        last_accessed = std::chrono::steady_clock::now();
        access_count.fetch_add(1, std::memory_order_relaxed);
        // GDSF priority will be recalculated during eviction based on current inflation factor
    }
};

class TokenCache {
private:
    // Performance constants
    static constexpr float CACHE_PREEMPTIVE_THRESHOLD = 0.9f;
    static constexpr float CACHE_TRIM_TARGET_RATIO = 0.7f;
    static constexpr size_t DEFAULT_CACHE_SIZE = 10000;
    static constexpr size_t INITIAL_RESERVE_SIZE = 512;
    
    // Unified bidirectional cache storage - single source of truth
    mutable std::unordered_map<std::string, std::unique_ptr<UnifiedCacheEntry>> text_to_entry;  // text key -> entry
    mutable std::unordered_map<std::string, UnifiedCacheEntry*> token_hash_to_entry;           // token hash -> entry
    
    // GDSF eviction policy state
    mutable double gdsf_inflation_factor{0.0}; // L value in GDSF algorithm
    
    // Vocab reference - stored during construction for direct tokenization/detokenization
    const llama_vocab* vocab;
    
    // Thread safety with shared_mutex for better read performance
    mutable std::shared_mutex cache_mutex;
    
    // Performance tracking (mutable for const method access)
    mutable CacheStats stats;
    
    // Cache management
    size_t max_cache_size;
    size_t cleanup_threshold;
    
    // Callback for cache invalidation notifications (optional)
    std::function<void(const std::string&)> invalidation_callback;
    
    // Helper methods
    [[nodiscard]] std::string GenerateTokenHash(const std::vector<int32_t>& tokens) const noexcept {
        if (tokens.empty()) [[unlikely]] {
            return "empty_tokens";
        }
        
        // Use STL algorithm for better optimization
        static const std::hash<int32_t> hasher{};
        static constexpr size_t HASH_CONSTANT = 0x9e3779b9;
        
        size_t hash_value = tokens.size();
        hash_value = std::accumulate(tokens.begin(), tokens.end(), hash_value,
            [](const size_t acc, const int32_t token) noexcept {
                return acc ^ (hasher(token) + HASH_CONSTANT + (acc << 6) + (acc >> 2));
            });
        
        return "tokens_" + std::to_string(hash_value);
    }
    
    // Update memory usage tracking
    void UpdateMemoryUsage(const int64_t delta) const noexcept {
        stats.memory_usage_bytes.fetch_add(delta, std::memory_order_relaxed);
    }
    
    // Update GDSF priority for accessed entry
    void UpdateGDSFPriority(const UnifiedCacheEntry& entry) const noexcept {
        entry.UpdateAccess();
        // GDSF priority = L + F/S where L is inflation factor, F is frequency, S is size
        entry.gdsf_priority = gdsf_inflation_factor + 
                             static_cast<double>(entry.access_count.load(std::memory_order_relaxed)) / 
                             static_cast<double>(entry.memory_size);
    }
    
    // Add unified entry to bidirectional cache
    void AddEntryInternal(std::string text_key, std::string text, std::vector<int32_t> tokens) const {
        // Trigger cleanup if approaching capacity
        if (text_to_entry.size() >= static_cast<size_t>(max_cache_size * CACHE_PREEMPTIVE_THRESHOLD)) [[unlikely]] {
            CleanupOldEntries();
        }
        
        auto entry = std::make_unique<UnifiedCacheEntry>(std::move(text), std::move(tokens));
        const size_t entry_memory = entry->memory_size;
        const std::string token_hash = GenerateTokenHash(entry->tokens);
        
        // Check if text key already exists
        if (const auto existing_it = text_to_entry.find(text_key); existing_it != text_to_entry.end()) [[unlikely]] {
            // Remove old reverse mapping
            const auto old_hash = GenerateTokenHash(existing_it->second->tokens);
            token_hash_to_entry.erase(old_hash);
            
            // Update existing entry
            const int64_t memory_delta = static_cast<int64_t>(entry_memory) - 
                                       static_cast<int64_t>(existing_it->second->memory_size);
            existing_it->second = std::move(entry);
            token_hash_to_entry[token_hash] = existing_it->second.get();
            UpdateMemoryUsage(memory_delta);
            UpdateGDSFPriority(*existing_it->second);
        } else [[likely]] {
            // Add new entry - this is the common path
            if (auto [inserted_it, was_inserted] = text_to_entry.emplace(std::move(text_key), std::move(entry)); was_inserted) [[likely]] {
                token_hash_to_entry[token_hash] = inserted_it->second.get();
                UpdateMemoryUsage(entry_memory);
                UpdateGDSFPriority(*inserted_it->second);
            }
        }
    }
    
    void CleanupOldEntries() const {
        const size_t target_size = static_cast<size_t>(max_cache_size * CACHE_TRIM_TARGET_RATIO);
        size_t initial_size = text_to_entry.size();
        size_t total_reclaimed = 0;
        
        // Debug logging for cache pressure
        if (initial_size > max_cache_size * 0.8f) {
            LOG_TokenCache("Cache trim triggered - pressure at " + 
                     std::to_string(static_cast<float>(initial_size) / max_cache_size * 100.0f) + 
                     "% (" + std::to_string(initial_size) + "/" + std::to_string(max_cache_size) + ")");
        }
        
        while (text_to_entry.size() > target_size && !text_to_entry.empty()) {
            // GDSF Algorithm: Find entry with minimum priority value
            // H = L + F/S where L = inflation factor, F = frequency, S = size
            
            // Update all priorities with current inflation factor before selection
            for (auto& [key, entry] : text_to_entry) {
                const uint32_t frequency = entry->access_count.load(std::memory_order_relaxed);
                entry->gdsf_priority = gdsf_inflation_factor + 
                                     static_cast<double>(frequency) / static_cast<double>(entry->memory_size);
            }
            
            // Find victim with minimum GDSF priority
            auto victim_it = std::min_element(text_to_entry.begin(), text_to_entry.end(),
                [](const auto& a, const auto& b) noexcept {
                    return a.second->gdsf_priority < b.second->gdsf_priority;
                });
            
            if (victim_it != text_to_entry.end()) [[likely]] {
                // Update inflation factor to the priority of the evicted entry
                // This ensures that subsequent entries need higher priority to avoid eviction
                gdsf_inflation_factor = victim_it->second->gdsf_priority;
                
                // Remove reverse mapping
                const auto token_hash = GenerateTokenHash(victim_it->second->tokens);
                token_hash_to_entry.erase(token_hash);
                
                // Track memory reclaimed
                total_reclaimed += victim_it->second->memory_size;
                UpdateMemoryUsage(-static_cast<int64_t>(victim_it->second->memory_size));
                
                // Remove main entry
                text_to_entry.erase(victim_it);
                stats.evictions.fetch_add(1, std::memory_order_relaxed);
            } else [[unlikely]] {
                LOG_DEBUG_TokenCache("Cache trim: No victim found, breaking");
                break;
            }
        }
        
        // Update reclaimed memory stats
        stats.memory_reclaimed_bytes.fetch_add(total_reclaimed, std::memory_order_relaxed);
        
        // Debug logging for trim results
        if (initial_size != text_to_entry.size()) {
            LOG_TokenCache("GDSF cache trim completed: " + 
                     std::to_string(initial_size - text_to_entry.size()) + " entries removed, " +
                     std::to_string(total_reclaimed) + " bytes reclaimed, inflation factor: " +
                     std::to_string(gdsf_inflation_factor));
        }
    }
    
public:
    explicit TokenCache(size_t cache_size = DEFAULT_CACHE_SIZE, 
                       const llama_vocab* vocab_ptr = nullptr) 
        : max_cache_size(cache_size)
        , cleanup_threshold(static_cast<size_t>(cache_size * 0.8)) // 80% of max size
        , vocab(vocab_ptr)
    {
        // Reserve initial capacity for better performance
        text_to_entry.reserve(INITIAL_RESERVE_SIZE);
        token_hash_to_entry.reserve(INITIAL_RESERVE_SIZE);
        
        LOG_TokenCache("TokenCache initialized with max size: " + std::to_string(max_cache_size) + 
                      ", using GDSF eviction policy" +
                      ", vocab: " + std::string(vocab ? "provided" : "null"));
    }
    
    ~TokenCache() {
        std::unique_lock<std::shared_mutex> lock(cache_mutex);
        LOG_TokenCache("TokenCache destroyed - Final stats: " + 
            std::to_string(stats.GetTotalHits()) + " hits / " + 
            std::to_string(stats.GetTotalRequests()) + " requests (" +
            std::to_string(static_cast<int>(stats.GetOverallHitRatio() * 100)) + "% hit rate)");
    }
    
    // Resize cache capacity
    void ResizeCache(size_t new_max_size) {
        std::unique_lock<std::shared_mutex> lock(cache_mutex);
        LOG_TokenCache("Resizing cache from " + std::to_string(max_cache_size) + 
                      " to " + std::to_string(new_max_size) + " entries");
        max_cache_size = new_max_size;
        cleanup_threshold = static_cast<size_t>(new_max_size * 0.8);
        
        if (text_to_entry.size() > max_cache_size) [[unlikely]] {
            CleanupOldEntries();
        }
    }
    
    // Set vocab after model loading (thread-safe)
    void SetVocab(const llama_vocab* vocab_ptr) {
        std::unique_lock<std::shared_mutex> lock(cache_mutex);
        vocab = vocab_ptr;
        LOG_TokenCache("Vocab updated: " + std::string(vocab ? "provided" : "null"));
    }
    
    // Optional callback registration for cache invalidation notifications
    // ContextInfo (higher) can register with TokenCache (lower) for cache events
    void RegisterInvalidationCallback(std::function<void(const std::string&)> callback) {
        std::unique_lock<std::shared_mutex> lock(cache_mutex);
        invalidation_callback = std::move(callback);
        LOG_TokenCache("Invalidation callback registered");
    }
    
    // ================================================================
    // ENHANCED BIDIRECTIONAL CACHE OPERATIONS
    // ================================================================
    
    // Text -> Tokens lookup (primary interface)
    [[nodiscard]] std::optional<std::vector<int32_t>> GetTokensFromText(std::string_view text_key) const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex);
        
        const std::string key_str(text_key);
        if (const auto it = text_to_entry.find(key_str); it != text_to_entry.end()) [[likely]] {
            stats.text_to_token_hits.fetch_add(1, std::memory_order_relaxed);
            UpdateGDSFPriority(*it->second);
            LOG_DEBUG_TokenCache("Cache hit for text (" + std::to_string(text_key.length()) + " chars)");
            return it->second->tokens;
        }
        
        stats.text_to_token_misses.fetch_add(1, std::memory_order_relaxed);
        LOG_DEBUG_TokenCache("Cache miss for text (" + std::to_string(text_key.length()) + " chars)");
        return std::nullopt;
    }
    
    // Tokens -> Text lookup (reverse lookup with hash collision protection)
    [[nodiscard]] std::optional<std::string> GetTextFromTokens(const std::vector<int32_t>& tokens) const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex);
        
        const std::string token_hash = GenerateTokenHash(tokens);
        if (const auto it = token_hash_to_entry.find(token_hash); it != token_hash_to_entry.end()) [[likely]] {
            auto& entry = *(it->second);
            
            // Verify exact match (hash collision protection)
            if (entry.tokens == tokens) [[likely]] {
                stats.token_to_text_hits.fetch_add(1, std::memory_order_relaxed);
                UpdateGDSFPriority(entry);
                LOG_DEBUG_TokenCache("Cache hit for tokens (" + std::to_string(tokens.size()) + " tokens)");
                return entry.text;
            } else {
                LOG_DEBUG_TokenCache("Hash collision detected for token sequence");
            }
        }
        
        stats.token_to_text_misses.fetch_add(1, std::memory_order_relaxed);
        LOG_DEBUG_TokenCache("Cache miss for tokens (" + std::to_string(tokens.size()) + " tokens)");
        return std::nullopt;
    }
    
    // Store bidirectional mapping: Text ↔ Tokens
    void StoreBidirectional(std::string text_key, std::string text, std::vector<int32_t> tokens) const {
        std::unique_lock<std::shared_mutex> lock(cache_mutex);
        AddEntryInternal(std::move(text_key), std::move(text), std::move(tokens));
    }
    
    // Batch operations for better performance
    void StoreBatch(std::vector<std::tuple<std::string, std::string, std::vector<int32_t>>> entries) const {
        if (entries.empty()) [[unlikely]] return;
        
        std::unique_lock<std::shared_mutex> lock(cache_mutex);
        
        // Reserve space to minimize rehashing during batch insertion
        const size_t new_capacity = text_to_entry.size() + entries.size();
        if (new_capacity > text_to_entry.bucket_count()) [[unlikely]] {
            text_to_entry.reserve(new_capacity);
            token_hash_to_entry.reserve(new_capacity);
        }
        
        // Use STL for_each for better optimization
        std::for_each(entries.begin(), entries.end(), [this](auto& entry) {
            auto& [text_key, text, tokens] = entry;
            AddEntryInternal(std::move(text_key), std::move(text), std::move(tokens));
        });
    }
    
    // Check existence
    [[nodiscard]] bool ContainsText(std::string_view text_key) const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex);
        return text_to_entry.find(std::string(text_key)) != text_to_entry.end();
    }
    
    [[nodiscard]] bool ContainsTokens(const std::vector<int32_t>& tokens) const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex);
        const std::string token_hash = GenerateTokenHash(tokens);
        if (const auto it = token_hash_to_entry.find(token_hash); it != token_hash_to_entry.end()) [[likely]] {
            return it->second->tokens == tokens; // Verify exact match
        }
        return false;
    }
    

    
    // ================================================================
    // CACHE INVALIDATION AND MANAGEMENT
    // ================================================================
    
    // Invalidate specific text entry
    void InvalidateText(const std::string& text) {
        std::unique_lock<std::shared_mutex> lock(cache_mutex);
        
        auto it = text_to_entry.find(text);
        if (it != text_to_entry.end()) {
            // Remove reverse mapping
            const auto token_hash = GenerateTokenHash(it->second->tokens);
            token_hash_to_entry.erase(token_hash);
            
            // Update memory tracking
            UpdateMemoryUsage(-static_cast<int64_t>(it->second->memory_size));
            
            text_to_entry.erase(it);
            LOG_DEBUG_TokenCache("Invalidated text cache entry");
            
            if (invalidation_callback) {
                invalidation_callback(text);
            }
        }
    }
    
    void ClearAll() {
        std::unique_lock<std::shared_mutex> lock(cache_mutex);
        
        size_t total_entries = text_to_entry.size();
        size_t memory_cleared = stats.memory_usage_bytes.load();
        
        text_to_entry.clear();
        token_hash_to_entry.clear();
        
        // Reset memory tracking
        stats.memory_usage_bytes.store(0, std::memory_order_relaxed);
        
        LOG_TokenCache("Cleared all cache entries (" + std::to_string(total_entries) + " total, " +
                      std::to_string(memory_cleared) + " bytes freed)");
        
        if (invalidation_callback) {
            invalidation_callback("all");
        }
    }
    
    // Memory compaction to defragment cache
    void CompactMemory() const {
        std::unique_lock<std::shared_mutex> lock(cache_mutex);
        
        // Rebuild the cache with optimal memory layout
        auto old_entries = std::move(text_to_entry);
        text_to_entry.clear();
        token_hash_to_entry.clear();
        
        text_to_entry.reserve(old_entries.size());
        token_hash_to_entry.reserve(old_entries.size());
        
        for (auto& [key, entry] : old_entries) {
            const std::string token_hash = GenerateTokenHash(entry->tokens);
            text_to_entry[key] = std::move(entry);
            token_hash_to_entry[token_hash] = text_to_entry[key].get();
        }
        
        LOG_TokenCache("Memory compaction completed");
    }
    
    // ================================================================
    // PERFORMANCE STATISTICS AND MONITORING
    // ================================================================
    
    // Enhanced statistics with detailed breakdown
    struct DetailedCacheStats {
        size_t unified_entries;
        size_t total_entries;
        size_t max_size;
        size_t memory_bytes;
        float hit_ratio;
        float fill_ratio;
        float memory_efficiency;  // operations per byte
        size_t evictions;
        size_t memory_reclaimed;
        
        // Per-operation stats
        size_t text_to_token_hits;
        size_t text_to_token_misses;
        size_t token_to_text_hits;
        size_t token_to_text_misses;
    };
    
    [[nodiscard]] DetailedCacheStats GetDetailedStats() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex);
        
        const auto total_hits = stats.GetTotalHits();
        const auto total_requests = stats.GetTotalRequests();
        const auto memory_bytes = stats.memory_usage_bytes.load(std::memory_order_relaxed);
        const auto unified_entries = text_to_entry.size();
        
        return DetailedCacheStats{
            .unified_entries = unified_entries,
            .total_entries = unified_entries,
            .max_size = max_cache_size,
            .memory_bytes = memory_bytes,
            .hit_ratio = (total_requests > 0) ? static_cast<float>(total_hits) / total_requests : 0.0f,
            .fill_ratio = static_cast<float>(unified_entries) / max_cache_size,
            .memory_efficiency = (memory_bytes > 0) ? static_cast<float>(total_requests) / memory_bytes : 0.0f,
            .evictions = stats.evictions.load(std::memory_order_relaxed),
            .memory_reclaimed = stats.memory_reclaimed_bytes.load(std::memory_order_relaxed),
            .text_to_token_hits = stats.text_to_token_hits.load(std::memory_order_relaxed),
            .text_to_token_misses = stats.text_to_token_misses.load(std::memory_order_relaxed),
            .token_to_text_hits = stats.token_to_text_hits.load(std::memory_order_relaxed),
            .token_to_text_misses = stats.token_to_text_misses.load(std::memory_order_relaxed)
        };
    }
    
    // Legacy statistics method
    CacheStats GetStats() const {
        std::shared_lock<std::shared_mutex> lock(cache_mutex);
        return stats;
    }
    
    size_t GetCacheSize() const {
        std::shared_lock<std::shared_mutex> lock(cache_mutex);
        return text_to_entry.size();
    }
    
    // Enhanced memory management
    size_t GetMemoryUsage() const {
        std::shared_lock<std::shared_mutex> lock(cache_mutex);
        return stats.memory_usage_bytes.load(std::memory_order_relaxed);
    }
    
    // Get memory breakdown for debugging
    size_t GetDetailedMemoryUsage() const {
        std::shared_lock<std::shared_mutex> lock(cache_mutex);
        
        size_t memory = 0;
        
        // Unified cache entries
        for (const auto& entry : text_to_entry) {
            memory += entry.second->memory_size;
        }
        
        return memory;
    }
    
    void LogStatistics() const {
        auto detailed_stats = GetDetailedStats();
        
        LOG_TokenCache("=== Enhanced Cache Statistics ===");
        LOG_TokenCache("Entries: " + std::to_string(detailed_stats.unified_entries) + " unified = " + 
                      std::to_string(detailed_stats.total_entries) + " total");
        LOG_TokenCache("Capacity: " + std::to_string(detailed_stats.total_entries) + "/" + 
                      std::to_string(detailed_stats.max_size) + " (" + 
                      std::to_string(static_cast<int>(detailed_stats.fill_ratio * 100)) + "% full)");
        LOG_TokenCache("Memory: " + std::to_string(detailed_stats.memory_bytes / 1024) + " KB " +
                      "(efficiency: " + std::to_string(detailed_stats.memory_efficiency) + " ops/byte)");
        LOG_TokenCache("Text→Token: " + std::to_string(detailed_stats.text_to_token_hits) + 
            " hits, " + std::to_string(detailed_stats.text_to_token_misses) + " misses");
        LOG_TokenCache("Token→Text: " + std::to_string(detailed_stats.token_to_text_hits) + 
            " hits, " + std::to_string(detailed_stats.token_to_text_misses) + " misses");
        LOG_TokenCache("Overall: " + std::to_string(static_cast<int>(detailed_stats.hit_ratio * 100)) + 
                      "% hit rate (" + std::to_string(detailed_stats.text_to_token_hits + detailed_stats.token_to_text_hits) + 
                      " total hits)");
        LOG_TokenCache("Evictions: " + std::to_string(detailed_stats.evictions) + 
                      " (" + std::to_string(detailed_stats.memory_reclaimed / 1024) + " KB reclaimed)");
        LOG_TokenCache("Policy: GDSF (GreedyDual-Size-Frequency)");
    }
    

    
    // ================================================================
    // TOKENIZATION AND DETOKENIZATION WITH INTEGRATED CACHING
    // ================================================================
    
    // Primary tokenization method - handles both caching and actual tokenization
    std::vector<int32_t> TokenizeText(const std::string& text, bool add_special = true) {
        if (!vocab || text.empty()) {
            LOG_ERROR_TokenCache("Vocab not available or text empty for tokenization");
            return {};
        }
        
        // Check cache first using enhanced bidirectional cache
        std::string cache_key = text + (add_special ? ":s" : ":n");
        if (auto cached_tokens = GetTokensFromText(cache_key)) {
            return *cached_tokens;
        }
        
        try {
            // Get required buffer size
            const int32_t n_tokens_required = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, add_special, true);
            if (n_tokens_required <= 0) [[unlikely]] {
                LOG_ERROR_TokenCache("Invalid token count required: " + std::to_string(n_tokens_required));
                return {};
            }
            
            // Tokenize
            std::vector<llama_token> llama_tokens(n_tokens_required);
            const int32_t n_tokens_actual = llama_tokenize(vocab, text.c_str(), text.size(),
                                                          llama_tokens.data(), llama_tokens.size(), add_special, true);
            
            if (n_tokens_actual < 0 || n_tokens_actual != n_tokens_required) [[unlikely]] {
                LOG_ERROR_TokenCache("Tokenization failed - expected: " + std::to_string(n_tokens_required) + 
                                   ", got: " + std::to_string(n_tokens_actual));
                return {};
            }
            
            // Convert llama_token to int32_t with better performance
            std::vector<int32_t> tokens;
            tokens.reserve(llama_tokens.size());
            std::transform(llama_tokens.begin(), llama_tokens.end(), std::back_inserter(tokens),
                          [](llama_token token) { return static_cast<int32_t>(token); });
            
            // Cache result using bidirectional storage
            StoreBidirectional(cache_key, text, tokens);
            
            LOG_DEBUG_TokenCache("Tokenized text (" + std::to_string(text.length()) + 
                               " chars → " + std::to_string(tokens.size()) + " tokens)");
            
            return tokens;
            
        } catch (const std::exception& e) {
            LOG_ERROR_TokenCache("Exception during tokenization: " + std::string(e.what()));
            return {};
        }
    }
    
    // Primary detokenization method - handles both caching and actual detokenization
    std::string DetokenizeTokens(const std::vector<int32_t>& tokens) {
        if (!vocab || tokens.empty()) {
            LOG_ERROR_TokenCache("Vocab not available or tokens empty for detokenization");
            return "";
        }
        
        // Check cache first using enhanced bidirectional cache
        if (auto cached_text = GetTextFromTokens(tokens)) {
            return *cached_text;
        }
        
        try {
            std::string result;
            result.reserve(tokens.size() * 4); // Rough estimate for better performance
            
            for (int32_t token_int : tokens) {
                llama_token token = static_cast<llama_token>(token_int);
                std::vector<char> buffer(32);
                
                int32_t result_length = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
                
                if (result_length < 0) [[unlikely]] {
                    // Buffer too small, resize and retry
                    buffer.resize(-result_length);
                    result_length = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
                }
                
                if (result_length > 0) [[likely]] {
                    result.append(buffer.data(), result_length);
                }
            }
            
            // Cache result using bidirectional storage
            if (!result.empty()) {
                StoreBidirectional(result, result, tokens);
            }
            
            LOG_DEBUG_TokenCache("Detokenized tokens (" + std::to_string(tokens.size()) + 
                               " tokens → " + std::to_string(result.length()) + " chars)");
            
            return result;
            
        } catch (const std::exception& e) {
            LOG_ERROR_TokenCache("Exception during detokenization: " + std::string(e.what()));
            return "";
        }
    }
};
