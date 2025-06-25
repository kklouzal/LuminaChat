#pragma once

#include "Logger.hpp"
#include <unordered_map>
#include <vector>
#include <string>
#include <mutex>
#include <functional>
#include <chrono>
#include <atomic>

// TokenCache: High-performance token caching system
// - Text-to-token and token-to-text caching
// - Performance statistics tracking
// - Single instance per ModelInfo, shared across contexts
// - Thread-safe operations
// - Template caching for performance

struct CacheStats {
    std::atomic<size_t> text_to_token_hits{0};
    std::atomic<size_t> text_to_token_misses{0};
    std::atomic<size_t> token_to_text_hits{0};
    std::atomic<size_t> token_to_text_misses{0};
    std::atomic<size_t> template_cache_hits{0};
    std::atomic<size_t> template_cache_misses{0};
    
    // Default constructor
    CacheStats() = default;
    
    // Copy constructor
    CacheStats(const CacheStats& other) 
        : text_to_token_hits{other.text_to_token_hits.load()}
        , text_to_token_misses{other.text_to_token_misses.load()}
        , token_to_text_hits{other.token_to_text_hits.load()}
        , token_to_text_misses{other.token_to_text_misses.load()}
        , template_cache_hits{other.template_cache_hits.load()}
        , template_cache_misses{other.template_cache_misses.load()} {}
    
    // Copy assignment operator
    CacheStats& operator=(const CacheStats& other) {
        if (this != &other) {
            text_to_token_hits = other.text_to_token_hits.load();
            text_to_token_misses = other.text_to_token_misses.load();
            token_to_text_hits = other.token_to_text_hits.load();
            token_to_text_misses = other.token_to_text_misses.load();
            template_cache_hits = other.template_cache_hits.load();
            template_cache_misses = other.template_cache_misses.load();
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
    
    float GetTemplateHitRatio() const {
        size_t total = template_cache_hits + template_cache_misses;
        return total > 0 ? static_cast<float>(template_cache_hits) / total : 0.0f;
    }
    
    size_t GetTotalHits() const {
        return text_to_token_hits + token_to_text_hits + template_cache_hits;
    }
    
    size_t GetTotalRequests() const {
        return text_to_token_hits + text_to_token_misses + 
               token_to_text_hits + token_to_text_misses +
               template_cache_hits + template_cache_misses;
    }
};

struct TokenCacheEntry {
    std::vector<int32_t> tokens;
    std::chrono::steady_clock::time_point last_accessed;
    size_t access_count;
    
    TokenCacheEntry() : last_accessed(std::chrono::steady_clock::now()), access_count(1) {}
    TokenCacheEntry(std::vector<int32_t> tok) : tokens(std::move(tok)), 
        last_accessed(std::chrono::steady_clock::now()), access_count(1) {}
};

struct TextCacheEntry {
    std::string text;
    std::chrono::steady_clock::time_point last_accessed;
    size_t access_count;
    
    TextCacheEntry() : last_accessed(std::chrono::steady_clock::now()), access_count(1) {}
    TextCacheEntry(std::string txt) : text(std::move(txt)), 
        last_accessed(std::chrono::steady_clock::now()), access_count(1) {}
};

class TokenCache {
private:
    // Cache storage
    std::unordered_map<std::string, TokenCacheEntry> text_to_token_cache;
    std::unordered_map<std::string, TextCacheEntry> token_to_text_cache; // Key is token sequence hash
    std::unordered_map<std::string, TokenCacheEntry> template_cache; // Rendered templates
    
    // Thread safety
    mutable std::mutex cache_mutex;
    
    // Performance tracking
    CacheStats stats;
    
    // Cache management
    static constexpr size_t MAX_CACHE_SIZE = 10000;
    static constexpr size_t CLEANUP_THRESHOLD = 8000;
    
    // Callback for cache invalidation notifications (optional)
    std::function<void(const std::string&)> invalidation_callback;
    
    // Helper methods
    std::string GenerateTokenHash(const std::vector<int32_t>& tokens) const {
        // Simple hash generation for token sequences
        std::string hash;
        hash.reserve(tokens.size() * 8); // Rough estimate
        for (int32_t token : tokens) {
            hash += std::to_string(token) + ",";
        }
        return hash;
    }
    
    void CleanupOldEntries() {
        // Remove least recently used entries when cache gets too large
        auto now = std::chrono::steady_clock::now();
        auto cutoff = now - std::chrono::hours(1); // Remove entries older than 1 hour
        
        // Cleanup text-to-token cache
        for (auto it = text_to_token_cache.begin(); it != text_to_token_cache.end();) {
            if (it->second.last_accessed < cutoff) {
                LOG_DEBUG_TokenCache("Removing old text-to-token cache entry");
                it = text_to_token_cache.erase(it);
            } else {
                ++it;
            }
        }
        
        // Cleanup token-to-text cache  
        for (auto it = token_to_text_cache.begin(); it != token_to_text_cache.end();) {
            if (it->second.last_accessed < cutoff) {
                LOG_DEBUG_TokenCache("Removing old token-to-text cache entry");
                it = token_to_text_cache.erase(it);
            } else {
                ++it;
            }
        }
        
        // Cleanup template cache
        for (auto it = template_cache.begin(); it != template_cache.end();) {
            if (it->second.last_accessed < cutoff) {
                LOG_DEBUG_TokenCache("Removing old template cache entry");
                it = template_cache.erase(it);
            } else {
                ++it;
            }
        }
    }
    
public:
    TokenCache() {
        LOG_TokenCache("TokenCache initialized");
    }
    
    ~TokenCache() {
        std::lock_guard<std::mutex> lock(cache_mutex);
        LOG_TokenCache("TokenCache destroyed - Final stats: " + 
            std::to_string(stats.GetTotalHits()) + " hits / " + 
            std::to_string(stats.GetTotalRequests()) + " requests");
    }
    
    // Optional callback registration for cache invalidation notifications
    // ContextInfo (higher) can register with TokenCache (lower) for cache events
    void RegisterInvalidationCallback(std::function<void(const std::string&)> callback) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        invalidation_callback = std::move(callback);
        LOG_TokenCache("Invalidation callback registered");
    }
    
    // Text-to-token caching
    bool GetTokens(const std::string& text, std::vector<int32_t>& tokens) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        
        auto it = text_to_token_cache.find(text);
        if (it != text_to_token_cache.end()) {
            tokens = it->second.tokens;
            it->second.last_accessed = std::chrono::steady_clock::now();
            it->second.access_count++;
            stats.text_to_token_hits++;
            LOG_DEBUG_TokenCache("Cache hit for text (" + std::to_string(text.length()) + " chars)");
            return true;
        }
        
        stats.text_to_token_misses++;
        LOG_DEBUG_TokenCache("Cache miss for text (" + std::to_string(text.length()) + " chars)");
        return false;
    }
    
    void StoreTokens(const std::string& text, const std::vector<int32_t>& tokens) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        
        // Check if cleanup needed
        if (text_to_token_cache.size() > CLEANUP_THRESHOLD) {
            CleanupOldEntries();
        }
        
        text_to_token_cache[text] = TokenCacheEntry(tokens);
        LOG_DEBUG_TokenCache("Stored tokens for text (" + std::to_string(text.length()) + 
            " chars → " + std::to_string(tokens.size()) + " tokens)");
    }
    
    // Token-to-text caching
    bool GetText(const std::vector<int32_t>& tokens, std::string& text) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        
        std::string hash = GenerateTokenHash(tokens);
        auto it = token_to_text_cache.find(hash);
        if (it != token_to_text_cache.end()) {
            text = it->second.text;
            it->second.last_accessed = std::chrono::steady_clock::now();
            it->second.access_count++;
            stats.token_to_text_hits++;
            LOG_DEBUG_TokenCache("Cache hit for tokens (" + std::to_string(tokens.size()) + " tokens)");
            return true;
        }
        
        stats.token_to_text_misses++;
        LOG_DEBUG_TokenCache("Cache miss for tokens (" + std::to_string(tokens.size()) + " tokens)");
        return false;
    }
    
    void StoreText(const std::vector<int32_t>& tokens, const std::string& text) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        
        // Check if cleanup needed
        if (token_to_text_cache.size() > CLEANUP_THRESHOLD) {
            CleanupOldEntries();
        }
        
        std::string hash = GenerateTokenHash(tokens);
        token_to_text_cache[hash] = TextCacheEntry(text);
        LOG_DEBUG_TokenCache("Stored text for tokens (" + std::to_string(tokens.size()) + 
            " tokens → " + std::to_string(text.length()) + " chars)");
    }
    
    // Template caching for ChatTemplateManager
    bool GetTemplateTokens(const std::string& template_hash, std::vector<int32_t>& tokens) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        
        auto it = template_cache.find(template_hash);
        if (it != template_cache.end()) {
            tokens = it->second.tokens;
            it->second.last_accessed = std::chrono::steady_clock::now();
            it->second.access_count++;
            stats.template_cache_hits++;
            LOG_DEBUG_TokenCache("Template cache hit (" + template_hash.substr(0, 16) + "...)");
            return true;
        }
        
        stats.template_cache_misses++;
        LOG_DEBUG_TokenCache("Template cache miss (" + template_hash.substr(0, 16) + "...)");
        return false;
    }
    
    void StoreTemplateTokens(const std::string& template_hash, const std::vector<int32_t>& tokens) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        
        // Check if cleanup needed
        if (template_cache.size() > CLEANUP_THRESHOLD) {
            CleanupOldEntries();
        }
        
        template_cache[template_hash] = TokenCacheEntry(tokens);
        LOG_DEBUG_TokenCache("Stored template tokens (" + template_hash.substr(0, 16) + 
            "... → " + std::to_string(tokens.size()) + " tokens)");
    }
    
    // Cache invalidation
    void InvalidateText(const std::string& text) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        
        auto it = text_to_token_cache.find(text);
        if (it != text_to_token_cache.end()) {
            text_to_token_cache.erase(it);
            LOG_DEBUG_TokenCache("Invalidated text cache entry");
            
            if (invalidation_callback) {
                invalidation_callback(text);
            }
        }
    }
    
    void InvalidateTemplate(const std::string& template_hash) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        
        auto it = template_cache.find(template_hash);
        if (it != template_cache.end()) {
            template_cache.erase(it);
            LOG_DEBUG_TokenCache("Invalidated template cache entry");
            
            if (invalidation_callback) {
                invalidation_callback("template:" + template_hash);
            }
        }
    }
    
    void ClearAll() {
        std::lock_guard<std::mutex> lock(cache_mutex);
        
        size_t total_entries = text_to_token_cache.size() + token_to_text_cache.size() + template_cache.size();
        
        text_to_token_cache.clear();
        token_to_text_cache.clear();
        template_cache.clear();
        
        LOG_TokenCache("Cleared all cache entries (" + std::to_string(total_entries) + " total)");
        
        if (invalidation_callback) {
            invalidation_callback("all");
        }
    }
    
    // Performance statistics
    CacheStats GetStats() const {
        std::lock_guard<std::mutex> lock(cache_mutex);
        return stats;
    }
    
    size_t GetCacheSize() const {
        std::lock_guard<std::mutex> lock(cache_mutex);
        return text_to_token_cache.size() + token_to_text_cache.size() + template_cache.size();
    }
    
    // Memory management
    size_t GetMemoryUsage() const {
        std::lock_guard<std::mutex> lock(cache_mutex);
        
        size_t memory = 0;
        
        // Text-to-token cache
        for (const auto& entry : text_to_token_cache) {
            memory += entry.first.capacity(); // Text string
            memory += entry.second.tokens.capacity() * sizeof(int32_t); // Token vector
        }
        
        // Token-to-text cache
        for (const auto& entry : token_to_text_cache) {
            memory += entry.first.capacity(); // Hash string
            memory += entry.second.text.capacity(); // Text string
        }
        
        // Template cache
        for (const auto& entry : template_cache) {
            memory += entry.first.capacity(); // Template hash
            memory += entry.second.tokens.capacity() * sizeof(int32_t); // Token vector
        }
        
        return memory;
    }
    
    void LogStatistics() const {
        auto stats_copy = GetStats();
        size_t cache_size = GetCacheSize();
        size_t memory_usage = GetMemoryUsage();
        
        LOG_TokenCache("=== Cache Statistics ===");
        LOG_TokenCache("Total entries: " + std::to_string(cache_size));
        LOG_TokenCache("Memory usage: " + std::to_string(memory_usage / 1024) + " KB");
        LOG_TokenCache("Text→Token: " + std::to_string(stats_copy.text_to_token_hits) + 
            " hits, " + std::to_string(stats_copy.text_to_token_misses) + 
            " misses (" + std::to_string(static_cast<int>(stats_copy.GetTextToTokenHitRatio() * 100)) + "% hit rate)");
        LOG_TokenCache("Token→Text: " + std::to_string(stats_copy.token_to_text_hits) + 
            " hits, " + std::to_string(stats_copy.token_to_text_misses) + 
            " misses (" + std::to_string(static_cast<int>(stats_copy.GetTokenToTextHitRatio() * 100)) + "% hit rate)");
        LOG_TokenCache("Templates: " + std::to_string(stats_copy.template_cache_hits) + 
            " hits, " + std::to_string(stats_copy.template_cache_misses) + 
            " misses (" + std::to_string(static_cast<int>(stats_copy.GetTemplateHitRatio() * 100)) + "% hit rate)");
        LOG_TokenCache("Overall: " + std::to_string(stats_copy.GetTotalHits()) + 
            " hits / " + std::to_string(stats_copy.GetTotalRequests()) + " requests");
    }
    
    // Wrapper methods for consistent API (used by tests)
    std::optional<std::vector<int32_t>> GetCachedTokens(const std::string& text) {
        std::vector<int32_t> tokens;
        if (GetTokens(text, tokens)) {
            return tokens;
        }
        return std::nullopt;
    }
    
    void CacheTokens(const std::string& text, const std::vector<int32_t>& tokens) {
        StoreTokens(text, tokens);
    }
    
    std::optional<std::string> GetCachedText(const std::vector<int32_t>& tokens) {
        std::string text;
        if (GetText(tokens, text)) {
            return text;
        }
        return std::nullopt;
    }
    
    void CacheText(const std::vector<int32_t>& tokens, const std::string& text) {
        StoreText(tokens, text);
    }
    
    std::optional<std::vector<int32_t>> GetCachedTemplate(const std::string& template_key) {
        std::vector<int32_t> tokens;
        if (GetTemplateTokens(template_key, tokens)) {
            return tokens;
        }
        return std::nullopt;
    }
    
    void CacheTemplate(const std::string& template_key, const std::vector<int32_t>& tokens) {
        StoreTemplateTokens(template_key, tokens);
    }
    
    void InvalidateCache() {
        ClearAll();
    }
};
