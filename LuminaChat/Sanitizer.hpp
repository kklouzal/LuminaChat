#pragma once

#include "Logger.hpp"
#include "ErrorHandling.hpp"
#include "SettingsManager.hpp"
#include <string>
#include <vector>
#include <unordered_set>
#include <mutex>
#include <functional>
#include <algorithm>
#include <cctype>
#include <regex>
#include <sstream>

/**
 * Content sanitization and filtering system integrating existing TextSanitizer and Blacklist functionality.
 * 
 * Features:
 * - Pattern-based content filtering
 * - Discord formatting cleanup
 * - Unicode sanitization
 * - Blacklist management with retroactive cleanup
 * - Plugin integration via callback system
 * - Settings-driven configuration
 */

class Sanitizer {
public:
    // Callback type for retroactive cleanup operations
    using RetroactiveCleanupCallback = std::function<void(const std::string&)>;
    
    // Callback type for advanced filtering notifications (optional plugin integration)
    using FilterCallback = std::function<void(const std::string&, bool)>;

private:
    // Thread-safe container for blacklisted patterns
    std::unordered_set<std::string> blacklisted_patterns;
    mutable std::mutex blacklist_mutex;
    
    // Callback for performing retroactive cleanup when new patterns are added
    RetroactiveCleanupCallback cleanup_callback;
    
    // Optional callback for advanced filtering notifications
    FilterCallback filter_callback;
    
    // Settings integration
    SettingsManager* settings_manager = nullptr;
    
    // Configuration flags
    bool sanitization_enabled = true;
    bool blacklist_enabled = true;
    bool discord_formatting_cleanup = true;
    bool unicode_cleanup = true;
    bool whitespace_normalization = true;

public:
    explicit Sanitizer(SettingsManager* settings = nullptr) 
        : settings_manager(settings) {
        LoadSettings();
        LOG_Sanitizer("Sanitizer initialized");
    }

    ~Sanitizer() = default;

    // Copy and move constructors (thread-safe)
    Sanitizer(const Sanitizer& other) {
        std::lock_guard<std::mutex> lock(other.blacklist_mutex);
        blacklisted_patterns = other.blacklisted_patterns;
        cleanup_callback = other.cleanup_callback;
        filter_callback = other.filter_callback;
        settings_manager = other.settings_manager;
        LoadSettings();
    }

    Sanitizer& operator=(const Sanitizer& other) {
        if (this != &other) {
            std::lock(blacklist_mutex, other.blacklist_mutex);
            std::lock_guard<std::mutex> lock1(blacklist_mutex, std::adopt_lock);
            std::lock_guard<std::mutex> lock2(other.blacklist_mutex, std::adopt_lock);
            
            blacklisted_patterns = other.blacklisted_patterns;
            cleanup_callback = other.cleanup_callback;
            filter_callback = other.filter_callback;
            settings_manager = other.settings_manager;
            LoadSettings();
        }
        return *this;
    }

    // Main sanitization interface
    bool SanitizeInput(std::string& input) {
        if (!sanitization_enabled) {
            return true;
        }

        if (input.empty()) {
            return true;
        }

        std::string original_input = input;
        
        // Step 1: Text sanitization (Discord formatting, Unicode, whitespace)
        input = SanitizeText(input);
        
        // Step 2: Blacklist filtering
        bool allowed = IsContentAllowed(input);
        
        // Step 3: Optional filter callback notification
        if (filter_callback) {
            SafeExecute([&]() {
                filter_callback(original_input, allowed);
            }, "execute filter callback for input", "Sanitizer");
        }
        
        if (!allowed) {
            LOG_Sanitizer("Input blocked by blacklist filter");
            input.clear(); // Clear blocked content
            return false;
        }
        
        return true;
    }

    // Filter assistant responses (separate from input sanitization)
    bool FilterAssistantResponse(std::string& response) {
        if (!blacklist_enabled || response.empty()) {
            return true;
        }

        bool allowed = IsContentAllowed(response);
        
        if (filter_callback) {
            SafeExecute([&]() {
                filter_callback(response, allowed);
            }, "execute filter callback for assistant response", "Sanitizer");
        }
        
        if (!allowed) {
            LOG_Sanitizer("Assistant response blocked by blacklist filter");
            response.clear();
            return false;
        }
        
        return true;
    }

    // Text sanitization (existing TextSanitizer functionality)
    std::string SanitizeText(const std::string& text) {
        if (text.empty()) {
            return text;
        }

        std::string sanitized = text;
        
        // Discord-specific formatting cleanup
        if (discord_formatting_cleanup) {
            sanitized = CleanDiscordFormatting(sanitized);
        }
        
        // Unicode character cleanup
        if (unicode_cleanup) {
            sanitized = RemoveInvalidUnicode(sanitized);
        }
        
        // Whitespace normalization
        if (whitespace_normalization) {
            sanitized = NormalizeWhitespace(sanitized);
        }
        
        return sanitized;
    }

    // Blacklist management
    bool IsContentAllowed(const std::string& content) {
        if (!blacklist_enabled || content.empty()) {
            return true;
        }
        
        std::lock_guard<std::mutex> lock(blacklist_mutex);
        
        // Fast exact match check first
        if (blacklisted_patterns.find(content) != blacklisted_patterns.end()) {
            return false;
        }
        
        // Pattern matching
        return !std::any_of(blacklisted_patterns.begin(), blacklisted_patterns.end(),
            [&content](const std::string& pattern) {
                return MatchesPattern(content, pattern);
            });
    }

    void LoadBlacklist(const std::string& blacklist_path) {
        // Implementation for loading blacklist from file
        // This would read from a file and populate blacklisted_patterns
        LOG_Sanitizer("Loading blacklist from: " + blacklist_path);
        // TODO: Implement file loading logic
    }

    // Blacklist pattern management
    bool AddPattern(const std::string& pattern) {
        if (pattern.empty()) {
            return false;
        }
        
        bool pattern_was_added = false;
        RetroactiveCleanupCallback callback_to_call = nullptr;
        
        {
            std::lock_guard<std::mutex> lock(blacklist_mutex);
            auto result = blacklisted_patterns.insert(pattern);
            pattern_was_added = result.second;
            
            if (pattern_was_added) {
                LOG_Sanitizer("Added pattern to blacklist: " + 
                    (pattern.length() > 50 ? pattern.substr(0, 50) + "..." : pattern));
                callback_to_call = cleanup_callback;
            }
        }
        
        // Perform retroactive cleanup if callback is available
        if (pattern_was_added && callback_to_call) {
            LOG_Sanitizer("Performing retroactive cleanup for newly blacklisted pattern");
            SafeExecute([&]() {
                callback_to_call(pattern);
            }, "perform retroactive cleanup", "Sanitizer");
        }
        
        return pattern_was_added;
    }

    bool RemovePattern(const std::string& pattern) {
        if (pattern.empty()) {
            return false;
        }
        
        std::lock_guard<std::mutex> lock(blacklist_mutex);
        auto it = blacklisted_patterns.find(pattern);
        if (it != blacklisted_patterns.end()) {
            blacklisted_patterns.erase(it);
            LOG_Sanitizer("Removed pattern from blacklist: " + 
                (pattern.length() > 50 ? pattern.substr(0, 50) + "..." : pattern));
            return true;
        }
        return false;
    }

    std::vector<std::string> GetAllPatterns() const {
        std::lock_guard<std::mutex> lock(blacklist_mutex);
        return std::vector<std::string>(blacklisted_patterns.begin(), blacklisted_patterns.end());
    }

    size_t GetPatternCount() const {
        std::lock_guard<std::mutex> lock(blacklist_mutex);
        return blacklisted_patterns.size();
    }

    // Callback registration (for circular dependency management)
    void RegisterCleanupCallback(RetroactiveCleanupCallback callback) {
        std::lock_guard<std::mutex> lock(blacklist_mutex);
        cleanup_callback = std::move(callback);
    }

    void RegisterFilterCallback(FilterCallback callback) {
        filter_callback = std::move(callback);
    }

    // Configuration
    void SetSanitizationEnabled(bool enabled) { sanitization_enabled = enabled; }
    void SetBlacklistEnabled(bool enabled) { blacklist_enabled = enabled; }
    void SetDiscordCleanupEnabled(bool enabled) { discord_formatting_cleanup = enabled; }
    void SetUnicodeCleanupEnabled(bool enabled) { unicode_cleanup = enabled; }
    void SetWhitespaceNormalizationEnabled(bool enabled) { whitespace_normalization = enabled; }

    bool IsSanitizationEnabled() const { return sanitization_enabled; }
    bool IsBlacklistEnabled() const { return blacklist_enabled; }

private:
    void LoadSettings() {
        if (!settings_manager) {
            return;
        }
        
        sanitization_enabled = settings_manager->GetBool("Plugins", "sanitization", true);
        blacklist_enabled = settings_manager->GetBool("Sanitizer", "blacklist_enabled", true);
        discord_formatting_cleanup = settings_manager->GetBool("Sanitizer", "discord_cleanup", true);
        unicode_cleanup = settings_manager->GetBool("Sanitizer", "unicode_cleanup", true);
        whitespace_normalization = settings_manager->GetBool("Sanitizer", "whitespace_normalization", true);
        
        LOG_Sanitizer("Loaded settings - Sanitization: " + std::string(sanitization_enabled ? "enabled" : "disabled"));
    }

    // Text sanitization implementation (from existing TextSanitizer)
    std::string CleanDiscordFormatting(const std::string& text) {
        if (text.empty()) {
            return text;
        }
        
        try {
            // Pre-compile static regex patterns for better performance
            static const std::regex user_mention_regex{R"(<@!?\d+>)"};
            static const std::regex channel_mention_regex{R"(<#\d+>)"};
            static const std::regex role_mention_regex{R"(<@&\d+>)"};
            static const std::regex custom_emoji_regex{R"(<:\w+:\d+>)"};
            static const std::regex animated_emoji_regex{R"(<a:\w+:\d+>)"};
            static const std::regex timestamp_regex{R"(<t:\d+:[tTdDfFR]>)"};
            
            std::string cleaned = text;
            
            // Remove Discord-specific formatting
            cleaned = std::regex_replace(cleaned, user_mention_regex, "[User]");
            cleaned = std::regex_replace(cleaned, channel_mention_regex, "[Channel]");
            cleaned = std::regex_replace(cleaned, role_mention_regex, "[Role]");
            cleaned = std::regex_replace(cleaned, custom_emoji_regex, "[Emoji]");
            cleaned = std::regex_replace(cleaned, animated_emoji_regex, "[AnimatedEmoji]");
            cleaned = std::regex_replace(cleaned, timestamp_regex, "[Timestamp]");
            
            return cleaned;
        } catch (const std::exception& e) {
            HandleError("Discord formatting cleanup failed: " + std::string(e.what()), "Sanitizer");
            return text;
        }
    }

    std::string RemoveInvalidUnicode(const std::string& text) {
        if (text.empty()) {
            return text;
        }
        
        std::string result;
        result.reserve(text.length());
        
        size_t invalid_sequences = 0;
        size_t invalid_codepoints = 0;
        
        for (size_t i = 0; i < text.length(); ) {
            const unsigned char c = static_cast<unsigned char>(text[i]);
            
            // Handle ASCII characters (0-127) - most common case
            if (c < 128) {
                // Skip control characters except common whitespace
                if (c < 32 && c != '\n' && c != '\r' && c != '\t') {
                    ++i;
                    continue;
                }
                result += text[i];
                ++i;
            }
            // Handle UTF-8 sequences
            else {
                const int utf8_len = GetUtf8SequenceLength(c);
                if (utf8_len == 0 || i + utf8_len > text.length()) {
                    // Invalid UTF-8 sequence, skip this byte
                    invalid_sequences++;
                    ++i;
                    continue;
                }
                
                // Validate the complete UTF-8 sequence
                bool valid_sequence = true;
                for (int j = 1; j < utf8_len && (i + j) < text.length(); ++j) {
                    const unsigned char next_byte = static_cast<unsigned char>(text[i + j]);
                    if ((next_byte & 0xC0) != 0x80) {
                        valid_sequence = false;
                        break;
                    }
                }
                
                if (valid_sequence && (i + utf8_len <= text.length())) {
                    // Check for problematic Unicode codepoints
                    const uint32_t codepoint = DecodeUtf8Codepoint(text.substr(i, utf8_len));
                    if (IsValidCodepoint(codepoint)) {
                        result += text.substr(i, utf8_len);
                    } else {
                        invalid_codepoints++;
                    }
                    i += utf8_len;
                } else {
                    // Invalid sequence, skip this byte
                    invalid_sequences++;
                    ++i;
                }
            }
        }
        
        // Debug logging for significant Unicode cleanup
        if (invalid_sequences > 0 || invalid_codepoints > 0) {
            LOG_DEBUG_Sanitizer("Unicode cleanup removed " + std::to_string(invalid_sequences) + 
                " invalid sequences and " + std::to_string(invalid_codepoints) + " invalid codepoints");
        }
        
        return result;
    }

    std::string NormalizeWhitespace(const std::string& text) {
        if (text.empty()) {
            return text;
        }
        
        try {
            std::string normalized = text;
            
            // Replace multiple consecutive spaces/tabs with single space
            normalized = std::regex_replace(normalized, std::regex(R"([ \t]+)"), " ");
            
            // Limit consecutive newlines to maximum of 2
            normalized = std::regex_replace(normalized, std::regex(R"(\n{3,})"), "\n\n");
            
            // Remove trailing spaces at end of lines
            normalized = std::regex_replace(normalized, std::regex(R"([ \t]+(\n|$))"), "$1");
            
            // Trim leading and trailing whitespace
            normalized = std::regex_replace(normalized, std::regex(R"(^\s+|\s+$)"), "");
            
            return normalized;
        } catch (const std::exception& e) {
            HandleError("Whitespace normalization failed: " + std::string(e.what()), "Sanitizer");
            // Fallback to basic trim
            const size_t start = text.find_first_not_of(" \t\n\r");
            if (start == std::string::npos) return "";
            const size_t end = text.find_last_not_of(" \t\n\r");
            return text.substr(start, end - start + 1);
        }
    }

    // Pattern matching (case-insensitive)
    static bool MatchesPattern(const std::string& response, const std::string& pattern) {
        if (response.empty() || pattern.empty()) {
            return false;
        }
        
        // Fast exact match check first
        if (response == pattern) {
            return true;
        }
        
        // Case-insensitive substring search
        const auto it = std::search(response.begin(), response.end(), pattern.begin(), pattern.end(),
            [](const char a, const char b) { 
                return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b)); 
            });
        
        return it != response.end();
    }

    // UTF-8 helper functions
    static int GetUtf8SequenceLength(const unsigned char first_byte) {
        if ((first_byte & 0x80) == 0) return 1;      // 0xxxxxxx
        if ((first_byte & 0xE0) == 0xC0) return 2;   // 110xxxxx
        if ((first_byte & 0xF0) == 0xE0) return 3;   // 1110xxxx
        if ((first_byte & 0xF8) == 0xF0) return 4;   // 11110xxx
        return 0; // Invalid UTF-8 start byte
    }

    static uint32_t DecodeUtf8Codepoint(const std::string& utf8_sequence) {
        if (utf8_sequence.empty()) return 0;
        
        const unsigned char first = static_cast<unsigned char>(utf8_sequence[0]);
        
        if (utf8_sequence.length() == 1) {
            return first; // ASCII
        }
        
        uint32_t codepoint = 0;
        
        if (utf8_sequence.length() == 2) {
            codepoint = (first & 0x1F) << 6;
            codepoint |= (static_cast<unsigned char>(utf8_sequence[1]) & 0x3F);
        }
        else if (utf8_sequence.length() == 3) {
            codepoint = (first & 0x0F) << 12;
            codepoint |= (static_cast<unsigned char>(utf8_sequence[1]) & 0x3F) << 6;
            codepoint |= (static_cast<unsigned char>(utf8_sequence[2]) & 0x3F);
        }
        else if (utf8_sequence.length() == 4) {
            codepoint = (first & 0x07) << 18;
            codepoint |= (static_cast<unsigned char>(utf8_sequence[1]) & 0x3F) << 12;
            codepoint |= (static_cast<unsigned char>(utf8_sequence[2]) & 0x3F) << 6;
            codepoint |= (static_cast<unsigned char>(utf8_sequence[3]) & 0x3F);
        }
        
        return codepoint;
    }

    static bool IsValidCodepoint(const uint32_t codepoint) {
        // Skip null character
        if (codepoint == 0) return false;
        
        // Skip surrogates (should not appear in UTF-8)
        if (codepoint >= 0xD800 && codepoint <= 0xDFFF) {
            return false;
        }
        
        // Skip non-characters
        if ((codepoint & 0xFFFE) == 0xFFFE) {
            return false;
        }
        
        // Skip specific problematic non-character range
        if (codepoint >= 0xFDD0 && codepoint <= 0xFDEF) {
            return false;
        }
        
        return true;
    }
};
