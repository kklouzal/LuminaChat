#pragma once
#include <string>
#include <sstream>
#include <regex>
#include <algorithm>

class TextSanitizer {
public:
    // Sanitize text by removing invalid Unicode characters and Discord-specific formatting
    [[nodiscard]] static std::string sanitize_text(const std::string& text) {
        if (text.empty()) [[unlikely]] {
            return text;
        }

        std::string sanitized = text;
        
        // First, clean Discord-specific formatting
        sanitized = clean_discord_formatting(sanitized);
        
        // Then remove invalid Unicode characters
        sanitized = remove_invalid_unicode(sanitized);
        
        // Normalize whitespace
        sanitized = normalize_whitespace(sanitized);
        
        return sanitized;
    }

private:    // Remove Discord-specific formatting that might cause tokenization issues
    [[nodiscard]] static std::string clean_discord_formatting(const std::string& text) noexcept {
        if (text.empty()) [[unlikely]] {
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
            
            // Remove user mentions <@!123456789> or <@123456789>
            cleaned = std::regex_replace(cleaned, user_mention_regex, "[User]");
            
            // Remove channel mentions <#123456789>
            cleaned = std::regex_replace(cleaned, channel_mention_regex, "[Channel]");
            
            // Remove role mentions <@&123456789>
            cleaned = std::regex_replace(cleaned, role_mention_regex, "[Role]");
            
            // Remove custom emojis <:emojiname:123456789>
            cleaned = std::regex_replace(cleaned, custom_emoji_regex, "[Emoji]");
            
            // Remove animated emojis <a:emojiname:123456789>
            cleaned = std::regex_replace(cleaned, animated_emoji_regex, "[AnimatedEmoji]");
            
            // Remove timestamp formatting <t:1234567890:F>
            cleaned = std::regex_replace(cleaned, timestamp_regex, "[Timestamp]");
            
            return cleaned;
        } catch (const std::exception&) {
            // If regex fails, return original text rather than empty string
            return text;
        }
    }    // Remove invalid Unicode characters that might cause tokenization issues
    [[nodiscard]] static std::string remove_invalid_unicode(const std::string& text) noexcept {
        if (text.empty()) [[unlikely]] {
            return text;
        }
        
        std::string result;
        result.reserve(text.length());
        
        for (size_t i = 0; i < text.length(); ) {
            const unsigned char c = static_cast<unsigned char>(text[i]);
            
            // Handle ASCII characters (0-127) - most common case
            if (c < 128) [[likely]] {
                // Skip control characters except common whitespace
                if (c < 32 && c != '\n' && c != '\r' && c != '\t') [[unlikely]] {
                    ++i;
                    continue;
                }
                result += text[i];
                ++i;
            }
            // Handle UTF-8 sequences
            else [[unlikely]] {
                const int utf8_len = get_utf8_sequence_length(c);
                if (utf8_len == 0 || i + utf8_len > text.length()) [[unlikely]] {
                    // Invalid UTF-8 sequence, skip this byte
                    ++i;
                    continue;
                }
                
                // Validate the complete UTF-8 sequence with bounds checking
                bool valid_sequence = true;
                for (int j = 1; j < utf8_len && (i + j) < text.length(); ++j) {
                    const unsigned char next_byte = static_cast<unsigned char>(text[i + j]);
                    if ((next_byte & 0xC0) != 0x80) [[unlikely]] {
                        valid_sequence = false;
                        break;
                    }
                }
                
                if (valid_sequence && (i + utf8_len <= text.length())) [[likely]] {
                    // Check for problematic Unicode codepoints
                    const uint32_t codepoint = decode_utf8_codepoint(text.substr(i, utf8_len));
                    if (is_valid_codepoint(codepoint)) [[likely]] {
                        result += text.substr(i, utf8_len);
                    }
                    i += utf8_len;
                } else [[unlikely]] {
                    // Invalid sequence, skip this byte
                    ++i;
                }
            }
        }
        
        return result;
    }

    // Get the expected length of a UTF-8 sequence based on the first byte
    static int get_utf8_sequence_length(unsigned char first_byte) {
        if ((first_byte & 0x80) == 0) return 1;      // 0xxxxxxx
        if ((first_byte & 0xE0) == 0xC0) return 2;   // 110xxxxx
        if ((first_byte & 0xF0) == 0xE0) return 3;   // 1110xxxx
        if ((first_byte & 0xF8) == 0xF0) return 4;   // 11110xxx
        return 0; // Invalid UTF-8 start byte
    }

    // Decode a UTF-8 sequence to a Unicode codepoint
    static uint32_t decode_utf8_codepoint(const std::string& utf8_sequence) {
        if (utf8_sequence.empty()) return 0;
        
        unsigned char first = static_cast<unsigned char>(utf8_sequence[0]);
        
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
    }    // Check if a Unicode codepoint is valid and safe for tokenization
    static bool is_valid_codepoint(uint32_t codepoint) {
        // Skip null character
        if (codepoint == 0) return false;
        
        // Only skip the most problematic ranges, be less aggressive
        // Skip surrogates (should not appear in UTF-8)
        if (codepoint >= 0xD800 && codepoint <= 0xDFFF) {
            return false;
        }
        
        // Skip non-characters (U+FFFE and U+FFFF in any plane)
        if ((codepoint & 0xFFFE) == 0xFFFE) {
            return false;
        }
        
        // Skip specific problematic non-character range
        if (codepoint >= 0xFDD0 && codepoint <= 0xFDEF) {
            return false;
        }
        
        // Allow most Unicode including emojis, symbols, and other characters
        // Modern LLMs can handle these without issues
        return true;
    }    // Normalize whitespace to prevent tokenization issues while preserving structure
    static std::string normalize_whitespace(const std::string& text) {
        if (text.empty()) return text;
        
        try {
            std::string normalized = text;
            
            // Replace multiple consecutive spaces/tabs with single space (but preserve newlines)
            normalized = std::regex_replace(normalized, std::regex(R"([ \t]+)"), " ");
            
            // Limit consecutive newlines to maximum of 2 (preserve paragraph structure)
            normalized = std::regex_replace(normalized, std::regex(R"(\n{3,})"), "\n\n");
            
            // Remove trailing spaces at end of lines
            normalized = std::regex_replace(normalized, std::regex(R"([ \t]+(\n|$))"), "$1");
            
            // Trim leading and trailing whitespace from entire string
            normalized = std::regex_replace(normalized, std::regex(R"(^\s+|\s+$)"), "");
            
            return normalized;
        } catch (const std::exception&) {
            // If regex fails, do basic whitespace cleanup
            std::string result = text;
            // Just trim leading/trailing whitespace as fallback
            size_t start = result.find_first_not_of(" \t\n\r");
            if (start == std::string::npos) return "";
            size_t end = result.find_last_not_of(" \t\n\r");
            return result.substr(start, end - start + 1);
        }
    }
};
