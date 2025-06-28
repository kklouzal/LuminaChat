#pragma once

#include <string>
#include <cctype>

namespace LuminaChat {

/**
 * Common utility functions for the LuminaChat application
 */
namespace Utilities {

    /**
     * Trim whitespace and trailing linebreaks from text
     * This ensures clean text without trailing formatting artifacts
     * @param text The text to trim
     * @return Trimmed text with leading and trailing whitespace removed
     */
    inline std::string TrimString(const std::string& text) {
        if (text.empty()) {
            return text;
        }
        
        // Find the last non-whitespace character
        size_t end = text.length();
        while (end > 0 && (std::isspace(text[end - 1]) || text[end - 1] == '\n' || text[end - 1] == '\r')) {
            end--;
        }
        
        // Find the first non-whitespace character (in case there's leading whitespace)
        size_t start = 0;
        while (start < end && std::isspace(text[start])) {
            start++;
        }
        
        return text.substr(start, end - start);
    }

} // namespace Utilities

} // namespace LuminaChat
