#pragma once

#include <string>
#include <cctype>
#include <vector>
#include <regex>
#include "Logger.hpp"

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

    /**
     * Estimate token count for text without full tokenization
     * This is a rough estimation used for context size planning
     * @param text The text to estimate token count for
     * @return Estimated number of tokens (rough approximation)
     */
    inline size_t EstimateTokenCount(const std::string& text) {
        // Rough estimation: ~4 characters per token on average
        // This is a simplification - actual tokenization varies by model
        return text.length() / 4;
    }

    /**
     * Extract clean response text from raw model output
     * Removes template tokens, normalizes whitespace, and cleans formatting
     * @param raw_response The raw response from the model
     * @return Cleaned response text ready for display
     */
    inline std::string ExtractCleanResponse(const std::string& raw_response) {
        if (raw_response.empty()) {
            return raw_response;
        }
        
        std::string cleaned = raw_response;
        
        // Remove common template tokens that may appear in generated output
        const std::vector<std::string> template_tokens = {
            "<|start_header_id|>assistant<|end_header_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|im_start|>assistant",
            "<|im_start|>",
            "<|im_end|>",
            "### Assistant:",
            "Assistant:",
            "<s>",
            "</s>",
            "[INST]",
            "[/INST]"
        };
        
        // Remove template tokens
        for (const auto& token : template_tokens) {
            size_t pos = 0;
            while ((pos = cleaned.find(token, pos)) != std::string::npos) {
                cleaned.erase(pos, token.length());
                // Don't increment pos to catch consecutive occurrences
            }
        }
        
        // Clean up leading/trailing whitespace and normalize spacing
        cleaned = TrimString(cleaned);
        
        // Remove excessive newlines (but preserve intentional paragraph breaks)
        std::regex multiple_newlines(R"(\n\s*\n\s*\n+)");
        cleaned = std::regex_replace(cleaned, multiple_newlines, "\n\n");
        
        LOG_DEBUG("Utilities", "Response extraction: " + std::to_string(raw_response.length()) + 
                 " chars -> " + std::to_string(cleaned.length()) + " chars (clean)");
        
        return cleaned;
    }

} // namespace Utilities

} // namespace LuminaChat
