#pragma once

#include <string>
#include <cctype>

// Safe string trimming function that properly handles UTF-8 characters
[[nodiscard]] inline std::string safe_trim(const std::string& str) noexcept {
    if (str.empty()) [[unlikely]] {
        return str;
    }
    
    size_t start = 0;
    const size_t end_pos = str.size();
    size_t end = end_pos;

    // Trim from start - safely cast to unsigned char to avoid isspace issues with negative values
    while (start < end && std::isspace(static_cast<unsigned char>(str[start]))) [[unlikely]] {
        ++start;
    }

    // Trim from end - safely cast to unsigned char to avoid isspace issues with negative values  
    while (end > start && std::isspace(static_cast<unsigned char>(str[end - 1]))) [[unlikely]] {
        --end;
    }

    // Check if no trimming needed (common case)
    if (start == 0 && end == end_pos) [[likely]] {
        return str;
    }

    return str.substr(start, end - start);
}