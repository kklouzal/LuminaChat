#pragma once

#include <string>
#include <cctype>

// Safe string trimming function that properly handles UTF-8 characters
inline std::string safe_trim(const std::string& str) {
    size_t start = 0;
    size_t end = str.size();

    // Trim from start - safely cast to unsigned char to avoid isspace issues with negative values
    while (start < end && isspace(static_cast<unsigned char>(str[start]))) {
        start += 1;
    }

    // Trim from end - safely cast to unsigned char to avoid isspace issues with negative values
    while (end > start && isspace(static_cast<unsigned char>(str[end - 1]))) {
        end -= 1;
    }

    return str.substr(start, end - start);
}