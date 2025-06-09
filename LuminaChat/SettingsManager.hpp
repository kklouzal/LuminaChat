// SettingsManager.hpp - header-only implementation for settings management
// Handles loading and saving configuration in INI format with sections
//
// File Specific Directives:
// Manage settings in a structured INI format with sections for each setting.
// Handles saving/loading of all application settings.
// .ini file is saved in the same directory as the executable.
//
// Project Settings:
// C++ Language Standard: ISO C++20 Standard (/std:c++20)
// C Language Standard: ISO C17 (2018) Standard (/std:c17)
// Character Set: Use Unicode Character Set
// Whole Program Optimization: Use Link Time Code Generation
// Optimization: Maximum Optimization (Favor Speed) (/O2)
// Enable Intrinsic Functions: Yes (/Oi)
// Favor Size or Speed: Favor fast code (/Ot)
// Whole Program Optimization: Yes (/GL)
// Enable String Pooling: Yes (/GF)
// Runtime Library: Multi-threaded DLL (/MD)
// Enable Run-Time Type Information (RTTI) YES (/GR)
// Link Time Code Generation: Use Link Time Code Generation (/LTCG)
//
// CRITICAL CODING DIRECTIVES:
// 1. Minimalism & Performance: Deliver lean, efficient solutions that avoid unnecessary bloat.
// 2. Consistent Coding Style: Maintain uniform style and structure for clear, maintainable code.
// 3. Clear Documentation: Provide concise comments explaining complex logic and key decisions.
// 4. Eliminate Redundancy: Remove unused, obsolete, and legacy code along with excess includes.
// 5. Optimize Function Structure: Adjust function boundaries to reduce overlap and clarify responsibilities.
// 6. Preserve Core Functionality: Streamline code while safeguarding essential features.
// 7. Cross-Platform Standards: Use fixed-width types and proper initialization to ensure portability.
// 8. Smart Caching: Cache frequently used variables to reduce repeated allocations.
// 9. Ensure Logical Consistency: Review code flow to maintain coherent, error-free execution.
// 10. Continuous Refinement: Regularly refactor and verify that updates preserve stable functionality.
#pragma once

#include <string>
#include <cstdint>
#include <fstream>
#include <algorithm>
#include <wx/stdpaths.h>
#include <wx/filename.h>

// Forward declare log callback function
void settings_manager_log_callback(const std::string& message);

class SettingsManager {
private:
    // Cache the settings file path to avoid repeated filesystem calls
    static std::string s_cached_settings_path;
    
    static const std::string& EscapeString(const std::string& input) {
        static thread_local std::string result; // Reuse buffer to reduce allocations
        result.clear();
        result.reserve(input.size() * 1.2f); // Pre-allocate with small buffer
        
        for (char c : input) {
            switch (c) {
                case '\n': result += "\\n"; break;
                case '\r': result += "\\r"; break;
                case '\t': result += "\\t"; break;
                case '\\': result += "\\\\"; break;
                case '=':  result += "\\="; break;
                default:   result += c; break;
            }
        }
        return result;
    }
    
    static const std::string& UnescapeString(const std::string& input) {
        static thread_local std::string result; // Reuse buffer to reduce allocations
        result.clear();
        result.reserve(input.size());
        
        for (size_t i = 0; i < input.size(); ++i) {
            if (input[i] == '\\' && i + 1 < input.size()) {
                switch (input[i + 1]) {
                    case 'n': result += '\n'; ++i; break;
                    case 'r': result += '\r'; ++i; break;
                    case 't': result += '\t'; ++i; break;
                    case '\\': result += '\\'; ++i; break;
                    case '=': result += '='; ++i; break;
                    default: result += input[i]; break;
                }
            } else {
                result += input[i];
            }
        }
        return result;
    }
    
    // Optimized string trimming with in-place modification
    static void TrimString(std::string& str) {
        constexpr const char* whitespace = " \t\r\n";
        str.erase(0, str.find_first_not_of(whitespace));
        str.erase(str.find_last_not_of(whitespace) + 1);
    }
    
    // Validate and clamp integer values
    static int32_t ValidateInt32(const std::string& value, int32_t default_val, int32_t min_val, int32_t max_val) noexcept {
        try {
            int32_t result = std::stoi(value);
            return std::clamp(result, min_val, max_val);
        } catch (...) {
            return default_val;
        }
    }

    // Helper function for thread-safe logging
    static void log_message(const std::string& message) {
        settings_manager_log_callback(message);
    }

public:
    static const std::string& GetSettingsFilePath() {
        if (s_cached_settings_path.empty()) {
            // Get the directory where the executable is located
            wxString exeDir = wxStandardPaths::Get().GetExecutablePath();
            wxFileName exePath(exeDir);
            wxString appDir = exePath.GetPath();
            
            // Create the settings file path in the same directory as the executable
            wxFileName configFile(appDir, "luminachat.ini");
            s_cached_settings_path = configFile.GetFullPath().ToStdString();
            
            log_message("Settings file path: " + s_cached_settings_path);
        }
        return s_cached_settings_path;
    }
    
    static void SaveSettings(const std::string& model_path, int32_t context_size, int32_t gpu_layers, 
                           int32_t predict_tokens, const std::string& chat_template, 
                           const std::string& identity_directive, const std::string& other_directives,
                           const std::string& discord_bot_token, const std::string& discord_channel_ids,
                           const std::string& discord_isolated_channel_ids, const std::string& discord_shared_history_channel_ids,
                           bool discord_allow_dms) {
        const std::string& filepath = GetSettingsFilePath();
        std::ofstream file(filepath, std::ios::out | std::ios::trunc);
        
        if (!file.is_open()) {
            log_message("Error: Failed to save settings to: " + filepath);
            return;
        }
        
        // Use a single write operation with pre-built string for better performance
        file << "[General]\n"
                "ModelPath=" << EscapeString(model_path) << '\n' <<
                "ContextSize=" << context_size << '\n' <<
                "GpuLayers=" << gpu_layers << '\n' <<
                "PredictTokens=" << predict_tokens << "\n\n"
                
                "[ChatTemplate]\n"
                "Template=" << EscapeString(chat_template) << "\n\n"
                
                "[SystemPrompt]\n"
                "IdentityDirective=" << EscapeString(identity_directive) << '\n' <<
                "OtherDirectives=" << EscapeString(other_directives) << "\n\n"
                
                "[Discord]\n"
                "BotToken=" << EscapeString(discord_bot_token) << '\n' <<
                "ChannelIds=" << EscapeString(discord_channel_ids) << '\n' <<
                "IsolatedChannelIds=" << EscapeString(discord_isolated_channel_ids) << '\n' <<
                "SharedHistoryChannelIds=" << EscapeString(discord_shared_history_channel_ids) << '\n' <<
                "AllowDMs=" << (discord_allow_dms ? '1' : '0') << '\n';
        
        if (file.good()) {
            log_message("Settings saved to: " + filepath);
        } else {
            log_message("Error: Failed to write settings to: " + filepath);
        }
    }
    
    static void LoadSettings(std::string& model_path, int32_t& context_size, int32_t& gpu_layers, 
                           int32_t& predict_tokens, std::string& chat_template, 
                           std::string& identity_directive, std::string& other_directives,
                           std::string& discord_bot_token, std::string& discord_channel_ids,
                           std::string& discord_isolated_channel_ids, std::string& discord_shared_history_channel_ids,
                           bool& discord_allow_dms) {
        // Set defaults first
        context_size = (context_size == 0) ? 2048 : context_size;
        predict_tokens = (predict_tokens == 0) ? 256 : predict_tokens;
        discord_allow_dms = true; // Default to true
        
        const std::string& filepath = GetSettingsFilePath();
        std::ifstream file(filepath);
        
        if (!file.is_open()) {
            log_message("Settings file not found: " + filepath + " (using defaults)");
            return;
        }
        
        log_message("Loading settings from: " + filepath);
        std::string line, current_section;
        int32_t loaded_count = 0;
        
        while (std::getline(file, line)) {
            TrimString(line);
            
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#' || line[0] == ';') continue;
            
            // Check for section headers
            if (line[0] == '[' && line.back() == ']') {
                current_section = line.substr(1, line.length() - 2);
                continue;
            }
            
            // Parse key=value pairs
            const size_t pos = line.find('=');
            if (pos == std::string::npos) continue;
            
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            TrimString(key);
            TrimString(value);
            
            // Handle settings based on section using string_view for efficiency
            if (current_section == "General") {
                if (key == "ModelPath") {
                    model_path = UnescapeString(value);
                    ++loaded_count;
                } else if (key == "ContextSize") {
                    context_size = ValidateInt32(value, 2048, 1, 131072);
                    ++loaded_count;
                } else if (key == "GpuLayers") {
                    gpu_layers = ValidateInt32(value, 0, 0, 999);
                    ++loaded_count;
                } else if (key == "PredictTokens") {
                    predict_tokens = ValidateInt32(value, 256, 1, 4096);
                    ++loaded_count;
                }
            } else if (current_section == "ChatTemplate") {
                if (key == "Template") {
                    chat_template = UnescapeString(value);
                    ++loaded_count;
                }
            } else if (current_section == "SystemPrompt") {
                if (key == "IdentityDirective") {
                    identity_directive = UnescapeString(value);
                    ++loaded_count;
                } else if (key == "OtherDirectives") {
                    other_directives = UnescapeString(value);
                    ++loaded_count;
                }
            } else if (current_section == "Discord") {
                if (key == "BotToken") {
                    discord_bot_token = UnescapeString(value);
                    ++loaded_count;
                } else if (key == "ChannelIds") {
                    discord_channel_ids = UnescapeString(value);
                    ++loaded_count;
                } else if (key == "IsolatedChannelIds") {
                    discord_isolated_channel_ids = UnescapeString(value);
                    ++loaded_count;
                } else if (key == "SharedHistoryChannelIds") {
                    discord_shared_history_channel_ids = UnescapeString(value);
                    ++loaded_count;
                } else if (key == "AllowDMs") {
                    discord_allow_dms = (value == "1" || value == "true" || value == "True" || value == "TRUE");
                    ++loaded_count;
                }
            }
        }
        
        log_message("Settings loading complete. Loaded " + std::to_string(loaded_count) + " settings.");
    }
};

// Static member definition
std::string SettingsManager::s_cached_settings_path;

// Log callback function declaration
extern void settings_manager_log_callback(const std::string& message);
