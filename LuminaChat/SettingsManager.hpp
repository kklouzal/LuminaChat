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
#include <iostream>
#include <algorithm>
#include <wx/stdpaths.h>
#include <wx/filename.h>

// Forward declare log callback function
void settings_manager_log_callback(const std::string& message);

class SettingsManager {
private:
    static std::string EscapeString(const std::string& input) {
        std::string result;
        for (char c : input) {
            if (c == '\n') {
                result += "\\n";
            } else if (c == '\r') {
                result += "\\r";
            } else if (c == '\t') {
                result += "\\t";
            } else if (c == '\\') {
                result += "\\\\";
            } else if (c == '=') {
                result += "\\=";
            } else {
                result += c;
            }
        }
        return result;
    }
    
    static std::string UnescapeString(const std::string& input) {
        std::string result;
        for (size_t i = 0; i < input.size(); ++i) {
            if (input[i] == '\\' && i + 1 < input.size()) {
                if (input[i + 1] == 'n') {
                    result += '\n';
                    i++;
                } else if (input[i + 1] == 'r') {
                    result += '\r';
                    i++;
                } else if (input[i + 1] == 't') {
                    result += '\t';
                    i++;
                } else if (input[i + 1] == '\\') {
                    result += '\\';
                    i++;
                } else if (input[i + 1] == '=') {
                    result += '=';
                    i++;
                } else {
                    result += input[i];
                }
            } else {
                result += input[i];
            }
        }
        return result;
    }
    
    // Validate and clamp integer values
    static int32_t ValidateInt32(const std::string& value, int32_t default_val, int32_t min_val, int32_t max_val) {
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
    static std::string GetSettingsFilePath() {
        // Get the directory where the executable is located
        wxString exeDir = wxStandardPaths::Get().GetExecutablePath();
        wxFileName exePath(exeDir);
        wxString appDir = exePath.GetPath();
        
        // Create the settings file path in the same directory as the executable
        wxFileName configFile(appDir, "luminachat.ini");
        std::string filepath = configFile.GetFullPath().ToStdString();
        
        log_message("Settings file path: " + filepath);
        return filepath;
    }
    
    static void SaveSettings(const std::string& model_path, int32_t context_size, int32_t gpu_layers, 
                           int32_t predict_tokens, const std::string& chat_template, 
                           const std::string& identity_directive, const std::string& other_directives,
                           const std::string& discord_bot_token, const std::string& discord_channel_ids,
                           const std::string& discord_isolated_channel_ids, const std::string& discord_shared_history_channel_ids,
                           bool discord_allow_dms) {
        std::string filepath = GetSettingsFilePath();
        std::ofstream file(filepath);
        
        if (file.is_open()) {
            // [General] section
            file << "[General]" << std::endl;
            file << "ModelPath=" << EscapeString(model_path) << std::endl;
            file << "ContextSize=" << context_size << std::endl;
            file << "GpuLayers=" << gpu_layers << std::endl;
            file << "PredictTokens=" << predict_tokens << std::endl;
            file << std::endl;
            
            // [ChatTemplate] section
            file << "[ChatTemplate]" << std::endl;
            file << "Template=" << EscapeString(chat_template) << std::endl;
            file << std::endl;
            
            // [SystemPrompt] section
            file << "[SystemPrompt]" << std::endl;
            file << "IdentityDirective=" << EscapeString(identity_directive) << std::endl;
            file << "OtherDirectives=" << EscapeString(other_directives) << std::endl;
            file << std::endl;
            
            // [Discord] section
            file << "[Discord]" << std::endl;
            file << "BotToken=" << EscapeString(discord_bot_token) << std::endl;
            file << "ChannelIds=" << EscapeString(discord_channel_ids) << std::endl;
            file << "IsolatedChannelIds=" << EscapeString(discord_isolated_channel_ids) << std::endl;
            file << "SharedHistoryChannelIds=" << EscapeString(discord_shared_history_channel_ids) << std::endl;
            file << "AllowDMs=" << (discord_allow_dms ? "1" : "0") << std::endl;
            
            file.close();
            
            log_message("Settings saved to: " + filepath);
        } else {
            log_message("Error: Failed to save settings to: " + filepath);
        }
    }
    
    static void LoadSettings(std::string& model_path, int32_t& context_size, int32_t& gpu_layers, 
                           int32_t& predict_tokens, std::string& chat_template, 
                           std::string& identity_directive, std::string& other_directives,
                           std::string& discord_bot_token, std::string& discord_channel_ids,
                           std::string& discord_isolated_channel_ids, std::string& discord_shared_history_channel_ids,
                           bool& discord_allow_dms) {
        std::string filepath = GetSettingsFilePath();
        std::ifstream file(filepath);
        
        // Set defaults first
        if (model_path.empty()) model_path = "";
        if (context_size == 0) context_size = 2048;
        if (gpu_layers == 0) gpu_layers = 0;
        if (predict_tokens == 0) predict_tokens = 256;
        discord_allow_dms = true; // Default to true
        
        if (file.is_open()) {
            log_message("Loading settings from: " + filepath);
            std::string line;
            std::string current_section;
            int loaded_count = 0;
            
            while (std::getline(file, line)) {
                // Trim whitespace
                line.erase(0, line.find_first_not_of(" \t\r\n"));
                line.erase(line.find_last_not_of(" \t\r\n") + 1);
                
                // Skip empty lines and comments
                if (line.empty() || line[0] == '#' || line[0] == ';') {
                    continue;
                }
                
                // Check for section headers
                if (line[0] == '[' && line.back() == ']') {
                    current_section = line.substr(1, line.length() - 2);
                    log_message("Reading section: [" + current_section + "]");
                    continue;
                }
                
                // Parse key=value pairs
                size_t pos = line.find('=');
                if (pos != std::string::npos) {
                    std::string key = line.substr(0, pos);
                    std::string value = line.substr(pos + 1);
                    
                    key.erase(0, key.find_first_not_of(" \t"));
                    key.erase(key.find_last_not_of(" \t") + 1);
                    value.erase(0, value.find_first_not_of(" \t"));
                    value.erase(value.find_last_not_of(" \t") + 1);
                    
                    // Handle settings based on section
                    if (current_section == "General") {
                        if (key == "ModelPath") {
                            model_path = UnescapeString(value);
                            loaded_count++;
                            log_message("Loaded ModelPath: " + std::string(model_path.empty() ? "(empty)" : "configured"));
                        } else if (key == "ContextSize") {
                            context_size = ValidateInt32(value, 2048, 1, 131072);
                            loaded_count++;
                            log_message("Loaded ContextSize: " + std::to_string(context_size));
                        } else if (key == "GpuLayers") {
                            gpu_layers = ValidateInt32(value, 0, 0, 999);
                            loaded_count++;
                            log_message("Loaded GpuLayers: " + std::to_string(gpu_layers));
                        } else if (key == "PredictTokens") {
                            predict_tokens = ValidateInt32(value, 256, 1, 4096);
                            loaded_count++;
                            log_message("Loaded PredictTokens: " + std::to_string(predict_tokens));
                        }
                    } else if (current_section == "ChatTemplate") {
                        if (key == "Template") {
                            chat_template = UnescapeString(value);
                            loaded_count++;
                            log_message("Loaded ChatTemplate: " + std::string(chat_template.empty() ? "(empty)" : "configured"));
                        }
                    } else if (current_section == "SystemPrompt") {
                        if (key == "IdentityDirective") {
                            identity_directive = UnescapeString(value);
                            loaded_count++;
                            log_message("Loaded IdentityDirective: " + std::string(identity_directive.empty() ? "(empty)" : "configured"));
                        } else if (key == "OtherDirectives") {
                            other_directives = UnescapeString(value);
                            loaded_count++;
                            log_message("Loaded OtherDirectives: " + std::string(other_directives.empty() ? "(empty)" : "configured"));
                        }
                    } else if (current_section == "Discord") {
                        if (key == "BotToken") {
                            discord_bot_token = UnescapeString(value);
                            loaded_count++;
                            log_message("Loaded BotToken: " + std::string(discord_bot_token.empty() ? "(empty)" : "configured"));
                        } else if (key == "ChannelIds") {
                            discord_channel_ids = UnescapeString(value);
                            loaded_count++;
                            log_message("Loaded ChannelIds: " + discord_channel_ids);
                        } else if (key == "IsolatedChannelIds") {
                            discord_isolated_channel_ids = UnescapeString(value);
                            loaded_count++;
                            log_message("Loaded IsolatedChannelIds: " + discord_isolated_channel_ids);
                        } else if (key == "SharedHistoryChannelIds") {
                            discord_shared_history_channel_ids = UnescapeString(value);
                            loaded_count++;
                            log_message("Loaded SharedHistoryChannelIds: " + discord_shared_history_channel_ids);
                        } else if (key == "AllowDMs") {
                            discord_allow_dms = (value == "1" || value == "true" || value == "True" || value == "TRUE");
                            loaded_count++;
                            log_message("Loaded AllowDMs: " + std::string(discord_allow_dms ? "true" : "false"));
                        }
                    } else {
                        log_message("Unknown section/setting: [" + current_section + "] " + key + "=" + value);
                    }
                }
            }
            file.close();
            
            log_message("Settings loading complete. Loaded " + std::to_string(loaded_count) + " settings.");
        } else {
            log_message("Settings file not found: " + filepath + " (using defaults)");
        }
    }
};

// Log callback function declaration
extern void settings_manager_log_callback(const std::string& message);
