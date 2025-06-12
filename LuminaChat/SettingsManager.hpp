// SettingsManager.hpp - header-only implementation for settings management
// Handles loading and saving configuration in INI format with sections
//
// Project Settings:
// C++ Language Standard: ISO C++20 Standard (/std:c++20)
// C Language Standard: ISO C17 (2018) Standard (/std:c17)
// Optimization: Maximum Optimization (Favor Speed) (/O2)
// Favor Size or Speed: Favor fast code (/Ot)
// Runtime Library: Multi-threaded DLL (/MD)
// Enable Run-Time Type Information (RTTI) YES (/GR)
//
// CRITICAL CODING DIRECTIVES:
// 1.  Minimalism & Performance: Deliver lean, efficient solutions; do not create or preserve unused helpers or wrappers.
// 2.  Redundancy Elimination: Remove unused, obsolete, and legacy code—including unneeded interfaces and includes.
// 3.  Consistent Style: Adopt a uniform coding style and structure for clarity and maintainability.
// 4.  Documentation: Write concise comments that explain complex logic and key design decisions.
// 5.  Zero Magic & Strong Typing: Replace magic literals with named constants, enums, or constexpr; prefer scoped enums.
// 6.  Function Boundaries: Define clear responsibilities; reduce overlap and avoid unnecessary layers.
// 7.  Core Preservation: Streamline code while safeguarding essential features; favor direct access over extra abstractions.
// 8.  Const-Correctness & Immutability: Mark variables, parameters, and methods as const wherever possible.
// 9.  RAII & Resource Safety: Encapsulate resource acquisition/release in constructors/destructors or smart pointers.
// 10. Standard Library Preference: Favor STL algorithms and containers over custom loops and buffers.
// 11. Cross-Platform Portability: Use fixed-width types and proper initialization to guarantee identical behavior everywhere.
// 12. Thread Safety: Define and document thread-safety contracts; protect shared state with mutexes, atomics, or thread-safe containers.
// 13. Smart Caching: Cache frequently used values to minimize allocations and improve performance.
// 14. Logical Consistency: Verify code flow to ensure coherent, error-free execution paths.
// 15. Continuous Refinement: Regularly refactor and confirm that updates preserve stable functionality.

#pragma once

#include <string>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <wx/stdpaths.h>
#include <wx/filename.h>
#include "LogHandler.hpp"

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

public:
    static std::string GetSettingsFilePath() {
        // Get the directory where the executable is located
        wxString exeDir = wxStandardPaths::Get().GetExecutablePath();
        wxFileName exePath(exeDir);
        wxString appDir = exePath.GetPath();
        
        // Create the settings file path in the same directory as the executable
        wxFileName configFile(appDir, "luminachat.ini");
        std::string filepath = configFile.GetFullPath().ToStdString();
        
        SETTINGS_LOG("Settings file path: " + filepath);
        return filepath;
    }
    
    static void SaveSettings(const std::string& model_path, int32_t context_size, int32_t gpu_layers, 
                           int32_t predict_tokens, const std::string& chat_template, 
                           const std::string& identity_directive, const std::string& other_directives,
                           const std::string& discord_bot_token,
                           const std::string& discord_isolated_channel_ids, const std::string& discord_shared_history_channel_ids,
                           bool discord_allow_dms, bool discord_pull_history, int32_t discord_history_fill_percentage,
                           const std::string& summarizer_model_path, int32_t summarizer_context_size, int32_t summarizer_gpu_layers,
                           int32_t summarizer_predict_tokens, const std::string& summarizer_system_prompt, const std::string& summarizer_chat_template) {
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
            file << "IsolatedChannelIds=" << EscapeString(discord_isolated_channel_ids) << std::endl;
            file << "SharedHistoryChannelIds=" << EscapeString(discord_shared_history_channel_ids) << std::endl;
            file << "AllowDMs=" << (discord_allow_dms ? "1" : "0") << std::endl;
            file << "PullHistory=" << (discord_pull_history ? "1" : "0") << std::endl;
            file << "HistoryFillPercentage=" << discord_history_fill_percentage << std::endl;
            file << std::endl;
            
            // [Summarizer] section
            file << "[Summarizer]" << std::endl;
            file << "ModelPath=" << EscapeString(summarizer_model_path) << std::endl;
            file << "ContextSize=" << summarizer_context_size << std::endl;
            file << "GpuLayers=" << summarizer_gpu_layers << std::endl;
            file << "PredictTokens=" << summarizer_predict_tokens << std::endl;
            file << "SystemPrompt=" << EscapeString(summarizer_system_prompt) << std::endl;
            file << "ChatTemplate=" << EscapeString(summarizer_chat_template) << std::endl;
            
            file.close();
            
            SETTINGS_LOG("Settings saved to: " + filepath);
        } else {
            SETTINGS_LOG("Error: Failed to save settings to: " + filepath);
        }
    }
    
    static void LoadSettings(std::string& model_path, int32_t& context_size, int32_t& gpu_layers, 
                           int32_t& predict_tokens, std::string& chat_template, 
                           std::string& identity_directive, std::string& other_directives,
                           std::string& discord_bot_token,
                           std::string& discord_isolated_channel_ids, std::string& discord_shared_history_channel_ids,
                           bool& discord_allow_dms, bool& discord_pull_history, int32_t& discord_history_fill_percentage,
                           std::string& summarizer_model_path, int32_t& summarizer_context_size, int32_t& summarizer_gpu_layers,
                           int32_t& summarizer_predict_tokens, std::string& summarizer_system_prompt, std::string& summarizer_chat_template) {
        std::string filepath = GetSettingsFilePath();
        std::ifstream file(filepath);
        
        // Set defaults first
        if (model_path.empty()) model_path = "";
        if (context_size == 0) context_size = 2048;
        if (gpu_layers == 0) gpu_layers = 0;
        if (predict_tokens == 0) predict_tokens = 256;
        discord_allow_dms = true; // Default to true
        discord_pull_history = true; // Default to true
        discord_history_fill_percentage = 50; // Default to 50%
        
        // Summarizer defaults
        if (summarizer_context_size == 0) summarizer_context_size = 1024;
        if (summarizer_gpu_layers == 0) summarizer_gpu_layers = 0;
        if (summarizer_predict_tokens == 0) summarizer_predict_tokens = 128;
        if (summarizer_system_prompt.empty()) {
            summarizer_system_prompt = "You are a helpful AI assistant that provides concise summaries. "
                                     "Focus on key points and maintain clarity while keeping responses brief.";
        }
        
        if (file.is_open()) {
            SETTINGS_LOG("Loading settings from: " + filepath);
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
                    SETTINGS_LOG("Reading section: [" + current_section + "]");
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
                            SETTINGS_LOG("Loaded ModelPath: " + std::string(model_path.empty() ? "(empty)" : "configured"));
                        } else if (key == "ContextSize") {
                            context_size = ValidateInt32(value, 2048, 1, 131072);
                            loaded_count++;
                            SETTINGS_LOG("Loaded ContextSize: " + std::to_string(context_size));
                        } else if (key == "GpuLayers") {
                            gpu_layers = ValidateInt32(value, 0, 0, 999);
                            loaded_count++;
                            SETTINGS_LOG("Loaded GpuLayers: " + std::to_string(gpu_layers));
                        } else if (key == "PredictTokens") {
                            predict_tokens = ValidateInt32(value, 256, 1, 4096);
                            loaded_count++;
                            SETTINGS_LOG("Loaded PredictTokens: " + std::to_string(predict_tokens));
                        }
                    } else if (current_section == "ChatTemplate") {
                        if (key == "Template") {
                            chat_template = UnescapeString(value);
                            loaded_count++;
                            SETTINGS_LOG("Loaded ChatTemplate: " + std::string(chat_template.empty() ? "(empty)" : "configured"));
                        }
                    } else if (current_section == "SystemPrompt") {
                        if (key == "IdentityDirective") {
                            identity_directive = UnescapeString(value);
                            loaded_count++;
                            SETTINGS_LOG("Loaded IdentityDirective: " + std::string(identity_directive.empty() ? "(empty)" : "configured"));
                        } else if (key == "OtherDirectives") {
                            other_directives = UnescapeString(value);
                            loaded_count++;
                            SETTINGS_LOG("Loaded OtherDirectives: " + std::string(other_directives.empty() ? "(empty)" : "configured"));
                        }
                    } else if (current_section == "Discord") {
                        if (key == "BotToken") {
                            discord_bot_token = UnescapeString(value);
                            loaded_count++;
                            SETTINGS_LOG("Loaded BotToken: " + std::string(discord_bot_token.empty() ? "(empty)" : "configured"));
                        } else if (key == "IsolatedChannelIds") {
                            discord_isolated_channel_ids = UnescapeString(value);
                            loaded_count++;
                            SETTINGS_LOG("Loaded IsolatedChannelIds: " + discord_isolated_channel_ids);
                        } else if (key == "SharedHistoryChannelIds") {
                            discord_shared_history_channel_ids = UnescapeString(value);
                            loaded_count++;
                            SETTINGS_LOG("Loaded SharedHistoryChannelIds: " + discord_shared_history_channel_ids);
                        } else if (key == "AllowDMs") {
                            discord_allow_dms = (value == "1" || value == "true" || value == "True" || value == "TRUE");
                            loaded_count++;
                            SETTINGS_LOG("Loaded AllowDMs: " + std::string(discord_allow_dms ? "true" : "false"));
                        } else if (key == "PullHistory") {
                            discord_pull_history = (value == "1" || value == "true" || value == "True" || value == "TRUE");
                            loaded_count++;
                            SETTINGS_LOG("Loaded PullHistory: " + std::string(discord_pull_history ? "true" : "false"));
                        } else if (key == "HistoryFillPercentage") {
                            discord_history_fill_percentage = ValidateInt32(value, 50, 10, 80);
                            loaded_count++;
                            SETTINGS_LOG("Loaded HistoryFillPercentage: " + std::to_string(discord_history_fill_percentage) + "%");
                        }
                        // Legacy support: ignore old ChannelIds setting if present
                        else if (key == "ChannelIds") {
                            SETTINGS_LOG("Ignored legacy ChannelIds setting (no longer used)");
                        }
                    } else if (current_section == "Summarizer") {
                        if (key == "ModelPath") {
                            summarizer_model_path = UnescapeString(value);
                            loaded_count++;
                            SETTINGS_LOG("Loaded Summarizer ModelPath: " + std::string(summarizer_model_path.empty() ? "(empty)" : "configured"));
                        } else if (key == "ContextSize") {
                            summarizer_context_size = ValidateInt32(value, 1024, 1, 32768);
                            loaded_count++;
                            SETTINGS_LOG("Loaded Summarizer ContextSize: " + std::to_string(summarizer_context_size));
                        } else if (key == "GpuLayers") {
                            summarizer_gpu_layers = ValidateInt32(value, 0, 0, 999);
                            loaded_count++;
                            SETTINGS_LOG("Loaded Summarizer GpuLayers: " + std::to_string(summarizer_gpu_layers));
                        } else if (key == "PredictTokens") {
                            summarizer_predict_tokens = ValidateInt32(value, 128, 1, 2048);
                            loaded_count++;
                            SETTINGS_LOG("Loaded Summarizer PredictTokens: " + std::to_string(summarizer_predict_tokens));
                        } else if (key == "SystemPrompt") {
                            summarizer_system_prompt = UnescapeString(value);
                            loaded_count++;
                            SETTINGS_LOG("Loaded Summarizer SystemPrompt: " + std::string(summarizer_system_prompt.empty() ? "(empty)" : "configured"));
                        } else if (key == "ChatTemplate") {
                            summarizer_chat_template = UnescapeString(value);
                            loaded_count++;
                            SETTINGS_LOG("Loaded Summarizer ChatTemplate: " + std::string(summarizer_chat_template.empty() ? "(empty)" : "configured"));
                        }
                    } else {
                        SETTINGS_LOG("Unknown section/setting: [" + current_section + "] " + key + "=" + value);
                    }
                }
            }
            file.close();
            
            SETTINGS_LOG("Settings loading complete. Loaded " + std::to_string(loaded_count) + " settings.");
        } else {
            SETTINGS_LOG("Settings file not found: " + filepath + " (using defaults)");
        }
    }
};

//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODE DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//