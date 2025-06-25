#pragma once

#include "Logger.hpp"
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <mutex>
#include <cstdint>
// wxWidgets includes for file path management
#include <wx/stdpaths.h>
#include <wx/filename.h>

/**
 * Configuration persistence and access using existing .ini format.
 * 
 * Features:
 * - Compatible with existing LuminaChat .ini files
 * - Model paths, context sizes, GPU layer configurations
 * - Discord bot tokens and channel settings
 * - UI preferences and logging levels
 * - Plugin enable/disable flags
 * - Template configuration support
 */

class SettingsManager {
private:
    struct Section {
        std::unordered_map<std::string, std::string> keys;
    };

    std::unordered_map<std::string, Section> sections;
    std::string ini_file_path;
    mutable std::mutex settings_mutex;
    bool settings_dirty = false;

    // String escaping for INI format
    static std::string EscapeString(const std::string& input) noexcept {
        std::string result;
        result.reserve(input.length() + input.length() / 4); // Reserve extra space for escapes
        for (const char c : input) {
            switch (c) {
                case '\n': result += "\\n"; break;
                case '\r': result += "\\r"; break;
                case '\t': result += "\\t"; break;
                case '\\': result += "\\\\"; break;
                case '=': result += "\\="; break;
                default: result += c; break;
            }
        }
        return result;
    }

    static std::string UnescapeString(const std::string& input) noexcept {
        std::string result;
        result.reserve(input.length()); // Reserve space for efficiency
        for (size_t i = 0; i < input.size(); ++i) {
            if (input[i] == '\\' && i + 1 < input.size()) {
                switch (input[i + 1]) {
                    case 'n': result += '\n'; i++; break;
                    case 'r': result += '\r'; i++; break;
                    case 't': result += '\t'; i++; break;
                    case '\\': result += '\\'; i++; break;
                    case '=': result += '='; i++; break;
                    default: result += input[i]; break;
                }
            } else {
                result += input[i];
            }
        }
        return result;
    }

    std::string Trim(const std::string& str) {
        const char* whitespace = " \t\r\n";
        size_t start = str.find_first_not_of(whitespace);
        if (start == std::string::npos) return "";
        size_t end = str.find_last_not_of(whitespace);
        return str.substr(start, end - start + 1);
    }

    std::string ToLower(const std::string& str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
    }

public:
    // Initialize the settings manager
    bool Initialize() {
        std::string default_path = GetSettingsFilePath();
        if (!LoadSettings(default_path)) {
            // If loading failed, create with defaults and save
            LOG_SettingsManager("Creating new settings file with defaults");
            SetDefaults();
            return SaveSettings();
        }
        return true;
    }

    // Get settings file path using wxWidgets
    static std::string GetSettingsFilePath() noexcept {
        // Get the directory where the executable is located
        wxString exeDir = wxStandardPaths::Get().GetExecutablePath();
        wxFileName exePath(exeDir);
        wxString appDir = exePath.GetPath();
        
        // Create the settings file path in the same directory as the executable
        wxFileName configFile(appDir, "luminachat.ini");
        std::string filepath = configFile.GetFullPath().ToStdString();
        
        LOG_SettingsManager("Settings file path: " + filepath);
        return filepath;
    }
    // Load settings from .ini file
    bool LoadSettings(const std::string& ini_path = "") {
        std::lock_guard<std::mutex> lock(settings_mutex);
        
        // Use provided path or get default path
        std::string file_path = ini_path.empty() ? GetSettingsFilePath() : ini_path;
        
        LOG_SettingsManager("Attempting to load settings from: " + file_path);
        
        std::ifstream file(file_path);
        if (!file.is_open()) {
            LOG_ERROR_SettingsManager("Failed to open settings file: " + file_path);
            // Initialize with defaults if file doesn't exist
            SetDefaults();
            ini_file_path = file_path;
            return false;
        }

        LOG_SettingsManager("File opened successfully, clearing existing sections");
        sections.clear();
        ini_file_path = file_path;
        
        std::string line;
        std::string current_section = "";
        int line_count = 0;
        
        while (std::getline(file, line)) {
            line_count++;
            line = Trim(line);
            
            LOG_SettingsManager("Processing line " + std::to_string(line_count) + ": '" + line + "'");
            
            // Skip empty lines and comments
            if (line.empty() || line[0] == ';' || line[0] == '#') {
                continue;
            }
            
            // Section headers
            if (line.front() == '[' && line.back() == ']') {
                current_section = line.substr(1, line.length() - 2);
                sections[current_section] = Section{};
                LOG_SettingsManager("Found section: [" + current_section + "]");
                continue;
            }
            
            // Key-value pairs
            size_t equals_pos = line.find('=');
            if (equals_pos != std::string::npos && !current_section.empty()) {
                std::string key = Trim(line.substr(0, equals_pos));
                std::string value = Trim(line.substr(equals_pos + 1));
                
                // Remove quotes if present
                if (value.length() >= 2 && value.front() == '"' && value.back() == '"') {
                    value = value.substr(1, value.length() - 2);
                }
                
                // Unescape the value
                value = UnescapeString(value);
                
                sections[current_section].keys[key] = value;
                LOG_SettingsManager("Set [" + current_section + "]." + key + " = '" + value + "'");
            }
        }

        file.close();
        settings_dirty = false;
        LOG_SettingsManager("Successfully loaded " + std::to_string(line_count) + " lines from: " + file_path);
        return true;
    }

    // Save settings to .ini file
    bool SaveSettings() {
        std::lock_guard<std::mutex> lock(settings_mutex);
        
        if (ini_file_path.empty()) {
            ini_file_path = GetSettingsFilePath();
        }

        std::ofstream file(ini_file_path);
        if (!file.is_open()) {
            LOG_ERROR("SettingsManager", "Failed to open settings file for writing: " + ini_file_path);
            return false;
        }

        // Write sections in a consistent order
        for (const auto& [section_name, section] : sections) {
            file << "[" << section_name << "]\n";
            
            for (const auto& [key, value] : section.keys) {
                // Escape the value and write it
                std::string escaped_value = EscapeString(value);
                
                // Quote values that contain spaces or special characters after escaping
                bool needs_quotes = escaped_value.find_first_of(" \t\"") != std::string::npos;
                if (needs_quotes) {
                    file << key << "=\"" << escaped_value << "\"\n";
                } else {
                    file << key << "=" << escaped_value << "\n";
                }
            }
            file << "\n";
        }

        file.close();
        settings_dirty = false;
        LOG_SettingsManager("Saved settings to: " + ini_file_path);
        return true;
    }

    // Get string value
    std::string GetString(const std::string& section, const std::string& key, const std::string& default_value = "") {
        std::lock_guard<std::mutex> lock(settings_mutex);
        
        auto section_it = sections.find(section);
        if (section_it == sections.end()) {
            return default_value;
        }

        auto key_it = section_it->second.keys.find(key);
        if (key_it == section_it->second.keys.end()) {
            return default_value;
        }

        return key_it->second;
    }

    // Validation limits (from old implementation)
    static constexpr int32_t MIN_CONTEXT_SIZE = 1;
    static constexpr int32_t MAX_CONTEXT_SIZE = 131072;
    static constexpr int32_t MIN_GPU_LAYERS = 0;
    static constexpr int32_t MAX_GPU_LAYERS = 999;
    static constexpr int32_t MIN_PREDICT_TOKENS = 1;
    static constexpr int32_t MAX_PREDICT_TOKENS = 4096;

    // Validate and clamp integer values
    static int32_t ValidateInt32(const std::string& value, const int32_t default_val, const int32_t min_val, const int32_t max_val) noexcept {
        try {
            const int32_t result = std::stoi(value);
            return std::clamp(result, min_val, max_val);
        } catch (...) {
            return default_val;
        }
    }

    // Get integer value with validation
    int GetInt(const std::string& section, const std::string& key, int default_value = 0) {
        std::string str_value = GetString(section, key);
        if (str_value.empty()) {
            return default_value;
        }

        try {
            int result = std::stoi(str_value);
            // Apply validation for known settings
            if (section == "Models") {
                if (key.find("context_size") != std::string::npos) {
                    return std::clamp(result, MIN_CONTEXT_SIZE, MAX_CONTEXT_SIZE);
                } else if (key.find("gpu_layers") != std::string::npos) {
                    return std::clamp(result, MIN_GPU_LAYERS, MAX_GPU_LAYERS);
                }
            }
            return result;
        } catch (const std::exception&) {
            LOG_WARNING("SettingsManager", "Failed to parse int value for [" + section + "]." + key + ": " + str_value);
            return default_value;
        }
    }

    // Get boolean value
    bool GetBool(const std::string& section, const std::string& key, bool default_value = false) {
        std::string str_value = ToLower(GetString(section, key));
        if (str_value.empty()) {
            return default_value;
        }

        return str_value == "true" || str_value == "1" || str_value == "yes" || str_value == "on";
    }

    // Get float value
    float GetFloat(const std::string& section, const std::string& key, float default_value = 0.0f) {
        std::lock_guard<std::mutex> lock(settings_mutex);
        
        auto section_it = sections.find(section);
        if (section_it == sections.end()) {
            LOG_SettingsManager("GetFloat [" + section + "]." + key + " - section not found, using default: " + std::to_string(default_value));
            return default_value;
        }
        
        auto key_it = section_it->second.keys.find(key);
        if (key_it == section_it->second.keys.end()) {
            LOG_SettingsManager("GetFloat [" + section + "]." + key + " - key not found, using default: " + std::to_string(default_value));
            return default_value;
        }
        
        try {
            float value = std::stof(key_it->second);
            LOG_SettingsManager("GetFloat [" + section + "]." + key + " = " + std::to_string(value));
            return value;
        } catch (const std::exception& e) {
            LOG_ERROR_SettingsManager("GetFloat [" + section + "]." + key + " - conversion failed: " + std::string(e.what()) + ", using default: " + std::to_string(default_value));
            return default_value;
        }
    }
    
    // Set string value
    void SetString(const std::string& section, const std::string& key, const std::string& value) {
        std::lock_guard<std::mutex> lock(settings_mutex);
        sections[section].keys[key] = value;
        settings_dirty = true;
    }

    // Set integer value
    void SetInt(const std::string& section, const std::string& key, int value) {
        SetString(section, key, std::to_string(value));
    }

    // Set boolean value
    void SetBool(const std::string& section, const std::string& key, bool value) {
        std::lock_guard<std::mutex> lock(settings_mutex);
        sections[section].keys[key] = value ? "true" : "false";
        settings_dirty = true;
        LOG_SettingsManager("SetBool [" + section + "]." + key + " = " + (value ? "true" : "false"));
    }
    
    // Float support methods
    void SetFloat(const std::string& section, const std::string& key, float value) {
        std::lock_guard<std::mutex> lock(settings_mutex);
        sections[section].keys[key] = std::to_string(value);
        settings_dirty = true;
        LOG_SettingsManager("SetFloat [" + section + "]." + key + " = " + std::to_string(value));
    }
    
    // Template management
    // Note: Main chat templates are managed by ChatTemplateManager with hardcoded defaults.
    // SettingsManager only stores user customizations and specialized templates (e.g. summary).
    std::string GetChatTemplate(const std::string& template_name) {
        return GetString("Templates", template_name);
    }

    void SetChatTemplate(const std::string& template_name, const std::string& template_content) {
        SetString("Templates", template_name, template_content);
    }

    // Check if settings need saving
    bool IsDirty() const {
        std::lock_guard<std::mutex> lock(settings_mutex);
        return settings_dirty;
    }

    // Set default values for new installations
    void SetDefaults() {
        std::lock_guard<std::mutex> lock(settings_mutex);
        
        // Model configuration
        SetString_Unlocked("Models", "main_model_path", "");
        SetString_Unlocked("Models", "summary_model_path", "");
        SetInt_Unlocked("Models", "main_context_size", 8192);
        SetInt_Unlocked("Models", "summary_context_size", 4096);
        SetInt_Unlocked("Models", "main_gpu_layers", 999);
        SetInt_Unlocked("Models", "summary_gpu_layers", 999);

        // Discord configuration
        SetString_Unlocked("Discord", "bot_token", "");
        SetString_Unlocked("Discord", "default_channel", "");
        SetBool_Unlocked("Discord", "auto_respond", true);
        SetBool_Unlocked("Discord", "history_backfill", true);
        SetInt_Unlocked("Discord", "backfill_limit", 100);

        // UI preferences
        SetString_Unlocked("UI", "theme", "dark");
        SetInt_Unlocked("UI", "window_width", 800);
        SetInt_Unlocked("UI", "window_height", 600);
        SetBool_Unlocked("UI", "auto_scroll", true);

        // Logging configuration
        SetString_Unlocked("Logging", "level", "INFO");
        SetBool_Unlocked("Logging", "file_output", false);
        SetString_Unlocked("Logging", "file_path", "luminachat.log");

        // Plugin configuration
        SetBool_Unlocked("Plugins", "sanitization", true);
        SetBool_Unlocked("Plugins", "summarization", true);
        SetBool_Unlocked("Plugins", "discord_channel_management", true);
        SetBool_Unlocked("Plugins", "history_backfill", true);

        // Context management
        SetFloat_Unlocked("Context", "prune_threshold", 0.8f);
        SetFloat_Unlocked("Context", "prune_target", 0.4f);
        SetBool_Unlocked("Context", "auto_summarize", true);

        // Default summary template (main chat template is now handled by ChatTemplateManager)
        SetChatTemplate_Unlocked("summary", GetDefaultSummaryTemplate());

        settings_dirty = true;
        LOG_SettingsManager("Initialized default settings");
    }

private:
    // Unlocked versions for use within locked contexts (like SetDefaults)
    void SetString_Unlocked(const std::string& section, const std::string& key, const std::string& value) {
        sections[section].keys[key] = value;
        settings_dirty = true;
    }

    void SetInt_Unlocked(const std::string& section, const std::string& key, int value) {
        sections[section].keys[key] = std::to_string(value);
        settings_dirty = true;
    }

    void SetBool_Unlocked(const std::string& section, const std::string& key, bool value) {
        sections[section].keys[key] = value ? "true" : "false";
        settings_dirty = true;
    }
    
    void SetFloat_Unlocked(const std::string& section, const std::string& key, float value) {
        sections[section].keys[key] = std::to_string(value);
        settings_dirty = true;
    }
    
    void SetChatTemplate_Unlocked(const std::string& template_name, const std::string& template_content) {
        sections["Templates"].keys[template_name] = template_content;
        settings_dirty = true;
    }

    // Note: Default chat template is now handled by ChatTemplateManager::GetDefaultTemplate()
    // This method is kept only for the summary template
    std::string GetDefaultSummaryTemplate() {
        return R"({{- bos_token }}

<|start_header_id|>system_message<|end_header_id|>
You are a helpful summarization assistant. Create a concise but comprehensive summary of the provided conversation, preserving key context, topics discussed, and important details.
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Please summarize the following conversation:

{{ content_to_summarize }}
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>)";
    }

    // Auto-save if settings are dirty
    bool AutoSave() {
        if (IsDirty()) {
            return SaveSettings();
        }
        return true;
    }

    // Backward compatibility methods for easier migration from old SettingsManager
    void SaveAllSettings() {
        SaveSettings();
    }

    // Validate channel IDs (useful for Discord settings)
    std::string ValidateChannelIds(const std::string& input) {
        std::string result;
        std::stringstream ss(input);
        std::string channel_id;
        
        while (std::getline(ss, channel_id, ',')) {
            channel_id = Trim(channel_id);
            if (!channel_id.empty() && std::all_of(channel_id.begin(), channel_id.end(), ::isdigit)) {
                if (!result.empty()) result += ",";
                result += channel_id;
            }
        }
        
        return result;
    }
};
