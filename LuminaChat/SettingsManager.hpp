#pragma once

#include "Logger.hpp"
#include "ErrorHandling.hpp"
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
        bool loaded_successfully = LoadSettings(default_path);
        
        if (!loaded_successfully) {
            // If loading failed, create with defaults and save
            LOG_SettingsManager("Creating new settings file with defaults");
            SetDefaults();
            return SaveSettings();
        }
        
        // Settings loaded successfully, now sanitize and validate
        LOG_SettingsManager("Settings loaded, performing sanitization and validation...");
        SanitizeSettings();      // Remove deprecated/unused settings
        ValidateAndFixSettings(); // Fix invalid values
        
        // Save if sanitization/validation made changes
        if (settings_dirty) {
            LOG_SettingsManager("Saving sanitized settings...");
            SaveSettings();
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
            LOG_ERROR_SettingsManager("Failed to open settings file for writing: " + ini_file_path);
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
            LOG_WARNING_SettingsManager("Failed to parse int value for [" + section + "]." + key + ": " + str_value);
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
    
    // Template management removed - ChatTemplateManager now handles all template functionality dynamically
    // SettingsManager only stores user customizations for template variables (environment, identity, system prompt)

    // Check if settings need saving
    bool IsDirty() const {
        std::lock_guard<std::mutex> lock(settings_mutex);
        return settings_dirty;
    }

    // Set default values for new installations
    void SetDefaults() {
        std::lock_guard<std::mutex> lock(settings_mutex);
        
        // Model configuration
        SetString_Unlocked("Models", "outer_model_path", "");
        SetString_Unlocked("Models", "inner_model_path", "");
        SetString_Unlocked("Models", "summary_model_path", "");
        SetInt_Unlocked("Models", "outer_context_size", 8192);
        SetInt_Unlocked("Models", "inner_context_size", 8192);
        SetInt_Unlocked("Models", "summary_context_size", 4096);
        SetInt_Unlocked("Models", "outer_gpu_layers", 999);
        SetInt_Unlocked("Models", "inner_gpu_layers", 999);
        SetInt_Unlocked("Models", "summary_gpu_layers", 999);

        // Discord configuration
        SetString_Unlocked("Discord", "bot_token", "");
        SetString_Unlocked("Discord", "default_channel", "");
        SetString_Unlocked("Discord", "allowed_channels", "");
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

        // Note: All templates are now handled dynamically by ChatTemplateManager
        // No more hardcoded template storage needed

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

    // Settings sanitization - remove deprecated/unused settings
    void SanitizeSettings() {
        std::lock_guard<std::mutex> lock(settings_mutex);
        
        LOG_SettingsManager("Starting settings sanitization...");
        
        // Define the current valid settings structure
        static const std::unordered_map<std::string, std::vector<std::string>> VALID_SETTINGS = {
            {"Models", {
                // Current voice model settings
                "outer_model_path", "inner_model_path", "summary_model_path",
                "outer_context_size", "inner_context_size", "summary_context_size", 
                "outer_gpu_layers", "inner_gpu_layers", "summary_gpu_layers",
                "outer_system_prompt", "inner_system_prompt",
                
                // Legacy template configuration (kept for backwards compatibility)
                "environment_description", "identity_directive", "system_prompt",
                
                // Plugin model paths
                "emotag_model_path", "emotag_system_prompt", "emotag_analysis_window", "emotag_include_user"
            }},
            {"Templates", {
                // Template configuration (new dedicated section)
                "environment_description", "identity_directive", "system_prompt", "last_selected_persona"
            }},
            {"Personas", {
                // Persona management - dynamic keys are handled separately
                // Note: Persona keys follow pattern "persona_N_name" and "persona_N_directive" where N is 0-99
            }},
            {"Discord", {
                "bot_token", "default_channel", "allowed_channels", "auto_respond", "history_backfill", "backfill_limit"
            }},
            {"UI", {
                "theme", "window_width", "window_height", "auto_scroll"
            }},
            {"Logging", {
                "level", "file_output", "file_path"
            }},
            {"Plugins", {
                "sanitization", "summarization", "discord_channel_management", "history_backfill"
            }},
            {"Context", {
                "prune_threshold", "prune_target", "auto_summarize"
            }}
        };
        
        bool settings_changed = false;
        std::vector<std::string> removed_sections;
        std::vector<std::pair<std::string, std::string>> removed_keys;
        
        // Check each section in the loaded settings
        auto section_it = sections.begin();
        while (section_it != sections.end()) {
            const std::string& section_name = section_it->first;
            auto& section = section_it->second;
            
            // Check if this section is valid
            auto valid_section_it = VALID_SETTINGS.find(section_name);
            if (valid_section_it == VALID_SETTINGS.end()) {
                // This entire section is deprecated
                LOG_SettingsManager("Removing deprecated section: [" + section_name + "]");
                removed_sections.push_back(section_name);
                section_it = sections.erase(section_it);
                settings_changed = true;
                continue;
            }
            
            // Section is valid, check individual keys
            const auto& valid_keys = valid_section_it->second;
            auto key_it = section.keys.begin();
            while (key_it != section.keys.end()) {
                const std::string& key_name = key_it->first;
                bool key_is_valid = false;
                
                // Special handling for Personas section - allow dynamic persona keys
                if (section_name == "Personas") {
                    // Check if key matches persona pattern: persona_N_name or persona_N_directive
                    if (key_name.substr(0, 8) == "persona_") {
                        size_t underscore_pos = key_name.find('_', 8);
                        if (underscore_pos != std::string::npos) {
                            std::string suffix = key_name.substr(underscore_pos + 1);
                            std::string number_part = key_name.substr(8, underscore_pos - 8);
                            
                            // Check if number part is valid (0-99) and suffix is valid
                            if ((suffix == "name" || suffix == "directive") && 
                                std::all_of(number_part.begin(), number_part.end(), ::isdigit) &&
                                !number_part.empty() && std::stoi(number_part) >= 0 && std::stoi(number_part) <= 99) {
                                key_is_valid = true;
                            }
                        }
                    }
                } else {
                    // Standard validation for other sections
                    key_is_valid = (std::find(valid_keys.begin(), valid_keys.end(), key_name) != valid_keys.end());
                }
                
                if (!key_is_valid) {
                    // This key is deprecated
                    LOG_SettingsManager("Removing deprecated key: [" + section_name + "]." + key_name + " = '" + key_it->second + "'");
                    removed_keys.emplace_back(section_name, key_name);
                    key_it = section.keys.erase(key_it);
                    settings_changed = true;
                } else {
                    ++key_it;
                }
            }
            
            ++section_it;
        }
        
        // Log summary of sanitization
        if (settings_changed) {
            settings_dirty = true;
            LOG_SettingsManager("Settings sanitization complete:");
            LOG_SettingsManager("- Removed " + std::to_string(removed_sections.size()) + " deprecated sections");
            LOG_SettingsManager("- Removed " + std::to_string(removed_keys.size()) + " deprecated keys");
            
            // Detailed logging of what was removed
            for (const auto& section : removed_sections) {
                LOG_SettingsManager("  Removed section: [" + section + "]");
            }
            for (const auto& key_pair : removed_keys) {
                LOG_SettingsManager("  Removed key: [" + key_pair.first + "]." + key_pair.second);
            }
        } else {
            LOG_SettingsManager("Settings sanitization complete - no deprecated settings found");
        }
    }
    
    // Validate setting values and fix any that are invalid
    void ValidateAndFixSettings() {
        std::lock_guard<std::mutex> lock(settings_mutex);
        
        LOG_SettingsManager("Starting settings validation...");
        bool settings_changed = false;
        
        // Validate Models section
        if (sections.find("Models") != sections.end()) {
            auto& models_section = sections["Models"];
            
            // Validate context sizes
            for (const std::string& context_key : {"outer_context_size", "inner_context_size", "summary_context_size"}) {
                if (models_section.keys.find(context_key) != models_section.keys.end()) {
                    try {
                        int value = std::stoi(models_section.keys[context_key]);
                        int clamped_value = std::clamp(value, MIN_CONTEXT_SIZE, MAX_CONTEXT_SIZE);
                        if (value != clamped_value) {
                            LOG_SettingsManager("Fixed invalid " + context_key + ": " + std::to_string(value) + " -> " + std::to_string(clamped_value));
                            models_section.keys[context_key] = std::to_string(clamped_value);
                            settings_changed = true;
                        }
                    } catch (...) {
                        // Invalid value, set to default
                        int default_value = (context_key == "summary_context_size") ? 4096 : 8192;
                        LOG_SettingsManager("Fixed invalid " + context_key + ": non-numeric -> " + std::to_string(default_value));
                        models_section.keys[context_key] = std::to_string(default_value);
                        settings_changed = true;
                    }
                }
            }
            
            // Validate GPU layers
            for (const std::string& gpu_key : {"outer_gpu_layers", "inner_gpu_layers", "summary_gpu_layers"}) {
                if (models_section.keys.find(gpu_key) != models_section.keys.end()) {
                    try {
                        int value = std::stoi(models_section.keys[gpu_key]);
                        int clamped_value = std::clamp(value, MIN_GPU_LAYERS, MAX_GPU_LAYERS);
                        if (value != clamped_value) {
                            LOG_SettingsManager("Fixed invalid " + gpu_key + ": " + std::to_string(value) + " -> " + std::to_string(clamped_value));
                            models_section.keys[gpu_key] = std::to_string(clamped_value);
                            settings_changed = true;
                        }
                    } catch (...) {
                        // Invalid value, set to default
                        LOG_SettingsManager("Fixed invalid " + gpu_key + ": non-numeric -> 999");
                        models_section.keys[gpu_key] = "999";
                        settings_changed = true;
                    }
                }
            }
            
            // Validate emotag analysis window (should be 1-10)
            if (models_section.keys.find("emotag_analysis_window") != models_section.keys.end()) {
                try {
                    int value = std::stoi(models_section.keys["emotag_analysis_window"]);
                    int clamped_value = std::clamp(value, 1, 10);
                    if (value != clamped_value) {
                        LOG_SettingsManager("Fixed invalid emotag_analysis_window: " + std::to_string(value) + " -> " + std::to_string(clamped_value));
                        models_section.keys["emotag_analysis_window"] = std::to_string(clamped_value);
                        settings_changed = true;
                    }
                } catch (...) {
                    LOG_SettingsManager("Fixed invalid emotag_analysis_window: non-numeric -> 3");
                    models_section.keys["emotag_analysis_window"] = "3";
                    settings_changed = true;
                }
            }
        }
        
        // Validate Context section
        if (sections.find("Context") != sections.end()) {
            auto& context_section = sections["Context"];
            
            // Validate threshold values (should be 0.0 to 1.0)
            for (const std::string& threshold_key : {"prune_threshold", "prune_target"}) {
                if (context_section.keys.find(threshold_key) != context_section.keys.end()) {
                    try {
                        float value = std::stof(context_section.keys[threshold_key]);
                        float clamped_value = std::clamp(value, 0.0f, 1.0f);
                        if (std::abs(value - clamped_value) > 0.001f) {
                            LOG_SettingsManager("Fixed invalid " + threshold_key + ": " + std::to_string(value) + " -> " + std::to_string(clamped_value));
                            context_section.keys[threshold_key] = std::to_string(clamped_value);
                            settings_changed = true;
                        }
                    } catch (...) {
                        float default_value = (threshold_key == "prune_threshold") ? 0.8f : 0.4f;
                        LOG_SettingsManager("Fixed invalid " + threshold_key + ": non-numeric -> " + std::to_string(default_value));
                        context_section.keys[threshold_key] = std::to_string(default_value);
                        settings_changed = true;
                    }
                }
            }
        }
        
        if (settings_changed) {
            settings_dirty = true;
            LOG_SettingsManager("Settings validation complete - some values were corrected");
        } else {
            LOG_SettingsManager("Settings validation complete - all values are valid");
        }
    }
};
