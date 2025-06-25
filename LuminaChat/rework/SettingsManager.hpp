#pragma once

#include "Logger.hpp"
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <mutex>

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
    // Load settings from .ini file
    bool LoadSettings(const std::string& ini_path) {
        std::lock_guard<std::mutex> lock(settings_mutex);
        
        LOG_SettingsManager("Attempting to load settings from: " + ini_path);
        
        std::ifstream file(ini_path);
        if (!file.is_open()) {
            LOG_ERROR_SettingsManager("Failed to open settings file: " + ini_path);
            return false;
        }

        LOG_SettingsManager("File opened successfully, clearing existing sections");
        sections.clear();
        ini_file_path = ini_path;
        
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
                
                sections[current_section].keys[key] = value;
                LOG_SettingsManager("Set [" + current_section + "]." + key + " = '" + value + "'");
            }
        }

        file.close();
        settings_dirty = false;
        LOG_SettingsManager("Successfully loaded " + std::to_string(line_count) + " lines from: " + ini_path);
        return true;
    }

    // Save settings to .ini file
    bool SaveSettings() {
        std::lock_guard<std::mutex> lock(settings_mutex);
        
        if (ini_file_path.empty()) {
            LOG_ERROR("SettingsManager", "No ini file path set for saving");
            return false;
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
                // Quote values that contain spaces or special characters
                bool needs_quotes = value.find_first_of(" \t\"") != std::string::npos;
                if (needs_quotes) {
                    file << key << "=\"" << value << "\"\n";
                } else {
                    file << key << "=" << value << "\n";
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

    // Get integer value
    int GetInt(const std::string& section, const std::string& key, int default_value = 0) {
        std::string str_value = GetString(section, key);
        if (str_value.empty()) {
            return default_value;
        }

        try {
            return std::stoi(str_value);
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

private:
    // Set default values for new installations
    void SetDefaults() {
        // Model configuration
        SetString("Models", "main_model_path", "");
        SetString("Models", "summary_model_path", "");
        SetInt("Models", "main_context_size", 8192);
        SetInt("Models", "summary_context_size", 4096);
        SetInt("Models", "main_gpu_layers", -1);
        SetInt("Models", "summary_gpu_layers", -1);

        // Discord configuration
        SetString("Discord", "bot_token", "");
        SetString("Discord", "default_channel", "");
        SetBool("Discord", "auto_respond", true);
        SetBool("Discord", "history_backfill", true);
        SetInt("Discord", "backfill_limit", 100);

        // UI preferences
        SetString("UI", "theme", "dark");
        SetInt("UI", "window_width", 800);
        SetInt("UI", "window_height", 600);
        SetBool("UI", "auto_scroll", true);

        // Logging configuration
        SetString("Logging", "level", "INFO");
        SetBool("Logging", "file_output", false);
        SetString("Logging", "file_path", "luminachat.log");

        // Plugin configuration
        SetBool("Plugins", "sanitization", true);
        SetBool("Plugins", "summarization", true);
        SetBool("Plugins", "discord_channel_management", true);
        SetBool("Plugins", "history_backfill", true);

        // Context management
        SetFloat("Context", "prune_threshold", 0.8f);
        SetFloat("Context", "prune_target", 0.4f);
        SetBool("Context", "auto_summarize", true);

        // Default templates
        SetChatTemplate("default", GetDefaultChatTemplate());
        SetChatTemplate("summary", GetDefaultSummaryTemplate());

        LOG_SettingsManager("Initialized default settings");
    }

    std::string GetDefaultChatTemplate() {
        return R"({{- bos_token }}

<|start_header_id|>env<|end_header_id|>
{{ overarching_environment }}
<|eot_id|>

<|start_header_id|>persona<|end_header_id|>
{{ identity_directive }}
<|eot_id|>

<|start_header_id|>system_message<|end_header_id|>
{{ system_prompt }}
<|eot_id|>

{% if old_chat_summary %}
<|start_header_id|>old_chat_summary<|end_header_id|>
{{ old_chat_summary }}
<|eot_id|>
{% endif %}

{% if past_sessions and past_sessions|length > 0 %}
  {% for memory in past_sessions %}
<|start_header_id|>memory_{{ loop.index }}<|end_header_id|>
{{ memory }}
<|eot_id|>
  {% endfor %}
{% endif %}

{% if summary %}
<|start_header_id|>summary<|end_header_id|>
{{ summary }}
<|eot_id|>
{% endif %}

{%- for msg in messages %}
  {% if msg.role == "assistant" %}
<|start_header_id|>assistant<|end_header_id|>
{{ msg.content | trim }}<|eot_id|>
  {% else %}
<|start_header_id|>user<|end_header_id|>
[{{ msg.role }}] {{ msg.content | trim }}<|eot_id|>
  {% endif %}
{%- endfor %}

<|start_header_id|>assistant<|end_header_id|>)";
    }

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
};
