#pragma once

// LuminaChat_Settings.hpp - Settings and configuration management
// This header contains all settings loading/saving, template configuration,
// and UI settings persistence functionality.

#include <string>

// === Settings and Configuration Methods ===

inline std::string LuminaChatFrame::GetEnvironmentDescriptionFromUI() const {
    if (settings_ui) {
        return settings_ui->GetEnvironmentDescription();
    }
    return "";
}

inline std::string LuminaChatFrame::GetIdentityDirectiveFromUI() const {
    if (settings_ui) {
        return settings_ui->GetIdentityDirective();
    }
    return "";
}

inline std::string LuminaChatFrame::GetOuterVoiceSystemPromptFromUI() const {
    if (outer_voice_ui) {
        return outer_voice_ui->GetSystemPrompt();
    }
    return "";
}

inline std::string LuminaChatFrame::GetInnerVoiceSystemPromptFromUI() const {
    if (inner_voice_ui) {
        return inner_voice_ui->GetSystemPrompt();
    }
    return "";
}

inline void LuminaChatFrame::ApplyTemplateSettingsToContext(ContextInfo* context, const std::string& context_id) {
    if (!context) {
        LOG_WARNING_LuminaChat("Cannot apply template settings to null context");
        return;
    }
    
    // Don't apply template settings to summary contexts
    if (context_id.find("summary") != std::string::npos) {
        LOG_LuminaChat("Skipping template settings for summary context: " + context_id);
        return;
    }
    
    std::string environment_description = GetEnvironmentDescriptionFromUI();
    std::string identity_directive = GetIdentityDirectiveFromUI();
    
    // Get the appropriate system prompt based on context ID
    std::string system_prompt;
    if (context_id.find("outer") != std::string::npos) {
        system_prompt = GetOuterVoiceSystemPromptFromUI();
        LOG_LuminaChat("Using Outer Voice system prompt for context: " + context_id + " (length: " + std::to_string(system_prompt.length()) + ")");
        if (!system_prompt.empty()) {
            LOG_LuminaChat("Outer Voice system prompt content (first 50 chars): '" + system_prompt.substr(0, 50) + (system_prompt.length() > 50 ? "..." : "") + "'");
        }
    } else if (context_id.find("inner") != std::string::npos) {
        system_prompt = GetInnerVoiceSystemPromptFromUI();
        LOG_LuminaChat("Using Inner Voice system prompt for context: " + context_id + " (length: " + std::to_string(system_prompt.length()) + ")");
        if (!system_prompt.empty()) {
            LOG_LuminaChat("Inner Voice system prompt content (first 50 chars): '" + system_prompt.substr(0, 50) + (system_prompt.length() > 50 ? "..." : "") + "'");
        }
    } else {
        // Default to outer voice for unspecified contexts
        system_prompt = GetOuterVoiceSystemPromptFromUI();
        LOG_LuminaChat("Using default (Outer Voice) system prompt for context: " + context_id + " (length: " + std::to_string(system_prompt.length()) + ")");
        if (!system_prompt.empty()) {
            LOG_LuminaChat("Default system prompt content (first 50 chars): '" + system_prompt.substr(0, 50) + (system_prompt.length() > 50 ? "..." : "") + "'");
        }
    }
    
    if (!environment_description.empty()) {
        context->UpdateEnvironment(environment_description);
        LOG_LuminaChat("Applied environment description to context: " + context_id);
    }
    
    if (!identity_directive.empty()) {
        context->UpdateIdentity(identity_directive);
        LOG_LuminaChat("Applied identity directive to context: " + context_id);
    }
    
    // Always update system prompt to ensure proper clearing when empty
    context->UpdateSystemPrompt(system_prompt);
    if (!system_prompt.empty()) {
        LOG_LuminaChat("Applied system prompt to context: " + context_id + " (first 50 chars: '" + system_prompt.substr(0, 50) + (system_prompt.length() > 50 ? "..." : "") + "')");
    } else {
        LOG_LuminaChat("Cleared system prompt for context: " + context_id);
    }
}

inline void LuminaChatFrame::LoadUISettings() {
    if (!settings_manager) {
        LOG_WARNING_LuminaChat("Cannot load UI settings - SettingsManager not initialized");
        return;
    }
    
    LOG_LuminaChat("Loading UI settings from configuration...");
    
    try {
        // Load outer voice UI configuration
        if (outer_voice_ui) {
            outer_voice_ui->LoadSettings();
        }
        
        // Load inner voice UI configuration
        if (inner_voice_ui) {
            inner_voice_ui->LoadSettings();
        }
        
        // Load general settings UI configuration
        if (settings_ui) {
            settings_ui->LoadSettings();
        }
        
        // Load plugin settings through UI managers
        if (summarization_ui) {
            summarization_ui->LoadSettings(settings_manager.get());
        }
        if (emotag_ui) {
            emotag_ui->LoadSettings(settings_manager.get());
        }
        if (context_pruning_ui) {
            context_pruning_ui->LoadSettings(settings_manager.get());
        }
        
        // Load Discord settings
        if (discord_token_text) {
            std::string discord_token = settings_manager->GetString("Discord", "bot_token", "");
            discord_token_text->SetValue(discord_token);
            if (!discord_token.empty()) {
                LOG_LuminaChat("Loaded Discord bot token from settings");
            }
        }
        
        if (allowed_channels_text) {
            std::string allowed_channels = settings_manager->GetString("Discord", "allowed_channels", "");
            LOG_LuminaChat("Loading allowed channels from settings: '" + allowed_channels + "'");
            allowed_channels_text->SetValue(allowed_channels);
            if (!allowed_channels.empty()) {
                LOG_LuminaChat("Loaded Discord allowed channels from settings");
                UpdateDiscordAllowedChannels();
            } else {
                LOG_LuminaChat("No allowed channels found in settings");
            }
        }
        
        if (auto_respond_checkbox) {
            bool auto_respond = settings_manager->GetBool("Discord", "auto_respond", true);
            auto_respond_checkbox->SetValue(auto_respond);
            LOG_LuminaChat("Loaded Discord auto-respond setting: " + std::string(auto_respond ? "enabled" : "disabled"));
        }
        
        // Load logging level
        if (log_level_choice) {
            std::string log_level = settings_manager->GetString("Logging", "level", "INFO");
            int selection = 1; // Default to INFO
            if (log_level == "DEBUG") selection = 0;
            else if (log_level == "INFO") selection = 1;
            else if (log_level == "WARNING") selection = 2;
            else if (log_level == "ERROR") selection = 3;
            
            log_level_choice->SetSelection(selection);
            LOG_LuminaChat("Loaded log level: " + log_level);
        }
        
        // Update UI to reflect loaded values
        UpdateUI();
        
        LOG_LuminaChat("UI settings loaded successfully");
        
    } catch (const std::exception& e) {
        LOG_ERROR_LuminaChat("Error loading UI settings: " + std::string(e.what()));
    }
}

inline void LuminaChatFrame::SaveUISettings() {
    if (!settings_manager) {
        LOG_WARNING_LuminaChat("Cannot save UI settings - SettingsManager not initialized");
        return;
    }
    
    LOG_LuminaChat("Saving UI settings to configuration...");
    
    try {
        // Save outer voice UI configuration
        if (outer_voice_ui) {
            outer_voice_ui->SaveSettings();
        }
        
        // Save inner voice UI configuration
        if (inner_voice_ui) {
            inner_voice_ui->SaveSettings();
        }
        
        // Save general settings UI configuration
        if (settings_ui) {
            settings_ui->SaveSettings();
        }
        
        // Save plugin settings through UI managers
        if (summarization_ui) {
            summarization_ui->SaveSettings(settings_manager.get());
        }
        if (emotag_ui) {
            emotag_ui->SaveSettings(settings_manager.get());
        }
        if (context_pruning_ui) {
            context_pruning_ui->SaveSettings(settings_manager.get());
        }
        
        // Save Discord settings
        if (discord_token_text) {
            std::string discord_token = discord_token_text->GetValue().ToStdString();
            settings_manager->SetString("Discord", "bot_token", discord_token);
        }
        
        if (allowed_channels_text) {
            std::string allowed_channels = allowed_channels_text->GetValue().ToStdString();
            LOG_LuminaChat("Saving allowed channels to settings: '" + allowed_channels + "'");
            settings_manager->SetString("Discord", "allowed_channels", allowed_channels);
        }
        
        if (auto_respond_checkbox) {
            bool auto_respond = auto_respond_checkbox->GetValue();
            settings_manager->SetBool("Discord", "auto_respond", auto_respond);
        }
        
        // Save logging level
        if (log_level_choice) {
            int selection = log_level_choice->GetSelection();
            std::string log_level = "INFO"; // Default
            switch (selection) {
                case 0: log_level = "DEBUG"; break;
                case 1: log_level = "INFO"; break;
                case 2: log_level = "WARNING"; break;
                case 3: log_level = "ERROR"; break;
            }
            settings_manager->SetString("Logging", "level", log_level);
        }
        
        // Actually save the settings to file
        if (settings_manager->SaveSettings()) {
            LOG_LuminaChat("UI settings saved successfully");
        } else {
            LOG_ERROR_LuminaChat("Failed to save UI settings to file");
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR_LuminaChat("Error saving UI settings: " + std::string(e.what()));
    }
}
