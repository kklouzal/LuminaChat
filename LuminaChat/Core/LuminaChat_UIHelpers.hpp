#pragma once

// LuminaChat_UIHelpers.hpp - UI helper and update methods
// This header contains UI update utilities, status management, error/success messages,
// and other helper methods for maintaining the user interface state.

#include <string>
#include <sstream>
#include <algorithm>
#include "LuminaChat_UIUtilities.hpp"

// === UI Helper and Update Methods ===

inline void LuminaChatFrame::UpdateUI() {
    // Update Voice Settings UIs
    if (outer_voice_ui) {
        outer_voice_ui->UpdateUI();
    }
    if (inner_voice_ui) {
        inner_voice_ui->UpdateUI();
    }
    
    // Update General Settings UI
    if (settings_ui) {
        settings_ui->UpdateUI();
    }
    
    send_button->Enable(model_loaded && running);
    connect_discord_button->Enable(running && !discord_token_text->GetValue().IsEmpty());
    
    if (model_loaded) {
        SetStatusText("Model Loaded", 1);
    } else if (model_loading) {
        SetStatusText("Loading Model...", 1);
    } else {
        SetStatusText("No Model", 1);
    }
    
    // Update context status whenever UI is updated
    UpdateContextStatus();
}

inline void LuminaChatFrame::UpdateContextStatus() {
    if (!llama_manager || !model_loaded || !running) {
        SetStatusText("Inner: --/-- | Outer: --/--", 2);
        return;
    }
    
    try {
        auto* inner_context = llama_manager->GetContextInfo("inner_context");
        auto* outer_context = llama_manager->GetContextInfo("outer_context");
        
        std::string inner_status = "Inner: --/--";
        std::string outer_status = "Outer: --/--";
        
        if (inner_context) {
            int inner_used = static_cast<int>(inner_context->GetActualContextTokens());
            int inner_total = static_cast<int>(inner_context->GetMaxContextTokens());
            inner_status = "Inner: " + std::to_string(inner_used) + "/" + std::to_string(inner_total);
        }
        
        if (outer_context) {
            int outer_used = static_cast<int>(outer_context->GetActualContextTokens());
            int outer_total = static_cast<int>(outer_context->GetMaxContextTokens());
            outer_status = "Outer: " + std::to_string(outer_used) + "/" + std::to_string(outer_total);
        }
        
        SetStatusText(inner_status + " | " + outer_status, 2);
        
    } catch (const std::exception& e) {
        LOG_ERROR_LuminaChat("Error updating context status: " + std::string(e.what()));
        SetStatusText("Context Error", 2);
    }
}

inline void LuminaChatFrame::SetGenerationUIState(bool generating) {
    send_button->Enable(!generating && model_loaded && running);
    stop_button->Enable(generating);
    chat_input->Enable(!generating);
    
    if (generating) {
        SetStatusText("Generating Response...", 0);
    } else {
        SetStatusText("Ready", 0);
    }
}

inline void LuminaChatFrame::UpdateSummaryPluginStatus(const std::string& status, const wxColour& color) {
    if (summarization_ui) {
        summarization_ui->UpdateStatus(status, color);
    }
}

inline void LuminaChatFrame::UpdateEmoTagPluginStatus(const std::string& status, const wxColour& color) {
    if (emotag_ui) {
        emotag_ui->UpdateStatus(status, color);
    }
}

inline void LuminaChatFrame::UpdateSummaryPluginDebugInfo() {
    if (summarization_ui && summarization_plugin) {
        summarization_ui->UpdateDebugInfo(summarization_plugin.get());
    }
}

inline void LuminaChatFrame::UpdateEmoTagPluginDebugInfo() {
    if (emotag_ui && emotag_plugin) {
        emotag_ui->UpdateDebugInfo(emotag_plugin.get());
    }
}

// === Common UI Message Methods ===
// Note: These methods are maintained for backward compatibility.
// New code should use the centralized error handling methods in ErrorHandling.hpp directly

// === Discord Integration Helper Methods ===

inline void LuminaChatFrame::UpdateDiscordAllowedChannels() {
    if (!discord_manager || !allowed_channels_text) {
        return;
    }
    
    try {
        // Parse the comma-separated channel IDs using utility function
        std::string channels_str = allowed_channels_text->GetValue().ToStdString();
        auto channel_ids = LuminaChatValidationUtilities::ParseDiscordChannelIDs(channels_str);
        
        // Update the Discord manager with the new allowed channels
        discord_manager->SetAllowedChannels(channel_ids);
        
        LOG_LuminaChat(wxString::Format("Updated allowed Discord channels: %d channels configured", 
                     static_cast<int>(channel_ids.size())).ToStdString());
        
    } catch (const std::exception& e) {
        HandleError("Error updating Discord allowed channels: " + std::string(e.what()), "Discord Integration");
    }
}

inline void LuminaChatFrame::RefreshDiscordChannelList() {
    if (!discord_manager || !discord_connected) {
        return;
    }
    
    try {
        auto channels = discord_manager->GetChannels();
        channels_list->Clear();
        active_channel_choice->Clear();
        
        LOG_LuminaChat(wxString::Format("Loaded %d Discord channels", static_cast<int>(channels.size())).ToStdString());
        
        for (const auto& channel : channels) {
            wxString channel_display = wxString::Format("%s [%s] (%s)%s", 
                                                       channel.name, 
                                                       channel.guild_name,
                                                       channel.id,
                                                       channel.is_allowed ? " *ALLOWED*" : "");
            channels_list->Append(channel_display);
            active_channel_choice->Append(channel.name);
        }
        
        if (!channels.empty()) {
            active_channel_choice->SetSelection(0);
        }
        
        // Update status
        size_t guild_count = discord_manager->GetGuildCount();
        size_t allowed_count = 0;
        for (const auto& channel : channels) {
            if (channel.is_allowed) allowed_count++;
        }
        
        LOG_LuminaChat(wxString::Format("Discord status: %d guilds, %d channels (%d allowed)", 
                     static_cast<int>(guild_count), 
                     static_cast<int>(channels.size()),
                     static_cast<int>(allowed_count)).ToStdString());
        
    } catch (const std::exception& e) {
        LOG_ERROR_LuminaChat(wxString::Format("Error refreshing Discord channel list: %s", e.what()).ToStdString());
    }
}
