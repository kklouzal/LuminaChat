#pragma once

// LuminaChat_UIUtilities.hpp - Common UI utility functions and patterns
// This header contains reusable UI construction patterns and validation functions
// to reduce code duplication across UI creation methods.

#include <wx/wx.h>
#include <wx/stattext.h>
#include <wx/textctrl.h>
#include <wx/button.h>
#include <wx/choice.h>
#include <wx/checkbox.h>
#include <wx/sizer.h>
#include <string>
#include <vector>
#include "../Logger.hpp"

// === Common UI Construction Patterns ===

namespace LuminaChatUIUtilities {
    
    // Standard label creation with consistent styling
    inline wxStaticText* CreateLabel(wxWindow* parent, const wxString& text, bool bold = false) {
        auto* label = new wxStaticText(parent, wxID_ANY, text);
        if (bold) {
            wxFont font = label->GetFont();
            font.SetWeight(wxFONTWEIGHT_BOLD);
            label->SetFont(font);
        }
        return label;
    }
    
    // Standard text control creation with validation
    inline wxTextCtrl* CreateTextCtrl(wxWindow* parent, const wxString& value = wxEmptyString, 
                                     int style = 0, const wxString& tooltip = wxEmptyString) {
        auto* textctrl = new wxTextCtrl(parent, wxID_ANY, value, wxDefaultPosition, wxDefaultSize, style);
        if (!tooltip.IsEmpty()) {
            textctrl->SetToolTip(tooltip);
        }
        return textctrl;
    }
    
    // Standard button creation with consistent styling
    inline wxButton* CreateButton(wxWindow* parent, int id, const wxString& label, 
                                 const wxString& tooltip = wxEmptyString) {
        auto* button = new wxButton(parent, id, label);
        if (!tooltip.IsEmpty()) {
            button->SetToolTip(tooltip);
        }
        return button;
    }
    
    // Create a horizontal sizer with label and control
    inline wxBoxSizer* CreateLabeledControl(wxWindow* parent, const wxString& label_text, 
                                           wxWindow* control, bool label_bold = false) {
        auto* sizer = new wxBoxSizer(wxHORIZONTAL);
        auto* label = CreateLabel(parent, label_text, label_bold);
        
        sizer->Add(label, 0, wxALIGN_CENTER_VERTICAL | wxALL, LuminaChatConstants::CONTROL_SPACING);
        sizer->Add(control, 1, wxEXPAND | wxALL, LuminaChatConstants::CONTROL_SPACING);
        
        return sizer;
    }
    
    // Create a choice control with multiple options
    inline wxChoice* CreateChoice(wxWindow* parent, const std::vector<wxString>& choices, 
                                 int default_selection = 0, const wxString& tooltip = wxEmptyString) {
        auto* choice = new wxChoice(parent, wxID_ANY);
        for (const auto& choice_text : choices) {
            choice->Append(choice_text);
        }
        if (default_selection >= 0 && default_selection < static_cast<int>(choices.size())) {
            choice->SetSelection(default_selection);
        }
        if (!tooltip.IsEmpty()) {
            choice->SetToolTip(tooltip);
        }
        return choice;
    }
    
    // Create a checkbox with auto-save functionality
    template<typename Frame>
    inline wxCheckBox* CreateAutoSaveCheckbox(wxWindow* parent, Frame* frame, const wxString& label,
                                             const std::string& settings_section, const std::string& settings_key,
                                             bool default_value = false, const wxString& tooltip = wxEmptyString) {
        auto* checkbox = new wxCheckBox(parent, wxID_ANY, label);
        checkbox->SetValue(default_value);
        
        if (!tooltip.IsEmpty()) {
            checkbox->SetToolTip(tooltip);
        }
        
        // Bind auto-save event
        checkbox->Bind(wxEVT_CHECKBOX, [frame, checkbox, settings_section, settings_key, label](wxCommandEvent& event) {
            if (frame->settings_manager && checkbox) {
                bool value = checkbox->GetValue();
                frame->settings_manager->SetBool(settings_section, settings_key, value);
                frame->settings_manager->SaveSettings();
                LOG_UI(label.ToStdString() + " auto-saved: " + std::string(value ? "enabled" : "disabled"));
            }
            event.Skip();
        });
        
        return checkbox;
    }
    
    // Create a text control with auto-save functionality
    template<typename Frame>
    inline wxTextCtrl* CreateAutoSaveTextCtrl(wxWindow* parent, Frame* frame, const wxString& value,
                                             const std::string& settings_section, const std::string& settings_key,
                                             int style = 0, const wxString& tooltip = wxEmptyString,
                                             std::function<void()> on_save_callback = nullptr) {
        auto* textctrl = CreateTextCtrl(parent, value, style, tooltip);
        
        // Bind auto-save event on kill focus
        textctrl->Bind(wxEVT_KILL_FOCUS, [frame, textctrl, settings_section, settings_key, on_save_callback](wxFocusEvent& event) {
            if (frame->settings_manager && textctrl) {
                std::string text_value = textctrl->GetValue().ToStdString();
                frame->settings_manager->SetString(settings_section, settings_key, text_value);
                frame->settings_manager->SaveSettings();
                
                if (on_save_callback) {
                    on_save_callback();
                }
                
                LOG_UI("Auto-saved setting: " + settings_section + "." + settings_key);
            }
            event.Skip();
        });
        
        return textctrl;
    }
}

// === Input Validation Utilities ===

namespace LuminaChatValidationUtilities {
    
    // Validate Discord channel ID format
    inline bool IsValidDiscordChannelID(const std::string& channel_id) {
        if (channel_id.empty() || channel_id.length() > LuminaChatValidation::MAX_DISCORD_CHANNEL_ID_LENGTH) {
            return false;
        }
        return std::all_of(channel_id.begin(), channel_id.end(), ::isdigit);
    }
    
    // Validate context size
    inline bool IsValidContextSize(int context_size) {
        return context_size >= LuminaChatValidation::MIN_CONTEXT_SIZE && 
               context_size <= LuminaChatValidation::MAX_CONTEXT_SIZE;
    }
    
    // Validate temperature value
    inline bool IsValidTemperature(float temperature) {
        return temperature >= LuminaChatValidation::MIN_TEMPERATURE && 
               temperature <= LuminaChatValidation::MAX_TEMPERATURE;
    }
    
    // Validate max tokens value
    inline bool IsValidMaxTokens(int max_tokens) {
        return max_tokens >= LuminaChatValidation::MIN_MAX_TOKENS && 
               max_tokens <= LuminaChatValidation::MAX_MAX_TOKENS;
    }
    
    // Parse comma-separated Discord channel IDs
    inline std::vector<std::string> ParseDiscordChannelIDs(const std::string& channels_str) {
        std::vector<std::string> channel_ids;
        
        if (channels_str.empty()) {
            return channel_ids;
        }
        
        std::stringstream ss(channels_str);
        std::string channel_id;
        
        while (std::getline(ss, channel_id, ',')) {
            // Trim whitespace
            channel_id.erase(0, channel_id.find_first_not_of(" \t\n\r"));
            channel_id.erase(channel_id.find_last_not_of(" \t\n\r") + 1);
            
            if (IsValidDiscordChannelID(channel_id)) {
                channel_ids.push_back(channel_id);
            }
        }
        
        return channel_ids;
    }
}
