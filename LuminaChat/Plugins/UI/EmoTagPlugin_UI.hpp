#pragma once

#include <wx/wx.h>
#include <wx/panel.h>
#include <wx/textctrl.h>
#include <wx/button.h>
#include <wx/stattext.h>
#include <wx/slider.h>
#include <wx/checkbox.h>
#include <wx/sizer.h>
#include <wx/scrolwin.h>
#include <wx/filedlg.h>
#include "../EmoTagPlugin.hpp"
#include "../../SettingsManager.hpp"

class EmoTagPluginUI : public wxEvtHandler {
public:
    EmoTagPluginUI(wxWindow* parent);
    ~EmoTagPluginUI() = default;

    // Panel creation
    wxPanel* CreatePanel();
    
    // UI update methods
    void UpdateStatus(const std::string& status, const wxColour& color = wxNullColour);
    void UpdateDebugInfo(LuminaChat::EmoTagPlugin* plugin);
    
    // Settings persistence
    void LoadSettings(SettingsManager* settings_manager);
    void SaveSettings(SettingsManager* settings_manager);
    
    // Accessors for main frame
    std::string GetModelPath() const;
    std::string GetSystemPrompt() const;
    int GetAnalysisWindowSize() const;
    bool GetIncludeUserMessages() const;

private:
    wxWindow* parent_window;
    
    // UI Components
    wxPanel* emotag_panel;
    wxTextCtrl* emotag_model_path_text;
    wxButton* browse_emotag_model_button;
    wxStaticText* emotag_status_text;
    wxTextCtrl* emotag_log_output_text;
    wxTextCtrl* emotag_last_generation_text;
    wxTextCtrl* emotag_system_prompt_text;
    wxSlider* emotag_window_size_slider;
    wxStaticText* emotag_window_size_text;
    wxCheckBox* emotag_include_user_checkbox;
    
    // Event IDs
    enum {
        ID_BrowseEmoTagModel = 3100,
        ID_WindowSizeSlider,
        ID_IncludeUserCheckbox
    };
    
    // Event handlers
    void OnBrowseModel(wxCommandEvent& event);
    void OnWindowSizeChanged(wxCommandEvent& event);
    void OnIncludeUserChanged(wxCommandEvent& event);
    
    // Internal methods
    void CreateModelConfigSection(wxScrolledWindow* parent, wxBoxSizer* sizer);
    void CreateStatusSection(wxScrolledWindow* parent, wxBoxSizer* sizer);
    void CreateConfigSection(wxScrolledWindow* parent, wxBoxSizer* sizer);
    void CreatePromptSection(wxScrolledWindow* parent, wxBoxSizer* sizer);
    void UpdateWindowSizeLabel();
};

// Implementation
inline EmoTagPluginUI::EmoTagPluginUI(wxWindow* parent) 
    : parent_window(parent) {
}

inline wxPanel* EmoTagPluginUI::CreatePanel() {
    emotag_panel = new wxPanel(parent_window, wxID_ANY);
    
    wxScrolledWindow* scroll = new wxScrolledWindow(emotag_panel, wxID_ANY);
    scroll->SetScrollRate(5, 5);
    
    wxBoxSizer* main_sizer = new wxBoxSizer(wxVERTICAL);
    
    // Create sections
    CreateModelConfigSection(scroll, main_sizer);
    CreateConfigSection(scroll, main_sizer);
    CreateStatusSection(scroll, main_sizer);
    CreatePromptSection(scroll, main_sizer);
    
    scroll->SetSizer(main_sizer);
    
    wxBoxSizer* panel_sizer = new wxBoxSizer(wxVERTICAL);
    panel_sizer->Add(scroll, 1, wxEXPAND | wxALL, 5);
    emotag_panel->SetSizer(panel_sizer);
    
    return emotag_panel;
}

inline void EmoTagPluginUI::CreateModelConfigSection(wxScrolledWindow* parent, wxBoxSizer* sizer) {
    wxStaticBoxSizer* model_box = new wxStaticBoxSizer(wxVERTICAL, parent, "EmoTag Model Configuration");
    
    // Model path
    wxBoxSizer* path_sizer = new wxBoxSizer(wxHORIZONTAL);
    path_sizer->Add(new wxStaticText(parent, wxID_ANY, "Model Path:"), 0, wxALIGN_CENTER_VERTICAL | wxRIGHT, 5);
    
    emotag_model_path_text = new wxTextCtrl(parent, wxID_ANY, "", wxDefaultPosition, wxSize(400, -1));
    path_sizer->Add(emotag_model_path_text, 1, wxEXPAND | wxRIGHT, 5);
    
    browse_emotag_model_button = new wxButton(parent, ID_BrowseEmoTagModel, "Browse...");
    browse_emotag_model_button->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &EmoTagPluginUI::OnBrowseModel, this);
    path_sizer->Add(browse_emotag_model_button, 0);
    
    model_box->Add(path_sizer, 0, wxEXPAND | wxALL, 5);
    
    sizer->Add(model_box, 0, wxEXPAND | wxALL, 5);
}

inline void EmoTagPluginUI::CreateConfigSection(wxScrolledWindow* parent, wxBoxSizer* sizer) {
    wxStaticBoxSizer* config_box = new wxStaticBoxSizer(wxVERTICAL, parent, "Analysis Configuration");
    
    // Analysis window size
    wxBoxSizer* window_sizer = new wxBoxSizer(wxHORIZONTAL);
    window_sizer->Add(new wxStaticText(parent, wxID_ANY, "Analysis Window Size:"), 0, wxALIGN_CENTER_VERTICAL | wxRIGHT, 5);
    
    emotag_window_size_slider = new wxSlider(parent, ID_WindowSizeSlider, 3, 1, 10, wxDefaultPosition, wxSize(200, -1));
    emotag_window_size_slider->Bind(wxEVT_COMMAND_SLIDER_UPDATED, &EmoTagPluginUI::OnWindowSizeChanged, this);
    window_sizer->Add(emotag_window_size_slider, 0, wxRIGHT, 10);
    
    emotag_window_size_text = new wxStaticText(parent, wxID_ANY, "3 responses");
    window_sizer->Add(emotag_window_size_text, 0, wxALIGN_CENTER_VERTICAL);
    
    config_box->Add(window_sizer, 0, wxEXPAND | wxALL, 5);
    
    // Include user messages checkbox
    emotag_include_user_checkbox = new wxCheckBox(parent, ID_IncludeUserCheckbox, "Include user messages in analysis");
    emotag_include_user_checkbox->Bind(wxEVT_COMMAND_CHECKBOX_CLICKED, &EmoTagPluginUI::OnIncludeUserChanged, this);
    emotag_include_user_checkbox->SetValue(false);
    config_box->Add(emotag_include_user_checkbox, 0, wxEXPAND | wxALL, 5);
    
    sizer->Add(config_box, 0, wxEXPAND | wxALL, 5);
}

inline void EmoTagPluginUI::CreateStatusSection(wxScrolledWindow* parent, wxBoxSizer* sizer) {
    wxStaticBoxSizer* status_box = new wxStaticBoxSizer(wxVERTICAL, parent, "Status & Debug Information");
    
    // Status text
    emotag_status_text = new wxStaticText(parent, wxID_ANY, "Not initialized");
    status_box->Add(emotag_status_text, 0, wxEXPAND | wxALL, 5);
    
    // Log output
    status_box->Add(new wxStaticText(parent, wxID_ANY, "Log Output:"), 0, wxEXPAND | wxTOP, 5);
    emotag_log_output_text = new wxTextCtrl(parent, wxID_ANY, "", wxDefaultPosition, wxSize(-1, 100), 
                                          wxTE_MULTILINE | wxTE_READONLY);
    emotag_log_output_text->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
    status_box->Add(emotag_log_output_text, 0, wxEXPAND | wxALL, 5);
    
    // Last generation
    status_box->Add(new wxStaticText(parent, wxID_ANY, "Last Generation:"), 0, wxEXPAND | wxTOP, 5);
    emotag_last_generation_text = new wxTextCtrl(parent, wxID_ANY, "", wxDefaultPosition, wxSize(-1, 100), 
                                                wxTE_MULTILINE | wxTE_READONLY);
    emotag_last_generation_text->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
    status_box->Add(emotag_last_generation_text, 0, wxEXPAND | wxALL, 5);
    
    sizer->Add(status_box, 1, wxEXPAND | wxALL, 5);
}

inline void EmoTagPluginUI::CreatePromptSection(wxScrolledWindow* parent, wxBoxSizer* sizer) {
    wxStaticBoxSizer* prompt_box = new wxStaticBoxSizer(wxVERTICAL, parent, "System Prompt");
    
    emotag_system_prompt_text = new wxTextCtrl(parent, wxID_ANY, 
        "Analyze the emotional tone and state of the following AI responses. Provide a brief emotional analysis focusing on mood, engagement level, and conversational dynamics.",
        wxDefaultPosition, wxSize(-1, 100), wxTE_MULTILINE);
    
    prompt_box->Add(emotag_system_prompt_text, 1, wxEXPAND | wxALL, 5);
    sizer->Add(prompt_box, 0, wxEXPAND | wxALL, 5);
}

inline void EmoTagPluginUI::UpdateStatus(const std::string& status, const wxColour& color) {
    if (emotag_status_text) {
        emotag_status_text->SetLabel(status);
        if (color.IsOk()) {
            emotag_status_text->SetForegroundColour(color);
        }
        emotag_status_text->GetParent()->Layout();
    }
}

inline void EmoTagPluginUI::UpdateDebugInfo(LuminaChat::EmoTagPlugin* plugin) {
    if (!plugin) return;
    
    // Update log output only if content has changed
    if (emotag_log_output_text) {
        auto log_history = plugin->GetLogHistory();
        std::string log_text;
        for (const auto& entry : log_history) {
            log_text += entry + "\n";
        }
        
        // Only update if content has actually changed
        wxString current_content = emotag_log_output_text->GetValue();
        if (current_content != log_text) {
            // Preserve scroll position
            long insertion_point = emotag_log_output_text->GetInsertionPoint();
            long scroll_pos = emotag_log_output_text->GetScrollPos(wxVERTICAL);
            
            emotag_log_output_text->SetValue(log_text);
            
            // Restore scroll position if we weren't at the end
            if (insertion_point != current_content.length()) {
                emotag_log_output_text->SetInsertionPoint(insertion_point);
                emotag_log_output_text->SetScrollPos(wxVERTICAL, scroll_pos);
            } else {
                // If we were at the end, stay at the end (auto-scroll)
                emotag_log_output_text->SetInsertionPointEnd();
            }
        }
    }
    
    // Update last generation only if content has changed
    if (emotag_last_generation_text) {
        auto last_gen = plugin->GetLastGeneration();
        std::string gen_text;
        if (last_gen.has_generation) {
            gen_text = "Timestamp: " + last_gen.timestamp + "\n\n";
            gen_text += "Input:\n" + last_gen.input + "\n\n";
            gen_text += "Output:\n" + last_gen.output;
        } else {
            gen_text = "No generation recorded yet.";
        }
        
        // Only update if content has actually changed
        wxString current_content = emotag_last_generation_text->GetValue();
        if (current_content != gen_text) {
            // Preserve scroll position
            long insertion_point = emotag_last_generation_text->GetInsertionPoint();
            long scroll_pos = emotag_last_generation_text->GetScrollPos(wxVERTICAL);
            
            emotag_last_generation_text->SetValue(gen_text);
            
            // Restore scroll position if we weren't at the end
            if (insertion_point != current_content.length()) {
                emotag_last_generation_text->SetInsertionPoint(insertion_point);
                emotag_last_generation_text->SetScrollPos(wxVERTICAL, scroll_pos);
            } else {
                // If we were at the end, stay at the end (auto-scroll)
                emotag_last_generation_text->SetInsertionPointEnd();
            }
        }
    }
}

inline void EmoTagPluginUI::LoadSettings(SettingsManager* settings_manager) {
    if (!settings_manager) return;
    
    if (emotag_model_path_text) {
        std::string model_path = settings_manager->GetString("Models", "emotag_model_path", "");
        emotag_model_path_text->SetValue(model_path);
    }
    
    if (emotag_system_prompt_text) {
        std::string system_prompt = settings_manager->GetString("EmoTag", "system_prompt", 
            "Analyze the emotional tone and state of the following AI responses. Provide a brief emotional analysis focusing on mood, engagement level, and conversational dynamics.");
        emotag_system_prompt_text->SetValue(system_prompt);
    }
    
    if (emotag_window_size_slider && emotag_window_size_text) {
        int window_size = settings_manager->GetInt("EmoTag", "analysis_window_size", 3);
        emotag_window_size_slider->SetValue(window_size);
        UpdateWindowSizeLabel();
    }
    
    if (emotag_include_user_checkbox) {
        bool include_user = settings_manager->GetBool("EmoTag", "include_user_messages", false);
        emotag_include_user_checkbox->SetValue(include_user);
    }
}

inline void EmoTagPluginUI::SaveSettings(SettingsManager* settings_manager) {
    if (!settings_manager) return;
    
    if (emotag_model_path_text) {
        std::string emotag_model_path = emotag_model_path_text->GetValue().ToStdString();
        settings_manager->SetString("Models", "emotag_model_path", emotag_model_path);
    }
    
    if (emotag_system_prompt_text) {
        std::string emotag_prompt = emotag_system_prompt_text->GetValue().ToStdString();
        settings_manager->SetString("EmoTag", "system_prompt", emotag_prompt);
    }
    
    if (emotag_window_size_slider) {
        int window_size = emotag_window_size_slider->GetValue();
        settings_manager->SetInt("EmoTag", "analysis_window_size", window_size);
    }
    
    if (emotag_include_user_checkbox) {
        bool include_user = emotag_include_user_checkbox->GetValue();
        settings_manager->SetBool("EmoTag", "include_user_messages", include_user);
    }
}

inline std::string EmoTagPluginUI::GetModelPath() const {
    if (emotag_model_path_text) {
        return emotag_model_path_text->GetValue().ToStdString();
    }
    return "";
}

inline std::string EmoTagPluginUI::GetSystemPrompt() const {
    if (emotag_system_prompt_text) {
        return emotag_system_prompt_text->GetValue().ToStdString();
    }
    return "";
}

inline int EmoTagPluginUI::GetAnalysisWindowSize() const {
    if (emotag_window_size_slider) {
        return emotag_window_size_slider->GetValue();
    }
    return 3; // Default value
}

inline bool EmoTagPluginUI::GetIncludeUserMessages() const {
    if (emotag_include_user_checkbox) {
        return emotag_include_user_checkbox->GetValue();
    }
    return false; // Default value
}

inline void EmoTagPluginUI::UpdateWindowSizeLabel() {
    if (emotag_window_size_text && emotag_window_size_slider) {
        int value = emotag_window_size_slider->GetValue();
        emotag_window_size_text->SetLabel(wxString::Format("%d response%s", value, value == 1 ? "" : "s"));
    }
}

inline void EmoTagPluginUI::OnBrowseModel(wxCommandEvent& event) {
    wxFileDialog dialog(parent_window, "Choose EmoTag Model File", "", "", 
                       "GGUF files (*.gguf)|*.gguf|All files (*.*)|*.*", 
                       wxFD_OPEN | wxFD_FILE_MUST_EXIST);
    
    if (dialog.ShowModal() == wxID_OK && emotag_model_path_text) {
        emotag_model_path_text->SetValue(dialog.GetPath());
    }
}

inline void EmoTagPluginUI::OnWindowSizeChanged(wxCommandEvent& event) {
    UpdateWindowSizeLabel();
}

inline void EmoTagPluginUI::OnIncludeUserChanged(wxCommandEvent& event) {
    // This could trigger immediate re-analysis or just update the setting
    // For now, we'll just let the setting change take effect on next analysis
}
