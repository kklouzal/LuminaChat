#pragma once

#include <wx/wx.h>
#include <wx/panel.h>
#include <wx/textctrl.h>
#include <wx/button.h>
#include <wx/stattext.h>
#include <wx/sizer.h>
#include <wx/scrolwin.h>
#include <wx/filedlg.h>
#include "../SummarizationPlugin.hpp"
#include "../../SettingsManager.hpp"

class SummarizationPluginUI : public wxEvtHandler {
public:
    SummarizationPluginUI(wxWindow* parent);
    ~SummarizationPluginUI() = default;

    // Panel creation
    wxPanel* CreatePanel();
    
    // UI update methods
    void UpdateStatus(const std::string& status, const wxColour& color = wxNullColour);
    void UpdateDebugInfo(LuminaChat::SummarizationPlugin* plugin);
    
    // Settings persistence
    void LoadSettings(SettingsManager* settings_manager);
    void SaveSettings(SettingsManager* settings_manager);
    
    // Accessors for main frame
    std::string GetModelPath() const;
    std::string GetSystemPrompt() const;

private:
    wxWindow* parent_window;
    
    // UI Components
    wxPanel* summary_panel;
    wxTextCtrl* summary_model_path_text;
    wxButton* browse_summary_model_button;
    wxStaticText* summary_status_text;
    wxTextCtrl* summary_log_output_text;
    wxTextCtrl* summary_last_generation_text;
    wxTextCtrl* summary_system_prompt_text;
    
    // Event IDs
    enum {
        ID_BrowseSummaryModel = 3000
    };
    
    // Event handlers
    void OnBrowseModel(wxCommandEvent& event);
    
    // Internal methods
    void CreateModelConfigSection(wxScrolledWindow* parent, wxBoxSizer* sizer);
    void CreateStatusSection(wxScrolledWindow* parent, wxBoxSizer* sizer);
    void CreatePromptSection(wxScrolledWindow* parent, wxBoxSizer* sizer);
};

// Implementation
inline SummarizationPluginUI::SummarizationPluginUI(wxWindow* parent) 
    : parent_window(parent) {
}

inline wxPanel* SummarizationPluginUI::CreatePanel() {
    summary_panel = new wxPanel(parent_window, wxID_ANY);
    
    wxScrolledWindow* scroll = new wxScrolledWindow(summary_panel, wxID_ANY);
    scroll->SetScrollRate(5, 5);
    
    wxBoxSizer* main_sizer = new wxBoxSizer(wxVERTICAL);
    
    // Create sections
    CreateModelConfigSection(scroll, main_sizer);
    CreateStatusSection(scroll, main_sizer);
    CreatePromptSection(scroll, main_sizer);
    
    scroll->SetSizer(main_sizer);
    
    wxBoxSizer* panel_sizer = new wxBoxSizer(wxVERTICAL);
    panel_sizer->Add(scroll, 1, wxEXPAND | wxALL, 5);
    summary_panel->SetSizer(panel_sizer);
    
    return summary_panel;
}

inline void SummarizationPluginUI::CreateModelConfigSection(wxScrolledWindow* parent, wxBoxSizer* sizer) {
    wxStaticBoxSizer* model_box = new wxStaticBoxSizer(wxVERTICAL, parent, "Summary Model Configuration");
    
    // Model path
    wxBoxSizer* path_sizer = new wxBoxSizer(wxHORIZONTAL);
    path_sizer->Add(new wxStaticText(parent, wxID_ANY, "Model Path:"), 0, wxALIGN_CENTER_VERTICAL | wxRIGHT, 5);
    
    summary_model_path_text = new wxTextCtrl(parent, wxID_ANY, "", wxDefaultPosition, wxSize(400, -1));
    path_sizer->Add(summary_model_path_text, 1, wxEXPAND | wxRIGHT, 5);
    
    browse_summary_model_button = new wxButton(parent, ID_BrowseSummaryModel, "Browse...");
    browse_summary_model_button->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &SummarizationPluginUI::OnBrowseModel, this);
    path_sizer->Add(browse_summary_model_button, 0);
    
    model_box->Add(path_sizer, 0, wxEXPAND | wxALL, 5);
    
    sizer->Add(model_box, 0, wxEXPAND | wxALL, 5);
}

inline void SummarizationPluginUI::CreateStatusSection(wxScrolledWindow* parent, wxBoxSizer* sizer) {
    wxStaticBoxSizer* status_box = new wxStaticBoxSizer(wxVERTICAL, parent, "Status & Debug Information");
    
    // Status text
    summary_status_text = new wxStaticText(parent, wxID_ANY, "Not initialized");
    status_box->Add(summary_status_text, 0, wxEXPAND | wxALL, 5);
    
    // Log output
    status_box->Add(new wxStaticText(parent, wxID_ANY, "Log Output:"), 0, wxEXPAND | wxTOP, 5);
    summary_log_output_text = new wxTextCtrl(parent, wxID_ANY, "", wxDefaultPosition, wxSize(-1, 100), 
                                           wxTE_MULTILINE | wxTE_READONLY);
    summary_log_output_text->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
    status_box->Add(summary_log_output_text, 0, wxEXPAND | wxALL, 5);
    
    // Last generation
    status_box->Add(new wxStaticText(parent, wxID_ANY, "Last Generation:"), 0, wxEXPAND | wxTOP, 5);
    summary_last_generation_text = new wxTextCtrl(parent, wxID_ANY, "", wxDefaultPosition, wxSize(-1, 100), 
                                                 wxTE_MULTILINE | wxTE_READONLY);
    summary_last_generation_text->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
    status_box->Add(summary_last_generation_text, 0, wxEXPAND | wxALL, 5);
    
    sizer->Add(status_box, 1, wxEXPAND | wxALL, 5);
}

inline void SummarizationPluginUI::CreatePromptSection(wxScrolledWindow* parent, wxBoxSizer* sizer) {
    wxStaticBoxSizer* prompt_box = new wxStaticBoxSizer(wxVERTICAL, parent, "System Prompt");
    
    summary_system_prompt_text = new wxTextCtrl(parent, wxID_ANY, 
        "You are a conversation summarizer. Summarize the following conversation while preserving key context and emotional tone.",
        wxDefaultPosition, wxSize(-1, 100), wxTE_MULTILINE);
    
    prompt_box->Add(summary_system_prompt_text, 1, wxEXPAND | wxALL, 5);
    sizer->Add(prompt_box, 0, wxEXPAND | wxALL, 5);
}

inline void SummarizationPluginUI::UpdateStatus(const std::string& status, const wxColour& color) {
    if (summary_status_text) {
        summary_status_text->SetLabel(status);
        if (color.IsOk()) {
            summary_status_text->SetForegroundColour(color);
        }
        summary_status_text->GetParent()->Layout();
    }
}

inline void SummarizationPluginUI::UpdateDebugInfo(LuminaChat::SummarizationPlugin* plugin) {
    if (!plugin) return;
    
    // Update log output only if content has changed
    if (summary_log_output_text) {
        auto log_history = plugin->GetLogHistory();
        std::string log_text;
        for (const auto& entry : log_history) {
            log_text += entry + "\n";
        }
        
        // Only update if content has actually changed
        wxString current_content = summary_log_output_text->GetValue();
        wxString new_log_text = wxString::FromUTF8(log_text);
        if (current_content != new_log_text) {
            // Preserve scroll position
            long insertion_point = summary_log_output_text->GetInsertionPoint();
            long scroll_pos = summary_log_output_text->GetScrollPos(wxVERTICAL);
            
            summary_log_output_text->SetValue(new_log_text);
            
            // Restore scroll position if we weren't at the end
            if (insertion_point != current_content.length()) {
                summary_log_output_text->SetInsertionPoint(insertion_point);
                summary_log_output_text->SetScrollPos(wxVERTICAL, scroll_pos);
            } else {
                // If we were at the end, stay at the end (auto-scroll)
                summary_log_output_text->SetInsertionPointEnd();
            }
        }
    }
    
    // Update last generation only if content has changed
    if (summary_last_generation_text) {
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
        wxString current_content = summary_last_generation_text->GetValue();
        wxString new_gen_text = wxString::FromUTF8(gen_text);
        if (current_content != new_gen_text) {
            // Preserve scroll position
            long insertion_point = summary_last_generation_text->GetInsertionPoint();
            long scroll_pos = summary_last_generation_text->GetScrollPos(wxVERTICAL);
            
            summary_last_generation_text->SetValue(new_gen_text);
            
            // Restore scroll position if we weren't at the end
            if (insertion_point != current_content.length()) {
                summary_last_generation_text->SetInsertionPoint(insertion_point);
                summary_last_generation_text->SetScrollPos(wxVERTICAL, scroll_pos);
            } else {
                // If we were at the end, stay at the end (auto-scroll)
                summary_last_generation_text->SetInsertionPointEnd();
            }
        }
    }
}

inline void SummarizationPluginUI::LoadSettings(SettingsManager* settings_manager) {
    if (!settings_manager) return;
    
    if (summary_model_path_text) {
        std::string model_path = settings_manager->GetString("Models", "summary_model_path", "");
        summary_model_path_text->SetValue(model_path);
    }
    
    if (summary_system_prompt_text) {
        std::string system_prompt = settings_manager->GetString("Summary", "system_prompt", 
            "You are a conversation summarizer. Summarize the following conversation while preserving key context and emotional tone.");
        summary_system_prompt_text->SetValue(system_prompt);
    }
}

inline void SummarizationPluginUI::SaveSettings(SettingsManager* settings_manager) {
    if (!settings_manager) return;
    
    if (summary_model_path_text) {
        std::string summary_model_path = summary_model_path_text->GetValue().ToStdString();
        settings_manager->SetString("Models", "summary_model_path", summary_model_path);
    }
    
    if (summary_system_prompt_text) {
        std::string summary_prompt = summary_system_prompt_text->GetValue().ToStdString();
        settings_manager->SetString("Summary", "system_prompt", summary_prompt);
    }
}

inline std::string SummarizationPluginUI::GetModelPath() const {
    if (summary_model_path_text) {
        return summary_model_path_text->GetValue().ToStdString();
    }
    return "";
}

inline std::string SummarizationPluginUI::GetSystemPrompt() const {
    if (summary_system_prompt_text) {
        return summary_system_prompt_text->GetValue().ToStdString();
    }
    return "";
}

inline void SummarizationPluginUI::OnBrowseModel(wxCommandEvent& event) {
    wxFileDialog dialog(parent_window, "Choose Summary Model File", "", "", 
                       "GGUF files (*.gguf)|*.gguf|All files (*.*)|*.*", 
                       wxFD_OPEN | wxFD_FILE_MUST_EXIST);
    
    if (dialog.ShowModal() == wxID_OK && summary_model_path_text) {
        summary_model_path_text->SetValue(dialog.GetPath());
    }
}
