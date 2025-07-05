#pragma once

#include <wx/wx.h>
#include <wx/panel.h>
#include <wx/textctrl.h>
#include <wx/slider.h>
#include <wx/stattext.h>
#include <wx/checkbox.h>
#include <wx/sizer.h>
#include <wx/scrolwin.h>
#include "../ContextPruningPlugin.hpp"
#include "../../SettingsManager.hpp"

class ContextPruningPluginUI : public wxEvtHandler {
public:
    ContextPruningPluginUI(wxWindow* parent);
    ~ContextPruningPluginUI() = default;

    // Panel creation
    wxPanel* CreatePanel();
    
    // UI update methods
    void UpdateStatus(const std::string& status, const wxColour& color = wxNullColour);
    void UpdateDebugInfo(LuminaChat::ContextPruningPlugin* plugin);
    
    // Settings persistence
    void LoadSettings(SettingsManager* settings_manager);
    void SaveSettings(SettingsManager* settings_manager);
    
    // Accessors for configuration
    float GetPruningThreshold() const;
    float GetTargetUsage() const;
    float GetEmergencyThreshold() const;
    bool GetAutoTriggerEnabled() const;

private:
    wxWindow* parent_window;
    
    // UI Components
    wxPanel* context_pruning_panel;
    wxSlider* pruning_threshold_slider;
    wxStaticText* pruning_threshold_text;
    wxSlider* target_usage_slider;
    wxStaticText* target_usage_text;
    wxSlider* emergency_threshold_slider;
    wxStaticText* emergency_threshold_text;
    wxCheckBox* auto_trigger_checkbox;
    wxStaticText* status_text;
    wxTextCtrl* activity_log_text;
    wxTextCtrl* statistics_text;
    
    // Event IDs
    enum {
        ID_PruningThresholdSlider = 3200,
        ID_TargetUsageSlider,
        ID_EmergencyThresholdSlider,
        ID_AutoTriggerCheckbox
    };
    
    // Event handlers
    void OnPruningThresholdChanged(wxCommandEvent& event);
    void OnTargetUsageChanged(wxCommandEvent& event);
    void OnEmergencyThresholdChanged(wxCommandEvent& event);
    void OnAutoTriggerChanged(wxCommandEvent& event);
    
    // Internal methods
    void CreateThresholdSection(wxScrolledWindow* parent, wxBoxSizer* sizer);
    void CreateStatusSection(wxScrolledWindow* parent, wxBoxSizer* sizer);
    void UpdateSliderLabels();
};

// Implementation
inline ContextPruningPluginUI::ContextPruningPluginUI(wxWindow* parent) 
    : parent_window(parent) {
}

inline wxPanel* ContextPruningPluginUI::CreatePanel() {
    context_pruning_panel = new wxPanel(parent_window, wxID_ANY);
    
    wxScrolledWindow* scroll = new wxScrolledWindow(context_pruning_panel, wxID_ANY);
    scroll->SetScrollRate(5, 5);
    
    wxBoxSizer* main_sizer = new wxBoxSizer(wxVERTICAL);
    
    // Create sections
    CreateThresholdSection(scroll, main_sizer);
    CreateStatusSection(scroll, main_sizer);
    
    scroll->SetSizer(main_sizer);
    
    wxBoxSizer* panel_sizer = new wxBoxSizer(wxVERTICAL);
    panel_sizer->Add(scroll, 1, wxEXPAND | wxALL, 5);
    context_pruning_panel->SetSizer(panel_sizer);
    
    return context_pruning_panel;
}

inline void ContextPruningPluginUI::CreateThresholdSection(wxScrolledWindow* parent, wxBoxSizer* sizer) {
    wxStaticBoxSizer* threshold_box = new wxStaticBoxSizer(wxVERTICAL, parent, "Pruning Thresholds & Configuration");
    
    // Pruning threshold (when to start pruning)
    wxBoxSizer* pruning_sizer = new wxBoxSizer(wxHORIZONTAL);
    pruning_sizer->Add(new wxStaticText(parent, wxID_ANY, "Pruning Threshold:"), 0, wxALIGN_CENTER_VERTICAL | wxRIGHT, 5);
    
    pruning_threshold_slider = new wxSlider(parent, ID_PruningThresholdSlider, 75, 50, 95, 
                                          wxDefaultPosition, wxSize(200, -1));
    pruning_threshold_slider->Bind(wxEVT_COMMAND_SLIDER_UPDATED, &ContextPruningPluginUI::OnPruningThresholdChanged, this);
    pruning_sizer->Add(pruning_threshold_slider, 1, wxEXPAND | wxRIGHT, 10);
    
    pruning_threshold_text = new wxStaticText(parent, wxID_ANY, "75%");
    pruning_sizer->Add(pruning_threshold_text, 0, wxALIGN_CENTER_VERTICAL);
    
    threshold_box->Add(pruning_sizer, 0, wxEXPAND | wxALL, 5);
    
    // Target usage (how much to reduce to)
    wxBoxSizer* target_sizer = new wxBoxSizer(wxHORIZONTAL);
    target_sizer->Add(new wxStaticText(parent, wxID_ANY, "Target Usage:"), 0, wxALIGN_CENTER_VERTICAL | wxRIGHT, 5);
    
    target_usage_slider = new wxSlider(parent, ID_TargetUsageSlider, 40, 20, 70, 
                                     wxDefaultPosition, wxSize(200, -1));
    target_usage_slider->Bind(wxEVT_COMMAND_SLIDER_UPDATED, &ContextPruningPluginUI::OnTargetUsageChanged, this);
    target_sizer->Add(target_usage_slider, 1, wxEXPAND | wxRIGHT, 10);
    
    target_usage_text = new wxStaticText(parent, wxID_ANY, "40%");
    target_sizer->Add(target_usage_text, 0, wxALIGN_CENTER_VERTICAL);
    
    threshold_box->Add(target_sizer, 0, wxEXPAND | wxALL, 5);
    
    // Emergency threshold (immediate pruning)
    wxBoxSizer* emergency_sizer = new wxBoxSizer(wxHORIZONTAL);
    emergency_sizer->Add(new wxStaticText(parent, wxID_ANY, "Emergency Threshold:"), 0, wxALIGN_CENTER_VERTICAL | wxRIGHT, 5);
    
    emergency_threshold_slider = new wxSlider(parent, ID_EmergencyThresholdSlider, 90, 80, 98, 
                                            wxDefaultPosition, wxSize(200, -1));
    emergency_threshold_slider->Bind(wxEVT_COMMAND_SLIDER_UPDATED, &ContextPruningPluginUI::OnEmergencyThresholdChanged, this);
    emergency_sizer->Add(emergency_threshold_slider, 1, wxEXPAND | wxRIGHT, 10);
    
    emergency_threshold_text = new wxStaticText(parent, wxID_ANY, "90%");
    emergency_sizer->Add(emergency_threshold_text, 0, wxALIGN_CENTER_VERTICAL);
    
    threshold_box->Add(emergency_sizer, 0, wxEXPAND | wxALL, 5);
    
    // Auto-trigger checkbox
    auto_trigger_checkbox = new wxCheckBox(parent, ID_AutoTriggerCheckbox, "Enable automatic pruning");
    auto_trigger_checkbox->Bind(wxEVT_COMMAND_CHECKBOX_CLICKED, &ContextPruningPluginUI::OnAutoTriggerChanged, this);
    auto_trigger_checkbox->SetValue(true);
    threshold_box->Add(auto_trigger_checkbox, 0, wxEXPAND | wxALL, 5);
    
    // Help text
    wxStaticText* help_text = new wxStaticText(parent, wxID_ANY, 
        "Pruning Threshold: Context usage % at which pruning begins\n"
        "Target Usage: Reduce context to this % after pruning\n"
        "Emergency Threshold: Immediate pruning at this %");
    help_text->SetFont(wxFont(8, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_ITALIC, wxFONTWEIGHT_NORMAL));
    threshold_box->Add(help_text, 0, wxEXPAND | wxALL, 5);
    
    sizer->Add(threshold_box, 0, wxEXPAND | wxALL, 5);
}

inline void ContextPruningPluginUI::CreateStatusSection(wxScrolledWindow* parent, wxBoxSizer* sizer) {
    wxStaticBoxSizer* status_box = new wxStaticBoxSizer(wxVERTICAL, parent, "Status & Activity");
    
    // Status text
    status_text = new wxStaticText(parent, wxID_ANY, "Plugin not initialized");
    status_box->Add(status_text, 0, wxEXPAND | wxALL, 5);
    
    // Statistics display
    status_box->Add(new wxStaticText(parent, wxID_ANY, "Plugin Statistics:"), 0, wxEXPAND | wxTOP, 5);
    statistics_text = new wxTextCtrl(parent, wxID_ANY, "", wxDefaultPosition, wxSize(-1, 80), 
                                   wxTE_MULTILINE | wxTE_READONLY);
    statistics_text->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
    status_box->Add(statistics_text, 0, wxEXPAND | wxALL, 5);
    
    // Activity log
    status_box->Add(new wxStaticText(parent, wxID_ANY, "Activity Log:"), 0, wxEXPAND | wxTOP, 5);
    activity_log_text = new wxTextCtrl(parent, wxID_ANY, "", wxDefaultPosition, wxSize(-1, 150), 
                                     wxTE_MULTILINE | wxTE_READONLY);
    activity_log_text->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
    status_box->Add(activity_log_text, 1, wxEXPAND | wxALL, 5);
    
    sizer->Add(status_box, 1, wxEXPAND | wxALL, 5);
}

inline void ContextPruningPluginUI::UpdateStatus(const std::string& status, const wxColour& color) {
    if (status_text) {
        status_text->SetLabel(status);
        if (color.IsOk()) {
            status_text->SetForegroundColour(color);
        }
        status_text->GetParent()->Layout();
    }
}

inline void ContextPruningPluginUI::UpdateDebugInfo(LuminaChat::ContextPruningPlugin* plugin) {
    if (!plugin) return;
    
    // Update statistics
    if (statistics_text) {
        auto stats = plugin->GetStats();
        std::string stats_text = wxString::Format(
            "Contexts Monitored: %zu\n"
            "Pruning Requests: %zu\n"
            "Emergency Prunings: %zu\n"
            "Plugin Ready: %s",
            stats.contexts_currently_monitored,
            stats.pruning_requests_handled,
            stats.emergency_prunings_triggered,
            stats.plugin_ready ? "Yes" : "No"
        ).ToStdString();
        statistics_text->SetValue(stats_text);
    }
    
    // Note: The plugin doesn't have a GetLogHistory method like the other plugins,
    // so we'll keep the activity log simple for now. This could be extended
    // if logging is added to the ContextPruningPlugin.
}

inline void ContextPruningPluginUI::LoadSettings(SettingsManager* settings_manager) {
    if (!settings_manager) return;
    
    if (pruning_threshold_slider && pruning_threshold_text) {
        int threshold = static_cast<int>(settings_manager->GetFloat("ContextPruning", "pruning_threshold", 0.75f) * 100);
        pruning_threshold_slider->SetValue(threshold);
        UpdateSliderLabels();
    }
    
    if (target_usage_slider && target_usage_text) {
        int target = static_cast<int>(settings_manager->GetFloat("ContextPruning", "target_usage", 0.40f) * 100);
        target_usage_slider->SetValue(target);
        UpdateSliderLabels();
    }
    
    if (emergency_threshold_slider && emergency_threshold_text) {
        int emergency = static_cast<int>(settings_manager->GetFloat("ContextPruning", "emergency_threshold", 0.90f) * 100);
        emergency_threshold_slider->SetValue(emergency);
        UpdateSliderLabels();
    }
    
    if (auto_trigger_checkbox) {
        bool auto_trigger = settings_manager->GetBool("ContextPruning", "auto_trigger", true);
        auto_trigger_checkbox->SetValue(auto_trigger);
    }
}

inline void ContextPruningPluginUI::SaveSettings(SettingsManager* settings_manager) {
    if (!settings_manager) return;
    
    if (pruning_threshold_slider) {
        float threshold = static_cast<float>(pruning_threshold_slider->GetValue()) / 100.0f;
        settings_manager->SetFloat("ContextPruning", "pruning_threshold", threshold);
    }
    
    if (target_usage_slider) {
        float target = static_cast<float>(target_usage_slider->GetValue()) / 100.0f;
        settings_manager->SetFloat("ContextPruning", "target_usage", target);
    }
    
    if (emergency_threshold_slider) {
        float emergency = static_cast<float>(emergency_threshold_slider->GetValue()) / 100.0f;
        settings_manager->SetFloat("ContextPruning", "emergency_threshold", emergency);
    }
    
    if (auto_trigger_checkbox) {
        bool auto_trigger = auto_trigger_checkbox->GetValue();
        settings_manager->SetBool("ContextPruning", "auto_trigger", auto_trigger);
    }
}

inline float ContextPruningPluginUI::GetPruningThreshold() const {
    if (pruning_threshold_slider) {
        return static_cast<float>(pruning_threshold_slider->GetValue()) / 100.0f;
    }
    return 0.75f; // Default value
}

inline float ContextPruningPluginUI::GetTargetUsage() const {
    if (target_usage_slider) {
        return static_cast<float>(target_usage_slider->GetValue()) / 100.0f;
    }
    return 0.40f; // Default value
}

inline float ContextPruningPluginUI::GetEmergencyThreshold() const {
    if (emergency_threshold_slider) {
        return static_cast<float>(emergency_threshold_slider->GetValue()) / 100.0f;
    }
    return 0.90f; // Default value
}

inline bool ContextPruningPluginUI::GetAutoTriggerEnabled() const {
    if (auto_trigger_checkbox) {
        return auto_trigger_checkbox->GetValue();
    }
    return true; // Default value
}

inline void ContextPruningPluginUI::UpdateSliderLabels() {
    if (pruning_threshold_text && pruning_threshold_slider) {
        pruning_threshold_text->SetLabel(wxString::Format("%d%%", pruning_threshold_slider->GetValue()));
    }
    
    if (target_usage_text && target_usage_slider) {
        target_usage_text->SetLabel(wxString::Format("%d%%", target_usage_slider->GetValue()));
    }
    
    if (emergency_threshold_text && emergency_threshold_slider) {
        emergency_threshold_text->SetLabel(wxString::Format("%d%%", emergency_threshold_slider->GetValue()));
    }
}

inline void ContextPruningPluginUI::OnPruningThresholdChanged(wxCommandEvent& event) {
    UpdateSliderLabels();
}

inline void ContextPruningPluginUI::OnTargetUsageChanged(wxCommandEvent& event) {
    UpdateSliderLabels();
}

inline void ContextPruningPluginUI::OnEmergencyThresholdChanged(wxCommandEvent& event) {
    UpdateSliderLabels();
}

inline void ContextPruningPluginUI::OnAutoTriggerChanged(wxCommandEvent& event) {
    // This could trigger immediate configuration update
    // For now, we'll just let the setting change take effect
}
