#pragma once

#include <wx/wx.h>
#include <wx/notebook.h>
#include <wx/textctrl.h>
#include <wx/button.h>
#include <wx/slider.h>
#include <wx/gauge.h>
#include <wx/stattext.h>
#include <wx/sizer.h>
#include <wx/scrolwin.h>
#include <wx/filedlg.h>
#include <memory>
#include <functional>
#include "../SettingsManager.hpp"

// Forward declarations
class LlamaManager;
class ContextInfo;

/**
 * Settings UI Manager - Handles general application settings UI
 * 
 * Responsibilities:
 * - Central model loading coordination (reads from individual voice tabs)
 * - Template configuration UI (environment, identity, system prompt)
 * - Settings loading/saving to SettingsManager for general settings
 * - Application startup and model loading orchestration
 * - UI state management and validation
 */
class SettingsUI {
public:
    // Callback types for communication with main frame
    using LogCallback = std::function<void(const std::string&)>;
    using LoadModelsCallback = std::function<void()>;
    using ContextApplyCallback = std::function<void(ContextInfo*, const std::string&)>;

    explicit SettingsUI(wxWindow* parent);
    ~SettingsUI() = default;

    // Panel creation
    wxPanel* CreatePanel();

    // External dependencies injection
    void SetSettingsManager(SettingsManager* settings_manager);
    void SetLlamaManager(LlamaManager* llama_manager);
    void SetCallbacks(LogCallback log_cb, LoadModelsCallback load_models_cb, ContextApplyCallback context_apply_cb);

    // Settings persistence
    void LoadSettings();
    void SaveSettings();

    // UI state management
    void UpdateUI();
    void UpdateModelProgress(int progress);
    void SetModelLoadingState(bool loading);
    void SetModelLoadedState(bool loaded);

    // Template settings access for context application
    std::string GetEnvironmentDescription() const;
    std::string GetIdentityDirective() const;

private:
    // UI Components - General Settings
    wxPanel* settings_panel{nullptr};
    wxScrolledWindow* scrolled_window{nullptr};

    // Central Model Loading
    wxButton* load_models_button{nullptr};
    wxGauge* model_progress{nullptr};
    wxStaticText* status_text{nullptr};

    // Template Configuration
    wxTextCtrl* environment_description_text{nullptr};
    wxTextCtrl* identity_directive_text{nullptr};

    // External dependencies
    wxWindow* parent_window{nullptr};
    SettingsManager* settings_manager{nullptr};
    LlamaManager* llama_manager{nullptr};

    // Callbacks
    LogCallback log_callback;
    LoadModelsCallback load_models_callback;
    ContextApplyCallback context_apply_callback;

    // State tracking
    bool models_loaded{false};
    bool models_loading{false};

    // UI Creation methods
    void CreateModelLoadingSection(wxBoxSizer* main_sizer);
    void CreateTemplateConfigurationSection(wxBoxSizer* main_sizer);
    
    // Event handlers
    void OnLoadModels(wxCommandEvent& event);
    void OnEnvironmentDescriptionFocusLost(wxFocusEvent& event);
    void OnIdentityDirectiveFocusLost(wxFocusEvent& event);

    // Helper methods
    void LogMessage(const std::string& message);
    void SetupEvents();
    
    // Settings helpers
    void LoadTemplateSettings();
    void SaveTemplateSettings();

    // Constants for IDs
    enum {
        ID_LoadModels = 20001
    };
};

/**
 * Settings UI Implementation
 */

inline SettingsUI::SettingsUI(wxWindow* parent) 
    : parent_window(parent) {
}

inline wxPanel* SettingsUI::CreatePanel() {
    if (settings_panel) {
        return settings_panel;
    }

    settings_panel = new wxPanel(parent_window);
    
    // Create scrolled window for settings content
    scrolled_window = new wxScrolledWindow(settings_panel, wxID_ANY, 
                                          wxDefaultPosition, wxDefaultSize, 
                                          wxVSCROLL | wxHSCROLL);
    scrolled_window->SetScrollRate(10, 10);

    // Main content sizer
    wxBoxSizer* main_sizer = new wxBoxSizer(wxVERTICAL);

    // Create configuration sections
    CreateModelLoadingSection(main_sizer);
    CreateTemplateConfigurationSection(main_sizer);
    
    // Add stretch spacer
    main_sizer->AddStretchSpacer();

    scrolled_window->SetSizer(main_sizer);

    // Panel layout
    wxBoxSizer* panel_sizer = new wxBoxSizer(wxVERTICAL);
    panel_sizer->Add(scrolled_window, 1, wxEXPAND | wxALL, 5);
    settings_panel->SetSizer(panel_sizer);

    // Setup event handlers
    SetupEvents();

    return settings_panel;
}

inline void SettingsUI::CreateModelLoadingSection(wxBoxSizer* main_sizer) {
    // Central model loading group
    wxStaticBoxSizer* loading_box = new wxStaticBoxSizer(wxVERTICAL, scrolled_window, "Model Loading");
    
    // Status text
    status_text = new wxStaticText(scrolled_window, wxID_ANY, "Ready to load models from individual voice configurations");
    loading_box->Add(status_text, 0, wxEXPAND | wxALL, 5);
    
    // Load button and progress
    wxBoxSizer* load_sizer = new wxBoxSizer(wxHORIZONTAL);
    load_models_button = new wxButton(scrolled_window, ID_LoadModels, "Load All Models");
    load_models_button->SetToolTip("Load all models based on configurations from Outer Voice and Inner Voice tabs");
    load_sizer->Add(load_models_button, 0, wxALL, 5);
    model_progress = new wxGauge(scrolled_window, wxID_ANY, 100);
    load_sizer->Add(model_progress, 1, wxEXPAND | wxALL, 5);
    loading_box->Add(load_sizer, 0, wxEXPAND);

    main_sizer->Add(loading_box, 0, wxEXPAND | wxALL, 5);
}

inline void SettingsUI::CreateTemplateConfigurationSection(wxBoxSizer* main_sizer) {
    // Template configuration group
    wxStaticBoxSizer* template_box = new wxStaticBoxSizer(wxVERTICAL, scrolled_window, "Template Configuration");
    
    // Environment Description
    template_box->Add(new wxStaticText(scrolled_window, wxID_ANY, "Environment Description:"), 0, wxALL, 5);
    environment_description_text = new wxTextCtrl(scrolled_window, wxID_ANY, wxEmptyString,
                                                 wxDefaultPosition, wxSize(-1, 120),
                                                 wxTE_MULTILINE | wxTE_WORDWRAP);
    environment_description_text->SetToolTip("Define the environment and setting where the conversation takes place. This will be used in template variable replacement for environment-related sections.");
    template_box->Add(environment_description_text, 0, wxEXPAND | wxALL, 5);

    // Identity Directive
    template_box->Add(new wxStaticText(scrolled_window, wxID_ANY, "Identity Directive:"), 0, wxALL, 5);
    identity_directive_text = new wxTextCtrl(scrolled_window, wxID_ANY, wxEmptyString,
                                           wxDefaultPosition, wxSize(-1, 120),
                                           wxTE_MULTILINE | wxTE_WORDWRAP);
    identity_directive_text->SetToolTip("Define the AI's core identity and behavioral guidelines. This will be used in template variable replacement for identity-related sections.");
    template_box->Add(identity_directive_text, 0, wxEXPAND | wxALL, 5);

    main_sizer->Add(template_box, 0, wxEXPAND | wxALL, 5);
}

inline void SettingsUI::SetupEvents() {
    if (load_models_button) {
        load_models_button->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &SettingsUI::OnLoadModels, this);
    }
    if (environment_description_text) {
        environment_description_text->Bind(wxEVT_KILL_FOCUS, &SettingsUI::OnEnvironmentDescriptionFocusLost, this);
    }
    if (identity_directive_text) {
        identity_directive_text->Bind(wxEVT_KILL_FOCUS, &SettingsUI::OnIdentityDirectiveFocusLost, this);
    }
}

// External dependencies injection
inline void SettingsUI::SetSettingsManager(SettingsManager* settings_manager) {
    this->settings_manager = settings_manager;
}

inline void SettingsUI::SetLlamaManager(LlamaManager* llama_manager) {
    this->llama_manager = llama_manager;
}

inline void SettingsUI::SetCallbacks(LogCallback log_cb, LoadModelsCallback load_models_cb, ContextApplyCallback context_apply_cb) {
    log_callback = log_cb;
    load_models_callback = load_models_cb;
    context_apply_callback = context_apply_cb;
}

// Settings persistence
inline void SettingsUI::LoadSettings() {
    if (!settings_manager) {
        LogMessage("WARNING: Cannot load settings - SettingsManager not initialized");
        return;
    }

    LogMessage("Loading settings UI configuration...");
    LoadTemplateSettings();
    UpdateUI();
    LogMessage("Settings UI loaded successfully");
}

inline void SettingsUI::LoadTemplateSettings() {
    if (!settings_manager) return;

    // Load environment description
    if (environment_description_text) {
        std::string env_desc = settings_manager->GetString("Templates", "environment_description", "");
        environment_description_text->SetValue(env_desc);
        if (!env_desc.empty()) {
            LogMessage("Loaded environment description from settings");
        }
    }

    // Load identity directive
    if (identity_directive_text) {
        std::string identity = settings_manager->GetString("Templates", "identity_directive", "");
        identity_directive_text->SetValue(identity);
        if (!identity.empty()) {
            LogMessage("Loaded identity directive from settings");
        }
    }
}

inline void SettingsUI::SaveSettings() {
    if (!settings_manager) {
        LogMessage("WARNING: Cannot save settings - SettingsManager not initialized");
        return;
    }

    LogMessage("Saving settings UI configuration...");
    SaveTemplateSettings();
    settings_manager->SaveSettings();
    LogMessage("Settings UI saved successfully");
}

inline void SettingsUI::SaveTemplateSettings() {
    if (!settings_manager) return;

    // Save environment description
    if (environment_description_text) {
        std::string env_desc = environment_description_text->GetValue().ToStdString();
        settings_manager->SetString("Templates", "environment_description", env_desc);
    }

    // Save identity directive
    if (identity_directive_text) {
        std::string identity = identity_directive_text->GetValue().ToStdString();
        settings_manager->SetString("Templates", "identity_directive", identity);
    }
}

// UI state management
inline void SettingsUI::UpdateUI() {
    // Update button state based on loading status
    if (load_models_button) {
        if (models_loading) {
            load_models_button->SetLabel("Loading Models...");
            load_models_button->Enable(false);
        } else if (models_loaded) {
            load_models_button->SetLabel("Models Loaded");
            load_models_button->Enable(false);
        } else {
            load_models_button->SetLabel("Load All Models");
            load_models_button->Enable(true);
        }
    }
    
    // Update status text
    if (status_text) {
        if (models_loading) {
            status_text->SetLabel("Loading models from voice configurations...");
        } else if (models_loaded) {
            status_text->SetLabel("All models loaded successfully");
        } else {
            status_text->SetLabel("Ready to load models from individual voice configurations");
        }
    }
}

inline void SettingsUI::UpdateModelProgress(int progress) {
    if (model_progress) {
        model_progress->SetValue(progress);
    }
}

inline void SettingsUI::SetModelLoadingState(bool loading) {
    models_loading = loading;
    UpdateUI();
}

inline void SettingsUI::SetModelLoadedState(bool loaded) {
    models_loaded = loaded;
    UpdateUI();
}

// Template settings access
inline std::string SettingsUI::GetEnvironmentDescription() const {
    if (environment_description_text) {
        return environment_description_text->GetValue().ToStdString();
    }
    return "";
}

inline std::string SettingsUI::GetIdentityDirective() const {
    if (identity_directive_text) {
        return identity_directive_text->GetValue().ToStdString();
    }
    return "";
}

// Event handlers
inline void SettingsUI::OnLoadModels(wxCommandEvent& event) {
    if (load_models_callback) {
        LogMessage("Initiating load of all models from voice configurations...");
        load_models_callback();
    }
}

// Event handlers
inline void SettingsUI::OnEnvironmentDescriptionFocusLost(wxFocusEvent& event) {
    if (settings_manager && environment_description_text) {
        settings_manager->SetString("Templates", "environment_description", environment_description_text->GetValue().ToStdString());
        settings_manager->SaveSettings();
    }
    event.Skip();
}

inline void SettingsUI::OnIdentityDirectiveFocusLost(wxFocusEvent& event) {
    if (settings_manager && identity_directive_text) {
        settings_manager->SetString("Templates", "identity_directive", identity_directive_text->GetValue().ToStdString());
        settings_manager->SaveSettings();
    }
    event.Skip();
}

// Helper methods
inline void SettingsUI::LogMessage(const std::string& message) {
    if (log_callback) {
        log_callback(message);
    }
}
