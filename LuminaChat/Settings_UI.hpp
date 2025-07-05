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

// Forward declarations
class SettingsManager;
class LlamaManager;
class ContextInfo;

/**
 * Settings UI Manager - Handles all general application settings UI
 * 
 * Responsibilities:
 * - Model configuration UI (path, context size, GPU layers)
 * - Template configuration UI (environment, identity, system prompt)
 * - Settings loading/saving to SettingsManager
 * - Model loading coordination
 * - UI state management and validation
 */
class SettingsUI {
public:
    // Callback types for communication with main frame
    using LogCallback = std::function<void(const std::string&)>;
    using ModelLoadCallback = std::function<void()>;
    using ContextApplyCallback = std::function<void(ContextInfo*, const std::string&)>;

    explicit SettingsUI(wxWindow* parent);
    ~SettingsUI() = default;

    // Panel creation
    wxPanel* CreatePanel();

    // External dependencies injection
    void SetSettingsManager(SettingsManager* settings_manager);
    void SetLlamaManager(LlamaManager* llama_manager);
    void SetCallbacks(LogCallback log_cb, ModelLoadCallback model_load_cb, ContextApplyCallback context_apply_cb);

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
    std::string GetSystemPrompt() const;

    // Model configuration access
    std::string GetModelPath() const;
    int GetContextSize() const;
    int GetGPULayers() const;

private:
    // UI Components - Model Configuration
    wxPanel* settings_panel{nullptr};
    wxScrolledWindow* scrolled_window{nullptr};
    
    // Model path selection
    wxTextCtrl* model_path_text{nullptr};
    wxButton* browse_model_button{nullptr};
    
    // Model configuration sliders
    wxSlider* context_size_slider{nullptr};
    wxStaticText* context_size_label{nullptr};
    wxSlider* gpu_layers_slider{nullptr};
    wxStaticText* gpu_layers_label{nullptr};
    
    // Model loading
    wxButton* load_model_button{nullptr};
    wxGauge* model_progress{nullptr};

    // Template Configuration
    wxTextCtrl* environment_description_text{nullptr};
    wxTextCtrl* identity_directive_text{nullptr};
    wxTextCtrl* system_prompt_text{nullptr};

    // External dependencies
    wxWindow* parent_window{nullptr};
    SettingsManager* settings_manager{nullptr};
    LlamaManager* llama_manager{nullptr};

    // Callbacks
    LogCallback log_callback;
    ModelLoadCallback model_load_callback;
    ContextApplyCallback context_apply_callback;

    // State tracking
    bool model_loaded{false};
    bool model_loading{false};
    std::string current_model_id{"main_model"};

    // UI Creation methods
    void CreateModelConfigurationSection(wxBoxSizer* main_sizer);
    void CreateTemplateConfigurationSection(wxBoxSizer* main_sizer);
    
    // Event handlers
    void OnBrowseModel(wxCommandEvent& event);
    void OnLoadModel(wxCommandEvent& event);
    void OnContextSizeChange(wxCommandEvent& event);
    void OnGPULayersChange(wxCommandEvent& event);
    void OnModelPathChange(wxCommandEvent& event);
    void OnEnvironmentDescriptionFocusLost(wxFocusEvent& event);
    void OnIdentityDirectiveFocusLost(wxFocusEvent& event);
    void OnSystemPromptFocusLost(wxFocusEvent& event);

    // Helper methods
    void LogMessage(const std::string& message);
    void SetupSliderEvents();
    void SetupTextEvents();
    void ValidateModelConfiguration();
    void UpdateSliderLabels();
    
    // Settings helpers
    void LoadModelSettings();
    void LoadTemplateSettings();
    void SaveModelSettings();
    void SaveTemplateSettings();

    // Constants for IDs
    enum {
        ID_BrowseModel = 20001,
        ID_LoadModel = 20002
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
    CreateModelConfigurationSection(main_sizer);
    CreateTemplateConfigurationSection(main_sizer);
    
    // Add stretch spacer
    main_sizer->AddStretchSpacer();

    scrolled_window->SetSizer(main_sizer);

    // Panel layout
    wxBoxSizer* panel_sizer = new wxBoxSizer(wxVERTICAL);
    panel_sizer->Add(scrolled_window, 1, wxEXPAND | wxALL, 5);
    settings_panel->SetSizer(panel_sizer);

    // Setup event handlers
    SetupSliderEvents();
    SetupTextEvents();

    return settings_panel;
}

inline void SettingsUI::CreateModelConfigurationSection(wxBoxSizer* main_sizer) {
    // Model configuration group
    wxStaticBoxSizer* model_box = new wxStaticBoxSizer(wxVERTICAL, scrolled_window, "Model Configuration");
    
    // Model path selection
    wxBoxSizer* path_sizer = new wxBoxSizer(wxHORIZONTAL);
    path_sizer->Add(new wxStaticText(scrolled_window, wxID_ANY, "Model Path:"), 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    model_path_text = new wxTextCtrl(scrolled_window, wxID_ANY);
    path_sizer->Add(model_path_text, 1, wxEXPAND | wxALL, 5);
    browse_model_button = new wxButton(scrolled_window, ID_BrowseModel, "Browse...");
    path_sizer->Add(browse_model_button, 0, wxALL, 5);
    model_box->Add(path_sizer, 0, wxEXPAND);

    // Context size control
    wxBoxSizer* context_sizer = new wxBoxSizer(wxHORIZONTAL);
    context_sizer->Add(new wxStaticText(scrolled_window, wxID_ANY, "Context Size:"), 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    context_size_slider = new wxSlider(scrolled_window, wxID_ANY, 4096, 512, 32768, 
                                      wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL | wxSL_LABELS);
    context_sizer->Add(context_size_slider, 1, wxEXPAND | wxALL, 5);
    context_size_label = new wxStaticText(scrolled_window, wxID_ANY, "4096");
    context_sizer->Add(context_size_label, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    model_box->Add(context_sizer, 0, wxEXPAND);

    // GPU layers control
    wxBoxSizer* gpu_sizer = new wxBoxSizer(wxHORIZONTAL);
    gpu_sizer->Add(new wxStaticText(scrolled_window, wxID_ANY, "GPU Layers:"), 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    gpu_layers_slider = new wxSlider(scrolled_window, wxID_ANY, 999, 0, 999, 
                                    wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL | wxSL_LABELS);
    gpu_sizer->Add(gpu_layers_slider, 1, wxEXPAND | wxALL, 5);
    gpu_layers_label = new wxStaticText(scrolled_window, wxID_ANY, "999");
    gpu_sizer->Add(gpu_layers_label, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    model_box->Add(gpu_sizer, 0, wxEXPAND);

    // Load button and progress
    wxBoxSizer* load_sizer = new wxBoxSizer(wxHORIZONTAL);
    load_model_button = new wxButton(scrolled_window, ID_LoadModel, "Load Model");
    load_sizer->Add(load_model_button, 0, wxALL, 5);
    model_progress = new wxGauge(scrolled_window, wxID_ANY, 100);
    load_sizer->Add(model_progress, 1, wxEXPAND | wxALL, 5);
    model_box->Add(load_sizer, 0, wxEXPAND);

    main_sizer->Add(model_box, 0, wxEXPAND | wxALL, 5);
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

    // System Prompt
    template_box->Add(new wxStaticText(scrolled_window, wxID_ANY, "System Prompt:"), 0, wxALL, 5);
    system_prompt_text = new wxTextCtrl(scrolled_window, wxID_ANY, wxEmptyString,
                                       wxDefaultPosition, wxSize(-1, 120),
                                       wxTE_MULTILINE | wxTE_WORDWRAP);
    system_prompt_text->SetToolTip("Define the system-level instructions and context. This will be used in template variable replacement for system prompt sections.");
    template_box->Add(system_prompt_text, 0, wxEXPAND | wxALL, 5);

    main_sizer->Add(template_box, 0, wxEXPAND | wxALL, 5);
}

inline void SettingsUI::SetupSliderEvents() {
    if (context_size_slider) {
        context_size_slider->Bind(wxEVT_SLIDER, &SettingsUI::OnContextSizeChange, this);
    }
    if (gpu_layers_slider) {
        gpu_layers_slider->Bind(wxEVT_SLIDER, &SettingsUI::OnGPULayersChange, this);
    }
    if (browse_model_button) {
        browse_model_button->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &SettingsUI::OnBrowseModel, this);
    }
    if (load_model_button) {
        load_model_button->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &SettingsUI::OnLoadModel, this);
    }
}

inline void SettingsUI::SetupTextEvents() {
    if (model_path_text) {
        model_path_text->Bind(wxEVT_TEXT, &SettingsUI::OnModelPathChange, this);
    }
    if (environment_description_text) {
        environment_description_text->Bind(wxEVT_KILL_FOCUS, &SettingsUI::OnEnvironmentDescriptionFocusLost, this);
    }
    if (identity_directive_text) {
        identity_directive_text->Bind(wxEVT_KILL_FOCUS, &SettingsUI::OnIdentityDirectiveFocusLost, this);
    }
    if (system_prompt_text) {
        system_prompt_text->Bind(wxEVT_KILL_FOCUS, &SettingsUI::OnSystemPromptFocusLost, this);
    }
}

// External dependencies injection
inline void SettingsUI::SetSettingsManager(SettingsManager* settings_manager) {
    this->settings_manager = settings_manager;
}

inline void SettingsUI::SetLlamaManager(LlamaManager* llama_manager) {
    this->llama_manager = llama_manager;
}

inline void SettingsUI::SetCallbacks(LogCallback log_cb, ModelLoadCallback model_load_cb, ContextApplyCallback context_apply_cb) {
    log_callback = log_cb;
    model_load_callback = model_load_cb;
    context_apply_callback = context_apply_cb;
}

// Settings persistence
inline void SettingsUI::LoadSettings() {
    if (!settings_manager) {
        LogMessage("WARNING: Cannot load settings - SettingsManager not initialized");
        return;
    }

    LogMessage("Loading settings UI configuration...");
    LoadModelSettings();
    LoadTemplateSettings();
    UpdateUI();
    LogMessage("Settings UI loaded successfully");
}

inline void SettingsUI::LoadModelSettings() {
    if (!settings_manager) return;

    // Load model path
    if (model_path_text) {
        std::string model_path = settings_manager->GetString("Models", "main_model_path", "");
        model_path_text->SetValue(model_path);
        if (!model_path.empty()) {
            LogMessage("Loaded model path: " + model_path);
        }
    }

    // Load context size
    if (context_size_slider) {
        int context_size = settings_manager->GetInt("Models", "main_context_size", 4096);
        context_size_slider->SetValue(context_size);
        LogMessage("Loaded context size: " + std::to_string(context_size));
    }

    // Load GPU layers
    if (gpu_layers_slider) {
        int gpu_layers = settings_manager->GetInt("Models", "main_gpu_layers", 999);
        gpu_layers_slider->SetValue(gpu_layers);
        LogMessage("Loaded GPU layers: " + std::to_string(gpu_layers));
    }
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

    // Load system prompt
    if (system_prompt_text) {
        std::string system_prompt = settings_manager->GetString("Templates", "system_prompt", "");
        system_prompt_text->SetValue(system_prompt);
        if (!system_prompt.empty()) {
            LogMessage("Loaded system prompt from settings");
        }
    }
}

inline void SettingsUI::SaveSettings() {
    if (!settings_manager) {
        LogMessage("WARNING: Cannot save settings - SettingsManager not initialized");
        return;
    }

    LogMessage("Saving settings UI configuration...");
    SaveModelSettings();
    SaveTemplateSettings();
    settings_manager->SaveSettings();
    LogMessage("Settings UI saved successfully");
}

inline void SettingsUI::SaveModelSettings() {
    if (!settings_manager) return;

    // Save model path
    if (model_path_text) {
        std::string model_path = model_path_text->GetValue().ToStdString();
        settings_manager->SetString("Models", "main_model_path", model_path);
    }

    // Save context size
    if (context_size_slider) {
        int context_size = context_size_slider->GetValue();
        settings_manager->SetInt("Models", "main_context_size", context_size);
    }

    // Save GPU layers
    if (gpu_layers_slider) {
        int gpu_layers = gpu_layers_slider->GetValue();
        settings_manager->SetInt("Models", "main_gpu_layers", gpu_layers);
    }
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

    // Save system prompt
    if (system_prompt_text) {
        std::string system_prompt = system_prompt_text->GetValue().ToStdString();
        settings_manager->SetString("Templates", "system_prompt", system_prompt);
    }
}

// UI state management
inline void SettingsUI::UpdateUI() {
    UpdateSliderLabels();
    ValidateModelConfiguration();
}

inline void SettingsUI::UpdateSliderLabels() {
    if (context_size_label && context_size_slider) {
        context_size_label->SetLabel(wxString::Format("%d", context_size_slider->GetValue()));
    }
    if (gpu_layers_label && gpu_layers_slider) {
        gpu_layers_label->SetLabel(wxString::Format("%d", gpu_layers_slider->GetValue()));
    }
}

inline void SettingsUI::ValidateModelConfiguration() {
    if (load_model_button && model_path_text) {
        bool has_model_path = !model_path_text->GetValue().IsEmpty();
        
        if (model_loading) {
            // During loading, show as "Loading..." and keep enabled for potential cancellation
            load_model_button->SetLabel("Loading Model...");
            load_model_button->Enable(false); // Disable for now, could add cancellation later
        } else {
            // Normal state - show "Load Model" and enable based on path availability
            load_model_button->SetLabel("Load Model");
            load_model_button->Enable(has_model_path);
        }
    }
}

inline void SettingsUI::UpdateModelProgress(int progress) {
    if (model_progress) {
        model_progress->SetValue(progress);
    }
}

inline void SettingsUI::SetModelLoadingState(bool loading) {
    model_loading = loading;
    UpdateUI();
}

inline void SettingsUI::SetModelLoadedState(bool loaded) {
    model_loaded = loaded;
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

inline std::string SettingsUI::GetSystemPrompt() const {
    if (system_prompt_text) {
        return system_prompt_text->GetValue().ToStdString();
    }
    return "";
}

// Model configuration access
inline std::string SettingsUI::GetModelPath() const {
    if (model_path_text) {
        return model_path_text->GetValue().ToStdString();
    }
    return "";
}

inline int SettingsUI::GetContextSize() const {
    if (context_size_slider) {
        return context_size_slider->GetValue();
    }
    return 4096; // Default fallback
}

inline int SettingsUI::GetGPULayers() const {
    if (gpu_layers_slider) {
        return gpu_layers_slider->GetValue();
    }
    return 999; // Default fallback
}

// Event handlers
inline void SettingsUI::OnBrowseModel(wxCommandEvent& event) {
    wxFileDialog openFileDialog(parent_window, "Choose Model File", "", "",
                               "GGUF files (*.gguf)|*.gguf|All files (*.*)|*.*",
                               wxFD_OPEN | wxFD_FILE_MUST_EXIST);

    if (openFileDialog.ShowModal() == wxID_CANCEL) {
        return;
    }

    wxString path = openFileDialog.GetPath();
    if (model_path_text) {
        model_path_text->SetValue(path);
        OnModelPathChange(event); // Trigger save
    }
    
    LogMessage("Selected model file: " + path.ToStdString());
}

inline void SettingsUI::OnLoadModel(wxCommandEvent& event) {
    if (model_load_callback) {
        model_load_callback();
    }
}

inline void SettingsUI::OnContextSizeChange(wxCommandEvent& event) {
    UpdateUI();
    if (settings_manager && context_size_slider) {
        settings_manager->SetInt("Models", "main_context_size", context_size_slider->GetValue());
        settings_manager->SaveSettings();
    }
}

inline void SettingsUI::OnGPULayersChange(wxCommandEvent& event) {
    UpdateUI();
    if (settings_manager && gpu_layers_slider) {
        settings_manager->SetInt("Models", "main_gpu_layers", gpu_layers_slider->GetValue());
        settings_manager->SaveSettings();
    }
}

inline void SettingsUI::OnModelPathChange(wxCommandEvent& event) {
    if (settings_manager && model_path_text) {
        settings_manager->SetString("Models", "main_model_path", model_path_text->GetValue().ToStdString());
        // Note: Don't auto-save on every keystroke for performance
    }
    UpdateUI();
}

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

inline void SettingsUI::OnSystemPromptFocusLost(wxFocusEvent& event) {
    if (settings_manager && system_prompt_text) {
        settings_manager->SetString("Templates", "system_prompt", system_prompt_text->GetValue().ToStdString());
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
