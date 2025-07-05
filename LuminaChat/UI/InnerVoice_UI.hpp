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
 * Inner Voice Settings UI Manager - Handles inner voice model configuration
 * 
 * Responsibilities:
 * - Inner Voice configuration UI (path, context size, GPU layers)
 * - Settings loading/saving to SettingsManager for inner voice
 * - UI state management and validation
 * 
 * Note: Model loading is now handled centrally by the General Settings tab
 */
class InnerVoiceUI {
public:
    // Callback types for communication with main frame
    using LogCallback = std::function<void(const std::string&)>;
    using ContextApplyCallback = std::function<void(ContextInfo*, const std::string&)>;

    explicit InnerVoiceUI(wxWindow* parent);
    ~InnerVoiceUI() = default;

    // Panel creation
    wxPanel* CreatePanel();

    // External dependencies injection
    void SetSettingsManager(SettingsManager* settings_manager);
    void SetLlamaManager(LlamaManager* llama_manager);
    void SetCallbacks(LogCallback log_cb, ContextApplyCallback context_apply_cb);

    // Settings persistence
    void LoadSettings();
    void SaveSettings();

    // UI state management
    void UpdateUI();

    // Inner Voice configuration access
    std::string GetModelPath() const;
    int GetContextSize() const;
    int GetGPULayers() const;

private:
    // UI Components - Inner Voice Configuration
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

    // External dependencies
    wxWindow* parent_window{nullptr};
    SettingsManager* settings_manager{nullptr};
    LlamaManager* llama_manager{nullptr};

    // Callbacks
    LogCallback log_callback;
    ContextApplyCallback context_apply_callback;

    // State tracking
    std::string current_model_id{"inner_model"};

    // UI Creation methods
    void CreateInnerVoiceConfigurationSection(wxBoxSizer* main_sizer);
    
    // Event handlers
    void OnBrowseModel(wxCommandEvent& event);
    void OnContextSizeChange(wxCommandEvent& event);
    void OnGPULayersChange(wxCommandEvent& event);
    void OnModelPathChange(wxCommandEvent& event);

    // Helper methods
    void LogMessage(const std::string& message);
    void SetupSliderEvents();
    void ValidateModelConfiguration();
    void UpdateSliderLabels();
    
    // Settings helpers
    void LoadInnerVoiceSettings();
    void SaveInnerVoiceSettings();

    // Constants for IDs
    enum {
        ID_BrowseInnerModel = 21001
    };
};

/**
 * Inner Voice Settings UI Implementation
 */

inline InnerVoiceUI::InnerVoiceUI(wxWindow* parent) 
    : parent_window(parent) {
}

inline wxPanel* InnerVoiceUI::CreatePanel() {
    settings_panel = new wxPanel(parent_window);
    
    // Create scrolled window for the settings
    scrolled_window = new wxScrolledWindow(settings_panel, wxID_ANY);
    scrolled_window->SetScrollRate(5, 5);
    
    // Main vertical sizer for the scrolled content
    wxBoxSizer* main_sizer = new wxBoxSizer(wxVERTICAL);
    
    // Create sections
    CreateInnerVoiceConfigurationSection(main_sizer);
    
    // Set up scrolled window
    scrolled_window->SetSizer(main_sizer);
    scrolled_window->FitInside();
    
    // Panel layout
    wxBoxSizer* panel_sizer = new wxBoxSizer(wxVERTICAL);
    panel_sizer->Add(scrolled_window, 1, wxEXPAND | wxALL, 5);
    settings_panel->SetSizer(panel_sizer);
    
    // Set up events
    SetupSliderEvents();
    
    return settings_panel;
}

inline void InnerVoiceUI::CreateInnerVoiceConfigurationSection(wxBoxSizer* main_sizer) {
    // Inner Voice configuration group
    wxStaticBoxSizer* model_box = new wxStaticBoxSizer(wxVERTICAL, scrolled_window, "Inner Voice Configuration");
    
    // Model path selection
    wxBoxSizer* path_sizer = new wxBoxSizer(wxHORIZONTAL);
    path_sizer->Add(new wxStaticText(scrolled_window, wxID_ANY, "Model Path:"), 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    model_path_text = new wxTextCtrl(scrolled_window, wxID_ANY);
    path_sizer->Add(model_path_text, 1, wxEXPAND | wxALL, 5);
    browse_model_button = new wxButton(scrolled_window, ID_BrowseInnerModel, "Browse...");
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

    // Add info text about unified model loading
    wxStaticText* info_text = new wxStaticText(scrolled_window, wxID_ANY, "Note: Use the \"Load Model\" button in the General Settings tab to load models.");
    info_text->SetForegroundColour(wxColour(100, 100, 100)); // Gray text
    model_box->Add(info_text, 0, wxEXPAND | wxALL, 5);

    main_sizer->Add(model_box, 0, wxEXPAND | wxALL, 5);
}

inline void InnerVoiceUI::SetupSliderEvents() {
    if (context_size_slider) {
        context_size_slider->Bind(wxEVT_SLIDER, &InnerVoiceUI::OnContextSizeChange, this);
    }
    if (gpu_layers_slider) {
        gpu_layers_slider->Bind(wxEVT_SLIDER, &InnerVoiceUI::OnGPULayersChange, this);
    }
    if (browse_model_button) {
        browse_model_button->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &InnerVoiceUI::OnBrowseModel, this);
    }
    if (model_path_text) {
        model_path_text->Bind(wxEVT_TEXT, &InnerVoiceUI::OnModelPathChange, this);
    }
}

// External dependencies injection
inline void InnerVoiceUI::SetSettingsManager(SettingsManager* settings_manager) {
    this->settings_manager = settings_manager;
}

inline void InnerVoiceUI::SetLlamaManager(LlamaManager* llama_manager) {
    this->llama_manager = llama_manager;
}

inline void InnerVoiceUI::SetCallbacks(LogCallback log_cb, ContextApplyCallback context_apply_cb) {
    log_callback = log_cb;
    context_apply_callback = context_apply_cb;
}

// Settings persistence
inline void InnerVoiceUI::LoadSettings() {
    if (!settings_manager) {
        LogMessage("WARNING: Cannot load settings - SettingsManager not initialized");
        return;
    }

    LogMessage("Loading inner voice settings UI configuration...");
    LoadInnerVoiceSettings();
    UpdateUI();
    LogMessage("Inner voice settings UI loaded successfully");
}

inline void InnerVoiceUI::LoadInnerVoiceSettings() {
    if (!settings_manager) return;

    // Load model path
    if (model_path_text) {
        std::string model_path = settings_manager->GetString("Models", "inner_model_path", "");
        model_path_text->SetValue(model_path);
        if (!model_path.empty()) {
            LogMessage("Loaded inner voice model path: " + model_path);
        }
    }

    // Load context size
    if (context_size_slider) {
        int context_size = settings_manager->GetInt("Models", "inner_context_size", 4096);
        context_size_slider->SetValue(context_size);
        LogMessage("Loaded inner voice context size: " + std::to_string(context_size));
    }

    // Load GPU layers
    if (gpu_layers_slider) {
        int gpu_layers = settings_manager->GetInt("Models", "inner_gpu_layers", 999);
        gpu_layers_slider->SetValue(gpu_layers);
        LogMessage("Loaded inner voice GPU layers: " + std::to_string(gpu_layers));
    }
}

inline void InnerVoiceUI::SaveSettings() {
    if (!settings_manager) {
        LogMessage("WARNING: Cannot save settings - SettingsManager not initialized");
        return;
    }

    LogMessage("Saving inner voice settings UI configuration...");
    SaveInnerVoiceSettings();
    settings_manager->SaveSettings();
    LogMessage("Inner voice settings UI saved successfully");
}

inline void InnerVoiceUI::SaveInnerVoiceSettings() {
    if (!settings_manager) return;

    // Save model path
    if (model_path_text) {
        std::string model_path = model_path_text->GetValue().ToStdString();
        settings_manager->SetString("Models", "inner_model_path", model_path);
    }

    // Save context size
    if (context_size_slider) {
        int context_size = context_size_slider->GetValue();
        settings_manager->SetInt("Models", "inner_context_size", context_size);
    }

    // Save GPU layers
    if (gpu_layers_slider) {
        int gpu_layers = gpu_layers_slider->GetValue();
        settings_manager->SetInt("Models", "inner_gpu_layers", gpu_layers);
    }
}

// UI state management
inline void InnerVoiceUI::UpdateUI() {
    UpdateSliderLabels();
    ValidateModelConfiguration();
}

inline void InnerVoiceUI::UpdateSliderLabels() {
    if (context_size_label && context_size_slider) {
        context_size_label->SetLabel(wxString::Format("%d", context_size_slider->GetValue()));
    }
    if (gpu_layers_label && gpu_layers_slider) {
        gpu_layers_label->SetLabel(wxString::Format("%d", gpu_layers_slider->GetValue()));
    }
}

inline void InnerVoiceUI::ValidateModelConfiguration() {
    // No validation needed since model loading is handled centrally
    // This method is kept for consistency and potential future use
}

// Inner Voice configuration access
inline std::string InnerVoiceUI::GetModelPath() const {
    if (model_path_text) {
        return model_path_text->GetValue().ToStdString();
    }
    return "";
}

inline int InnerVoiceUI::GetContextSize() const {
    if (context_size_slider) {
        return context_size_slider->GetValue();
    }
    return 4096; // Default fallback
}

inline int InnerVoiceUI::GetGPULayers() const {
    if (gpu_layers_slider) {
        return gpu_layers_slider->GetValue();
    }
    return 999; // Default fallback
}

// Event handlers
inline void InnerVoiceUI::OnBrowseModel(wxCommandEvent& event) {
    wxFileDialog openFileDialog(parent_window, "Choose Inner Voice Model File", "", "",
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
    
    LogMessage("Selected inner voice model file: " + path.ToStdString());
}

inline void InnerVoiceUI::OnContextSizeChange(wxCommandEvent& event) {
    UpdateUI();
    if (settings_manager && context_size_slider) {
        settings_manager->SetInt("Models", "inner_context_size", context_size_slider->GetValue());
        settings_manager->SaveSettings();
    }
}

inline void InnerVoiceUI::OnGPULayersChange(wxCommandEvent& event) {
    UpdateUI();
    if (settings_manager && gpu_layers_slider) {
        settings_manager->SetInt("Models", "inner_gpu_layers", gpu_layers_slider->GetValue());
        settings_manager->SaveSettings();
    }
}

inline void InnerVoiceUI::OnModelPathChange(wxCommandEvent& event) {
    if (settings_manager && model_path_text) {
        settings_manager->SetString("Models", "inner_model_path", model_path_text->GetValue().ToStdString());
        // Note: Don't auto-save on every keystroke for performance
    }
    UpdateUI();
}

// Helper methods
inline void InnerVoiceUI::LogMessage(const std::string& message) {
    if (log_callback) {
        log_callback(message);
    }
}
