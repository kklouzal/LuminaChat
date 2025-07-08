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
#include "../Logger.hpp"

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
    using ContextApplyCallback = std::function<void(ContextInfo*, const std::string&)>;

    explicit InnerVoiceUI(wxWindow* parent);
    ~InnerVoiceUI() = default;

    // Panel creation
    wxPanel* CreatePanel();

    // External dependencies injection
    void SetSettingsManager(SettingsManager* settings_manager);
    void SetLlamaManager(LlamaManager* llama_manager);
    void SetCallbacks(ContextApplyCallback context_apply_cb);

    // Settings persistence
    void LoadSettings();
    void SaveSettings();

    // UI state management
    void UpdateUI();

    // Template display functionality for debugging
    void UpdateTemplateDisplay(const std::string& template_content);

    // Inner Voice configuration access
    std::string GetModelPath() const;
    int GetContextSize() const;
    int GetGPULayers() const;
    std::string GetSystemPrompt() const;

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
    
    // System prompt configuration
    wxTextCtrl* system_prompt_text{nullptr};

    // Template display components for debugging
    wxNotebook* inner_notebook{nullptr};
    wxPanel* config_panel{nullptr};
    wxPanel* template_panel{nullptr};
    wxTextCtrl* template_display{nullptr};
    std::string last_finalized_template;

    // External dependencies
    wxWindow* parent_window{nullptr};
    SettingsManager* settings_manager{nullptr};
    LlamaManager* llama_manager{nullptr};

    // Callbacks
    ContextApplyCallback context_apply_callback;

    // State tracking
    std::string current_model_id{"inner_model"};

    // UI Creation methods
    void CreateInnerVoiceConfigurationSection(wxBoxSizer* main_sizer);
    void CreateTemplateSection(wxPanel* parent, wxBoxSizer* main_sizer);
    
    // Event handlers
    void OnBrowseModel(wxCommandEvent& event);
    void OnContextSizeChange(wxCommandEvent& event);
    void OnGPULayersChange(wxCommandEvent& event);
    void OnModelPathChange(wxCommandEvent& event);
    void OnSystemPromptFocusLost(wxFocusEvent& event);

    // Helper methods
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
    
    // Create notebook for organizing content into tabs
    inner_notebook = new wxNotebook(settings_panel, wxID_ANY);
    
    // Configuration tab
    config_panel = new wxPanel(inner_notebook);
    inner_notebook->AddPage(config_panel, "Configuration");
    
    // Create scrolled window for the configuration
    scrolled_window = new wxScrolledWindow(config_panel, wxID_ANY);
    scrolled_window->SetScrollRate(5, 5);
    scrolled_window->EnableScrolling(true, true);
    scrolled_window->SetCanFocus(false); // Prevent scrolled window from stealing focus
    
    // Main vertical sizer for the scrolled content
    wxBoxSizer* config_sizer = new wxBoxSizer(wxVERTICAL);
    
    // Create configuration section
    CreateInnerVoiceConfigurationSection(config_sizer);
    
    // Set up scrolled window
    scrolled_window->SetSizer(config_sizer);
    scrolled_window->FitInside();
    
    // Config panel layout
    wxBoxSizer* config_panel_sizer = new wxBoxSizer(wxVERTICAL);
    config_panel_sizer->Add(scrolled_window, 1, wxEXPAND | wxALL, 5);
    config_panel->SetSizer(config_panel_sizer);
    
    // Template Display tab
    template_panel = new wxPanel(inner_notebook);
    inner_notebook->AddPage(template_panel, "Template Display");
    
    // Template section sizer
    wxBoxSizer* template_sizer = new wxBoxSizer(wxVERTICAL);
    CreateTemplateSection(template_panel, template_sizer);
    template_panel->SetSizer(template_sizer);
    
    // Main panel layout with notebook
    wxBoxSizer* panel_sizer = new wxBoxSizer(wxVERTICAL);
    panel_sizer->Add(inner_notebook, 1, wxEXPAND | wxALL, 5);
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

    // System prompt control
    model_box->Add(new wxStaticText(scrolled_window, wxID_ANY, "Inner Voice System Prompt:"), 0, wxALL, 5);
    system_prompt_text = new wxTextCtrl(scrolled_window, wxID_ANY, wxEmptyString,
                                       wxDefaultPosition, wxSize(-1, 120),
                                       wxTE_MULTILINE | wxTE_WORDWRAP | wxWANTS_CHARS);
    system_prompt_text->SetToolTip("Define the system-level instructions and context specific to the Inner Voice. This will be used in template variable replacement for the inner voice system prompt sections.");
    
    // Ensure the text control can receive focus properly by binding mouse events
    system_prompt_text->Bind(wxEVT_LEFT_DOWN, [this](wxMouseEvent& event) {
        if (system_prompt_text && !system_prompt_text->HasFocus()) {
            system_prompt_text->SetFocus();
        }
        event.Skip(); // Allow normal processing
    });
    
    model_box->Add(system_prompt_text, 0, wxEXPAND | wxALL, 5);

    // Add info text about unified model loading
    wxStaticText* info_text = new wxStaticText(scrolled_window, wxID_ANY, "Note: Use the \"Load Model\" button in the General Settings tab to load models.");
    info_text->SetForegroundColour(wxColour(100, 100, 100)); // Gray text
    model_box->Add(info_text, 0, wxEXPAND | wxALL, 5);

    main_sizer->Add(model_box, 0, wxEXPAND | wxALL, 5);
}

inline void InnerVoiceUI::CreateTemplateSection(wxPanel* parent, wxBoxSizer* main_sizer) {
    // Template display (read-only monospace text control)
    template_display = new wxTextCtrl(parent, wxID_ANY, wxEmptyString,
                                     wxDefaultPosition, wxDefaultSize,
                                     wxTE_READONLY | wxTE_MULTILINE | wxHSCROLL | wxVSCROLL);
    template_display->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
    
    // Initial explanatory text
    template_display->SetValue(
        "Inner Voice Template Display\n"
        "============================\n\n"
        "This displays the exact template sent to the Inner Voice AI model\n"
        "during the two-stage reasoning process. The template includes:\n\n"
        "* Environment description\n"
        "* Identity directive\n"
        "* System prompt\n"
        "* Conversation history\n"
        "* Template variables\n\n"
        "The template will be updated automatically when you send a message\n"
        "and the inner voice reasoning begins.\n\n"
        "Use this to debug template processing and verify that variables\n"
        "are being substituted correctly for the inner voice context."
    );
    
    // Add to sizer - template display takes all available space
    main_sizer->Add(template_display, 1, wxEXPAND | wxALL, 5);
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
    if (system_prompt_text) {
        system_prompt_text->Bind(wxEVT_KILL_FOCUS, &InnerVoiceUI::OnSystemPromptFocusLost, this);
    }
}

// External dependencies injection
inline void InnerVoiceUI::SetSettingsManager(SettingsManager* settings_manager) {
    this->settings_manager = settings_manager;
}

inline void InnerVoiceUI::SetLlamaManager(LlamaManager* llama_manager) {
    this->llama_manager = llama_manager;
}

inline void InnerVoiceUI::SetCallbacks(ContextApplyCallback context_apply_cb) {
    context_apply_callback = context_apply_cb;
}

// Settings persistence
inline void InnerVoiceUI::LoadSettings() {
    if (!settings_manager) {
        LOG_WARNING("InnerVoice", "Cannot load settings - SettingsManager not initialized");
        return;
    }

    LOG_INFO("InnerVoice", "Loading inner voice settings UI configuration...");
    LoadInnerVoiceSettings();
    UpdateUI();
    LOG_INFO("InnerVoice", "Inner voice settings UI loaded successfully");
}

inline void InnerVoiceUI::LoadInnerVoiceSettings() {
    if (!settings_manager) return;

    // Load model path
    if (model_path_text) {
        std::string model_path = settings_manager->GetString("Models", "inner_model_path", "");
        model_path_text->SetValue(model_path);
        if (!model_path.empty()) {
            LOG_INFO("InnerVoice", "Loaded inner voice model path: " + model_path);
        }
    }

    // Load context size
    if (context_size_slider) {
        int context_size = settings_manager->GetInt("Models", "inner_context_size", 4096);
        context_size_slider->SetValue(context_size);
        LOG_INFO("InnerVoice", "Loaded inner voice context size: " + std::to_string(context_size));
    }

    // Load GPU layers
    if (gpu_layers_slider) {
        int gpu_layers = settings_manager->GetInt("Models", "inner_gpu_layers", 999);
        gpu_layers_slider->SetValue(gpu_layers);
        LOG_INFO("InnerVoice", "Loaded inner voice GPU layers: " + std::to_string(gpu_layers));
    }

    // Load system prompt
    if (system_prompt_text) {
        std::string system_prompt = settings_manager->GetString("Models", "inner_system_prompt", "");
        system_prompt_text->SetValue(system_prompt);
        if (!system_prompt.empty()) {
            LOG_INFO("InnerVoice", "Loaded inner voice system prompt");
        }
    }
}

inline void InnerVoiceUI::SaveSettings() {
    if (!settings_manager) {
        LOG_WARNING("UI", "Cannot save settings - SettingsManager not initialized");
        return;
    }

    LOG_INFO("UI", "Saving inner voice settings UI configuration...");
    SaveInnerVoiceSettings();
    settings_manager->SaveSettings();
    LOG_INFO("UI", "Inner voice settings UI saved successfully");
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

    // Save system prompt
    if (system_prompt_text) {
        std::string system_prompt = system_prompt_text->GetValue().ToStdString();
        settings_manager->SetString("Models", "inner_system_prompt", system_prompt);
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

// Template display functionality for debugging
inline void InnerVoiceUI::UpdateTemplateDisplay(const std::string& template_content) {
    if (template_display) {
        // Try UTF-8 conversion first, fallback to default if it fails
        wxString wx_content = wxString::FromUTF8(template_content);
        if (wx_content.IsEmpty() && !template_content.empty()) {
            // UTF-8 conversion failed, try default conversion
            wx_content = wxString(template_content);
            LOG_WARNING("UI", "UTF-8 conversion failed for inner voice template, using default conversion");
        }
        template_display->SetValue(wx_content);
        last_finalized_template = template_content; // Keep track of the last finalized template
        LOG_DEBUG("UI", "Updated inner voice template display with " + std::to_string(template_content.length()) + " characters");
    }
    LOG_DEBUG("UI", "Template Content: " + template_content);
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

inline std::string InnerVoiceUI::GetSystemPrompt() const {
    if (system_prompt_text) {
        return system_prompt_text->GetValue().ToStdString();
    }
    return ""; // Default fallback
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
    
    LOG_INFO("UI", "Selected inner voice model file: " + path.ToStdString());
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

inline void InnerVoiceUI::OnSystemPromptFocusLost(wxFocusEvent& event) {
    if (settings_manager && system_prompt_text) {
        settings_manager->SetString("Models", "inner_system_prompt", system_prompt_text->GetValue().ToStdString());
        settings_manager->SaveSettings();
    }
}
