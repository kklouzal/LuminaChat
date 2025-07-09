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
#include <wx/listbox.h>
#include <memory>
#include <functional>
#include <vector>
#include "../SettingsManager.hpp"
#include "../Logger.hpp"

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
    using LoadModelsCallback = std::function<void()>;
    using ContextApplyCallback = std::function<void(ContextInfo*, const std::string&)>;

    explicit SettingsUI(wxWindow* parent);
    ~SettingsUI() = default;

    // Panel creation
    wxPanel* CreatePanel();

    // External dependencies injection
    void SetSettingsManager(SettingsManager* settings_manager);
    void SetLlamaManager(LlamaManager* llama_manager);
    void SetCallbacks(LoadModelsCallback load_models_cb, ContextApplyCallback context_apply_cb);

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
    
    // Persona Management
    wxListBox* persona_list{nullptr};
    wxTextCtrl* persona_name_text{nullptr};
    wxButton* save_persona_button{nullptr};
    wxButton* load_persona_button{nullptr};
    wxButton* delete_persona_button{nullptr};

    // External dependencies
    wxWindow* parent_window{nullptr};
    SettingsManager* settings_manager{nullptr};
    LlamaManager* llama_manager{nullptr};

    // Callbacks
    LoadModelsCallback load_models_callback;
    ContextApplyCallback context_apply_callback;

    // State tracking
    bool models_loaded{false};
    bool models_loading{false};

    // UI Creation methods
    void CreateModelLoadingSection(wxBoxSizer* main_sizer);
    void CreateTemplateConfigurationSection(wxBoxSizer* main_sizer);
    void CreatePersonaManagementSection(wxBoxSizer* main_sizer);
    
    // Event handlers
    void OnLoadModels(wxCommandEvent& event);
    void OnEnvironmentDescriptionFocusLost(wxFocusEvent& event);
    void OnIdentityDirectiveFocusLost(wxFocusEvent& event);
    void OnPersonaListSelection(wxCommandEvent& event);
    void OnSavePersona(wxCommandEvent& event);
    void OnLoadPersona(wxCommandEvent& event);
    void OnDeletePersona(wxCommandEvent& event);

    // Helper methods
    void SetupEvents();
    
    // Settings helpers
    void LoadTemplateSettings();
    void SaveTemplateSettings();
    void LoadPersonaList();
    void RefreshPersonaList();
    void SavePersona(const std::string& name, const std::string& directive);
    void LoadPersona(const std::string& name);
    void DeletePersona(const std::string& name);
    std::string GetSelectedPersonaName() const;
    void UpdatePersonaButtons();
    
    // Content validation
    bool ValidateDirectiveContent(const std::string& content);
    
    // Emergency cleanup methods
    void ClearCurrentIdentityDirective();
    void ValidateAndCleanAllPersonas();

    // Constants for IDs
    enum {
        ID_LoadModels = 20001,
        ID_PersonaList,
        ID_SavePersona,
        ID_LoadPersona,
        ID_DeletePersona
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
    CreatePersonaManagementSection(main_sizer);
    
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

inline void SettingsUI::CreatePersonaManagementSection(wxBoxSizer* main_sizer) {
    // Persona management group
    wxStaticBoxSizer* persona_box = new wxStaticBoxSizer(wxVERTICAL, scrolled_window, "Identity Persona Management");
    
    // Persona list
    persona_box->Add(new wxStaticText(scrolled_window, wxID_ANY, "Saved Personas:"), 0, wxALL, 5);
    persona_list = new wxListBox(scrolled_window, ID_PersonaList, wxDefaultPosition, wxSize(-1, 120));
    persona_list->SetToolTip("Select a saved persona to load its identity directive");
    persona_box->Add(persona_list, 0, wxEXPAND | wxALL, 5);
    
    // Persona name input
    wxBoxSizer* name_sizer = new wxBoxSizer(wxHORIZONTAL);
    name_sizer->Add(new wxStaticText(scrolled_window, wxID_ANY, "Persona Name:"), 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    persona_name_text = new wxTextCtrl(scrolled_window, wxID_ANY, wxEmptyString, wxDefaultPosition, wxSize(200, -1));
    persona_name_text->SetToolTip("Enter a name for the current persona when saving");
    name_sizer->Add(persona_name_text, 1, wxEXPAND | wxALL, 5);
    persona_box->Add(name_sizer, 0, wxEXPAND);
    
    // Persona management buttons
    wxBoxSizer* button_sizer = new wxBoxSizer(wxHORIZONTAL);
    save_persona_button = new wxButton(scrolled_window, ID_SavePersona, "Save Persona");
    save_persona_button->SetToolTip("Save the current identity directive as a named persona");
    button_sizer->Add(save_persona_button, 0, wxALL, 5);
    
    load_persona_button = new wxButton(scrolled_window, ID_LoadPersona, "Load Persona");
    load_persona_button->SetToolTip("Load the selected persona's identity directive");
    button_sizer->Add(load_persona_button, 0, wxALL, 5);
    
    delete_persona_button = new wxButton(scrolled_window, ID_DeletePersona, "Delete Persona");
    delete_persona_button->SetToolTip("Delete the selected persona permanently");
    button_sizer->Add(delete_persona_button, 0, wxALL, 5);
    
    persona_box->Add(button_sizer, 0, wxEXPAND);

    main_sizer->Add(persona_box, 0, wxEXPAND | wxALL, 5);
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
    if (persona_list) {
        persona_list->Bind(wxEVT_LISTBOX, &SettingsUI::OnPersonaListSelection, this);
    }
    if (save_persona_button) {
        save_persona_button->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &SettingsUI::OnSavePersona, this);
    }
    if (load_persona_button) {
        load_persona_button->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &SettingsUI::OnLoadPersona, this);
    }
    if (delete_persona_button) {
        delete_persona_button->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &SettingsUI::OnDeletePersona, this);
    }
}

// External dependencies injection
inline void SettingsUI::SetSettingsManager(SettingsManager* settings_manager) {
    this->settings_manager = settings_manager;
}

inline void SettingsUI::SetLlamaManager(LlamaManager* llama_manager) {
    this->llama_manager = llama_manager;
}

inline void SettingsUI::SetCallbacks(LoadModelsCallback load_models_cb, ContextApplyCallback context_apply_cb) {
    load_models_callback = load_models_cb;
    context_apply_callback = context_apply_cb;
}

// Settings persistence
inline void SettingsUI::LoadSettings() {
    if (!settings_manager) {
        LOG_WARNING("Settings", "Cannot load settings - SettingsManager not initialized");
        return;
    }

    LOG_INFO("Settings", "Loading settings UI configuration...");
    
    // CRITICAL FIX: Validate current identity directive and clean up if needed
    std::string current_directive = settings_manager->GetString("Templates", "identity_directive", "");
    if (!current_directive.empty() && !ValidateDirectiveContent(current_directive)) {
        LOG_ERROR("Settings", "CRITICAL: Detected corrupted identity directive - clearing it to fix Discord output loop");
        ClearCurrentIdentityDirective();
    }
    
    // Clean up any corrupted personas
    ValidateAndCleanAllPersonas();
    
    LoadTemplateSettings();
    LoadPersonaList();
    UpdateUI();
    LOG_INFO("Settings", "Settings UI loaded successfully");
}

inline void SettingsUI::LoadTemplateSettings() {
    if (!settings_manager) return;

    // Load environment description
    if (environment_description_text) {
        std::string env_desc = settings_manager->GetString("Templates", "environment_description", "");
        environment_description_text->SetValue(env_desc);
        if (!env_desc.empty()) {
            LOG_INFO("Settings", "Loaded environment description from settings");
        }
    }

    // Load identity directive
    if (identity_directive_text) {
        std::string identity = settings_manager->GetString("Templates", "identity_directive", "");
        identity_directive_text->SetValue(identity);
        if (!identity.empty()) {
            LOG_INFO("Settings", "Loaded identity directive from settings");
        }
    }
}

inline void SettingsUI::SaveSettings() {
    if (!settings_manager) {
        LOG_WARNING("Settings", "Cannot save settings - SettingsManager not initialized");
        return;
    }

    LOG_INFO("Settings", "Saving settings UI configuration...");
    SaveTemplateSettings();
    settings_manager->SaveSettings();
    LOG_INFO("Settings", "Settings UI saved successfully");
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
    
    // Update persona management buttons
    UpdatePersonaButtons();
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
        LOG_INFO("Settings", "Initiating load of all models from voice configurations...");
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
        std::string directive = identity_directive_text->GetValue().ToStdString();
        
        // CRITICAL FIX: Validate identity directive before saving
        if (ValidateDirectiveContent(directive)) {
            settings_manager->SetString("Templates", "identity_directive", directive);
            settings_manager->SaveSettings();
        } else {
            LOG_ERROR("Settings", "Identity directive contains problematic content - not saved");
            // Optionally reload the last good value
            std::string last_good = settings_manager->GetString("Templates", "identity_directive", "");
            identity_directive_text->SetValue(last_good);
        }
    }
    event.Skip();
}

// Persona Management Methods
inline void SettingsUI::LoadPersonaList() {
    if (!settings_manager || !persona_list) return;
    
    RefreshPersonaList();
    
    // Load the last selected persona if available
    std::string last_persona = settings_manager->GetString("Templates", "last_selected_persona", "");
    if (!last_persona.empty()) {
        int index = persona_list->FindString(last_persona);
        if (index != wxNOT_FOUND) {
            persona_list->SetSelection(index);
            LOG_INFO("Settings", "Auto-selected last used persona: " + last_persona);
        }
    }
    
    UpdatePersonaButtons();
}

inline void SettingsUI::RefreshPersonaList() {
    if (!settings_manager || !persona_list) return;
    
    persona_list->Clear();
    
    // Get all persona names from settings
    std::vector<std::string> persona_names;
    for (int i = 0; i < 100; ++i) { // Reasonable limit
        std::string key = "persona_" + std::to_string(i) + "_name";
        std::string name = settings_manager->GetString("Personas", key, "");
        if (!name.empty()) {
            persona_names.push_back(name);
        }
    }
    
    // Add to listbox
    for (const auto& name : persona_names) {
        persona_list->Append(name);
    }
    
    LOG_INFO("Settings", "Loaded " + std::to_string(persona_names.size()) + " saved personas");
}

inline void SettingsUI::SavePersona(const std::string& name, const std::string& directive) {
    if (!settings_manager || name.empty()) return;
    
    // CRITICAL FIX: Validate directive content before saving
    if (!ValidateDirectiveContent(directive)) {
        LOG_ERROR("Settings", "Cannot save persona '" + name + "' - content validation failed");
        return;
    }
    
    // Find an available slot or update existing
    int slot = -1;
    for (int i = 0; i < 100; ++i) {
        std::string key = "persona_" + std::to_string(i) + "_name";
        std::string existing_name = settings_manager->GetString("Personas", key, "");
        if (existing_name.empty() || existing_name == name) {
            slot = i;
            break;
        }
    }
    
    if (slot == -1) {
        LOG_ERROR("Settings", "Cannot save persona - no available slots");
        return;
    }
    
    // Save persona data
    std::string name_key = "persona_" + std::to_string(slot) + "_name";
    std::string directive_key = "persona_" + std::to_string(slot) + "_directive";
    
    settings_manager->SetString("Personas", name_key, name);
    settings_manager->SetString("Personas", directive_key, directive);
    settings_manager->SaveSettings();
    
    LOG_INFO("Settings", "Saved persona: " + name + " (slot " + std::to_string(slot) + ")");
    
    // Refresh the list and select the saved persona
    RefreshPersonaList();
    int index = persona_list->FindString(name);
    if (index != wxNOT_FOUND) {
        persona_list->SetSelection(index);
    }
    UpdatePersonaButtons();
}

inline void SettingsUI::LoadPersona(const std::string& name) {
    if (!settings_manager || !identity_directive_text || name.empty()) return;
    
    // Find the persona
    for (int i = 0; i < 100; ++i) {
        std::string name_key = "persona_" + std::to_string(i) + "_name";
        std::string existing_name = settings_manager->GetString("Personas", name_key, "");
        if (existing_name == name) {
            std::string directive_key = "persona_" + std::to_string(i) + "_directive";
            std::string directive = settings_manager->GetString("Personas", directive_key, "");
            
            // CRITICAL FIX: Validate directive content to prevent loops
            if (ValidateDirectiveContent(directive)) {
                identity_directive_text->SetValue(directive);
                
                // Save as current identity directive and remember as last selected
                settings_manager->SetString("Templates", "identity_directive", directive);
                settings_manager->SetString("Templates", "last_selected_persona", name);
                settings_manager->SaveSettings();
                
                LOG_INFO("Settings", "Loaded persona: " + name);
            } else {
                LOG_ERROR("Settings", "Persona '" + name + "' contains invalid or potentially problematic content");
            }
            return;
        }
    }
    
    LOG_ERROR("Settings", "Persona not found: " + name);
}

inline void SettingsUI::DeletePersona(const std::string& name) {
    if (!settings_manager || name.empty()) return;
    
    // Find and delete the persona
    for (int i = 0; i < 100; ++i) {
        std::string name_key = "persona_" + std::to_string(i) + "_name";
        std::string existing_name = settings_manager->GetString("Personas", name_key, "");
        if (existing_name == name) {
            std::string directive_key = "persona_" + std::to_string(i) + "_directive";
            
            settings_manager->SetString("Personas", name_key, "");
            settings_manager->SetString("Personas", directive_key, "");
            settings_manager->SaveSettings();
            
            LOG_INFO("Settings", "Deleted persona: " + name);
            
            // Clear the persona name field if it matches
            if (persona_name_text && persona_name_text->GetValue().ToStdString() == name) {
                persona_name_text->SetValue("");
            }
            
            // Refresh the list
            RefreshPersonaList();
            UpdatePersonaButtons();
            return;
        }
    }
    
    LOG_ERROR("Settings", "Persona not found for deletion: " + name);
}

inline std::string SettingsUI::GetSelectedPersonaName() const {
    if (!persona_list) return "";
    
    int selection = persona_list->GetSelection();
    if (selection == wxNOT_FOUND) return "";
    
    return persona_list->GetString(selection).ToStdString();
}

inline void SettingsUI::UpdatePersonaButtons() {
    bool has_selection = !GetSelectedPersonaName().empty();
    bool has_name = persona_name_text && !persona_name_text->GetValue().IsEmpty();
    bool has_directive = identity_directive_text && !identity_directive_text->GetValue().IsEmpty();
    
    if (load_persona_button) {
        load_persona_button->Enable(has_selection);
    }
    if (delete_persona_button) {
        delete_persona_button->Enable(has_selection);
    }
    if (save_persona_button) {
        save_persona_button->Enable(has_name && has_directive);
    }
}

// New Event Handlers
inline void SettingsUI::OnPersonaListSelection(wxCommandEvent& event) {
    std::string selected_name = GetSelectedPersonaName();
    if (!selected_name.empty() && persona_name_text) {
        persona_name_text->SetValue(selected_name);
    }
    UpdatePersonaButtons();
}

inline void SettingsUI::OnSavePersona(wxCommandEvent& event) {
    if (!persona_name_text || !identity_directive_text) return;
    
    std::string name = persona_name_text->GetValue().Trim().ToStdString();
    std::string directive = identity_directive_text->GetValue().ToStdString();
    
    if (name.empty()) {
        LOG_ERROR("Settings", "Cannot save persona - name is empty");
        return;
    }
    
    if (directive.empty()) {
        LOG_ERROR("Settings", "Cannot save persona - identity directive is empty");
        return;
    }
    
    // Confirm overwrite if persona already exists
    int existing_index = persona_list->FindString(name);
    if (existing_index != wxNOT_FOUND) {
        // Use unified error handling system for user confirmation
        HandleWarning("A persona with this name already exists. Please choose a different name or modify the existing persona.", "Persona Management", true);
        return;
    }
    
    SavePersona(name, directive);
}

inline void SettingsUI::OnLoadPersona(wxCommandEvent& event) {
    std::string selected_name = GetSelectedPersonaName();
    if (selected_name.empty()) {
        LOG_ERROR("Settings", "No persona selected to load");
        return;
    }
    
    LoadPersona(selected_name);
}

inline void SettingsUI::OnDeletePersona(wxCommandEvent& event) {
    std::string selected_name = GetSelectedPersonaName();
    if (selected_name.empty()) {
        HandleError("No persona selected to delete", "Persona Management", true);
        return;
    }
    
    // Use unified error handling system to confirm deletion
    HandleWarning("To delete persona '" + selected_name + "', please confirm by clicking the delete button again.", "Persona Management", true);
    
    // Store the deletion candidate for confirmation
    static std::string deletion_candidate = "";
    if (deletion_candidate == selected_name) {
        // Second click confirmed - proceed with deletion
        DeletePersona(selected_name);
        deletion_candidate = "";
    } else {
        // First click - set candidate
        deletion_candidate = selected_name;
    }
}

// Content validation to prevent problematic personas
inline bool SettingsUI::ValidateDirectiveContent(const std::string& content) {
    if (content.empty()) return true; // Empty content is valid
    
    // Check for repetitive patterns that could cause AI loops
    std::vector<std::string> problematic_patterns = {
        "[ behaviors ]",
        "[behaviors]", 
        "behaviors",
        "behavior",
        "behaviour",
        "behaviours"
    };
    
    // Count occurrences of each pattern
    for (const auto& pattern : problematic_patterns) {
        size_t count = 0;
        size_t pos = 0;
        while ((pos = content.find(pattern, pos)) != std::string::npos) {
            count++;
            pos += pattern.length();
            
            // If we find more than 3 occurrences of any pattern, it's likely problematic
            if (count > 3) {
                LOG_WARNING("Settings", "Identity directive contains repetitive pattern: " + pattern + " (found " + std::to_string(count) + " times)");
                return false;
            }
        }
    }
    
    // Check for excessive length (over 5000 characters might be problematic)
    if (content.length() > 5000) {
        LOG_WARNING("Settings", "Identity directive is very long (" + std::to_string(content.length()) + " characters) - this might cause issues");
        return false;
    }
    
    // Check for common template injection patterns
    if (content.find("{{") != std::string::npos || content.find("}}") != std::string::npos ||
        content.find("{%") != std::string::npos || content.find("%}") != std::string::npos) {
        LOG_WARNING("Settings", "Identity directive contains template syntax that could cause processing issues");
        return false;
    }
    
    return true;
}

// Emergency cleanup methods to fix corruption issues
inline void SettingsUI::ClearCurrentIdentityDirective() {
    if (!settings_manager) return;
    
    LOG_ERROR("Settings", "EMERGENCY FIX: Clearing potentially corrupted identity directive");
    
    // Clear the current identity directive
    settings_manager->SetString("Templates", "identity_directive", "");
    settings_manager->SetString("Templates", "last_selected_persona", "");
    settings_manager->SaveSettings();
    
    // Clear the UI field
    if (identity_directive_text) {
        identity_directive_text->SetValue("");
    }
    
    LOG_INFO("Settings", "Identity directive cleared - Discord output should return to normal");
}

inline void SettingsUI::ValidateAndCleanAllPersonas() {
    if (!settings_manager) return;
    
    LOG_INFO("Settings", "Validating and cleaning all saved personas...");
    
    int cleaned_count = 0;
    
    // Check all persona slots
    for (int i = 0; i < 100; ++i) {
        std::string name_key = "persona_" + std::to_string(i) + "_name";
        std::string directive_key = "persona_" + std::to_string(i) + "_directive";
        
        std::string name = settings_manager->GetString("Personas", name_key, "");
        std::string directive = settings_manager->GetString("Personas", directive_key, "");
        
        if (!name.empty() && !directive.empty()) {
            if (!ValidateDirectiveContent(directive)) {
                LOG_WARNING("Settings", "CLEANING: Removing corrupted persona '" + name + "'");
                settings_manager->SetString("Personas", name_key, "");
                settings_manager->SetString("Personas", directive_key, "");
                cleaned_count++;
            }
        }
    }
    
    if (cleaned_count > 0) {
        settings_manager->SaveSettings();
        RefreshPersonaList();
        LOG_INFO("Settings", "Cleaned " + std::to_string(cleaned_count) + " corrupted personas");
    } else {
        LOG_INFO("Settings", "All personas are valid - no cleanup needed");
    }
}
