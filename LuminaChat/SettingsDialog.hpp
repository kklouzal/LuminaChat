#pragma once

#include <wx/wx.h>
#include <wx/filepicker.h>
#include <wx/spinctrl.h>
#include <wx/notebook.h>

class SettingsDialog : public wxDialog {
private:
    // Model Settings
    wxFilePickerCtrl* model_picker;
    wxSpinCtrl* context_size_spin;
    wxSpinCtrl* gpu_layers_spin;
    wxSpinCtrl* predict_tokens_spin;
    
    // Chat Template Settings
    wxTextCtrl* chat_template_text;
    
    // System Prompt Settings
    wxTextCtrl* identity_directive_text;
    wxTextCtrl* other_directives_text;
    
    // Discord Settings
    wxTextCtrl* discord_token_text;
    wxTextCtrl* discord_channels_text;
    wxTextCtrl* discord_isolated_channels_text;
    wxTextCtrl* discord_shared_history_channels_text; // ADDED: New textbox for shared history channels
    wxCheckBox* discord_allow_dms_check;
    
    // Reference variables
    std::string& model_path_ref;
    int32_t& context_size_ref;
    int32_t& gpu_layers_ref;
    int32_t& predict_tokens_ref;
    std::string& chat_template_ref;
    std::string& identity_directive_ref;
    std::string& other_directives_ref;
    std::string& discord_bot_token_ref;
    std::string& discord_channel_ids_ref;
    std::string& discord_isolated_channel_ids_ref;
    std::string& discord_shared_history_channel_ids_ref; // ADDED: New reference
    bool& discord_allow_dms_ref;

public:
    SettingsDialog(wxWindow* parent, 
                   std::string& model_path, int32_t& context_size, int32_t& gpu_layers, int32_t& predict_tokens,
                   std::string& chat_template, std::string& identity_directive, std::string& other_directives,
                   std::string& discord_bot_token, std::string& discord_channel_ids, 
                   std::string& discord_isolated_channel_ids, std::string& discord_shared_history_channel_ids,
                   bool& discord_allow_dms);

private:
    void CreateUI();
    void OnOK(wxCommandEvent& event);
    void OnCancel(wxCommandEvent& event);
    void LoadCurrentValues();
    void SaveValues();
};
