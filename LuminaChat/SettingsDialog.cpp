#include "SettingsDialog.hpp"
#include "SettingsManager.hpp"

SettingsDialog::SettingsDialog(wxWindow* parent, 
                               std::string& model_path, int32_t& context_size, int32_t& gpu_layers, int32_t& predict_tokens,
                               std::string& chat_template, std::string& identity_directive, std::string& other_directives,
                               std::string& discord_bot_token, std::string& discord_channel_ids, 
                               std::string& discord_isolated_channel_ids, std::string& discord_shared_history_channel_ids,
                               bool& discord_allow_dms)
    : wxDialog(parent, wxID_ANY, "Settings", wxDefaultPosition, wxSize(600, 500), wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER),
      model_path_ref(model_path), context_size_ref(context_size), gpu_layers_ref(gpu_layers), predict_tokens_ref(predict_tokens),
      chat_template_ref(chat_template), identity_directive_ref(identity_directive), other_directives_ref(other_directives),
      discord_bot_token_ref(discord_bot_token), discord_channel_ids_ref(discord_channel_ids), 
      discord_isolated_channel_ids_ref(discord_isolated_channel_ids), discord_shared_history_channel_ids_ref(discord_shared_history_channel_ids),
      discord_allow_dms_ref(discord_allow_dms) {
    
    CreateUI();
    LoadCurrentValues();
}

void SettingsDialog::CreateUI() {
    wxNotebook* notebook = new wxNotebook(this, wxID_ANY);
    
    // Model Settings Tab
    wxPanel* model_panel = new wxPanel(notebook);
    wxBoxSizer* model_sizer = new wxBoxSizer(wxVERTICAL);
    
    model_sizer->Add(new wxStaticText(model_panel, wxID_ANY, "Model File:"), 0, wxALL, 5);
    model_picker = new wxFilePickerCtrl(model_panel, wxID_ANY, "", "Select Model File", 
                                        "GGUF files (*.gguf)|*.gguf|All files (*.*)|*.*");
    model_sizer->Add(model_picker, 0, wxEXPAND | wxALL, 5);
    
    model_sizer->Add(new wxStaticText(model_panel, wxID_ANY, "Context Size:"), 0, wxALL, 5);
    context_size_spin = new wxSpinCtrl(model_panel, wxID_ANY, "", wxDefaultPosition, wxDefaultSize, 
                                       wxSP_ARROW_KEYS, 1, 131072, 2048);
    model_sizer->Add(context_size_spin, 0, wxEXPAND | wxALL, 5);
    
    model_sizer->Add(new wxStaticText(model_panel, wxID_ANY, "GPU Layers:"), 0, wxALL, 5);
    gpu_layers_spin = new wxSpinCtrl(model_panel, wxID_ANY, "", wxDefaultPosition, wxDefaultSize, 
                                     wxSP_ARROW_KEYS, 0, 999, 0);
    model_sizer->Add(gpu_layers_spin, 0, wxEXPAND | wxALL, 5);
    
    model_sizer->Add(new wxStaticText(model_panel, wxID_ANY, "Max Prediction Tokens:"), 0, wxALL, 5);
    predict_tokens_spin = new wxSpinCtrl(model_panel, wxID_ANY, "", wxDefaultPosition, wxDefaultSize, 
                                         wxSP_ARROW_KEYS, 1, 4096, 256);
    model_sizer->Add(predict_tokens_spin, 0, wxEXPAND | wxALL, 5);
    
    model_panel->SetSizer(model_sizer);
    notebook->AddPage(model_panel, "Model");
    
    // Chat Template Tab
    wxPanel* template_panel = new wxPanel(notebook);
    wxBoxSizer* template_sizer = new wxBoxSizer(wxVERTICAL);
    
    template_sizer->Add(new wxStaticText(template_panel, wxID_ANY, "Custom Chat Template (leave empty to use model default):"), 
                        0, wxALL, 5);
    chat_template_text = new wxTextCtrl(template_panel, wxID_ANY, "", wxDefaultPosition, wxDefaultSize, 
                                        wxTE_MULTILINE | wxTE_WORDWRAP);
    template_sizer->Add(chat_template_text, 1, wxEXPAND | wxALL, 5);
    
    template_panel->SetSizer(template_sizer);
    notebook->AddPage(template_panel, "Chat Template");
    
    // System Prompt Tab
    wxPanel* prompt_panel = new wxPanel(notebook);
    wxBoxSizer* prompt_sizer = new wxBoxSizer(wxVERTICAL);
    
    prompt_sizer->Add(new wxStaticText(prompt_panel, wxID_ANY, "Identity Directive:"), 0, wxALL, 5);
    identity_directive_text = new wxTextCtrl(prompt_panel, wxID_ANY, "", wxDefaultPosition, wxDefaultSize, 
                                             wxTE_MULTILINE | wxTE_WORDWRAP);
    prompt_sizer->Add(identity_directive_text, 1, wxEXPAND | wxALL, 5);
    
    prompt_sizer->Add(new wxStaticText(prompt_panel, wxID_ANY, "Other Directives:"), 0, wxALL, 5);
    other_directives_text = new wxTextCtrl(prompt_panel, wxID_ANY, "", wxDefaultPosition, wxDefaultSize, 
                                           wxTE_MULTILINE | wxTE_WORDWRAP);
    prompt_sizer->Add(other_directives_text, 1, wxEXPAND | wxALL, 5);
    
    prompt_panel->SetSizer(prompt_sizer);
    notebook->AddPage(prompt_panel, "System Prompt");
    
    // Discord Settings Tab
    wxPanel* discord_panel = new wxPanel(notebook);
    wxBoxSizer* discord_sizer = new wxBoxSizer(wxVERTICAL);
    
    discord_sizer->Add(new wxStaticText(discord_panel, wxID_ANY, "Bot Token:"), 0, wxALL, 5);
    discord_token_text = new wxTextCtrl(discord_panel, wxID_ANY, "", wxDefaultPosition, wxDefaultSize, wxTE_PASSWORD);
    discord_sizer->Add(discord_token_text, 0, wxEXPAND | wxALL, 5);
    
    discord_sizer->Add(new wxStaticText(discord_panel, wxID_ANY, "Allowed Channel IDs (comma-separated, empty = all):"), 0, wxALL, 5);
    discord_channels_text = new wxTextCtrl(discord_panel, wxID_ANY);
    discord_sizer->Add(discord_channels_text, 0, wxEXPAND | wxALL, 5);
    
    discord_sizer->Add(new wxStaticText(discord_panel, wxID_ANY, "Isolated Context Channel IDs (comma-separated):"), 0, wxALL, 5);
    discord_isolated_channels_text = new wxTextCtrl(discord_panel, wxID_ANY);
    discord_sizer->Add(discord_isolated_channels_text, 0, wxEXPAND | wxALL, 5);
    
    // ADDED: New textbox for shared history channels
    discord_sizer->Add(new wxStaticText(discord_panel, wxID_ANY, "Shared Context History Channel IDs (comma-separated):"), 0, wxALL, 5);
    discord_shared_history_channels_text = new wxTextCtrl(discord_panel, wxID_ANY);
    discord_sizer->Add(discord_shared_history_channels_text, 0, wxEXPAND | wxALL, 5);
    
    discord_allow_dms_check = new wxCheckBox(discord_panel, wxID_ANY, "Allow Direct Messages");
    discord_sizer->Add(discord_allow_dms_check, 0, wxALL, 5);
    
    // Add help text
    wxStaticText* help_text = new wxStaticText(discord_panel, wxID_ANY, 
        "• Allowed channels: Bot will only respond in these channels (empty = all)\n"
        "• Isolated channels: Each gets its own conversation context\n"
        "• Shared history channels: Only these shared channels will have message history loaded\n"
        "• DMs always get isolated contexts when enabled");
    help_text->Wrap(550);
    discord_sizer->Add(help_text, 0, wxALL | wxEXPAND, 5);
    
    discord_panel->SetSizer(discord_sizer);
    notebook->AddPage(discord_panel, "Discord");
    
    // Buttons
    wxSizer* button_sizer = CreateButtonSizer(wxOK | wxCANCEL);
    
    // Main layout
    wxBoxSizer* main_sizer = new wxBoxSizer(wxVERTICAL);
    main_sizer->Add(notebook, 1, wxEXPAND | wxALL, 10);
    main_sizer->Add(button_sizer, 0, wxEXPAND | wxALL, 10);
    
    SetSizer(main_sizer);
    
    // Bind events
    Bind(wxEVT_COMMAND_BUTTON_CLICKED, &SettingsDialog::OnOK, this, wxID_OK);
    Bind(wxEVT_COMMAND_BUTTON_CLICKED, &SettingsDialog::OnCancel, this, wxID_CANCEL);
}

void SettingsDialog::LoadCurrentValues() {
    model_picker->SetPath(wxString::FromUTF8(model_path_ref));
    context_size_spin->SetValue(context_size_ref);
    gpu_layers_spin->SetValue(gpu_layers_ref);
    predict_tokens_spin->SetValue(predict_tokens_ref);
    chat_template_text->SetValue(wxString::FromUTF8(chat_template_ref));
    identity_directive_text->SetValue(wxString::FromUTF8(identity_directive_ref));
    other_directives_text->SetValue(wxString::FromUTF8(other_directives_ref));
    discord_token_text->SetValue(wxString::FromUTF8(discord_bot_token_ref));
    discord_channels_text->SetValue(wxString::FromUTF8(discord_channel_ids_ref));
    discord_isolated_channels_text->SetValue(wxString::FromUTF8(discord_isolated_channel_ids_ref));
    discord_shared_history_channels_text->SetValue(wxString::FromUTF8(discord_shared_history_channel_ids_ref)); // ADDED: Load new setting
    discord_allow_dms_check->SetValue(discord_allow_dms_ref);
}

void SettingsDialog::SaveValues() {
    model_path_ref = model_picker->GetPath().ToStdString();
    context_size_ref = context_size_spin->GetValue();
    gpu_layers_ref = gpu_layers_spin->GetValue();
    predict_tokens_ref = predict_tokens_spin->GetValue();
    chat_template_ref = chat_template_text->GetValue().ToStdString();
    identity_directive_ref = identity_directive_text->GetValue().ToStdString();
    other_directives_ref = other_directives_text->GetValue().ToStdString();
    discord_bot_token_ref = discord_token_text->GetValue().ToStdString();
    discord_channel_ids_ref = discord_channels_text->GetValue().ToStdString();
    discord_isolated_channel_ids_ref = discord_isolated_channels_text->GetValue().ToStdString();
    discord_shared_history_channel_ids_ref = discord_shared_history_channels_text->GetValue().ToStdString(); // ADDED: Save new setting
    discord_allow_dms_ref = discord_allow_dms_check->GetValue();
    
    SettingsManager::SaveSettings(model_path_ref, context_size_ref, gpu_layers_ref, predict_tokens_ref,
                                  chat_template_ref, identity_directive_ref, other_directives_ref,
                                  discord_bot_token_ref, discord_channel_ids_ref, discord_isolated_channel_ids_ref,
                                  discord_shared_history_channel_ids_ref, discord_allow_dms_ref); // ADDED: Save new setting
}

void SettingsDialog::OnOK(wxCommandEvent& event) {
    SaveValues();
    EndModal(wxID_OK);
}

void SettingsDialog::OnCancel(wxCommandEvent& event) {
    EndModal(wxID_CANCEL);
}
