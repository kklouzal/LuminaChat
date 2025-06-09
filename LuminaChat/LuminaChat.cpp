// LuminaChat.cpp - Main application file for LuminaChat with wxWidgets GUI
// Handles initialization, UI setup, and main application logic
//
// File Specific Directives:
// Manages all wxWidgets GUI components and application lifecycle.
//
// Project Settings:
// C++ Language Standard: ISO C++20 Standard (/std:c++20)
// C Language Standard: ISO C17 (2018) Standard (/std:c17)
// Character Set: Use Unicode Character Set
// Whole Program Optimization: Use Link Time Code Generation
// Optimization: Maximum Optimization (Favor Speed) (/O2)
// Enable Intrinsic Functions: Yes (/Oi)
// Favor Size or Speed: Favor fast code (/Ot)
// Whole Program Optimization: Yes (/GL)
// Enable String Pooling: Yes (/GF)
// Runtime Library: Multi-threaded DLL (/MD)
// Enable Run-Time Type Information (RTTI) YES (/GR)
// Link Time Code Generation: Use Link Time Code Generation (/LTCG)
//
// CODING DIRECTIVES:
// 1. Keep the codebase minimalistic, focused on functionality and efficiency.
// 2. Stay consistent with similar coding styles and patterns throughout the codebase.
// 3. Comment code thoroughly, where necessary, to explain complex logic or decisions.
// 4. Always eliminate unused code, dead code, legacy code, and cleanup includes.
// 5. Combine or split functions where necessary to eliminate redundancy.
// 6. Focus on overall codebase reduction without sacrificing functionality.
// 7. Use consistent _t fixed-width variable types to ensure portability across platforms.
// 8. Cache frequently used variables to avoid repeated allocations.
// 9. Ensure there are no logical errors and the execution paths flow as expected.
// 10. Refactor where necessary to maintain clean code, efficient code, and to conform to the above settings and directives.
// 11. After making changes, go back and make sure the codebase has been updated to incorporate the new changes and that it still adheres to the coding directives.
// 12. NEVER BREAK FUNCTIONALITY THAT IS ALREADY WORKING.

#include <wx/wx.h>
#include <wx/filedlg.h>
#include <wx/notebook.h>
#include <wx/textctrl.h>
#include <wx/richtext/richtextctrl.h>
#include <wx/button.h>
#include <wx/sizer.h>
#include <wx/panel.h>
#include <wx/stattext.h>
#include <wx/stdpaths.h>
#include <wx/filename.h>
#include <wx/thread.h>
#include <wx/event.h>
#include <wx/gauge.h>
#include <string>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <memory>
#include <atomic>
#include <iostream>
#include <streambuf>
#include "LlamaManager.hpp"
#include "DiscordManager.hpp"
#include "SettingsManager.hpp"

// Forward declarations
class LuminaChatFrame;
class ModelWorkerThread;

// ADDED: Custom stream buffer for redirecting cout/cerr to wxWidgets
class wxLogStreamBuffer : public std::streambuf {
private:
    LuminaChatFrame* frame;
    std::string buffer;
    
public:
    wxLogStreamBuffer(LuminaChatFrame* f) : frame(f) {}
    
protected:
    virtual int_type overflow(int_type c) override {
        if (c != EOF) {
            buffer += static_cast<char>(c);
            if (c == '\n') {
                FlushBuffer();
            }
        }
        return c;
    }
    
    virtual std::streamsize xsputn(const char* s, std::streamsize count) override {
        buffer.append(s, count);
        // Check for newlines and flush if found
        if (buffer.find('\n') != std::string::npos) {
            FlushBuffer();
        }
        return count;
    }
    
private:
    void FlushBuffer(); // Declaration only - implementation after LuminaChatFrame is defined
};

// Event IDs
enum {
    ID_START = 1000,
    ID_STOP,
    ID_SETTINGS,
    ID_INPUT_TEXT,
    ID_BROWSE_MODEL,
    ID_MODEL_LOADED,
    ID_RESPONSE_READY,
    ID_PROGRESS_UPDATE,
    ID_CONNECT_DISCORD
};

// Custom events for thread communication
wxDECLARE_EVENT(wxEVT_MODEL_LOADED, wxCommandEvent);
wxDECLARE_EVENT(wxEVT_RESPONSE_READY, wxCommandEvent);
wxDECLARE_EVENT(wxEVT_PROGRESS_UPDATE, wxCommandEvent);

wxDEFINE_EVENT(wxEVT_MODEL_LOADED, wxCommandEvent);
wxDEFINE_EVENT(wxEVT_RESPONSE_READY, wxCommandEvent);
wxDEFINE_EVENT(wxEVT_PROGRESS_UPDATE, wxCommandEvent);

// Progress callback function for model loading
bool model_loading_progress_callback(float progress, void *user_data);

// Llama.cpp log callback function for logs tab
void llama_log_callback(ggml_log_level level, const char * message, void * user_data);

// Worker thread for model operations
class ModelWorkerThread : public wxThread {
private:
    LlamaManager* llama_manager;
    wxEvtHandler* parent;
    std::string model_path;
    int32_t context_size;
    int32_t gpu_layers;
    int32_t predict_tokens;
    std::string chat_template;
    std::atomic<bool> should_stop{false};

public:
    enum Operation {
        LOAD_MODEL,
        GENERATE_RESPONSE
    };

    Operation operation;
    std::string input_text;
    std::string input_username; // ADDED: Store username for generation
    std::string result;
    bool success = false;

    ModelWorkerThread(wxEvtHandler* parent, LlamaManager* manager, Operation op)
        : wxThread(wxTHREAD_DETACHED), parent(parent), llama_manager(manager), operation(op) {}

    void SetModelParams(const std::string& path, int32_t ctx, int32_t gpu, int32_t pred, const std::string& tmpl) {
        model_path = path;
        context_size = ctx;
        gpu_layers = gpu;
        predict_tokens = pred;
        chat_template = tmpl;
    }

    // UPDATED: Add username parameter for context-aware generation
    void SetInput(const std::string& input, const std::string& username = "User") {
        input_text = input;
        input_username = username;
    }

    void RequestStop() {
        should_stop = true;
    }

protected:
    virtual ExitCode Entry() override {
        if (operation == LOAD_MODEL) {
            success = LoadModel();
            
            wxCommandEvent event(wxEVT_MODEL_LOADED);
            event.SetInt(success ? 1 : 0);
            if (!should_stop) {
                wxQueueEvent(parent, event.Clone());
            }
        } else if (operation == GENERATE_RESPONSE) {
            result = GenerateResponse();
            success = !result.empty() && result.find("Error:") != 0;
            
            wxCommandEvent event(wxEVT_RESPONSE_READY);
            event.SetString(wxString::FromUTF8(result));
            if (!should_stop) {
                wxQueueEvent(parent, event.Clone());
            }
        }
        
        return (ExitCode)0;
    }

private:
    bool LoadModel() {
        if (should_stop) return false;
        
        if (!llama_manager->initialize()) {
            return false;
        }

        // Install the log callback before loading model
        llama_log_set(llama_log_callback, parent);

        llama_manager->set_context_size(context_size);
        llama_manager->set_gpu_layers(gpu_layers);
        llama_manager->set_predict_tokens(predict_tokens);

        if (should_stop) return false;

        // Pass the parent frame directly as void* for progress callback
        if (!llama_manager->load_model(model_path, parent)) {
            return false;
        }

        if (should_stop) return false;

        if (!chat_template.empty()) {
            llama_manager->set_custom_chat_template(chat_template);
        }

        return true;
    }

    std::string GenerateResponse() {
        if (should_stop || input_text.empty()) {
            return "";
        }
        
        // UPDATED: Use username parameter for generation
        return llama_manager->generate_response(input_text, input_username);
    }
};

// MOVED: Global pointer declaration before LuminaChatFrame class
LuminaChatFrame* g_main_frame = nullptr;

// MOVED: Declaration only - implementation after LuminaChatFrame is defined
void llama_manager_log_callback(const std::string& message);

// MOVED: Declaration only - implementation after LuminaChatFrame is defined
void discord_manager_log_callback(const std::string& message);

// ADDED: Settings manager log callback declaration
void settings_manager_log_callback(const std::string& message);

// Settings Dialog
class SettingsDialog : public wxDialog {
private:
    // UI controls
    wxTextCtrl* model_path_text;
    wxTextCtrl* context_size_text;
    wxTextCtrl* gpu_layers_text;
    wxTextCtrl* predict_tokens_text;
    wxTextCtrl* chat_template_text;
    wxTextCtrl* identity_directive_text;
    wxTextCtrl* other_directives_text;
    wxTextCtrl* discord_bot_token_text;
    wxTextCtrl* discord_channel_ids_text;
    wxTextCtrl* discord_isolated_channel_ids_text;
    wxTextCtrl* discord_shared_history_channel_ids_text; // ADDED: New textbox for shared history channels
    wxCheckBox* discord_allow_dms_checkbox;
    
    // References to settings
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
    std::string& discord_shared_history_channel_ids_ref; // ADDED: Reference for shared history channels
    bool& discord_allow_dms_ref;

    // Validation helper
    bool ValidateAndSetInt32(wxTextCtrl* control, int32_t& target, int32_t default_val, 
                           int32_t min_val, int32_t max_val, const wxString& field_name) {
        long value;
        if (control->GetValue().ToLong(&value) && value >= min_val && value <= max_val) {
            target = static_cast<int32_t>(value);
            return true;
        } else {
            wxMessageBox(wxString::Format("Invalid %s. Using default value (%d).", field_name, default_val), 
                        "Warning", wxOK | wxICON_WARNING);
            target = default_val;
            return false;
        }
    }

public:
    SettingsDialog(wxWindow* parent, std::string& model_path, int32_t& context_size, int32_t& gpu_layers, 
                  int32_t& predict_tokens, std::string& chat_template, std::string& identity_directive, 
                  std::string& other_directives, std::string& discord_bot_token, std::string& discord_channel_ids,
                  std::string& discord_isolated_channel_ids, std::string& discord_shared_history_channel_ids, // ADDED: New parameter
                  bool& discord_allow_dms) 
        : wxDialog(parent, wxID_ANY, "Settings", wxDefaultPosition, wxSize(700, 600)),
          model_path_ref(model_path), context_size_ref(context_size), gpu_layers_ref(gpu_layers), 
          predict_tokens_ref(predict_tokens), chat_template_ref(chat_template),
          identity_directive_ref(identity_directive), other_directives_ref(other_directives),
          discord_bot_token_ref(discord_bot_token), discord_channel_ids_ref(discord_channel_ids),
          discord_isolated_channel_ids_ref(discord_isolated_channel_ids), discord_shared_history_channel_ids_ref(discord_shared_history_channel_ids), // ADDED: New reference
          discord_allow_dms_ref(discord_allow_dms) {
        
        wxNotebook* notebook = new wxNotebook(this, wxID_ANY);
        
        // Model Settings Tab
        wxPanel* model_panel = new wxPanel(notebook);
        wxBoxSizer* model_sizer = new wxBoxSizer(wxVERTICAL);
        
        // Model file path
        model_sizer->Add(new wxStaticText(model_panel, wxID_ANY, "Model File Path:"), 0, wxALL, 5);
        
        wxBoxSizer* path_sizer = new wxBoxSizer(wxHORIZONTAL);
        model_path_text = new wxTextCtrl(model_panel, wxID_ANY, model_path, wxDefaultPosition, wxDefaultSize, wxTE_READONLY);
        wxButton* browse_btn = new wxButton(model_panel, ID_BROWSE_MODEL, "Browse...");
        
        path_sizer->Add(model_path_text, 1, wxEXPAND | wxRIGHT, 5);
        path_sizer->Add(browse_btn, 0, wxALIGN_CENTER_VERTICAL);
        
        model_sizer->Add(path_sizer, 0, wxEXPAND | wxALL, 5);
        
        // Context size
        model_sizer->Add(new wxStaticText(model_panel, wxID_ANY, "Context Size (tokens):"), 0, wxALL, 5);
        context_size_text = new wxTextCtrl(model_panel, wxID_ANY, wxString::Format("%d", context_size));
        model_sizer->Add(context_size_text, 0, wxEXPAND | wxALL, 5);
        
        // GPU layers
        model_sizer->Add(new wxStaticText(model_panel, wxID_ANY, "GPU Offload Layers (0 = CPU only):"), 0, wxALL, 5);
        gpu_layers_text = new wxTextCtrl(model_panel, wxID_ANY, wxString::Format("%d", gpu_layers));
        model_sizer->Add(gpu_layers_text, 0, wxEXPAND | wxALL, 5);
        
        // Predict tokens
        model_sizer->Add(new wxStaticText(model_panel, wxID_ANY, "Max Prediction Tokens:"), 0, wxALL, 5);
        predict_tokens_text = new wxTextCtrl(model_panel, wxID_ANY, wxString::Format("%d", predict_tokens));
        model_sizer->Add(predict_tokens_text, 0, wxEXPAND | wxALL, 5);
        
        model_panel->SetSizer(model_sizer);
        notebook->AddPage(model_panel, "Model Settings");
        
        // System Prompt Tab
        wxPanel* system_panel = new wxPanel(notebook);
        wxBoxSizer* system_sizer = new wxBoxSizer(wxVERTICAL);
        
        // Identity Directive (1/3 height)
        system_sizer->Add(new wxStaticText(system_panel, wxID_ANY, "Identity Directive:"), 0, wxALL, 5);
        identity_directive_text = new wxTextCtrl(system_panel, wxID_ANY, wxString::FromUTF8(identity_directive), 
                                               wxDefaultPosition, wxDefaultSize, 
                                               wxTE_MULTILINE | wxTE_WORDWRAP);
        identity_directive_text->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
        system_sizer->Add(identity_directive_text, 1, wxEXPAND | wxALL, 5);
        
        // Other Directives (2/3 height)
        system_sizer->Add(new wxStaticText(system_panel, wxID_ANY, "Other Directives:"), 0, wxALL, 5);
        other_directives_text = new wxTextCtrl(system_panel, wxID_ANY, wxString::FromUTF8(other_directives), 
                                             wxDefaultPosition, wxDefaultSize, 
                                             wxTE_MULTILINE | wxTE_WORDWRAP);
        other_directives_text->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
        system_sizer->Add(other_directives_text, 2, wxEXPAND | wxALL, 5);
        
        system_panel->SetSizer(system_sizer);
        notebook->AddPage(system_panel, "System Prompt");
        
        // Chat Template Tab
        wxPanel* template_panel = new wxPanel(notebook);
        wxBoxSizer* template_sizer = new wxBoxSizer(wxVERTICAL);
        
        template_sizer->Add(new wxStaticText(template_panel, wxID_ANY, "Chat Template (Jinja2 format):"), 0, wxALL, 5);
        template_sizer->Add(new wxStaticText(template_panel, wxID_ANY, "Leave empty to use model's default template"), 0, wxALL, 5);
        
        chat_template_text = new wxTextCtrl(template_panel, wxID_ANY, wxString::FromUTF8(chat_template), 
                                           wxDefaultPosition, wxDefaultSize, 
                                           wxTE_MULTILINE | wxTE_WORDWRAP);
        chat_template_text->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
        
        template_sizer->Add(chat_template_text, 1, wxEXPAND | wxALL, 5);
        
        // Add helpful text
        wxStaticText* help_text = new wxStaticText(template_panel, wxID_ANY, 
            "Common variables: {{ messages }}, {{ add_generation_prompt }}\n"
            "Example: {% for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}");
        help_text->SetFont(wxFont(8, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_ITALIC, wxFONTWEIGHT_NORMAL));
        template_sizer->Add(help_text, 0, wxALL, 5);
        
        template_panel->SetSizer(template_sizer);
        notebook->AddPage(template_panel, "Chat Template");
        
        // Discord Settings Tab
        wxPanel* discord_panel = new wxPanel(notebook);
        wxBoxSizer* discord_sizer = new wxBoxSizer(wxVERTICAL);
        
        // Bot Token
        discord_sizer->Add(new wxStaticText(discord_panel, wxID_ANY, "Discord Bot Token:"), 0, wxALL, 5);
        discord_sizer->Add(new wxStaticText(discord_panel, wxID_ANY, "Enter your Discord bot's authentication token"), 0, wxLEFT | wxRIGHT | wxBOTTOM, 5);
        
        discord_bot_token_text = new wxTextCtrl(discord_panel, wxID_ANY, wxString::FromUTF8(discord_bot_token), 
                                               wxDefaultPosition, wxDefaultSize, 
                                               wxTE_PASSWORD); // Hide token for security
        discord_bot_token_text->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
        discord_sizer->Add(discord_bot_token_text, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 5);
        
        // Allow DMs checkbox
        discord_allow_dms_checkbox = new wxCheckBox(discord_panel, wxID_ANY, "Allow Direct Messages (DMs)");
        discord_allow_dms_checkbox->SetValue(discord_allow_dms);
        discord_sizer->Add(discord_allow_dms_checkbox, 0, wxALL, 5);
        discord_sizer->Add(new wxStaticText(discord_panel, wxID_ANY, "When unchecked, the bot will auto-reply to DMs that the feature is disabled"), 0, wxLEFT | wxRIGHT | wxBOTTOM, 5);
        
        // Channel IDs
        discord_sizer->Add(new wxStaticText(discord_panel, wxID_ANY, "Allowed Channel IDs (comma separated):"), 0, wxALL, 5);
        discord_sizer->Add(new wxStaticText(discord_panel, wxID_ANY, "List of Discord channel IDs where the bot should respond (leave empty for all channels)"), 0, wxLEFT | wxRIGHT | wxBOTTOM, 5);
        
        discord_channel_ids_text = new wxTextCtrl(discord_panel, wxID_ANY, wxString::FromUTF8(discord_channel_ids), 
                                                 wxDefaultPosition, wxSize(-1, 60), 
                                                 wxTE_MULTILINE | wxTE_WORDWRAP);
        discord_channel_ids_text->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
        discord_sizer->Add(discord_channel_ids_text, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 5);
        
        // Isolated Context Channel IDs
        discord_sizer->Add(new wxStaticText(discord_panel, wxID_ANY, "Isolated Context Channel IDs (comma separated):"), 0, wxALL, 5);
        discord_sizer->Add(new wxStaticText(discord_panel, wxID_ANY, "Channels that get their own separate context (each user gets individual context)"), 0, wxLEFT | wxRIGHT | wxBOTTOM, 5);
        
        discord_isolated_channel_ids_text = new wxTextCtrl(discord_panel, wxID_ANY, wxString::FromUTF8(discord_isolated_channel_ids), 
                                                           wxDefaultPosition, wxSize(-1, 60), 
                                                           wxTE_MULTILINE | wxTE_WORDWRAP);
        discord_isolated_channel_ids_text->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
        discord_sizer->Add(discord_isolated_channel_ids_text, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 5);
        
        // Shared History Channel IDs
        discord_sizer->Add(new wxStaticText(discord_panel, wxID_ANY, "Shared History Channel IDs (comma separated):"), 0, wxALL, 5);
        discord_sizer->Add(new wxStaticText(discord_panel, wxID_ANY, "Channels to pull message history from (leave empty for none)"), 0, wxLEFT | wxRIGHT | wxBOTTOM, 5);
        
        discord_shared_history_channel_ids_text = new wxTextCtrl(discord_panel, wxID_ANY, wxString::FromUTF8(discord_shared_history_channel_ids), 
                                                                 wxDefaultPosition, wxSize(-1, 60), 
                                                                 wxTE_MULTILINE | wxTE_WORDWRAP);
        discord_shared_history_channel_ids_text->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
        discord_sizer->Add(discord_shared_history_channel_ids_text, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 5);
        
        // Help text for Discord settings
        wxStaticText* discord_help = new wxStaticText(discord_panel, wxID_ANY, 
            "Instructions:\n"
            "1. Create a Discord application at https://discord.com/developers/applications\n"
            "2. Create a bot and copy the token above\n"
            "3. Right-click Discord channels and 'Copy ID' to get channel IDs\n"
            "4. Separate multiple channel IDs with commas (e.g., 123456789,987654321)\n"
            "5. Channels NOT in 'Isolated Context' list will use the shared main chat context\n"
            "6. Channels IN 'Isolated Context' list will get their own separate context (shared by all users in that channel)\n"
            "7. Direct Messages (DMs) use individual isolated contexts when enabled (one per user for privacy)\n"
            "8. When DMs are disabled, users will receive an auto-reply explaining the feature is turned off");
        discord_help->SetFont(wxFont(8, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_ITALIC, wxFONTWEIGHT_NORMAL));
        discord_sizer->Add(discord_help, 1, wxEXPAND | wxALL, 5);
        
        discord_panel->SetSizer(discord_sizer);
        notebook->AddPage(discord_panel, "Discord Settings");
        
        // Dialog buttons
        wxBoxSizer* btn_sizer = new wxBoxSizer(wxHORIZONTAL);
        btn_sizer->Add(new wxButton(this, wxID_OK, "OK"), 0, wxRIGHT, 5);
        btn_sizer->Add(new wxButton(this, wxID_CANCEL, "Cancel"), 0);
        
        wxBoxSizer* main_sizer = new wxBoxSizer(wxVERTICAL);
        main_sizer->Add(notebook, 1, wxEXPAND | wxALL, 10);
        main_sizer->Add(btn_sizer, 0, wxALIGN_RIGHT | wxALL, 10);
        
        SetSizer(main_sizer);
        
        Bind(wxEVT_COMMAND_BUTTON_CLICKED, &SettingsDialog::OnBrowseModel, this, ID_BROWSE_MODEL);
        Bind(wxEVT_COMMAND_BUTTON_CLICKED, &SettingsDialog::OnOK, this, wxID_OK);
    }

private:
    void OnBrowseModel(wxCommandEvent& event) {
        wxFileDialog file_dialog(this, "Choose Model File", "", "", 
                                "GGUF files (*.gguf)|*.gguf|All files (*.*)|*.*",
                                wxFD_OPEN | wxFD_FILE_MUST_EXIST);
        
        if (file_dialog.ShowModal() == wxID_OK) {
            model_path_text->SetValue(file_dialog.GetPath());
        }
    }
    
    void OnOK(wxCommandEvent& event) {
        model_path_ref = model_path_text->GetValue().ToStdString();
        
        // Validate all numeric fields
        ValidateAndSetInt32(context_size_text, context_size_ref, 2048, 1, 131072, "context size");
        ValidateAndSetInt32(gpu_layers_text, gpu_layers_ref, 0, 0, 999, "GPU layers");
        ValidateAndSetInt32(predict_tokens_text, predict_tokens_ref, 256, 1, 4096, "prediction tokens");
        
        // Get text fields
        chat_template_ref = chat_template_text->GetValue().ToUTF8().data();
        identity_directive_ref = identity_directive_text->GetValue().ToUTF8().data();
        other_directives_ref = other_directives_text->GetValue().ToUTF8().data();
        discord_bot_token_ref = discord_bot_token_text->GetValue().ToUTF8().data();
        discord_channel_ids_ref = discord_channel_ids_text->GetValue().ToUTF8().data();
        discord_isolated_channel_ids_ref = discord_isolated_channel_ids_text->GetValue().ToUTF8().data();
        discord_shared_history_channel_ids_ref = discord_shared_history_channel_ids_text->GetValue().ToUTF8().data(); // ADDED: Get shared history channels
        discord_allow_dms_ref = discord_allow_dms_checkbox->GetValue();
        
        // Validate Discord channel IDs format if provided
        if (!discord_channel_ids_ref.empty()) {
            std::string cleaned_ids = ValidateChannelIds(discord_channel_ids_ref);
            if (cleaned_ids != discord_channel_ids_ref) {
                discord_channel_ids_ref = cleaned_ids;
                wxMessageBox("Allowed Channel IDs have been cleaned up. Invalid entries were removed.", 
                           "Channel IDs Modified", wxOK | wxICON_INFORMATION);
            }
        }
        
        // Validate isolated channel IDs format if provided
        if (!discord_isolated_channel_ids_ref.empty()) {
            std::string cleaned_ids = ValidateChannelIds(discord_isolated_channel_ids_ref);
            if (cleaned_ids != discord_isolated_channel_ids_ref) {
                discord_isolated_channel_ids_ref = cleaned_ids;
                wxMessageBox("Isolated Context Channel IDs have been cleaned up. Invalid entries were removed.", 
                           "Channel IDs Modified", wxOK | wxICON_INFORMATION);
            }
        }
        
        // Validate shared history channel IDs format if provided
        if (!discord_shared_history_channel_ids_ref.empty()) {
            std::string cleaned_ids = ValidateChannelIds(discord_shared_history_channel_ids_ref);
            if (cleaned_ids != discord_shared_history_channel_ids_ref) {
                discord_shared_history_channel_ids_ref = cleaned_ids;
                wxMessageBox("Shared History Channel IDs have been cleaned up. Invalid entries were removed.", 
                           "Channel IDs Modified", wxOK | wxICON_INFORMATION);
            }
        }
        
        // Save settings
        SettingsManager::SaveSettings(model_path_ref, context_size_ref, gpu_layers_ref, 
                                    predict_tokens_ref, chat_template_ref, 
                                    identity_directive_ref, other_directives_ref,
                                    discord_bot_token_ref, discord_channel_ids_ref,
                                    discord_isolated_channel_ids_ref, discord_shared_history_channel_ids_ref, discord_allow_dms_ref);
        
        EndModal(wxID_OK);
    }
    
    // Validate and clean up channel IDs format
    std::string ValidateChannelIds(const std::string& input) {
        std::string result;
        std::string current_id;
        
        for (char c : input) {
            if (std::isdigit(c)) {
                current_id += c;
            } else if (c == ',' || c == ' ' || c == '\n' || c == '\r' || c == '\t') {
                if (!current_id.empty()) {
                    if (!result.empty()) result += ",";
                    result += current_id;
                    current_id.clear();
                }
            }
            // Skip other characters
        }
        
        // Add last ID if any
        if (!current_id.empty()) {
            if (!result.empty()) result += ",";
            result += current_id;
        }
        
        return result;
    }
};

// Main Frame with optimized performance
class LuminaChatFrame : public wxFrame {
private:
    // Core components
    std::unique_ptr<LlamaManager> llama_manager;
    std::unique_ptr<DiscordManager> discord_manager;
    
    // ADDED: Stream buffers for console redirection
    std::unique_ptr<wxLogStreamBuffer> cout_buffer;
    std::unique_ptr<wxLogStreamBuffer> cerr_buffer;
    std::streambuf* original_cout;
    std::streambuf* original_cerr;
    
    // Settings
    std::string model_path;
    int32_t context_size;
    int32_t gpu_layers;
    int32_t predict_tokens;
    std::string chat_template;
    std::string identity_directive;
    std::string other_directives;
    std::string discord_bot_token;
    std::string discord_channel_ids;
    std::string discord_isolated_channel_ids;
    std::string discord_shared_history_channel_ids; // ADDED: New setting for shared channels with history backfill
    bool discord_allow_dms = true; // ADDED: New setting for DM handling

    // State
    bool is_started;
    std::atomic<bool> is_processing{false};
    
    // UI controls
    wxButton* start_btn;
    wxButton* stop_btn;
    wxButton* settings_btn;
    wxButton* discord_btn;
    wxGauge* progress_bar;
    wxStaticText* progress_label;
    wxStaticText* timings_label;
    wxRichTextCtrl* chat_history;
    wxTextCtrl* input_text;
    wxTextCtrl* logs_text;
    wxNotebook* main_notebook;
    wxPanel* main_panel;  // Store reference to main panel
    
    ModelWorkerThread* worker_thread = nullptr;

    // ADDED: Context management
    static constexpr const char* DEFAULT_CONTEXT_ID = "main_chat";
    bool context_created = false;

public:
    LuminaChatFrame() : wxFrame(nullptr, wxID_ANY, "LuminaChat", wxDefaultPosition, wxSize(800, 600)),
                        llama_manager(std::make_unique<LlamaManager>()),
                        discord_manager(std::make_unique<DiscordManager>()),
                        context_size(2048), gpu_layers(0), predict_tokens(256), is_started(false),
                        original_cout(nullptr), original_cerr(nullptr) {
        
        // ADDED: Set global pointer for log callback
        g_main_frame = this;
        
        SettingsManager::LoadSettings(model_path, context_size, gpu_layers, predict_tokens, 
                                    chat_template, identity_directive, other_directives,
                                    discord_bot_token, discord_channel_ids, discord_isolated_channel_ids, 
                                    discord_shared_history_channel_ids, discord_allow_dms);
        
        CreateUI();
        
        // ADDED: Redirect cout and cerr to logs panel after UI is created
        SetupConsoleRedirection();
        
        UpdateButtonStates();
        UpdateWindowTitle();

        // Bind custom events
        Bind(wxEVT_MODEL_LOADED, &LuminaChatFrame::OnModelLoaded, this);
        Bind(wxEVT_RESPONSE_READY, &LuminaChatFrame::OnResponseReady, this);
        Bind(wxEVT_PROGRESS_UPDATE, &LuminaChatFrame::OnProgressUpdate, this);
    }

    ~LuminaChatFrame() {
        // ADDED: Restore original cout/cerr
        if (original_cout) {
            std::cout.rdbuf(original_cout);
        }
        if (original_cerr) {
            std::cerr.rdbuf(original_cerr);
        }
        
        // ADDED: Clear global pointer
        g_main_frame = nullptr;
        
        if (worker_thread) {
            worker_thread->RequestStop();
            worker_thread = nullptr;
        }
    }

    // Add method to append to logs from callback (thread-safe)
    void AppendToLogsThreadSafe(const wxString& message) {
        // Use CallAfter to ensure this runs on the main thread
        CallAfter([this, message]() {
            if (logs_text) {
                logs_text->AppendText(message);
                logs_text->SetInsertionPointEnd();
            }
        });
    }

private:
    // ADDED: Setup console output redirection
    void SetupConsoleRedirection() {
        // Store original stream buffers
        original_cout = std::cout.rdbuf();
        original_cerr = std::cerr.rdbuf();
        
        // Create new stream buffers that redirect to wxWidgets
        cout_buffer = std::make_unique<wxLogStreamBuffer>(this);
        cerr_buffer = std::make_unique<wxLogStreamBuffer>(this);
        
        // Redirect cout and cerr
        std::cout.rdbuf(cout_buffer.get());
        std::cerr.rdbuf(cerr_buffer.get());
        
        // Add initial message to logs
        AppendToLogsThreadSafe("Console output redirected to logs panel\n");
    }

    void UpdateWindowTitle() {
        if (!model_path.empty()) {
            wxFileName modelFile(wxString::FromUTF8(model_path));
            SetTitle(wxString::Format("LuminaChat - %s", modelFile.GetName()));
        } else {
            SetTitle("LuminaChat");
        }
    }
    
    std::string GetCombinedSystemPrompt() const {
        std::string combined;
        if (!identity_directive.empty()) {
            combined = identity_directive;
            if (!other_directives.empty()) {
                combined += "\n\n" + other_directives;
            }
        } else if (!other_directives.empty()) {
            combined = other_directives;
        }
        return combined;
    }
    
    void SetDefaultSystemPromptIfEmpty() {
        if (identity_directive.empty() && other_directives.empty()) {
            identity_directive = "You are a helpful AI assistant named Lumina.";
            other_directives = "Answer each user request thoughtfully and to the best of your ability. Be clear and concise in your responses.";
            
            SettingsManager::SaveSettings(model_path, context_size, gpu_layers, predict_tokens, 
                                        chat_template, identity_directive, other_directives,
                                        discord_bot_token, discord_channel_ids, discord_isolated_channel_ids, 
                                        discord_shared_history_channel_ids, discord_allow_dms); // ADDED: Save new setting
        }
    }

    void CreateUI() {
        main_panel = new wxPanel(this);
        
        // Top toolbar
        wxBoxSizer* toolbar_sizer = new wxBoxSizer(wxHORIZONTAL);
        start_btn = new wxButton(main_panel, ID_START, "Start");
        stop_btn = new wxButton(main_panel, ID_STOP, "Stop");
        settings_btn = new wxButton(main_panel, ID_SETTINGS, "Settings");
        discord_btn = new wxButton(main_panel, ID_CONNECT_DISCORD, "Connect Discord");
        
        toolbar_sizer->Add(start_btn, 0, wxRIGHT, 5);
        toolbar_sizer->Add(stop_btn, 0, wxRIGHT, 5);
        toolbar_sizer->Add(settings_btn, 0, wxRIGHT, 5);
        toolbar_sizer->Add(discord_btn, 0);
        toolbar_sizer->AddStretchSpacer();
        
        // Progress bar and label
        progress_label = new wxStaticText(main_panel, wxID_ANY, "Ready");
        progress_bar = new wxGauge(main_panel, wxID_ANY, 100, wxDefaultPosition, wxSize(-1, 20));
        progress_bar->SetValue(0);
        progress_bar->Hide(); // Initially hidden
        
        // Timings label
        timings_label = new wxStaticText(main_panel, wxID_ANY, "");
        
        // Create notebook with tabs
        main_notebook = new wxNotebook(main_panel, wxID_ANY);
        
        // Chat Tab
        wxPanel* chat_panel = new wxPanel(main_notebook);
        wxBoxSizer* chat_sizer = new wxBoxSizer(wxVERTICAL);
        
        // Chat history (read-only rich text control for colored backgrounds)
        chat_history = new wxRichTextCtrl(chat_panel, wxID_ANY, "", wxDefaultPosition, wxDefaultSize,
                                         wxRE_READONLY | wxRE_MULTILINE | wxVSCROLL);
        chat_history->SetFont(wxFont(10, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
        
        // Input text box
        input_text = new wxTextCtrl(chat_panel, ID_INPUT_TEXT, "", wxDefaultPosition, wxDefaultSize,
                                   wxTE_PROCESS_ENTER);
        
        chat_sizer->Add(chat_history, 1, wxEXPAND | wxALL, 5);
        chat_sizer->Add(input_text, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 5);
        
        chat_panel->SetSizer(chat_sizer);
        main_notebook->AddPage(chat_panel, "Chat");
        
        // Logs Tab
        wxPanel* logs_panel = new wxPanel(main_notebook);
        wxBoxSizer* logs_sizer = new wxBoxSizer(wxVERTICAL);
        
        // Logs text box (read-only)
        logs_text = new wxTextCtrl(logs_panel, wxID_ANY, "", wxDefaultPosition, wxDefaultSize,
                                  wxTE_MULTILINE | wxTE_READONLY | wxTE_WORDWRAP);
        logs_text->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
        
        logs_sizer->Add(logs_text, 1, wxEXPAND | wxALL, 5);
        
        logs_panel->SetSizer(logs_sizer);
        main_notebook->AddPage(logs_panel, "Logs");
        
        // Layout - fixed alignment flags to prevent assertion
        wxBoxSizer* main_sizer = new wxBoxSizer(wxVERTICAL);
        main_sizer->Add(toolbar_sizer, 0, wxEXPAND | wxALL, 10);
        
        // Progress and timings row - fixed incompatible alignment flags
        wxBoxSizer* status_sizer = new wxBoxSizer(wxHORIZONTAL);
        status_sizer->Add(progress_label, 0, wxALIGN_CENTER_VERTICAL);
        status_sizer->AddSpacer(20);
        // Remove wxEXPAND when using wxALIGN_CENTER_VERTICAL in horizontal sizer
        status_sizer->Add(timings_label, 1, wxALIGN_CENTER_VERTICAL);
        
        main_sizer->Add(status_sizer, 0, wxEXPAND | wxLEFT | wxRIGHT, 10);
        
        // Progress bar in its own sizer with proper show/hide behavior
        main_sizer->Add(progress_bar, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 10);
        
        main_sizer->Add(main_notebook, 1, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 10);
        
        main_panel->SetSizer(main_sizer);
        
        // Set minimum window size to prevent UI from becoming unusable
        SetMinSize(wxSize(600, 400));
        
        // Bind events
        Bind(wxEVT_COMMAND_BUTTON_CLICKED, &LuminaChatFrame::OnStart, this, ID_START);
        Bind(wxEVT_COMMAND_BUTTON_CLICKED, &LuminaChatFrame::OnStop, this, ID_STOP);
        Bind(wxEVT_COMMAND_BUTTON_CLICKED, &LuminaChatFrame::OnSettings, this, ID_SETTINGS);
        Bind(wxEVT_COMMAND_BUTTON_CLICKED, &LuminaChatFrame::OnConnectDiscord, this, ID_CONNECT_DISCORD);
        Bind(wxEVT_COMMAND_TEXT_ENTER, &LuminaChatFrame::OnInputEnter, this, ID_INPUT_TEXT);
        
        // Set initial message
        AddWelcomeMessage();
    }
    
    // Add welcome message with proper formatting
    void AddWelcomeMessage() {
        wxRichTextAttr attr;
        attr.SetTextColour(*wxBLACK);
        attr.SetBackgroundColour(wxColour(240, 240, 240)); // Light gray background
        attr.SetAlignment(wxTEXT_ALIGNMENT_CENTER);
        attr.SetParagraphSpacingAfter(5);
        
        chat_history->BeginStyle(attr);
        chat_history->WriteText("Welcome to LuminaChat!\nClick 'Settings' to select a model, then 'Start' to begin.");
        chat_history->EndStyle();
        chat_history->Newline();
        chat_history->Newline();
    }
    
    // Add AI message with light green background
    void AddAIMessage(const wxString& message) {
        // Create AI message style with light green background
        wxRichTextAttr aiAttr;
        aiAttr.SetTextColour(*wxBLACK);
        aiAttr.SetBackgroundColour(wxColour(144, 238, 144)); // Light green
        aiAttr.SetLeftIndent(50);   // Indent from left
        aiAttr.SetRightIndent(50);  // Indent from right
        aiAttr.SetParagraphSpacingBefore(0);  // Remove all paragraph spacing
        aiAttr.SetParagraphSpacingAfter(0);   // Remove all paragraph spacing
        aiAttr.SetLineSpacing(100); // Normal line spacing (100%)
        
        // Split message by double newlines (paragraph breaks) and single newlines
        wxArrayString paragraphs = wxSplit(message, '\n', '\0');
        
        // Start the AI message
        chat_history->BeginStyle(aiAttr);
        chat_history->WriteText("AI: ");

        // Add each paragraph/line separately
        for (size_t i = 0; i < paragraphs.GetCount(); ++i) {
            wxString para = paragraphs[i].Trim();
            
            if (!para.IsEmpty()) {
                if (i > 0) {
                    // Add a small space between non-empty paragraphs
                    chat_history->WriteText(" ");
                }
                chat_history->WriteText(para);
            } else if (i > 0 && i < paragraphs.GetCount() - 1) {
                // For empty lines (paragraph breaks), add a bit more space
                chat_history->WriteText(" - - ");
            }
        }
        
        chat_history->EndStyle();
        
        // Manual spacing between messages
        chat_history->WriteText("\n\n");
    }
    
    // Add user message with light blue background
    void AddUserMessage(const wxString& message) {
        // Create user message style with light blue background
        wxRichTextAttr userAttr;
        userAttr.SetTextColour(*wxBLACK);
        userAttr.SetBackgroundColour(wxColour(173, 216, 230)); // Light blue
        userAttr.SetLeftIndent(50);   // Indent from left
        userAttr.SetRightIndent(50);  // Indent from right
        userAttr.SetParagraphSpacingBefore(0);  // Remove all paragraph spacing
        userAttr.SetParagraphSpacingAfter(0);   // Remove all paragraph spacing
        userAttr.SetLineSpacing(100); // Normal line spacing (100%)
        
        // Add spacing before user message
        chat_history->WriteText("\n");
        
        // Split message by newlines for user messages too
        wxArrayString paragraphs = wxSplit(message, '\n', '\0');
        
        // Start the user message
        chat_history->BeginStyle(userAttr);
        chat_history->WriteText("You: ");
        
        // Add each paragraph/line separately
        for (size_t i = 0; i < paragraphs.GetCount(); ++i) {
            wxString para = paragraphs[i].Trim();
            
            if (!para.IsEmpty()) {
                if (i > 0) {
                    chat_history->WriteText(" ");
                }
                chat_history->WriteText(para);
            } else if (i > 0 && i < paragraphs.GetCount() - 1) {
                chat_history->WriteText("  ");
            }
        }
        
        chat_history->EndStyle();
        
        // Single newline after user message
        chat_history->WriteText("\n");
    }
    
    // Add system message with gray background
    void AddSystemMessage(const wxString& message) {
        wxRichTextAttr systemAttr;
        systemAttr.SetTextColour(wxColour(100, 100, 100)); // Dark gray text
        systemAttr.SetBackgroundColour(wxColour(245, 245, 245)); // Very light gray
        systemAttr.SetLeftIndent(30);
        systemAttr.SetRightIndent(30);
        systemAttr.SetParagraphSpacingBefore(0);  // Remove all paragraph spacing
        systemAttr.SetParagraphSpacingAfter(0);   // Remove all paragraph spacing
        systemAttr.SetLineSpacing(100); // Normal line spacing
        systemAttr.SetFontStyle(wxFONTSTYLE_ITALIC);
        
        chat_history->BeginStyle(systemAttr);
        chat_history->WriteText("System: " + message);
        chat_history->EndStyle();
        chat_history->WriteText("\n\n");
    }

private:
    void UpdateButtonStates() {
        start_btn->Enable(!is_started && !is_processing);
        stop_btn->Enable(is_started);
        input_text->Enable(is_started && !is_processing);
        settings_btn->Enable(!is_processing);
        
        // Update Discord button text and state
        if (discord_manager && discord_manager->is_bot_running()) {
            discord_btn->SetLabel("Disconnect Discord");
            discord_btn->Enable(true);
        } else {
            discord_btn->SetLabel("Connect Discord");
            discord_btn->Enable(true);
        }
    }
    
    void OnStart(wxCommandEvent& event) {
        if (model_path.empty()) {
            wxMessageBox("Please select a model file in Settings first.", "No Model Selected", 
                        wxOK | wxICON_WARNING);
            return;
        }
        
        if (is_processing) return;

        SetDefaultSystemPromptIfEmpty();

        is_processing = true;
        UpdateButtonStates();
        
        // Show progress bar with proper layout update
        progress_label->SetLabel("Loading model...");
        progress_bar->SetValue(0);
        progress_bar->Show();
        main_panel->Layout();
        
        // ADDED: Use cout for logging (will be redirected to logs panel)
        std::cout << "Initializing LuminaChat..." << std::endl;
        std::cout << "Context Size: " << context_size << " tokens" << std::endl;
        std::cout << "GPU Layers: " << gpu_layers << std::endl;
        std::cout << "Max Prediction: " << predict_tokens << " tokens" << std::endl;
        std::cout << "Loading model: " << model_path << std::endl;
        
        if (!GetCombinedSystemPrompt().empty()) {
            std::cout << "System prompt configured" << std::endl;
        }
        
        // Start model loading thread
        worker_thread = new ModelWorkerThread(this, llama_manager.get(), ModelWorkerThread::LOAD_MODEL);
        worker_thread->SetModelParams(model_path, context_size, gpu_layers, predict_tokens, chat_template);
        
        if (worker_thread->Run() != wxTHREAD_NO_ERROR) {
            delete worker_thread;
            worker_thread = nullptr;
            is_processing = false;
            progress_bar->Hide();
            progress_label->SetLabel("Ready");
            main_panel->Layout();
            UpdateButtonStates();
            AddSystemMessage("Error: Failed to start model loading thread");
        }
    }
    
    void OnProgressUpdate(wxCommandEvent& event) {
        int32_t percent = event.GetInt();
        progress_bar->SetValue(percent);
        progress_label->SetLabel(wxString::Format("Loading model... %d%%", percent));
        
        // Force UI update with proper refresh
        progress_bar->Refresh();
        progress_label->Refresh();
    }
    
    void OnModelLoaded(wxCommandEvent& event) {
        is_processing = false;
        worker_thread = nullptr;
        
        // Hide progress bar with proper layout update
        progress_bar->Hide();
        progress_label->SetLabel("Ready");
        main_panel->Layout();
        
        bool success = event.GetInt() == 1;
        
        if (success) {
            // Handle chat template first
            if (chat_template.empty()) {
                std::string model_template = llama_manager->get_model_chat_template();
                if (!model_template.empty()) {
                    chat_template = model_template;
                    SettingsManager::SaveSettings(model_path, context_size, gpu_layers, predict_tokens, 
                                                chat_template, identity_directive, other_directives,
                                                discord_bot_token, discord_channel_ids, discord_isolated_channel_ids, 
                                                discord_shared_history_channel_ids, discord_allow_dms);
                    std::cout << "Loaded chat template from model" << std::endl;
                }
            } else {
                llama_manager->set_custom_chat_template(chat_template);
                std::cout << "Using custom chat template" << std::endl;
            }
            
            // SIMPLIFIED: Create main chat context with system prompt from settings
            std::string combined_prompt = GetCombinedSystemPrompt();
            if (!llama_manager->create_context(DEFAULT_CONTEXT_ID, combined_prompt)) {
                std::cerr << "Error: Failed to create main chat context" << std::endl;
                success = false;
            } else {
                context_created = true;
                std::cout << "Main chat context created with system prompt" << std::endl;
            }
        }
        
        if (success) {
            is_started = true;
            
            // Connect Discord manager to LlamaManager when model is loaded
            if (discord_manager && discord_manager->is_bot_running()) {
                discord_manager->set_llama_manager(llama_manager.get());
                discord_manager->set_main_context_id(DEFAULT_CONTEXT_ID);
                std::cout << "Discord bot connected to loaded model" << std::endl;
            }
            
            chat_history->Clear();
            AddSystemMessage("LuminaChat ready! Type your message below.");
            input_text->SetFocus();
            
            llama_manager->reset_timings();
            timings_label->SetLabel("");
        } else {
            std::cerr << "Error: Failed to load model or create context" << std::endl;
        }
        
        UpdateButtonStates();
    }
    
    void OnStop(wxCommandEvent& event) {
        if (worker_thread) {
            worker_thread->RequestStop();
            worker_thread = nullptr;
        }
        
        // Hide progress bar if visible with proper layout update
        if (progress_bar->IsShown()) {
            progress_bar->Hide();
            progress_label->SetLabel("Ready");
            main_panel->Layout(); // Use main_panel->Layout() instead of GetSizer()->Layout()
        }
        
        // ADDED: Disconnect Discord manager from LlamaManager when model is stopped
        if (discord_manager && discord_manager->is_bot_running()) {
            discord_manager->set_llama_manager(nullptr);
            std::cout << "Discord bot disconnected from model" << std::endl;
        }
        
        llama_manager->cleanup();
        context_created = false; // ADDED: Reset context state
        is_started = false;
        is_processing = false;
        UpdateButtonStates();
        std::cout << "LuminaChat stopped." << std::endl << std::endl;
        timings_label->SetLabel("");
    }
    
    void OnConnectDiscord(wxCommandEvent& event) {
        if (!discord_manager) {
            AddSystemMessage("Error: Discord manager not available");
            return;
        }
        
        if (discord_manager->is_bot_running()) {
            // Stop Discord bot
            discord_manager->shutdown();
            discord_btn->SetLabel("Connect Discord");
            AddSystemMessage("Discord bot disconnected");
        } else {
            // Configure and start Discord bot
            if (discord_bot_token.empty()) {
                AddSystemMessage("Discord bot token not configured. Please check Settings.");
                return;
            }
            
            // Configure bot
            DiscordBotConfig config;
            config.bot_token = discord_bot_token;
            
            if (!discord_manager->configure(config)) {
                AddSystemMessage("Failed to configure Discord bot");
                return;
            }
            
            // Set up integration with LlamaManager
            discord_manager->set_llama_manager(llama_manager.get());
            discord_manager->set_main_context_id(DEFAULT_CONTEXT_ID);
            discord_manager->set_allowed_channels(discord_channel_ids);
            discord_manager->set_isolated_channels(discord_isolated_channel_ids);
            discord_manager->set_shared_history_channels(discord_shared_history_channel_ids); // ADDED: Set shared history channels
            discord_manager->set_allow_dms(discord_allow_dms);
            
            // Start bot
            if (discord_manager->start()) {
                discord_btn->SetLabel("Disconnect Discord");
                AddSystemMessage("Discord bot connecting...");
                
                // Start a timer to periodically report backfill status
                std::thread([this]() {
                    bool backfill_started = false;
                    for (int i = 0; i < 60; ++i) { // Check for up to 60 seconds
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                        if (discord_manager) {
                            auto status = discord_manager->get_backfill_status();
                            
                            if (status.in_progress) {
                                if (!backfill_started) {
                                    std::cout << "Chat history backfill started for accessible channels..." << std::endl;
                                    backfill_started = true;
                                }
                                
                                if (i % 5 == 0 && status.channels_processed > 0) { // Report every 5 seconds
                                    std::cout << "Backfill progress: " << status.channels_complete 
                                             << "/" << status.channels_processed << " channels processed, " 
                                             << status.total_messages_fetched << " messages fetched" << std::endl;
                                }
                            } else if (backfill_started || status.total_messages_fetched > 0) {
                                std::cout << "Chat history backfill completed: " 
                                         << status.total_messages_fetched << " messages processed from " 
                                         << status.channels_processed << " accessible channels" << std::endl;
                                break;
                            } else if (i > 10) { // Give some time for backfill to start
                                std::cout << "No accessible channels found for history backfill or backfill disabled" << std::endl;
                                break;
                            }
                        }
                    }
                }).detach();
            } else {
                AddSystemMessage("Failed to start Discord bot");
            }
        }
    }

    void OnSettings(wxCommandEvent& event) {
        SettingsDialog dialog(this, model_path, context_size, gpu_layers, predict_tokens, 
                            chat_template, identity_directive, other_directives,
                            discord_bot_token, discord_channel_ids, discord_isolated_channel_ids, 
                            discord_shared_history_channel_ids, discord_allow_dms); // ADDED: Pass new setting
        if (dialog.ShowModal() == wxID_OK) {
            UpdateWindowTitle();
            
            if (is_started) {
                wxMessageBox("Settings have been saved. Please restart the model for changes to take effect.", 
                           "Settings Updated", wxOK | wxICON_INFORMATION);
            }
        }
    }

    void OnInputEnter(wxCommandEvent& event) {
        if (!is_started || is_processing) return;
        
        // ADDED: Check if context is created
        if (!context_created) {
            AddSystemMessage("Error: Chat context not available. Please restart the model.");
            return;
        }
        
        wxString input = input_text->GetValue().Trim();
        if (input.IsEmpty()) return;
        
        // Display user input with blue background
        AddUserMessage(input);
        input_text->Clear();
        
        is_processing = true;
        UpdateButtonStates();
        
        // Start response generation in background thread
        worker_thread = new ModelWorkerThread(this, llama_manager.get(), ModelWorkerThread::GENERATE_RESPONSE);
        worker_thread->SetInput(input.ToStdString(), "User"); // UPDATED: Pass username
        
        if (worker_thread->Run() != wxTHREAD_NO_ERROR) {
            delete worker_thread;
            worker_thread = nullptr;
            is_processing = false;
            UpdateButtonStates();
            AddSystemMessage("Error: Failed to start response generation thread");
        }
    }
    
    void OnResponseReady(wxCommandEvent& event) {
        is_processing = false;
        worker_thread = nullptr; // Thread is detached and will clean itself up
        
        wxString response = event.GetString();
        
        // Check for error responses
        if (response.StartsWith("Error:")) {
            AddSystemMessage(response);
        } else {
            AddAIMessage(response);
            
            // Update timings after successful inference
            UpdateTimingsDisplay();
        }
        
        UpdateButtonStates();
        input_text->SetFocus();
        
        // Auto-scroll to bottom
        chat_history->SetInsertionPointEnd();
        chat_history->ShowPosition(chat_history->GetLastPosition());
    }

    void UpdateTimingsDisplay() {
        if (!is_started || !llama_manager) {
            return;
        }
        
        // Get timings from llama manager
        auto timings = llama_manager->get_timings();
        if (timings.n_eval > 0) {
            double tokens_per_sec = 1000.0 * timings.n_eval / timings.t_eval_ms;
            timings_label->SetLabel(wxString::Format("%.2f tok/s (%d tokens in %.2fms)", 
                                                   tokens_per_sec, 
                                                   timings.n_eval, 
                                                   timings.t_eval_ms));
        }
    }
};

// ADDED: Implementation of wxLogStreamBuffer::FlushBuffer now that LuminaChatFrame is defined
void wxLogStreamBuffer::FlushBuffer() {
    if (!buffer.empty() && frame) {
        // Remove trailing newlines for cleaner display
        while (!buffer.empty() && (buffer.back() == '\n' || buffer.back() == '\r')) {
            buffer.pop_back();
        }
        
        if (!buffer.empty()) {
            // Use thread-safe method to append to logs
            wxString message = wxString::FromUTF8(buffer) + "\n";
            frame->AppendToLogsThreadSafe(message);
        }
        buffer.clear();
    }
}

// MOVED: Implementation of llama manager log callback now that LuminaChatFrame is defined
void llama_manager_log_callback(const std::string& message) {
    if (g_main_frame) {
        g_main_frame->AppendToLogsThreadSafe(wxString::FromUTF8(message) + "\n");
    }
}

// MOVED: Implementation of discord manager log callback now that LuminaChatFrame is defined
void discord_manager_log_callback(const std::string& message) {
    if (g_main_frame) {
        g_main_frame->AppendToLogsThreadSafe(wxString::FromUTF8("[Discord] " + message) + "\n");
    }
}

// ADDED: Settings manager log callback implementation after LuminaChatFrame
void settings_manager_log_callback(const std::string& message) {
    if (g_main_frame) {
        g_main_frame->AppendToLogsThreadSafe(wxString::FromUTF8(message + "\n"));
    } else {
        std::cout << "[SettingsManager] " << message << std::endl;
    }
}

// Progress callback function for model loading - implementation
bool model_loading_progress_callback(float progress, void *user_data) {
    wxEvtHandler* handler = static_cast<wxEvtHandler*>(user_data);
    if (handler) {
        int32_t percent = static_cast<int32_t>(progress * 100.0f);
        
        // Create progress event
        wxCommandEvent event(wxEVT_PROGRESS_UPDATE);
        event.SetInt(percent);
        
        // Post event to UI thread
        wxQueueEvent(handler, event.Clone());
    }
    
    // Return true to continue loading
    return true;
}

// Llama.cpp log callback function for logs tab - implementation
void llama_log_callback(ggml_log_level level, const char * message, void * user_data) {
    LuminaChatFrame* frame = static_cast<LuminaChatFrame*>(user_data);
    if (frame && message) {
        const char* level_str = (level == GGML_LOG_LEVEL_ERROR ? "ERROR" :
                                level == GGML_LOG_LEVEL_WARN  ? "WARN" :
                                level == GGML_LOG_LEVEL_INFO  ? "INFO" : "DEBUG");
        
        // Format the log message
        wxString log_message = wxString::Format("[%s] %s", level_str, message);
        
        // Use thread-safe method to append to logs
        frame->AppendToLogsThreadSafe(log_message);
    }
}

// Application Class
class LuminaChatApp : public wxApp {
public:
    bool OnInit() override {
        // Set application name for better file organization
        SetAppName("LuminaChat");
        SetVendorName("LuminaChat");
        
        LuminaChatFrame* frame = new LuminaChatFrame();
        frame->Show(true);
        return true;
    }
};

wxIMPLEMENT_APP(LuminaChatApp);

// Explicit main function for proper linking
int32_t main(int32_t argc, char* argv[]) {
    return wxEntry(argc, argv);
}
