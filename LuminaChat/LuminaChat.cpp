// LuminaChat.cpp - Main application file for LuminaChat with wxWidgets GUI
// Handles initialization, UI setup, and main application logic
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
// 5. Use consistent _t fixed-width variable types to ensure portability across platforms.
// 6. Cache frequently used variables to avoid repeated allocations.
// 7. Ensure there are no logical errors and the execution paths flow as expected.
// 8. Refactor where necessary to maintain clean code, efficient code, and to conform to the above settings and directives.
// 9. NEVER BREAK FUNCTIONALITY THAT IS ALREADY WORKING.

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
#include "LlamaManager.hpp"

// Event IDs
enum {
    ID_START = 1000,
    ID_STOP,
    ID_SETTINGS,
    ID_INPUT_TEXT,
    ID_BROWSE_MODEL,
    ID_MODEL_LOADED,
    ID_RESPONSE_READY,
    ID_PROGRESS_UPDATE
};

// Custom events for thread communication
wxDECLARE_EVENT(wxEVT_MODEL_LOADED, wxCommandEvent);
wxDECLARE_EVENT(wxEVT_RESPONSE_READY, wxCommandEvent);
wxDECLARE_EVENT(wxEVT_PROGRESS_UPDATE, wxCommandEvent);

wxDEFINE_EVENT(wxEVT_MODEL_LOADED, wxCommandEvent);
wxDEFINE_EVENT(wxEVT_RESPONSE_READY, wxCommandEvent);
wxDEFINE_EVENT(wxEVT_PROGRESS_UPDATE, wxCommandEvent);

// Forward declarations
class LuminaChatFrame;
class ModelWorkerThread;

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

    void SetInput(const std::string& input) {
        input_text = input;
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
        
        return llama_manager->generate_response(input_text);
    }
};

// Settings management functions
class SettingsManager {
private:
    // Helper functions to escape/unescape strings for INI storage
    static std::string EscapeString(const std::string& input) {
        std::string result = input;
        size_t pos = 0;
        while ((pos = result.find('\n', pos)) != std::string::npos) {
            result.replace(pos, 1, "\\n");
            pos += 2;
        }
        pos = 0;
        while ((pos = result.find('\r', pos)) != std::string::npos) {
            result.replace(pos, 1, "\\r");
            pos += 2;
        }
        return result;
    }
    
    static std::string UnescapeString(const std::string& input) {
        std::string result = input;
        size_t pos = 0;
        while ((pos = result.find("\\n", pos)) != std::string::npos) {
            result.replace(pos, 2, "\n");
            pos += 1;
        }
        pos = 0;
        while ((pos = result.find("\\r", pos)) != std::string::npos) {
            result.replace(pos, 2, "\r");
            pos += 1;
        }
        return result;
    }
    
    // Validate and clamp integer values
    static int32_t ValidateInt32(const std::string& value, int32_t default_val, int32_t min_val, int32_t max_val) {
        try {
            int32_t parsed = std::stoi(value);
            return (parsed >= min_val && parsed <= max_val) ? parsed : default_val;
        } catch (...) {
            return default_val;
        }
    }

public:
    static std::string GetSettingsFilePath() {
        wxStandardPaths& stdPaths = wxStandardPaths::Get();
        wxString exeDir = stdPaths.GetExecutablePath().BeforeLast(wxFileName::GetPathSeparator());
        wxString iniPath = exeDir + wxFileName::GetPathSeparator() + wxT("LuminaChat.ini");
        return iniPath.ToStdString();
    }
    
    static void SaveSettings(const std::string& model_path, int32_t context_size, int32_t gpu_layers, 
                           int32_t predict_tokens, const std::string& chat_template, 
                           const std::string& identity_directive, const std::string& other_directives) {
        std::ofstream file(GetSettingsFilePath());
        
        if (file.is_open()) {
            file << "[General]\n"
                 << "ModelPath=" << model_path << "\n"
                 << "ContextSize=" << context_size << "\n"
                 << "GpuLayers=" << gpu_layers << "\n"
                 << "PredictTokens=" << predict_tokens << "\n"
                 << "\n[ChatTemplate]\n"
                 << "Template=" << EscapeString(chat_template) << "\n"
                 << "\n[SystemPrompt]\n"
                 << "IdentityDirective=" << EscapeString(identity_directive) << "\n"
                 << "OtherDirectives=" << EscapeString(other_directives) << "\n";
        }
    }
    
    static void LoadSettings(std::string& model_path, int32_t& context_size, int32_t& gpu_layers, 
                           int32_t& predict_tokens, std::string& chat_template, 
                           std::string& identity_directive, std::string& other_directives) {
        std::ifstream file(GetSettingsFilePath());
        
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                if (line.empty() || line[0] == '[') continue;
                
                size_t equalPos = line.find('=');
                if (equalPos == std::string::npos) continue;
                
                std::string key = line.substr(0, equalPos);
                std::string value = line.substr(equalPos + 1);
                
                if (key == "ModelPath") {
                    model_path = value;
                } else if (key == "ContextSize") {
                    context_size = ValidateInt32(value, 2048, 1, 131072);
                } else if (key == "GpuLayers") {
                    gpu_layers = ValidateInt32(value, 0, 0, 999);
                } else if (key == "PredictTokens") {
                    predict_tokens = ValidateInt32(value, 256, 1, 4096);
                } else if (key == "Template") {
                    chat_template = UnescapeString(value);
                } else if (key == "IdentityDirective") {
                    identity_directive = UnescapeString(value);
                } else if (key == "OtherDirectives") {
                    other_directives = UnescapeString(value);
                }
            }
        }
    }
};

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
    
    // References to settings
    std::string& model_path_ref;
    int32_t& context_size_ref;
    int32_t& gpu_layers_ref;
    int32_t& predict_tokens_ref;
    std::string& chat_template_ref;
    std::string& identity_directive_ref;
    std::string& other_directives_ref;

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
                  std::string& other_directives) 
        : wxDialog(parent, wxID_ANY, "Settings", wxDefaultPosition, wxSize(700, 500)),
          model_path_ref(model_path), context_size_ref(context_size), gpu_layers_ref(gpu_layers), 
          predict_tokens_ref(predict_tokens), chat_template_ref(chat_template),
          identity_directive_ref(identity_directive), other_directives_ref(other_directives) {
        
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
        
        // Save settings
        SettingsManager::SaveSettings(model_path_ref, context_size_ref, gpu_layers_ref, 
                                    predict_tokens_ref, chat_template_ref, 
                                    identity_directive_ref, other_directives_ref);
        
        EndModal(wxID_OK);
    }
};

// Main Frame with optimized performance
class LuminaChatFrame : public wxFrame {
private:
    // Core components
    std::unique_ptr<LlamaManager> llama_manager;
    
    // Settings
    std::string model_path;
    int32_t context_size;
    int32_t gpu_layers;
    int32_t predict_tokens;
    std::string chat_template;
    std::string identity_directive;
    std::string other_directives;
    
    // State
    bool is_started;
    std::atomic<bool> is_processing{false};
    
    // UI controls
    wxButton* start_btn;
    wxButton* stop_btn;
    wxButton* settings_btn;
    wxGauge* progress_bar;
    wxStaticText* progress_label;
    wxStaticText* timings_label;
    wxRichTextCtrl* chat_history;
    wxTextCtrl* input_text;
    wxTextCtrl* logs_text;
    wxNotebook* main_notebook;
    wxPanel* main_panel;  // Store reference to main panel
    
    ModelWorkerThread* worker_thread = nullptr;

public:
    LuminaChatFrame() : wxFrame(nullptr, wxID_ANY, "LuminaChat", wxDefaultPosition, wxSize(800, 600)),
                        llama_manager(std::make_unique<LlamaManager>()),
                        context_size(2048), gpu_layers(0), predict_tokens(256), is_started(false) {
        
        SettingsManager::LoadSettings(model_path, context_size, gpu_layers, predict_tokens, 
                                    chat_template, identity_directive, other_directives);
        
        CreateUI();
        UpdateButtonStates();
        UpdateWindowTitle();

        // Bind custom events
        Bind(wxEVT_MODEL_LOADED, &LuminaChatFrame::OnModelLoaded, this);
        Bind(wxEVT_RESPONSE_READY, &LuminaChatFrame::OnResponseReady, this);
        Bind(wxEVT_PROGRESS_UPDATE, &LuminaChatFrame::OnProgressUpdate, this);
    }

    ~LuminaChatFrame() {
        if (worker_thread) {
            worker_thread->RequestStop();
            worker_thread = nullptr;
        }
    }

private:
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
                                        chat_template, identity_directive, other_directives);
        }
    }

    void CreateUI() {
        main_panel = new wxPanel(this);
        
        // Top toolbar
        wxBoxSizer* toolbar_sizer = new wxBoxSizer(wxHORIZONTAL);
        start_btn = new wxButton(main_panel, ID_START, "Start");
        stop_btn = new wxButton(main_panel, ID_STOP, "Stop");
        settings_btn = new wxButton(main_panel, ID_SETTINGS, "Settings");
        
        toolbar_sizer->Add(start_btn, 0, wxRIGHT, 5);
        toolbar_sizer->Add(stop_btn, 0, wxRIGHT, 5);
        toolbar_sizer->Add(settings_btn, 0);
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
        main_panel->Layout(); // Use main_panel->Layout() instead of GetSizer()->Layout()
        
        // Log loading information
        logs_text->AppendText(wxString::Format(
            "Initializing LuminaChat...\n"
            "Context Size: %d tokens\n"
            "GPU Layers: %d\n"
            "Max Prediction: %d tokens\n"
            "Loading model: %s\n",
            context_size, gpu_layers, predict_tokens, model_path));
        
        if (!GetCombinedSystemPrompt().empty()) {
            logs_text->AppendText("System prompt configured\n");
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
        main_panel->Layout(); // Use main_panel->Layout() instead of GetSizer()->Layout()
        
        bool success = event.GetInt() == 1;
        
        if (success) {
            // Handle chat template
            if (chat_template.empty()) {
                std::string model_template = llama_manager->get_model_chat_template();
                if (!model_template.empty()) {
                    chat_template = model_template;
                    SettingsManager::SaveSettings(model_path, context_size, gpu_layers, predict_tokens, 
                                                chat_template, identity_directive, other_directives);
                    logs_text->AppendText("Loaded chat template from model\n");
                }
            } else {
                llama_manager->set_custom_chat_template(chat_template);
                logs_text->AppendText("Using custom chat template\n");
            }
            
            // Set system prompt
            std::string combined_prompt = GetCombinedSystemPrompt();
            if (!combined_prompt.empty()) {
                llama_manager->set_system_prompt(combined_prompt);
                logs_text->AppendText("System prompt applied\n");
            }
            
            is_started = true;
            
            chat_history->Clear();
            AddSystemMessage("LuminaChat ready! Type your message below.");
            input_text->SetFocus();
            
            llama_manager->reset_timings();
            timings_label->SetLabel("");
        } else {
            logs_text->AppendText("Error: Failed to load model\n");
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
        
        llama_manager->cleanup();
        is_started = false;
        is_processing = false;
        UpdateButtonStates();
        logs_text->AppendText("LuminaChat stopped.\n\n");
        timings_label->SetLabel("");
    }
    
    void OnSettings(wxCommandEvent& event) {
        SettingsDialog dialog(this, model_path, context_size, gpu_layers, predict_tokens, 
                            chat_template, identity_directive, other_directives);
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
        
        wxString input = input_text->GetValue().Trim();
        if (input.IsEmpty()) return;
        
        // Display user input with blue background
        AddUserMessage(input);
        input_text->Clear();
        
        is_processing = true;
        UpdateButtonStates();
        
        // Start response generation in background thread
        worker_thread = new ModelWorkerThread(this, llama_manager.get(), ModelWorkerThread::GENERATE_RESPONSE);
        worker_thread->SetInput(input.ToStdString());
        
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

public:
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
