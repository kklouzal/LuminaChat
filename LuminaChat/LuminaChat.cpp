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
// 6. Ensure there are no logical errors and the execution paths flow as expected.
// 7. Refactor where necessary to maintain clean code, efficient code, and to conform to the above settings and directives.
// 8. NEVER BREAK FUNCTIONALITY THAT IS ALREADY WORKING.

#include <wx/wx.h>
#include <wx/filedlg.h>
#include <wx/notebook.h>
#include <wx/textctrl.h>
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
public:
    static std::string GetSettingsFilePath() {
        wxStandardPaths& stdPaths = wxStandardPaths::Get();
        wxString exeDir = stdPaths.GetExecutablePath().BeforeLast(wxFileName::GetPathSeparator());
        wxString iniPath = exeDir + wxFileName::GetPathSeparator() + wxT("LuminaChat.ini");
        return iniPath.ToStdString();
    }
    
    static void SaveSettings(const std::string& model_path, int32_t context_size, int32_t gpu_layers, int32_t predict_tokens, const std::string& chat_template) {
        std::string filePath = GetSettingsFilePath();
        std::ofstream file(filePath);
        
        if (file.is_open()) {
            file << "[General]\n";
            file << "ModelPath=" << model_path << "\n";
            file << "ContextSize=" << context_size << "\n";
            file << "GpuLayers=" << gpu_layers << "\n";
            file << "PredictTokens=" << predict_tokens << "\n";
            file << "\n[ChatTemplate]\n";
            
            // Escape newlines for storage
            std::string escaped_template = chat_template;
            size_t pos = 0;
            while ((pos = escaped_template.find('\n', pos)) != std::string::npos) {
                escaped_template.replace(pos, 1, "\\n");
                pos += 2;
            }
            while ((pos = escaped_template.find('\r', pos)) != std::string::npos) {
                escaped_template.replace(pos, 1, "\\r");
                pos += 2;
            }
            
            file << "Template=" << escaped_template << "\n";
            file.close();
        }
    }
    
    static void LoadSettings(std::string& model_path, int32_t& context_size, int32_t& gpu_layers, int32_t& predict_tokens, std::string& chat_template) {
        std::string filePath = GetSettingsFilePath();
        std::ifstream file(filePath);
        
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                // Skip section headers and empty lines
                if (line.empty() || line[0] == '[') {
                    continue;
                }
                
                size_t equalPos = line.find('=');
                if (equalPos != std::string::npos) {
                    std::string key = line.substr(0, equalPos);
                    std::string value = line.substr(equalPos + 1);
                    
                    if (key == "ModelPath") {
                        model_path = value;
                    } else if (key == "ContextSize") {
                        try {
                            context_size = std::stoi(value);
                            if (context_size <= 0 || context_size > 131072) {
                                context_size = 2048;
                            }
                        } catch (...) {
                            context_size = 2048;
                        }
                    } else if (key == "GpuLayers") {
                        try {
                            gpu_layers = std::stoi(value);
                            if (gpu_layers < 0 || gpu_layers > 999) {
                                gpu_layers = 0;
                            }
                        } catch (...) {
                            gpu_layers = 0;
                        }
                    } else if (key == "PredictTokens") {
                        try {
                            predict_tokens = std::stoi(value);
                            if (predict_tokens <= 0 || predict_tokens > 4096) {
                                predict_tokens = 256;
                            }
                        } catch (...) {
                            predict_tokens = 256;
                        }
                    } else if (key == "Template") {
                        // Unescape newlines
                        chat_template = value;
                        size_t pos = 0;
                        while ((pos = chat_template.find("\\n", pos)) != std::string::npos) {
                            chat_template.replace(pos, 2, "\n");
                            pos += 1;
                        }
                        pos = 0;
                        while ((pos = chat_template.find("\\r", pos)) != std::string::npos) {
                            chat_template.replace(pos, 2, "\r");
                            pos += 1;
                        }
                    }
                }
            }
            file.close();
        }
    }
};

// Settings Dialog
class SettingsDialog : public wxDialog {
private:
    wxTextCtrl* model_path_text;
    wxTextCtrl* context_size_text;
    wxTextCtrl* gpu_layers_text;
    wxTextCtrl* predict_tokens_text;
    wxTextCtrl* chat_template_text;
    std::string& model_path_ref;
    int32_t& context_size_ref;
    int32_t& gpu_layers_ref;
    int32_t& predict_tokens_ref;
    std::string& chat_template_ref;

public:
    SettingsDialog(wxWindow* parent, std::string& model_path, int32_t& context_size, int32_t& gpu_layers, int32_t& predict_tokens, std::string& chat_template) 
        : wxDialog(parent, wxID_ANY, "Settings", wxDefaultPosition, wxSize(700, 500)),
          model_path_ref(model_path), context_size_ref(context_size), gpu_layers_ref(gpu_layers), 
          predict_tokens_ref(predict_tokens), chat_template_ref(chat_template) {
        
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
        
        long context_size_val;
        if (context_size_text->GetValue().ToLong(&context_size_val) && context_size_val > 0 && context_size_val <= 131072) {
            context_size_ref = static_cast<int32_t>(context_size_val);
        } else {
            wxMessageBox("Invalid context size. Using default value (2048).", "Warning", wxOK | wxICON_WARNING);
            context_size_ref = 2048;
        }
        
        long gpu_layers_val;
        if (gpu_layers_text->GetValue().ToLong(&gpu_layers_val) && gpu_layers_val >= 0 && gpu_layers_val <= 999) {
            gpu_layers_ref = static_cast<int32_t>(gpu_layers_val);
        } else {
            wxMessageBox("Invalid GPU layers. Using default value (0).", "Warning", wxOK | wxICON_WARNING);
            gpu_layers_ref = 0;
        }
        
        long predict_tokens_val;
        if (predict_tokens_text->GetValue().ToLong(&predict_tokens_val) && predict_tokens_val > 0 && predict_tokens_val <= 4096) {
            predict_tokens_ref = static_cast<int32_t>(predict_tokens_val);
        } else {
            wxMessageBox("Invalid prediction tokens. Using default value (256).", "Warning", wxOK | wxICON_WARNING);
            predict_tokens_ref = 256;
        }
        
        // Get chat template
        chat_template_ref = chat_template_text->GetValue().ToUTF8().data();
        
        // Save settings to INI file including chat template
        SettingsManager::SaveSettings(model_path_ref, context_size_ref, gpu_layers_ref, predict_tokens_ref, chat_template_ref);
        
        EndModal(wxID_OK);
    }
};

// Main Frame with optimized performance
class LuminaChatFrame : public wxFrame {
private:
    std::unique_ptr<LlamaManager> llama_manager;
    std::string model_path;
    int32_t context_size;
    int32_t gpu_layers;
    int32_t predict_tokens;
    std::string chat_template;
    bool is_started;
    std::atomic<bool> is_processing{false};
    
    wxButton* start_btn;
    wxButton* stop_btn;
    wxButton* settings_btn;
    wxGauge* progress_bar;
    wxStaticText* progress_label;
    wxStaticText* timings_label;
    wxTextCtrl* chat_history;
    wxTextCtrl* input_text;
    wxTextCtrl* logs_text;
    wxNotebook* main_notebook;
    
    ModelWorkerThread* worker_thread = nullptr;

public:
    LuminaChatFrame() : wxFrame(nullptr, wxID_ANY, "LuminaChat", wxDefaultPosition, wxSize(800, 600)),
                        llama_manager(std::make_unique<LlamaManager>()),
                        context_size(2048), gpu_layers(0), predict_tokens(256), is_started(false) {
        
        SettingsManager::LoadSettings(model_path, context_size, gpu_layers, predict_tokens, chat_template);
        
        CreateUI();
        UpdateButtonStates();
        
        if (!model_path.empty()) {
            wxFileName modelFile(wxString::FromUTF8(model_path));
            SetTitle(wxString::Format("LuminaChat - %s", modelFile.GetName()));
        }

        // Bind custom events
        Bind(wxEVT_MODEL_LOADED, &LuminaChatFrame::OnModelLoaded, this);
        Bind(wxEVT_RESPONSE_READY, &LuminaChatFrame::OnResponseReady, this);
        Bind(wxEVT_PROGRESS_UPDATE, &LuminaChatFrame::OnProgressUpdate, this);
    }

    ~LuminaChatFrame() {
        if (worker_thread) {
            worker_thread->RequestStop();
            // Don't wait for detached thread, just clean up
            worker_thread = nullptr;
        }
    }

private:
    void CreateUI() {
        wxPanel* main_panel = new wxPanel(this);
        
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
        
        // Chat history (read-only)
        chat_history = new wxTextCtrl(chat_panel, wxID_ANY, "", wxDefaultPosition, wxDefaultSize,
                                     wxTE_MULTILINE | wxTE_READONLY | wxTE_WORDWRAP);
        chat_history->SetFont(wxFont(10, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
        
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
        
        // Layout
        wxBoxSizer* main_sizer = new wxBoxSizer(wxVERTICAL);
        main_sizer->Add(toolbar_sizer, 0, wxEXPAND | wxALL, 10);
        
        // Progress and timings row
        wxBoxSizer* status_sizer = new wxBoxSizer(wxHORIZONTAL);
        status_sizer->Add(progress_label, 0, wxALIGN_CENTER_VERTICAL);
        status_sizer->AddSpacer(20);
        status_sizer->Add(timings_label, 0, wxALIGN_CENTER_VERTICAL);
        status_sizer->AddStretchSpacer();
        
        main_sizer->Add(status_sizer, 0, wxEXPAND | wxLEFT | wxRIGHT, 10);
        main_sizer->Add(progress_bar, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 10);
        main_sizer->Add(main_notebook, 1, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 10);
        
        main_panel->SetSizer(main_sizer);
        
        // Bind events
        Bind(wxEVT_COMMAND_BUTTON_CLICKED, &LuminaChatFrame::OnStart, this, ID_START);
        Bind(wxEVT_COMMAND_BUTTON_CLICKED, &LuminaChatFrame::OnStop, this, ID_STOP);
        Bind(wxEVT_COMMAND_BUTTON_CLICKED, &LuminaChatFrame::OnSettings, this, ID_SETTINGS);
        Bind(wxEVT_COMMAND_TEXT_ENTER, &LuminaChatFrame::OnInputEnter, this, ID_INPUT_TEXT);
        
        // Set initial message
        chat_history->SetValue("Welcome to LuminaChat!\nClick 'Settings' to select a model, then 'Start' to begin.\n\n");
    }
    
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
        
        if (is_processing) {
            return;
        }

        is_processing = true;
        UpdateButtonStates();
        
        // Show progress bar and update label
        progress_label->SetLabel("Loading model...");
        progress_bar->SetValue(0);
        progress_bar->Show();
        Layout(); // Refresh layout to show progress bar
        
        logs_text->AppendText("Initializing LuminaChat...\n");
        logs_text->AppendText(wxString::Format("Context Size: %d tokens\n", context_size));
        logs_text->AppendText(wxString::Format("GPU Layers: %d\n", gpu_layers));
        logs_text->AppendText(wxString::Format("Max Prediction: %d tokens\n", predict_tokens));
        logs_text->AppendText("Loading model: " + model_path + "\n");
        
        // Start model loading in background thread
        worker_thread = new ModelWorkerThread(this, llama_manager.get(), ModelWorkerThread::LOAD_MODEL);
        worker_thread->SetModelParams(model_path, context_size, gpu_layers, predict_tokens, chat_template);
        
        if (worker_thread->Run() != wxTHREAD_NO_ERROR) {
            delete worker_thread;
            worker_thread = nullptr;
            is_processing = false;
            progress_bar->Hide();
            progress_label->SetLabel("Ready");
            Layout();
            UpdateButtonStates();
            chat_history->AppendText("Error: Failed to start model loading thread\n");
        }
    }
    
    void OnProgressUpdate(wxCommandEvent& event) {
        int32_t percent = event.GetInt();
        progress_bar->SetValue(percent);
        progress_label->SetLabel(wxString::Format("Loading model... %d%%", percent));
        
        // Force UI update
        Update();
    }
    
    void OnModelLoaded(wxCommandEvent& event) {
        is_processing = false;
        worker_thread = nullptr;
        
        // Hide progress bar and reset label
        progress_bar->Hide();
        progress_label->SetLabel("Ready");
        Layout();
        
        bool success = event.GetInt() == 1;
        
        if (success) {
            if (chat_template.empty()) {
                std::string model_template = llama_manager->get_model_chat_template();
                if (!model_template.empty()) {
                    chat_template = model_template;
                    SettingsManager::SaveSettings(model_path, context_size, gpu_layers, predict_tokens, chat_template);
                    logs_text->AppendText("Loaded chat template from model\n");
                }
            } else {
                llama_manager->set_custom_chat_template(chat_template);
                logs_text->AppendText("Using custom chat template\n");
            }
            
            is_started = true;
            chat_history->AppendText("LuminaChat ready! Type your message below.\n\n");
            input_text->SetFocus();
            
            // Reset timings for fresh measurement
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
        
        // Hide progress bar if visible
        if (progress_bar->IsShown()) {
            progress_bar->Hide();
            progress_label->SetLabel("Ready");
            Layout();
        }
        
        llama_manager->cleanup();
        is_started = false;
        is_processing = false;
        UpdateButtonStates();
        logs_text->AppendText("LuminaChat stopped.\n\n");
        timings_label->SetLabel("");
    }
    
    void OnSettings(wxCommandEvent& event) {
        SettingsDialog dialog(this, model_path, context_size, gpu_layers, predict_tokens, chat_template);
        if (dialog.ShowModal() == wxID_OK) {
            // Update window title if model path changed
            if (!model_path.empty()) {
                wxFileName modelFile(wxString::FromUTF8(model_path));
                SetTitle(wxString::Format("LuminaChat - %s", modelFile.GetName()));
            } else {
                SetTitle("LuminaChat");
            }
            
            // If the model is currently running and settings changed, inform user
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
        
        // Display user input immediately
        chat_history->AppendText("You: " + input + "\n");
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
            chat_history->AppendText("Error: Failed to start response generation thread\n");
        }
    }
    
    void OnResponseReady(wxCommandEvent& event) {
        is_processing = false;
        worker_thread = nullptr; // Thread is detached and will clean itself up
        
        wxString response = event.GetString();
        
        // Check for error responses
        if (response.StartsWith("Error:")) {
            chat_history->AppendText("System: " + response + "\n\n");
        } else {
            chat_history->AppendText("AI: " + response + "\n\n");
            
            // Update timings after successful inference
            UpdateTimingsDisplay();
        }
        
        UpdateButtonStates();
        input_text->SetFocus();
        
        // Auto-scroll to bottom
        chat_history->SetInsertionPointEnd();
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
    
    int32_t OnExit() override {
        return wxApp::OnExit();
    }
};

wxIMPLEMENT_APP(LuminaChatApp);

// Explicit main function for proper linking
int32_t main(int32_t argc, char* argv[]) {
    return wxEntry(argc, argv);
}
