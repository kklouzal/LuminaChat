// LuminaChat.cpp - Main application file for LuminaChat with wxWidgets GUI
//
// Project Settings:
// C++ Language Standard: ISO C++20 Standard (/std:c++20)
// C Language Standard: ISO C17 (2018) Standard (/std:c17)
// Optimization: Maximum Optimization (Favor Speed) (/O2)
// Favor Size or Speed: Favor fast code (/Ot)
// Runtime Library: Multi-threaded DLL (/MD)
// Enable Run-Time Type Information (RTTI) YES (/GR)
//
// CRITICAL CODING DIRECTIVES:
// 1.  Minimalism & Performance: Deliver lean, efficient solutions; do not create or preserve unused helpers or wrappers.
// 2.  Consistent Style: Adopt a uniform coding style and structure for clarity and maintainability.
// 3.  Documentation: Write concise comments that explain complex logic and key design decisions.
// 4.  Redundancy Elimination: Remove unused, obsolete, and legacy code—including unneeded interfaces and includes.
// 5.  Function Boundaries: Define clear responsibilities; reduce overlap and avoid unnecessary layers.
// 6.  Core Preservation: Streamline code while safeguarding essential features; favor direct access over extra abstractions.
// 7.  Cross-Platform Portability: Use fixed-width types and proper initialization to guarantee identical behavior everywhere.
// 8.  Smart Caching: Cache frequently used values to minimize allocations and improve performance.
// 9.  Logical Consistency: Verify code flow to ensure coherent, error-free execution paths.
// 10. Continuous Refinement: Regularly refactor and confirm that updates preserve stable functionality.
// 11. Const-Correctness & Immutability: Mark variables, parameters, and methods as const wherever possible.
// 12. RAII & Resource Safety: Encapsulate resource acquisition/release in constructors/destructors or smart pointers.
// 13. Zero Magic & Strong Typing: Replace magic literals with named constants, enums, or constexpr; prefer scoped enums.
// 14. Standard Library Preference: Favor STL algorithms and containers over custom loops and buffers.
// 15. Thread Safety: Define and document thread-safety contracts; protect shared state with mutexes, atomics, or thread-safe containers.

#include <wx/wx.h>
#include <wx/filedlg.h>
#include <wx/notebook.h>
#include <wx/textctrl.h>
#include <wx/richtext/richtextctrl.h>
#include <wx/button.h>
#include <wx/sizer.h>
#include <wx/panel.h>
#include <wx/stattext.h>
#include <wx/filename.h>
#include <wx/thread.h>
#include <wx/event.h>
#include <wx/gauge.h>
#include <wx/checkbox.h>
#include <wx/slider.h>
#include <wx/scrolwin.h>
#include <wx/timer.h>
#include <wx/datetime.h>
#include <string>
#include <cstdint>
#include <memory>
#include <atomic>
#include <thread>
#include <chrono>
#include "LlamaManager.hpp"
#include "DiscordManager.hpp" 
#include "SettingsManager.hpp"
#include "LogHandler.hpp"

// Forward declarations
class LuminaChatFrame;
class ModelWorkerThread;

// Optimized stream buffer for console redirection with improved performance
class wxLogStreamBuffer : public std::streambuf {
private:
    LuminaChatFrame* const frame;
    std::string buffer;
    
    // Pre-compiled constants for better performance
    static constexpr char NEWLINE = '\n';
    static constexpr size_t BUFFER_RESERVE_SIZE = 1024;
    
public:
    explicit wxLogStreamBuffer(LuminaChatFrame* f) : frame(f) { 
        buffer.reserve(BUFFER_RESERVE_SIZE);
    }
    
protected:
    int_type overflow(int_type c) override {
        if (c != EOF) {
            buffer += static_cast<char>(c);
            if (c == NEWLINE) FlushBuffer();
        }
        return c;
    }
    
    std::streamsize xsputn(const char* s, std::streamsize count) override {
        const size_t old_size = buffer.size();
        buffer.append(s, count);
        
        // Check for newlines only in the newly added portion
        if (buffer.find(NEWLINE, old_size) != std::string::npos) {
            FlushBuffer();
        }
        return count;
    }
    
private:
    void FlushBuffer();
};

// Event IDs - consolidated and organized
enum class EventId : int32_t {
    START = 1000,
    STOP,
    SETTINGS,
    INPUT_TEXT,
    BROWSE_MODEL,
    CONNECT_DISCORD,
    PRUNE_SUMMARIZE,
    MODEL_LOADED,
    RESPONSE_READY,
    PROGRESS_UPDATE,
    CONTEXT_MONITOR_TIMER
};

// Custom events for thread communication
wxDECLARE_EVENT(wxEVT_MODEL_LOADED, wxCommandEvent);
wxDECLARE_EVENT(wxEVT_RESPONSE_READY, wxCommandEvent);
wxDECLARE_EVENT(wxEVT_PROGRESS_UPDATE, wxCommandEvent);

wxDEFINE_EVENT(wxEVT_MODEL_LOADED, wxCommandEvent);
wxDEFINE_EVENT(wxEVT_RESPONSE_READY, wxCommandEvent);
wxDEFINE_EVENT(wxEVT_PROGRESS_UPDATE, wxCommandEvent);

// Forward declarations for callback functions
bool model_loading_progress_callback(float progress, void* user_data);
void llama_log_callback(ggml_log_level level, const char* message, void* user_data);

// Global frame pointer for callbacks
LuminaChatFrame* g_main_frame = nullptr;

// Optimized worker thread with better resource management
class ModelWorkerThread : public wxThread {
public:
    enum class Operation : uint8_t { LOAD_MODEL, GENERATE_RESPONSE };

private:
    LlamaManager* const llama_manager;
    wxEvtHandler* const parent;
    
    // Consolidated configuration structure
    struct Config {
        std::string model_path;
        std::string input_text;
        std::string input_username{"User"};
        std::string chat_template;
        std::string result;
        std::string summarizer_model_path;
        std::string summarizer_chat_template;
        int32_t context_size{2048};
        int32_t gpu_layers{0};
        int32_t predict_tokens{256};
        int32_t summarizer_context_size{1024};
        int32_t summarizer_gpu_layers{0};
        int32_t summarizer_predict_tokens{128};
        
        Config() { 
            result.reserve(2048);
            input_text.reserve(512);
        }
    } config;
    
    const Operation operation;
    std::atomic<bool> should_stop{false};
    
public:
    std::atomic<bool> success{false};

    ModelWorkerThread(wxEvtHandler* parent_handler, LlamaManager* manager, Operation op)
        : wxThread(wxTHREAD_DETACHED), llama_manager(manager), parent(parent_handler), operation(op) {}

    void SetModelParams(std::string path, int32_t ctx, int32_t gpu, int32_t pred, std::string tmpl,
                       std::string sum_path = "", int32_t sum_ctx = 1024, int32_t sum_gpu = 0, int32_t sum_pred = 128,
                       std::string sum_tmpl = "") {
        config.model_path = std::move(path);
        config.context_size = ctx;
        config.gpu_layers = gpu;
        config.predict_tokens = pred;
        config.chat_template = std::move(tmpl);
        config.summarizer_model_path = std::move(sum_path);
        config.summarizer_context_size = sum_ctx;
        config.summarizer_gpu_layers = sum_gpu;
        config.summarizer_predict_tokens = sum_pred;
        config.summarizer_chat_template = std::move(sum_tmpl);
    }

    void SetInput(std::string input, std::string username = "User") {
        config.input_text = std::move(input);
        config.input_username = std::move(username);
    }

    void RequestStop() noexcept { should_stop = true; }

protected:
    ExitCode Entry() override {
        try {
            success = (operation == Operation::LOAD_MODEL) ? LoadModel() : GenerateResponse();
            PostCompletionEvent();
        } catch (const std::exception& e) {
            config.result = "Error: " + std::string(e.what());
            success = false;
            PostCompletionEvent();
        }
        return static_cast<ExitCode>(0);
    }

private:
    bool LoadModel() {
        if (should_stop || !llama_manager) return false;
        
        if (!llama_manager->initialize()) return false;
        
        llama_log_set(llama_log_callback, parent);
        
        // Load main model with "main_model" ID and chat template
        if (should_stop || !llama_manager->load_model(config.model_path, "main_model", 
                                                     config.context_size, config.gpu_layers, 
                                                     config.predict_tokens, parent, config.chat_template)) return false;
        
        // Load summarizer model if path is provided and different from main model
        if (!config.summarizer_model_path.empty() && config.summarizer_model_path != config.model_path) {
            if (should_stop || !llama_manager->load_model(config.summarizer_model_path, "summary_model",
                                                         config.summarizer_context_size, config.summarizer_gpu_layers,
                                                         config.summarizer_predict_tokens, parent, config.summarizer_chat_template)) {
                LLAMA_LOG("Warning: Failed to load summarizer model, continuing with main model only");
                // Don't return false - main model is loaded successfully
            } else {
                LLAMA_LOG("Summarizer model loaded successfully");
            }
        }
        
        return true;
    }

    bool GenerateResponse() {
        if (should_stop || config.input_text.empty() || !llama_manager) return false;
        
        config.result = llama_manager->generate_response(config.input_text, config.input_username);
        return !config.result.empty() && !config.result.starts_with("Error:");
    }
    
    void PostCompletionEvent() {
        if (should_stop || !parent) return;
        
        if (operation == Operation::LOAD_MODEL) {
            wxCommandEvent event(wxEVT_MODEL_LOADED);
            event.SetInt(success ? 1 : 0);
            wxQueueEvent(parent, event.Clone());
        } else {
            wxCommandEvent event(wxEVT_RESPONSE_READY);
            event.SetString(wxString::FromUTF8(config.result));
            wxQueueEvent(parent, event.Clone());
        }
    }
};

// Optimized Settings Dialog with better validation and organization
class SettingsDialog : public wxDialog {
private:
    // Control structures
    struct Controls {
        // Model settings
        wxTextCtrl* model_path;
        wxTextCtrl* context_size;
        wxTextCtrl* gpu_layers;
        wxTextCtrl* predict_tokens;
        wxTextCtrl* chat_template;
        
        // System prompt
        wxTextCtrl* identity_directive;
        wxTextCtrl* other_directives;
        
        // Discord settings
        wxTextCtrl* bot_token;
        wxTextCtrl* isolated_channels;
        wxTextCtrl* shared_channels;
        wxCheckBox* allow_dms;
        wxCheckBox* pull_history;
        wxSlider* history_percentage;
        wxStaticText* percentage_label;
        
        // Summarizer settings
        wxTextCtrl* summarizer_model_path;
        wxTextCtrl* summarizer_context_size;
        wxTextCtrl* summarizer_gpu_layers;
        wxTextCtrl* summarizer_predict_tokens;
        wxTextCtrl* summarizer_system_prompt;
        wxTextCtrl* summarizer_chat_template;
    } ctrls;
    
    // Configuration references
    struct ConfigRefs {
        std::string& model_path;
        int32_t& context_size;
        int32_t& gpu_layers;
        int32_t& predict_tokens;
        std::string& chat_template;
        std::string& identity_directive;
        std::string& other_directives;
        std::string& discord_bot_token;
        std::string& discord_isolated_channels;
        std::string& discord_shared_channels;
        bool& discord_allow_dms;
        bool& discord_pull_history;
        int32_t& discord_history_percentage;
        std::string& summarizer_model_path;
        int32_t& summarizer_context_size;
        int32_t& summarizer_gpu_layers;
        int32_t& summarizer_predict_tokens;
        std::string& summarizer_system_prompt;
        std::string& summarizer_chat_template;
    } config;

    // UI elements
    const wxFont monospace_font{9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL};
    const wxFont help_font{8, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_ITALIC, wxFONTWEIGHT_NORMAL};

public:
    SettingsDialog(wxWindow* parent, std::string& model_path, int32_t& context_size, int32_t& gpu_layers, 
                  int32_t& predict_tokens, std::string& chat_template, std::string& identity_directive, 
                  std::string& other_directives, std::string& discord_bot_token,
                  std::string& discord_isolated_channels, std::string& discord_shared_channels,
                  bool& discord_allow_dms, bool& discord_pull_history, int32_t& discord_history_percentage,
                  std::string& summarizer_model_path, int32_t& summarizer_context_size, int32_t& summarizer_gpu_layers,
                  int32_t& summarizer_predict_tokens, std::string& summarizer_system_prompt, std::string& summarizer_chat_template) 
        : wxDialog(parent, wxID_ANY, "Settings", wxDefaultPosition, wxSize(700, 600))
        , config{model_path, context_size, gpu_layers, predict_tokens, chat_template,
                identity_directive, other_directives, discord_bot_token, discord_isolated_channels,
                discord_shared_channels, discord_allow_dms, discord_pull_history, discord_history_percentage,
                summarizer_model_path, summarizer_context_size, summarizer_gpu_layers, summarizer_predict_tokens,
                summarizer_system_prompt, summarizer_chat_template} {
        
        InitializeUI();
        BindEvents();
    }

private:
    void InitializeUI() {
        auto* notebook = new wxNotebook(this, wxID_ANY);
        
        CreateModelSettingsTab(notebook);
        CreateSystemPromptTab(notebook);
        CreateChatTemplateTab(notebook);
        CreateSummarizerTab(notebook);
        CreateDiscordSettingsTab(notebook);
        
        // Dialog layout
        auto* btn_sizer = new wxBoxSizer(wxHORIZONTAL);
        btn_sizer->Add(new wxButton(this, wxID_OK, "OK"), 0, wxRIGHT, 5);
        btn_sizer->Add(new wxButton(this, wxID_CANCEL, "Cancel"), 0);
        
        auto* main_sizer = new wxBoxSizer(wxVERTICAL);
        main_sizer->Add(notebook, 1, wxEXPAND | wxALL, 10);
        main_sizer->Add(btn_sizer, 0, wxALIGN_RIGHT | wxALL, 10);
        
        SetSizer(main_sizer);
    }
    
    void CreateModelSettingsTab(wxNotebook* notebook) {
        auto* panel = new wxPanel(notebook);
        auto* scrolled = new wxScrolledWindow(panel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxVSCROLL);
        scrolled->SetScrollRate(0, 20);
        
        auto* sizer = new wxBoxSizer(wxVERTICAL);
        
        // Model file path with browse button
        sizer->Add(new wxStaticText(scrolled, wxID_ANY, "Model File Path:"), 0, wxALL, 5);
        
        auto* path_sizer = new wxBoxSizer(wxHORIZONTAL);
        ctrls.model_path = new wxTextCtrl(scrolled, wxID_ANY, config.model_path, wxDefaultPosition, wxDefaultSize, wxTE_READONLY);
        auto* browse_btn = new wxButton(scrolled, static_cast<int>(EventId::BROWSE_MODEL), "Browse...");
        
        path_sizer->Add(ctrls.model_path, 1, wxEXPAND | wxRIGHT, 5);
        path_sizer->Add(browse_btn, 0, wxALIGN_CENTER_VERTICAL);
        sizer->Add(path_sizer, 0, wxEXPAND | wxALL, 5);
        
        // Numeric settings with validation hints
        AddNumericSetting(scrolled, sizer, "Context Size (tokens):", ctrls.context_size, config.context_size);
        AddNumericSetting(scrolled, sizer, "GPU Offload Layers (0 = CPU only):", ctrls.gpu_layers, config.gpu_layers);
        AddNumericSetting(scrolled, sizer, "Max Prediction Tokens:", ctrls.predict_tokens, config.predict_tokens);
        
        scrolled->SetSizer(sizer);
        
        auto* panel_sizer = new wxBoxSizer(wxVERTICAL);
        panel_sizer->Add(scrolled, 1, wxEXPAND);
        panel->SetSizer(panel_sizer);
        
        notebook->AddPage(panel, "Model Settings");
    }
    
    void AddNumericSetting(wxWindow* parent, wxBoxSizer* sizer, const wxString& label, 
                          wxTextCtrl*& control, int32_t value) {
        sizer->Add(new wxStaticText(parent, wxID_ANY, label), 0, wxALL, 5);
        control = new wxTextCtrl(parent, wxID_ANY, wxString::Format("%d", value));
        sizer->Add(control, 0, wxEXPAND | wxALL, 5);
    }
    
    void CreateSystemPromptTab(wxNotebook* notebook) {
        auto* panel = new wxPanel(notebook);
        auto* scrolled = new wxScrolledWindow(panel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxVSCROLL);
        scrolled->SetScrollRate(0, 20);
        
        auto* sizer = new wxBoxSizer(wxVERTICAL);
        
        // Identity Directive (1/3 height)
        sizer->Add(new wxStaticText(scrolled, wxID_ANY, "Identity Directive:"), 0, wxALL, 5);
        ctrls.identity_directive = new wxTextCtrl(scrolled, wxID_ANY, wxString::FromUTF8(config.identity_directive), 
                                               wxDefaultPosition, wxSize(-1, 100), 
                                               wxTE_MULTILINE | wxTE_WORDWRAP);
        ctrls.identity_directive->SetFont(monospace_font);
        sizer->Add(ctrls.identity_directive, 0, wxEXPAND | wxALL, 5);
        
        // Other Directives (2/3 height)
        sizer->Add(new wxStaticText(scrolled, wxID_ANY, "Other Directives:"), 0, wxALL, 5);
        ctrls.other_directives = new wxTextCtrl(scrolled, wxID_ANY, wxString::FromUTF8(config.other_directives), 
                                             wxDefaultPosition, wxSize(-1, 200), 
                                             wxTE_MULTILINE | wxTE_WORDWRAP);
        ctrls.other_directives->SetFont(monospace_font);
        sizer->Add(ctrls.other_directives, 0, wxEXPAND | wxALL, 5);
        
        scrolled->SetSizer(sizer);
        
        auto* panel_sizer = new wxBoxSizer(wxVERTICAL);
        panel_sizer->Add(scrolled, 1, wxEXPAND);
        panel->SetSizer(panel_sizer);
        
        notebook->AddPage(panel, "System Prompt");
    }
    
    void CreateChatTemplateTab(wxNotebook* notebook) {
        auto* panel = new wxPanel(notebook);
        auto* scrolled = new wxScrolledWindow(panel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxVSCROLL);
        scrolled->SetScrollRate(0, 20);
        
        auto* sizer = new wxBoxSizer(wxVERTICAL);
        
        sizer->Add(new wxStaticText(scrolled, wxID_ANY, "Chat Template (Jinja2 format):"), 0, wxALL, 5);
        sizer->Add(new wxStaticText(scrolled, wxID_ANY, "Leave empty to use model's default template"), 0, wxALL, 5);
        
        ctrls.chat_template = new wxTextCtrl(scrolled, wxID_ANY, wxString::FromUTF8(config.chat_template), 
                                           wxDefaultPosition, wxSize(-1, 200), 
                                           wxTE_MULTILINE | wxTE_WORDWRAP);
        ctrls.chat_template->SetFont(monospace_font);
        
        sizer->Add(ctrls.chat_template, 0, wxEXPAND | wxALL, 5);
        
        // Add helpful text
        wxStaticText* help_text = new wxStaticText(scrolled, wxID_ANY, 
            "Common variables: {{ messages }}, {{ add_generation_prompt }}\n"
            "Example: {% for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}");
        help_text->SetFont(help_font);
        sizer->Add(help_text, 0, wxALL, 5);
        
        scrolled->SetSizer(sizer);
        
        auto* panel_sizer = new wxBoxSizer(wxVERTICAL);
        panel_sizer->Add(scrolled, 1, wxEXPAND);
        panel->SetSizer(panel_sizer);
        
        notebook->AddPage(panel, "Chat Template");
    }
    
    void CreateSummarizerTab(wxNotebook* notebook) {
        auto* panel = new wxPanel(notebook);
        auto* scrolled = new wxScrolledWindow(panel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxVSCROLL);
        scrolled->SetScrollRate(0, 20);
        
        auto* sizer = new wxBoxSizer(wxVERTICAL);
        
        // Model file path with browse button
        sizer->Add(new wxStaticText(scrolled, wxID_ANY, "Summarization Model File Path:"), 0, wxALL, 5);
        
        auto* path_sizer = new wxBoxSizer(wxHORIZONTAL);
        ctrls.summarizer_model_path = new wxTextCtrl(scrolled, wxID_ANY, config.summarizer_model_path, 
                                                   wxDefaultPosition, wxDefaultSize, wxTE_READONLY);
        auto* browse_btn = new wxButton(scrolled, static_cast<int>(EventId::BROWSE_MODEL) + 100, "Browse...");
        
        path_sizer->Add(ctrls.summarizer_model_path, 1, wxEXPAND | wxRIGHT, 5);
        path_sizer->Add(browse_btn, 0, wxALIGN_CENTER_VERTICAL);
        sizer->Add(path_sizer, 0, wxEXPAND | wxALL, 5);
        
        // Numeric settings
        AddNumericSetting(scrolled, sizer, "Context Size (tokens):", ctrls.summarizer_context_size, config.summarizer_context_size);
        AddNumericSetting(scrolled, sizer, "GPU Offload Layers (0 = CPU only):", ctrls.summarizer_gpu_layers, config.summarizer_gpu_layers);
        AddNumericSetting(scrolled, sizer, "Max Prediction Tokens:", ctrls.summarizer_predict_tokens, config.summarizer_predict_tokens);
        
        // System Prompt
        sizer->Add(new wxStaticText(scrolled, wxID_ANY, "System Prompt:"), 0, wxALL, 5);
        ctrls.summarizer_system_prompt = new wxTextCtrl(scrolled, wxID_ANY, wxString::FromUTF8(config.summarizer_system_prompt), 
                                                      wxDefaultPosition, wxSize(-1, 120), 
                                                      wxTE_MULTILINE | wxTE_WORDWRAP);
        ctrls.summarizer_system_prompt->SetFont(monospace_font);
        sizer->Add(ctrls.summarizer_system_prompt, 0, wxEXPAND | wxALL, 5);
        
        // Chat Template
        sizer->Add(new wxStaticText(scrolled, wxID_ANY, "Chat Template (leave empty to use model default):"), 0, wxALL, 5);
        ctrls.summarizer_chat_template = new wxTextCtrl(scrolled, wxID_ANY, wxString::FromUTF8(config.summarizer_chat_template), 
                                                       wxDefaultPosition, wxSize(-1, 120), 
                                                       wxTE_MULTILINE | wxTE_WORDWRAP);
        ctrls.summarizer_chat_template->SetFont(monospace_font);
        sizer->Add(ctrls.summarizer_chat_template, 0, wxEXPAND | wxALL, 5);
        
        scrolled->SetSizer(sizer);
        
        auto* panel_sizer = new wxBoxSizer(wxVERTICAL);
        panel_sizer->Add(scrolled, 1, wxEXPAND);
        panel->SetSizer(panel_sizer);
        
        notebook->AddPage(panel, "Summarizer");
        
        // Bind browse button event
        browse_btn->Bind(wxEVT_COMMAND_BUTTON_CLICKED, [this](wxCommandEvent&) {
            wxFileDialog file_dialog(this, "Choose Summarizer Model File", "", "", 
                                    "GGUF files (*.gguf)|*.gguf|All files (*.*)|*.*",
                                    wxFD_OPEN | wxFD_FILE_MUST_EXIST);
            
            if (file_dialog.ShowModal() == wxID_OK) {
                ctrls.summarizer_model_path->SetValue(file_dialog.GetPath());
            }
        });
    }
    
    void CreateDiscordSettingsTab(wxNotebook* notebook) {
        auto* panel = new wxPanel(notebook);
        auto* scrolled = new wxScrolledWindow(panel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxVSCROLL);
        scrolled->SetScrollRate(0, 20);
        
        auto* sizer = new wxBoxSizer(wxVERTICAL);
        
        // Streamlined Discord settings creation
        AddTextSetting(scrolled, sizer, "Discord Bot Token:", ctrls.bot_token, config.discord_bot_token, wxTE_PASSWORD);
        
        ctrls.allow_dms = new wxCheckBox(scrolled, wxID_ANY, "Allow Direct Messages");
        ctrls.allow_dms->SetValue(config.discord_allow_dms);
        sizer->Add(ctrls.allow_dms, 0, wxALL, 5);
        
        // History settings in horizontal layout
        auto* history_sizer = new wxBoxSizer(wxHORIZONTAL);
        ctrls.pull_history = new wxCheckBox(scrolled, wxID_ANY, "Pull Message History");
        ctrls.pull_history->SetValue(config.discord_pull_history);
        
        ctrls.history_percentage = new wxSlider(scrolled, wxID_ANY, config.discord_history_percentage, 
                                               10, 80, wxDefaultPosition, wxSize(120, -1));
        ctrls.percentage_label = new wxStaticText(scrolled, wxID_ANY, 
                                                 wxString::Format("%d%%", config.discord_history_percentage));
        
        history_sizer->Add(ctrls.pull_history, 0, wxALIGN_CENTER_VERTICAL | wxRIGHT, 20);
        history_sizer->Add(ctrls.history_percentage, 0, wxALIGN_CENTER_VERTICAL | wxRIGHT, 10);
        history_sizer->Add(ctrls.percentage_label, 0, wxALIGN_CENTER_VERTICAL);
        sizer->Add(history_sizer, 0, wxALL, 5);
        
        AddTextSetting(scrolled, sizer, "Isolated Context Channels:", ctrls.isolated_channels, 
                      config.discord_isolated_channels, wxTE_MULTILINE);
        AddTextSetting(scrolled, sizer, "Shared Context Channels:", ctrls.shared_channels, 
                      config.discord_shared_channels, wxTE_MULTILINE);
        
        scrolled->SetSizer(sizer);
        
        auto* panel_sizer = new wxBoxSizer(wxVERTICAL);
        panel_sizer->Add(scrolled, 1, wxEXPAND);
        panel->SetSizer(panel_sizer);
        
        notebook->AddPage(panel, "Discord Settings");
    }
    
    // Helper method to reduce code duplication
    void AddTextSetting(wxWindow* parent, wxBoxSizer* sizer, const wxString& label, 
                       wxTextCtrl*& control, const std::string& value, long style = 0) {
        sizer->Add(new wxStaticText(parent, wxID_ANY, label), 0, wxALL, 5);
        control = new wxTextCtrl(parent, wxID_ANY, wxString::FromUTF8(value), 
                                wxDefaultPosition, (style & wxTE_MULTILINE) ? wxSize(-1, 80) : wxDefaultSize, style);
        if (style & wxTE_MULTILINE) {
            control->SetFont(monospace_font);
        }
        sizer->Add(control, 0, wxEXPAND | wxALL, 5);
    }
    
    void BindEvents() {
        Bind(wxEVT_COMMAND_BUTTON_CLICKED, &SettingsDialog::OnBrowseModel, this, static_cast<int>(EventId::BROWSE_MODEL));
        Bind(wxEVT_COMMAND_BUTTON_CLICKED, &SettingsDialog::OnOK, this, wxID_OK);
        
        // Bind slider event for real-time percentage update
        if (ctrls.history_percentage) {
            ctrls.history_percentage->Bind(wxEVT_SLIDER, [this](wxCommandEvent&) {
                if (ctrls.percentage_label) {
                    ctrls.percentage_label->SetLabel(wxString::Format("%d%%", 
                                                    ctrls.history_percentage->GetValue()));
                }
            });
        }
    }
    
    // Simplified validation with better error handling
    template<typename T>
    bool ValidateNumeric(wxTextCtrl* control, T& target, T min_val, T max_val, 
                        T default_val, const wxString& field_name) {
        if (!control) return false;
        
        long value;
        if (control->GetValue().ToLong(&value) && value >= min_val && value <= max_val) {
            target = static_cast<T>(value);
            return true;
        }
        
        target = default_val;
        control->SetValue(wxString::Format("%ld", static_cast<long>(default_val)));
        
        wxMessageBox(wxString::Format("Invalid %s. Using default: %ld", field_name, 
                    static_cast<long>(default_val)), "Validation Error", wxOK | wxICON_WARNING);
        return false;
    }
    
    void OnBrowseModel(wxCommandEvent& event) {
        wxFileDialog file_dialog(this, "Choose Model File", "", "", 
                                "GGUF files (*.gguf)|*.gguf|All files (*.*)|*.*",
                                wxFD_OPEN | wxFD_FILE_MUST_EXIST);
        
        if (file_dialog.ShowModal() == wxID_OK) {
            ctrls.model_path->SetValue(file_dialog.GetPath());
        }
    }
    
    void OnOK(wxCommandEvent& event) {
        // Update all configuration values
        config.model_path = ctrls.model_path->GetValue().ToStdString();
        
        ValidateNumeric(ctrls.context_size, config.context_size, 1, 131072, 2048, "context size");
        ValidateNumeric(ctrls.gpu_layers, config.gpu_layers, 0, 999, 0, "GPU layers");
        ValidateNumeric(ctrls.predict_tokens, config.predict_tokens, 1, 4096, 256, "prediction tokens");
        
        config.chat_template = ctrls.chat_template->GetValue().ToUTF8().data();
        config.identity_directive = ctrls.identity_directive->GetValue().ToUTF8().data();
        config.other_directives = ctrls.other_directives->GetValue().ToUTF8().data();
        config.discord_bot_token = ctrls.bot_token->GetValue().ToUTF8().data();
        config.discord_isolated_channels = ValidateChannelIds(ctrls.isolated_channels->GetValue().ToUTF8().data());
        config.discord_shared_channels = ValidateChannelIds(ctrls.shared_channels->GetValue().ToUTF8().data());
        config.discord_allow_dms = ctrls.allow_dms->GetValue();
        config.discord_pull_history = ctrls.pull_history->GetValue();
        config.discord_history_percentage = ctrls.history_percentage->GetValue();
        
        // Summarizer settings
        config.summarizer_model_path = ctrls.summarizer_model_path->GetValue().ToStdString();
        ValidateNumeric(ctrls.summarizer_context_size, config.summarizer_context_size, 1, 32768, 1024, "summarizer context size");
        ValidateNumeric(ctrls.summarizer_gpu_layers, config.summarizer_gpu_layers, 0, 999, 0, "summarizer GPU layers");
        ValidateNumeric(ctrls.summarizer_predict_tokens, config.summarizer_predict_tokens, 1, 2048, 128, "summarizer prediction tokens");
        config.summarizer_system_prompt = ctrls.summarizer_system_prompt->GetValue().ToUTF8().data();
        config.summarizer_chat_template = ctrls.summarizer_chat_template->GetValue().ToUTF8().data();
        
        // Save settings with all parameters
        SettingsManager::SaveSettings(config.model_path, config.context_size, config.gpu_layers, 
                                    config.predict_tokens, config.chat_template, 
                                    config.identity_directive, config.other_directives,
                                    config.discord_bot_token, config.discord_isolated_channels, 
                                    config.discord_shared_channels, config.discord_allow_dms, 
                                    config.discord_pull_history, config.discord_history_percentage,
                                    config.summarizer_model_path, config.summarizer_context_size,
                                    config.summarizer_gpu_layers, config.summarizer_predict_tokens,
                                    config.summarizer_system_prompt, config.summarizer_chat_template);
        
        EndModal(wxID_OK);
    }

    // Optimized channel ID validation
    std::string ValidateChannelIds(const std::string& input) {
        if (input.empty()) return {};
        
        std::string result;
        result.reserve(input.length());
        
        std::string current_id;
        current_id.reserve(32);
        
        for (char c : input) {
            if (std::isdigit(c)) {
                current_id += c;
            } else if (!current_id.empty() && (c == ',' || std::isspace(c))) {
                if (!result.empty()) result += ',';
                result += current_id;
                current_id.clear();
            }
        }
        
        if (!current_id.empty()) {
            if (!result.empty()) result += ',';
            result += current_id;
        }
        
        return result;
    }
};

// Main Frame with optimized performance and simplified structure
class LuminaChatFrame : public wxFrame {
private:
    // Core components
    std::unique_ptr<LlamaManager> llama_manager;
    std::unique_ptr<DiscordManager> discord_manager;
    
    // Console redirection
    std::unique_ptr<wxLogStreamBuffer> cout_buffer, cerr_buffer;
    std::streambuf* original_cout{nullptr};
    std::streambuf* original_cerr{nullptr};
    
    // Consolidated configuration structure
    struct AppConfig {
        std::string model_path, chat_template;
        std::string identity_directive, other_directives;
        std::string discord_token, discord_isolated_channels, discord_shared_channels;
        std::string summarizer_model_path, summarizer_system_prompt, summarizer_chat_template;
        int32_t context_size{2048}, gpu_layers{0}, predict_tokens{256};
        int32_t summarizer_context_size{1024}, summarizer_gpu_layers{0}, summarizer_predict_tokens{128};
        int32_t discord_history_percentage{50};
        bool discord_allow_dms{true}, discord_pull_history{true};
    } config;    // State management
    std::atomic<bool> is_started{false}, is_processing{false}, context_created{false};
    
    // Context monitoring timer
    wxTimer* context_monitor_timer;    // UI controls with better organization
    struct UIControls {
        wxButton *start_btn, *stop_btn, *settings_btn, *discord_btn, *prune_btn;
        wxGauge* progress_bar;
        wxGauge* context_progress_bar;
        wxStaticText *progress_label, *timings_label, *context_label;
        wxRichTextCtrl* chat_history;
        wxTextCtrl *input_text, *logs_text, *summaries_text;
        wxNotebook* notebook;
        wxPanel* main_panel;
    } ui;
    
    ModelWorkerThread* worker_thread{nullptr};

public:
    LuminaChatFrame() : wxFrame(nullptr, wxID_ANY, "LuminaChat", wxDefaultPosition, wxSize(800, 600)) {
        g_main_frame = this;
        
        llama_manager = std::make_unique<LlamaManager>();
        discord_manager = std::make_unique<DiscordManager>();
        
        // Initialize context monitoring timer
        context_monitor_timer = new wxTimer(this, static_cast<int>(EventId::CONTEXT_MONITOR_TIMER));
          // Set up unified logging
        LogHandler::set_output_callback([this](const std::string& msg) {
            AppendToLogsThreadSafe(wxString::FromUTF8(msg));
        });
        
        // Set up summary logging callbacks
        llama_manager->set_summary_input_callback([this](const std::string& input) {
            AppendSummaryInput(wxString::FromUTF8(input));
        });
        
        llama_manager->set_summary_output_callback([this](const std::string& output) {
            AppendSummaryOutput(wxString::FromUTF8(output));
        });
        
        LoadConfiguration();
        CreateUI();
        SetupConsoleRedirection();
        UpdateButtonStates();
        BindEvents();
    }    ~LuminaChatFrame() {
        if (context_monitor_timer) {
            context_monitor_timer->Stop();
            delete context_monitor_timer;
        }
        RestoreConsoleStreams();
        CleanupWorkerThread();
        g_main_frame = nullptr;
    }

    void AppendToLogsThreadSafe(const wxString& message) {
        if (!message.IsEmpty()) {
            CallAfter([this, message]() {
                if (ui.logs_text) {
                    ui.logs_text->AppendText(message);
                    ui.logs_text->SetInsertionPointEnd();
                }
            });
        }
    }

    void AppendToSummariesThreadSafe(const wxString& message) {
        if (!message.IsEmpty()) {
            CallAfter([this, message]() {
                if (ui.summaries_text) {
                    ui.summaries_text->AppendText(message);
                    ui.summaries_text->SetInsertionPointEnd();
                }
            });
        }
    }

    void AppendSummaryInput(const wxString& input) {
        wxString timestamp = wxDateTime::Now().Format("%H:%M:%S");
        wxString formatted = wxString::Format("[%s] INPUT:\n%s\n\n", timestamp, input);
        AppendToSummariesThreadSafe(formatted);
    }
    
    void AppendSummaryOutput(const wxString& output) {
        wxString timestamp = wxDateTime::Now().Format("%H:%M:%S");
        wxString formatted = wxString::Format("[%s] OUTPUT:\n%s\n\n%s\n\n", 
                                            timestamp, output, wxString(60, '-'));
        AppendToSummariesThreadSafe(formatted);
    }

private:
    void LoadConfiguration() {
        SettingsManager::LoadSettings(config.model_path, config.context_size, config.gpu_layers, 
                                    config.predict_tokens, config.chat_template, 
                                    config.identity_directive, config.other_directives,
                                    config.discord_token, config.discord_isolated_channels, 
                                    config.discord_shared_channels, config.discord_allow_dms,
                                    config.discord_pull_history, config.discord_history_percentage,
                                    config.summarizer_model_path, config.summarizer_context_size,
                                    config.summarizer_gpu_layers, config.summarizer_predict_tokens,
                                    config.summarizer_system_prompt, config.summarizer_chat_template);
    }
    
    void CreateUI() {
        ui.main_panel = new wxPanel(this);
        
        CreateToolbar();
        CreateStatusArea();
        CreateNotebook();
        LayoutComponents();
        
        SetMinSize(wxSize(600, 400));
        AddWelcomeMessage();
    }
      void CreateToolbar() {
        ui.start_btn = new wxButton(ui.main_panel, static_cast<int>(EventId::START), "Start");
        ui.stop_btn = new wxButton(ui.main_panel, static_cast<int>(EventId::STOP), "Stop");
        ui.settings_btn = new wxButton(ui.main_panel, static_cast<int>(EventId::SETTINGS), "Settings");
        ui.discord_btn = new wxButton(ui.main_panel, static_cast<int>(EventId::CONNECT_DISCORD), "Connect Discord");
        ui.prune_btn = new wxButton(ui.main_panel, static_cast<int>(EventId::PRUNE_SUMMARIZE), "Prune && Summarize");
    }void CreateStatusArea() {
        ui.progress_label = new wxStaticText(ui.main_panel, wxID_ANY, "Ready");
        ui.progress_bar = new wxGauge(ui.main_panel, wxID_ANY, 100, wxDefaultPosition, wxSize(-1, 20));
        ui.progress_bar->Hide();
        
        ui.context_label = new wxStaticText(ui.main_panel, wxID_ANY, "Context: N/A", wxDefaultPosition, wxSize(120, -1));
        ui.context_progress_bar = new wxGauge(ui.main_panel, wxID_ANY, 100, wxDefaultPosition, wxSize(200, 15));
        
        ui.timings_label = new wxStaticText(ui.main_panel, wxID_ANY, "");
    }
      void CreateNotebook() {
        ui.notebook = new wxNotebook(ui.main_panel, wxID_ANY);
        
        CreateChatTab();
        CreateSummariesTab();
        CreateLogsTab();
    }
    
    void CreateChatTab() {
        auto* panel = new wxPanel(ui.notebook);
        auto* sizer = new wxBoxSizer(wxVERTICAL);
        
        ui.chat_history = new wxRichTextCtrl(panel, wxID_ANY, "", wxDefaultPosition, wxDefaultSize,
                                           wxRE_READONLY | wxRE_MULTILINE | wxVSCROLL);
        ui.input_text = new wxTextCtrl(panel, static_cast<int>(EventId::INPUT_TEXT), "", 
                                      wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER);
        
        sizer->Add(ui.chat_history, 1, wxEXPAND | wxALL, 5);
        sizer->Add(ui.input_text, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 5);
        
        panel->SetSizer(sizer);
        ui.notebook->AddPage(panel, "Chat");
    }
      void CreateSummariesTab() {
        auto* panel = new wxPanel(ui.notebook);
        auto* sizer = new wxBoxSizer(wxVERTICAL);
        
        ui.summaries_text = new wxTextCtrl(panel, wxID_ANY, "", wxDefaultPosition, wxDefaultSize,
                                         wxTE_MULTILINE | wxTE_READONLY | wxTE_WORDWRAP);
        ui.summaries_text->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
        
        // Add initial welcome message with helpful information
        ui.summaries_text->SetValue("Summary Monitor - Track summarization inputs and outputs\n"
                                   "============================================================\n\n"
                                   "This tab shows when the AI summarizes old conversation messages\n"
                                   "to make room for new ones when the context gets full.\n\n"
                                   "INPUT: Shows the messages being summarized\n"
                                   "OUTPUT: Shows the generated summary\n\n"
                                   "Note: Summarization requires a summary model to be configured\n"
                                   "in the Settings -> Summarizer tab.\n\n"
                                   "Waiting for first summarization...\n\n");
        
        sizer->Add(ui.summaries_text, 1, wxEXPAND | wxALL, 5);
        panel->SetSizer(sizer);
        ui.notebook->AddPage(panel, "Summaries");
    }
    
    void CreateLogsTab() {
        auto* panel = new wxPanel(ui.notebook);
        auto* sizer = new wxBoxSizer(wxVERTICAL);
        
        ui.logs_text = new wxTextCtrl(panel, wxID_ANY, "", wxDefaultPosition, wxDefaultSize,
                                     wxTE_MULTILINE | wxTE_READONLY | wxTE_WORDWRAP);
        ui.logs_text->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
        
        sizer->Add(ui.logs_text, 1, wxEXPAND | wxALL, 5);
        panel->SetSizer(sizer);
        ui.notebook->AddPage(panel, "Logs");
    }    void LayoutComponents() {
        auto* toolbar_sizer = new wxBoxSizer(wxHORIZONTAL);
        toolbar_sizer->Add(ui.start_btn, 0, wxRIGHT, 5);
        toolbar_sizer->Add(ui.stop_btn, 0, wxRIGHT, 5);  
        toolbar_sizer->Add(ui.settings_btn, 0, wxRIGHT, 5);
        toolbar_sizer->Add(ui.discord_btn, 0, wxRIGHT, 5);
        toolbar_sizer->Add(ui.prune_btn, 0);
        toolbar_sizer->AddStretchSpacer();
        
        // Context monitoring area - fixed sizing and spacing
        auto* context_sizer = new wxBoxSizer(wxHORIZONTAL);
        context_sizer->Add(ui.context_label, 0, wxALIGN_CENTER_VERTICAL | wxRIGHT, 5);
        context_sizer->Add(ui.context_progress_bar, 0, wxALIGN_CENTER_VERTICAL);
        context_sizer->AddStretchSpacer();
        
        auto* status_sizer = new wxBoxSizer(wxHORIZONTAL);
        status_sizer->Add(ui.progress_label, 0, wxALIGN_CENTER_VERTICAL);
        status_sizer->AddSpacer(20);
        status_sizer->Add(ui.timings_label, 1, wxALIGN_CENTER_VERTICAL);
        
        auto* main_sizer = new wxBoxSizer(wxVERTICAL);
        main_sizer->Add(toolbar_sizer, 0, wxEXPAND | wxALL, 10);
        main_sizer->Add(context_sizer, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 10);
        main_sizer->Add(status_sizer, 0, wxEXPAND | wxLEFT | wxRIGHT, 10);
        main_sizer->Add(ui.progress_bar, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 10);
        main_sizer->Add(ui.notebook, 1, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 10);
        
        ui.main_panel->SetSizer(main_sizer);
    }      void BindEvents() {
        Bind(wxEVT_COMMAND_BUTTON_CLICKED, &LuminaChatFrame::OnStart, this, static_cast<int>(EventId::START));
        Bind(wxEVT_COMMAND_BUTTON_CLICKED, &LuminaChatFrame::OnStop, this, static_cast<int>(EventId::STOP));
        Bind(wxEVT_COMMAND_BUTTON_CLICKED, &LuminaChatFrame::OnSettings, this, static_cast<int>(EventId::SETTINGS));
        Bind(wxEVT_COMMAND_BUTTON_CLICKED, &LuminaChatFrame::OnConnectDiscord, this, static_cast<int>(EventId::CONNECT_DISCORD));
        Bind(wxEVT_COMMAND_BUTTON_CLICKED, &LuminaChatFrame::OnPruneSummarize, this, static_cast<int>(EventId::PRUNE_SUMMARIZE));
        Bind(wxEVT_COMMAND_TEXT_ENTER, &LuminaChatFrame::OnInputEnter, this, static_cast<int>(EventId::INPUT_TEXT));
        
        Bind(wxEVT_MODEL_LOADED, &LuminaChatFrame::OnModelLoaded, this);
        Bind(wxEVT_RESPONSE_READY, &LuminaChatFrame::OnResponseReady, this);
        Bind(wxEVT_PROGRESS_UPDATE, &LuminaChatFrame::OnProgressUpdate, this);
        
        // Bind context monitoring timer
        Bind(wxEVT_TIMER, &LuminaChatFrame::OnContextMonitorTimer, this, static_cast<int>(EventId::CONTEXT_MONITOR_TIMER));
    }
    
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
        if (!config.model_path.empty()) {
            wxFileName modelFile(wxString::FromUTF8(config.model_path));
            SetTitle(wxString::Format("LuminaChat - %s", modelFile.GetName()));
        } else {
            SetTitle("LuminaChat");
        }
    }
    
    void RestoreConsoleStreams() noexcept {
        try {
            if (original_cout) {
                std::cout.rdbuf(original_cout);
            }
            if (original_cerr) {
                std::cerr.rdbuf(original_cerr);
            }
        } catch (...) {
            // Ignore errors during cleanup
        }
    }
    
    void CleanupWorkerThread() noexcept {
        if (worker_thread) {
            worker_thread->RequestStop();
            worker_thread = nullptr;
        }
    }
      void UpdateButtonStates() noexcept {
        const bool started = is_started.load();
        const bool processing = is_processing.load();
        
        if (ui.start_btn) ui.start_btn->Enable(!started && !processing);
        if (ui.stop_btn) ui.stop_btn->Enable(started);
        if (ui.input_text) ui.input_text->Enable(started && !processing);
        if (ui.settings_btn) ui.settings_btn->Enable(!processing);
        if (ui.prune_btn) ui.prune_btn->Enable(started && !processing);
        
        UpdateDiscordButtonState();
    }
    
    void UpdateDiscordButtonState() noexcept {
        if (!ui.discord_btn || !discord_manager) return;
        
        ui.discord_btn->SetLabel(discord_manager->is_bot_running() ? 
                               "Disconnect Discord" : "Connect Discord");
    }
    
    // Optimized message formatting with better memory management
    void AddChatMessage(const wxString& message, const wxString& sender, const wxColour& bg_color) {
        wxRichTextAttr attr;
        attr.SetTextColour(*wxBLACK);
        attr.SetBackgroundColour(bg_color);
        attr.SetLeftIndent(50);
        attr.SetRightIndent(50);
        attr.SetParagraphSpacingBefore(0);
        attr.SetParagraphSpacingAfter(0);
        
        if (sender != "System") ui.chat_history->WriteText("\n");
        
        ui.chat_history->BeginStyle(attr);
        ui.chat_history->WriteText(sender + ": " + message);
        ui.chat_history->EndStyle();
        ui.chat_history->WriteText(sender == "You" ? "\n" : "\n\n");
        
        ui.chat_history->SetInsertionPointEnd();
        ui.chat_history->ShowPosition(ui.chat_history->GetLastPosition());
    }
    
    void AddUserMessage(const wxString& message) {
        AddChatMessage(message, "You", wxColour(173, 216, 230));
    }
    
    void AddAIMessage(const wxString& message) {
        AddChatMessage(message, "AI", wxColour(144, 238, 144));
    }
    
    void AddSystemMessage(const wxString& message) {
        wxRichTextAttr attr;
        attr.SetTextColour(wxColour(100, 100, 100));
        attr.SetBackgroundColour(wxColour(245, 245, 245));
        attr.SetLeftIndent(30);
        attr.SetRightIndent(30);
        attr.SetFontStyle(wxFONTSTYLE_ITALIC);
        
        ui.chat_history->BeginStyle(attr);
        ui.chat_history->WriteText("System: " + message);
        ui.chat_history->EndStyle();
        ui.chat_history->WriteText("\n\n");
    }
    
    void AddWelcomeMessage() {
        AddSystemMessage("Welcome to LuminaChat!\nClick 'Settings' to select a model, then 'Start' to begin.");
    }
      std::string GetCombinedSystemPrompt() const {
        if (config.identity_directive.empty() && config.other_directives.empty()) {
            return {};
        }
        
        std::string combined = config.identity_directive;
        if (!config.identity_directive.empty() && !config.other_directives.empty()) {
            combined += "\n\n";
        }
        combined += config.other_directives;
        
        return combined;
    }
      // Optimized event handler methods
    void OnPruneSummarize(wxCommandEvent& event) {
        if (!is_started || is_processing || !llama_manager) {
            wxMessageBox("Please start the model first.", "Model Not Started", 
                        wxOK | wxICON_WARNING);
            return;
        }
        
        // Ensure we're using the main chat context
        if (!llama_manager->switch_to_context("main_chat")) {
            wxMessageBox("Failed to switch to main chat context.", "Context Error", 
                        wxOK | wxICON_ERROR);
            return;
        }
        
        // Check if there's a conversation to prune
        if (llama_manager->get_message_count() < 3) {
            wxMessageBox("Need at least 3 messages in the conversation to prune.", 
                        "Insufficient Messages", wxOK | wxICON_INFORMATION);
            return;
        }
        
        // Check if summarization is available
        if (!llama_manager->is_summarization_available()) {
            wxMessageBox("Summarization is not available. Please configure a summary model in Settings.", 
                        "Summarization Unavailable", wxOK | wxICON_WARNING);
            return;
        }
        
        try {
            // Show progress
            AddSystemMessage("Starting prune and summarize (keeping 90% of context)...");
            
            // Perform pruning with 90% keep ratio (10% prune)
            bool success = llama_manager->prune_conversation_with_summary(0.9f);
            
            if (success) {
                AddSystemMessage("Context pruned and summarized successfully.");
                UpdateContextProgress();
            } else {
                AddSystemMessage("Failed to prune and summarize context.");
            }
            
        } catch (const std::exception& e) {
            AddSystemMessage(wxString::Format("Error during pruning: %s", e.what()));
        }
    }
    
    void OnStart(wxCommandEvent& event) {
        if (config.model_path.empty()) {
            wxMessageBox("Please select a model file in Settings first.", "No Model Selected", 
                        wxOK | wxICON_WARNING);
            return;
        }
        
        if (is_processing) return;

        // Set default system prompt if empty
        if (config.identity_directive.empty() && config.other_directives.empty()) {
            config.identity_directive = "You are a helpful AI assistant named Lumina.";
            config.other_directives = "Answer each user request thoughtfully and to the best of your ability.";
            
            SettingsManager::SaveSettings(config.model_path, config.context_size, config.gpu_layers, 
                                        config.predict_tokens, config.chat_template, 
                                        config.identity_directive, config.other_directives,
                                        config.discord_token, config.discord_isolated_channels, 
                                        config.discord_shared_channels, config.discord_allow_dms,
                                        config.discord_pull_history, config.discord_history_percentage,
                                        config.summarizer_model_path, config.summarizer_context_size,
                                        config.summarizer_gpu_layers, config.summarizer_predict_tokens,
                                        config.summarizer_system_prompt, config.summarizer_chat_template);
        }

        is_processing = true;
        UpdateButtonStates();
        
        ui.progress_label->SetLabel("Loading model...");
        ui.progress_bar->SetValue(0);
        ui.progress_bar->Show();
        ui.main_panel->Layout();
        
        LLAMA_LOG("Loading model: " + config.model_path);
        
        worker_thread = new ModelWorkerThread(this, llama_manager.get(), ModelWorkerThread::Operation::LOAD_MODEL);
        worker_thread->SetModelParams(config.model_path, config.context_size, config.gpu_layers, 
                                     config.predict_tokens, config.chat_template,
                                     config.summarizer_model_path, config.summarizer_context_size,
                                     config.summarizer_gpu_layers, config.summarizer_predict_tokens,
                                     config.summarizer_chat_template);
        
        if (worker_thread->Run() != wxTHREAD_NO_ERROR) {
            delete worker_thread;
            worker_thread = nullptr;
            is_processing = false;
            ui.progress_bar->Hide();
            ui.progress_label->SetLabel("Ready");
            ui.main_panel->Layout();
            UpdateButtonStates();
            AddSystemMessage("Error: Failed to start model loading");
        }
    }
    
    void OnProgressUpdate(wxCommandEvent& event) {
        int32_t percent = event.GetInt();
        ui.progress_bar->SetValue(percent);
        ui.progress_label->SetLabel(wxString::Format("Loading model... %d%%", percent));
        
        // Force UI update with proper refresh
        ui.progress_bar->Refresh();
        ui.progress_label->Refresh();
    }
    
    void OnModelLoaded(wxCommandEvent& event) {
        is_processing = false;
        worker_thread = nullptr;
        
        // Hide progress bar with proper layout update
        ui.progress_bar->Hide();
        ui.progress_label->SetLabel("Ready");
        ui.main_panel->Layout();
        
        bool success = event.GetInt() == 1;
        
        if (success) {
            // Handle chat template first - only save model template if no custom template was provided
            if (config.chat_template.empty()) {
                std::string model_template = llama_manager->get_model_chat_template();
                if (!model_template.empty()) {
                    config.chat_template = model_template;
                    SettingsManager::SaveSettings(config.model_path, config.context_size, config.gpu_layers, config.predict_tokens, 
                                                config.chat_template, 
                                                config.identity_directive, config.other_directives,
                                                config.discord_token, config.discord_isolated_channels, 
                                                config.discord_shared_channels, config.discord_allow_dms,
                                                config.discord_pull_history, config.discord_history_percentage,
                                                config.summarizer_model_path, config.summarizer_context_size,
                                                config.summarizer_gpu_layers, config.summarizer_predict_tokens,
                                                config.summarizer_system_prompt, config.summarizer_chat_template);
                    LLAMA_LOG("Saved model's default chat template to settings");
                }
            }
            
            // Create main chat context with system prompt from settings
            std::string combined_prompt = GetCombinedSystemPrompt();
            if (!llama_manager->create_context("main_chat", "main_model", combined_prompt)) {
                LLAMA_LOG_ERROR("Failed to create main chat context");
                success = false;
            } else {
                context_created = true;
                LLAMA_LOG("Main chat context created with system prompt");
                
                // Create summary context if summarizer model is loaded
                if (!config.summarizer_model_path.empty() && config.summarizer_model_path != config.model_path) {
                    std::string summary_prompt = config.summarizer_system_prompt.empty() ? 
                        "You are a helpful assistant that summarizes conversations concisely and accurately." : 
                        config.summarizer_system_prompt;
                    
                    if (llama_manager->create_context("summary_context", "summary_model", summary_prompt, true)) {
                        LLAMA_LOG("Summary context created successfully with summarizer model");
                        // Note: Summarizer chat template was already set during model loading
                    } else {
                        LLAMA_LOG("Warning: Failed to create summary context, summarization features may be limited");
                    }
                }
            }
        }
          if (success) {
            is_started = true;
            
            // Start context monitoring timer (update every 2 seconds)
            if (context_monitor_timer) {
                context_monitor_timer->Start(2000);
            }
              // Connect Discord manager to LlamaManager when model is loaded
            if (discord_manager && discord_manager->is_bot_running()) {
                discord_manager->set_llama_manager(llama_manager.get());
                discord_manager->set_main_context_id("main_chat");
                discord_manager->set_model_id("main_model");
                DISCORD_LOG("Discord bot connected to loaded model");
            }
            
            ui.chat_history->Clear();
            AddSystemMessage("LuminaChat ready! Type your message below.");
            ui.input_text->SetFocus();
            
            llama_manager->reset_timings();
            ui.timings_label->SetLabel("");
        } else {
            LLAMA_LOG_ERROR("Failed to load model or create context");
        }
        
        UpdateButtonStates();
    }
      void OnStop(wxCommandEvent& event) {
        if (worker_thread) {
            worker_thread->RequestStop();
            worker_thread = nullptr;
        }
        
        // Stop context monitoring timer
        if (context_monitor_timer) {
            context_monitor_timer->Stop();
        }
        
        // Hide progress bar if visible with proper layout update
        if (ui.progress_bar->IsShown()) {
            ui.progress_bar->Hide();
            ui.progress_label->SetLabel("Ready");
            ui.main_panel->Layout();
        }
          // Reset context progress display
        ui.context_label->SetLabel("Buffer: N/A");
        ui.context_progress_bar->SetValue(0);
        ui.context_progress_bar->SetForegroundColour(wxSystemSettings::GetColour(wxSYS_COLOUR_HIGHLIGHT));
        
        // Disconnect Discord manager from LlamaManager when model is stopped
        if (discord_manager && discord_manager->is_bot_running()) {
            discord_manager->set_llama_manager(nullptr);
            DISCORD_LOG("Discord bot disconnected from model");
        }
        
        llama_manager->cleanup();
        context_created = false;
        is_started = false;
        is_processing = false;
        UpdateButtonStates();
        LLAMA_LOG("LuminaChat stopped.");
        ui.timings_label->SetLabel("");
    }
    
    void OnConnectDiscord(wxCommandEvent& event) {
        if (!discord_manager) {
            AddSystemMessage("Error: Discord manager not available");
            return;
        }
        
        if (discord_manager->is_bot_running()) {
            // Stop Discord bot
            discord_manager->shutdown();
            ui.discord_btn->SetLabel("Connect Discord");
            AddSystemMessage("Discord bot disconnected");
        } else {
            // Configure and start Discord bot
            if (config.discord_token.empty()) {
                AddSystemMessage("Discord bot token not configured. Please check Settings.");
                return;
            }
            
            // Configure bot
            DiscordBotConfig bot_config;
            bot_config.bot_token = config.discord_token;
            
            if (!discord_manager->configure(bot_config)) {
                AddSystemMessage("Failed to configure Discord bot");
                return;
            }
              // Set up integration with LlamaManager
            discord_manager->set_llama_manager(llama_manager.get());
            discord_manager->set_main_context_id("main_chat");
            discord_manager->set_model_id("main_model");
            discord_manager->set_isolated_channels(config.discord_isolated_channels);
            discord_manager->set_shared_history_channels(config.discord_shared_channels);
            discord_manager->set_allow_dms(config.discord_allow_dms);
            discord_manager->set_history_settings(config.discord_pull_history, config.discord_history_percentage);
            
            // Start bot
            if (discord_manager->start()) {
                ui.discord_btn->SetLabel("Disconnect Discord");
                AddSystemMessage("Discord bot connecting...");
                
                // UPDATED: Simplified backfill monitoring for new structure
                std::thread([this]() {
                    bool backfill_reported = false;
                    
                    for (int i = 0; i < 60; ++i) { // Check for up to 5 minutes
                        std::this_thread::sleep_for(std::chrono::seconds(5));
                        
                        if (discord_manager) {
                            auto status = discord_manager->get_backfill_status();
                            
                            if (status.in_progress && !backfill_reported) {
                                DISCORD_HISTORY_LOG("Chat history backfill started for " + 
                                                   std::to_string(status.total_channels) + " configured channels...");
                                backfill_reported = true;
                            }
                            
                            if (status.in_progress && i % 3 == 0) { // Report every 15 seconds
                                DISCORD_HISTORY_LOG("Backfill progress: " + 
                                                   std::to_string(status.completed_channels) + "/" + 
                                                   std::to_string(status.total_channels) + " channels completed, " +
                                                   std::to_string(status.total_messages_fetched) + " messages fetched");
                            }
                            
                            if (!status.in_progress && backfill_reported) {
                                DISCORD_HISTORY_LOG("Chat history backfill completed: " + 
                                                   std::to_string(status.total_messages_fetched) + 
                                                   " messages fetched from " + 
                                                   std::to_string(status.total_channels) + " channels");
                                break;
                            }
                            
                            if (!status.in_progress && i > 4) { // Give time for backfill to start
                                if (config.discord_pull_history && status.total_channels == 0) {
                                    DISCORD_HISTORY_LOG("No accessible channels found for history backfill");
                                } else if (!config.discord_pull_history) {
                                    DISCORD_HISTORY_LOG("Message history backfill disabled in settings");
                                }
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
        SettingsDialog dialog(this, config.model_path, config.context_size, config.gpu_layers, config.predict_tokens, 
                            config.chat_template, config.identity_directive, config.other_directives,
                            config.discord_token, config.discord_isolated_channels, 
                            config.discord_shared_channels, config.discord_allow_dms,
                            config.discord_pull_history, config.discord_history_percentage,
                            config.summarizer_model_path, config.summarizer_context_size,
                            config.summarizer_gpu_layers, config.summarizer_predict_tokens,
                            config.summarizer_system_prompt, config.summarizer_chat_template);
        if (dialog.ShowModal() == wxID_OK) {
            UpdateWindowTitle();
            
            if (is_started) {
                wxMessageBox("Settings have been saved. Please restart the model for changes to take effect.", 
                           "Settings Updated", wxOK | wxICON_INFORMATION);
            }
        }
    }    void OnInputEnter(wxCommandEvent& event) {
        if (!is_started || is_processing) return;
        
        // Check if context is created
        if (!context_created) {
            AddSystemMessage("Error: Chat context not available. Please restart the model.");
            return;
        }
        
        // Ensure we're using the main chat context before processing the message
        if (!llama_manager->switch_to_context("main_chat")) {
            AddSystemMessage("Error: Failed to switch to main chat context.");
            return;
        }
        
        wxString input = ui.input_text->GetValue().Trim();
        if (input.IsEmpty()) return;
        
        // Display user input with blue background
        AddUserMessage(input);
        ui.input_text->Clear();
        
        is_processing = true;
        UpdateButtonStates();
        
        // Start response generation in background thread
        worker_thread = new ModelWorkerThread(this, llama_manager.get(), ModelWorkerThread::Operation::GENERATE_RESPONSE);
        worker_thread->SetInput(input.ToStdString(), "User");
        
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
            
            // Update context progress immediately after response
            UpdateContextProgress();
        }
        
        UpdateButtonStates();
        ui.input_text->SetFocus();
        
        // Auto-scroll to bottom
        ui.chat_history->SetInsertionPointEnd();
        ui.chat_history->ShowPosition(ui.chat_history->GetLastPosition());
    }    void UpdateTimingsDisplay() {
        if (!is_started || !llama_manager) {
            return;
        }
        
        // Get timings from llama manager
        auto timings = llama_manager->get_timings();
        if (timings.n_eval > 0) {
            double tokens_per_sec = 1000.0 * timings.n_eval / timings.t_eval_ms;
            
            // Get additional statistics for comprehensive display
            auto perf_stats = llama_manager->get_performance_stats();
            
            ui.timings_label->SetLabel(wxString::Format("Generated: %d tokens, %.2f tok/s (%.2fms)", 
                                                       timings.n_eval, 
                                                       tokens_per_sec,
                                                       timings.t_eval_ms));
        } else {
            ui.timings_label->SetLabel("");
        }
    }
    
    void OnContextMonitorTimer(wxTimerEvent& event) {
        UpdateContextProgress();
    }    void UpdateContextProgress() {
        if (!is_started || !llama_manager || !context_created || !llama_manager->has_context("main_chat")) {
            ui.context_label->SetLabel("Buffer: N/A");
            ui.context_progress_bar->SetValue(0);
            return;
        }
        
        // Get context usage information directly without switching contexts
        int32_t context_size = llama_manager->get_context_size_for("main_chat");
        int32_t context_usage = llama_manager->get_context_usage_for("main_chat");
        
        if (context_size > 0) {
            float usage_percentage = static_cast<float>(context_usage) / static_cast<float>(context_size) * 100.0f;
            int32_t progress_value = static_cast<int32_t>(usage_percentage);
            
            ui.context_progress_bar->SetValue(std::min(progress_value, 100));
            
            // More descriptive label showing context buffer usage vs conversation tokens
            ui.context_label->SetLabel(wxString::Format("Buffer: %d/%d", context_usage, context_size));
            
            // Change color based on usage (visual feedback)
            if (usage_percentage > 90.0f) {
                ui.context_progress_bar->SetForegroundColour(wxColour(255, 0, 0)); // Red
            } else if (usage_percentage > 75.0f) {
                ui.context_progress_bar->SetForegroundColour(wxColour(255, 165, 0)); // Orange
            } else {
                ui.context_progress_bar->SetForegroundColour(wxColour(0, 128, 0)); // Green
            }
        } else {
            ui.context_label->SetLabel("Buffer: 0/0");
            ui.context_progress_bar->SetValue(0);
        }
    }
};

// Implementation of wxLogStreamBuffer::FlushBuffer
void wxLogStreamBuffer::FlushBuffer() {
    if (buffer.empty() || !frame) return;
    
    // Remove trailing whitespace efficiently
    while (!buffer.empty() && (buffer.back() == '\n' || buffer.back() == '\r')) {
        buffer.pop_back();
    }
    
    if (!buffer.empty()) {
        frame->AppendToLogsThreadSafe(wxString::FromUTF8(buffer) + "\n");
    }
    
    buffer.clear();
}

// Callback implementations
bool model_loading_progress_callback(float progress, void *user_data) {
    if (auto* handler = static_cast<wxEvtHandler*>(user_data)) {
        wxCommandEvent event(wxEVT_PROGRESS_UPDATE);
        event.SetInt(static_cast<int32_t>(progress * 100.0f));
        wxQueueEvent(handler, event.Clone());
    }
    return true;
}

void llama_log_callback(ggml_log_level level, const char* message, void* user_data) {
    if (auto* frame = static_cast<LuminaChatFrame*>(user_data); frame && message) {
        const char* level_str = (level == GGML_LOG_LEVEL_ERROR ? "ERROR" :
                                level == GGML_LOG_LEVEL_WARN  ? "WARN" :
                                level == GGML_LOG_LEVEL_INFO  ? "INFO" : "DEBUG");
        
        frame->AppendToLogsThreadSafe(wxString::Format("[%s] %s", level_str, message));
    }
}

// Application Class
class LuminaChatApp : public wxApp {
public:
    bool OnInit() override {
        SetAppName("LuminaChat");
        SetVendorName("LuminaChat");
        
        try {
            (new LuminaChatFrame())->Show(true);
            return true;
        } catch (const std::exception& e) {
            wxMessageBox(wxString::Format("Failed to initialize: %s", e.what()), 
                        "Error", wxOK | wxICON_ERROR);
            return false;
        }
    }
};

wxIMPLEMENT_APP(LuminaChatApp);
// Cross-platform entry point* argv[]) {
int32_t main(int32_t argc, char* argv[]) {   return wxEntry(argc, argv);
    return wxEntry(argc, argv);
}

//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODE DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
