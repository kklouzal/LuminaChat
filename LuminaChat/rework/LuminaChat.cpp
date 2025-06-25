// LuminaChat.cpp - Clean architecture implementation with wxWidgets UI
//
// This is the final implementation that integrates all rework components
// following the strict dependency hierarchy defined in Rework.md:
//
// Dependency Order (lower items can reference items above them):
// 1. Logger.hpp
// 2. SettingsManager.hpp  
// 3. Sanitizer.hpp
// 4. DiscordManager.hpp
// 5. ContextSizeManager.hpp
// 6. TokenCache.hpp
// 7. ModelInfo.hpp
// 8. ChatTemplateManager.hpp
// 9. ContextInfo.hpp
// 10. LlamaManager.hpp
// 11. Orchestrator.hpp

#include <wx/wx.h>
#include <wx/notebook.h>
#include <wx/textctrl.h>
#include <wx/richtext/richtextctrl.h>
#include <wx/button.h>
#include <wx/sizer.h>
#include <wx/panel.h>
#include <wx/stattext.h>
#include <wx/timer.h>
#include <wx/listbox.h>
#include <wx/choice.h>
#include <wx/checkbox.h>
#include <wx/slider.h>
#include <wx/gauge.h>
#include <wx/filedlg.h>

// Include rework components in strict dependency order
#include "Logger.hpp"
#include "SettingsManager.hpp"
#include "Sanitizer.hpp"
#include "DiscordManager.hpp"
#include "ContextSizeManager.hpp"
#include "TokenCache.hpp"
#include "ModelInfo.hpp"
#include "ChatTemplateManager.hpp"
#include "ContextInfo.hpp"
#include "LlamaManager.hpp"
#include "Orchestrator.hpp"

#include <memory>
#include <thread>
#include <atomic>

// wxWidgets application class
class LuminaChatApp : public wxApp {
public:
    bool OnInit() override;
};

// Main frame class implementing the clean rework architecture
class LuminaChatFrame : public wxFrame {
public:
    LuminaChatFrame();
    ~LuminaChatFrame();

    // UI event handlers
    void OnExit(wxCommandEvent& event);
    void OnAbout(wxCommandEvent& event);
    void OnSendMessage(wxCommandEvent& event);
    void OnLoadModel(wxCommandEvent& event);
    void OnConnectDiscord(wxCommandEvent& event);
    void OnTimer(wxTimerEvent& event);
    void OnClose(wxCloseEvent& event);
    void OnClearChat(wxCommandEvent& event);
    void OnClearLogs(wxCommandEvent& event);

    // Core system lifecycle
    void Start();
    void Stop();
    
    // Helper methods for initialization
    void RegisterCallbacks();
    void LoadDefaultModels();
    
    // Callback handlers (registered with lower-level components)
    void OnOrchestratorOutput(std::string_view output, InputSource source);
    void OnLogMessage(std::string_view log_message);

private:
    // UI Components organized by panel
    wxNotebook* notebook;
    
    // Chat Panel
    wxPanel* chat_panel;
    wxRichTextCtrl* chat_display;
    wxTextCtrl* chat_input;
    wxButton* send_button;
    wxButton* clear_button;
    
    // Settings Panel
    wxPanel* settings_panel;
    wxTextCtrl* model_path_text;
    wxButton* browse_model_button;
    wxSlider* context_size_slider;
    wxStaticText* context_size_label;
    wxSlider* gpu_layers_slider;
    wxStaticText* gpu_layers_label;
    wxButton* load_model_button;
    wxGauge* model_progress;
    
    // Discord Panel
    wxPanel* discord_panel;
    wxTextCtrl* discord_token_text;
    wxButton* connect_discord_button;
    wxListBox* channels_list;
    wxChoice* active_channel_choice;
    wxCheckBox* auto_respond_checkbox;
    
    // Logs Panel
    wxPanel* logs_panel;
    wxTextCtrl* logs_display;
    wxButton* clear_logs_button;
    wxChoice* log_level_choice;
    
    // Core rework components (in dependency order)
    std::unique_ptr<Logger> logger;
    std::unique_ptr<SettingsManager> settings_manager;
    std::unique_ptr<Sanitizer> sanitizer;
    std::unique_ptr<DiscordManager> discord_manager;
    std::unique_ptr<ContextSizeManager> context_size_manager;
    std::unique_ptr<LlamaManager> llama_manager;
    std::unique_ptr<Orchestrator> orchestrator;
    
    // System state
    std::atomic<bool> running{false};
    std::atomic<bool> model_loaded{false};
    std::atomic<bool> discord_connected{false};
    
    // Timer for scheduled operations
    wxTimer* system_timer;
    
    // Current active context and model IDs
    std::string current_context_id{"main_chat"};
    std::string current_model_id{"main_model"};
    
    // UI creation methods
    void CreateChatPanel();
    void CreateSettingsPanel();
    void CreateDiscordPanel();
    void CreateLogsPanel();
    
    // UI update methods
    void UpdateUI();
    void UpdateModelProgress(int progress);
    void AddChatMessage(const std::string& sender, const std::string& message, const wxColour& color = wxNullColour);
    void AddLogMessage(const std::string& message);
    
    DECLARE_EVENT_TABLE()
};

// Event IDs
enum {
    ID_Send = 1000,
    ID_LoadModel,
    ID_ConnectDiscord,
    ID_Timer,
    ID_ClearChat,
    ID_ClearLogs,
    ID_BrowseModel
};

// Event table mapping
wxBEGIN_EVENT_TABLE(LuminaChatFrame, wxFrame)
    EVT_MENU(wxID_EXIT, LuminaChatFrame::OnExit)
    EVT_MENU(wxID_ABOUT, LuminaChatFrame::OnAbout)
    EVT_BUTTON(ID_Send, LuminaChatFrame::OnSendMessage)
    EVT_BUTTON(ID_LoadModel, LuminaChatFrame::OnLoadModel)
    EVT_BUTTON(ID_ConnectDiscord, LuminaChatFrame::OnConnectDiscord)
    EVT_BUTTON(ID_BrowseModel, LuminaChatFrame::OnLoadModel)
    EVT_BUTTON(ID_ClearChat, LuminaChatFrame::OnClearChat)
    EVT_BUTTON(ID_ClearLogs, LuminaChatFrame::OnClearLogs)
    EVT_TIMER(ID_Timer, LuminaChatFrame::OnTimer)
    EVT_CLOSE(LuminaChatFrame::OnClose)
wxEND_EVENT_TABLE()

// Application implementation
wxIMPLEMENT_APP(LuminaChatApp);

bool LuminaChatApp::OnInit() {
    LuminaChatFrame* frame = new LuminaChatFrame();
    frame->Show(true);
    frame->Start();
    return true;
}

// Frame constructor - creates UI layout
LuminaChatFrame::LuminaChatFrame() 
    : wxFrame(nullptr, wxID_ANY, "LuminaChat - Rework Architecture", 
              wxDefaultPosition, wxSize(1200, 800)) {
    
    // Create menu bar
    wxMenuBar* menuBar = new wxMenuBar;
    wxMenu* fileMenu = new wxMenu;
    fileMenu->Append(wxID_EXIT, "E&xit\tCtrl-Q", "Quit LuminaChat");
    
    wxMenu* helpMenu = new wxMenu;
    helpMenu->Append(wxID_ABOUT, "&About\tF1", "About LuminaChat");
    
    menuBar->Append(fileMenu, "&File");
    menuBar->Append(helpMenu, "&Help");
    SetMenuBar(menuBar);
    
    // Create status bar
    CreateStatusBar(2);
    SetStatusText("Welcome to LuminaChat!", 0);
    SetStatusText("No Model", 1);
    
    // Create main notebook for tabbed interface
    notebook = new wxNotebook(this, wxID_ANY);
    
    // Create all UI panels
    CreateChatPanel();
    CreateSettingsPanel();
    CreateDiscordPanel();
    CreateLogsPanel();
    
    // Main layout
    wxBoxSizer* main_sizer = new wxBoxSizer(wxVERTICAL);
    main_sizer->Add(notebook, 1, wxEXPAND | wxALL, 5);
    SetSizer(main_sizer);
    
    // Create system timer
    system_timer = new wxTimer(this, ID_Timer);
    
    Center();
}

LuminaChatFrame::~LuminaChatFrame() {
    Stop();
}

void LuminaChatFrame::CreateChatPanel() {
    chat_panel = new wxPanel(notebook);
    notebook->AddPage(chat_panel, "Chat", true);
    
    // Chat display area
    chat_display = new wxRichTextCtrl(chat_panel, wxID_ANY, wxEmptyString,
                                      wxDefaultPosition, wxDefaultSize,
                                      wxRE_READONLY | wxRE_MULTILINE);
    chat_display->SetBackgroundColour(wxColour(248, 249, 250));
    
    // Input area
    chat_input = new wxTextCtrl(chat_panel, wxID_ANY, wxEmptyString,
                               wxDefaultPosition, wxDefaultSize,
                               wxTE_PROCESS_ENTER | wxTE_MULTILINE);
    chat_input->SetMinSize(wxSize(-1, 80));
    
    // Control buttons
    send_button = new wxButton(chat_panel, ID_Send, "Send");
    clear_button = new wxButton(chat_panel, ID_ClearChat, "Clear");
    
    // Layout
    wxBoxSizer* button_sizer = new wxBoxSizer(wxHORIZONTAL);
    button_sizer->Add(send_button, 0, wxALL, 5);
    button_sizer->Add(clear_button, 0, wxALL, 5);
    button_sizer->AddStretchSpacer();
    
    wxBoxSizer* chat_sizer = new wxBoxSizer(wxVERTICAL);
    chat_sizer->Add(chat_display, 1, wxEXPAND | wxALL, 5);
    chat_sizer->Add(chat_input, 0, wxEXPAND | wxALL, 5);
    chat_sizer->Add(button_sizer, 0, wxEXPAND);
    
    chat_panel->SetSizer(chat_sizer);
    
    // Bind enter key to send
    chat_input->Bind(wxEVT_TEXT_ENTER, &LuminaChatFrame::OnSendMessage, this);
}

void LuminaChatFrame::CreateSettingsPanel() {
    settings_panel = new wxPanel(notebook);
    notebook->AddPage(settings_panel, "Model Settings");
    
    // Model configuration group
    wxStaticBoxSizer* model_box = new wxStaticBoxSizer(wxVERTICAL, settings_panel, "Model Configuration");
    
    // Model path selection
    wxBoxSizer* path_sizer = new wxBoxSizer(wxHORIZONTAL);
    path_sizer->Add(new wxStaticText(settings_panel, wxID_ANY, "Model Path:"), 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    model_path_text = new wxTextCtrl(settings_panel, wxID_ANY);
    path_sizer->Add(model_path_text, 1, wxEXPAND | wxALL, 5);
    browse_model_button = new wxButton(settings_panel, ID_BrowseModel, "Browse...");
    path_sizer->Add(browse_model_button, 0, wxALL, 5);
    model_box->Add(path_sizer, 0, wxEXPAND);
    
    // Context size control
    wxBoxSizer* context_sizer = new wxBoxSizer(wxHORIZONTAL);
    context_sizer->Add(new wxStaticText(settings_panel, wxID_ANY, "Context Size:"), 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    context_size_slider = new wxSlider(settings_panel, wxID_ANY, 4096, 512, 32768, 
                                      wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL | wxSL_LABELS);
    context_sizer->Add(context_size_slider, 1, wxEXPAND | wxALL, 5);
    context_size_label = new wxStaticText(settings_panel, wxID_ANY, "4096");
    context_sizer->Add(context_size_label, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    model_box->Add(context_sizer, 0, wxEXPAND);
    
    // GPU layers control
    wxBoxSizer* gpu_sizer = new wxBoxSizer(wxHORIZONTAL);
    gpu_sizer->Add(new wxStaticText(settings_panel, wxID_ANY, "GPU Layers:"), 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    gpu_layers_slider = new wxSlider(settings_panel, wxID_ANY, 0, 0, 100, 
                                    wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL | wxSL_LABELS);
    gpu_sizer->Add(gpu_layers_slider, 1, wxEXPAND | wxALL, 5);
    gpu_layers_label = new wxStaticText(settings_panel, wxID_ANY, "0");
    gpu_sizer->Add(gpu_layers_label, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    model_box->Add(gpu_sizer, 0, wxEXPAND);
    
    // Load button and progress
    wxBoxSizer* load_sizer = new wxBoxSizer(wxHORIZONTAL);
    load_model_button = new wxButton(settings_panel, ID_LoadModel, "Load Model");
    load_sizer->Add(load_model_button, 0, wxALL, 5);
    model_progress = new wxGauge(settings_panel, wxID_ANY, 100);
    load_sizer->Add(model_progress, 1, wxEXPAND | wxALL, 5);
    model_box->Add(load_sizer, 0, wxEXPAND);
    
    // Main settings layout
    wxBoxSizer* settings_sizer = new wxBoxSizer(wxVERTICAL);
    settings_sizer->Add(model_box, 0, wxEXPAND | wxALL, 5);
    settings_sizer->AddStretchSpacer();
    
    settings_panel->SetSizer(settings_sizer);
}

void LuminaChatFrame::CreateDiscordPanel() {
    discord_panel = new wxPanel(notebook);
    notebook->AddPage(discord_panel, "Discord Bot");
    
    // Discord connection group
    wxStaticBoxSizer* discord_box = new wxStaticBoxSizer(wxVERTICAL, discord_panel, "Bot Connection");
    
    // Token input
    wxBoxSizer* token_sizer = new wxBoxSizer(wxHORIZONTAL);
    token_sizer->Add(new wxStaticText(discord_panel, wxID_ANY, "Bot Token:"), 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    discord_token_text = new wxTextCtrl(discord_panel, wxID_ANY, wxEmptyString, 
                                       wxDefaultPosition, wxDefaultSize, wxTE_PASSWORD);
    token_sizer->Add(discord_token_text, 1, wxEXPAND | wxALL, 5);
    connect_discord_button = new wxButton(discord_panel, ID_ConnectDiscord, "Connect");
    token_sizer->Add(connect_discord_button, 0, wxALL, 5);
    discord_box->Add(token_sizer, 0, wxEXPAND);
    
    // Channel management group
    wxStaticBoxSizer* channels_box = new wxStaticBoxSizer(wxVERTICAL, discord_panel, "Channel Management");
    
    channels_list = new wxListBox(discord_panel, wxID_ANY);
    channels_box->Add(channels_list, 1, wxEXPAND | wxALL, 5);
    
    wxBoxSizer* channel_controls_sizer = new wxBoxSizer(wxHORIZONTAL);
    channel_controls_sizer->Add(new wxStaticText(discord_panel, wxID_ANY, "Active Channel:"), 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    active_channel_choice = new wxChoice(discord_panel, wxID_ANY);
    channel_controls_sizer->Add(active_channel_choice, 1, wxEXPAND | wxALL, 5);
    auto_respond_checkbox = new wxCheckBox(discord_panel, wxID_ANY, "Auto-respond");
    channel_controls_sizer->Add(auto_respond_checkbox, 0, wxALL, 5);
    channels_box->Add(channel_controls_sizer, 0, wxEXPAND);
    
    // Layout
    wxBoxSizer* discord_sizer = new wxBoxSizer(wxVERTICAL);
    discord_sizer->Add(discord_box, 0, wxEXPAND | wxALL, 5);
    discord_sizer->Add(channels_box, 1, wxEXPAND | wxALL, 5);
    
    discord_panel->SetSizer(discord_sizer);
}

void LuminaChatFrame::CreateLogsPanel() {
    logs_panel = new wxPanel(notebook);
    notebook->AddPage(logs_panel, "System Logs");
    
    // Log display
    logs_display = new wxTextCtrl(logs_panel, wxID_ANY, wxEmptyString,
                                 wxDefaultPosition, wxDefaultSize,
                                 wxTE_READONLY | wxTE_MULTILINE | wxHSCROLL | wxVSCROLL);
    logs_display->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
    
    // Log controls
    wxBoxSizer* log_controls_sizer = new wxBoxSizer(wxHORIZONTAL);
    log_controls_sizer->Add(new wxStaticText(logs_panel, wxID_ANY, "Log Level:"), 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    log_level_choice = new wxChoice(logs_panel, wxID_ANY);
    log_level_choice->Append("DEBUG");
    log_level_choice->Append("INFO");
    log_level_choice->Append("WARNING");
    log_level_choice->Append("ERROR");
    log_level_choice->SetSelection(1); // Default to INFO
    log_controls_sizer->Add(log_level_choice, 0, wxALL, 5);
    log_controls_sizer->AddStretchSpacer();
    clear_logs_button = new wxButton(logs_panel, ID_ClearLogs, "Clear Logs");
    log_controls_sizer->Add(clear_logs_button, 0, wxALL, 5);
    
    // Layout
    wxBoxSizer* logs_sizer = new wxBoxSizer(wxVERTICAL);
    logs_sizer->Add(logs_display, 1, wxEXPAND | wxALL, 5);
    logs_sizer->Add(log_controls_sizer, 0, wxEXPAND);
    
    logs_panel->SetSizer(logs_sizer);
}

// System lifecycle management
void LuminaChatFrame::Start() {
    AddLogMessage("Initializing LuminaChat Rework Architecture with llama.cpp integration...");
    
    try {
        // Initialize components in strict dependency order
        logger = std::make_unique<Logger>();
        AddLogMessage("Logger initialized");
        
        settings_manager = std::make_unique<SettingsManager>();
        if (!settings_manager->LoadSettings("config.ini")) {
            AddLogMessage("Warning: Failed to load settings file, using defaults");
        } else {
            AddLogMessage("Settings Manager initialized and loaded");
        }
        
        sanitizer = std::make_unique<Sanitizer>();
        sanitizer->LoadBlacklist("blacklist.txt");
        AddLogMessage("Sanitizer initialized");
        
        discord_manager = std::make_unique<DiscordManager>();
        AddLogMessage("Discord Manager initialized");
        
        context_size_manager = std::make_unique<ContextSizeManager>();
        AddLogMessage("Context Size Manager initialized");
        
        // Initialize LlamaManager with settings integration
        llama_manager = std::make_unique<LlamaManager>(settings_manager.get());
        if (!llama_manager->Initialize()) {
            AddLogMessage("ERROR: Failed to initialize LlamaManager");
            throw std::runtime_error("LlamaManager initialization failed");
        }
        AddLogMessage("Llama Manager initialized with llama.cpp backend");
        
        orchestrator = std::make_unique<Orchestrator>(llama_manager.get());
        AddLogMessage("Orchestrator initialized");
        
        // Register callbacks (higher components register with lower)
        RegisterCallbacks();
        AddLogMessage("Callback dependencies registered");
        
        // Load default models from settings
        LoadDefaultModels();
        
        running = true;
        system_timer->Start(100);
        
        AddLogMessage("LuminaChat started successfully with full llama.cpp integration");
        SetStatusText("System Ready - llama.cpp Integrated", 0);
        UpdateUI();
        
    } catch (const std::exception& e) {
        wxString error_msg = wxString::Format("Failed to start LuminaChat: %s", e.what());
        AddLogMessage(error_msg.ToStdString());
        wxMessageBox(error_msg, "Startup Error", wxOK | wxICON_ERROR);
        SetStatusText("Startup Failed", 0);
    }
}

void LuminaChatFrame::Stop() {
    if (!running) return;
    
    AddLogMessage("Shutting down LuminaChat...");
    running = false;
    
    if (system_timer && system_timer->IsRunning()) {
        system_timer->Stop();
    }
    
    // Clean shutdown in reverse dependency order
    orchestrator.reset();
    llama_manager.reset();
    context_size_manager.reset();
    discord_manager.reset();
    sanitizer.reset();
    settings_manager.reset();
    logger.reset();
    
    AddLogMessage("LuminaChat stopped cleanly");
    SetStatusText("Stopped", 0);
}

// Callback implementations
void LuminaChatFrame::OnOrchestratorOutput(std::string_view output, InputSource source) {
    wxString source_name;
    wxColour color;
    
    switch (source) {
        case InputSource::UI:
            source_name = "Assistant";
            color = wxColour(0, 100, 200);
            break;
        case InputSource::DISCORD:
            source_name = "Discord Bot";
            color = wxColour(114, 137, 218);
            break;
        case InputSource::SYSTEM:
            source_name = "System";
            color = wxColour(200, 100, 0);
            break;
        case InputSource::SCHEDULED_TASK:
            source_name = "Scheduled";
            color = wxColour(100, 200, 100);
            break;
    }
    
    AddChatMessage(source_name.ToStdString(), std::string(output), color);
}

void LuminaChatFrame::OnLogMessage(std::string_view log_message) {
    AddLogMessage(std::string(log_message));
}

// UI helper methods
void LuminaChatFrame::AddChatMessage(const std::string& sender, const std::string& message, const wxColour& color) {
    wxDateTime now = wxDateTime::Now();
    
    chat_display->BeginSuppressUndo();
    chat_display->SetInsertionPointEnd();
    
    chat_display->BeginTextColour(wxColour(128, 128, 128));
    chat_display->WriteText(wxString::Format("[%s] ", now.Format("%H:%M:%S")));
    chat_display->EndTextColour();
    
    if (color.IsOk()) {
        chat_display->BeginTextColour(color);
    }
    chat_display->BeginBold();
    chat_display->WriteText(sender + ": ");
    chat_display->EndBold();
    if (color.IsOk()) {
        chat_display->EndTextColour();
    }
    
    chat_display->WriteText(message + "\n");
    
    chat_display->EndSuppressUndo();
    chat_display->ScrollIntoView(chat_display->GetLastPosition(), WXK_DOWN);
}

void LuminaChatFrame::AddLogMessage(const std::string& message) {
    wxDateTime now = wxDateTime::Now();
    wxString log_entry = wxString::Format("[%s] %s\n", now.Format("%Y-%m-%d %H:%M:%S"), message);
    
    logs_display->SetInsertionPointEnd();
    logs_display->WriteText(log_entry);
    logs_display->SetInsertionPointEnd();
}

void LuminaChatFrame::UpdateUI() {
    context_size_label->SetLabel(wxString::Format("%d", context_size_slider->GetValue()));
    gpu_layers_label->SetLabel(wxString::Format("%d", gpu_layers_slider->GetValue()));
    
    send_button->Enable(model_loaded && running);
    load_model_button->Enable(running && !model_path_text->GetValue().IsEmpty());
    connect_discord_button->Enable(running && !discord_token_text->GetValue().IsEmpty());
    
    if (model_loaded) {
        SetStatusText("Model Loaded", 1);
    } else {
        SetStatusText("No Model", 1);
    }
}

// Helper methods for initialization
void LuminaChatFrame::RegisterCallbacks() {
    // UI Output callbacks - LuminaChat (higher) registers with lower components
    if (orchestrator) {
        orchestrator->RegisterOutputCallback([this](std::string_view output, InputSource source) {
            CallAfter([this, output_str = std::string(output), source]() {
                OnOrchestratorOutput(output_str, source);
            });
        });
    }
    
    // Communication callbacks - Orchestrator (higher) registers with lower components
    if (discord_manager && orchestrator) {
        discord_manager->RegisterMessageCallback([this](const std::string& content, const std::string& channel_id, const std::string& username) {
            if (orchestrator) {
                orchestrator->OnRawDiscordMessage(content, channel_id, username);
            }
        });
    }
    
    if (context_size_manager && orchestrator) {
        context_size_manager->RegisterSummarizationCallback([this](const std::string& context_id, const std::string& content) {
            if (orchestrator) {
                orchestrator->RequestSummarization(context_id, content);
            }
        });
    }
}

void LuminaChatFrame::LoadDefaultModels() {
    if (!llama_manager || !settings_manager) {
        AddLogMessage("ERROR: Cannot load models - components not initialized");
        return;
    }
    
    AddLogMessage("Loading default models from settings...");
    
    // Load main model
    if (!llama_manager->LoadModelFromSettings("main_model", "main")) {
        AddLogMessage("WARNING: Failed to load main model from settings");
    } else {
        AddLogMessage("Main model loaded successfully");
        
        // Create default context with main model
        auto* context = llama_manager->GetOrCreateContextInfo("main_context", "main_model", "default");
        if (context) {
            AddLogMessage("Main context created successfully");
        }
    }
    
    // Load summary model (optional)
    if (!llama_manager->LoadModelFromSettings("summary_model", "summary")) {
        AddLogMessage("Summary model not configured or failed to load (optional)");
    } else {
        AddLogMessage("Summary model loaded successfully");
        
        // Create summary context
        auto* summary_context = llama_manager->GetOrCreateContextInfo("summary_context", "summary_model", "summary");
        if (summary_context) {
            AddLogMessage("Summary context created successfully");
        }
    }
}

// Event handlers
void LuminaChatFrame::OnExit(wxCommandEvent& event) {
    Close(true);
}

void LuminaChatFrame::OnAbout(wxCommandEvent& event) {
    wxMessageBox("LuminaChat - Clean Rework Architecture\n\n"
                 "Features:\n"
                 "• Clean component hierarchy with no circular dependencies\n"
                 "• Plugin architecture foundation for AI workflows\n"
                 "• Dynamic chat template management\n"
                 "• Discord bot integration with auto-response\n"
                 "• Advanced context size management\n"
                 "• Content sanitization and filtering\n"
                 "• Performance-optimized token caching\n\n"
                 "Built with wxWidgets following the rework architecture principles.",
                 "About LuminaChat",
                 wxOK | wxICON_INFORMATION);
}

void LuminaChatFrame::OnSendMessage(wxCommandEvent& event) {
    if (!running || !llama_manager || !llama_manager->IsReady()) {
        AddLogMessage("Cannot send message: system not ready");
        return;
    }
    
    wxString input = chat_input->GetValue().Trim();
    if (input.IsEmpty()) {
        return;
    }
    
    try {
        chat_input->Clear();
        AddChatMessage("You", input.ToStdString(), wxColour(50, 150, 50));
        
        // Use orchestrator to route input (it will handle sanitization and context routing)
        if (orchestrator) {
            orchestrator->InputReceived(input.ToStdString(), current_context_id, InputSource::UI);
        } else {
            // Fallback: direct context interaction for testing
            auto* context = llama_manager->GetContextInfo(current_context_id);
            if (context) {
                AddLogMessage("Processing message with context: " + current_context_id);
                std::string response = context->HandleInput(input.ToStdString(), "user");
                if (!response.empty() && response.substr(0, 6) != "Error:") {
                    AddChatMessage("Assistant", response, wxColour(50, 50, 150));
                } else {
                    AddLogMessage("Error generating response: " + response);
                }
            } else {
                AddLogMessage("Error: Context not found: " + current_context_id);
            }
        }
        
    } catch (const std::exception& e) {
        AddLogMessage(wxString::Format("Error sending message: %s", e.what()).ToStdString());
        wxMessageBox(wxString::Format("Error: %s", e.what()), "Send Error", wxOK | wxICON_ERROR);
    }
}

void LuminaChatFrame::OnLoadModel(wxCommandEvent& event) {
    // Handle browse button
    if (event.GetId() == ID_BrowseModel) {
        wxFileDialog openFileDialog(this, "Select GGUF Model File", "", "",
                                   "GGUF Model files (*.gguf)|*.gguf|All files (*.*)|*.*",
                                   wxFD_OPEN | wxFD_FILE_MUST_EXIST);
        
        if (openFileDialog.ShowModal() == wxID_OK) {
            model_path_text->SetValue(openFileDialog.GetPath());
        }
        UpdateUI();
        return;
    }
    
    // Handle load model
    wxString model_path = model_path_text->GetValue().Trim();
    if (model_path.IsEmpty()) {
        wxMessageBox("Please select a model file first", "No Model Selected", wxOK | wxICON_WARNING);
        return;
    }
    
    if (!running) {
        AddLogMessage("Cannot load model: system not running");
        return;
    }
    
    try {
        AddLogMessage(wxString::Format("Loading model: %s", model_path).ToStdString());
        
        int context_size = context_size_slider->GetValue();
        int gpu_layers = gpu_layers_slider->GetValue();
        
        load_model_button->Enable(false);
        model_progress->SetValue(0);
        
        // Simulate loading progress
        for (int i = 0; i <= 100; i += 10) {
            model_progress->SetValue(i);
            wxSafeYield();
            wxMilliSleep(50);
        }
        
        // Load model through LlamaManager
        ModelConfig config;
        config.model_path = model_path.ToStdString();
        config.context_size = context_size;
        config.gpu_layers = gpu_layers;
        
        if (llama_manager->LoadModel(current_model_id, config)) {
            model_loaded = true;
            SetStatusText("Model Loaded", 1);
            
            auto* context_info = llama_manager->GetOrCreateContextInfo(current_context_id, current_model_id);
            if (context_info) {
                AddLogMessage("Model loaded successfully");
                AddChatMessage("System", "Model loaded and ready for conversation!", wxColour(0, 150, 0));
            } else {
                AddLogMessage("ERROR: Failed to create context after model loading");
            }
        } else {
            AddLogMessage("ERROR: Failed to load model");
            wxMessageBox("Failed to load model. Check the file path and try again.", "Model Load Error", wxOK | wxICON_ERROR);
        }
        
    } catch (const std::exception& e) {
        AddLogMessage(wxString::Format("Error loading model: %s", e.what()).ToStdString());
        wxMessageBox(wxString::Format("Failed to load model: %s", e.what()), 
                     "Model Load Error", wxOK | wxICON_ERROR);
        model_loaded = false;
    }
    
    model_progress->SetValue(0);
    load_model_button->Enable(true);
    UpdateUI();
}

void LuminaChatFrame::OnConnectDiscord(wxCommandEvent& event) {
    if (discord_connected) {
        discord_manager->Disconnect();
        discord_connected = false;
        connect_discord_button->SetLabel("Connect");
        channels_list->Clear();
        active_channel_choice->Clear();
        AddLogMessage("Disconnected from Discord");
        AddChatMessage("System", "Discord bot disconnected", wxColour(200, 100, 0));
        UpdateUI();
        return;
    }
    
    wxString token = discord_token_text->GetValue().Trim();
    if (token.IsEmpty()) {
        wxMessageBox("Please enter a Discord bot token first", "No Token", wxOK | wxICON_WARNING);
        return;
    }
    
    if (!running) {
        AddLogMessage("Cannot connect to Discord: system not running");
        return;
    }
    
    try {
        AddLogMessage("Connecting to Discord...");
        connect_discord_button->Enable(false);
        
        if (discord_manager && discord_manager->Connect(token.ToStdString())) {
            discord_connected = true;
            AddLogMessage("Connected to Discord successfully");
            AddChatMessage("System", "Discord bot connected and ready!", wxColour(114, 137, 218));
            
            auto channels = discord_manager->GetChannels();
            channels_list->Clear();
            active_channel_choice->Clear();
            
            for (const auto& channel : channels) {
                wxString channel_display = wxString::Format("%s (%s)", channel.name, channel.id);
                channels_list->Append(channel_display);
                active_channel_choice->Append(channel.name);
            }
            
            if (!channels.empty()) {
                active_channel_choice->SetSelection(0);
            }
            
            connect_discord_button->SetLabel("Disconnect");
            
        } else {
            throw std::runtime_error("Discord API connection failed");
        }
        
    } catch (const std::exception& e) {
        AddLogMessage(wxString::Format("Error connecting to Discord: %s", e.what()).ToStdString());
        wxMessageBox(wxString::Format("Failed to connect to Discord: %s", e.what()),
                     "Discord Connection Error", wxOK | wxICON_ERROR);
        discord_connected = false;
    }
    
    connect_discord_button->Enable(true);
    UpdateUI();
}

void LuminaChatFrame::OnClearChat(wxCommandEvent& event) {
    chat_display->Clear();
    AddLogMessage("Chat display cleared");
}

void LuminaChatFrame::OnClearLogs(wxCommandEvent& event) {
    logs_display->Clear();
}

void LuminaChatFrame::OnTimer(wxTimerEvent& event) {
    if (running && orchestrator) {
        orchestrator->ProcessScheduledTasks();
        UpdateUI();
    }
}

void LuminaChatFrame::OnClose(wxCloseEvent& event) {
    Stop();
    event.Skip();
}
