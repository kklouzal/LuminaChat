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
// 5. ContextPruningPlugin.hpp
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
#include <wx/event.h>
#include <wx/scrolwin.h>

// Include rework components in strict dependency order
#include "Logger.hpp"
#include "SettingsManager.hpp"
#include "Sanitizer.hpp"
#include "DiscordManager.hpp"
#include "Plugins/ContextPruningPlugin.hpp"
#include "TokenCache.hpp"
#include "ModelInfo.hpp"
#include "ChatTemplateManager.hpp"
#include "Context/ContextInfo.hpp"
#include "LlamaManager.hpp"
#include "Orchestrator.hpp"
#include "Plugins/SummarizationPlugin.hpp"
#include "Plugins/EmoTagPlugin.hpp"
#include "Plugins/UI/SummarizationPlugin_UI.hpp"
#include "Plugins/UI/EmoTagPlugin_UI.hpp"
#include "Plugins/UI/ContextPruningPlugin_UI.hpp"
#include "UI/Settings_UI.hpp"
#include "UI/OuterVoice_UI.hpp"
#include "UI/InnerVoice_UI.hpp"

#include <memory>
#include <thread>
#include <atomic>

// Configuration constants
namespace LuminaChatConstants {
    // Timer and update intervals
    static constexpr int DEBUG_UPDATE_INTERVAL = 10;  // Timer ticks between debug updates
    static constexpr int TIMER_INTERVAL_MS = 100;     // Main timer interval in milliseconds
    
    // UI layout constants
    static constexpr int DEFAULT_WINDOW_WIDTH = 800;
    static constexpr int DEFAULT_WINDOW_HEIGHT = 600;
    static constexpr int CONTROL_SPACING = 5;
    static constexpr int PANEL_BORDER = 10;
    
    // Model configuration defaults
    static constexpr int DEFAULT_CONTEXT_SIZE = 4096;
    static constexpr int MIN_CONTEXT_SIZE = 512;
    static constexpr int MAX_CONTEXT_SIZE = 32768;
    static constexpr int DEFAULT_GPU_LAYERS = 999;
    static constexpr int MIN_GPU_LAYERS = 0;
    static constexpr int MAX_GPU_LAYERS = 999;
    
    // Discord configuration
    static constexpr int DEFAULT_BACKFILL_LIMIT = 100;
    
    // Chat display formatting
    static constexpr int TIMESTAMP_FORMAT_LENGTH = 8;  // "[HH:MM:SS]"
    
    // Plugin monitoring intervals
    static constexpr int EMOTION_ANALYSIS_DELAY_MS = 1000;  // Delay after generation for emotion analysis
    static constexpr int CONTEXT_MONITORING_DELAY_MS = 1000; // Delay after generation for context monitoring
}

// UI Color constants
namespace LuminaChatColors {
    static const wxColour SUCCESS_GREEN{0, 150, 0};
    static const wxColour ERROR_RED{150, 50, 50};
    static const wxColour WARNING_ORANGE{150, 100, 50};
    static const wxColour INFO_BLUE{50, 50, 150};
    static const wxColour DISCORD_BLUE{114, 137, 218};
    static const wxColour NEUTRAL_GRAY{100, 100, 100};
    static const wxColour BACKGROUND_LIGHT{248, 249, 250};
    static const wxColour TIMESTAMP_GRAY{128, 128, 128};
    static const wxColour TEMPLATE_BACKGROUND{250, 250, 250};
}

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
    void OnStopGeneration(wxCommandEvent& event);
    void OnConnectDiscord(wxCommandEvent& event);
    void OnTimer(wxTimerEvent& event);
    void OnClose(wxCloseEvent& event);
    void OnClearChat(wxCommandEvent& event);
    void OnClearLogs(wxCommandEvent& event);
    void OnClearTemplate(wxCommandEvent& event);
    void OnLogLevelChanged(wxCommandEvent& event);

    // Settings UI integration
    void OnLoadModelFromSettingsUI(const std::string& model_path, int context_size, int gpu_layers);

    // Core system lifecycle
    void Start();
    void Stop();
    
    // Helper methods for initialization
    void RegisterCallbacks();
    void LoadDefaultModels();
    void InitializePlugins();  // New method for plugin initialization after model loading
    
    // Template configuration helpers
    std::string GetEnvironmentDescriptionFromUI() const;
    std::string GetIdentityDirectiveFromUI() const;
    std::string GetSystemPromptFromUI() const;
    void ApplyTemplateSettingsToContext(ContextInfo* context, const std::string& context_id);

    // Message generation helper methods
    void CaptureTemplateForInspection(ContextInfo* context);
    GenerationCallbacks CreateGenerationCallbacks();
    void ExecuteMessageGeneration(ContextInfo* context, const std::string& input, const GenerationCallbacks& callbacks);
    void HandleMessageGenerationError(const std::string& error_message);

    // Common UI helper methods
    void ShowErrorMessage(const std::string& message, const std::string& context = "");
    void ShowSuccessMessage(const std::string& message);
    void ShowWarningMessage(const std::string& message);
    void LogAndDisplayError(const std::string& log_msg, const std::string& user_msg = "");
    void LogAndDisplaySuccess(const std::string& log_msg, const std::string& user_msg = "");

    // UI Settings persistence
    void LoadUISettings();
    void SaveUISettings();

    // Callback handlers (registered with lower-level components)
    void OnLogMessage(std::string_view log_message);

private:
    // UI Components organized by panel
    wxNotebook* notebook;
    
    // Chat Panel
    wxPanel* chat_panel;
    wxRichTextCtrl* chat_display;
    wxTextCtrl* chat_input;
    wxButton* send_button;
    wxButton* stop_button;
    wxButton* clear_button;
    
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
    
    // Template Panel
    wxPanel* template_panel;
    wxTextCtrl* template_display;
    wxButton* clear_template_button;
    
    // Plugin UI Managers
    std::unique_ptr<SummarizationPluginUI> summarization_ui;
    std::unique_ptr<EmoTagPluginUI> emotag_ui;
    std::unique_ptr<ContextPruningPluginUI> context_pruning_ui;
    
    // Voice Settings UI Managers
    std::unique_ptr<OuterVoiceUI> outer_voice_ui;
    std::unique_ptr<InnerVoiceUI> inner_voice_ui;
    
    // General Settings UI Manager
    std::unique_ptr<SettingsUI> settings_ui;
    
    // Core rework components (in dependency order)
    std::unique_ptr<SettingsManager> settings_manager;
    std::unique_ptr<Sanitizer> sanitizer;
    std::unique_ptr<DiscordManager> discord_manager;
    std::unique_ptr<LlamaManager> llama_manager;
    std::unique_ptr<Orchestrator> orchestrator;
    std::unique_ptr<LuminaChat::SummarizationPlugin> summarization_plugin;
    std::unique_ptr<LuminaChat::EmoTagPlugin> emotag_plugin;
    std::unique_ptr<LuminaChat::ContextPruningPlugin> context_pruning_plugin;
    
    // System state
    std::atomic<bool> running{false};
    std::atomic<bool> model_loaded{false};
    std::atomic<bool> model_loading{false};
    std::atomic<bool> discord_connected{false};
    
    // Timer for scheduled operations
    wxTimer* system_timer;
    
    // Current active context and model IDs
    std::string current_context_id{"outer_chat"};
    std::string current_model_id{"outer_model"};
    
    // Streaming state management
    std::atomic<bool> is_streaming{false};
    std::string current_assistant_message;
    long assistant_message_start_pos = -1;
    std::string last_finalized_template;  // Store the last template for inspection
    
    // UI creation methods
    void CreateChatPanel();
    void CreateDiscordPanel();
    void CreateLogsPanel();
    void CreateTemplatePanel();
    
    // UI update methods
    void UpdateUI();
    void UpdateModelProgress(int progress);
    void UpdateContextStatus();  // Track and display context usage
    void AddChatMessage(const std::string& sender, const std::string& message, const wxColour& color = wxNullColour);
    void StartStreamingMessage(const std::string& sender, const wxColour& color = wxNullColour);
    void AppendToStreamingMessage(const std::string& text);
    void ReplaceStreamingMessage(const std::string& text);  // Replace entire streaming content
    void EndStreamingMessage();
    void SetGenerationUIState(bool generating);  // Enable/disable UI during generation
    void AddLogMessage(const std::string& message);
    void UpdateTemplateDisplay(const std::string& template_content);  // Update template inspection tab
    void UpdateSummaryPluginStatus(const std::string& status, const wxColour& color = wxNullColour);  // Update summary plugin status
    void UpdateEmoTagPluginStatus(const std::string& status, const wxColour& color = wxNullColour);  // Update emotag plugin status
    void UpdateSummaryPluginDebugInfo();  // Update summary plugin debug textboxes
    void UpdateEmoTagPluginDebugInfo();   // Update emotag plugin debug textboxes
    
    // Individual plugin initialization methods
    void InitializeSummarizationPlugin();
    void InitializeEmoTagPlugin();
    void InitializeContextPruningPlugin();
    
    DECLARE_EVENT_TABLE()
};

// Event IDs
enum {
    ID_Send = 1000,
    ID_Stop,
    ID_ConnectDiscord,
    ID_Timer,
    ID_ClearChat,
    ID_ClearLogs,
    ID_ClearTemplate
};

// Event table mapping
wxBEGIN_EVENT_TABLE(LuminaChatFrame, wxFrame)
    EVT_MENU(wxID_EXIT, LuminaChatFrame::OnExit)
    EVT_MENU(wxID_ABOUT, LuminaChatFrame::OnAbout)
    EVT_BUTTON(ID_Send, LuminaChatFrame::OnSendMessage)
    EVT_BUTTON(ID_Stop, LuminaChatFrame::OnStopGeneration)
    EVT_BUTTON(ID_ConnectDiscord, LuminaChatFrame::OnConnectDiscord)
    EVT_BUTTON(ID_ClearChat, LuminaChatFrame::OnClearChat)
    EVT_BUTTON(ID_ClearLogs, LuminaChatFrame::OnClearLogs)
    EVT_BUTTON(ID_ClearTemplate, LuminaChatFrame::OnClearTemplate)
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
    
    // Create status bar with 3 sections: [System Status] [Model Status] [Context Status]
    CreateStatusBar(3);
    SetStatusText("Welcome to LuminaChat!", 0);
    SetStatusText("No Model", 1);
    SetStatusText("Context: --/--", 2);
    
    // Create main notebook for tabbed interface
    notebook = new wxNotebook(this, wxID_ANY);
    
    // Create all UI panels
    CreateChatPanel();
    CreateDiscordPanel();
    CreateLogsPanel();
    CreateTemplatePanel();
    
    // Create Voice Settings UI managers
    outer_voice_ui = std::make_unique<OuterVoiceUI>(this);
    inner_voice_ui = std::make_unique<InnerVoiceUI>(this);
    
    // Create General Settings UI manager
    settings_ui = std::make_unique<SettingsUI>(this);
    
    // Create plugin UI managers
    summarization_ui = std::make_unique<SummarizationPluginUI>(this);
    emotag_ui = std::make_unique<EmoTagPluginUI>(this);
    context_pruning_ui = std::make_unique<ContextPruningPluginUI>(this);
    
    // Add voice settings panels to notebook
    notebook->AddPage(outer_voice_ui->CreatePanel(), "Outer Voice Settings");
    notebook->AddPage(inner_voice_ui->CreatePanel(), "Inner Voice Settings");
    
    // Add general settings panel to notebook
    notebook->AddPage(settings_ui->CreatePanel(), "General Settings");
    
    // Add plugin panels to notebook
    notebook->AddPage(summarization_ui->CreatePanel(), "Summary Settings");
    notebook->AddPage(emotag_ui->CreatePanel(), "EmoTag Settings");
    notebook->AddPage(context_pruning_ui->CreatePanel(), "Context Pruning");
    
    // Main layout
    wxBoxSizer* main_sizer = new wxBoxSizer(wxVERTICAL);
    main_sizer->Add(notebook, 1, wxEXPAND | wxALL, LuminaChatConstants::CONTROL_SPACING);
    SetSizer(main_sizer);
    
    // Create system timer
    system_timer = new wxTimer(this, ID_Timer);
    
    Center();
}

LuminaChatFrame::~LuminaChatFrame() {
    Stop();
}

// === UI Creation Methods ===

void LuminaChatFrame::CreateChatPanel() {
    chat_panel = new wxPanel(notebook);
    notebook->AddPage(chat_panel, "Chat", true);
    
    // Chat display area
    chat_display = new wxRichTextCtrl(chat_panel, wxID_ANY, wxEmptyString,
                                      wxDefaultPosition, wxDefaultSize,
                                      wxRE_READONLY | wxRE_MULTILINE);
    chat_display->SetBackgroundColour(LuminaChatColors::BACKGROUND_LIGHT);
    
    // Input area
    chat_input = new wxTextCtrl(chat_panel, wxID_ANY, wxEmptyString,
                               wxDefaultPosition, wxDefaultSize,
                               wxTE_PROCESS_ENTER | wxTE_MULTILINE);
    chat_input->SetMinSize(wxSize(-1, 80));
    
    // Control buttons
    send_button = new wxButton(chat_panel, ID_Send, "Send");
    stop_button = new wxButton(chat_panel, ID_Stop, "Stop");
    stop_button->Enable(false);  // Initially disabled
    clear_button = new wxButton(chat_panel, ID_ClearChat, "Clear");
    
    // Layout
    wxBoxSizer* button_sizer = new wxBoxSizer(wxHORIZONTAL);
    button_sizer->Add(send_button, 0, wxALL, LuminaChatConstants::CONTROL_SPACING);
    button_sizer->Add(stop_button, 0, wxALL, LuminaChatConstants::CONTROL_SPACING);
    button_sizer->Add(clear_button, 0, wxALL, LuminaChatConstants::CONTROL_SPACING);
    button_sizer->AddStretchSpacer();
    
    wxBoxSizer* chat_sizer = new wxBoxSizer(wxVERTICAL);
    chat_sizer->Add(chat_display, 1, wxEXPAND | wxALL, 5);
    chat_sizer->Add(chat_input, 0, wxEXPAND | wxALL, 5);
    chat_sizer->Add(button_sizer, 0, wxEXPAND);
    
    chat_panel->SetSizer(chat_sizer);
    
    // Bind enter key to send
    chat_input->Bind(wxEVT_TEXT_ENTER, &LuminaChatFrame::OnSendMessage, this);
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
    
    // Auto-save Discord token on kill focus
    discord_token_text->Bind(wxEVT_KILL_FOCUS, [this](wxFocusEvent& event) {
        if (settings_manager) {
            settings_manager->SetString("Discord", "bot_token", discord_token_text->GetValue().ToStdString());
            settings_manager->SaveSettings();
        }
        event.Skip();
    });
    
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
    
    // Auto-save auto-respond setting
    auto_respond_checkbox->Bind(wxEVT_CHECKBOX, [this](wxCommandEvent& event) {
        if (settings_manager) {
            settings_manager->SetBool("Discord", "auto_respond", auto_respond_checkbox->GetValue());
            settings_manager->SaveSettings();
        }
    });
    
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
    
    // Bind log level change event
    log_level_choice->Bind(wxEVT_CHOICE, [this](wxCommandEvent& event) {
        OnLogLevelChanged(event);
    });
    
    log_controls_sizer->AddStretchSpacer();
    clear_logs_button = new wxButton(logs_panel, ID_ClearLogs, "Clear Logs");
    log_controls_sizer->Add(clear_logs_button, 0, wxALL, 5);
    
    // Layout
    wxBoxSizer* logs_sizer = new wxBoxSizer(wxVERTICAL);
    logs_sizer->Add(logs_display, 1, wxEXPAND | wxALL, 5);
    logs_sizer->Add(log_controls_sizer, 0, wxEXPAND);
    
    logs_panel->SetSizer(logs_sizer);
}

void LuminaChatFrame::CreateTemplatePanel() {
    template_panel = new wxPanel(notebook);
    notebook->AddPage(template_panel, "Template");
    
    // Template display (read-only monospace text control)
    template_display = new wxTextCtrl(template_panel, wxID_ANY, wxEmptyString,
                                     wxDefaultPosition, wxDefaultSize,
                                     wxTE_READONLY | wxTE_MULTILINE | wxHSCROLL | wxVSCROLL);
    template_display->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
    template_display->SetBackgroundColour(LuminaChatColors::TEMPLATE_BACKGROUND);
    
    // Initial text explaining the purpose
    template_display->SetValue(
        "Template Debugging Inspection\n"
        "===================\n\n"
    );
    
    // Clear button
    wxBoxSizer* template_controls_sizer = new wxBoxSizer(wxHORIZONTAL);
    template_controls_sizer->AddStretchSpacer();
    clear_template_button = new wxButton(template_panel, ID_ClearTemplate, "Clear Template");
    template_controls_sizer->Add(clear_template_button, 0, wxALL, 5);
    
    // Layout
    wxBoxSizer* template_sizer = new wxBoxSizer(wxVERTICAL);
    template_sizer->Add(template_display, 1, wxEXPAND | wxALL, 5);
    template_sizer->Add(template_controls_sizer, 0, wxEXPAND);
    
    template_panel->SetSizer(template_sizer);
}

// === System Lifecycle Management ===

/**
 * Initialize the LuminaChat system following the clean rework architecture.
 * This method sets up all components in their strict dependency order:
 * 1. Logger integration
 * 2. Settings Manager
 * 3. Sanitizer
 * 4. Discord Manager
 * 5. Llama Manager
 * 6. Orchestrator
 * 
 * Plugins are initialized separately after model loading.
 */
void LuminaChatFrame::Start() {
    AddLogMessage("Initializing LuminaChat Rework Architecture with llama.cpp integration...");
    
    try {
        // Initialize components in strict dependency order
        // Register GLOBAL Logger output callback to display logs in UI
        GetLogger().RegisterOutputCallback([this](std::string_view log_message) {
            CallAfter([this, log_str = std::string(log_message)]() {
                OnLogMessage(log_str);
            });
        });
        
        AddLogMessage("Global Logger connected to UI");
        
        settings_manager = std::make_unique<SettingsManager>();
        if (!settings_manager->Initialize()) {
            AddLogMessage("Warning: Failed to initialize settings manager, using defaults");
        } else {
            AddLogMessage("Settings Manager initialized successfully");
        }
        
        sanitizer = std::make_unique<Sanitizer>();
        sanitizer->LoadBlacklist("blacklist.txt");
        AddLogMessage("Sanitizer initialized");
        
        discord_manager = std::make_unique<DiscordManager>();
        AddLogMessage("Discord Manager initialized");
        
        // Initialize LlamaManager with settings integration
        llama_manager = std::make_unique<LlamaManager>(settings_manager.get());
        if (!llama_manager->Initialize()) {
            AddLogMessage("ERROR: Failed to initialize LlamaManager");
            throw std::runtime_error("LlamaManager initialization failed");
        }
        AddLogMessage("Llama Manager initialized with llama.cpp backend");

        orchestrator = std::make_unique<Orchestrator>(llama_manager.get());
        if (!orchestrator->Initialize()) {
            AddLogMessage("ERROR: Failed to initialize Orchestrator");
            throw std::runtime_error("Orchestrator initialization failed");
        }
        AddLogMessage("Orchestrator initialized");
        
        // Note: SummarizationPlugin initialization is now deferred until model loading
        AddLogMessage("Plugin initialization will be handled when user loads a model");
        
        // Initialize Outer Voice UI with dependencies
        if (outer_voice_ui) {
            outer_voice_ui->SetSettingsManager(settings_manager.get());
            outer_voice_ui->SetLlamaManager(llama_manager.get());
            outer_voice_ui->SetCallbacks(
                [this](const std::string& msg) { AddLogMessage(msg); },
                [this](ContextInfo* ctx, const std::string& id) { ApplyTemplateSettingsToContext(ctx, id); }
            );
            AddLogMessage("Outer Voice UI dependencies configured");
        }
        
        // Initialize General Settings UI with dependencies
        if (settings_ui) {
            settings_ui->SetSettingsManager(settings_manager.get());
            settings_ui->SetCallbacks(
                [this](const std::string& msg) { AddLogMessage(msg); },
                [this](const std::string& path, int ctx_size, int gpu_layers) { 
                    OnLoadModelFromSettingsUI(path, ctx_size, gpu_layers); 
                },
                [this](ContextInfo* ctx, const std::string& id) { ApplyTemplateSettingsToContext(ctx, id); }
            );
            AddLogMessage("General Settings UI dependencies configured");
        }
        
        // Initialize Inner Voice UI with dependencies
        if (inner_voice_ui) {
            inner_voice_ui->SetSettingsManager(settings_manager.get());
            inner_voice_ui->SetLlamaManager(llama_manager.get());
            inner_voice_ui->SetCallbacks(
                [this](const std::string& msg) { AddLogMessage(msg); },
                [this](ContextInfo* ctx, const std::string& id) { ApplyTemplateSettingsToContext(ctx, id); }
            );
            AddLogMessage("Inner Voice UI dependencies configured");
        }
        
        // Register callbacks (higher components register with lower)
        RegisterCallbacks();
        AddLogMessage("Callback dependencies registered");
        
        // Load UI settings from configuration
        LoadUISettings();
        
        // Note: Model loading is now unified in the General Settings tab
        AddLogMessage("Ready for model loading - use General Settings tab to load model");
        
        running = true;
        system_timer->Start(LuminaChatConstants::TIMER_INTERVAL_MS);
        
        AddLogMessage("LuminaChat started successfully - model loading available in General Settings tab");
        SetStatusText("System Ready - Use General Settings tab to load model", 0);
        UpdateUI();
        
        // Set initial plugin status
        UpdateSummaryPluginStatus("Waiting for model loading", LuminaChatColors::NEUTRAL_GRAY);
        UpdateEmoTagPluginStatus("Waiting for model loading", LuminaChatColors::NEUTRAL_GRAY);
        
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
    
    // Save UI settings before shutdown
    SaveUISettings();
    
    running = false;
    
    if (system_timer && system_timer->IsRunning()) {
        system_timer->Stop();
    }
    
    // Clean shutdown in reverse dependency order
    if (emotag_plugin) {
        emotag_plugin->Shutdown();
        emotag_plugin.reset();
        AddLogMessage("EmoTagPlugin stopped");
    }
    if (summarization_plugin) {
        summarization_plugin->Shutdown();
        summarization_plugin.reset();
        AddLogMessage("SummarizationPlugin stopped");
    }
    
    // Clear progress callback to avoid potential use-after-free
    if (llama_manager) {
        llama_manager->ClearProgressCallback();
    }
    
    // Wait briefly for any ongoing model loading to complete
    // Note: We use model_loading atomic flag to check if loading is in progress
    int wait_count = 0;
    while (model_loading.load() && wait_count < 50) { // Max 5 seconds wait
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        wait_count++;
    }
    
    if (model_loading.load()) {
        AddLogMessage("Warning: Model loading still in progress during shutdown");
    }
    
    orchestrator.reset();
    llama_manager.reset();
    discord_manager.reset();
    sanitizer.reset();
    settings_manager.reset();
    
    AddLogMessage("LuminaChat stopped cleanly");
    SetStatusText("Stopped", 0);
}

// Callback implementations
void LuminaChatFrame::OnLogMessage(std::string_view log_message) {
    AddLogMessage(std::string(log_message));
}

void LuminaChatFrame::OnLogLevelChanged(wxCommandEvent& event) {
    int selection = log_level_choice->GetSelection();
    Logger::LogLevel new_level;
    
    switch (selection) {
        case 0: // DEBUG
            new_level = Logger::LogLevel::DBG;
            AddLogMessage("Log level changed to DEBUG");
            break;
        case 1: // INFO
            new_level = Logger::LogLevel::INF;
            AddLogMessage("Log level changed to INFO");
            break;
        case 2: // WARNING
            new_level = Logger::LogLevel::WRN;
            AddLogMessage("Log level changed to WARNING");
            break;
        case 3: // ERROR
            new_level = Logger::LogLevel::ERR;
            AddLogMessage("Log level changed to ERROR");
            break;
        default:
            new_level = Logger::LogLevel::INF;
            AddLogMessage("Unknown log level selected, defaulting to INFO");
            break;
    }
    
    // Set level on the GLOBAL Logger instance that the macros use
    GetLogger().SetLogLevel(new_level);
}

// === UI Helper Methods ===

void LuminaChatFrame::AddChatMessage(const std::string& sender, const std::string& message, const wxColour& color) {
    wxDateTime now = wxDateTime::Now();
    
    chat_display->BeginSuppressUndo();
    chat_display->SetInsertionPointEnd();
    
    chat_display->BeginTextColour(LuminaChatColors::TIMESTAMP_GRAY);
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
    // Update Voice Settings UIs
    if (outer_voice_ui) {
        outer_voice_ui->UpdateUI();
    }
    if (inner_voice_ui) {
        inner_voice_ui->UpdateUI();
    }
    
    // Update General Settings UI
    if (settings_ui) {
        settings_ui->UpdateUI();
    }
    
    send_button->Enable(model_loaded && running);
    connect_discord_button->Enable(running && !discord_token_text->GetValue().IsEmpty());
    
    if (model_loaded) {
        SetStatusText("Model Loaded", 1);
    } else if (model_loading) {
        SetStatusText("Loading Model...", 1);
    } else {
        SetStatusText("No Model", 1);
    }
    
    // Update context status whenever UI is updated
    UpdateContextStatus();
}

void LuminaChatFrame::UpdateContextStatus() {
    if (!llama_manager || !model_loaded || !running) {
        SetStatusText("Context: --/--", 2);
        return;
    }
    
    try {
        // Get the current context info
        auto* context = llama_manager->GetContextInfo(current_context_id);
        if (!context) {
            SetStatusText("Context: No Context", 2);
            return;
        }
        
        // Get context usage information directly
        int used_tokens = static_cast<int>(context->GetActualContextTokens());
        int max_tokens = static_cast<int>(context->GetMaxContextTokens());
        
        // Format the status text
        wxString context_status = wxString::Format("Context: %d/%d", used_tokens, max_tokens);
        
        // Add visual indicators based on usage percentage
        if (max_tokens > 0) {
            double usage_percent = (double)used_tokens / max_tokens * 100.0;
            if (usage_percent >= 90.0) {
                context_status += " (FULL)";
            } else if (usage_percent >= 75.0) {
                context_status += " (HIGH)";
            } else if (usage_percent >= 50.0) {
                context_status += " (MED)";
            }
        }
        
        SetStatusText(context_status, 2);
        
    } catch (const std::exception& e) {
        AddLogMessage(wxString::Format("Error updating context status: %s", e.what()).ToStdString());
        SetStatusText("Context: Error", 2);
    }
}

void LuminaChatFrame::SetGenerationUIState(bool generating) {
    send_button->Enable(!generating && model_loaded && running);
    stop_button->Enable(generating);
    chat_input->Enable(!generating);
    
    if (generating) {
        send_button->SetLabel("Generating...");
        SetStatusText("Generating response...", 0);
    } else {
        send_button->SetLabel("Send");
        if (model_loaded) {
            SetStatusText("System Ready - Model Loaded", 0);
        } else {
            SetStatusText("System Ready - Click 'Load Model' to begin", 0);
        }
        // Refocus the chat input when generation is complete
        chat_input->SetFocus();
    }
}

// Helper methods for initialization
void LuminaChatFrame::RegisterCallbacks() {
    // Communication callbacks - Orchestrator (higher) registers with lower components
    if (discord_manager && orchestrator) {
        discord_manager->RegisterMessageCallback([this](const std::string& content, const std::string& channel_id, const std::string& username) {
            if (orchestrator) {
                orchestrator->OnRawDiscordMessage(content, channel_id, username);
            }
        });
    }
    
    // Register progress callback for model loading UI updates
    if (llama_manager && settings_ui) {
        llama_manager->RegisterProgressCallback([this](const std::string& model_id, float progress) {
            // Convert progress to percentage and update UI progress bar
            int percentage = static_cast<int>(progress * 100);
            
            // Use CallAfter to ensure UI updates happen on the main thread
            CallAfter([this, percentage, model_id]() {
                // Safety check: ensure UI components are still valid and app is running
                if (!running.load() || !settings_ui) {
                    return; // Application is shutting down or UI is destroyed
                }
                
                try {
                    settings_ui->UpdateModelProgress(percentage);
                    
                    // Update status bar with progress information
                    if (percentage < 100) {
                        SetStatusText(wxString::Format("Loading model: %d%%", percentage), 1);
                    } else {
                        SetStatusText("Model Loaded", 1);
                    }
                    
                    // Add log message for progress tracking
                    if (percentage % 10 == 0 || percentage >= 95) {
                        AddLogMessage(wxString::Format("Model loading progress: %d%% (%s)", percentage, model_id).ToStdString());
                    }
                } catch (...) {
                    // Ignore any UI update errors during shutdown
                }
            });
        });
        AddLogMessage("Progress callback registered with LlamaManager");
    }
    
    // Note: UI now uses direct async streaming instead of Orchestrator callbacks
    // This eliminates the synchronous callback tech debt
}

void LuminaChatFrame::LoadDefaultModels() {
    if (!llama_manager || !settings_manager) {
        AddLogMessage("ERROR: Cannot load models - components not initialized");
        return;
    }
    
    AddLogMessage("Loading default models from settings...");
    
    // Load outer model
    if (!llama_manager->LoadModelFromSettings("outer_model", "outer")) {
        AddLogMessage("WARNING: Failed to load outer model from settings");
    } else {
        AddLogMessage("Outer model loaded successfully");
        
        // Get outer model context size from settings
        int32_t outer_context_size = settings_manager->GetInt("Models", "outer_context_size", 4096);
        
        // Create default context with outer model
        auto* context = llama_manager->GetOrCreateContextInfo("outer_context", "outer_model", outer_context_size);
        if (context) {
            AddLogMessage("Outer context created successfully with context size: " + std::to_string(outer_context_size));
            // Apply template settings from UI (identity directive and system prompt)
            ApplyTemplateSettingsToContext(context, "outer_context");
        }
    }
    
    // Note: Summary model loading is now handled by SummarizationPlugin during its initialization
    AddLogMessage("Summary model initialization will be handled by SummarizationPlugin");
}

/**
 * Initialize all plugins after successful model loading.
 * This method coordinates the initialization of:
 * - SummarizationPlugin (for conversation summarization)
 * - EmoTagPlugin (for emotional analysis)
 * - ContextPruningPlugin (for context size management)
 * 
 * Each plugin is registered with the Orchestrator for coordination.
 */
void LuminaChatFrame::InitializePlugins() {
    if (!llama_manager || !orchestrator || !model_loaded) {
        AddLogMessage("ERROR: Cannot initialize plugins - prerequisites not met");
        return;
    }
    
    AddLogMessage("Initializing plugins after successful model loading...");
    
    try {
        InitializeSummarizationPlugin();
        InitializeEmoTagPlugin();
        InitializeContextPruningPlugin();
        
        AddLogMessage("All plugins initialized successfully");
        
    } catch (const std::exception& e) {
        AddLogMessage(wxString::Format("Error initializing plugins: %s", e.what()).ToStdString());
        UpdateSummaryPluginStatus("Initialization failed", LuminaChatColors::ERROR_RED);
    }
}

// === Individual Plugin Initialization Methods ===

void LuminaChatFrame::InitializeSummarizationPlugin() {
    summarization_plugin = std::make_unique<LuminaChat::SummarizationPlugin>(orchestrator.get());
    
    // Register callback for plugin status updates
    if (summarization_plugin) {
        summarization_plugin->SetStatusCallback([this](const std::string& status, bool is_error) {
            CallAfter([this, status, is_error]() {
                wxColour color = is_error ? LuminaChatColors::ERROR_RED : LuminaChatColors::SUCCESS_GREEN;
                UpdateSummaryPluginStatus(status, color);
            });
        });
    }
    
    if (!summarization_plugin->Initialize()) {
        AddLogMessage("Failed to initialize SummarizationPlugin");
        UpdateSummaryPluginStatus("Initialization failed", LuminaChatColors::ERROR_RED);
    } else {
        AddLogMessage("SummarizationPlugin initialized successfully");
        UpdateSummaryPluginStatus("Initialized and ready", LuminaChatColors::SUCCESS_GREEN);
        
        // Register the plugin with the Orchestrator for coordination
        orchestrator->RegisterSummarizationPlugin(summarization_plugin.get());
    }
}

void LuminaChatFrame::InitializeEmoTagPlugin() {
    emotag_plugin = std::make_unique<LuminaChat::EmoTagPlugin>(orchestrator.get());
    
    // Register callback for plugin status updates
    if (emotag_plugin) {
        emotag_plugin->SetStatusCallback([this](const std::string& status, bool is_error) {
            CallAfter([this, status, is_error]() {
                wxColour color = is_error ? LuminaChatColors::ERROR_RED : LuminaChatColors::SUCCESS_GREEN;
                UpdateEmoTagPluginStatus(status, color);
            });
        });
    }

    // Initialize the plugin
    if (!emotag_plugin->Initialize()) {
        AddLogMessage("Failed to initialize EmoTagPlugin");
        UpdateEmoTagPluginStatus("Initialization failed", LuminaChatColors::ERROR_RED);
        return;
    }
    
    // Register the plugin with the Orchestrator for coordination
    orchestrator->RegisterEmoTagPlugin(emotag_plugin.get());
        
    // Configure plugin with default settings (UI manager will load actual settings later)
    emotag_plugin->SetAnalysisWindow(3); // Default window size
    AddLogMessage("Configured EmoTagPlugin with default analysis window size: 3");
    
    AddLogMessage("EmoTagPlugin initialized successfully");
    UpdateEmoTagPluginStatus("Initialized and ready", LuminaChatColors::SUCCESS_GREEN);
}

void LuminaChatFrame::InitializeContextPruningPlugin() {
    context_pruning_plugin = std::make_unique<LuminaChat::ContextPruningPlugin>(orchestrator.get());
    
    if (!context_pruning_plugin->Initialize()) {
        AddLogMessage("Failed to initialize ContextPruningPlugin");
    } else {
        AddLogMessage("ContextPruningPlugin initialized successfully");
        
        // Configure more aggressive thresholds for better context management
        context_pruning_plugin->SetPruningThreshold(0.65f);  // Trigger at 65% instead of 75%
        context_pruning_plugin->SetTargetUsage(0.35f);       // Reduce to 35% instead of 40%
        context_pruning_plugin->SetEmergencyThreshold(0.85f); // Emergency at 85% instead of 90%
        
        // Register with Orchestrator for context monitoring
        orchestrator->RegisterContextPruningPlugin(context_pruning_plugin.get());
        AddLogMessage("ContextPruningPlugin registered with Orchestrator for monitoring");
    }
}

// === Event Handlers ===

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

/**
 * Handle user message sending and AI response generation.
 * This method orchestrates the complete message flow:
 * 1. Validates system state and user input
 * 2. Prepares the generation context with template settings
 * 3. Captures template for debugging inspection
 * 4. Initiates streaming response generation
 * 5. Handles completion and error scenarios
 */
void LuminaChatFrame::OnSendMessage(wxCommandEvent& event) {
    // Validate system state
    if (!running || !llama_manager || !llama_manager->IsReady()) {
        LogAndDisplayError("Cannot send message: system not ready", "System not ready for message processing");
        return;
    }
    
    // Validate input
    wxString input = chat_input->GetValue().Trim();
    if (input.IsEmpty()) {
        return;
    }
    
    try {
        // Clear input and display user message
        chat_input->Clear();
        AddChatMessage("You", input.ToStdString(), LuminaChatColors::SUCCESS_GREEN);
        
        // Get and validate context
        auto* context = llama_manager->GetContextInfo(current_context_id);
        if (!context) {
            LogAndDisplayError("Context not found: " + current_context_id, "Context not available");
            return;
        }
        
        // Apply current template settings from UI before processing message
        ApplyTemplateSettingsToContext(context, current_context_id);
        AddLogMessage("Processing message with context: " + current_context_id);
        
        // Capture template for inspection
        CaptureTemplateForInspection(context);
        
        // Create and execute generation request
        auto callbacks = CreateGenerationCallbacks();
        ExecuteMessageGeneration(context, input.ToStdString(), callbacks);
        
    } catch (const std::exception& e) {
        HandleMessageGenerationError(e.what());
    }
}

void LuminaChatFrame::OnLoadModelFromSettingsUI(const std::string& model_path, int context_size, int gpu_layers) {
    if (!settings_ui || !llama_manager || !settings_manager) {
        AddLogMessage("ERROR: Settings UI or required components not initialized");
        return;
    }

    if (model_path.empty()) {
        AddLogMessage("Please select a model file first");
        AddChatMessage("System", "Please select a model file first", LuminaChatColors::WARNING_ORANGE);
        return;
    }
    
    if (!running) {
        AddLogMessage("Cannot load model: system not running");
        return;
    }
    
    if (model_loading) {
        AddLogMessage("Model loading already in progress");
        return;
    }
    
    // Set loading state immediately
    model_loading = true;
    settings_ui->SetModelLoadingState(true);
    settings_ui->UpdateModelProgress(0);
    SetStatusText("Starting model load...", 1);
    
    AddLogMessage(wxString::Format("Starting unified model loading: %s", model_path).ToStdString());
    
    // Create model config
    ModelConfig config;
    config.model_path = model_path;
    config.context_size = context_size;
    config.gpu_layers = gpu_layers;
    
    // Start model loading in a detached background thread
    std::thread([this, config]() {
        bool success = false;
        std::string error_message;
        
        try {
            // Double-check that we're still supposed to be loading
            if (!model_loading.load()) {
                return; // Loading was cancelled before we started
            }
            
            // Load model through LlamaManager (this is the blocking operation)
            success = llama_manager->LoadModel(current_model_id, config);
            
            if (!success) {
                error_message = "Failed to load model. Check the file path and try again.";
            }
            
        } catch (const std::exception& e) {
            success = false;
            error_message = "Error loading model: " + std::string(e.what());
        } catch (...) {
            success = false;
            error_message = "Unknown error occurred during model loading";
        }
        
        // Final check - only update UI if we're still supposed to be loading
        if (!model_loading.load()) {
            return; // Loading was cancelled, don't update UI
        }
        
        // Update UI on the main thread using CallAfter
        CallAfter([this, success, error_message, config]() {
            try {
                if (success) {
                    model_loaded = true;
                    settings_ui->UpdateModelProgress(100);
                    SetStatusText("Model Loaded", 1);
                    
                    // Create context with the configured size
                    auto* context_info = llama_manager->GetOrCreateContextInfo(current_context_id, current_model_id, config.context_size);
                    
                    if (context_info) {
                        // Apply template settings from Settings UI
                        ApplyTemplateSettingsToContext(context_info, current_context_id);
                        AddLogMessage("Model loaded successfully from unified interface");
                        
                        // Initialize plugins after successful model loading
                        InitializePlugins();
                        
                        ShowSuccessMessage("Model loaded and ready for conversation!");
                    } else {
                        AddLogMessage("ERROR: Failed to create context after model loading");
                        ShowErrorMessage("Failed to create context after model loading");
                        model_loaded = false;
                    }
                } else {
                    model_loaded = false;
                    if (!error_message.empty()) {
                        LogAndDisplayError("Model loading failed: " + error_message, error_message);
                    } else {
                        ShowErrorMessage("Failed to load model. Check the file path and try again.");
                    }
                }
                
            } catch (const std::exception& e) {
                LogAndDisplayError("Error in model loading completion: " + std::string(e.what()),
                                  "Failed to complete model loading: " + std::string(e.what()));
                model_loaded = false;
            }
            
            // Always reset loading state
            model_loading = false;
            if (settings_ui) {
                settings_ui->UpdateModelProgress(0);
                settings_ui->SetModelLoadingState(false);
            }
            UpdateUI();
        });
        
    }).detach(); // Detach the thread so it runs independently
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
        ShowErrorMessage("Please enter a Discord bot token first");
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
            AddChatMessage("System", "Discord bot connected and ready!", LuminaChatColors::DISCORD_BLUE);
            
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
        LogAndDisplayError(wxString::Format("Error connecting to Discord: %s", e.what()).ToStdString(),
                          wxString::Format("Failed to connect to Discord: %s", e.what()).ToStdString());
        discord_connected = false;
    }
    
    connect_discord_button->Enable(true);
    UpdateUI();
}

void LuminaChatFrame::OnClearChat(wxCommandEvent& event) {
    chat_display->Clear();
    
    // Also clear the context history to reset token count
    if (llama_manager && model_loaded) {
        auto* context = llama_manager->GetContextInfo(current_context_id);
        if (context) {
            context->ClearContext();  // This clears both context and resets token count
            AddLogMessage("Chat context cleared - token count reset");
            UpdateContextStatus();  // Update status bar to reflect cleared context
        }
    }
    
    AddLogMessage("Chat display cleared");
}

void LuminaChatFrame::OnClearLogs(wxCommandEvent& event) {
    logs_display->Clear();
}

void LuminaChatFrame::OnTimer(wxTimerEvent& event) {
    if (running && orchestrator) {
        orchestrator->ProcessScheduledTasks();
        UpdateUI();
        
        // Update plugin debug info at regular intervals
        static int debug_update_counter = 0;
        debug_update_counter++;
        if (debug_update_counter >= LuminaChatConstants::DEBUG_UPDATE_INTERVAL) {
            debug_update_counter = 0;
            UpdateSummaryPluginDebugInfo();
            UpdateEmoTagPluginDebugInfo();
        }
    }
}

void LuminaChatFrame::OnClose(wxCloseEvent& event) {
    Stop();
    event.Skip();
}

void LuminaChatFrame::StartStreamingMessage(const std::string& sender, const wxColour& color) {
    is_streaming = true;
    current_assistant_message.clear();
    
    wxDateTime now = wxDateTime::Now();
    
    chat_display->BeginSuppressUndo();
    chat_display->SetInsertionPointEnd();
    
    chat_display->BeginTextColour(LuminaChatColors::TIMESTAMP_GRAY);
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
    
    // Store the position where the assistant message content starts
    assistant_message_start_pos = chat_display->GetLastPosition();
    
    chat_display->EndSuppressUndo();
}

void LuminaChatFrame::AppendToStreamingMessage(const std::string& text) {
    if (!is_streaming) {
        return;
    }
    
    current_assistant_message += text;
    
    chat_display->BeginSuppressUndo();
    chat_display->SetInsertionPointEnd();
    chat_display->WriteText(text);
    chat_display->EndSuppressUndo();
    chat_display->ScrollIntoView(chat_display->GetLastPosition(), WXK_DOWN);
}

void LuminaChatFrame::ReplaceStreamingMessage(const std::string& text) {
    if (!is_streaming) {
        return;
    }
    
    // Calculate the range of the current streaming message
    if (assistant_message_start_pos == -1) {
        // No streaming message started yet, just append
        AppendToStreamingMessage(text);
        return;
    }
    
    chat_display->BeginSuppressUndo();
    
    // Select and replace the current streaming message content
    long current_end = chat_display->GetLastPosition();
    chat_display->SetSelection(assistant_message_start_pos, current_end);
    chat_display->WriteText(text);
    
    chat_display->EndSuppressUndo();
    chat_display->ScrollIntoView(chat_display->GetLastPosition(), WXK_DOWN);
    
    // Update the cached message content
    current_assistant_message = text;
}

void LuminaChatFrame::EndStreamingMessage() {
    if (!is_streaming) {
        return;
    }
    
    is_streaming = false;
    
    chat_display->BeginSuppressUndo();
    chat_display->SetInsertionPointEnd();
    chat_display->WriteText("\n");
    chat_display->EndSuppressUndo();
    chat_display->ScrollIntoView(chat_display->GetLastPosition(), WXK_DOWN);
    
    current_assistant_message.clear();
    assistant_message_start_pos = -1;
    
    // Set focus back to the input field so users can immediately type their next message
    chat_input->SetFocus();
}

void LuminaChatFrame::OnStopGeneration(wxCommandEvent& event) {
    if (!running || !llama_manager) {
        return;
    }
    
    AddLogMessage("Stopping generation...");
    
    // Get the current context and stop generation
    auto* context = llama_manager->GetContextInfo(current_context_id);
    if (context && context->IsGenerating()) {
        context->StopGeneration();
        AddLogMessage("Generation stop requested");
    } else {
        AddLogMessage("No active generation to stop");
    }
}

void LuminaChatFrame::UpdateTemplateDisplay(const std::string& template_content) {
    last_finalized_template = template_content;
    
    // Create a comprehensive header for the template with debugging info
    wxDateTime now = wxDateTime::Now();
    
    std::ostringstream display_stream;
    display_stream << "=== FINALIZED CHAT TEMPLATE INSPECTION ===\n";
    display_stream << "Generated: " << now.Format("%Y-%m-%d %H:%M:%S").ToStdString() << "\n";
    display_stream << "Context ID: " << current_context_id << "\n";
    display_stream << "Model ID: " << current_model_id << "\n";
    display_stream << "Template Length: " << template_content.length() << " characters\n";
    display_stream << "===========================================\n\n";
    
    // Show the actual template that was dynamically generated by ChatTemplateManager
    display_stream << "TEMPLATE CONTENT (dynamically generated by ChatTemplateManager):\n";
    display_stream << "--------------------------------------------------------------\n";
    display_stream << template_content;
    
    template_display->SetValue(display_stream.str());
    template_display->SetInsertionPoint(0);  // Scroll to top
}

void LuminaChatFrame::OnClearTemplate(wxCommandEvent& event) {
    template_display->SetValue(
        "Template Inspection\n"
        "===================\n\n"
        "This tab shows the exact finalized chat template that was sent to the AI\n"
        "after all variable substitutions and processing has occurred.\n\n"
        "The template will be updated each time you send a message to the AI.\n"
        "Use this to debug template processing and verify that variables are\n"
        "being substituted correctly.\n\n"
        "Template content will appear here after sending your first message..."
    );
    last_finalized_template.clear();
    AddLogMessage("Template display cleared");
}

void LuminaChatFrame::UpdateSummaryPluginStatus(const std::string& status, const wxColour& color) {
    if (summarization_ui) {
        summarization_ui->UpdateStatus(status, color);
    }
}

void LuminaChatFrame::UpdateEmoTagPluginStatus(const std::string& status, const wxColour& color) {
    if (emotag_ui) {
        emotag_ui->UpdateStatus(status, color);
    }
}

void LuminaChatFrame::UpdateSummaryPluginDebugInfo() {
    if (summarization_ui && summarization_plugin) {
        summarization_ui->UpdateDebugInfo(summarization_plugin.get());
    }
}

void LuminaChatFrame::UpdateEmoTagPluginDebugInfo() {
    if (emotag_ui && emotag_plugin) {
        emotag_ui->UpdateDebugInfo(emotag_plugin.get());
    }
}

// Helper methods for initialization
std::string LuminaChatFrame::GetEnvironmentDescriptionFromUI() const {
    if (settings_ui) {
        return settings_ui->GetEnvironmentDescription();
    }
    return "";
}

std::string LuminaChatFrame::GetIdentityDirectiveFromUI() const {
    if (settings_ui) {
        return settings_ui->GetIdentityDirective();
    }
    return "";
}

std::string LuminaChatFrame::GetSystemPromptFromUI() const {
    if (settings_ui) {
        return settings_ui->GetSystemPrompt();
    }
    return "";
}

void LuminaChatFrame::ApplyTemplateSettingsToContext(ContextInfo* context, const std::string& context_id) {
    if (!context) {
        AddLogMessage("WARNING: Cannot apply template settings to null context");
        return;
    }
    
    // Don't apply template settings to summary contexts
    if (context_id.find("summary") != std::string::npos) {
        AddLogMessage("Skipping template settings for summary context: " + context_id);
        return;
    }
    
    std::string environment_description = GetEnvironmentDescriptionFromUI();
    std::string identity_directive = GetIdentityDirectiveFromUI();
    std::string system_prompt = GetSystemPromptFromUI();
    
    if (!environment_description.empty()) {
        context->UpdateEnvironment(environment_description);
        AddLogMessage("Applied environment description to context: " + context_id);
    }
    
    if (!identity_directive.empty()) {
        context->UpdateIdentity(identity_directive);
        AddLogMessage("Applied identity directive to context: " + context_id);
    }
    
    if (!system_prompt.empty()) {
        context->UpdateSystemPrompt(system_prompt);
        AddLogMessage("Applied system prompt to context: " + context_id);
    }
}

void LuminaChatFrame::LoadUISettings() {
    if (!settings_manager) {
        AddLogMessage("WARNING: Cannot load UI settings - SettingsManager not initialized");
        return;
    }
    
    AddLogMessage("Loading UI settings from configuration...");
    
    try {
        // Load outer voice UI configuration
        if (outer_voice_ui) {
            outer_voice_ui->LoadSettings();
        }
        
        // Load inner voice UI configuration
        if (inner_voice_ui) {
            inner_voice_ui->LoadSettings();
        }
        
        // Load general settings UI configuration
        if (settings_ui) {
            settings_ui->LoadSettings();
        }
        
        // Load plugin settings through UI managers
        if (summarization_ui) {
            summarization_ui->LoadSettings(settings_manager.get());
        }
        if (emotag_ui) {
            emotag_ui->LoadSettings(settings_manager.get());
        }
        if (context_pruning_ui) {
            context_pruning_ui->LoadSettings(settings_manager.get());
        }
        
        // Load Discord settings
        if (discord_token_text) {
            std::string discord_token = settings_manager->GetString("Discord", "bot_token", "");
            discord_token_text->SetValue(discord_token);
            if (!discord_token.empty()) {
                AddLogMessage("Loaded Discord bot token from settings");
            }
        }
        
        if (auto_respond_checkbox) {
            bool auto_respond = settings_manager->GetBool("Discord", "auto_respond", true);
            auto_respond_checkbox->SetValue(auto_respond);
            AddLogMessage("Loaded Discord auto-respond setting: " + std::string(auto_respond ? "enabled" : "disabled"));
        }
        
        // Load logging level
        if (log_level_choice) {
            std::string log_level = settings_manager->GetString("Logging", "level", "INFO");
            int selection = 1; // Default to INFO
            if (log_level == "DEBUG") selection = 0;
            else if (log_level == "INFO") selection = 1;
            else if (log_level == "WARNING") selection = 2;
            else if (log_level == "ERROR") selection = 3;
            
            log_level_choice->SetSelection(selection);
            AddLogMessage("Loaded log level: " + log_level);
        }
        
        // Update UI to reflect loaded values
        UpdateUI();
        
        AddLogMessage("UI settings loaded successfully");
        
    } catch (const std::exception& e) {
        AddLogMessage("Error loading UI settings: " + std::string(e.what()));
    }
}

void LuminaChatFrame::SaveUISettings() {
    if (!settings_manager) {
        AddLogMessage("WARNING: Cannot save UI settings - SettingsManager not initialized");
        return;
    }
    
    AddLogMessage("Saving UI settings to configuration...");
    
    try {
        // Save outer voice UI configuration
        if (outer_voice_ui) {
            outer_voice_ui->SaveSettings();
        }
        
        // Save inner voice UI configuration
        if (inner_voice_ui) {
            inner_voice_ui->SaveSettings();
        }
        
        // Save general settings UI configuration
        if (settings_ui) {
            settings_ui->SaveSettings();
        }
        
        // Save plugin settings through UI managers
        if (summarization_ui) {
            summarization_ui->SaveSettings(settings_manager.get());
        }
        if (emotag_ui) {
            emotag_ui->SaveSettings(settings_manager.get());
        }
        if (context_pruning_ui) {
            context_pruning_ui->SaveSettings(settings_manager.get());
        }
        
        // Save Discord settings
        if (discord_token_text) {
            std::string discord_token = discord_token_text->GetValue().ToStdString();
            settings_manager->SetString("Discord", "bot_token", discord_token);
        }
        
        if (auto_respond_checkbox) {
            bool auto_respond = auto_respond_checkbox->GetValue();
            settings_manager->SetBool("Discord", "auto_respond", auto_respond);
        }
        
        // Save logging level
        if (log_level_choice) {
            int selection = log_level_choice->GetSelection();
            std::string log_level = "INFO"; // Default
            switch (selection) {
                case 0: log_level = "DEBUG"; break;
                case 1: log_level = "INFO"; break;
                case 2: log_level = "WARNING"; break;
                case 3: log_level = "ERROR"; break;
            }
            settings_manager->SetString("Logging", "level", log_level);
        }
        
        // Actually save the settings to file
        if (settings_manager->SaveSettings()) {
            AddLogMessage("UI settings saved successfully");
        } else {
            AddLogMessage("ERROR: Failed to save UI settings to file");
        }
        
    } catch (const std::exception& e) {
        AddLogMessage("Error saving UI settings: " + std::string(e.what()));
    }
}

// === Message Generation Helper Methods ===

void LuminaChatFrame::CaptureTemplateForInspection(ContextInfo* context) {
    try {
        // Build the full prompt to capture the template after variable substitution
        std::string finalized_template = context->BuildFullPrompt();
        
        // Update the template inspection tab with the finalized template
        UpdateTemplateDisplay(finalized_template);
        AddLogMessage("Template inspection updated with finalized template");
        
    } catch (const std::exception& template_e) {
        AddLogMessage("Warning: Could not capture template for inspection: " + std::string(template_e.what()));
    }
}

GenerationCallbacks LuminaChatFrame::CreateGenerationCallbacks() {
    return GenerationCallbacks(
        // Token callback - called for each token as it's generated
        [this](const std::string& complete_response) {
            // Since we now receive the complete response each time, replace the content
            this->CallAfter([this, complete_response]() {
                ReplaceStreamingMessage(complete_response);
            });
        },
        
        // Completion callback - called when generation is done
        [this](const std::string& full_response, bool success) {
            this->CallAfter([this, full_response, success]() {
                EndStreamingMessage();
                SetGenerationUIState(false);  // Re-enable UI after generation
                UpdateContextStatus();  // Update context usage in status bar
                
                // PERFORMANCE FIX: Delay context monitoring to avoid interference with generation cleanup
                if (orchestrator) {
                    // Use a timer to delay monitoring by the configured delay after generation completes
                    auto timer = new wxTimer();
                    timer->Bind(wxEVT_TIMER, [this, timer](wxTimerEvent&) {
                        if (orchestrator) {
                            orchestrator->MonitorAllContextSizes();
                        }
                        delete timer;
                    });
                    timer->StartOnce(LuminaChatConstants::CONTEXT_MONITORING_DELAY_MS);
                }
                
                if (success) {
                    AddLogMessage("Response generation completed successfully");
                    
                    // Request emotional analysis for the context after AI response
                    if (orchestrator) {
                        auto* context = orchestrator->GetLlamaManager()->GetContextInfo(current_context_id);
                        auto* emotag_plugin = orchestrator->GetEmoTagPlugin();
                        if (context && emotag_plugin) {
                            emotag_plugin->RequestEmotionalAnalysis(current_context_id, context->GetMessageHistory());
                            AddLogMessage("Requested emotional analysis for context: " + current_context_id);
                            
                            // Immediately trigger processing of emotion analysis buffer
                            // instead of waiting for scheduled task
                            std::thread([this]() {
                                if (orchestrator) {
                                    orchestrator->ProcessEmotionAnalysisBuffer();
                                }
                            }).detach();
                        }
                    }
                } else {
                    AddLogMessage("Response generation was stopped or failed");
                    if (full_response.empty()) {
                        AddChatMessage("System", "Response generation was interrupted.", LuminaChatColors::ERROR_RED);
                    }
                }
            });
        },
        
        // Error callback - called if there's an error
        [this](const std::string& error_message) {
            this->CallAfter([this, error_message]() {
                if (is_streaming) {
                    EndStreamingMessage();
                }
                SetGenerationUIState(false);  // Re-enable UI on error
                UpdateContextStatus();  // Update context usage in status bar
                
                // Trigger context monitoring even on error to check size
                if (orchestrator) {
                    orchestrator->MonitorAllContextSizes();
                }
                
                AddLogMessage("Error generating response: " + error_message);
                AddChatMessage("System", "Error: " + error_message, LuminaChatColors::ERROR_RED);
            });
        }
    );
}

void LuminaChatFrame::ExecuteMessageGeneration(ContextInfo* context, const std::string& input, const GenerationCallbacks& callbacks) {
    // Start streaming message display
    StartStreamingMessage("Assistant", LuminaChatColors::INFO_BLUE);
    SetGenerationUIState(true);  // Disable UI during generation
    
    // Start async generation
    bool started = context->HandleInputAsync(input, callbacks, "user");
    if (!started) {
        EndStreamingMessage();
        SetGenerationUIState(false);  // Re-enable UI on failure
        LogAndDisplayError("Failed to start async generation", "Failed to start response generation");
    }
}

void LuminaChatFrame::HandleMessageGenerationError(const std::string& error_message) {
    if (is_streaming) {
        EndStreamingMessage();
    }
    SetGenerationUIState(false);  // Re-enable UI on exception
    LogAndDisplayError("Error sending message: " + error_message, "Error: " + error_message);
}

// === Common UI Helper Methods ===

void LuminaChatFrame::ShowErrorMessage(const std::string& message, const std::string& context) {
    std::string full_message = context.empty() ? message : context + ": " + message;
    AddLogMessage("ERROR: " + full_message);
    AddChatMessage("System", message, LuminaChatColors::ERROR_RED);
}

void LuminaChatFrame::ShowSuccessMessage(const std::string& message) {
    AddLogMessage(message);
    AddChatMessage("System", message, LuminaChatColors::SUCCESS_GREEN);
}

void LuminaChatFrame::ShowWarningMessage(const std::string& message) {
    AddLogMessage("WARNING: " + message);
    AddChatMessage("System", message, LuminaChatColors::WARNING_ORANGE);
}

void LuminaChatFrame::LogAndDisplayError(const std::string& log_msg, const std::string& user_msg) {
    AddLogMessage("ERROR: " + log_msg);
    if (!user_msg.empty()) {
        AddChatMessage("System", user_msg, LuminaChatColors::ERROR_RED);
    }
}

void LuminaChatFrame::LogAndDisplaySuccess(const std::string& log_msg, const std::string& user_msg) {
    AddLogMessage(log_msg);
    if (!user_msg.empty()) {
        AddChatMessage("System", user_msg, LuminaChatColors::SUCCESS_GREEN);
    }
}
