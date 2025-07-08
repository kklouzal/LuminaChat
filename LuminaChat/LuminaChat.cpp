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
#include <wx/splitter.h>

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
#include <sstream>

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
    void OnLogLevelChanged(wxCommandEvent& event);

    // Settings UI integration
    void OnLoadAllModelsFromVoiceSettings();

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
    std::string GetOuterVoiceSystemPromptFromUI() const;
    std::string GetInnerVoiceSystemPromptFromUI() const;
    void ApplyTemplateSettingsToContext(ContextInfo* context, const std::string& context_id);

    // Message generation helper methods
    void CaptureTemplateForInspection(ContextInfo* context, const std::string& context_id = "");
    GenerationCallbacks CreateGenerationCallbacks();
    void ExecuteMessageGeneration(ContextInfo* context, const std::string& input, const GenerationCallbacks& callbacks);
    void ExecuteTwoStageReasoning(ContextInfo* inner_context, ContextInfo* outer_context, const std::string& input);
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
    wxSplitterWindow* chat_splitter;  // Splitter for resizable layout
    wxPanel* inner_voice_panel;       // Panel for inner voice section
    wxPanel* main_chat_panel;         // Panel for main chat section
    wxTextCtrl* inner_voice_display;  // New textbox for inner voice output
    wxRichTextCtrl* chat_display;
    wxTextCtrl* chat_input;
    wxButton* send_button;
    wxButton* stop_button;
    wxButton* clear_button;
    
    // Discord Panel
    wxPanel* discord_panel;
    wxTextCtrl* discord_token_text;
    wxButton* connect_discord_button;
    wxTextCtrl* allowed_channels_text;
    wxListBox* channels_list;
    wxChoice* active_channel_choice;
    wxCheckBox* auto_respond_checkbox;
    
    // Logs Panel
    wxPanel* logs_panel;
    wxTextCtrl* logs_display;
    wxButton* clear_logs_button;
    wxChoice* log_level_choice;
    
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
    std::string current_context_id{"outer_context"};
    std::string current_model_id{"outer_model"};
    
    // Streaming state management
    std::atomic<bool> is_streaming{false};
    std::string current_assistant_message;
    long assistant_message_start_pos = -1;
    
    // Inner voice streaming state management
    std::atomic<bool> is_inner_voice_streaming{false};
    std::string current_inner_voice_message;
    
    // UI creation methods
    void CreateChatPanel();
    void CreateDiscordPanel();
    void CreateLogsPanel();
    
    // UI update methods
    void UpdateUI();
    void UpdateModelProgress(int progress);
    void UpdateContextStatus();  // Track and display context usage
    void UpdateDiscordAllowedChannels();  // Update Discord manager with allowed channels
    void RefreshDiscordChannelList();  // Refresh the Discord channel list from the manager
    void AddChatMessage(const std::string& sender, const std::string& message, const wxColour& color = wxNullColour);
    void StartStreamingMessage(const std::string& sender, const wxColour& color = wxNullColour);
    void AppendToStreamingMessage(const std::string& text);
    void ReplaceStreamingMessage(const std::string& text);  // Replace entire streaming content
    void EndStreamingMessage();
    
    // Inner voice streaming methods
    void StartInnerVoiceStreaming();
    void AppendToInnerVoiceStreaming(const std::string& text);
    void ReplaceInnerVoiceStreaming(const std::string& text);
    void EndInnerVoiceStreaming();
    
    void SetGenerationUIState(bool generating);  // Enable/disable UI during generation
    void AddLogMessage(const std::string& message);
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
    ID_ClearLogs
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
    SetStatusText("Inner: --/-- | Outer: --/--", 2);
    
    // Create main notebook for tabbed interface
    notebook = new wxNotebook(this, wxID_ANY);
    
    // Create all UI panels
    CreateChatPanel();
    CreateDiscordPanel();
    CreateLogsPanel();
    
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
    
    // Create splitter window for resizable layout
    chat_splitter = new wxSplitterWindow(chat_panel, wxID_ANY, 
                                        wxDefaultPosition, wxDefaultSize,
                                        wxSP_3D | wxSP_LIVE_UPDATE);
    chat_splitter->SetMinimumPaneSize(80);  // Minimum size for each pane
    
    // Create top panel for inner voice display
    inner_voice_panel = new wxPanel(chat_splitter);
    wxStaticText* inner_voice_label = new wxStaticText(inner_voice_panel, wxID_ANY, "Inner Voice Context (Real-time):");
    inner_voice_label->SetFont(wxFont(9, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD));
    
    inner_voice_display = new wxTextCtrl(inner_voice_panel, wxID_ANY, wxEmptyString,
                                        wxDefaultPosition, wxDefaultSize,
                                        wxTE_READONLY | wxTE_MULTILINE | wxHSCROLL | wxVSCROLL);
    inner_voice_display->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
    inner_voice_display->SetBackgroundColour(wxColour(248, 248, 255)); // Light blue tint
    inner_voice_display->SetValue("Inner voice output will appear here during generation...");
    
    // Layout for inner voice panel
    wxBoxSizer* inner_voice_sizer = new wxBoxSizer(wxVERTICAL);
    inner_voice_sizer->Add(inner_voice_label, 0, wxALL, 2);
    inner_voice_sizer->Add(inner_voice_display, 1, wxEXPAND | wxALL, 5);
    inner_voice_panel->SetSizer(inner_voice_sizer);
    
    // Create bottom panel for main chat and input
    main_chat_panel = new wxPanel(chat_splitter);
    
    // Chat display area
    wxStaticText* chat_label = new wxStaticText(main_chat_panel, wxID_ANY, "Main Chat:");
    chat_label->SetFont(wxFont(9, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD));
    
    chat_display = new wxRichTextCtrl(main_chat_panel, wxID_ANY, wxEmptyString,
                                      wxDefaultPosition, wxDefaultSize,
                                      wxRE_READONLY | wxRE_MULTILINE);
    chat_display->SetBackgroundColour(LuminaChatColors::BACKGROUND_LIGHT);
    
    // Input area
    chat_input = new wxTextCtrl(main_chat_panel, wxID_ANY, wxEmptyString,
                               wxDefaultPosition, wxDefaultSize,
                               wxTE_PROCESS_ENTER | wxTE_MULTILINE);
    chat_input->SetMinSize(wxSize(-1, 80));
    
    // Control buttons
    send_button = new wxButton(main_chat_panel, ID_Send, "Send");
    stop_button = new wxButton(main_chat_panel, ID_Stop, "Stop");
    stop_button->Enable(false);  // Initially disabled
    clear_button = new wxButton(main_chat_panel, ID_ClearChat, "Clear");
    
    // Layout for control buttons
    wxBoxSizer* button_sizer = new wxBoxSizer(wxHORIZONTAL);
    button_sizer->Add(send_button, 0, wxALL, LuminaChatConstants::CONTROL_SPACING);
    button_sizer->Add(stop_button, 0, wxALL, LuminaChatConstants::CONTROL_SPACING);
    button_sizer->Add(clear_button, 0, wxALL, LuminaChatConstants::CONTROL_SPACING);
    button_sizer->AddStretchSpacer();
    
    // Layout for main chat panel
    wxBoxSizer* main_chat_sizer = new wxBoxSizer(wxVERTICAL);
    main_chat_sizer->Add(chat_label, 0, wxALL, 2);
    main_chat_sizer->Add(chat_display, 1, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 5);
    main_chat_sizer->Add(chat_input, 0, wxEXPAND | wxALL, 5);
    main_chat_sizer->Add(button_sizer, 0, wxEXPAND);
    main_chat_panel->SetSizer(main_chat_sizer);
    
    // Set up the splitter window
    chat_splitter->SplitHorizontally(inner_voice_panel, main_chat_panel, 150); // Initial split at 150 pixels for inner voice
    
    // Layout for the main chat panel
    wxBoxSizer* chat_panel_sizer = new wxBoxSizer(wxVERTICAL);
    chat_panel_sizer->Add(chat_splitter, 1, wxEXPAND);
    chat_panel->SetSizer(chat_panel_sizer);
    
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
    connect_discord_button = new wxButton(discord_panel, ID_ConnectDiscord, "Connect Discord");
    token_sizer->Add(connect_discord_button, 0, wxALL, 5);
    discord_box->Add(token_sizer, 0, wxEXPAND);
    
    // Allowed channels input
    wxBoxSizer* allowed_channels_sizer = new wxBoxSizer(wxHORIZONTAL);
    allowed_channels_sizer->Add(new wxStaticText(discord_panel, wxID_ANY, "Allowed Channels (comma-separated IDs):"), 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    allowed_channels_text = new wxTextCtrl(discord_panel, wxID_ANY, wxEmptyString);
    allowed_channels_text->SetToolTip("Enter Discord channel IDs separated by commas (e.g., 123456789,987654321)");
    allowed_channels_sizer->Add(allowed_channels_text, 1, wxEXPAND | wxALL, 5);
    discord_box->Add(allowed_channels_sizer, 0, wxEXPAND);
    
    // Auto-save Discord token on kill focus
    discord_token_text->Bind(wxEVT_KILL_FOCUS, [this](wxFocusEvent& event) {
        if (settings_manager) {
            settings_manager->SetString("Discord", "bot_token", discord_token_text->GetValue().ToStdString());
            settings_manager->SaveSettings();
        }
        event.Skip();
    });
    
    // Auto-save allowed channels on kill focus
    allowed_channels_text->Bind(wxEVT_KILL_FOCUS, [this](wxFocusEvent& event) {
        if (settings_manager) {
            std::string channels_value = allowed_channels_text->GetValue().ToStdString();
            AddLogMessage("Saving allowed channels: '" + channels_value + "'");
            settings_manager->SetString("Discord", "allowed_channels", channels_value);
            settings_manager->SaveSettings();
        }
        // Update the Discord manager with new allowed channels
        UpdateDiscordAllowedChannels();
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
        
        // Register Discord response callback to send responses back through DiscordManager
        orchestrator->RegisterDiscordResponseCallback([this](const DiscordChannelResponse& response) {
            if (discord_manager && discord_connected) {
                bool sent = discord_manager->SendAIResponseAdvanced(
                    response.target_channel,
                    response.response_content,
                    response.internal_reasoning,
                    response.context_current,
                    response.context_maximum,
                    response.model_name,
                    response.processing_time
                );
                
                if (sent) {
                    AddLogMessage("Discord embed response sent to channel " + response.target_channel + 
                                 " (context: " + std::to_string(response.context_current) + "/" + 
                                 std::to_string(response.context_maximum) + ")");
                } else {
                    AddLogMessage("ERROR: Failed to send Discord embed response to channel " + response.target_channel);
                }
            } else {
                AddLogMessage("ERROR: Cannot send Discord response - manager not available or not connected");
            }
        });
        AddLogMessage("Discord response callback registered");
        
        // Set up Discord manager dependencies
        discord_manager->SetLlamaManager(llama_manager.get());
        discord_manager->SetOrchestrator(orchestrator.get());
        AddLogMessage("Discord Manager dependencies configured");
        
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
                [this]() { OnLoadAllModelsFromVoiceSettings(); },
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
        AddLogMessage("Ready for model loading - configure individual voice models in their respective tabs, then use General Settings to load all models");
        
        running = true;
        system_timer->Start(LuminaChatConstants::TIMER_INTERVAL_MS);
        
        AddLogMessage("LuminaChat started successfully - configure voice models in individual tabs, then load via General Settings");
        SetStatusText("System Ready - Configure voice models in individual tabs", 0);
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
    chat_display->WriteText(wxString::FromUTF8(sender) + ": ");
    chat_display->EndBold();
    if (color.IsOk()) {
        chat_display->EndTextColour();
    }
    
    chat_display->WriteText(wxString::FromUTF8(message) + "\n");
    
    chat_display->EndSuppressUndo();
    chat_display->ScrollIntoView(chat_display->GetLastPosition(), WXK_DOWN);
}

void LuminaChatFrame::AddLogMessage(const std::string& message) {
    wxDateTime now = wxDateTime::Now();
    wxString log_entry = wxString::Format("[%s] %s\n", now.Format("%Y-%m-%d %H:%M:%S"), wxString::FromUTF8(message));
    
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
        SetStatusText("Inner: --/-- | Outer: --/--", 2);
        return;
    }
    
    try {
        // Get both context infos
        auto* inner_context = llama_manager->GetContextInfo("inner_context");
        auto* outer_context = llama_manager->GetContextInfo("outer_context");
        
        wxString status_text;
        
        // Format inner context info
        if (inner_context) {
            int inner_used = static_cast<int>(inner_context->GetActualContextTokens());
            int inner_max = static_cast<int>(inner_context->GetMaxContextTokens());
            status_text += wxString::Format("Inner: %d/%d", inner_used, inner_max);
        } else {
            status_text += "Inner: --/--";
        }
        
        status_text += " | ";
        
        // Format outer context info
        if (outer_context) {
            int outer_used = static_cast<int>(outer_context->GetActualContextTokens());
            int outer_max = static_cast<int>(outer_context->GetMaxContextTokens());
            status_text += wxString::Format("Outer: %d/%d", outer_used, outer_max);
        } else {
            status_text += "Outer: --/--";
        }
        
        SetStatusText(status_text, 2);
        
    } catch (const std::exception& e) {
        AddLogMessage(wxString::Format("Error updating context status: %s", e.what()).ToStdString());
        SetStatusText("Inner: Error | Outer: Error", 2);
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
                 "* Clean component hierarchy with no circular dependencies\n"
                 "* Plugin architecture foundation for AI workflows\n"
                 "* Dynamic chat template management\n"
                 "* Discord bot integration with auto-response\n"
                 "* Advanced context size management\n"
                 "* Content sanitization and filtering\n"
                 "* Performance-optimized token caching\n\n"
                 "Built with wxWidgets following the rework architecture principles.",
                 "About LuminaChat",
                 wxOK | wxICON_INFORMATION);
}

/**
 * Handle user message sending with two-stage reasoning process.
 * This method orchestrates the complete message flow:
 * 1. Validates system state and user input
 * 2. Stage 1: Inner voice generates reasoning/thoughts
 * 3. Stage 2: Outer voice receives reasoning via UpdateReasoningThoughts
 * 4. Stage 3: Outer voice generates final response with reasoning context
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
        
        // Get and validate both contexts
        auto* inner_context = llama_manager->GetContextInfo("inner_context");
        auto* outer_context = llama_manager->GetContextInfo("outer_context");
        
        // Detailed diagnostics for missing contexts
        if (!inner_context || !outer_context) {
            std::string missing = "";
            std::string diagnostics = "Context availability check:\n";
            
            if (!inner_context) {
                missing += "inner_context ";
                diagnostics += "- Inner context: NOT FOUND\n";
            } else {
                diagnostics += "- Inner context: OK\n";
            }
            
            if (!outer_context) {
                missing += "outer_context ";
                diagnostics += "- Outer context: NOT FOUND\n";
            } else {
                diagnostics += "- Outer context: OK\n";
            }
            
            // Add model loading status
            diagnostics += "Model loading status:\n";
            diagnostics += "- model_loaded flag: " + std::string(model_loaded ? "true" : "false") + "\n";
            diagnostics += "- model_loading flag: " + std::string(model_loading ? "true" : "false") + "\n";
            
            // Check if models are actually loaded
            if (llama_manager) {
                diagnostics += "- LlamaManager ready: " + std::string(llama_manager->IsReady() ? "true" : "false") + "\n";
                
                // Check if models exist
                try {
                    auto* outer_model = llama_manager->GetModelInfo("outer_model");
                    auto* inner_model = llama_manager->GetModelInfo("inner_model");
                    diagnostics += "- Outer model exists: " + std::string(outer_model ? "true" : "false") + "\n";
                    diagnostics += "- Inner model exists: " + std::string(inner_model ? "true" : "false") + "\n";
                } catch (...) {
                    diagnostics += "- Unable to check model status\n";
                }
            }
            
            AddLogMessage(diagnostics);
            LogAndDisplayError("Required contexts not found: " + missing, 
                             "Required AI contexts not available. Please ensure you have:\n"
                             "1. Configured model paths in Outer Voice and Inner Voice tabs\n"
                             "2. Clicked 'Load All Models' in General Settings tab\n"
                             "3. Waited for models to finish loading\n\n"
                             "Check the System Logs tab for detailed diagnostics.");
            return;
        }
        
        // Apply template settings to both contexts
        ApplyTemplateSettingsToContext(inner_context, "inner_context");
        ApplyTemplateSettingsToContext(outer_context, "outer_context");
        AddLogMessage("Processing message with two-stage reasoning: inner -> outer");
        
        // Start the two-stage reasoning process
        ExecuteTwoStageReasoning(inner_context, outer_context, input.ToStdString());
        
    } catch (const std::exception& e) {
        HandleMessageGenerationError(e.what());
    }
}

void LuminaChatFrame::OnLoadAllModelsFromVoiceSettings() {
    if (!settings_ui || !llama_manager || !settings_manager || !outer_voice_ui || !inner_voice_ui) {
        AddLogMessage("ERROR: Required UI components not initialized");
        return;
    }
    
    if (!running) {
        AddLogMessage("Cannot load models: system not running");
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
    SetStatusText("Starting model loading from voice configurations...", 1);
    
    AddLogMessage("Starting model loading from individual voice configurations...");
    
    // Get model configurations from individual voice tabs
    std::string outer_model_path = outer_voice_ui->GetModelPath();
    int outer_context_size = outer_voice_ui->GetContextSize();
    int outer_gpu_layers = outer_voice_ui->GetGPULayers();
    
    std::string inner_model_path = inner_voice_ui->GetModelPath();
    int inner_context_size = inner_voice_ui->GetContextSize();
    int inner_gpu_layers = inner_voice_ui->GetGPULayers();
    
    // Validate that at least one model is configured
    if (outer_model_path.empty() && inner_model_path.empty()) {
        AddLogMessage("No models configured in voice settings");
        AddChatMessage("System", "Please configure models in the Outer Voice or Inner Voice tabs first", LuminaChatColors::WARNING_ORANGE);
        model_loading = false;
        settings_ui->SetModelLoadingState(false);
        return;
    }
    
    // Start model loading in a detached background thread
    std::thread([this, outer_model_path, outer_context_size, outer_gpu_layers, 
                inner_model_path, inner_context_size, inner_gpu_layers]() {
        bool outer_success = false;
        bool inner_success = false;
        std::string error_message;
        int total_progress = 0;
        
        try {
            // Load outer voice model if configured
            if (!outer_model_path.empty()) {
                if (!model_loading.load()) return; // Check for cancellation
                
                AddLogMessage("Loading outer voice model: " + outer_model_path);
                ModelConfig outer_config;
                outer_config.model_path = outer_model_path;
                outer_config.context_size = outer_context_size;
                outer_config.gpu_layers = outer_gpu_layers;
                
                outer_success = llama_manager->LoadModel("outer_model", outer_config);
                total_progress += 50; // 50% progress for first model
                
                CallAfter([this, total_progress]() {
                    if (settings_ui) settings_ui->UpdateModelProgress(total_progress);
                });
                
                if (outer_success) {
                    AddLogMessage("Outer voice model loaded successfully");
                    
                    // Create outer context
                    auto* outer_context = llama_manager->GetOrCreateContextInfo("outer_context", "outer_model", outer_context_size);
                    if (outer_context) {
                        ApplyTemplateSettingsToContext(outer_context, "outer_context");
                        AddLogMessage("Outer voice context created and configured");
                    }
                } else {
                    error_message += "Failed to load outer voice model. ";
                }
            }
            
            // Load inner voice model if configured
            if (!inner_model_path.empty()) {
                if (!model_loading.load()) return; // Check for cancellation
                
                AddLogMessage("Loading inner voice model: " + inner_model_path);
                ModelConfig inner_config;
                inner_config.model_path = inner_model_path;
                inner_config.context_size = inner_context_size;
                inner_config.gpu_layers = inner_gpu_layers;
                
                inner_success = llama_manager->LoadModel("inner_model", inner_config);
                total_progress = 100; // Complete progress
                
                CallAfter([this, total_progress]() {
                    if (settings_ui) settings_ui->UpdateModelProgress(total_progress);
                });
                
                if (inner_success) {
                    AddLogMessage("Inner voice model loaded successfully");
                    
                    // Create inner context
                    auto* inner_context = llama_manager->GetOrCreateContextInfo("inner_context", "inner_model", inner_context_size);
                    if (inner_context) {
                        ApplyTemplateSettingsToContext(inner_context, "inner_context");
                        AddLogMessage("Inner voice context created and configured");
                    }
                } else {
                    error_message += "Failed to load inner voice model. ";
                }
            }
            
        } catch (const std::exception& e) {
            error_message = "Error loading models: " + std::string(e.what());
        } catch (...) {
            error_message = "Unknown error occurred during model loading";
        }
        
        // Final check - only update UI if we're still supposed to be loading
        if (!model_loading.load()) {
            return; // Loading was cancelled, don't update UI
        }
        
        // Update UI on the main thread using CallAfter
        CallAfter([this, outer_success, inner_success, error_message]() {
            try {
                bool any_success = outer_success || inner_success;
                
                if (any_success) {
                    model_loaded = true;
                    settings_ui->UpdateModelProgress(100);
                    settings_ui->SetModelLoadedState(true);
                    SetStatusText("Models Loaded", 1);
                    
                    // Initialize plugins after successful model loading
                    InitializePlugins();
                    
                    if (outer_success && inner_success) {
                        ShowSuccessMessage("Both voice models loaded and ready!");
                        AddLogMessage("All voice models loaded successfully");
                    } else if (outer_success) {
                        ShowSuccessMessage("Outer voice model loaded and ready!");
                        AddLogMessage("Outer voice model loaded successfully");
                    } else if (inner_success) {
                        ShowSuccessMessage("Inner voice model loaded and ready!");
                        AddLogMessage("Inner voice model loaded successfully");
                    }
                } else {
                    model_loaded = false;
                    if (!error_message.empty()) {
                        LogAndDisplayError("Model loading failed: " + error_message, error_message);
                    } else {
                        ShowErrorMessage("Failed to load models. Check the file paths in voice settings.");
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
        connect_discord_button->SetLabel("Connect Discord");
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
        connect_discord_button->SetLabel("Connecting...");
        
        // Update allowed channels before connecting
        UpdateDiscordAllowedChannels();
        
        // Set up message callback for Discord manager
        discord_manager->RegisterMessageCallback([this](const std::string& content, const std::string& channel_id, const std::string& username) {
            // Log the message
            AddLogMessage(wxString::Format("Discord message from %s in channel %s: %s", 
                         username, channel_id, content.substr(0, 50)).ToStdString());
            
            // Route to orchestrator for processing
            if (orchestrator) {
                orchestrator->OnRawDiscordMessage(content, channel_id, username);
            } else {
                AddLogMessage("ERROR: Orchestrator not available for Discord message processing");
            }
        });
        
        // Start connection in a separate thread since D++ is async
        std::thread([this, token]() {
            try {
                if (discord_manager->Connect(token.ToStdString())) {
                    // Connection successful - update UI on main thread
                    CallAfter([this]() {
                        discord_connected = true;
                        AddLogMessage("Connected to Discord successfully");
                        AddChatMessage("System", "Discord bot connected and ready!", LuminaChatColors::DISCORD_BLUE);
                        
                        // Wait a moment for guilds to load, then update channel list
                        wxTimer* timer = new wxTimer();
                        timer->Bind(wxEVT_TIMER, [this, timer](wxTimerEvent&) {
                            RefreshDiscordChannelList();
                            delete timer;
                        });
                        timer->StartOnce(2000);
                        
                        connect_discord_button->SetLabel("Disconnect");
                        connect_discord_button->Enable(true);
                        UpdateUI();
                    });
                } else {
                    // Connection failed
                    CallAfter([this]() {
                        std::string error_details = discord_manager->GetLastError();
                        std::string error_msg = error_details.empty() ? "Failed to connect to Discord" : error_details;
                        
                        AddLogMessage("ERROR: " + error_msg);
                        AddChatMessage("System", "Discord connection failed: " + error_msg, wxColour(255, 0, 0));
                        connect_discord_button->SetLabel("Connect Discord");
                        connect_discord_button->Enable(true);
                        discord_connected = false;
                        UpdateUI();
                    });
                }
            } catch (const std::exception& e) {
                CallAfter([this, e]() {
                    LogAndDisplayError(wxString::Format("Error connecting to Discord: %s", e.what()).ToStdString(),
                                      wxString::Format("Failed to connect to Discord: %s", e.what()).ToStdString());
                    connect_discord_button->SetLabel("Connect Discord");
                    connect_discord_button->Enable(true);
                    discord_connected = false;
                    UpdateUI();
                });
            }
        }).detach();
        
    } catch (const std::exception& e) {
        LogAndDisplayError(wxString::Format("Error starting Discord connection: %s", e.what()).ToStdString(),
                          wxString::Format("Failed to start Discord connection: %s", e.what()).ToStdString());
        connect_discord_button->SetLabel("Connect Discord");
        connect_discord_button->Enable(true);
        discord_connected = false;
        UpdateUI();
    }
}

void LuminaChatFrame::OnClearChat(wxCommandEvent& event) {
    chat_display->Clear();
    
    // Clear inner voice display as well
    if (inner_voice_display) {
        inner_voice_display->SetValue("Inner voice output will appear here during generation...");
    }
    
    // Also clear the context history to reset token count
    if (llama_manager && model_loaded) {
        auto* context = llama_manager->GetContextInfo(current_context_id);
        if (context) {
            context->ClearContext();  // This clears both context and resets token count
            AddLogMessage("Chat context cleared - token count reset");
            UpdateContextStatus();  // Update status bar to reflect cleared context
        }
    }
    
    AddLogMessage("Chat display and inner voice display cleared");
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
    chat_display->WriteText(wxString::FromUTF8(sender) + ": ");
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
    chat_display->WriteText(wxString::FromUTF8(text));
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
    chat_display->WriteText(wxString::FromUTF8(text));
    
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

void LuminaChatFrame::StartInnerVoiceStreaming() {
    is_inner_voice_streaming = true;
    current_inner_voice_message.clear();
    
    if (inner_voice_display) {
        inner_voice_display->SetValue("");
    }
}

void LuminaChatFrame::AppendToInnerVoiceStreaming(const std::string& text) {
    if (!is_inner_voice_streaming || !inner_voice_display) {
        return;
    }
    
    current_inner_voice_message += text;
    
    // Update the display by appending to the current content
    wxString current_content = inner_voice_display->GetValue();
    inner_voice_display->SetValue(current_content + wxString::FromUTF8(text));
    
    // Auto-scroll to the end to show the latest content
    inner_voice_display->SetInsertionPointEnd();
    inner_voice_display->ShowPosition(inner_voice_display->GetLastPosition());
}

void LuminaChatFrame::ReplaceInnerVoiceStreaming(const std::string& text) {
    if (!is_inner_voice_streaming || !inner_voice_display) {
        return;
    }
    
    current_inner_voice_message = text;
    
    // Replace the entire content with the new text
    inner_voice_display->SetValue(wxString::FromUTF8(text));
    
    // Auto-scroll to the end
    inner_voice_display->SetInsertionPointEnd();
    inner_voice_display->ShowPosition(inner_voice_display->GetLastPosition());
}

void LuminaChatFrame::EndInnerVoiceStreaming() {
    if (!is_inner_voice_streaming) {
        return;
    }
    
    is_inner_voice_streaming = false;
    
    if (inner_voice_display) {
        // Add a final newline and completion message
        wxString final_content = inner_voice_display->GetValue() + "\n";
        inner_voice_display->SetValue(final_content);
        inner_voice_display->SetInsertionPointEnd();
    }
    
    current_inner_voice_message.clear();
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

std::string LuminaChatFrame::GetOuterVoiceSystemPromptFromUI() const {
    if (outer_voice_ui) {
        return outer_voice_ui->GetSystemPrompt();
    }
    return "";
}

std::string LuminaChatFrame::GetInnerVoiceSystemPromptFromUI() const {
    if (inner_voice_ui) {
        return inner_voice_ui->GetSystemPrompt();
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
    
    // Get the appropriate system prompt based on context ID
    std::string system_prompt;
    if (context_id.find("outer") != std::string::npos) {
        system_prompt = GetOuterVoiceSystemPromptFromUI();
        AddLogMessage("Using Outer Voice system prompt for context: " + context_id + " (length: " + std::to_string(system_prompt.length()) + ")");
        if (!system_prompt.empty()) {
            AddLogMessage("Outer Voice system prompt content (first 50 chars): '" + system_prompt.substr(0, 50) + (system_prompt.length() > 50 ? "..." : "") + "'");
        }
    } else if (context_id.find("inner") != std::string::npos) {
        system_prompt = GetInnerVoiceSystemPromptFromUI();
        AddLogMessage("Using Inner Voice system prompt for context: " + context_id + " (length: " + std::to_string(system_prompt.length()) + ")");
        if (!system_prompt.empty()) {
            AddLogMessage("Inner Voice system prompt content (first 50 chars): '" + system_prompt.substr(0, 50) + (system_prompt.length() > 50 ? "..." : "") + "'");
        }
    } else {
        // Default to outer voice for unspecified contexts
        system_prompt = GetOuterVoiceSystemPromptFromUI();
        AddLogMessage("Using default (Outer Voice) system prompt for context: " + context_id + " (length: " + std::to_string(system_prompt.length()) + ")");
        if (!system_prompt.empty()) {
            AddLogMessage("Default system prompt content (first 50 chars): '" + system_prompt.substr(0, 50) + (system_prompt.length() > 50 ? "..." : "") + "'");
        }
    }
    
    if (!environment_description.empty()) {
        context->UpdateEnvironment(environment_description);
        AddLogMessage("Applied environment description to context: " + context_id);
    }
    
    if (!identity_directive.empty()) {
        context->UpdateIdentity(identity_directive);
        AddLogMessage("Applied identity directive to context: " + context_id);
    }
    
    // Always update system prompt to ensure proper clearing when empty
    context->UpdateSystemPrompt(system_prompt);
    if (!system_prompt.empty()) {
        AddLogMessage("Applied system prompt to context: " + context_id + " (first 50 chars: '" + system_prompt.substr(0, 50) + (system_prompt.length() > 50 ? "..." : "") + "')");
    } else {
        AddLogMessage("Cleared system prompt for context: " + context_id);
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
        
        if (allowed_channels_text) {
            std::string allowed_channels = settings_manager->GetString("Discord", "allowed_channels", "");
            AddLogMessage("Loading allowed channels from settings: '" + allowed_channels + "'");
            allowed_channels_text->SetValue(allowed_channels);
            if (!allowed_channels.empty()) {
                AddLogMessage("Loaded Discord allowed channels from settings");
                UpdateDiscordAllowedChannels();
            } else {
                AddLogMessage("No allowed channels found in settings");
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
        
        if (allowed_channels_text) {
            std::string allowed_channels = allowed_channels_text->GetValue().ToStdString();
            AddLogMessage("Saving allowed channels to settings: '" + allowed_channels + "'");
            settings_manager->SetString("Discord", "allowed_channels", allowed_channels);
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

void LuminaChatFrame::CaptureTemplateForInspection(ContextInfo* context, const std::string& context_id) {
    try {
        // Build the full prompt to capture the template after variable substitution
        std::string finalized_template = context->BuildFullPrompt();
        
        // Determine which voice UI should display the template
        std::string actual_context_id = context_id.empty() ? current_context_id : context_id;
        
        if (actual_context_id == "inner_context" && inner_voice_ui) {
            inner_voice_ui->UpdateTemplateDisplay(finalized_template);
            AddLogMessage("Inner voice template inspection updated with finalized template");
        } else if (actual_context_id == "outer_context" && outer_voice_ui) {
            outer_voice_ui->UpdateTemplateDisplay(finalized_template);
            AddLogMessage("Outer voice template inspection updated with finalized template");
        } else {
            // Fallback: try to determine from current state or log warning
            std::string context_debug_info = "Cannot determine voice UI - actual_context_id: " + actual_context_id + ", current_context_id: " + current_context_id;
            AddLogMessage("Warning: " + context_debug_info);
        }
        
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

void LuminaChatFrame::ExecuteTwoStageReasoning(ContextInfo* inner_context, ContextInfo* outer_context, const std::string& input) {
    // Set UI state for processing
    SetGenerationUIState(true);
    AddLogMessage("Starting two-stage reasoning process...");
    
    // CRITICAL: Share chat history between contexts so inner voice has the same context as outer voice
    AddLogMessage("Synchronizing chat history between inner and outer voice contexts...");
    const auto& outer_history = outer_context->GetMessageHistory();
    
    // Clear inner context history first to ensure clean state
    inner_context->ClearMessageHistory();
    
    // Copy all historical messages from outer context to inner context
    for (const auto& [role, content] : outer_history) {
        inner_context->AddHistoricalMessage(role, content);
    }
    
    AddLogMessage("Chat history synchronized: " + std::to_string(outer_history.size()) + " messages copied to inner voice context");
    
    // Start inner voice streaming display
    StartInnerVoiceStreaming();
    
    // Stage 1: Generate inner voice reasoning (stream to inner voice display)
    AddLogMessage("Stage 1: Inner voice reasoning...");
    
    // Create callbacks for inner voice (Stage 1) - stream to inner voice display
    auto inner_callbacks = GenerationCallbacks(
        // Token callback - stream to inner voice display
        [this](const std::string& token) {
            this->CallAfter([this, token]() {
                AppendToInnerVoiceStreaming(token);
            });
        },
        
        // Completion callback for inner voice
        [this, outer_context, input](const std::string& inner_response, bool success) {
            this->CallAfter([this, outer_context, input, inner_response, success]() {
                // End inner voice streaming
                EndInnerVoiceStreaming();
                
                if (success && !inner_response.empty()) {
                    AddLogMessage("Stage 1 Complete: Inner voice generated reasoning");
                    AddLogMessage("Inner reasoning length: " + std::to_string(inner_response.length()) + " characters");
                    
                    // Stage 2: Update outer voice with reasoning thoughts
                    AddLogMessage("Stage 2: Updating outer voice with internal reflection...");
                    try {
                        outer_context->UpdateInternalReflection(inner_response);
                        AddLogMessage("Stage 2 Complete: Internal reflection updated in outer voice");
                        
                        // Stage 3: Generate final response with outer voice
                        AddLogMessage("Stage 3: Generating final response with outer voice...");
                        
                        // Capture template for inspection (outer voice with reasoning)
                        CaptureTemplateForInspection(outer_context, "outer_context");
                        
                        // Create callbacks for outer voice final response (with streaming)
                        auto outer_callbacks = CreateGenerationCallbacks();
                        
                        // Start streaming message display for final response
                        StartStreamingMessage("Assistant", LuminaChatColors::INFO_BLUE);
                        
                        // Generate final response with outer voice
                        bool started = outer_context->HandleInputAsync(input, outer_callbacks, "user");
                        if (!started) {
                            EndStreamingMessage();
                            SetGenerationUIState(false);
                            LogAndDisplayError("Failed to start outer voice generation", "Failed to generate final response");
                        }
                        
                    } catch (const std::exception& e) {
                        SetGenerationUIState(false);
                        LogAndDisplayError("Error in Stage 2: " + std::string(e.what()), "Error updating reasoning thoughts");
                    }
                    
                } else {
                    // Inner voice failed
                    SetGenerationUIState(false);
                    std::string error_msg = success ? "Inner voice generated empty response" : "Inner voice generation failed";
                    LogAndDisplayError("Stage 1 Failed: " + error_msg, "Inner voice reasoning failed");
                }
            });
        },
        
        // Error callback for inner voice
        [this](const std::string& error_message) {
            this->CallAfter([this, error_message]() {
                // End inner voice streaming on error
                EndInnerVoiceStreaming();
                SetGenerationUIState(false);
                LogAndDisplayError("Stage 1 Error: " + error_message, "Inner voice reasoning error: " + error_message);
            });
        }
    );
    
    // Capture inner voice template for inspection before starting reasoning
    CaptureTemplateForInspection(inner_context, "inner_context");
    
    // Start inner voice reasoning (Stage 1)
    bool started = inner_context->HandleInputAsync(input, inner_callbacks, "user");
    if (!started) {
        EndInnerVoiceStreaming();  // End streaming if generation fails to start
        SetGenerationUIState(false);
        LogAndDisplayError("Failed to start inner voice reasoning", "Failed to start reasoning process");
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

void LuminaChatFrame::UpdateDiscordAllowedChannels() {
    if (!discord_manager || !allowed_channels_text) {
        return;
    }
    
    try {
        // Parse the comma-separated channel IDs
        std::string channels_str = allowed_channels_text->GetValue().ToStdString();
        std::vector<std::string> channel_ids;
        
        if (!channels_str.empty()) {
            // Split by comma and trim whitespace
            std::stringstream ss(channels_str);
            std::string channel_id;
            
            while (std::getline(ss, channel_id, ',')) {
                // Trim whitespace
                channel_id.erase(0, channel_id.find_first_not_of(" \t\n\r"));
                channel_id.erase(channel_id.find_last_not_of(" \t\n\r") + 1);
                
                // Validate that it's a valid Discord channel ID (numeric)
                if (!channel_id.empty() && std::all_of(channel_id.begin(), channel_id.end(), ::isdigit)) {
                    channel_ids.push_back(channel_id);
                }
            }
        }
        
        // Update the Discord manager with the new allowed channels
        discord_manager->SetAllowedChannels(channel_ids);
        
        AddLogMessage(wxString::Format("Updated allowed Discord channels: %d channels configured", 
                     static_cast<int>(channel_ids.size())).ToStdString());
        
    } catch (const std::exception& e) {
        AddLogMessage(wxString::Format("Error updating Discord allowed channels: %s", e.what()).ToStdString());
    }
}

void LuminaChatFrame::RefreshDiscordChannelList() {
    if (!discord_manager || !discord_connected) {
        return;
    }
    
    try {
        auto channels = discord_manager->GetChannels();
        channels_list->Clear();
        active_channel_choice->Clear();
        
        AddLogMessage(wxString::Format("Loaded %d Discord channels", static_cast<int>(channels.size())).ToStdString());
        
        for (const auto& channel : channels) {
            wxString channel_display = wxString::Format("%s [%s] (%s)%s", 
                                                       channel.name, 
                                                       channel.guild_name,
                                                       channel.id,
                                                       channel.is_allowed ? " *ALLOWED*" : "");
            channels_list->Append(channel_display);
            active_channel_choice->Append(channel.name);
        }
        
        if (!channels.empty()) {
            active_channel_choice->SetSelection(0);
        }
        
        // Update status
        size_t guild_count = discord_manager->GetGuildCount();
        size_t allowed_count = 0;
        for (const auto& channel : channels) {
            if (channel.is_allowed) allowed_count++;
        }
        
        AddLogMessage(wxString::Format("Discord status: %d guilds, %d channels (%d allowed)", 
                     static_cast<int>(guild_count), 
                     static_cast<int>(channels.size()),
                     static_cast<int>(allowed_count)).ToStdString());
        
    } catch (const std::exception& e) {
        AddLogMessage(wxString::Format("Error refreshing Discord channel list: %s", e.what()).ToStdString());
    }
}
