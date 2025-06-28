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
#include <wx/event.h>
#include <wx/scrolwin.h>

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
#include "SummarizationPlugin.hpp"
#include "EmoTagPlugin.hpp"

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
    void OnStopGeneration(wxCommandEvent& event);
    void OnLoadModel(wxCommandEvent& event);
    void OnConnectDiscord(wxCommandEvent& event);
    void OnTimer(wxTimerEvent& event);
    void OnClose(wxCloseEvent& event);
    void OnClearChat(wxCommandEvent& event);
    void OnClearLogs(wxCommandEvent& event);
    void OnClearTemplate(wxCommandEvent& event);
    void OnLoadSummaryModel(wxCommandEvent& event);
    void OnLoadEmoTagModel(wxCommandEvent& event);
    void OnLogLevelChanged(wxCommandEvent& event);

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
    wxTextCtrl* environment_description_text;
    wxTextCtrl* identity_directive_text;
    wxTextCtrl* system_prompt_text;
    
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
    
    // Summary Settings Panel
    wxPanel* summary_panel;
    wxTextCtrl* summary_model_path_text;
    wxButton* browse_summary_model_button;
    wxStaticText* summary_status_text;
    wxTextCtrl* summary_log_output_text;
    wxTextCtrl* summary_last_generation_text;
    wxTextCtrl* summary_system_prompt_text;
    
    // EmoTag Settings Panel
    wxPanel* emotag_panel;
    wxTextCtrl* emotag_model_path_text;
    wxButton* browse_emotag_model_button;
    wxStaticText* emotag_status_text;
    wxTextCtrl* emotag_log_output_text;
    wxTextCtrl* emotag_last_generation_text;
    wxTextCtrl* emotag_system_prompt_text;
    wxSlider* emotag_window_size_slider;
    wxStaticText* emotag_window_size_text;
    wxCheckBox* emotag_include_user_checkbox;
    
    // Template Panel
    wxPanel* template_panel;
    wxTextCtrl* template_display;
    wxButton* clear_template_button;
    
    // Core rework components (in dependency order)
    std::unique_ptr<SettingsManager> settings_manager;
    std::unique_ptr<Sanitizer> sanitizer;
    std::unique_ptr<DiscordManager> discord_manager;
    std::unique_ptr<ContextSizeManager> context_size_manager;
    std::unique_ptr<LlamaManager> llama_manager;
    std::unique_ptr<Orchestrator> orchestrator;
    std::unique_ptr<LuminaChat::SummarizationPlugin> summarization_plugin;
    std::unique_ptr<LuminaChat::EmoTagPlugin> emotag_plugin;
    
    // System state
    std::atomic<bool> running{false};
    std::atomic<bool> model_loaded{false};
    std::atomic<bool> model_loading{false};
    std::atomic<bool> discord_connected{false};
    
    // Timer for scheduled operations
    wxTimer* system_timer;
    
    // Current active context and model IDs
    std::string current_context_id{"main_chat"};
    std::string current_model_id{"main_model"};
    
    // Streaming state management
    std::atomic<bool> is_streaming{false};
    std::string current_assistant_message;
    long assistant_message_start_pos = -1;
    std::string last_finalized_template;  // Store the last template for inspection
    
    // UI creation methods
    void CreateChatPanel();
    void CreateSettingsPanel();
    void CreateDiscordPanel();
    void CreateLogsPanel();
    void CreateSummaryPanel();
    void CreateEmoTagPanel();
    void CreateTemplatePanel();
    
    // UI update methods
    void UpdateUI();
    void UpdateModelProgress(int progress);
    void UpdateContextStatus();  // Track and display context usage
    void AddChatMessage(const std::string& sender, const std::string& message, const wxColour& color = wxNullColour);
    void StartStreamingMessage(const std::string& sender, const wxColour& color = wxNullColour);
    void AppendToStreamingMessage(const std::string& text);
    void EndStreamingMessage();
    void SetGenerationUIState(bool generating);  // Enable/disable UI during generation
    void AddLogMessage(const std::string& message);
    void UpdateTemplateDisplay(const std::string& template_content);  // Update template inspection tab
    void UpdateSummaryPluginStatus(const std::string& status, const wxColour& color = wxNullColour);  // Update summary plugin status
    void UpdateEmoTagPluginStatus(const std::string& status, const wxColour& color = wxNullColour);  // Update emotag plugin status
    void UpdateSummaryPluginDebugInfo();  // Update summary plugin debug textboxes
    void UpdateEmoTagPluginDebugInfo();   // Update emotag plugin debug textboxes
    
    DECLARE_EVENT_TABLE()
};

// Event IDs
enum {
    ID_Send = 1000,
    ID_Stop,
    ID_LoadModel,
    ID_ConnectDiscord,
    ID_Timer,
    ID_ClearChat,
    ID_ClearLogs,
    ID_ClearTemplate,
    ID_BrowseModel,
    ID_BrowseSummaryModel,
    ID_BrowseEmoTagModel
};

// Event table mapping
wxBEGIN_EVENT_TABLE(LuminaChatFrame, wxFrame)
    EVT_MENU(wxID_EXIT, LuminaChatFrame::OnExit)
    EVT_MENU(wxID_ABOUT, LuminaChatFrame::OnAbout)
    EVT_BUTTON(ID_Send, LuminaChatFrame::OnSendMessage)
    EVT_BUTTON(ID_Stop, LuminaChatFrame::OnStopGeneration)
    EVT_BUTTON(ID_LoadModel, LuminaChatFrame::OnLoadModel)
    EVT_BUTTON(ID_ConnectDiscord, LuminaChatFrame::OnConnectDiscord)
    EVT_BUTTON(ID_BrowseModel, LuminaChatFrame::OnLoadModel)
    EVT_BUTTON(ID_ClearChat, LuminaChatFrame::OnClearChat)
    EVT_BUTTON(ID_ClearLogs, LuminaChatFrame::OnClearLogs)
    EVT_BUTTON(ID_ClearTemplate, LuminaChatFrame::OnClearTemplate)
    EVT_BUTTON(ID_BrowseSummaryModel, LuminaChatFrame::OnLoadSummaryModel)
    EVT_BUTTON(ID_BrowseEmoTagModel, LuminaChatFrame::OnLoadEmoTagModel)
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
    CreateSettingsPanel();
    CreateDiscordPanel();
    CreateLogsPanel();
    CreateSummaryPanel();
    CreateEmoTagPanel();
    CreateTemplatePanel();
    
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
    stop_button = new wxButton(chat_panel, ID_Stop, "Stop");
    stop_button->Enable(false);  // Initially disabled
    clear_button = new wxButton(chat_panel, ID_ClearChat, "Clear");
    
    // Layout
    wxBoxSizer* button_sizer = new wxBoxSizer(wxHORIZONTAL);
    button_sizer->Add(send_button, 0, wxALL, 5);
    button_sizer->Add(stop_button, 0, wxALL, 5);
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
    
    // Create a scrolled window to contain all settings
    wxScrolledWindow* scrolled_window = new wxScrolledWindow(settings_panel, wxID_ANY, 
                                                            wxDefaultPosition, wxDefaultSize, 
                                                            wxVSCROLL | wxHSCROLL);
    scrolled_window->SetScrollRate(10, 10);
    
    // Model configuration group
    wxStaticBoxSizer* model_box = new wxStaticBoxSizer(wxVERTICAL, scrolled_window, "Model Configuration");
    
    // Model path selection
    wxBoxSizer* path_sizer = new wxBoxSizer(wxHORIZONTAL);
    path_sizer->Add(new wxStaticText(scrolled_window, wxID_ANY, "Model Path:"), 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    model_path_text = new wxTextCtrl(scrolled_window, wxID_ANY);
    path_sizer->Add(model_path_text, 1, wxEXPAND | wxALL, 5);
    browse_model_button = new wxButton(scrolled_window, ID_BrowseModel, "Browse...");
    path_sizer->Add(browse_model_button, 0, wxALL, 5);
    model_box->Add(path_sizer, 0, wxEXPAND);
    
    // Bind text change event to save model path
    model_path_text->Bind(wxEVT_TEXT, [this](wxCommandEvent& event) {
        if (settings_manager) {
            settings_manager->SetString("Models", "main_model_path", model_path_text->GetValue().ToStdString());
            // Don't auto-save on every keystroke for performance, just mark dirty
        }
    });
    
    // Context size control
    wxBoxSizer* context_sizer = new wxBoxSizer(wxHORIZONTAL);
    context_sizer->Add(new wxStaticText(scrolled_window, wxID_ANY, "Context Size:"), 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    context_size_slider = new wxSlider(scrolled_window, wxID_ANY, 4096, 512, 32768, 
                                      wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL | wxSL_LABELS);
    context_sizer->Add(context_size_slider, 1, wxEXPAND | wxALL, 5);
    context_size_label = new wxStaticText(scrolled_window, wxID_ANY, "4096");
    context_sizer->Add(context_size_label, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    model_box->Add(context_sizer, 0, wxEXPAND);
    
    // Bind slider change event to save settings
    context_size_slider->Bind(wxEVT_SLIDER, [this](wxCommandEvent& event) {
        UpdateUI();
        if (settings_manager) {
            settings_manager->SetInt("Models", "main_context_size", context_size_slider->GetValue());
            settings_manager->SaveSettings();
        }
    });
    
    // GPU layers control
    wxBoxSizer* gpu_sizer = new wxBoxSizer(wxHORIZONTAL);
    gpu_sizer->Add(new wxStaticText(scrolled_window, wxID_ANY, "GPU Layers:"), 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    gpu_layers_slider = new wxSlider(scrolled_window, wxID_ANY, 999, 0, 999, 
                                    wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL | wxSL_LABELS);
    gpu_sizer->Add(gpu_layers_slider, 1, wxEXPAND | wxALL, 5);
    gpu_layers_label = new wxStaticText(scrolled_window, wxID_ANY, "999");
    gpu_sizer->Add(gpu_layers_label, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    model_box->Add(gpu_sizer, 0, wxEXPAND);
    
    // Bind slider change event to save settings
    gpu_layers_slider->Bind(wxEVT_SLIDER, [this](wxCommandEvent& event) {
        UpdateUI();
        if (settings_manager) {
            settings_manager->SetInt("Models", "main_gpu_layers", gpu_layers_slider->GetValue());
            settings_manager->SaveSettings();
        }
    });
    
    // Load button and progress
    wxBoxSizer* load_sizer = new wxBoxSizer(wxHORIZONTAL);
    load_model_button = new wxButton(scrolled_window, ID_LoadModel, "Load Model");
    load_sizer->Add(load_model_button, 0, wxALL, 5);
    model_progress = new wxGauge(scrolled_window, wxID_ANY, 100);
    load_sizer->Add(model_progress, 1, wxEXPAND | wxALL, 5);
    model_box->Add(load_sizer, 0, wxEXPAND);
    
    // Template configuration group
    wxStaticBoxSizer* template_box = new wxStaticBoxSizer(wxVERTICAL, scrolled_window, "Template Configuration");
    
    // Environment Description textbox
    template_box->Add(new wxStaticText(scrolled_window, wxID_ANY, "Environment Description:"), 0, wxALL, 5);
    environment_description_text = new wxTextCtrl(scrolled_window, wxID_ANY, wxEmptyString,
                                                 wxDefaultPosition, wxSize(-1, 120),
                                                 wxTE_MULTILINE | wxTE_WORDWRAP);
    environment_description_text->SetToolTip("Define the environment and setting where the conversation takes place. This will be used in template variable replacement for environment-related sections.");
    template_box->Add(environment_description_text, 0, wxEXPAND | wxALL, 5);
    
    // Auto-save on kill focus (when user moves away from field)
    environment_description_text->Bind(wxEVT_KILL_FOCUS, [this](wxFocusEvent& event) {
        if (settings_manager) {
            settings_manager->SetString("Templates", "environment_description", environment_description_text->GetValue().ToStdString());
            settings_manager->SaveSettings();
        }
        event.Skip();
    });
    
    // Identity Directive textbox
    template_box->Add(new wxStaticText(scrolled_window, wxID_ANY, "Identity Directive:"), 0, wxALL, 5);
    identity_directive_text = new wxTextCtrl(scrolled_window, wxID_ANY, wxEmptyString,
                                           wxDefaultPosition, wxSize(-1, 120),
                                           wxTE_MULTILINE | wxTE_WORDWRAP);
    identity_directive_text->SetToolTip("Define the AI's core identity and behavioral guidelines. This will be used in template variable replacement for identity-related sections.");
    template_box->Add(identity_directive_text, 0, wxEXPAND | wxALL, 5);
    
    // Auto-save on kill focus
    identity_directive_text->Bind(wxEVT_KILL_FOCUS, [this](wxFocusEvent& event) {
        if (settings_manager) {
            settings_manager->SetString("Templates", "identity_directive", identity_directive_text->GetValue().ToStdString());
            settings_manager->SaveSettings();
        }
        event.Skip();
    });
    
    // System Prompt textbox
    template_box->Add(new wxStaticText(scrolled_window, wxID_ANY, "System Prompt:"), 0, wxALL, 5);
    system_prompt_text = new wxTextCtrl(scrolled_window, wxID_ANY, wxEmptyString,
                                       wxDefaultPosition, wxSize(-1, 120),
                                       wxTE_MULTILINE | wxTE_WORDWRAP);
    system_prompt_text->SetToolTip("Define the system-level instructions and context. This will be used in template variable replacement for system prompt sections.");
    template_box->Add(system_prompt_text, 0, wxEXPAND | wxALL, 5);
    
    // Auto-save on kill focus
    system_prompt_text->Bind(wxEVT_KILL_FOCUS, [this](wxFocusEvent& event) {
        if (settings_manager) {
            settings_manager->SetString("Templates", "system_prompt", system_prompt_text->GetValue().ToStdString());
            settings_manager->SaveSettings();
        }
        event.Skip();
    });
    
    // Main settings layout for scrolled content
    wxBoxSizer* scrolled_sizer = new wxBoxSizer(wxVERTICAL);
    scrolled_sizer->Add(model_box, 0, wxEXPAND | wxALL, 5);
    scrolled_sizer->Add(template_box, 0, wxEXPAND | wxALL, 5);
    scrolled_sizer->AddStretchSpacer();
    
    scrolled_window->SetSizer(scrolled_sizer);
    
    // Main panel layout with scrolled window
    wxBoxSizer* settings_sizer = new wxBoxSizer(wxVERTICAL);
    settings_sizer->Add(scrolled_window, 1, wxEXPAND | wxALL, 5);
    
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

void LuminaChatFrame::CreateSummaryPanel() {
    summary_panel = new wxPanel(notebook);
    notebook->AddPage(summary_panel, "Summary Settings");
    
    // Create a scrolled window to contain all settings
    wxScrolledWindow* scrolled_window = new wxScrolledWindow(summary_panel, wxID_ANY, 
                                                            wxDefaultPosition, wxDefaultSize, 
                                                            wxVSCROLL | wxHSCROLL);
    scrolled_window->SetScrollRate(10, 10);
    
    // Summary model configuration group
    wxStaticBoxSizer* model_box = new wxStaticBoxSizer(wxVERTICAL, scrolled_window, "Summary Model Configuration");
    
    // Model path selection
    wxBoxSizer* path_sizer = new wxBoxSizer(wxHORIZONTAL);
    path_sizer->Add(new wxStaticText(scrolled_window, wxID_ANY, "Summary Model Path:"), 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    summary_model_path_text = new wxTextCtrl(scrolled_window, wxID_ANY);
    path_sizer->Add(summary_model_path_text, 1, wxEXPAND | wxALL, 5);
    browse_summary_model_button = new wxButton(scrolled_window, ID_BrowseSummaryModel, "Browse...");
    path_sizer->Add(browse_summary_model_button, 0, wxALL, 5);
    model_box->Add(path_sizer, 0, wxEXPAND);
    
    // Bind text change event to save model path
    summary_model_path_text->Bind(wxEVT_TEXT, [this](wxCommandEvent& event) {
        if (settings_manager) {
            settings_manager->SetString("Models", "summary_model_path", summary_model_path_text->GetValue().ToStdString());
            // Don't auto-save on every keystroke for performance, just mark dirty
        }
    });
    
    // Auto-configuration note
    wxStaticText* auto_config_note = new wxStaticText(scrolled_window, wxID_ANY, 
        "Note: Context size will be automatically set to 25% of main model context size.\n"
        "GPU layers will match the main model setting.\n"
        "Model loading is managed automatically by the SummarizationPlugin.");
    auto_config_note->SetFont(auto_config_note->GetFont().Italic());
    auto_config_note->SetForegroundColour(wxColour(100, 100, 100));
    model_box->Add(auto_config_note, 0, wxALL, 5);
    
    // Status display
    wxStaticBoxSizer* status_box = new wxStaticBoxSizer(wxVERTICAL, scrolled_window, "Plugin Status");
    summary_status_text = new wxStaticText(scrolled_window, wxID_ANY, "SummarizationPlugin: Not initialized");
    summary_status_text->SetFont(summary_status_text->GetFont().Bold());
    summary_status_text->SetForegroundColour(wxColour(150, 100, 50));
    status_box->Add(summary_status_text, 0, wxALL, 5);
    
    // Log Output debug display
    status_box->Add(new wxStaticText(scrolled_window, wxID_ANY, "Log Output:"), 0, wxLEFT | wxRIGHT | wxTOP, 5);
    summary_log_output_text = new wxTextCtrl(scrolled_window, wxID_ANY, "Plugin logs will appear here...",
                                            wxDefaultPosition, wxSize(-1, 120),
                                            wxTE_READONLY | wxTE_MULTILINE | wxTE_WORDWRAP);
    summary_log_output_text->SetFont(wxFont(8, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
    summary_log_output_text->SetBackgroundColour(wxColour(245, 245, 245));
    status_box->Add(summary_log_output_text, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 5);
    
    // Last Generation debug display
    status_box->Add(new wxStaticText(scrolled_window, wxID_ANY, "Last Generation:"), 0, wxLEFT | wxRIGHT | wxTOP, 5);
    summary_last_generation_text = new wxTextCtrl(scrolled_window, wxID_ANY, "No generation data available yet...",
                                                 wxDefaultPosition, wxSize(-1, 120),
                                                 wxTE_READONLY | wxTE_MULTILINE | wxTE_WORDWRAP);
    summary_last_generation_text->SetFont(wxFont(8, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
    summary_last_generation_text->SetBackgroundColour(wxColour(245, 245, 245));
    status_box->Add(summary_last_generation_text, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 5);
    
    // Summary system prompt configuration group
    wxStaticBoxSizer* prompt_box = new wxStaticBoxSizer(wxVERTICAL, scrolled_window, "Summary System Prompt");
    
    // System Prompt textbox
    prompt_box->Add(new wxStaticText(scrolled_window, wxID_ANY, "System Prompt for Summarization:"), 0, wxALL, 5);
    summary_system_prompt_text = new wxTextCtrl(scrolled_window, wxID_ANY, wxEmptyString,
                                               wxDefaultPosition, wxSize(-1, 200),
                                               wxTE_MULTILINE | wxTE_WORDWRAP);
    summary_system_prompt_text->SetToolTip("Define the system prompt for the summarization model. This will instruct the AI on how to create summaries of conversation history.");
    
    // Set default summary prompt if empty
    summary_system_prompt_text->SetValue(
        "You are a helpful AI assistant that creates concise summaries of conversations. "
        "When given a conversation history, provide a clear and informative summary that captures "
        "the key points, decisions, and context. Focus on preserving important information while "
        "being concise. Format your summary in a structured way with bullet points when appropriate."
    );
    
    prompt_box->Add(summary_system_prompt_text, 1, wxEXPAND | wxALL, 5);
    
    // Auto-save on kill focus
    summary_system_prompt_text->Bind(wxEVT_KILL_FOCUS, [this](wxFocusEvent& event) {
        if (settings_manager) {
            settings_manager->SetString("Summary", "system_prompt", summary_system_prompt_text->GetValue().ToStdString());
            settings_manager->SaveSettings();
        }
        event.Skip();
    });
    
    // Main settings layout for scrolled content
    wxBoxSizer* scrolled_sizer = new wxBoxSizer(wxVERTICAL);
    scrolled_sizer->Add(model_box, 0, wxEXPAND | wxALL, 5);
    scrolled_sizer->Add(status_box, 0, wxEXPAND | wxALL, 5);
    scrolled_sizer->Add(prompt_box, 1, wxEXPAND | wxALL, 5);
    scrolled_sizer->AddStretchSpacer();
    
    scrolled_window->SetSizer(scrolled_sizer);
    
    // Main panel layout with scrolled window
    wxBoxSizer* summary_sizer = new wxBoxSizer(wxVERTICAL);
    summary_sizer->Add(scrolled_window, 1, wxEXPAND | wxALL, 5);
    
    summary_panel->SetSizer(summary_sizer);
}

void LuminaChatFrame::CreateEmoTagPanel() {
    emotag_panel = new wxPanel(notebook);
    notebook->AddPage(emotag_panel, "EmoTag Settings");
    
    // Create a scrolled window to contain all settings
    wxScrolledWindow* scrolled_window = new wxScrolledWindow(emotag_panel, wxID_ANY, 
                                                            wxDefaultPosition, wxDefaultSize, 
                                                            wxVSCROLL | wxHSCROLL);
    scrolled_window->SetScrollRate(10, 10);
    
    // EmoTag model configuration group
    wxStaticBoxSizer* model_box = new wxStaticBoxSizer(wxVERTICAL, scrolled_window, "Emotion Analysis Model Configuration");
    
    // Model path selection
    wxBoxSizer* path_sizer = new wxBoxSizer(wxHORIZONTAL);
    path_sizer->Add(new wxStaticText(scrolled_window, wxID_ANY, "Emotion Model Path:"), 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    emotag_model_path_text = new wxTextCtrl(scrolled_window, wxID_ANY);
    path_sizer->Add(emotag_model_path_text, 1, wxEXPAND | wxALL, 5);
    browse_emotag_model_button = new wxButton(scrolled_window, ID_BrowseEmoTagModel, "Browse...");
    path_sizer->Add(browse_emotag_model_button, 0, wxALL, 5);
    model_box->Add(path_sizer, 0, wxEXPAND);
    
    // Bind text change event to save model path
    emotag_model_path_text->Bind(wxEVT_TEXT, [this](wxCommandEvent& event) {
        if (settings_manager) {
            settings_manager->SetString("Models", "emotion_model_path", emotag_model_path_text->GetValue().ToStdString());
            // Don't auto-save on every keystroke for performance, just mark dirty
        }
    });
    
    // Auto-configuration note
    wxStaticText* auto_config_note = new wxStaticText(scrolled_window, wxID_ANY, 
        "Note: Context size will be automatically set to 30% of main model context size.\n"
        "GPU layers will match the main model setting.\n"
        "Model loading is managed automatically by the EmoTagPlugin.");
    auto_config_note->SetFont(auto_config_note->GetFont().Italic());
    auto_config_note->SetForegroundColour(wxColour(100, 100, 100));
    model_box->Add(auto_config_note, 0, wxALL, 5);
    
    // Status display
    wxStaticBoxSizer* status_box = new wxStaticBoxSizer(wxVERTICAL, scrolled_window, "Plugin Status");
    emotag_status_text = new wxStaticText(scrolled_window, wxID_ANY, "EmoTagPlugin: Not initialized");
    emotag_status_text->SetFont(emotag_status_text->GetFont().Bold());
    emotag_status_text->SetForegroundColour(wxColour(150, 100, 50));
    status_box->Add(emotag_status_text, 0, wxALL, 5);
    
    // Log Output debug display
    status_box->Add(new wxStaticText(scrolled_window, wxID_ANY, "Log Output:"), 0, wxLEFT | wxRIGHT | wxTOP, 5);
    emotag_log_output_text = new wxTextCtrl(scrolled_window, wxID_ANY, "Plugin logs will appear here...",
                                           wxDefaultPosition, wxSize(-1, 120),
                                           wxTE_READONLY | wxTE_MULTILINE | wxTE_WORDWRAP);
    emotag_log_output_text->SetFont(wxFont(8, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
    emotag_log_output_text->SetBackgroundColour(wxColour(245, 245, 245));
    status_box->Add(emotag_log_output_text, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 5);
    
    // Last Generation debug display
    status_box->Add(new wxStaticText(scrolled_window, wxID_ANY, "Last Generation:"), 0, wxLEFT | wxRIGHT | wxTOP, 5);
    emotag_last_generation_text = new wxTextCtrl(scrolled_window, wxID_ANY, "No generation data available yet...",
                                                wxDefaultPosition, wxSize(-1, 120),
                                                wxTE_READONLY | wxTE_MULTILINE | wxTE_WORDWRAP);
    emotag_last_generation_text->SetFont(wxFont(8, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
    emotag_last_generation_text->SetBackgroundColour(wxColour(245, 245, 245));
    status_box->Add(emotag_last_generation_text, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 5);
    
    // Plugin configuration group
    wxStaticBoxSizer* config_box = new wxStaticBoxSizer(wxVERTICAL, scrolled_window, "Analysis Configuration");
    
    // Analysis Window Size
    wxBoxSizer* window_size_sizer = new wxBoxSizer(wxHORIZONTAL);
    window_size_sizer->Add(new wxStaticText(scrolled_window, wxID_ANY, "Analysis Window Size:"), 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    emotag_window_size_slider = new wxSlider(scrolled_window, wxID_ANY, 3, 1, 10, wxDefaultPosition, wxSize(200, -1));
    emotag_window_size_slider->SetToolTip("Number of recent messages to keep and analyze (1-10)");
    window_size_sizer->Add(emotag_window_size_slider, 1, wxEXPAND | wxALL, 5);
    emotag_window_size_text = new wxStaticText(scrolled_window, wxID_ANY, "3");
    emotag_window_size_text->SetFont(emotag_window_size_text->GetFont().Bold());
    window_size_sizer->Add(emotag_window_size_text, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5);
    config_box->Add(window_size_sizer, 0, wxEXPAND);
    
    // Bind slider change event
    emotag_window_size_slider->Bind(wxEVT_SLIDER, [this](wxCommandEvent& event) {
        int value = emotag_window_size_slider->GetValue();
        emotag_window_size_text->SetLabel(wxString::Format("%d", value));
        if (settings_manager) {
            settings_manager->SetInt("Emotion", "analysis_window_size", value);
            settings_manager->SaveSettings();
        }
        if (emotag_plugin) {
            emotag_plugin->SetAnalysisWindow(static_cast<size_t>(value));
        }
    });
    
    // Include User Messages checkbox
    emotag_include_user_checkbox = new wxCheckBox(scrolled_window, wxID_ANY, "Include User Messages in Analysis");
    emotag_include_user_checkbox->SetToolTip("When enabled, user messages will be included in emotional analysis for richer context");
    emotag_include_user_checkbox->SetValue(false); // Default to false
    config_box->Add(emotag_include_user_checkbox, 0, wxALL, 5);
    
    // Bind checkbox change event
    emotag_include_user_checkbox->Bind(wxEVT_CHECKBOX, [this](wxCommandEvent& event) {
        bool value = emotag_include_user_checkbox->GetValue();
        if (settings_manager) {
            settings_manager->SetBool("Emotion", "include_user_messages", value);
            settings_manager->SaveSettings();
        }
        if (emotag_plugin) {
            emotag_plugin->SetIncludeUserMessages(value);
        }
    });
    
    // Configuration note
    wxStaticText* config_note = new wxStaticText(scrolled_window, wxID_ANY, 
        "Analysis is triggered immediately after each AI response.\n"
        "Window size determines how many recent messages are analyzed.\n"
        "Including user messages provides fuller context but uses more tokens.");
    config_note->SetFont(config_note->GetFont().Italic());
    config_note->SetForegroundColour(wxColour(100, 100, 100));
    config_box->Add(config_note, 0, wxALL, 5);
    
    // EmoTag system prompt configuration group
    wxStaticBoxSizer* prompt_box = new wxStaticBoxSizer(wxVERTICAL, scrolled_window, "Emotional Analysis System Prompt");
    
    // System Prompt textbox
    prompt_box->Add(new wxStaticText(scrolled_window, wxID_ANY, "System Prompt for Emotional Analysis:"), 0, wxALL, 5);
    emotag_system_prompt_text = new wxTextCtrl(scrolled_window, wxID_ANY, wxEmptyString,
                                               wxDefaultPosition, wxSize(-1, 200),
                                               wxTE_MULTILINE | wxTE_WORDWRAP);
    emotag_system_prompt_text->SetToolTip("Define the system prompt for the emotional analysis model. This will instruct the AI on how to analyze the emotional state of AI responses.");
    
    // Set default emotion prompt if empty
    emotag_system_prompt_text->SetValue(
        "You are an emotional state analyzer. When given AI assistant responses, analyze the emotional tone, "
        "mood, and psychological state conveyed in the text. Provide a brief emotional overview that captures "
        "the assistant's apparent emotional state, confidence level, and overall demeanor. "
        "Focus on identifying patterns like: confident, uncertain, empathetic, analytical, cheerful, "
        "cautious, enthusiastic, or reserved. Keep your analysis concise and actionable."
    );
    
    prompt_box->Add(emotag_system_prompt_text, 1, wxEXPAND | wxALL, 5);
    
    // Auto-save on kill focus
    emotag_system_prompt_text->Bind(wxEVT_KILL_FOCUS, [this](wxFocusEvent& event) {
        if (settings_manager) {
            settings_manager->SetString("Emotion", "system_prompt", emotag_system_prompt_text->GetValue().ToStdString());
            settings_manager->SaveSettings();
        }
        event.Skip();
    });
    
    // Main settings layout for scrolled content
    wxBoxSizer* scrolled_sizer = new wxBoxSizer(wxVERTICAL);
    scrolled_sizer->Add(model_box, 0, wxEXPAND | wxALL, 5);
    scrolled_sizer->Add(status_box, 0, wxEXPAND | wxALL, 5);
    scrolled_sizer->Add(config_box, 0, wxEXPAND | wxALL, 5);
    scrolled_sizer->Add(prompt_box, 1, wxEXPAND | wxALL, 5);
    scrolled_sizer->AddStretchSpacer();
    
    scrolled_window->SetSizer(scrolled_sizer);
    
    // Main panel layout with scrolled window
    wxBoxSizer* emotag_sizer = new wxBoxSizer(wxVERTICAL);
    emotag_sizer->Add(scrolled_window, 1, wxEXPAND | wxALL, 5);
    
    emotag_panel->SetSizer(emotag_sizer);
}

void LuminaChatFrame::CreateTemplatePanel() {
    template_panel = new wxPanel(notebook);
    notebook->AddPage(template_panel, "Template");
    
    // Template display (read-only monospace text control)
    template_display = new wxTextCtrl(template_panel, wxID_ANY, wxEmptyString,
                                     wxDefaultPosition, wxDefaultSize,
                                     wxTE_READONLY | wxTE_MULTILINE | wxHSCROLL | wxVSCROLL);
    template_display->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
    template_display->SetBackgroundColour(wxColour(250, 250, 250));
    
    // Initial text explaining the purpose
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

// System lifecycle management
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
        
        // Note: SummarizationPlugin initialization is now deferred until model loading
        AddLogMessage("Plugin initialization will be handled when user loads a model");
        
        // Register callbacks (higher components register with lower)
        RegisterCallbacks();
        AddLogMessage("Callback dependencies registered");
        
        // Load UI settings from configuration
        LoadUISettings();
        
        // Note: Model loading is now manual - user must click "Load Model" button
        AddLogMessage("Ready for manual model loading - click 'Load Model' button to proceed");
        
        running = true;
        system_timer->Start(100);
        
        AddLogMessage("LuminaChat started successfully - ready for manual model loading");
        SetStatusText("System Ready - Click 'Load Model' to begin", 0);
        UpdateUI();
        
        // Set initial plugin status
        UpdateSummaryPluginStatus("Waiting for model loading", wxColour(100, 100, 100));
        UpdateEmoTagPluginStatus("Waiting for model loading", wxColour(100, 100, 100));
        
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
        emotag_plugin->Stop();
        emotag_plugin.reset();
        AddLogMessage("EmoTagPlugin stopped");
    }
    if (summarization_plugin) {
        summarization_plugin->Stop();
        summarization_plugin.reset();
        AddLogMessage("SummarizationPlugin stopped");
    }
    orchestrator.reset();
    llama_manager.reset();
    context_size_manager.reset();
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
    load_model_button->Enable(running && !model_path_text->GetValue().IsEmpty() && !model_loading);
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
        
        // Get context usage information from stats
        const auto& stats = context->GetStats();
        int used_tokens = static_cast<int>(stats.current_context_tokens);
        int max_tokens = static_cast<int>(stats.max_context_tokens);
        
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
    
    // Note: UI now uses direct async streaming instead of Orchestrator callbacks
    // This eliminates the synchronous callback tech debt
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
        
        // Get main model context size from settings
        int32_t main_context_size = settings_manager->GetInt("Models", "main_context_size", 4096);
        
        // Create default context with main model
        auto* context = llama_manager->GetOrCreateContextInfo("main_context", "main_model", main_context_size);
        if (context) {
            AddLogMessage("Main context created successfully with context size: " + std::to_string(main_context_size));
            // Apply template settings from UI (identity directive and system prompt)
            ApplyTemplateSettingsToContext(context, "main_context");
        }
    }
    
    // Note: Summary model loading is now handled by SummarizationPlugin during its initialization
    AddLogMessage("Summary model initialization will be handled by SummarizationPlugin");
}

void LuminaChatFrame::InitializePlugins() {
    if (!llama_manager || !orchestrator || !model_loaded) {
        AddLogMessage("ERROR: Cannot initialize plugins - prerequisites not met");
        return;
    }
    
    AddLogMessage("Initializing plugins after successful model loading...");
    
    try {
        // Initialize and start SummarizationPlugin
        summarization_plugin = std::make_unique<LuminaChat::SummarizationPlugin>(orchestrator.get());
        
        // Register callback for plugin status updates
        if (summarization_plugin) {
            // TODO: When SummarizationPlugin supports status callbacks, register here
            // summarization_plugin->RegisterStatusCallback([this](const std::string& status, bool is_error) {
            //     CallAfter([this, status, is_error]() {
            //         wxColour color = is_error ? wxColour(150, 50, 50) : wxColour(0, 150, 0);
            //         UpdateSummaryPluginStatus(status, color);
            //     });
            // });
        }
        
        summarization_plugin->Start();
        AddLogMessage("SummarizationPlugin initialized and started");
        
        // Update plugin status
        UpdateSummaryPluginStatus("Initialized and ready", wxColour(0, 150, 0));
        
        // Initialize and start EmoTagPlugin
        emotag_plugin = std::make_unique<LuminaChat::EmoTagPlugin>(orchestrator.get());
        
        // Register callback for plugin status updates
        if (emotag_plugin) {
            emotag_plugin->SetStatusCallback([this](const std::string& status, bool is_error) {
                CallAfter([this, status, is_error]() {
                    wxColour color = is_error ? wxColour(150, 50, 50) : wxColour(0, 150, 0);
                    UpdateEmoTagPluginStatus(status, color);
                });
            });
        }
        
        emotag_plugin->Start();
        
        // Configure plugin with current UI settings
        if (emotag_window_size_slider) {
            int window_size = emotag_window_size_slider->GetValue();
            emotag_plugin->SetAnalysisWindow(static_cast<size_t>(window_size));
            AddLogMessage("Configured EmoTagPlugin analysis window size: " + std::to_string(window_size));
        }
        
        if (emotag_include_user_checkbox) {
            bool include_user = emotag_include_user_checkbox->GetValue();
            emotag_plugin->SetIncludeUserMessages(include_user);
            AddLogMessage("Configured EmoTagPlugin include user messages: " + std::string(include_user ? "enabled" : "disabled"));
        }
        
        AddLogMessage("EmoTagPlugin initialized and started");
        
        // Update plugin status
        UpdateEmoTagPluginStatus("Initialized and ready", wxColour(0, 150, 0));
        
        AddLogMessage("All plugins initialized successfully");
        
    } catch (const std::exception& e) {
        AddLogMessage(wxString::Format("Error initializing plugins: %s", e.what()).ToStdString());
        UpdateSummaryPluginStatus("Initialization failed", wxColour(150, 50, 50));
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
        
        // Get context and use async streaming for all UI output
        auto* context = llama_manager->GetContextInfo(current_context_id);
        if (!context) {
            AddLogMessage("Error: Context not found: " + current_context_id);
            AddChatMessage("System", "Error: Context not available", wxColour(150, 50, 50));
            return;
        }
        
        // Apply current template settings from UI before processing message
        ApplyTemplateSettingsToContext(context, current_context_id);
        
        AddLogMessage("Processing message with context: " + current_context_id);
        
        // Record user message in EmoTagPlugin if enabled
        if (emotag_plugin) {
            emotag_plugin->RecordUserMessage(current_context_id, input.ToStdString());
            AddLogMessage("User message recorded in EmoTagPlugin for context: " + current_context_id);
        }
        
        // Get the finalized template for inspection BEFORE starting generation
        std::string finalized_template;
        try {
            // Build the full prompt to capture the template after variable substitution
            finalized_template = context->BuildFullPrompt();
            
            // Update the template inspection tab with the finalized template
            UpdateTemplateDisplay(finalized_template);
            AddLogMessage("Template inspection updated with finalized template");
            
        } catch (const std::exception& template_e) {
            AddLogMessage("Warning: Could not capture template for inspection: " + std::string(template_e.what()));
        }
        
        // Create callbacks for streaming response
        GenerationCallbacks callbacks(
            // Token callback - called for each token as it's generated
            [this](const std::string& token_text) {
                // Update UI on main thread
                this->CallAfter([this, token_text]() {
                    AppendToStreamingMessage(token_text);
                });
            },
            
            // Completion callback - called when generation is done
            [this](const std::string& full_response, bool success) {
                this->CallAfter([this, full_response, success]() {
                    EndStreamingMessage();
                    SetGenerationUIState(false);  // Re-enable UI after generation
                    UpdateContextStatus();  // Update context usage in status bar
                    if (success) {
                        AddLogMessage("Response generation completed successfully");
                        
                        // Record AI response in EmoTagPlugin for emotional analysis
                        if (emotag_plugin && !full_response.empty()) {
                            emotag_plugin->RecordAIResponse(current_context_id, full_response);
                            AddLogMessage("AI response recorded in EmoTagPlugin for emotional analysis");
                        }
                    } else {
                        AddLogMessage("Response generation was stopped or failed");
                        if (full_response.empty()) {
                            AddChatMessage("System", "Response generation was interrupted.", wxColour(150, 50, 50));
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
                    AddLogMessage("Error generating response: " + error_message);
                    AddChatMessage("System", "Error: " + error_message, wxColour(150, 50, 50));
                });
            }
        );
        
        // Start streaming message display
        StartStreamingMessage("Assistant", wxColour(50, 50, 150));
        SetGenerationUIState(true);  // Disable UI during generation
        
        // Start async generation
        bool started = context->HandleInputAsync(input.ToStdString(), callbacks, "user");
        if (!started) {
            EndStreamingMessage();
            SetGenerationUIState(false);  // Re-enable UI on failure
            AddLogMessage("Failed to start async generation");
            AddChatMessage("System", "Failed to start response generation", wxColour(150, 50, 50));
        }
        
    } catch (const std::exception& e) {
        if (is_streaming) {
            EndStreamingMessage();
        }
        SetGenerationUIState(false);  // Re-enable UI on exception
        AddLogMessage(wxString::Format("Error sending message: %s", e.what()).ToStdString());
        AddChatMessage("System", wxString::Format("Error: %s", e.what()).ToStdString(), wxColour(150, 50, 50));
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
        AddLogMessage("Please select a model file first");
        AddChatMessage("System", "Please select a model file first", wxColour(150, 100, 50));
        return;
    }
    
    if (!running) {
        AddLogMessage("Cannot load model: system not running");
        return;
    }
    
    // Check if already loading
    if (model_loading) {
        AddLogMessage("Model loading already in progress");
        return;
    }
    
    try {
        LOG_DEBUG_LuminaChat("Starting model load process");
        AddLogMessage(wxString::Format("Loading model: %s", model_path).ToStdString());
        
        // Set loading state FIRST
        model_loading = true;
        UpdateUI(); // This will disable the button and update status
        
        LOG_DEBUG_LuminaChat("Set loading state, calling UpdateUI()");
        
        int context_size = context_size_slider->GetValue();
        int gpu_layers = gpu_layers_slider->GetValue();
        
        LOG_DEBUG_LuminaChat(wxString::Format("Got slider values - context_size: %d, gpu_layers: %d", context_size, gpu_layers).ToStdString());
        
        model_progress->SetValue(0);
        
        LOG_DEBUG_LuminaChat("About to call LlamaManager::LoadModel");
        
        // Load model through LlamaManager
        ModelConfig config;
        config.model_path = model_path.ToStdString();
        config.context_size = context_size;
        config.gpu_layers = gpu_layers;
        
        if (llama_manager->LoadModel(current_model_id, config)) {
            LOG_DEBUG_LuminaChat("LlamaManager::LoadModel returned true - success");
            
            LOG_DEBUG_LuminaChat("Setting model_loaded = true");
            model_loaded = true;
            
            LOG_DEBUG_LuminaChat("Setting progress bar to 100%");
            model_progress->SetValue(100);  // Show completion
            
            LOG_DEBUG_LuminaChat("Setting status text to Model Loaded");
            SetStatusText("Model Loaded", 1);
            
            LOG_DEBUG_LuminaChat("About to call GetOrCreateContextInfo");
            try {
                // Get main model context size from settings (architectural fix)
                int32_t main_context_size = settings_manager->GetInt("Models", "main_context_size", 4096);
                
                LOG_DEBUG_LuminaChat("Calling GetOrCreateContextInfo with context_id=" + current_context_id + 
                                   ", model_id=" + current_model_id + 
                                   ", context_size=" + std::to_string(main_context_size));
                auto* context_info = llama_manager->GetOrCreateContextInfo(current_context_id, current_model_id, main_context_size);
                LOG_DEBUG_LuminaChat("GetOrCreateContextInfo call completed");
                
                if (context_info) {
                    LOG_DEBUG_LuminaChat("GetOrCreateContextInfo returned valid context");
                    // Apply template settings from UI (identity directive and system prompt)
                    ApplyTemplateSettingsToContext(context_info, current_context_id);
                    AddLogMessage("Model loaded successfully");
                    
                    // Now initialize plugins after successful model loading
                    InitializePlugins();
                    
                    AddChatMessage("System", "Model loaded and ready for conversation!", wxColour(0, 150, 0));
                } else {
                    LOG_ERROR_LuminaChat("GetOrCreateContextInfo returned null");
                    AddLogMessage("ERROR: Failed to create context after model loading");
                }
            } catch (const std::exception& inner_e) {
                LOG_ERROR_LuminaChat(wxString::Format("Exception in GetOrCreateContextInfo: %s", inner_e.what()).ToStdString());
                throw; // Re-throw to be caught by outer handler
            }
        } else {
            LOG_ERROR_LuminaChat("LlamaManager::LoadModel returned false - failure");
            AddLogMessage("ERROR: Failed to load model");
            AddChatMessage("System", "Failed to load model. Check the file path and try again.", wxColour(150, 50, 50));
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR_LuminaChat(wxString::Format("Exception caught in OnLoadModel: %s", e.what()).ToStdString());
        AddLogMessage(wxString::Format("Error loading model: %s", e.what()).ToStdString());
        AddChatMessage("System", wxString::Format("Failed to load model: %s", e.what()).ToStdString(), wxColour(150, 50, 50));
        model_loaded = false;
    }
    
    // Always reset loading state and update UI
    LOG_DEBUG_LuminaChat("Resetting loading state and updating UI");
    model_loading = false;
    model_progress->SetValue(0);
    UpdateUI();
}

void LuminaChatFrame::OnLoadSummaryModel(wxCommandEvent& event) {
    // Handle browse button for summary model
    if (event.GetId() == ID_BrowseSummaryModel) {
        wxFileDialog openFileDialog(this, "Select Summary GGUF Model File", "", "",
                                   "GGUF Model files (*.gguf)|*.gguf|All files (*.*)|*.*",
                                   wxFD_OPEN | wxFD_FILE_MUST_EXIST);
        
        if (openFileDialog.ShowModal() == wxID_OK) {
            summary_model_path_text->SetValue(openFileDialog.GetPath());
            
            // Save the path immediately when selected
            if (settings_manager) {
                settings_manager->SetString("Models", "summary_model_path", openFileDialog.GetPath().ToStdString());
                settings_manager->SaveSettings();
                AddLogMessage("Summary model path updated: " + openFileDialog.GetPath().ToStdString());
            }
        }
        return;
    }
    
    // Note: Direct model loading is now handled by SummarizationPlugin
    AddLogMessage("Summary model loading is managed by SummarizationPlugin - use the configuration above and restart the plugin");
}

void LuminaChatFrame::OnLoadEmoTagModel(wxCommandEvent& event) {
    // Handle browse button for emotion model
    if (event.GetId() == ID_BrowseEmoTagModel) {
        wxFileDialog openFileDialog(this, "Select Emotion Analysis GGUF Model File", "", "",
                                   "GGUF Model files (*.gguf)|*.gguf|All files (*.*)|*.*",
                                   wxFD_OPEN | wxFD_FILE_MUST_EXIST);
        
        if (openFileDialog.ShowModal() == wxID_OK) {
            emotag_model_path_text->SetValue(openFileDialog.GetPath());
            
            // Save the path immediately when selected
            if (settings_manager) {
                settings_manager->SetString("Models", "emotion_model_path", openFileDialog.GetPath().ToStdString());
                settings_manager->SaveSettings();
                AddLogMessage("Emotion model path updated: " + openFileDialog.GetPath().ToStdString());
            }
        }
        return;
    }
    
    // Note: Direct model loading is now handled by EmoTagPlugin
    AddLogMessage("Emotion model loading is managed by EmoTagPlugin - use the configuration above and restart the plugin");
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
        AddLogMessage("Please enter a Discord bot token first");
        AddChatMessage("System", "Please enter a Discord bot token first", wxColour(150, 100, 50));
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
        AddChatMessage("System", wxString::Format("Failed to connect to Discord: %s", e.what()).ToStdString(), wxColour(150, 50, 50));
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
        
        // Update plugin debug info every 10 timer ticks (about once per second if timer is 100ms)
        static int debug_update_counter = 0;
        debug_update_counter++;
        if (debug_update_counter >= 10) {
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
    
    // Show the actual template that will be processed by llama.cpp
    display_stream << "TEMPLATE CONTENT (sent to llama.cpp Jinja2 interpreter):\n";
    display_stream << "--------------------------------------------------------\n";
    display_stream << template_content;
    
    // Add footer with analysis
    display_stream << "\n\n=== TEMPLATE ANALYSIS ===\n";
    display_stream << "Contains 'for msg in messages': " << (template_content.find("for msg in messages") != std::string::npos ? "YES" : "NO") << "\n";
    display_stream << "Contains '{{': " << (template_content.find("{{") != std::string::npos ? "YES" : "NO") << "\n";
    display_stream << "Contains 'bos_token': " << (template_content.find("bos_token") != std::string::npos ? "YES" : "NO") << "\n";
    display_stream << "Contains 'assistant<|end_header_id': " << (template_content.find("assistant<|end_header_id") != std::string::npos ? "YES" : "NO") << "\n";
    display_stream << "=========================\n";
    
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
    if (summary_status_text) {
        summary_status_text->SetLabel("SummarizationPlugin: " + status);
        if (color.IsOk()) {
            summary_status_text->SetForegroundColour(color);
        }
        summary_status_text->GetParent()->Layout();  // Refresh the layout
    }
}

void LuminaChatFrame::UpdateEmoTagPluginStatus(const std::string& status, const wxColour& color) {
    if (emotag_status_text) {
        emotag_status_text->SetLabel("EmoTagPlugin: " + status);
        if (color.IsOk()) {
            emotag_status_text->SetForegroundColour(color);
        }
        emotag_status_text->GetParent()->Layout();  // Refresh the layout
    }
}

void LuminaChatFrame::UpdateSummaryPluginDebugInfo() {
    if (!summarization_plugin) return;
    
    // Update log output
    if (summary_log_output_text) {
        auto logs = summarization_plugin->GetLogHistory();
        wxString log_content;
        
        // Show last 10 log entries (to fit in 5-line textbox)
        size_t start_idx = logs.size() > 10 ? logs.size() - 10 : 0;
        for (size_t i = start_idx; i < logs.size(); ++i) {
            if (!log_content.IsEmpty()) log_content += "\n";
            log_content += logs[i];
        }
        
        if (log_content.IsEmpty()) {
            log_content = "No log entries yet...";
        }
        
        // Only update if content has changed
        if (summary_log_output_text->GetValue() != log_content) {
            summary_log_output_text->SetValue(log_content);
        }
    }
    
    // Update last generation
    if (summary_last_generation_text) {
        auto gen_info = summarization_plugin->GetLastGeneration();
        wxString gen_content;
        
        if (gen_info.has_generation) {
            gen_content = wxString::Format(
                "Context: %s\nTime: %s\n\nINPUT:\n%s\n\nOUTPUT:\n%s",
                gen_info.context_id,
                gen_info.timestamp,
                gen_info.input,
                gen_info.output
            );
        } else {
            gen_content = "No generation data available yet...";
        }
        
        // Only update if content has changed to preserve scroll position
        if (summary_last_generation_text->GetValue() != gen_content) {
            // Store current scroll position
            long insertion_point = summary_last_generation_text->GetInsertionPoint();
            
            summary_last_generation_text->SetValue(gen_content);
            
            // Restore scroll position if the content got longer (new generation)
            // But if it's the same length or shorter, keep at top for readability
            if (gen_content.length() > summary_last_generation_text->GetValue().length() && insertion_point > 0) {
                summary_last_generation_text->SetInsertionPoint(insertion_point);
            }
        }
    }
}

void LuminaChatFrame::UpdateEmoTagPluginDebugInfo() {
    if (!emotag_plugin) return;
    
    // Update log output
    if (emotag_log_output_text) {
        auto logs = emotag_plugin->GetLogHistory();
        wxString log_content;
        
        // Show last 10 log entries (to fit in 5-line textbox)
        size_t start_idx = logs.size() > 10 ? logs.size() - 10 : 0;
        for (size_t i = start_idx; i < logs.size(); ++i) {
            if (!log_content.IsEmpty()) log_content += "\n";
            log_content += logs[i];
        }
        
        if (log_content.IsEmpty()) {
            log_content = "No log entries yet...";
        }
        
        // Only update if content has changed
        if (emotag_log_output_text->GetValue() != log_content) {
            emotag_log_output_text->SetValue(log_content);
        }
    }
    
    // Update last generation
    if (emotag_last_generation_text) {
        auto gen_info = emotag_plugin->GetLastGeneration();
        wxString gen_content;
        
        if (gen_info.has_generation) {
            gen_content = wxString::Format(
                "Context: %s\nTime: %s\n\nINPUT:\n%s\n\nOUTPUT:\n%s",
                gen_info.context_id,
                gen_info.timestamp,
                gen_info.input,
                gen_info.output
            );
        } else {
            gen_content = "No generation data available yet...";
        }
        
        // Only update if content has changed to preserve scroll position
        if (emotag_last_generation_text->GetValue() != gen_content) {
            // Store current scroll position
            long insertion_point = emotag_last_generation_text->GetInsertionPoint();
            
            emotag_last_generation_text->SetValue(gen_content);
            
            // Restore scroll position if the content got longer (new generation)
            // But if it's the same length or shorter, keep at top for readability
            if (gen_content.length() > emotag_last_generation_text->GetValue().length() && insertion_point > 0) {
                emotag_last_generation_text->SetInsertionPoint(insertion_point);
            }
        }
    }
}

// Helper methods for initialization
std::string LuminaChatFrame::GetEnvironmentDescriptionFromUI() const {
    if (environment_description_text) {
        return environment_description_text->GetValue().ToStdString();
    }
    return "";
}

std::string LuminaChatFrame::GetIdentityDirectiveFromUI() const {
    if (identity_directive_text) {
        return identity_directive_text->GetValue().ToStdString();
    }
    return "";
}

std::string LuminaChatFrame::GetSystemPromptFromUI() const {
    if (system_prompt_text) {
        return system_prompt_text->GetValue().ToStdString();
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
        // Load model settings
        std::string model_path = settings_manager->GetString("Models", "main_model_path", "");
        if (!model_path.empty() && model_path_text) {
            model_path_text->SetValue(model_path);
            AddLogMessage("Loaded model path: " + model_path);
        }
        
        // Load model configuration sliders
        if (context_size_slider) {
            int context_size = settings_manager->GetInt("Models", "main_context_size", 4096);
            context_size_slider->SetValue(context_size);
            AddLogMessage("Loaded context size: " + std::to_string(context_size));
        }
        
        if (gpu_layers_slider) {
            int gpu_layers = settings_manager->GetInt("Models", "main_gpu_layers", 999);
            gpu_layers_slider->SetValue(gpu_layers);
            AddLogMessage("Loaded GPU layers: " + std::to_string(gpu_layers));
        }
        
        // Load template configuration
        if (environment_description_text) {
            std::string env_desc = settings_manager->GetString("Templates", "environment_description", "");
            environment_description_text->SetValue(env_desc);
            if (!env_desc.empty()) {
                AddLogMessage("Loaded environment description from settings");
            }
        }
        
        if (identity_directive_text) {
            std::string identity = settings_manager->GetString("Templates", "identity_directive", "");
            identity_directive_text->SetValue(identity);
            if (!identity.empty()) {
                AddLogMessage("Loaded identity directive from settings");
            }
        }
        
        if (system_prompt_text) {
            std::string system_prompt = settings_manager->GetString("Templates", "system_prompt", "");
            system_prompt_text->SetValue(system_prompt);
            if (!system_prompt.empty()) {
                AddLogMessage("Loaded system prompt from settings");
            }
        }
        
        // Load summary model settings
        if (summary_model_path_text) {
            std::string summary_model_path = settings_manager->GetString("Models", "summary_model_path", "");
            summary_model_path_text->SetValue(summary_model_path);
            if (!summary_model_path.empty()) {
                AddLogMessage("Loaded summary model path: " + summary_model_path);
            }
        }
        
        if (summary_system_prompt_text) {
            std::string default_summary_prompt = 
                "You are a helpful AI assistant that creates concise summaries of conversations. "
                "When given a conversation history, provide a clear and informative summary that captures "
                "the key points, decisions, and context. Focus on preserving important information while "
                "being concise. Format your summary in a structured way with bullet points when appropriate.";
                
            std::string summary_prompt = settings_manager->GetString("Summary", "system_prompt", default_summary_prompt);
            summary_system_prompt_text->SetValue(summary_prompt);
            AddLogMessage("Loaded summary system prompt from settings");
        }
        
        // Load emotion model settings
        if (emotag_model_path_text) {
            std::string emotag_model_path = settings_manager->GetString("Models", "emotion_model_path", "");
            emotag_model_path_text->SetValue(emotag_model_path);
            if (!emotag_model_path.empty()) {
                AddLogMessage("Loaded emotion model path: " + emotag_model_path);
            }
        }
        
        if (emotag_system_prompt_text) {
            std::string default_emotion_prompt = 
                "You are an emotional state analyzer. When given AI assistant responses, analyze the emotional tone, "
                "mood, and psychological state conveyed in the text. Provide a brief emotional overview that captures "
                "the assistant's apparent emotional state, confidence level, and overall demeanor. "
                "Focus on identifying patterns like: confident, uncertain, empathetic, analytical, cheerful, "
                "cautious, enthusiastic, or reserved. Keep your analysis concise and actionable.";
                
            std::string emotion_prompt = settings_manager->GetString("Emotion", "system_prompt", default_emotion_prompt);
            emotag_system_prompt_text->SetValue(emotion_prompt);
            AddLogMessage("Loaded emotion system prompt from settings");
        }
        
        // Load EmoTag configuration parameters
        if (emotag_window_size_slider && emotag_window_size_text) {
            int window_size = settings_manager->GetInt("Emotion", "analysis_window_size", 3);
            emotag_window_size_slider->SetValue(window_size);
            emotag_window_size_text->SetLabel(wxString::Format("%d", window_size));
            AddLogMessage("Loaded emotion analysis window size: " + std::to_string(window_size));
        }
        
        if (emotag_include_user_checkbox) {
            bool include_user = settings_manager->GetBool("Emotion", "include_user_messages", false);
            emotag_include_user_checkbox->SetValue(include_user);
            AddLogMessage("Loaded emotion include user messages setting: " + std::string(include_user ? "enabled" : "disabled"));
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
        // Save model settings
        if (model_path_text) {
            std::string model_path = model_path_text->GetValue().ToStdString();
            settings_manager->SetString("Models", "main_model_path", model_path);
        }
        
        if (context_size_slider) {
            int context_size = context_size_slider->GetValue();
            settings_manager->SetInt("Models", "main_context_size", context_size);
        }
        
        if (gpu_layers_slider) {
            int gpu_layers = gpu_layers_slider->GetValue();
            settings_manager->SetInt("Models", "main_gpu_layers", gpu_layers);
        }
        
        // Save template configuration
        if (environment_description_text) {
            std::string env_desc = environment_description_text->GetValue().ToStdString();
            settings_manager->SetString("Templates", "environment_description", env_desc);
        }
        
        if (identity_directive_text) {
            std::string identity = identity_directive_text->GetValue().ToStdString();
            settings_manager->SetString("Templates", "identity_directive", identity);
        }
        
        if (system_prompt_text) {
            std::string system_prompt = system_prompt_text->GetValue().ToStdString();
            settings_manager->SetString("Templates", "system_prompt", system_prompt);
        }
        
        // Save summary model settings
        if (summary_model_path_text) {
            std::string summary_model_path = summary_model_path_text->GetValue().ToStdString();
            settings_manager->SetString("Models", "summary_model_path", summary_model_path);
        }
        
        if (summary_system_prompt_text) {
            std::string summary_prompt = summary_system_prompt_text->GetValue().ToStdString();
            settings_manager->SetString("Summary", "system_prompt", summary_prompt);
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


