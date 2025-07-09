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
#include "../Logger.hpp"
#include "../ErrorHandling.hpp"
#include "../SettingsManager.hpp"
#include "../Sanitizer.hpp"
#include "../DiscordManager.hpp"
#include "../Plugins/ContextPruningPlugin.hpp"
#include "../TokenCache.hpp"
#include "../ModelInfo.hpp"
#include "../ChatTemplateManager.hpp"
#include "../Context/ContextInfo.hpp"
#include "../LlamaManager.hpp"
#include "../Orchestrator.hpp"
#include "../Plugins/SummarizationPlugin.hpp"
#include "../Plugins/EmoTagPlugin.hpp"
#include "../Plugins/UI/SummarizationPlugin_UI.hpp"
#include "../Plugins/UI/EmoTagPlugin_UI.hpp"
#include "../Plugins/UI/ContextPruningPlugin_UI.hpp"
#include "../UI/Settings_UI.hpp"
#include "../UI/OuterVoice_UI.hpp"
#include "../UI/InnerVoice_UI.hpp"

#include <memory>
#include <thread>
#include <atomic>
#include <sstream>
#include <functional>

// Include centralized constants and patterns
#include "LuminaChat_Constants.hpp"
#include "../ErrorHandling.hpp"

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

    // UI Settings persistence
    void LoadUISettings();
    void SaveUISettings();

    // Callback handlers (registered with lower-level components)
    void OnLogMessage(const std::string& log_message);

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

// Use centralized event IDs from LuminaChat_Constants.hpp
using namespace LuminaChatEventIDs;

// Include broken-out implementation files (header-only)
#include "LuminaChat_UIUtilities.hpp"
#include "LuminaChat_StreamingDisplay.hpp"
#include "LuminaChat_UIHelpers.hpp"
#include "LuminaChat_Settings.hpp"
#include "LuminaChat_UICreation.hpp"
#include "LuminaChat_MessageGeneration.hpp"

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
    : wxFrame(nullptr, wxID_ANY, LuminaChatConstants::APP_NAME + wxString(" - ") + LuminaChatConstants::APP_DESCRIPTION, 
              wxDefaultPosition, wxSize(LuminaChatConstants::DEFAULT_WINDOW_WIDTH, LuminaChatConstants::DEFAULT_WINDOW_HEIGHT)) {
    
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
    if (running) return;
    
    try {
        LOG_LuminaChat("Starting LuminaChat system...");
        
        // === STEP 1: Register ErrorHandler UI callbacks for unified error handling ===
        // This connects the global ErrorHandler to our UI components
        LuminaChat::ErrorHandler::GetInstance().RegisterErrorCallback([this](const std::string& message, const std::string& context) {
            // Use CallAfter to ensure UI updates happen on the main thread
            CallAfter([this, message, context]() {
                std::string display_message = context.empty() ? message : context + ": " + message;
                AddChatMessage("System", display_message, LuminaChatColors::ERROR_RED);
            });
        });
        
        LuminaChat::ErrorHandler::GetInstance().RegisterWarningCallback([this](const std::string& message, const std::string& context) {
            CallAfter([this, message, context]() {
                std::string display_message = context.empty() ? message : context + ": " + message;
                AddChatMessage("System", display_message, LuminaChatColors::WARNING_ORANGE);
            });
        });
        
        LuminaChat::ErrorHandler::GetInstance().RegisterSuccessCallback([this](const std::string& message, const std::string& context) {
            CallAfter([this, message, context]() {
                std::string display_message = context.empty() ? message : context + ": " + message;
                AddChatMessage("System", display_message, LuminaChatColors::SUCCESS_GREEN);
            });
        });
        
        LuminaChat::ErrorHandler::GetInstance().RegisterInfoCallback([this](const std::string& message, const std::string& context) {
            CallAfter([this, message, context]() {
                std::string display_message = context.empty() ? message : context + ": " + message;
                AddChatMessage("System", display_message, LuminaChatColors::INFO_BLUE);
            });
        });
        
        LOG_LuminaChat("ErrorHandler UI callbacks registered for unified error handling");
        
        // === STEP 2: Initialize core components in dependency order ===
        // Register Logger UI callback first to ensure logs are displayed
        GetLogger().RegisterOutputCallback([this](std::string_view log_message) {
            // Convert string_view to string to ensure data lifetime across threads
            std::string log_str(log_message);
            // Use CallAfter to ensure UI updates happen on the main thread
            CallAfter([this, log_str]() {
                OnLogMessage(log_str);
            });
        });
        LOG_LuminaChat("Logger UI callback registered successfully");
        
        // Initialize Settings Manager with proper settings file path
        settings_manager = std::make_unique<SettingsManager>();
        if (!settings_manager->LoadSettings("")) {  // Empty string uses default path (luminachat.ini)
            HandleWarning("Failed to load luminachat.ini, using defaults", "Settings");
        } else {
            LOG_LuminaChat("Settings loaded successfully");
        }
        
        // Initialize Sanitizer
        sanitizer = std::make_unique<Sanitizer>();
        LOG_LuminaChat("Sanitizer initialized");
        
        // Initialize Discord Manager
        discord_manager = std::make_unique<DiscordManager>();
        LOG_LuminaChat("Discord Manager initialized");
        
        // Initialize Llama Manager with SettingsManager dependency
        llama_manager = std::make_unique<LlamaManager>(settings_manager.get());
        if (!llama_manager->Initialize()) {
            HandleError("Failed to initialize LlamaManager", "LlamaManager");
            return;
        }
        LOG_LuminaChat("Llama Manager initialized successfully");
        
        // Initialize Orchestrator (depends on LlamaManager)
        orchestrator = std::make_unique<Orchestrator>(llama_manager.get());
        if (!orchestrator->Initialize()) {
            LOG_ERROR_LuminaChat("Failed to initialize Orchestrator");
            HandleError("Failed to initialize Orchestrator", "Initialization", true);
            return;
        }
        LOG_LuminaChat("Orchestrator initialized");
        
        // === STEP 3: Set up inter-component dependencies ===
        // Register Discord response callback with orchestrator
        orchestrator->RegisterDiscordResponseCallback([this](const DiscordChannelResponse& response) {
            if (discord_manager && discord_manager->IsConnected()) {
                bool sent = discord_manager->SendAIResponse(response.target_channel, 
                                                          response.response_content,
                                                          response.internal_reasoning,
                                                          response.context_current,
                                                          response.context_maximum);
                if (sent) {
                    LOG_LuminaChat("Discord response sent successfully to channel " + response.target_channel +
                                 " (context: " + std::to_string(response.context_current) + "/" + 
                                 std::to_string(response.context_maximum) + ")");
                } else {
                    LOG_ERROR_LuminaChat("Failed to send Discord response to channel " + response.target_channel);
                }
            } else {
                LOG_ERROR_LuminaChat("Cannot send Discord response - manager not available or not connected");
            }
        });
        LOG_LuminaChat("Discord response callback registered");
        
        // Set up Discord manager dependencies
        discord_manager->SetLlamaManager(llama_manager.get());
        discord_manager->SetOrchestrator(orchestrator.get());
        LOG_LuminaChat("Discord Manager dependencies configured");
        
        // Note: SummarizationPlugin initialization is now deferred until model loading
        LOG_LuminaChat("Plugin initialization will be handled when user loads a model");
        
        // Initialize Outer Voice UI with dependencies
        if (outer_voice_ui) {
            outer_voice_ui->SetSettingsManager(settings_manager.get());
            outer_voice_ui->SetLlamaManager(llama_manager.get());
            outer_voice_ui->SetCallbacks(
                [this](ContextInfo* ctx, const std::string& id) { ApplyTemplateSettingsToContext(ctx, id); }
            );
            LOG_LuminaChat("Outer Voice UI dependencies configured");
        }
        
        // Initialize General Settings UI with dependencies
        if (settings_ui) {
            settings_ui->SetSettingsManager(settings_manager.get());
            settings_ui->SetCallbacks(
                [this]() { OnLoadAllModelsFromVoiceSettings(); },
                [this](ContextInfo* ctx, const std::string& id) { ApplyTemplateSettingsToContext(ctx, id); }
            );
            LOG_LuminaChat("General Settings UI dependencies configured");
        }
        
        // Initialize Inner Voice UI with dependencies
        if (inner_voice_ui) {
            inner_voice_ui->SetSettingsManager(settings_manager.get());
            inner_voice_ui->SetLlamaManager(llama_manager.get());
            inner_voice_ui->SetCallbacks(
                [this](ContextInfo* ctx, const std::string& id) { ApplyTemplateSettingsToContext(ctx, id); }
            );
            LOG_LuminaChat("Inner Voice UI dependencies configured");
        }
        
        // Register callbacks (higher components register with lower)
        RegisterCallbacks();
        LOG_LuminaChat("Callback dependencies registered");
        
        // Load UI settings from configuration
        LoadUISettings();
        
        // Note: Model loading is now unified in the General Settings tab
        LOG_LuminaChat("Ready for model loading - configure individual voice models in their respective tabs, then use General Settings to load all models");
        
        running = true;
        system_timer->Start(LuminaChatConstants::TIMER_INTERVAL_MS);
        
        LOG_LuminaChat("LuminaChat started successfully - configure voice models in individual tabs, then load via General Settings");
        SetStatusText("System Ready - Configure voice models in individual tabs", 0);
        UpdateUI();
        
        // Set initial plugin status
        UpdateSummaryPluginStatus("Waiting for model loading", LuminaChatColors::NEUTRAL_GRAY);
        UpdateEmoTagPluginStatus("Waiting for model loading", LuminaChatColors::NEUTRAL_GRAY);
        
    } catch (const std::exception& e) {
        wxString error_msg = wxString::Format("Failed to start LuminaChat: %s", e.what());
        LOG_ERROR_LuminaChat(error_msg.ToStdString());
        HandleError("Startup failed: " + std::string(e.what()), "System Initialization", true);
        SetStatusText("Startup Failed", 0);
    }
}

void LuminaChatFrame::Stop() {
    if (!running) return;
    
    LOG_LuminaChat("Shutting down LuminaChat...");
    
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
        LOG_LuminaChat("EmoTagPlugin stopped");
    }
    if (summarization_plugin) {
        summarization_plugin->Shutdown();
        summarization_plugin.reset();
        LOG_LuminaChat("SummarizationPlugin stopped");
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
        LOG_WARNING_LuminaChat("Model loading still in progress during shutdown");
    }
    
    orchestrator.reset();
    llama_manager.reset();
    discord_manager.reset();
    sanitizer.reset();
    settings_manager.reset();
    
    LOG_LuminaChat("LuminaChat stopped cleanly");
    SetStatusText("Stopped", 0);
}

// Callback implementations
void LuminaChatFrame::OnLogMessage(const std::string& log_message) {
    // This method receives log messages from the global Logger and displays them in the UI
    // Logger already provides fully formatted log entries with timestamps, so we just display them directly
    
    // Guard against empty messages
    if (log_message.empty()) {
        return;
    }
    
    // Convert string to wxString safely
    wxString log_entry = wxString::FromUTF8(log_message) + "\n";
    
    // Safety check for conversion failure
    if (log_entry.IsEmpty() && !log_message.empty()) {
        // UTF-8 conversion failed, try default conversion
        log_entry = wxString(log_message) + "\n";
    }
    
    logs_display->SetInsertionPointEnd();
    logs_display->WriteText(log_entry);
    logs_display->SetInsertionPointEnd();
}

void LuminaChatFrame::OnLogLevelChanged(wxCommandEvent& event) {
    int selection = log_level_choice->GetSelection();
    Logger::LogLevel new_level;
    
    switch (selection) {
        case 0: // DEBUG
            new_level = Logger::LogLevel::DBG;
            LOG_LuminaChat("Log level changed to DEBUG");
            break;
        case 1: // INFO
            new_level = Logger::LogLevel::INF;
            LOG_LuminaChat("Log level changed to INFO");
            break;
        case 2: // WARNING
            new_level = Logger::LogLevel::WRN;
            LOG_LuminaChat("Log level changed to WARNING");
            break;
        case 3: // ERROR
            new_level = Logger::LogLevel::ERR;
            LOG_LuminaChat("Log level changed to ERROR");
            break;
        default:
            new_level = Logger::LogLevel::INF;
            LOG_LuminaChat("Unknown log level selected, defaulting to INFO");
            break;
    }
    
    // Set level on the GLOBAL Logger instance that the macros use
    GetLogger().SetLogLevel(new_level);
}

// === UI Helper Methods ===

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
                
                SafeExecute([&]() {
                    settings_ui->UpdateModelProgress(percentage);
                    
                    // Update status bar with progress information
                    if (percentage < 100) {
                        SetStatusText(wxString::Format("Loading model: %d%%", percentage), 1);
                    } else {
                        SetStatusText("Model Loaded", 1);
                    }
                    
                    // Add log message for progress tracking
                    if (percentage % 10 == 0 || percentage >= 95) {
                        LOG_LuminaChat(wxString::Format("Model loading progress: %d%% (%s)", percentage, model_id).ToStdString());
                    }
                }, "update model loading progress", "UI");
            });
        });
        LOG_LuminaChat("Progress callback registered with LlamaManager");
    }
    
    // Note: UI now uses direct async streaming instead of Orchestrator callbacks
    // This eliminates the synchronous callback tech debt
}

void LuminaChatFrame::LoadDefaultModels() {
    if (!llama_manager || !settings_manager) {
        LOG_ERROR_LuminaChat("Cannot load models - components not initialized");
        return;
    }
    
    LOG_LuminaChat("Loading default models from settings...");
    
    // Load outer model
    if (!llama_manager->LoadModelFromSettings("outer_model", "outer")) {
        LOG_WARNING_LuminaChat("Failed to load outer model from settings");
    } else {
        LOG_LuminaChat("Outer model loaded successfully");
        
        // Get outer model context size from settings
        int32_t outer_context_size = settings_manager->GetInt("Models", "outer_context_size", 4096);
        
        // Create default context with outer model
        auto* context = llama_manager->GetOrCreateContextInfo("outer_context", "outer_model", outer_context_size);
        if (context) {
            LOG_LuminaChat("Outer context created successfully with context size: " + std::to_string(outer_context_size));
            // Apply template settings from UI (identity directive and system prompt)
            ApplyTemplateSettingsToContext(context, "outer_context");
        }
    }
    
    // Note: Summary model loading is now handled by SummarizationPlugin during its initialization
    LOG_LuminaChat("Summary model initialization will be handled by SummarizationPlugin");
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
        LOG_ERROR_LuminaChat("Cannot initialize plugins - prerequisites not met");
        return;
    }
    
    LOG_LuminaChat("Initializing plugins after successful model loading...");
    
    SafeExecute([&]() {
        InitializeSummarizationPlugin();
        InitializeEmoTagPlugin();
        InitializeContextPruningPlugin();
        
        LOG_LuminaChat("All plugins initialized successfully");
    }, "initialize plugins", "Plugin Initialization");
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
        LOG_ERROR_LuminaChat("Failed to initialize SummarizationPlugin");
        UpdateSummaryPluginStatus("Initialization failed", LuminaChatColors::ERROR_RED);
    } else {
        LOG_LuminaChat("SummarizationPlugin initialized successfully");
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
        LOG_ERROR_LuminaChat("Failed to initialize EmoTagPlugin");
        UpdateEmoTagPluginStatus("Initialization failed", LuminaChatColors::ERROR_RED);
        return;
    }
    
    // Register the plugin with the Orchestrator for coordination
    orchestrator->RegisterEmoTagPlugin(emotag_plugin.get());
        
    // Configure plugin with default settings (UI manager will load actual settings later)
    emotag_plugin->SetAnalysisWindow(3); // Default window size
    LOG_LuminaChat("Configured EmoTagPlugin with default analysis window size: 3");
    
    LOG_LuminaChat("EmoTagPlugin initialized successfully");
    UpdateEmoTagPluginStatus("Initialized and ready", LuminaChatColors::SUCCESS_GREEN);
}

void LuminaChatFrame::InitializeContextPruningPlugin() {
    context_pruning_plugin = std::make_unique<LuminaChat::ContextPruningPlugin>(orchestrator.get());
    
    if (!context_pruning_plugin->Initialize()) {
        LOG_ERROR_LuminaChat("Failed to initialize ContextPruningPlugin");
    } else {
        LOG_LuminaChat("ContextPruningPlugin initialized successfully");
        
        // Configure more aggressive thresholds for better context management
        context_pruning_plugin->SetPruningThreshold(0.65f);  // Trigger at 65% instead of 75%
        context_pruning_plugin->SetTargetUsage(0.35f);       // Reduce to 35% instead of 40%
        context_pruning_plugin->SetEmergencyThreshold(0.85f); // Emergency at 85% instead of 90%
        
        // Register with Orchestrator for context monitoring
        orchestrator->RegisterContextPruningPlugin(context_pruning_plugin.get());
        LOG_LuminaChat("ContextPruningPlugin registered with Orchestrator for monitoring");
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
        HandleError("Cannot send message: system not ready", "Message Processing", true);
        return;
    }
    
    // Validate input
    wxString input = chat_input->GetValue().Trim();
    if (input.IsEmpty()) {
        return;
    }
    
    SafeExecute([this, input]() {
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
                TryExecute([&]() {
                    auto* outer_model = llama_manager->GetModelInfo("outer_model");
                    auto* inner_model = llama_manager->GetModelInfo("inner_model");
                    diagnostics += "- Outer model exists: " + std::string(outer_model ? "true" : "false") + "\n";
                    diagnostics += "- Inner model exists: " + std::string(inner_model ? "true" : "false") + "\n";
                }, "check model status", "Model Status");
                
                if (!TryExecute([&]() {
                    auto* outer_model = llama_manager->GetModelInfo("outer_model");
                    auto* inner_model = llama_manager->GetModelInfo("inner_model");
                    diagnostics += "- Outer model exists: " + std::string(outer_model ? "true" : "false") + "\n";
                    diagnostics += "- Inner model exists: " + std::string(inner_model ? "true" : "false") + "\n";
                }, "check model status", "Model Status")) {
                    diagnostics += "- Unable to check model status\n";
                }
            }
            
            LOG_ERROR_LuminaChat(diagnostics);
            HandleError("Required contexts not found: " + missing + 
                       "\n\nPlease ensure you have:\n"
                       "1. Configured model paths in Outer Voice and Inner Voice tabs\n"
                       "2. Clicked 'Load All Models' in General Settings tab\n"
                       "3. Waited for models to finish loading\n\n"
                       "Check the System Logs tab for detailed diagnostics.", "Message Processing", true);
            return;
        }
        
        // Apply template settings to both contexts
        ApplyTemplateSettingsToContext(inner_context, "inner_context");
        ApplyTemplateSettingsToContext(outer_context, "outer_context");
        LOG_LuminaChat("Processing message with two-stage reasoning: inner -> outer");
        
        // Start the two-stage reasoning process
        ExecuteTwoStageReasoning(inner_context, outer_context, input.ToStdString());
        
    }, "process message", "Message Processing");
}

void LuminaChatFrame::OnLoadAllModelsFromVoiceSettings() {
    if (!settings_ui || !llama_manager || !settings_manager || !outer_voice_ui || !inner_voice_ui) {
        LOG_ERROR_LuminaChat("Required UI components not initialized");
        return;
    }
    
    if (!running) {
        LOG_ERROR_LuminaChat("Cannot load models: system not running");
        return;
    }
    
    if (model_loading) {
        LOG_WARNING_LuminaChat("Model loading already in progress");
        return;
    }
    
    // Set loading state immediately
    model_loading = true;
    settings_ui->SetModelLoadingState(true);
    settings_ui->UpdateModelProgress(0);
    SetStatusText("Starting model loading from voice configurations...", 1);
    
    LOG_LuminaChat("Starting model loading from individual voice configurations...");
    
    // Get model configurations from individual voice tabs
    std::string outer_model_path = outer_voice_ui->GetModelPath();
    int outer_context_size = outer_voice_ui->GetContextSize();
    int outer_gpu_layers = outer_voice_ui->GetGPULayers();
    
    std::string inner_model_path = inner_voice_ui->GetModelPath();
    int inner_context_size = inner_voice_ui->GetContextSize();
    int inner_gpu_layers = inner_voice_ui->GetGPULayers();
    
    // Validate that at least one model is configured
    if (outer_model_path.empty() && inner_model_path.empty()) {
        LOG_WARNING_LuminaChat("No models configured in voice settings");
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
        
        SafeExecute([&]() {
            // Load outer voice model if configured
            if (!outer_model_path.empty()) {
                if (!model_loading.load()) return; // Check for cancellation
                
                LOG_LuminaChat("Loading outer voice model: " + outer_model_path);
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
                    LOG_LuminaChat("Outer voice model loaded successfully");
                    
                    // Create outer context
                    auto* outer_context = llama_manager->GetOrCreateContextInfo("outer_context", "outer_model", outer_context_size);
                    if (outer_context) {
                        ApplyTemplateSettingsToContext(outer_context, "outer_context");
                        LOG_LuminaChat("Outer voice context created and configured");
                    }
                } else {
                    error_message += "Failed to load outer voice model. ";
                }
            }
            
            // Load inner voice model if configured
            if (!inner_model_path.empty()) {
                if (!model_loading.load()) return; // Check for cancellation
                
                LOG_LuminaChat("Loading inner voice model: " + inner_model_path);
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
                    LOG_LuminaChat("Inner voice model loaded successfully");
                    
                    // Create inner context
                    auto* inner_context = llama_manager->GetOrCreateContextInfo("inner_context", "inner_model", inner_context_size);
                    if (inner_context) {
                        ApplyTemplateSettingsToContext(inner_context, "inner_context");
                        LOG_LuminaChat("Inner voice context created and configured");
                    }
                } else {
                    error_message += "Failed to load inner voice model. ";
                }
            }
        }, "load voice models", "Model Loading");
        
        // Final check - only update UI if we're still supposed to be loading
        if (!model_loading.load()) {
            return; // Loading was cancelled, don't update UI
        }
        
        // Update UI on the main thread using CallAfter
        CallAfter([this, outer_success, inner_success, error_message]() {
            SafeExecute([&]() {
                bool any_success = outer_success || inner_success;
                
                if (any_success) {
                    model_loaded = true;
                    settings_ui->UpdateModelProgress(100);
                    settings_ui->SetModelLoadedState(true);
                    SetStatusText("Models Loaded", 1);
                    
                    // Initialize plugins after successful model loading
                    InitializePlugins();
                    
                    if (outer_success && inner_success) {
                        HandleSuccess("Both voice models loaded and ready!", "Model Loading", true);
                        LOG_LuminaChat("All voice models loaded successfully");
                    } else if (outer_success) {
                        HandleSuccess("Outer voice model loaded and ready!", "Model Loading", true);
                        LOG_LuminaChat("Outer voice model loaded successfully");
                    } else if (inner_success) {
                        HandleSuccess("Inner voice model loaded and ready!", "Model Loading", true);
                        LOG_LuminaChat("Inner voice model loaded successfully");
                    }
                } else {
                    model_loaded = false;
                    if (!error_message.empty()) {
                        HandleError("Model loading failed: " + error_message, "Model Loading", true);
                    } else {
                        HandleError("Failed to load models. Check the file paths in voice settings.", "Model Loading", true);
                    }
                }
            }, "handle model loading completion", "Model Loading");
            
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
        LOG_LuminaChat("Disconnected from Discord");
        AddChatMessage("System", "Discord bot disconnected", wxColour(200, 100, 0));
        UpdateUI();
        return;
    }
    
    wxString token = discord_token_text->GetValue().Trim();
    if (token.IsEmpty()) {
        HandleError("Please enter a Discord bot token first", "Discord Setup", true);
        return;
    }
    
    if (!running) {
        LOG_ERROR_LuminaChat("Cannot connect to Discord: system not running");
        return;
    }
    
    SafeExecute([this, token]() {
        LOG_LuminaChat("Connecting to Discord...");
        connect_discord_button->Enable(false);
        connect_discord_button->SetLabel("Connecting...");
        
        // Update allowed channels before connecting
        UpdateDiscordAllowedChannels();
        
        // Set up message callback for Discord manager
        discord_manager->RegisterMessageCallback([this](const std::string& content, const std::string& channel_id, const std::string& username) {
            // Log the message
            LOG_LuminaChat(wxString::Format("Discord message from %s in channel %s: %s", 
                         username, channel_id, content.substr(0, 50)).ToStdString());
            
            // Route to orchestrator for processing
            if (orchestrator) {
                orchestrator->OnRawDiscordMessage(content, channel_id, username);
            } else {
                LOG_ERROR_LuminaChat("Orchestrator not available for Discord message processing");
            }
        });
        
        // Start connection in a separate thread since D++ is async
        std::thread([this, token]() {
            SafeExecute([this, token]() {
                if (discord_manager->Connect(token.ToStdString())) {
                    // Connection successful - update UI on main thread
                    CallAfter([this]() {
                        discord_connected = true;
                        LOG_LuminaChat("Connected to Discord successfully");
                        HandleSuccess("Discord bot connected and ready!", "Discord Connection");
                        
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
                        
                        HandleError("Discord connection failed: " + error_msg, "Discord Connection");
                        connect_discord_button->SetLabel("Connect Discord");
                        connect_discord_button->Enable(true);
                        discord_connected = false;
                        UpdateUI();
                    });
                }
            }, "Discord connection", "Discord Connection");
        }).detach();
        
    }, "Discord connection setup", "Discord Connection");
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
            LOG_LuminaChat("Chat context cleared - token count reset");
            UpdateContextStatus();  // Update status bar to reflect cleared context
        }
    }
    
    LOG_LuminaChat("Chat display and inner voice display cleared");
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

void LuminaChatFrame::OnStopGeneration(wxCommandEvent& event) {
    if (!running || !llama_manager) {
        return;
    }        LOG_LuminaChat("Stopping generation...");
        
        // Get the current context and stop generation
        auto* context = llama_manager->GetContextInfo(current_context_id);
        if (context && context->IsGenerating()) {
            context->StopGeneration();
            LOG_LuminaChat("Generation stop requested");
        } else {
            LOG_LuminaChat("No active generation to stop");
        }
}
