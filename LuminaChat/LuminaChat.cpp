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
// 1.  Minimalism & Performance: Deliver lean, efficient solutions; do not create or preserve unused helpers, wrappers, trivial accessors (setters/getters), or scaffolding.
// 2.  Redundancy Elimination: Remove unused, obsolete, and legacy code—including unneeded interfaces, includes, helper or accessor methods.
// 3.  Consistent Style: Adopt a uniform coding style and structure for clarity and maintainability.
// 4.  Documentation: Write concise comments that explain complex logic and key design decisions.
// 5.  Zero Magic & Strong Typing: Replace magic literals with named constants, enums, or constexpr; prefer scoped enums over raw ints.
// 6.  Function Boundaries: Define clear responsibilities; reduce overlap and avoid unnecessary layers of indirection.
// 7.  Core Preservation: Streamline code while safeguarding essential features; favor direct variable or object access/passing over extra abstractions (e.g., setters/getters).
// 8.  Const-Correctness & Immutability: Mark variables, parameters, and methods as const wherever possible.
// 9.  RAII & Resource Safety: Encapsulate resource acquisition/release in constructors/destructors or smart pointers.
// 10. Standard Library Preference: Favor STL algorithms and containers over custom loops and buffers for clarity and safety.
// 11. Cross-Platform Portability: Use fixed-width types and proper initialization to guarantee identical behavior everywhere.
// 12. Thread Safety: Define and document thread-safety contracts; protect shared state with mutexes, atomics, or thread-safe containers.
// 13. Smart Caching: Cache frequently used values to minimize allocations and improve performance.
// 14. Logical Consistency: Verify code flow to ensure coherent, error-free execution paths.
// 15. Continuous Refinement: Regularly refactor and confirm that updates preserve stable functionality.

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
#include <wx/listbox.h>
#include <wx/choice.h>
#include <string>
#include <cstdint>
#include <memory>
#include <atomic>
#include <thread>
#include <chrono>
#include <algorithm>
#include <vector>
#include <iterator>
#include "LlamaManager.hpp"
#include "DiscordManager.hpp" 
#include "SettingsManager.hpp"
#include "LogHandler.hpp"

// SummarizerConstants are defined in LlamaSummarizer.hpp

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
    VIEW_SUMMARY_SLOTS,
    MODEL_LOADED,
    RESPONSE_READY,
    PROGRESS_UPDATE,
    CONTEXT_MONITOR_TIMER
};

// Custom events for thread communication
wxDECLARE_EVENT(wxEVT_MODEL_LOADED, wxCommandEvent);
wxDECLARE_EVENT(wxEVT_RESPONSE_READY, wxCommandEvent);
wxDECLARE_EVENT(wxEVT_PROGRESS_UPDATE, wxCommandEvent);
wxDECLARE_EVENT(wxEVT_TOKEN_STREAM, wxCommandEvent);

wxDEFINE_EVENT(wxEVT_MODEL_LOADED, wxCommandEvent);
wxDEFINE_EVENT(wxEVT_RESPONSE_READY, wxCommandEvent);
wxDEFINE_EVENT(wxEVT_PROGRESS_UPDATE, wxCommandEvent);
wxDEFINE_EVENT(wxEVT_TOKEN_STREAM, wxCommandEvent);

// Forward declarations for callback functions
bool model_loading_progress_callback(float progress, void* user_data);
void llama_log_callback(ggml_log_level level, const char* message, void* user_data);

// UI Constants (Directive #5: Zero Magic & Strong Typing)
namespace UIConstants {
    constexpr int32_t CONTEXT_MONITOR_INTERVAL_MS = 2000;
    constexpr float MS_TO_SECONDS = 1000.0f;
    constexpr int32_t DISCORD_CONNECTION_DELAY_SEC = 5;
    constexpr int32_t MIN_MESSAGES_FOR_PRUNING = 10;
    constexpr int32_t WELCOME_MESSAGE_HEIGHT = 100;
    constexpr int32_t DIRECTIVE_MESSAGE_HEIGHT = 200;
    constexpr int32_t TEMPLATE_MESSAGE_HEIGHT = 200;
    constexpr int32_t SUMMARY_PROMPT_HEIGHT = 120;
    constexpr int32_t CHANNEL_LIST_HEIGHT = 80;
    constexpr int32_t PROGRESS_BAR_HEIGHT = 16;
    constexpr int32_t STATS_LABEL_HEIGHT = 60;
    constexpr int32_t SLIDER_WIDTH = 120;
    constexpr int32_t CONTEXT_LABEL_WIDTH = 220;
    constexpr int32_t STATS_LABEL_WIDTH_SMALL = 160;
    constexpr int32_t STATS_LABEL_WIDTH_LARGE = 180;
    constexpr int32_t MINIMUM_WINDOW_WIDTH = 600;
    constexpr int32_t MINIMUM_WINDOW_HEIGHT = 400;
    constexpr int32_t DEFAULT_WINDOW_WIDTH = 800;
    constexpr int32_t DEFAULT_WINDOW_HEIGHT = 600;
    constexpr int32_t SETTINGS_DIALOG_WIDTH = 700;
    constexpr int32_t SETTINGS_DIALOG_HEIGHT = 600;
}

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
    // RAII-safe thread execution with comprehensive error handling (Directives #9, #14)
    ExitCode Entry() override {
        try {
            success = (operation == Operation::LOAD_MODEL) ? LoadModel() : GenerateResponse();
            PostCompletionEvent();
        } catch (const std::exception& e) {
            // Ensure graceful error handling for all operations (Directive #14: Logical Consistency)
            config.result = "Error: " + std::string(e.what());
            success = false;
            PostCompletionEvent();
        } catch (...) {
            // Handle unexpected exceptions to prevent thread termination
            config.result = "Error: Unknown exception occurred";
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
        
        // Use direct context access for main chat to avoid potential context switching issues
        auto main_context = llama_manager->get_context_info("main_chat");
        if (!main_context) {
            config.result = "Error: Failed to access main chat context";
            return false;
        }
        
        // Create streaming callback that posts tokens to UI thread in real-time
        auto stream_callback = [this](std::string_view token_text) {
            if (should_stop || !parent) return;
            
            // Thread-safe streaming to UI
            wxCommandEvent event(wxEVT_TOKEN_STREAM);
            event.SetString(wxString::FromUTF8(token_text.data(), token_text.length()));
            wxQueueEvent(parent, event.Clone());
        };
        
        // Use the streaming version of generate_response
        config.result = llama_manager->generate_response_streaming(config.input_text, main_context, config.input_username, stream_callback);
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
        std::string discord_token, discord_isolated_channels;
        std::string summarizer_model_path, summarizer_system_prompt, summarizer_chat_template;
        std::string blacklist_entries;
        int32_t context_size{2048}, gpu_layers{0}, predict_tokens{256};
        int32_t summarizer_context_size{1024}, summarizer_gpu_layers{0}, summarizer_predict_tokens{128};
        int32_t discord_history_percentage{50};
        bool discord_allow_dms{true}, discord_pull_history{true};
    } config;
      // State management
    std::atomic<bool> is_started{false}, is_processing{false}, context_created{false};
    
    // Streaming state management
    std::atomic<bool> is_streaming_response{false};
    long current_stream_position{0};
    
    // Context monitoring timer
    wxTimer* context_monitor_timer;    // UI controls with better organization

    struct UIControls {
        wxButton *start_btn, *stop_btn, *settings_btn, *discord_btn, *prune_btn, *summary_slots_btn, *test_summaries_btn;
        wxGauge* progress_bar;
        
        // Detailed context breakdown UI elements
        wxGauge *overall_progress_bar, *summary_progress_bar, *history_progress_bar, *ai_space_progress_bar, *buffer_progress_bar;
        wxStaticText *overall_label, *summary_label, *history_label, *ai_space_label, *buffer_label;
        
        wxStaticText *progress_label, *cache_stats_label, *gen_stats_label;
        wxStaticBoxSizer* cache_stats_box;
        wxStaticBoxSizer* gen_stats_box;
        wxStaticBoxSizer* context_buffer_box;
        wxRichTextCtrl* chat_history;
        wxTextCtrl *input_text, *logs_text, *summaries_text;
        wxTextCtrl* message_history_list;
        wxChoice* context_selector;
        wxNotebook* notebook;
        wxPanel* main_panel;
    }ui;
      // Thread safety: Worker thread management (Directive #12)
    ModelWorkerThread* worker_thread{nullptr};

public:
    // Const-correct constructor (Directive #8: Const-Correctness)
    LuminaChatFrame() : wxFrame(nullptr, wxID_ANY, "LuminaChat", wxDefaultPosition, 
                               wxSize(UIConstants::DEFAULT_WINDOW_WIDTH, UIConstants::DEFAULT_WINDOW_HEIGHT)) {
        g_main_frame = this;
        
        llama_manager = std::make_unique<LlamaManager>();
        discord_manager = std::make_unique<DiscordManager>();

        // Initialize context monitoring timer
        context_monitor_timer = new wxTimer(this, static_cast<int>(EventId::CONTEXT_MONITOR_TIMER));
          // Set up unified logging
        LogHandler::set_output_callback([this](std::string_view msg) {
            AppendToLogsThreadSafe(wxString::FromUTF8(msg.data(), msg.length()));
        });
        
        // Set up summarizer-specific logging to route to summaries tab
        LogHandler::set_summarizer_callback([this](std::string_view msg) {
            AppendToSummariesThreadSafe(wxString::FromUTF8(msg.data(), msg.length()));
        });        LoadConfiguration();
        CreateUI();
        SetupConsoleRedirection();
        UpdateButtonStates();
        BindEvents();
    }

    // RAII-compliant destructor with proper cleanup order (Directive #9)
    ~LuminaChatFrame() {
        // Clean up in reverse order of construction
        if (context_monitor_timer && context_monitor_timer->IsRunning()) {
            context_monitor_timer->Stop();
        }
        delete context_monitor_timer;
        context_monitor_timer = nullptr;
        
        RestoreConsoleStreams();
        CleanupWorkerThread();
        g_main_frame = nullptr;
    }
    
    // Thread-safe logging methods (Directives #8, #12)
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
    
    // Summary display methods (Directive #8)
    void AppendSummaryInput(const wxString& input) {
        const wxString timestamp = wxDateTime::Now().Format("%H:%M:%S");
        const wxString formatted = wxString::Format("[%s] INPUT:\n%s\n\n", timestamp, input);
        AppendToSummariesThreadSafe(formatted);
    }    
    void AppendSummaryOutput(const wxString& output) {
        const wxString timestamp = wxDateTime::Now().Format("%H:%M:%S");
        const wxString formatted = wxString::Format("[%s] OUTPUT:\n%s\n\n%s\n\n", 
                                            timestamp, output, wxString(60, '-'));
        AppendToSummariesThreadSafe(formatted);
    }

private:
    void LoadConfiguration() {
        SettingsManager::LoadSettings(config.model_path, config.context_size, config.gpu_layers, 
                                    config.predict_tokens, config.chat_template, 
                                    config.identity_directive, config.other_directives,
                                    config.discord_token, config.discord_isolated_channels, 
                                    config.discord_allow_dms,
                                    config.discord_pull_history, config.discord_history_percentage,
                                    config.summarizer_model_path, config.summarizer_context_size,
                                    config.summarizer_gpu_layers, config.summarizer_predict_tokens,
                                    config.summarizer_system_prompt, config.summarizer_chat_template,
                                    config.blacklist_entries);
    }
    
    void CreateUI() {
        ui.main_panel = new wxPanel(this);
        
        CreateToolbar();
        CreateStatusArea();
        CreateNotebook();
        LayoutComponents();
        SetMinSize(wxSize(UIConstants::MINIMUM_WINDOW_WIDTH, UIConstants::MINIMUM_WINDOW_HEIGHT));
        AddWelcomeMessage();
    }
    
    void CreateToolbar() {
        ui.start_btn = new wxButton(ui.main_panel, static_cast<int>(EventId::START), "Start");
        ui.stop_btn = new wxButton(ui.main_panel, static_cast<int>(EventId::STOP), "Stop");
        ui.settings_btn = new wxButton(ui.main_panel, static_cast<int>(EventId::SETTINGS), "Settings");
        ui.discord_btn = new wxButton(ui.main_panel, static_cast<int>(EventId::CONNECT_DISCORD), "Connect Discord");
        ui.prune_btn = new wxButton(ui.main_panel, static_cast<int>(EventId::PRUNE_SUMMARIZE), "Prune && Summarize");
        ui.summary_slots_btn = new wxButton(ui.main_panel, static_cast<int>(EventId::VIEW_SUMMARY_SLOTS), "View Summaries");
    }
    
    void CreateStatusArea() {
        ui.progress_label = new wxStaticText(ui.main_panel, wxID_ANY, "Ready");
        ui.progress_bar = new wxGauge(ui.main_panel, wxID_ANY, 100, wxDefaultPosition, 
                                     wxSize(-1, UIConstants::PROGRESS_BAR_HEIGHT));
        ui.progress_bar->Hide();
        
        // Create context/buffer, generation, and cache statistics areas
        CreateContextBufferArea();
        CreateGenerationStatsArea();
        CreateCacheStatsArea();
    }    void CreateContextBufferArea() {
        ui.context_buffer_box = new wxStaticBoxSizer(wxVERTICAL, ui.main_panel, "Context Usage Breakdown");
          // Overall context usage (main indicator)
        auto* overall_sizer = new wxBoxSizer(wxHORIZONTAL);
        ui.overall_progress_bar = new wxGauge(ui.main_panel, wxID_ANY, 100, wxDefaultPosition, 
                                             wxSize(150, UIConstants::PROGRESS_BAR_HEIGHT));
        ui.overall_label = new wxStaticText(ui.main_panel, wxID_ANY, "Overall: N/A", wxDefaultPosition, 
                                          wxSize(UIConstants::CONTEXT_LABEL_WIDTH, -1));
        overall_sizer->Add(ui.overall_progress_bar, 1, wxEXPAND | wxRIGHT, 5);
        overall_sizer->Add(ui.overall_label, 0, wxALIGN_CENTER_VERTICAL);
        ui.context_buffer_box->Add(overall_sizer, 0, wxEXPAND | wxALL, 2);
        
        // Summary usage
        auto* summary_sizer = new wxBoxSizer(wxHORIZONTAL);
        ui.summary_progress_bar = new wxGauge(ui.main_panel, wxID_ANY, 100, wxDefaultPosition, 
                                             wxSize(150, UIConstants::PROGRESS_BAR_HEIGHT));
        ui.summary_label = new wxStaticText(ui.main_panel, wxID_ANY, "Summaries: N/A", wxDefaultPosition, 
                                          wxSize(UIConstants::CONTEXT_LABEL_WIDTH, -1));
        summary_sizer->Add(ui.summary_progress_bar, 1, wxEXPAND | wxRIGHT, 5);
        summary_sizer->Add(ui.summary_label, 0, wxALIGN_CENTER_VERTICAL);
        ui.context_buffer_box->Add(summary_sizer, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 2);
        
        // Active history usage
        auto* history_sizer = new wxBoxSizer(wxHORIZONTAL);
        ui.history_progress_bar = new wxGauge(ui.main_panel, wxID_ANY, 100, wxDefaultPosition, 
                                             wxSize(150, UIConstants::PROGRESS_BAR_HEIGHT));
        ui.history_label = new wxStaticText(ui.main_panel, wxID_ANY, "History: N/A", wxDefaultPosition, 
                                          wxSize(UIConstants::CONTEXT_LABEL_WIDTH, -1));
        history_sizer->Add(ui.history_progress_bar, 1, wxEXPAND | wxRIGHT, 5);
        history_sizer->Add(ui.history_label, 0, wxALIGN_CENTER_VERTICAL);
        ui.context_buffer_box->Add(history_sizer, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 2);
        
        // AI response space allocation
        auto* ai_space_sizer = new wxBoxSizer(wxHORIZONTAL);
        ui.ai_space_progress_bar = new wxGauge(ui.main_panel, wxID_ANY, 100, wxDefaultPosition, 
                                              wxSize(150, UIConstants::PROGRESS_BAR_HEIGHT));
        ui.ai_space_label = new wxStaticText(ui.main_panel, wxID_ANY, "AI Space: N/A", wxDefaultPosition, 
                                           wxSize(UIConstants::CONTEXT_LABEL_WIDTH, -1));
        ai_space_sizer->Add(ui.ai_space_progress_bar, 1, wxEXPAND | wxRIGHT, 5);
        ai_space_sizer->Add(ui.ai_space_label, 0, wxALIGN_CENTER_VERTICAL);
        ui.context_buffer_box->Add(ai_space_sizer, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 2);
        
        // Emergency buffer
        auto* buffer_sizer = new wxBoxSizer(wxHORIZONTAL);
        ui.buffer_progress_bar = new wxGauge(ui.main_panel, wxID_ANY, 100, wxDefaultPosition, 
                                            wxSize(150, UIConstants::PROGRESS_BAR_HEIGHT));
        ui.buffer_label = new wxStaticText(ui.main_panel, wxID_ANY, "Buffer: N/A", wxDefaultPosition, 
                                         wxSize(UIConstants::CONTEXT_LABEL_WIDTH, -1));
        buffer_sizer->Add(ui.buffer_progress_bar, 1, wxEXPAND | wxRIGHT, 5);
        buffer_sizer->Add(ui.buffer_label, 0, wxALIGN_CENTER_VERTICAL);
        ui.context_buffer_box->Add(buffer_sizer, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 2);
    }
      void CreateGenerationStatsArea() {
        ui.gen_stats_box = new wxStaticBoxSizer(wxVERTICAL, ui.main_panel, "Generation Stats");
        
        ui.gen_stats_label = new wxStaticText(ui.main_panel, wxID_ANY, "Ready\nTokens: 0\nSpeed: 0.0 tok/s\nTime: 0.0s", 
                                             wxDefaultPosition, wxSize(UIConstants::STATS_LABEL_WIDTH_SMALL, UIConstants::STATS_LABEL_HEIGHT));
        ui.gen_stats_label->SetFont(wxFont(8, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
        ui.gen_stats_box->Add(ui.gen_stats_label, 1, wxEXPAND | wxALL, 5);
    }
      void CreateCacheStatsArea() {
        ui.cache_stats_box = new wxStaticBoxSizer(wxVERTICAL, ui.main_panel, "Token Cache Stats");
        
        ui.cache_stats_label = new wxStaticText(ui.main_panel, wxID_ANY, "Hits: 0 | Misses: 0\nRequests: 0\nEntries: 0\nHit Ratio: 0.0%", 
                                               wxDefaultPosition, wxSize(UIConstants::STATS_LABEL_WIDTH_LARGE, UIConstants::STATS_LABEL_HEIGHT));
        ui.cache_stats_label->SetFont(wxFont(8, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
        ui.cache_stats_box->Add(ui.cache_stats_label, 1, wxEXPAND | wxALL, 5);
        
        // Initialize cache stats display
        UpdateCacheStats();
    }
      void CreateNotebook() {
        ui.notebook = new wxNotebook(ui.main_panel, wxID_ANY);
        
        CreateChatTab();
        CreateSummariesTab();
        CreateMessageHistoryTab();
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
                                   "============================================================\n"
                                   "Waiting for first summarization...\n\n");
        
        sizer->Add(ui.summaries_text, 1, wxEXPAND | wxALL, 5);
        panel->SetSizer(sizer);
        ui.notebook->AddPage(panel, "Summaries");
    }
      void CreateMessageHistoryTab() {
        auto* panel = new wxPanel(ui.notebook);
        auto* sizer = new wxBoxSizer(wxVERTICAL);
        
        // Add a top controls row with refresh button and context selector
        auto* controls_sizer = new wxBoxSizer(wxHORIZONTAL);
        auto* refresh_btn = new wxButton(panel, wxID_ANY, "Refresh Message History");
        
        // Create context selector dropdown
        ui.context_selector = new wxChoice(panel, wxID_ANY);
        ui.context_selector->SetMinSize(wxSize(150, -1));
        
        controls_sizer->Add(refresh_btn, 0, wxALIGN_CENTER_VERTICAL | wxRIGHT, 10);
        controls_sizer->Add(new wxStaticText(panel, wxID_ANY, "Context:"), 0, wxALIGN_CENTER_VERTICAL | wxRIGHT, 5);
        controls_sizer->Add(ui.context_selector, 0, wxALIGN_CENTER_VERTICAL);
        controls_sizer->AddStretchSpacer();
        
        sizer->Add(controls_sizer, 0, wxEXPAND | wxALL, 5);
          // Create the text control for message history (multi-line display)
        ui.message_history_list = new wxTextCtrl(panel, wxID_ANY, "", wxDefaultPosition, wxDefaultSize,
                                                wxTE_MULTILINE | wxTE_READONLY | wxTE_WORDWRAP);
        ui.message_history_list->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
        
        sizer->Add(ui.message_history_list, 1, wxEXPAND | wxALL, 5);
        
        // Bind refresh button event
        refresh_btn->Bind(wxEVT_COMMAND_BUTTON_CLICKED, [this](wxCommandEvent&) {
            RefreshMessageHistory();
        });
        
        panel->SetSizer(sizer);
        ui.notebook->AddPage(panel, "Message History");
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
    }
      void LayoutComponents() {
        auto* toolbar_sizer = new wxBoxSizer(wxHORIZONTAL);
        toolbar_sizer->Add(ui.start_btn, 0, wxRIGHT, 5);
        toolbar_sizer->Add(ui.stop_btn, 0, wxRIGHT, 5);  
        toolbar_sizer->Add(ui.settings_btn, 0, wxRIGHT, 5);
        toolbar_sizer->Add(ui.discord_btn, 0, wxRIGHT, 5);
        toolbar_sizer->Add(ui.prune_btn, 0, wxRIGHT, 5);
        toolbar_sizer->Add(ui.summary_slots_btn, 0);
        toolbar_sizer->AddStretchSpacer();
        
        // Top info area with three sections evenly spaced and center-aligned
        auto* top_info_sizer = new wxBoxSizer(wxHORIZONTAL);
        
        // Add flexible space before the first box
        top_info_sizer->AddStretchSpacer(1);
        
        // Context buffer box (left)
        top_info_sizer->Add(ui.context_buffer_box, 0, wxALIGN_CENTER);
        top_info_sizer->AddStretchSpacer(1);
        
        // Generation statistics box (center)
        top_info_sizer->Add(ui.gen_stats_box, 0, wxALIGN_CENTER);
        top_info_sizer->AddStretchSpacer(1);
        
        // Cache statistics box (right)
        top_info_sizer->Add(ui.cache_stats_box, 0, wxALIGN_CENTER);
        
        // Add flexible space after the last box
        top_info_sizer->AddStretchSpacer(1);
        
        // Status area for progress label only
        auto* status_sizer = new wxBoxSizer(wxHORIZONTAL);
        status_sizer->Add(ui.progress_label, 0, wxALIGN_CENTER_VERTICAL);
        status_sizer->AddStretchSpacer();

        auto* main_sizer = new wxBoxSizer(wxVERTICAL);
        main_sizer->Add(toolbar_sizer, 0, wxEXPAND | wxALL, 10);
        main_sizer->Add(top_info_sizer, 0, wxEXPAND | wxLEFT | wxRIGHT | wxTOP, 5);  // Reduced top margin from 10 to 5
        main_sizer->Add(status_sizer, 0, wxEXPAND | wxLEFT | wxRIGHT, 10);
        main_sizer->Add(ui.progress_bar, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 10);
        main_sizer->Add(ui.notebook, 1, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 10);
          ui.main_panel->SetSizer(main_sizer);
    }
      void BindEvents() {
        Bind(wxEVT_COMMAND_BUTTON_CLICKED, &LuminaChatFrame::OnStart, this, static_cast<int>(EventId::START));
        Bind(wxEVT_COMMAND_BUTTON_CLICKED, &LuminaChatFrame::OnStop, this, static_cast<int>(EventId::STOP));
        Bind(wxEVT_COMMAND_BUTTON_CLICKED, &LuminaChatFrame::OnSettings, this, static_cast<int>(EventId::SETTINGS));
        Bind(wxEVT_COMMAND_BUTTON_CLICKED, &LuminaChatFrame::OnConnectDiscord, this, static_cast<int>(EventId::CONNECT_DISCORD));        Bind(wxEVT_COMMAND_BUTTON_CLICKED, &LuminaChatFrame::OnPruneSummarize, this, static_cast<int>(EventId::PRUNE_SUMMARIZE));
        Bind(wxEVT_COMMAND_BUTTON_CLICKED, &LuminaChatFrame::OnViewSummarySlots, this, static_cast<int>(EventId::VIEW_SUMMARY_SLOTS));
        Bind(wxEVT_COMMAND_TEXT_ENTER, &LuminaChatFrame::OnInputEnter, this, static_cast<int>(EventId::INPUT_TEXT));
        
        Bind(wxEVT_MODEL_LOADED, &LuminaChatFrame::OnModelLoaded, this);
        Bind(wxEVT_RESPONSE_READY, &LuminaChatFrame::OnResponseReady, this);
        Bind(wxEVT_PROGRESS_UPDATE, &LuminaChatFrame::OnProgressUpdate, this);
        Bind(wxEVT_TOKEN_STREAM, &LuminaChatFrame::OnTokenStream, this);
        
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
    
    // Thread-safe worker cleanup with proper RAII (Directives #9, #12)
    void CleanupWorkerThread() noexcept {
        if (worker_thread) {
            worker_thread->RequestStop();
            worker_thread = nullptr;
            // Thread is detached and will clean itself up
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
        if (ui.summary_slots_btn) ui.summary_slots_btn->Enable(started && !processing);
        
        UpdateDiscordButtonState();
    }
    
    void UpdateDiscordButtonState() noexcept {
        if (!ui.discord_btn || !discord_manager) return;
        
        ui.discord_btn->SetLabel(discord_manager->is_running.load() ? 
                               "Disconnect Discord" : "Connect Discord");
    }
    
    // Optimized message formatting with smart caching (Directives #8, #13)
    void AddChatMessage(const wxString& message, const wxString& sender, const wxColour& bg_color) {
        // Cache attribute to avoid repeated allocations (Directive #13: Smart Caching)
        static thread_local wxRichTextAttr cached_attr;
        
        cached_attr.SetTextColour(*wxBLACK);
        cached_attr.SetBackgroundColour(bg_color);
        cached_attr.SetLeftIndent(50);
        cached_attr.SetRightIndent(50);
        cached_attr.SetParagraphSpacingBefore(0);
        cached_attr.SetParagraphSpacingAfter(0);
        
        if (sender != "System") {
            // Add spacing for non-system messages
        }
        
        ui.chat_history->BeginStyle(cached_attr);
        ui.chat_history->WriteText(sender + ": " + message);
        ui.chat_history->EndStyle();
        ui.chat_history->WriteText(sender == "You" ? "\n" : "\n\n");
        
        ui.chat_history->SetInsertionPointEnd();
        ui.chat_history->ShowPosition(ui.chat_history->GetLastPosition());
    }
    
    // Convenience methods for message types (Directive #8)
    void AddUserMessage(const wxString& message) {
        AddChatMessage(message, "You", wxColour(173, 216, 230));
    }
    
    void AddAIMessage(const wxString& message) {
        AddChatMessage(message, "AI", wxColour(144, 238, 144));
    }
    
    void AddSystemMessage(const wxString& message) {
        // Cache system message attributes (Directive #13: Smart Caching)
        static thread_local wxRichTextAttr system_attr;
        static thread_local bool attr_initialized = false;
        
        if (!attr_initialized) {
            system_attr.SetTextColour(wxColour(100, 100, 100));
            system_attr.SetBackgroundColour(wxColour(245, 245, 245));
            system_attr.SetLeftIndent(30);
            system_attr.SetRightIndent(30);
            system_attr.SetFontStyle(wxFONTSTYLE_ITALIC);
            attr_initialized = true;
        }
        
        ui.chat_history->BeginStyle(system_attr);
        ui.chat_history->WriteText("System: " + message);
        ui.chat_history->EndStyle();
        ui.chat_history->WriteText("\n\n");
    }
    
    void AddWelcomeMessage() {
        AddSystemMessage("Welcome to LuminaChat!\nClick 'Settings' to select a model, then 'Start' to begin.");
    }
    
    // System prompt composition with proper default handling (Directive #14: Logical Consistency)
    std::string GetCombinedSystemPrompt() const {
        if (config.identity_directive.empty() && config.other_directives.empty()) {
            return "You are a helpful AI assistant."; // Provide meaningful default
        }
        
        std::string combined = config.identity_directive;
        if (!config.identity_directive.empty() && !config.other_directives.empty()) {
            combined += "\n\n";
        }
        combined += config.other_directives;
          return combined;
    }

    // Optimized event handler methods with proper validation (Directive #6: Function Boundaries)
    void OnPruneSummarize(wxCommandEvent& event) {
        if (!is_started || is_processing || !llama_manager) {
            wxMessageBox("Please start the model first.", "Model Not Started", 
                        wxOK | wxICON_WARNING);
            return;
        }
        
        // Check if there's sufficient conversation to prune (Directive #5: Named Constants)
        auto context_info = llama_manager->get_context_info("main_chat");
        if (context_info->message_history.size() < UIConstants::MIN_MESSAGES_FOR_PRUNING) {
            wxMessageBox(wxString::Format("Need at least %d messages in the conversation to prune.", UIConstants::MIN_MESSAGES_FOR_PRUNING), 
                        "Insufficient Messages", wxOK | wxICON_INFORMATION);
            return;
        }
        
        // Check if summarization is available
        if (!llama_manager->has_context("summary_context")) {
            wxMessageBox("Summarization is not available. Please configure a summary model in Settings.", 
                        "Summarization Unavailable", wxOK | wxICON_WARNING);
            return;
        }
        
        // Get the main chat context directly instead of switching
        auto main_context = llama_manager->get_context_info("main_chat");
        if (!main_context) {
            wxMessageBox("Failed to access main chat context.", "Context Error", 
                        wxOK | wxICON_ERROR);
            return;
        }
        
        try {
            // Show progress
            AddSystemMessage("Starting prune and summarize (keeping 90% of context)...");
            
            // Perform pruning with 90% keep ratio (10% prune) using direct context access
            bool success = main_context->prune_with_summarization(0.9f);
              if (success) {
                // Update context after pruning using direct context access
                success = llama_manager->update_context_from_history(main_context);
                if (success) {
                    AddSystemMessage("Context pruned and summarized successfully.");
                    UpdateContextProgress();
                } else {
                    AddSystemMessage("Pruning succeeded but failed to update context.");
                }
            } else {
                AddSystemMessage("Failed to prune and summarize context.");
            }
            
        } catch (const std::exception& e) {
            AddSystemMessage(wxString::Format("Error during pruning: %s", e.what()));
        }
    }
      void OnViewSummarySlots(wxCommandEvent& event) {
        if (!is_started || !llama_manager) {
            wxMessageBox("Please start the model first.", "Model Not Started", 
                        wxOK | wxICON_WARNING);
            return;
        }
        
        // Get summary info for the main chat context
        auto main_context = llama_manager->get_context_info("main_chat");
        if (!main_context) {
            wxMessageBox("Failed to access main chat context.", "Context Error", 
                        wxOK | wxICON_ERROR);
            return;
        }
        
        //auto summary_info = llama_manager->get_summary_slot_info(main_context);
		auto summary_info = main_context->summarizer->get_summary_slot_info();
        
        wxString message;
        message << "Summary Slot System Status:\n\n";
        message << wxString::Format("Used slots: %zu / %zu\n\n", summary_info.used_slots, summary_info.total_slots);
        
        if (summary_info.used_slots == 0) {
            message << "No summaries generated yet.\n\n";
        } else {
            message << "Summaries (chronological order, oldest to newest):\n";
            message << wxString(50, '=') << "\n\n";
              for (size_t i = 0; const auto& summary : summary_info.summaries) {
                message << wxString::Format("Slot %zu:\n", ++i);
                
                // Show full summary without truncation
                wxString summary_text = wxString::FromUTF8(summary);
                
                message << summary_text << "\n\n";
                message << wxString(30, '-') << "\n\n";
            }
            
            message << "When slot " << summary_info.total_slots + 1 << " is needed, slot 1 & 2 will be\n";
            message << "automatically merged and all slots will shift left.";
        }
        
        wxMessageBox(message, "Summary Slots", wxOK | wxICON_INFORMATION);
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
                                        config.discord_allow_dms,
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
        
        // Thread-safe worker thread assignment (Directive #12: Thread Safety)
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
                auto model_info = llama_manager->get_model_info("main_model");
                std::string model_template = model_info->get_chat_template();
                if (!model_template.empty()) {
                    config.chat_template = model_template;                    SettingsManager::SaveSettings(config.model_path, config.context_size, config.gpu_layers, config.predict_tokens, 
                                                config.chat_template, 
                                                config.identity_directive, config.other_directives,
                                                config.discord_token, config.discord_isolated_channels, 
                                                config.discord_allow_dms,
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
                        // Initialize summarizer resources for all existing contexts
                        llama_manager->initialize_summarizer_resources();
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
                context_monitor_timer->Start(UIConstants::CONTEXT_MONITOR_INTERVAL_MS);
            }
              // Connect Discord manager to LlamaManager when model is loaded
            if (discord_manager && discord_manager->is_running.load()) {
                discord_manager->set_llama_manager(llama_manager.get());
                discord_manager->main_context_id = "main_chat";
                discord_manager->model_id = "main_model";
                DISCORD_LOG("Discord bot connected to loaded model");
            }            
            ui.chat_history->Clear();
            AddSystemMessage("LuminaChat ready! Type your message below.");
            ui.input_text->SetFocus();
            
            auto* main_context = llama_manager->get_context_info("main_chat");
            if (main_context) {
                main_context->reset_performance_stats();
            }
              // Reset generation stats display
            ui.gen_stats_label->SetLabel("Ready\nTokens: 0\nSpeed: 0.0 tok/s\nTime: 0.0s");            // Initialize cache stats display after model is loaded
            UpdateCacheStats();
            
            // Initialize message history display
            RefreshMessageHistory();
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
        }          // Reset context progress display
        ui.overall_label->SetLabel("Overall: N/A");
        ui.summary_label->SetLabel("Summaries: N/A");
        ui.history_label->SetLabel("History: N/A");
        ui.ai_space_label->SetLabel("AI Space: N/A");
        ui.buffer_label->SetLabel("Buffer: N/A");
        
        ui.overall_progress_bar->SetValue(0);
        ui.summary_progress_bar->SetValue(0);
        ui.history_progress_bar->SetValue(0);
        ui.ai_space_progress_bar->SetValue(0);
        ui.buffer_progress_bar->SetValue(0);
        
        // Reset colors to default
        ui.overall_progress_bar->SetForegroundColour(wxSystemSettings::GetColour(wxSYS_COLOUR_HIGHLIGHT));
        ui.summary_progress_bar->SetForegroundColour(wxSystemSettings::GetColour(wxSYS_COLOUR_HIGHLIGHT));
        ui.history_progress_bar->SetForegroundColour(wxSystemSettings::GetColour(wxSYS_COLOUR_HIGHLIGHT));
        ui.ai_space_progress_bar->SetForegroundColour(wxSystemSettings::GetColour(wxSYS_COLOUR_HIGHLIGHT));
        ui.buffer_progress_bar->SetForegroundColour(wxSystemSettings::GetColour(wxSYS_COLOUR_HIGHLIGHT));
        
        // Disconnect Discord manager from LlamaManager when model is stopped
        if (discord_manager && discord_manager->is_running.load()) {
            discord_manager->set_llama_manager(nullptr);
            DISCORD_LOG("Discord bot disconnected from model");
        }
          llama_manager->cleanup();
        context_created = false;
        is_started = false;
        is_processing = false;
        
        // Reset streaming state
        is_streaming_response = false;
        current_stream_position = 0;          UpdateButtonStates();
        LLAMA_LOG("LuminaChat stopped.");
        ui.gen_stats_label->SetLabel("Ready\nTokens: 0\nSpeed: 0.0 tok/s\nTime: 0.0s");        // Clear message history display when model is stopped
        if (ui.message_history_list) {
            ui.message_history_list->SetValue("Model stopped - no message history available");
        }
    }
      void OnConnectDiscord(wxCommandEvent& event) {
        if (!discord_manager) {
            AddSystemMessage("Error: Discord manager not available");
            return;
        }
        
        if (discord_manager->is_running.load()) {
            // Stop Discord bot
            discord_manager->shutdown();
            ui.discord_btn->SetLabel("Connect Discord");
            AddSystemMessage("Discord bot disconnected");
        } else {
            // Validate prerequisites before attempting connection
            if (!is_started || !llama_manager || !context_created) {
                AddSystemMessage("Error: Please start the AI model first before connecting Discord");
                return;
            }
            
            if (!llama_manager->has_context("main_chat")) {
                AddSystemMessage("Error: Main chat context not available. Please restart the model.");
                return;
            }
            
            // Configure and start Discord bot
            if (config.discord_token.empty()) {
                AddSystemMessage("Discord bot token not configured. Please check Settings.");
                return;
            }
            
            AddSystemMessage("Validating Discord configuration...");
            
            // Configure bot
            DiscordBotConfig bot_config;
            bot_config.bot_token = config.discord_token;
            
            if (!discord_manager->configure(bot_config)) {
                AddSystemMessage("Failed to configure Discord bot - check token format");
                return;
            }
            
            AddSystemMessage("Setting up Discord integration...");
            
            // Set up integration with LlamaManager BEFORE starting
            discord_manager->set_llama_manager(llama_manager.get());
            discord_manager->main_context_id = "main_chat";
            discord_manager->model_id = "main_model";
            discord_manager->set_isolated_channels(config.discord_isolated_channels);
            discord_manager->set_allow_dms(config.discord_allow_dms);
            discord_manager->set_history_settings(config.discord_pull_history, config.discord_history_percentage);
            
            AddSystemMessage("Starting Discord bot...");
            
            // Start bot
            if (discord_manager->start()) {
                ui.discord_btn->SetLabel("Disconnect Discord");
                AddSystemMessage("Discord bot connecting...");
                
                // Simplified backfill monitoring for new structure
                std::thread([this]() {
                    bool backfill_reported = false;
                    
                    for (int i = 0; i < 60; ++i) { // Check for up to 5 minutes
                        std::this_thread::sleep_for(std::chrono::seconds(UIConstants::DISCORD_CONNECTION_DELAY_SEC));
                        
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
                AddSystemMessage("Failed to start Discord bot - check logs for details");
            }
        }
    }

    void OnSettings(wxCommandEvent& event) {        SettingsDialog dialog(this, config.model_path, config.context_size, config.gpu_layers, config.predict_tokens, 
                            config.chat_template, config.identity_directive, config.other_directives,
                            config.discord_token, config.discord_isolated_channels, 
                            config.discord_allow_dms,
                            config.discord_pull_history, config.discord_history_percentage,
                            config.summarizer_model_path, config.summarizer_context_size,
                            config.summarizer_gpu_layers, config.summarizer_predict_tokens,
                            config.summarizer_system_prompt, config.summarizer_chat_template,
                            config.blacklist_entries);
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
        
        // Verify main chat context exists instead of switching to it
        if (!llama_manager->has_context("main_chat")) {
            AddSystemMessage("Error: Main chat context not available.");
            return;
        }
        
        wxString input = ui.input_text->GetValue().Trim();
        if (input.IsEmpty()) return;
          // Display user input with blue background
        AddUserMessage(input);
        ui.input_text->Clear();
        
        // Reset streaming state for new generation
        is_streaming_response = false;
        current_stream_position = 0;
        
        is_processing = true;
        UpdateButtonStates();
        
        // Thread-safe response generation startup (Directive #12: Thread Safety)
        worker_thread = new ModelWorkerThread(this, llama_manager.get(), ModelWorkerThread::Operation::GENERATE_RESPONSE);
        worker_thread->SetInput(input.ToStdString(), "User");
        
        if (worker_thread->Run() != wxTHREAD_NO_ERROR) {
            delete worker_thread;
            worker_thread = nullptr;
            is_processing = false;
            UpdateButtonStates();
            AddSystemMessage("Error: Failed to start response generation thread");
        }
    }      void OnResponseReady(wxCommandEvent& event) {
        is_processing = false;
        worker_thread = nullptr; // Thread is detached and will clean itself up
        
        wxString response = event.GetString();
        
        // Check for error responses
        if (response.StartsWith("Error:")) {
            AddSystemMessage(response);
        } else {            // If we were streaming, finish the stream formatting
            if (is_streaming_response) {
                is_streaming_response = false;
                
                // End the style formatting for the streamed message
                ui.chat_history->EndStyle();
                ui.chat_history->WriteText("\n\n");
            } else {
                // Fallback: display the complete response if streaming didn't work
                AddAIMessage(response);
            }
            
            // Update generation stats after successful inference
            UpdateGenerationStats();
            
            // Update context progress immediately after response
            UpdateContextProgress();
            
            // Update cache statistics after response generation
            UpdateCacheStats();
            
            // Refresh message history to show the new conversation state
            RefreshMessageHistory();
        }
        
        UpdateButtonStates();
        ui.input_text->SetFocus();
        
        // Auto-scroll to bottom
        ui.chat_history->SetInsertionPointEnd();
        ui.chat_history->ShowPosition(ui.chat_history->GetLastPosition());
    }// Token streaming event handler for real-time response display
    void OnTokenStream(wxCommandEvent& event) {
        wxString token_text = event.GetString();
        
        // Check if we need to start a new AI message
        if (!is_streaming_response) {
            is_streaming_response = true;
            current_stream_position = ui.chat_history->GetLastPosition();
            
            // Start AI message formatting with distinctive appearance
            wxRichTextAttr ai_attr;
            ai_attr.SetTextColour(*wxBLACK);
            ai_attr.SetBackgroundColour(wxColour(144, 238, 144)); // Light green background
            ai_attr.SetLeftIndent(50);
            ai_attr.SetRightIndent(50);
            
            ui.chat_history->BeginStyle(ai_attr);
            ui.chat_history->BeginBold();
            ui.chat_history->WriteText("AI: ");
            ui.chat_history->EndBold();
        }
        
        // Handle line breaks properly - replace \n with proper line breaks
        wxString processed_text = token_text;
        processed_text.Replace("\n", "\n", true); // Ensure newlines are preserved
        
        // For wxRichTextCtrl, we need to handle newlines explicitly
        if (processed_text.Contains('\n')) {
            // Split on newlines and write each part, adding line breaks manually
            wxArrayString lines = wxSplit(processed_text, '\n');
            for (size_t i = 0; i < lines.GetCount(); ++i) {
                if (i > 0) {
                    // Add a line break for subsequent lines
                    ui.chat_history->Newline();
                }
                if (!lines[i].IsEmpty()) {
                    ui.chat_history->WriteText(lines[i]);
                }
            }
        } else {
            // No newlines, write directly
            ui.chat_history->WriteText(processed_text);
        }
        
        ui.chat_history->SetInsertionPointEnd();
        ui.chat_history->ShowPosition(ui.chat_history->GetLastPosition());
        
        // Force immediate UI update for smooth streaming effect
        ui.chat_history->Update();
    }
    
    void UpdateGenerationStats() {
        if (!is_started || !llama_manager) {
            ui.gen_stats_label->SetLabel("Ready\nTokens: 0\nSpeed: 0.0 tok/s\nTime: 0.0s");
            return;
        }
        
        auto context_info = llama_manager->get_context_info("main_chat");
        if (!context_info) {
            ui.gen_stats_label->SetLabel("No Context\nTokens: 0\nSpeed: 0.0 tok/s\nTime: 0.0s");
            return;
        }
        
        const auto& stats = context_info->get_performance_stats();

        if (stats.last_total_generation_tokens > 0) {
            double tokens_per_sec = UIConstants::MS_TO_SECONDS * stats.last_total_generation_tokens / (stats.last_decode_time_us / 1000.0);
            double total_time_sec = (stats.last_decode_time_us / 1000.0) / UIConstants::MS_TO_SECONDS;
            
            wxString gen_display = wxString::Format(
                "Generated\nTokens: %lld\nSpeed: %.1f tok/s\nTime: %.1fs",
                stats.last_total_generation_tokens,
                tokens_per_sec,
                total_time_sec
            );
            ui.gen_stats_label->SetLabel(gen_display);
        } else {
            ui.gen_stats_label->SetLabel("Ready\nTokens: 0\nSpeed: 0.0 tok/s\nTime: 0.0s");
        }
    }
    
    void OnContextMonitorTimer(wxTimerEvent& event) {
        UpdateContextProgress();
        UpdateCacheStats();
        UpdateGenerationStats();
    }
      void UpdateCacheStats() {
        if (!llama_manager) {
            ui.cache_stats_label->SetLabel("Hits: 0 | Misses: 0\nRequests: 0\nEntries: 0\nHit Ratio: 0.0%");
            return;
        }
        
        auto stats = llama_manager->get_token_cache_stats();
        size_t misses = stats.requests > stats.hits ? stats.requests - stats.hits : 0;
        
        wxString cache_display = wxString::Format(
            "Hits: %zu | Misses: %zu\nRequests: %zu\nEntries: %zu\nHit Ratio: %.1f%%",
            stats.hits, misses,
            stats.requests, stats.entries,
            stats.hit_ratio * 100.0f
        );
        
        ui.cache_stats_label->SetLabel(cache_display);
    }
      void UpdateContextProgress() {
        if (!is_started || !llama_manager || !context_created || !llama_manager->has_context("main_chat")) {
            // Reset all progress bars and labels when not available
            ui.overall_label->SetLabel("Overall: N/A");
            ui.summary_label->SetLabel("Summaries: N/A");
            ui.history_label->SetLabel("History: N/A");
            ui.ai_space_label->SetLabel("AI Space: N/A");
            ui.buffer_label->SetLabel("Buffer: N/A");
            
            ui.overall_progress_bar->SetValue(0);
            ui.summary_progress_bar->SetValue(0);
            ui.history_progress_bar->SetValue(0);
            ui.ai_space_progress_bar->SetValue(0);
            ui.buffer_progress_bar->SetValue(0);
            return;
        }

        // Get context usage information using ContextSizeManager
        auto context_info = llama_manager->get_context_info("main_chat");
        if (!context_info || !context_info->model_info) {
            return;
        }
        
        // Get detailed analysis from ContextSizeManager
        auto analysis = analyze_context_usage(*context_info, *context_info->model_info);
        
        if (analysis.context_size > 0) {
            // Overall context usage - main indicator
            float overall_percentage = static_cast<float>(analysis.total_used_tokens) / analysis.context_size * 100.0f;
            ui.overall_progress_bar->SetValue(static_cast<int32_t>(std::min(overall_percentage, 100.0f)));
            
            // Color coding for overall progress bar based on usage
            if (overall_percentage >= 90.0f) {
                ui.overall_progress_bar->SetForegroundColour(wxColour(220, 20, 20)); // Red - critical
            } else if (overall_percentage >= 75.0f) {
                ui.overall_progress_bar->SetForegroundColour(wxColour(255, 165, 0)); // Orange - warning
            } else {
                ui.overall_progress_bar->SetForegroundColour(wxSystemSettings::GetColour(wxSYS_COLOUR_HIGHLIGHT)); // Normal
            }
            
            // Summary usage breakdown
            float summary_percentage = static_cast<float>(analysis.summary_tokens) / analysis.context_size * 100.0f;
            ui.summary_progress_bar->SetValue(static_cast<int32_t>(std::min(summary_percentage, 100.0f)));
            
            // Color summary bar based on hard cap (30%)
            if (analysis.summary_hard_cap_exceeded) {
                ui.summary_progress_bar->SetForegroundColour(wxColour(220, 20, 20)); // Red - exceeded hard cap
            } else if (summary_percentage >= 25.0f) {
                ui.summary_progress_bar->SetForegroundColour(wxColour(255, 165, 0)); // Orange - approaching cap
            } else {
                ui.summary_progress_bar->SetForegroundColour(wxColour(100, 200, 100)); // Green - healthy
            }
            
            // Active history usage breakdown
            float history_percentage = static_cast<float>(analysis.active_history_tokens) / analysis.context_size * 100.0f;
            ui.history_progress_bar->SetValue(static_cast<int32_t>(std::min(history_percentage, 100.0f)));
            ui.history_progress_bar->SetForegroundColour(wxColour(100, 150, 255)); // Blue for active content
            
            // AI response space allocation
            float ai_space_percentage = static_cast<float>(analysis.required_ai_space) / analysis.context_size * 100.0f;
            ui.ai_space_progress_bar->SetValue(static_cast<int32_t>(std::min(ai_space_percentage, 100.0f)));
            
            // Color AI space based on minimum requirements
            if (analysis.available_tokens < analysis.required_ai_space) {
                ui.ai_space_progress_bar->SetForegroundColour(wxColour(220, 20, 20)); // Red - insufficient space
            } else {
                ui.ai_space_progress_bar->SetForegroundColour(wxColour(150, 100, 255)); // Purple for AI space
            }
            
            // Emergency buffer allocation
            float buffer_percentage = static_cast<float>(analysis.emergency_buffer_space) / analysis.context_size * 100.0f;
            ui.buffer_progress_bar->SetValue(static_cast<int32_t>(std::min(buffer_percentage, 100.0f)));
            
            // Color buffer based on violation status
            if (analysis.emergency_buffer_violated) {
                ui.buffer_progress_bar->SetForegroundColour(wxColour(220, 20, 20)); // Red - buffer violated
            } else {
                ui.buffer_progress_bar->SetForegroundColour(wxColour(255, 200, 100)); // Yellow for buffer space
            }
            
            // Update labels with detailed information
            ui.overall_label->SetLabel(wxString::Format("Overall: %d/%d (%.1f%%)", 
                analysis.total_used_tokens, analysis.context_size, overall_percentage));
                
            ui.summary_label->SetLabel(wxString::Format("Summaries: %d tokens (%.1f%%) [%zu slots]", 
                analysis.summary_tokens, summary_percentage, analysis.summary_slot_stats.slot_count));
                
            ui.history_label->SetLabel(wxString::Format("History: %d tokens (%.1f%%)", 
                analysis.active_history_tokens, history_percentage));
                
            ui.ai_space_label->SetLabel(wxString::Format("AI Space: %d tokens (%.1f%%)", 
                analysis.required_ai_space, ai_space_percentage));
                
            ui.buffer_label->SetLabel(wxString::Format("Buffer: %d tokens (%.1f%%)", 
                analysis.emergency_buffer_space, buffer_percentage));
                
        } else {
            // Reset to default values
            ui.overall_label->SetLabel("Overall: 0/0");
            ui.summary_label->SetLabel("Summaries: 0 tokens");
            ui.history_label->SetLabel("History: 0 tokens");
            ui.ai_space_label->SetLabel("AI Space: 0 tokens");
            ui.buffer_label->SetLabel("Buffer: 0 tokens");
            
            ui.overall_progress_bar->SetValue(0);
            ui.summary_progress_bar->SetValue(0);
            ui.history_progress_bar->SetValue(0);
            ui.ai_space_progress_bar->SetValue(0);
            ui.buffer_progress_bar->SetValue(0);
        }
    }      void RefreshMessageHistory() {        if (!ui.message_history_list || !is_started || !llama_manager) {
            if (ui.message_history_list) {
                ui.message_history_list->SetValue("Model not started - no message history available");
            }
            if (ui.context_selector) {
                ui.context_selector->Clear();
                ui.context_selector->Append("No contexts available");
                ui.context_selector->SetSelection(0);
                ui.context_selector->Enable(false);
            }
            return;
        }
        
        // First, update the context selector dropdown with all available contexts
        UpdateContextSelector();
        
        // Clear existing content
        ui.message_history_list->SetValue("");
          // Get selected context from dropdown
        wxString selected_context = "main_chat"; // Default
        if (ui.context_selector && ui.context_selector->GetSelection() != wxNOT_FOUND) {
            selected_context = ui.context_selector->GetStringSelection();
        }
        
        // Get the selected context
        auto context_info = llama_manager->get_context_info(selected_context.ToStdString());
        if (!context_info) {
            ui.message_history_list->SetValue(wxString::Format("Context '%s' not found or not available", selected_context));
            return;
        }
        
        // Add each message to the text control with full content
        if (context_info->message_history.empty()) {
            ui.message_history_list->SetValue("No messages in history");        } else {
            wxString full_history;
            for (size_t i = 0; i < context_info->message_history.size(); ++i) {
                const auto& msg = context_info->message_history[i];
                
                // Format: [index] role: content (full content, no truncation)
                std::string content = msg.second;  // content is the second element of the pair
                
                // Keep newlines intact for proper multi-line display
                wxString formatted = wxString::Format("[%zu] %s:\n%s\n\n", 
                                                    i, 
                                                    wxString::FromUTF8(msg.first),   // role is the first element of the pair
                                                    wxString::FromUTF8(content));
                full_history += formatted;
            }
            ui.message_history_list->SetValue(full_history);
        }
        
        // Scroll to the bottom to show most recent messages
        ui.message_history_list->SetInsertionPointEnd();
    }void UpdateContextSelector() {
        if (!ui.context_selector || !llama_manager) return;
        
        // Store current selection
        wxString current_selection = ui.context_selector->GetStringSelection();
        
        // Clear and repopulate the dropdown
        ui.context_selector->Clear();
        
        if (!is_started) {
            ui.context_selector->Append("No contexts available");
            ui.context_selector->SetSelection(0);
            ui.context_selector->Enable(false);
            return;
        }
        
        // Get all available contexts directly from LlamaManager
        auto context_ids = llama_manager->get_context_ids();
        
        if (context_ids.empty()) {
            ui.context_selector->Append("No contexts available");
            ui.context_selector->SetSelection(0);
            ui.context_selector->Enable(false);
        } else {
            // Add all available contexts
            for (const auto& context_id : context_ids) {
                ui.context_selector->Append(wxString::FromUTF8(context_id.c_str()));
            }
            
            // Try to restore previous selection, otherwise default to "main_chat" or first item
            int selection_index = ui.context_selector->FindString(current_selection);
            if (selection_index == wxNOT_FOUND) {
                selection_index = ui.context_selector->FindString("main_chat");
                if (selection_index == wxNOT_FOUND && ui.context_selector->GetCount() > 0) {
                    selection_index = 0;
                }
            }
            
            if (selection_index != wxNOT_FOUND) {
                ui.context_selector->SetSelection(selection_index);
            }
            ui.context_selector->Enable(true);
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

// RAII-compliant application class with comprehensive error handling (Directives #4, #9)
class LuminaChatApp : public wxApp {
public:
    // Cross-platform application initialization with proper error handling (Directive #11)
    bool OnInit() override {
        SetAppName("LuminaChat");
        SetVendorName("LuminaChat");
        
        try {
            auto* frame = new LuminaChatFrame();
            frame->Show(true);
            return true;
        } catch (const std::exception& e) {
            wxMessageBox(wxString::Format("Failed to initialize application: %s", e.what()), 
                        "Initialization Error", wxOK | wxICON_ERROR);
            return false;
        } catch (...) {
            // Handle any unexpected exceptions (Directive #14: Logical Consistency)
            wxMessageBox("Unknown error occurred during application initialization", 
                        "Critical Error", wxOK | wxICON_ERROR);
            return false;
        }
    }
};

wxIMPLEMENT_APP(LuminaChatApp);

// Cross-platform entry point with fixed-width types (Directive #11: Portability)
int32_t main(int32_t argc, char* argv[]) {
    return wxEntry(argc, argv);
}

//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODE DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
