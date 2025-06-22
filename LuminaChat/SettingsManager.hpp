// SettingsManager.hpp - header-only implementation for settings management
// Handles loading and saving configuration in INI format with sections
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

#pragma once

#include <string>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <algorithm>
// wxWidgets includes for Settings Dialog
#include <wx/dialog.h>
#include <wx/textctrl.h>
#include <wx/button.h>
#include <wx/checkbox.h>
#include <wx/slider.h>
#include <wx/sizer.h>
#include <wx/panel.h>
#include <wx/notebook.h>
#include <wx/stattext.h>
#include <wx/filedlg.h>
#include <wx/msgdlg.h>
#include <wx/scrolwin.h>
#include <wx/stdpaths.h>
#include <wx/filename.h>
#include "LogHandler.hpp"

// UI Constants and Event IDs for Settings Dialog
namespace SettingsUIConstants {
    constexpr int32_t WELCOME_MESSAGE_HEIGHT = 100;
    constexpr int32_t DIRECTIVE_MESSAGE_HEIGHT = 200;
    constexpr int32_t TEMPLATE_MESSAGE_HEIGHT = 200;
    constexpr int32_t SUMMARY_PROMPT_HEIGHT = 120;
    constexpr int32_t CHANNEL_LIST_HEIGHT = 80;
    constexpr int32_t BLACKLIST_ENTRIES_HEIGHT = 300;
    constexpr int32_t SLIDER_WIDTH = 120;
    constexpr int32_t SETTINGS_DIALOG_WIDTH = 700;
    constexpr int32_t SETTINGS_DIALOG_HEIGHT = 600;
}

enum class SettingsEventId : int32_t {
    BROWSE_MODEL = 2000,
    BROWSE_SUMMARIZER_MODEL = 2100,
    SAVE_APPLY_BLACKLIST = 2200
};

class SettingsManager {
private:
    // Named constants to replace magic literals (Directive #5)
    static constexpr int32_t DEFAULT_CONTEXT_SIZE = 2048;
    static constexpr int32_t DEFAULT_GPU_LAYERS = 0;
    static constexpr int32_t DEFAULT_PREDICT_TOKENS = 256;
    static constexpr int32_t DEFAULT_HISTORY_FILL_PERCENTAGE = 50;
    static constexpr int32_t DEFAULT_SUMMARIZER_CONTEXT_SIZE = 1024;
    static constexpr int32_t DEFAULT_SUMMARIZER_PREDICT_TOKENS = 128;
    
    // Validation limits
    static constexpr int32_t MIN_CONTEXT_SIZE = 1;
    static constexpr int32_t MAX_CONTEXT_SIZE = 131072;
    static constexpr int32_t MIN_GPU_LAYERS = 0;
    static constexpr int32_t MAX_GPU_LAYERS = 999;
    static constexpr int32_t MIN_PREDICT_TOKENS = 1;
    static constexpr int32_t MAX_PREDICT_TOKENS = 4096;
    static constexpr int32_t MIN_HISTORY_PERCENTAGE = 10;
    static constexpr int32_t MAX_HISTORY_PERCENTAGE = 80;
    static constexpr int32_t MAX_SUMMARIZER_CONTEXT = 32768;
    static constexpr int32_t MAX_SUMMARIZER_PREDICT = 2048;    static std::string EscapeString(const std::string& input) noexcept {
        std::string result;
        result.reserve(input.length() + input.length() / 4); // Reserve extra space for escapes
        for (const char c : input) {
            switch (c) {
                case '\n': result += "\\n"; break;
                case '\r': result += "\\r"; break;
                case '\t': result += "\\t"; break;
                case '\\': result += "\\\\"; break;
                case '=': result += "\\="; break;
                default: result += c; break;
            }
        }
        return result;
    }
      static std::string UnescapeString(const std::string& input) noexcept {
        std::string result;
        result.reserve(input.length()); // Reserve space for efficiency
        for (size_t i = 0; i < input.size(); ++i) {
            if (input[i] == '\\' && i + 1 < input.size()) [[unlikely]] {
                switch (input[i + 1]) {
                    case 'n': result += '\n'; i++; break;
                    case 'r': result += '\r'; i++; break;
                    case 't': result += '\t'; i++; break;
                    case '\\': result += '\\'; i++; break;
                    case '=': result += '='; i++; break;
                    default: result += input[i]; break;
                }
            } else [[likely]] {
                result += input[i];
            }
        }
        return result;
    }    // Validate and clamp integer values
    static int32_t ValidateInt32(const std::string& value, const int32_t default_val, const int32_t min_val, const int32_t max_val) noexcept {
        try {
            const int32_t result = std::stoi(value);
            return std::clamp(result, min_val, max_val);
        } catch (...) {
            return default_val;
        }
    }

public:    static std::string GetSettingsFilePath() noexcept {
        // Get the directory where the executable is located
        wxString exeDir = wxStandardPaths::Get().GetExecutablePath();
        wxFileName exePath(exeDir);
        wxString appDir = exePath.GetPath();
        
        // Create the settings file path in the same directory as the executable
        wxFileName configFile(appDir, "luminachat.ini");
        std::string filepath = configFile.GetFullPath().ToStdString();
        
        SETTINGS_LOG("Settings file path: " + filepath);
        return filepath;
    }
      static void SaveSettings(const std::string& model_path, int32_t context_size, int32_t gpu_layers, 
                           int32_t predict_tokens, const std::string& chat_template, 
                           const std::string& identity_directive, const std::string& other_directives,
                           const std::string& discord_bot_token,
                           const std::string& discord_isolated_channel_ids, const std::string& discord_shared_history_channel_ids,
                           bool discord_allow_dms, bool discord_pull_history, int32_t discord_history_fill_percentage,
                           const std::string& summarizer_model_path, int32_t summarizer_context_size, int32_t summarizer_gpu_layers,
                           int32_t summarizer_predict_tokens, const std::string& summarizer_system_prompt, const std::string& summarizer_chat_template,
                           const std::string& blacklist_entries = "") {
        std::string filepath = GetSettingsFilePath();
        std::ofstream file(filepath);
        
        if (file.is_open()) {
            // [General] section
            file << "[General]" << std::endl;
            file << "ModelPath=" << EscapeString(model_path) << std::endl;
            file << "ContextSize=" << context_size << std::endl;
            file << "GpuLayers=" << gpu_layers << std::endl;
            file << "PredictTokens=" << predict_tokens << std::endl;
            file << std::endl;
            
            // [ChatTemplate] section
            file << "[ChatTemplate]" << std::endl;
            file << "Template=" << EscapeString(chat_template) << std::endl;
            file << std::endl;
            
            // [SystemPrompt] section
            file << "[SystemPrompt]" << std::endl;
            file << "IdentityDirective=" << EscapeString(identity_directive) << std::endl;
            file << "OtherDirectives=" << EscapeString(other_directives) << std::endl;
            file << std::endl;
            
            // [Discord] section
            file << "[Discord]" << std::endl;
            file << "BotToken=" << EscapeString(discord_bot_token) << std::endl;
            file << "IsolatedChannelIds=" << EscapeString(discord_isolated_channel_ids) << std::endl;
            file << "SharedHistoryChannelIds=" << EscapeString(discord_shared_history_channel_ids) << std::endl;
            file << "AllowDMs=" << (discord_allow_dms ? "1" : "0") << std::endl;
            file << "PullHistory=" << (discord_pull_history ? "1" : "0") << std::endl;
            file << "HistoryFillPercentage=" << discord_history_fill_percentage << std::endl;
            file << std::endl;
              // [Summarizer] section
            file << "[Summarizer]" << std::endl;
            file << "ModelPath=" << EscapeString(summarizer_model_path) << std::endl;
            file << "ContextSize=" << summarizer_context_size << std::endl;
            file << "GpuLayers=" << summarizer_gpu_layers << std::endl;
            file << "PredictTokens=" << summarizer_predict_tokens << std::endl;
            file << "SystemPrompt=" << EscapeString(summarizer_system_prompt) << std::endl;
            file << "ChatTemplate=" << EscapeString(summarizer_chat_template) << std::endl;
            file << std::endl;
            
            // [Blacklist] section
            file << "[Blacklist]" << std::endl;
            file << "Entries=" << EscapeString(blacklist_entries) << std::endl;
            
            file.close();
            
            SETTINGS_LOG("Settings saved to: " + filepath);
        } else {
            SETTINGS_LOG("Error: Failed to save settings to: " + filepath);
        }
    }
      static void LoadSettings(std::string& model_path, int32_t& context_size, int32_t& gpu_layers, 
                           int32_t& predict_tokens, std::string& chat_template, 
                           std::string& identity_directive, std::string& other_directives,
                           std::string& discord_bot_token,
                           std::string& discord_isolated_channel_ids, std::string& discord_shared_history_channel_ids,
                           bool& discord_allow_dms, bool& discord_pull_history, int32_t& discord_history_fill_percentage,
                           std::string& summarizer_model_path, int32_t& summarizer_context_size, int32_t& summarizer_gpu_layers,
                           int32_t& summarizer_predict_tokens, std::string& summarizer_system_prompt, std::string& summarizer_chat_template,
                           std::string& blacklist_entries) {
        std::string filepath = GetSettingsFilePath();
        std::ifstream file(filepath);
        
        // Set defaults first
        if (model_path.empty()) model_path = "";
        if (context_size == 0) context_size = DEFAULT_CONTEXT_SIZE;
        if (gpu_layers == 0) gpu_layers = DEFAULT_GPU_LAYERS;
        if (predict_tokens == 0) predict_tokens = DEFAULT_PREDICT_TOKENS;
        discord_allow_dms = true; // Default to true
        discord_pull_history = true; // Default to true
        discord_history_fill_percentage = DEFAULT_HISTORY_FILL_PERCENTAGE; // Default to 50%
        
        // Summarizer defaults
        if (summarizer_context_size == 0) summarizer_context_size = DEFAULT_SUMMARIZER_CONTEXT_SIZE;
        if (summarizer_gpu_layers == 0) summarizer_gpu_layers = DEFAULT_GPU_LAYERS;
        if (summarizer_predict_tokens == 0) summarizer_predict_tokens = DEFAULT_SUMMARIZER_PREDICT_TOKENS;
        if (summarizer_system_prompt.empty()) {
            summarizer_system_prompt = "You are a helpful AI assistant that provides concise summaries. "
                                     "Focus on key points and maintain clarity while keeping responses brief.";
        }
        
        if (file.is_open()) {
            SETTINGS_LOG("Loading settings from: " + filepath);
            std::string line;
            std::string current_section;
            int loaded_count = 0;
            
            while (std::getline(file, line)) {
                // Trim whitespace
                line.erase(0, line.find_first_not_of(" \t\r\n"));
                line.erase(line.find_last_not_of(" \t\r\n") + 1);
                
                // Skip empty lines and comments
                if (line.empty() || line[0] == '#' || line[0] == ';') {
                    continue;
                }
                
                // Check for section headers
                if (line[0] == '[' && line.back() == ']') {
                    current_section = line.substr(1, line.length() - 2);
                    SETTINGS_LOG("Reading section: [" + current_section + "]");
                    continue;
                }
                
                // Parse key=value pairs
                size_t pos = line.find('=');
                if (pos != std::string::npos) {
                    std::string key = line.substr(0, pos);
                    std::string value = line.substr(pos + 1);
                    
                    key.erase(0, key.find_first_not_of(" \t"));
                    key.erase(key.find_last_not_of(" \t") + 1);
                    value.erase(0, value.find_first_not_of(" \t"));
                    value.erase(value.find_last_not_of(" \t") + 1);
                    
                    // Handle settings based on section
                    if (current_section == "General") {
                        if (key == "ModelPath") {
                            model_path = UnescapeString(value);
                            loaded_count++;
                            SETTINGS_LOG("Loaded ModelPath: " + std::string(model_path.empty() ? "(empty)" : "configured"));
                        } else if (key == "ContextSize") {
                            context_size = ValidateInt32(value, DEFAULT_CONTEXT_SIZE, MIN_CONTEXT_SIZE, MAX_CONTEXT_SIZE);
                            loaded_count++;
                            SETTINGS_LOG("Loaded ContextSize: " + std::to_string(context_size));
                        } else if (key == "GpuLayers") {
                            gpu_layers = ValidateInt32(value, DEFAULT_GPU_LAYERS, MIN_GPU_LAYERS, MAX_GPU_LAYERS);
                            loaded_count++;
                            SETTINGS_LOG("Loaded GpuLayers: " + std::to_string(gpu_layers));
                        } else if (key == "PredictTokens") {
                            predict_tokens = ValidateInt32(value, DEFAULT_PREDICT_TOKENS, MIN_PREDICT_TOKENS, MAX_PREDICT_TOKENS);
                            loaded_count++;
                            SETTINGS_LOG("Loaded PredictTokens: " + std::to_string(predict_tokens));
                        }
                    } else if (current_section == "ChatTemplate") {
                        if (key == "Template") {
                            chat_template = UnescapeString(value);
                            loaded_count++;
                            SETTINGS_LOG("Loaded ChatTemplate: " + std::string(chat_template.empty() ? "(empty)" : "configured"));
                        }
                    } else if (current_section == "SystemPrompt") {
                        if (key == "IdentityDirective") {
                            identity_directive = UnescapeString(value);
                            loaded_count++;
                            SETTINGS_LOG("Loaded IdentityDirective: " + std::string(identity_directive.empty() ? "(empty)" : "configured"));
                        } else if (key == "OtherDirectives") {
                            other_directives = UnescapeString(value);
                            loaded_count++;
                            SETTINGS_LOG("Loaded OtherDirectives: " + std::string(other_directives.empty() ? "(empty)" : "configured"));
                        }
                    } else if (current_section == "Discord") {
                        if (key == "BotToken") {
                            discord_bot_token = UnescapeString(value);
                            loaded_count++;
                            SETTINGS_LOG("Loaded BotToken: " + std::string(discord_bot_token.empty() ? "(empty)" : "configured"));
                        } else if (key == "IsolatedChannelIds") {
                            discord_isolated_channel_ids = UnescapeString(value);
                            loaded_count++;
                            SETTINGS_LOG("Loaded IsolatedChannelIds: " + discord_isolated_channel_ids);
                        } else if (key == "SharedHistoryChannelIds") {
                            discord_shared_history_channel_ids = UnescapeString(value);
                            loaded_count++;
                            SETTINGS_LOG("Loaded SharedHistoryChannelIds: " + discord_shared_history_channel_ids);
                        } else if (key == "AllowDMs") {
                            discord_allow_dms = (value == "1" || value == "true" || value == "True" || value == "TRUE");
                            loaded_count++;
                            SETTINGS_LOG("Loaded AllowDMs: " + std::string(discord_allow_dms ? "true" : "false"));
                        } else if (key == "PullHistory") {
                            discord_pull_history = (value == "1" || value == "true" || value == "True" || value == "TRUE");
                            loaded_count++;
                            SETTINGS_LOG("Loaded PullHistory: " + std::string(discord_pull_history ? "true" : "false"));
                        } else if (key == "HistoryFillPercentage") {
                            discord_history_fill_percentage = ValidateInt32(value, DEFAULT_HISTORY_FILL_PERCENTAGE, MIN_HISTORY_PERCENTAGE, MAX_HISTORY_PERCENTAGE);
                            loaded_count++;
                            SETTINGS_LOG("Loaded HistoryFillPercentage: " + std::to_string(discord_history_fill_percentage) + "%");                        }
                    } else if (current_section == "Summarizer") {
                        if (key == "ModelPath") {
                            summarizer_model_path = UnescapeString(value);
                            loaded_count++;
                            SETTINGS_LOG("Loaded Summarizer ModelPath: " + std::string(summarizer_model_path.empty() ? "(empty)" : "configured"));
                        } else if (key == "ContextSize") {
                            summarizer_context_size = ValidateInt32(value, DEFAULT_SUMMARIZER_CONTEXT_SIZE, MIN_CONTEXT_SIZE, MAX_SUMMARIZER_CONTEXT);
                            loaded_count++;
                            SETTINGS_LOG("Loaded Summarizer ContextSize: " + std::to_string(summarizer_context_size));
                        } else if (key == "GpuLayers") {
                            summarizer_gpu_layers = ValidateInt32(value, DEFAULT_GPU_LAYERS, MIN_GPU_LAYERS, MAX_GPU_LAYERS);
                            loaded_count++;
                            SETTINGS_LOG("Loaded Summarizer GpuLayers: " + std::to_string(summarizer_gpu_layers));
                        } else if (key == "PredictTokens") {
                            summarizer_predict_tokens = ValidateInt32(value, DEFAULT_SUMMARIZER_PREDICT_TOKENS, MIN_PREDICT_TOKENS, MAX_SUMMARIZER_PREDICT);
                            loaded_count++;
                            SETTINGS_LOG("Loaded Summarizer PredictTokens: " + std::to_string(summarizer_predict_tokens));
                        } else if (key == "SystemPrompt") {
                            summarizer_system_prompt = UnescapeString(value);
                            loaded_count++;
                            SETTINGS_LOG("Loaded Summarizer SystemPrompt: " + std::string(summarizer_system_prompt.empty() ? "(empty)" : "configured"));                        } else if (key == "ChatTemplate") {
                            summarizer_chat_template = UnescapeString(value);
                            loaded_count++;
                            SETTINGS_LOG("Loaded Summarizer ChatTemplate: " + std::string(summarizer_chat_template.empty() ? "(empty)" : "configured"));
                        }
                    } else if (current_section == "Blacklist") {
                        if (key == "Entries") {
                            blacklist_entries = UnescapeString(value);
                            loaded_count++;
                            SETTINGS_LOG("Loaded Blacklist Entries: " + std::string(blacklist_entries.empty() ? "(empty)" : "configured"));
                        }
                    } else {
                        SETTINGS_LOG("Unknown section/setting: [" + current_section + "] " + key + "=" + value);
                    }
                }
            }
            file.close();
            
            SETTINGS_LOG("Settings loading complete. Loaded " + std::to_string(loaded_count) + " settings.");
        } else {
            SETTINGS_LOG("Settings file not found: " + filepath + " (using defaults)");
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
        
        // Blacklist settings
        wxTextCtrl* blacklist_entries;
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
        int32_t& summarizer_gpu_layers;        int32_t& summarizer_predict_tokens;
        std::string& summarizer_system_prompt;
        std::string& summarizer_chat_template;
        std::string& blacklist_entries;
    } config;

    // UI elements
    const wxFont monospace_font{9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL};
    const wxFont help_font{8, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_ITALIC, wxFONTWEIGHT_NORMAL};

public:    SettingsDialog(wxWindow* parent, std::string& model_path, int32_t& context_size, int32_t& gpu_layers, 
                  int32_t& predict_tokens, std::string& chat_template, std::string& identity_directive, 
                  std::string& other_directives, std::string& discord_bot_token,
                  std::string& discord_isolated_channels, std::string& discord_shared_channels,
                  bool& discord_allow_dms, bool& discord_pull_history, int32_t& discord_history_percentage,
                  std::string& summarizer_model_path, int32_t& summarizer_context_size, int32_t& summarizer_gpu_layers,
                  int32_t& summarizer_predict_tokens, std::string& summarizer_system_prompt, std::string& summarizer_chat_template,
                  std::string& blacklist_entries) 
        : wxDialog(parent, wxID_ANY, "Settings", wxDefaultPosition, 
                  wxSize(SettingsUIConstants::SETTINGS_DIALOG_WIDTH, SettingsUIConstants::SETTINGS_DIALOG_HEIGHT))
        , config{model_path, context_size, gpu_layers, predict_tokens, chat_template,
                identity_directive, other_directives, discord_bot_token, discord_isolated_channels,
                discord_shared_channels, discord_allow_dms, discord_pull_history, discord_history_percentage,
                summarizer_model_path, summarizer_context_size, summarizer_gpu_layers, summarizer_predict_tokens,
                summarizer_system_prompt, summarizer_chat_template, blacklist_entries} {
        
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
        CreateBlacklistTab(notebook);
        
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
        auto* browse_btn = new wxButton(scrolled, static_cast<int>(SettingsEventId::BROWSE_MODEL), "Browse...");
        
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
                                               wxDefaultPosition, wxSize(-1, SettingsUIConstants::WELCOME_MESSAGE_HEIGHT), 
                                               wxTE_MULTILINE | wxTE_WORDWRAP);
        ctrls.identity_directive->SetFont(monospace_font);
        sizer->Add(ctrls.identity_directive, 0, wxEXPAND | wxALL, 5);
        
        // Other Directives (2/3 height)
        sizer->Add(new wxStaticText(scrolled, wxID_ANY, "Other Directives:"), 0, wxALL, 5);
        ctrls.other_directives = new wxTextCtrl(scrolled, wxID_ANY, wxString::FromUTF8(config.other_directives), 
                                             wxDefaultPosition, wxSize(-1, SettingsUIConstants::DIRECTIVE_MESSAGE_HEIGHT), 
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
                                           wxDefaultPosition, wxSize(-1, SettingsUIConstants::TEMPLATE_MESSAGE_HEIGHT), 
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
        auto* browse_btn = new wxButton(scrolled, static_cast<int>(SettingsEventId::BROWSE_SUMMARIZER_MODEL), "Browse...");
        
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
                                                      wxDefaultPosition, wxSize(-1, SettingsUIConstants::SUMMARY_PROMPT_HEIGHT), 
                                                      wxTE_MULTILINE | wxTE_WORDWRAP);
        ctrls.summarizer_system_prompt->SetFont(monospace_font);
        sizer->Add(ctrls.summarizer_system_prompt, 0, wxEXPAND | wxALL, 5);
        
        // Chat Template
        sizer->Add(new wxStaticText(scrolled, wxID_ANY, "Chat Template (leave empty to use model default):"), 0, wxALL, 5);
        ctrls.summarizer_chat_template = new wxTextCtrl(scrolled, wxID_ANY, wxString::FromUTF8(config.summarizer_chat_template), 
                                                       wxDefaultPosition, wxSize(-1, SettingsUIConstants::SUMMARY_PROMPT_HEIGHT), 
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
                                               10, 80, wxDefaultPosition, wxSize(SettingsUIConstants::SLIDER_WIDTH, -1));
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
    
    void CreateBlacklistTab(wxNotebook* notebook) {
        auto* panel = new wxPanel(notebook);
        auto* scrolled = new wxScrolledWindow(panel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxVSCROLL);
        scrolled->SetScrollRate(0, 20);
        
        auto* sizer = new wxBoxSizer(wxVERTICAL);
        
        // Help text
        auto* help_text = new wxStaticText(scrolled, wxID_ANY, 
            "Add responses that should be excluded from conversation history.\n"
            "Enter one blacklisted phrase or pattern per line:");
        help_text->SetFont(help_font);
        sizer->Add(help_text, 0, wxALL, 5);
        
        // Blacklist entries text area
        ctrls.blacklist_entries = new wxTextCtrl(scrolled, wxID_ANY, 
                                                wxString::FromUTF8(config.blacklist_entries), 
                                                wxDefaultPosition, 
                                                wxSize(-1, SettingsUIConstants::BLACKLIST_ENTRIES_HEIGHT), 
                                                wxTE_MULTILINE | wxTE_WORDWRAP);
        ctrls.blacklist_entries->SetFont(monospace_font);
        sizer->Add(ctrls.blacklist_entries, 1, wxEXPAND | wxALL, 5);
        
        // Save and Apply button
        auto* save_apply_btn = new wxButton(scrolled, static_cast<int>(SettingsEventId::SAVE_APPLY_BLACKLIST), "Save and Apply Blacklist");
        sizer->Add(save_apply_btn, 0, wxALIGN_RIGHT | wxALL, 5);
        
        // Additional help text
        auto* usage_help = new wxStaticText(scrolled, wxID_ANY, 
            "Tips:\n"
            "• Case-insensitive matching\n"
            "• Partial matches are supported\n"
            "• Empty lines are ignored\n"
            "• Changes apply to new responses and existing history");
        usage_help->SetFont(help_font);
        sizer->Add(usage_help, 0, wxALL, 5);
        
        scrolled->SetSizer(sizer);
        
        auto* panel_sizer = new wxBoxSizer(wxVERTICAL);
        panel_sizer->Add(scrolled, 1, wxEXPAND);
        panel->SetSizer(panel_sizer);
        
        notebook->AddPage(panel, "Blacklist");
        
        // Bind save and apply button event
        save_apply_btn->Bind(wxEVT_COMMAND_BUTTON_CLICKED, [this](wxCommandEvent&) {
            OnSaveApplyBlacklist();
        });
    }

    void BindEvents() {
        Bind(wxEVT_COMMAND_BUTTON_CLICKED, &SettingsDialog::OnBrowseModel, this, static_cast<int>(SettingsEventId::BROWSE_MODEL));
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
    
    // Improved validation with better error handling (Directives #8, #14)
    template<typename T>
    bool ValidateNumeric(wxTextCtrl* control, T& target, const T min_val, const T max_val, 
                        const T default_val, const wxString& field_name) {
        if (!control) {
            target = default_val;
            return false;
        }
        
        long value;
        if (control->GetValue().ToLong(&value) && value >= static_cast<long>(min_val) && value <= static_cast<long>(max_val)) {
            target = static_cast<T>(value);
            return true;
        }
        
        // Reset to default on validation failure (Directive #14: Logical Consistency)
        target = default_val;
        control->SetValue(wxString::Format("%ld", static_cast<long>(default_val)));
        
        wxMessageBox(wxString::Format("Invalid %s (must be %ld-%ld). Using default: %ld", 
                    field_name, static_cast<long>(min_val), static_cast<long>(max_val),
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
        ValidateNumeric(ctrls.summarizer_predict_tokens, config.summarizer_predict_tokens, 1, 2048, 128, "summarizer prediction tokens");        config.summarizer_system_prompt = ctrls.summarizer_system_prompt->GetValue().ToUTF8().data();        config.summarizer_chat_template = ctrls.summarizer_chat_template->GetValue().ToUTF8().data();
        
        // Blacklist settings
        config.blacklist_entries = ctrls.blacklist_entries->GetValue().ToUTF8().data();
        
        // Save settings with all parameters
        SettingsManager::SaveSettings(config.model_path, config.context_size, config.gpu_layers, 
                    config.predict_tokens, config.chat_template, 
                    config.identity_directive, config.other_directives,
                    config.discord_bot_token, config.discord_isolated_channels, 
                    config.discord_shared_channels, config.discord_allow_dms, 
                    config.discord_pull_history, config.discord_history_percentage,
                    config.summarizer_model_path, config.summarizer_context_size,
                    config.summarizer_gpu_layers, config.summarizer_predict_tokens,
                    config.summarizer_system_prompt, config.summarizer_chat_template,
                    config.blacklist_entries);
        
        EndModal(wxID_OK);
    }
    
    // Optimized channel ID validation with STL algorithms (Directives #10, #14)
    std::string ValidateChannelIds(const std::string& input) {
        if (input.empty()) return {};
        
        std::string result;
        result.reserve(input.length());
        
        std::string current_id;
        current_id.reserve(32); // Discord snowflake IDs are typically 18-19 digits
        
        // Use STL algorithm for better performance (Directive #10)
        for (const char c : input) {
            if (std::isdigit(static_cast<unsigned char>(c))) {
                current_id += c;
            } else if (!current_id.empty() && (c == ',' || std::isspace(static_cast<unsigned char>(c)))) {
                if (!result.empty()) result += ',';
                result += current_id;
                current_id.clear();
            }
        }
        
        // Handle final ID if input doesn't end with delimiter
        if (!current_id.empty()) {
            if (!result.empty()) result += ',';
            result += current_id;
        }
        
        return result;
    }
    
    void OnSaveApplyBlacklist() {
        // Update the blacklist configuration
        config.blacklist_entries = ctrls.blacklist_entries->GetValue().ToUTF8().data();
        
        // Save settings with current values and updated blacklist
        SettingsManager::SaveSettings(config.model_path, config.context_size, config.gpu_layers, 
                    config.predict_tokens, config.chat_template, 
                    config.identity_directive, config.other_directives,
                    config.discord_bot_token, config.discord_isolated_channels, 
                    config.discord_shared_channels, config.discord_allow_dms, 
                    config.discord_pull_history, config.discord_history_percentage,
                    config.summarizer_model_path, config.summarizer_context_size,
                    config.summarizer_gpu_layers, config.summarizer_predict_tokens,
                    config.summarizer_system_prompt, config.summarizer_chat_template,
                    config.blacklist_entries);
        
        wxMessageBox("Blacklist settings saved and applied successfully!", "Blacklist Updated", 
                    wxOK | wxICON_INFORMATION);
    }
    
    // Helper method to reduce code duplication
    void AddTextSetting(wxWindow* parent, wxBoxSizer* sizer, const wxString& label, 
                       wxTextCtrl*& control, const std::string& value, long style = 0) {
        sizer->Add(new wxStaticText(parent, wxID_ANY, label), 0, wxALL, 5);
        control = new wxTextCtrl(parent, wxID_ANY, wxString::FromUTF8(value), 
                                wxDefaultPosition, (style & wxTE_MULTILINE) ? wxSize(-1, SettingsUIConstants::CHANNEL_LIST_HEIGHT) : wxDefaultSize, style);
        if (style & wxTE_MULTILINE) {
            control->SetFont(monospace_font);
        }
        sizer->Add(control, 0, wxEXPAND | wxALL, 5);
    }
};

//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODE DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//