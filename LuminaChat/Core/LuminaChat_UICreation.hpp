#pragma once

// LuminaChat_UICreation.hpp - UI panel creation and layout methods
// This header contains all the wxWidgets UI creation logic for the different
// panels (Chat, Discord, Logs), keeping the layout code organized and separate.

#include <wx/wx.h>
#include <wx/notebook.h>
#include <wx/textctrl.h>
#include <wx/richtext/richtextctrl.h>
#include <wx/button.h>
#include <wx/sizer.h>
#include <wx/panel.h>
#include <wx/stattext.h>
#include <wx/listbox.h>
#include "../Logger.hpp"
#include <wx/choice.h>
#include <wx/checkbox.h>
#include <wx/splitter.h>
#include <wx/scrolwin.h>
#include "LuminaChat_UIUtilities.hpp"

// === UI Panel Creation Methods ===

inline void LuminaChatFrame::CreateChatPanel() {
    chat_panel = new wxPanel(notebook);
    notebook->AddPage(chat_panel, "Chat", true);
    
    // Create splitter window for resizable layout
    chat_splitter = new wxSplitterWindow(chat_panel, wxID_ANY, 
                                        wxDefaultPosition, wxDefaultSize,
                                        wxSP_3D | wxSP_LIVE_UPDATE);
    chat_splitter->SetMinimumPaneSize(LuminaChatConstants::SPLITTER_MIN_SIZE);  // Minimum size for each pane
    
    // Create top panel for inner voice display
    inner_voice_panel = new wxPanel(chat_splitter);
    wxStaticText* inner_voice_label = new wxStaticText(inner_voice_panel, wxID_ANY, "Inner Voice Context (Real-time):");
    inner_voice_label->SetFont(wxFont(9, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD));
    
    inner_voice_display = new wxTextCtrl(inner_voice_panel, wxID_ANY, wxEmptyString,
                                        wxDefaultPosition, wxDefaultSize,
                                        wxTE_READONLY | wxTE_MULTILINE | wxHSCROLL | wxVSCROLL);
    inner_voice_display->SetFont(wxFont(LuminaChatConstants::MONOSPACE_FONT_SIZE, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
    inner_voice_display->SetBackgroundColour(LuminaChatColors::INNER_VOICE_BACKGROUND); // Light blue tint
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
    chat_input->SetMinSize(wxSize(-1, LuminaChatConstants::CHAT_INPUT_MIN_HEIGHT));
    
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
    chat_splitter->SplitHorizontally(inner_voice_panel, main_chat_panel, LuminaChatConstants::INNER_VOICE_INITIAL_HEIGHT); // Initial split at configured height for inner voice
    
    // Layout for the main chat panel
    wxBoxSizer* chat_panel_sizer = new wxBoxSizer(wxVERTICAL);
    chat_panel_sizer->Add(chat_splitter, 1, wxEXPAND);
    chat_panel->SetSizer(chat_panel_sizer);
    
    // Bind enter key to send
    chat_input->Bind(wxEVT_TEXT_ENTER, &LuminaChatFrame::OnSendMessage, this);
}

inline void LuminaChatFrame::CreateDiscordPanel() {
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
        if (settings_manager && discord_token_text) {
            std::string token = discord_token_text->GetValue().ToStdString();
            settings_manager->SetString("Discord", "bot_token", token);
            settings_manager->SaveSettings();
            LOG_UI("Discord token auto-saved");
        }
        event.Skip();
    });
    
    // Auto-save allowed channels on kill focus
    allowed_channels_text->Bind(wxEVT_KILL_FOCUS, [this](wxFocusEvent& event) {
        if (settings_manager && allowed_channels_text) {
            std::string channels = allowed_channels_text->GetValue().ToStdString();
            settings_manager->SetString("Discord", "allowed_channels", channels);
            settings_manager->SaveSettings();
            UpdateDiscordAllowedChannels(); // Update the manager immediately
            LOG_UI("Discord allowed channels auto-saved and updated");
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
        if (settings_manager && auto_respond_checkbox) {
            bool auto_respond = auto_respond_checkbox->GetValue();
            settings_manager->SetBool("Discord", "auto_respond", auto_respond);
            settings_manager->SaveSettings();
            LOG_UI("Discord auto-respond setting auto-saved: " + std::string(auto_respond ? "enabled" : "disabled"));
        }
        event.Skip();
    });
    
    // Layout
    wxBoxSizer* discord_sizer = new wxBoxSizer(wxVERTICAL);
    discord_sizer->Add(discord_box, 0, wxEXPAND | wxALL, 5);
    discord_sizer->Add(channels_box, 1, wxEXPAND | wxALL, 5);
    
    discord_panel->SetSizer(discord_sizer);
}

inline void LuminaChatFrame::CreateLogsPanel() {
    logs_panel = new wxPanel(notebook);
    notebook->AddPage(logs_panel, "System Logs");
    
    // Log display
    logs_display = new wxTextCtrl(logs_panel, wxID_ANY, wxEmptyString,
                                 wxDefaultPosition, wxDefaultSize,
                                 wxTE_READONLY | wxTE_MULTILINE | wxHSCROLL | wxVSCROLL);
    logs_display->SetFont(wxFont(LuminaChatConstants::MONOSPACE_FONT_SIZE, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
    
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
