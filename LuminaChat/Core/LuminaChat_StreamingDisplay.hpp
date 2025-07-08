#pragma once

// LuminaChat_StreamingDisplay.hpp - Streaming display and message management methods
// This header contains all the streaming message display functionality for both
// the main chat display and the inner voice display, keeping the UI update logic organized.

#include <wx/wx.h>
#include <wx/richtext/richtextctrl.h>

// === Streaming Message Display Methods ===

inline void LuminaChatFrame::StartStreamingMessage(const std::string& sender, const wxColour& color) {
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

inline void LuminaChatFrame::AppendToStreamingMessage(const std::string& text) {
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

inline void LuminaChatFrame::ReplaceStreamingMessage(const std::string& text) {
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

inline void LuminaChatFrame::EndStreamingMessage() {
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
    
    // Note: Focus is now set in the completion callback after UI state changes
}

// === Inner Voice Streaming Methods ===

inline void LuminaChatFrame::StartInnerVoiceStreaming() {
    is_inner_voice_streaming = true;
    current_inner_voice_message.clear();
    
    if (inner_voice_display) {
        inner_voice_display->SetValue("");
    }
}

inline void LuminaChatFrame::AppendToInnerVoiceStreaming(const std::string& text) {
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

inline void LuminaChatFrame::ReplaceInnerVoiceStreaming(const std::string& text) {
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

inline void LuminaChatFrame::EndInnerVoiceStreaming() {
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

// === Chat and Log Message Methods ===

inline void LuminaChatFrame::AddChatMessage(const std::string& sender, const std::string& message, const wxColour& color) {
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
