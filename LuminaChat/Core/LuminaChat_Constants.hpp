#pragma once

// LuminaChat_Constants.hpp - Centralized constants and configuration
// This header consolidates all application constants to avoid duplication
// and provide a single source of truth for configuration values.

#include <wx/wx.h>

// === Application Configuration Constants ===
namespace LuminaChatConstants {
    // UI Layout and spacing
    static constexpr int DEFAULT_WINDOW_WIDTH = 1200;
    static constexpr int DEFAULT_WINDOW_HEIGHT = 800;
    static constexpr int CONTROL_SPACING = 5;
    static constexpr int PANEL_MARGIN = 10;
    static constexpr int SPLITTER_MIN_SIZE = 80;
    static constexpr int INNER_VOICE_INITIAL_HEIGHT = 150;
    static constexpr int CHAT_INPUT_MIN_HEIGHT = 80;
    
    // Font configurations
    static constexpr int MONOSPACE_FONT_SIZE = 9;
    static constexpr int LABEL_FONT_SIZE = 9;
    
    // Model loading and processing
    static constexpr int MODEL_LOAD_TIMEOUT_MS = 300000;  // 5 minutes
    static constexpr int TIMER_INTERVAL_MS = 1000;        // 1 second for UI updates
    
    // Plugin monitoring intervals
    static constexpr int EMOTION_ANALYSIS_DELAY_MS = 1000;  // Delay after generation for emotion analysis
    static constexpr int CONTEXT_MONITORING_DELAY_MS = 1000; // Delay after generation for context monitoring
    static constexpr int PLUGIN_STATUS_UPDATE_INTERVAL_MS = 5000; // Plugin status update frequency
    static constexpr int DEBUG_UPDATE_INTERVAL = 5; // Update debug info every N timer ticks
    
    // Discord configuration
    static constexpr int DISCORD_RECONNECT_DELAY_MS = 30000; // 30 seconds
    static constexpr int DISCORD_CHANNEL_REFRESH_INTERVAL_MS = 60000; // 1 minute
    
    // Context and memory management
    static constexpr int DEFAULT_CONTEXT_SIZE = 4096;
    static constexpr int MAX_MESSAGE_HISTORY_SIZE = 1000;
    static constexpr int CONTEXT_WARNING_THRESHOLD = 0.8; // 80% full
    static constexpr int CONTEXT_CRITICAL_THRESHOLD = 0.95; // 95% full
    
    // File paths and extensions
    static const char* SETTINGS_FILE_NAME = "LuminaChat_settings.ini";
    static const char* LOG_FILE_NAME = "LuminaChat.log";
    static const char* MODEL_FILE_EXTENSIONS[] = {".gguf", ".bin", ".safetensors"};
    
    // Application metadata
    static const char* APP_NAME = "LuminaChat";
    static const char* APP_VERSION = "1.0.0";
    static const char* APP_DESCRIPTION = "AI Chat Application with Two-Stage Reasoning";
}

// === UI Color Constants ===
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
    static const wxColour INNER_VOICE_BACKGROUND{248, 248, 255}; // Light blue tint
    static const wxColour CONTEXT_WARNING{255, 180, 0};         // Orange for context warnings
    static const wxColour CONTEXT_CRITICAL{255, 100, 100};      // Red for context critical
}

// === Event ID Constants ===
namespace LuminaChatEventIDs {
    enum {
        ID_Send = 1000,
        ID_Stop,
        ID_ConnectDiscord,
        ID_Timer,
        ID_ClearChat,
        ID_ClearLogs,
        ID_LoadModel,
        ID_SaveSettings,
        ID_RefreshChannels,
        ID_ToggleAutoRespond,
        // Reserve space for future event IDs
        ID_RESERVED_END = 1100
    };
}

// === Default Settings Values ===
namespace LuminaChatDefaults {
    // Model settings
    static const char* DEFAULT_MODEL_PATH = "";
    static constexpr int DEFAULT_CONTEXT_SIZE = 4096;
    static constexpr float DEFAULT_TEMPERATURE = 0.7f;
    static constexpr int DEFAULT_MAX_TOKENS = 512;
    
    // Discord settings
    static const char* DEFAULT_DISCORD_TOKEN = "";
    static const char* DEFAULT_ALLOWED_CHANNELS = "";
    static constexpr bool DEFAULT_AUTO_RESPOND = true;
    
    // Logging settings
    static const char* DEFAULT_LOG_LEVEL = "INFO";
    static constexpr bool DEFAULT_LOG_TO_FILE = true;
    
    // UI settings
    static constexpr bool DEFAULT_SHOW_INNER_VOICE = true;
    static constexpr bool DEFAULT_AUTO_SCROLL = true;
    static constexpr bool DEFAULT_WORD_WRAP = true;
}

// === Validation Constants ===
namespace LuminaChatValidation {
    static constexpr int MIN_CONTEXT_SIZE = 512;
    static constexpr int MAX_CONTEXT_SIZE = 32768;
    static constexpr float MIN_TEMPERATURE = 0.0f;
    static constexpr float MAX_TEMPERATURE = 2.0f;
    static constexpr int MIN_MAX_TOKENS = 1;
    static constexpr int MAX_MAX_TOKENS = 4096;
    static constexpr int MAX_DISCORD_CHANNEL_ID_LENGTH = 20;
    static constexpr int MAX_LOG_MESSAGE_LENGTH = 10000;
}
