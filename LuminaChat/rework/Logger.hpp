#pragma once

#include <string>
#include <string_view>
#include <functional>
#include <mutex>
#include <sstream>
#include <chrono>
#include <iomanip>

/**
 * Centralized logging system for all LuminaChat components.
 * 
 * Features:
 * - Thread-safe logging with component-specific macros
 * - UI output callback registration (breaks circular dependency)
 * - Automatic timestamping and component identification
 */

class Logger {
public:
    enum class LogLevel {
        DEBUG = 0,
        INFO = 1,
        WARNING = 2,
        ERROR = 3
    };

private:
    std::function<void(std::string_view)> output_callback;
    std::mutex callback_mutex;
    LogLevel current_level = LogLevel::INFO;

    std::string GetTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        ss << "." << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }

public:
    // Callback registration for UI output (higher component registers with lower)
    void RegisterOutputCallback(std::function<void(std::string_view)> callback) {
        std::lock_guard<std::mutex> lock(callback_mutex);
        output_callback = std::move(callback);
    }

    void SetLogLevel(LogLevel level) {
        current_level = level;
    }

    LogLevel GetLogLevel() const {
        return current_level;
    }

    void LogMessage(LogLevel level, const std::string& component, const std::string& message) {
        if (level < current_level) {
            return;
        }

        std::string level_str;
        switch (level) {
            case LogLevel::DEBUG:   level_str = "DEBUG"; break;
            case LogLevel::INFO:    level_str = "INFO";  break;
            case LogLevel::WARNING: level_str = "WARN";  break;
            case LogLevel::ERROR:   level_str = "ERROR"; break;
        }

        std::stringstream log_line;
        log_line << "[" << GetTimestamp() << "] [" << level_str << "] [" << component << "] " << message;

        // Thread-safe callback execution
        std::lock_guard<std::mutex> lock(callback_mutex);
        if (output_callback) {
            output_callback(log_line.str());
        }
    }
};

// Global logger instance
inline Logger& GetLogger() {
    static Logger logger;
    return logger;
}

// Component-specific logging macros (unique per file as specified in design)
#define LOG_DEBUG(component, message) \
    GetLogger().LogMessage(Logger::LogLevel::DEBUG, component, message)

#define LOG_INFO(component, message) \
    GetLogger().LogMessage(Logger::LogLevel::INFO, component, message)

#define LOG_WARNING(component, message) \
    GetLogger().LogMessage(Logger::LogLevel::WARNING, component, message)

#define LOG_ERROR(component, message) \
    GetLogger().LogMessage(Logger::LogLevel::ERROR, component, message)

// File-specific convenience macros (to be customized per component file)
#define LOG_Logger(message) LOG_INFO("Logger", message)
#define LOG_SettingsManager(message) LOG_INFO("SettingsManager", message)
#define LOG_Sanitizer(message) LOG_INFO("Sanitizer", message)
#define LOG_DiscordManager(message) LOG_INFO("DiscordManager", message)
#define LOG_ContextSizeManager(message) LOG_INFO("ContextSizeManager", message)
#define LOG_TokenCache(message) LOG_INFO("TokenCache", message)
#define LOG_ModelInfo(message) LOG_INFO("ModelInfo", message)
#define LOG_ChatTemplateManager(message) LOG_INFO("ChatTemplateManager", message)
#define LOG_ContextInfo(message) LOG_INFO("ContextInfo", message)
#define LOG_LlamaManager(message) LOG_INFO("LlamaManager", message)
#define LOG_Orchestrator(message) LOG_INFO("Orchestrator", message)
#define LOG_LuminaChat(message) LOG_INFO("LuminaChat", message)
