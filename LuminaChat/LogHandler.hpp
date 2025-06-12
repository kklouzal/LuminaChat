// LogHandler.hpp - header-only implementation for unified logging system
// Handles centralized logging with component identification and thread safety.
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
// 1.  Minimalism & Performance: Deliver lean, efficient solutions; do not create or preserve unused helpers or wrappers.
// 2.  Redundancy Elimination: Remove unused, obsolete, and legacy code—including unneeded interfaces and includes.
// 3.  Consistent Style: Adopt a uniform coding style and structure for clarity and maintainability.
// 4.  Documentation: Write concise comments that explain complex logic and key design decisions.
// 5.  Zero Magic & Strong Typing: Replace magic literals with named constants, enums, or constexpr; prefer scoped enums.
// 6.  Function Boundaries: Define clear responsibilities; reduce overlap and avoid unnecessary layers.
// 7.  Core Preservation: Streamline code while safeguarding essential features; favor direct access over extra abstractions.
// 8.  Const-Correctness & Immutability: Mark variables, parameters, and methods as const wherever possible.
// 9.  RAII & Resource Safety: Encapsulate resource acquisition/release in constructors/destructors or smart pointers.
// 10. Standard Library Preference: Favor STL algorithms and containers over custom loops and buffers.
// 11. Cross-Platform Portability: Use fixed-width types and proper initialization to guarantee identical behavior everywhere.
// 12. Thread Safety: Define and document thread-safety contracts; protect shared state with mutexes, atomics, or thread-safe containers.
// 13. Smart Caching: Cache frequently used values to minimize allocations and improve performance.
// 14. Logical Consistency: Verify code flow to ensure coherent, error-free execution paths.
// 15. Continuous Refinement: Regularly refactor and confirm that updates preserve stable functionality.

#pragma once

#include <string>
#include <functional>
#include <mutex>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <sstream>

// Log levels for filtering
enum class LogLevel : int {
    DBG = 0,
    INF = 1,
    WRN = 2,
    ERR = 3
};

// Component identifiers for structured logging
enum class LogComponent {
    MAIN,
    LLAMA_MANAGER,
    DISCORD_MANAGER,
    DISCORD_HISTORY,
    SETTINGS_MANAGER,
    SUMMARIZER_MANAGER,
    UI
};

class LogHandler {
private:
    // Thread-safe callback mechanism
    std::function<void(const std::string&)> output_callback;
    mutable std::mutex callback_mutex;
    
    // Logging configuration
    std::atomic<LogLevel> min_log_level{LogLevel::INF};
    std::atomic<bool> include_timestamps{true};
    std::atomic<bool> include_component_tags{true};
    
    // Component name mapping
    static const char* get_component_name(LogComponent component) {
        switch (component) {
            case LogComponent::MAIN: return "Main";
            case LogComponent::LLAMA_MANAGER: return "LlamaManager";
            case LogComponent::DISCORD_MANAGER: return "Discord";
            case LogComponent::DISCORD_HISTORY: return "DiscordHistory";
            case LogComponent::SETTINGS_MANAGER: return "Settings";
            case LogComponent::SUMMARIZER_MANAGER: return "Summarizer";
            case LogComponent::UI: return "UI";
            default: return "Unknown";
        }
    }
    
    // Level name mapping
    static const char* get_level_name(LogLevel level) {
        switch (level) {
            case LogLevel::DBG: return "DEBUG";
            case LogLevel::INF: return "INFO";
            case LogLevel::WRN: return "WARN";
            case LogLevel::ERR: return "ERROR";
            default: return "UNKNOWN";
        }
    }
    
    // Format message with timestamp and component
    std::string format_message(LogLevel level, LogComponent component, const std::string& message) const {
        std::ostringstream oss;
        
        // Add timestamp if enabled
        if (include_timestamps) {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) % 1000;
            
            oss << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S");
            oss << "." << std::setfill('0') << std::setw(3) << ms.count() << "] ";
        }
        
        // Add component tag if enabled
        if (include_component_tags) {
            oss << "[" << get_component_name(component) << "] ";
        }
        
        // Add level for warnings and errors
        if (level >= LogLevel::WRN) {
            oss << "[" << get_level_name(level) << "] ";
        }
        
        oss << message;
        return oss.str();
    }
    
    // Singleton instance
    static LogHandler& instance() {
        static LogHandler handler;
        return handler;
    }
    
    LogHandler() = default;

public:
    // Delete copy constructor and assignment operator
    LogHandler(const LogHandler&) = delete;
    LogHandler& operator=(const LogHandler&) = delete;
    
    // Set output callback (thread-safe)
    static void set_output_callback(std::function<void(const std::string&)> callback) {
        auto& handler = instance();
        std::lock_guard<std::mutex> lock(handler.callback_mutex);
        handler.output_callback = callback;
    }
    
    // Configuration methods
    static void set_min_log_level(LogLevel level) {
        instance().min_log_level = level;
    }
    
    static void set_include_timestamps(bool include) {
        instance().include_timestamps = include;
    }
    
    static void set_include_component_tags(bool include) {
        instance().include_component_tags = include;
    }
    
    // Main logging method
    static void log(LogLevel level, LogComponent component, const std::string& message) {
        auto& handler = instance();
        
        // Check if we should log this level
        if (level < handler.min_log_level) {
            return;
        }
        
        // Format the message
        std::string formatted = handler.format_message(level, component, message);
        
        // Send to callback if available
        {
            std::lock_guard<std::mutex> lock(handler.callback_mutex);
            if (handler.output_callback) {
                handler.output_callback(formatted + "\n");
            }
        }
    }
    
    // Convenience methods for different levels
    static void debug(LogComponent component, const std::string& message) {
        log(LogLevel::DBG, component, message);
    }
    
    static void info(LogComponent component, const std::string& message) {
        log(LogLevel::INF, component, message);
    }
    
    static void warning(LogComponent component, const std::string& message) {
        log(LogLevel::WRN, component, message);
    }
    
    static void error(LogComponent component, const std::string& message) {
        log(LogLevel::ERR, component, message);
    }
    
    // Component-specific convenience methods
    static void llama_log(const std::string& message, LogLevel level = LogLevel::INF) {
        log(level, LogComponent::LLAMA_MANAGER, message);
    }
    
    static void discord_log(const std::string& message, LogLevel level = LogLevel::INF) {
        log(level, LogComponent::DISCORD_MANAGER, message);
    }
    
    static void discord_history_log(const std::string& message, LogLevel level = LogLevel::INF) {
        log(level, LogComponent::DISCORD_HISTORY, message);
    }
    
    static void settings_log(const std::string& message, LogLevel level = LogLevel::INF) {
        log(level, LogComponent::SETTINGS_MANAGER, message);
    }
    
    static void summarizer_log(const std::string& message, LogLevel level = LogLevel::INF) {
        log(level, LogComponent::SUMMARIZER_MANAGER, message);
    }
    
    static void ui_log(const std::string& message, LogLevel level = LogLevel::INF) {
        log(level, LogComponent::UI, message);
    }
};

// Essential macros only
#define LOG_DEBUG(component, message) LogHandler::debug(LogComponent::component, message)
#define LOG_INFO(component, message) LogHandler::info(LogComponent::component, message)
#define LOG_WARNING(component, message) LogHandler::warning(LogComponent::component, message)
#define LOG_ERROR(component, message) LogHandler::error(LogComponent::component, message)

// Component-specific macros
#define LLAMA_LOG(message) LogHandler::llama_log(message)
#define LLAMA_LOG_ERROR(message) LogHandler::llama_log(message, LogLevel::ERR)
#define DISCORD_LOG(message) LogHandler::discord_log(message)
#define DISCORD_LOG_ERROR(message) LogHandler::discord_log(message, LogLevel::ERR)
#define DISCORD_HISTORY_LOG(message) LogHandler::discord_history_log(message)
#define SETTINGS_LOG(message) LogHandler::settings_log(message)
#define SUMMARIZER_LOG(message) LogHandler::summarizer_log(message)
#define SUMMARIZER_LOG_ERROR(message) LogHandler::summarizer_log(message, LogLevel::ERR)
#define UI_LOG(message) LogHandler::ui_log(message)

//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODING DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//
