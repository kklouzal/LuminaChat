#pragma once

// ErrorHandling.hpp - Global Error Handling System using Logger.hpp
// This provides a centralized error handling system that works across the entire codebase
// and optionally integrates with UI components for user notifications.

#include "Logger.hpp"
#include <functional>
#include <memory>
#include <mutex>

namespace LuminaChat {

/**
 * Global Error Handling System
 * 
 * This system provides consistent error, warning, and success handling across
 * the entire LuminaChat codebase. It uses the high-performance Logger.hpp as
 * its foundation and supports optional UI notification callbacks.
 * 
 * Features:
 * - Uses the existing Logger.hpp system for all actual logging
 * - Optional UI notification callbacks for user-facing messages
 * - Available to all components, not just UI code
 * - Consistent formatting and behavior across the application
 * - Exception safety helpers for critical operations
 */
class ErrorHandler {
public:
    // UI notification callback types
    using UINotificationCallback = std::function<void(const std::string& message, const std::string& context)>;
    
    // Get the global ErrorHandler instance
    static ErrorHandler& GetInstance() {
        static ErrorHandler instance;
        return instance;
    }
    
    // Register callbacks for UI notifications (optional)
    void RegisterErrorCallback(UINotificationCallback callback) {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        error_callback_ = std::move(callback);
    }
    
    void RegisterWarningCallback(UINotificationCallback callback) {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        warning_callback_ = std::move(callback);
    }
    
    void RegisterSuccessCallback(UINotificationCallback callback) {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        success_callback_ = std::move(callback);
    }
    
    void RegisterInfoCallback(UINotificationCallback callback) {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        info_callback_ = std::move(callback);
    }
    
    // Primary error handling methods
    void HandleError(const std::string& message, const std::string& context = "", bool notify_ui = true) {
        // Always log the error using the Logger system
        if (context.empty()) {
            LOG_ERROR("ErrorHandler", message);
        } else {
            LOG_ERROR(context, message);
        }
        
        // Optionally notify UI
        if (notify_ui) {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (error_callback_) {
                error_callback_(message, context);
            }
        }
    }
    
    void HandleWarning(const std::string& message, const std::string& context = "", bool notify_ui = true) {
        // Always log the warning using the Logger system
        if (context.empty()) {
            LOG_WARNING("ErrorHandler", message);
        } else {
            LOG_WARNING(context, message);
        }
        
        // Optionally notify UI
        if (notify_ui) {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (warning_callback_) {
                warning_callback_(message, context);
            }
        }
    }
    
    void HandleSuccess(const std::string& message, const std::string& context = "", bool notify_ui = true) {
        // Log as info (success is not a log level in Logger.hpp)
        if (context.empty()) {
            LOG_INFO("ErrorHandler", message);
        } else {
            LOG_INFO(context, message);
        }
        
        // Optionally notify UI
        if (notify_ui) {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (success_callback_) {
                success_callback_(message, context);
            }
        }
    }
    
    void HandleInfo(const std::string& message, const std::string& context = "", bool notify_ui = false) {
        // Always log the info using the Logger system
        if (context.empty()) {
            LOG_INFO("ErrorHandler", message);
        } else {
            LOG_INFO(context, message);
        }
        
        // Optionally notify UI (default false for info)
        if (notify_ui) {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            if (info_callback_) {
                info_callback_(message, context);
            }
        }
    }
    
    // Exception safety helpers
    template<typename Func>
    void SafeExecute(Func&& operation, const std::string& operation_name, const std::string& context = "") {
        try {
            operation();
        } catch (const std::exception& e) {
            HandleError("Exception in " + operation_name + ": " + e.what(), context, true);
        } catch (...) {
            HandleError("Unknown exception in " + operation_name, context, true);
        }
    }
    
    // Utility for operations that should not fail
    template<typename Func>
    bool TryExecute(Func&& operation, const std::string& operation_name, const std::string& context = "") {
        try {
            operation();
            return true;
        } catch (const std::exception& e) {
            HandleError("Failed to " + operation_name + ": " + e.what(), context, false);
            return false;
        } catch (...) {
            HandleError("Failed to " + operation_name + ": unknown exception", context, false);
            return false;
        }
    }

private:
    ErrorHandler() = default;
    ~ErrorHandler() = default;
    
    // Non-copyable, non-movable
    ErrorHandler(const ErrorHandler&) = delete;
    ErrorHandler& operator=(const ErrorHandler&) = delete;
    ErrorHandler(ErrorHandler&&) = delete;
    ErrorHandler& operator=(ErrorHandler&&) = delete;
    
    std::mutex callback_mutex_;
    UINotificationCallback error_callback_;
    UINotificationCallback warning_callback_;
    UINotificationCallback success_callback_;
    UINotificationCallback info_callback_;
};

} // namespace LuminaChat

// Global convenience functions for use throughout the codebase
inline void HandleError(const std::string& message, const std::string& context = "", bool notify_ui = true) {
    LuminaChat::ErrorHandler::GetInstance().HandleError(message, context, notify_ui);
}

inline void HandleWarning(const std::string& message, const std::string& context = "", bool notify_ui = true) {
    LuminaChat::ErrorHandler::GetInstance().HandleWarning(message, context, notify_ui);
}

inline void HandleSuccess(const std::string& message, const std::string& context = "", bool notify_ui = true) {
    LuminaChat::ErrorHandler::GetInstance().HandleSuccess(message, context, notify_ui);
}

inline void HandleInfo(const std::string& message, const std::string& context = "", bool notify_ui = false) {
    LuminaChat::ErrorHandler::GetInstance().HandleInfo(message, context, notify_ui);
}

template<typename Func>
void SafeExecute(Func&& operation, const std::string& operation_name, const std::string& context = "") {
    LuminaChat::ErrorHandler::GetInstance().SafeExecute(std::forward<Func>(operation), operation_name, context);
}

template<typename Func>
bool TryExecute(Func&& operation, const std::string& operation_name, const std::string& context = "") {
    return LuminaChat::ErrorHandler::GetInstance().TryExecute(std::forward<Func>(operation), operation_name, context);
}

// Convenience macros for common scenarios
#define HANDLE_ERROR(message) HandleError(message, "", true)
#define HANDLE_WARNING(message) HandleWarning(message, "", true)
#define HANDLE_SUCCESS(message) HandleSuccess(message, "", true)
#define HANDLE_INFO(message) HandleInfo(message, "", false)

#define HANDLE_ERROR_CTX(message, context) HandleError(message, context, true)
#define HANDLE_WARNING_CTX(message, context) HandleWarning(message, context, true)
#define HANDLE_SUCCESS_CTX(message, context) HandleSuccess(message, context, true)
#define HANDLE_INFO_CTX(message, context) HandleInfo(message, context, false)

#define SAFE_EXECUTE(operation, name) SafeExecute([&]() { operation; }, name, "")
#define SAFE_EXECUTE_CTX(operation, name, context) SafeExecute([&]() { operation; }, name, context)

#define TRY_EXECUTE(operation, name) TryExecute([&]() { operation; }, name, "")
#define TRY_EXECUTE_CTX(operation, name, context) TryExecute([&]() { operation; }, name, context)
