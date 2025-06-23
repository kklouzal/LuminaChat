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
#include <functional>
#include <mutex>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <optional>
#include <shared_mutex>
#include <array>
#include <charconv>

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
    LLAMA_CONTEXT,
    LLAMA_RESPONSE,
    CONTEXT_SIZE_MANAGER,
    TOKEN_CACHE,
    DISCORD_MANAGER,
    DISCORD_HISTORY,
    SETTINGS_MANAGER,
    SUMMARIZER_MANAGER,
    BLACKLIST_MANAGER,
    SANITIZER,
    UI,
    PERFORMANCE
};

class LogHandler {
private:
    // Thread-safe callback mechanism - optimized with shared_mutex for better read performance
    std::optional<std::function<void(std::string_view)>> output_callback;
    std::optional<std::function<void(std::string_view)>> summarizer_callback;
    mutable std::shared_mutex callback_mutex;
    
    // Dynamic logging configuration - allows runtime debug level adjustment
    std::atomic<LogLevel> min_level{LogLevel::INF};
    static constexpr bool INCLUDE_TIMESTAMPS = true;
    static constexpr bool INCLUDE_COMPONENT_TAGS = true;
    
    // Performance tracing for hot paths - minimal overhead when disabled
    std::atomic<bool> performance_tracing_enabled{false};
      // Performance optimization: cache time formatting to avoid repeated work
    mutable std::chrono::seconds last_time_cache{0};
    mutable std::array<char, 17> time_cache{};  // Pre-allocated buffer for time string "[HH:MM:SS.mmm] " + null
    mutable std::mutex time_cache_mutex;
      // Fast integer to string conversion using C++17 std::to_chars
    static constexpr size_t format_milliseconds(char* buffer, int ms) noexcept {
        // Fast path for common cases
        if (ms < 10) {
            buffer[0] = '0';
            buffer[1] = '0';
            buffer[2] = '0' + static_cast<char>(ms);
            return 3;
        } else if (ms < 100) {
            buffer[0] = '0';
            buffer[1] = '0' + static_cast<char>(ms / 10);
            buffer[2] = '0' + static_cast<char>(ms % 10);
            return 3;
        } else {
            buffer[0] = '0' + static_cast<char>(ms / 100);
            buffer[1] = '0' + static_cast<char>((ms / 10) % 10);
            buffer[2] = '0' + static_cast<char>(ms % 10);
            return 3;
        }
    }
    
    // Optimized time string formatting with caching
    std::string_view get_formatted_time() const {
        if constexpr (!INCLUDE_TIMESTAMPS) {
            return {};
        }
        
        const auto now = std::chrono::system_clock::now();
        const auto now_seconds = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch());
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
          // Use cached time if it's the same second
        {
            std::lock_guard<std::mutex> lock(time_cache_mutex);
            if (now_seconds == last_time_cache) {
                // Update only milliseconds part (positions 10, 11, 12)
                format_milliseconds(time_cache.data() + 10, static_cast<int>(ms.count()));
                return std::string_view(time_cache.data(), 15); // "[HH:MM:SS.mmm] "
            }
            
            // Format new time
            const auto time_t = std::chrono::system_clock::to_time_t(now);
            std::tm tm_buf{};
#ifdef _WIN32
            localtime_s(&tm_buf, &time_t);
#else
            localtime_r(&time_t, &tm_buf);
#endif
            
            // Fast formatting using direct char manipulation
            time_cache[0] = '[';
            time_cache[1] = '0' + static_cast<char>(tm_buf.tm_hour / 10);
            time_cache[2] = '0' + static_cast<char>(tm_buf.tm_hour % 10);
            time_cache[3] = ':';
            time_cache[4] = '0' + static_cast<char>(tm_buf.tm_min / 10);
            time_cache[5] = '0' + static_cast<char>(tm_buf.tm_min % 10);
            time_cache[6] = ':';
            time_cache[7] = '0' + static_cast<char>(tm_buf.tm_sec / 10);
            time_cache[8] = '0' + static_cast<char>(tm_buf.tm_sec % 10);
            time_cache[9] = '.';
            format_milliseconds(time_cache.data() + 10, static_cast<int>(ms.count()));
            time_cache[13] = ']';
            time_cache[14] = ' ';
            time_cache[15] = '\0';
            
            last_time_cache = now_seconds;
            return std::string_view(time_cache.data(), 15); // "[HH:MM:SS.mmm] "
        }
    }    // Component name mapping - constexpr for compile-time optimization
    static constexpr std::string_view get_component_name(LogComponent component) noexcept {
        switch (component) {
            case LogComponent::MAIN: return "Main";
            case LogComponent::LLAMA_MANAGER: return "LlamaManager";
            case LogComponent::LLAMA_CONTEXT: return "LlamaContext";
            case LogComponent::LLAMA_RESPONSE: return "LlamaResponse";
            case LogComponent::CONTEXT_SIZE_MANAGER: return "ContextSizeManager";
            case LogComponent::TOKEN_CACHE: return "TokenCache";
            case LogComponent::DISCORD_MANAGER: return "Discord";
            case LogComponent::DISCORD_HISTORY: return "DiscordHistory";
            case LogComponent::SETTINGS_MANAGER: return "Settings";
            case LogComponent::SUMMARIZER_MANAGER: return "Summarizer";
            case LogComponent::BLACKLIST_MANAGER: return "BlacklistManager";
            case LogComponent::SANITIZER: return "Sanitizer";
            case LogComponent::UI: return "UI";
            case LogComponent::PERFORMANCE: return "Performance";
            default: return "Unknown";
        }
    }
    
    // Level name mapping - constexpr for compile-time optimization
    static constexpr std::string_view get_level_name(LogLevel level) noexcept {
        switch (level) {
            case LogLevel::DBG: return "DEBUG";
            case LogLevel::INF: return "INFO";
            case LogLevel::WRN: return "WARN";
            case LogLevel::ERR: return "ERROR";
            default: return "UNKNOWN";
        }
    }    // Optimized message formatting using direct string operations instead of streams
    std::string format_message(LogLevel level, LogComponent component, std::string_view message) const {
        // Pre-calculate approximate size to reduce allocations
        size_t estimated_size = message.length();
        if constexpr (INCLUDE_TIMESTAMPS) {
            estimated_size += 16; // "[HH:MM:SS.mmm] "
        }
        if constexpr (INCLUDE_COMPONENT_TAGS) {
            estimated_size += get_component_name(component).length() + 3; // "[component] "
        }
        if (level >= LogLevel::WRN) {
            estimated_size += get_level_name(level).length() + 3; // "[level] "
        }
        
        std::string result;
        result.reserve(estimated_size);
        
        // Add timestamp if enabled - using cached formatting
        if constexpr (INCLUDE_TIMESTAMPS) {
            const auto time_str = get_formatted_time();
            result.append(time_str);
        }
        
        // Add component tag if enabled
        if constexpr (INCLUDE_COMPONENT_TAGS) {
            result.append("[");
            result.append(get_component_name(component));
            result.append("] ");
        }
        
        // Add level for warnings and errors only
        if (level >= LogLevel::WRN) [[unlikely]] {
            result.append("[");
            result.append(get_level_name(level));
            result.append("] ");
        }
        
        result.append(message);
        return result;
    }
      // Singleton instance
    static LogHandler& instance() noexcept {
        static LogHandler handler;
        return handler;
    }
    
    LogHandler() = default;

public:
    // Delete copy constructor and assignment operator for singleton pattern
    LogHandler(const LogHandler&) = delete;
    LogHandler& operator=(const LogHandler&) = delete;
      // Set output callback (thread-safe) - optimized with move semantics and optional
    static void set_output_callback(std::function<void(std::string_view)> callback) {
        auto& handler = instance();
        std::unique_lock<std::shared_mutex> lock(handler.callback_mutex);
        handler.output_callback = std::move(callback);
    }
    
    // Set summarizer-specific callback (thread-safe) - optimized with move semantics and optional
    static void set_summarizer_callback(std::function<void(std::string_view)> callback) {
        auto& handler = instance();
        std::unique_lock<std::shared_mutex> lock(handler.callback_mutex);
        handler.summarizer_callback = std::move(callback);    }
      // Main logging method - optimized with shared_mutex and minimal string copies
    static void log(LogLevel level, LogComponent component, std::string_view message) {
        auto& handler = instance();
        
        // Early exit for filtered levels - branch prediction hint
        if (level < handler.min_level.load(std::memory_order_relaxed)) [[likely]] {
            return;
        }
        
        // Format the message once
        std::string formatted = handler.format_message(level, component, message);
        formatted.append("\n"); // Single append instead of concatenation
        
        // Use shared_lock for better read performance, convert to string_view for zero-copy
        std::shared_lock<std::shared_mutex> lock(handler.callback_mutex);
          // Route summarizer logs to specialized callback if available
        if (component == LogComponent::SUMMARIZER_MANAGER && handler.summarizer_callback.has_value()) [[unlikely]] {
            std::invoke(handler.summarizer_callback.value(), std::string_view{formatted});
        } else if (handler.output_callback.has_value()) [[likely]] {
            std::invoke(handler.output_callback.value(), std::string_view{formatted});
        }
    }
      // Core logging methods - const-correct parameters
    static void debug(LogComponent component, std::string_view message) {
        log(LogLevel::DBG, component, message);
    }
    
    static void info(LogComponent component, std::string_view message) {
        log(LogLevel::INF, component, message);
    }
    
    static void warning(LogComponent component, std::string_view message) {
        log(LogLevel::WRN, component, message);
    }
      static void error(LogComponent component, std::string_view message) {
        log(LogLevel::ERR, component, message);
    }
    
    // Dynamic logging level control for runtime debug adjustment
    static void set_min_log_level(LogLevel level) {
        auto& handler = instance();
        handler.min_level.store(level, std::memory_order_relaxed);
    }
    
    static LogLevel get_min_log_level() {
        auto& handler = instance();
        return handler.min_level.load(std::memory_order_relaxed);
    }
    
    // Performance tracing control for hot path debugging
    static void enable_performance_tracing(bool enable = true) {
        auto& handler = instance();
        handler.performance_tracing_enabled.store(enable, std::memory_order_relaxed);
    }
    
    static bool is_performance_tracing_enabled() {
        auto& handler = instance();
        return handler.performance_tracing_enabled.load(std::memory_order_relaxed);
    }
    
    // Fast debug level check for conditional debug code
    static bool is_debug_enabled() {
        auto& handler = instance();
        return handler.min_level.load(std::memory_order_relaxed) <= LogLevel::DBG;
    }
};

// Unified logging macros - streamlined per directive #2 (Redundancy Elimination)
#define LOG_DEBUG(component, message) LogHandler::debug(LogComponent::component, message)
#define LOG_INFO(component, message) LogHandler::info(LogComponent::component, message)
#define LOG_WARNING(component, message) LogHandler::warning(LogComponent::component, message)
#define LOG_ERROR(component, message) LogHandler::error(LogComponent::component, message)

// Performance tracing macros for hot paths - minimal overhead when disabled
#define PERF_TRACE(component, message) \
    do { if (LogHandler::is_performance_tracing_enabled()) [[unlikely]] { \
        LogHandler::debug(LogComponent::component, "[PERF] " message); \
    } } while(0)

// Conditional debug macros for expensive debug operations
#define DEBUG_IF_ENABLED(component, code) \
    do { if (LogHandler::is_debug_enabled()) [[unlikely]] { code; } } while(0)

#define DEBUG_LOG_IF_ENABLED(component, message) \
    do { if (LogHandler::is_debug_enabled()) [[unlikely]] { \
        LogHandler::debug(LogComponent::component, message); \
    } } while(0)

// Component-specific convenience macros - for backward compatibility and convenience
#define LLAMA_LOG(message) LogHandler::info(LogComponent::LLAMA_MANAGER, message)
#define LLAMA_LOG_DEBUG(message) LogHandler::debug(LogComponent::LLAMA_MANAGER, message)
#define LLAMA_LOG_ERROR(message) LogHandler::error(LogComponent::LLAMA_MANAGER, message)

#define LLAMA_CONTEXT_LOG(message) LogHandler::info(LogComponent::LLAMA_CONTEXT, message)
#define LLAMA_CONTEXT_LOG_DEBUG(message) LogHandler::debug(LogComponent::LLAMA_CONTEXT, message)
#define LLAMA_CONTEXT_LOG_ERROR(message) LogHandler::error(LogComponent::LLAMA_CONTEXT, message)

#define LLAMA_RESPONSE_LOG(message) LogHandler::info(LogComponent::LLAMA_RESPONSE, message)
#define LLAMA_RESPONSE_LOG_DEBUG(message) LogHandler::debug(LogComponent::LLAMA_RESPONSE, message)
#define LLAMA_RESPONSE_LOG_ERROR(message) LogHandler::error(LogComponent::LLAMA_RESPONSE, message)

#define CONTEXT_SIZE_LOG(message) LogHandler::info(LogComponent::CONTEXT_SIZE_MANAGER, message)
#define CONTEXT_SIZE_LOG_DEBUG(message) LogHandler::debug(LogComponent::CONTEXT_SIZE_MANAGER, message)
#define CONTEXT_SIZE_LOG_ERROR(message) LogHandler::error(LogComponent::CONTEXT_SIZE_MANAGER, message)

#define TOKEN_CACHE_LOG(message) LogHandler::info(LogComponent::TOKEN_CACHE, message)
#define TOKEN_CACHE_LOG_DEBUG(message) LogHandler::debug(LogComponent::TOKEN_CACHE, message)
#define TOKEN_CACHE_LOG_ERROR(message) LogHandler::error(LogComponent::TOKEN_CACHE, message)

#define DISCORD_LOG(message) LogHandler::info(LogComponent::DISCORD_MANAGER, message)
#define DISCORD_LOG_DEBUG(message) LogHandler::debug(LogComponent::DISCORD_MANAGER, message)
#define DISCORD_LOG_ERROR(message) LogHandler::error(LogComponent::DISCORD_MANAGER, message)

#define DISCORD_HISTORY_LOG(message) LogHandler::info(LogComponent::DISCORD_HISTORY, message)
#define DISCORD_HISTORY_LOG_DEBUG(message) LogHandler::debug(LogComponent::DISCORD_HISTORY, message)
#define DISCORD_HISTORY_LOG_ERROR(message) LogHandler::error(LogComponent::DISCORD_HISTORY, message)

#define SETTINGS_LOG(message) LogHandler::info(LogComponent::SETTINGS_MANAGER, message)
#define SETTINGS_LOG_DEBUG(message) LogHandler::debug(LogComponent::SETTINGS_MANAGER, message)
#define SETTINGS_LOG_ERROR(message) LogHandler::error(LogComponent::SETTINGS_MANAGER, message)

#define SUMMARIZER_LOG(message) LogHandler::info(LogComponent::SUMMARIZER_MANAGER, message)
#define SUMMARIZER_LOG_DEBUG(message) LogHandler::debug(LogComponent::SUMMARIZER_MANAGER, message)
#define SUMMARIZER_LOG_ERROR(message) LogHandler::error(LogComponent::SUMMARIZER_MANAGER, message)

#define BLACKLIST_LOG(message) LogHandler::info(LogComponent::BLACKLIST_MANAGER, message)
#define BLACKLIST_LOG_DEBUG(message) LogHandler::debug(LogComponent::BLACKLIST_MANAGER, message)
#define BLACKLIST_LOG_ERROR(message) LogHandler::error(LogComponent::BLACKLIST_MANAGER, message)

#define SANITIZER_LOG(message) LogHandler::info(LogComponent::SANITIZER, message)
#define SANITIZER_LOG_DEBUG(message) LogHandler::debug(LogComponent::SANITIZER, message)
#define SANITIZER_LOG_ERROR(message) LogHandler::error(LogComponent::SANITIZER, message)

#define UI_LOG(message) LogHandler::info(LogComponent::UI, message)
#define UI_LOG_DEBUG(message) LogHandler::debug(LogComponent::UI, message)
#define UI_LOG_ERROR(message) LogHandler::error(LogComponent::UI, message)

#define MAIN_LOG(message) LogHandler::info(LogComponent::MAIN, message)
#define MAIN_LOG_DEBUG(message) LogHandler::debug(LogComponent::MAIN, message)
#define MAIN_LOG_ERROR(message) LogHandler::error(LogComponent::MAIN, message)

#define PERFORMANCE_LOG(message) LogHandler::info(LogComponent::PERFORMANCE, message)
#define PERFORMANCE_LOG_DEBUG(message) LogHandler::debug(LogComponent::PERFORMANCE, message)
#define PERFORMANCE_LOG_ERROR(message) LogHandler::error(LogComponent::PERFORMANCE, message)

//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODING DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//
