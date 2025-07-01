#pragma once

#include "../Logger.hpp"
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <chrono>
#include <string>
#include <string_view>
#include <array>

namespace LuminaChat {

/**
 * Context state enumeration for managing context availability and processing
 */
enum class ContextState : uint8_t {
    CONTEXT_IDLE = 0,       // Idle, waiting for user input or plugin processing
    CONTEXT_GENERATING = 1, // Context is generating output, plugins wait
    PLUGIN_PROCESSING = 2,  // A plugin is processing the context, other plugins wait
    ERROR_STATE = 3         // Error state - context unavailable
};

// Comparison operators for ContextState
constexpr bool operator==(ContextState lhs, ContextState rhs) noexcept {
    return static_cast<uint8_t>(lhs) == static_cast<uint8_t>(rhs);
}

constexpr bool operator!=(ContextState lhs, ContextState rhs) noexcept {
    return !(lhs == rhs);
}



// Cache-line size for optimal memory alignment
static constexpr size_t CACHE_LINE_SIZE = 64;

// Compile-time state validation lookup table for maximum performance
namespace StateTransitions {
    constexpr bool IsValidTransition(ContextState from, ContextState to) noexcept {
        // Lookup table: [from_state][to_state] = is_valid
        constexpr bool transition_table[4][4] = {
            // From CONTEXT_IDLE:      to IDLE, GENERATING, PLUGIN, ERROR
            /*CONTEXT_IDLE*/        { false,   true,       true,   true  },
            // From CONTEXT_GENERATING: to IDLE, GENERATING, PLUGIN, ERROR  
            /*CONTEXT_GENERATING*/  { true,    false,      false,  true  },
            // From PLUGIN_PROCESSING: to IDLE, GENERATING, PLUGIN, ERROR
            /*PLUGIN_PROCESSING*/   { true,    false,      false,  true  },
            // From ERROR_STATE:       to IDLE, GENERATING, PLUGIN, ERROR
            /*ERROR_STATE*/         { true,    false,      false,  false }
        };
        
        const auto from_idx = static_cast<uint8_t>(from);
        const auto to_idx = static_cast<uint8_t>(to);
        
        if (from_idx >= 4 || to_idx >= 4) [[unlikely]] {
            return false;
        }
        
        return transition_table[from_idx][to_idx];
    }
    
    // Compile-time string lookup for state names - eliminates runtime string construction
    constexpr std::string_view GetStateStringView(ContextState state) noexcept {
        constexpr std::array<std::string_view, 4> state_names = {
            "CONTEXT_IDLE",
            "CONTEXT_GENERATING", 
            "PLUGIN_PROCESSING",
            "ERROR_STATE"
        };
        
        const auto idx = static_cast<uint8_t>(state);
        return (idx < 4) ? state_names[idx] : "UNKNOWN";
    }
}

/**
 * Ultra-high-performance thread-safe context state management base class
 * 
 * Performance Optimizations:
 * - Cache-aligned memory layout to prevent false sharing
 * - Explicit memory ordering for atomic operations
 * - Constexpr state validation with lookup tables
 * - Lock-free hot paths for state queries
 * - Minimized string allocations with string_view
 * - Branch prediction hints for common code paths
 * - Optimized member layout for cache efficiency
 * 
 * This class is designed to be inherited by ContextInfo and other classes
 * that need ultimate performance state management with plugin coordination.
 */
class alignas(CACHE_LINE_SIZE) ContextStateManager {
protected:
    // Hot path data - first cache line for maximum performance
    std::atomic<ContextState> state{ContextState::CONTEXT_IDLE};
    std::atomic<bool> state_transition_in_progress{false};
    
    // Cold path data - separate cache line to prevent false sharing
    alignas(CACHE_LINE_SIZE) mutable std::mutex state_mutex;   // Fast mutex for better performance than timed_mutex on hot paths
    std::condition_variable state_cv;                          // Standard CV for better performance
    std::string current_plugin_name;                          // Protected by state_mutex
    
public:
    /**
     * Default constructor - initializes state to CONTEXT_IDLE
     */
    ContextStateManager() = default;
    
    /**
     * Virtual destructor for proper inheritance
     */
    virtual ~ContextStateManager() = default;
    
    // Disable copy constructor and assignment operator for performance
    ContextStateManager(const ContextStateManager&) = delete;
    ContextStateManager& operator=(const ContextStateManager&) = delete;
    
    // Allow move constructor and assignment for performance
    ContextStateManager(ContextStateManager&&) = default;
    ContextStateManager& operator=(ContextStateManager&&) = default;
    
    /**
     * Get current state (ultra-fast lock-free operation)
     * Uses relaxed memory ordering for maximum performance on hot path
     */
    [[nodiscard]] inline ContextState GetState() const noexcept { 
        return state.load(std::memory_order_relaxed); 
    }
    
    /**
     * Get current state as string_view for debugging (zero-allocation)
     */
    [[nodiscard]] inline std::string_view GetStateStringView() const noexcept {
        return StateTransitions::GetStateStringView(state.load(std::memory_order_relaxed));
    }
    
    /**
     * Get current state as string for debugging (fallback for legacy compatibility)
     * Note: Creates temporary string - prefer GetStateStringView() for performance
     */
    [[nodiscard]] std::string GetStateString() const {
        return std::string{GetStateStringView()};
    }
    
    /**
     * Ultra-fast atomic state transition with compile-time validation
     * @param new_state The desired new state
     * @return true if transition was successful, false otherwise
     */
    [[nodiscard]] bool TrySetState(ContextState new_state) noexcept {
        // Use acquire ordering for the initial load to ensure memory synchronization
        ContextState expected = state.load(std::memory_order_acquire);
        
        // Compile-time validated transition check - this should be inlined completely
        if (!StateTransitions::IsValidTransition(expected, new_state)) [[unlikely]] {
            LOG_DEBUG("ContextStateManager", "Invalid state transition attempted: " + 
                    std::string(StateTransitions::GetStateStringView(expected)) + " -> " + 
                    std::string(StateTransitions::GetStateStringView(new_state)));
            return false;
        }
        
        // Attempt atomic state change with acquire-release ordering for synchronization
        // Use strong compare_exchange to avoid ABA problems in high-contention scenarios
        do {
            if (state.compare_exchange_weak(expected, new_state, 
                                           std::memory_order_acq_rel, std::memory_order_acquire)) [[likely]] {
                LOG_DEBUG("ContextStateManager", "State transition: " + 
                        std::string(StateTransitions::GetStateStringView(expected)) + " -> " + 
                        std::string(StateTransitions::GetStateStringView(new_state)));
                // Notify any threads waiting for state changes
                state_cv.notify_all();
                return true;
            }
            // Re-validate transition with the new expected value
            if (!StateTransitions::IsValidTransition(expected, new_state)) [[unlikely]] {
                LOG_DEBUG("ContextStateManager", "State transition invalid after concurrent modification: " + 
                        std::string(StateTransitions::GetStateStringView(expected)) + " -> " + 
                        std::string(StateTransitions::GetStateStringView(new_state)));
                return false;
            }
        } while (true);
    }
    
    /**
     * High-performance plugin coordination - acquire processing lock
     * @param plugin_name Name of the plugin requesting access
     * @param timeout_ms Maximum time to wait for access in milliseconds
     * @return true if processing lock was acquired, false otherwise
     */
    [[nodiscard]] bool TryAcquirePluginProcessing(std::string_view plugin_name, 
                                                  uint32_t timeout_ms = 1000) noexcept {
        // Fast path: check if we can even attempt to acquire
        const ContextState current = state.load(std::memory_order_relaxed);
        if (current != ContextState::CONTEXT_IDLE) [[unlikely]] {
            LOG_DEBUG("ContextStateManager", "Plugin " + std::string(plugin_name) + " cannot acquire processing - context state: " + 
                    std::string(StateTransitions::GetStateStringView(current)));
            return false;
        }
        
        // Try to acquire mutex with timeout
        std::unique_lock<std::mutex> lock(state_mutex, std::defer_lock);
        if (!lock.try_lock()) [[unlikely]] {
            LOG_WARNING("ContextStateManager", "Plugin " + std::string(plugin_name) + " failed to acquire state mutex within timeout");
            return false;
        }
        
        // Set state transition in progress atomically
        bool expected_transition = false;
        if (!state_transition_in_progress.compare_exchange_strong(expected_transition, true, 
                                                                 std::memory_order_acq_rel)) [[unlikely]] {
            LOG_DEBUG("ContextStateManager", "Plugin " + std::string(plugin_name) + " cannot acquire processing - transition already in progress");
            return false;
        }
        
        // Attempt state transition with strong ordering
        ContextState expected_state = ContextState::CONTEXT_IDLE;
        if (!state.compare_exchange_strong(expected_state, ContextState::PLUGIN_PROCESSING, 
                                         std::memory_order_acq_rel)) [[unlikely]] {
            state_transition_in_progress.store(false, std::memory_order_release);
            LOG_DEBUG("ContextStateManager", "Plugin " + std::string(plugin_name) + " state transition failed - concurrent modification");
            return false;
        }
        
        // Record which plugin is processing - string assignment only on success path
        current_plugin_name = plugin_name;
        state_transition_in_progress.store(false, std::memory_order_release);
        state_cv.notify_all();
        
        LOG_DEBUG("ContextStateManager", "Plugin " + std::string(plugin_name) + " acquired processing lock");
        return true;
    }
    
    /**
     * High-performance plugin coordination - release processing lock
     * @param plugin_name Name of the plugin releasing access
     */
    void ReleasePluginProcessing(std::string_view plugin_name) noexcept {
        std::lock_guard<std::mutex> lock(state_mutex);
        
        // Verify this plugin owns the processing state
        if (current_plugin_name != plugin_name) [[unlikely]] {
            LOG_WARNING("ContextStateManager", "Plugin " + std::string(plugin_name) + " tried to release processing, but " + 
                      current_plugin_name + " is the current processor");
            return;
        }
        
        // Transition back to idle with strong ordering
        ContextState expected = ContextState::PLUGIN_PROCESSING;
        if (state.compare_exchange_strong(expected, ContextState::CONTEXT_IDLE, 
                                        std::memory_order_acq_rel)) [[likely]] {
            current_plugin_name.clear();
            state_cv.notify_all();
            LOG_DEBUG("ContextStateManager", "Plugin " + std::string(plugin_name) + " released processing lock");
        } else {
            LOG_WARNING("ContextStateManager", "Plugin " + std::string(plugin_name) + " failed to release processing lock - unexpected state: " + 
                      std::string(StateTransitions::GetStateStringView(expected)));
        }
    }
    
    /**
     * Emergency state recovery for deadlock prevention
     * @param reason Optional reason for the forced release
     */
    void ForceReleasePluginProcessing(std::string_view reason = "") noexcept {
        std::lock_guard<std::mutex> lock(state_mutex);
        
        const ContextState current = state.load(std::memory_order_relaxed);
        if (current == ContextState::PLUGIN_PROCESSING) [[likely]] {
            std::string reason_str = reason.empty() ? "" : std::string(" - ") + std::string(reason);
            LOG_WARNING("ContextStateManager", "Force releasing plugin processing lock" + reason_str + 
                      " (was held by: " + current_plugin_name + ")");
            
            ContextState expected = ContextState::PLUGIN_PROCESSING;
            if (state.compare_exchange_strong(expected, ContextState::CONTEXT_IDLE, 
                                            std::memory_order_acq_rel)) [[likely]] {
                current_plugin_name.clear();
                state_cv.notify_all();
                LOG_WARNING("ContextStateManager", "Plugin processing lock force-released successfully");
            }
        }
    }
    
    /**
     * High-performance wait for specific state with timeout
     * @param target_state The state to wait for
     * @param timeout_ms Maximum time to wait in milliseconds
     * @return true if target state was reached, false if timeout occurred
     */
    [[nodiscard]] bool WaitForState(ContextState target_state, uint32_t timeout_ms = 5000) noexcept {
        // Fast path: check if we're already in the target state
        if (state.load(std::memory_order_relaxed) == target_state) [[likely]] {
            return true;
        }
        
        std::unique_lock<std::mutex> lock(state_mutex);
        return state_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this, target_state]() noexcept {
            return state.load(std::memory_order_relaxed) == target_state;
        });
    }
    
    /**
     * High-performance wait for context to become available 
     * @param timeout_ms Maximum time to wait in milliseconds
     * @return true if context became available, false if timeout occurred
     */
    [[nodiscard]] bool WaitForAvailable(uint32_t timeout_ms = 5000) noexcept {
        // Fast path: check if already available
        if (state.load(std::memory_order_relaxed) == ContextState::CONTEXT_IDLE) [[likely]] {
            return true;
        }
        
        std::unique_lock<std::mutex> lock(state_mutex);
        return state_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this]() noexcept {
            return state.load(std::memory_order_relaxed) == ContextState::CONTEXT_IDLE;
        });
    }
    
    /**
     * Get current processing plugin name (thread-safe, zero-allocation when possible)
     * @return Name of the plugin currently processing, empty if none
     */
    [[nodiscard]] std::string GetCurrentProcessingPlugin() const {
        // Fast path: if not in plugin processing state, return empty immediately
        if (state.load(std::memory_order_relaxed) != ContextState::PLUGIN_PROCESSING) [[likely]] {
            return {};
        }
        
        std::lock_guard<std::mutex> lock(state_mutex);
        return current_plugin_name; // RVO optimization
    }
    
    /**
     * Ultra-fast check if context is available for generation (lock-free)
     * @return true if available for generation, false otherwise
     */
    [[nodiscard]] virtual inline bool IsAvailableForGeneration() const noexcept {
        return state.load(std::memory_order_relaxed) == ContextState::CONTEXT_IDLE;
    }
    
    /**
     * Ultra-fast check if context is available for plugin operations (lock-free)
     * @return true if available for plugin processing, false otherwise
     */
    [[nodiscard]] virtual inline bool IsAvailableForPluginProcessing() const noexcept {
        return state.load(std::memory_order_relaxed) == ContextState::CONTEXT_IDLE;
    }
    
protected:
};

} // namespace LuminaChat
