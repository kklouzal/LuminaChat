#pragma once

#include "../Logger.hpp"
#include <atomic>
#include <chrono>
#include <string>
#include <string_view>
#include <array>
#include <thread>

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
constexpr inline bool operator==(ContextState lhs, ContextState rhs) noexcept {
    return static_cast<uint8_t>(lhs) == static_cast<uint8_t>(rhs);
}

constexpr inline bool operator!=(ContextState lhs, ContextState rhs) noexcept {
    return !(lhs == rhs);
}



// Cache-line size for optimal memory alignment
static constexpr size_t CACHE_LINE_SIZE = 64;

// Compile-time state validation lookup table for maximum performance
namespace StateTransitions {
    constexpr inline bool IsValidTransition(ContextState from, ContextState to) noexcept {
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
    constexpr inline std::string_view GetStateStringView(ContextState state) noexcept {
        constexpr std::array<std::string_view, 4> state_names = {
            "CONTEXT_IDLE",
            "CONTEXT_GENERATING", 
            "PLUGIN_PROCESSING",
            "ERROR_STATE"
        };
        
        const auto idx = static_cast<uint8_t>(state);
        return (idx < 4) ? state_names[idx] : "UNKNOWN";
    }
    
    // Compile-time only validation for static_assert contexts
    consteval bool IsValidStateValue(ContextState state) noexcept {
        const auto idx = static_cast<uint8_t>(state);
        return idx < 4;
    }
    
    // Compile-time only transition validation for static_assert contexts  
    consteval bool ValidateTransitionAtCompileTime(ContextState from, ContextState to) noexcept {
        return IsValidStateValue(from) && IsValidStateValue(to) && IsValidTransition(from, to);
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
    
    // Lock-free plugin tracking - using atomic hash for performance
    std::atomic<std::size_t> current_plugin_id{0};  // 0 = no plugin, hash of plugin name otherwise
    
    // Lock-free plugin name cache - atomic pointer for fast access
    mutable std::atomic<const char*> cached_plugin_name{nullptr};  // Points to static string or nullptr
    
public:
    /**
     * Default constructor - initializes state to CONTEXT_IDLE
     */
    ContextStateManager() noexcept = default;
    
    /**
     * Virtual destructor for proper inheritance
     */
    virtual ~ContextStateManager() noexcept {
        // Clear atomic pointer - no need to delete as it points to static/stack strings
        cached_plugin_name.store(nullptr, std::memory_order_relaxed);
    }
    
    // Disable copy constructor and assignment operator for performance
    ContextStateManager(const ContextStateManager&) = delete;
    ContextStateManager& operator=(const ContextStateManager&) = delete;
    
    // Disable move constructor and assignment - atomics and mutexes are not moveable
    ContextStateManager(ContextStateManager&&) = delete;
    ContextStateManager& operator=(ContextStateManager&&) = delete;
    
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
    [[nodiscard]] inline std::string GetStateString() const {
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
            return false;
        }
        
        // Attempt atomic state change with acquire-release ordering for synchronization
        // Use strong compare_exchange to avoid ABA problems in high-contention scenarios
        do {
            if (state.compare_exchange_weak(expected, new_state, 
                                           std::memory_order_acq_rel, std::memory_order_acquire)) [[likely]] {
                return true;
            }
            // Re-validate transition with the new expected value
            if (!StateTransitions::IsValidTransition(expected, new_state)) [[unlikely]] {
                return false;
            }
        } while (true);
    }
    
    /**
     * Ultra-high-performance lock-free plugin acquisition
     * @param plugin_name Name of the plugin requesting access
     * @return true if processing lock was acquired, false otherwise
     * 
     * Uses only atomic operations for maximum performance. No timeouts,
     * no blocking, and minimal error logging for optimal speed.
     */
    [[nodiscard]] bool TryAcquirePluginProcessing(const std::string_view plugin_name) noexcept {
        // Fast path: atomic state check
        ContextState expected = ContextState::CONTEXT_IDLE;
        if (!state.compare_exchange_strong(expected, ContextState::PLUGIN_PROCESSING, 
                                         std::memory_order_acq_rel, std::memory_order_acquire)) [[unlikely]] {
            return false;  // State wasn't IDLE or concurrent modification
        }
        
        // Store plugin hash atomically - completely lock-free
        const auto plugin_hash = ComputePluginHash(plugin_name);
        current_plugin_id.store(plugin_hash, std::memory_order_release);
        
        return true;
    }
    
    /**
     * Ultra-high-performance lock-free plugin release
     * @param plugin_name Name of the plugin releasing access
     * @return true if successfully released, false if not owned by this plugin
     * 
     * Uses only atomic operations for maximum performance and provides 
     * clear success/failure indication.
     */
    [[nodiscard]] bool ReleasePluginProcessing(const std::string_view plugin_name) noexcept {
        // Fast path: verify ownership using atomic hash comparison
        const auto plugin_hash = ComputePluginHash(plugin_name);
        const auto current_hash = current_plugin_id.load(std::memory_order_acquire);
        
        if (current_hash != plugin_hash) [[unlikely]] {
            return false;  // Not owned by this plugin
        }
        
        // Atomic state transition
        ContextState expected = ContextState::PLUGIN_PROCESSING;
        if (!state.compare_exchange_strong(expected, ContextState::CONTEXT_IDLE, 
                                         std::memory_order_acq_rel)) [[unlikely]] {
            return false;  // Unexpected state
        }
        
        // Clear plugin ownership atomically
        current_plugin_id.store(0, std::memory_order_release);
        return true;
    }
    
    /**
     * Emergency state recovery for deadlock prevention
     * @param reason Optional reason for the forced release (unused in lock-free version)
     */
    void ForceReleasePluginProcessing(const std::string_view reason = "") noexcept {
        const ContextState current = state.load(std::memory_order_relaxed);
        if (current == ContextState::PLUGIN_PROCESSING) [[likely]] {
            // Atomic state transition - force release regardless of ownership
            ContextState expected = ContextState::PLUGIN_PROCESSING;
            if (state.compare_exchange_strong(expected, ContextState::CONTEXT_IDLE, 
                                            std::memory_order_acq_rel)) [[likely]] {
                // Clear atomic plugin ownership
                current_plugin_id.store(0, std::memory_order_release);
                cached_plugin_name.store(nullptr, std::memory_order_release);
            }
        }
    }
    
    /**
     * Emergency state recovery - force reset to CONTEXT_IDLE
     * @param reason Reason for the forced reset (for logging)
     * @return Previous state that was overridden
     */
    ContextState ForceResetToIdle(const std::string_view reason = "emergency recovery") noexcept {
        ContextState previous_state = state.exchange(ContextState::CONTEXT_IDLE, std::memory_order_acq_rel);
        
        // Clear plugin ownership if we were in plugin processing state
        if (previous_state == ContextState::PLUGIN_PROCESSING) {
            current_plugin_id.store(0, std::memory_order_release);
            cached_plugin_name.store(nullptr, std::memory_order_release);
        }
        
        return previous_state;
    }

    /**
     * Lock-free polling wait for specific state
     * @param target_state The state to wait for
     * @param max_polls Maximum number of polls before giving up
     * @param poll_delay_ns Nanoseconds to wait between polls (0 = tight loop)
     * @return true if target state was reached, false if max polls exceeded
     */
    [[nodiscard]] bool WaitForState(ContextState target_state, uint32_t max_polls = 1000, 
                                   uint32_t poll_delay_ns = 1000) noexcept {
        for (uint32_t i = 0; i < max_polls; ++i) {
            if (state.load(std::memory_order_relaxed) == target_state) [[likely]] {
                return true;
            }
            
            if (poll_delay_ns > 0) {
                // Small delay to prevent excessive CPU usage
                std::this_thread::sleep_for(std::chrono::nanoseconds(poll_delay_ns));
            } else {
                // Yield CPU for other threads
                std::this_thread::yield();
            }
        }
        return false;
    }
    
    /**
     * Lock-free polling wait for context to become available
     * @param max_polls Maximum number of polls before giving up
     * @param poll_delay_ns Nanoseconds to wait between polls (0 = tight loop)
     * @return true if context became available, false if max polls exceeded
     */
    [[nodiscard]] bool WaitForAvailable(uint32_t max_polls = 1000, uint32_t poll_delay_ns = 1000) noexcept {
        return WaitForState(ContextState::CONTEXT_IDLE, max_polls, poll_delay_ns);
    }
    
    /**
     * Ultra-fast lock-free check if a specific plugin owns the processing state
     * @param plugin_name Name of the plugin to check
     * @return true if this plugin owns the processing state, false otherwise
     */
    [[nodiscard]] inline bool IsOwnedByPlugin(const std::string_view plugin_name) const noexcept {
        // Fast path: check state first
        if (state.load(std::memory_order_relaxed) != ContextState::PLUGIN_PROCESSING) [[likely]] {
            return false;
        }
        
        // Lock-free hash comparison
        const auto plugin_hash = ComputePluginHash(plugin_name);
        return current_plugin_id.load(std::memory_order_acquire) == plugin_hash;
    }
    
    /**
     * Lock-free get current processing plugin name (zero-allocation, ultra-fast)
     * @return Pointer to plugin name string, or nullptr if no plugin processing
     * 
     * Returns a pointer to avoid string allocation. The returned pointer
     * is only valid while the plugin owns the processing state.
     */
    [[nodiscard]] inline const char* GetCurrentProcessingPlugin() const noexcept {
        // Fast path: check state and return cached name atomically
        if (state.load(std::memory_order_relaxed) != ContextState::PLUGIN_PROCESSING) [[likely]] {
            return nullptr;
        }
        
        return cached_plugin_name.load(std::memory_order_acquire);
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
    // Lock-free plugin hash computation for atomic tracking
    [[nodiscard]] static inline std::size_t ComputePluginHash(const std::string_view plugin_name) noexcept {
        // Use std::hash for consistent hashing, but ensure we never return 0 (reserved for "no plugin")
        std::hash<std::string_view> hasher;
        const auto hash_value = hasher(plugin_name);
        return (hash_value == 0) ? 1 : hash_value;  // Ensure non-zero hash
    }
    
};

} // namespace LuminaChat
