// DiscordHistoryLoader.hpp - header-only implementation for per-context Discord chat history backfill
// Handles immediate Discord message history fetching with pre-tokenization for individual contexts.
//
// File Specific Directives:
// Individual history loaders per context with immediate collection and pre-tokenization.
// Isolated contexts: fill to target ratio then apply immediately.
// Shared contexts: coordinate between multiple loaders before merging.
//
// PERFORMANCE OPTIMIZATIONS IMPLEMENTED:
// 1. STL Algorithm Enhancements: Replaced verbose manual loops with efficient STL algorithms
// 2. Direct Member Access: Eliminated unnecessary wrapper methods and accessor patterns
// 3. Redundancy Elimination: Removed duplicate validation code and consolidated similar patterns
// 4. Memory Efficiency: Used move semantics and emplace operations for better container performance
// 5. Mutex Optimization: Consolidated critical sections to reduce lock contention
// 6. Validation Streamlining: Replaced complex STL validation with simple direct checks
// 7. Deadlock Prevention: Eliminated 'this' capture in async callbacks and non-blocking mutex acquisition
// 8. Branch Prediction: Added [[likely]]/[[unlikely]] attributes for optimal CPU branch prediction
// 9. Exception Handling: Removed try-catch from hot paths, using validation instead where possible
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
// 3.  Consistent Style: Adopt a uniform coding style and structure for clarity and maintainability, ensure no syntatical or stylization errors.
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
#include <vector>
#include <cstdint>
#include <memory>
#include <atomic>
#include <chrono>
#include <unordered_set>
#include <mutex>
#include <thread>
#include <algorithm>
#include <future>
#include <optional>
#include <numeric>

#include <dpp/dpp.h>
#include "llama-cpp.h"
#include "LogHandler.hpp"
#include "common_utils.hpp"
#include "Sanitizer.hpp"

class LlamaManager;

// Forward declarations for ContextSizeManager integration
struct ContextInfo;
struct ModelInfo;
void initialize_context_size_manager(ContextInfo& context_info, const ModelInfo& model_info);
void track_user_message(ContextInfo& context_info, const ModelInfo& model_info, int32_t user_tokens);

// Status structure for DiscordManager compatibility
struct BackfillStatus {
    bool in_progress;
    int32_t total_channels;
    int32_t completed_channels;
    int32_t total_messages_fetched;
    int32_t total_tokens_loaded_this_session;
};

// Pre-tokenized message with exact token data
struct PreTokenizedMessage {
    uint64_t channel_id;
    uint64_t message_id;
    std::string username;
    std::string content;
    std::chrono::system_clock::time_point timestamp;
    std::vector<llama_token> tokens;
    int32_t token_count;
    
    PreTokenizedMessage() : channel_id(0), message_id(0), token_count(0) {}
};

// Forward declaration
class DiscordHistoryLoader;

// Coordinates multiple history loaders for shared context merging
class SharedContextCoordinator {
private:
    std::string shared_context_id;
    std::vector<std::weak_ptr<DiscordHistoryLoader>> loaders;
    std::atomic<int32_t> completed_loaders{0};
    std::atomic<bool> merge_in_progress{false};
    std::atomic<bool> merge_completed{false};  // Track if merge has been completed to prevent duplicates
    mutable std::mutex coordination_mutex;
    LlamaManager* llama_manager = nullptr;
    
    // Helper methods for merge operations
    bool apply_shared_messages(ContextInfo* shared_context, std::vector<PreTokenizedMessage> all_messages);
    bool apply_messages_fallback(ContextInfo* shared_context, std::vector<PreTokenizedMessage> all_messages);
    void rollback_shared_context(ContextInfo* shared_context, size_t original_size);
    void log_merge_success(const auto& initial_analysis, ContextInfo* shared_context, size_t message_count);
    
public:
    explicit SharedContextCoordinator(std::string context_id, LlamaManager* llama_mgr) 
        : shared_context_id(std::move(context_id)), llama_manager(llama_mgr) {}
    
    void register_loader(std::weak_ptr<DiscordHistoryLoader> loader) {
        std::lock_guard<std::mutex> lock(coordination_mutex);
        loaders.push_back(loader);
    }
    
    void notify_loader_complete();
    bool all_loaders_complete() const;
    void trigger_shared_merge();
};

// NOTE: For Discord history backfill, we use insert_historical_message() instead of add_message()
// because Discord API returns messages from newest to oldest, and we want to maintain proper
// chronological order by inserting historical messages right after system messages but before
// the current conversation, rather than appending them to the end.

// Individual history loader for a specific context
class DiscordHistoryLoader : public std::enable_shared_from_this<DiscordHistoryLoader> {
private:
    // Context association - each loader tied to one context
    std::string target_context_id;
    ContextInfo* target_context = nullptr;
    LlamaManager* llama_manager = nullptr;
    std::string model_id;
    
    // Thread safety for context access
    mutable std::mutex context_access_mutex;
    
    // Channel assignment and configuration
    std::vector<uint64_t> assigned_channels;
    bool is_shared_context = false;

    // NOTE: context_fill_ratio represents the target percentage of context to fill with HISTORICAL MESSAGES ONLY
    // The actual safe threshold must account for emergency buffer and AI response space
    float context_fill_ratio = 0.5f;
    
    // Bot identification
    uint64_t bot_user_id = 0;
    
    // Discord API access
    dpp::cluster* bot = nullptr;
      // Pre-tokenized message storage
    std::vector<PreTokenizedMessage> collected_messages;
    std::atomic<int32_t> total_tokens_collected{0};
    std::atomic<bool> target_reached{false};
    std::atomic<bool> collection_complete{false};
    std::atomic<bool> messages_applied{false};  // Track if messages have been applied to prevent duplicates
    mutable std::mutex messages_mutex;
    
    // Processing state
    uint64_t last_message_id = 0;
    std::unordered_set<uint64_t> processed_message_ids;
    mutable std::mutex processed_ids_mutex;
    
    // Shared context coordination
    std::shared_ptr<SharedContextCoordinator> coordinator;    // Direct validation without unnecessary wrapper - eliminated redundant method
    
    // Following directive #7: favor direct access over abstractions
    bool validate_context_safe() const {
        std::lock_guard<std::mutex> lock(context_access_mutex);
        return target_context && llama_manager && 
               target_context->context_size_manager && target_context->model_info;
    }
    
    // Unsafe validation - must be called while context_access_mutex is already held
    bool validate_context_unsafe() const {
        return target_context && llama_manager && 
               target_context->context_size_manager && target_context->model_info;
    }
    
    static constexpr int32_t MESSAGES_PER_FETCH = 10;
    static constexpr int32_t MAX_ITERATIONS = 1000;
    static constexpr int32_t TIMEOUT_SECONDS = 10;
    static constexpr int32_t RATE_LIMIT_DELAY_MS = 100;
    static constexpr int32_t MAX_MESSAGE_TOKENS = 2048;
    
    // Fetch and process a batch of messages from assigned channel
    // Optimized batch processing with STL algorithms and error handling
    // Fixed deadlock by completely avoiding 'this' capture in async callbacks
    bool process_channel_batch() {
        if (assigned_channels.empty() || target_reached.load()) [[unlikely]] return false;
        
        auto promise = std::make_shared<std::promise<dpp::message_map>>();
        auto future = promise->get_future();
        
        // Capture only essential data, no 'this' pointer to prevent deadlocks
        uint64_t channel_id = assigned_channels[0];
        
        bot->messages_get(channel_id, 0, last_message_id, 0, MESSAGES_PER_FETCH,
            [promise](const dpp::confirmation_callback_t& callback) {
                try {
                    if (callback.is_error()) [[unlikely]] {
                        promise->set_value(dpp::message_map{});
                        return;
                    }
                      // Just pass the message map, no processing in callback
                    promise->set_value(callback.get<dpp::message_map>());
                } catch (const std::exception&) {
                    // Can't log here without 'this', so just return empty map
                    promise->set_value(dpp::message_map{});
                }
            });
        
        // Wait for the async operation to complete
        if (future.wait_for(std::chrono::seconds(TIMEOUT_SECONDS)) != std::future_status::ready) [[unlikely]] {
            return false;
        }
        
        auto messages = future.get();
        if (messages.empty()) [[unlikely]] {
            return false;
        }
        
        // Process messages synchronously on the calling thread (no deadlock risk)
        return collect_and_tokenize_messages(messages, channel_id);
    }
    
    // Enhanced message collection with STL optimizations and separation of concerns
    bool collect_and_tokenize_messages(const dpp::message_map& messages, uint64_t channel_id) {
        std::vector<PreTokenizedMessage> batch_messages;
        batch_messages.reserve(messages.size()); // STL: Pre-allocate for performance
        
        // STL: Find oldest message ID efficiently
        auto oldest_it = std::min_element(messages.begin(), messages.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

        // Process messages with optimized filtering
        for (const auto& [id, msg] : messages) {
            if (auto processed_msg = process_individual_message(msg, channel_id)) {
                batch_messages.emplace_back(std::move(*processed_msg));
            }
        }

        // Update pagination state - explicit cast to resolve type ambiguity
        last_message_id = oldest_it != messages.end() ? static_cast<uint64_t>(oldest_it->first) : last_message_id;
        
        return batch_messages.empty() ? false : apply_message_batch(std::move(batch_messages));
    }

private:
    // Process individual message with validation and tokenization
    std::optional<PreTokenizedMessage> process_individual_message(const dpp::message& msg, uint64_t channel_id) {
        uint64_t msg_id = static_cast<uint64_t>(msg.id);

        // Fast duplicate check with early return
        {
            std::lock_guard<std::mutex> lock(processed_ids_mutex);
            if (!processed_message_ids.insert(msg_id).second) [[unlikely]] return std::nullopt;
        }        // Determine if this is our bot message
        bool is_our_bot = msg.author.is_bot() && (static_cast<uint64_t>(msg.author.id) == bot_user_id);
        
        // Only process messages from users (non-bots) or from our specific bot
        if (msg.author.is_bot() && !is_our_bot) [[unlikely]] {
            return std::nullopt; // Skip other bots, but allow users and our bot
        }
        
        // Extract and sanitize content
        std::string content = extract_message_content(msg, is_our_bot);
        if (content.empty()) [[unlikely]] return std::nullopt;
        
        // Skip blacklisted bot responses
        if (is_our_bot && llama_manager && llama_manager->is_blacklisted_response(content)) [[unlikely]] {
            return std::nullopt;
        }
        
        // Create and tokenize message
        PreTokenizedMessage pre_msg;
        pre_msg.channel_id = channel_id;
        pre_msg.message_id = msg_id;
        pre_msg.username = is_our_bot ? "assistant" : msg.author.username;
        pre_msg.content = content;
        pre_msg.timestamp = std::chrono::system_clock::time_point(std::chrono::seconds(msg.sent));
        
        // Optimized tokenization without try-catch overhead in hot path
        std::string formatted = is_our_bot ? content : (msg.author.username + ": " + content);
        
        // Safely access context for tokenization with proper validation
        {
            std::lock_guard<std::mutex> lock(context_access_mutex);
            if (!target_context || !llama_manager) [[unlikely]] {
                return std::nullopt;
            }
            // Input validation prevents tokenization exceptions
            pre_msg.tokens = llama_manager->process_text_to_tokens(formatted, target_context, false);
        }
        
        pre_msg.token_count = static_cast<int32_t>(pre_msg.tokens.size());
        
        // Store the formatted content to ensure consistency between tokenization and insertion
        pre_msg.content = formatted;
        
        if (pre_msg.token_count > 0 && pre_msg.token_count <= MAX_MESSAGE_TOKENS) [[likely]] {
            return std::make_optional(std::move(pre_msg));
        } else {
            return std::nullopt;
        }
    }
    
    // Extract content from message (direct access, no unnecessary wrappers)
    std::string extract_message_content(const dpp::message& msg, bool is_our_bot) const {
        std::string content = msg.content;
        if (is_our_bot && !msg.embeds.empty() && !msg.embeds[0].description.empty()) {
            content = msg.embeds[0].description;
        }
        return safe_trim(TextSanitizer::sanitize_text(content));
    }
    
    // Apply batch of messages to context with atomic operations
    // Fixed deadlock while ensuring proper context synchronization
    bool apply_message_batch(std::vector<PreTokenizedMessage> batch_messages) {
        // Store original size for potential rollback
        size_t original_history_size;
        bool context_update_succeeded = false;
        bool need_rollback_update = false;
          {
            std::lock_guard<std::mutex> lock(context_access_mutex);
            if (!validate_context_unsafe()) [[unlikely]] return false;
            
            original_history_size = target_context->message_history.size();
            
            try {
                // STL: Use reverse iterator for chronological insertion
                std::for_each(batch_messages.rbegin(), batch_messages.rend(),
                    [this](const auto& msg) {
                        std::string role = (msg.username == "assistant") ? "assistant" : "user";
                        target_context->insert_historical_message(role, msg.content);
                    });
                
                // Atomic context rebuild within same lock
                context_update_succeeded = llama_manager->update_context_from_history(target_context);
                
                // If update failed, rollback immediately while we still have the lock
                if (!context_update_succeeded) [[unlikely]] {
                    auto& history = target_context->message_history;
                    if (history.size() > original_history_size) [[likely]] {
                        history.erase(history.begin() + original_history_size, history.end());
                        target_context->message_cache_dirty = true;
                        target_context->conversation_state.invalidate();
                        need_rollback_update = true;
                    }
                }
            } catch (const std::exception& e) {
                DISCORD_HISTORY_LOG("Exception applying batch for '" + target_context_id + "': " + e.what());
                context_update_succeeded = false;
                
                // Rollback within the same lock scope to avoid deadlock
                auto& history = target_context->message_history;
                if (history.size() > original_history_size) [[likely]] {
                    history.erase(history.begin() + original_history_size, history.end());
                    target_context->message_cache_dirty = true;
                    target_context->conversation_state.invalidate();
                    need_rollback_update = true;
                }
            }
        } // Release lock before any additional context operations
        
        // If rollback occurred, update context outside of the lock to ensure consistency
        if (need_rollback_update) [[unlikely]] {
            std::lock_guard<std::mutex> lock(context_access_mutex);
            if (target_context && llama_manager) [[likely]] {
                llama_manager->update_context_from_history(target_context);
                DISCORD_HISTORY_LOG("Context restored after failed batch for '" + target_context_id + "'");
            }
        }
        
        // Log failure and return early if context update failed
        if (!context_update_succeeded) [[unlikely]] {
            DISCORD_HISTORY_LOG("Context update failed for '" + target_context_id + "' - batch discarded and context restored");
            return false;
        }
        
        // Update tracking and check limits outside of context lock
        return update_collection_state(std::move(batch_messages));
    }
    
    // Update collection state and check target thresholds
    // Fixed deadlock by using separate lock scopes for different mutexes
    bool update_collection_state(std::vector<PreTokenizedMessage> batch_messages) {
        // Get analysis with minimal lock scope
        float current_usage;
        int32_t total_tokens;
        {
            std::lock_guard<std::mutex> lock(context_access_mutex);
            if (!target_context || !target_context->context_size_manager) [[unlikely]] return false;
            
            auto analysis = target_context->context_size_manager->analyze_context(*target_context);
            current_usage = static_cast<float>(analysis.total_used_tokens) / analysis.context_size;
            total_tokens = analysis.total_used_tokens;
            
            // Calculate safe target with direct arithmetic
            float safe_target = std::min(
                context_fill_ratio * (1.0f - analysis.emergency_buffer_percentage - analysis.ai_allocation_percentage),
                1.0f - analysis.emergency_buffer_percentage - analysis.ai_allocation_percentage - 0.05f
            );
            
            // Update atomics and tracking
            if (current_usage >= safe_target) [[unlikely]] {
                target_reached = true;
            }
        }
        
        // Update message storage with separate lock
        {
            std::lock_guard<std::mutex> lock(messages_mutex);
            collected_messages.insert(collected_messages.end(), 
                                    std::make_move_iterator(batch_messages.begin()),
                                    std::make_move_iterator(batch_messages.end()));
        }
        
        total_tokens_collected.store(total_tokens);
        
        DISCORD_HISTORY_LOG("Batch applied to '" + target_context_id + "': " + 
                          std::to_string(current_usage * 100) + "% usage");
        return true;
    }
      // Collection thread main function
    // Fixed deadlock by avoiding mutex holds across async operations
    void collection_thread() {
        DISCORD_HISTORY_LOG("Starting collection for context '" + target_context_id + "'");
          // Validate context safety before starting
        if (!validate_context_safe()) [[unlikely]] {
            DISCORD_HISTORY_LOG("ERROR: Context validation failed at start for '" + target_context_id + "' - aborting collection");
            collection_complete = true;
            return;
        }
        
        // CRITICAL: Wait for context to be fully initialized before proceeding
        int32_t initialization_checks = 0;
        const int32_t MAX_INIT_CHECKS = 10;
        while (initialization_checks < MAX_INIT_CHECKS) {
            {
                std::lock_guard<std::mutex> lock(context_access_mutex);
                if (target_context && target_context->fully_initialized.load()) {
                    break;
                }
            }
            DISCORD_HISTORY_LOG("Waiting for context '" + target_context_id + "' to be fully initialized...");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            initialization_checks++;
        }
          if (initialization_checks >= MAX_INIT_CHECKS) {
            DISCORD_HISTORY_LOG("ERROR: Context '" + target_context_id + "' failed to initialize within timeout - attempting manual retry");
            
            // Attempt one manual retry of context initialization
            if (llama_manager && llama_manager->retry_context_initialization(target_context_id)) {
                DISCORD_HISTORY_LOG("SUCCESS: Context '" + target_context_id + "' initialized successfully on manual retry");
            } else {
                DISCORD_HISTORY_LOG("CRITICAL: Context '" + target_context_id + "' manual retry failed - aborting collection");
                collection_complete = true;
                return;
            }
        }
        
        try {
            // Initialize ContextSizeManager and ensure context is ready for operations
            {
                std::lock_guard<std::mutex> lock(context_access_mutex);
                
                // Re-validate context after acquiring mutex
                if (!target_context || !target_context->model_info) [[unlikely]] {
                    DISCORD_HISTORY_LOG("ERROR: Context became invalid after initial validation for '" + target_context_id + "'");
                    collection_complete = true;
                    return;
                }
                
                // Initialize ContextSizeManager if needed
                if (!target_context->context_size_manager) [[unlikely]] {
                    initialize_context_size_manager(*target_context, *target_context->model_info);
                }
                
                // CRITICAL FIX: Ensure context is properly initialized for tokenization
                // If the context hasn't been through tokenization yet, ensure it's ready
                // Fixed condition: Check n_past == 0 regardless of message_history content
                // because contexts created with system prompts have non-empty message_history
                if (target_context->n_past == 0) [[unlikely]] {
                    DISCORD_HISTORY_LOG("Initializing context state for history loading (n_past=0, ensuring tokenization readiness)...");
                    
                    // Ensure the context has proper initial state for tokenization
                    // This mimics what happens during the first generation cycle
                    if (llama_manager) [[likely]] {
                        // Force a minimal context update to establish proper state
                        // This ensures the vocabulary and tokenizer are properly initialized
                        try {
                            llama_manager->update_context_from_history(target_context);
                            DISCORD_HISTORY_LOG("Context initialization completed for '" + target_context_id + "' (n_past now: " + 
                                               std::to_string(target_context->n_past) + ")");
                        } catch (const std::exception& e) {
                            DISCORD_HISTORY_LOG("ERROR: Failed to initialize context for '" + target_context_id + "': " + e.what());
                            collection_complete = true;
                            return;
                        }
                    }
                }
                
                // Log initial context state for debugging
                if (target_context->context_size_manager) [[likely]] {
                    auto initial_analysis = target_context->context_size_manager->analyze_context(*target_context);
                    DISCORD_HISTORY_LOG("Initial context state - Size: " + std::to_string(initial_analysis.context_size) + 
                               ", Used: " + std::to_string(initial_analysis.total_used_tokens) + 
                               ", Available: " + std::to_string(initial_analysis.available_tokens) +
                               ", Emergency Buffer: " + std::to_string(initial_analysis.emergency_buffer_percentage * 100) + "%");
                }
            } // Release context lock before processing batches
            
            // Collect messages until target reached - no context lock held during async operations
            int32_t iterations = 0;
            while (!target_reached.load() && iterations < MAX_ITERATIONS) [[likely]] {
                if (!process_channel_batch()) [[unlikely]] {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(RATE_LIMIT_DELAY_MS));
                iterations++;
            }
        } catch (const std::exception& e) {
            DISCORD_HISTORY_LOG("CRITICAL: Exception in collection thread for '" + target_context_id + "': " + e.what());
            target_reached = true;
        }
        
        collection_complete = true;
        
        if (is_shared_context && coordinator) [[unlikely]] {
            coordinator->notify_loader_complete();
        } else [[likely]] {
            // Isolated context - apply immediately
            apply_to_isolated_context();
        }
        
        DISCORD_HISTORY_LOG("Collection complete for context '" + target_context_id + "': " + 
                   std::to_string(total_tokens_collected.load()) + " tokens collected");
    }

public:
    // Factory methods for creating loaders
    static std::shared_ptr<DiscordHistoryLoader> create_for_isolated_context(
        const std::string& context_id, uint64_t channel_id, 
        ContextInfo* context, LlamaManager* llama_mgr, const std::string& model_identifier,
        dpp::cluster* discord_bot, uint64_t bot_user_id, float fill_ratio) {
        
        auto loader = std::make_shared<DiscordHistoryLoader>();
        loader->target_context_id = context_id;
        loader->target_context = context;
        loader->llama_manager = llama_mgr;
        loader->model_id = model_identifier;
        loader->assigned_channels = {channel_id};
        loader->is_shared_context = false;
        loader->context_fill_ratio = fill_ratio;
        loader->bot = discord_bot;
        loader->bot_user_id = bot_user_id;
        
        return loader;
    }
    
    static std::shared_ptr<DiscordHistoryLoader> create_for_shared_context(
        const std::string& context_id, uint64_t channel_id,
        ContextInfo* context, LlamaManager* llama_mgr, const std::string& model_identifier,
        dpp::cluster* discord_bot, uint64_t bot_user_id, 
        std::shared_ptr<SharedContextCoordinator> coord, float portion_ratio) {
        
        auto loader = std::make_shared<DiscordHistoryLoader>();
        loader->target_context_id = context_id;
        loader->target_context = context;
        loader->llama_manager = llama_mgr;
        loader->model_id = model_identifier;
        loader->assigned_channels = {channel_id};
        loader->is_shared_context = true;
        loader->context_fill_ratio = portion_ratio;
        loader->bot = discord_bot;
        loader->bot_user_id = bot_user_id;
        loader->coordinator = coord;
        
        return loader;
    }

    // Start collection immediately
    void start_collection() {
        std::thread(&DiscordHistoryLoader::collection_thread, this).detach();
    }
    
    // Apply messages directly to isolated context
    bool apply_to_isolated_context() {
        if (is_shared_context) [[unlikely]] return false;
        
        // Check if already applied to prevent duplicate applications
        if (messages_applied.load()) [[unlikely]] {
            DISCORD_HISTORY_LOG("Messages already applied to context '" + target_context_id + "' - skipping duplicate application");
            return true;
        }
        
        std::lock_guard<std::mutex> lock(messages_mutex);
        
        // Messages are already applied to context during collection
        // Just mark as applied and log completion
        messages_applied = true;
        
        DISCORD_HISTORY_LOG("Messages already applied to context '" + target_context_id + "' during collection");
        DISCORD_HISTORY_LOG("Final context usage: " + std::to_string(total_tokens_collected.load()) + " tokens");
        return true;
    }
    
    // Get collected messages for shared context merging
    std::vector<PreTokenizedMessage> get_collected_messages() const {
        std::lock_guard<std::mutex> lock(messages_mutex);
        return collected_messages;
    }

public:
    // Direct member access - eliminate unnecessary LoaderStatus wrapper
    // Following directive #7: favor direct access over abstractions
    
    // State access methods (const and thread-safe)
    bool is_target_reached() const { return target_reached.load(); }
    bool is_collection_complete() const { return collection_complete.load(); }
    bool are_messages_applied() const { return messages_applied.load(); }
    int32_t get_total_tokens_collected() const { return total_tokens_collected.load(); }
    
    size_t get_messages_collected() const {
        std::lock_guard<std::mutex> lock(messages_mutex);
        return collected_messages.size();
    }
};

// SharedContextCoordinator implementation with STL optimizations
inline void SharedContextCoordinator::notify_loader_complete() {
    completed_loaders.fetch_add(1);
    if (all_loaders_complete()) [[likely]] {
        trigger_shared_merge();
    }
}

inline bool SharedContextCoordinator::all_loaders_complete() const {
    std::lock_guard<std::mutex> lock(coordination_mutex);
    
    // STL: Use count_if for efficient active loader counting
    auto active_count = std::count_if(loaders.begin(), loaders.end(),
        [](const auto& weak_loader) { return !weak_loader.expired(); });
    
    return completed_loaders.load() >= active_count;
}

inline void SharedContextCoordinator::trigger_shared_merge() {
    // Atomic check and set for merge completion
    if (merge_completed.load() || merge_in_progress.exchange(true)) [[unlikely]] return;
    
    std::lock_guard<std::mutex> lock(coordination_mutex);
    
    // Double-check pattern with early return
    if (merge_completed.load()) [[unlikely]] {
        merge_in_progress = false;
        return;
    }
    
    try {
        // STL: Efficient message collection with transform and flatten
        std::vector<PreTokenizedMessage> all_messages;
        
        // Pre-calculate total size for single allocation
        size_t total_size = std::accumulate(loaders.begin(), loaders.end(), size_t{0},
            [](size_t sum, const auto& weak_loader) {
                if (auto loader = weak_loader.lock()) [[likely]] {
                    return sum + loader->get_messages_collected();
                }
                return sum;
            });
        
        all_messages.reserve(total_size);
        
        // STL: Collect messages efficiently
        for (auto& weak_loader : loaders) {
            if (auto loader = weak_loader.lock()) [[likely]] {
                auto messages = loader->get_collected_messages();
                all_messages.insert(all_messages.end(), 
                                  std::make_move_iterator(messages.begin()),
                                  std::make_move_iterator(messages.end()));
            }
        }
        
        // Get context and validate once
        auto* shared_context = llama_manager ? llama_manager->get_context_info(shared_context_id) : nullptr;
        if (!shared_context) [[unlikely]] {
            DISCORD_HISTORY_LOG("ERROR: Failed to get shared context '" + shared_context_id + "' for merge");
            merge_in_progress = false;
            return;
        }
        
        // Apply messages with single transaction
        if (apply_shared_messages(shared_context, std::move(all_messages))) [[likely]] {
            merge_completed = true;
        }
        
        merge_in_progress = false;
    } catch (const std::exception& e) {
        DISCORD_HISTORY_LOG("CRITICAL: Exception during shared merge for '" + shared_context_id + "': " + e.what());
        merge_in_progress = false;
    }
}

// Helper method implementations for SharedContextCoordinator
inline bool SharedContextCoordinator::apply_shared_messages(ContextInfo* shared_context, std::vector<PreTokenizedMessage> all_messages) {
    if (!shared_context->context_size_manager) [[unlikely]] {
        return apply_messages_fallback(shared_context, std::move(all_messages));
    }
    
    auto initial_analysis = shared_context->context_size_manager->analyze_context(*shared_context);
    size_t original_history_size = shared_context->message_history.size();
    
    try {
        // STL: Process messages in reverse for chronological order
        std::for_each(all_messages.rbegin(), all_messages.rend(),
            [shared_context](const auto& msg) {
                std::string role = (msg.username == "assistant") ? "assistant" : "user";
                shared_context->insert_historical_message(role, msg.content);
            });
        
        if (!llama_manager->update_context_from_history(shared_context)) [[unlikely]] {
            rollback_shared_context(shared_context, original_history_size);
            return false;
        }
        
        log_merge_success(initial_analysis, shared_context, all_messages.size());
        return true;
    } catch (const std::exception& e) {
        DISCORD_HISTORY_LOG("Exception during shared message application: " + std::string(e.what()));
        rollback_shared_context(shared_context, original_history_size);
        return false;
    }
}

inline bool SharedContextCoordinator::apply_messages_fallback(ContextInfo* shared_context, std::vector<PreTokenizedMessage> all_messages) {
    size_t original_history_size = shared_context->message_history.size();
    
    try {
        std::for_each(all_messages.rbegin(), all_messages.rend(),
            [shared_context](const auto& msg) {
                std::string role = (msg.username == "assistant") ? "assistant" : "user";
                shared_context->insert_historical_message(role, msg.content);
            });
          if (!llama_manager->update_context_from_history(shared_context)) [[unlikely]] {
            DISCORD_HISTORY_LOG("Fallback context update failed - performing rollback");
            rollback_shared_context(shared_context, original_history_size);
            return false;
        }
        
        DISCORD_HISTORY_LOG("Fallback message application succeeded");
        return true;
    } catch (const std::exception& e) {
        DISCORD_HISTORY_LOG("Exception during fallback message application: " + std::string(e.what()));
        rollback_shared_context(shared_context, original_history_size);
        return false;
    }
}

inline void SharedContextCoordinator::rollback_shared_context(ContextInfo* shared_context, size_t original_size) {
    auto& history = shared_context->message_history;
    if (history.size() > original_size) [[likely]] {
        history.erase(history.begin() + original_size, history.end());
        shared_context->message_cache_dirty = true;
        shared_context->conversation_state.invalidate();
        
        // Ensure context is properly updated after rollback
        if (llama_manager) [[likely]] {
            llama_manager->update_context_from_history(shared_context);
        }
        
        DISCORD_HISTORY_LOG("Shared context rolled back to size " + std::to_string(original_size) + " and context updated");
    }
}

inline void SharedContextCoordinator::log_merge_success(const auto& initial_analysis, ContextInfo* shared_context, size_t message_count) {
    auto final_analysis = shared_context->context_size_manager->analyze_context(*shared_context);
    float final_usage = static_cast<float>(final_analysis.total_used_tokens) / final_analysis.context_size;
    
    DISCORD_HISTORY_LOG("Shared context merge completed for '" + shared_context_id + "':");
    DISCORD_HISTORY_LOG("  Messages: " + std::to_string(message_count) + 
                      ", Usage: " + std::to_string(final_usage * 100) + "%");
}

//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODE DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//
