// DiscordHistoryLoader.hpp - header-only implementation for per-context Discord chat history backfill
// Handles immediate Discord message history fetching with pre-tokenization for individual contexts.
//
// File Specific Directives:
// Individual history loaders per context with immediate collection and pre-tokenization.
// Isolated contexts: fill to target ratio then apply immediately.
// Shared contexts: coordinate between multiple loaders before merging.
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
    
public:
    explicit SharedContextCoordinator(const std::string& context_id, LlamaManager* llama_mgr) 
        : shared_context_id(context_id), llama_manager(llama_mgr) {}
    
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
    std::shared_ptr<SharedContextCoordinator> coordinator;
      // Safe context validation with null checks
    bool validate_context_safe() const {
        std::lock_guard<std::mutex> lock(context_access_mutex);
        if (!target_context) {
            DISCORD_HISTORY_LOG("ERROR: target_context is null for '" + target_context_id + "'");
            return false;
        }
        if (!llama_manager) {
            DISCORD_HISTORY_LOG("ERROR: llama_manager is null for '" + target_context_id + "'");
            return false;
        }
        if (!target_context->context_size_manager) {
            DISCORD_HISTORY_LOG("ERROR: context_size_manager is null for '" + target_context_id + "'");
            return false;
        }
        if (!target_context->model_info) {
            DISCORD_HISTORY_LOG("ERROR: model_info is null for '" + target_context_id + "'");
            return false;
        }
        return true;
    }    // Safely restore context state after failed rebuild
    // NOTE: This function must be called while holding context_access_mutex
    bool restore_context_state(const std::vector<PreTokenizedMessage>& failed_messages) {
        if (!target_context) return false;
        
        try {
            // Remove the messages we just added
            // Note: Direct access to message_history is safe here since we're in error recovery
            for (size_t i = 0; i < failed_messages.size(); ++i) {
                if (!target_context->message_history.empty()) {
                    target_context->message_history.pop_back();
                }
            }
            
            // Mark cache as dirty and reset conversation state
            target_context->message_cache_dirty = true;
            if (target_context->conversation_state.needs_rebuild == false) {
                target_context->conversation_state.invalidate();
            }
            
            // Try to rebuild from remaining message history to restore valid state
            // Note: update_context_from_history is now thread-safe with internal mutex
            bool restore_success = llama_manager->update_context_from_history(target_context);
            if (!restore_success) {
                DISCORD_HISTORY_LOG("CRITICAL: Failed to restore context state for '" + target_context_id + "' - context may be corrupted");
                return false;
            }
            
            DISCORD_HISTORY_LOG("Successfully restored context state for '" + target_context_id + "' after failed rebuild");
            return true;
        } catch (const std::exception& e) {
            DISCORD_HISTORY_LOG("CRITICAL: Exception while restoring context state for '" + target_context_id + "': " + e.what());
            return false;
        }
    }
    static constexpr int32_t MESSAGES_PER_FETCH = 10;
    static constexpr int32_t MAX_ITERATIONS = 1000;
    static constexpr int32_t TIMEOUT_SECONDS = 10;
    static constexpr int32_t RATE_LIMIT_DELAY_MS = 100;
    static constexpr int32_t MAX_MESSAGE_TOKENS = 2048;
    
    // Fetch and process a batch of messages from assigned channel
    bool process_channel_batch() {
        if (assigned_channels.empty() || target_reached.load()) return false;
        
        uint64_t channel_id = assigned_channels[0]; // Single channel per loader
        
        auto promise = std::make_shared<std::promise<bool>>();
        auto future = promise->get_future();
        
        bot->messages_get(channel_id, 0, last_message_id, 0, MESSAGES_PER_FETCH,
            [this, channel_id, promise](const dpp::confirmation_callback_t& callback) {
                try {
                    if (callback.is_error()) {
                        promise->set_value(false);
                        return;
                    }
                    
                    auto messages = callback.get<dpp::message_map>();
                    if (messages.empty()) {
                        promise->set_value(false);
                        return;
                    }
                    
                    promise->set_value(collect_and_tokenize_messages(messages, channel_id));
                } catch (const std::exception& e) {
                    DISCORD_HISTORY_LOG("Exception processing channel " + std::to_string(channel_id) + ": " + e.what());
                    promise->set_value(false);
                }
            });
        
        auto status = future.wait_for(std::chrono::seconds(TIMEOUT_SECONDS));
        return status == std::future_status::ready && future.get();
    }    // Collect and immediately tokenize messages
    bool collect_and_tokenize_messages(const dpp::message_map& messages, uint64_t channel_id) {
        std::vector<PreTokenizedMessage> batch_messages;
        uint64_t oldest_id = UINT64_MAX;
        int32_t batch_tokens = 0;
        
        for (const auto& [id, msg] : messages) {
            uint64_t msg_id = static_cast<uint64_t>(msg.id);
            oldest_id = std::min(oldest_id, msg_id);
            
            // Skip duplicates
            {
                std::lock_guard<std::mutex> lock(processed_ids_mutex);
                if (processed_message_ids.count(msg_id)) continue;
                processed_message_ids.insert(msg_id);
            }
              // Check if processable message
            bool is_our_bot = msg.author.is_bot() && (static_cast<uint64_t>(msg.author.id) == bot_user_id);
            if (!msg.author.is_bot() || is_our_bot) {
                // Extract content first for blacklist checking
                std::string content = msg.content;
                if (is_our_bot && !msg.embeds.empty() && !msg.embeds[0].description.empty()) {
                    content = msg.embeds[0].description;
                }
                content = safe_trim(TextSanitizer::sanitize_text(content));
                  // Skip bot messages that match blacklist patterns
                if (is_our_bot && llama_manager && llama_manager->is_blacklisted_response(content)) {
                    DISCORD_HISTORY_LOG("Skipping blacklisted bot message from history: '" + 
                                      (content.length() > 50 ? content.substr(0, 50) + "..." : content) + "'");
                    continue;
                }
                
                PreTokenizedMessage pre_msg;
                pre_msg.channel_id = channel_id;
                pre_msg.message_id = msg_id;
                pre_msg.username = is_our_bot ? "assistant" : msg.author.username;
                pre_msg.content = content;
                pre_msg.timestamp = std::chrono::system_clock::time_point(std::chrono::seconds(msg.sent));
                
                // Simple tokenization for storage - exact counting happens after rebuild
                std::string role = is_our_bot ? "assistant" : "user";
                std::string formatted = is_our_bot ? content : (msg.author.username + ": " + content);
                
                // Store basic token info for reference (not used for capacity planning)
                try {
                    auto base_tokens = llama_manager->process_text_to_tokens(formatted, target_context, false);
                    pre_msg.tokens = base_tokens;
                    pre_msg.token_count = static_cast<int32_t>(base_tokens.size()); // Basic count for reference only

                    if (pre_msg.token_count > 0 && pre_msg.token_count <= MAX_MESSAGE_TOKENS) {
                        batch_messages.push_back(pre_msg);
                        batch_tokens += pre_msg.token_count; // This is just for initial filtering
                    }
                } catch (const std::exception& e) {
                    DISCORD_HISTORY_LOG("ERROR: Exception during tokenization for message " + std::to_string(msg_id) + ": " + e.what());
                    continue; // Skip this message and continue
                }
            }
        }        // Add batch to context and rebuild to get exact token count
        if (!batch_messages.empty()) {
            // Note: No sorting needed since we're fetching from newest to oldest
            // and will insert them in the correct chronological position using insert_historical_message()
            
            // Store context state before modification for potential rollback
            // Hold mutex for entire critical section to prevent concurrent access
            std::lock_guard<std::mutex> lock(context_access_mutex);
            
            // Validate context safety while holding mutex
            if (!target_context) {
                DISCORD_HISTORY_LOG("ERROR: target_context is null for '" + target_context_id + "'");
                return false;
            }
            if (!llama_manager) {
                DISCORD_HISTORY_LOG("ERROR: llama_manager is null for '" + target_context_id + "'");
                return false;
            }
            if (!target_context->context_size_manager) {
                DISCORD_HISTORY_LOG("ERROR: context_size_manager is null for '" + target_context_id + "'");
                return false;
            }
            if (!target_context->model_info) {
                DISCORD_HISTORY_LOG("ERROR: model_info is null for '" + target_context_id + "'");
                return false;
            }
            
            size_t original_history_size = target_context->message_history.size();            // Temporarily add messages to context using historical insertion
            std::string role;
            try {
                // Add messages while holding the mutex to prevent concurrent modifications
                // Use reverse iterator to process messages from oldest to newest since we fetched newest to oldest
                for (auto it = batch_messages.rbegin(); it != batch_messages.rend(); ++it) {
                    const auto& msg = *it;
                    role = (msg.username == "assistant") ? "assistant" : "user";
                    target_context->insert_historical_message(role, msg.content);
                }
            } catch (const std::exception& e) {
                DISCORD_HISTORY_LOG("ERROR: Exception while adding messages to context for '" + target_context_id + "': " + e.what());
                return false;
            }
              // Perform full context rebuild to get exact token count
            bool rebuild_success = false;
            try {
                // Rebuild context while holding mutex to prevent concurrent access
                rebuild_success = llama_manager->update_context_from_history(target_context);
            } catch (const std::exception& e) {
                DISCORD_HISTORY_LOG("ERROR: Exception during context rebuild for '" + target_context_id + "': " + e.what());
                rebuild_success = false;
            }
            
            if (!rebuild_success) {
                DISCORD_HISTORY_LOG("ERROR: Failed to rebuild context during collection for '" + target_context_id + "'");
                
                // Attempt to restore context to valid state
                if (!restore_context_state(batch_messages)) {
                    DISCORD_HISTORY_LOG("CRITICAL: Failed to restore context state for '" + target_context_id + "' - context may be permanently corrupted");
                    // Mark as failed to prevent further operations
                    target_reached = true; // Stop collection to prevent further corruption
                }
                return false;
            }
              // Get exact context analysis after rebuild
            EnhancedContextAnalysis analysis;
            try {
                // Analyze context while holding mutex
                analysis = target_context->context_size_manager->analyze_context(*target_context);
            } catch (const std::exception& e) {
                DISCORD_HISTORY_LOG("ERROR: Exception during context analysis for '" + target_context_id + "': " + e.what());
                return false;
            }
            
            float current_usage = static_cast<float>(analysis.total_used_tokens) / analysis.context_size;
            
            // Calculate target threshold
            float safe_target = std::min(
                context_fill_ratio * (1.0f - analysis.emergency_buffer_percentage - analysis.ai_allocation_percentage),
                1.0f - analysis.emergency_buffer_percentage - analysis.ai_allocation_percentage - 0.05f  // Extra 5% safety margin
            );
            
            DISCORD_HISTORY_LOG("Context '" + target_context_id + "' after adding " + std::to_string(batch_messages.size()) + " messages:");
            DISCORD_HISTORY_LOG("  Exact token count: " + std::to_string(analysis.total_used_tokens) + " / " + std::to_string(analysis.context_size));
            DISCORD_HISTORY_LOG("  Current usage: " + std::to_string(current_usage * 100) + "%");
            DISCORD_HISTORY_LOG("  Target threshold: " + std::to_string(safe_target * 100) + "%");
              // Check if we've reached our target
            if (current_usage >= safe_target) {
                target_reached = true;
                DISCORD_HISTORY_LOG("Target reached for context '" + target_context_id + "' at " + std::to_string(current_usage * 100) + "% usage");
            }
              // Update our tracking (messages are already added to context)
            {
                std::lock_guard<std::mutex> lock(messages_mutex);
                collected_messages.insert(collected_messages.end(), batch_messages.begin(), batch_messages.end());
            }
            total_tokens_collected.store(analysis.total_used_tokens);
              // Track with ContextSizeManager
            try {
                // Note: ContextSizeManager operations are thread-safe
                if (target_context->context_size_manager && target_context->model_info) {
                    for (const auto& msg : batch_messages) {
                        if (msg.username != "assistant") {
                            track_user_message(*target_context, *target_context->model_info, msg.token_count);
                        }
                    }
                }
            } catch (const std::exception& e) {
                DISCORD_HISTORY_LOG("ERROR: Exception during ContextSizeManager tracking for '" + target_context_id + "': " + e.what());
                // Continue despite tracking failure
            }
            
            // Update pagination
            if (oldest_id != UINT64_MAX) {
                last_message_id = oldest_id;
            }
            
            return true; // Continue iteration
        } else {
            // No valid messages in this batch
            return false;
        }
    }    // Collection thread main function
    void collection_thread() {
        DISCORD_HISTORY_LOG("Starting collection for context '" + target_context_id + "'");
        
        // Validate context safety before starting
        if (!validate_context_safe()) {
            DISCORD_HISTORY_LOG("ERROR: Context validation failed at start for '" + target_context_id + "' - aborting collection");
            collection_complete = true;
            return;
        }        try {
            // Initialize ContextSizeManager and log initial state with proper mutex protection
            {
                std::lock_guard<std::mutex> lock(context_access_mutex);
                
                // Re-validate context after acquiring mutex
                if (!target_context || !target_context->model_info) {
                    DISCORD_HISTORY_LOG("ERROR: Context became invalid after initial validation for '" + target_context_id + "'");
                    collection_complete = true;
                    return;
                }
                
                // Initialize ContextSizeManager if needed
                if (!target_context->context_size_manager) {
                    initialize_context_size_manager(*target_context, *target_context->model_info);
                }
                
                // Log initial context state for debugging
                if (target_context->context_size_manager) {
                    auto initial_analysis = target_context->context_size_manager->analyze_context(*target_context);
                    DISCORD_HISTORY_LOG("Initial context state - Size: " + std::to_string(initial_analysis.context_size) + 
                               ", Used: " + std::to_string(initial_analysis.total_used_tokens) + 
                               ", Available: " + std::to_string(initial_analysis.available_tokens) +
                               ", Emergency Buffer: " + std::to_string(initial_analysis.emergency_buffer_percentage * 100) + "%");
                }
            }
            // Collect messages until target reached
            int32_t iterations = 0;
            while (!target_reached.load() && iterations < MAX_ITERATIONS) {
                if (!process_channel_batch()) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(RATE_LIMIT_DELAY_MS));
                iterations++;
            }
            
        } catch (const std::exception& e) {
            DISCORD_HISTORY_LOG("CRITICAL: Exception in collection thread for '" + target_context_id + "': " + e.what());
            // Mark as failed to prevent further operations
            target_reached = true;
        }
        
        collection_complete = true;
        
        if (is_shared_context && coordinator) {
            coordinator->notify_loader_complete();
        } else {
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
    
    // Status checks
    bool is_ready_to_apply() const { return target_reached.load(); }
    bool is_collection_complete() const { return collection_complete.load(); }    // Apply messages directly to isolated context
    bool apply_to_isolated_context() {
        if (is_shared_context) return false;
        
        // Check if already applied to prevent duplicate applications
        if (messages_applied.load()) {
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
    
    // Status information
    struct LoaderStatus {
        bool target_reached;
        bool collection_complete;
        int32_t total_tokens_collected;
        size_t messages_collected;
    };
      LoaderStatus get_status() const {
        std::lock_guard<std::mutex> lock(messages_mutex);
        return {
            target_reached.load(),
            collection_complete.load(), 
            total_tokens_collected.load(),
            collected_messages.size()
        };
    }
    
    // Check if messages have been applied (for debugging and status monitoring)
    bool are_messages_applied() const { return messages_applied.load(); }
};

// SharedContextCoordinator implementation
inline void SharedContextCoordinator::notify_loader_complete() {
    completed_loaders.fetch_add(1);
    
    if (all_loaders_complete()) {
        trigger_shared_merge();
    }
}

inline bool SharedContextCoordinator::all_loaders_complete() const {
    std::lock_guard<std::mutex> lock(coordination_mutex);
    int32_t active_loaders = 0;
    
    // Count active loaders (cannot modify container in const method)
    for (const auto& weak_loader : loaders) {
        if (auto loader = weak_loader.lock()) {
            active_loaders++;
        }
    }
    
    return completed_loaders.load() >= active_loaders;
}

inline void SharedContextCoordinator::trigger_shared_merge() {
    // Check if merge has already been completed
    if (merge_completed.load()) {
        DISCORD_HISTORY_LOG("Shared context merge already completed for '" + shared_context_id + "' - skipping duplicate merge");
        return;
    }
    
    if (merge_in_progress.exchange(true)) return;
    
    std::lock_guard<std::mutex> lock(coordination_mutex);
    
    // Double-check after acquiring lock
    if (merge_completed.load()) {
        DISCORD_HISTORY_LOG("Shared context merge already completed for '" + shared_context_id + "' - skipping duplicate merge");
        merge_in_progress = false;
        return;
    }
    
    try {        // Collect all messages from all loaders
        std::vector<PreTokenizedMessage> all_messages;
        
        for (auto& weak_loader : loaders) {
            if (auto loader = weak_loader.lock()) {
                auto messages = loader->get_collected_messages();
                all_messages.insert(all_messages.end(), messages.begin(), messages.end());
            }
        }
        
        // Note: No sorting needed since we'll insert messages in chronological order using insert_historical_message()
        
        // Get shared context and apply messages
        ContextInfo* shared_context = nullptr;
        if (llama_manager) {
            shared_context = llama_manager->get_context_info(shared_context_id);
        }
        
        if (!shared_context) {
            DISCORD_HISTORY_LOG("ERROR: Failed to get shared context '" + shared_context_id + "' for merge");
            merge_in_progress = false;
            return;
        }
        
        // Calculate total tokens from all collected messages
        int32_t total_collected_tokens = 0;
        for (const auto& msg : all_messages) {
            total_collected_tokens += msg.token_count;
        }
        
        // Final safety check with exact rebuild
        if (shared_context->context_size_manager) {
            auto initial_analysis = shared_context->context_size_manager->analyze_context(*shared_context);
              // Store original state for potential rollback
            size_t original_history_size = shared_context->message_history.size();
            
            // Add all messages using historical insertion to maintain chronological order
            // Process from newest to oldest (reverse order) so they're inserted in correct chronological sequence
            for (auto it = all_messages.rbegin(); it != all_messages.rend(); ++it) {
                const auto& msg = *it;
                std::string role = (msg.username == "assistant") ? "assistant" : "user";
                shared_context->insert_historical_message(role, msg.content);
            }
            
            // Rebuild to get exact token count
            bool rebuild_success = llama_manager->update_context_from_history(shared_context);
            if (!rebuild_success) {
                DISCORD_HISTORY_LOG("ERROR: Failed to rebuild shared context '" + shared_context_id + "' during merge");
                
                // Attempt rollback
                while (shared_context->message_history.size() > original_history_size && !shared_context->message_history.empty()) {
                    shared_context->message_history.pop_back();
                }
                shared_context->message_cache_dirty = true;
                
                // Try to restore original state
                if (!llama_manager->update_context_from_history(shared_context)) {
                    DISCORD_HISTORY_LOG("CRITICAL: Failed to restore shared context '" + shared_context_id + "' after failed merge - context may be corrupted");
                }
                
                merge_in_progress = false;
                return;
            }
            
            auto final_analysis = shared_context->context_size_manager->analyze_context(*shared_context);
            float final_usage = static_cast<float>(final_analysis.total_used_tokens) / final_analysis.context_size;
            
            DISCORD_HISTORY_LOG("Shared context merge completed for '" + shared_context_id + "':");
            DISCORD_HISTORY_LOG("  Initial tokens: " + std::to_string(initial_analysis.total_used_tokens));
            DISCORD_HISTORY_LOG("  Final tokens: " + std::to_string(final_analysis.total_used_tokens));
            DISCORD_HISTORY_LOG("  Added messages: " + std::to_string(all_messages.size()));
            DISCORD_HISTORY_LOG("  Final usage: " + std::to_string(final_usage * 100) + "%");
            
            if (final_usage > 0.95f) {
                DISCORD_HISTORY_LOG("WARNING: Shared context usage exceeds 95% after merge");
            }        } else {
            // Fallback if no context size manager
            // Add all messages using historical insertion to maintain chronological order
            for (auto it = all_messages.rbegin(); it != all_messages.rend(); ++it) {
                const auto& msg = *it;
                std::string role = (msg.username == "assistant") ? "assistant" : "user";
                shared_context->insert_historical_message(role, msg.content);
            }
            
            if (!llama_manager->update_context_from_history(shared_context)) {
                DISCORD_HISTORY_LOG("ERROR: Failed to rebuild shared context '" + shared_context_id + "' during merge");
                merge_in_progress = false;
                return;
            }
            
            DISCORD_HISTORY_LOG("Successfully merged " + std::to_string(all_messages.size()) + 
                       " messages into shared context '" + shared_context_id + "'");
        }
        
        // Mark as completed
        merge_completed = true;
        merge_in_progress = false;
        
    } catch (const std::exception& e) {
        DISCORD_HISTORY_LOG("CRITICAL: Exception during shared context merge for '" + shared_context_id + "': " + e.what());
        merge_in_progress = false;
    }
}

//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODE DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//
