// DiscordHistoryLoader.hpp - header-only implementation for Discord chat history backfill
// Handles Discord message history fetching and context population for LuminaChat backend.
//
// File Specific Directives:
// Manage Discord chat history backfill with context capacity tracking and rate limiting.
// Handle message fetching, token estimation, and context population efficiently.
//
// Project Settings:
// C++ Language Standard: ISO C++20 Standard (/std:c++20)
// C Language Standard: ISO C17 (2018) Standard (/std:c17)
// Optimization: Maximum Optimization (Favor Speed) (/O2)
// Favor Size or Speed: Favor fast code (/Ot)
// Runtime Library: Multi-threaded DLL (/MD)
// Enable Run-Time Type Information (RTTI) YES (/GR)
#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <functional>
#include <thread>
#include <algorithm>

#include <dpp/dpp.h>

// Forward declarations
class LlamaManager;

class DiscordHistoryLoader {
public:
    // History message structure
    struct HistoryMessage {
        uint64_t message_id;
        uint64_t user_id;
        uint64_t channel_id;
        std::string username;
        std::string content;
        std::chrono::system_clock::time_point timestamp;
    };
    
    // Channel backfill state tracking
    struct ChannelBackfillState {
        uint64_t last_message_id = 0;
        bool backfill_complete = false;
        int32_t messages_fetched = 0;
        int32_t estimated_tokens_added = 0;
        std::string target_context_id;
    };
    
    // Context capacity tracking
    struct ContextBackfillInfo {
        std::string context_id;
        int32_t estimated_tokens = 0;
        int32_t capacity_limit = 0;
        std::vector<uint64_t> associated_channels;
        bool is_main_context = false;
    };
    
    // Backfill status for reporting
    struct BackfillStatus {
        bool in_progress;
        int32_t channels_processed;
        int32_t total_messages_fetched;
        int32_t channels_complete;
        std::unordered_map<std::string, ContextBackfillInfo> context_usage;
    };

private:
    // Core dependencies
    dpp::cluster* bot;
    LlamaManager* llama_manager;
    std::string main_context_id;
    
    // Logging callback
    std::function<void(const std::string&)> log_callback;
    
    // Channel configuration
    const std::unordered_set<uint64_t>* isolated_channels;
    const std::unordered_set<uint64_t>* shared_history_channels;
    
    // Backfill state
    std::atomic<bool> backfill_in_progress{false};
    std::unordered_map<uint64_t, ChannelBackfillState> channel_backfill_state;
    std::unordered_map<std::string, ContextBackfillInfo> context_backfill_info;
    mutable std::mutex backfill_mutex;
    mutable std::mutex context_backfill_mutex;
    
    // Constants
    static constexpr int32_t MESSAGES_PER_FETCH = 10;
    static constexpr float MAX_CONTEXT_FILL_RATIO = 0.5f;
    
    // Helper to estimate token count
    int32_t estimate_message_tokens(const std::string& username, const std::string& content) const {
        const int32_t base_overhead = 50;
        const int32_t role_overhead = username.length() + 20;
        const int32_t content_tokens = static_cast<int32_t>((content.length() + 2) / 3);
        return base_overhead + role_overhead + content_tokens;
    }
    
    // Check if channel should have history backfilled
    bool should_backfill_channel(uint64_t channel_id) const {
        // Always backfill isolated channels
        if (isolated_channels && isolated_channels->find(channel_id) != isolated_channels->end()) {
            return true;
        }
        
        // For shared channels, only backfill if explicitly listed
        return shared_history_channels && 
               shared_history_channels->find(channel_id) != shared_history_channels->end();
    }
    
    bool is_isolated_channel(uint64_t channel_id) const {
        return isolated_channels && isolated_channels->find(channel_id) != isolated_channels->end();
    }
    
    void log_message(const std::string& message) const {
        if (log_callback) {
            log_callback(message);
        }
    }
    
    // Fetch message history from a channel
    void fetch_channel_history(uint64_t channel_id, uint64_t before_message_id = 0) {
        if (!bot) {
            log_message("Cannot fetch history: bot not available");
            return;
        }
        
        try {
            bot->messages_get(channel_id, before_message_id, 0, 0, MESSAGES_PER_FETCH, 
                [this, channel_id](const dpp::confirmation_callback_t& callback) {
                    if (callback.is_error()) {
                        log_message("Error fetching history for channel " + std::to_string(channel_id) + 
                                   ": " + callback.get_error().human_readable);
                        return;
                    }
                    
                    auto messages = callback.get<dpp::message_map>();
                    process_fetched_history(channel_id, messages);
                });
        } catch (const std::exception& e) {
            log_message("Exception while fetching channel history: " + std::string(e.what()));
        }
    }
    
    // Process fetched message history
    void process_fetched_history(uint64_t channel_id, const dpp::message_map& messages) {
        std::lock_guard<std::mutex> lock(backfill_mutex);
        
        auto& state = channel_backfill_state[channel_id];
        std::vector<HistoryMessage> channel_messages;
        
        // Convert D++ messages to our format, filtering out bots
        for (const auto& [id, msg] : messages) {
            if (msg.author.is_bot() || msg.content.empty()) {
                continue;
            }
            
            HistoryMessage hist_msg;
            hist_msg.message_id = msg.id;
            hist_msg.user_id = msg.author.id;
            hist_msg.channel_id = channel_id;
            hist_msg.username = msg.author.username;
            hist_msg.content = msg.content;
            hist_msg.timestamp = std::chrono::system_clock::time_point(
                std::chrono::seconds(msg.sent)
            );
            
            channel_messages.push_back(hist_msg);
        }
        
        if (channel_messages.empty()) {
            state.backfill_complete = true;
            log_message("No more messages to fetch for channel " + std::to_string(channel_id));
            check_backfill_completion();
            return;
        }
        
        // Sort messages by timestamp (oldest first for proper context order)
        std::sort(channel_messages.begin(), channel_messages.end(),
            [](const HistoryMessage& a, const HistoryMessage& b) {
                return a.timestamp < b.timestamp;
            });
        
        // Update state
        state.last_message_id = channel_messages.back().message_id;
        state.messages_fetched += static_cast<int32_t>(channel_messages.size());
        
        // Calculate token usage for these messages
        int32_t message_tokens = 0;
        for (const auto& msg : channel_messages) {
            message_tokens += estimate_message_tokens(msg.username, msg.content);
        }
        state.estimated_tokens_added += message_tokens;
        
        // Add to appropriate context and check capacity
        bool added_successfully = add_history_to_context(channel_id, channel_messages);
        
        log_message("Processed " + std::to_string(channel_messages.size()) + 
                   " history messages for channel " + std::to_string(channel_id) + 
                   " (total: " + std::to_string(state.messages_fetched) + 
                   ", tokens: " + std::to_string(message_tokens) + ")");
        if (added_successfully) {
            log_message("Successfully added history messages to context for channel " + std::to_string(channel_id));
        } else {
            log_message("Failed to add history messages to context for channel " + std::to_string(channel_id));
        }
        // Check if we should fetch more based on context capacity
        if (/*messages.size() == MESSAGES_PER_FETCH &&*/ added_successfully && should_continue_backfill(channel_id)) {
            // Fetch more messages with a small delay
            log_message("Queuing next backfill fetch for channel " + std::to_string(channel_id));
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            fetch_channel_history(channel_id, state.last_message_id);
        } else {
            state.backfill_complete = true;
            if (!added_successfully) {
                log_message("Stopped backfill for channel " + std::to_string(channel_id) + 
                           " - context capacity reached");
            }
            check_backfill_completion();
        }
    }
    
    // Add history messages to appropriate context with capacity checking
    bool add_history_to_context(uint64_t channel_id, const std::vector<HistoryMessage>& messages) {
        if (!llama_manager || messages.empty()) {
            return false;
        }
        
        std::string context_id = get_target_context_id(channel_id);
        if (context_id.empty()) {
            log_message("Failed to determine target context for channel: " + std::to_string(channel_id));
            return false;
        }
        
        // Check if context exists, create if missing (only for isolated channels during backfill)
        if (!llama_manager->has_context(context_id)) {
            // Only create isolated channel contexts during backfill
            if (is_isolated_channel(channel_id)) {
                log_message("Context '" + context_id + "' not found, creating for history backfill...");
                
                // Create context with empty system prompt (inherits from main)
                if (!llama_manager->create_context(context_id, "")) {
                    log_message("Failed to create context '" + context_id + "' for history backfill");
                    return false;
                }
                
                log_message("Successfully created context '" + context_id + "' for history backfill");
            } else {
                // For shared contexts, they should already exist
                log_message("Shared context '" + context_id + "' not available for history backfill");
                return false;
            }
        }
        
        // Check context capacity before adding messages
        if (!check_and_update_context_capacity(context_id, channel_id, messages)) {
            return false;
        }
        
        // Switch to context and add messages
        std::string original_context = llama_manager->get_active_context();
        if (!llama_manager->switch_to_context(context_id)) {
            log_message("Failed to switch to context for history: " + context_id);
            return false;
        }
        
        // Get token count before adding messages for validation
        int32_t tokens_before = llama_manager->get_message_history_token_count();
        
        // Add messages as conversation history
        for (const auto& msg : messages) {
            llama_manager->add_message_to_history(msg.username, msg.content);
        }
        
        // Batch update the context after adding all messages
        bool success = llama_manager->update_context_from_history();
        
        if (success) {
            // Validate actual token usage vs estimates
            int32_t tokens_after = llama_manager->get_message_history_token_count();
            int32_t actual_tokens_added = tokens_after - tokens_before;
            
            // Update our tracking with actual values
            update_actual_token_usage(context_id, messages, actual_tokens_added);
        } else {
            log_message("Failed to update context after adding history messages to " + context_id);
        }
        
        // Restore original context
        if (!original_context.empty()) {
            llama_manager->switch_to_context(original_context);
        }
        
        return success;
    }
    
    std::string get_target_context_id(uint64_t channel_id) const {
        if (is_isolated_channel(channel_id)) {
            return "discord_channel_" + std::to_string(channel_id);
        }
        return main_context_id;
    }
    
    void update_actual_token_usage(const std::string& context_id, const std::vector<HistoryMessage>& messages, int32_t actual_tokens_added) {
        std::lock_guard<std::mutex> capacity_lock(context_backfill_mutex);
        auto& info = context_backfill_info[context_id];
        
        int32_t estimated_added = 0;
        for (const auto& msg : messages) {
            estimated_added += estimate_message_tokens(msg.username, msg.content);
        }
        
        info.estimated_tokens += actual_tokens_added;
        
        float accuracy = (estimated_added > 0) ? ((float)actual_tokens_added / estimated_added * 100.0f) : 100.0f;
        log_message("Added " + std::to_string(messages.size()) + " history messages to context " + context_id +
                   " - Actual: " + std::to_string(actual_tokens_added) + " tokens, " +
                   "Estimated: " + std::to_string(estimated_added) + " tokens (" +
                   std::to_string(accuracy) + "% accuracy)");
    }
    
    // Enhanced context capacity tracking with actual token validation
    bool check_and_update_context_capacity(const std::string& context_id, uint64_t channel_id, const std::vector<HistoryMessage>& messages) {
        std::lock_guard<std::mutex> lock(context_backfill_mutex);
        
        // Initialize context info if not exists
        if (context_backfill_info.find(context_id) == context_backfill_info.end()) {
            ContextBackfillInfo info;
            info.context_id = context_id;
            info.is_main_context = (context_id == main_context_id);
            
            if (llama_manager) {
                int32_t context_size = llama_manager->get_context_size();
                info.capacity_limit = static_cast<int32_t>(context_size * MAX_CONTEXT_FILL_RATIO);
                
                // Get actual current usage for this context
                std::string original_context = llama_manager->get_active_context();
                if (llama_manager->switch_to_context(context_id)) {
                    info.estimated_tokens = llama_manager->get_message_history_token_count();
                    if (!original_context.empty()) {
                        llama_manager->switch_to_context(original_context);
                    }
                } else {
                    log_message("Failed to switch to context '" + context_id + "' for capacity check");
                    info.estimated_tokens = 0;
                }
            } else {
                info.capacity_limit = static_cast<int32_t>(2048 * MAX_CONTEXT_FILL_RATIO);
                info.estimated_tokens = 0;
            }
            
            context_backfill_info[context_id] = info;
            log_message("Initialized backfill tracking for context '" + context_id + 
                       "' (limit: " + std::to_string(info.capacity_limit) + " tokens, " +
                       "current: " + std::to_string(info.estimated_tokens) + " tokens)");
        }
        
        auto& info = context_backfill_info[context_id];
        
        // Add this channel to the context's associated channels if not already present
        if (std::find(info.associated_channels.begin(), info.associated_channels.end(), channel_id) == info.associated_channels.end()) {
            info.associated_channels.push_back(channel_id);
        }
        
        // Calculate tokens for new messages
        int32_t new_tokens = 0;
        for (const auto& msg : messages) {
            new_tokens += estimate_message_tokens(msg.username, msg.content);
        }
        
        // Check if adding these messages would exceed capacity
        if (info.estimated_tokens + new_tokens > info.capacity_limit) {
            log_message("Context '" + context_id + "' would exceed capacity limit (" +
                       std::to_string(info.estimated_tokens + new_tokens) + "/" +
                       std::to_string(info.capacity_limit) + " tokens) - stopping backfill for channel " +
                       std::to_string(channel_id));
            return false;
        }
        
        // Update estimated token count
        info.estimated_tokens += new_tokens;
        
        float usage_percent = (float)info.estimated_tokens / info.capacity_limit * 100.0f;
        log_message("Context '" + context_id + "' estimated usage: " + 
                   std::to_string(info.estimated_tokens) + "/" + 
                   std::to_string(info.capacity_limit) + " tokens (" +
                   std::to_string(usage_percent) + "%) after adding " +
                   std::to_string(messages.size()) + " messages from channel " +
                   std::to_string(channel_id));
        
        return true;
    }
    
    // More accurate should_continue_backfill with actual token tracking
    bool should_continue_backfill(uint64_t channel_id) const {
        if (!llama_manager) {
            return false;
        }
        
        auto state_it = channel_backfill_state.find(channel_id);
        if (state_it == channel_backfill_state.end()) {
            return false;
        }
        
        // Determine which context this channel uses
        std::string target_context = get_target_context_id(channel_id);
        
        // Check context capacity with actual usage
        std::lock_guard<std::mutex> capacity_lock(context_backfill_mutex);
        auto capacity_it = context_backfill_info.find(target_context);
        if (capacity_it != context_backfill_info.end()) {
            const auto& info = capacity_it->second;
            
            // Use actual token count if available, fall back to estimate
            int32_t current_usage = info.estimated_tokens;
            
            // Get actual current usage for more accurate decisions
            std::string original_context = llama_manager->get_active_context();
            if (llama_manager->switch_to_context(target_context)) {
                current_usage = llama_manager->get_message_history_token_count();
                if (!original_context.empty()) {
                    llama_manager->switch_to_context(original_context);
                }
            }
            
            // Stop if context is approaching our target (50% usage)
            if (current_usage >= info.capacity_limit) {
                log_message("Stopping backfill for channel " + std::to_string(channel_id) + 
                           " - context '" + target_context + "' reached target usage: " +
                           std::to_string(current_usage) + "/" + std::to_string(info.capacity_limit) + " tokens");
                return false;
            }
            
            // Also check if we're very close to the limit (within 90% of target)
            int32_t soft_limit = static_cast<int32_t>(info.capacity_limit * 0.9f);
            if (current_usage >= soft_limit) {
                log_message("Approaching backfill limit for channel " + std::to_string(channel_id) + 
                           " - context '" + target_context + "' at " +
                           std::to_string(current_usage) + "/" + std::to_string(info.capacity_limit) + " tokens");
                return false;
            }
        }
        
        // Also limit to reasonable number of messages per channel to prevent runaway backfill
        const int32_t max_messages_per_channel = 200;
        if (state_it->second.messages_fetched >= max_messages_per_channel) {
            log_message("Stopping backfill for channel " + std::to_string(channel_id) + 
                       " - reached message limit (" + std::to_string(max_messages_per_channel) + ")");
            return false;
        }
        
        return true;
    }
    
    // Enhanced backfill completion check with context reporting
    void check_backfill_completion() {
        bool all_complete = true;
        for (const auto& [channel_id, state] : channel_backfill_state) {
            if (!state.backfill_complete) {
                all_complete = false;
                break;
            }
        }
        
        if (all_complete) {
            backfill_in_progress = false;
            
            // Report final usage per context with actual token counts
            std::lock_guard<std::mutex> capacity_lock(context_backfill_mutex);
            for (const auto& [context_id, info] : context_backfill_info) {
                // Get actual final usage
                int32_t actual_usage = info.estimated_tokens;
                if (llama_manager) {
                    std::string original_context = llama_manager->get_active_context();
                    if (llama_manager->switch_to_context(context_id)) {
                        actual_usage = llama_manager->get_message_history_token_count();
                        if (!original_context.empty()) {
                            llama_manager->switch_to_context(original_context);
                        }
                    }
                }
                
                float usage_percent = (float)actual_usage / info.capacity_limit * 100.0f;
                float target_percent = MAX_CONTEXT_FILL_RATIO * 100.0f;
                
                log_message("Final backfill for context '" + context_id + "': " +
                           std::to_string(actual_usage) + "/" +
                           std::to_string(info.capacity_limit) + " tokens (" +
                           std::to_string(usage_percent) + "% of " + std::to_string(target_percent) + "% target) across " +
                           std::to_string(info.associated_channels.size()) + " channels");
                
                // Warn if we didn't reach a reasonable fill level
                if (usage_percent < (target_percent * 0.1f)) {
                    log_message("Warning: Context '" + context_id + "' has very low usage after backfill - " +
                               "may indicate insufficient message history in channels");
                }
            }
            
            log_message("Chat history backfill completed for all contexts");
        }
    }
    
    // Process guilds to find channels for backfill
    void process_guilds_for_backfill(const dpp::guild_map& guilds) {
        for (const auto& [guild_id, guild] : guilds) {
            // Get channels for this guild
            bot->channels_get(guild_id, [this, guild_id](const dpp::confirmation_callback_t& callback) {
                if (callback.is_error()) {
                    log_message("Error getting channels for guild " + std::to_string(guild_id));
                    return;
                }
                
                auto channels = callback.get<dpp::channel_map>();
                for (const auto& [channel_id, channel] : channels) {
                    // Only process text channels that we should backfill
                    if (channel.is_text_channel() && should_backfill_channel(channel_id)) {
                        std::lock_guard<std::mutex> lock(backfill_mutex);
                        channel_backfill_state[channel_id] = ChannelBackfillState{};
                        
                        // Log which type of channel is being backfilled
                        std::string channel_type = is_isolated_channel(channel_id) ? "isolated" : "shared";
                        log_message("Starting backfill for " + channel_type + " channel: " + std::to_string(channel_id) + " (" + channel.name + ")");
                        
                        // Start fetching with a small delay between channels
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        fetch_channel_history(channel_id);
                    }
                }
            });
        }
    }

public:
    DiscordHistoryLoader() = default;
    ~DiscordHistoryLoader() = default;
    
    // Configuration
    void configure(dpp::cluster* discord_bot, LlamaManager* llama_mgr, const std::string& main_ctx_id) {
        bot = discord_bot;
        llama_manager = llama_mgr;
        main_context_id = main_ctx_id;
    }
    
    void set_channel_configuration(const std::unordered_set<uint64_t>* allowed,
                                  const std::unordered_set<uint64_t>* isolated,
                                  const std::unordered_set<uint64_t>* shared_history) {
        // Ignore allowed parameter (all channels are now allowed)
        isolated_channels = isolated;
        shared_history_channels = shared_history;
    }
    
    void set_log_callback(std::function<void(const std::string&)> callback) {
        log_callback = callback;
    }
    
    // Main backfill control
    void start_backfill() {
        if (backfill_in_progress || !bot || !llama_manager) {
            return;
        }
        
        backfill_in_progress = true;
        log_message("Starting chat history backfill...");
        
        {
            std::lock_guard<std::mutex> lock(backfill_mutex);
            channel_backfill_state.clear();
        }
        
        {
            std::lock_guard<std::mutex> capacity_lock(context_backfill_mutex);
            context_backfill_info.clear();
        }
        
        // Get list of guilds the bot has access to
        try {
            bot->current_user_get_guilds([this](const dpp::confirmation_callback_t& callback) {
                if (callback.is_error()) {
                    log_message("Error getting guilds for backfill: " + callback.get_error().human_readable);
                    backfill_in_progress = false;
                    return;
                }
                
                auto guilds = callback.get<dpp::guild_map>();
                
                if (guilds.empty()) {
                    log_message("Bot is not in any guilds - only DM backfill possible");
                    backfill_in_progress = false;
                    return;
                }
                
                log_message("Found " + std::to_string(guilds.size()) + " guilds, checking channel access...");
                process_guilds_for_backfill(guilds);
            });
        } catch (const std::exception& e) {
            log_message("Exception starting backfill: " + std::string(e.what()));
            backfill_in_progress = false;
        }
    }
    
    // Status reporting
    BackfillStatus get_status() const {
        std::lock_guard<std::mutex> lock(backfill_mutex);
        std::lock_guard<std::mutex> capacity_lock(context_backfill_mutex);
        
        BackfillStatus status;
        status.in_progress = backfill_in_progress;
        status.channels_processed = static_cast<int32_t>(channel_backfill_state.size());
        status.total_messages_fetched = 0;
        status.channels_complete = 0;
        
        for (const auto& [channel_id, state] : channel_backfill_state) {
            status.total_messages_fetched += state.messages_fetched;
            if (state.backfill_complete) {
                status.channels_complete++;
            }
        }
        
        // Copy context usage information
        status.context_usage = context_backfill_info;
        
        return status;
    }
    
    bool is_in_progress() const {
        return backfill_in_progress;
    }
};
