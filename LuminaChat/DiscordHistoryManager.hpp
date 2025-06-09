// DiscordHistoryManager.hpp - header-only implementation for Discord chat history backfilling
// Handles message history fetching, context capacity management, and round-robin distribution.
//
// File Specific Directives:
// Manage Discord message history backfilling with per-context capacity tracking.
// Implement round-robin fetching for shared channels to ensure fair distribution.
// Handle isolated channel contexts and shared main context coordination.
// Shared Channels use the main application context, while isolated channels and DMs use their own contexts.
// Message history is fetched in batches and context fill ratios are maintained to avoid overfilling contexts.
// Each unique context fills independantly tracking it's own usage and capacity.
//
// CODING DIRECTIVES:
// 1. Keep the codebase minimalistic, focused on functionality and efficiency.
// 2. Stay consistent with similar coding styles and patterns throughout the codebase.
// 3. Comment code thoroughly, where necessary, to explain complex logic or decisions.
// 4. Always eliminate unused code, dead code, legacy code, and cleanup includes.
// 5. Combine or split functions where necessary to eliminate redundancy.
// 6. Focus on overall codebase reduction without sacrificing functionality.
// 7. Use consistent _t fixed-width variable types to ensure portability across platforms.
// 8. Cache frequently used variables to avoid repeated allocations.
// 9. Ensure there are no logical errors and the execution paths flow as expected.
// 10. Refactor where necessary to maintain clean code, efficient code, and to conform to the above settings and directives.
// 11. After making changes, go back and make sure the codebase has been updated to incorporate the new changes and that it still adheres to the coding directives.
// 12. NEVER BREAK FUNCTIONALITY THAT IS ALREADY WORKING.

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <atomic>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <algorithm>
#include <dpp/dpp.h>

// Forward declarations
class LlamaManager;
void discord_manager_log_callback(const std::string& message);

class DiscordHistoryManager {
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
    
    // Forward declare ContextBackfillInfo before BackfillStatus
    struct ContextBackfillInfo {
        std::string context_id;
        int32_t actual_tokens = 0;
        int32_t capacity_limit = 0;
        std::vector<uint64_t> associated_channels;
        bool is_main_context = false;
        std::chrono::system_clock::time_point last_updated;
        std::unordered_map<uint64_t, int32_t> channel_contributions;
    };
    
    // Backfill status information
    struct BackfillStatus {
        bool in_progress = false;
        int32_t channels_processed = 0;
        int32_t total_messages_fetched = 0;
        int32_t channels_complete = 0;
        std::unordered_map<std::string, ContextBackfillInfo> context_usage;
    };
    
private:
    // Constants
    static constexpr int32_t MESSAGES_PER_FETCH = 10;
    static constexpr float MAX_CONTEXT_FILL_RATIO = 0.5f;
    
    // Dependencies
    dpp::cluster* bot;
    LlamaManager* llama_manager;
    std::string main_context_id;
    
    // Channel configuration
    const std::unordered_set<uint64_t>* allowed_channels;
    const std::unordered_set<uint64_t>* isolated_channels;
    const std::unordered_set<uint64_t>* shared_history_channels;
    const std::unordered_map<uint64_t, std::string>* channel_contexts;
    
    // Backfill state tracking
    struct ChannelBackfillState {
        uint64_t last_message_id = 0;
        bool backfill_complete = false;
        int32_t messages_fetched = 0;
        int32_t estimated_tokens_added = 0;
        std::string target_context_id;
        bool is_shared_channel = false;
        int32_t round_robin_position = 0;
    };
    
    std::unordered_map<uint64_t, ChannelBackfillState> channel_backfill_state;
    mutable std::mutex backfill_mutex;
    std::atomic<bool> backfill_in_progress{false};
    
    // Round-robin management for shared channels
    struct SharedChannelRoundRobin {
        std::vector<uint64_t> active_channels;
        int32_t current_index = 0;
        int32_t messages_per_round = 5;
        bool round_complete = false;
        
        uint64_t get_next_channel() {
            if (active_channels.empty()) return 0;
            uint64_t channel = active_channels[current_index];
            current_index = (current_index + 1) % static_cast<int32_t>(active_channels.size());
            if (current_index == 0) round_complete = true;
            return channel;
        }
        
        void remove_channel(uint64_t channel_id) {
            auto it = std::find(active_channels.begin(), active_channels.end(), channel_id);
            if (it != active_channels.end()) {
                int32_t removed_index = static_cast<int32_t>(std::distance(active_channels.begin(), it));
                active_channels.erase(it);
                if (removed_index < current_index) {
                    current_index--;
                } else if (removed_index == current_index && current_index >= static_cast<int32_t>(active_channels.size())) {
                    current_index = 0;
                }
            }
        }
    };
    
    SharedChannelRoundRobin shared_round_robin;
    mutable std::mutex round_robin_mutex;
    
    // Context capacity tracking
    std::unordered_map<std::string, ContextBackfillInfo> context_backfill_info;
    mutable std::mutex context_backfill_mutex;
    
    // Helper methods
    int32_t estimate_message_tokens(const std::string& username, const std::string& content) const {
        const int32_t base_overhead = 50;
        const int32_t role_overhead = username.length() + 20;
        const int32_t content_tokens = static_cast<int32_t>((content.length() + 2) / 3);
        return base_overhead + role_overhead + content_tokens;
    }
    
    bool is_isolated_channel(uint64_t channel_id) const {
        return isolated_channels && isolated_channels->count(channel_id) > 0;
    }
    
    bool is_channel_allowed(uint64_t channel_id) const {
        return allowed_channels && (allowed_channels->empty() || allowed_channels->count(channel_id) > 0);
    }
    
    bool should_backfill_shared_channel(uint64_t channel_id) const {
        if (isolated_channels && isolated_channels->count(channel_id) > 0) return true;
        return shared_history_channels && shared_history_channels->count(channel_id) > 0;
    }
    
    void log_message(const std::string& message) const {
        discord_manager_log_callback(message);
    }
    
    // ADDED: Helper methods for safe context operations
    bool create_context_safely(const std::string& context_id) {
        if (!llama_manager) return false;
        
        try {
            return llama_manager->create_context(context_id, "");
        } catch (const std::exception& e) {
            log_message("Exception creating context '" + context_id + "': " + std::string(e.what()));
            return false;
        }
    }
    
    bool switch_context_safely(const std::string& context_id) {
        if (!llama_manager) return false;
        
        try {
            return llama_manager->switch_to_context(context_id);
        } catch (const std::exception& e) {
            log_message("Exception switching to context '" + context_id + "': " + std::string(e.what()));
            return false;
        }
    }
    
    void fetch_channel_history(uint64_t channel_id, uint64_t before_message_id = 0) {
        if (!bot || !llama_manager) {
            log_message("Cannot fetch history: bot or llama_manager not available");
            return;
        }
        
        // Determine fetch size based on channel type
        int32_t fetch_size = MESSAGES_PER_FETCH;
        
        {
            std::lock_guard<std::mutex> lock(backfill_mutex);
            auto state_it = channel_backfill_state.find(channel_id);
            if (state_it != channel_backfill_state.end() && state_it->second.is_shared_channel) {
                fetch_size = shared_round_robin.messages_per_round;
            }
        }
        
        try {
            bot->messages_get(channel_id, before_message_id, 0, 0, fetch_size, 
                [this, channel_id](const dpp::confirmation_callback_t& callback) {
                    if (callback.is_error()) {
                        log_message("Error fetching history for channel " + std::to_string(channel_id) + 
                                   ": " + callback.get_error().human_readable);
                        handle_channel_backfill_complete(channel_id, false);
                        return;
                    }
                    
                    auto messages = callback.get<dpp::message_map>();
                    process_fetched_history(channel_id, messages);
                });
        } catch (const std::exception& e) {
            log_message("Exception while fetching channel history: " + std::string(e.what()));
            handle_channel_backfill_complete(channel_id, false);
        }
    }
    
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
            handle_channel_backfill_complete(channel_id, true);
            return;
        }
        
        // Sort messages by timestamp (oldest first)
        std::sort(channel_messages.begin(), channel_messages.end(),
            [](const HistoryMessage& a, const HistoryMessage& b) {
                return a.timestamp < b.timestamp;
            });
        
        // Update state
        state.last_message_id = channel_messages.back().message_id;
        state.messages_fetched += static_cast<int32_t>(channel_messages.size());
        
        // Add to appropriate context
        int32_t actual_tokens_added = 0;
        bool added_successfully = add_history_to_context(channel_id, channel_messages, actual_tokens_added);
        
        state.estimated_tokens_added += actual_tokens_added;
        
        log_message("Processed " + std::to_string(channel_messages.size()) + 
                   " history messages for channel " + std::to_string(channel_id) + 
                   " (total: " + std::to_string(state.messages_fetched) + 
                   ", tokens: " + std::to_string(actual_tokens_added) + ")");
        
        // Handle continuation
        if (added_successfully && should_continue_backfill(channel_id)) {
            if (state.is_shared_channel) {
                schedule_next_shared_channel_fetch();
            } else {
                if (messages.size() >= static_cast<size_t>(shared_round_robin.messages_per_round)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(250));
                    fetch_channel_history(channel_id, state.last_message_id);
                } else {
                    handle_channel_backfill_complete(channel_id, true);
                }
            }
        } else {
            handle_channel_backfill_complete(channel_id, !added_successfully);
        }
    }
    
    void handle_channel_backfill_complete(uint64_t channel_id, bool natural_completion) {
        {
            std::lock_guard<std::mutex> lock(backfill_mutex);
            auto state_it = channel_backfill_state.find(channel_id);
            if (state_it != channel_backfill_state.end()) {
                state_it->second.backfill_complete = true;
                
                if (!natural_completion) {
                    log_message("Stopped backfill for channel " + std::to_string(channel_id) + 
                               " - context capacity reached or error occurred");
                }
            }
        }
        
        {
            std::lock_guard<std::mutex> rr_lock(round_robin_mutex);
            shared_round_robin.remove_channel(channel_id);
        }
        
        check_backfill_completion();
    }
    
    void schedule_next_shared_channel_fetch() {
        std::lock_guard<std::mutex> rr_lock(round_robin_mutex);
        
        if (shared_round_robin.active_channels.empty()) {
            return;
        }
        
        uint64_t next_channel = shared_round_robin.get_next_channel();
        
        std::thread([this, next_channel]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            
            std::lock_guard<std::mutex> lock(backfill_mutex);
            auto state_it = channel_backfill_state.find(next_channel);
            if (state_it != channel_backfill_state.end() && !state_it->second.backfill_complete) {
                fetch_channel_history(next_channel, state_it->second.last_message_id);
            }
        }).detach();
    }
    
    bool add_history_to_context(uint64_t channel_id, const std::vector<HistoryMessage>& messages, int32_t& actual_tokens_added) {
        if (!llama_manager || messages.empty()) {
            actual_tokens_added = 0;
            return false;
        }
        
        bool is_isolated_chan = is_isolated_channel(channel_id);
        std::string context_id;
        
        if (is_isolated_chan) {
            // Get or create isolated channel context
            if (channel_contexts) {
                auto it = channel_contexts->find(channel_id);
                if (it != channel_contexts->end()) {
                    context_id = it->second;
                } else {
                    context_id = "discord_channel_" + std::to_string(channel_id);
                    if (!create_context_safely(context_id)) {
                        log_message("Failed to create isolated context for channel " + std::to_string(channel_id));
                        actual_tokens_added = 0;
                        return false;
                    }
                    log_message("Created isolated context for channel history: " + context_id);
                }
            } else {
                context_id = "discord_channel_" + std::to_string(channel_id);
                if (!create_context_safely(context_id)) {
                    log_message("Failed to create isolated context for channel " + std::to_string(channel_id));
                    actual_tokens_added = 0;
                    return false;
                }
            }
        } else {
            context_id = main_context_id;
        }
        
        if (context_id.empty() || !llama_manager->has_context(context_id)) {
            log_message("Context not available for history backfill: " + context_id);
            actual_tokens_added = 0;
            return false;
        }
        
        if (!check_context_capacity_before_add(context_id, channel_id, messages)) {
            actual_tokens_added = 0;
            return false;
        }
        
        // Switch to context and add messages
        std::string original_context = llama_manager->get_active_context();
        if (!switch_context_safely(context_id)) {
            log_message("Failed to switch to context for history: " + context_id);
            actual_tokens_added = 0;
            return false;
        }
        
        int32_t tokens_before = llama_manager->get_message_history_token_count();
        
        for (const auto& msg : messages) {
            llama_manager->add_message_to_history(msg.username, msg.content);
        }
        
        bool success = llama_manager->update_context_from_history();
        
        if (success) {
            int32_t tokens_after = llama_manager->get_message_history_token_count();
            actual_tokens_added = tokens_after - tokens_before;
            
            update_context_usage_tracking(context_id, channel_id, actual_tokens_added);
            
            log_message("Added " + std::to_string(messages.size()) + " history messages to context " + context_id +
                       " - Added: " + std::to_string(actual_tokens_added) + " tokens");
        } else {
            log_message("Failed to update context after adding history messages to " + context_id);
            actual_tokens_added = 0;
        }
        
        // Restore original context
        if (!original_context.empty()) {
            llama_manager->switch_to_context(original_context);
        }
        
        return success;
    }
    
    bool check_context_capacity_before_add(const std::string& context_id, uint64_t channel_id, const std::vector<HistoryMessage>& messages) {
        std::lock_guard<std::mutex> lock(context_backfill_mutex);
        
        if (context_backfill_info.find(context_id) == context_backfill_info.end()) {
            initialize_context_backfill_info(context_id);
        }
        
        auto& info = context_backfill_info[context_id];
        
        int32_t current_usage = get_actual_context_usage(context_id);
        info.actual_tokens = current_usage;
        info.last_updated = std::chrono::system_clock::now();
        
        int32_t estimated_new_tokens = 0;
        for (const auto& msg : messages) {
            estimated_new_tokens += estimate_message_tokens(msg.username, msg.content);
        }
        
        if (current_usage + estimated_new_tokens > info.capacity_limit) {
            log_message("Context '" + context_id + "' would exceed capacity limit (" +
                       std::to_string(current_usage + estimated_new_tokens) + "/" +
                       std::to_string(info.capacity_limit) + " tokens) - stopping backfill");
            return false;
        }
        
        float usage_percent = (float)(current_usage + estimated_new_tokens) / info.capacity_limit * 100.0f;
        log_message("Context '" + context_id + "' projected usage: " + 
                   std::to_string(current_usage + estimated_new_tokens) + "/" + 
                   std::to_string(info.capacity_limit) + " tokens (" +
                   std::to_string(usage_percent) + "%)");
        
        return true;
    }
    
    void initialize_context_backfill_info(const std::string& context_id) {
        ContextBackfillInfo info;
        info.context_id = context_id;
        info.is_main_context = (context_id == main_context_id);
        info.last_updated = std::chrono::system_clock::now();
        
        if (llama_manager) {
            int32_t context_size = llama_manager->get_context_size();
            info.capacity_limit = static_cast<int32_t>(context_size * MAX_CONTEXT_FILL_RATIO);
            info.actual_tokens = get_actual_context_usage(context_id);
        } else {
            info.capacity_limit = static_cast<int32_t>(2048 * MAX_CONTEXT_FILL_RATIO);
            info.actual_tokens = 0;
        }
        
        context_backfill_info[context_id] = info;
        log_message("Initialized backfill tracking for context '" + context_id + 
                   "' (limit: " + std::to_string(info.capacity_limit) + " tokens, " +
                   "current: " + std::to_string(info.actual_tokens) + " tokens)");
    }
    
    int32_t get_actual_context_usage(const std::string& context_id) {
        if (!llama_manager) return 0;
        
        std::string original_context = llama_manager->get_active_context();
        int32_t usage = 0;
        
        if (llama_manager->switch_to_context(context_id)) {
            usage = llama_manager->get_message_history_token_count();
            if (!original_context.empty()) {
                llama_manager->switch_to_context(original_context);
            }
        }
        
        return usage;
    }
    
    void update_context_usage_tracking(const std::string& context_id, uint64_t channel_id, int32_t tokens_added) {
        std::lock_guard<std::mutex> lock(context_backfill_mutex);
        
        auto& info = context_backfill_info[context_id];
        info.actual_tokens += tokens_added;
        info.last_updated = std::chrono::system_clock::now();
        
        if (info.is_main_context) {
            info.channel_contributions[channel_id] += tokens_added;
        }
        
        if (std::find(info.associated_channels.begin(), info.associated_channels.end(), channel_id) == info.associated_channels.end()) {
            info.associated_channels.push_back(channel_id);
        }
    }
    
    bool should_continue_backfill(uint64_t channel_id) const {
        if (!llama_manager) return false;
        
        auto state_it = channel_backfill_state.find(channel_id);
        if (state_it == channel_backfill_state.end()) return false;
        
        std::string target_context;
        if (is_isolated_channel(channel_id)) {
            auto ctx_it = channel_contexts->find(channel_id);
            target_context = (ctx_it != channel_contexts->end()) ? ctx_it->second : ("discord_channel_" + std::to_string(channel_id));
        } else {
            target_context = main_context_id;
        }
        
        std::lock_guard<std::mutex> capacity_lock(context_backfill_mutex);
        auto capacity_it = context_backfill_info.find(target_context);
        if (capacity_it != context_backfill_info.end()) {
            const auto& info = capacity_it->second;
            
            int32_t current_usage = const_cast<DiscordHistoryManager*>(this)->get_actual_context_usage(target_context);
            
            if (current_usage >= info.capacity_limit) {
                return false;
            }
            
            int32_t soft_limit = static_cast<int32_t>(info.capacity_limit * 0.9f);
            if (current_usage >= soft_limit) {
                return false;
            }
        }
        
        const int32_t max_messages_per_channel = 200;
        return state_it->second.messages_fetched < max_messages_per_channel;
    }
    
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
            
            std::lock_guard<std::mutex> capacity_lock(context_backfill_mutex);
            for (const auto& [context_id, info] : context_backfill_info) {
                int32_t actual_usage = info.actual_tokens;
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
                
                if (usage_percent < (target_percent * 0.1f)) {
                    log_message("Warning: Context '" + context_id + "' has very low usage after backfill - " +
                               "may indicate insufficient message history in channels");
                }
            }
            
            log_message("Chat history backfill completed for all contexts");
        }
    }
    
    void process_guilds_for_backfill(const dpp::guild_map& guilds) {
        std::vector<uint64_t> shared_channels_to_backfill;
        std::vector<uint64_t> isolated_channels_to_backfill;
        
        // Track how many guild channel requests are pending
        std::atomic<int> pending_requests{static_cast<int>(guilds.size())};
        
        for (const auto& [guild_id, guild] : guilds) {
            bot->channels_get(guild_id, [this, guild_id, &shared_channels_to_backfill, &isolated_channels_to_backfill, &pending_requests](const dpp::confirmation_callback_t& callback) {
                if (callback.is_error()) {
                    log_message("Error getting channels for guild " + std::to_string(guild_id));
                    pending_requests--;
                    return;
                }
                
                auto channels = callback.get<dpp::channel_map>();
                std::vector<uint64_t> local_shared;
                std::vector<uint64_t> local_isolated;
                
                for (const auto& [channel_id, channel] : channels) {
                    if (channel.is_text_channel() && is_channel_allowed(channel_id) && should_backfill_shared_channel(channel_id)) {
                        std::lock_guard<std::mutex> lock(backfill_mutex);
                        
                        ChannelBackfillState state;
                        state.is_shared_channel = !is_isolated_channel(channel_id);
                        state.target_context_id = state.is_shared_channel ? main_context_id : ("discord_channel_" + std::to_string(channel_id));
                        
                        channel_backfill_state[channel_id] = state;
                        
                        if (state.is_shared_channel) {
                            local_shared.push_back(channel_id);
                        } else {
                            local_isolated.push_back(channel_id);
                        }
                        
                        std::string channel_type = state.is_shared_channel ? "shared" : "isolated";
                        log_message("Prepared " + channel_type + " channel for backfill: " + std::to_string(channel_id) + " (" + channel.name + ")");
                    }
                }
                
                // Add to global lists
                shared_channels_to_backfill.insert(shared_channels_to_backfill.end(), local_shared.begin(), local_shared.end());
                isolated_channels_to_backfill.insert(isolated_channels_to_backfill.end(), local_isolated.begin(), local_isolated.end());
                
                // Check if this was the last request
                if (--pending_requests == 0) {
                    start_coordinated_backfill(shared_channels_to_backfill, isolated_channels_to_backfill);
                }
            });
        }
    }
    
    void start_coordinated_backfill(const std::vector<uint64_t>& shared_channels, const std::vector<uint64_t>& isolated_channels) {
        {
            std::lock_guard<std::mutex> rr_lock(round_robin_mutex);
            shared_round_robin.active_channels = shared_channels;
            shared_round_robin.current_index = 0;
            shared_round_robin.round_complete = false;
        }
        
        log_message("Starting backfill: " + std::to_string(shared_channels.size()) + " shared channels (round-robin), " +
                   std::to_string(isolated_channels.size()) + " isolated channels");
        
        for (uint64_t channel_id : isolated_channels) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            fetch_channel_history(channel_id);
        }
        
        if (!shared_channels.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            fetch_channel_history(shared_channels[0]);
        }
    }

public:
    DiscordHistoryManager() = default;
    ~DiscordHistoryManager() = default;
    
    // Configuration
    void configure(dpp::cluster* bot_ptr, LlamaManager* llama_ptr, const std::string& main_ctx_id) {
        bot = bot_ptr;
        llama_manager = llama_ptr;
        main_context_id = main_ctx_id;
    }
    
    void set_channel_configuration(
        const std::unordered_set<uint64_t>* allowed,
        const std::unordered_set<uint64_t>* isolated,
        const std::unordered_set<uint64_t>* shared_history,
        const std::unordered_map<uint64_t, std::string>* channel_ctx
    ) {
        allowed_channels = allowed;
        isolated_channels = isolated;
        shared_history_channels = shared_history;
        channel_contexts = channel_ctx;
    }
    
    // Main interface
    bool start_backfill() {
        if (backfill_in_progress || !bot || !llama_manager) {
            log_message("Cannot start backfill - already in progress or dependencies not available");
            return false;
        }
        
        backfill_in_progress = true;
        log_message("Starting chat history backfill with round-robin for shared channels...");
        
        {
            std::lock_guard<std::mutex> lock(backfill_mutex);
            channel_backfill_state.clear();
        }
        
        {
            std::lock_guard<std::mutex> capacity_lock(context_backfill_mutex);
            context_backfill_info.clear();
        }
        
        {
            std::lock_guard<std::mutex> rr_lock(round_robin_mutex);
            shared_round_robin = SharedChannelRoundRobin{};
        }
        
        try {
            bot->current_user_get_guilds([this](const dpp::confirmation_callback_t& callback) {
                if (callback.is_error()) {
                    log_message("Error getting guilds for backfill: " + callback.get_error().human_readable);
                    backfill_in_progress = false;
                    return;
                }
                
                auto guilds = callback.get<dpp::guild_map>();
                
                if (guilds.empty()) {
                    log_message("Bot is not in any guilds - backfill completed with no channels");
                    backfill_in_progress = false;
                    return;
                }
                
                log_message("Found " + std::to_string(guilds.size()) + " guilds, initializing backfill...");
                process_guilds_for_backfill(guilds);
            });
            return true;
        } catch (const std::exception& e) {
            log_message("Exception starting backfill: " + std::string(e.what()));
            backfill_in_progress = false;
            return false;
        }
    }
    
    void stop_backfill() {
        if (!backfill_in_progress) {
            return;
        }
        
        backfill_in_progress = false;
        
        {
            std::lock_guard<std::mutex> lock(backfill_mutex);
            channel_backfill_state.clear();
        }
        
        {
            std::lock_guard<std::mutex> rr_lock(round_robin_mutex);
            shared_round_robin = SharedChannelRoundRobin{};
        }
        
        log_message("Chat history backfill stopped");
    }
    
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
        
        status.context_usage = context_backfill_info;
        
        return status;
    }
    
    bool is_backfill_in_progress() const { 
        return backfill_in_progress; 
    }
};
