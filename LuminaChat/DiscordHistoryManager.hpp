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
// CRITICAL CODING DIRECTIVES:
// 1. Minimalism & Performance: Deliver lean, efficient solutions that avoid unnecessary bloat.
// 2. Consistent Coding Style: Maintain uniform style and structure for clear, maintainable code.
// 3. Clear Documentation: Provide concise comments explaining complex logic and key decisions.
// 4. Eliminate Redundancy: Remove unused, obsolete, and legacy code along with excess includes.
// 5. Optimize Function Structure: Adjust function boundaries to reduce overlap and clarify responsibilities.
// 6. Preserve Core Functionality: Streamline code while safeguarding essential features.
// 7. Cross-Platform Standards: Use fixed-width types and proper initialization to ensure portability.
// 8. Smart Caching: Cache frequently used variables to reduce repeated allocations.
// 9. Ensure Logical Consistency: Review code flow to maintain coherent, error-free execution.
// 10. Continuous Refinement: Regularly refactor and verify that updates preserve stable functionality.

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
        uint64_t message_id = 0;
        uint64_t user_id = 0;
        uint64_t channel_id = 0;
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
    
    // OPTIMIZED: Consolidated context operations with better error handling
    bool ensure_context_available(const std::string& context_id) {
        if (!llama_manager || context_id.empty()) return false;
        
        try {
            return llama_manager->has_context(context_id) || 
                   llama_manager->create_context(context_id, "");
        } catch (const std::exception&) {
            return false;
        }
    }
    
    bool switch_context_safely(const std::string& context_id) {
        if (!llama_manager || context_id.empty()) return false;
        
        try {
            return llama_manager->has_context(context_id) && 
                   llama_manager->switch_to_context(context_id);
        } catch (const std::exception&) {
            return false;
        }
    }

    // STREAMLINED: Simplified fetch with better error handling
    void fetch_channel_history(uint64_t channel_id, uint64_t before_message_id = 0) {
        if (!bot || !llama_manager) return;
        
        int32_t fetch_size = MESSAGES_PER_FETCH;
        {
            std::lock_guard<std::mutex> lock(backfill_mutex);
            auto state_it = channel_backfill_state.find(channel_id);
            if (state_it != channel_backfill_state.end() && state_it->second.is_shared_channel) {
                fetch_size = shared_round_robin.messages_per_round;
            }
        }
        
        bot->messages_get(channel_id, before_message_id, 0, 0, fetch_size, 
            [this, channel_id](const dpp::confirmation_callback_t& callback) {
                if (callback.is_error()) {
                    handle_channel_backfill_complete(channel_id, false);
                    return;
                }
                process_fetched_history(channel_id, callback.get<dpp::message_map>());
            });
    }
    
    // OPTIMIZED: Reduced function complexity and improved token tracking
    void process_fetched_history(uint64_t channel_id, const dpp::message_map& messages) {
        std::lock_guard<std::mutex> lock(backfill_mutex);
        
        auto& state = channel_backfill_state[channel_id];
        std::vector<HistoryMessage> channel_messages;
        channel_messages.reserve(messages.size());
        
        // Convert and filter messages
        for (const auto& [id, msg] : messages) {
            if (msg.author.is_bot() || msg.content.empty()) continue;
            
            channel_messages.emplace_back(HistoryMessage{
                msg.id, msg.author.id, channel_id, msg.author.username, msg.content,
                std::chrono::system_clock::time_point(std::chrono::seconds(msg.sent))
            });
        }
        
        if (channel_messages.empty()) {
            handle_channel_backfill_complete(channel_id, true);
            return;
        }
        
        // Sort by timestamp (oldest first)
        std::sort(channel_messages.begin(), channel_messages.end(),
            [](const HistoryMessage& a, const HistoryMessage& b) {
                return a.timestamp < b.timestamp;
            });
        
        // Update state and process
        state.last_message_id = channel_messages.back().message_id;
        state.messages_fetched += static_cast<int32_t>(channel_messages.size());
        
        int32_t tokens_added = 0;
        bool success = add_history_to_context(channel_id, channel_messages, tokens_added);
        state.estimated_tokens_added += tokens_added;
        
        // Handle continuation
        if (success && should_continue_backfill(channel_id)) {
            if (state.is_shared_channel) {
                schedule_next_shared_channel_fetch();
            } else if (messages.size() >= static_cast<size_t>(shared_round_robin.messages_per_round)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                fetch_channel_history(channel_id, state.last_message_id);
            } else {
                handle_channel_backfill_complete(channel_id, true);
            }
        } else {
            handle_channel_backfill_complete(channel_id, !success);
        }
    }
    
    // CONSOLIDATED: Simplified channel completion handling
    void handle_channel_backfill_complete(uint64_t channel_id, bool natural_completion) {
        {
            std::lock_guard<std::mutex> lock(backfill_mutex);
            auto state_it = channel_backfill_state.find(channel_id);
            if (state_it != channel_backfill_state.end()) {
                state_it->second.backfill_complete = true;
            }
        }
        
        {
            std::lock_guard<std::mutex> rr_lock(round_robin_mutex);
            shared_round_robin.remove_channel(channel_id);
        }
        
        check_backfill_completion();
    }
    
    // ADDED: Missing round-robin scheduler for shared channels
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
    
    // OPTIMIZED: Streamlined context operations with better caching
    bool add_history_to_context(uint64_t channel_id, const std::vector<HistoryMessage>& messages, int32_t& actual_tokens_added) {
        if (!llama_manager || messages.empty()) {
            actual_tokens_added = 0;
            return false;
        }
        
        // Determine target context
        std::string context_id;
        bool is_isolated_chan = is_isolated_channel(channel_id);
        
        if (is_isolated_chan) {
            if (channel_contexts) {
                auto it = channel_contexts->find(channel_id);
                context_id = (it != channel_contexts->end()) ? it->second : 
                            ("discord_channel_" + std::to_string(channel_id));
            } else {
                actual_tokens_added = 0;
                return false;
            }
        } else {
            context_id = main_context_id;
        }
        
        if (context_id.empty() || !ensure_context_available(context_id)) {
            actual_tokens_added = 0;
            return false;
        }
        
        if (!check_context_capacity_before_add(context_id, channel_id, messages)) {
            actual_tokens_added = 0;
            return false;
        }
        
        // Switch context and add messages
        std::string original_context = llama_manager->get_active_context();
        if (!switch_context_safely(context_id)) {
            actual_tokens_added = 0;
            return false;
        }
        
        // CACHED: Get token count once before processing
        int32_t tokens_before = 0;
        try {
            tokens_before = llama_manager->get_message_history_token_count();
        } catch (const std::exception&) {
            tokens_before = 0;
        }
        
        // Add all messages
        for (const auto& msg : messages) {
            llama_manager->add_message_to_history(msg.username, msg.content);
        }
        
        bool success = llama_manager->update_context_from_history();
        
        if (success) {
            try {
                int32_t tokens_after = llama_manager->get_message_history_token_count();
                actual_tokens_added = std::max(0, tokens_after - tokens_before);
            } catch (const std::exception&) {
                actual_tokens_added = 0;
            }
            
            update_context_usage_tracking(context_id, channel_id, actual_tokens_added);
        } else {
            actual_tokens_added = 0;
        }
        
        // Restore original context
        if (!original_context.empty() && original_context != context_id) {
            switch_context_safely(original_context);
        }
        
        return success;
    }

    // OPTIMIZED: Reduced mutex contention and simplified capacity checking
    bool check_context_capacity_before_add(const std::string& context_id, uint64_t channel_id, const std::vector<HistoryMessage>& messages) {
        std::lock_guard<std::mutex> lock(context_backfill_mutex);
        
        if (context_backfill_info.find(context_id) == context_backfill_info.end()) {
            initialize_context_backfill_info(context_id);
        }
        
        auto& info = context_backfill_info[context_id];
        info.actual_tokens = get_actual_context_usage(context_id);
        info.last_updated = std::chrono::system_clock::now();
        
        // CACHED: Calculate estimated tokens once
        int32_t estimated_new_tokens = 0;
        for (const auto& msg : messages) {
            estimated_new_tokens += estimate_message_tokens(msg.username, msg.content);
        }
        
        return (info.actual_tokens + estimated_new_tokens) <= info.capacity_limit;
    }

    // SIMPLIFIED: Streamlined initialization with better defaults
    void initialize_context_backfill_info(const std::string& context_id) {
        if (context_id.empty()) return;
        
        ContextBackfillInfo info{
            .context_id = context_id,
            .actual_tokens = 0,
            .capacity_limit = static_cast<int32_t>(2048 * MAX_CONTEXT_FILL_RATIO),
            .is_main_context = (context_id == main_context_id),
            .last_updated = std::chrono::system_clock::now()
        };
        
        if (llama_manager) {
            try {
                int32_t context_size = llama_manager->get_context_size();
                info.capacity_limit = static_cast<int32_t>(context_size * MAX_CONTEXT_FILL_RATIO);
                info.actual_tokens = get_actual_context_usage(context_id);
            } catch (const std::exception&) {
                // Use defaults already set
            }
        }
        
        context_backfill_info[context_id] = info;
    }
    
    // OPTIMIZED: Cached context switching for better performance
    int32_t get_actual_context_usage(const std::string& context_id) {
        if (!llama_manager) return 0;
        
        std::string original_context = llama_manager->get_active_context();
        int32_t usage = 0;
        
        if (original_context == context_id) {
            // Already in target context
            usage = llama_manager->get_message_history_token_count();
        } else if (llama_manager->switch_to_context(context_id)) {
            usage = llama_manager->get_message_history_token_count();
            if (!original_context.empty()) {
                llama_manager->switch_to_context(original_context);
            }
        }
        
        return usage;
    }
    
    // STREAMLINED: Simplified usage tracking
    void update_context_usage_tracking(const std::string& context_id, uint64_t channel_id, int32_t tokens_added) {
        std::lock_guard<std::mutex> lock(context_backfill_mutex);
        
        auto& info = context_backfill_info[context_id];
        info.actual_tokens += tokens_added;
        info.last_updated = std::chrono::system_clock::now();
        
        if (info.is_main_context) {
            info.channel_contributions[channel_id] += tokens_added;
        }
        
        // OPTIMIZED: Only add if not already present
        if (std::find(info.associated_channels.begin(), info.associated_channels.end(), channel_id) == info.associated_channels.end()) {
            info.associated_channels.push_back(channel_id);
        }
    }
    
    // CACHED: Improved continuation logic with better performance
    bool should_continue_backfill(uint64_t channel_id) const {
        if (!llama_manager) return false;
        
        auto state_it = channel_backfill_state.find(channel_id);
        if (state_it == channel_backfill_state.end()) return false;
        
        // Check message limit first (cheapest check)
        const int32_t max_messages_per_channel = 200;
        if (state_it->second.messages_fetched >= max_messages_per_channel) {
            return false;
        }
        
        // Determine target context
        std::string target_context = is_isolated_channel(channel_id) ? 
            (channel_contexts ? 
                (channel_contexts->count(channel_id) ? channel_contexts->at(channel_id) : 
                 ("discord_channel_" + std::to_string(channel_id))) : "") :
            main_context_id;
        
        // Check capacity limits
        std::lock_guard<std::mutex> capacity_lock(context_backfill_mutex);
        auto capacity_it = context_backfill_info.find(target_context);
        if (capacity_it != context_backfill_info.end()) {
            const auto& info = capacity_it->second;
            int32_t current_usage = const_cast<DiscordHistoryManager*>(this)->get_actual_context_usage(target_context);
            
            // Use 90% soft limit for early termination
            int32_t soft_limit = static_cast<int32_t>(info.capacity_limit * 0.9f);
            return current_usage < soft_limit;
        }
        
        return true;
    }
    
    // OPTIMIZED: Reduced complexity in completion checking
    void check_backfill_completion() {
        // Quick check without detailed logging
        bool all_complete = std::all_of(channel_backfill_state.begin(), channel_backfill_state.end(),
            [](const auto& pair) { return pair.second.backfill_complete; });
        
        if (all_complete) {
            backfill_in_progress = false;
            log_message("Chat history backfill completed for all contexts");
        }
    }
    
    // STREAMLINED: Simplified guild processing with better error handling
    void process_guilds_for_backfill(const dpp::guild_map& guilds) {
        std::vector<uint64_t> shared_channels_to_backfill, isolated_channels_to_backfill;
        std::atomic<int> pending_requests{static_cast<int>(guilds.size())};
        
        for (const auto& [guild_id, guild] : guilds) {
            bot->channels_get(guild_id, [this, &shared_channels_to_backfill, &isolated_channels_to_backfill, &pending_requests](const dpp::confirmation_callback_t& callback) {
                if (!callback.is_error()) {
                    auto channels = callback.get<dpp::channel_map>();
                    std::vector<uint64_t> local_shared, local_isolated;
                    
                    for (const auto& [channel_id, channel] : channels) {
                        if (channel.is_text_channel() && is_channel_allowed(channel_id) && should_backfill_shared_channel(channel_id)) {
                            bool is_shared = !is_isolated_channel(channel_id);
                            ChannelBackfillState state{
                                .last_message_id = 0,
                                .backfill_complete = false,
                                .messages_fetched = 0,
                                .estimated_tokens_added = 0,
                                .target_context_id = is_shared ? main_context_id : ("discord_channel_" + std::to_string(channel_id)),
                                .is_shared_channel = is_shared,
                                .round_robin_position = 0
                            };
                            
                            {
                                std::lock_guard<std::mutex> lock(backfill_mutex);
                                channel_backfill_state[channel_id] = state;
                            }
                            
                            (is_shared ? local_shared : local_isolated).push_back(channel_id);
                        }
                    }
                    
                    // Thread-safe concatenation
                    shared_channels_to_backfill.insert(shared_channels_to_backfill.end(), local_shared.begin(), local_shared.end());
                    isolated_channels_to_backfill.insert(isolated_channels_to_backfill.end(), local_isolated.begin(), local_isolated.end());
                }
                
                if (--pending_requests == 0) {
                    start_coordinated_backfill(shared_channels_to_backfill, isolated_channels_to_backfill);
                }
            });
        }
    }
    
    // OPTIMIZED: Simplified backfill coordination
    void start_coordinated_backfill(const std::vector<uint64_t>& shared_channels, const std::vector<uint64_t>& isolated_channels) {
        {
            std::lock_guard<std::mutex> rr_lock(round_robin_mutex);
            shared_round_robin.active_channels = shared_channels;
            shared_round_robin.current_index = 0;
            shared_round_robin.round_complete = false;
        }
        
        log_message("Starting backfill: " + std::to_string(shared_channels.size()) + " shared, " +
                   std::to_string(isolated_channels.size()) + " isolated channels");
        
        // Start isolated channels with staggered timing
        for (size_t i = 0; i < isolated_channels.size(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100 * i));
            fetch_channel_history(isolated_channels[i]);
        }
        
        // Start first shared channel
        if (!shared_channels.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            fetch_channel_history(shared_channels[0]);
        }
    }

public:
    DiscordHistoryManager() 
        : bot(nullptr)
        , llama_manager(nullptr)
        , allowed_channels(nullptr)
        , isolated_channels(nullptr)
        , shared_history_channels(nullptr)
        , channel_contexts(nullptr)
    {
    }
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
        if (backfill_in_progress) return true;
        
        if (!bot || !llama_manager || main_context_id.empty() || !llama_manager->has_context(main_context_id)) {
            return false;
        }
        
        backfill_in_progress = true;
        
        // Clear previous state efficiently
        {
            std::lock_guard<std::mutex> lock(backfill_mutex);
            channel_backfill_state.clear();
        }
        {
            std::lock_guard<std::mutex> capacity_lock(context_backfill_mutex);
            context_backfill_info.clear();
            initialize_context_backfill_info(main_context_id);
        }
        {
            std::lock_guard<std::mutex> rr_lock(round_robin_mutex);
            shared_round_robin = SharedChannelRoundRobin{};
        }
        
        bot->current_user_get_guilds([this](const dpp::confirmation_callback_t& callback) {
            if (callback.is_error() || callback.get<dpp::guild_map>().empty()) {
                backfill_in_progress = false;
                return;
            }
            process_guilds_for_backfill(callback.get<dpp::guild_map>());
        });
        
        return true;
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
