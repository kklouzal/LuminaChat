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
#include <thread>
#include <algorithm>
#include <future>
#include <sstream>

#include <dpp/dpp.h>
#include "LogHandler.hpp"

class LlamaManager;

// SIMPLIFIED: Remove complex worker structures, keep only essential ones
struct HistoryMessage {
    uint64_t message_id;
    uint64_t user_id;
    uint64_t channel_id;
    std::string username;
    std::string content;
    std::chrono::system_clock::time_point timestamp;
};

struct ChannelState {
    uint64_t last_message_id = 0;
    bool fetch_complete = false;
    int32_t messages_fetched = 0;
    std::string context_id;
    bool is_isolated = false;
};

class DiscordHistoryLoader {
private:
    // Core dependencies
    dpp::cluster* bot = nullptr;
    LlamaManager* llama_manager = nullptr;
    std::string main_context_id;
    
    // Channel configuration
    const std::unordered_set<uint64_t>* isolated_channels = nullptr;
    const std::unordered_set<uint64_t>* shared_history_channels = nullptr;
    
    // Settings
    bool history_enabled = true;
    int32_t context_fill_percentage = 50;
    
    // SIMPLIFIED: Direct channel processing
    std::unordered_map<uint64_t, ChannelState> channel_states;
    std::vector<uint64_t> shared_channels_list; // For round-robin
    size_t shared_channel_index = 0;
    mutable std::mutex state_mutex;
    
    // State
    std::atomic<bool> backfill_in_progress{false};
    
    // Constants
    static constexpr int32_t MESSAGES_PER_FETCH = 10;
    static constexpr float BASE_MAX_CONTEXT_FILL_RATIO = 0.01f;
    static constexpr int32_t MAX_RETRIES = 3;
    
    // SIMPLIFIED: Token estimation
    static constexpr int32_t estimate_message_tokens(const std::string& username, const std::string& content) {
        return 50 + static_cast<int32_t>(username.length()) + 20 + static_cast<int32_t>((content.length() + 2) / 3);
    }
    
    bool should_backfill_channel(uint64_t channel_id) const {
        return (isolated_channels && isolated_channels->count(channel_id)) ||
               (shared_history_channels && shared_history_channels->count(channel_id));
    }
    
    bool is_isolated_channel(uint64_t channel_id) const {
        return isolated_channels && isolated_channels->count(channel_id);
    }
    
    void log_message(const std::string& message) const {
        DISCORD_HISTORY_LOG(message);
    }
    
    // SIMPLIFIED: Direct context switching and capacity checking
    bool can_add_more_messages(const std::string& context_id) {
        if (!llama_manager) return false;
        
        std::string original_context = llama_manager->get_active_context();
        bool result = false;
        
        if (llama_manager->switch_to_context(context_id)) {
            int32_t current_tokens = llama_manager->get_message_history_token_count();
            int32_t context_size = llama_manager->get_context_size();
            float fill_ratio = BASE_MAX_CONTEXT_FILL_RATIO * context_fill_percentage;
            int32_t capacity_limit = static_cast<int32_t>(context_size * fill_ratio);
            
            result = current_tokens < capacity_limit;
            
            if (!original_context.empty()) {
                llama_manager->switch_to_context(original_context);
            }
        }
        
        return result;
    }
    
    // SIMPLIFIED: Direct message fetching and processing
    bool process_channel_batch(uint64_t channel_id) {
        auto& state = channel_states[channel_id];
        if (state.fetch_complete || !can_add_more_messages(state.context_id)) {
            return false;
        }
        
        log_message("Fetching messages from channel " + std::to_string(channel_id));
        
        auto promise = std::make_shared<std::promise<bool>>();
        auto future = promise->get_future();
        
        bot->messages_get(channel_id, state.last_message_id, 0, 0, MESSAGES_PER_FETCH,
            [this, channel_id, promise](const dpp::confirmation_callback_t& callback) {
                bool success = false;
                
                try {
                    if (callback.is_error()) {
                        log_message("Error fetching messages for channel " + std::to_string(channel_id));
                        promise->set_value(false);
                        return;
                    }
                    
                    auto messages = callback.get<dpp::message_map>();
                    if (messages.empty()) {
                        std::lock_guard<std::mutex> lock(state_mutex);
                        channel_states[channel_id].fetch_complete = true;
                        promise->set_value(false);
                        return;
                    }
                    
                    success = add_messages_to_context(messages, channel_id);
                    promise->set_value(success);
                    
                } catch (const std::exception& e) {
                    log_message("Exception processing channel " + std::to_string(channel_id) + ": " + e.what());
                    promise->set_value(false);
                }
            });
        
        // Wait for result with timeout
        auto status = future.wait_for(std::chrono::seconds(10));
        if (status == std::future_status::ready) {
            return future.get();
        } else {
            std::lock_guard<std::mutex> lock(state_mutex);
            channel_states[channel_id].fetch_complete = true;
            return false;
        }
    }
    
    bool add_messages_to_context(const dpp::message_map& messages, uint64_t channel_id) {
        if (!llama_manager || messages.empty()) return false;
        
        auto& state = channel_states[channel_id];
        
        // Switch to appropriate context
        std::string original_context = llama_manager->get_active_context();
        if (!llama_manager->switch_to_context(state.context_id)) {
            log_message("Failed to switch to context: " + state.context_id);
            return false;
        }
        
        // Process messages in chronological order
        std::vector<std::pair<std::chrono::system_clock::time_point, dpp::message>> sorted_messages;
        for (const auto& [id, msg] : messages) {
            if (!msg.author.is_bot() && !msg.content.empty()) {
                sorted_messages.emplace_back(
                    std::chrono::system_clock::time_point(std::chrono::seconds(msg.sent)), msg
                );
            }
        }
        
        std::sort(sorted_messages.begin(), sorted_messages.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        
        // Add messages to context
        int32_t added_count = 0;
        uint64_t latest_id = state.last_message_id;
        
        for (const auto& [timestamp, msg] : sorted_messages) {
            llama_manager->add_message_to_history(msg.author.username, msg.content);
            added_count++;
            // FIXED: Ensure both arguments to std::max are the same type
            latest_id = std::max(latest_id, static_cast<uint64_t>(msg.id));
        }
        
        bool success = llama_manager->update_context_from_history();
        
        if (success) {
            std::lock_guard<std::mutex> lock(state_mutex);
            state.messages_fetched += added_count;
            state.last_message_id = latest_id;
            
            log_message("Added " + std::to_string(added_count) + " messages to context " + state.context_id);
        }
        
        // Restore original context
        if (!original_context.empty()) {
            llama_manager->switch_to_context(original_context);
        }
        
        return success;
    }
    
    // SIMPLIFIED: Round-robin for shared channels
    uint64_t get_next_shared_channel() {
        if (shared_channels_list.empty()) return 0;
        
        uint64_t channel_id = shared_channels_list[shared_channel_index];
        shared_channel_index = (shared_channel_index + 1) % shared_channels_list.size();
        return channel_id;
    }
    
    // SIMPLIFIED: Main processing loop
    void process_all_channels() {
        const int32_t MAX_ITERATIONS = 1000;
        int32_t iteration = 0;
        
        while (backfill_in_progress && iteration < MAX_ITERATIONS) {
            bool made_progress = false;
            
            // Process isolated channels
            for (const auto& [channel_id, state] : channel_states) {
                if (state.is_isolated && !state.fetch_complete) {
                    if (process_channel_batch(channel_id)) {
                        made_progress = true;
                    }
                }
            }
            
            // Process one shared channel (round-robin)
            if (!shared_channels_list.empty()) {
                uint64_t channel_id = get_next_shared_channel();
                auto it = channel_states.find(channel_id);
                if (it != channel_states.end() && !it->second.fetch_complete) {
                    if (process_channel_batch(channel_id)) {
                        made_progress = true;
                    }
                }
            }
            
            iteration++;
            
            if (!made_progress) {
                // Check if all channels are complete
                bool all_complete = true;
                for (const auto& [channel_id, state] : channel_states) {
                    if (!state.fetch_complete) {
                        all_complete = false;
                        break;
                    }
                }
                
                if (all_complete) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
            }
        }
        
        backfill_in_progress = false;
        log_message("Chat history backfill completed");
    }

public:
    struct BackfillStatus {
        bool in_progress;
        int32_t total_channels;
        int32_t completed_channels;
        int32_t total_messages_fetched;
    };

    DiscordHistoryLoader() = default;
    ~DiscordHistoryLoader() = default;
    
    void configure(dpp::cluster* discord_bot, LlamaManager* llama_mgr, const std::string& main_ctx_id) {
        bot = discord_bot;
        llama_manager = llama_mgr;
        main_context_id = main_ctx_id;
    }
    
    void set_channel_configuration(const std::unordered_set<uint64_t>* allowed,
                                  const std::unordered_set<uint64_t>* isolated,
                                  const std::unordered_set<uint64_t>* shared_history) {
        isolated_channels = isolated;
        shared_history_channels = shared_history;
    }
    
    void set_history_settings(bool enabled, int32_t fill_percentage) {
        history_enabled = enabled;
        context_fill_percentage = std::clamp(fill_percentage, 10, 80);
    }
    
    void start_backfill() {
        if (!history_enabled || backfill_in_progress || !bot || !llama_manager) return;
        
        backfill_in_progress = true;
        log_message("Starting simplified chat history backfill...");
        
        // Get channels and setup states
        bot->current_user_get_guilds([this](const dpp::confirmation_callback_t& callback) {
            if (callback.is_error()) {
                log_message("Error getting guilds: " + callback.get_error().human_readable);
                backfill_in_progress = false;
                return;
            }
            
            auto guilds = callback.get<dpp::guild_map>();
            setup_channel_states(guilds);
        });
    }
    
    void setup_channel_states(const dpp::guild_map& guilds) {
        std::lock_guard<std::mutex> lock(state_mutex);
        channel_states.clear();
        shared_channels_list.clear();
        shared_channel_index = 0;
        
        auto process_guilds = [this, guilds]() {
            for (const auto& [guild_id, guild] : guilds) {
                bot->channels_get(guild_id, [this, guild_id](const dpp::confirmation_callback_t& callback) {
                    if (!callback.is_error()) {
                        auto channels = callback.get<dpp::channel_map>();
                        
                        std::lock_guard<std::mutex> lock(state_mutex);
                        for (const auto& [channel_id, channel] : channels) {
                            if (channel.is_text_channel() && should_backfill_channel(channel_id)) {
                                ChannelState state;
                                state.is_isolated = is_isolated_channel(channel_id);
                                
                                if (state.is_isolated) {
                                    state.context_id = "discord_channel_" + std::to_string(channel_id);
                                    // Create isolated context if needed
                                    if (!llama_manager->has_context(state.context_id)) {
                                        llama_manager->create_context(state.context_id, "");
                                    }
                                } else {
                                    state.context_id = main_context_id;
                                    shared_channels_list.push_back(channel_id);
                                }
                                
                                channel_states[channel_id] = state;
                                log_message("Configured channel " + std::to_string(channel_id) + 
                                           " for " + (state.is_isolated ? "isolated" : "shared") + " context");
                            }
                        }
                    }
                });
            }
            
            // Start processing after a short delay
            std::thread([this]() {
                std::this_thread::sleep_for(std::chrono::seconds(2));
                if (backfill_in_progress) {
                    process_all_channels();
                }
            }).detach();
        };
        
        std::thread(process_guilds).detach();
    }
    
    BackfillStatus get_status() const {
        std::lock_guard<std::mutex> lock(state_mutex);
        
        BackfillStatus status;
        status.in_progress = backfill_in_progress;
        status.total_channels = static_cast<int32_t>(channel_states.size());
        status.completed_channels = 0;
        status.total_messages_fetched = 0;
        
        for (const auto& [channel_id, state] : channel_states) {
            if (state.fetch_complete) {
                status.completed_channels++;
            }
            status.total_messages_fetched += state.messages_fetched;
        }
        
        return status;
    }
    
    bool is_in_progress() const { return backfill_in_progress; }
};
