// DiscordManager.hpp - header-only implementation for Discord integration via D++
// Handles Discord bot functionality and integration with LuminaChat backend.
//
// File Specific Directives:
// Manage Discord bot lifecycle, message handling, and integration with LlamaManager.
// Handle Discord rate limiting and async operations properly.
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

#include <iostream>
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
#include <functional>
#include <string_view>
#include <sstream>
#include <dpp/dpp.h>

#include "DiscordHistoryLoader.hpp"
#include "LogHandler.hpp"

// Discord bot configuration structure
struct DiscordBotConfig {
    std::string bot_token;
    std::string application_id;
    uint64_t guild_id = 0;  // 0 = global commands
    bool auto_reconnect = true;
    bool enable_message_cache = true;
    uint32_t message_cache_size = 100;
    uint32_t rate_limit_buffer_ms = 100;
};

// Discord message context for tracking conversations
struct DiscordMessageContext {
    uint64_t user_id;
    uint64_t channel_id;
    uint64_t guild_id;
    std::string username;
    std::string channel_name;
    std::chrono::system_clock::time_point timestamp;
    bool is_dm;
};

class DiscordManager {
private:
    // Core components - consolidated
    std::unique_ptr<dpp::cluster> bot;
    std::unique_ptr<DiscordHistoryLoader> history_loader;
    LlamaManager* llama_manager;
    
    // Configuration
    DiscordBotConfig config;
    std::string main_context_id;
    
    // State - using atomic where possible
    std::atomic<bool> is_running{false};
    std::atomic<bool> is_connected{false};
    std::atomic<bool> should_stop{false};
    
    // Channel configuration - consolidated mutex
    std::unordered_set<uint64_t> isolated_channels;
    std::unordered_set<uint64_t> shared_history_channels;
    bool allow_dms = true;
    bool pull_message_history = true;
    int32_t history_fill_percentage = 50;
    mutable std::mutex channel_config_mutex; // Renamed for clarity
    
    // Message and context management - consolidated
    std::unordered_map<uint64_t, std::vector<DiscordMessageContext>> user_conversations;
    std::unordered_map<uint64_t, std::string> user_contexts;
    std::unordered_map<uint64_t, std::string> channel_contexts;
    std::unordered_map<uint64_t, std::chrono::system_clock::time_point> last_response_time;
    mutable std::mutex data_mutex; // Single mutex for all data structures
    
    // Performance tracking - atomic only
    std::atomic<uint64_t> total_messages_processed{0};
    std::atomic<uint64_t> total_responses_sent{0};
    std::chrono::system_clock::time_point last_activity;
    
    // Constants - cached for performance
    static constexpr std::chrono::milliseconds MIN_RESPONSE_INTERVAL{2000};
    static constexpr size_t MAX_MESSAGE_LENGTH = 2000;
    static constexpr int32_t MAX_CONTEXT_FILL_PERCENTAGE = 80;
    static constexpr int32_t MIN_CONTEXT_FILL_PERCENTAGE = 10;

public:
    using BackfillStatus = DiscordHistoryLoader::BackfillStatus;
    
    BackfillStatus get_backfill_status() const {
        return history_loader ? history_loader->get_status() : BackfillStatus{};
    }

private:
    void setup_event_handlers() {
        if (!bot) return;
        
        bot->on_ready([this](const dpp::ready_t& event) {
            is_connected = true;
            log_message("Discord bot ready! Logged in as: " + bot->me.username);
        });
        
        bot->on_message_create([this](const dpp::message_create_t& event) {
            handle_message(event);
        });
        
        bot->on_log([this](const dpp::log_t& event) {
            if (event.severity >= dpp::ll_warning) {
                log_message("[D++] " + event.message);
            }
        });
        
        bot->on_guild_create([this](const dpp::guild_create_t& event) {
            if (!is_connected) {
                is_connected = true;
                log_message("Discord bot connected to guild: " + event.created.name);
            }
        });
    }
    
    void handle_message(const dpp::message_create_t& event) {
        if (event.msg.author.is_bot() || event.msg.content.empty()) return;
        
        const bool is_dm = (event.msg.guild_id == 0);
        
        // Early exit for disabled DMs
        if (is_dm && !allow_dms) {
            send_message(event.msg.channel_id, 
                "Sorry, Direct Messages are currently disabled. Please use the appropriate server channels.");
            return;
        }
        
        // Early exit for unconfigured channels
        if (!is_dm && !is_channel_configured(event.msg.channel_id)) return;
        
        // Rate limiting check
        if (is_rate_limited(event.msg.author.id)) return;
        
        // Process message
        std::string response = process_user_message(
            event.msg.content, event.msg.author.username, 
            event.msg.author.id, event.msg.channel_id, event.msg.guild_id
        );
        
        if (!response.empty() && !response.starts_with("Error:")) {
            send_message(event.msg.channel_id, response);
        }
    }
    
    std::string process_user_message(const std::string& message, const std::string& username, 
                                   uint64_t user_id, uint64_t channel_id, uint64_t guild_id) {
        if (!llama_manager) return "Error: AI backend not available";
        
        total_messages_processed++;
        last_activity = std::chrono::system_clock::now();
        
        std::string context_id = get_or_create_user_context(user_id, username, channel_id, guild_id);
        if (context_id.empty()) return "Error: Failed to access chat context";
        
        if (!llama_manager->switch_to_context(context_id)) {
            return "Error: Failed to access your chat context";
        }
        
        std::string response = llama_manager->generate_response(message, username);
        return response.empty() ? "I'm not sure how to respond to that. Could you try rephrasing?" : response;
    }
    
    std::vector<std::string> split_message(const std::string& message, size_t max_length = MAX_MESSAGE_LENGTH) const {
        std::vector<std::string> parts;
        if (message.length() <= max_length) {
            parts.push_back(message);
            return parts;
        }
        
        size_t start = 0;
        while (start < message.length()) {
            size_t end = std::min(start + max_length, message.length());
            if (end < message.length()) {
                size_t last_space = message.find_last_of(" \n\t", end);
                if (last_space != std::string::npos && last_space > start) {
                    end = last_space;
                }
            }
            parts.push_back(message.substr(start, end - start));
            start = end;
            while (start < message.length() && std::isspace(message[start])) start++;
        }
        return parts;
    }
    
    void log_message(const std::string& message) const {
        DISCORD_LOG(message);
    }
    
    void parse_channel_ids(const std::string& channel_ids_str, std::unordered_set<uint64_t>& target_set) {
        std::lock_guard<std::mutex> lock(channel_config_mutex);
        target_set.clear();
        
        std::stringstream ss(channel_ids_str);
        std::string id_str;
        while (std::getline(ss, id_str, ',')) {
            id_str.erase(0, id_str.find_first_not_of(" \t\n\r"));
            id_str.erase(id_str.find_last_not_of(" \t\n\r") + 1);
            
            if (!id_str.empty()) {
                try {
                    target_set.insert(std::stoull(id_str));
                } catch (const std::exception&) {
                    log_message("Warning: Invalid channel ID '" + id_str + "'");
                }
            }
        }
    }
    
    bool is_channel_configured(uint64_t channel_id) const {
        std::lock_guard<std::mutex> lock(channel_config_mutex);
        return isolated_channels.count(channel_id) > 0 || shared_history_channels.count(channel_id) > 0;
    }
    
    bool is_isolated_channel(uint64_t channel_id) const {
        std::lock_guard<std::mutex> lock(channel_config_mutex);
        return isolated_channels.count(channel_id) > 0;
    }
    
    bool is_rate_limited(uint64_t user_id) {
        std::lock_guard<std::mutex> lock(data_mutex);
        auto now = std::chrono::system_clock::now();
        auto it = last_response_time.find(user_id);
        
        if (it != last_response_time.end() && (now - it->second) < MIN_RESPONSE_INTERVAL) {
            return true;
        }
        
        last_response_time[user_id] = now;
        return false;
    }
    
    std::string get_or_create_user_context(uint64_t user_id, const std::string& username, 
                                         uint64_t channel_id, uint64_t guild_id) {
        const bool is_dm = (guild_id == 0);
        const bool is_isolated_chan = is_isolated_channel(channel_id);
        
        // Use shared main context for regular channels
        if (!is_isolated_chan && !is_dm) {
            return (llama_manager && llama_manager->has_context(main_context_id)) ? main_context_id : "";
        }
        
        std::lock_guard<std::mutex> lock(data_mutex);
        
        if (is_dm) {
            auto it = user_contexts.find(user_id);
            if (it != user_contexts.end() && llama_manager && llama_manager->has_context(it->second)) {
                return it->second;
            }
            
            std::string context_id = "discord_dm_" + std::to_string(user_id);
            
            // FIXED: Check if context already exists before trying to create
            if (llama_manager && llama_manager->has_context(context_id)) {
                user_contexts[user_id] = context_id;
                return context_id;
            }
            
            if (llama_manager && llama_manager->create_context(context_id, "")) {
                user_contexts[user_id] = context_id;
                return context_id;
            }
        } else if (is_isolated_chan) {
            auto it = channel_contexts.find(channel_id);
            if (it != channel_contexts.end() && llama_manager && llama_manager->has_context(it->second)) {
                return it->second;
            }
            
            std::string context_id = "discord_channel_" + std::to_string(channel_id);
            
            // FIXED: Check if context already exists before trying to create
            if (llama_manager && llama_manager->has_context(context_id)) {
                channel_contexts[channel_id] = context_id;
                return context_id;
            }
            
            if (llama_manager && llama_manager->create_context(context_id, "")) {
                channel_contexts[channel_id] = context_id;
                return context_id;
            }
        }
        
        return "";
    }
    
    void cleanup_contexts() {
        std::lock_guard<std::mutex> lock(data_mutex);
        
        if (llama_manager) {
            for (const auto& [user_id, context_id] : user_contexts) {
                if (context_id != main_context_id) {
                    llama_manager->remove_context(context_id);
                }
            }
            for (const auto& [channel_id, context_id] : channel_contexts) {
                if (context_id != main_context_id) {
                    llama_manager->remove_context(context_id);
                }
            }
        }
        
        user_contexts.clear();
        channel_contexts.clear();
        user_conversations.clear();
        last_response_time.clear();
    }

public:
    DiscordManager() : llama_manager(nullptr), history_loader(std::make_unique<DiscordHistoryLoader>()) {
        last_activity = std::chrono::system_clock::now();
    }
    
    ~DiscordManager() { shutdown(); }
    
    bool configure(const DiscordBotConfig& bot_config) {
        if (is_running || bot_config.bot_token.empty()) return false;
        config = bot_config;
        return true;
    }
    
    void set_isolated_channels(const std::string& channel_ids) {
        parse_channel_ids(channel_ids, isolated_channels);
    }
    
    void set_shared_history_channels(const std::string& channel_ids) {
        parse_channel_ids(channel_ids, shared_history_channels);
    }
    
    void set_allow_dms(bool allow) {
        std::lock_guard<std::mutex> lock(channel_config_mutex);
        allow_dms = allow;
    }
    
    void set_main_context_id(const std::string& context_id) {
        main_context_id = context_id;
    }
    
    void set_history_settings(bool pull_history, int32_t fill_percentage) {
        std::lock_guard<std::mutex> lock(channel_config_mutex);
        pull_message_history = pull_history;
        history_fill_percentage = std::clamp(fill_percentage, MIN_CONTEXT_FILL_PERCENTAGE, MAX_CONTEXT_FILL_PERCENTAGE);
        
        if (history_loader) {
            history_loader->set_history_settings(pull_message_history, history_fill_percentage);
        }
    }
    
    void set_llama_manager(LlamaManager* manager) {
        llama_manager = manager;
        if (manager) {
            if (history_loader && bot) {
                history_loader->configure(bot.get(), manager, main_context_id);
                history_loader->set_channel_configuration(nullptr, &isolated_channels, &shared_history_channels);
                history_loader->set_history_settings(pull_message_history, history_fill_percentage);
            }
        } else {
            cleanup_contexts();
        }
    }
    
    bool initialize() {
        if (config.bot_token.empty()) return false;
        
        try {
            uint32_t intents = dpp::i_default_intents | dpp::i_message_content;
            bot = std::make_unique<dpp::cluster>(config.bot_token, intents);
            setup_event_handlers();
            
            if (history_loader && llama_manager) {
                history_loader->configure(bot.get(), llama_manager, main_context_id);
                history_loader->set_channel_configuration(nullptr, &isolated_channels, &shared_history_channels);
                history_loader->set_history_settings(pull_message_history, history_fill_percentage);
            }
            
            return true;
        } catch (const std::exception& e) {
            log_message("Error: Failed to initialize Discord bot: " + std::string(e.what()));
            return false;
        }
    }
    
    bool start() {
        if (is_running || !initialize()) return false;
        
        try {
            bot->start(dpp::st_return);
            is_running = true;
            should_stop = false;
            
            // Start history backfill if enabled
            if (pull_message_history && llama_manager) {
                std::thread([this]() {
                    std::this_thread::sleep_for(std::chrono::seconds(5));
                    if (is_connected && history_loader) {
                        history_loader->start_backfill();
                    }
                }).detach();
            }
            
            return true;
        } catch (const std::exception& e) {
            log_message("Error: Failed to start Discord bot: " + std::string(e.what()));
            return false;
        }
    }
    
    void shutdown() {
        if (!is_running) return;
        
        should_stop = true;
        
        if (bot) {
            bot->shutdown();
            bot.reset();
        }
        
        is_running = false;
        is_connected = false;
        cleanup_contexts();
    }
    
    bool send_message(uint64_t channel_id, const std::string& message) {
        if (!is_running || !is_connected || !bot || message.empty()) return false;
        
        try {
            auto message_parts = split_message(message);
            for (const auto& part : message_parts) {
                bot->message_create(dpp::message(channel_id, part));
                if (message_parts.size() > 1) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                }
            }
            
            total_responses_sent++;
            last_activity = std::chrono::system_clock::now();
            return true;
        } catch (const std::exception& e) {
            log_message("Error sending message: " + std::string(e.what()));
            return false;
        }
    }
    
    // Status and statistics
    bool is_bot_running() const { return is_running; }
    bool is_bot_connected() const { return is_connected; }
    
    struct BotStatistics {
        uint64_t messages_processed;
        uint64_t responses_sent;
        uint64_t active_conversations;
        bool is_running;
        bool is_connected;
        std::chrono::system_clock::time_point last_activity;
    };
    
    BotStatistics get_statistics() const {
        std::lock_guard<std::mutex> lock(data_mutex);
        return {
            total_messages_processed.load(),
            total_responses_sent.load(),
            user_conversations.size(),
            is_running.load(),
            is_connected.load(),
            last_activity
        };
    }
};
//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODING DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//