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
// Character Set: Use Unicode Character Set
// Whole Program Optimization: Use Link Time Code Generation
// Optimization: Maximum Optimization (Favor Speed) (/O2)
// Enable Intrinsic Functions: Yes (/Oi)
// Favor Size or Speed: Favor fast code (/Ot)
// Whole Program Optimization: Yes (/GL)
// Enable String Pooling: Yes (/GF)
// Runtime Library: Multi-threaded DLL (/MD)
// Enable Run-Time Type Information (RTTI) YES (/GR)
// Link Time Code Generation: Use Link Time Code Generation (/LTCG)
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

// ADDED: Include D++ library
#include <dpp/dpp.h>

// Forward declare the LlamaManager for integration
class LlamaManager;

// Forward declare log callback function
void discord_manager_log_callback(const std::string& message);

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
    // UPDATED: Core Discord bot components with actual D++ implementation
    std::unique_ptr<dpp::cluster> bot;
    
    // Configuration and state
    DiscordBotConfig config;
    std::atomic<bool> is_running{false};
    std::atomic<bool> is_connected{false};
    std::atomic<bool> should_stop{false};
    
    // Integration with LlamaManager
    LlamaManager* llama_manager;
    
    // ADDED: Channel filtering
    std::unordered_set<uint64_t> allowed_channels;
    mutable std::mutex channel_mutex;
    
    // Message handling - made mutable for const methods
    mutable std::mutex message_mutex;
    std::unordered_map<uint64_t, std::vector<DiscordMessageContext>> user_conversations;
    
    // Performance tracking
    std::atomic<uint64_t> total_messages_processed{0};
    std::atomic<uint64_t> total_responses_sent{0};
    std::chrono::system_clock::time_point last_activity;
    
    // ADDED: Response rate limiting
    std::unordered_map<uint64_t, std::chrono::system_clock::time_point> last_response_time;
    mutable std::mutex rate_limit_mutex;
    const std::chrono::milliseconds min_response_interval{2000}; // 2 seconds between responses per user
    
    // Helper function for thread-safe logging
    void log_message(const std::string& message) const {
        discord_manager_log_callback(message);
    }
    
    // ADDED: Parse channel IDs from comma-separated string
    void parse_channel_ids(const std::string& channel_ids_str) {
        std::lock_guard<std::mutex> lock(channel_mutex);
        allowed_channels.clear();
        
        if (channel_ids_str.empty()) {
            return; // Empty means allow all channels
        }
        
        std::stringstream ss(channel_ids_str);
        std::string id_str;
        
        while (std::getline(ss, id_str, ',')) {
            // Trim whitespace
            id_str.erase(0, id_str.find_first_not_of(" \t\n\r"));
            id_str.erase(id_str.find_last_not_of(" \t\n\r") + 1);
            
            if (!id_str.empty()) {
                try {
                    uint64_t channel_id = std::stoull(id_str);
                    allowed_channels.insert(channel_id);
                    log_message("Added allowed channel: " + std::to_string(channel_id));
                } catch (const std::exception& e) {
                    log_message("Warning: Invalid channel ID '" + id_str + "': " + e.what());
                }
            }
        }
        
        log_message("Configured " + std::to_string(allowed_channels.size()) + " allowed channels");
    }
    
    // ADDED: Check if channel is allowed
    bool is_channel_allowed(uint64_t channel_id) const {
        std::lock_guard<std::mutex> lock(channel_mutex);
        return allowed_channels.empty() || allowed_channels.count(channel_id) > 0;
    }
    
    // ADDED: Rate limiting check
    bool is_rate_limited(uint64_t user_id) {
        std::lock_guard<std::mutex> lock(rate_limit_mutex);
        auto now = std::chrono::system_clock::now();
        auto it = last_response_time.find(user_id);
        
        if (it != last_response_time.end()) {
            auto time_since_last = now - it->second;
            if (time_since_last < min_response_interval) {
                return true; // Still rate limited
            }
        }
        
        last_response_time[user_id] = now;
        return false;
    }
    
    // Message context management
    void store_message_context(const DiscordMessageContext& context) {
        std::lock_guard<std::mutex> lock(message_mutex);
        auto& conversation = user_conversations[context.user_id];
        conversation.push_back(context);
        
        // Limit conversation history per user
        if (conversation.size() > config.message_cache_size) {
            conversation.erase(conversation.begin());
        }
    }
    
    // Get conversation context for a user
    std::vector<DiscordMessageContext> get_user_context(uint64_t user_id) const {
        std::lock_guard<std::mutex> lock(message_mutex);
        auto it = user_conversations.find(user_id);
        return (it != user_conversations.end()) ? it->second : std::vector<DiscordMessageContext>{};
    }

public:
    DiscordManager() : llama_manager(nullptr) {
        // Initialize with default configuration
        config.auto_reconnect = true;
        config.enable_message_cache = true;
        config.message_cache_size = 100;
        config.rate_limit_buffer_ms = 100;
        
        last_activity = std::chrono::system_clock::now();
    }
    
    ~DiscordManager() {
        shutdown();
    }
    
    // UPDATED: Configuration methods with channel ID parsing
    bool configure(const DiscordBotConfig& bot_config) {
        if (is_running) {
            log_message("Error: Cannot configure while bot is running");
            return false;
        }
        
        if (bot_config.bot_token.empty()) {
            log_message("Error: Bot token cannot be empty");
            return false;
        }
        
        config = bot_config;
        log_message("Discord bot configuration updated");
        return true;
    }
    
    // ADDED: Method to set allowed channels
    void set_allowed_channels(const std::string& channel_ids) {
        parse_channel_ids(channel_ids);
    }
    
    // Integration with LlamaManager
    void set_llama_manager(LlamaManager* manager) {
        llama_manager = manager;
        if (manager) {
            log_message("LlamaManager integration enabled");
        } else {
            log_message("LlamaManager integration disabled");
        }
    }
    
    // UPDATED: Bot lifecycle management with actual D++ implementation
    bool initialize() {
        if (config.bot_token.empty()) {
            log_message("Error: Bot token not configured");
            return false;
        }
        
        try {
            // Create D++ cluster with intents
            uint32_t intents = dpp::i_default_intents | dpp::i_message_content;
            bot = std::make_unique<dpp::cluster>(config.bot_token, intents);
            
            // Set up event handlers
            setup_event_handlers();
            
            log_message("Discord bot initialized with D++");
            return true;
        } catch (const std::exception& e) {
            log_message("Error: Failed to initialize Discord bot: " + std::string(e.what()));
            return false;
        }
    }
    
    bool start() {
        if (is_running) {
            log_message("Warning: Bot is already running");
            return true;
        }
        
        if (!initialize()) {
            return false;
        }
        
        try {
            // Start the bot
            bot->start(dpp::st_return);
            
            is_running = true;
            should_stop = false;
            
            log_message("Discord bot started successfully");
            return true;
        } catch (const std::exception& e) {
            log_message("Error: Failed to start Discord bot: " + std::string(e.what()));
            return false;
        }
    }
    
    void shutdown() {
        if (!is_running) {
            return;
        }
        
        should_stop = true;
        
        try {
            if (bot) {
                bot->shutdown();
                bot.reset();
            }
        } catch (const std::exception& e) {
            log_message("Warning during bot shutdown: " + std::string(e.what()));
        }
        
        is_running = false;
        is_connected = false;
        
        // Clear conversation cache
        {
            std::lock_guard<std::mutex> lock(message_mutex);
            user_conversations.clear();
        }
        
        // Clear rate limiting data
        {
            std::lock_guard<std::mutex> lock(rate_limit_mutex);
            last_response_time.clear();
        }
        
        log_message("Discord bot shutdown complete");
    }
    
    // Status and statistics
    bool is_bot_running() const {
        return is_running;
    }
    
    bool is_bot_connected() const {
        return is_connected;
    }
    
    struct BotStatistics {
        uint64_t messages_processed;
        uint64_t responses_sent;
        uint64_t active_conversations;
        bool is_running;
        bool is_connected;
        std::chrono::system_clock::time_point last_activity;
    };
    
    BotStatistics get_statistics() const {
        std::lock_guard<std::mutex> lock(message_mutex);
        return {
            total_messages_processed.load(),
            total_responses_sent.load(),
            user_conversations.size(),
            is_running.load(),
            is_connected.load(),
            last_activity
        };
    }
    
    // UPDATED: Message sending with actual D++ implementation
    bool send_message(uint64_t channel_id, const std::string& message) {
        if (!is_running || !is_connected || !bot) {
            log_message("Error: Bot not running or not connected");
            return false;
        }
        
        if (message.empty()) {
            log_message("Error: Cannot send empty message");
            return false;
        }
        
        try {
            // Split long messages if needed (Discord has 2000 char limit)
            std::vector<std::string> message_parts = split_message(message, 2000);
            
            for (const auto& part : message_parts) {
                dpp::message msg(channel_id, part);
                bot->message_create(msg);
                
                // Small delay between parts to avoid rate limiting
                if (message_parts.size() > 1) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                }
            }
            
            total_responses_sent++;
            last_activity = std::chrono::system_clock::now();
            
            log_message("Message sent to channel " + std::to_string(channel_id) + ": " + 
                       message.substr(0, 50) + (message.size() > 50 ? "..." : ""));
            return true;
        } catch (const std::exception& e) {
            log_message("Error sending message: " + std::string(e.what()));
            return false;
        }
    }
    
    // Conversation management
    void clear_user_conversation(uint64_t user_id) {
        std::lock_guard<std::mutex> lock(message_mutex);
        user_conversations.erase(user_id);
        log_message("Cleared conversation for user " + std::to_string(user_id));
    }
    
    void clear_all_conversations() {
        std::lock_guard<std::mutex> lock(message_mutex);
        size_t count = user_conversations.size();
        user_conversations.clear();
        log_message("Cleared " + std::to_string(count) + " conversations");
    }
    
    // Utility methods
    std::string get_bot_status() const {
        if (!is_running) {
            return "Stopped";
        } else if (!is_connected) {
            return "Starting...";
        } else {
            return "Connected";
        }
    }

private:
    // UPDATED: Event handlers for D++ integration
    void setup_event_handlers() {
        if (!bot) return;
        
        // Bot ready event
        bot->on_ready([this](const dpp::ready_t& event) {
            handle_ready(event);
        });
        
        // Message creation event
        bot->on_message_create([this](const dpp::message_create_t& event) {
            handle_message(event);
        });
        
        // Logging event
        bot->on_log([this](const dpp::log_t& event) {
            handle_log(event);
        });
        
        // Guild create event (for connection status)
        bot->on_guild_create([this](const dpp::guild_create_t& event) {
            if (!is_connected) {
                is_connected = true;
                log_message("Discord bot connected to guild: " + event.created.name);
            }
        });
        
        log_message("Discord event handlers configured");
    }
    
    // ADDED: Event handler implementations
    void handle_ready(const dpp::ready_t& event) {
        is_connected = true;
        log_message("Discord bot ready! Logged in as: " + bot->me.username + "#" + std::to_string(bot->me.discriminator));
        log_message("Bot is in " + std::to_string(event.guild_count) + " guilds");
    }
    
    void handle_message(const dpp::message_create_t& event) {
        try {
            // Ignore messages from bots (including ourselves)
            if (event.msg.author.is_bot()) {
                return;
            }
            
            // Check if channel is allowed
            if (!is_channel_allowed(event.msg.channel_id)) {
                return;
            }
            
            // Check rate limiting
            if (is_rate_limited(event.msg.author.id)) {
                log_message("Rate limited user: " + event.msg.author.username);
                return;
            }
            
            // Get message content
            std::string message_content = event.msg.content;
            if (message_content.empty()) {
                return;
            }
            
            // Create message context
            DiscordMessageContext context;
            context.user_id = event.msg.author.id;
            context.channel_id = event.msg.channel_id;
            context.guild_id = event.msg.guild_id;
            context.username = event.msg.author.username;
            context.timestamp = std::chrono::system_clock::now();
            context.is_dm = (event.msg.guild_id == 0);
            
            // Try to get channel name
            if (event.msg.guild_id != 0) {
                bot->channel_get(event.msg.channel_id, [this, context](const dpp::confirmation_callback_t& callback) mutable {
                    if (!callback.is_error()) {
                        auto channel = callback.get<dpp::channel>();
                        context.channel_name = channel.name;
                    }
                });
            } else {
                context.channel_name = "DM";
            }
            
            store_message_context(context);
            
            log_message("Processing message from " + event.msg.author.username + 
                       " in " + context.channel_name + ": " + 
                       message_content.substr(0, 100) + (message_content.size() > 100 ? "..." : ""));
            
            // Process message with LlamaManager - pass username
            std::string response = process_user_message(message_content, event.msg.author.username, event.msg.author.id);
            
            if (!response.empty() && !response.starts_with("Error:")) {
                // Send response
                send_message(event.msg.channel_id, response);
            } else if (response.starts_with("Error:")) {
                log_message("Error processing message: " + response);
                // Optionally send error message to user
                send_message(event.msg.channel_id, "I'm having trouble processing your message right now. Please try again later.");
            }
            
        } catch (const std::exception& e) {
            log_message("Exception in handle_message: " + std::string(e.what()));
        }
    }
    
    void handle_log(const dpp::log_t& event) {
        // Filter out verbose logs, only show warnings and errors
        if (event.severity >= dpp::ll_warning) {
            std::string severity_str;
            switch (event.severity) {
                case dpp::ll_trace: severity_str = "TRACE"; break;
                case dpp::ll_debug: severity_str = "DEBUG"; break;
                case dpp::ll_info: severity_str = "INFO"; break;
                case dpp::ll_warning: severity_str = "WARN"; break;
                case dpp::ll_error: severity_str = "ERROR"; break;
                case dpp::ll_critical: severity_str = "CRITICAL"; break;
                default: severity_str = "UNKNOWN"; break;
            }
            
            log_message("[D++:" + severity_str + "] " + event.message);
        }
    }
    
    // ADDED: Message processing with LlamaManager integration - updated to accept username
    std::string process_user_message(const std::string& message, const std::string& username, uint64_t user_id) {
        if (!llama_manager) {
            return "Error: AI backend not available";
        }
        
        total_messages_processed++;
        last_activity = std::chrono::system_clock::now();
        
        try {
            // Generate response using LlamaManager with username
            std::string response = llama_manager->generate_response(message, username);
            
            if (response.empty()) {
                return "I'm not sure how to respond to that. Could you try rephrasing your question?";
            }
            
            return response;
            
        } catch (const std::exception& e) {
            log_message("Exception in process_user_message: " + std::string(e.what()));
            return "Error: Exception occurred while processing message";
        }
    }
    
    // ADDED: Helper function to split long messages
    std::vector<std::string> split_message(const std::string& message, size_t max_length) const {
        std::vector<std::string> parts;
        
        if (message.length() <= max_length) {
            parts.push_back(message);
            return parts;
        }
        
        size_t start = 0;
        while (start < message.length()) {
            size_t end = std::min(start + max_length, message.length());
            
            // Try to split at word boundary
            if (end < message.length()) {
                size_t last_space = message.find_last_of(" \n\t", end);
                if (last_space != std::string::npos && last_space > start) {
                    end = last_space;
                }
            }
            
            parts.push_back(message.substr(start, end - start));
            start = end;
            
            // Skip whitespace at the beginning of next part
            while (start < message.length() && std::isspace(message[start])) {
                start++;
            }
        }
        
        return parts;
    }
};

// Log callback function declaration
extern void discord_manager_log_callback(const std::string& message);
