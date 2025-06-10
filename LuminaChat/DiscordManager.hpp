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

// ADDED: Include D++ library
#include <dpp/dpp.h>

// ADDED: Include history loader
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
    // UPDATED: Core Discord bot components with actual D++ implementation
    std::unique_ptr<dpp::cluster> bot;
    
    // ADDED: Mutex to protect all D++ API calls (shared with history loader)
    mutable std::mutex discord_api_mutex;
    
    // Configuration and state
    DiscordBotConfig config;
    std::atomic<bool> is_running{false};
    std::atomic<bool> is_connected{false};
    std::atomic<bool> should_stop{false};
    
    // Integration with LlamaManager
    LlamaManager* llama_manager;
    std::string main_context_id; // ADDED: ID of the main shared context
    
    // ADDED: History loader integration
    std::unique_ptr<DiscordHistoryLoader> history_loader;
    
    // UPDATED: Channel filtering - removed allowed_channels
    std::unordered_set<uint64_t> isolated_channels;
    std::unordered_set<uint64_t> shared_history_channels;
    bool allow_dms = true;
    bool pull_message_history = true;
    int32_t history_fill_percentage = 50;
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
    
    // ADDED: Context management for Discord users and channels
    std::unordered_map<uint64_t, std::string> user_contexts; // user_id -> context_id (for DMs only)
    std::unordered_map<uint64_t, std::string> channel_contexts; // channel_id -> context_id (for isolated channels)
    mutable std::mutex context_mutex;

public:
    // UPDATED: Enhanced backfill status with per-context information
    using BackfillStatus = DiscordHistoryLoader::BackfillStatus;
    
    BackfillStatus get_backfill_status() const {
        if (history_loader) {
            return history_loader->get_status();
        }
        return BackfillStatus{};
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
            
            // Determine if this is a DM using guild_id
            bool is_dm = (event.msg.guild_id == 0);
            
            // Check if DMs are disabled and handle auto-reply
            if (is_dm && !allow_dms) {
                // Send auto-reply for disabled DMs
                std::string auto_reply = "Sorry, Direct Messages are currently disabled. Please use the appropriate server channels to chat with me.";
                send_message(event.msg.channel_id, auto_reply);
                log_message("Auto-replied to DM from " + event.msg.author.username + " (DMs disabled)");
                return;
            }
            
            // FIXED: For guild channels, only respond if channel is in our configured lists
            if (!is_dm && !is_channel_configured(event.msg.channel_id)) {
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
            context.is_dm = is_dm;
            
            // Set channel name based on context
            if (is_dm) {
                context.channel_name = "DM";
            } else {
                // Try to get channel name for guild channels
                context.channel_name = "Unknown"; // Fallback
                bot->channel_get(event.msg.channel_id, [this, context](const dpp::confirmation_callback_t& callback) mutable {
                    if (!callback.is_error()) {
                        auto channel = callback.get<dpp::channel>();
                        context.channel_name = channel.name;
                    }
                });
            }
            
            store_message_context(context);
            
            // Improved context type logging
            std::string context_type;
            if (is_dm) {
                context_type = "DM (isolated)";
            } else if (is_isolated_channel(event.msg.channel_id)) {
                context_type = "isolated";
            } else {
                context_type = "shared";
            }
            
            log_message("Processing message from " + event.msg.author.username + 
                       " in " + context.channel_name + " (" + context_type + " context): " + 
                       message_content.substr(0, 100) + (message_content.size() > 100 ? "..." : ""));
            
            // Process message with guild_id for proper DM detection
            std::string response = process_user_message(message_content, event.msg.author.username, 
                                                      event.msg.author.id, event.msg.channel_id, event.msg.guild_id);
            
            if (!response.empty() && !response.starts_with("Error:")) {
                // Send response
                send_message(event.msg.channel_id, response);
            } else if (response.starts_with("Error:")) {
                log_message("Error processing message: " + response);
                // Send error message to user
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
    
    // FIXED: Message processing with correct function signature
    std::string process_user_message(const std::string& message, const std::string& username, uint64_t user_id, uint64_t channel_id, uint64_t guild_id) {
        if (!llama_manager) {
            return "Error: AI backend not available";
        }
        
        total_messages_processed++;
        last_activity = std::chrono::system_clock::now();
        
        try {
            // Get or create context for this user and channel
            std::string context_id = get_or_create_user_context(user_id, username, channel_id, guild_id);
            if (context_id.empty()) {
                return "Error: Failed to access chat context";
            }
            
            // Switch to appropriate context (shared, isolated channel, or DM)
            if (!llama_manager->switch_to_context(context_id)) {
                log_message("Failed to switch to context '" + context_id + "' for user: " + username);
                return "Error: Failed to access your chat context";
            }
            
            // Improved context usage logging
            bool is_dm = (guild_id == 0);
            bool is_isolated_chan = is_isolated_channel(channel_id);
            
            if (context_id == main_context_id) {
                log_message("Using shared main context for user: " + username + " in channel: " + std::to_string(channel_id));
            } else if (is_dm) {
                log_message("Using DM context '" + context_id + "' for user: " + username);
            } else if (is_isolated_chan) {
                log_message("Using isolated channel context '" + context_id + "' for channel: " + std::to_string(channel_id) + " (user: " + username + ")");
            } else {
                log_message("Using context '" + context_id + "' for user: " + username);
            }
            
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
    
    // Helper function for thread-safe logging
    void log_message(const std::string& message) const {
        DISCORD_LOG(message);
    }
    
    // UPDATED: Parse isolated channel IDs
    void parse_isolated_channel_ids(const std::string& channel_ids_str) {
        std::lock_guard<std::mutex> lock(channel_mutex);
        isolated_channels.clear();
        
        if (channel_ids_str.empty()) {
            return;
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
                    isolated_channels.insert(channel_id);
                    log_message("Added isolated context channel: " + std::to_string(channel_id));
                } catch (const std::exception& e) {
                    log_message("Warning: Invalid isolated channel ID '" + id_str + "': " + e.what());
                }
            }
        }
        
        log_message("Configured " + std::to_string(isolated_channels.size()) + " isolated context channels");
    }
    
    // ADDED: Parse channel IDs from comma-separated string (for shared history channels)
    void parse_shared_history_channel_ids(const std::string& channel_ids_str) {
        std::lock_guard<std::mutex> lock(channel_mutex);
        shared_history_channels.clear();
        
        if (channel_ids_str.empty()) {
            return; // Empty means no shared history channels
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
                    shared_history_channels.insert(channel_id);
                    log_message("Added shared history channel: " + std::to_string(channel_id));
                } catch (const std::exception& e) {
                    log_message("Warning: Invalid shared history channel ID '" + id_str + "': " + e.what());
                }
            }
        }
        
        log_message("Configured " + std::to_string(shared_history_channels.size()) + " shared channels for history backfill");
    }
    
    // ADDED: Check if channel should have history backfilled (for shared channels)
    bool should_backfill_shared_channel(uint64_t channel_id) const {
        std::lock_guard<std::mutex> lock(channel_mutex);
        
        // If channel is isolated, always backfill (handled by existing logic)
        if (isolated_channels.count(channel_id) > 0) {
            return true;
        }
        
        // For shared channels, only backfill if explicitly listed
        return shared_history_channels.count(channel_id) > 0;
    }

    // FIXED: Replace is_channel_allowed with proper channel filtering
    bool is_channel_configured(uint64_t channel_id) const {
        std::lock_guard<std::mutex> lock(channel_mutex);
        // Channel is configured if it's in either isolated channels or shared history channels
        return isolated_channels.count(channel_id) > 0 || 
               shared_history_channels.count(channel_id) > 0;
    }
    
    // ADDED: Check if channel should use isolated context
    bool is_isolated_channel(uint64_t channel_id) const {
        std::lock_guard<std::mutex> lock(channel_mutex);
        return isolated_channels.count(channel_id) > 0;
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
    
    // SIMPLIFIED: Helper to get or create context with consistent system prompt handling
    std::string get_or_create_user_context(uint64_t user_id, const std::string& username, uint64_t channel_id, uint64_t guild_id) {
        bool is_dm = (guild_id == 0);
        bool is_isolated_chan = is_isolated_channel(channel_id);
        
        // For regular channels, use shared main context
        if (!is_isolated_chan && !is_dm) {
            if (!main_context_id.empty() && llama_manager && llama_manager->has_context(main_context_id)) {
                return main_context_id;
            } else {
                log_message("Warning: Main context '" + main_context_id + "' not available for shared channel");
                return "";
            }
        }
        
        std::lock_guard<std::mutex> lock(context_mutex);
        
        // Handle DMs - per-user isolated contexts
        if (is_dm) {
            auto it = user_contexts.find(user_id);
            
            if (it != user_contexts.end()) {
                if (llama_manager && llama_manager->has_context(it->second)) {
                    return it->second;
                } else {
                    user_contexts.erase(it);
                }
            }
            
            // Create new DM context - empty system prompt will inherit from main
            if (llama_manager) {
                std::string context_id = "discord_dm_" + std::to_string(user_id);
                log_message("Creating DM context '" + context_id + "' for user: " + username);
                
                if (llama_manager->create_context(context_id, "")) {
                    user_contexts[user_id] = context_id;
                    return context_id;
                } else {
                    log_message("Failed to create DM context for user: " + username);
                    return "";
                }
            }
        }
        // Handle isolated channels - per-channel contexts shared by all users
        else if (is_isolated_chan) {
            auto it = channel_contexts.find(channel_id);
            
            if (it != channel_contexts.end()) {
                if (llama_manager && llama_manager->has_context(it->second)) {
                    return it->second;
                } else {
                    channel_contexts.erase(it);
                }
            }
            
            // Create new isolated channel context - empty system prompt will inherit from main
            if (llama_manager) {
                std::string context_id = "discord_channel_" + std::to_string(channel_id);
                log_message("Creating isolated channel context '" + context_id + "' for channel: " + std::to_string(channel_id));
                
                if (llama_manager->create_context(context_id, "")) {
                    channel_contexts[channel_id] = context_id;
                    return context_id;
                } else {
                    log_message("Failed to create isolated channel context for channel: " + std::to_string(channel_id));
                    return "";
                }
            }
        }
        
        return "";
    }
    
    // UPDATED: Clean up user context (DMs only)
    void cleanup_user_context(uint64_t user_id) {
        std::lock_guard<std::mutex> lock(context_mutex);
        
        auto it = user_contexts.find(user_id);
        if (it != user_contexts.end()) {
            if (llama_manager) {
                // Only remove if it's not the main shared context
                if (it->second != main_context_id) {
                    llama_manager->remove_context(it->second);
                    log_message("Removed DM context for Discord user: " + std::to_string(user_id));
                }
            }
            user_contexts.erase(it);
        }
    }
    
    // ADDED: Clean up channel context
    void cleanup_channel_context(uint64_t channel_id) {
        std::lock_guard<std::mutex> lock(context_mutex);
        
        auto it = channel_contexts.find(channel_id);
        if (it != channel_contexts.end()) {
            if (llama_manager) {
                // Only remove if it's not the main shared context
                if (it->second != main_context_id) {
                    llama_manager->remove_context(it->second);
                    log_message("Removed isolated channel context for channel: " + std::to_string(channel_id));
                }
            }
            channel_contexts.erase(it);
        }
    }

public:
    DiscordManager() : llama_manager(nullptr) {
        // Initialize with default configuration
        config.auto_reconnect = true;
        config.enable_message_cache = true;
        config.message_cache_size = 100;
        config.rate_limit_buffer_ms = 100;
        
        last_activity = std::chrono::system_clock::now();
        
        // Initialize history loader
        history_loader = std::make_unique<DiscordHistoryLoader>();
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
    
    // ADDED: Method to set isolated channels
    void set_isolated_channels(const std::string& channel_ids) {
        parse_isolated_channel_ids(channel_ids);
    }
    
    // ADDED: Method to set shared history channels
    void set_shared_history_channels(const std::string& channel_ids) {
        parse_shared_history_channel_ids(channel_ids);
    }
    
    // ADDED: Method to set DM allowance
    void set_allow_dms(bool allow) {
        std::lock_guard<std::mutex> lock(channel_mutex);
        allow_dms = allow;
        log_message("Direct Messages: " + std::string(allow ? "Enabled" : "Disabled"));
    }
    
    // ADDED: Method to set main context ID for shared channels
    void set_main_context_id(const std::string& context_id) {
        main_context_id = context_id;
        log_message("Set main shared context ID: " + context_id);
    }
    
    // ADDED: Method to set history settings
    void set_history_settings(bool pull_history, int32_t fill_percentage) {
        std::lock_guard<std::mutex> lock(channel_mutex);
        pull_message_history = pull_history;
        history_fill_percentage = std::clamp(fill_percentage, 10, 80);
        log_message("History settings: Pull=" + std::string(pull_history ? "Enabled" : "Disabled") + 
                   ", Fill=" + std::to_string(history_fill_percentage) + "%");
        
        // Update history loader if available
        if (history_loader) {
            history_loader->set_history_settings(pull_message_history, history_fill_percentage);
        }
    }
    
    // Integration with LlamaManager
    void set_llama_manager(LlamaManager* manager) {
        llama_manager = manager;
        if (manager) {
            log_message("LlamaManager integration enabled");
            
            // UPDATED: Configure history loader with Discord API mutex
            if (history_loader && bot) {
                history_loader->configure(bot.get(), manager, main_context_id);
                history_loader->set_channel_configuration(nullptr, &isolated_channels, &shared_history_channels);
                history_loader->set_history_settings(pull_message_history, history_fill_percentage);
                
                // ADDED: Share Discord API mutex with history loader
                // Note: We'll need to modify DiscordHistoryLoader to accept the mutex reference
            }
        } else {
            log_message("LlamaManager integration disabled");
            // Clean up all user contexts when disconnecting
            cleanup_all_user_contexts();
        }
    }
    
    // ADDED: Clean up all user and channel contexts
    void cleanup_all_user_contexts() {
        std::lock_guard<std::mutex> lock(context_mutex);
        
        if (llama_manager) {
            // Clean up DM contexts
            for (const auto& [user_id, context_id] : user_contexts) {
                if (context_id != main_context_id) {
                    llama_manager->remove_context(context_id);
                }
            }
            
            // Clean up isolated channel contexts
            for (const auto& [channel_id, context_id] : channel_contexts) {
                if (context_id != main_context_id) {
                    llama_manager->remove_context(context_id);
                }
            }
        }
        
        size_t dm_count = user_contexts.size();
        size_t channel_count = channel_contexts.size();
        user_contexts.clear();
        channel_contexts.clear();
        
        if (dm_count > 0 || channel_count > 0) {
            log_message("Cleaned up " + std::to_string(dm_count) + " DM contexts and " + 
                       std::to_string(channel_count) + " isolated channel contexts (preserved main context)");
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
            
            // Configure history loader with bot instance
            if (history_loader && llama_manager) {
                history_loader->configure(bot.get(), llama_manager, main_context_id);
                history_loader->set_channel_configuration(nullptr, &isolated_channels, &shared_history_channels);
                history_loader->set_history_settings(pull_message_history, history_fill_percentage);
            }
            
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
            // PROTECTED: Start the bot with mutex protection
            {
                std::lock_guard<std::mutex> lock(discord_api_mutex);
                bot->start(dpp::st_return);
            }
            
            is_running = true;
            should_stop = false;
            
            log_message("Discord bot started successfully");
            
            // Start chat history backfill after a short delay to ensure connection (only if enabled)
            if (pull_message_history && llama_manager) {
                std::thread([this]() {
                    std::this_thread::sleep_for(std::chrono::seconds(5)); // Increased delay for worker system
                    if (is_connected && llama_manager && history_loader) {
                        log_message("Starting chat history backfill with worker system...");
                        history_loader->start_backfill();
                    }
                }).detach();
            } else {
                if (!pull_message_history) {
                    log_message("Chat history backfill disabled in settings");
                } else {
                    log_message("Chat history backfill skipped - LlamaManager not available");
                }
            }
            
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
            // PROTECTED: Shutdown the bot with mutex protection
            if (bot) {
                std::lock_guard<std::mutex> lock(discord_api_mutex);
                bot->shutdown();
                bot.reset();
            }
        } catch (const std::exception& e) {
            log_message("Warning during bot shutdown: " + std::string(e.what()));
        }
        
        is_running = false;
        is_connected = false;
        
        // Clean up all user contexts
        cleanup_all_user_contexts();
        
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
    
    // UPDATED: Message sending with mutex protection
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
            
            // PROTECTED: All D++ API calls must be protected by mutex
            std::lock_guard<std::mutex> lock(discord_api_mutex);
            
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
        
        // Only clean up DM context for this user, not channel contexts
        cleanup_user_context(user_id);
        
        log_message("Cleared conversation and DM context for user " + std::to_string(user_id));
    }
    
    // ADDED: Clear conversation for a specific channel
    void clear_channel_conversation(uint64_t channel_id) {
        // Clear message history for all users who were active in this channel
        std::lock_guard<std::mutex> lock(message_mutex);
        
        // Remove conversations that were in this channel
        auto it = user_conversations.begin();
        while (it != user_conversations.end()) {
            auto& conversations = it->second;
            conversations.erase(
                std::remove_if(conversations.begin(), conversations.end(),
                    [channel_id](const DiscordMessageContext& ctx) {
                        return ctx.channel_id == channel_id;
                    }),
                conversations.end()
            );
            
            if (conversations.empty()) {
                it = user_conversations.erase(it);
            } else {
                ++it;
            }
        }
        
        // Clean up the isolated channel context
        cleanup_channel_context(channel_id);
        
        log_message("Cleared conversation and isolated context for channel " + std::to_string(channel_id));
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
};

//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODING DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//