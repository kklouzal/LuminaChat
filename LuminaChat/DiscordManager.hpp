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
// 1.  Minimalism & Performance: Deliver lean, efficient solutions; do not create or preserve unused helpers, wrappers, trivial accessors (setters/getters), or scaffolding.
// 2.  Redundancy Elimination: Remove unused, obsolete, and legacy code—including unneeded interfaces, includes, helper or accessor methods.
// 3.  Consistent Style: Adopt a uniform coding style and structure for clarity and maintainability.
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
#include <thread>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <string_view>
#include <sstream>
#include <dpp/dpp.h>

#include "DiscordHistoryLoader.hpp"
#include "LogHandler.hpp"
#include "Sanitizer.hpp"

// Discord bot configuration structure
struct DiscordBotConfig {
    std::string bot_token;
    std::string application_id;
    uint64_t guild_id = 0;
    bool auto_reconnect = true;
    uint32_t rate_limit_buffer_ms = 100;
};

class DiscordManager {
private:
    // Core components
    std::unique_ptr<dpp::cluster> bot;
    std::unique_ptr<DiscordHistoryLoader> history_loader;
    LlamaManager* llama_manager;

      // Configuration
    DiscordBotConfig config;

    // State
    std::atomic<bool> is_connected{false};
    std::atomic<bool> should_stop{false};
    
    // Channel configuration
    std::unordered_set<uint64_t> isolated_channels;
    std::unordered_set<uint64_t> shared_history_channels;
    bool allow_dms = true;
    bool pull_message_history = true;
    int32_t history_fill_percentage = 50;
    mutable std::mutex channel_config_mutex;
    
    // Context management
    std::unordered_map<uint64_t, std::string> user_contexts;
    std::unordered_map<uint64_t, std::string> channel_contexts;
    std::unordered_map<uint64_t, std::chrono::system_clock::time_point> last_response_time;
    mutable std::mutex data_mutex;
    
    // Performance tracking
    std::atomic<uint64_t> total_messages_processed{0};
    std::atomic<uint64_t> total_responses_sent{0};
    std::chrono::system_clock::time_point last_activity;
      // Constants
    static constexpr std::chrono::milliseconds MIN_RESPONSE_INTERVAL{2000};
    static constexpr size_t MAX_MESSAGE_LENGTH = 2000;
    static constexpr int32_t MAX_CONTEXT_FILL_PERCENTAGE = 80;
    static constexpr int32_t MIN_CONTEXT_FILL_PERCENTAGE = 10;
    static constexpr int32_t HEARTBEAT_DELAY_SEC = 5;
    static constexpr int32_t RETRY_DELAY_MS = 500;

public:
    // State
    std::atomic<bool> is_running{ false };

    // Configuration
    std::string main_context_id;

    std::string model_id = "main_model";

    using BackfillStatus = DiscordHistoryLoader::BackfillStatus;
    
    BackfillStatus get_backfill_status() const {
        return history_loader ? history_loader->get_status() : BackfillStatus{};
    }

private:
    void setup_event_handlers() {
        if (!bot) return;
        
        bot->on_ready([this](const dpp::ready_t& event) {
            is_connected = true;
            DISCORD_LOG("Discord bot ready! Logged in as: " + bot->me.username);
            
            // Pass bot identity to history loader for recognizing own messages
            if (history_loader) {
                history_loader->set_bot_identity(static_cast<uint64_t>(bot->me.id), bot->me.username);
            }
        });
        
        bot->on_message_create([this](const dpp::message_create_t& event) {
            handle_message(event);
        });
        
        bot->on_log([this](const dpp::log_t& event) {
            if (event.severity >= dpp::ll_warning) {
                DISCORD_LOG("[D++] " + event.message);
            }
        });
        
        bot->on_guild_create([this](const dpp::guild_create_t& event) {
            if (!is_connected) {
                is_connected = true;
                DISCORD_LOG("Discord bot connected to guild: " + event.created.name);
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
        
        // Sanitize the incoming message to prevent tokenization issues
        std::string sanitized_message = TextSanitizer::sanitize_text(message);
        if (sanitized_message.empty()) {
            return "I'm sorry, but your message couldn't be processed. Please try rephrasing your message.";
        }
        
        total_messages_processed++;
        last_activity = std::chrono::system_clock::now();
        
        std::string context_id = get_or_create_user_context(user_id, username, channel_id, guild_id);
        if (context_id.empty()) return "Error: Failed to access chat context";
        
        // Use direct context access instead of switching
        ContextInfo* target_context = llama_manager->get_context_info(context_id);
        if (!target_context) {
            return "Error: Failed to access your chat context";
        }
        
        std::string response = llama_manager->generate_response(sanitized_message, target_context, username);
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
                    DISCORD_LOG("Warning: Invalid channel ID '" + id_str + "'");
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
      // Get system prompt from main context for new Discord contexts
    std::string get_system_prompt_for_new_context() const {
        if (!llama_manager || main_context_id.empty()) return "";
        
        // Get the main context info to retrieve system prompt
        auto context_info = llama_manager->get_context_info(main_context_id);
        // get the system message from the main context
        std::string system_prompt = context_info->system_message;
        
        return system_prompt;
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
            
            // Check if context already exists before trying to create
            if (llama_manager && llama_manager->has_context(context_id)) {                user_contexts[user_id] = context_id;
                return context_id;
            }
              // Use new API with model_id parameter and proper system prompt
            std::string system_prompt = get_system_prompt_for_new_context();
            if (llama_manager && !model_id.empty() && llama_manager->create_context(context_id, model_id, system_prompt)) {
                user_contexts[user_id] = context_id;
                return context_id;
            }
        } else if (is_isolated_chan) {
            auto it = channel_contexts.find(channel_id);
            if (it != channel_contexts.end() && llama_manager && llama_manager->has_context(it->second)) {
                return it->second;
            }
            
            std::string context_id = "discord_channel_" + std::to_string(channel_id);
            
            // Check if context already exists before trying to create
            if (llama_manager && llama_manager->has_context(context_id)) {                channel_contexts[channel_id] = context_id;
                return context_id;
            }
              // Use new API with model_id parameter and proper system prompt
            std::string system_prompt = get_system_prompt_for_new_context();
            if (llama_manager && !model_id.empty() && llama_manager->create_context(context_id, model_id, system_prompt)) {
                channel_contexts[channel_id] = context_id;
                return context_id;
            }
        }
          return "";
    }
    
    // Helper function to get context ID for a specific channel/user
    std::string get_context_for_channel(uint64_t channel_id, uint64_t user_id = 0, uint64_t guild_id = 0) const {
        const bool is_dm = (guild_id == 0);
        const bool is_isolated_chan = is_isolated_channel(channel_id);
        
        // Use shared main context for regular channels
        if (!is_isolated_chan && !is_dm) {
            return main_context_id;
        }
        
        std::lock_guard<std::mutex> lock(data_mutex);
        
        if (is_dm && user_id != 0) {
            auto it = user_contexts.find(user_id);
            if (it != user_contexts.end()) {
                return it->second;
            }
        } else if (is_isolated_chan) {
            auto it = channel_contexts.find(channel_id);
            if (it != channel_contexts.end()) {
                return it->second;
            }
        }
        
        return main_context_id; // Fallback to main context
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

    void set_history_settings(bool pull_history, int32_t fill_percentage) {
        std::lock_guard<std::mutex> lock(channel_config_mutex);
        pull_message_history = pull_history;
        history_fill_percentage = std::clamp(fill_percentage, MIN_CONTEXT_FILL_PERCENTAGE, MAX_CONTEXT_FILL_PERCENTAGE);
        
        if (history_loader) {
            history_loader->set_history_settings(pull_message_history, history_fill_percentage);
        }
    }
    
    void set_llama_manager(LlamaManager* manager) {
        llama_manager = manager;        if (manager) {
            // Ensure model_id is set before configuring history loader
            if (model_id.empty()) {
                model_id = "main_model";
            }
            
            if (history_loader && bot) {
                history_loader->configure(bot.get(), manager, main_context_id, model_id);
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
                // Ensure model_id is set before configuring history loader
                if (model_id.empty()) {
                    model_id = "main_model";
                }
                
                history_loader->configure(bot.get(), llama_manager, main_context_id, model_id);
                history_loader->set_channel_configuration(nullptr, &isolated_channels, &shared_history_channels);
                history_loader->set_history_settings(pull_message_history, history_fill_percentage);
            }
            
            return true;
        } catch (const std::exception& e) {
            DISCORD_LOG("Error: Failed to initialize Discord bot: " + std::string(e.what()));
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
                    std::this_thread::sleep_for(std::chrono::seconds(HEARTBEAT_DELAY_SEC));
                    if (is_connected && history_loader) {
                        history_loader->start_backfill();
                    }
                }).detach();
            }
            
            return true;
        } catch (const std::exception& e) {
            DISCORD_LOG("Error: Failed to start Discord bot: " + std::string(e.what()));
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
                // Get context information for this channel
                std::string context_id = get_context_for_channel(channel_id, 0, 0);
                std::string footer_text = "🤖 LuminaChat AI";
                
                if (llama_manager && !context_id.empty()) {
                    // Get ContextInfo for the context ID
                    auto context_info = llama_manager->get_context_info(context_id);
                    
                    int32_t context_usage = context_info->n_past;
                    int32_t context_size = context_info->get_context_size();
                    
                    if (context_size > 0) {
                        footer_text += " • Context: " + std::to_string(context_usage) + "/" + std::to_string(context_size);
                    }
                }
                
                // Create a nice looking embed for bot responses
                dpp::embed embed = dpp::embed()
                    .set_color(0x00ff9f)  // Nice green color
                    .set_description(part)
                    .set_footer(dpp::embed_footer().set_text(footer_text))
                    .set_timestamp(time(nullptr));
                
                dpp::message msg(channel_id, "");
                msg.add_embed(embed);
                bot->message_create(msg);
                
                if (message_parts.size() > 1) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAY_MS));
                }
            }
            
            total_responses_sent++;
            last_activity = std::chrono::system_clock::now();
            return true;
        } catch (const std::exception& e) {
            DISCORD_LOG("Error sending message: " + std::string(e.what()));
            return false;
        }
    }
      // Status and statistics  
    // Direct access to is_running (Directive #7: favor direct access over thin accessors)
    // Removed unused is_bot_connected() method (Directive #2: Redundancy Elimination)
    
    struct BotStatistics {
        uint64_t messages_processed;
        uint64_t responses_sent;
        bool is_running;
        bool is_connected;
        std::chrono::system_clock::time_point last_activity;
    };
    
    BotStatistics get_statistics() const {
        std::lock_guard<std::mutex> lock(data_mutex);
        return {
            total_messages_processed.load(),
            total_responses_sent.load(),
            is_running.load(),
            is_connected.load(),
            last_activity
        };
    }
};
//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODING DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//