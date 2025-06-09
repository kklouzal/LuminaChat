// DiscordManager.hpp - header-only implementation for Discord integration via D++
// Handles Discord bot functionality and integration with LuminaChat backend.
//
// File Specific Directives:
// Manage Discord bot lifecycle, message handling, and integration with LlamaManager.
// Handle Discord rate limiting and async operations properly.
// Incoming discord messages are processed and disptched to the LlamaManager for processing in the appropriate context.
// Bot messages are filtered out and ignored.
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

// OPTIMIZED: Minimal essential includes only
#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <string_view>

#include <dpp/dpp.h>
#include "DiscordHistoryManager.hpp"

// Forward declarations
class LlamaManager;
void discord_manager_log_callback(const std::string& message);

// STREAMLINED: Core configuration structure
struct DiscordBotConfig {
    std::string bot_token;
    uint64_t guild_id = 0;
    bool auto_reconnect = true;
    uint32_t message_cache_size = 100;
    uint32_t rate_limit_buffer_ms = 100;
};

// SIMPLIFIED: Essential message context
struct DiscordMessageContext {
    uint64_t user_id = 0;
    uint64_t channel_id = 0;
    uint64_t guild_id = 0;
    std::string username;
    std::chrono::system_clock::time_point timestamp;
    bool is_dm = false;
};

class DiscordManager {
private:
    // CORE: Essential bot components
    std::unique_ptr<dpp::cluster> bot;
    DiscordBotConfig config;
    std::atomic<bool> is_running{false};
    std::atomic<bool> is_connected{false};
    std::atomic<bool> should_stop{false};
    
    // INTEGRATION: LlamaManager connection
    LlamaManager* llama_manager;
    std::string main_context_id;
    
    // OPTIMIZED: Channel management with single mutex
    mutable std::mutex channel_mutex;
    std::unordered_set<uint64_t> allowed_channels;
    std::unordered_set<uint64_t> isolated_channels;
    std::unordered_set<uint64_t> shared_history_channels;
    bool allow_dms = true;
    
    // EFFICIENT: Context management
    mutable std::mutex context_mutex;
    std::unordered_map<uint64_t, std::string> user_contexts; // DM contexts
    std::unordered_map<uint64_t, std::string> channel_contexts; // Isolated channel contexts
    
    // PERFORMANCE: Rate limiting and caching
    mutable std::mutex rate_limit_mutex;
    std::unordered_map<uint64_t, std::chrono::system_clock::time_point> last_response_time;
    static constexpr std::chrono::milliseconds min_response_interval{2000};
    
    // STATISTICS: Performance tracking
    std::atomic<uint64_t> total_messages_processed{0};
    std::atomic<uint64_t> total_responses_sent{0};
    std::chrono::system_clock::time_point last_activity;
    
    // DELEGATION: History management
    std::unique_ptr<DiscordHistoryManager> history_manager;
    
    // CACHED: Pre-allocated working buffers
    mutable std::vector<std::string> message_parts_buffer;

public:
    using BackfillStatus = DiscordHistoryManager::BackfillStatus;
    
    BackfillStatus get_backfill_status() const {
        return history_manager ? history_manager->get_status() : BackfillStatus{};
    }

private:
    // OPTIMIZED: Event handler setup
    void setup_event_handlers() {
        if (!bot) return;
        
        bot->on_ready([this](const dpp::ready_t& event) {
            is_connected = true;
            log_message("Discord bot ready: " + bot->me.username);
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
                log_message("Connected to guild: " + event.created.name);
            }
        });
    }
    
    // STREAMLINED: Core message handling
    void handle_message(const dpp::message_create_t& event) {
        if (event.msg.author.is_bot()) return;
        
        const bool is_dm = (event.msg.guild_id == 0);
        
        // Handle DM restrictions
        if (is_dm && !allow_dms) {
            send_message(event.msg.channel_id, 
                "Sorry, Direct Messages are currently disabled. Please use the appropriate server channels.");
            return;
        }
        
        // Check channel permissions for guild messages
        if (!is_dm && !is_channel_allowed(event.msg.channel_id)) return;
        
        // Rate limiting
        if (is_rate_limited(event.msg.author.id)) return;
        
        const std::string& content = event.msg.content;
        if (content.empty()) return;
        
        // Process message
        std::string response = process_user_message(content, event.msg.author.username, 
                                                  event.msg.author.id, event.msg.channel_id, event.msg.guild_id);
        
        if (!response.empty() && !response.starts_with("Error:")) {
            send_message(event.msg.channel_id, response);
        } else if (response.starts_with("Error:")) {
            log_message("Error processing message: " + response);
            send_message(event.msg.channel_id, "I'm having trouble processing your message. Please try again later.");
        }
    }
    
    // OPTIMIZED: Message processing with efficient context management
    std::string process_user_message(const std::string& message, const std::string& username, 
                                   uint64_t user_id, uint64_t channel_id, uint64_t guild_id) {
        if (!llama_manager) return "Error: AI backend not available";
        
        ++total_messages_processed;
        last_activity = std::chrono::system_clock::now();
        
        // Get context efficiently
        const std::string context_id = get_or_create_context(user_id, username, channel_id, guild_id);
        if (context_id.empty()) return "Error: Failed to access chat context";
        
        // Switch context and generate response
        if (!llama_manager->switch_to_context(context_id)) {
            return "Error: Failed to access your chat context";
        }
        
        const std::string response = llama_manager->generate_response(message, username);
        return response.empty() ? "I'm not sure how to respond to that. Could you try rephrasing?" : response;
    }
    
    // EFFICIENT: Message splitting with cached buffer
    void split_message_to_buffer(const std::string& message, size_t max_length = 2000) const {
        message_parts_buffer.clear();
        
        if (message.length() <= max_length) {
            message_parts_buffer.push_back(message);
            return;
        }
        
        size_t start = 0;
        while (start < message.length()) {
            size_t end = std::min(start + max_length, message.length());
            
            // Find word boundary
            if (end < message.length()) {
                const size_t last_space = message.find_last_of(" \n\t", end);
                if (last_space != std::string::npos && last_space > start) {
                    end = last_space;
                }
            }
            
            message_parts_buffer.emplace_back(message.substr(start, end - start));
            start = end;
            
            // Skip whitespace
            while (start < message.length() && std::isspace(message[start])) ++start;
        }
    }
    
    // UTILITY: Thread-safe logging
    void log_message(const std::string& message) const {
        discord_manager_log_callback(message);
    }
    
    // OPTIMIZED: Channel management helpers
    void parse_channel_ids(const std::string& channel_ids_str, std::unordered_set<uint64_t>& target_set) {
        std::lock_guard<std::mutex> lock(channel_mutex);
        target_set.clear();
        
        if (channel_ids_str.empty()) return;
        
        size_t start = 0;
        while (start < channel_ids_str.length()) {
            const size_t end = channel_ids_str.find(',', start);
            const size_t len = (end == std::string::npos) ? std::string::npos : end - start;
            
            std::string id_str = channel_ids_str.substr(start, len);
            
            // Trim whitespace
            const size_t first = id_str.find_first_not_of(" \t\n\r");
            if (first != std::string::npos) {
                const size_t last = id_str.find_last_not_of(" \t\n\r");
                id_str = id_str.substr(first, last - first + 1);
                
                try {
                    target_set.insert(std::stoull(id_str));
                } catch (const std::exception&) {
                    log_message("Invalid channel ID: " + id_str);
                }
            }
            
            start = (end == std::string::npos) ? channel_ids_str.length() : end + 1;
        }
    }
    
    // FAST: Channel permission checks
    bool is_channel_allowed(uint64_t channel_id) const {
        std::lock_guard<std::mutex> lock(channel_mutex);
        return allowed_channels.empty() || allowed_channels.count(channel_id);
    }
    
    bool is_isolated_channel(uint64_t channel_id) const {
        std::lock_guard<std::mutex> lock(channel_mutex);
        return isolated_channels.count(channel_id) > 0;
    }
    
    // EFFICIENT: Rate limiting
    bool is_rate_limited(uint64_t user_id) {
        std::lock_guard<std::mutex> lock(rate_limit_mutex);
        const auto now = std::chrono::system_clock::now();
        
        const auto it = last_response_time.find(user_id);
        if (it != last_response_time.end() && (now - it->second) < min_response_interval) {
            return true;
        }
        
        last_response_time[user_id] = now;
        return false;
    }
    
    // STREAMLINED: Context management
    std::string get_or_create_context(uint64_t user_id, const std::string& username, 
                                    uint64_t channel_id, uint64_t guild_id) {
        const bool is_dm = (guild_id == 0);
        const bool is_isolated = is_isolated_channel(channel_id);
        
        // Use shared main context for regular channels
        if (!is_isolated && !is_dm) {
            return (llama_manager && llama_manager->has_context(main_context_id)) ? main_context_id : "";
        }
        
        std::lock_guard<std::mutex> lock(context_mutex);
        
        if (is_dm) {
            // DM context management
            auto it = user_contexts.find(user_id);
            if (it != user_contexts.end() && llama_manager && llama_manager->has_context(it->second)) {
                return it->second;
            }
            
            // Create new DM context
            const std::string context_id = "discord_dm_" + std::to_string(user_id);
            if (llama_manager && llama_manager->create_context(context_id, "")) {
                user_contexts[user_id] = context_id;
                return context_id;
            }
        } else if (is_isolated) {
            // Isolated channel context management
            auto it = channel_contexts.find(channel_id);
            if (it != channel_contexts.end() && llama_manager && llama_manager->has_context(it->second)) {
                return it->second;
            }
            
            // Create new isolated channel context
            const std::string context_id = "discord_channel_" + std::to_string(channel_id);
            if (llama_manager && llama_manager->create_context(context_id, "")) {
                channel_contexts[channel_id] = context_id;
                return context_id;
            }
        }
        
        return "";
    }
    
    // CLEANUP: Context removal helpers
    void cleanup_user_context(uint64_t user_id) {
        std::lock_guard<std::mutex> lock(context_mutex);
        const auto it = user_contexts.find(user_id);
        if (it != user_contexts.end()) {
            if (llama_manager && it->second != main_context_id) {
                llama_manager->remove_context(it->second);
            }
            user_contexts.erase(it);
        }
    }
    
    void cleanup_channel_context(uint64_t channel_id) {
        std::lock_guard<std::mutex> lock(context_mutex);
        const auto it = channel_contexts.find(channel_id);
        if (it != channel_contexts.end()) {
            if (llama_manager && it->second != main_context_id) {
                llama_manager->remove_context(it->second);
            }
            channel_contexts.erase(it);
        }
    }

public:
    DiscordManager() : llama_manager(nullptr), last_activity(std::chrono::system_clock::now()) {
        message_parts_buffer.reserve(4); // Pre-allocate for typical message splitting
    }
    
    ~DiscordManager() {
        shutdown();
    }
    
    // CONFIGURATION: Streamlined setup methods
    bool configure(const DiscordBotConfig& bot_config) {
        if (is_running) return false;
        if (bot_config.bot_token.empty()) return false;
        
        config = bot_config;
        return true;
    }
    
    void set_allowed_channels(const std::string& channel_ids) {
        parse_channel_ids(channel_ids, allowed_channels);
    }
    
    void set_isolated_channels(const std::string& channel_ids) {
        parse_channel_ids(channel_ids, isolated_channels);
    }
    
    void set_shared_history_channels(const std::string& channel_ids) {
        parse_channel_ids(channel_ids, shared_history_channels);
    }
    
    void set_allow_dms(bool allow) {
        std::lock_guard<std::mutex> lock(channel_mutex);
        allow_dms = allow;
    }
    
    void set_main_context_id(const std::string& context_id) {
        main_context_id = context_id;
    }
    
    // INTEGRATION: LlamaManager connection
    void set_llama_manager(LlamaManager* manager) {
        llama_manager = manager;
        if (!manager) cleanup_all_contexts();
    }
    
    void cleanup_all_contexts() {
        std::lock_guard<std::mutex> lock(context_mutex);
        
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
    }
    
    // LIFECYCLE: Bot management
    bool initialize() {
        if (config.bot_token.empty()) return false;
        
        try {
            const uint32_t intents = dpp::i_default_intents | dpp::i_message_content;
            bot = std::make_unique<dpp::cluster>(config.bot_token, intents);
            setup_event_handlers();
            return true;
        } catch (const std::exception& e) {
            log_message("Failed to initialize bot: " + std::string(e.what()));
            return false;
        }
    }
    
    bool start() {
        if (is_running) return true;
        if (!initialize()) return false;
        
        try {
            bot->start(dpp::st_return);
            is_running = true;
            should_stop = false;
            
            // Start history backfill after connection
            std::thread([this]() {
                std::this_thread::sleep_for(std::chrono::seconds(3));
                if (is_connected && llama_manager) {
                    start_chat_history_backfill();
                }
            }).detach();
            
            return true;
        } catch (const std::exception& e) {
            log_message("Failed to start bot: " + std::string(e.what()));
            return false;
        }
    }
    
    bool start_chat_history_backfill() {
        if (!llama_manager || main_context_id.empty()) return false;
        
        if (!history_manager) {
            history_manager = std::make_unique<DiscordHistoryManager>();
            history_manager->configure(bot.get(), llama_manager, main_context_id);
            
            std::lock_guard<std::mutex> lock(channel_mutex);
            history_manager->set_channel_configuration(
                &allowed_channels, &isolated_channels, 
                &shared_history_channels, &channel_contexts
            );
        }
        
        return history_manager->start_backfill();
    }
    
    void shutdown() {
        if (!is_running) return;
        
        should_stop = true;
        
        if (history_manager) {
            history_manager->stop_backfill();
            history_manager.reset();
        }
        
        if (bot) {
            bot->shutdown();
            bot.reset();
        }
        
        is_running = false;
        is_connected = false;
        
        cleanup_all_contexts();
        
        // Clear rate limiting data
        {
            std::lock_guard<std::mutex> lock(rate_limit_mutex);
            last_response_time.clear();
        }
    }
    
    // STATUS: Query methods
    bool is_bot_running() const { return is_running; }
    bool is_bot_connected() const { return is_connected; }
    
    struct BotStatistics {
        uint64_t messages_processed;
        uint64_t responses_sent;
        bool is_running;
        bool is_connected;
        std::chrono::system_clock::time_point last_activity;
    };
    
    BotStatistics get_statistics() const {
        return {
            total_messages_processed.load(),
            total_responses_sent.load(),
            is_running.load(),
            is_connected.load(),
            last_activity
        };
    }
    
    // MESSAGING: Optimized message sending
    bool send_message(uint64_t channel_id, const std::string& message) {
        if (!is_running || !is_connected || !bot || message.empty()) return false;
        
        try {
            split_message_to_buffer(message);
            
            for (const auto& part : message_parts_buffer) {
                bot->message_create(dpp::message(channel_id, part));
                
                if (message_parts_buffer.size() > 1) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                }
            }
            
            ++total_responses_sent;
            last_activity = std::chrono::system_clock::now();
            return true;
        } catch (const std::exception& e) {
            log_message("Error sending message: " + std::string(e.what()));
            return false;
        }
    }
    
    // CONVERSATION: Management methods
    void clear_user_conversation(uint64_t user_id) {
        cleanup_user_context(user_id);
    }
    
    void clear_channel_conversation(uint64_t channel_id) {
        cleanup_channel_context(channel_id);
    }
    
    std::string get_bot_status() const {
        if (!is_running) return "Stopped";
        return is_connected ? "Connected" : "Starting...";
    }
};

extern void discord_manager_log_callback(const std::string& message);
