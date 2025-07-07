// DiscordManager.hpp - Discord bot interface for LuminaChat using D++
//
// This provides a Discord bot connection interface that integrates
// with the Orchestrator for message routing and processing.
// 
// Implementation uses D++ (Discord Plus Plus) library for Discord API integration.

#pragma once

#include <string>
#include <functional>
#include <vector>
#include <atomic>
#include <memory>
#include <unordered_set>
#include <mutex>
#include <ctime>
#include <sstream>
#include <thread>
#include <queue>
#include <condition_variable>
#include <chrono>
#include <map>
#include <dpp/dpp.h>
#include <dpp/nlohmann/json.hpp>

// Forward declaration - InputSource is defined in Orchestrator.hpp
enum class InputSource;

// Discord channel information
struct DiscordChannel {
    std::string id;
    std::string name;
    std::string guild_id;
    std::string guild_name;
    bool auto_respond = false;
    bool is_allowed = false;
};

// Enhanced Discord user information
struct DiscordUser {
    std::string id;           // User ID (unique identifier)
    std::string username;     // Display username 
    std::string global_name;  // Global display name (if available)
    std::string original_username; // Original Discord username
};

// Discord message structure
struct DiscordMessage {
    std::string id;
    std::string channel_id;
    std::string guild_id;
    std::string username;
    std::string content;
    std::string timestamp;
};

// Enhanced message structure with user info
struct EnhancedDiscordMessage {
    std::string id;
    std::string channel_id;
    std::string guild_id;
    DiscordUser user;
    std::string content;
    std::string timestamp;
};

// Queued message for processing
struct QueuedMessage {
    std::string channel_id;
    DiscordUser user;
    std::string content;
    std::chrono::steady_clock::time_point timestamp;
    std::string user_channel_key; // username + channel_id for grouping
};

// Message group for consolidation
struct MessageGroup {
    std::string channel_id;
    DiscordUser user;
    std::vector<std::string> messages;
    std::chrono::steady_clock::time_point last_message_time;
    bool processing = false;
};

// Forward declarations
class LlamaManager;
class Orchestrator;

// Discord bot manager using D++
class DiscordManager {
private:
    std::atomic<bool> connected{false};
    std::atomic<bool> connecting{false};
    std::string bot_token;
    std::string last_error;
    mutable std::mutex error_mutex;
    std::vector<DiscordChannel> available_channels;
    std::unordered_set<std::string> allowed_channel_ids;
    mutable std::mutex channels_mutex;
    
    // D++ bot instance
    std::unique_ptr<dpp::cluster> bot;
    
    // Message queuing and processing
    std::queue<QueuedMessage> message_queue;
    std::map<std::string, MessageGroup> message_groups; // key: username + channel_id
    mutable std::map<std::string, bool> channel_generation_busy; // key: channel_id, value: is_generating
    mutable std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::thread processing_thread;
    std::atomic<bool> processing_active{false};
    
    // Debounce settings
    static constexpr int DEBOUNCE_SECONDS = 3;
    
    // Callback for raw message reception
    std::function<void(const std::string&, const std::string&, const std::string&)> message_callback;
    
    // Enhanced callback for detailed user information (optional)
    std::function<void(const std::string&, const std::string&, const DiscordUser&)> enhanced_message_callback;

    // Integration with LuminaChat components
    LlamaManager* llama_manager = nullptr;
    Orchestrator* orchestrator = nullptr;
    
    // Note: Private methods implemented inline below to avoid circular dependencies
    
public:
    DiscordManager() = default;
    ~DiscordManager() { 
        StopProcessing();
        Disconnect(); 
    }
    
    // Component integration
    void SetLlamaManager(LlamaManager* manager) { llama_manager = manager; }
    void SetOrchestrator(Orchestrator* orch) { orchestrator = orch; }
    
    // Connection management - implemented inline below
    bool IsConnected() const { return connected; }
    bool IsConnecting() const { return connecting; }
    
    // Status information - implemented inline below
    
    // Implementation
    inline bool Connect(const std::string& token) {
        if (token.empty()) {
            return false;
        }
        
        if (connected || connecting) {
            return false;
        }
        
        try {
            connecting = true;
            bot_token = token;
            
            // Create D++ bot instance
            bot = std::make_unique<dpp::cluster>(token, dpp::i_default_intents | dpp::i_message_content);
            
            // Set up event handlers
            bot->on_ready([this](const dpp::ready_t& event) {
                OnReady(event);
            });
            
            bot->on_message_create([this](const dpp::message_create_t& event) {
                OnMessage(event);
            });
            
            bot->on_guild_create([this](const dpp::guild_create_t& event) {
                OnGuildCreate(event);
            });
            
            // Start message processing thread
            StartProcessing();
            
                  // Start the bot (non-blocking)
        bot->start(dpp::st_return);
            
            return true;
            
        } catch (const std::exception& e) {
            connecting = false;
            connected = false;
            StopProcessing();
            bot.reset();
            
            // Store error details
            {
                std::lock_guard<std::mutex> lock(error_mutex);
                last_error = "Connection failed: " + std::string(e.what());
            }
            
            return false;
        }
    }

    inline void Disconnect() {
        connected = false;
        connecting = false;
        
        StopProcessing();
        
        if (bot) {
            bot->shutdown();
            bot.reset();
        }
        
        std::lock_guard<std::mutex> lock(channels_mutex);
        available_channels.clear();
        allowed_channel_ids.clear();
    }

    inline std::vector<DiscordChannel> GetChannels() const {
        std::lock_guard<std::mutex> lock(channels_mutex);
        return available_channels;
    }

    inline bool SetChannelAutoRespond(const std::string& channel_id, bool auto_respond) {
        std::lock_guard<std::mutex> lock(channels_mutex);
        for (auto& channel : available_channels) {
            if (channel.id == channel_id) {
                channel.auto_respond = auto_respond;
                return true;
            }
        }
        return false;
    }

    inline void SetAllowedChannels(const std::vector<std::string>& channel_ids) {
        std::lock_guard<std::mutex> lock(channels_mutex);
        allowed_channel_ids.clear();
        for (const auto& id : channel_ids) {
            allowed_channel_ids.insert(id);
        }
        
        // Update channel allowed status
        for (auto& channel : available_channels) {
            channel.is_allowed = allowed_channel_ids.count(channel.id) > 0;
        }
        
        // Setup context pairs for allowed channels
        SetupContextPairs();
    }

    inline void AddAllowedChannel(const std::string& channel_id) {
        std::lock_guard<std::mutex> lock(channels_mutex);
        allowed_channel_ids.insert(channel_id);
        
        // Update channel allowed status
        for (auto& channel : available_channels) {
            if (channel.id == channel_id) {
                channel.is_allowed = true;
                break;
            }
        }
        
        SetupContextPairs();
    }

    inline void RemoveAllowedChannel(const std::string& channel_id) {
        std::lock_guard<std::mutex> lock(channels_mutex);
        allowed_channel_ids.erase(channel_id);
        
        // Update channel allowed status
        for (auto& channel : available_channels) {
            if (channel.id == channel_id) {
                channel.is_allowed = false;
                break;
            }
        }
    }

    inline std::vector<std::string> GetAllowedChannels() const {
        std::lock_guard<std::mutex> lock(channels_mutex);
        return std::vector<std::string>(allowed_channel_ids.begin(), allowed_channel_ids.end());
    }

    inline void RegisterMessageCallback(std::function<void(const std::string&, const std::string&, const std::string&)> callback) {
        message_callback = std::move(callback);
    }

    inline void RegisterEnhancedMessageCallback(std::function<void(const std::string&, const std::string&, const DiscordUser&)> callback) {
        enhanced_message_callback = std::move(callback);
    }

    inline bool SendMessage(const std::string& channel_id, const std::string& content) {
        if (!connected || !bot || content.empty()) {
            return false;
        }
        
        try {
            dpp::snowflake channel_snowflake = std::stoull(channel_id);
            dpp::message msg(channel_snowflake, content);
            bot->message_create(msg);
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    inline bool SendAIResponse(const std::string& channel_id, 
                              const std::string& external_response, 
                              const std::string& internal_response = "",
                              size_t context_current = 0, 
                              size_t context_maximum = 0) {
        if (!connected || !bot || external_response.empty()) {
            return false;
        }
        
        try {
            dpp::snowflake channel_snowflake = std::stoull(channel_id);
            
            // Create an embed for the AI response
            dpp::embed embed = dpp::embed()
                .set_color(0x00A0E4)  // Professional blue color
                .set_timestamp(time(0));
            
            // Add internal response as header if provided
            if (!internal_response.empty()) {
                embed.set_title("🤖 Internal Response")
                     .add_field("", internal_response, false);
            }
            
            // Add external response as main body
            embed.set_description(external_response);
            
            // Add context usage footer
            std::string footer_text = "Context usage: " + std::to_string(context_current);
            if (context_maximum > 0) {
                footer_text += "/" + std::to_string(context_maximum);
            }
            embed.set_footer(dpp::embed_footer().set_text(footer_text));
            
            // Create message with embed
            dpp::message msg(channel_snowflake, "");
            msg.add_embed(embed);
            
            bot->message_create(msg);
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    inline bool SendAIResponseAdvanced(const std::string& channel_id,
                                      const std::string& external_response,
                                      const std::string& internal_response = "",
                                      size_t context_current = 0,
                                      size_t context_maximum = 0,
                                      const std::string& model_name = "",
                                      const std::string& processing_time = "") {
        if (!connected || !bot || external_response.empty()) {
            return false;
        }
        
        try {
            dpp::snowflake channel_snowflake = std::stoull(channel_id);
            
            // Create an elegant embed for the AI response
            dpp::embed embed = dpp::embed()
                .set_color(0x5865F2)  // Discord blurple
                .set_timestamp(time(0))
                .set_author("LuminaChat AI", "", "");
            
            // Add internal response section if provided with improved formatting
            if (!internal_response.empty()) {
                try {
                    // Parse JSON and format in a user-friendly way
                    auto json_data = nlohmann::json::parse(internal_response);
                    
                    std::string formatted_analysis = "";
                    
                    // Helper lambda to safely extract and truncate strings
                    auto safe_extract = [](const nlohmann::json& data, const std::string& key, size_t max_length = 200) -> std::string {
                        if (!data.contains(key)) return "";
                        std::string value = data[key].get<std::string>();
                        if (value.length() > max_length) {
                            value = value.substr(0, max_length - 3) + "...";
                        }
                        // Escape any problematic characters for Discord
                        std::string escaped = "";
                        for (char c : value) {
                            if (c == '*' || c == '_' || c == '`' || c == '~') {
                                escaped += "\\";
                            }
                            escaped += c;
                        }
                        return escaped;
                    };
                    
                    // Speaker - show who the message is from
                    if (json_data.contains("speaker")) {
                        std::string speaker = safe_extract(json_data, "speaker", 100);
                        if (!speaker.empty()) {
                            formatted_analysis += "**Speaker:** " + speaker + "\n";
                        }
                    }
                    
                    // Raw Input - most important, so show it prominently
                    if (json_data.contains("raw_input")) {
                        std::string raw_input = safe_extract(json_data, "raw_input", 300);
                        if (!raw_input.empty()) {
                            formatted_analysis += "**Raw Input:** " + raw_input + "\n\n";
                        }
                    }
                    
                    // Intent
                    if (json_data.contains("intent")) {
                        std::string intent = safe_extract(json_data, "intent", 150);
                        if (!intent.empty()) {
                            formatted_analysis += "**Intent:** " + intent + "\n";
                        }
                    }
                    
                    // Key Points - limit to prevent overflow
                    if (json_data.contains("key_points")) {
                        auto key_points = json_data["key_points"];
                        if (key_points.is_array() && !key_points.empty()) {
                            formatted_analysis += "**Key Points:**\n";
                            int count = 1;
                            int max_points = 3; // Limit to prevent overflow
                            for (const auto& point : key_points) {
                                if (count > max_points) break;
                                std::string point_text = point.get<std::string>();
                                if (point_text.length() > 100) {
                                    point_text = point_text.substr(0, 97) + "...";
                                }
                                formatted_analysis += std::to_string(count++) + ". " + point_text + "\n";
                            }
                            if (key_points.size() > max_points) {
                                formatted_analysis += "... and " + std::to_string(key_points.size() - max_points) + " more\n";
                            }
                        } else if (key_points.is_string()) {
                            std::string point_text = safe_extract(json_data, "key_points", 150);
                            if (!point_text.empty()) {
                                formatted_analysis += "**Key Points:** " + point_text + "\n";
                            }
                        }
                    }
                    
                    // Reasoning Steps - show thought process
                    if (json_data.contains("reasoning_steps")) {
                        auto reasoning_steps = json_data["reasoning_steps"];
                        if (reasoning_steps.is_array() && !reasoning_steps.empty()) {
                            formatted_analysis += "**Reasoning:**\n";
                            int count = 1;
                            int max_steps = 2; // Limit to prevent overflow
                            for (const auto& step : reasoning_steps) {
                                if (count > max_steps) break;
                                std::string step_text = step.get<std::string>();
                                if (step_text.length() > 80) {
                                    step_text = step_text.substr(0, 77) + "...";
                                }
                                formatted_analysis += std::to_string(count++) + ". " + step_text + "\n";
                            }
                            if (reasoning_steps.size() > max_steps) {
                                formatted_analysis += "... and " + std::to_string(reasoning_steps.size() - max_steps) + " more steps\n";
                            }
                        }
                    }
                    
                    // Response Plan - strategy for reply
                    if (json_data.contains("response_plan")) {
                        std::string response_plan = safe_extract(json_data, "response_plan", 180);
                        if (!response_plan.empty()) {
                            formatted_analysis += "**Response Plan:** " + response_plan + "\n";
                        }
                    }
                    
                    // Strategy - fallback for older format compatibility
                    if (json_data.contains("strategy") && !json_data.contains("response_plan")) {
                        std::string strategy = safe_extract(json_data, "strategy", 200);
                        if (!strategy.empty()) {
                            formatted_analysis += "**Strategy:** " + strategy;
                        }
                    }
                    
                    // Remove trailing newlines
                    while (!formatted_analysis.empty() && formatted_analysis.back() == '\n') {
                        formatted_analysis.pop_back();
                    }
                    
                    // Discord embed fields have a 1024 character limit
                    if (formatted_analysis.length() > 1020) {
                        formatted_analysis = formatted_analysis.substr(0, 1017) + "...";
                    }
                    
                    // Only add the field if we have meaningful content
                    if (!formatted_analysis.empty()) {
                        embed.add_field("🧠 Internal Thoughts", formatted_analysis, false);
                    }
                    
                } catch (const nlohmann::json::exception&) {
                    // Fallback to simple text if JSON parsing fails
                    std::string fallback_text = internal_response;
                    if (fallback_text.length() > 1020) {
                        fallback_text = fallback_text.substr(0, 1017) + "...";
                    }
                    embed.add_field("🧠 Internal Thoughts", fallback_text, false);
                }
            }
            
            // Main response body
            embed.set_description(external_response);
            
            // Build footer with context and timing information
            std::string footer_text = "";
            
            // Context usage information
            std::string context_text = std::to_string(context_current);
            if (context_maximum > 0) {
                context_text += " / " + std::to_string(context_maximum);
                float usage_percent = context_maximum > 0 ? (float(context_current) / float(context_maximum)) * 100.0f : 0.0f;
                
                // Add visual indicator for context usage
                std::string indicator = "🟢"; // Green for low usage
                if (usage_percent > 80.0f) indicator = "🔴"; // Red for high usage
                else if (usage_percent > 60.0f) indicator = "🟡"; // Yellow for medium usage
                
                context_text = indicator + " " + context_text + " tokens";
            } else {
                context_text += " tokens";
            }
            
            footer_text = "Context: " + context_text;
            
            // Add timing information to footer if available
            if (!processing_time.empty()) {
                footer_text += " • Time: " + processing_time;
            }
            
            embed.set_footer(dpp::embed_footer().set_text(footer_text));
            
            // Create message with embed
            dpp::message msg(channel_snowflake, "");
            msg.add_embed(embed);
            
            bot->message_create(msg);
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    // Debug method to help troubleshoot JSON formatting issues
    inline bool SendAIResponseDebug(const std::string& channel_id,
                                   const std::string& external_response,
                                   const std::string& internal_response = "",
                                   size_t context_current = 0,
                                   size_t context_maximum = 0,
                                   const std::string& model_name = "",
                                   const std::string& processing_time = "") {
        if (!connected || !bot || external_response.empty()) {
            return false;
        }
        
        try {
            dpp::snowflake channel_snowflake = std::stoull(channel_id);
            
            // Create an elegant embed for the AI response
            dpp::embed embed = dpp::embed()
                .set_color(0x5865F2)  // Discord blurple
                .set_timestamp(time(0))
                .set_author("LuminaChat AI (Debug Mode)", "", "");
            
            // Add raw internal response for debugging
            if (!internal_response.empty()) {
                std::string debug_text = "**Raw JSON:**\n```json\n";
                std::string truncated_json = internal_response;
                if (truncated_json.length() > 800) {
                    truncated_json = truncated_json.substr(0, 800) + "...";
                }
                debug_text += truncated_json + "\n```";
                
                // Try to parse and show what we extract
                try {
                    auto json_data = nlohmann::json::parse(internal_response);
                    debug_text += "\n**Parsed Fields:**\n";
                    
                    if (json_data.contains("speaker")) {
                        debug_text += "• Speaker: `" + json_data["speaker"].get<std::string>() + "`\n";
                    }
                    if (json_data.contains("raw_input")) {
                        std::string raw_input = json_data["raw_input"].get<std::string>();
                        if (raw_input.length() > 100) {
                            raw_input = raw_input.substr(0, 97) + "...";
                        }
                        debug_text += "• Raw Input: `" + raw_input + "`\n";
                    }
                    if (json_data.contains("intent")) {
                        debug_text += "• Intent: `" + json_data["intent"].get<std::string>() + "`\n";
                    }
                    if (json_data.contains("key_points") && json_data["key_points"].is_array()) {
                        debug_text += "• Key Points: " + std::to_string(json_data["key_points"].size()) + " items\n";
                    }
                    if (json_data.contains("reasoning_steps") && json_data["reasoning_steps"].is_array()) {
                        debug_text += "• Reasoning Steps: " + std::to_string(json_data["reasoning_steps"].size()) + " steps\n";
                    }
                    if (json_data.contains("response_plan")) {
                        debug_text += "• Response Plan: `" + json_data["response_plan"].get<std::string>() + "`\n";
                    }
                    if (json_data.contains("strategy")) {
                        debug_text += "• Strategy: `" + json_data["strategy"].get<std::string>() + "`\n";
                    }
                } catch (const nlohmann::json::exception& e) {
                    debug_text += "\n**JSON Parse Error:** " + std::string(e.what());
                }
                
                // Discord has a 1024 character limit for embed fields
                if (debug_text.length() > 1020) {
                    debug_text = debug_text.substr(0, 1017) + "...";
                }
                
                embed.add_field("🔍 Debug Info", debug_text, false);
            }
            
            // Main response body
            embed.set_description(external_response);
            
            // Build footer with context and timing information
            std::string footer_text = "Context: " + std::to_string(context_current);
            if (context_maximum > 0) {
                footer_text += " / " + std::to_string(context_maximum) + " tokens";
            }
            if (!processing_time.empty()) {
                footer_text += " • Time: " + processing_time;
            }
            embed.set_footer(dpp::embed_footer().set_text(footer_text));
            
            // Create message with embed
            dpp::message msg(channel_snowflake, "");
            msg.add_embed(embed);
            
            bot->message_create(msg);
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    inline std::vector<DiscordMessage> GetChannelHistory(const std::string& channel_id, size_t limit) {
        std::vector<DiscordMessage> history;
        
        if (!connected || !bot) {
            return history;
        }
        
        try {
            // Note: This is a simplified implementation
            // In a full implementation, you'd use bot->messages_get() with proper callbacks
            // For now, return empty to indicate this feature needs async implementation
            return history;
        } catch (const std::exception&) {
            return history;
        }
    }

    inline std::string GetConnectionStatus() const {
        if (connected) {
            // Note: get_guild_cache() is not available in this D++ version
            // We'll show channel count instead
            size_t channel_count = GetChannelCount();
            return "Connected (" + std::to_string(channel_count) + " channels)";
        } else if (connecting) {
            return "Connecting...";
        } else {
            return "Disconnected";
        }
    }

    inline std::string GetLastError() const {
        std::lock_guard<std::mutex> lock(error_mutex);
        return last_error;
    }

    inline size_t GetGuildCount() const {
        // Note: get_guild_cache() is not available in this D++ version
        // Return 0 for now or implement alternative counting
        return 0;
    }

    inline size_t GetChannelCount() const {
        std::lock_guard<std::mutex> lock(channels_mutex);
        return available_channels.size();
    }

    inline size_t GetQueuedMessageCount() const {
        std::lock_guard<std::mutex> lock(queue_mutex);
        return message_groups.size();
    }

    inline std::string GetQueueStatus() const {
        std::lock_guard<std::mutex> lock(queue_mutex);
        std::string status = "Queued groups: " + std::to_string(message_groups.size());
        
        if (!message_groups.empty()) {
            status += " (";
            bool first = true;
            for (const auto& [key, group] : message_groups) {
                if (!first) status += ", ";
                status += group.user.username + ":" + std::to_string(group.messages.size());
                first = false;
            }
            status += ")";
        }
        
        // Add busy channels info
        size_t busy_count = 0;
        for (const auto& [channel_id, busy] : channel_generation_busy) {
            if (busy) busy_count++;
        }
        
        if (busy_count > 0) {
            status += " • Busy channels: " + std::to_string(busy_count);
        }
        
        return status;
    }

    // Method to mark channel as not busy (called when generation completes)
    inline void OnGenerationComplete(const std::string& channel_id) {
        MarkChannelBusy(channel_id, false);
    }

    // Event handlers
    inline void OnReady(const dpp::ready_t& event) {
        connected = true;
        connecting = false;
        
        // Bot is ready - channels will be loaded via guild_create events
    }

    // Message processing methods
    inline void StartProcessing() {
        processing_active = true;
        processing_thread = std::thread(&DiscordManager::ProcessMessageQueue, this);
    }

    inline void StopProcessing() {
        if (processing_active) {
            processing_active = false;
            queue_cv.notify_all();
            if (processing_thread.joinable()) {
                processing_thread.join();
            }
            
            // Clear pending messages
            std::lock_guard<std::mutex> lock(queue_mutex);
            std::queue<QueuedMessage> empty_queue;
            message_queue.swap(empty_queue);
            message_groups.clear();
            channel_generation_busy.clear();
        }
    }

    inline void ProcessMessageQueue() {
        while (processing_active) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            
            // Wait for messages or shutdown signal
            queue_cv.wait_for(lock, std::chrono::milliseconds(500), [this] {
                return !message_queue.empty() || !processing_active;
            });
            
            if (!processing_active) {
                break;
            }
            
            auto now = std::chrono::steady_clock::now();
            
            // Process expired message groups (debounce period has passed)
            std::vector<std::string> groups_to_process;
            for (auto& [key, group] : message_groups) {
                if (!group.processing) {
                    auto time_since_last = std::chrono::duration_cast<std::chrono::seconds>(
                        now - group.last_message_time).count();
                    
                    if (time_since_last >= DEBOUNCE_SECONDS) {
                        groups_to_process.push_back(key);
                        group.processing = true;
                    }
                }
            }
            
            // Process each expired group
            for (const auto& key : groups_to_process) {
                auto it = message_groups.find(key);
                if (it != message_groups.end()) {
                    MessageGroup group = it->second;
                    message_groups.erase(it);
                    
                    // Check if this channel is currently generating
                    bool channel_busy = channel_generation_busy[group.channel_id];
                    if (channel_busy) {
                        // Re-queue for later processing
                        message_groups[key] = group;
                        message_groups[key].processing = false;
                        continue;
                    }
                    
                    lock.unlock();
                    
                    // Consolidate messages from this user
                    std::string consolidated_content;
                    for (size_t i = 0; i < group.messages.size(); ++i) {
                        if (i > 0) {
                            consolidated_content += " ";
                        }
                        consolidated_content += group.messages[i];
                    }
                    
                    // Mark channel as busy before processing
                    MarkChannelBusy(group.channel_id, true);
                    
                    // Process the consolidated message
                    ProcessSingleMessage(group.channel_id, group.user, consolidated_content);
                    
                    lock.lock();
                }
            }
        }
    }

    inline void ProcessSingleMessage(const std::string& channel_id, 
                                   const DiscordUser& user, 
                                   const std::string& content) {
        // Call message callback if registered
        if (enhanced_message_callback) {
            // Use enhanced callback if available (provides full user info)
            enhanced_message_callback(content, channel_id, user);
        } else if (message_callback) {
            // Fall back to basic callback (backward compatibility)
            message_callback(content, channel_id, user.username);
        }
        
        // Route through orchestrator if available
        if (orchestrator) {
            // Note: We can't call orchestrator methods directly here due to circular dependencies
            // The orchestrator integration is handled through the message callback
            // The message callback in LuminaChat.cpp will route to the orchestrator
            
            // For now, we'll mark the channel as not busy after a timeout
            // In a full implementation, we'd get a callback when generation completes
            std::thread([this, channel_id]() {
                // Wait for a reasonable generation time (adjust as needed)
                std::this_thread::sleep_for(std::chrono::seconds(30));
                MarkChannelBusy(channel_id, false);
            }).detach();
        } else {
            // No orchestrator - mark as not busy immediately
            MarkChannelBusy(channel_id, false);
        }
    }

    inline void QueueMessage(const std::string& channel_id, 
                           const DiscordUser& user, 
                           const std::string& content) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        
        std::string user_channel_key = user.username + "_" + channel_id;
        auto now = std::chrono::steady_clock::now();
        
        // Add to or create message group
        auto& group = message_groups[user_channel_key];
        group.channel_id = channel_id;
        group.user = user;
        group.messages.push_back(content);
        group.last_message_time = now;
        group.processing = false; // Reset processing flag for new messages
        
        queue_cv.notify_one();
    }

    inline void MarkChannelBusy(const std::string& channel_id, bool busy) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        channel_generation_busy[channel_id] = busy;
    }

    inline bool IsChannelBusy(const std::string& channel_id) const {
        std::lock_guard<std::mutex> lock(queue_mutex);
        auto it = channel_generation_busy.find(channel_id);
        return it != channel_generation_busy.end() && it->second;
    }

    inline void OnMessage(const dpp::message_create_t& event) {
        // Ignore bot messages
        if (event.msg.author.is_bot()) {
            return;
        }
        
        std::string channel_id = std::to_string(event.msg.channel_id);
        
        // Add channel to our list if we haven't seen it before
        {
            std::lock_guard<std::mutex> lock(channels_mutex);
            bool channel_exists = false;
            for (const auto& channel : available_channels) {
                if (channel.id == channel_id) {
                    channel_exists = true;
                    break;
                }
            }
            
            if (!channel_exists && bot) {
                // Try to get channel info
                dpp::channel* channel_info = dpp::find_channel(event.msg.channel_id);
                if (channel_info && channel_info->get_type() == dpp::CHANNEL_TEXT) {
                    DiscordChannel dc;
                    dc.id = channel_id;
                    dc.name = channel_info->name;
                    dc.guild_id = std::to_string(channel_info->guild_id);
                    
                    // Try to get guild name
                    dpp::guild* guild_info = dpp::find_guild(channel_info->guild_id);
                    if (guild_info) {
                        dc.guild_name = guild_info->name;
                    }
                    
                    dc.auto_respond = false;
                    dc.is_allowed = allowed_channel_ids.count(dc.id) > 0;
                    
                    available_channels.push_back(dc);
                }
            }
        }
        
        // Check if channel is allowed
        if (!IsChannelAllowed(channel_id)) {
            return;
        }
        
        // Extract message data
        std::string content = event.msg.content;
        
        // Extract comprehensive user information
        DiscordUser user = CreateDiscordUser(event.msg.author);
        
        // Queue the message for processing instead of immediate processing
        QueueMessage(channel_id, user, content);
    }

    inline void OnGuildCreate(const dpp::guild_create_t& event) {
        // Load channels from this guild
        LoadChannelsFromGuild(event.created);
    }

    // Helper methods
    inline void LoadChannelsFromGuild(const dpp::guild& guild) {
        std::lock_guard<std::mutex> lock(channels_mutex);
        
        if (!bot) {
            return;
        }
        
        // For now, we'll use a simpler approach
        // In a full implementation, you'd fetch channels asynchronously
        // For this implementation, we'll just note that the guild exists
        // and rely on the allowed_channel_ids being set manually
        
        // The channels will be populated as messages are received
        // This is a simplified approach for the initial implementation
        
        // Setup context pairs for allowed channels (if any)
        SetupContextPairs();
    }

    inline void SetupContextPairs() {
        // Note: This would require LlamaManager to be included
        // For now, this is a no-op to avoid circular dependencies
        // In practice, this would be called from the main application
        // after all components are initialized
    }

    inline bool IsChannelAllowed(const std::string& channel_id) const {
        std::lock_guard<std::mutex> lock(channels_mutex);
        return allowed_channel_ids.count(channel_id) > 0;
    }

    // Helper method to extract username with fallback handling
    inline std::string ExtractUsername(const dpp::user& user, const std::string& channel_id) {
        std::string username;
        
        // Try to get the most user-friendly name in this order:
        // 1. Global name (new Discord feature)
        // 2. Username 
        // 3. User ID as fallback
        
        // Check for global name first (newer Discord feature)
        if (!user.global_name.empty()) {
            username = user.global_name;
        }
        // Fall back to regular username
        else if (!user.username.empty()) {
            username = user.username;
        }
        // Ultimate fallback to user ID
        else {
            username = "User_" + std::to_string(user.id);
        }
        
        // Sanitize username for safety (remove or replace problematic characters)
        std::string sanitized_username;
        for (char c : username) {
            // Allow alphanumeric, spaces, underscores, hyphens, and common Unicode ranges
            if (std::isalnum(c) || c == ' ' || c == '_' || c == '-' || 
                (static_cast<unsigned char>(c) >= 128)) { // Allow Unicode characters
                sanitized_username += c;
            } else {
                sanitized_username += '_'; // Replace problematic chars with underscore
            }
        }
        
        // Ensure username is not empty after sanitization
        if (sanitized_username.empty()) {
            sanitized_username = "User_" + std::to_string(user.id);
        }
        
        // Limit length to prevent issues with very long usernames
        if (sanitized_username.length() > 50) {
            sanitized_username = sanitized_username.substr(0, 47) + "...";
        }
        
        return sanitized_username;
    }

    // Helper method to create DiscordUser struct from dpp::user
    inline DiscordUser CreateDiscordUser(const dpp::user& user) {
        DiscordUser discord_user;
        discord_user.id = std::to_string(user.id);
        discord_user.original_username = user.username;
        discord_user.global_name = user.global_name;
        discord_user.username = ExtractUsername(user, ""); // Get the display name
        return discord_user;
    }
};
