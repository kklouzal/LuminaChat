// DiscordManager.hpp - header-only implementation for Discord integration via D++
// Handles Discord bot functionality and integration with LuminaChat backend.
//
// File Specific Directives:
// Manage Discord bot lifecycle, message handling, and integration with LlamaManager.
// Handle Discord rate limiting and async operations properly.
//
// OPTIMIZATIONS IMPLEMENTED:
// 1. STL Algorithm Enhancements: Replaced manual loops with efficient STL algorithms in parse_channel_ids, get_backfill_status, and cleanup_contexts
// 2. Direct Context Access: All context operations use direct llama_manager->get_context_info() calls for simplicity and maintainability
// 3. Direct Member Access: Leveraged atomic variables directly rather than thin accessor methods (Directive #7)  
// 4. String Operation Optimizations: Used string.reserve() and efficient concatenation patterns to reduce allocations
// 5. Redundancy Elimination: Consolidated similar cleanup patterns and removed duplicate validation code
// 6. Memory-Efficient Containers: Used STL transform_reduce patterns for efficient aggregation operations
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
// 3.  Consistent Style: Adopt a uniform coding style and structure for clarity and maintainability, ensure no syntatical or stylization errors.
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
#include <queue>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <dpp/dpp.h>

#include "DiscordHistoryLoader.hpp"
#include "LogHandler.hpp"
#include "Sanitizer.hpp"

// Forward declarations
class LlamaManager;

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
    LlamaManager* llama_manager;

    // Configuration
    DiscordBotConfig config;

    // State
    std::atomic<bool> is_connected{false};
    std::atomic<bool> should_stop{false};    // Channel configuration - moved to private with atomic access pattern
    std::unordered_set<uint64_t> isolated_channels;
    mutable std::mutex channel_config_mutex;

    // Context management with direct access pattern
    std::unordered_map<uint64_t, std::string> user_contexts;
    std::unordered_map<uint64_t, std::string> channel_contexts;
    std::unordered_map<uint64_t, std::chrono::system_clock::time_point> last_response_time;

    mutable std::mutex data_mutex;

    // Message queue system to prevent concurrent context access
    struct PendingMessage {
        std::string content;
        std::string username;
        uint64_t user_id;
        uint64_t channel_id;
        uint64_t guild_id;
        std::chrono::system_clock::time_point timestamp;
    };

    // Multi-message collection system for handling rapid successive messages
    struct MessageCollector {
        std::vector<PendingMessage> messages;
        std::chrono::system_clock::time_point last_message_time;
        std::unique_ptr<std::thread> timer_thread;
        std::atomic<bool> timer_active{false};
        uint64_t user_id;
        std::string context_id;
    };
    
    std::unordered_map<std::string, std::queue<PendingMessage>> context_message_queues;
    std::unordered_map<std::string, std::atomic<bool>> context_processing_flags;
    std::unordered_map<std::string, std::unique_ptr<MessageCollector>> pending_collectors;
    mutable std::mutex message_queue_mutex;
    mutable std::mutex collectors_mutex;
      // Individual history loaders per context
    std::unordered_map<std::string, std::shared_ptr<DiscordHistoryLoader>> context_history_loaders;
    mutable std::mutex loaders_mutex;
    
    // Performance tracking
    std::atomic<uint64_t> total_messages_processed{0};
    std::atomic<uint64_t> total_responses_sent{0};
    std::chrono::system_clock::time_point last_activity;
    
    // Constants
    static constexpr std::chrono::milliseconds MIN_RESPONSE_INTERVAL{2000};
    static constexpr std::chrono::milliseconds MESSAGE_COLLECTION_DELAY{3000};
    static constexpr size_t MAX_MESSAGE_LENGTH = 2000;
    static constexpr int32_t MAX_CONTEXT_FILL_PERCENTAGE = 80;
    static constexpr int32_t MIN_CONTEXT_FILL_PERCENTAGE = 10;
    static constexpr int32_t HEARTBEAT_DELAY_SEC = 5;
    static constexpr int32_t RETRY_DELAY_MS = 500;

public:
    // State - direct access for performance (Directive #7)
    std::atomic<bool> is_running{false};
    std::atomic<bool> allow_dms{true};
    std::atomic<bool> pull_message_history{true};
    std::atomic<int32_t> history_fill_percentage{50};

    // Configuration - direct access
    std::string main_context_id;
    std::string model_id = "main_model";
        BackfillStatus get_backfill_status() const {
        std::lock_guard<std::mutex> lock(loaders_mutex);
        
        BackfillStatus status{};
        status.total_channels = static_cast<int32_t>(context_history_loaders.size());
        
        // Use STL algorithms with direct method access per directive #7
        status.completed_channels = std::count_if(context_history_loaders.begin(), context_history_loaders.end(),
            [](const auto& pair) {
                return pair.second && pair.second->is_collection_complete();
            });
        
        status.in_progress = (status.completed_channels < status.total_channels);
        
        // Accumulate totals using direct getter methods
        status.total_messages_fetched = std::accumulate(context_history_loaders.begin(), context_history_loaders.end(), 0,
            [](int32_t sum, const auto& pair) {
                return sum + (pair.second ? static_cast<int32_t>(pair.second->get_messages_collected()) : 0);
            });
        
        status.total_tokens_loaded_this_session = std::accumulate(context_history_loaders.begin(), context_history_loaders.end(), 0,
            [](int32_t sum, const auto& pair) {
                return sum + (pair.second ? pair.second->get_total_tokens_collected() : 0);
            });
        
        return status;
    }

private:

void setup_event_handlers() {
        if (!bot) [[unlikely]] return;
        
        bot->on_ready([this](const dpp::ready_t& event) {
            try {
                is_connected = true;
                DISCORD_LOG("Discord bot ready! Logged in as: " + bot->me.username);
                
                // Verify LlamaManager is available before proceeding
                if (!llama_manager) {
                    DISCORD_LOG("ERROR: LlamaManager not available during bot ready event");
                    return;
                }
                
                if (main_context_id.empty()) {
                    DISCORD_LOG("ERROR: main_context_id not set during bot ready event");
                    return;
                }
                
                if (model_id.empty()) {
                    DISCORD_LOG("ERROR: model_id not set during bot ready event");
                    return;
                }
                
                // Verify main context exists
                if (!llama_manager->has_context(main_context_id)) {
                    DISCORD_LOG("ERROR: Main context '" + main_context_id + "' does not exist");
                    return;
                }
                
                DISCORD_LOG("Prerequisites verified - creating isolated channel contexts");
                
                // Proactively create contexts for isolated channels
                create_isolated_channel_contexts();
                
                // CRITICAL: Add delay to ensure all contexts are fully initialized
                // before any Discord message processing begins - increased for better reliability
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                LLAMA_LOG("Discord bot initialization completed - contexts ready for message processing");
            } catch (const std::exception& e) {
                DISCORD_LOG("CRITICAL ERROR in on_ready handler: " + std::string(e.what()));
            } catch (...) {
                DISCORD_LOG("CRITICAL ERROR: Unknown exception in on_ready handler");
            }
        });
        
        bot->on_message_create([this](const dpp::message_create_t& event) {
            try {
                handle_message(event);
            } catch (const std::exception& e) {
                DISCORD_LOG("ERROR in message handler: " + std::string(e.what()));
            } catch (...) {
                DISCORD_LOG("ERROR: Unknown exception in message handler");
            }
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
        if (event.msg.author.is_bot() || event.msg.content.empty()) [[unlikely]] return;
        
        const bool is_dm = (event.msg.guild_id == 0);
          // Early exit for disabled DMs
        if (is_dm && !allow_dms.load()) [[unlikely]] {
            send_message(event.msg.channel_id, 
                "Sorry, Direct Messages are currently disabled. Please use the appropriate server channels.");
            return;
        }
          // Early exit for channels that have no accessible context
        // Note: Non-isolated channels use the main_context_id, so only block if main context is missing
        if (!is_dm && !is_isolated_channel(event.msg.channel_id) && 
            (!llama_manager || !llama_manager->has_context(main_context_id))) [[unlikely]] {
            DISCORD_LOG("Blocking message from unconfigured channel " + std::to_string(event.msg.channel_id) + 
                       " - no main context available");
            return;
        }
        
        // Smart response logic: only respond when directly addressed or in general conversation
        // If user mentions/replies to others but not the bot, just add to history (listening mode)
        // NEW: Messages from the same user are collected for 3 seconds before generating a response
        // This allows users to send multiple messages in quick succession without triggering multiple AI responses
        bool should_generate_response = true;
        bool is_mentioning_other_user = false;
        bool bot_is_mentioned_or_replied_to = false;
        
        // Check for mentions first
        for (const auto& mention : event.msg.mentions) {
            if (mention.first.id == bot->me.id) {
                bot_is_mentioned_or_replied_to = true;
            } else {
                is_mentioning_other_user = true;
            }
        }
        
        // Check if this is a reply
        if (event.msg.message_reference.message_id != 0) {
            // This is a reply - we need to be more careful about determining who it's replying to
            // For now, we'll be conservative: if there are no bot mentions and there are other mentions,
            // or if this is clearly a conversation between users, treat it as listening mode
            // In practice, you might want to fetch the referenced message to check its author
            
            // If the bot is specifically mentioned in a reply, always respond
            if (!bot_is_mentioned_or_replied_to && is_mentioning_other_user) {
                should_generate_response = false;
            }
            // If it's a reply with no mentions at all, assume it might be to the bot 
            // (conservative approach to avoid missing direct replies)
        }
        
        // Logic: If mentioning others but not the bot, just listen
        if (is_mentioning_other_user && !bot_is_mentioned_or_replied_to) {
            should_generate_response = false;
        }
          // Always add to message history for context, but only generate response if appropriate
        if (should_generate_response) [[likely]] {
            // Rate limiting check only for responses
            if (is_rate_limited(event.msg.author.id)) [[unlikely]] return;
            
            // Queue message for processing (prevents concurrent context access)
            queue_user_message(
                event.msg.content, event.msg.author.username, 
                event.msg.author.id, event.msg.channel_id, event.msg.guild_id
            );
        } else {
            // Just add to history without generating a response
            add_message_to_history_only(
                event.msg.content, event.msg.author.username,
                event.msg.author.id, event.msg.channel_id, event.msg.guild_id
            );
        }
    }
    
    // Queue a message for processing with message collection delay
    void queue_user_message(const std::string& message, const std::string& username, 
                           uint64_t user_id, uint64_t channel_id, uint64_t guild_id) {
        // Sanitize the incoming message to prevent tokenization issues
        std::string sanitized_message = TextSanitizer::sanitize_text(message);
        if (sanitized_message.empty()) [[unlikely]] {
            send_message(channel_id, "I'm sorry, but your message couldn't be processed. Please try rephrasing your message.");
            return;
        }
        
        std::string context_id = get_or_create_user_context(user_id, username, channel_id, guild_id);
        if (context_id.empty()) [[unlikely]] {
            send_message(channel_id, "Error: Failed to access chat context");
            return;
        }
        
        // Create unique key for this user in this context
        std::string collector_key;
        collector_key.reserve(context_id.length() + 20); // Reserve space for efficiency
        collector_key.append(context_id).append("_").append(std::to_string(user_id));
        
        PendingMessage new_message{
            sanitized_message, username, user_id, channel_id, guild_id,
            std::chrono::system_clock::now()
        };        // Add message to history immediately using direct context access - retry initialization if needed
        ContextInfo* target_context = llama_manager ? llama_manager->get_context_info(context_id) : nullptr;
        if (target_context && target_context->fully_initialized.load()) [[likely]] {
            target_context->add_message("user", username + ": " + sanitized_message);
            ++total_messages_processed;
            last_activity = std::chrono::system_clock::now();
        } else if (target_context) [[unlikely]] {
            // Context exists but not fully initialized - attempt retry
            LLAMA_LOG("WARNING: Context '" + context_id + "' not initialized for message queue - attempting retry");
            
            if (llama_manager->retry_context_initialization(context_id)) {
                LLAMA_LOG("SUCCESS: Context '" + context_id + "' initialized successfully for message queue");
                target_context->add_message("user", username + ": " + sanitized_message);
                ++total_messages_processed;
                last_activity = std::chrono::system_clock::now();
            } else {
                LLAMA_LOG("ERROR: Context '" + context_id + "' retry initialization failed - message not added to queue history");
            }
        }
        
        {
            std::lock_guard<std::mutex> lock(collectors_mutex);
              // Get or create collector for this user/context combination
            auto& collector = pending_collectors[collector_key];
            if (!collector) [[unlikely]] {
                collector = std::make_unique<MessageCollector>();
                collector->user_id = user_id;
                collector->context_id = context_id;
            }
            
            // Add message to collector
            collector->messages.push_back(new_message);
            collector->last_message_time = std::chrono::system_clock::now();
            
            // If timer is already running, let it continue (it will pick up this new message)
            if (!collector->timer_active.exchange(true)) [[likely]] {
                // Start new timer
                collector->timer_thread = std::make_unique<std::thread>([this, collector_key]() {
                    std::this_thread::sleep_for(MESSAGE_COLLECTION_DELAY);
                    process_collected_messages(collector_key);
                });
                collector->timer_thread->detach();
            }
        }
        
        DISCORD_LOG("Added message to collector for user " + std::to_string(user_id) + " in context '" + context_id + "'");
    }
    
    // Process collected messages after delay period
    void process_collected_messages(const std::string& collector_key) {
        std::vector<PendingMessage> messages_to_process;
        std::string context_id;
          {
            std::lock_guard<std::mutex> lock(collectors_mutex);
            auto it = pending_collectors.find(collector_key);
            if (it == pending_collectors.end() || !it->second) [[unlikely]] {
                return; // Collector was already processed or removed
            }
            
            auto& collector = it->second;
              // Check if we should wait longer (if a very recent message was added)
            auto now = std::chrono::system_clock::now();
            auto time_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(now - collector->last_message_time);
            
            if (time_since_last < MESSAGE_COLLECTION_DELAY) [[unlikely]] {
                // Recent message detected, restart timer
                collector->timer_active = false;
                collector->timer_thread = std::make_unique<std::thread>([this, collector_key]() {
                    std::this_thread::sleep_for(MESSAGE_COLLECTION_DELAY);
                    process_collected_messages(collector_key);
                });
                collector->timer_thread->detach();
                collector->timer_active = true;
                return;
            }
            
            // Time's up, process all collected messages
            messages_to_process = std::move(collector->messages);
            context_id = collector->context_id;
            
            // Clean up collector
            pending_collectors.erase(it);
        }
        
        if (messages_to_process.empty()) [[unlikely]] return;
        
        // Combine all messages into a single prompt for the AI
        std::string combined_message;
        combined_message.reserve(std::accumulate(messages_to_process.begin(), messages_to_process.end(), 
                                               size_t{0}, [](size_t sum, const auto& msg) { 
                                                   return sum + msg.content.size() + 1; 
                                               }));
        
        const std::string& username = messages_to_process.front().username;
        const uint64_t channel_id = messages_to_process.front().channel_id;
        
        for (auto it = messages_to_process.begin(); it != messages_to_process.end(); ++it) {
            if (it != messages_to_process.begin()) combined_message += '\n';
            combined_message += it->content;
        }
        
        DISCORD_LOG("Processing " + std::to_string(messages_to_process.size()) + 
                   " collected messages for context '" + context_id + "'");
        
        // Add combined message to processing queue
        {
            std::lock_guard<std::mutex> lock(message_queue_mutex);
            auto& queue = context_message_queues[context_id];
            queue.emplace(combined_message, username, messages_to_process.front().user_id, 
                         channel_id, messages_to_process.front().guild_id,
                         std::chrono::system_clock::now());
            
            // Initialize processing flag if needed
            context_processing_flags.try_emplace(context_id, false);
        }
        
        // Start processing this context's queue
        process_context_queue(context_id);
    }
    
    // Process queued messages for a specific context (one at a time)
    void process_context_queue(const std::string& context_id) {        // Check if already processing this context
        {
            std::lock_guard<std::mutex> lock(message_queue_mutex);
            if (context_processing_flags[context_id].exchange(true)) [[unlikely]] {
                DISCORD_LOG("Context '" + context_id + "' already processing - message queued");
                return; // Already processing this context
            }
        }        // Process messages in queue sequentially
        std::thread([this, context_id]() {
            try {
                while (true) {
                    PendingMessage msg;
                    bool has_message = false;
                    
                    // Get next message from queue
                    {
                        std::lock_guard<std::mutex> lock(message_queue_mutex);
                        if (!context_message_queues[context_id].empty()) [[likely]] {
                            msg = context_message_queues[context_id].front();
                            context_message_queues[context_id].pop();
                            has_message = true;
                        }
                    }
                    
                    if (!has_message) [[unlikely]] {
                        break; // No more messages
                    }
                    
                    // Process the message (this will be serialized per context)
                    std::string response = process_user_message_direct(msg.content, msg.username, 
                                                                     msg.user_id, msg.channel_id, msg.guild_id);
                    
                    // Send response if not empty and not an error
                    if (!response.empty() && !response.starts_with("Error:")) [[likely]] {
                        send_message(msg.channel_id, response);
                    }
                    
                    total_messages_processed++;
                    last_activity = std::chrono::system_clock::now();
                }
                
                // Mark as no longer processing
                {
                    std::lock_guard<std::mutex> lock(message_queue_mutex);
                    context_processing_flags[context_id] = false;
                }
                
            } catch (const std::exception& e) {
                DISCORD_LOG("CRITICAL: Exception in context queue processing for '" + context_id + "': " + e.what());
                std::lock_guard<std::mutex> lock(message_queue_mutex);
                context_processing_flags[context_id] = false;
            }
        }).detach();
    }      std::string process_user_message_direct(const std::string& message, const std::string& username, 
                                           uint64_t user_id, uint64_t channel_id, uint64_t guild_id) {
        if (!llama_manager) [[unlikely]] return "Error: AI backend not available";        
        
        std::string context_id = get_or_create_user_context(user_id, username, channel_id, guild_id);
        if (context_id.empty()) [[unlikely]] return "Error: Failed to access chat context";
          // Use direct context access for better performance
        ContextInfo* target_context = llama_manager ? llama_manager->get_context_info(context_id) : nullptr;
        if (!target_context) [[unlikely]] {
            return "Error: Failed to access your chat context";
        }
          // CRITICAL: Verify context is ready for generation before proceeding, retry if needed
        if (!target_context->fully_initialized.load()) [[unlikely]] {
            LLAMA_LOG("WARNING: Discord attempting to use uninitialized context '" + context_id + "' - attempting retry initialization");
            
            // Attempt retry initialization
            if (llama_manager->retry_context_initialization(context_id)) {
                LLAMA_LOG("SUCCESS: Context '" + context_id + "' initialized successfully for Discord generation");
            } else {
                LLAMA_LOG("ERROR: Context '" + context_id + "' retry initialization failed - deferring message");
                return "The AI is still initializing for this channel. Please try again in a moment.";
            }
        }
        
        std::string response = llama_manager->generate_response(message, target_context, username);
        return response.empty() ? "I'm not sure how to respond to that. Could you try rephrasing?" : response;
    }
      // Add message to history without generating a response (for listening mode)
    void add_message_to_history_only(const std::string& message, const std::string& username,
                                    uint64_t user_id, uint64_t channel_id, uint64_t guild_id) {
        if (!llama_manager) [[unlikely]] return;
        
        // Sanitize the incoming message to prevent tokenization issues
        std::string sanitized_message = TextSanitizer::sanitize_text(message);
        if (sanitized_message.empty()) [[unlikely]] return;
        
        std::string context_id = get_or_create_user_context(user_id, username, channel_id, guild_id);
        if (context_id.empty()) [[unlikely]] return;        // Use direct context access to add message to history - retry initialization if needed
        ContextInfo* target_context = llama_manager ? llama_manager->get_context_info(context_id) : nullptr;
        if (target_context && target_context->fully_initialized.load()) [[likely]] {
            target_context->add_message("user", username + ": " + sanitized_message);
            ++total_messages_processed;
            last_activity = std::chrono::system_clock::now();
            DISCORD_LOG("Added message to history (listening mode) for context '" + context_id + "'");        } else if (target_context) [[unlikely]] {
            // Context exists but not fully initialized - attempt retry
            LLAMA_LOG("WARNING: Context '" + context_id + "' not initialized for history-only mode - attempting retry");
            
            if (llama_manager->retry_context_initialization(context_id)) {
                LLAMA_LOG("SUCCESS: Context '" + context_id + "' initialized successfully for history-only mode");
                target_context->add_message("user", username + ": " + sanitized_message);
                ++total_messages_processed;
                last_activity = std::chrono::system_clock::now();
                DISCORD_LOG("Added message to history (listening mode) for context '" + context_id + "' after retry");
            } else {
                LLAMA_LOG("ERROR: Context '" + context_id + "' retry initialization failed - message not added to history");
            }
        }
    }
      
    std::vector<std::string> split_message(const std::string& message, size_t max_length = MAX_MESSAGE_LENGTH) const {
        if (message.length() <= max_length) [[likely]] {
            return {message};
        }
        
        std::vector<std::string> parts;
        parts.reserve((message.length() + max_length - 1) / max_length); // Pre-allocate capacity
        
        size_t start = 0;
        while (start < message.length()) {
            size_t end = std::min(start + max_length, message.length());
            if (end < message.length()) {
                size_t last_space = message.find_last_of(" \n\t", end);
                if (last_space != std::string::npos && last_space > start) {
                    end = last_space;
                }
            }
            parts.emplace_back(message.substr(start, end - start));
            
            // Skip whitespace using STL algorithm
            start = end;
            auto it = std::find_if_not(message.begin() + start, message.end(), 
                                     [](char c) { return std::isspace(static_cast<unsigned char>(c)); });
            start = static_cast<size_t>(it - message.begin());
        }
        return parts;
    }
      void parse_channel_ids(const std::string& channel_ids_str, std::unordered_set<uint64_t>& target_set) {
        std::lock_guard<std::mutex> lock(channel_config_mutex);
        target_set.clear();
        
        if (channel_ids_str.empty()) [[unlikely]] return;
        
        // Reserve space based on comma count for efficiency
        target_set.reserve(std::count(channel_ids_str.begin(), channel_ids_str.end(), ',') + 1);
        
        // STL-optimized parsing with single-pass algorithm
        size_t start = 0;
        while (start < channel_ids_str.length()) {
            // Find next delimiter or end
            auto comma_pos = std::find(channel_ids_str.begin() + start, channel_ids_str.end(), ',');
            size_t end_pos = comma_pos - channel_ids_str.begin();
            
            // Extract and trim in single operation
            auto id_start = std::find_if_not(channel_ids_str.begin() + start, channel_ids_str.begin() + end_pos,
                                           [](char c) { return std::isspace(static_cast<unsigned char>(c)); });
            auto id_end = std::find_if_not(std::reverse_iterator(channel_ids_str.begin() + end_pos),
                                         std::reverse_iterator(channel_ids_str.begin() + start),
                                         [](char c) { return std::isspace(static_cast<unsigned char>(c)); }).base();            if (id_start < id_end) [[likely]] {
                try {
                    target_set.emplace(std::stoull(std::string(id_start, id_end)));
                } catch (const std::exception&) {
                    // Skip invalid IDs silently for performance
                }
            }
            
            start = (comma_pos == channel_ids_str.end()) ? channel_ids_str.length() : end_pos + 1;
        }
    }    bool is_isolated_channel(uint64_t channel_id) const {
        std::lock_guard<std::mutex> lock(channel_config_mutex);
        return isolated_channels.contains(channel_id);
    }
    
    bool is_rate_limited(uint64_t user_id) {
        const auto now = std::chrono::system_clock::now();
        std::lock_guard<std::mutex> lock(data_mutex);
          auto& last_time = last_response_time[user_id]; // Direct access with automatic insertion
        if (last_time != std::chrono::system_clock::time_point{} && (now - last_time) < MIN_RESPONSE_INTERVAL) [[unlikely]] {
            return true;
        }
        
        last_time = now;
        return false;
    }      // Create contexts for all isolated channels that the bot can access
    void create_isolated_channel_contexts() {
        if (!llama_manager || model_id.empty()) [[unlikely]] return;
        
        std::lock_guard<std::mutex> channel_lock(channel_config_mutex);
        std::string system_prompt = get_system_prompt_for_new_context();
        
        for (uint64_t channel_id : isolated_channels) {
            std::string context_id = "discord_channel_" + std::to_string(channel_id);
              // Check if context already exists
            if (llama_manager->has_context(context_id)) [[likely]] {
                std::lock_guard<std::mutex> data_lock(data_mutex);
                channel_contexts[channel_id] = context_id;
                DISCORD_LOG("Using existing context for isolated channel: " + std::to_string(channel_id));
            } else {
                // Create new context with mandatory warmup
                DISCORD_LOG("Creating new context for isolated channel: " + std::to_string(channel_id));
                if (llama_manager->create_context(context_id, model_id, system_prompt)) {
                    std::lock_guard<std::mutex> data_lock(data_mutex);
                    channel_contexts[channel_id] = context_id;
                    DISCORD_LOG("Created context '" + context_id + "' for isolated channel: " + std::to_string(channel_id));
                      // CRITICAL: Verify context is actually ready before proceeding, retry if needed
                    auto context_info = llama_manager->get_context_info(context_id);
                    if (context_info && !context_info->fully_initialized.load()) {
                        DISCORD_LOG("WARNING: Context '" + context_id + "' not fully initialized after creation - attempting retry");
                        
                        // Attempt retry initialization
                        if (llama_manager->retry_context_initialization(context_id)) {
                            DISCORD_LOG("SUCCESS: Context '" + context_id + "' initialized successfully on retry");
                        } else {
                            DISCORD_LOG("ERROR: Context '" + context_id + "' failed retry initialization - will require manual recovery");
                        }
                    }
                } else {
                    DISCORD_LOG("Warning: Failed to create context for isolated channel: " + std::to_string(channel_id));
                    continue;
                }
            }
              // Create and start individual history loader for this context
            if (pull_message_history.load() && bot) {
                auto context_info = llama_manager->get_context_info(context_id);
                if (context_info) {
                    const float fill_ratio = static_cast<float>(history_fill_percentage.load()) / 100.0f;                    auto loader = DiscordHistoryLoader::create_for_context(
                        context_id, channel_id, context_info, llama_manager, model_id,
                        bot.get(), static_cast<uint64_t>(bot->me.id), fill_ratio);
                    
                    loader->start_collection(); // Start immediately
                    
                    std::lock_guard<std::mutex> loaders_lock(loaders_mutex);
                    context_history_loaders[context_id] = loader;
                    
                    DISCORD_LOG("Started history collection for channel: " + std::to_string(channel_id));
                }
            }
        }
          DISCORD_LOG("Initialized " + std::to_string(isolated_channels.size()) + " isolated channel contexts");
    }// Get system prompt from main context for new Discord contexts
    std::string get_system_prompt_for_new_context() const {
        if (!llama_manager || main_context_id.empty()) [[unlikely]] return "";
        
        // Get the main context info to retrieve system prompt
        auto context_info = llama_manager->get_context_info(main_context_id);
        if (!context_info) [[unlikely]] return ""; // Check for null pointer
        
        // get the system message from the main context
        std::string system_prompt = context_info->system_message;
        
        return system_prompt;
    }
      std::string get_or_create_user_context(uint64_t user_id, const std::string& username, 
                                         uint64_t channel_id, uint64_t guild_id) {
        const bool is_dm = (guild_id == 0);
        const bool is_isolated_chan = is_isolated_channel(channel_id);
        
        // Use main context for regular channels that aren't isolated
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
            if (llama_manager && llama_manager->has_context(context_id)) {
                user_contexts[user_id] = context_id;
                return context_id;
            }
              // Use new API with model_id parameter and proper system prompt
            std::string system_prompt = get_system_prompt_for_new_context();
            if (llama_manager && !model_id.empty() && llama_manager->create_context(context_id, model_id, system_prompt)) {
                user_contexts[user_id] = context_id;
                  // CRITICAL: Verify context is ready for use, retry if needed
                auto created_context = llama_manager->get_context_info(context_id);
                if (created_context && !created_context->fully_initialized.load()) {
                    LLAMA_LOG("WARNING: Created Discord context '" + context_id + "' not fully initialized - attempting retry");
                    
                    // Attempt retry initialization
                    if (llama_manager->retry_context_initialization(context_id)) {
                        LLAMA_LOG("SUCCESS: Discord context '" + context_id + "' initialized successfully on retry");
                    } else {
                        LLAMA_LOG("ERROR: Discord context '" + context_id + "' failed retry initialization");
                    }
                }
                
                return context_id;
            }
        } else if (is_isolated_chan) {
            // For isolated channels, context should already exist from proactive creation
            auto it = channel_contexts.find(channel_id);
            if (it != channel_contexts.end() && llama_manager && llama_manager->has_context(it->second)) {
                return it->second;
            }
            
            // Fallback: create context if somehow missing (shouldn't happen normally)
            std::string context_id = "discord_channel_" + std::to_string(channel_id);
            if (llama_manager && llama_manager->has_context(context_id)) {
                channel_contexts[channel_id] = context_id;
                return context_id;
            }
              DISCORD_LOG("Warning: Creating missing context for isolated channel: " + std::to_string(channel_id));
            std::string system_prompt = get_system_prompt_for_new_context();
            if (llama_manager && !model_id.empty() && llama_manager->create_context(context_id, model_id, system_prompt)) {
                channel_contexts[channel_id] = context_id;
                  // CRITICAL: Verify context is ready for use, retry if needed
                auto created_context = llama_manager->get_context_info(context_id);
                if (created_context && !created_context->fully_initialized.load()) {
                    LLAMA_LOG("WARNING: Created fallback Discord context '" + context_id + "' not fully initialized - attempting retry");
                    
                    // Attempt retry initialization
                    if (llama_manager->retry_context_initialization(context_id)) {
                        LLAMA_LOG("SUCCESS: Fallback Discord context '" + context_id + "' initialized successfully on retry");
                    } else {
                        LLAMA_LOG("ERROR: Fallback Discord context '" + context_id + "' failed retry initialization");
                    }
                }
                
                return context_id;
            }
        }        return "";
    }
    
    // Helper function to get context ID for a specific channel/user
    std::string get_context_for_channel(uint64_t channel_id, uint64_t user_id = 0, uint64_t guild_id = 0) const {
        const bool is_dm = (guild_id == 0);
        const bool is_isolated_chan = is_isolated_channel(channel_id);
        
        // Use main context for regular channels that aren't isolated
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
        // Clean up message collectors
        {
            std::lock_guard<std::mutex> collectors_lock(collectors_mutex);
            pending_collectors.clear();
        }
          // Clean up individual history loaders
        {
            std::lock_guard<std::mutex> loaders_lock(loaders_mutex);
            context_history_loaders.clear();
        }
        
        // Clean up contexts in LlamaManager
        if (llama_manager) [[likely]] {
            std::lock_guard<std::mutex> lock(data_mutex);
            
            // Use STL algorithms for efficient processing - single pass for both containers
            auto cleanup_contexts = [this](const auto& container) {
                std::for_each(container.begin(), container.end(), [this](const auto& pair) {
                                 if (pair.second != main_context_id) [[likely]] {
                                     llama_manager->remove_context(pair.second);
                                 }
                             });
            };            
            cleanup_contexts(user_contexts);
            cleanup_contexts(channel_contexts);
        }
        
        // Clean up message queues
        {
            std::lock_guard<std::mutex> queue_lock(message_queue_mutex);
            context_message_queues.clear();
            context_processing_flags.clear();
        }
        
        // Clear local containers
        {
            std::lock_guard<std::mutex> lock(data_mutex);
            user_contexts.clear();
            channel_contexts.clear();
            last_response_time.clear();
        }
    }

public:

    DiscordManager() : llama_manager(nullptr) {
        last_activity = std::chrono::system_clock::now();
    }
    
    ~DiscordManager() { shutdown(); }    bool configure(const DiscordBotConfig& bot_config) {
        if (is_running || bot_config.bot_token.empty()) [[unlikely]] return false;
        
        // Validate bot token format (Discord bot tokens are typically 70+ characters and contain dots)
        if (bot_config.bot_token.length() < 50 || bot_config.bot_token.find('.') == std::string::npos) {
            DISCORD_LOG("Error: Invalid bot token format. Please check your Discord bot token in settings.");
            return false;
        }
        
        config = bot_config;
        DISCORD_LOG("Discord bot configured with token: " + bot_config.bot_token.substr(0, 10) + "...");
        return true;
    }
      void set_isolated_channels(const std::string& channel_ids) {
        parse_channel_ids(channel_ids, isolated_channels);
    }
    
    void set_allow_dms(bool allow) {
        allow_dms.store(allow);
    }
    
    void set_history_settings(bool pull_history, int32_t fill_percentage) {
        pull_message_history.store(pull_history);
        history_fill_percentage.store(std::clamp(fill_percentage, MIN_CONTEXT_FILL_PERCENTAGE, MAX_CONTEXT_FILL_PERCENTAGE));
        
        DISCORD_LOG("History settings updated: pull=" + std::string(pull_history ? "true" : "false") + 
                   ", fill=" + std::to_string(fill_percentage) + "%");
    }
      void set_llama_manager(LlamaManager* manager) {
        llama_manager = manager;
        
        if (manager) [[likely]] {
            // Ensure model_id is set
            if (model_id.empty()) [[unlikely]] {
                model_id = "main_model";
            }
            DISCORD_LOG("LlamaManager set with model_id: " + model_id);
        } else {
            cleanup_contexts();
        }
    }    bool initialize() {
        if (config.bot_token.empty()) [[unlikely]] return false;
        
        try {
            DISCORD_LOG("Initializing Discord bot with DPP library...");
            uint32_t intents = dpp::i_default_intents | dpp::i_message_content;
            bot = std::make_unique<dpp::cluster>(config.bot_token, intents);
            setup_event_handlers();
            
            DISCORD_LOG("Discord bot initialized successfully");
            return true;
        } catch (const std::exception& e) {
            DISCORD_LOG("Error: Failed to initialize Discord bot: " + std::string(e.what()));
            return false;
        }
    }bool start() {
        if (is_running || !initialize()) [[unlikely]] return false;
        
        // Validate that essential components are ready before starting
        if (!llama_manager) {
            DISCORD_LOG("ERROR: Cannot start Discord bot - LlamaManager not set");
            return false;
        }
        
        if (main_context_id.empty()) {
            DISCORD_LOG("ERROR: Cannot start Discord bot - main_context_id not set");
            return false;
        }
        
        if (!llama_manager->has_context(main_context_id)) {
            DISCORD_LOG("ERROR: Cannot start Discord bot - main context '" + main_context_id + "' does not exist");
            return false;
        }
        
        DISCORD_LOG("Starting Discord bot with validated configuration...");
        
        try {
            bot->start(dpp::st_return);
            is_running = true;
            should_stop = false;
            
            DISCORD_LOG("Discord bot start() called successfully");
            return true;
        } catch (const std::exception& e) {
            DISCORD_LOG("Error: Failed to start Discord bot: " + std::string(e.what()));
            return false;
        }
    }
      void shutdown() {
        if (!is_running) [[unlikely]] return;
        
        should_stop = true;
        
        if (bot) [[likely]] {
            bot->shutdown();
            bot.reset();
        }
        
        is_running = false;
        is_connected = false;
        cleanup_contexts();
    }
      bool send_message(uint64_t channel_id, const std::string& message) {
        if (!is_running.load() || !is_connected.load() || !bot || message.empty()) [[unlikely]] return false;
        
        // Keep try-catch here as Discord API calls can fail due to network/rate limiting
        try {
            const auto message_parts = split_message(message);
            const std::string context_id = get_context_for_channel(channel_id, 0, 0);
            std::string footer_text = "🤖 LuminaChat AI";
              // Get context information once outside the loop using direct access
            if (llama_manager && !context_id.empty()) [[likely]] {
                if (const auto* context_info = llama_manager->get_context_info(context_id)) [[likely]] {
                    const int32_t context_usage = context_info->n_past;
                    const int32_t context_size = context_info->get_context_size();
                    
                    if (context_size > 0) [[likely]] {
                        footer_text += " • Context: " + std::to_string(context_usage) + "/" + std::to_string(context_size);
                    }
                }
            }
            
            const auto timestamp = time(nullptr);
            for (const auto& part : message_parts) {
                // Create embed once per part
                dpp::embed embed = dpp::embed()
                    .set_color(0x00ff9f)
                    .set_description(part)
                    .set_footer(dpp::embed_footer().set_text(footer_text))
                    .set_timestamp(timestamp);
                
                dpp::message msg(channel_id, "");
                msg.add_embed(std::move(embed));
                bot->message_create(std::move(msg));
                  if (message_parts.size() > 1) [[unlikely]] {
                    std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAY_MS));
                }
            }
            
            total_responses_sent.fetch_add(1);
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