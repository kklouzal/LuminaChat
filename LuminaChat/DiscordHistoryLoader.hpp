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

// Essential structures only
struct ChannelState {
    uint64_t last_message_id = 0;
    bool fetch_complete = false;
    int32_t messages_fetched = 0;
    std::string context_id;
    bool is_isolated = false;
};

struct PendingMessage {
    uint64_t channel_id = 0;
    std::string username;
    std::string content;
    std::chrono::system_clock::time_point timestamp;
    uint64_t message_id = 0;
    std::vector<llama_token> tokenized_content;
    int32_t actual_token_count = 0;
    std::string target_context_id;
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
    
    // Channel processing
    std::unordered_map<uint64_t, ChannelState> channel_states;
    std::vector<uint64_t> shared_channels_list;
    size_t shared_channel_index = 0;
    mutable std::mutex state_mutex;
    
    // Message collection
    std::vector<PendingMessage> pending_messages;
    std::mutex pending_messages_mutex;
    
    // Capacity tracking
    std::unordered_map<std::string, int32_t> context_capacity_limits;
    std::unordered_map<std::string, int32_t> context_current_tokens;
    std::mutex capacity_mutex;
    
    // State
    std::atomic<bool> backfill_in_progress{false};
    
    // Constants
    static constexpr int32_t MESSAGES_PER_FETCH = 10;
    static constexpr float BASE_MAX_CONTEXT_FILL_RATIO = 0.01f;
    static constexpr int32_t MAX_RETRIES = 3;
    
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
    
    // NEW: Initialize context capacities once at start
    void initialize_context_capacities() {
        std::lock_guard<std::mutex> lock(capacity_mutex);
        context_capacity_limits.clear();
        context_current_tokens.clear();
        
        if (!llama_manager) return;
        
        std::string original_context = llama_manager->get_active_context();
        
        // Get unique context IDs
        std::unordered_set<std::string> unique_contexts;
        for (const auto& [channel_id, state] : channel_states) {
            unique_contexts.insert(state.context_id);
        }
        
        // Initialize capacity data for each context
        for (const std::string& context_id : unique_contexts) {
            if (llama_manager->switch_to_context(context_id)) {
                int32_t current_tokens = llama_manager->get_message_history_token_count();
                int32_t context_size = llama_manager->get_context_size();
                float fill_ratio = BASE_MAX_CONTEXT_FILL_RATIO * context_fill_percentage;
                int32_t capacity_limit = static_cast<int32_t>(context_size * fill_ratio);
                
                context_current_tokens[context_id] = current_tokens;
                context_capacity_limits[context_id] = capacity_limit;
                
                log_message("Context '" + context_id + "' capacity: " + std::to_string(current_tokens) + 
                           "/" + std::to_string(capacity_limit) + " tokens");
            }
        }
        
        // Restore original context
        if (!original_context.empty()) {
            llama_manager->switch_to_context(original_context);
        }
        
        log_message("Initialized capacity tracking for " + std::to_string(unique_contexts.size()) + " contexts");
    }
    
    // SIMPLIFIED: Check capacity without context switching
    bool can_add_more_messages_estimated_no_switch(const std::string& context_id, int32_t additional_tokens) {
        std::lock_guard<std::mutex> lock(capacity_mutex);
        
        auto capacity_it = context_capacity_limits.find(context_id);
        auto current_it = context_current_tokens.find(context_id);
        
        if (capacity_it == context_capacity_limits.end() || current_it == context_current_tokens.end()) {
            return false; // Unknown context
        }
        
        return (current_it->second + additional_tokens) < capacity_it->second;
    }
    
    // NEW: Update estimated token usage without context switching
    void update_estimated_token_usage(const std::string& context_id, int32_t additional_tokens) {
        std::lock_guard<std::mutex> lock(capacity_mutex);
        context_current_tokens[context_id] += additional_tokens;
    }
    
    // SIMPLIFIED: Direct message fetching and processing
    bool process_channel_batch(uint64_t channel_id) {
        auto& state = channel_states[channel_id];
        if (state.fetch_complete) {
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
                    
                    success = collect_messages_for_later(messages, channel_id);
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
    
    // MODIFIED: Collect messages with exact tokenization using LlamaManager
    bool collect_messages_for_later(const dpp::message_map& messages, uint64_t channel_id) {
        if (messages.empty()) return false;
        
        auto& state = channel_states[channel_id];
        
        // Process messages with exact tokenization
        std::vector<PendingMessage> batch_messages;
        uint64_t oldest_id = UINT64_MAX; // Track oldest message ID for next iteration
        int32_t total_exact_tokens = 0;
        
        for (const auto& [id, msg] : messages) {
            if (!msg.author.is_bot() && !msg.content.empty()) {
                PendingMessage pending;
                pending.channel_id = channel_id;
                pending.username = msg.author.username;
                pending.content = msg.content;
                pending.timestamp = std::chrono::system_clock::time_point(std::chrono::seconds(msg.sent));
                pending.message_id = static_cast<uint64_t>(msg.id);
                pending.target_context_id = state.context_id;
                
                // NEW: Use LlamaManager's tokenization instead of redundant local function
                std::string formatted_message = msg.author.username + ": " + msg.content;
                pending.tokenized_content = llama_manager->tokenize_text(formatted_message, false);
                
                // Check for reasonable token count limits
                if (pending.tokenized_content.size() > 2048) {
                    log_message("Warning: Message from " + msg.author.username + " would produce " + 
                               std::to_string(pending.tokenized_content.size()) + " tokens, skipping");
                    continue;
                }
                
                pending.actual_token_count = static_cast<int32_t>(pending.tokenized_content.size());
                
                if (pending.actual_token_count > 0) {
                    batch_messages.push_back(pending);
                    total_exact_tokens += pending.actual_token_count;
                    // FIXED: Track the oldest message ID (smallest value) for next iteration
                    oldest_id = std::min(oldest_id, pending.message_id);
                } else {
                    log_message("Warning: Tokenization produced 0 tokens for message from " + 
                               msg.author.username + ", skipping");
                }
            } else {
                // FIXED: Still need to track message IDs even for bot messages to ensure proper pagination
                uint64_t msg_id = static_cast<uint64_t>(msg.id);
                oldest_id = std::min(oldest_id, msg_id);
            }
        }
        
        // Check if we can add all messages using exact token counts
        if (!batch_messages.empty() && can_add_more_messages_estimated_no_switch(state.context_id, total_exact_tokens)) {
            // Add to pending collection
            {
                std::lock_guard<std::mutex> lock(pending_messages_mutex);
                pending_messages.insert(pending_messages.end(), batch_messages.begin(), batch_messages.end());
            }
            
            // Update exact token usage
            update_estimated_token_usage(state.context_id, total_exact_tokens);
            
            // Update state
            {
                std::lock_guard<std::mutex> lock(state_mutex);
                state.messages_fetched += static_cast<int32_t>(batch_messages.size());
                // FIXED: Set last_message_id to oldest message for next API call to fetch older messages
                if (oldest_id != UINT64_MAX) {
                    state.last_message_id = oldest_id;
                }
            }
            
            log_message("Collected " + std::to_string(batch_messages.size()) + " messages from channel " + 
                       std::to_string(channel_id) + " (exact " + std::to_string(total_exact_tokens) + 
                       " tokens), next fetch before ID: " + std::to_string(oldest_id));
            return true;
        } else if (!batch_messages.empty()) {
            // Mark channel as complete if we can't fit more
            std::lock_guard<std::mutex> lock(state_mutex);
            channel_states[channel_id].fetch_complete = true;
            
            // Get current capacity info for logging
            std::string capacity_info = "unknown";
            {
                std::lock_guard<std::mutex> cap_lock(capacity_mutex);
                auto current_it = context_current_tokens.find(state.context_id);
                auto capacity_it = context_capacity_limits.find(state.context_id);
                if (current_it != context_current_tokens.end() && capacity_it != context_capacity_limits.end()) {
                    capacity_info = std::to_string(current_it->second) + "/" + std::to_string(capacity_it->second);
                }
            }
            
            log_message("Channel " + std::to_string(channel_id) + " completed - context capacity reached (" + capacity_info + ")");
        } else {
            // FIXED: Even if no messages were added, update the last_message_id to continue pagination
            std::lock_guard<std::mutex> lock(state_mutex);
            if (oldest_id != UINT64_MAX) {
                state.last_message_id = oldest_id;
                log_message("Updated last_message_id for channel " + std::to_string(channel_id) + 
                           " to " + std::to_string(oldest_id) + " (no messages collected this batch)");
            }
        }
        
        return false;
    }
    
    // MODIFIED: Apply collected messages using pre-tokenized content
    void apply_collected_messages() {
        std::lock_guard<std::mutex> lock(pending_messages_mutex);
        if (pending_messages.empty()) return;
        
        log_message("Applying " + std::to_string(pending_messages.size()) + " collected tokenized messages to contexts...");
        
        // Sort messages by timestamp for chronological order
        std::sort(pending_messages.begin(), pending_messages.end(),
            [](const PendingMessage& a, const PendingMessage& b) {
                return a.timestamp < b.timestamp;
            });
        
        // Group messages by target context
        std::unordered_map<std::string, std::vector<PendingMessage>> context_groups;
        for (const auto& msg : pending_messages) {
            context_groups[msg.target_context_id].push_back(msg);
        }
        
        // Apply messages to each context using pre-tokenized content
        std::string original_context = llama_manager->get_active_context();
        
        for (const auto& [context_id, messages] : context_groups) {
            log_message("Switching to context '" + context_id + "' to add " + std::to_string(messages.size()) + " tokenized messages");
            
            if (llama_manager->switch_to_context(context_id)) {
                int32_t added_count = 0;
                int32_t total_tokens_added = 0;
                
                for (const auto& msg : messages) {
                    // Use the regular add_message_to_history for now
                    // In the future, we could add a method to LlamaManager that accepts pre-tokenized content
                    llama_manager->add_message_to_history(msg.username, msg.content);
                    added_count++;
                    total_tokens_added += msg.actual_token_count;
                }
                
                if (llama_manager->update_context_from_history()) {
                    log_message("Successfully added " + std::to_string(added_count) + " messages (" + 
                               std::to_string(total_tokens_added) + " tokens) to context " + context_id);
                } else {
                    log_message("Failed to update context " + context_id + " from history");
                }
            } else {
                log_message("Failed to switch to context " + context_id);
            }
        }
        
        // Restore original context
        if (!original_context.empty()) {
            llama_manager->switch_to_context(original_context);
            log_message("Restored original context: " + original_context);
        }
        
        // Clear processed messages
        pending_messages.clear();
        log_message("Tokenized message application phase completed");
    }
    
    // SIMPLIFIED: Round-robin for shared channels
    uint64_t get_next_shared_channel() {
        if (shared_channels_list.empty()) return 0;
        
        uint64_t channel_id = shared_channels_list[shared_channel_index];
        shared_channel_index = (shared_channel_index + 1) % shared_channels_list.size();
        return channel_id;
    }
    
    // MODIFIED: Updated main processing loop with proper isolated vs shared handling
    void process_all_channels() {
        const int32_t MAX_ITERATIONS = 1000;
        int32_t iteration = 0;
        
        // Initialize context capacities once
        log_message("Initializing context capacity tracking...");
        initialize_context_capacities();
        
        // Phase 1: Collect all messages
        log_message("Phase 1: Collecting messages from all channels...");
        
        while (backfill_in_progress && iteration < MAX_ITERATIONS) {
            bool made_progress = false;
            
            // Process isolated channels sequentially to completion
            // Since each has its own context, we can fill them completely without affecting others
            std::vector<uint64_t> isolated_channels_to_process;
            {
                std::lock_guard<std::mutex> lock(state_mutex);
                for (const auto& [channel_id, state] : channel_states) {
                    if (state.is_isolated && !state.fetch_complete) {
                        isolated_channels_to_process.push_back(channel_id);
                    }
                }
            }
            
            // Process all isolated channels until complete or no progress
            for (uint64_t channel_id : isolated_channels_to_process) {
                auto it = channel_states.find(channel_id);
                if (it != channel_states.end() && !it->second.fetch_complete) {
                    log_message("Processing isolated channel " + std::to_string(channel_id) + " to completion...");
                    
                    // Keep processing this isolated channel until it's done
                    while (!it->second.fetch_complete && backfill_in_progress) {
                        if (process_channel_batch(channel_id)) {
                            made_progress = true;
                            // Small delay between batches for rate limiting
                            std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        } else {
                            break; // Channel completed or error
                        }
                    }
                    
                    if (it->second.fetch_complete) {
                        log_message("Completed isolated channel " + std::to_string(channel_id));
                    }
                }
            }
            
            // Process shared channels using round-robin (only one batch per iteration)
            // This ensures fair distribution of the main context among shared channels
            if (!shared_channels_list.empty()) {
                uint64_t channel_id = get_next_shared_channel();
                auto it = channel_states.find(channel_id);
                if (it != channel_states.end() && !it->second.fetch_complete) {
                    if (process_channel_batch(channel_id)) {
                        made_progress = true;
                        log_message("Processed batch for shared channel " + std::to_string(channel_id) + 
                                   " (round-robin)");
                    }
                }
            }
            
            iteration++;
            
            if (!made_progress) {
                // Check if all channels are complete
                bool all_complete = true;
                {
                    std::lock_guard<std::mutex> lock(state_mutex);
                    for (const auto& [channel_id, state] : channel_states) {
                        if (!state.fetch_complete) {
                            all_complete = false;
                            break;
                        }
                    }
                }
                
                if (all_complete) {
                    log_message("All channels completed");
                    break;
                }
                
                // If no progress but channels remain, wait and try again
                log_message("No progress made, waiting before retry...");
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            } else {
                // Short delay between iterations when making progress
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
            }
        }
        
        // Phase 2: Apply all collected messages
        log_message("Phase 2: Applying collected messages to contexts...");
        apply_collected_messages();
        
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

//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODE DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//