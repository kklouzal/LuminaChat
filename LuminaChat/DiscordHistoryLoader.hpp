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
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <thread>
#include <algorithm>
#include <future>

#include <dpp/dpp.h>
#include "LogHandler.hpp"
#include "common_utils.hpp"

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
    std::string model_id; // Store model ID for context creation
    
    // Bot identification for recognizing own messages
    uint64_t bot_user_id = 0;
    std::string bot_username;
    
    // Channel configuration
    const std::unordered_set<uint64_t>* isolated_channels = nullptr;
    const std::unordered_set<uint64_t>* shared_history_channels = nullptr;
    
    // Settings
    bool history_enabled = true;
    int32_t context_fill_percentage = 50;
    
    // Channel processing
    std::unordered_map<uint64_t, ChannelState> channel_states;
    std::vector<uint64_t> shared_channels_list;
    size_t shared_channel_index = 0;    mutable std::mutex state_mutex;
    
    // Message collection
    std::vector<PendingMessage> pending_messages;
    std::mutex pending_messages_mutex;
    
    // Deduplication tracking
    std::unordered_set<uint64_t> processed_message_ids;
    std::mutex processed_ids_mutex;
    
    // Capacity tracking
    std::unordered_map<std::string, int32_t> context_capacity_limits;
    std::unordered_map<std::string, int32_t> context_current_tokens;
    std::mutex capacity_mutex;
    
    // State
    std::atomic<bool> backfill_in_progress{false};    // Constants
    static constexpr int32_t MESSAGES_PER_FETCH = 10;
    static constexpr float BASE_MAX_CONTEXT_FILL_RATIO = 0.01f;
    static constexpr int32_t MAX_ITERATIONS = 1000;
    static constexpr int32_t TIMEOUT_SECONDS = 10;
    static constexpr int32_t RATE_LIMIT_DELAY_MS = 100;
    static constexpr int32_t RETRY_DELAY_MS = 1000;
    static constexpr int32_t PROGRESS_DELAY_MS = 250;
    static constexpr int32_t INITIALIZATION_DELAY_SEC = 2;
    static constexpr int32_t MAX_MESSAGE_TOKENS = 2048;
    static constexpr int32_t MIN_FILL_PERCENTAGE = 10;
    static constexpr int32_t MAX_FILL_PERCENTAGE = 80;
    
    bool should_backfill_channel(uint64_t channel_id) const {
        return (isolated_channels && isolated_channels->count(channel_id)) ||
               (shared_history_channels && shared_history_channels->count(channel_id));
    }
    
    bool is_isolated_channel(uint64_t channel_id) const {
        return isolated_channels && isolated_channels->count(channel_id);
    }
    
    // Initialize context capacities once at start
    void initialize_context_capacities() {
        std::lock_guard<std::mutex> lock(capacity_mutex);
        context_capacity_limits.clear();
        context_current_tokens.clear();
        
        if (!llama_manager) return;
        
        // Get unique context IDs
        std::unordered_set<std::string> unique_contexts;
        for (const auto& [channel_id, state] : channel_states) {
            unique_contexts.insert(state.context_id);
        }
        
        // Initialize capacity data for each context
        for (const std::string& context_id : unique_contexts) {
            // Get the context info from LlamaManager
            auto context = llama_manager->get_context_info(context_id);

            int32_t current_tokens = context->message_history_token_count;
            int32_t context_size = context->get_context_size();
            float fill_ratio = BASE_MAX_CONTEXT_FILL_RATIO * context_fill_percentage;
            int32_t capacity_limit = static_cast<int32_t>(context_size * fill_ratio);
            
            context_current_tokens[context_id] = current_tokens;
            context_capacity_limits[context_id] = capacity_limit;
            
            DISCORD_HISTORY_LOG("Context '" + context_id + "' capacity: " + std::to_string(current_tokens) + 
                        "/" + std::to_string(capacity_limit) + " tokens");
        }
        
        DISCORD_HISTORY_LOG("Initialized capacity tracking for " + std::to_string(unique_contexts.size()) + " contexts");
    }
    
    // Check capacity without context switching
    bool can_add_more_messages_estimated_no_switch(const std::string& context_id, int32_t additional_tokens) {
        std::lock_guard<std::mutex> lock(capacity_mutex);
        
        auto capacity_it = context_capacity_limits.find(context_id);
        auto current_it = context_current_tokens.find(context_id);
        
        if (capacity_it == context_capacity_limits.end() || current_it == context_current_tokens.end()) {
            return false; // Unknown context
        }
        
        return (current_it->second + additional_tokens) < capacity_it->second;
    }
    
    // Update estimated token usage without context switching
    void update_estimated_token_usage(const std::string& context_id, int32_t additional_tokens) {
        std::lock_guard<std::mutex> lock(capacity_mutex);
        context_current_tokens[context_id] += additional_tokens;
    }
    
    // Direct message fetching and processing
    bool process_channel_batch(uint64_t channel_id) {
        auto& state = channel_states[channel_id];
        if (state.fetch_complete) {
            return false;
        }
        
        DISCORD_HISTORY_LOG("Fetching messages from channel " + std::to_string(channel_id));
        
        auto promise = std::make_shared<std::promise<bool>>();
        auto future = promise->get_future();
          bot->messages_get(channel_id, 0, state.last_message_id, 0, MESSAGES_PER_FETCH,
            [this, channel_id, promise](const dpp::confirmation_callback_t& callback) {
                bool success = false;
                
                try {
                    if (callback.is_error()) {
                        DISCORD_HISTORY_LOG("Error fetching messages for channel " + std::to_string(channel_id));
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
                    DISCORD_HISTORY_LOG("Exception processing channel " + std::to_string(channel_id) + ": " + e.what());
                    promise->set_value(false);
                }
            });
        
        // Wait for result with timeout
        auto status = future.wait_for(std::chrono::seconds(TIMEOUT_SECONDS));
        if (status == std::future_status::ready) {
            return future.get();
        } else {
            std::lock_guard<std::mutex> lock(state_mutex);
            channel_states[channel_id].fetch_complete = true;
            return false;
        }
    }
    
    // Collect messages with exact tokenization using LlamaManager
    bool collect_messages_for_later(const dpp::message_map& messages, uint64_t channel_id) {
        if (messages.empty()) return false;
        
        auto& state = channel_states[channel_id];        // Process messages with exact tokenization
        std::vector<PendingMessage> batch_messages;
        uint64_t oldest_id = UINT64_MAX; // Track oldest message ID for next iteration
        int32_t total_exact_tokens = 0;
        int32_t new_messages_count = 0;
        int32_t duplicate_count = 0;
          for (const auto& [id, msg] : messages) {
            uint64_t msg_id = static_cast<uint64_t>(msg.id);
            
            // Track the oldest message ID for pagination
            oldest_id = std::min(oldest_id, msg_id);
            
            // Check for duplicate messages (safety net - should be rare with correct API usage)
            {
                std::lock_guard<std::mutex> lock(processed_ids_mutex);
                if (processed_message_ids.count(msg_id)) {
                    duplicate_count++;
                    DISCORD_HISTORY_LOG("Skipping duplicate message ID: " + std::to_string(msg_id));
                    continue;
                }
            }
            
            new_messages_count++;
            
            // Check if this is a user message OR our bot's own message
            bool is_our_bot = msg.author.is_bot() && (static_cast<uint64_t>(msg.author.id) == bot_user_id);
            
            // For bot messages, allow empty content if there are embeds; for user messages, content must be non-empty
            bool has_content = !msg.content.empty();
            bool has_embed_content = is_our_bot && !msg.embeds.empty() && !msg.embeds[0].description.empty();
            bool should_process = (!msg.author.is_bot() || is_our_bot) && (has_content || has_embed_content);
            
            if (should_process) {
                PendingMessage pending;
                pending.channel_id = channel_id;
                
                // Use "assistant" role for our bot's messages, username for user messages
                if (is_our_bot) {
                    pending.username = "assistant";
                } else {
                    pending.username = msg.author.username;
                }
                
                // For bot messages, extract content from embeds if they exist, otherwise use regular content
                std::string actual_content = msg.content;
                if (is_our_bot && !msg.embeds.empty()) {
                    // Extract content from the first embed's description
                    const auto& embed = msg.embeds[0];
                    if (!embed.description.empty()) {
                        actual_content = embed.description;
                    }
                }                pending.content = safe_trim(actual_content);
                pending.timestamp = std::chrono::system_clock::time_point(std::chrono::seconds(msg.sent));
                pending.message_id = msg_id;
                pending.target_context_id = state.context_id;
                
                // Use LlamaManager's context-specific tokenization method
                std::string formatted_message;
                if (is_our_bot) {
                    // For bot messages, don't include username prefix since it's "assistant"
                    formatted_message = actual_content;
                } else {
                    formatted_message = msg.author.username + ": " + actual_content;
                }
                
                // Get the specific context for tokenization to ensure consistency
                ContextInfo* target_context = llama_manager->get_context_info(state.context_id);
                if (target_context) {
                    pending.tokenized_content = llama_manager->process_text_to_tokens(formatted_message, target_context, false);
                } else {
                    DISCORD_HISTORY_LOG("Warning: Could not find context " + state.context_id + " for tokenization, skipping message");
                    continue; // Skip this message if we can't find the appropriate context
                }
                
                // Check for reasonable token count limits
                if (pending.tokenized_content.size() > MAX_MESSAGE_TOKENS) {
                    DISCORD_HISTORY_LOG("Warning: Message from " + (is_our_bot ? "bot" : msg.author.username) + " would produce " + 
                               std::to_string(pending.tokenized_content.size()) + " tokens, skipping");
                    continue;}
                
                pending.actual_token_count = static_cast<int32_t>(pending.tokenized_content.size());
                
                if (pending.actual_token_count > 0) {
                    batch_messages.push_back(pending);
                    total_exact_tokens += pending.actual_token_count;
                    
                    // Mark message as processed to prevent duplicates
                    {
                        std::lock_guard<std::mutex> lock(processed_ids_mutex);
                        processed_message_ids.insert(msg_id);
                    }
                } else {
                    DISCORD_HISTORY_LOG("Warning: Tokenization produced 0 tokens for message from " + 
                               (is_our_bot ? "bot" : msg.author.username) + ", skipping");
                }
            }
            // Note: we don't update any tracking for non-processable messages (other bots)
            // This ensures pagination is only based on messages we actually care about
        }
        
        DISCORD_HISTORY_LOG("Batch analysis - Channel " + std::to_string(channel_id) + 
                   ": " + std::to_string(new_messages_count) + " new, " + 
                   std::to_string(duplicate_count) + " duplicates, " + 
                   std::to_string(batch_messages.size()) + " processable");
        
        // Determine how to handle pagination based on what we found
        bool should_continue_pagination = false;
        
        // Check if we can add all messages using exact token counts
        if (!batch_messages.empty() && can_add_more_messages_estimated_no_switch(state.context_id, total_exact_tokens)) {
            // Add to pending collection
            {
                std::lock_guard<std::mutex> lock(pending_messages_mutex);
                pending_messages.insert(pending_messages.end(), batch_messages.begin(), batch_messages.end());
            }
            
            // Update exact token usage
            update_estimated_token_usage(state.context_id, total_exact_tokens);
              // Update state with success
            {
                std::lock_guard<std::mutex> lock(state_mutex);
                state.messages_fetched += static_cast<int32_t>(batch_messages.size());
                  // Update pagination to continue from oldest message in this batch
                if (oldest_id != UINT64_MAX) {
                    state.last_message_id = oldest_id;
                    should_continue_pagination = true;
                }
            }
            
            if (should_continue_pagination) {
                DISCORD_HISTORY_LOG("Collected " + std::to_string(batch_messages.size()) + " messages from channel " + 
                           std::to_string(channel_id) + " (exact " + std::to_string(total_exact_tokens) + 
                           " tokens), next fetch before ID: " + std::to_string(oldest_id));
            }
            return should_continue_pagination;        } else if (!batch_messages.empty()) {
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
            
            DISCORD_HISTORY_LOG("Channel " + std::to_string(channel_id) + " completed - context capacity reached (" + capacity_info + ")");
            return false;
        } else {
            // No processable messages found in this batch
            std::lock_guard<std::mutex> lock(state_mutex);
            
            if (new_messages_count == 0) {
                // No new messages - likely reached the end of available messages
                state.fetch_complete = true;
                DISCORD_HISTORY_LOG("Channel " + std::to_string(channel_id) + " completed - no new messages found");
                return false;
            } else {
                // We had new messages but none were processable (all bots, etc.)                // Continue pagination to find processable messages
                if (oldest_id != UINT64_MAX) {
                    state.last_message_id = oldest_id;
                    DISCORD_HISTORY_LOG("No processable messages for channel " + std::to_string(channel_id) + 
                               ", continuing pagination from " + std::to_string(oldest_id));
                    return true;
                } else {
                    // Shouldn't happen but handle gracefully
                    state.fetch_complete = true;
                    DISCORD_HISTORY_LOG("Channel " + std::to_string(channel_id) + " completed - unable to determine pagination");
                    return false;
                }
            }
        }
    }
    
    // Apply collected messages using pre-tokenized content
    void apply_collected_messages() {
        std::lock_guard<std::mutex> lock(pending_messages_mutex);
        if (pending_messages.empty()) return;
        
        DISCORD_HISTORY_LOG("Applying " + std::to_string(pending_messages.size()) + " collected tokenized messages to contexts...");
        
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
        
        // Apply messages to each context using direct context access instead of switching
        for (const auto& [context_id, messages] : context_groups) {
            DISCORD_HISTORY_LOG("Adding " + std::to_string(messages.size()) + " tokenized messages to context '" + context_id + "' directly");
            
            ContextInfo* target_context = llama_manager->get_context_info(context_id);
            if (target_context) {
                int32_t added_count = 0;
                int32_t total_tokens_added = 0;
                
                for (const auto& msg : messages) {
                    // Use the context-specific overload to avoid context switching
                    DISCORD_HISTORY_LOG("Adding message from " + msg.username + " to specified context");
                    target_context->add_message(msg.username, msg.content);
                    added_count++;
                    total_tokens_added += msg.actual_token_count;
                }
                
                if (llama_manager->update_context_from_history(target_context)) {
                    DISCORD_HISTORY_LOG("Successfully added " + std::to_string(added_count) + " messages (" + 
                               std::to_string(total_tokens_added) + " tokens) to context " + context_id);
                } else {
                    DISCORD_HISTORY_LOG("Failed to update context " + context_id + " from history");
                }
            } else {
                DISCORD_HISTORY_LOG("Failed to get context info for " + context_id);
            }
        }
        
        // No need to restore context since we didn't switch
        
        // Clear processed messages
        pending_messages.clear();
        DISCORD_HISTORY_LOG("Tokenized message application phase completed");
    }
    
    // Round-robin for shared channels
    uint64_t get_next_shared_channel() {
        if (shared_channels_list.empty()) return 0;
        
        uint64_t channel_id = shared_channels_list[shared_channel_index];
        shared_channel_index = (shared_channel_index + 1) % shared_channels_list.size();
        return channel_id;
    }
    
    // Updated main processing loop with proper isolated vs shared handling
    void process_all_channels() {
        const int32_t MAX_ITERATIONS = DiscordHistoryLoader::MAX_ITERATIONS;
        int32_t iteration = 0;
        
        // Initialize context capacities once
        DISCORD_HISTORY_LOG("Initializing context capacity tracking...");
        initialize_context_capacities();
        
        // Phase 1: Collect all messages
        DISCORD_HISTORY_LOG("Phase 1: Collecting messages from all channels...");
        
        while (backfill_in_progress && iteration < MAX_ITERATIONS) {
            bool made_progress = false;
              // Process isolated channels sequentially to completion using STL algorithms (Directive #14)
            // Since each has its own context, we can fill them completely without affecting others
            std::vector<uint64_t> isolated_channels_to_process;
            {
                std::lock_guard<std::mutex> lock(state_mutex);
                std::for_each(channel_states.begin(), channel_states.end(), [&](const auto& pair) {
                    const auto& [channel_id, state] = pair;
                    if (state.is_isolated && !state.fetch_complete) {
                        isolated_channels_to_process.push_back(channel_id);
                    }
                });
            }
            
            // Process all isolated channels until complete or no progress
            for (uint64_t channel_id : isolated_channels_to_process) {
                auto it = channel_states.find(channel_id);
                if (it != channel_states.end() && !it->second.fetch_complete) {
                    DISCORD_HISTORY_LOG("Processing isolated channel " + std::to_string(channel_id) + " to completion...");
                    
                    // Keep processing this isolated channel until it's done
                    while (!it->second.fetch_complete && backfill_in_progress) {
                        if (process_channel_batch(channel_id)) {
                            made_progress = true;
                            // Small delay between batches for rate limiting
                            std::this_thread::sleep_for(std::chrono::milliseconds(RATE_LIMIT_DELAY_MS));
                        } else {
                            break; // Channel completed or error
                        }
                    }
                    
                    if (it->second.fetch_complete) {
                        DISCORD_HISTORY_LOG("Completed isolated channel " + std::to_string(channel_id));
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
                        DISCORD_HISTORY_LOG("Processed batch for shared channel " + std::to_string(channel_id) + 
                                   " (round-robin)");
                    }
                }
            }
            
            iteration++;
            
            if (!made_progress) {                // Check if all channels are complete using STL algorithms (Directive #14)
                bool all_complete = true;
                {
                    std::lock_guard<std::mutex> lock(state_mutex);
                    all_complete = std::all_of(channel_states.begin(), channel_states.end(),
                        [](const auto& pair) { return pair.second.fetch_complete; });
                }
                
                if (all_complete) {
                    DISCORD_HISTORY_LOG("All channels completed");
                    break;
                }
                
                // If no progress but channels remain, wait and try again
                DISCORD_HISTORY_LOG("No progress made, waiting before retry...");
                std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_DELAY_MS));
            } else {
                // Short delay between iterations when making progress
                std::this_thread::sleep_for(std::chrono::milliseconds(PROGRESS_DELAY_MS));
            }
        }
        
        // Phase 2: Apply all collected messages
        DISCORD_HISTORY_LOG("Phase 2: Applying collected messages to contexts...");
        apply_collected_messages();
        
        backfill_in_progress = false;
        DISCORD_HISTORY_LOG("Chat history backfill completed");
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
      void configure(dpp::cluster* discord_bot, LlamaManager* llama_mgr, const std::string& main_ctx_id, const std::string& model_identifier = "") {
        bot = discord_bot;
        llama_manager = llama_mgr;
        main_context_id = main_ctx_id;
        model_id = model_identifier;
    }
    
    // Set bot identification information for recognizing own messages
    void set_bot_identity(uint64_t user_id, const std::string& username) {
        bot_user_id = user_id;
        bot_username = username;
        DISCORD_HISTORY_LOG("Bot identity set: ID=" + std::to_string(user_id) + ", Username=" + username);
    }
    
    void set_channel_configuration(const std::unordered_set<uint64_t>* allowed,
                                  const std::unordered_set<uint64_t>* isolated,
                                  const std::unordered_set<uint64_t>* shared_history) {
        isolated_channels = isolated;
        shared_history_channels = shared_history;
    }
      void set_history_settings(bool enabled, int32_t fill_percentage) {
        history_enabled = enabled;
        context_fill_percentage = std::clamp(fill_percentage, MIN_FILL_PERCENTAGE, MAX_FILL_PERCENTAGE);
    }
      void start_backfill() {
        if (!history_enabled || backfill_in_progress || !bot || !llama_manager) return;
        
        backfill_in_progress = true;
        
        // Clear processed message IDs for new backfill session
        {
            std::lock_guard<std::mutex> lock(processed_ids_mutex);
            processed_message_ids.clear();
        }
        
        DISCORD_HISTORY_LOG("Starting simplified chat history backfill...");
        
        // Get channels and setup states
        bot->current_user_get_guilds([this](const dpp::confirmation_callback_t& callback) {
            if (callback.is_error()) {
                DISCORD_HISTORY_LOG("Error getting guilds: " + callback.get_error().human_readable);
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
        
        // Clear pending messages and processed IDs for fresh start
        {
            std::lock_guard<std::mutex> pmsg_lock(pending_messages_mutex);
            pending_messages.clear();
        }
        
        // Validate prerequisites before proceeding
        if (model_id.empty()) {
            DISCORD_HISTORY_LOG("Error: No model_id configured for context creation");
            backfill_in_progress = false;
            return;
        }
        
        if (!llama_manager) {
            DISCORD_HISTORY_LOG("Error: No LlamaManager configured");
            backfill_in_progress = false;
            return;
        }
        
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
                                    
                                    // Validate model_id and create context with proper system prompt
                                    if (!model_id.empty()) {
                                        if (!llama_manager->has_context(state.context_id)) {
                                            // Get the main shared context info to retrieve system prompt
                                            auto context_info = llama_manager->get_context_info(main_context_id);

                                            // Get system prompt from main context
                                            std::string system_prompt = context_info->system_message;
                                            
                                            if (llama_manager->create_context(state.context_id, model_id, system_prompt)) {
                                                DISCORD_HISTORY_LOG("Created isolated context '" + state.context_id + 
                                                           "' with model '" + model_id + "' and system prompt");
                                            } else {
                                                DISCORD_HISTORY_LOG("Error: Failed to create isolated context '" + 
                                                           state.context_id + "' with model '" + model_id + "'");
                                                continue; // Skip this channel if context creation failed
                                            }
                                        } else {
                                            DISCORD_HISTORY_LOG("Using existing isolated context '" + state.context_id + "'");
                                        }
                                    } else {
                                        DISCORD_HISTORY_LOG("Error: Cannot create isolated context - no model_id");
                                        continue; // Skip this channel
                                    }
                                } else {
                                    state.context_id = main_context_id;
                                    shared_channels_list.push_back(channel_id);
                                    
                                    // Validate main context exists
                                    if (!llama_manager->has_context(main_context_id)) {
                                        DISCORD_HISTORY_LOG("Error: Main context '" + main_context_id + "' does not exist");
                                        continue; // Skip this channel
                                    }
                                }
                                
                                channel_states[channel_id] = state;
                                DISCORD_HISTORY_LOG("Configured channel " + std::to_string(channel_id) + 
                                           " for " + (state.is_isolated ? "isolated" : "shared") + " context: " + state.context_id);
                            }
                        }
                    } else {
                        DISCORD_HISTORY_LOG("Error getting channels for guild " + std::to_string(guild_id) + 
                                   ": " + callback.get_error().human_readable);
                    }
                });
            }
            
            // Start processing after a short delay
            std::thread([this]() {
                std::this_thread::sleep_for(std::chrono::seconds(INITIALIZATION_DELAY_SEC));
                if (backfill_in_progress) {
                    // Validate we have channels to process before starting
                    {
                        std::lock_guard<std::mutex> lock(state_mutex);
                        if (channel_states.empty()) {
                            DISCORD_HISTORY_LOG("No channels configured for backfill, stopping");
                            backfill_in_progress = false;
                            return;
                        }
                        DISCORD_HISTORY_LOG("Starting backfill for " + std::to_string(channel_states.size()) + " channels");
                    }
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