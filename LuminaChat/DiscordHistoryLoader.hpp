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
#include <functional>
#include <thread>
#include <algorithm>
#include <queue>
#include <condition_variable>
#include <future>

#include <dpp/dpp.h>
#include "LogHandler.hpp"

// Forward declarations
class LlamaManager;

// History message structure
struct HistoryMessage {
    uint64_t message_id;
    uint64_t user_id;
    uint64_t channel_id;
    std::string username;
    std::string content;
    std::chrono::system_clock::time_point timestamp;
};

// Worker-specific channel state
struct WorkerChannelState {
    uint64_t last_message_id = 0;
    bool fetch_complete = false;
    int32_t messages_fetched = 0;
    int32_t estimated_tokens_added = 0;
};

// Worker context information - FIXED: Make copyable for status reporting
struct WorkerContextInfo {
    std::string context_id;
    int32_t estimated_tokens = 0;
    int32_t capacity_limit = 0;
    bool is_main_context = false;
    bool worker_active = false;  // CHANGED: Remove atomic for copyability
    bool worker_complete = false; // CHANGED: Remove atomic for copyability
};

// Worker result for context updating
struct WorkerResult {
    std::string context_id;
    std::vector<HistoryMessage> messages;
    bool success = false;
    std::string error_message;
    int32_t total_messages_processed = 0;
    int32_t estimated_tokens_used = 0;
};

class DiscordHistoryLoader {
private:
    // Core dependencies
    dpp::cluster* bot;
    LlamaManager* llama_manager;
    std::string main_context_id;
    
    // Channel configuration
    const std::unordered_set<uint64_t>* isolated_channels;
    const std::unordered_set<uint64_t>* shared_history_channels;
    
    // History settings
    bool history_enabled = true;
    int32_t context_fill_percentage = 50;
    
    // Worker management
    std::vector<std::unique_ptr<class DiscordHistoryWorker>> workers;
    std::unordered_map<std::string, std::unique_ptr<WorkerContextInfo>> context_info_map;
    mutable std::mutex workers_mutex;
    mutable std::mutex context_info_mutex;
    
    // ADDED: Mutex to protect D++ API calls from worker threads
    mutable std::mutex discord_api_mutex;
    
    // Backfill state
    std::atomic<bool> backfill_in_progress{false};
    std::atomic<int32_t> active_workers{0};
    
    // ADDED: Worker numbering for logging
    std::atomic<int32_t> worker_counter{0};
    
    // Constants
    static constexpr int32_t MESSAGES_PER_FETCH = 10;
    static constexpr float BASE_MAX_CONTEXT_FILL_RATIO = 0.01f; // 1% per percentage point

    // Helper to estimate token count
    int32_t estimate_message_tokens(const std::string& username, const std::string& content) const {
        const int32_t base_overhead = 50;
        const int32_t role_overhead = username.length() + 20;
        const int32_t content_tokens = static_cast<int32_t>((content.length() + 2) / 3);
        return base_overhead + role_overhead + content_tokens;
    }
    
    // Check if channel should have history backfilled
    bool should_backfill_channel(uint64_t channel_id) const {
        // Always backfill isolated channels
        if (isolated_channels && isolated_channels->find(channel_id) != isolated_channels->end()) {
            return true;
        }
        
        // For shared channels, only backfill if explicitly listed
        return shared_history_channels && 
               shared_history_channels->find(channel_id) != shared_history_channels->end();
    }
    
    bool is_isolated_channel(uint64_t channel_id) const {
        return isolated_channels && isolated_channels->find(channel_id) != isolated_channels->end();
    }
    
    void log_message(const std::string& message) const {
        DISCORD_HISTORY_LOG(message);
    }
    
    // Initialize context info for worker
    std::unique_ptr<WorkerContextInfo> create_context_info(const std::string& context_id) {
        auto info = std::make_unique<WorkerContextInfo>();
        info->context_id = context_id;
        info->is_main_context = (context_id == main_context_id);
        
        if (llama_manager) {
            int32_t context_size = llama_manager->get_context_size();
            float fill_ratio = BASE_MAX_CONTEXT_FILL_RATIO * context_fill_percentage;
            info->capacity_limit = static_cast<int32_t>(context_size * fill_ratio);
            
            // Get actual current usage for this context
            std::string original_context = llama_manager->get_active_context();
            if (llama_manager->switch_to_context(context_id)) {
                info->estimated_tokens = llama_manager->get_message_history_token_count();
                if (!original_context.empty()) {
                    llama_manager->switch_to_context(original_context);
                }
            } else {
                info->estimated_tokens = 0;
            }
        } else {
            float fill_ratio = BASE_MAX_CONTEXT_FILL_RATIO * context_fill_percentage;
            info->capacity_limit = static_cast<int32_t>(2048 * fill_ratio);
            info->estimated_tokens = 0;
        }
        
        return info;
    }
    
    // Create worker for isolated context (1 channel per worker)
    std::unique_ptr<DiscordHistoryWorker> create_isolated_worker(uint64_t channel_id) {
        std::string context_id = "discord_channel_" + std::to_string(channel_id);
        auto context_info = create_context_info(context_id);
        
        log_message("Creating isolated worker for channel " + std::to_string(channel_id) + 
                   " with context '" + context_id + "' (limit: " + 
                   std::to_string(context_info->capacity_limit) + " tokens)");
        
        {
            std::lock_guard<std::mutex> lock(context_info_mutex);
            context_info_map[context_id] = std::move(context_info);
        }
        
        // ADDED: Assign worker number
        int32_t worker_number = worker_counter.fetch_add(1) + 1;
        
        return std::make_unique<DiscordHistoryWorker>(
            bot, llama_manager, context_id, std::vector<uint64_t>{channel_id}, 
            false, context_fill_percentage, &discord_api_mutex, worker_number
        );
    }
    
    // Create worker for shared context (multiple channels, round-robin)
    std::unique_ptr<DiscordHistoryWorker> create_shared_worker() {
        if (main_context_id.empty()) {
            log_message("Error: Main context ID not set for shared worker");
            return nullptr;
        }
        
        // Collect all shared channels
        std::vector<uint64_t> shared_channels;
        if (shared_history_channels) {
            for (uint64_t channel_id : *shared_history_channels) {
                shared_channels.push_back(channel_id);
            }
        }
        
        if (shared_channels.empty()) {
            log_message("No shared channels configured for shared worker");
            return nullptr;
        }
        
        auto context_info = create_context_info(main_context_id);
        
        log_message("Creating shared worker for " + std::to_string(shared_channels.size()) + 
                   " channels with context '" + main_context_id + "' (limit: " + 
                   std::to_string(context_info->capacity_limit) + " tokens)");
        
        {
            std::lock_guard<std::mutex> lock(context_info_mutex);
            context_info_map[main_context_id] = std::move(context_info);
        }
        
        // ADDED: Assign worker number
        int32_t worker_number = worker_counter.fetch_add(1) + 1;
        
        return std::make_unique<DiscordHistoryWorker>(
            bot, llama_manager, main_context_id, shared_channels, 
            true, context_fill_percentage, &discord_api_mutex, worker_number
        );
    }
    
    // Process worker completion
    void handle_worker_completion(const std::string& context_id, const WorkerResult& result) {
        {
            std::lock_guard<std::mutex> lock(context_info_mutex);
            auto it = context_info_map.find(context_id);
            if (it != context_info_map.end()) {
                it->second->worker_complete = true;
                it->second->worker_active = false;
            }
        }
        
        // FIXED: Use fetch_sub to ensure proper atomic decrement
        int32_t remaining = active_workers.fetch_sub(1) - 1;
        
        if (result.success) {
            log_message("Worker completed successfully for context '" + context_id + 
                       "': " + std::to_string(result.total_messages_processed) + 
                       " messages, " + std::to_string(result.estimated_tokens_used) + " tokens");
        } else {
            log_message("Worker failed for context '" + context_id + "': " + result.error_message);
        }
        
        // FIXED: Check completion based on atomic counter, not active_workers <= 0
        if (remaining <= 0) {
            check_backfill_completion();
        }
    }
    
    // Check if backfill is complete
    void check_backfill_completion() {
        backfill_in_progress = false;
        
        // Report final usage per context
        std::lock_guard<std::mutex> lock(context_info_mutex);
        for (const auto& [context_id, info] : context_info_map) {
            // Get actual final usage
            int32_t actual_usage = info->estimated_tokens;
            if (llama_manager) {
                std::string original_context = llama_manager->get_active_context();
                if (llama_manager->switch_to_context(context_id)) {
                    actual_usage = llama_manager->get_message_history_token_count();
                    if (!original_context.empty()) {
                        llama_manager->switch_to_context(original_context);
                    }
                }
            }
            
            float usage_percent = (float)actual_usage / info->capacity_limit * 100.0f;
            float target_percent = BASE_MAX_CONTEXT_FILL_RATIO * context_fill_percentage * 100.0f;
            
            log_message("Final backfill for context '" + context_id + "': " +
                       std::to_string(actual_usage) + "/" +
                       std::to_string(info->capacity_limit) + " tokens (" +
                       std::to_string(usage_percent) + "% of " + std::to_string(target_percent) + "% target)");
        }
        
        log_message("Chat history backfill completed for all contexts");
    }
    
    // Process guilds to create workers
    void process_guilds_for_backfill(const dpp::guild_map& guilds) {
        // Use shared pointers for thread-safe async processing
        auto isolated_channels_shared = std::make_shared<std::vector<uint64_t>>();
        auto shared_channels_shared = std::make_shared<std::vector<uint64_t>>();
        auto guilds_processed = std::make_shared<std::atomic<int32_t>>(0);
        int32_t total_guilds = static_cast<int32_t>(guilds.size());
        
        if (total_guilds == 0) {
            log_message("No guilds found for backfill");
            backfill_in_progress = false;
            return;
        }
        
        for (const auto& [guild_id, guild] : guilds) {
            // PROTECTED: Get channels for this guild with mutex protection
            {
                std::lock_guard<std::mutex> lock(discord_api_mutex);
                bot->channels_get(guild_id, [this, guild_id, isolated_channels_shared, shared_channels_shared, guilds_processed, total_guilds](const dpp::confirmation_callback_t& callback) {
                    if (callback.is_error()) {
                        log_message("Error getting channels for guild " + std::to_string(guild_id));
                    } else {
                        auto channels = callback.get<dpp::channel_map>();
                        for (const auto& [channel_id, channel] : channels) {
                            if (channel.is_text_channel() && should_backfill_channel(channel_id)) {
                                if (is_isolated_channel(channel_id)) {
                                    isolated_channels_shared->push_back(channel_id);
                                } else {
                                    shared_channels_shared->push_back(channel_id);
                                }
                            }
                        }
                    }
                    
                    // Check if we've processed all guilds
                    int32_t processed = guilds_processed->fetch_add(1) + 1;
                    if (processed >= total_guilds) {
                        // Once we've processed all guilds, create workers
                        create_all_workers(*isolated_channels_shared, !shared_channels_shared->empty());
                    }
                });
            }
        }
    }
    
    // Create all workers based on channel configuration - FIXED: Defer thread creation
    void create_all_workers(const std::vector<uint64_t>& isolated_channel_list, bool has_shared_channels) {
        std::lock_guard<std::mutex> lock(workers_mutex);
        workers.clear();
        
        // Create workers for isolated channels (1 worker per channel)
        for (uint64_t channel_id : isolated_channel_list) {
            auto worker = create_isolated_worker(channel_id);
            if (worker) {
                workers.push_back(std::move(worker));
            }
        }
        
        // Create worker for shared channels (1 worker for all shared channels)
        if (has_shared_channels) {
            auto worker = create_shared_worker();
            if (worker) {
                workers.push_back(std::move(worker));
            }
        }
        
        if (workers.empty()) {
            log_message("No workers created - no configured channels found");
            backfill_in_progress = false;
            return;
        }
        
        log_message("Created " + std::to_string(workers.size()) + " workers for backfill");
        
        // FIXED: Set active workers count before starting threads
        active_workers.store(static_cast<int32_t>(workers.size()));
        
        // Mark all workers as active in context info (without accessing worker methods)
        {
            std::lock_guard<std::mutex> info_lock(context_info_mutex);
            for (const auto& [context_id, info] : context_info_map) {
                info->worker_active = true;
                info->worker_complete = false; // ADDED: Ensure clean state
            }
        }
        
        // Start all workers using a deferred approach
        start_all_workers();
    }

    // Helper method to start all workers (defined after DiscordHistoryWorker)
    void start_all_workers();

public:
    // Backfill status for reporting
    struct BackfillStatus {
        bool in_progress;
        int32_t active_workers;
        int32_t total_workers;
        std::unordered_map<std::string, WorkerContextInfo> context_usage;
    };

    DiscordHistoryLoader() = default;
    ~DiscordHistoryLoader() = default;
    
    // Configuration
    void configure(dpp::cluster* discord_bot, LlamaManager* llama_mgr, const std::string& main_ctx_id) {
        bot = discord_bot;
        llama_manager = llama_mgr;
        main_context_id = main_ctx_id;
    }
    
    void set_channel_configuration(const std::unordered_set<uint64_t>* allowed,
                                  const std::unordered_set<uint64_t>* isolated,
                                  const std::unordered_set<uint64_t>* shared_history) {
        // Ignore allowed parameter (all channels are now allowed)
        isolated_channels = isolated;
        shared_history_channels = shared_history;
    }
    
    // Set history settings
    void set_history_settings(bool enabled, int32_t fill_percentage) {
        history_enabled = enabled;
        context_fill_percentage = std::clamp(fill_percentage, 10, 80);
        log_message("History loader settings updated: Enabled=" + std::string(enabled ? "true" : "false") + 
                   ", Fill=" + std::to_string(context_fill_percentage) + "%");
    }
    
    // Main backfill control
    void start_backfill() {
        if (!history_enabled) {
            log_message("Message history backfill is disabled in settings");
            return;
        }
        
        if (backfill_in_progress || !bot || !llama_manager) {
            return;
        }
        
        backfill_in_progress = true;
        active_workers = 0;
        worker_counter = 0; // ADDED: Reset worker counter
        
        log_message("Starting chat history backfill with " + std::to_string(context_fill_percentage) + "% context fill target...");
        
        {
            std::lock_guard<std::mutex> lock(workers_mutex);
            workers.clear();
        }
        
        {
            std::lock_guard<std::mutex> lock(context_info_mutex);
            context_info_map.clear();
        }
        
        // PROTECTED: Get list of guilds the bot has access to with mutex protection
        try {
            std::lock_guard<std::mutex> lock(discord_api_mutex);
            bot->current_user_get_guilds([this](const dpp::confirmation_callback_t& callback) {
                if (callback.is_error()) {
                    log_message("Error getting guilds for backfill: " + callback.get_error().human_readable);
                    backfill_in_progress = false;
                    return;
                }
                
                auto guilds = callback.get<dpp::guild_map>();
                
                if (guilds.empty()) {
                    log_message("Bot is not in any guilds - only DM backfill possible");
                    backfill_in_progress = false;
                    return;
                }
                
                log_message("Found " + std::to_string(guilds.size()) + " guilds, creating workers...");
                process_guilds_for_backfill(guilds);
            });
        } catch (const std::exception& e) {
            log_message("Exception starting backfill: " + std::string(e.what()));
            backfill_in_progress = false;
        }
    }
    
    // Status reporting - FIXED: Copy context info properly
    BackfillStatus get_status() const {
        std::lock_guard<std::mutex> workers_lock(workers_mutex);
        std::lock_guard<std::mutex> info_lock(context_info_mutex);
        
        BackfillStatus status;
        status.in_progress = backfill_in_progress;
        status.active_workers = active_workers;
        status.total_workers = 0; // Workers vector is cleared after starting threads
        
        // Copy context usage information - FIXED: Proper dereferencing
        for (const auto& [context_id, info_ptr] : context_info_map) {
            if (info_ptr) {
                status.context_usage[context_id] = *info_ptr;
            }
        }
        
        return status;
    }
    
    bool is_in_progress() const {
        return backfill_in_progress;
    }
};

// DiscordHistoryWorker class definition - MOVED AFTER DiscordHistoryLoader
class DiscordHistoryWorker {
private:
    dpp::cluster* bot;
    LlamaManager* llama_manager;
    std::string context_id;
    std::vector<uint64_t> assigned_channels;
    bool is_shared_worker;
    int32_t context_fill_percentage;
    
    // ADDED: Reference to Discord API mutex from parent loader
    std::mutex* discord_api_mutex;
    
    // Worker state
    std::unordered_map<uint64_t, WorkerChannelState> channel_states;
    int32_t current_channel_index = 0; // For round-robin in shared worker
    std::atomic<bool> should_stop{false};
    
    // Safety mechanisms
    int32_t max_iterations_per_channel = 50; // Prevent infinite loops
    
    // ADDED: Worker identification for logging
    int32_t worker_number;
    
    // ADDED: Message processing statistics per channel
    struct ChannelStats {
        int32_t messages_processed = 0;
        int32_t messages_skipped = 0;
        uint64_t last_message_id = 0;
    };
    std::unordered_map<uint64_t, ChannelStats> channel_stats;
    
    // Constants
    static constexpr int32_t MESSAGES_PER_FETCH = 10;
    static constexpr float BASE_MAX_CONTEXT_FILL_RATIO = 0.01f;
    
    void log_message(const std::string& message) const {
        DISCORD_HISTORY_LOG("[Worker#" + std::to_string(worker_number) + ":" + context_id + "] " + message);
    }
    
    // FIXED: Log worker progress for a specific channel with more detail
    void log_worker_progress(uint64_t channel_id) const {
        auto stats_it = channel_stats.find(channel_id);
        if (stats_it != channel_stats.end()) {
            const auto& stats = stats_it->second;
            DISCORD_HISTORY_LOG("Worker #" + std::to_string(worker_number) + 
                               " Progress for Context #" + context_id + 
                               " Polling Channel #" + std::to_string(channel_id) + 
                               ": Processed " + std::to_string(stats.messages_processed) + 
                               " messages, Skipped " + std::to_string(stats.messages_skipped) + 
                               " messages, Last Message ID: " + std::to_string(stats.last_message_id));
        }
    }
    
    // Helper to estimate token count
    int32_t estimate_message_tokens(const std::string& username, const std::string& content) const {
        const int32_t base_overhead = 50;
        const int32_t role_overhead = username.length() + 20;
        const int32_t content_tokens = static_cast<int32_t>((content.length() + 2) / 3);
        return base_overhead + role_overhead + content_tokens;
    }
    
    // FIXED: Enhanced fetch with immediate progress logging
    std::future<std::vector<HistoryMessage>> fetch_channel_messages(uint64_t channel_id, uint64_t before_message_id = 0) {
        auto promise = std::make_shared<std::promise<std::vector<HistoryMessage>>>();
        auto future = promise->get_future();
        
        if (!bot || !discord_api_mutex) {
            promise->set_value({});
            return future;
        }
        
        // ADDED: Log the fetch attempt
        log_message("Fetching messages from channel " + std::to_string(channel_id) + 
                   " before message ID " + std::to_string(before_message_id));
        
        try {
            std::lock_guard<std::mutex> lock(*discord_api_mutex);
            bot->messages_get(channel_id, before_message_id, 0, 0, MESSAGES_PER_FETCH,
                [this, channel_id, promise](const dpp::confirmation_callback_t& callback) {
                    std::vector<HistoryMessage> messages;
                    
                    try {
                        if (callback.is_error()) {
                            log_message("Error fetching history for channel " + std::to_string(channel_id) + 
                                       ": " + callback.get_error().human_readable);
                            promise->set_value(messages);
                            return;
                        }
                        
                        auto dpp_messages = callback.get<dpp::message_map>();
                        log_message("Retrieved " + std::to_string(dpp_messages.size()) + " raw messages from channel " + std::to_string(channel_id));
                        
                        // Track statistics for this fetch
                        int32_t messages_processed = 0;
                        int32_t messages_skipped = 0;
                        uint64_t last_message_id = 0;
                        
                        // Convert D++ messages to our format, filtering out bots
                        for (const auto& [id, msg] : dpp_messages) {
                            if (msg.id > last_message_id) {
                                last_message_id = msg.id;
                            }
                            
                            if (msg.author.is_bot() || msg.content.empty()) {
                                messages_skipped++;
                                continue;
                            }
                            
                            HistoryMessage hist_msg;
                            hist_msg.message_id = msg.id;
                            hist_msg.user_id = msg.author.id;
                            hist_msg.channel_id = channel_id;
                            hist_msg.username = msg.author.username;
                            hist_msg.content = msg.content;
                            hist_msg.timestamp = std::chrono::system_clock::time_point(
                                std::chrono::seconds(msg.sent)
                            );
                            
                            messages.push_back(hist_msg);
                            messages_processed++;
                        }
                        
                        // Update channel statistics
                        auto& stats = channel_stats[channel_id];
                        stats.messages_processed += messages_processed;
                        stats.messages_skipped += messages_skipped;
                        if (last_message_id > 0) {
                            stats.last_message_id = last_message_id;
                        }
                        
                        // ADDED: Log processing results immediately
                        log_message("Channel " + std::to_string(channel_id) + 
                                   " batch: " + std::to_string(messages_processed) + " processed, " + 
                                   std::to_string(messages_skipped) + " skipped");
                        
                        // Sort messages by timestamp (oldest first for proper context order)
                        std::sort(messages.begin(), messages.end(),
                            [](const HistoryMessage& a, const HistoryMessage& b) {
                                return a.timestamp < b.timestamp;
                            });
                        
                        promise->set_value(messages);
                    } catch (const std::exception& e) {
                        log_message("Exception in fetch callback for channel " + std::to_string(channel_id) + ": " + std::string(e.what()));
                        promise->set_value({});
                    }
                });
        } catch (const std::exception& e) {
            log_message("Exception in fetch_channel_messages for channel " + std::to_string(channel_id) + ": " + std::string(e.what()));
            promise->set_value({});
        }
        
        return future;
    }
    
    // Add messages to context and update state
    bool add_messages_to_context(const std::vector<HistoryMessage>& messages, uint64_t channel_id) {
        if (!llama_manager || messages.empty()) {
            return false;
        }
        
        // FIXED: Check if context exists, create if missing with retry
        bool context_created = false;
        if (!llama_manager->has_context(context_id)) {
            if (context_id.find("discord_channel_") == 0) {
                log_message("Context '" + context_id + "' not found, creating for history backfill...");
                
                int32_t retry_count = 0;
                while (retry_count < 3 && !llama_manager->create_context(context_id, "")) {
                    retry_count++;
                    log_message("Failed to create context '" + context_id + "', retry " + std::to_string(retry_count) + "/3");
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                }
                
                if (retry_count >= 3) {
                    log_message("Failed to create context '" + context_id + "' after 3 retries");
                    return false;
                }
                
                context_created = true;
                log_message("Successfully created context '" + context_id + "' for history backfill");
            } else {
                log_message("Shared context '" + context_id + "' not available for history backfill");
                return false;
            }
        }
        
        // FIXED: Switch to context with retry mechanism and validation
        std::string original_context = llama_manager->get_active_context();
        int32_t switch_retry_count = 0;
        bool switch_success = false;
        
        while (switch_retry_count < 3 && !switch_success) {
            switch_success = llama_manager->switch_to_context(context_id);
            if (!switch_success) {
                switch_retry_count++;
                log_message("Failed to switch to context '" + context_id + "', retry " + std::to_string(switch_retry_count) + "/3");
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        }
        
        if (!switch_success) {
            log_message("Failed to switch to context '" + context_id + "' after 3 retries");
            return false;
        }
        
        // Verify we're in the correct context
        if (llama_manager->get_active_context() != context_id) {
            log_message("Context switch verification failed - expected '" + context_id + "', got '" + llama_manager->get_active_context() + "'");
            return false;
        }
        
        // Get token count before adding messages
        int32_t tokens_before = llama_manager->get_message_history_token_count();
        
        // Add messages as conversation history
        for (const auto& msg : messages) {
            llama_manager->add_message_to_history(msg.username, msg.content);
        }
        
        // FIXED: Batch update with error handling and validation
        bool success = false;
        int32_t update_retry_count = 0;
        
        while (update_retry_count < 3 && !success) {
            success = llama_manager->update_context_from_history();
            if (!success) {
                update_retry_count++;
                log_message("Failed to update context from history, retry " + std::to_string(update_retry_count) + "/3");
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
            }
        }
        
        if (success) {
            // Update channel state
            auto& state = channel_states[channel_id];
            state.messages_fetched += static_cast<int32_t>(messages.size());
            if (!messages.empty()) {
                state.last_message_id = messages.back().message_id;
            }
            
            // Validate actual token usage
            int32_t tokens_after = llama_manager->get_message_history_token_count();
            int32_t actual_tokens_added = tokens_after - tokens_before;
            state.estimated_tokens_added += actual_tokens_added;
            
            log_message("Added " + std::to_string(messages.size()) + " messages from channel " + 
                       std::to_string(channel_id) + " (" + std::to_string(actual_tokens_added) + " tokens, total: " + std::to_string(tokens_after) + ")");
        } else {
            log_message("Failed to update context after adding history messages to " + context_id + " after 3 retries");
        }
        
        // FIXED: Restore original context with retry
        if (!original_context.empty()) {
            int32_t restore_retry_count = 0;
            while (restore_retry_count < 3 && !llama_manager->switch_to_context(original_context)) {
                restore_retry_count++;
                log_message("Failed to restore original context '" + original_context + "', retry " + std::to_string(restore_retry_count) + "/3");
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        }
        
        return success;
    }
    
    // FIXED: Enhanced capacity checking with proper context switching
    bool should_continue_fetching() {
        if (!llama_manager) {
            log_message("LlamaManager not available for capacity check");
            return false;
        }
        
        // Get current context usage with proper switching
        std::string original_context = llama_manager->get_active_context();
        int32_t current_usage = 0;
        bool switch_success = false;
        
        // Try to switch to our context
        int32_t retry_count = 0;
        while (retry_count < 3 && !switch_success) {
            switch_success = llama_manager->switch_to_context(context_id);
            if (!switch_success) {
                retry_count++;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        
        if (switch_success) {
            current_usage = llama_manager->get_message_history_token_count();
            
            // Restore original context
            if (!original_context.empty()) {
                llama_manager->switch_to_context(original_context);
            }
        } else {
            log_message("Failed to switch to context '" + context_id + "' for capacity check");
            return false;
        }
        
        // Calculate capacity limit
        int32_t context_size = llama_manager->get_context_size();
        float fill_ratio = BASE_MAX_CONTEXT_FILL_RATIO * context_fill_percentage;
        int32_t capacity_limit = static_cast<int32_t>(context_size * fill_ratio);
        
        bool should_continue = current_usage < capacity_limit;
        
        if (!should_continue) {
            log_message("Context capacity reached: " + std::to_string(current_usage) + "/" + std::to_string(capacity_limit) + " tokens");
        }
        
        return should_continue;
    }

    // Get next channel for round-robin (shared worker only)
    uint64_t get_next_channel() {
        if (assigned_channels.empty()) return 0;
        
        uint64_t channel_id = assigned_channels[current_channel_index];
        current_channel_index = (current_channel_index + 1) % assigned_channels.size();
        return channel_id;
    }

public:
    DiscordHistoryWorker(dpp::cluster* discord_bot, LlamaManager* llama_mgr, 
                        const std::string& ctx_id, const std::vector<uint64_t>& channels,
                        bool shared_worker, int32_t fill_percentage, std::mutex* api_mutex = nullptr, int32_t worker_num = 1)
        : bot(discord_bot), llama_manager(llama_mgr), context_id(ctx_id), 
          assigned_channels(channels), is_shared_worker(shared_worker),
          context_fill_percentage(fill_percentage),
          discord_api_mutex(api_mutex), worker_number(worker_num) {
        
        // Initialize channel states and statistics
        for (uint64_t channel_id : assigned_channels) {
            channel_states[channel_id] = WorkerChannelState{};
            channel_stats[channel_id] = ChannelStats{};
        }
        
        // ADDED: Log worker initialization
        log_message("Worker #" + std::to_string(worker_number) + " initialized for context '" + 
                   context_id + "' with " + std::to_string(assigned_channels.size()) + " channels");
    }
    
    std::string get_context_id() const { return context_id; }
    
    // Enhanced worker execution with detailed progress logging
    WorkerResult run() {
        WorkerResult result;
        result.context_id = context_id;
        result.success = false;
        
        // ADDED: Force initial log message
        DISCORD_HISTORY_LOG("Worker #" + std::to_string(worker_number) + " STARTING execution for " + 
                           std::to_string(assigned_channels.size()) + " channels (shared: " + 
                           (is_shared_worker ? "yes" : "no") + ")");
        
        try {
            std::vector<HistoryMessage> all_messages;
            int32_t total_iterations = 0;
            const int32_t max_total_iterations = max_iterations_per_channel * assigned_channels.size();
            int32_t consecutive_failures = 0;
            const int32_t max_consecutive_failures = 5;
            
            while (should_continue_fetching() && !should_stop) {
                bool made_progress = false;
                
                // ADDED: Log iteration start
                log_message("Starting iteration " + std::to_string(total_iterations + 1));
                
                if (is_shared_worker) {
                    // Round-robin through channels
                    for (size_t i = 0; i < assigned_channels.size() && should_continue_fetching() && !should_stop; ++i) {
                        uint64_t channel_id = get_next_channel();
                        auto& state = channel_states[channel_id];
                        
                        log_message("Processing channel " + std::to_string(channel_id) + 
                                   " (iteration " + std::to_string(total_iterations + 1) + ")");
                        
                        if (state.fetch_complete) {
                            log_message("Channel " + std::to_string(channel_id) + " already completed, skipping");
                            continue;
                        }
                        
                        // Add iteration limit per channel
                        if (state.messages_fetched > max_iterations_per_channel * MESSAGES_PER_FETCH) {
                            log_message("Channel " + std::to_string(channel_id) + " reached iteration limit, marking complete");
                            log_worker_progress(channel_id);
                            state.fetch_complete = true;
                            continue;
                        }
                        
                        // Fetch with error handling
                        auto future = fetch_channel_messages(channel_id, state.last_message_id);
                        auto status = future.wait_for(std::chrono::seconds(15));
                        
                        if (status != std::future_status::ready) {
                            log_message("Timeout fetching messages from channel " + std::to_string(channel_id));
                            log_worker_progress(channel_id);
                            consecutive_failures++;
                            if (consecutive_failures >= max_consecutive_failures) {
                                log_message("Too many consecutive failures (" + std::to_string(consecutive_failures) + "), stopping worker");
                                result.error_message = "Too many consecutive fetch failures";
                                return result;
                            }
                            state.fetch_complete = true;
                            continue;
                        }
                        
                        std::vector<HistoryMessage> messages;
                        try {
                            messages = future.get();
                        } catch (const std::exception& e) {
                            log_message("Exception getting messages from channel " + std::to_string(channel_id) + ": " + std::string(e.what()));
                            log_worker_progress(channel_id);
                            consecutive_failures++;
                            if (consecutive_failures >= max_consecutive_failures) {
                                log_message("Too many consecutive failures, stopping worker");
                                result.error_message = "Too many consecutive fetch exceptions";
                                return result;
                            }
                            continue;
                        }
                        
                        if (messages.empty()) {
                            state.fetch_complete = true;
                            log_message("No more messages available for channel " + std::to_string(channel_id));
                            log_worker_progress(channel_id);
                            continue;
                        }
                        
                        // Reset failure counter on success
                        consecutive_failures = 0;
                        
                        // Add messages to context
                        if (add_messages_to_context(messages, channel_id)) {
                            all_messages.insert(all_messages.end(), messages.begin(), messages.end());
                            made_progress = true;
                            log_message("Successfully processed " + std::to_string(messages.size()) + " messages from channel " + std::to_string(channel_id));
                            log_worker_progress(channel_id);
                        } else {
                            log_message("Failed to add messages from channel " + std::to_string(channel_id) + ", stopping worker");
                            log_worker_progress(channel_id);
                            result.error_message = "Failed to add messages to context";
                            return result;
                        }
                        
                        // Small delay between channels
                        std::this_thread::sleep_for(std::chrono::milliseconds(250));
                    }
                } else {
                    // Single channel processing for isolated worker
                    uint64_t channel_id = assigned_channels[0];
                    auto& state = channel_states[channel_id];
                    
                    log_message("Processing isolated channel " + std::to_string(channel_id) + 
                               " (iteration " + std::to_string(total_iterations + 1) + ")");
                    
                    if (!state.fetch_complete) {
                        // Add iteration limit
                        if (state.messages_fetched > max_iterations_per_channel * MESSAGES_PER_FETCH) {
                            log_message("Isolated channel " + std::to_string(channel_id) + " reached iteration limit");
                            log_worker_progress(channel_id);
                            state.fetch_complete = true;
                        } else {
                            auto future = fetch_channel_messages(channel_id, state.last_message_id);
                            auto status = future.wait_for(std::chrono::seconds(15));
                            
                            if (status != std::future_status::ready) {
                                log_message("Timeout fetching messages from isolated channel " + std::to_string(channel_id));
                                log_worker_progress(channel_id);
                                consecutive_failures++;
                                if (consecutive_failures >= max_consecutive_failures) {
                                    log_message("Too many consecutive failures, stopping isolated worker");
                                    result.error_message = "Too many consecutive fetch failures";
                                    return result;
                                }
                                state.fetch_complete = true;
                            } else {
                                std::vector<HistoryMessage> messages;
                                try {
                                    messages = future.get();
                                } catch (const std::exception& e) {
                                    log_message("Exception getting messages from isolated channel " + std::to_string(channel_id) + ": " + std::string(e.what()));
                                    log_worker_progress(channel_id);
                                    consecutive_failures++;
                                    if (consecutive_failures >= max_consecutive_failures) {
                                        log_message("Too many consecutive failures, stopping isolated worker");
                                        result.error_message = "Too many consecutive fetch exceptions";
                                        return result;
                                    }
                                    state.fetch_complete = true;
                                    continue;
                                }
                                
                                if (messages.empty()) {
                                    state.fetch_complete = true;
                                    log_message("No more messages available for isolated channel " + std::to_string(channel_id));
                                    log_worker_progress(channel_id);
                                } else {
                                    // Reset failure counter on success
                                    consecutive_failures = 0;
                                    
                                    if (add_messages_to_context(messages, channel_id)) {
                                        all_messages.insert(all_messages.end(), messages.begin(), messages.end());
                                        made_progress = true;
                                        log_message("Successfully processed " + std::to_string(messages.size()) + " messages from isolated channel " + std::to_string(channel_id));
                                        log_worker_progress(channel_id);
                                    } else {
                                        log_message("Failed to add messages from isolated channel " + std::to_string(channel_id) + ", stopping worker");
                                        log_worker_progress(channel_id);
                                        result.error_message = "Failed to add messages to context";
                                        return result;
                                    }
                                }
                            }
                        }
                        
                        std::this_thread::sleep_for(std::chrono::milliseconds(250));
                    }
                }
                
                total_iterations++;
                
                // ADDED: Log iteration completion
                log_message("Completed iteration " + std::to_string(total_iterations) + 
                           ", made progress: " + (made_progress ? "yes" : "no"));
                
                // Safety check for infinite loops
                if (total_iterations > max_total_iterations) {
                    log_message("Worker reached maximum total iterations (" + std::to_string(max_total_iterations) + "), stopping");
                    // Log final progress for all channels
                    for (uint64_t channel_id : assigned_channels) {
                        log_worker_progress(channel_id);
                    }
                    break;
                }
                
                if (!made_progress) {
                    // Check if all channels are complete
                    bool all_complete = true;
                    for (const auto& [channel_id, state] : channel_states) {
                        if (!state.fetch_complete) {
                            all_complete = false;
                            break;
                        }
                    }
                    
                    if (all_complete) {
                        log_message("All assigned channels completed");
                        // Log final progress for all channels
                        for (uint64_t channel_id : assigned_channels) {
                            log_worker_progress(channel_id);
                        }
                        break;
                    }
                    
                    // Delay when no progress is made
                    log_message("No progress made, waiting before next iteration...");
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                }
            }
            
            if (should_stop) {
                log_message("Worker stopped by request");
                result.error_message = "Worker stopped";
            } else {
                // Calculate totals
                for (const auto& [channel_id, state] : channel_states) {
                    result.total_messages_processed += state.messages_fetched;
                    result.estimated_tokens_used += state.estimated_tokens_added;
                }
                
                result.messages = std::move(all_messages);
                result.success = true;
                
                log_message("Worker completed successfully: " + 
                           std::to_string(result.total_messages_processed) + " messages, " +
                           std::to_string(result.estimated_tokens_used) + " tokens");
            }
            
        } catch (const std::exception& e) {
            result.error_message = "Worker exception: " + std::string(e.what());
            log_message(result.error_message);
        }
        
        return result;
    }
    
    void stop() {
        should_stop = true;
        log_message("Worker stop requested");
    }
};

// Now define the helper methods after DiscordHistoryWorker is fully defined
inline void DiscordHistoryLoader::start_all_workers() {
    // Move workers to local vector to avoid accessing after transfer
    std::vector<std::unique_ptr<DiscordHistoryWorker>> workers_to_start;
    {
        std::lock_guard<std::mutex> lock(workers_mutex);
        workers_to_start = std::move(workers);
        workers.clear(); // Clear the member vector
    }
    
    // FIXED: Add worker completion tracking
    log_message("Starting " + std::to_string(workers_to_start.size()) + " workers...");
    
    // Start each worker in its own thread
    for (auto& worker : workers_to_start) {
        // FIXED: Get context ID before moving the worker
        std::string worker_context_id = worker->get_context_id();
        std::thread worker_thread([this, worker_context_id](std::unique_ptr<DiscordHistoryWorker> worker_instance) {
            try {
                WorkerResult result = worker_instance->run();
                
                // FIXED: Use simple deferred callback approach
                std::thread completion_thread([this, result]() {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    handle_worker_completion(result.context_id, result);
                });
                completion_thread.detach();
            } catch (const std::exception& e) {
                // ADDED: Handle exceptions in worker threads
                WorkerResult error_result;
                error_result.context_id = worker_context_id;
                error_result.success = false;
                error_result.error_message = "Worker thread exception: " + std::string(e.what());
                
                std::thread error_thread([this, error_result]() {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    handle_worker_completion(error_result.context_id, error_result);
                });
                error_thread.detach();
            }
        }, std::move(worker));
        worker_thread.detach();
    }
}
