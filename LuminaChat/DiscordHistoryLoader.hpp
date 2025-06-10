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
#include <thread>
#include <algorithm>
#include <future>
#include <sstream>

#include <dpp/dpp.h>
#include "LogHandler.hpp"

class LlamaManager;

struct HistoryMessage {
    uint64_t message_id;
    uint64_t user_id;
    uint64_t channel_id;
    std::string username;
    std::string content;
    std::chrono::system_clock::time_point timestamp;
};

// ADDED: Missing WorkerChannelState structure
struct WorkerChannelState {
    uint64_t last_message_id = 0;
    bool fetch_complete = false;
    int32_t messages_fetched = 0;
    int32_t estimated_tokens_added = 0;
};

struct WorkerContextInfo {
    std::string context_id;
    int32_t estimated_tokens = 0;
    int32_t capacity_limit = 0;
    bool is_main_context = false;
    bool worker_active = false;
    bool worker_complete = false;
};

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
    dpp::cluster* bot = nullptr;
    LlamaManager* llama_manager = nullptr;
    std::string main_context_id;
    
    // Channel configuration - const pointers for read-only access
    const std::unordered_set<uint64_t>* isolated_channels = nullptr;
    const std::unordered_set<uint64_t>* shared_history_channels = nullptr;
    
    // Settings
    bool history_enabled = true;
    int32_t context_fill_percentage = 50;
    
    // ADDED: Missing workers vector and consolidated mutex
    std::vector<std::unique_ptr<class DiscordHistoryWorker>> workers;
    std::unordered_map<std::string, std::unique_ptr<WorkerContextInfo>> context_info_map;
    mutable std::mutex worker_mutex;
    
    // State
    std::atomic<bool> backfill_in_progress{false};
    std::atomic<int32_t> active_workers{0};
    std::atomic<int32_t> worker_counter{0};
    
    // Constants - cached for performance
    static constexpr int32_t MESSAGES_PER_FETCH = 10;
    static constexpr float BASE_MAX_CONTEXT_FILL_RATIO = 0.01f;
    static constexpr int32_t MAX_RETRIES = 3;
    static constexpr std::chrono::seconds WORKER_TIMEOUT{15};
    
    // Token estimation - optimized calculation
    static constexpr int32_t estimate_message_tokens(const std::string& username, const std::string& content) {
        return 50 + static_cast<int32_t>(username.length()) + 20 + static_cast<int32_t>((content.length() + 2) / 3);
    }
    
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
    
    std::unique_ptr<WorkerContextInfo> create_context_info(const std::string& context_id) {
        auto info = std::make_unique<WorkerContextInfo>();
        info->context_id = context_id;
        info->is_main_context = (context_id == main_context_id);
        
        if (llama_manager) {
            int32_t context_size = llama_manager->get_context_size();
            float fill_ratio = BASE_MAX_CONTEXT_FILL_RATIO * context_fill_percentage;
            info->capacity_limit = static_cast<int32_t>(context_size * fill_ratio);
            
            // Get current usage efficiently
            std::string original_context = llama_manager->get_active_context();
            if (llama_manager->switch_to_context(context_id)) {
                info->estimated_tokens = llama_manager->get_message_history_token_count();
                if (!original_context.empty()) {
                    llama_manager->switch_to_context(original_context);
                }
            } else {
                info->capacity_limit = static_cast<int32_t>(2048 * BASE_MAX_CONTEXT_FILL_RATIO * context_fill_percentage);
            }
        } else {
            info->capacity_limit = static_cast<int32_t>(2048 * BASE_MAX_CONTEXT_FILL_RATIO * context_fill_percentage);
        }
        
        return info;
    }
    
    void handle_worker_completion(const std::string& context_id, const WorkerResult& result) {
        {
            std::lock_guard<std::mutex> lock(worker_mutex);
            auto it = context_info_map.find(context_id);
            if (it != context_info_map.end()) {
                it->second->worker_complete = true;
                it->second->worker_active = false;
            }
        }
        
        int32_t remaining = active_workers.fetch_sub(1) - 1;
        
        if (result.success) {
            log_message("Worker completed for '" + context_id + 
                       "': " + std::to_string(result.total_messages_processed) + " messages");
        } else {
            log_message("Worker failed for '" + context_id + "': " + result.error_message);
        }
        
        if (remaining <= 0) {
            check_backfill_completion();
        }
    }
    
    void check_backfill_completion() {
        backfill_in_progress = false;
        log_message("Chat history backfill completed for all contexts");
    }
    
    // ADDED: Missing method declarations
    std::unique_ptr<class DiscordHistoryWorker> create_isolated_worker(uint64_t channel_id);
    std::unique_ptr<class DiscordHistoryWorker> create_shared_worker();
    void create_all_workers(const std::vector<uint64_t>& isolated_channel_list, bool has_shared_channels);
    void process_guilds_for_backfill(const dpp::guild_map& guilds);
    void start_all_workers();

public:
    struct BackfillStatus {
        bool in_progress;
        int32_t active_workers;
        int32_t total_workers;
        std::unordered_map<std::string, WorkerContextInfo> context_usage;
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
        active_workers = 0;
        worker_counter = 0;
        
        log_message("Starting chat history backfill with " + std::to_string(context_fill_percentage) + "% context fill target");
        
        {
            std::lock_guard<std::mutex> lock(worker_mutex);
            context_info_map.clear();
        }
        
        try {
            bot->current_user_get_guilds([this](const dpp::confirmation_callback_t& callback) {
                if (callback.is_error()) {
                    log_message("Error getting guilds: " + callback.get_error().human_readable);
                    backfill_in_progress = false;
                    return;
                }
                
                auto guilds = callback.get<dpp::guild_map>();
                if (guilds.empty()) {
                    log_message("Bot is not in any guilds");
                    backfill_in_progress = false;
                    return;
                }
                
                process_guilds_for_backfill(guilds);
            });
        } catch (const std::exception& e) {
            log_message("Exception starting backfill: " + std::string(e.what()));
            backfill_in_progress = false;
        }
    }
    
    BackfillStatus get_status() const {
        std::lock_guard<std::mutex> lock(worker_mutex);
        
        BackfillStatus status;
        status.in_progress = backfill_in_progress;
        status.active_workers = active_workers;
        status.total_workers = 0;
        
        for (const auto& [context_id, info_ptr] : context_info_map) {
            if (info_ptr) {
                status.context_usage[context_id] = *info_ptr;
            }
        }
        
        return status;
    }
    
    bool is_in_progress() const { return backfill_in_progress; }
};

// DiscordHistoryWorker class definition
class DiscordHistoryWorker {
private:
    dpp::cluster* bot;
    LlamaManager* llama_manager;
    std::string context_id;
    std::vector<uint64_t> assigned_channels;
    bool is_shared_worker;
    int32_t context_fill_percentage;
    
    std::mutex* discord_api_mutex;
    
    // Worker state
    std::unordered_map<uint64_t, WorkerChannelState> channel_states;
    int32_t current_channel_index = 0;
    std::atomic<bool> should_stop{false};
    
    // Safety mechanisms
    int32_t max_iterations_per_channel = 50;
    
    // Worker identification for logging
    int32_t worker_number;
    
    // Message processing statistics per channel
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
    
    int32_t estimate_message_tokens(const std::string& username, const std::string& content) const {
        const int32_t base_overhead = 50;
        const int32_t role_overhead = username.length() + 20;
        const int32_t content_tokens = static_cast<int32_t>((content.length() + 2) / 3);
        return base_overhead + role_overhead + content_tokens;
    }
    
    std::future<std::vector<HistoryMessage>> fetch_channel_messages(uint64_t channel_id, uint64_t before_message_id = 0);
    bool add_messages_to_context(const std::vector<HistoryMessage>& messages, uint64_t channel_id);
    bool should_continue_fetching();
    uint64_t get_next_channel();

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
        
        log_message("Worker #" + std::to_string(worker_number) + " initialized for context '" + 
                   context_id + "' with " + std::to_string(assigned_channels.size()) + " channels");
    }
    
    std::string get_context_id() const { return context_id; }
    
    WorkerResult run();
    void stop();
};

// ADDED: Implementation of missing methods
inline std::unique_ptr<DiscordHistoryWorker> DiscordHistoryLoader::create_isolated_worker(uint64_t channel_id) {
    std::string context_id = "discord_channel_" + std::to_string(channel_id);
    auto context_info = create_context_info(context_id);
    
    log_message("Creating isolated worker for channel " + std::to_string(channel_id) + 
               " with context '" + context_id + "'");
    
    {
        std::lock_guard<std::mutex> lock(worker_mutex);
        context_info_map[context_id] = std::move(context_info);
    }
    
    int32_t worker_number = worker_counter.fetch_add(1) + 1;
    
    return std::make_unique<DiscordHistoryWorker>(
        bot, llama_manager, context_id, std::vector<uint64_t>{channel_id}, 
        false, context_fill_percentage, nullptr, worker_number
    );
}

inline std::unique_ptr<DiscordHistoryWorker> DiscordHistoryLoader::create_shared_worker() {
    if (main_context_id.empty()) {
        log_message("Error: Main context ID not set for shared worker");
        return nullptr;
    }
    
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
               " channels with context '" + main_context_id + "'");
    
    {
        std::lock_guard<std::mutex> lock(worker_mutex);
        context_info_map[main_context_id] = std::move(context_info);
    }
    
    int32_t worker_number = worker_counter.fetch_add(1) + 1;
    
    return std::make_unique<DiscordHistoryWorker>(
        bot, llama_manager, main_context_id, shared_channels, 
        true, context_fill_percentage, nullptr, worker_number
    );
}

inline void DiscordHistoryLoader::process_guilds_for_backfill(const dpp::guild_map& guilds) {
    auto isolated_channels_shared = std::make_shared<std::vector<uint64_t>>();
    auto shared_channels_shared = std::make_shared<std::vector<uint64_t>>();
    auto guilds_processed = std::make_shared<std::atomic<int32_t>>(0);
    int32_t total_guilds = static_cast<int32_t>(guilds.size());
    
    if (total_guilds == 0) {
        log_message("No guilds found for backfill");
        backfill_in_progress = false;
        return;
    }
    
    log_message("Processing " + std::to_string(total_guilds) + " guilds for channel discovery...");
    
    for (const auto& [guild_id, guild] : guilds) {
        log_message("Getting channels for guild: " + guild.name + " (ID: " + std::to_string(guild_id) + ")");
        
        bot->channels_get(guild_id, [this, guild_id, isolated_channels_shared, shared_channels_shared, guilds_processed, total_guilds](const dpp::confirmation_callback_t& callback) {
            if (callback.is_error()) {
                log_message("Error getting channels for guild " + std::to_string(guild_id) + ": " + callback.get_error().human_readable);
            } else {
                auto channels = callback.get<dpp::channel_map>();
                log_message("Found " + std::to_string(channels.size()) + " channels in guild " + std::to_string(guild_id));
                
                int32_t text_channels = 0;
                int32_t configured_channels = 0;
                
                for (const auto& [channel_id, channel] : channels) {
                    if (channel.is_text_channel()) {
                        text_channels++;
                        log_message("Text channel found: " + channel.name + " (ID: " + std::to_string(channel_id) + ")");
                        
                        if (should_backfill_channel(channel_id)) {
                            configured_channels++;
                            if (is_isolated_channel(channel_id)) {
                                isolated_channels_shared->push_back(channel_id);
                                log_message("Added isolated channel: " + channel.name + " (ID: " + std::to_string(channel_id) + ")");
                            } else {
                                shared_channels_shared->push_back(channel_id);
                                log_message("Added shared channel: " + channel.name + " (ID: " + std::to_string(channel_id) + ")");
                            }
                        } else {
                            log_message("Channel " + channel.name + " (ID: " + std::to_string(channel_id) + ") not configured for backfill");
                        }
                    }
                }
                
                log_message("Guild " + std::to_string(guild_id) + " summary: " + 
                           std::to_string(text_channels) + " text channels, " + 
                           std::to_string(configured_channels) + " configured for backfill");
            }
            
            int32_t processed = guilds_processed->fetch_add(1) + 1;
            log_message("Processed guild " + std::to_string(processed) + "/" + std::to_string(total_guilds));
            
            if (processed >= total_guilds) {
                log_message("All guilds processed. Isolated channels: " + std::to_string(isolated_channels_shared->size()) + 
                           ", Shared channels: " + std::to_string(shared_channels_shared->size()));
                create_all_workers(*isolated_channels_shared, !shared_channels_shared->empty());
            }
        });
    }
}

inline void DiscordHistoryLoader::create_all_workers(const std::vector<uint64_t>& isolated_channel_list, bool has_shared_channels) {
    std::lock_guard<std::mutex> lock(worker_mutex);
    workers.clear();
    
    log_message("Creating workers for " + std::to_string(isolated_channel_list.size()) + " isolated channels and " + 
               (has_shared_channels ? "shared channels" : "no shared channels"));
    
    // Debug: Check channel configuration
    if (isolated_channels) {
        log_message("Configured isolated channels: " + std::to_string(isolated_channels->size()));
    } else {
        log_message("No isolated channels configured (pointer is null)");
    }
    
    if (shared_history_channels) {
        log_message("Configured shared history channels: " + std::to_string(shared_history_channels->size()));
    } else {
        log_message("No shared history channels configured (pointer is null)");
    }
    
    // Create workers for isolated channels
    for (uint64_t channel_id : isolated_channel_list) {
        log_message("Creating isolated worker for channel ID: " + std::to_string(channel_id));
        auto worker = create_isolated_worker(channel_id);
        if (worker) {
            workers.push_back(std::move(worker));
            log_message("Successfully created isolated worker for channel " + std::to_string(channel_id));
        } else {
            log_message("Failed to create isolated worker for channel " + std::to_string(channel_id));
        }
    }
    
    // Create worker for shared channels
    if (has_shared_channels) {
        log_message("Creating shared worker for shared channels");
        auto worker = create_shared_worker();
        if (worker) {
            workers.push_back(std::move(worker));
            log_message("Successfully created shared worker");
        } else {
            log_message("Failed to create shared worker");
        }
    }
    
    if (workers.empty()) {
        log_message("No workers created - no configured channels found");
        log_message("This may indicate that no channels are properly configured for history backfill");
        log_message("Check isolated_channels and shared_history_channels configuration");
        backfill_in_progress = false;
        return;
    }
    
    log_message("Created " + std::to_string(workers.size()) + " workers for backfill");
    
    active_workers.store(static_cast<int32_t>(workers.size()));
    start_all_workers();
}

inline void DiscordHistoryLoader::start_all_workers() {
    // Move workers to local vector to avoid accessing after transfer
    std::vector<std::unique_ptr<DiscordHistoryWorker>> workers_to_start;
    {
        std::lock_guard<std::mutex> lock(worker_mutex);
        workers_to_start = std::move(workers);
        workers.clear();
    }
    
    log_message("Starting " + std::to_string(workers_to_start.size()) + " workers...");
    
    // Start each worker in its own thread
    for (auto& worker : workers_to_start) {
        std::string worker_context_id = worker->get_context_id();
        std::thread worker_thread([this, worker_context_id](std::unique_ptr<DiscordHistoryWorker> worker_instance) {
            try {
                WorkerResult result = worker_instance->run();
                
                // Use deferred callback approach
                std::thread completion_thread([this, result]() {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    handle_worker_completion(result.context_id, result);
                });
                completion_thread.detach();
            } catch (const std::exception& e) {
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

// DiscordHistoryWorker method implementations
inline std::future<std::vector<HistoryMessage>> DiscordHistoryWorker::fetch_channel_messages(uint64_t channel_id, uint64_t before_message_id) {
    auto promise = std::make_shared<std::promise<std::vector<HistoryMessage>>>();
    auto future = promise->get_future();
    
    if (!bot) {
        promise->set_value({});
        return future;
    }
    
    log_message("Fetching messages from channel " + std::to_string(channel_id));
    
    try {
        bot->messages_get(channel_id, before_message_id, 0, 0, MESSAGES_PER_FETCH,
            [this, channel_id, promise](const dpp::confirmation_callback_t& callback) {
                std::vector<HistoryMessage> messages;
                
                try {
                    if (callback.is_error()) {
                        log_message("Error fetching history for channel " + std::to_string(channel_id));
                        promise->set_value(messages);
                        return;
                    }
                    
                    auto dpp_messages = callback.get<dpp::message_map>();
                    
                    int32_t messages_processed = 0;
                    int32_t messages_skipped = 0;
                    uint64_t last_message_id = 0;
                    
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
                    
                    // Sort messages by timestamp (oldest first)
                    std::sort(messages.begin(), messages.end(),
                        [](const HistoryMessage& a, const HistoryMessage& b) {
                            return a.timestamp < b.timestamp;
                        });
                    
                    promise->set_value(messages);
                } catch (const std::exception& e) {
                    log_message("Exception in fetch callback: " + std::string(e.what()));
                    promise->set_value({});
                }
            });
    } catch (const std::exception& e) {
        log_message("Exception in fetch_channel_messages: " + std::string(e.what()));
        promise->set_value({});
    }
    
    return future;
}

inline bool DiscordHistoryWorker::add_messages_to_context(const std::vector<HistoryMessage>& messages, uint64_t channel_id) {
    if (!llama_manager || messages.empty()) {
        return false;
    }
    
    // Check if context exists, create if missing
    if (!llama_manager->has_context(context_id)) {
        if (context_id.find("discord_channel_") == 0) {
            log_message("Creating context '" + context_id + "' for history backfill");
            if (!llama_manager->create_context(context_id, "")) {
                log_message("Failed to create context '" + context_id + "'");
                return false;
            }
        } else {
            log_message("Shared context '" + context_id + "' not available");
            return false;
        }
    }
    
    // Switch to context
    std::string original_context = llama_manager->get_active_context();
    if (!llama_manager->switch_to_context(context_id)) {
        log_message("Failed to switch to context '" + context_id + "'");
        return false;
    }
    
    // Get token count before adding messages
    int32_t tokens_before = llama_manager->get_message_history_token_count();
    
    // Add messages as conversation history
    for (const auto& msg : messages) {
        llama_manager->add_message_to_history(msg.username, msg.content);
    }
    
    // Update context
    bool success = llama_manager->update_context_from_history();
    
    if (success) {
        // Update channel state
        auto& state = channel_states[channel_id];
        state.messages_fetched += static_cast<int32_t>(messages.size());
        if (!messages.empty()) {
            state.last_message_id = messages.back().message_id;
        }
        
        int32_t tokens_after = llama_manager->get_message_history_token_count();
        int32_t actual_tokens_added = tokens_after - tokens_before;
        state.estimated_tokens_added += actual_tokens_added;
        
        log_message("Added " + std::to_string(messages.size()) + " messages (" + 
                   std::to_string(actual_tokens_added) + " tokens)");
    }
    
    // Restore original context
    if (!original_context.empty()) {
        llama_manager->switch_to_context(original_context);
    }
    
    return success;
}

inline bool DiscordHistoryWorker::should_continue_fetching() {
    if (!llama_manager) {
        return false;
    }
    
    std::string original_context = llama_manager->get_active_context();
    int32_t current_usage = 0;
    
    if (llama_manager->switch_to_context(context_id)) {
        current_usage = llama_manager->get_message_history_token_count();
        if (!original_context.empty()) {
            llama_manager->switch_to_context(original_context);
        }
    } else {
        return false;
    }
    
    int32_t context_size = llama_manager->get_context_size();
    float fill_ratio = BASE_MAX_CONTEXT_FILL_RATIO * context_fill_percentage;
    int32_t capacity_limit = static_cast<int32_t>(context_size * fill_ratio);
    
    return current_usage < capacity_limit;
}

inline uint64_t DiscordHistoryWorker::get_next_channel() {
    if (assigned_channels.empty()) return 0;
    
    uint64_t channel_id = assigned_channels[current_channel_index];
    current_channel_index = (current_channel_index + 1) % assigned_channels.size();
    return channel_id;
}

inline WorkerResult DiscordHistoryWorker::run() {
    WorkerResult result;
    result.context_id = context_id;
    result.success = false;
    
    log_message("Worker starting execution for " + std::to_string(assigned_channels.size()) + " channels");
    
    try {
        std::vector<HistoryMessage> all_messages;
        int32_t total_iterations = 0;
        const int32_t max_total_iterations = max_iterations_per_channel * assigned_channels.size();
        
        while (should_continue_fetching() && !should_stop && total_iterations < max_total_iterations) {
            bool made_progress = false;
            
            if (is_shared_worker) {
                // Round-robin through channels
                for (size_t i = 0; i < assigned_channels.size() && should_continue_fetching() && !should_stop; ++i) {
                    uint64_t channel_id = get_next_channel();
                    auto& state = channel_states[channel_id];
                    
                    if (state.fetch_complete) continue;
                    
                    auto future = fetch_channel_messages(channel_id, state.last_message_id);
                    auto status = future.wait_for(std::chrono::seconds(15));
                    
                    if (status != std::future_status::ready) {
                        state.fetch_complete = true;
                        continue;
                    }
                    
                    std::vector<HistoryMessage> messages = future.get();
                    if (messages.empty()) {
                        state.fetch_complete = true;
                        continue;
                    }
                    
                    if (add_messages_to_context(messages, channel_id)) {
                        all_messages.insert(all_messages.end(), messages.begin(), messages.end());
                        made_progress = true;
                    }
                    
                    std::this_thread::sleep_for(std::chrono::milliseconds(250));
                }
            } else {
                // Single channel processing
                uint64_t channel_id = assigned_channels[0];
                auto& state = channel_states[channel_id];
                
                if (!state.fetch_complete) {
                    auto future = fetch_channel_messages(channel_id, state.last_message_id);
                    auto status = future.wait_for(std::chrono::seconds(15));
                    
                    if (status == std::future_status::ready) {
                        std::vector<HistoryMessage> messages = future.get();
                        if (messages.empty()) {
                            state.fetch_complete = true;
                        } else if (add_messages_to_context(messages, channel_id)) {
                            all_messages.insert(all_messages.end(), messages.begin(), messages.end());
                            made_progress = true;
                        }
                    } else {
                        state.fetch_complete = true;
                    }
                    
                    std::this_thread::sleep_for(std::chrono::milliseconds(250));
                }
            }
            
            total_iterations++;
            
            if (!made_progress) {
                bool all_complete = true;
                for (const auto& [channel_id, state] : channel_states) {
                    if (!state.fetch_complete) {
                        all_complete = false;
                        break;
                    }
                }
                
                if (all_complete) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
        }
        
        // Calculate totals
        for (const auto& [channel_id, state] : channel_states) {
            result.total_messages_processed += state.messages_fetched;
            result.estimated_tokens_used += state.estimated_tokens_added;
        }
        
        result.messages = std::move(all_messages);
        result.success = true;
        
        log_message("Worker completed: " + std::to_string(result.total_messages_processed) + " messages");
        
    } catch (const std::exception& e) {
        result.error_message = "Worker exception: " + std::string(e.what());
        log_message(result.error_message);
    }
    
    return result;
}

inline void DiscordHistoryWorker::stop() {
    should_stop = true;
    log_message("Worker stop requested");
}
