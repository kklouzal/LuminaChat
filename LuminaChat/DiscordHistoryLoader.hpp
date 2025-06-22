// DiscordHistoryLoader.hpp - header-only implementation for isolated Discord chat history backfill
// Handles immediate Discord message history fetching with pre-tokenization for individual contexts.
//
// File Specific Directives:
// Individual history loaders per context with immediate collection and pre-tokenization.
// Each loader is tied to exactly one context and one Discord channel.
// All contexts are isolated and independent, no shared context coordination required.
//
// PERFORMANCE OPTIMIZATIONS IMPLEMENTED:
// 1. STL Algorithm Enhancements: Replaced verbose manual loops with efficient STL algorithms
// 2. Direct Member Access: Eliminated unnecessary wrapper methods and accessor patterns
// 3. Redundancy Elimination: Removed duplicate validation code and consolidated similar patterns
// 4. Memory Efficiency: Used move semantics and emplace operations for better container performance
// 5. Mutex Optimization: Consolidated critical sections to reduce lock contention
// 6. Validation Streamlining: Replaced complex STL validation with simple direct checks
// 7. Deadlock Prevention: Eliminated 'this' capture in async callbacks and non-blocking mutex acquisition
// 8. Branch Prediction: Added [[likely]]/[[unlikely]] attributes for optimal CPU branch prediction
// 9. Exception Handling: Removed try-catch from hot paths, using validation instead where possible
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
#include <chrono>
#include <unordered_set>
#include <mutex>
#include <thread>
#include <algorithm>
#include <future>
#include <optional>
#include <numeric>

#include <dpp/dpp.h>
#include "llama-cpp.h"
#include "LogHandler.hpp"
#include "common_utils.hpp"
#include "Sanitizer.hpp"

class LlamaManager;

// Forward declarations for ContextSizeManager integration
struct ContextInfo;
struct ModelInfo;
struct EnhancedContextAnalysis;
void initialize_context_size_manager(ContextInfo& context_info, const ModelInfo& model_info);
EnhancedContextAnalysis analyze_context_usage(ContextInfo& context_info, const ModelInfo& model_info);
void track_user_message(ContextInfo& context_info, const ModelInfo& model_info, int32_t user_tokens);

// Status structure for DiscordManager compatibility
struct BackfillStatus {
    bool in_progress;
    int32_t total_channels;
    int32_t completed_channels;
    int32_t total_messages_fetched;
    int32_t total_tokens_loaded_this_session;
};

// Pre-tokenized message simplified structure
struct PreTokenizedMessage {
    uint64_t channel_id;
    uint64_t message_id;
    std::string username;
    std::string content;
    std::chrono::system_clock::time_point timestamp;
    int32_t token_count;
    
    PreTokenizedMessage() : channel_id(0), message_id(0), token_count(0) {}
};

// Forward declaration
class DiscordHistoryLoader;

// NOTE: For Discord history backfill, we use insert_historical_message() instead of add_message()
// because Discord API returns messages from newest to oldest, and we want to maintain proper
// chronological order by inserting historical messages right after system messages but before
// the current conversation, rather than appending them to the end.

// Individual history loader for a specific isolated context
class DiscordHistoryLoader {
private:
    // Context association - each loader tied to one context
    std::string target_context_id;
    ContextInfo* target_context = nullptr;
    LlamaManager* llama_manager = nullptr;
    std::string model_id;
    
    // Thread safety for context access
    mutable std::mutex context_access_mutex;    // Target channel for this isolated context loader
    uint64_t target_channel_id;

    // NOTE: context_fill_ratio represents the target percentage of context to fill with HISTORICAL MESSAGES ONLY
    // The actual safe threshold must account for emergency buffer and AI response space
    float context_fill_ratio = 0.5f;
    
    // Bot identification
    uint64_t bot_user_id = 0;
    
    // Discord API access
    dpp::cluster* bot = nullptr;
      // Pre-tokenized message storage
    std::vector<PreTokenizedMessage> collected_messages;
    std::atomic<int32_t> total_tokens_collected{0};
    std::atomic<bool> target_reached{false};
    std::atomic<bool> collection_complete{false};
    std::atomic<bool> messages_applied{false};  // Track if messages have been applied to prevent duplicates
    mutable std::mutex messages_mutex;    // Processing state
    uint64_t last_message_id = 0;
    std::unordered_set<uint64_t> processed_message_ids;
    mutable std::mutex processed_ids_mutex;
    
    // Direct validation without unnecessary wrapper - eliminated redundant method
    
    // Following directive #7: favor direct access over abstractions
    bool validate_context_safe() const {
        std::lock_guard<std::mutex> lock(context_access_mutex);
        return target_context && llama_manager && 
               target_context->context_size_manager && target_context->model_info;
    }
    
    // Unsafe validation - must be called while context_access_mutex is already held
    bool validate_context_unsafe() const {
        return target_context && llama_manager && 
               target_context->context_size_manager && target_context->model_info;
    }
    
    static constexpr int32_t MESSAGES_PER_FETCH = 10;
    static constexpr int32_t MAX_ITERATIONS = 1000;
    static constexpr int32_t TIMEOUT_SECONDS = 10;
    static constexpr int32_t RATE_LIMIT_DELAY_MS = 100;
    static constexpr int32_t MAX_MESSAGE_TOKENS = 2048;
      // Fetch and process a batch of messages from the target channel
    // Optimized batch processing with STL algorithms and error handling
    // Fixed deadlock by completely avoiding 'this' capture in async callbacks
    bool process_channel_batch() {
        if (target_reached.load()) [[unlikely]] return false;
        
        auto promise = std::make_shared<std::promise<dpp::message_map>>();
        auto future = promise->get_future();
        
        // Capture only essential data, no 'this' pointer to prevent deadlocks
        uint64_t channel_id = target_channel_id;
        
        bot->messages_get(channel_id, 0, last_message_id, 0, MESSAGES_PER_FETCH,
            [promise](const dpp::confirmation_callback_t& callback) {
                try {
                    if (callback.is_error()) [[unlikely]] {
                        promise->set_value(dpp::message_map{});
                        return;
                    }
                      // Just pass the message map, no processing in callback
                    promise->set_value(callback.get<dpp::message_map>());
                } catch (const std::exception&) {
                    // Can't log here without 'this', so just return empty map
                    promise->set_value(dpp::message_map{});
                }
            });
        
        // Wait for the async operation to complete
        if (future.wait_for(std::chrono::seconds(TIMEOUT_SECONDS)) != std::future_status::ready) [[unlikely]] {
            return false;
        }
        
        auto messages = future.get();
        if (messages.empty()) [[unlikely]] {
            return false;
        }
        
        // Process messages synchronously on the calling thread (no deadlock risk)
        return collect_and_tokenize_messages(messages, channel_id);
    }
    
    // Enhanced message collection with STL optimizations and separation of concerns
    bool collect_and_tokenize_messages(const dpp::message_map& messages, uint64_t channel_id) {
        std::vector<PreTokenizedMessage> batch_messages;
        batch_messages.reserve(messages.size()); // STL: Pre-allocate for performance
        
        // STL: Find oldest message ID efficiently
        auto oldest_it = std::min_element(messages.begin(), messages.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });        // Process messages with optimized filtering
        for (const auto& [id, msg] : messages) {
            if (auto processed_msg = process_individual_message(msg, channel_id)) {
                batch_messages.emplace_back(std::move(*processed_msg));
            }
        }
        
        // Update pagination state - explicit cast to resolve type ambiguity
        last_message_id = oldest_it != messages.end() ? static_cast<uint64_t>(oldest_it->first) : last_message_id;
          // Apply batch to context and check fill ratio
        return apply_message_batch(std::move(batch_messages));
    }
    
    // Process individual message - simplified without pre-tokenization
    std::optional<PreTokenizedMessage> process_individual_message(const dpp::message& msg, uint64_t channel_id) {
        uint64_t msg_id = static_cast<uint64_t>(msg.id);

        // Fast duplicate check with early return
        {
            std::lock_guard<std::mutex> lock(processed_ids_mutex);
            if (!processed_message_ids.insert(msg_id).second) return std::nullopt;
        }
        
        // Determine if this is our bot message
        bool is_our_bot = msg.author.is_bot() && (static_cast<uint64_t>(msg.author.id) == bot_user_id);
        
        // Only process messages from users (non-bots) or from our specific bot
        if (msg.author.is_bot() && !is_our_bot) {
            return std::nullopt;
        }
        
        // Extract and sanitize content
        std::string content = extract_message_content(msg, is_our_bot);
        if (content.empty()) return std::nullopt;
        
        // Skip blacklisted bot responses
        if (is_our_bot && llama_manager && llama_manager->is_blacklisted_response(content)) {
            return std::nullopt;
        }
        
        // Create message without pre-tokenization
        PreTokenizedMessage pre_msg;
        pre_msg.channel_id = channel_id;
        pre_msg.message_id = msg_id;
        pre_msg.username = is_our_bot ? "assistant" : msg.author.username;
        pre_msg.timestamp = std::chrono::system_clock::time_point(std::chrono::seconds(msg.sent));
        
        // Store formatted content - tokenization will happen during context rebuild
        std::string formatted = is_our_bot ? content : (msg.author.username + ": " + content);
        pre_msg.content = formatted;
        pre_msg.token_count = 0; // Will be calculated during context rebuild
        
        return std::make_optional(std::move(pre_msg));
    }
    
    // Extract content from message (direct access, no unnecessary wrappers)
    std::string extract_message_content(const dpp::message& msg, bool is_our_bot) const {        std::string content = msg.content;
        if (is_our_bot && !msg.embeds.empty() && !msg.embeds[0].description.empty()) {
            content = msg.embeds[0].description;
        }
        return safe_trim(TextSanitizer::sanitize_text(content));
    }
    
    // Apply batch of messages to context - exactly as specified
    bool apply_message_batch(std::vector<PreTokenizedMessage> batch_messages) {
        if (batch_messages.empty()) return false;
        
        std::lock_guard<std::mutex> lock(context_access_mutex);
        if (!validate_context_unsafe()) return false;
        
        // Step 2: Loop through batch, add to context 1-by-1
        for (auto it = batch_messages.rbegin(); it != batch_messages.rend(); ++it) {            std::string role = (it->username == "assistant") ? "assistant" : "user";
            target_context->insert_historical_message(role, it->content);
        }
        
        // Step 3: Trigger FULL context rebuild (NOT incremental)
        bool success = llama_manager->update_context_from_history(target_context);
        
        if (!success) {
            DISCORD_HISTORY_LOG("Context update failed for '" + target_context_id + "'");
            return false;
        }
        
        // CRITICAL FIX: Reset conversation state flags after successful context rebuild
        // This prevents unexpected incremental rebuilds when the user later generates responses
        target_context->conversation_state.needs_rebuild = false;
        target_context->message_cache_dirty = false;
          // Step 4: Check context usage size
        if (!target_context->context_size_manager) return false;
        
        auto analysis = analyze_context_usage(*target_context, *target_context->model_info);
        float current_usage = static_cast<float>(analysis.total_used_tokens) / analysis.context_size;
        
        // Calculate safe target with direct arithmetic
        float safe_target = std::min(
            context_fill_ratio * (1.0f - analysis.emergency_buffer_percentage - analysis.ai_allocation_percentage),
            1.0f - analysis.emergency_buffer_percentage - analysis.ai_allocation_percentage - 0.05f
        );
        
        // Store messages for tracking
        {
            std::lock_guard<std::mutex> lock_msgs(messages_mutex);
            collected_messages.insert(collected_messages.end(), 
                                    std::make_move_iterator(batch_messages.begin()),
                                    std::make_move_iterator(batch_messages.end()));
        }
        
        // Update token count and check if target reached
        total_tokens_collected.store(analysis.total_used_tokens);
        
        if (current_usage >= safe_target) {
            target_reached = true;
        }
        
        DISCORD_HISTORY_LOG("Batch applied to '" + target_context_id + "': " + 
                          std::to_string(current_usage * 100) + "% usage (target: " +
                          std::to_string(safe_target * 100) + "%)");
          return true;
    }
    
    // Collection thread main function - simplified
    void collection_thread() {
        DISCORD_HISTORY_LOG("Starting collection for context '" + target_context_id + "'");
        
        // Basic validation
        if (!validate_context_safe()) {
            DISCORD_HISTORY_LOG("ERROR: Context validation failed for '" + target_context_id + "'");
            collection_complete = true;
            return;
        }
        
        // Wait for context initialization
        while (true) {
            {
                std::lock_guard<std::mutex> lock(context_access_mutex);
                if (target_context && target_context->fully_initialized.load()) {
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // Initialize ContextSizeManager if needed
        {
            std::lock_guard<std::mutex> lock(context_access_mutex);
            if (!target_context->context_size_manager) {
                initialize_context_size_manager(*target_context, *target_context->model_info);
            }
        }
        
        // Collect messages until target reached
        int32_t iterations = 0;
        while (!target_reached.load() && iterations < MAX_ITERATIONS) {
            if (!process_channel_batch()) {
                break;
            }            std::this_thread::sleep_for(std::chrono::milliseconds(RATE_LIMIT_DELAY_MS));
            iterations++;
        }
        
        collection_complete = true;
        apply_to_isolated_context();
          DISCORD_HISTORY_LOG("Collection complete for context '" + target_context_id + "': " + 
                           std::to_string(total_tokens_collected.load()) + " tokens collected");
    }

public:    // Factory method for creating isolated context loaders
    static std::shared_ptr<DiscordHistoryLoader> create_for_context(
        const std::string& context_id, uint64_t channel_id, 
        ContextInfo* context, LlamaManager* llama_mgr, const std::string& model_identifier,
        dpp::cluster* discord_bot, uint64_t bot_user_id, float fill_ratio) {
        
        auto loader = std::make_shared<DiscordHistoryLoader>();
        loader->target_context_id = context_id;
        loader->target_context = context;
        loader->llama_manager = llama_mgr;
        loader->model_id = model_identifier;
        loader->target_channel_id = channel_id;
        loader->context_fill_ratio = fill_ratio;
        loader->bot = discord_bot;
        loader->bot_user_id = bot_user_id;
        
        return loader;
    }    // Start collection immediately
    void start_collection() {
        std::thread(&DiscordHistoryLoader::collection_thread, this).detach();
    }
    
    // Apply messages directly to context
    bool apply_to_isolated_context() {
        // Check if already applied to prevent duplicate applications
        if (messages_applied.load()) {
            DISCORD_HISTORY_LOG("Messages already applied to context '" + target_context_id + "' - skipping duplicate application");
            return true;
        }
        
        // Messages are already applied to context during collection
        // Just mark as applied and log completion
        messages_applied = true;
          DISCORD_HISTORY_LOG("Messages applied to context '" + target_context_id + "' during collection");
        DISCORD_HISTORY_LOG("Final context usage: " + std::to_string(total_tokens_collected.load()) + " tokens");
        return true;
    }
    
    // Get collected messages for reference
    std::vector<PreTokenizedMessage> get_collected_messages() const {
        std::lock_guard<std::mutex> lock(messages_mutex);
        return collected_messages;
    }

public:
    // Direct member access - eliminate unnecessary LoaderStatus wrapper
    // Following directive #7: favor direct access over abstractions
    
    // State access methods (const and thread-safe)
    bool is_target_reached() const { return target_reached.load(); }
    bool is_collection_complete() const { return collection_complete.load(); }
    bool are_messages_applied() const { return messages_applied.load(); }
    int32_t get_total_tokens_collected() const { return total_tokens_collected.load(); }
    
    size_t get_messages_collected() const {
        std::lock_guard<std::mutex> lock(messages_mutex);
        return collected_messages.size();
    }
};

//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODING DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//
