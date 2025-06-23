// LlamaSummarizer.hpp - header-only implementation for message summarization functionality
// Handles conversation summarization using dedicated summary models.
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
#include <algorithm>
#include <functional>
#include "LogHandler.hpp"
#include "ContextSizeManager.hpp"

// Forward declarations to avoid circular dependency
struct ContextInfo;
struct ModelInfo;

// Constants for summarization configuration
namespace SummarizerConstants {
    // Summary slot management
    constexpr size_t MAX_SUMMARY_SLOTS = 5;
    
    // Performance constants
    constexpr size_t SUMMARY_CONTENT_RESERVE_SIZE = 4096;
    constexpr size_t MAX_MESSAGE_HISTORY_SIZE = 1000;
    
    // String formatting
    constexpr char SEPARATOR_CHAR = '=';
    constexpr size_t SEPARATOR_LENGTH = 50;
}

// Thread Safety Contract: This class is NOT thread-safe.
// External synchronization is required for concurrent access.
//
// MESSAGE HISTORY STRUCTURE AND CHRONOLOGICAL ORDER:
// =====================================================
// The LlamaSummarizer maintains a specific order in message history to ensure proper context flow:
//
// 1. Original System Message (if any) - from ContextInfo::system_message
// 2. Summary Messages (chronologically ordered: oldest → newest)
//    - [Previous conversation summary]: <oldest_summary>
//    - [Previous conversation summary]: <older_summary>
//    - [Previous conversation summary]: <newer_summary>
//    - [Previous conversation summary]: <newest_summary>
// 3. All other messages (conversation history, other system messages, etc.)
//
// SUMMARY SLOT CHRONOLOGICAL ORDER:
// =================================
// summary_slots[0] = oldest summary (represents earliest conversation period)
// summary_slots[1] = older summary
// summary_slots[2] = newer summary
// summary_slots[3] = newer summary  
// summary_slots[4] = newest summary (represents most recent summarized period)
//
// During rollover (when 6th summary would be added):
// - summary_slots[0] and summary_slots[1] are merged into a combined summary
// - The combined summary replaces summary_slots[0] (maintaining oldest position)
// - Remaining slots shift: [2]→[1], [3]→[2], [4]→[3]
// - New summary becomes summary_slots[4]
// - Result: [merged_oldest] [former_slot2] [former_slot3] [former_slot4] [new_summary]
//
class LlamaSummarizer {
private:
    // Reference to the parent context that owns this summarizer
    ContextInfo* parent_context;
    
    // Reference to the summary model for validation/checks
    ModelInfo* summary_model;
    
    // Reference to the summary context for direct operations
    ContextInfo* summary_context;
      // Callback for generating responses (decouples from LlamaManager implementation)
    std::function<std::string(const std::string&, const std::string&, ContextInfo*)> generate_response_callback;
    
    // Callback to notify when summary slots have been modified (for history updates)
    std::function<void()> summary_slots_modified_callback;
    
    // Context ID for summary operations (for logging/validation)
    static constexpr const char* SUMMARY_CONTEXT_ID = "summary_context";

public:

    // 5-slot summary system: maintains chronological order of conversation summaries
    // When the 6th summary is generated, slot 0 is dropped, slots shift left, and new summary goes to slot 4
    // Direct access per Directive #7: favor direct access over thin accessors
    std::vector<std::string> summary_slots;
    
    explicit LlamaSummarizer(ContextInfo* parent_ctx, ModelInfo* summary_mdl = nullptr, ContextInfo* summary_ctx = nullptr,
                           std::function<std::string(const std::string&, const std::string&, ContextInfo*)> response_callback = nullptr) 
        : parent_context(parent_ctx), summary_model(summary_mdl), summary_context(summary_ctx), 
          generate_response_callback(std::move(response_callback)) {
        summary_slots.reserve(SummarizerConstants::MAX_SUMMARY_SLOTS);
    }
      // Method to set summary context and callback after creation (for late initialization)
    void set_summary_resources(ModelInfo* summary_mdl, ContextInfo* summary_ctx,
                              std::function<std::string(const std::string&, const std::string&, ContextInfo*)> response_callback) {
        summary_model = summary_mdl;
        summary_context = summary_ctx;
        generate_response_callback = std::move(response_callback);
    }
    
    // Set callback for when summary slots are modified (for message history updates)
    void set_summary_slots_modified_callback(std::function<void()> callback) {
        summary_slots_modified_callback = std::move(callback);
    }
    
    // Delete copy constructor and assignment operator
    LlamaSummarizer(const LlamaSummarizer&) = delete;
    LlamaSummarizer& operator=(const LlamaSummarizer&) = delete;

    // Add summary to the 5-slot chronological summary system
    void add_summary_to_slots(const std::string& new_summary);

    // Enhanced prune message history using summary model to condense pruned messages
    void prune_message_history(std::vector<std::pair<std::string, std::string>>& message_history, 
                              float keep_ratio);
                                  
    // Summarize a collection of messages using the summary context
    std::string summarize_messages(const std::vector<std::pair<std::string, std::string>>& messages_to_summarize);    // Summary slot information structure
    struct SummarySlotInfo {
        size_t total_slots = 0;
        size_t used_slots = 0;
        std::vector<std::string> summaries; // In chronological order (oldest to newest)
    };
    
    // Get information about summary slots
    SummarySlotInfo get_summary_slot_info() const {
        SummarySlotInfo info;
        info.total_slots = SummarizerConstants::MAX_SUMMARY_SLOTS;
        info.used_slots = summary_slots.size();
        info.summaries = summary_slots;
        return info;
    }    // Enhanced summarization for very large message collections
    // Breaks down large summarization tasks into manageable chunks to avoid token limits
    std::string summarize_large_message_collection(const std::vector<std::pair<std::string, std::string>>& messages_to_summarize) {
        if (messages_to_summarize.empty()) [[unlikely]] {
            return "";
        }
        
        // For smaller collections, use the standard approach
        if (messages_to_summarize.size() <= 50) [[likely]] {
            return summarize_messages(messages_to_summarize);
        }
        
        SUMMARIZER_LOG("Large message collection detected (" + std::to_string(messages_to_summarize.size()) + 
                       " messages), using chunked summarization approach");
        
        // Break messages into chunks of ~25 messages each
        constexpr size_t CHUNK_SIZE = 25;
        std::vector<std::string> chunk_summaries;
        
        for (size_t i = 0; i < messages_to_summarize.size(); i += CHUNK_SIZE) {
            size_t end_idx = std::min(i + CHUNK_SIZE, messages_to_summarize.size());
            std::vector<std::pair<std::string, std::string>> chunk(
                messages_to_summarize.begin() + i, 
                messages_to_summarize.begin() + end_idx
            );
            
            std::string chunk_summary = summarize_messages(chunk);
            if (!chunk_summary.empty()) [[likely]] {
                chunk_summaries.push_back(std::move(chunk_summary));
                SUMMARIZER_LOG("Successfully summarized chunk " + std::to_string(i/CHUNK_SIZE + 1) + 
                               " (" + std::to_string(chunk.size()) + " messages)");
            }
        }
        
        // If we have multiple chunk summaries, combine them into a final summary
        if (chunk_summaries.size() > 1) [[likely]] {
            
            std::vector<std::pair<std::string, std::string>> final_summary_input;
            final_summary_input.reserve(chunk_summaries.size());
            for (size_t i = 0; i < chunk_summaries.size(); ++i) {
                final_summary_input.emplace_back("system", 
                    "Summary part " + std::to_string(i + 1) + ": " + std::move(chunk_summaries[i]));
            }              std::string final_summary = summarize_messages(final_summary_input);
            if (!final_summary.empty()) [[likely]] {
                SUMMARIZER_LOG("Successfully created final summary from " + std::to_string(chunk_summaries.size()) + " chunks");
                return final_summary;
            }
        }
        
        // Fallback: return the first chunk summary if final combination failed
        return chunk_summaries.empty() ? "" : std::move(chunk_summaries[0]);}
    
    // Direct summary generation using stored context pointers (eliminates LlamaManager dependency)
    std::string generate_summary_directly(const std::string& summarization_request);
    
    // ContextSizeManager integration methods for coordinated summary management
    void integrate_with_context_size_manager(EnhancedContextSizeManager* context_manager, ContextInfo* context_ref);
    void track_summary_with_context_manager(const std::string& summary, int32_t estimated_tokens = 0);
    bool should_perform_rollover_summarization() const;
    void enforce_slot_limits(); // Force cleanup of excessive slots
    
    // Remove any existing summary messages from message history
    // This prevents accumulation of stale summaries when slots are updated
    void remove_injected_summaries_from_history(std::vector<std::pair<std::string, std::string>>& message_history) {
        auto it = message_history.begin();
        size_t removed_count = 0;
        
        while (it != message_history.end()) {
            if (it->first == "system" && it->second.find("[Previous conversation summary]: ") == 0) {
                it = message_history.erase(it);
                removed_count++;
            } else {
                ++it;
            }
        }
          if (removed_count > 0) [[unlikely]] {
            SUMMARIZER_LOG("Removed " + std::to_string(removed_count) + " stale summary messages from history");
        }
    }    // Inject summary slots into message history as system messages
    // Summaries are placed immediately after the original system message (if any)
    // and maintain chronological order (oldest to newest)
    void inject_summaries_into_history(std::vector<std::pair<std::string, std::string>>& message_history) {
        // First, remove any existing injected summaries to prevent duplicates/stale data
        remove_injected_summaries_from_history(message_history);
        
        if (summary_slots.empty()) [[unlikely]] {
            SUMMARIZER_LOG("No summaries to inject - summary slots are empty");
            return;
        }
        
        // Find insertion point: right after the ORIGINAL system message only
        // The original system message (if present) should be the first message with role "system"
        // and should NOT be a summary message or other auto-generated system message
        size_t insert_pos = 0;
          // Check if first message is the original system message
        if (!message_history.empty() && 
            message_history[0].first == "system" &&
            message_history[0].second.find("[Previous conversation summary]: ") != 0 &&
            message_history[0].second.find("[Note: ") != 0) [[likely]] {
            // Original system message found at position 0, insert summaries after it
            insert_pos = 1;
            SUMMARIZER_LOG("Found original system message at position 0, inserting summaries after it");
        } else [[unlikely]] {
            // No original system message, insert summaries at the beginning
            insert_pos = 0;
            SUMMARIZER_LOG("No original system message found, inserting summaries at beginning");
        }
        
        // Insert summaries in chronological order (oldest to newest)
        // This maintains the conversation flow: system -> old summaries -> recent summaries -> conversation
        size_t injected_count = 0;
        for (const auto& summary : summary_slots) {
            if (!summary.empty()) [[likely]] {
                message_history.insert(message_history.begin() + insert_pos, 
                                     std::make_pair("system", "[Previous conversation summary]: " + summary));
                insert_pos++; // Move insertion point forward for next summary
                injected_count++;
            }
        }
        
        SUMMARIZER_LOG("Injected " + std::to_string(injected_count) + 
                       " summaries in chronological order (oldest to newest) at position " + 
                       std::to_string(insert_pos - injected_count));
        
        // CRITICAL: Ensure ContextSizeManager is synchronized immediately after injection
        // This prevents the summary token count from being lost due to timing issues
        if (context_size_manager && injected_count > 0) {
            SUMMARIZER_LOG("Performing immediate ContextSizeManager sync after summary injection");
            sync_slots_with_context_manager();
        }
    }// Check if summaries exist that should be injected into message history
    bool has_summaries_to_inject() const {
        return !summary_slots.empty() && 
               std::any_of(summary_slots.begin(), summary_slots.end(), 
                          [](const std::string& s) { return !s.empty(); });
    }    // Notify that summary slots have been modified (triggers message history updates)
    void notify_summary_slots_modified() {
        if (summary_slots_modified_callback) [[likely]] {
            SUMMARIZER_LOG("Notifying that summary slots have been modified");
            summary_slots_modified_callback();
        }
    }    // Update message history with current summary slots (removes stale, adds current)
    void refresh_summaries_in_history(std::vector<std::pair<std::string, std::string>>& message_history) {
        SUMMARIZER_LOG("Refreshing summaries in message history after slot modification");
        inject_summaries_into_history(message_history);
    }    
    // Synchronize summary slots with ContextSizeManager (public interface for external sync)
    void sync_slots_with_context_manager();
    
    // Validate summary slot chronological order (for debugging)
    bool validate_summary_chronological_order() const {
        if (summary_slots.size() <= 1) [[likely]] return true;
        
        // Summary slots should be in chronological order: oldest (index 0) to newest (last index)
        // During rollover, slots 0 and 1 are merged, and the merged summary replaces position 0
        // This maintains the chronological order: [merged_oldest] [slot2] [slot3] [slot4] [new_summary]
          SUMMARIZER_LOG("Validating chronological order of " + std::to_string(summary_slots.size()) + " summary slots");
        for (size_t i = 0; i < summary_slots.size(); ++i) {
            if (summary_slots[i].empty()) [[unlikely]] {
                SUMMARIZER_LOG("Warning: Empty summary slot at position " + std::to_string(i));
                return false;
            }
        }
        
        SUMMARIZER_LOG("✓ All summary slots contain valid summaries in chronological order");
        return true;
    }
    
    // Get detailed summary slot information for debugging
    std::string get_summary_slots_debug_info() const {
        std::string info = "Summary Slots Debug Info:\n";
        info += "Total slots: " + std::to_string(summary_slots.size()) + "/" + std::to_string(SummarizerConstants::MAX_SUMMARY_SLOTS) + "\n";
        
        for (size_t i = 0; i < summary_slots.size(); ++i) {
            std::string preview = summary_slots[i].length() > 60 ? 
                                summary_slots[i].substr(0, 60) + "..." : 
                                summary_slots[i];
            info += "Slot " + std::to_string(i) + " (chronological position " + std::to_string(i + 1) + "): " + preview + "\n";
        }        
        return info;
    }

private:
    // Reference to parent context's ContextSizeManager for coordination
    EnhancedContextSizeManager* context_size_manager = nullptr;
    ContextInfo* context_info_ref = nullptr;
    
    // Helper to estimate tokens in a summary (rough approximation)
    int32_t estimate_summary_tokens(const std::string& summary) const;
};

// Implementation of member functions


inline void LlamaSummarizer::add_summary_to_slots(const std::string& new_summary) {
    if (new_summary.empty()) [[unlikely]] return;
    
    // Estimate tokens for the new summary
    int32_t summary_tokens = estimate_summary_tokens(new_summary);
    
    // INTEGRATION FIX: Use ContextSizeManager for all slot management
    if (context_size_manager) [[likely]] {
        // Get plan from ContextSizeManager
        auto plan = context_size_manager->plan_summary_addition(summary_tokens);
        
        SUMMARIZER_LOG("ContextSizeManager summary plan: " + plan.action_plan + 
                       " (estimated usage: " + std::to_string(plan.estimated_final_usage_percentage * 100) + "%)");
        
        // Execute the plan through ContextSizeManager
        bool success = context_size_manager->execute_summary_addition(summary_tokens, [this]() -> int32_t {
            // If merge is needed, create a merged summary
            if (summary_slots.size() >= 2) {
                std::string oldest_summary = summary_slots[0];
                std::string second_oldest_summary = summary_slots[1];
                
                // Create combined summary
                std::vector<std::pair<std::string, std::string>> rollover_messages;
                rollover_messages.emplace_back("system", "Previous summary 1: " + oldest_summary);
                rollover_messages.emplace_back("system", "Previous summary 2: " + second_oldest_summary);
                
                std::string combined_summary = summarize_messages(rollover_messages);
                if (!combined_summary.empty()) {
                    // Update our local slots to match the merge
                    summary_slots.erase(summary_slots.begin(), summary_slots.begin() + 2);
                    summary_slots.insert(summary_slots.begin(), combined_summary);
                    
                    return estimate_summary_tokens(combined_summary);
                }
            }
            return 0; // Merge failed
        });
        
        if (success) [[likely]] {
            // Add to our local slots
            summary_slots.push_back(new_summary);
            
            // Sync with ContextSizeManager state
            sync_slots_with_context_manager();
            
            SUMMARIZER_LOG("Successfully added summary via ContextSizeManager coordination");
        } else {
            SUMMARIZER_LOG("Warning: ContextSizeManager rejected summary addition");
        }    } else {
        // CRITICAL: ContextSizeManager is required for proper slot management
        SUMMARIZER_LOG_ERROR("CRITICAL: Cannot add summary - ContextSizeManager not available");
        SUMMARIZER_LOG_ERROR("Summary slot management requires ContextSizeManager integration");
        return; // Do not add summary without proper slot management
    }
    
    // Validate chronological order after modification
    if (!validate_summary_chronological_order()) [[unlikely]] {
        SUMMARIZER_LOG_ERROR("CRITICAL: Chronological order validation failed after adding summary");
    }
    
    // Log detailed slot information for debugging
    SUMMARIZER_LOG("Post-addition slot state:\n" + get_summary_slots_debug_info());
    
    // Notify that summary slots have been modified so message histories can be updated
    notify_summary_slots_modified();
}

inline void LlamaSummarizer::prune_message_history(std::vector<std::pair<std::string, std::string>>& message_history, 
                                                  float keep_ratio) {
    if (message_history.empty()) [[unlikely]] {
        SUMMARIZER_LOG("No messages to prune - message history is empty");
        return;
    }
    
    // Check if summary resources are available
    if (!summary_context || !summary_model || !generate_response_callback) [[unlikely]] {
        SUMMARIZER_LOG("Warning: Summary resources not available - performing simple pruning without summarization");
        
        // Fallback: simple pruning without summarization
        bool has_system = !message_history.empty() && message_history[0].first == "system";
        size_t system_offset = has_system ? 1 : 0;
        size_t total_messages = message_history.size() - system_offset;        size_t messages_to_keep = std::max(size_t(2), static_cast<size_t>(total_messages * keep_ratio));
        
        if (messages_to_keep >= total_messages) [[likely]] {
            return; // No pruning needed
        }
        
        // Keep system message + recent messages
        std::vector<std::pair<std::string, std::string>> new_history;
        if (has_system) {
            new_history.emplace_back(std::move(message_history[0]));
        }
        
        // Add a note about removed messages
        size_t removed_count = total_messages - messages_to_keep;
        new_history.emplace_back("system", "[Note: " + std::to_string(removed_count) + 
                                " older messages removed due to context limits]");
        
        // Keep recent messages
        size_t start_idx = message_history.size() - messages_to_keep;
        for (size_t i = start_idx; i < message_history.size(); ++i) {
            new_history.emplace_back(std::move(message_history[i]));
        }
        
        message_history = std::move(new_history);
        SUMMARIZER_LOG("Simple pruning completed - kept " + std::to_string(messages_to_keep) + 
                       " of " + std::to_string(total_messages) + " messages");
        return;
    }
      // Prevent recursion during summarization process
    static bool is_pruning = false;
    if (is_pruning) [[unlikely]] {
        SUMMARIZER_LOG("Pruning already in progress, skipping to prevent recursion");
        return;
    }
    is_pruning = true;
      // Always keep system message if present
    bool has_system = !message_history.empty() && message_history[0].first == "system";
    size_t system_offset = has_system ? 1 : 0;
    
    // SAFEGUARDS: Ensure meaningful summarization
    // Don't summarize unless we have a meaningful amount of content
    const size_t MIN_MESSAGES_TO_SUMMARIZE = 3;    // At least 3 messages worth summarizing
    const size_t MIN_MESSAGES_TO_KEEP = 2;         // Always keep at least 2 recent messages
    const size_t MIN_TOTAL_FOR_SUMMARIZATION = 5;  // Don't summarize unless we have at least 5 total messages
    
    // Calculate how many non-system messages to keep
    size_t total_messages = message_history.size() - system_offset;
    
    // Check if we have enough messages to justify summarization
    if (message_history.size() < MIN_TOTAL_FOR_SUMMARIZATION) {
        SUMMARIZER_LOG("Skipping summarization - insufficient message count (" + 
                      std::to_string(message_history.size()) + " < " + 
                      std::to_string(MIN_TOTAL_FOR_SUMMARIZATION) + ")");
        is_pruning = false;
        return;
    }
    
    // Calculate messages to keep based on ratio, but enforce minimums
    size_t messages_to_keep_by_ratio = static_cast<size_t>(total_messages * keep_ratio);
    size_t messages_to_keep = std::max({MIN_MESSAGES_TO_KEEP, size_t(2), messages_to_keep_by_ratio});
    
    // Calculate how many messages would be summarized
    size_t messages_to_summarize_count = (total_messages > messages_to_keep) ? (total_messages - messages_to_keep) : 0;
    
    // Don't proceed if we wouldn't summarize enough messages to make it worthwhile
    if (messages_to_summarize_count < MIN_MESSAGES_TO_SUMMARIZE) {
        SUMMARIZER_LOG("Skipping summarization - too few messages to summarize (" + 
                      std::to_string(messages_to_summarize_count) + " < " + 
                      std::to_string(MIN_MESSAGES_TO_SUMMARIZE) + ")");
        is_pruning = false;
        return;
    }
    
    SUMMARIZER_LOG("Summarization approved: " + std::to_string(message_history.size()) + 
                   " total messages, keeping " + std::to_string(messages_to_keep) + 
                   ", summarizing " + std::to_string(messages_to_summarize_count));
      if (messages_to_keep >= total_messages) {
        SUMMARIZER_LOG("No pruning needed - keeping " + std::to_string(messages_to_keep) + " of " + std::to_string(total_messages) + " messages");
        is_pruning = false;
        return;
    }
    
    // Extract messages to be pruned (everything except system message and messages to keep)
    size_t prune_start_idx = system_offset;
    size_t prune_end_idx = message_history.size() - messages_to_keep;
      if (prune_end_idx <= prune_start_idx) {
        SUMMARIZER_LOG("No messages to prune after index calculation");
        is_pruning = false;
        return; // Nothing to prune
    }
    
    // Collect messages to be summarized
    std::vector<std::pair<std::string, std::string>> messages_to_summarize;
    for (size_t i = prune_start_idx; i < prune_end_idx; ++i) {
        messages_to_summarize.emplace_back(message_history[i]);
    }
      // Attempt to summarize using enhanced approach for large collections
    std::string summary = summarize_large_message_collection(messages_to_summarize);
    
    // Calculate how many messages were summarized for logging
    size_t summarized_count = prune_end_idx - prune_start_idx;
    
    // Build new message history
    std::vector<std::pair<std::string, std::string>> new_history;
    new_history.reserve(system_offset + SummarizerConstants::MAX_SUMMARY_SLOTS + messages_to_keep);
    
    // Add system message if present
    if (has_system) {
        new_history.emplace_back(std::move(message_history[0]));
    }
      // Handle 5-slot summary system with ContextSizeManager coordination
    if (!summary.empty()) {
        // Track the summary creation with ContextSizeManager before adding to slots
        if (context_size_manager) {
            track_summary_with_context_manager(summary);
        }
        
        // Add new summary to the slot system (this will use ContextSizeManager logic)
        add_summary_to_slots(summary);
        SUMMARIZER_LOG("Successfully created summary for " + std::to_string(summarized_count) + " pruned messages");
        SUMMARIZER_LOG("Summary slots now contain " + std::to_string(summary_slots.size()) + " summaries");
    } else {
        // Fallback: keep more messages if summarization failed
        SUMMARIZER_LOG("Warning: Summarization failed or unavailable, keeping more messages instead");
        
        // Keep up to half of the messages that would have been pruned
        size_t fallback_keep = std::min(summarized_count / 2, size_t(3)); // Keep at least 1, max 3
        size_t fallback_start = prune_end_idx - fallback_keep;
        
        for (size_t i = fallback_start; i < prune_end_idx; ++i) {
            if (i < message_history.size()) {
                new_history.emplace_back(std::move(message_history[i]));
            }
        }
        
        // Add a system message explaining what happened
        new_history.emplace_back("system", "[Note: " + std::to_string(summarized_count - fallback_keep) + 
                                " older messages removed due to context limits]");    }
      
    // NOTE: Do NOT inject summary slots here as they will be injected by the context preparation logic
    // This prevents duplicate summary injection and ensures proper coordination with slot modifications
    
    // Add the recent messages to keep
    for (size_t i = prune_end_idx; i < message_history.size(); ++i) {
        new_history.emplace_back(std::move(message_history[i]));
    }
    
    message_history = std::move(new_history);    
    SUMMARIZER_LOG("Pruned " + std::to_string(summarized_count) + " messages into summary. " +
                   "New history has " + std::to_string(message_history.size()) + " messages");
    
    // Reset the pruning flag
    is_pruning = false;
}

inline std::string LlamaSummarizer::summarize_messages(const std::vector<std::pair<std::string, std::string>>& messages_to_summarize) {
    if (messages_to_summarize.empty()) [[unlikely]] {
        SUMMARIZER_LOG("Error: Empty messages to summarize");
        return "";
    }
    
    // Check if summary resources are available
    if (!summary_context || !summary_model || !generate_response_callback) [[unlikely]] {
        SUMMARIZER_LOG_ERROR("Summary resources not available - context, model, or callback missing");
        return "";
    }
    
    // Validate that the summary context is properly initialized
    if (!summary_context->context || !summary_context->model_info || !summary_context->batch_initialized) [[unlikely]] {
        SUMMARIZER_LOG_ERROR("Summary context is not properly initialized for message summarization");
        return "";
    }
    
    // Build the content to summarize with better formatting
    std::string content_to_summarize;
    content_to_summarize.reserve(SummarizerConstants::SUMMARY_CONTENT_RESERVE_SIZE);
    
    for (const auto& [role, content] : messages_to_summarize) {
        content_to_summarize += role + ": " + content + "\n\n";
    }
    
    // Create a more explicit summarization request with better prompt
    std::string summarization_request = "Please provide a concise but informative summary of the following conversation. Focus on the main topics discussed and key information exchanged:\n\n" + 
                                      content_to_summarize + 
                                      "\nProvide a clear summary:";
    
    SUMMARIZER_LOG("Starting summarization of " + std::to_string(messages_to_summarize.size()) + " messages");
    
    // Log the input being summarized in a user-friendly format
    std::string input_summary = "=== INPUT TO SUMMARIZE ===\n";
    for (const auto& [role, content] : messages_to_summarize) {
        input_summary += role + ": " + content + "\n\n";
    }
    input_summary += "========================\n";
    SUMMARIZER_LOG(input_summary);
    
    // Generate summary using the direct approach (no context switching needed)
    std::string summary = generate_summary_directly(summarization_request);
      // Check if the summary is actually an error message
    if (!summary.empty() && summary.size() >= 6 && summary.substr(0, 6) == "Error:") [[unlikely]] {
        SUMMARIZER_LOG_ERROR("Summary generation returned error: " + summary);
        summary = ""; // Treat as failed summarization
    } else if (!summary.empty()) [[likely]] {
        // Log the successful output in a user-friendly format
        std::string output_message = "=== GENERATED SUMMARY ===\n" + summary + "\n==========================\n";
        SUMMARIZER_LOG(output_message);
    }
      if (summary.empty()) [[unlikely]] {
        SUMMARIZER_LOG("Warning: Message summarization produced empty result");
    } else [[likely]] {
        SUMMARIZER_LOG("Successfully summarized " + std::to_string(messages_to_summarize.size()) + 
                       " messages into " + std::to_string(summary.length()) + " character summary");
        
        // Track summary creation with ContextSizeManager
        if (context_size_manager) [[likely]] {
            track_summary_with_context_manager(summary);
        }
    }
    
    return summary;
}

inline std::string LlamaSummarizer::generate_summary_directly(const std::string& summarization_request) {
    if (summarization_request.empty() || !summary_context || !summary_model || !generate_response_callback) [[unlikely]] {
        SUMMARIZER_LOG("Error: Empty request, null context/model, or missing callback");
        return "";
    }
    
    // Validate that the summary context is properly initialized
    if (!summary_context->context || !summary_context->model_info || !summary_context->batch_initialized) [[unlikely]] {
        SUMMARIZER_LOG_ERROR("Summary context is not properly initialized");
        return "";
    }
    
    // Validate that the summary context is associated with the correct model
    if (summary_context->model_info != summary_model) [[unlikely]] {
        SUMMARIZER_LOG_ERROR("Summary context is not associated with the expected summary model");
        return "";
    }
    
    SUMMARIZER_LOG("Starting direct summary generation using callback");
    
    // Use the callback to generate the response directly on the summary context
    // Removed try-catch block for better performance - callback should handle errors internally
    std::string summary = generate_response_callback(summarization_request, "user", summary_context);
    
    // Clean up the summary (remove any extra whitespace, newlines)
    if (!summary.empty()) [[likely]] {
        size_t start = summary.find_first_not_of(" \t\n\r");
        size_t end = summary.find_last_not_of(" \t\n\r");
        if (start != std::string::npos && end != std::string::npos) [[likely]] {
            summary = summary.substr(start, end - start + 1);
        }
        
        SUMMARIZER_LOG("Successfully generated summary using direct callback");
    } else [[unlikely]] {
        SUMMARIZER_LOG("Warning: Generated summary is empty");
    }
      return summary;
}

// ContextSizeManager integration implementations
inline void LlamaSummarizer::integrate_with_context_size_manager(EnhancedContextSizeManager* context_manager, ContextInfo* context_ref) {
    context_size_manager = context_manager;
    context_info_ref = context_ref;
    
    if (context_size_manager && context_info_ref) [[likely]] {
        SUMMARIZER_LOG("Integrated with ContextSizeManager and ContextInfo reference for coordinated summary management");
        
        // First, enforce slot limits to clean up any excessive slots
        enforce_slot_limits();
        
        // Then sync remaining slots with ContextSizeManager
        sync_slots_with_context_manager();
        
        SUMMARIZER_LOG("Integration complete - slot limits enforced and synchronized");
    } else [[unlikely]] {
        SUMMARIZER_LOG("Warning: Failed to integrate with ContextSizeManager - missing manager or context reference");
    }
}

inline void LlamaSummarizer::track_summary_with_context_manager(const std::string& summary, int32_t estimated_tokens) {
    if (!context_size_manager || summary.empty()) [[unlikely]] return;
    
    int32_t actual_tokens = estimated_tokens > 0 ? estimated_tokens : estimate_summary_tokens(summary);
    context_size_manager->track_summary_creation(actual_tokens);
    
    SUMMARIZER_LOG("Tracked summary creation with ContextSizeManager: " + std::to_string(actual_tokens) + " tokens");
}

inline bool LlamaSummarizer::should_perform_rollover_summarization() const {
    if (!context_size_manager || !context_info_ref) [[unlikely]] {
        // Cannot perform proper rollover analysis without ContextSizeManager
        SUMMARIZER_LOG_ERROR("CRITICAL: Cannot determine rollover need - ContextSizeManager not available");
        return false; // Conservative approach - don't rollover without proper analysis
    }
    
    // Get proper context analysis from ContextSizeManager
    auto analysis = context_size_manager->analyze_context(*context_info_ref);
      // Check if ContextSizeManager recommends merge based on real analysis
    auto recommendations = context_size_manager->get_optimization_recommendations(analysis);
    for (const auto& recommendation : recommendations) {
        if (recommendation.find("merge") != std::string::npos || 
            recommendation.find("summary") != std::string::npos) [[unlikely]] {
            SUMMARIZER_LOG("ContextSizeManager recommends rollover: " + recommendation);
            return true;
        }
    }
    
    // Also check direct analysis flags
    if (analysis.needs_summary_merge || analysis.summary_hard_cap_exceeded) [[unlikely]] {
        SUMMARIZER_LOG("ContextSizeManager analysis indicates rollover needed (merge: " + 
                      std::to_string(analysis.needs_summary_merge) + ", cap exceeded: " + 
                      std::to_string(analysis.summary_hard_cap_exceeded) + ")");
        return true;
    }

    return false;
}

inline void LlamaSummarizer::sync_slots_with_context_manager() {
    if (!context_size_manager) [[unlikely]] return;
    
    SUMMARIZER_LOG("Performing atomic synchronization with ContextSizeManager:");
    SUMMARIZER_LOG("  Current local slots: " + std::to_string(summary_slots.size()));
    
    // Use atomic synchronization method instead of destructive clear-and-rebuild
    // This ensures ContextSizeManager state is updated safely from our authoritative slots
    context_size_manager->sync_with_authoritative_slots(summary_slots);
    
    // Verify sync was successful
    auto analysis = context_size_manager->analyze_context(*context_info_ref);
    auto slot_stats = analysis.summary_slot_stats;
    
    SUMMARIZER_LOG("Atomic sync verification:");
    SUMMARIZER_LOG("  Local slots: " + std::to_string(summary_slots.size()));
    SUMMARIZER_LOG("  ContextSizeManager slots: " + std::to_string(slot_stats.slot_count));
    SUMMARIZER_LOG("  Usage: " + std::to_string(slot_stats.usage_percentage * 100) + "%");
    
    if (summary_slots.size() != slot_stats.slot_count) {
        SUMMARIZER_LOG_ERROR("CRITICAL: Slot count mismatch after atomic sync!");
        SUMMARIZER_LOG_ERROR("  This indicates a serious synchronization bug - attempting recovery");
        
        // Emergency recovery: try one more time with detailed logging
        SUMMARIZER_LOG("Emergency recovery - re-attempting atomic sync with detailed logging");
        for (size_t i = 0; i < summary_slots.size(); ++i) {
            SUMMARIZER_LOG("  Slot " + std::to_string(i) + " content length: " + 
                          std::to_string(summary_slots[i].length()) + " chars");
        }
        
        context_size_manager->sync_with_authoritative_slots(summary_slots);
        
        // Re-verify
        auto recovery_analysis = context_size_manager->analyze_context(*context_info_ref);
        auto recovery_stats = recovery_analysis.summary_slot_stats;
        SUMMARIZER_LOG("Recovery result: " + std::to_string(recovery_stats.slot_count) + " slots synced");
    } else {
        SUMMARIZER_LOG("✓ Atomic slot synchronization successful");
    }
}

inline int32_t LlamaSummarizer::estimate_summary_tokens(const std::string& summary) const {
    if (summary.empty()) [[unlikely]] return 0;
    
    // Rough estimation: ~4 characters per token (this is a conservative estimate)
    // For better accuracy, this should use the actual tokenizer, but that would require
    // more complex integration. This approximation is sufficient for planning purposes.
    constexpr float CHARS_PER_TOKEN = 4.0f;
      int32_t estimated = static_cast<int32_t>(summary.length() / CHARS_PER_TOKEN);
    
    // Add small buffer for safety (10%)
    estimated = static_cast<int32_t>(estimated * 1.1f);
    
    // Minimum of 10 tokens, maximum reasonable cap of 1000 tokens for a summary
    return std::clamp(estimated, 10, 1000);
}

inline void LlamaSummarizer::enforce_slot_limits() {
    if (!context_size_manager) {
        SUMMARIZER_LOG_ERROR("CRITICAL: Cannot enforce slot limits - ContextSizeManager not available");
        return;
    }
    
    // Use ContextSizeManager to determine proper slot management actions
    auto analysis = context_size_manager->analyze_context(*context_info_ref);
    auto slot_stats = analysis.summary_slot_stats;
    
    SUMMARIZER_LOG("Enforcing slot limits via ContextSizeManager:");
    SUMMARIZER_LOG("  Current slots: " + std::to_string(summary_slots.size()));
    SUMMARIZER_LOG("  ContextSizeManager recommends: " + std::to_string(slot_stats.slot_count));
    SUMMARIZER_LOG("  Hard cap exceeded: " + std::string(slot_stats.exceeds_hard_cap ? "YES" : "NO"));
    SUMMARIZER_LOG("  Needs merge: " + std::string(slot_stats.needs_merge ? "YES" : "NO"));
    
    // If ContextSizeManager indicates we need merging/reduction, handle it
    if (slot_stats.exceeds_hard_cap && summary_slots.size() > 1) {
        SUMMARIZER_LOG("ContextSizeManager indicates hard cap exceeded - merging oldest slots");
        
        // Merge two oldest slots
        std::string oldest = summary_slots[0];
        std::string second_oldest = summary_slots[1];
        
        std::vector<std::pair<std::string, std::string>> merge_messages;
        merge_messages.emplace_back("system", "Previous summary 1: " + oldest);
        merge_messages.emplace_back("system", "Previous summary 2: " + second_oldest);
        
        std::string merged_summary = summarize_messages(merge_messages);
        if (!merged_summary.empty()) {
            // Replace the two oldest with the merged one
            summary_slots.erase(summary_slots.begin(), summary_slots.begin() + 2);
            summary_slots.insert(summary_slots.begin(), merged_summary);
            
            // Update ContextSizeManager to reflect the change
            sync_slots_with_context_manager();
            
            SUMMARIZER_LOG("Successfully merged two oldest slots into one");
        } else {
            SUMMARIZER_LOG_ERROR("Failed to merge slots - removing oldest instead");
            summary_slots.erase(summary_slots.begin());
            sync_slots_with_context_manager();
        }
    }    
    SUMMARIZER_LOG("Slot limit enforcement complete");
}

//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODING DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//
