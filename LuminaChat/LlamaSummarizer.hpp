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
// 1.  Minimalism & Performance: Deliver lean, efficient solutions; do not create or preserve unused helpers or wrappers.
// 2.  Redundancy Elimination: Remove unused, obsolete, and legacy code—including unneeded interfaces and includes.
// 3.  Consistent Style: Adopt a uniform coding style and structure for clarity and maintainability.
// 4.  Documentation: Write concise comments that explain complex logic and key design decisions.
// 5.  Zero Magic & Strong Typing: Replace magic literals with named constants, enums, or constexpr; prefer scoped enums.
// 6.  Function Boundaries: Define clear responsibilities; reduce overlap and avoid unnecessary layers.
// 7.  Core Preservation: Streamline code while safeguarding essential features; favor direct access over extra abstractions.
// 8.  Const-Correctness & Immutability: Mark variables, parameters, and methods as const wherever possible.
// 9.  RAII & Resource Safety: Encapsulate resource acquisition/release in constructors/destructors or smart pointers.
// 10. Standard Library Preference: Favor STL algorithms and containers over custom loops and buffers.
// 11. Cross-Platform Portability: Use fixed-width types and proper initialization to guarantee identical behavior everywhere.
// 12. Thread Safety: Define and document thread-safety contracts; protect shared state with mutexes, atomics, or thread-safe containers.
// 13. Smart Caching: Cache frequently used values to minimize allocations and improve performance.
// 14. Logical Consistency: Verify code flow to ensure coherent, error-free execution paths.
// 15. Continuous Refinement: Regularly refactor and confirm that updates preserve stable functionality.

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <functional>
#include <algorithm>
#include "LogHandler.hpp"

// Forward declaration to avoid circular dependency
class LlamaManager;

// Constants for summarization configuration
namespace SummarizerConstants {
    // Summary slot management
    constexpr size_t MAX_SUMMARY_SLOTS = 5;
    
    // Context management ratios
    constexpr float MAX_CONTEXT_USAGE = 0.90f;
    constexpr float TARGET_CONTEXT_USAGE = 0.60f;
    constexpr float AGGRESSIVE_PRUNING_RATIO = 0.3f;
    
    // Performance constants
    constexpr size_t SUMMARY_CONTENT_RESERVE_SIZE = 4096;
    constexpr size_t MAX_MESSAGE_HISTORY_SIZE = 1000;
    
    // String formatting
    constexpr char SEPARATOR_CHAR = '=';
    constexpr size_t SEPARATOR_LENGTH = 50;
}

// Thread Safety Contract: This class is NOT thread-safe.
// External synchronization is required for concurrent access.
class LlamaSummarizer {
private:
    // Reference to the LlamaManager for context switching and generation
    LlamaManager* llama_manager;
    
    // Logging callbacks for summary operations
    std::function<void(const std::string&)> summary_input_callback;
    std::function<void(const std::string&)> summary_output_callback;
    
    // Context ID for summary operations
    static constexpr const char* SUMMARY_CONTEXT_ID = "summary_context";

public:
    explicit LlamaSummarizer(LlamaManager* manager) : llama_manager(manager) {}

    // Delete copy constructor and assignment operator
    LlamaSummarizer(const LlamaSummarizer&) = delete;
    LlamaSummarizer& operator=(const LlamaSummarizer&) = delete;

    // Set callbacks for summary logging
    void set_summary_input_callback(std::function<void(const std::string&)> callback) {
        summary_input_callback = std::move(callback);
    }
    
    void set_summary_output_callback(std::function<void(const std::string&)> callback) {
        summary_output_callback = std::move(callback);
    }

    // Check if summarization is available (summary context exists)
    bool is_summarization_available() const;

    // Add summary to the 5-slot chronological summary system
    void add_summary_to_slots(std::vector<std::string>& summary_slots, const std::string& new_summary);

    // Enhanced prune message history using summary model to condense pruned messages
    void prune_message_history(std::vector<std::pair<std::string, std::string>>& message_history, 
                              std::vector<std::string>& summary_slots, 
                              float keep_ratio);

    // Summarize a collection of messages using the summary context
    std::string summarize_messages(const std::vector<std::pair<std::string, std::string>>& messages_to_summarize);

    // Summary slot information structure
    struct SummarySlotInfo {
        size_t total_slots;
        size_t used_slots;
        std::vector<std::string> summaries; // In chronological order (oldest to newest)
    };
    
    // Get information about summary slots
    SummarySlotInfo get_summary_slot_info(const std::vector<std::string>& summary_slots) const {
        SummarySlotInfo info;
        info.total_slots = SummarizerConstants::MAX_SUMMARY_SLOTS;
        info.used_slots = summary_slots.size();
        info.summaries = summary_slots;
        return info;
    }

private:
    // Helper to log summary input to UI
    void log_summary_input(const std::vector<std::pair<std::string, std::string>>& messages_to_summarize) const {
        if (!summary_input_callback) return;
        
        std::string ui_input = "Summarizing " + std::to_string(messages_to_summarize.size()) + " messages:\n";
        ui_input += std::string(SummarizerConstants::SEPARATOR_LENGTH, SummarizerConstants::SEPARATOR_CHAR) + "\n";
        
        for (const auto& [role, content] : messages_to_summarize) {
            ui_input += role + ": " + content + "\n\n";
        }
        ui_input += std::string(SummarizerConstants::SEPARATOR_LENGTH, SummarizerConstants::SEPARATOR_CHAR) + "\n";
        summary_input_callback(ui_input);
    }
    
    // Helper to log summary output to UI
    void log_summary_output(const std::string& summary) const {
        if (summary_output_callback) {
            summary_output_callback(summary);
        }
    }
    
    // Helper to log error to UI
    void log_summary_error(const std::string& error_message) const {
        if (summary_output_callback) {
            summary_output_callback("ERROR: " + error_message);
        }
    }
};

// Implementation of member functions

inline bool LlamaSummarizer::is_summarization_available() const {
    if (!llama_manager) return false;
    return llama_manager->has_context(SUMMARY_CONTEXT_ID);
}

inline void LlamaSummarizer::add_summary_to_slots(std::vector<std::string>& summary_slots, const std::string& new_summary) {
    if (new_summary.empty()) return;
    
    // If we're at capacity (5 slots), implement rollover summarization
    if (summary_slots.size() >= SummarizerConstants::MAX_SUMMARY_SLOTS) {
        SUMMARIZER_LOG("Summary slots at capacity, performing rollover summarization");
        
        // Get the two oldest summaries (slots 0 and 1)
        std::string oldest_summary = summary_slots[0];
        std::string second_oldest_summary = summary_slots[1];
        
        // Create a combined summary from the two oldest
        std::vector<std::pair<std::string, std::string>> rollover_messages;
        rollover_messages.emplace_back("system", "Previous summary 1: " + oldest_summary);
        rollover_messages.emplace_back("system", "Previous summary 2: " + second_oldest_summary);
        
        std::string combined_summary = summarize_messages(rollover_messages);
        
        if (!combined_summary.empty()) {
            // Remove the two oldest summaries and replace with the combined one
            summary_slots.erase(summary_slots.begin(), summary_slots.begin() + 2);
            summary_slots.insert(summary_slots.begin(), combined_summary);
            SUMMARIZER_LOG("Rollover summarization successful - combined 2 oldest summaries into 1");
        } else {
            // Fallback: just remove the oldest if rollover summarization fails
            summary_slots.erase(summary_slots.begin());
            SUMMARIZER_LOG("Rollover summarization failed, removed oldest summary");
        }
    }
    
    // Add new summary to the end (newest position)
    summary_slots.push_back(new_summary);
    
    SUMMARIZER_LOG("Added new summary to slot " + std::to_string(summary_slots.size()) + 
                   " of " + std::to_string(SummarizerConstants::MAX_SUMMARY_SLOTS));
}

inline void LlamaSummarizer::prune_message_history(std::vector<std::pair<std::string, std::string>>& message_history, 
                                                  std::vector<std::string>& summary_slots, 
                                                  float keep_ratio) {
    if (message_history.empty()) {
        SUMMARIZER_LOG("No messages to prune - message history is empty");
        return;
    }
    
    // Prevent recursion during summarization process
    static bool is_pruning = false;
    if (is_pruning) {
        SUMMARIZER_LOG("Pruning already in progress, skipping to prevent recursion");
        return;
    }
    is_pruning = true;
    
    // Always keep system message if present
    bool has_system = !message_history.empty() && message_history[0].first == "system";
    size_t system_offset = has_system ? 1 : 0;
    
    // Calculate how many non-system messages to keep
    size_t total_messages = message_history.size() - system_offset;
    size_t messages_to_keep = std::max(size_t(2), static_cast<size_t>(total_messages * keep_ratio));
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
    
    // Attempt to summarize using summary context
    std::string summary = summarize_messages(messages_to_summarize);
    
    // Calculate how many messages were summarized for logging
    size_t summarized_count = prune_end_idx - prune_start_idx;
    
    // Build new message history
    std::vector<std::pair<std::string, std::string>> new_history;
    new_history.reserve(system_offset + SummarizerConstants::MAX_SUMMARY_SLOTS + messages_to_keep);
    
    // Add system message if present
    if (has_system) {
        new_history.emplace_back(std::move(message_history[0]));
    }
    
    // Handle 5-slot summary system
    if (!summary.empty()) {
        // Add new summary to the slot system
        add_summary_to_slots(summary_slots, summary);
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
                                " older messages removed due to context limits]");
    }
    
    // FIXED: Do NOT add summary slots to message history during regular pruning
    // Summary slots are maintained separately and only used for rollover summarization
    // The actual summaries are not part of the conversation context
    
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
    if (messages_to_summarize.empty() || !llama_manager) {
        SUMMARIZER_LOG("Error: Empty messages or null llama_manager");
        return "";
    }
    
    // Check if we have a summary context available
    if (!llama_manager->has_context(SUMMARY_CONTEXT_ID)) {
        SUMMARIZER_LOG("Warning: No summary context available for message summarization");
        log_summary_error("No summary context available - please load a summary model first");
        return "";
    }
    
    // Store current context to restore later
    std::string original_context_id = llama_manager->get_active_context();
    SUMMARIZER_LOG("Attempting to switch from context '" + original_context_id + "' to '" + SUMMARY_CONTEXT_ID + "'");
    
    // Switch to summary context temporarily
    if (!llama_manager->switch_to_context(SUMMARY_CONTEXT_ID)) {
        SUMMARIZER_LOG("Error: Failed to switch to summary context");
        log_summary_error("Failed to switch to summary context");
        return "";
    }
    
    SUMMARIZER_LOG("Successfully switched to summary context");    
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
    
    SUMMARIZER_LOG("Preparing to summarize " + std::to_string(messages_to_summarize.size()) + " messages");
    SUMMARIZER_LOG("Summary request length: " + std::to_string(summarization_request.length()) + " characters");
    
    // Log the input to the summaries tab
    log_summary_input(messages_to_summarize);
      // Generate summary using the summary context
    std::string summary;
    try {
        SUMMARIZER_LOG("Calling generate_response for summarization...");
        
        // Validate that we're actually in the summary context before generation
        if (llama_manager->get_active_context() != SUMMARY_CONTEXT_ID) {
            SUMMARIZER_LOG("Error: Not in summary context before generation");
            log_summary_error("Context switch verification failed");
            return "";
        }
        
        summary = llama_manager->generate_response(summarization_request, "user");
        SUMMARIZER_LOG("Generate_response returned: '" + summary + "'");
        
        // Check if the summary is actually an error message
        if (!summary.empty() && summary.size() >= 6 && summary.substr(0, 6) == "Error:") {
            SUMMARIZER_LOG("Summary generation returned error: " + summary);
            log_summary_error(summary);
            summary = ""; // Treat as failed summarization
        } else if (!summary.empty()) {
            // Clean up the summary (remove any extra whitespace, newlines)
            size_t start = summary.find_first_not_of(" \t\n\r");
            size_t end = summary.find_last_not_of(" \t\n\r");
            if (start != std::string::npos && end != std::string::npos) {
                summary = summary.substr(start, end - start + 1);
            }
            
            SUMMARIZER_LOG("Cleaned summary: '" + summary + "'");
            
            // Log the output to the summaries tab
            log_summary_output(summary);
        } else {
            SUMMARIZER_LOG("Generated summary is empty");
        }
    } catch (const std::exception& e) {
        SUMMARIZER_LOG("Exception during message summarization: " + std::string(e.what()));
        summary = "";
        log_summary_error(std::string(e.what()));
    } catch (...) {
        SUMMARIZER_LOG("Unknown exception during message summarization");
        summary = "";
        log_summary_error("Unknown exception during summarization");
    }    
    // Restore original context with error checking
    SUMMARIZER_LOG("Attempting to restore original context: '" + original_context_id + "'");
    if (!original_context_id.empty()) {
        if (!llama_manager->switch_to_context(original_context_id)) {
            SUMMARIZER_LOG("Error: Failed to restore original context '" + original_context_id + "'");
            log_summary_error("Failed to restore original context");
        } else {
            SUMMARIZER_LOG("Successfully restored original context");
        }
    }
    
    if (summary.empty()) {
        SUMMARIZER_LOG("Warning: Message summarization produced empty result");
    } else {
        SUMMARIZER_LOG("Successfully summarized " + std::to_string(messages_to_summarize.size()) + 
                       " messages into " + std::to_string(summary.length()) + " character summary");
    }
    
    return summary;
}

//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODING DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//
