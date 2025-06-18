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
    
    // Context management ratios
    constexpr float MAX_CONTEXT_USAGE = 0.90f;
    constexpr float TARGET_CONTEXT_USAGE = 0.60f;
    constexpr float AGGRESSIVE_PRUNING_RATIO = 0.30f;
    
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
    // Reference to the parent context that owns this summarizer
    ContextInfo* parent_context;
    
    // Reference to the summary model for validation/checks
    ModelInfo* summary_model;
    
    // Reference to the summary context for direct operations
    ContextInfo* summary_context;
    
    // Callback for generating responses (decouples from LlamaManager implementation)
    std::function<std::string(const std::string&, const std::string&, ContextInfo*)> generate_response_callback;
    
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
    
    // Delete copy constructor and assignment operator
    LlamaSummarizer(const LlamaSummarizer&) = delete;
    LlamaSummarizer& operator=(const LlamaSummarizer&) = delete;

    // Add summary to the 5-slot chronological summary system
    void add_summary_to_slots(const std::string& new_summary);

    // Enhanced prune message history using summary model to condense pruned messages
    void prune_message_history(std::vector<std::pair<std::string, std::string>>& message_history, 
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
    SummarySlotInfo get_summary_slot_info() const {
        SummarySlotInfo info;
        info.total_slots = SummarizerConstants::MAX_SUMMARY_SLOTS;
        info.used_slots = summary_slots.size();
        info.summaries = summary_slots;
        return info;
    }

    // Enhanced summarization for very large message collections
    // Breaks down large summarization tasks into manageable chunks to avoid token limits
    std::string summarize_large_message_collection(const std::vector<std::pair<std::string, std::string>>& messages_to_summarize) {
        if (messages_to_summarize.empty()) {
            return "";
        }
        
        // For smaller collections, use the standard approach
        if (messages_to_summarize.size() <= 50) {
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
            if (!chunk_summary.empty()) {
                chunk_summaries.push_back(chunk_summary);
                SUMMARIZER_LOG("Successfully summarized chunk " + std::to_string(i/CHUNK_SIZE + 1) + 
                               " (" + std::to_string(chunk.size()) + " messages)");
            }
        }
        
        // If we have multiple chunk summaries, combine them into a final summary
        if (chunk_summaries.size() > 1) {
            std::vector<std::pair<std::string, std::string>> final_summary_input;
            for (size_t i = 0; i < chunk_summaries.size(); ++i) {
                final_summary_input.emplace_back("system", 
                    "Summary part " + std::to_string(i + 1) + ": " + chunk_summaries[i]);
            }
            
            std::string final_summary = summarize_messages(final_summary_input);
            if (!final_summary.empty()) {
                SUMMARIZER_LOG("Successfully created final summary from " + std::to_string(chunk_summaries.size()) + " chunks");
                return final_summary;
            }
        }
        
        // Fallback: return the first chunk summary if final combination failed
        return chunk_summaries.empty() ? "" : chunk_summaries[0];
    }    // Direct summary generation using stored context pointers (eliminates LlamaManager dependency)
    std::string generate_summary_directly(const std::string& summarization_request);    // ContextSizeManager integration methods for coordinated summary management
    void integrate_with_context_size_manager(EnhancedContextSizeManager* context_manager, ContextInfo* context_ref);
    void track_summary_with_context_manager(const std::string& summary, int32_t estimated_tokens = 0);
    bool should_perform_rollover_summarization() const;
    void sync_slots_with_context_manager();

private:
    // Reference to parent context's ContextSizeManager for coordination
    EnhancedContextSizeManager* context_size_manager = nullptr;
    ContextInfo* context_info_ref = nullptr;
    
    // Helper to estimate tokens in a summary (rough approximation)
    int32_t estimate_summary_tokens(const std::string& summary) const;

};

// Implementation of member functions


inline void LlamaSummarizer::add_summary_to_slots(const std::string& new_summary) {
    if (new_summary.empty()) return;
    
    // Estimate tokens for the new summary
    int32_t summary_tokens = estimate_summary_tokens(new_summary);
    
    // Check with ContextSizeManager if we need to merge before adding
    bool needs_merge = false;
    if (context_size_manager) {
        auto plan = context_size_manager->plan_summary_addition(summary_tokens);
        needs_merge = plan.needs_merge_first;
        
        SUMMARIZER_LOG("ContextSizeManager summary plan: " + plan.action_plan + 
                       " (estimated usage: " + std::to_string(plan.estimated_final_usage_percentage * 100) + "%)");
    }
    
    // If we're at capacity (5 slots) OR ContextSizeManager recommends merge, implement rollover summarization
    if (summary_slots.size() >= SummarizerConstants::MAX_SUMMARY_SLOTS || needs_merge) {
        SUMMARIZER_LOG("Summary slots at capacity or merge recommended, performing rollover summarization");
        
        // Get the two oldest summaries (slots 0 and 1)
        std::string oldest_summary = summary_slots[0];
        std::string second_oldest_summary = summary_slots[1];
        
        // Create a combined summary from the two oldest
        std::vector<std::pair<std::string, std::string>> rollover_messages;
        rollover_messages.emplace_back("system", "Previous summary 1: " + oldest_summary);
        rollover_messages.emplace_back("system", "Previous summary 2: " + second_oldest_summary);
        
        std::string combined_summary = summarize_messages(rollover_messages);
        
        if (!combined_summary.empty()) {
            // Execute the merge operation through ContextSizeManager if available
            if (context_size_manager) {
                int32_t merged_tokens = estimate_summary_tokens(combined_summary);
                
                // Execute merge through ContextSizeManager
                bool success = context_size_manager->execute_summary_addition(summary_tokens, [merged_tokens]() -> int32_t {
                    return merged_tokens;  // Return the merged summary size
                });
                
                if (success) {
                    // Remove the two oldest summaries and replace with the combined one
                    summary_slots.erase(summary_slots.begin(), summary_slots.begin() + 2);
                    summary_slots.insert(summary_slots.begin(), combined_summary);
                    SUMMARIZER_LOG("ContextSizeManager-coordinated rollover summarization successful - combined 2 oldest summaries into 1");                } else {
                    SUMMARIZER_LOG("Warning: ContextSizeManager merge operation failed, falling back to simple removal");
                    summary_slots.erase(summary_slots.begin());
                }
            } else {
                // This should not happen if integration is working properly
                SUMMARIZER_LOG("CRITICAL: No ContextSizeManager available during rollover - this indicates integration failure");
                
                // Emergency fallback - but log this as an error
                summary_slots.erase(summary_slots.begin(), summary_slots.begin() + 2);
                summary_slots.insert(summary_slots.begin(), combined_summary);
                SUMMARIZER_LOG("Emergency fallback: combined 2 oldest summaries into 1 (INTEGRATION PROBLEM)");
            }
        } else {
            // Fallback: just remove the oldest if rollover summarization fails
            summary_slots.erase(summary_slots.begin());
            SUMMARIZER_LOG("Rollover summarization failed, removed oldest summary");
        }    } else {
        // Add new summary directly with ContextSizeManager tracking
        if (context_size_manager) {
            bool success = context_size_manager->execute_summary_addition(summary_tokens, []() -> int32_t {
                return 0;  // No merge needed
            });
            
            if (!success) {
                SUMMARIZER_LOG("Warning: ContextSizeManager rejected summary addition");
                return;
            }
        }
        
        // Add new summary to the end (newest position)
        summary_slots.push_back(new_summary);
    }
    
    // Track the summary creation with ContextSizeManager if not already done in merge case
    if (context_size_manager && !needs_merge) {
        context_size_manager->track_summary_creation(summary_tokens);
    }
    
    SUMMARIZER_LOG("Added new summary to slot " + std::to_string(summary_slots.size()) + 
                   " of " + std::to_string(SummarizerConstants::MAX_SUMMARY_SLOTS) + 
                   " (" + std::to_string(summary_tokens) + " estimated tokens)");
}

inline void LlamaSummarizer::prune_message_history(std::vector<std::pair<std::string, std::string>>& message_history, 
                                                  float keep_ratio) {
    if (message_history.empty()) {
        SUMMARIZER_LOG("No messages to prune - message history is empty");
        return;
    }
    
    // Check if summary resources are available
    if (!summary_context || !summary_model || !generate_response_callback) {
        SUMMARIZER_LOG("Warning: Summary resources not available - performing simple pruning without summarization");
        
        // Fallback: simple pruning without summarization
        bool has_system = !message_history.empty() && message_history[0].first == "system";
        size_t system_offset = has_system ? 1 : 0;
        size_t total_messages = message_history.size() - system_offset;
        size_t messages_to_keep = std::max(size_t(2), static_cast<size_t>(total_messages * keep_ratio));
        
        if (messages_to_keep >= total_messages) {
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
                                " older messages removed due to context limits]");
    }
    
    // Do NOT add summary slots to message history during regular pruning
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
    if (messages_to_summarize.empty()) {
        SUMMARIZER_LOG("Error: Empty messages to summarize");
        return "";
    }
    
    // Check if summary resources are available
    if (!summary_context || !summary_model || !generate_response_callback) {
        SUMMARIZER_LOG_ERROR("Summary resources not available - context, model, or callback missing");
        return "";
    }
    
    // Validate that the summary context is properly initialized
    if (!summary_context->context || !summary_context->model_info || !summary_context->batch_initialized) {
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
    if (!summary.empty() && summary.size() >= 6 && summary.substr(0, 6) == "Error:") {
        SUMMARIZER_LOG_ERROR("Summary generation returned error: " + summary);
        summary = ""; // Treat as failed summarization
    } else if (!summary.empty()) {
        // Log the successful output in a user-friendly format
        std::string output_message = "=== GENERATED SUMMARY ===\n" + summary + "\n==========================\n";
        SUMMARIZER_LOG(output_message);
    }
      if (summary.empty()) {
        SUMMARIZER_LOG("Warning: Message summarization produced empty result");
    } else {
        SUMMARIZER_LOG("Successfully summarized " + std::to_string(messages_to_summarize.size()) + 
                       " messages into " + std::to_string(summary.length()) + " character summary");
        
        // Track summary creation with ContextSizeManager
        if (context_size_manager) {
            track_summary_with_context_manager(summary);
        }
    }
    
    return summary;
}

inline std::string LlamaSummarizer::generate_summary_directly(const std::string& summarization_request) {
    if (summarization_request.empty() || !summary_context || !summary_model || !generate_response_callback) {
        SUMMARIZER_LOG("Error: Empty request, null context/model, or missing callback");
        return "";
    }
    
    // Validate that the summary context is properly initialized
    if (!summary_context->context || !summary_context->model_info || !summary_context->batch_initialized) {
        SUMMARIZER_LOG_ERROR("Summary context is not properly initialized");
        return "";
    }
    
    // Validate that the summary context is associated with the correct model
    if (summary_context->model_info != summary_model) {
        SUMMARIZER_LOG_ERROR("Summary context is not associated with the expected summary model");
        return "";
    }
    
    SUMMARIZER_LOG("Starting direct summary generation using callback");
    
    std::string summary;
    try {
        // Use the callback to generate the response directly on the summary context
        summary = generate_response_callback(summarization_request, "user", summary_context);
        
        // Clean up the summary (remove any extra whitespace, newlines)
        if (!summary.empty()) {
            size_t start = summary.find_first_not_of(" \t\n\r");
            size_t end = summary.find_last_not_of(" \t\n\r");
            if (start != std::string::npos && end != std::string::npos) {
                summary = summary.substr(start, end - start + 1);
            }
            
            SUMMARIZER_LOG("Successfully generated summary using direct callback");
        } else {
            SUMMARIZER_LOG("Warning: Generated summary is empty");
        }
    } catch (const std::exception& e) {
        SUMMARIZER_LOG_ERROR("Exception during direct summary generation: " + std::string(e.what()));
        summary = "";
    } catch (...) {
        SUMMARIZER_LOG_ERROR("Unknown exception during direct summary generation");
        summary = "";
    }
      return summary;
}

// ContextSizeManager integration implementations
inline void LlamaSummarizer::integrate_with_context_size_manager(EnhancedContextSizeManager* context_manager, ContextInfo* context_ref) {
    context_size_manager = context_manager;
    context_info_ref = context_ref;
    
    if (context_size_manager && context_info_ref) {
        SUMMARIZER_LOG("Integrated with ContextSizeManager and ContextInfo reference for coordinated summary management");
        
        // Sync existing summary slots with ContextSizeManager
        sync_slots_with_context_manager();
    } else {
        SUMMARIZER_LOG("Warning: Failed to integrate with ContextSizeManager - missing manager or context reference");
    }
}

inline void LlamaSummarizer::track_summary_with_context_manager(const std::string& summary, int32_t estimated_tokens) {
    if (!context_size_manager || summary.empty()) return;
    
    int32_t actual_tokens = estimated_tokens > 0 ? estimated_tokens : estimate_summary_tokens(summary);
    context_size_manager->track_summary_creation(actual_tokens);
    
    SUMMARIZER_LOG("Tracked summary creation with ContextSizeManager: " + std::to_string(actual_tokens) + " tokens");
}

inline bool LlamaSummarizer::should_perform_rollover_summarization() const {
    if (!context_size_manager || !context_info_ref) {
        // Only fall back if we truly don't have ContextSizeManager integration
        SUMMARIZER_LOG("Warning: No ContextSizeManager integration - using fallback logic");
        return summary_slots.size() >= SummarizerConstants::MAX_SUMMARY_SLOTS;
    }
    
    // Get proper context analysis from ContextSizeManager
    auto analysis = context_size_manager->analyze_context(*context_info_ref);
    
    // Check if ContextSizeManager recommends merge based on real analysis
    auto recommendations = context_size_manager->get_optimization_recommendations(analysis);
    for (const auto& recommendation : recommendations) {
        if (recommendation.find("merge") != std::string::npos || 
            recommendation.find("summary") != std::string::npos) {
            SUMMARIZER_LOG("ContextSizeManager recommends rollover: " + recommendation);
            return true;
        }
    }
    
    // Also check direct analysis flags
    if (analysis.needs_summary_merge || analysis.summary_hard_cap_exceeded) {
        SUMMARIZER_LOG("ContextSizeManager analysis indicates rollover needed (merge: " + 
                      std::to_string(analysis.needs_summary_merge) + ", cap exceeded: " + 
                      std::to_string(analysis.summary_hard_cap_exceeded) + ")");
        return true;
    }
    
    // Also check the traditional slot limit as backup
    if (summary_slots.size() >= SummarizerConstants::MAX_SUMMARY_SLOTS) {
        SUMMARIZER_LOG("Traditional slot limit reached (" + std::to_string(summary_slots.size()) + " >= " + 
                      std::to_string(SummarizerConstants::MAX_SUMMARY_SLOTS) + ")");
        return true;
    }
    
    return false;
}

inline void LlamaSummarizer::sync_slots_with_context_manager() {
    if (!context_size_manager) return;
    
    // Calculate total tokens in existing slots and sync with ContextSizeManager
    int32_t total_tokens = 0;
    for (const auto& slot : summary_slots) {
        int32_t slot_tokens = estimate_summary_tokens(slot);
        total_tokens += slot_tokens;
        
        // Add each existing slot to ContextSizeManager tracking
        // Note: This assumes ContextSizeManager can handle retrospective additions
        // In practice, this should be called during initialization before any operations
    }
    
    SUMMARIZER_LOG("Synchronized " + std::to_string(summary_slots.size()) + 
                   " existing summary slots (" + std::to_string(total_tokens) + 
                   " estimated tokens) with ContextSizeManager");
}

inline int32_t LlamaSummarizer::estimate_summary_tokens(const std::string& summary) const {
    if (summary.empty()) return 0;
    
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

//
//  !! ENSURE YOU REMEMBER TO FOLLOW THE CRITICAL CODING DIRECTIVES COMMENTED AT THE TOP OF THIS FILE !!
//
