#pragma once

#include "BasePlugin.hpp"
#include <unordered_set>

namespace LuminaChat {

// Request and response structures for the pipeline
struct SummarizationRequest {
    PrunedMessageBatch batch;
    std::string context_id;
    RequestPriority priority = RequestPriority::NORMAL;
    std::chrono::steady_clock::time_point queued_time;
    
    SummarizationRequest(PrunedMessageBatch b, const std::string& ctx_id, RequestPriority prio = RequestPriority::NORMAL)
        : batch(std::move(b)), context_id(ctx_id), priority(prio), queued_time(std::chrono::steady_clock::now()) {}
};

struct SummarizationResponse {
    std::string summary;
    std::string context_id;
    bool success;
    size_t messages_processed;
    std::string error_message;
    
    SummarizationResponse(const std::string& sum, const std::string& ctx_id, bool succ, size_t msg_count, const std::string& err = "")
        : summary(sum), context_id(ctx_id), success(succ), messages_processed(msg_count), error_message(err) {}
};

/**
 * Summarization Plugin - Provides summarization processing capabilities
 * 
 * This plugin serves as a pure processor for the Orchestrator's summarization pipeline.
 * It handles model initialization, context management, and the actual summarization work.
 * 
 * Architecture:
 * - Orchestrator coordinates workflow (scheduling, pipeline management)
 * - Plugin provides processing capability (model management, summarization logic)
 * - Orchestrator calls plugin methods through its pipeline processor
 * - Plugin applies results back to contexts when requested
 */
class SummarizationPlugin : public BasePlugin<PrunedMessageBatch, SummarizationRequest, SummarizationResponse> {
private:
    // Plugin-specific constants
    static constexpr const char* PLUGIN_NAME = "SummarizationPlugin";
    static constexpr const char* MODEL_ID = "summary_model";
    static constexpr const char* CONTEXT_ID = "plugin_summary_context";
    static constexpr const char* DEFAULT_SYSTEM_PROMPT = 
        "You are a helpful AI assistant that creates concise summaries of conversations. "
        "When given a conversation history, provide a clear and informative summary that captures "
        "the key points, decisions, and context. Focus on preserving important information while "
        "being concise. Format your summary in a structured way with bullet points when appropriate.";

public:
    explicit SummarizationPlugin(Orchestrator* orch) 
        : BasePlugin(orch, PLUGIN_NAME, MODEL_ID, CONTEXT_ID, DEFAULT_SYSTEM_PROMPT) {
        LogInfo("SummarizationPlugin initialized as processor service");
    }
    
    SummarizationResponse ProcessSummarizationRequest(const SummarizationRequest& request) {
        return ProcessRequest(request);
    }
    
    // Delegate to base class buffer management with clear naming
    std::vector<PrunedMessageBatch> GetAndClearPruningBuffer() {
        return GetAndClearBuffer();
    }
    
    bool HasPendingSummarization() const {
        return HasPendingWork();
    }
    
    void RequestSummarization(const std::string& context_id, const std::vector<std::pair<std::string, std::string>>& pruned_messages) {
        if (pruned_messages.empty()) return;
        
        LogInfo("Queuing " + std::to_string(pruned_messages.size()) + " messages for summarization: " + context_id);
        AddToBuffer(PrunedMessageBatch(context_id, pruned_messages));
    }
    
    // Specialized application methods
    bool ApplySummaryToContext(const std::string& context_id, const std::string& summary) {
        return ApplyResult(context_id, summary);
    }
    
    bool ApplyInnerVoiceSummary(const std::string& summary_id, const std::string& summary) {
        if (!llama_manager || summary.empty()) {
            LogError("LlamaManager unavailable or empty inner voice summary");
            return false;
        }
        
        try {
            // Extract original context ID: "context_inner_voice_summary" -> "context"
            size_t suffix_pos = summary_id.find("_inner_voice_summary");
            std::string original_context_id = (suffix_pos != std::string::npos) ? 
                summary_id.substr(0, suffix_pos) : summary_id;
            
            auto* context = llama_manager->GetContextInfo(original_context_id);
            if (!context) {
                LogWarning("Inner context not found: " + original_context_id);
                return false;
            }
            
            // Apply summary with inner voice-specific lock
            if (!context->TryAcquirePluginProcessing("SummarizationPlugin_InnerVoice")) {
                LogWarning("Inner voice context busy: " + original_context_id);
                return false;
            }
            
            context->ApplyCompletedSummary(summary);
            [[maybe_unused]] bool release_result = context->ReleasePluginProcessing("SummarizationPlugin_InnerVoice");
            
            stats.IncrementContextUpdate();
            LogInfo("Applied inner voice summary to: " + original_context_id + " - " + summary.substr(0, 100) + "...");
            return true;
            
        } catch (const std::exception& e) {
            LogError("Exception applying inner voice summary: " + std::string(e.what()));
            return false;
        }
    }
    
    bool ApplyCompressedSummary(const std::string& compression_id, const std::string& compressed_summary) {
        if (!llama_manager || compressed_summary.empty()) {
            LogError("LlamaManager unavailable or empty compressed summary");
            return false;
        }
        
        try {
            // Extract original context ID: "context_summary_compression" -> "context"
            size_t suffix_pos = compression_id.find("_summary_compression");
            std::string original_context_id = (suffix_pos != std::string::npos) ? 
                compression_id.substr(0, suffix_pos) : compression_id;
            
            auto* context = llama_manager->GetContextInfo(original_context_id);
            if (!context) {
                LogWarning("Context not found for compressed summary: " + original_context_id);
                return false;
            }
            
            // Apply compressed summary with compression-specific lock
            if (!context->TryAcquirePluginProcessing("SummarizationPlugin_Compression")) {
                LogWarning("Context busy for compression result: " + original_context_id);
                return false;
            }
            
            context->ApplyCompletedSummary(compressed_summary);
            [[maybe_unused]] bool release_result = context->ReleasePluginProcessing("SummarizationPlugin_Compression");
            
            stats.IncrementContextUpdate();
            LogInfo("Applied compressed summary to: " + original_context_id + " - " + compressed_summary.substr(0, 100) + "...");
            return true;
            
        } catch (const std::exception& e) {
            LogError("Exception applying compressed summary: " + std::string(e.what()));
            return false;
        }
    }
    
    // Simplified debug access using base class
    std::optional<std::string> GetLastGeneration() const {
        auto gen_info = BasePlugin::GetLastGeneration();
        if (!gen_info.has_generation) return {};
        
        return gen_info.context_id + " [" + gen_info.timestamp + "]: " + 
               gen_info.input.substr(0, 100) + " -> " + 
               gen_info.output.substr(0, 100);
    }

    void Shutdown() {
        BasePlugin::Shutdown();
        LogInfo("SummarizationPlugin shutdown");
    }

protected:
    // BasePlugin virtual method implementations
    SummarizationResponse ProcessRequest(const SummarizationRequest& request) override {
        if (!model_ready.load() || !llama_manager) {
            const char* error_msg = "Summary model not ready for processing batch";
            LogError(error_msg);
            return {error_msg, request.context_id, false, 0, error_msg};
        }
        
        try {
            LogInfo("Processing " + std::to_string(request.batch.pruned_messages.size()) + " messages for: " + request.context_id);
            
            std::string summary = ProcessSummarizationBatch(request.batch);
            if (summary.empty()) {
                const char* error_msg = "Failed to generate summary";
                LogError(error_msg);
                return {"", request.context_id, false, 0, error_msg};
            }
            
            // Update statistics and return success
            stats.IncrementBatch(request.batch.pruned_messages.size());
            LogInfo("Successfully generated summary for: " + request.context_id);
            return {summary, request.context_id, true, request.batch.pruned_messages.size(), ""};
            
        } catch (const std::exception& e) {
            std::string error_msg = "Exception: " + std::string(e.what());
            LogError(error_msg);
            return {"", request.context_id, false, 0, error_msg};
        }
    }
    
    bool ApplyResult(const std::string& context_id, const std::string& summary) override {
        if (!llama_manager || summary.empty()) {
            LogError("LlamaManager unavailable or empty summary");
            return false;
        }
        
        try {
            // Handle inner voice summaries with special processing
            if (context_id.find("_inner_voice_summary") != std::string::npos) {
                return ApplyInnerVoiceSummary(context_id, summary);
            }
            
            // Handle hierarchical summary compression results
            if (context_id.find("_summary_compression") != std::string::npos) {
                return ApplyCompressedSummary(context_id, summary);
            }
            
            auto* context = llama_manager->GetContextInfo(context_id);
            if (!context) {
                LogWarning("Context not found: " + context_id);
                return false;
            }
            
            // Apply summary with plugin lock protection
            if (!context->TryAcquirePluginProcessing("SummarizationPlugin")) {
                LogWarning("Context busy, cannot apply summary: " + context_id);
                return false;
            }
            
            context->ApplyCompletedSummary(summary);
            [[maybe_unused]] bool release_result = context->ReleasePluginProcessing("SummarizationPlugin");
            
            stats.IncrementContextUpdate();
            LogInfo("Applied summary to: " + context_id);
            return true;
            
        } catch (const std::exception& e) {
            LogError("Exception applying summary: " + std::string(e.what()));
            return false;
        }
    }
    
    std::string GetModelPath() const override {
        return settings_manager ? settings_manager->GetString("Models", "summary_model_path", "") : "";
    }
    
    std::string GetModelPathSettingKey() const override {
        return "summary_model_path";
    }
    
    std::string GetSystemPromptSettingKey() const override {
        return "Summary.system_prompt";
    }
    
    std::string GetPluginName() const override {
        return PLUGIN_NAME;
    }
    
    // Override logging to use plugin-specific macros
    void LogInfo(const std::string& message) {
        LOG_SummarizationPlugin(message);
        debug.AddLog("INFO", message);
    }
    
    void LogWarning(const std::string& message) {
        LOG_WARNING_SummarizationPlugin(message);
        debug.AddLog("WARN", message);
        HandleWarning(message, "SummarizationPlugin", false);
    }
    
    void LogError(const std::string& message) {
        LOG_ERROR_SummarizationPlugin(message);
        debug.AddLog("ERROR", message);
        HandleError(message, "SummarizationPlugin", false);
    }
    
    void LogDebug(const std::string& message) {
        LOG_DEBUG_SummarizationPlugin(message);
        debug.AddLog("DEBUG", message);
    }

private:
    // Plugin-specific processing logic
    std::string ProcessSummarizationBatch(const PrunedMessageBatch& batch) {
        if (!model_ready.load() || !llama_manager) {
            LogError("Summary model not ready");
            return "";
        }
        
        // Check if this is a summary compression request
        bool is_compression = (batch.context_id.find("_summary_compression") != std::string::npos);
        
        if (is_compression) {
            return ProcessSummaryCompression(batch);
        }
        
        // Regular message pair processing
        size_t total_messages = batch.pruned_messages.size();
        if (total_messages < 2) {
            LogInfo("Only " + std::to_string(total_messages) + " message(s) - skipping summarization");
            return "";
        }
        
        try {
            LogInfo("Processing " + std::to_string(total_messages / 2) + " message pairs");
            std::vector<std::string> pair_summaries;
            
            // Process message pairs
            for (size_t i = 0; i + 1 < total_messages; i += 2) {
                std::string pair_summary = ProcessMessagePair(batch, i);
                if (!pair_summary.empty()) {
                    pair_summaries.push_back(pair_summary);
                    LogInfo("Summarized pair " + std::to_string((i/2) + 1) + ": " + pair_summary.substr(0, 50) + "...");
                }
            }
            
            // Handle odd message
            if (total_messages % 2 == 1) {
                LogInfo("Leaving unpaired message in context");
            }
            
            // Combine summaries
            if (pair_summaries.empty()) {
                LogError("No pairs could be summarized");
                return "";
            }
            
            if (pair_summaries.size() == 1) {
                return "Summary: " + pair_summaries[0];
            }
            
            std::ostringstream final_stream;
            for (const auto& summary : pair_summaries) {
                final_stream << "* " << summary << "\n";
            }
            
            LogInfo("Successfully processed " + std::to_string(pair_summaries.size()) + " pairs");
            return final_stream.str();
            
        } catch (const std::exception& e) {
            LogError("Exception processing batch: " + std::string(e.what()));
            return "";
        }
    }
    
    std::string ProcessMessagePair(const PrunedMessageBatch& batch, size_t index) {
        std::ostringstream prompt;
        prompt << "Summarize this Q&A exchange in one clear sentence:\n\n"
               << batch.pruned_messages[index].first << ": " << batch.pruned_messages[index].second << "\n"
               << batch.pruned_messages[index + 1].first << ": " << batch.pruned_messages[index + 1].second << "\n"
               << "\nSummary:";
        
        std::string pair_prompt = prompt.str();
        std::string pair_summary = PerformGeneration(pair_prompt, batch.context_id);
        
        // Handle errors with context resize if needed
        if (pair_summary.empty() || PluginUtils::IsErrorResponse(pair_summary)) {
            LogWarning("Failed to summarize pair " + std::to_string((index/2) + 1) + " - attempting resize");
            return AttemptGenerationWithResize(pair_prompt, batch.context_id);
        }
        
        return pair_summary;
    }
    
    std::string ProcessSummaryCompression(const PrunedMessageBatch& batch) {
        LogInfo("Processing hierarchical summary compression for context: " + batch.context_id);
        
        try {
            // Build compression prompt from the multiple summaries
            std::ostringstream compression_prompt;
            compression_prompt << "Combine and compress the following summaries into a single, comprehensive summary that preserves the key information from both:\n\n";
            
            for (size_t i = 0; i < batch.pruned_messages.size(); ++i) {
                compression_prompt << batch.pruned_messages[i].first << ": " << batch.pruned_messages[i].second << "\n\n";
            }
            
            compression_prompt << "Combined Summary:";
            std::string prompt = compression_prompt.str();
            
            LogInfo("Built compression prompt (" + std::to_string(prompt.length()) + " chars) for context: " + batch.context_id);
            
            // Perform compression using base class generation method
            std::string compressed_result = PerformGeneration(prompt, batch.context_id);
            
            if (!compressed_result.empty() && !PluginUtils::IsErrorResponse(compressed_result)) {
                LogInfo("Successfully compressed summaries for context: " + batch.context_id + 
                       " - Result: " + compressed_result.substr(0, 100) + "...");
                return compressed_result;
            } else {
                LogError("Failed to compress summaries for context: " + batch.context_id);
                return "";
            }
            
        } catch (const std::exception& e) {
            LogError("Exception in summary compression: " + std::string(e.what()));
            return "";
        }
    }
}; // End of SummarizationPlugin class

} // namespace LuminaChat

// CRITICAL: These function definitions are placed here to combat circular dependencies
inline void Orchestrator::RequestSummarization(const PrunedMessageBatch& batch) {
    LOG_Orchestrator("Requesting summarization for context: " + batch.context_id);
    
    // Check if context is already being summarized
    if (GetContextState(batch.context_id) == ProcessingState::AWAITING_SUMMARIZATION) {
        LOG_Orchestrator("Context already awaiting summarization: " + batch.context_id);
        return;
    }
    
    // Set state to awaiting summarization
    SetContextState(batch.context_id, ProcessingState::AWAITING_SUMMARIZATION);
    
    // Create plugin's summarization request using the proper batch
    LuminaChat::SummarizationRequest request(batch, batch.context_id);
    
    // Queue for processing
    [[maybe_unused]] auto queue_result = summarization_pipeline.QueueRequest(request);
}

inline void Orchestrator::OnSummarizationComplete(const std::string& context_id, 
                                                 const LuminaChat::SummarizationResponse& response) {
    LOG_Orchestrator("Summarization complete for context: " + context_id);
    
    if (response.success) {
        // Use the SummarizationPlugin's ApplySummaryToContext method
        // This will handle both regular and inner voice summaries appropriately
        if (summarization_plugin && summarization_plugin->ApplySummaryToContext(context_id, response.summary)) {
            LOG_Orchestrator("Summary applied to context: " + context_id);
        } else {
            LOG_ERROR_Orchestrator("Failed to apply summary to context: " + context_id);
            SetContextState(context_id, ProcessingState::ERROR_STATE);
            return;
        }
        
        // Update statistics using lock-free atomic increment
        stats.summarizations_completed.fetch_add(1, std::memory_order_relaxed);
    } else {
        LOG_ERROR_Orchestrator("Summarization failed: " + response.error_message);
    }
    
    // Return context to normal processing
    SetContextState(context_id, ProcessingState::NORMAL_PROCESSING);
}

inline void Orchestrator::ProcessSummarizationRequest(const LuminaChat::SummarizationRequest& request, 
                                                     std::function<void(LuminaChat::SummarizationResponse)> callback) {
    LOG_Orchestrator("Processing summarization request for: " + request.context_id);
    
    try {
        // Delegate to SummarizationPlugin - plugin is required
        if (!summarization_plugin) {
            LOG_ERROR_Orchestrator("SummarizationPlugin not available - summarization requires plugin delegation");
            
            LuminaChat::SummarizationResponse error_response("", request.context_id, false, 0, 
                                                            "SummarizationPlugin not available - summarization requires plugin delegation");
            
            callback(error_response);
            OnSummarizationComplete(request.context_id, error_response);
            return;
        }
        
        LOG_Orchestrator("Delegating summarization to SummarizationPlugin for context: " + request.context_id);
        LuminaChat::SummarizationResponse response = summarization_plugin->ProcessSummarizationRequest(request);
        
        callback(response);
        
        // Trigger completion callback for the original orchestrator workflow
        OnSummarizationComplete(request.context_id, response);
        
    } catch (const std::exception& e) {
        LuminaChat::SummarizationResponse error_response("", request.context_id, false, 0, e.what());
        callback(error_response);
        OnSummarizationComplete(request.context_id, error_response);
    }
}

// CRITICAL: Orchestrator method implementations moved here to combat circular dependencies
inline void Orchestrator::ProcessPruningBuffer() {
    LOG_DEBUG_Orchestrator("ProcessPruningBuffer: Checking for pending summarization work...");
    
    // Check if summarization plugin has pending messages
    if (!summarization_plugin || !summarization_plugin->HasPendingSummarization()) [[likely]] {
        LOG_DEBUG_Orchestrator("ProcessPruningBuffer: No pending summarization work found");
        return; // No work to do
    }
    
    LOG_Orchestrator("Processing pruning buffer...");
    
    // Get all pending pruning batches from plugin
    auto pruning_batches = summarization_plugin->GetAndClearPruningBuffer();
    
    LOG_Orchestrator("Found " + std::to_string(pruning_batches.size()) + " pruning batches to process");
    
    // Process each batch through the summarization pipeline
    for (const auto& batch : pruning_batches) [[likely]] {
        if (!batch.needs_summarization) [[unlikely]] {
            continue; // Skip batches that don't need summarization
        }
        
        // Request summarization for this batch - most batches will need summarization
        RequestSummarization(batch);
        
        // Update statistics using lock-free atomic increment
        stats.summarizations_completed.fetch_add(1, std::memory_order_relaxed);
    }
    
    LOG_Orchestrator("Pruning buffer processing complete - processed " + 
                    std::to_string(pruning_batches.size()) + " batches");
}

// Orchestrator helper method for requesting summarization (moved here to avoid circular dependencies)
inline void Orchestrator::RequestSummarizationForPrunedMessages(const std::string& context_id, const std::vector<std::pair<std::string, std::string>>& pruned_messages) {
    if (!summarization_plugin || !summarization_plugin_available.load()) {
        LOG_WARNING_Orchestrator("SummarizationPlugin not available - pruned messages will be lost for context: " + context_id);
        return;
    }
    
    LOG_Orchestrator("Requesting summarization for " + std::to_string(pruned_messages.size()) + 
                    " pruned messages from context: " + context_id);
    
    summarization_plugin->RequestSummarization(context_id, pruned_messages);
}

// CRITICAL: ProcessInnerVoiceAnalysis implementation moved here to combat circular dependencies
inline void Orchestrator::ProcessInnerVoiceAnalysis() {
    LOG_DEBUG_Orchestrator("ProcessInnerVoiceAnalysis: Checking inner contexts for user input/AI output pairs...");
    
    if (!llama_manager || !summarization_plugin || !summarization_plugin_available.load()) {
        LOG_DEBUG_Orchestrator("ProcessInnerVoiceAnalysis: Required components not available");
        return;
    }
    
    // Track if there's currently a summarization in progress for each context to avoid overwhelming the pipeline
    static std::unordered_set<std::string> contexts_being_summarized;
    
    // Get all contexts that might be inner voice contexts
    // Look for contexts with "inner" in their ID or check for specific inner context patterns
    std::vector<std::string> inner_context_candidates = {"inner_context"};
    
    // Also check for any context IDs that contain "inner" (for future extensibility)
    // This could be expanded later if multiple inner contexts are supported
    
    for (const std::string& context_id : inner_context_candidates) {
        auto* context = llama_manager->GetContextInfo(context_id);
        if (!context) {
            continue; // Context doesn't exist
        }
        
        // Check if this context is already being summarized - if so, skip to avoid overwhelming the pipeline
        std::string summary_id = context_id + "_inner_voice_summary";
        if (GetContextState(summary_id) == ProcessingState::AWAITING_SUMMARIZATION) {
            LOG_DEBUG_Orchestrator("ProcessInnerVoiceAnalysis: Context " + context_id + " already has summarization in progress, skipping");
            continue;
        }
        
        // Get current message history
        const auto& message_history = context->GetMessageHistory();
        size_t current_message_count = message_history.size();
        
        // Always look for complete pairs from the beginning of the message history
        // This ensures we catch all pairs, even if previous summarizations were slow
        std::vector<std::pair<std::string, std::string>> available_pairs;
        
        LOG_DEBUG_Orchestrator("ProcessInnerVoiceAnalysis: Scanning " + std::to_string(current_message_count) + 
                              " messages in context: " + context_id + " for complete user/AI pairs");
        
        // Look for complete user input/AI output pairs from the beginning (oldest first)
        for (size_t i = 0; i + 1 < current_message_count; i += 2) {
            const auto& user_msg = message_history[i];
            const auto& ai_msg = message_history[i + 1];
            
            // Check if we have a user -> assistant pair
            if (user_msg.first == "user" && ai_msg.first == "assistant") {
                available_pairs.emplace_back(user_msg.second, ai_msg.second);
                LOG_DEBUG_Orchestrator("ProcessInnerVoiceAnalysis: Found user/AI pair at indices " + 
                                      std::to_string(i) + "-" + std::to_string(i+1) + " in " + context_id + 
                                      " - User: " + user_msg.second.substr(0, 50) + "..." +
                                      " - AI: " + ai_msg.second.substr(0, 50) + "...");
            }
        }
        
        // If we found complete pairs, process the oldest one first to ensure chronological order
        if (!available_pairs.empty()) {
            LOG_Orchestrator("ProcessInnerVoiceAnalysis: Found " + std::to_string(available_pairs.size()) + 
                           " complete pairs in context: " + context_id + " - processing oldest pair first");
            
            // Convert pairs to the format expected by SummarizationPlugin
            std::vector<std::pair<std::string, std::string>> formatted_pairs;
            // Take only the first (oldest) pair to process one at a time
            // This ensures we don't overwhelm the summarization pipeline and maintain chronological order
            const auto& [user_input, ai_output] = available_pairs[0];
                // Format as "User: <input>" and "Assistant: <output>" for better summarization context
                formatted_pairs.emplace_back("User", user_input);
                formatted_pairs.emplace_back("Assistant", ai_output);
            
            // Create a proper PrunedMessageBatch for inner voice summarization
            PrunedMessageBatch inner_voice_batch(context_id + "_inner_voice_summary", formatted_pairs);
            
            // Request summarization directly through the Orchestrator pipeline (not the plugin's buffer)
            RequestSummarization(inner_voice_batch);
            
            LOG_Orchestrator("ProcessInnerVoiceAnalysis: Queued oldest pair (1 of " + std::to_string(available_pairs.size()) + 
                           ") for summarization from inner voice context: " + context_id);
            
            // Remove only the processed pair from the message history (the first 2 messages)
            LOG_Orchestrator("ProcessInnerVoiceAnalysis: BEFORE removal - context " + context_id + 
                           " has " + std::to_string(context->GetMessageHistory().size()) + " messages");
            
            context->RemoveMessagePairs(0, 1); // Remove from index 0, count 1 pair (2 messages)
            
            LOG_Orchestrator("ProcessInnerVoiceAnalysis: AFTER removal - context " + context_id + 
                           " has " + std::to_string(context->GetMessageHistory().size()) + " messages");
            LOG_Orchestrator("ProcessInnerVoiceAnalysis: Removed oldest pair from message history in context: " + context_id);
            
            // Mark this context as having a summarization in progress by setting the summary context state
            SetContextState(context_id + "_inner_voice_summary", ProcessingState::AWAITING_SUMMARIZATION);
        } else {
            LOG_DEBUG_Orchestrator("ProcessInnerVoiceAnalysis: No complete pairs found in context: " + context_id);
        }
        
        LOG_DEBUG_Orchestrator("ProcessInnerVoiceAnalysis: Completed processing for context: " + context_id);
    }
}

// CRITICAL: ProcessSummarizationPipeline implementation to monitor pipeline health
inline void Orchestrator::ProcessSummarizationPipeline() {
    LOG_DEBUG_Orchestrator("ProcessSummarizationPipeline: Monitoring summarization pipeline health...");
    
    // Check if the summarization pipeline is shutdown (this shouldn't happen during normal operation)
    if (summarization_pipeline.IsShutdown()) {
        LOG_WARNING_Orchestrator("ProcessSummarizationPipeline: Summarization pipeline is shutdown, attempting to restart...");
        summarization_pipeline.Start();
        return;
    }
    
    // Get pipeline statistics for monitoring
    auto pipeline_stats = summarization_pipeline.GetStats();
    
    // Only log if there's actual activity or issues
    if (pipeline_stats.pending_requests > 0 || pipeline_stats.failed_requests > 0) {
        LOG_DEBUG_Orchestrator("ProcessSummarizationPipeline: Pipeline stats - Total: " + std::to_string(pipeline_stats.total_requests) +
                              ", Completed: " + std::to_string(pipeline_stats.completed_requests) +
                              ", Failed: " + std::to_string(pipeline_stats.failed_requests) +
                              ", Pending: " + std::to_string(pipeline_stats.pending_requests));
    }
    
    // Check for abnormal pipeline states that might indicate a problem
    if (pipeline_stats.pending_requests > 0 && summarization_pipeline.IsIdle()) {
        LOG_WARNING_Orchestrator("ProcessSummarizationPipeline: Pipeline shows " + std::to_string(pipeline_stats.pending_requests) + 
                                " pending requests but is idle - this may indicate a pipeline issue");
        
        // If we have a lot of stuck requests, this might indicate a real problem
        if (pipeline_stats.pending_requests > 10) {
            LOG_ERROR_Orchestrator("ProcessSummarizationPipeline: Too many stuck requests (" + 
                                  std::to_string(pipeline_stats.pending_requests) + "), pipeline may need attention");
        }
    } else if (pipeline_stats.pending_requests > 0 && summarization_pipeline.IsProcessing()) {
        LOG_DEBUG_Orchestrator("ProcessSummarizationPipeline: Pipeline is actively processing " + 
                              std::to_string(pipeline_stats.pending_requests) + " pending requests");
    }
    
    // Monitor for high failure rates
    if (pipeline_stats.failed_requests > 0 && pipeline_stats.total_requests > 0) {
        double failure_rate = static_cast<double>(pipeline_stats.failed_requests) / static_cast<double>(pipeline_stats.total_requests);
        if (failure_rate > 0.1) { // More than 10% failure rate
            LOG_WARNING_Orchestrator("ProcessSummarizationPipeline: High failure rate detected: " + 
                                    std::to_string(static_cast<int>(failure_rate * 100)) + "%");
        }
    }
    
    LOG_DEBUG_Orchestrator("ProcessSummarizationPipeline: Pipeline monitoring complete");
}