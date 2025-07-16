#pragma once

#include "BasePlugin.hpp"

namespace LuminaChat {

// Request and response structures for the pipeline
struct EmotionAnalysisRequest {
    EmotionalAnalysisBatch batch;
    std::string context_id;
    RequestPriority priority = RequestPriority::NORMAL;
    std::chrono::steady_clock::time_point queued_time;
    
    EmotionAnalysisRequest(EmotionalAnalysisBatch b, const std::string& ctx_id, RequestPriority prio = RequestPriority::NORMAL)
        : batch(std::move(b)), context_id(ctx_id), priority(prio), queued_time(std::chrono::steady_clock::now()) {}
};

struct EmotionAnalysisResponse {
    std::string emotional_state;
    std::string context_id;
    bool success;
    size_t responses_analyzed;
    std::string error_message;
    
    EmotionAnalysisResponse(const std::string& emotion, const std::string& ctx_id, bool succ, size_t resp_count, const std::string& err = "")
        : emotional_state(emotion), context_id(ctx_id), success(succ), responses_analyzed(resp_count), error_message(err) {}
};

/**
 * EmoTag Plugin - Provides emotional analysis processing capabilities
 * 
 * This plugin serves as a pure processor for the Orchestrator's emotion analysis pipeline.
 * It handles model initialization, context management, and the actual emotional analysis work.
 * 
 * Architecture:
 * - Orchestrator coordinates workflow (scheduling, pipeline management)
 * - Plugin provides processing capability (model management, emotional analysis logic)
 * - Orchestrator calls plugin methods through its pipeline processor
 * - Plugin applies results back to contexts when requested
 * 
 * Configuration:
 * - Analysis Window Size: Number of recent AI responses to analyze (default: 3)
 */
class EmoTagPlugin : public BasePlugin<EmotionalAnalysisBatch, EmotionAnalysisRequest, EmotionAnalysisResponse> {
private:
    // Plugin-specific constants
    static constexpr const char* PLUGIN_NAME = "EmoTagPlugin";
    static constexpr const char* MODEL_ID = "emotion_model";
    static constexpr const char* CONTEXT_ID = "plugin_emotion_context";
    static constexpr const char* DEFAULT_SYSTEM_PROMPT = 
        "You are an emotional state analyzer. When given AI assistant responses, analyze the emotional tone, "
        "mood, and psychological state conveyed in the text. Provide a brief emotional overview that captures "
        "the assistant's apparent emotional state, confidence level, and overall demeanor. "
        "Focus on identifying patterns like: confident, uncertain, empathetic, analytical, cheerful, "
        "cautious, enthusiastic, or reserved. Keep your analysis concise and actionable.";
    
    // Plugin-specific configuration
    std::atomic<size_t> analysis_window_size{3};

public:
    explicit EmoTagPlugin(Orchestrator* orch) 
        : BasePlugin(orch, PLUGIN_NAME, MODEL_ID, CONTEXT_ID, DEFAULT_SYSTEM_PROMPT) {
        LogInfo("EmoTagPlugin initialized as processor service");
    }
    
    void SetAnalysisWindow(size_t window_size) { analysis_window_size = window_size; }
    
    // Plugin-specific stats that include analysis window size
    struct EmoTagPluginStats {
        size_t batches_processed, responses_analyzed, emotional_states_generated, contexts_updated, analysis_window_size;
        bool emotion_model_ready;
        std::string emotion_model_path;
    };
    
    EmoTagPluginStats GetStats() const {
        auto base_stats = BasePlugin::GetStats();
        return {
            base_stats.batches_processed,
            base_stats.items_processed,  // responses_analyzed
            base_stats.results_generated, // emotional_states_generated
            base_stats.contexts_updated,
            analysis_window_size.load(),
            base_stats.model_ready,
            base_stats.model_path
        };
    }
    
    EmotionAnalysisResponse ProcessEmotionAnalysisRequest(const EmotionAnalysisRequest& request) {
        return ProcessRequest(request);
    }
    
    // Delegate to base class buffer management with clear naming
    std::vector<EmotionalAnalysisBatch> GetAndClearEmotionalAnalysisBuffer() {
        return GetAndClearBuffer();
    }
    
    bool HasPendingEmotionalAnalysis() const {
        return HasPendingWork();
    }
    
    void AddToEmotionalAnalysisBuffer(EmotionalAnalysisBatch&& batch) {
        AddToBuffer(std::move(batch));
    }
    
    /**
     * Request emotional analysis for a specific context
     */
    void RequestEmotionalAnalysis(const std::string& context_id, const std::vector<std::pair<std::string, std::string>>& message_history) {
        if (message_history.empty()) {
            LogInfo("No messages in history for emotional analysis request - context: " + context_id);
            return;
        }
        
        // Extract recent AI responses for analysis
        std::vector<std::string> ai_responses;
        const size_t max_responses = analysis_window_size.load();
        
        LogInfo("Searching for AI responses in " + std::to_string(message_history.size()) + " messages for context: " + context_id);
        
        // Walk backwards through message history to find AI responses
        for (auto it = message_history.rbegin(); 
             it != message_history.rend() && ai_responses.size() < max_responses; 
             ++it) {
            if (it->first == "assistant") {
                ai_responses.insert(ai_responses.begin(), it->second);
                LogInfo("Found assistant message (" + std::to_string(it->second.length()) + " chars) for emotional analysis");
            }
        }
        
        if (!ai_responses.empty()) {
            size_t response_count = ai_responses.size();
            AddToEmotionalAnalysisBuffer(EmotionalAnalysisBatch(context_id, std::move(ai_responses)));
            LogInfo("Requested emotional analysis for context " + context_id + 
                   " with " + std::to_string(response_count) + " AI responses");
        } else {
            LogInfo("No assistant messages found in " + std::to_string(message_history.size()) + " messages for context: " + context_id);
        }
    }
    
    // Specialized application method
    bool ApplyEmotionalStateToContext(const std::string& context_id, const std::string& emotional_state) {
        return ApplyResult(context_id, emotional_state);
    }

    void Shutdown() {
        BasePlugin::Shutdown();
        LogInfo("EmoTagPlugin shutdown");
    }

protected:
    // BasePlugin virtual method implementations
    EmotionAnalysisResponse ProcessRequest(const EmotionAnalysisRequest& request) override {
        LogInfo("Processing emotion analysis for context: " + request.context_id + 
               " (" + std::to_string(request.batch.ai_responses.size()) + " responses)");
        
        if (!model_ready.load() || !llama_manager) {
            std::string error_msg = "Emotion model not ready for processing batch";
            LogError(error_msg);
            return EmotionAnalysisResponse("", request.context_id, false, 0, error_msg);
        }
        
        try {
            std::string emotional_state = ProcessEmotionalAnalysisBatch(request.batch);
            
            if (emotional_state.empty()) {
                std::string error_msg = "Failed to generate emotional state for context: " + request.context_id;
                LogError(error_msg);
                return EmotionAnalysisResponse("", request.context_id, false, 0, error_msg);
            }
            
            stats.IncrementBatch(request.batch.ai_responses.size());
            LogInfo("Successfully generated emotional state for batch: " + request.context_id);
            
            return EmotionAnalysisResponse{
                emotional_state, request.context_id, true, request.batch.ai_responses.size()
            };
            
        } catch (const std::exception& e) {
            std::string error_msg = "Exception in emotion analysis processing: " + std::string(e.what());
            LogError(error_msg);
            return EmotionAnalysisResponse("", request.context_id, false, 0, error_msg);
        }
    }
    
    bool ApplyResult(const std::string& context_id, const std::string& emotional_state) override {
        if (!llama_manager || emotional_state.empty()) {
            LogError("LlamaManager not available or emotional state is empty");
            return false;
        }
        
        // Don't apply error states as valid emotional analysis
        if (PluginUtils::IsErrorResponse(emotional_state)) {
            LogWarning("Refusing to apply error response as emotional state for context: " + context_id + 
                      " - Error: " + emotional_state.substr(0, 100));
            return false;
        }
        
        try {
            auto* original_context = llama_manager->GetContextInfo(context_id);
            if (!original_context) {
                LogWarning("Original context not found for emotional state application: " + context_id);
                return false;
            }
            
            // Apply emotional state directly - UpdateEmotionalState is thread-safe
            original_context->UpdateEmotionalState(emotional_state);
            stats.IncrementContextUpdate();
            LogInfo("Applied emotional state to context: " + context_id);
            
            // Also apply to corresponding inner_voice context if applicable
            ApplyToInnerContext(context_id, emotional_state);
            return true;
            
        } catch (const std::exception& e) {
            LogError("Exception applying emotional state to context " + context_id + ": " + std::string(e.what()));
            return false;
        }
    }
    
    std::string GetModelPath() const override {
        return settings_manager ? settings_manager->GetString("Models", "emotag_model_path", "") : "";
    }
    
    std::string GetModelPathSettingKey() const override {
        return "emotag_model_path";
    }
    
    std::string GetSystemPromptSettingKey() const override {
        return "Emotion.system_prompt";
    }
    
    std::string GetPluginName() const override {
        return PLUGIN_NAME;
    }
    
    // Override logging to use plugin-specific macros
    void LogInfo(const std::string& message) {
        LOG_EmoTagPlugin(message);
        debug.AddLog("INFO", message);
    }
    
    void LogWarning(const std::string& message) {
        LOG_WARNING_EmoTagPlugin(message);
        debug.AddLog("WARN", message);
    }
    
    void LogError(const std::string& message) {
        LOG_ERROR_EmoTagPlugin(message);
        debug.AddLog("ERROR", message);
    }
    
    void LogDebug(const std::string& message) {
        LOG_DEBUG_EmoTagPlugin(message);
        debug.AddLog("DEBUG", message);
    }

private:
    /**
     * Apply emotional state to corresponding inner_voice context
     */
    void ApplyToInnerContext(const std::string& context_id, const std::string& emotional_state) {
        std::string inner_context_id;
        
        if (context_id.starts_with("outer_context_")) {
            // Discord context pattern: "outer_context_" + channel_id -> "inner_context_" + channel_id
            inner_context_id = "inner_context_" + context_id.substr(14);
        } else if (context_id == "outer_context") {
            // UI context pattern: "outer_context" -> "inner_context"
            inner_context_id = "inner_context";
        } else {
            return; // No corresponding inner context
        }
        
        auto* inner_context = llama_manager->GetContextInfo(inner_context_id);
        if (inner_context) {
            try {
                inner_context->UpdateEmotionalState(emotional_state);
                stats.IncrementContextUpdate();
                LogInfo("Applied emotional state to inner_voice context: " + inner_context_id);
            } catch (const std::exception& inner_e) {
                LogWarning("Failed to apply emotional state to inner_voice context: " + inner_context_id + 
                         " - Error: " + inner_e.what());
            }
        } else {
            LogDebug("No corresponding inner_voice context found for: " + inner_context_id);
        }
    }
    
    /**
     * Process a single batch of AI responses for emotional analysis
     */
    std::string ProcessEmotionalAnalysisBatch(const EmotionalAnalysisBatch& batch) {
        LogInfo("ProcessEmotionalAnalysisBatch called for context: " + batch.context_id);
        
        if (!model_ready.load() || !llama_manager) {
            LogError("Emotion model not ready for processing batch");
            return "";
        }
        
        try {
            // Build analysis prompt from AI responses
            std::ostringstream analysis_stream;
            analysis_stream << "Categorize my emotional state in 1-3 short sentences; identify the presence of any emotions present from the following:\n\n";

            for (size_t i = 0; i < batch.ai_responses.size(); ++i) {
                analysis_stream << (i + 1) << ": " << batch.ai_responses[i] << "\n";
            }

            analysis_stream << "\nMy emotional state:";
            std::string analysis_prompt = analysis_stream.str();
            
            LogInfo("Built analysis prompt (" + std::to_string(analysis_prompt.length()) + " chars) for context: " + batch.context_id);
            
            // Perform emotional analysis using base class generation method
            std::string emotional_state = PerformGeneration(analysis_prompt, batch.context_id);
            
            if (!emotional_state.empty() && !PluginUtils::IsErrorResponse(emotional_state)) {
                LogInfo("Successfully generated emotional analysis for context: " + batch.context_id + 
                       " - " + emotional_state.substr(0, 100) + 
                       (emotional_state.length() > 100 ? "..." : ""));
            } else {
                LogWarning("Failed to generate valid emotional analysis for context: " + batch.context_id);
            }
            
            return emotional_state;
            
        } catch (const std::exception& e) {
            LogError("Exception processing emotional analysis: " + std::string(e.what()));
            return "";
        }
    }
};

} // namespace LuminaChat

// CRITICAL: These function definitions are placed here to combat circular dependencies
inline void Orchestrator::RequestEmotionAnalysis(const EmotionalAnalysisBatch& batch, 
                                                LuminaChat::RequestPriority priority) {
    LOG_Orchestrator("Requesting emotion analysis for context: " + batch.context_id);
    
    // Create emotion analysis request using plugin structure
    LuminaChat::EmotionAnalysisRequest request(batch, batch.context_id, priority);
    
    // Check pipeline state before queuing
    LOG_Orchestrator("Pipeline state - IsProcessing: " + std::to_string(emotion_analysis_pipeline.IsProcessing()) + 
                    ", IsIdle: " + std::to_string(emotion_analysis_pipeline.IsIdle()) + 
                    ", IsShutdown: " + std::to_string(emotion_analysis_pipeline.IsShutdown()));
    
    // Log pipeline statistics before queuing
    auto stats = emotion_analysis_pipeline.GetStats();
    LOG_Orchestrator("Pipeline stats before queuing - Total: " + std::to_string(stats.total_requests.load()) + 
                    ", Completed: " + std::to_string(stats.completed_requests.load()) + 
                    ", Failed: " + std::to_string(stats.failed_requests.load()) + 
                    ", Pending: " + std::to_string(stats.pending_requests.load()));
    
    // Queue the request to the pipeline for asynchronous processing
    bool queued = emotion_analysis_pipeline.QueueRequest(request, priority, batch.context_id);
    
    if (queued) {
        LOG_Orchestrator("Successfully queued emotion analysis request for context: " + batch.context_id);
        
        // Log statistics after queuing
        auto stats_after = emotion_analysis_pipeline.GetStats();
        LOG_Orchestrator("Pipeline stats after queuing - Total: " + std::to_string(stats_after.total_requests.load()) + 
                        ", Pending: " + std::to_string(stats_after.pending_requests.load()));
    } else {
        LOG_ERROR_Orchestrator("Failed to queue emotion analysis request for context: " + batch.context_id + 
                              " (pipeline may be shutdown or queue full)");
    }
}

inline void Orchestrator::OnEmotionAnalysisComplete(const std::string& context_id, 
                                                   const LuminaChat::EmotionAnalysisResponse& response) {
    LOG_Orchestrator("Emotion analysis complete for context: " + context_id);
    
    if (response.success) {
        LOG_Orchestrator("Successfully analyzed emotional state for context: " + context_id + 
                        " - State: " + response.emotional_state.substr(0, 100) + 
                        (response.emotional_state.length() > 100 ? "..." : ""));
        
        // Apply emotional state to the context using EmoTagPlugin - plugin is required
        if (!emotag_plugin) {
            LOG_ERROR_Orchestrator("EmoTagPlugin not available - cannot apply emotional state for context: " + context_id);
            return;
        }
        
        if (!response.emotional_state.empty()) {
            bool applied = emotag_plugin->ApplyEmotionalStateToContext(context_id, response.emotional_state);
            if (applied) {
                LOG_Orchestrator("Applied emotional state to context: " + context_id);
            } else {
                LOG_ERROR_Orchestrator("Failed to apply emotional state to context: " + context_id);
            }
        } else {
            LOG_WARNING_Orchestrator("Generated emotional state is empty for context: " + context_id);
        }
        
        // Update statistics
        stats.emotion_analyses_completed.fetch_add(1, std::memory_order_relaxed);
    } else {
        LOG_ERROR_Orchestrator("Emotion analysis failed for context: " + context_id + " - Error: " + response.error_message);
    }
}

// CRITICAL: Orchestrator method implementations moved here to combat circular dependencies
inline void Orchestrator::ProcessEmotionAnalysisBuffer() {
    // Check if emotion plugin has pending analysis requests
    if (!emotag_plugin || !emotag_plugin->HasPendingEmotionalAnalysis()) [[likely]] {
        return; // No work to do
    }
    
    LOG_Orchestrator("Processing emotion analysis buffer...");
    
    // Get all pending emotional analysis batches from plugin
    auto analysis_batches = emotag_plugin->GetAndClearEmotionalAnalysisBuffer();
    
    if (!analysis_batches.empty()) [[likely]] {
        LOG_Orchestrator("Found " + std::to_string(analysis_batches.size()) + " emotion analysis batches to process");
        
        // Process each batch through the emotion analysis pipeline
        for (const auto& batch : analysis_batches) [[likely]] {
            LOG_Orchestrator("Processing batch for context: " + batch.context_id + " with " + std::to_string(batch.ai_responses.size()) + " responses");
            RequestEmotionAnalysis(batch, LuminaChat::RequestPriority::NORMAL);
        }
        
        LOG_Orchestrator("Emotion analysis buffer processing complete");
    }
}

inline void Orchestrator::ProcessEmotionAnalysisRequest(const LuminaChat::EmotionAnalysisRequest& request,
                                                       std::function<void(LuminaChat::EmotionAnalysisResponse)> callback) {
    LOG_Orchestrator("Processing emotion analysis request for: " + request.context_id);
    
    try {
        // Delegate to EmoTagPlugin - plugin is required
        if (!emotag_plugin) {
            LOG_ERROR_Orchestrator("EmoTagPlugin not available - emotion analysis requires plugin delegation");
            
            LuminaChat::EmotionAnalysisResponse error_response("", request.context_id, false, 0, 
                                                              "EmoTagPlugin not available - emotion analysis requires plugin delegation");
            
            callback(error_response);
            OnEmotionAnalysisComplete(request.context_id, error_response);
            return;
        }
        
        LOG_Orchestrator("Delegating emotion analysis to EmoTagPlugin for context: " + request.context_id + 
                        " (analyzing " + std::to_string(request.batch.ai_responses.size()) + " AI responses)");
        LuminaChat::EmotionAnalysisResponse response = emotag_plugin->ProcessEmotionAnalysisRequest(request);
        
        LOG_Orchestrator("EmoTagPlugin processing complete for context: " + request.context_id + 
                        " (success: " + (response.success ? "true" : "false") + ")");
        
        callback(response);
        
        // Trigger completion callback for the original orchestrator workflow
        OnEmotionAnalysisComplete(request.context_id, response);
        
    } catch (const std::exception& e) {
        LOG_ERROR_Orchestrator("Exception in ProcessEmotionAnalysisRequest: " + std::string(e.what()));
        LuminaChat::EmotionAnalysisResponse error_response("", request.context_id, false, 0, e.what());
        callback(error_response);
        OnEmotionAnalysisComplete(request.context_id, error_response);
    }
}