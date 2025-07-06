#pragma once

#include "Context/ContextInfo.hpp"
#include "ProcessingPipeline.hpp"
#include "Orchestrator.hpp"
#include "LlamaManager.hpp"
#include "SettingsManager.hpp"
#include "Logger.hpp"
#include "Utilities.hpp"
#include <chrono>
#include <memory>
#include <optional>
#include <sstream>
#include <iomanip>

namespace LuminaChat {

// Callback type for status updates
using StatusUpdateCallback = std::function<void(const std::string& status, bool is_error)>;

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
class EmoTagPlugin {
private:
    // No independent pipeline - Orchestrator coordinates workflow
    
    // Reference to orchestrator for context management
    Orchestrator* orchestrator = nullptr;
    
    // Emotion analysis model and context management
    LlamaManager* llama_manager = nullptr;
    SettingsManager* settings_manager = nullptr;
    std::string emotion_model_id = "emotion_model";
    std::string emotion_context_id = "plugin_emotion_context";
    std::atomic<bool> emotion_model_ready{false};
    
    // Status callback for UI updates
    StatusUpdateCallback status_callback;
    
    // Configuration
    std::atomic<size_t> analysis_window_size{3}; // Number of AI responses to analyze
    
    // Statistics
    std::atomic<size_t> batches_processed{0};
    std::atomic<size_t> responses_analyzed{0};
    std::atomic<size_t> emotional_states_generated{0};
    std::atomic<size_t> contexts_updated{0};
    
    // Buffer management for emotional analysis (moved from ContextInfo)
    mutable std::mutex emotional_analysis_buffer_mutex;
    std::vector<EmotionalAnalysisBatch> emotional_analysis_buffer;
    
    // Debugging features
    struct DebugGeneration {
        std::string input;
        std::string output;
        std::string context_id;
        std::chrono::system_clock::time_point timestamp;
    };
    
    mutable std::mutex debug_mutex;
    std::deque<std::string> log_history; // Plugin-specific log history
    static constexpr size_t MAX_LOG_HISTORY = 100; // Keep last 100 log entries
    std::optional<DebugGeneration> last_generation; // Last generation for debugging

public:
    explicit EmoTagPlugin(Orchestrator* orch) 
        : orchestrator(orch) {
        
        if (orchestrator) {
            llama_manager = orchestrator->GetLlamaManager();
            settings_manager = orchestrator->GetSettingsManager();
        }
        
        LogInfo("EmoTagPlugin initialized as processor service");
    }
    
    ~EmoTagPlugin() {
        Shutdown();
    }
    
    /**
     * Initialize the plugin (called by Orchestrator)
     */
    bool Initialize() {
        return InitializeEmotionModel();
    }
    
    /**
     * Shutdown the plugin and clean up resources
     */
    void Shutdown() {
        // Clean up emotion context and model
        emotion_model_ready = false;
        
        LogInfo("EmoTagPlugin shutdown");
    }
    
    /**
     * Set status update callback for UI notifications
     */
    void SetStatusCallback(StatusUpdateCallback callback) {
        status_callback = callback;
    }
    
    /**
     * Configure analysis parameters
     */
    void SetAnalysisWindow(size_t window_size) {
        analysis_window_size = window_size;
    }
    
    /**
     * Process an emotion analysis request (called by Orchestrator pipeline)
     * This is the main processing method used by the Orchestrator's pipeline
     */
    EmotionAnalysisResponse ProcessEmotionAnalysisRequest(const EmotionAnalysisRequest& request) {
        try {
            LogInfo("Processing emotion analysis for context: " + request.context_id + 
                   " (" + std::to_string(request.batch.ai_responses.size()) + " responses)");
            
            if (!emotion_model_ready.load() || !llama_manager) {
                std::string error_msg = "Emotion model not ready for processing batch";
                LogError(error_msg);
                return EmotionAnalysisResponse("", request.context_id, false, 0, error_msg);
            }
            
            // Process the batch to generate emotional state (existing logic)
            std::string emotional_state = ProcessEmotionalAnalysisBatch(request.batch);
            
            if (emotional_state.empty()) {
                std::string error_msg = "Failed to generate emotional state for context: " + request.context_id;
                LogError(error_msg);
                return EmotionAnalysisResponse("", request.context_id, false, 0, error_msg);
            }
            
            // Create success response with the generated emotional state
            EmotionAnalysisResponse response{
                emotional_state,
                request.context_id,
                true,
                request.batch.ai_responses.size()
            };
            
            LogInfo("Successfully generated emotional state for batch: " + request.context_id);
            
            // Update statistics
            responses_analyzed += request.batch.ai_responses.size();
            emotional_states_generated++;
            batches_processed++;
            
            return response;
            
        } catch (const std::exception& e) {
            std::string error_msg = "Exception processing emotion analysis: " + std::string(e.what());
            LogError(error_msg);
            return EmotionAnalysisResponse("", request.context_id, false, 0, error_msg);
        }
    }
    
    /**
     * Apply an emotional state to a context (called by Orchestrator after successful processing)
     */
    bool ApplyEmotionalStateToContext(const std::string& context_id, const std::string& emotional_state) {
        try {
            if (!llama_manager) {
                LogError("LlamaManager not available for applying emotional state");
                return false;
            }
            
            // Get the original context
            auto* original_context = llama_manager->GetContextInfo(context_id);
            if (!original_context) {
                LogWarning("Original context not found for emotional state application: " + context_id);
                return false;
            }
            
            // Apply the emotional state to the context
            if (emotional_state.empty()) {
                LogWarning("Emotional state is empty - skipping update for context: " + context_id);
                return false;
            }
            
            // Try to acquire plugin processing lock for safe emotional state application
            if (!original_context->TryAcquirePluginProcessing("EmoTagPlugin")) {
                LogWarning("Could not acquire plugin processing lock for context: " + context_id + " - context may be busy");
                return false;
            }
            
            try {
                // Apply the clean emotional state to the context
                original_context->UpdateEmotionalState(emotional_state);
                
                contexts_updated++;
                
                LogInfo("Applied emotional state to context: " + context_id);
                
                // Release plugin processing lock
                [[maybe_unused]] bool released = original_context->ReleasePluginProcessing("EmoTagPlugin");
                return true;
                
            } catch (const std::exception&) {
                // Release plugin processing lock on exception
                [[maybe_unused]] bool released = original_context->ReleasePluginProcessing("EmoTagPlugin");
                throw; // Re-throw the exception
            }
            
        } catch (const std::exception& e) {
            LogError("Exception applying emotional state to context: " + std::string(e.what()));
            return false;
        }
    }
    
    /**
     * Get plugin statistics
     */
    struct PluginStats {
        size_t batches_processed;
        size_t responses_analyzed;
        size_t emotional_states_generated;
        size_t contexts_updated;
        size_t analysis_window_size;
        bool emotion_model_ready;
        std::string emotion_model_path;
    };
    
    PluginStats GetStats() const {
        std::string model_path = "";
        if (settings_manager) {
            model_path = settings_manager->GetString("Models", "emotag_model_path", "");
        }
        
        return {
            batches_processed.load(),
            responses_analyzed.load(),
            emotional_states_generated.load(),
            contexts_updated.load(),
            analysis_window_size.load(),
            emotion_model_ready.load(),
            model_path
        };
    }
    
    /**
     * Check if the plugin is ready to process emotion analysis requests
     */
    bool IsReady() const {
        return emotion_model_ready.load();
    }
    
    // Buffer management methods (moved from ContextInfo)
    /**
     * Get and clear all pending emotional analysis batches
     */
    std::vector<EmotionalAnalysisBatch> GetAndClearEmotionalAnalysisBuffer() {
        std::lock_guard<std::mutex> lock(emotional_analysis_buffer_mutex);
        
        std::vector<EmotionalAnalysisBatch> result;
        result.swap(emotional_analysis_buffer);
        
        return result;
    }
    
    /**
     * Check if there are pending AI responses waiting for emotional analysis
     */
    bool HasPendingEmotionalAnalysis() const {
        std::lock_guard<std::mutex> lock(emotional_analysis_buffer_mutex);
        return !emotional_analysis_buffer.empty();
    }
    
    /**
     * Add an emotional analysis batch to the buffer
     */
    void AddToEmotionalAnalysisBuffer(EmotionalAnalysisBatch&& batch) {
        std::lock_guard<std::mutex> lock(emotional_analysis_buffer_mutex);
        emotional_analysis_buffer.emplace_back(std::move(batch));
    }
    
    /**
     * Request emotional analysis for a specific context (moved from ContextInfo)
     */
    void RequestEmotionalAnalysis(const std::string& context_id, const std::vector<std::pair<std::string, std::string>>& message_history) {
        if (message_history.empty()) {
            LogInfo("No messages in history for emotional analysis request - context: " + context_id);
            return;
        }
        
        // Extract recent AI responses for analysis
        std::vector<std::string> ai_responses;
        const size_t max_responses = analysis_window_size.load();
        const size_t history_size = message_history.size();
        
        LogInfo("Searching for AI responses in " + std::to_string(history_size) + " messages for context: " + context_id);
        
        // Walk backwards through message history to find AI responses
        for (auto it = message_history.rbegin(); it != message_history.rend() && ai_responses.size() < max_responses; ++it) {
            if (it->first == "assistant") {
                ai_responses.insert(ai_responses.begin(), it->second); // Insert at beginning to maintain order
                LogInfo("Found assistant message (" + std::to_string(it->second.length()) + " chars) for emotional analysis");
            }
        }
        
        if (!ai_responses.empty()) {
            size_t response_count = ai_responses.size();  // Store size before move
            AddToEmotionalAnalysisBuffer(EmotionalAnalysisBatch(context_id, std::move(ai_responses)));
            
            LogInfo("Requested emotional analysis for context " + context_id + 
                   " with " + std::to_string(response_count) + " AI responses");
        } else {
            LogInfo("No assistant messages found in " + std::to_string(history_size) + " messages for context: " + context_id);
        }
    }
    
    /**
     * Get plugin-specific log history for debugging
     */
    std::vector<std::string> GetLogHistory() const {
        std::lock_guard<std::mutex> lock(debug_mutex);
        return std::vector<std::string>(log_history.begin(), log_history.end());
    }
    
    /**
     * Get last generation info for debugging
     */
    struct LastGenerationInfo {
        bool has_generation;
        std::string input;
        std::string output;
        std::string context_id;
        std::string timestamp;
    };
    
    LastGenerationInfo GetLastGeneration() const {
        std::lock_guard<std::mutex> lock(debug_mutex);
        if (!last_generation.has_value()) {
            return {false, "", "", "", ""};
        }
        
        // Format timestamp
        auto time_t = std::chrono::system_clock::to_time_t(last_generation->timestamp);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        
        return {
            true,
            last_generation->input,
            last_generation->output,
            last_generation->context_id,
            ss.str()
        };
    }
    
    /**
     * Add a log entry to plugin-specific log history (for debugging)
     */
    void AddLogEntry(const std::string& log_message) {
        std::lock_guard<std::mutex> lock(debug_mutex);
        log_history.push_back(log_message);
        
        // Keep only the last MAX_LOG_HISTORY entries
        while (log_history.size() > MAX_LOG_HISTORY) {
            log_history.pop_front();
        }
    }
    
    // Helper methods for logging that also capture to debug history
    void LogInfo(const std::string& message) {
        LOG_EmoTagPlugin(message);
        AddLogEntry("[INFO] " + message);
    }
    
    void LogWarning(const std::string& message) {
        LOG_WARNING_EmoTagPlugin(message);
        AddLogEntry("[WARN] " + message);
    }
    
    void LogError(const std::string& message) {
        LOG_ERROR_EmoTagPlugin(message);
        AddLogEntry("[ERROR] " + message);
    }
    
    void LogDebug(const std::string& message) {
        LOG_DEBUG_EmoTagPlugin(message);
        AddLogEntry("[DEBUG] " + message);
    }
    
    /**
     * Dynamically resize emotion context to handle larger analysis inputs
     * @param needed_tokens The minimum tokens needed for processing
     * @return true if resize was successful, false otherwise
     */
    bool ResizeEmotionContext(size_t needed_tokens) {
        if (!llama_manager || !settings_manager) {
            LogError("LlamaManager or SettingsManager not available for context resize");
            return false;
        }
        
        // Get current emotion context
        auto* emotion_ctx = llama_manager->GetContextInfo(emotion_context_id);
        if (!emotion_ctx) {
            LogError("Emotion context not available for resize");
            return false;
        }
        
        // Calculate new context size (needed tokens + 25% buffer for system prompt and generation overhead)
        size_t new_context_size = static_cast<size_t>(needed_tokens * 1.25f) + 500;
        
        // Get outer model context size as hard limit
        int outer_context_size = settings_manager->GetInt("Models", "outer_context_size", 4096);
        if (new_context_size > static_cast<size_t>(outer_context_size)) {
            LogWarning("Cannot resize emotion context to " + std::to_string(new_context_size) + 
                " tokens - would exceed outer context limit of " + std::to_string(outer_context_size));
            return false;
        }
        
        LogInfo("Resizing emotion context from current size to " + std::to_string(new_context_size) + 
            " tokens (needed: " + std::to_string(needed_tokens) + ")");
        
        try {
            // Store system prompt before context destruction
            std::string system_prompt = settings_manager->GetString("Emotion", "system_prompt", "");
            
            // Remove the old context (this will free the llama context but keep the model loaded)
            llama_manager->RemoveContext(emotion_context_id);
            
            // Create new context with the larger size (model stays loaded)
            auto* new_context_ptr = llama_manager->GetOrCreateContextInfo(emotion_context_id, emotion_model_id, static_cast<int32_t>(new_context_size));
            if (!new_context_ptr) {
                LogError("Failed to create resized emotion context");
                return false;
            }
            
            // Reapply system prompt
            if (!system_prompt.empty()) {
                new_context_ptr->UpdateSystemPrompt(system_prompt);
                LogInfo("Reapplied system prompt to resized context");
            }
            
            LogInfo("Successfully resized emotion context to " + std::to_string(new_context_size) + " tokens");
            return true;
            
        } catch (const std::exception& e) {
            LogError("Exception during context resize: " + std::string(e.what()));
            return false;
        }
    }

private:
    /**
     * Initialize the emotion analysis model and context for the plugin
     */
    bool InitializeEmotionModel() {
        if (!llama_manager || !settings_manager) {
            LogError("LlamaManager or SettingsManager not available");
            if (status_callback) status_callback("Error: Core services not available", true);
            return false;
        }
        
        LogInfo("Initializing emotion model...");
        if (status_callback) status_callback("Initializing emotion model...", false);
        
        // Get emotion model configuration from settings
        std::string emotion_model_path = settings_manager->GetString("Models", "emotag_model_path", "");
        if (emotion_model_path.empty()) {
            LogWarning("No emotion model path configured - plugin will be disabled");
            if (status_callback) status_callback("No emotion model configured", true);
            return false;
        }
        
        // Get outer model settings to derive emotion model config
        int outer_context_size = settings_manager->GetInt("Models", "outer_context_size", 4096);
        int outer_gpu_layers = settings_manager->GetInt("Models", "outer_gpu_layers", 999);
        
        // Context size calculation for emotional analysis
        // Need enough space for: system prompt (~300 tokens) + AI responses (up to 1000 tokens each for 3 responses)
        // + formatting overhead (~100 tokens) + response generation (~200 tokens)
        // Total: ~3600 tokens minimum for typical analysis
        // Use 50% of outer context size with minimum 4500 tokens (matching SummarizationPlugin for reliability)
        int emotion_context_size = std::max(4500, (outer_context_size * 50) / 100);
        
        LogInfo("Loading emotion model: " + emotion_model_path + 
                        " (context: " + std::to_string(emotion_context_size) + 
                        " [30% of outer, optimized for response analysis], gpu_layers: " + std::to_string(outer_gpu_layers) + ")");
        
        if (status_callback) status_callback("Loading emotion model: " + emotion_model_path, false);
        
        try {
            // Load emotion model
            ModelConfig config;
            config.model_path = emotion_model_path;
            config.context_size = emotion_context_size;
            config.gpu_layers = outer_gpu_layers;
            
            if (!llama_manager->LoadModel(emotion_model_id, config)) {
                LogError("Failed to load emotion model: " + emotion_model_path);
                if (status_callback) status_callback("Failed to load emotion model", true);
                return false;
            }
            
            LogInfo("Emotion model loaded successfully");
            if (status_callback) status_callback("Creating emotion context...", false);
            
            // Create emotion context
            auto* context_ptr = llama_manager->GetOrCreateContextInfo(emotion_context_id, emotion_model_id, emotion_context_size);
            if (!context_ptr) {
                LogError("Failed to create emotion context");
                if (status_callback) status_callback("Failed to create emotion context", true);
                return false;
            }
            
            LogInfo("Emotion context created: " + emotion_context_id);
            if (status_callback) status_callback("Applying emotion system prompt...", false);
            
            // Apply emotion system prompt from settings
            std::string emotion_system_prompt = settings_manager->GetString("Emotion", "system_prompt",
                "You are an emotional state analyzer. When given AI assistant responses, analyze the emotional tone, "
                "mood, and psychological state conveyed in the text. Provide a brief emotional overview that captures "
                "the assistant's apparent emotional state, confidence level, and overall demeanor. "
                "Focus on identifying patterns like: confident, uncertain, empathetic, analytical, cheerful, "
                "cautious, enthusiastic, or reserved. Keep your analysis concise and actionable.");
            
            if (!emotion_system_prompt.empty()) {
                context_ptr->UpdateSystemPrompt(emotion_system_prompt);
                LogInfo("Applied emotion system prompt to context");
            }
            
            emotion_model_ready = true;
            LogInfo("Emotion model initialization completed successfully");
            if (status_callback) status_callback("Emotion model ready", false);
            return true;
            
        } catch (const std::exception& e) {
            LogError("Exception during emotion model initialization: " + std::string(e.what()));
            if (status_callback) status_callback("Error: " + std::string(e.what()), true);
            return false;
        }
    }
    
    /**
     * Process a single batch of AI responses for emotional analysis
     * Returns the generated emotional state or empty string on failure
     */
    std::string ProcessEmotionalAnalysisBatch(const EmotionalAnalysisBatch& batch) {
        if (!emotion_model_ready.load() || !llama_manager) {
            LogError("Emotion model not ready for processing batch");
            return "";
        }
        
        try {
            // Get the emotion context
            auto* emotion_ctx = llama_manager->GetContextInfo(emotion_context_id);
            if (!emotion_ctx) {
                LogError("Emotion context not available");
                return "";
            }
            
            // Build analysis prompt from AI responses
            std::ostringstream analysis_stream;
            analysis_stream << "Categorize my emotional state in 1-3 short sentences; identify the presence of any emotions present from the following:\n\n";

            for (size_t i = 0; i < batch.ai_responses.size(); ++i) {
                analysis_stream << (i + 1) << ": " << batch.ai_responses[i] << "\n";
            }

            analysis_stream << "\nMy emotional state:";
            std::string analysis_prompt = analysis_stream.str();

            
            LogInfo("Built analysis prompt (" + std::to_string(analysis_prompt.length()) + " chars) for context: " + batch.context_id);
            
            // Reset context for clean analysis
            emotion_ctx->ClearContext();
            
            std::string emotional_state;
            try {
                emotional_state = emotion_ctx->HandleInput(analysis_prompt, "user");
                LogInfo("Emotion analysis completed successfully for context: " + batch.context_id);
                
            } catch (const std::exception& analysis_e) {
                std::string error_msg = std::string(analysis_e.what());
                LogError("Exception during emotional analysis for context " + batch.context_id + ": " + error_msg);
                emotional_state = "Error: Exception during generation - " + error_msg;
            }
            
            // Always store generation for debugging (regardless of success/failure)
            {
                std::lock_guard<std::mutex> lock(debug_mutex);
                last_generation = DebugGeneration{
                    analysis_prompt,
                    emotional_state,
                    batch.context_id,
                    std::chrono::system_clock::now()
                };
            }
            
            // Check if we got an error (likely context overflow or exception) - attempt resize if needed
            if (emotional_state.empty() || 
                emotional_state.find("Error: Failed to process prompt") == 0 || 
                emotional_state.find("Error: Exception during generation") == 0) {
                
                LogWarning("Failed to analyze emotional state for context " + batch.context_id + 
                    " - attempting context resize...");
                
                // Extract needed token count from error message (if available)
                size_t needed_tokens = ExtractNeededTokensFromError(emotional_state);
                if (needed_tokens == 0) {
                    // Better token estimation: 3 chars per token + 25% buffer for system prompt overhead
                    needed_tokens = static_cast<size_t>((analysis_prompt.length() / 3.0f) * 1.25f) + 500; // +500 for system prompt and generation
                    LogInfo("Estimated needed tokens: " + std::to_string(needed_tokens));
                } else {
                    LogInfo("Extracted needed tokens from error: " + std::to_string(needed_tokens));
                }
                
                // Attempt to resize the emotion context
                if (ResizeEmotionContext(needed_tokens)) {
                    LogInfo("Context resize successful, retrying emotional analysis for context: " + batch.context_id);
                    
                    // Get the resized context and retry
                    emotion_ctx = llama_manager->GetContextInfo(emotion_context_id);
                    if (emotion_ctx) {
                        emotion_ctx->ClearContext();
                        try {
                            emotional_state = emotion_ctx->HandleInput(analysis_prompt, "user");
                        } catch (const std::exception& retry_e) {
                            emotional_state = "Error: Exception during retry generation - " + std::string(retry_e.what());
                            LogError("Exception during retry generation: " + std::string(retry_e.what()));
                        }
                        
                        // Update last_generation with retry result
                        {
                            std::lock_guard<std::mutex> lock(debug_mutex);
                            last_generation = DebugGeneration{
                                analysis_prompt,
                                emotional_state,
                                batch.context_id,
                                std::chrono::system_clock::now()
                            };
                        }
                        
                        if (emotional_state.empty() || 
                            emotional_state.find("Error: Failed to process prompt") == 0 || 
                            emotional_state.find("Error: Exception during") == 0) {
                            LogWarning("Emotional analysis still failed after context resize - skipping context: " + batch.context_id);
                            return "";
                        } else {
                            LogInfo("Emotional analysis succeeded after context resize for context: " + batch.context_id);
                        }
                    } else {
                        LogError("Lost emotion context after resize - skipping analysis");
                        return "";
                    }
                } else {
                    LogWarning("Context resize failed for emotional analysis - skipping context: " + batch.context_id + 
                        " (tokens too large or resize failed)");
                    return "";
                }
            }
            
            // Check if we got a valid response after potential retry
            if (emotional_state.empty() || 
                emotional_state.find("Error: Failed to process prompt") == 0 || 
                emotional_state.find("Error: Exception during") == 0) {
                LogWarning("Failed to generate valid emotional analysis for context: " + batch.context_id + " - Response: " + emotional_state);
                return "";
            }
            
            LogInfo("Successfully generated emotional analysis for context: " + batch.context_id + 
                   " - " + emotional_state.substr(0, 100) + 
                   (emotional_state.length() > 100 ? "..." : ""));
            
            return emotional_state;
            
        } catch (const std::exception& e) {
            LogError("Exception processing emotional analysis: " + std::string(e.what()));
            return "";
        }
    }
    
    /**
     * Extract needed token count from error message if available
     */
    size_t ExtractNeededTokensFromError(const std::string& error_message) {
        if (error_message.empty()) {
            return 0;
        }
        
        // Look for patterns like "Error: Failed to process prompt" with token information
        // Error messages from llama.cpp might contain token count information
        std::string search_patterns[] = {
            "needed ",
            "require ",
            "needs ",
            "tokens:"
        };
        
        for (const auto& pattern : search_patterns) {
            size_t pos = error_message.find(pattern);
            if (pos != std::string::npos) {
                // Look for a number after the pattern
                size_t start = pos + pattern.length();
                while (start < error_message.length() && !std::isdigit(error_message[start])) {
                    start++;
                }
                
                if (start < error_message.length()) {
                    // Extract the number
                    size_t end = start;
                    while (end < error_message.length() && std::isdigit(error_message[end])) {
                        end++;
                    }
                    
                    if (end > start) {
                        try {
                            return std::stoull(error_message.substr(start, end - start));
                        } catch (...) {
                            // Continue looking
                        }
                    }
                }
            }
        }
        
        return 0; // No token count found
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
        LOG_Orchestrator("Successfully analyzed emotional state for context: " + context_id);
        
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
        }
        
        // Update statistics
        // Update statistics using lock-free atomic increment
        stats.emotion_analyses_completed.fetch_add(1, std::memory_order_relaxed);
    } else {
        LOG_ERROR_Orchestrator("Emotion analysis failed: " + response.error_message);
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