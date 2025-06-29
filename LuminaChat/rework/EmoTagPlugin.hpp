#pragma once

#include "ContextInfo.hpp"
#include "ProcessingPipeline.hpp"
#include "Orchestrator.hpp"
#include "LlamaManager.hpp"
#include "SettingsManager.hpp"
#include "Logger.hpp"
#include "Utilities.hpp"
#include <chrono>
#include <thread>
#include <memory>
#include <optional>
#include <sstream>
#include <iomanip>

namespace LuminaChat {

// Callback type for status updates
using StatusUpdateCallback = std::function<void(const std::string& status, bool is_error)>;

/**
 * EmoTag Plugin - Handles background emotional analysis of AI responses
 * 
 * This plugin periodically checks for AI responses that need emotional analysis
 * and processes them through the emotional analysis pipeline.
 * 
 * Architecture:
 * - Core handles immediate conversation flow (critical path)
 * - Plugin handles emotional analysis work (async/background)
 * - Emotional states are applied back to contexts when complete
 * 
 * Configuration:
 * - Analysis Window Size: Number of recent AI responses to analyze (default: 3)
 */
class EmoTagPlugin {
private:
    std::unique_ptr<std::thread> processing_thread;
    std::atomic<bool> should_stop{false};
    std::chrono::milliseconds check_interval{2000}; // Check every 2 seconds
    
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
        
        LogInfo("EmoTagPlugin initialized");
    }
    
    ~EmoTagPlugin() {
        Stop();
    }
    
    /**
     * Start the plugin's background processing thread
     */
    void Start() {
        if (processing_thread && processing_thread->joinable()) {
            LogWarning("Plugin already running");
            return;
        }
        
        // Initialize emotion analysis model and context first
        if (!InitializeEmotionModel()) {
            LogError("Failed to initialize emotion model - plugin will not process emotional analysis");
            return;
        }
        
        should_stop = false;
        processing_thread = std::make_unique<std::thread>(&EmoTagPlugin::ProcessingLoop, this);
        LogInfo("EmoTagPlugin started with emotion model ready");
    }
    
    /**
     * Stop the plugin and wait for thread completion
     */
    void Stop() {
        should_stop = true;
        if (processing_thread && processing_thread->joinable()) {
            processing_thread->join();
        }
        
        // Clean up emotion context and model
        emotion_model_ready = false;
        
        LogInfo("EmoTagPlugin stopped");
    }
    
    /**
     * Set the check interval for processing emotional analysis
     */
    void SetCheckInterval(std::chrono::milliseconds interval) {
        check_interval = interval;
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
     * Get plugin statistics
     */
    struct PluginStats {
        size_t batches_processed;
        size_t responses_analyzed;
        size_t emotional_states_generated;
        size_t contexts_updated;
        size_t analysis_window_size;
        bool is_running;
        bool emotion_model_ready;
        std::string emotion_model_path;
    };
    
    PluginStats GetStats() const {
        std::string model_path = "";
        if (settings_manager) {
            model_path = settings_manager->GetString("Models", "emotion_model_path", "");
        }
        
        return {
            batches_processed.load(),
            responses_analyzed.load(),
            emotional_states_generated.load(),
            contexts_updated.load(),
            analysis_window_size.load(),
            !should_stop.load(),
            emotion_model_ready.load(),
            model_path
        };
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
        
        // Get main model context size as hard limit
        int main_context_size = settings_manager->GetInt("Models", "main_context_size", 4096);
        if (new_context_size > static_cast<size_t>(main_context_size)) {
            LogWarning("Cannot resize emotion context to " + std::to_string(new_context_size) + 
                " tokens - would exceed main context limit of " + std::to_string(main_context_size));
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
        std::string emotion_model_path = settings_manager->GetString("Models", "emotion_model_path", "");
        if (emotion_model_path.empty()) {
            LogWarning("No emotion model path configured - plugin will be disabled");
            if (status_callback) status_callback("No emotion model configured", true);
            return false;
        }
        
        // Get main model settings to derive emotion model config
        int main_context_size = settings_manager->GetInt("Models", "main_context_size", 4096);
        int main_gpu_layers = settings_manager->GetInt("Models", "main_gpu_layers", 999);
        
        // Context size calculation for emotional analysis
        // Need enough space for: system prompt (~300 tokens) + AI responses (up to 1000 tokens each for 3 responses)
        // + formatting overhead (~100 tokens) + response generation (~200 tokens)
        // Total: ~3600 tokens minimum for typical analysis
        // Use 50% of main context size with minimum 4500 tokens (matching SummarizationPlugin for reliability)
        int emotion_context_size = std::max(4500, (main_context_size * 50) / 100);
        
        LogInfo("Loading emotion model: " + emotion_model_path + 
                        " (context: " + std::to_string(emotion_context_size) + 
                        " [30% of main, optimized for response analysis], gpu_layers: " + std::to_string(main_gpu_layers) + ")");
        
        if (status_callback) status_callback("Loading emotion model: " + emotion_model_path, false);
        
        try {
            // Load emotion model
            ModelConfig config;
            config.model_path = emotion_model_path;
            config.context_size = emotion_context_size;
            config.gpu_layers = main_gpu_layers;
            
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
     * Main processing loop - runs in background thread
     */
    void ProcessingLoop() {
        LogInfo("Processing loop started");
        
        while (!should_stop.load()) {
            try {
                ProcessPendingEmotionalAnalysis();
                std::this_thread::sleep_for(check_interval);
            } catch (const std::exception& e) {
                LogError("Exception in processing loop: " + std::string(e.what()));
                std::this_thread::sleep_for(check_interval * 5); // Back off on error
            }
        }
        
        LogInfo("Processing loop ended");
    }
    
    /**
     * Check for and process any pending emotional analysis requests
     */
    void ProcessPendingEmotionalAnalysis() {
        // Skip processing if emotion model is not ready
        if (!emotion_model_ready.load()) {
            return;
        }
        
        // Check if there are any AI responses waiting for emotional analysis
        if (!ContextInfo::HasPendingEmotionalAnalysis()) {
            return; // No work to do
        }
        
        LogInfo("Found pending emotional analysis to process");
        
        // Get all pending emotional analysis batches
        auto analysis_batches = ContextInfo::GetAndClearEmotionalAnalysisBuffer();
        
        for (const auto& batch : analysis_batches) {
            ProcessEmotionalAnalysisBatch(batch);
        }
        
        batches_processed += analysis_batches.size();
    }
    
    /**
     * Process a single batch of AI responses for emotional analysis
     */
    void ProcessEmotionalAnalysisBatch(const EmotionalAnalysisBatch& batch) {
        LogInfo("Processing emotional analysis for context: " + batch.context_id + 
                               " (" + std::to_string(batch.ai_responses.size()) + " responses)");
        
        if (!emotion_model_ready.load() || !llama_manager) {
            LogError("Emotion model not ready for processing batch");
            return;
        }
        
        try {
            // Get the emotion context
            auto* emotion_ctx = llama_manager->GetContextInfo(emotion_context_id);
            if (!emotion_ctx) {
                LogError("Emotion context not available");
                return;
            }
            
            // Build analysis prompt from AI responses
            std::ostringstream analysis_stream;
            analysis_stream << "Categorize my emotional state; identify the presence of any emotions present from the following.\n";

            for (size_t i = 0; i < batch.ai_responses.size(); ++i) {
                analysis_stream << (i + 1) << ":\n" << batch.ai_responses[i] << "\n\n";
            }

            analysis_stream << "My emotional state:";
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
                            return;
                        } else {
                            LogInfo("Emotional analysis succeeded after context resize for context: " + batch.context_id);
                        }
                    } else {
                        LogError("Lost emotion context after resize - skipping analysis");
                        return;
                    }
                } else {
                    LogWarning("Context resize failed for emotional analysis - skipping context: " + batch.context_id + 
                        " (tokens too large or resize failed)");
                    return;
                }
            }
            
            // Check if we got a valid response after potential retry
            if (emotional_state.empty() || 
                emotional_state.find("Error: Failed to process prompt") == 0 || 
                emotional_state.find("Error: Exception during") == 0) {
                LogWarning("Failed to generate valid emotional analysis for context: " + batch.context_id + " - Response: " + emotional_state);
                return;
            }
            
            // Apply emotional state to original context
            ApplyEmotionalStateToContext(batch.context_id, emotional_state);
            
            responses_analyzed += batch.ai_responses.size();
            emotional_states_generated++;
            
            LogInfo("Successfully generated emotional analysis for context: " + batch.context_id + 
                           " - " + emotional_state.substr(0, 100) + 
                           (emotional_state.length() > 100 ? "..." : ""));
            
        } catch (const std::exception& e) {
            LogError("Exception processing emotional analysis: " + std::string(e.what()));
        }
    }
    
    /**
     * Apply generated emotional state back to the original context
     */
    void ApplyEmotionalStateToContext(const std::string& context_id, const std::string& emotional_state) {
        if (!llama_manager) {
            LogError("LlamaManager not available for applying emotional state");
            return;
        }
        
        try {
            // Get the original context
            auto* original_context = llama_manager->GetContextInfo(context_id);
            if (!original_context) {
                LogWarning("Original context not found for emotional state application: " + context_id);
                return;
            }
            
            // The emotional state should now be clean (no template tokens) thanks to ExtractCleanResponse
            LogDebug("Emotional state (" + std::to_string(emotional_state.length()) + " chars): " + 
                    emotional_state.substr(0, 200) + (emotional_state.length() > 200 ? "..." : ""));
            
            if (emotional_state.empty()) {
                LogWarning("Emotional state is empty - skipping update for context: " + context_id);
                return;
            }
            
            // Apply the clean emotional state to the context
            original_context->UpdateEmotionalState(emotional_state);
            
            contexts_updated++;
            
            LogInfo("Applied emotional state to context: " + context_id);
            
        } catch (const std::exception& e) {
            LogError("Exception applying emotional state to context: " + std::string(e.what()));
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
