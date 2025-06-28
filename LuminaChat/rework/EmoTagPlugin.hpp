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
#include <deque>

namespace LuminaChat {

// Callback type for status updates
using StatusUpdateCallback = std::function<void(const std::string& status, bool is_error)>;

/**
 * EmoTag Plugin - Handles emotional state analysis of AI responses
 * 
 * This plugin monitors AI responses and analyzes them immediately to determine
 * the emotional state of the AI assistant. The analysis is then fed back to the
 * main context to influence future responses.
 * 
 * Architecture:
 * - Core handles normal conversation flow
 * - Plugin captures AI responses and triggers immediate analysis
 * - Analysis window keeps the last N messages (AI responses + optionally user messages)
 * - Analysis is performed through a separate emotional analysis context
 * - Results are applied back to the main context via UpdateEmotionalState
 * 
 * Configuration:
 * - Analysis Window Size: Number of recent messages to keep and analyze (default: 3)
 * - Include User Messages: Whether to include user messages in analysis (default: false)
 */
class EmoTagPlugin {
private:
    std::unique_ptr<std::thread> processing_thread;
    std::atomic<bool> should_stop{false};
    std::chrono::milliseconds check_interval{5000}; // Check every 5 seconds (mainly for maintenance)
    
    // Reference to orchestrator for context management
    Orchestrator* orchestrator = nullptr;
    
    // Emotion analysis model and context management
    LlamaManager* llama_manager = nullptr;
    SettingsManager* settings_manager = nullptr;
    std::unique_ptr<ContextInfo> emotion_context;
    std::string emotion_model_id = "emotion_model";
    std::string emotion_context_id = "plugin_emotion_context";
    std::atomic<bool> emotion_model_ready{false};
    
    // Status callback for UI updates
    StatusUpdateCallback status_callback;
    
    // Configuration
    std::atomic<size_t> analysis_window_size{3}; // Number of AI responses to keep and analyze
    std::atomic<bool> include_user_messages{false}; // Whether to include user messages in analysis
    
    // Response tracking per context
    struct ContextResponseHistory {
        std::deque<std::pair<std::string, std::string>> messages; // role, content pairs
        std::string context_id;
    };
    std::unordered_map<std::string, ContextResponseHistory> context_histories;
    mutable std::mutex history_mutex;
    
    // Statistics
    std::atomic<size_t> responses_analyzed{0};
    std::atomic<size_t> emotional_states_generated{0};
    std::atomic<size_t> contexts_updated{0};

public:
    explicit EmoTagPlugin(Orchestrator* orch) 
        : orchestrator(orch) {
        
        if (orchestrator) {
            llama_manager = orchestrator->GetLlamaManager();
            settings_manager = orchestrator->GetSettingsManager();
        }
        
        LOG_EmoTagPlugin("EmoTagPlugin initialized");
    }
    
    ~EmoTagPlugin() {
        Stop();
    }
    
    /**
     * Start the plugin's background processing thread
     */
    void Start() {
        if (processing_thread && processing_thread->joinable()) {
            LOG_WARNING_EmoTagPlugin("Plugin already running");
            return;
        }
        
        // Initialize emotion analysis model and context first
        if (!InitializeEmotionModel()) {
            LOG_ERROR_EmoTagPlugin("Failed to initialize emotion model - plugin will not process emotional analysis");
            return;
        }
        
        should_stop = false;
        processing_thread = std::make_unique<std::thread>(&EmoTagPlugin::ProcessingLoop, this);
        LOG_EmoTagPlugin("EmoTagPlugin started with emotion model ready");
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
        emotion_context.reset();
        emotion_model_ready = false;
        
        LOG_EmoTagPlugin("EmoTagPlugin stopped");
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
    
    void SetIncludeUserMessages(bool include_users) {
        include_user_messages = include_users;
    }
    
    /**
     * Record an AI response for emotional analysis (triggers analysis)
     */
    void RecordAIResponse(const std::string& context_id, const std::string& response) {
        std::lock_guard<std::mutex> lock(history_mutex);
        
        auto& history = context_histories[context_id];
        history.context_id = context_id;
        history.messages.push_back({"assistant", response});
        
        // Keep only the last analysis_window_size messages
        size_t window = analysis_window_size.load();
        while (history.messages.size() > window) {
            history.messages.pop_front();
        }
        
        LOG_EmoTagPlugin("Recorded AI response for context: " + context_id + 
                        " (total messages: " + std::to_string(history.messages.size()) + ")");
        
        // Trigger analysis immediately after recording AI response
        ProcessEmotionalAnalysisForContext(context_id, history);
    }
    
    /**
     * Record a user message (optional, only stored if include_user_messages is true)
     */
    void RecordUserMessage(const std::string& context_id, const std::string& message) {
        if (!include_user_messages.load()) {
            return; // Skip if user messages are disabled
        }
        
        std::lock_guard<std::mutex> lock(history_mutex);
        
        auto& history = context_histories[context_id];
        history.context_id = context_id;
        history.messages.push_back({"user", message});
        
        // Keep only the last analysis_window_size messages
        size_t window = analysis_window_size.load();
        while (history.messages.size() > window) {
            history.messages.pop_front();
        }
        
        LOG_EmoTagPlugin("Recorded user message for context: " + context_id + 
                        " (total messages: " + std::to_string(history.messages.size()) + ")");
    }
    
    /**
     * Get plugin statistics
     */
    struct PluginStats {
        size_t responses_analyzed;
        size_t emotional_states_generated;
        size_t contexts_updated;
        size_t analysis_window_size;
        bool include_user_messages;
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
            responses_analyzed.load(),
            emotional_states_generated.load(),
            contexts_updated.load(),
            analysis_window_size.load(),
            include_user_messages.load(),
            !should_stop.load(),
            emotion_model_ready.load(),
            model_path
        };
    }

private:
    /**
     * Initialize the emotion analysis model and context for the plugin
     */
    bool InitializeEmotionModel() {
        if (!llama_manager || !settings_manager) {
            LOG_ERROR_EmoTagPlugin("LlamaManager or SettingsManager not available");
            if (status_callback) status_callback("Error: Core services not available", true);
            return false;
        }
        
        LOG_EmoTagPlugin("Initializing emotion analysis model...");
        if (status_callback) status_callback("Initializing emotion analysis model...", false);
        
        // Get emotion model configuration from settings
        std::string emotion_model_path = settings_manager->GetString("Models", "emotion_model_path", "");
        if (emotion_model_path.empty()) {
            LOG_WARNING_EmoTagPlugin("No emotion model path configured - plugin will be disabled");
            if (status_callback) status_callback("No emotion model configured", true);
            return false;
        }
        
        // Get main model settings to derive emotion model config
        int main_context_size = settings_manager->GetInt("Models", "main_context_size", 4096);
        int main_gpu_layers = settings_manager->GetInt("Models", "main_gpu_layers", 999);
        
        // Context size calculation for emotional analysis
        // Need enough space for: system prompt (~300 tokens) + AI responses (up to 1000 tokens each for 2 responses)
        // + formatting overhead (~100 tokens) + response generation (~200 tokens)
        // Total: ~2600 tokens minimum for typical analysis
        // Use 30% of main context size with minimum 3000 tokens
        int emotion_context_size = std::max(3000, (main_context_size * 30) / 100);
        
        LOG_EmoTagPlugin("Loading emotion model: " + emotion_model_path + 
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
                LOG_ERROR_EmoTagPlugin("Failed to load emotion model: " + emotion_model_path);
                if (status_callback) status_callback("Failed to load emotion model", true);
                return false;
            }
            
            LOG_EmoTagPlugin("Emotion model loaded successfully");
            if (status_callback) status_callback("Creating emotion context...", false);
            
            // Create emotion context
            auto* context_ptr = llama_manager->GetOrCreateContextInfo(emotion_context_id, emotion_model_id, emotion_context_size);
            if (!context_ptr) {
                LOG_ERROR_EmoTagPlugin("Failed to create emotion context");
                if (status_callback) status_callback("Failed to create emotion context", true);
                return false;
            }
            
            // Store the context (we don't own it, LlamaManager does)
            emotion_context = nullptr; // We'll access it through LlamaManager
            
            LOG_EmoTagPlugin("Emotion context created: " + emotion_context_id);
            if (status_callback) status_callback("Applying emotion system prompt...", false);
            
            // Apply emotion analysis system prompt from settings
            std::string emotion_system_prompt = settings_manager->GetString("Emotion", "system_prompt",
                "You are an emotional state analyzer. When given AI assistant responses, analyze the emotional tone, "
                "mood, and psychological state conveyed in the text. Provide a brief emotional overview that captures "
                "the assistant's apparent emotional state, confidence level, and overall demeanor. "
                "Focus on identifying patterns like: confident, uncertain, empathetic, analytical, cheerful, "
                "cautious, enthusiastic, or reserved. Keep your analysis concise and actionable.");
            
            if (!emotion_system_prompt.empty()) {
                context_ptr->UpdateSystemPrompt(emotion_system_prompt);
                LOG_EmoTagPlugin("Applied emotion system prompt to context");
            }
            
            emotion_model_ready = true;
            LOG_EmoTagPlugin("Emotion model initialization completed successfully");
            if (status_callback) status_callback("Emotion model ready", false);
            return true;
            
        } catch (const std::exception& e) {
            LOG_ERROR_EmoTagPlugin("Exception during emotion model initialization: " + std::string(e.what()));
            if (status_callback) status_callback("Error: " + std::string(e.what()), true);
            return false;
        }
    }
    
    /**
     * Main processing loop - runs in background thread for maintenance
     * Note: Analysis is now triggered immediately, this loop is mainly for cleanup
     */
    void ProcessingLoop() {
        LOG_EmoTagPlugin("Processing loop started (maintenance mode)");
        
        while (!should_stop.load()) {
            try {
                ProcessPendingEmotionalAnalysis(); // Mainly for maintenance now
                std::this_thread::sleep_for(check_interval);
            } catch (const std::exception& e) {
                LOG_ERROR_EmoTagPlugin("Exception in processing loop: " + std::string(e.what()));
                std::this_thread::sleep_for(check_interval * 2); // Back off on error
            }
        }
        
        LOG_EmoTagPlugin("Processing loop ended");
    }
    
    /**
     * Check for and process any contexts ready for emotional analysis
     * Note: This is now mainly for cleanup and maintenance since analysis is triggered directly
     */
    void ProcessPendingEmotionalAnalysis() {
        // Skip processing if emotion model is not ready
        if (!emotion_model_ready.load()) {
            return;
        }
        
        // This method is now primarily for maintenance/cleanup
        // Analysis is triggered immediately in RecordAIResponse
    }
    
    /**
     * Sanitize content to prevent template token conflicts
     */
    std::string SanitizeContentForAnalysis(const std::string& content) {
        std::string sanitized = content;
        
        // Replace reserved template tokens with safe alternatives
        // This prevents the template engine from treating them as control tokens
        size_t pos = 0;
        
        // Replace <|start_header_id|> with [START_HEADER]
        while ((pos = sanitized.find("<|start_header_id|>", pos)) != std::string::npos) {
            sanitized.replace(pos, 19, "[START_HEADER]");
            pos += 14; // Length of "[START_HEADER]"
        }
        
        // Replace <|end_header_id|> with [END_HEADER]
        pos = 0;
        while ((pos = sanitized.find("<|end_header_id|>", pos)) != std::string::npos) {
            sanitized.replace(pos, 17, "[END_HEADER]");
            pos += 12; // Length of "[END_HEADER]"
        }
        
        // Replace <|eot_id|> with [END_OF_TURN]
        pos = 0;
        while ((pos = sanitized.find("<|eot_id|>", pos)) != std::string::npos) {
            sanitized.replace(pos, 10, "[END_OF_TURN]");
            pos += 13; // Length of "[END_OF_TURN]"
        }
        
        return sanitized;
    }
    
    /**
     * Process emotional analysis for a specific context (internal method)
     */
    void ProcessEmotionalAnalysisForContext(const std::string& context_id, const ContextResponseHistory& history) {
        if (history.messages.empty()) {
            return; // Nothing to analyze
        }
        
        LOG_EmoTagPlugin("Processing emotional analysis for context: " + context_id + 
                        " (" + std::to_string(history.messages.size()) + " messages)");
        
        if (!emotion_model_ready.load() || !llama_manager) {
            LOG_ERROR_EmoTagPlugin("Emotion model not ready for analysis");
            return;
        }
        
        try {
            // Get the emotion context
            auto* emotion_ctx = llama_manager->GetContextInfo(emotion_context_id);
            if (!emotion_ctx) {
                LOG_ERROR_EmoTagPlugin("Emotion context not available");
                return;
            }
            
            // Build analysis prompt
            std::ostringstream analysis_stream;
            analysis_stream << "Analyze the emotional state of the AI assistant based on these recent messages:\n\n";
            
            for (size_t i = 0; i < history.messages.size(); ++i) {
                const auto& [role, content] = history.messages[i];
                if (role == "assistant") {
                    analysis_stream << "AI Response " << (i + 1) << ":\n";
                    analysis_stream << SanitizeContentForAnalysis(content) << "\n\n";
                } else if (role == "user" && include_user_messages.load()) {
                    analysis_stream << "User Message " << (i + 1) << ":\n";
                    analysis_stream << SanitizeContentForAnalysis(content) << "\n\n";
                }
            }
            
            analysis_stream << "Provide a brief emotional state overview (2-3 sentences) describing the assistant's "
                           << "emotional tone, confidence level, and overall demeanor based on these messages.\n\n"
                           << "Emotional Analysis:";
            
            std::string analysis_prompt = analysis_stream.str();
            
            LOG_EmoTagPlugin("Built analysis prompt (" + std::to_string(analysis_prompt.length()) + " chars) for context: " + context_id);
            
            // Reset context for clean analysis
            emotion_ctx->ClearContext();
            
            std::string emotional_state;
            try {
                emotional_state = emotion_ctx->HandleInput(analysis_prompt, "user");
                LOG_EmoTagPlugin("Emotion analysis completed successfully for context: " + context_id);
            } catch (const std::exception& analysis_e) {
                LOG_ERROR_EmoTagPlugin("Exception during emotional analysis for context " + context_id + ": " + std::string(analysis_e.what()));
                return;
            }
            
            // Check if we got a valid response
            if (emotional_state.empty() || emotional_state.find("Error: Failed to process prompt") == 0) {
                LOG_WARNING_EmoTagPlugin("Failed to generate emotional analysis for context: " + context_id);
                return;
            }
            
            // Sanitize emotional state before applying to prevent template token conflicts
            std::string sanitized_emotional_state = SanitizeContentForAnalysis(emotional_state);
            
            // Apply emotional state to original context
            ApplyEmotionalStateToContext(context_id, sanitized_emotional_state);
            
            responses_analyzed += history.messages.size();
            emotional_states_generated++;
            
            LOG_EmoTagPlugin("Successfully generated emotional analysis for context: " + context_id + 
                           " - " + emotional_state.substr(0, 100) + 
                           (emotional_state.length() > 100 ? "..." : ""));
            
        } catch (const std::exception& e) {
            LOG_ERROR_EmoTagPlugin("Exception processing emotional analysis: " + std::string(e.what()));
        }
    }
    
    /**
     * Apply generated emotional state back to the original context
     */
    void ApplyEmotionalStateToContext(const std::string& context_id, const std::string& emotional_state) {
        if (!llama_manager) {
            LOG_ERROR_EmoTagPlugin("LlamaManager not available for applying emotional state");
            return;
        }
        
        try {
            // Get the original context
            auto* original_context = llama_manager->GetContextInfo(context_id);
            if (!original_context) {
                LOG_WARNING_EmoTagPlugin("Original context not found for emotional state application: " + context_id);
                return;
            }
            
            // Apply the emotional state to the context
            original_context->UpdateEmotionalState(emotional_state);
            
            contexts_updated++;
            
            LOG_EmoTagPlugin("Applied emotional state to context: " + context_id);
            
        } catch (const std::exception& e) {
            LOG_ERROR_EmoTagPlugin("Exception applying emotional state to context: " + std::string(e.what()));
        }
    }
};

} // namespace LuminaChat
