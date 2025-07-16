#pragma once

#include "../Context/ContextInfo.hpp"
#include "../ProcessingPipeline.hpp"
#include "../Orchestrator.hpp"
#include "../LlamaManager.hpp"
#include "../SettingsManager.hpp"
#include "../Logger.hpp"
#include "../ErrorHandling.hpp"
#include "../Utilities.hpp"
#include <chrono>
#include <thread>
#include <memory>
#include <optional>
#include <sstream>
#include <iomanip>
#include <deque>
#include <mutex>
#include <atomic>
#include <functional>

namespace LuminaChat {

// Common callback type for status updates
using StatusUpdateCallback = std::function<void(const std::string& status, bool is_error)>;

/**
 * PluginUtils - Common utility functions used by all plugins
 * This static class provides the shared functionality that was duplicated across plugins
 */
class PluginUtils {
public:
    // Common error checking
    static bool IsErrorResponse(const std::string& response) {
        return response.empty() ||
               response.find("Error:") == 0 ||
               response.find("Failed to") == 0 ||
               response.find("Exception") != std::string::npos ||
               response.find("unavailable") != std::string::npos ||
               response.find("not ready") != std::string::npos;
    }
    
    // Common model configuration calculation
    static int CalculatePluginContextSize(SettingsManager* settings_manager) {
        if (!settings_manager) return 4500;
        int outer_context_size = settings_manager->GetInt("Models", "outer_context_size", 4096);
        return std::max(4500, (outer_context_size * 50) / 100);
    }
    
    // Common model loading configuration
    static ModelConfig CreatePluginModelConfig(const std::string& model_path, SettingsManager* settings_manager) {
        ModelConfig config;
        config.model_path = model_path;
        config.context_size = CalculatePluginContextSize(settings_manager);
        config.gpu_layers = settings_manager ? settings_manager->GetInt("Models", "outer_gpu_layers", 999) : 999;
        return config;
    }
    
    // Common context resize size calculation
    static size_t CalculateResizeTokens(size_t needed_tokens) {
        return static_cast<size_t>(needed_tokens * 1.25f) + 500;
    }
    
    // Common generation retry estimation
    static size_t EstimateNeededTokens(const std::string& prompt) {
        return static_cast<size_t>((prompt.length() / 3.0f) * 1.25f) + 500;
    }
};

/**
 * BasePlugin - Common infrastructure template for all LuminaChat plugins
 * 
 * This base class provides the unified architecture that both SummarizationPlugin
 * and EmoTagPlugin follow, eliminating ~95% code duplication while keeping
 * plugin-specific logging and behavior.
 * 
 * Template Parameters:
 * - BatchType: The data structure processed by this plugin (PrunedMessageBatch, EmotionalAnalysisBatch)
 * - RequestType: The pipeline request structure 
 * - ResponseType: The pipeline response structure
 */
template<typename BatchType, typename RequestType, typename ResponseType>
class BasePlugin {
protected:
    // Core references - identical across all plugins
    Orchestrator* orchestrator = nullptr;
    LlamaManager* llama_manager = nullptr;
    SettingsManager* settings_manager = nullptr;
    
    // Model state - unified pattern
    std::atomic<bool> model_ready{false};
    StatusUpdateCallback status_callback;
    std::string model_id;
    std::string context_id;
    std::string default_system_prompt;
    
    // Statistics - unified atomic pattern  
    struct Statistics {
        std::atomic<size_t> batches_processed{0};
        std::atomic<size_t> items_processed{0};  // messages_summarized OR responses_analyzed
        std::atomic<size_t> results_generated{0}; // summaries_applied OR emotional_states_generated  
        std::atomic<size_t> contexts_updated{0};
        
        void IncrementBatch(size_t item_count) {
            batches_processed.fetch_add(1, std::memory_order_relaxed);
            items_processed.fetch_add(item_count, std::memory_order_relaxed);
            results_generated.fetch_add(1, std::memory_order_relaxed);
        }
        
        void IncrementContextUpdate() {
            contexts_updated.fetch_add(1, std::memory_order_relaxed);
        }
    } stats;
    
    // Buffer management - unified pattern
    mutable std::mutex buffer_mutex;
    std::vector<BatchType> processing_buffer;
    
    // Debug infrastructure - unified pattern
    struct DebugInfo {
        mutable std::mutex mutex;
        std::deque<std::string> log_entries;
        std::optional<std::tuple<std::string, std::string, std::string, std::chrono::system_clock::time_point>> last_generation;
        static constexpr size_t MAX_LOG_ENTRIES = 100;
        
        void AddLog(const std::string& level, const std::string& message) {
            std::lock_guard<std::mutex> lock(mutex);
            log_entries.emplace_back("[" + level + "] " + message);
            while (log_entries.size() > MAX_LOG_ENTRIES) {
                log_entries.pop_front();
            }
        }
        
        void UpdateGeneration(const std::string& input, const std::string& output, const std::string& context_id) {
            std::lock_guard<std::mutex> lock(mutex);
            last_generation = std::make_tuple(input, output, context_id, std::chrono::system_clock::now());
        }
    } debug;

public:
    explicit BasePlugin(Orchestrator* orch, const std::string& plugin_name, 
                       const std::string& model_id_param, const std::string& context_id_param,
                       const std::string& default_prompt = "")
        : orchestrator(orch), model_id(model_id_param), context_id(context_id_param), 
          default_system_prompt(default_prompt) {
        if (orchestrator) {
            llama_manager = orchestrator->GetLlamaManager();
            settings_manager = orchestrator->GetSettingsManager();
        }
        // Note: Derived classes must call their own LogInfo for plugin-specific logging
    }
    
    virtual ~BasePlugin() {
        model_ready = false;
    }
    
    // Common lifecycle
    bool Initialize() { return InitializeModel(); }
    
    void Shutdown() {
        model_ready = false;
        // Note: Derived classes must call their own LogInfo for plugin-specific logging
    }
    
    void SetStatusCallback(StatusUpdateCallback callback) { 
        status_callback = std::move(callback); 
    }
    
    bool IsReady() const { return model_ready.load(); }
    
    // Common statistics structure
    struct PluginStats {
        size_t batches_processed, items_processed, results_generated, contexts_updated;
        bool model_ready;
        std::string model_path;
    };
    
    PluginStats GetStats() const {
        return {
            stats.batches_processed.load(),
            stats.items_processed.load(), 
            stats.results_generated.load(),
            stats.contexts_updated.load(),
            model_ready.load(),
            GetModelPath()
        };
    }
    
    // Common buffer management
    std::vector<BatchType> GetAndClearBuffer() {
        std::lock_guard<std::mutex> lock(buffer_mutex);
        return std::exchange(processing_buffer, {});
    }
    
    bool HasPendingWork() const {
        std::lock_guard<std::mutex> lock(buffer_mutex);
        return !processing_buffer.empty();
    }
    
    void AddToBuffer(BatchType&& batch) {
        std::lock_guard<std::mutex> lock(buffer_mutex);
        processing_buffer.emplace_back(std::move(batch));
    }
    
    // Common debug access
    std::vector<std::string> GetLogHistory() const {
        std::lock_guard<std::mutex> lock(debug.mutex);
        return std::vector<std::string>(debug.log_entries.begin(), debug.log_entries.end());
    }
    
    struct LastGenerationInfo {
        bool has_generation;
        std::string input, output, context_id, timestamp;
    };
    
    LastGenerationInfo GetLastGeneration() const {
        std::lock_guard<std::mutex> lock(debug.mutex);
        if (!debug.last_generation.has_value()) {
            return {false, "", "", "", ""};
        }
        
        auto& [input, output, context_id, timestamp] = *debug.last_generation;
        auto time_t = std::chrono::system_clock::to_time_t(timestamp);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        
        return {true, input, output, context_id, ss.str()};
    }

    // Virtual methods for plugin-specific behavior
    virtual ResponseType ProcessRequest(const RequestType& request) = 0;
    virtual bool ApplyResult(const std::string& context_id, const std::string& result) = 0;
    virtual std::string GetModelPath() const = 0;
    virtual std::string GetModelPathSettingKey() const = 0;
    virtual std::string GetSystemPromptSettingKey() const = 0;
    virtual std::string GetPluginName() const = 0;
    
    // Pure virtual logging methods - implemented by derived classes with their specific macros
    virtual void LogInfo(const std::string& message) = 0;
    virtual void LogWarning(const std::string& message) = 0;
    virtual void LogError(const std::string& message) = 0;
    virtual void LogDebug(const std::string& message) = 0;

protected:
    // Common debug tracking
    void StoreDebugGeneration(const std::string& input, const std::string& output, const std::string& context_id) {
        debug.UpdateGeneration(input, output, context_id);
    }

    // Common model initialization - implemented as template method pattern
    virtual bool InitializeModel() {
        if (!llama_manager || !settings_manager) {
            LogError("Core services not available");
            if (status_callback) status_callback("Error: Core services not available", true);
            return false;
        }
        
        std::string model_path = settings_manager->GetString("Models", GetModelPathSettingKey(), "");
        if (model_path.empty()) {
            LogWarning("No model path configured - plugin will be disabled");
            if (status_callback) status_callback("No model configured", true);
            return false;
        }
        
        LogInfo("Initializing model...");
        if (status_callback) status_callback("Initializing model...", false);
        
        try {
            // Use utility function for model configuration
            ModelConfig config = PluginUtils::CreatePluginModelConfig(model_path, settings_manager);
            
            LogInfo("Loading model: " + model_path + 
                    " (context: " + std::to_string(config.context_size) + 
                    " [50% of outer], gpu_layers: " + std::to_string(config.gpu_layers) + ")");
            
            if (status_callback) status_callback("Loading model: " + model_path, false);
            
            // Load model
            if (!llama_manager->LoadModel(model_id, config)) {
                LogError("Failed to load model: " + model_path);
                if (status_callback) status_callback("Failed to load model", true);
                return false;
            }
            
            LogInfo("Model loaded successfully");
            if (status_callback) status_callback("Creating context...", false);
            
            // Create context
            auto* context_ptr = llama_manager->GetOrCreateContextInfo(context_id, model_id, config.context_size);
            if (!context_ptr) {
                LogError("Failed to create context");
                if (status_callback) status_callback("Failed to create context", true);
                return false;
            }
            
            // Configure context
            context_ptr->SetPruningEligible(false);
            LogInfo("Marked context as not eligible for pruning: " + context_id);
            
            std::string system_prompt = settings_manager->GetString("Models", GetSystemPromptSettingKey(), default_system_prompt);
            if (!system_prompt.empty()) {
                context_ptr->UpdateSystemPrompt(system_prompt);
                LogInfo("Applied system prompt to context");
            }
            
            // Ensure context is in proper state
            if (context_ptr->GetState() == ContextState::ERROR_STATE) {
                LogWarning("Context created in error state - forcing reset");
                auto previous_state = context_ptr->ForceResetToIdle("initial creation recovery");
                LogInfo("Forced context reset from state " + std::to_string(static_cast<int>(previous_state)) + " to IDLE");
            }
            
            // Perform initialization test
            LogInfo("Performing initialization test on context...");
            std::string test_result = context_ptr->HandleInput("Test initialization. Respond with: Ready", "system");
            context_ptr->ClearContext();
            
            if (PluginUtils::IsErrorResponse(test_result)) {
                LogError("Context initialization test failed: " + test_result);
                if (status_callback) status_callback("Context test failed", true);
                return false;
            }
            
            LogInfo("Context initialization test passed");
            
            model_ready = true;
            LogInfo("Model initialization completed successfully");
            if (status_callback) status_callback("Model ready", false);
            return true;
            
        } catch (const std::exception& e) {
            LogError("Exception during model initialization: " + std::string(e.what()));
            if (status_callback) status_callback("Error: " + std::string(e.what()), true);
            return false;
        }
    }
    
    // Common context resize logic using utility functions
    bool ResizeContext(size_t needed_tokens) {
        if (!llama_manager || !settings_manager) {
            LogError("Core services unavailable for context resize");
            return false;
        }
        
        auto* plugin_ctx = llama_manager->GetContextInfo(context_id);
        if (!plugin_ctx) {
            LogError("Context unavailable for resize");
            return false;
        }
        
        // Use utility function for size calculation
        size_t new_context_size = PluginUtils::CalculateResizeTokens(needed_tokens);
        int outer_context_size = settings_manager->GetInt("Models", "outer_context_size", 4096);
        
        if (new_context_size > static_cast<size_t>(outer_context_size)) {
            LogWarning("Cannot resize to " + std::to_string(new_context_size) + " tokens - exceeds limit");
            return false;
        }
        
        LogInfo("Resizing context to " + std::to_string(new_context_size) + 
                " tokens (needed: " + std::to_string(needed_tokens) + ")");
        
        try {
            // Store system prompt, remove old context, create new one
            std::string system_prompt = settings_manager->GetString("Models", GetSystemPromptSettingKey(), default_system_prompt);
            
            llama_manager->RemoveContext(context_id);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            
            auto* new_context_ptr = llama_manager->GetOrCreateContextInfo(context_id, model_id, 
                                                                         static_cast<int32_t>(new_context_size));
            if (!new_context_ptr) {
                LogError("Failed to create resized context");
                return false;
            }
            
            new_context_ptr->SetPruningEligible(false);
            if (!system_prompt.empty()) {
                new_context_ptr->UpdateSystemPrompt(system_prompt);
            }
            
            // Ensure new context is in proper state
            if (new_context_ptr->GetState() == ContextState::ERROR_STATE) {
                LogWarning("New context created in error state - forcing reset");
                auto previous_state = new_context_ptr->ForceResetToIdle("resize recovery");
                LogInfo("Forced new context reset from state " + std::to_string(static_cast<int>(previous_state)) + " to IDLE");
            }
            
            LogInfo("Successfully resized context to " + std::to_string(new_context_size) + " tokens");
            return true;
            
        } catch (const std::exception& e) {
            LogError("Exception during context resize: " + std::string(e.what()));
            return false;
        }
    }
    
    // Common generation pattern - template method using plugin-specific logging
    std::string PerformGeneration(const std::string& prompt, const std::string& batch_context_id) {
        auto* plugin_ctx = llama_manager->GetContextInfo(context_id);
        if (!plugin_ctx) {
            LogError("Context not available");
            return "";
        }
        
        try {
            // Check if context is in error state and attempt recovery
            if (plugin_ctx->GetState() == ContextState::ERROR_STATE) {
                LogWarning("Context in error state - attempting recovery for: " + batch_context_id);
                auto previous_state = plugin_ctx->ForceResetToIdle("generation recovery");
                LogInfo("Forced context reset from state " + std::to_string(static_cast<int>(previous_state)) + " to IDLE");
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            
            // Perform generation with context management
            plugin_ctx->ClearContext();
            std::string result = plugin_ctx->HandleInput(prompt, "user");
            plugin_ctx->ClearContext();
            
            // Store generation for debugging
            StoreDebugGeneration(prompt, result, batch_context_id);
            
            LogInfo("Generation completed for: " + batch_context_id + 
                   " - Generated: " + result.substr(0, 100) + 
                   (result.length() > 100 ? "..." : ""));
            
            // Check if generation failed and attempt resize if needed
            if (PluginUtils::IsErrorResponse(result)) {
                LogWarning("Initial generation failed for " + batch_context_id + " - attempting resize and retry...");
                return AttemptGenerationWithResize(prompt, batch_context_id);
            }
            
            return result;
            
        } catch (const std::exception& e) {
            LogError("Exception in generation: " + std::string(e.what()));
            return "";
        }
    }
    
    // Common generation retry with resize using utility functions
    std::string AttemptGenerationWithResize(const std::string& prompt, const std::string& batch_context_id) {
        size_t needed_tokens = PluginUtils::EstimateNeededTokens(prompt);
        LogInfo("Estimated needed tokens: " + std::to_string(needed_tokens));
        
        if (!ResizeContext(needed_tokens)) {
            LogWarning("Context resize failed - skipping: " + batch_context_id);
            return "";
        }
        
        LogInfo("Context resize successful, retrying generation for: " + batch_context_id);
        
        auto* new_plugin_ctx = llama_manager->GetContextInfo(context_id);
        if (!new_plugin_ctx) {
            LogError("Lost context after resize - skipping generation");
            return "";
        }
        
        try {
            // Ensure new context is in good state
            if (new_plugin_ctx->GetState() == ContextState::ERROR_STATE) {
                LogWarning("Resized context still in error state - forcing reset");
                auto previous_state = new_plugin_ctx->ForceResetToIdle("post-resize recovery");
                LogInfo("Forced resized context reset from state " + std::to_string(static_cast<int>(previous_state)) + " to IDLE");
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            
            new_plugin_ctx->ClearContext();
            std::string retry_result = new_plugin_ctx->HandleInput(prompt, "user");
            new_plugin_ctx->ClearContext();
            
            // Update debug info with retry result
            StoreDebugGeneration(prompt, retry_result, batch_context_id);
            
            LogInfo("Retry generation completed for: " + batch_context_id);
            
            // Final validation
            if (PluginUtils::IsErrorResponse(retry_result)) {
                LogWarning("Generation still failed after context resize - skipping: " + batch_context_id);
                return "";
            }
            
            LogInfo("Generation succeeded after context resize for: " + batch_context_id);
            return retry_result;
            
        } catch (const std::exception& e) {
            LogError("Exception in generation retry: " + std::string(e.what()));
            return "";
        }
    }
};

} // namespace LuminaChat