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

namespace LuminaChat {

// Callback type for status updates
using StatusUpdateCallback = std::function<void(const std::string& status, bool is_error)>;

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
class SummarizationPlugin {
private:
    // No independent pipeline - Orchestrator coordinates workflow
    
    // Reference to orchestrator for context management
    Orchestrator* orchestrator = nullptr;
    
    // Summary model and context management
    LlamaManager* llama_manager = nullptr;
    SettingsManager* settings_manager = nullptr;
    std::unique_ptr<ContextInfo> summary_context;
    std::string summary_model_id = "summary_model";
    std::string summary_context_id = "plugin_summary_context";
    std::atomic<bool> summary_model_ready{false};
    
    // Status callback for UI updates
    StatusUpdateCallback status_callback;
    
    // Statistics
    std::atomic<size_t> batches_processed{0};
    std::atomic<size_t> messages_summarized{0};
    std::atomic<size_t> summaries_applied{0};
    
    // Buffer management for pruned messages (moved from ContextInfo)
    mutable std::mutex pruning_buffer_mutex;
    std::vector<PrunedMessageBatch> pruning_buffer;
    
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
    explicit SummarizationPlugin(Orchestrator* orch) 
        : orchestrator(orch) {
        
        if (orchestrator) {
            llama_manager = orchestrator->GetLlamaManager();
            settings_manager = orchestrator->GetSettingsManager();
        }
        
        LogInfo("SummarizationPlugin initialized as processor service");
    }
    
    ~SummarizationPlugin() {
        Shutdown();
    }
    
    /**
     * Initialize the plugin (called by Orchestrator)
     */
    bool Initialize() {
        return InitializeSummaryModel();
    }
    
    /**
     * Shutdown the plugin and clean up resources
     */
    void Shutdown() {
        // Clean up summary context and model
        summary_context.reset();
        summary_model_ready = false;
        
        LogInfo("SummarizationPlugin shutdown");
    }
    
    /**
     * Set status update callback for UI notifications
     */
    void SetStatusCallback(StatusUpdateCallback callback) {
        status_callback = callback;
    }
    
    /**
     * Process a summarization request (called by Orchestrator pipeline)
     * This is the main processing method used by the Orchestrator's pipeline
     */
    SummarizationResponse ProcessSummarizationRequest(const SummarizationRequest& request) {
        try {
            LogInfo("Processing summarization for context: " + request.context_id + 
                   " (" + std::to_string(request.batch.pruned_messages.size()) + " messages)");
            
            if (!summary_model_ready.load() || !llama_manager) {
                std::string error_msg = "Summary model not ready for processing batch";
                LogError(error_msg);
                return SummarizationResponse("", request.context_id, false, 0, error_msg);
            }
            
            // Process the batch to generate summary (existing logic)
            std::string summary = ProcessSummarizationBatch(request.batch);
            
            if (summary.empty()) {
                std::string error_msg = "Failed to generate summary for context: " + request.context_id;
                LogError(error_msg);
                return SummarizationResponse("", request.context_id, false, 0, error_msg);
            }
            
            // Create success response with the generated summary
            SummarizationResponse response{
                summary,
                request.context_id,
                true,
                request.batch.pruned_messages.size()
            };
            
            LogInfo("Successfully generated summary for batch: " + request.context_id);
            
            // Update statistics
            messages_summarized += request.batch.pruned_messages.size();
            batches_processed++;
            
            return response;
            
        } catch (const std::exception& e) {
            std::string error_msg = "Exception processing summarization: " + std::string(e.what());
            LogError(error_msg);
            return SummarizationResponse("", request.context_id, false, 0, error_msg);
        }
    }
    
    /**
     * Apply a summary to a context (called by Orchestrator after successful processing)
     */
    bool ApplySummaryToContext(const std::string& context_id, const std::string& summary) {
        try {
            if (!llama_manager) {
                LogError("LlamaManager not available for applying summary");
                return false;
            }
            
            // Check if this is an inner voice summary (indicated by "_inner_voice_summary" suffix)
            if (context_id.find("_inner_voice_summary") != std::string::npos) {
                return ApplyInnerVoiceSummary(context_id, summary);
            }
            
            // Get the original context
            auto* original_context = llama_manager->GetContextInfo(context_id);
            if (!original_context) {
                LogWarning("Original context not found for summary application: " + context_id);
                return false;
            }
            
            // Apply the summary to the context
            if (summary.empty()) {
                LogWarning("Summary is empty - skipping update for context: " + context_id);
                return false;
            }
            
            // Try to acquire plugin processing lock for safe summary application
            if (!original_context->TryAcquirePluginProcessing("SummarizationPlugin")) {
                LogWarning("Could not acquire plugin processing lock for context: " + context_id + " - context may be busy");
                return false;
            }
            
            try {
                // Apply the clean summary to the context
                original_context->ApplyCompletedSummary(summary);
                
                summaries_applied++;
                
                LogInfo("Applied summary to context: " + context_id);
                
                // Release plugin processing lock
                [[maybe_unused]] bool released = original_context->ReleasePluginProcessing("SummarizationPlugin");
                return true;
                
            } catch (const std::exception&) {
                // Release plugin processing lock on exception
                [[maybe_unused]] bool released = original_context->ReleasePluginProcessing("SummarizationPlugin");
                throw; // Re-throw the exception
            }
            
        } catch (const std::exception& e) {
            LogError("Exception applying summary to context: " + std::string(e.what()));
            return false;
        }
    }
    
    /**
     * Apply inner voice summary to the appropriate inner context
     * This method handles summaries of user input/AI output pairs from inner voice contexts
     * and applies them to build up a running summary of reasoning thoughts
     */
    bool ApplyInnerVoiceSummary(const std::string& summary_id, const std::string& summary) {
        try {
            if (!llama_manager) {
                LogError("LlamaManager not available for applying inner voice summary");
                return false;
            }
            
            // Extract the original context ID from the summary ID
            // Format: "inner_context_inner_voice_summary" -> "inner_context"
            std::string original_context_id = summary_id;
            size_t suffix_pos = original_context_id.find("_inner_voice_summary");
            if (suffix_pos != std::string::npos) {
                original_context_id = original_context_id.substr(0, suffix_pos);
            }
            
            // Get the inner context
            auto* inner_context = llama_manager->GetContextInfo(original_context_id);
            if (!inner_context) {
                LogWarning("Inner context not found for summary application: " + original_context_id);
                return false;
            }
            
            // Apply the summary to the inner context
            if (summary.empty()) {
                LogWarning("Inner voice summary is empty - skipping update for context: " + original_context_id);
                return false;
            }
            
            // Try to acquire plugin processing lock for safe summary application
            if (!inner_context->TryAcquirePluginProcessing("SummarizationPlugin_InnerVoice")) {
                LogWarning("Could not acquire plugin processing lock for inner voice context: " + original_context_id + " - context may be busy");
                return false;
            }
            
            try {
                // Apply the summary to the inner context (summary already includes "Summary: " prefix)
                inner_context->ApplyCompletedSummary(summary);
                
                summaries_applied++;
                
                LogInfo("Applied inner voice summary to context: " + original_context_id + 
                       " - " + summary.substr(0, 100) + (summary.length() > 100 ? "..." : ""));
                
                // Release plugin processing lock
                [[maybe_unused]] bool released = inner_context->ReleasePluginProcessing("SummarizationPlugin_InnerVoice");
                return true;
                
            } catch (const std::exception&) {
                // Release plugin processing lock on exception
                [[maybe_unused]] bool released = inner_context->ReleasePluginProcessing("SummarizationPlugin_InnerVoice");
                throw; // Re-throw the exception
            }
            
        } catch (const std::exception& e) {
            LogError("Exception applying inner voice summary to context: " + std::string(e.what()));
            return false;
        }
    }
    
    /**
     * Get plugin statistics 
     */
    struct PluginStats {
        size_t batches_processed;
        size_t messages_summarized;
        size_t summaries_applied;
        bool summary_model_ready;
        std::string summary_model_path;
    };
    
    PluginStats GetStats() const {
        std::string model_path = "";
        if (settings_manager) {
            model_path = settings_manager->GetString("Models", "summary_model_path", "");
        }
        
        return {
            batches_processed.load(),
            messages_summarized.load(),
            summaries_applied.load(),
            summary_model_ready.load(),
            model_path
        };
    }
    
    /**
     * Check if the plugin is ready to process summarization requests
     */
    bool IsReady() const {
        return summary_model_ready.load();
    }
    
    // Buffer management methods (moved from ContextInfo)
    /**
     * Get and clear all pending pruned message batches for summarization
     */
    std::vector<PrunedMessageBatch> GetAndClearPruningBuffer() {
        std::lock_guard<std::mutex> lock(pruning_buffer_mutex);
        
        std::vector<PrunedMessageBatch> result;
        result.swap(pruning_buffer);
        
        return result;
    }
    
    /**
     * Check if there are pending messages waiting for summarization
     */
    bool HasPendingSummarization() const {
        std::lock_guard<std::mutex> lock(pruning_buffer_mutex);
        return !pruning_buffer.empty();
    }
    
    /**
     * Add a pruned message batch to the buffer for summarization
     */
    void AddToPruningBuffer(PrunedMessageBatch&& batch) {
        std::lock_guard<std::mutex> lock(pruning_buffer_mutex);
        pruning_buffer.emplace_back(std::move(batch));
    }
    
    /**
     * Request summarization for a specific context (following EmoTagPlugin pattern)
     */
    void RequestSummarization(const std::string& context_id, const std::vector<std::pair<std::string, std::string>>& pruned_messages) {
        if (pruned_messages.empty()) {
            LogInfo("No messages provided for summarization request - context: " + context_id);
            return;
        }
        
        LogInfo("Received summarization request for context " + context_id + 
               " with " + std::to_string(pruned_messages.size()) + " pruned messages");
        
        // Create batch and add to buffer
        PrunedMessageBatch batch(context_id, pruned_messages);
        AddToPruningBuffer(std::move(batch));
        
        LogInfo("Queued summarization batch for context " + context_id + 
               " - " + std::to_string(pruned_messages.size()) + " messages in buffer");
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
        LOG_SummarizationPlugin(message);
        AddLogEntry("[INFO] " + message);
    }
    
    void LogWarning(const std::string& message) {
        LOG_WARNING_SummarizationPlugin(message);
        AddLogEntry("[WARN] " + message);
        // Also use unified error handling for consistency
        HandleWarning(message, "SummarizationPlugin", false);
    }
    
    void LogError(const std::string& message) {
        LOG_ERROR_SummarizationPlugin(message);
        AddLogEntry("[ERROR] " + message);
        // Also use unified error handling for consistency and UI notification
        HandleError(message, "SummarizationPlugin", false);
    }
    
    void LogDebug(const std::string& message) {
        LOG_DEBUG_SummarizationPlugin(message);
        AddLogEntry("[DEBUG] " + message);
    }
    
    /**
     * Dynamically resize summary context to handle larger message pairs
     * @param needed_tokens The minimum tokens needed for processing
     * @return true if resize was successful, false otherwise
     */
    bool ResizeSummaryContext(size_t needed_tokens) {
        if (!llama_manager || !settings_manager) {
            LOG_ERROR_SummarizationPlugin("LlamaManager or SettingsManager not available for context resize");
            return false;
        }
        
        // Get current summary context
        auto* summary_ctx = llama_manager->GetContextInfo(summary_context_id);
        if (!summary_ctx) {
            LOG_ERROR_SummarizationPlugin("Summary context not available for resize");
            return false;
        }
        
        // Calculate new context size (needed tokens + 5% buffer)
        size_t new_context_size = static_cast<size_t>(needed_tokens * 1.05f);
        
        // Get outer model context size as hard limit
        int outer_context_size = settings_manager->GetInt("Models", "outer_context_size", 4096);
        if (new_context_size > static_cast<size_t>(outer_context_size)) {
            LOG_WARNING_SummarizationPlugin("Cannot resize summary context to " + std::to_string(new_context_size) + 
                " tokens - would exceed outer context limit of " + std::to_string(outer_context_size));
            return false;
        }
        
        LOG_SummarizationPlugin("Resizing summary context from current size to " + std::to_string(new_context_size) + 
            " tokens (needed: " + std::to_string(needed_tokens) + ")");
        
        try {
            // Store system prompt before context destruction
            std::string system_prompt = settings_manager->GetString("Summary", "system_prompt", "");
            
            // Remove the old context (this will free the llama context but keep the model loaded)
            llama_manager->RemoveContext(summary_context_id);
            
            // Create new context with the larger size (model stays loaded)
            auto* new_context_ptr = llama_manager->GetOrCreateContextInfo(summary_context_id, summary_model_id, static_cast<int32_t>(new_context_size));
            if (!new_context_ptr) {
                LOG_ERROR_SummarizationPlugin("Failed to create resized summary context");
                return false;
            }
            
            // Reapply system prompt
            if (!system_prompt.empty()) {
                new_context_ptr->UpdateSystemPrompt(system_prompt);
                LOG_SummarizationPlugin("Reapplied system prompt to resized context");
            }
            
            LOG_SummarizationPlugin("Successfully resized summary context to " + std::to_string(new_context_size) + " tokens");
            return true;
            
        } catch (const std::exception& e) {
            LOG_ERROR_SummarizationPlugin("Exception during context resize: " + std::string(e.what()));
            return false;
        }
    }
    
private:
    /**
     * Initialize the summary model and context for the plugin
     */
    bool InitializeSummaryModel() {
        if (!llama_manager || !settings_manager) {
            LOG_ERROR_SummarizationPlugin("LlamaManager or SettingsManager not available");
            if (status_callback) status_callback("Error: Core services not available", true);
            return false;
        }
        
        LOG_SummarizationPlugin("Initializing summary model...");
        if (status_callback) status_callback("Initializing summary model...", false);
        
        // Get summary model configuration from settings
        std::string summary_model_path = settings_manager->GetString("Models", "summary_model_path", "");
        if (summary_model_path.empty()) {
            LOG_WARNING_SummarizationPlugin("No summary model path configured - plugin will be disabled");
            if (status_callback) status_callback("No summary model configured", true);
            return false;
        }
        
        // Get outer model settings to derive summary model config
        int outer_context_size = settings_manager->GetInt("Models", "outer_context_size", 4096);
        int outer_gpu_layers = settings_manager->GetInt("Models", "outer_gpu_layers", 999);
        
        // Context size calculation for pairwise summarization
        // Need enough space for: system prompt (~200 tokens) + 2 messages (up to 2000 tokens each) 
        // + formatting overhead (~100 tokens) + response generation (~200 tokens)
        // Total: ~4500 tokens minimum for large message pairs
        // Use 50% of outer context size with minimum 4500 tokens to handle large conversations
        int summary_context_size = std::max(4500, (outer_context_size * 50) / 100);
        
        LOG_SummarizationPlugin("Loading summary model: " + summary_model_path + 
                               " (context: " + std::to_string(summary_context_size) + 
                               " [50% of outer, optimized for large message pairs], gpu_layers: " + std::to_string(outer_gpu_layers) + ")");
        
        if (status_callback) status_callback("Loading summary model: " + summary_model_path, false);
        
        try {
            // Load summary model
            ModelConfig config;
            config.model_path = summary_model_path;
            config.context_size = summary_context_size;
            config.gpu_layers = outer_gpu_layers;
            
            if (!llama_manager->LoadModel(summary_model_id, config)) {
                LOG_ERROR_SummarizationPlugin("Failed to load summary model: " + summary_model_path);
                if (status_callback) status_callback("Failed to load summary model", true);
                return false;
            }
            
            LOG_SummarizationPlugin("Summary model loaded successfully");
            if (status_callback) status_callback("Creating summary context...", false);
            
            // Create summary context
            auto* context_ptr = llama_manager->GetOrCreateContextInfo(summary_context_id, summary_model_id, summary_context_size);
            if (!context_ptr) {
                LOG_ERROR_SummarizationPlugin("Failed to create summary context");
                if (status_callback) status_callback("Failed to create summary context", true);
                return false;
            }
            
            // Store the context (we don't own it, LlamaManager does)
            // But we keep a reference for easier access
            summary_context = nullptr; // We'll access it through LlamaManager
            
            LOG_SummarizationPlugin("Summary context created: " + summary_context_id);
            if (status_callback) status_callback("Applying summary system prompt...", false);
            
            // Apply summary system prompt from settings
            std::string summary_system_prompt = settings_manager->GetString("Summary", "system_prompt",
                "You are a helpful AI assistant that creates concise summaries of conversations. "
                "When given a conversation history, provide a clear and informative summary that captures "
                "the key points, decisions, and context. Focus on preserving important information while "
                "being concise. Format your summary in a structured way with bullet points when appropriate.");
            
            if (!summary_system_prompt.empty()) {
                context_ptr->UpdateSystemPrompt(summary_system_prompt);
                LOG_SummarizationPlugin("Applied summary system prompt to context");
            }
            
            summary_model_ready = true;
            LOG_SummarizationPlugin("Summary model initialization completed successfully");
            if (status_callback) status_callback("Summary model ready", false);
            return true;
            
        } catch (const std::exception& e) {
            LOG_ERROR_SummarizationPlugin("Exception during summary model initialization: " + std::string(e.what()));
            if (status_callback) status_callback("Error: " + std::string(e.what()), true);
            return false;
        }
    }
    
    /**
     * Process a single batch of pruned messages using pairwise summarization
     * Always summarizes 2 messages at a time (Q->A pairs) for maximum fidelity
     * Returns the generated summary or empty string on failure
     */
    std::string ProcessSummarizationBatch(const PrunedMessageBatch& batch) {
        if (!summary_model_ready.load() || !llama_manager) {
            LogError("Summary model not ready for processing batch");
            return "";
        }
        
        try {
            // Get the summary context
            auto* summary_ctx = llama_manager->GetContextInfo(summary_context_id);
            if (!summary_ctx) {
                LogError("Summary context not available");
                return "";
            }
            
            size_t total_messages = batch.pruned_messages.size();
            
            // Always use pairwise summarization - need at least 2 messages
            if (total_messages < 2) {
                LOG_SummarizationPlugin("Only " + std::to_string(total_messages) + 
                    " message(s) in batch - leaving in context without summarization");
                return "";
            }
            
            LOG_SummarizationPlugin("Using pairwise summarization for " + 
                std::to_string(total_messages) + " messages (" + 
                std::to_string(total_messages / 2) + " pairs)");
            
            std::vector<std::string> pair_summaries;
            
            // Process messages in pairs (Q->A), skip odd message at end
            for (size_t i = 0; i + 1 < total_messages; i += 2) {
                std::ostringstream pair_stream;
                pair_stream << "Summarize this Q&A exchange in one clear sentence:\n\n";
                pair_stream << batch.pruned_messages[i].first << ": " << batch.pruned_messages[i].second << "\n";
                pair_stream << batch.pruned_messages[i + 1].first << ": " << batch.pruned_messages[i + 1].second << "\n";
                pair_stream << "\nSummary:";
                
                std::string pair_prompt = pair_stream.str();
                
                // Reset context for each pair to ensure independence
                summary_ctx->ClearContext();
                
                std::string pair_summary;
                try {
                    pair_summary = summary_ctx->HandleInput(pair_prompt, "user");
                } catch (const std::exception& gen_e) {
                    pair_summary = "Error: Exception during generation - " + std::string(gen_e.what());
                    LOG_ERROR_SummarizationPlugin("Exception during pair generation: " + std::string(gen_e.what()));
                }
                
                // Always store generation for debugging (regardless of success/failure)
                {
                    std::lock_guard<std::mutex> lock(debug_mutex);
                    last_generation = DebugGeneration{
                        pair_prompt,
                        pair_summary,
                        batch.context_id,
                        std::chrono::system_clock::now()
                    };
                }
                
                // Check if we got an error (likely context overflow or exception)
                if (pair_summary.empty() || 
                    pair_summary.find("Error: Failed to process prompt") == 0 || 
                    pair_summary.find("Error: Exception during generation") == 0) {
                    LOG_WARNING_SummarizationPlugin("Failed to summarize pair " + std::to_string((i/2) + 1) + 
                        " - attempting context resize...");
                    
                    // Extract needed token count from error message (if available)
                    size_t needed_tokens = ExtractNeededTokensFromError(pair_summary);
                    if (needed_tokens == 0) {
                        // Estimate token count if we couldn't parse error message
                        needed_tokens = pair_prompt.length() / 4; // Rough estimation: 4 chars per token
                        LOG_SummarizationPlugin("Estimated needed tokens: " + std::to_string(needed_tokens));
                    } else {
                        LOG_SummarizationPlugin("Extracted needed tokens from error: " + std::to_string(needed_tokens));
                    }
                    
                    // Attempt to resize the summary context
                    if (ResizeSummaryContext(needed_tokens)) {
                        LOG_SummarizationPlugin("Context resize successful, retrying pair " + std::to_string((i/2) + 1));
                        
                        // Get the resized context and retry
                        summary_ctx = llama_manager->GetContextInfo(summary_context_id);
                        if (summary_ctx) {
                            summary_ctx->ClearContext();
                            try {
                                pair_summary = summary_ctx->HandleInput(pair_prompt, "user");
                            } catch (const std::exception& retry_e) {
                                pair_summary = "Error: Exception during retry generation - " + std::string(retry_e.what());
                                LOG_ERROR_SummarizationPlugin("Exception during retry generation: " + std::string(retry_e.what()));
                            }
                            
                            // Update last_generation with retry result
                            {
                                std::lock_guard<std::mutex> lock(debug_mutex);
                                last_generation = DebugGeneration{
                                    pair_prompt,
                                    pair_summary,
                                    batch.context_id,
                                    std::chrono::system_clock::now()
                                };
                            }
                            
                            if (pair_summary.empty() || 
                                pair_summary.find("Error: Failed to process prompt") == 0 || 
                                pair_summary.find("Error: Exception during") == 0) {
                                LOG_WARNING_SummarizationPlugin("Pair " + std::to_string((i/2) + 1) + 
                                    " still failed after context resize - skipping");
                                continue;
                            } else {
                                LOG_SummarizationPlugin("Pair " + std::to_string((i/2) + 1) + 
                                    " succeeded after context resize");
                            }
                        } else {
                            LOG_ERROR_SummarizationPlugin("Lost summary context after resize - skipping pair");
                            continue;
                        }
                    } else {
                        LOG_WARNING_SummarizationPlugin("Context resize failed for pair " + std::to_string((i/2) + 1) + 
                            " - skipping (tokens too large or resize failed)");
                        continue;
                    }
                }
                
                if (!pair_summary.empty()) {
                    pair_summaries.push_back(pair_summary);
                    LOG_SummarizationPlugin("Summarized pair " + std::to_string((i/2) + 1) + ": " + 
                        pair_summary.substr(0, 50) + (pair_summary.length() > 50 ? "..." : ""));
                }
            }
            
            // Handle odd message - leave it in original context
            if (total_messages % 2 == 1) {
                LOG_SummarizationPlugin("Leaving unpaired message in context: " + 
                    batch.pruned_messages[total_messages - 1].first);
            }
            
            // Create final summary from pairs
            std::string final_summary;
            if (pair_summaries.empty()) {
                LOG_ERROR_SummarizationPlugin("No pairs could be summarized - batch failed");
                return "";
            } else if (pair_summaries.size() == 1) {
                // Single pair - use directly
                final_summary = "Summary: " + pair_summaries[0];
            } else {
                // Multiple pairs - combine them
                std::ostringstream final_stream;
                for (size_t i = 0; i < pair_summaries.size(); ++i) {
                    final_stream << "* " << pair_summaries[i] << "\n";
                }
                final_summary = final_stream.str();
            }
            
            LOG_SummarizationPlugin("Successfully processed batch: " + 
                std::to_string(pair_summaries.size()) + " pairs summarized");
            
            return final_summary;
            
        } catch (const std::exception& e) {
            LOG_ERROR_SummarizationPlugin("Exception processing batch: " + std::string(e.what()));
            return "";
        }
    }

private:
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
    
    // Track state of inner contexts to avoid duplicate processing
    static std::unordered_map<std::string, size_t> last_processed_message_count;
    
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
        
        // Get current message history
        const auto& message_history = context->GetMessageHistory();
        size_t current_message_count = message_history.size();
        
        // Check if this context has new messages since last check
        auto last_count_it = last_processed_message_count.find(context_id);
        size_t last_processed_count = (last_count_it != last_processed_message_count.end()) ? last_count_it->second : 0;
        
        if (current_message_count <= last_processed_count) {
            continue; // No new messages to process
        }
        
        LOG_DEBUG_Orchestrator("ProcessInnerVoiceAnalysis: Found " + std::to_string(current_message_count - last_processed_count) + 
                              " new messages in context: " + context_id);
        
        // Look for complete user input/AI output pairs in the new messages
        std::vector<std::pair<std::string, std::string>> new_pairs;
        
        // Process messages from where we left off, looking for user/assistant pairs
        for (size_t i = last_processed_count; i + 1 < current_message_count; i += 2) {
            const auto& user_msg = message_history[i];
            const auto& ai_msg = message_history[i + 1];
            
            // Check if we have a user -> assistant pair
            if (user_msg.first == "user" && ai_msg.first == "assistant") {
                new_pairs.emplace_back(user_msg.second, ai_msg.second);
                LOG_DEBUG_Orchestrator("ProcessInnerVoiceAnalysis: Found user/AI pair in " + context_id + 
                                      " - User: " + user_msg.second.substr(0, 50) + "..." +
                                      " - AI: " + ai_msg.second.substr(0, 50) + "...");
            }
        }
        
        // If we found complete pairs, request summarization
        if (!new_pairs.empty()) {
            LOG_Orchestrator("ProcessInnerVoiceAnalysis: Requesting summarization for " + 
                           std::to_string(new_pairs.size()) + " inner voice pairs from context: " + context_id);
            
            // Convert pairs to the format expected by SummarizationPlugin
            std::vector<std::pair<std::string, std::string>> formatted_pairs;
            for (const auto& [user_input, ai_output] : new_pairs) {
                // Format as "User: <input>" and "Assistant: <output>" for better summarization context
                formatted_pairs.emplace_back("User", user_input);
                formatted_pairs.emplace_back("Assistant", ai_output);
            }
            
            // Create a proper PrunedMessageBatch for inner voice summarization
            PrunedMessageBatch inner_voice_batch(context_id + "_inner_voice_summary", formatted_pairs);
            
            // Request summarization directly through the Orchestrator pipeline (not the plugin's buffer)
            RequestSummarization(inner_voice_batch);
            
            LOG_Orchestrator("ProcessInnerVoiceAnalysis: Queued " + std::to_string(formatted_pairs.size()) + 
                           " messages for summarization from inner voice context: " + context_id);
            
            // Remove the processed pairs from the message history to prevent unbounded growth
            // The pairs we just found start at last_processed_count, so remove from there
            LOG_Orchestrator("ProcessInnerVoiceAnalysis: BEFORE removal - context " + context_id + 
                           " has " + std::to_string(context->GetMessageHistory().size()) + " messages");
            
            context->RemoveMessagePairs(last_processed_count, new_pairs.size());
            
            LOG_Orchestrator("ProcessInnerVoiceAnalysis: AFTER removal - context " + context_id + 
                           " has " + std::to_string(context->GetMessageHistory().size()) + " messages");
            LOG_Orchestrator("ProcessInnerVoiceAnalysis: Removed " + std::to_string(new_pairs.size()) + 
                           " pairs from message history starting at index " + std::to_string(last_processed_count) + 
                           " in context: " + context_id);
        }
        
        // Update our tracking of processed messages
        // Since we removed the pairs from the history, the message count has decreased
        // We need to adjust our tracking to account for the removed messages
        if (!new_pairs.empty()) {
            // After removing pairs, we don't advance the processed count because we removed the messages
            // The next processing should start from the same position (last_processed_count) since
            // we removed the messages that were there
            last_processed_message_count[context_id] = last_processed_count;
            LOG_DEBUG_Orchestrator("ProcessInnerVoiceAnalysis: Maintained last processed count at " + 
                                  std::to_string(last_processed_count) + " after removing pairs for " + context_id);
        } else {
            // No pairs found, update to current message count for next check
            last_processed_message_count[context_id] = current_message_count;
            LOG_DEBUG_Orchestrator("ProcessInnerVoiceAnalysis: Advanced last processed count to " + 
                                  std::to_string(current_message_count) + " (no pairs found) for " + context_id);
        }
        
        LOG_DEBUG_Orchestrator("ProcessInnerVoiceAnalysis: Updated last processed count for " + context_id + 
                              " to " + std::to_string(last_processed_message_count[context_id]));
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