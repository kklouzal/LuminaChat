#pragma once

#include "ContextInfo.hpp"
#include "ProcessingPipeline.hpp"
#include "Orchestrator.hpp"
#include "LlamaManager.hpp"
#include "SettingsManager.hpp"
#include "Logger.hpp"
#include <chrono>
#include <thread>
#include <memory>

namespace LuminaChat {

// Callback type for status updates
using StatusUpdateCallback = std::function<void(const std::string& status, bool is_error)>;

/**
 * Summarization Plugin - Handles background processing of pruned messages
 * 
 * This plugin periodically checks the pruning buffer for messages that need
 * summarization and processes them through the summarization pipeline.
 * 
 * Architecture:
 * - Core handles immediate context pruning (critical path)
 * - Plugin handles heavy summarization work (async/background)
 * - Summaries are applied back to contexts when complete
 */
class SummarizationPlugin {
private:
    std::unique_ptr<std::thread> processing_thread;
    std::atomic<bool> should_stop{false};
    std::chrono::milliseconds check_interval{1000}; // Check every second
    
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

public:
    explicit SummarizationPlugin(Orchestrator* orch) 
        : orchestrator(orch) {
        
        if (orchestrator) {
            llama_manager = orchestrator->GetLlamaManager();
            settings_manager = orchestrator->GetSettingsManager();
        }
        
        LOG_SummarizationPlugin("SummarizationPlugin initialized");
    }
    
    ~SummarizationPlugin() {
        Stop();
    }
    
    /**
     * Start the plugin's background processing thread
     */
    void Start() {
        if (processing_thread && processing_thread->joinable()) {
            LOG_WARNING_SummarizationPlugin("Plugin already running");
            return;
        }
        
        // Initialize summary model and context first
        if (!InitializeSummaryModel()) {
            LOG_ERROR_SummarizationPlugin("Failed to initialize summary model - plugin will not process summarizations");
            return;
        }
        
        should_stop = false;
        processing_thread = std::make_unique<std::thread>(&SummarizationPlugin::ProcessingLoop, this);
        LOG_SummarizationPlugin("SummarizationPlugin started with summary model ready");
    }
    
    /**
     * Stop the plugin and wait for thread completion
     */
    void Stop() {
        should_stop = true;
        if (processing_thread && processing_thread->joinable()) {
            processing_thread->join();
        }
        
        // Clean up summary context and model
        summary_context.reset();
        summary_model_ready = false;
        
        LOG_SummarizationPlugin("SummarizationPlugin stopped");
    }
    
    /**
     * Set the check interval for processing pruned messages
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
     * Get plugin statistics
     */
    struct PluginStats {
        size_t batches_processed;
        size_t messages_summarized;
        size_t summaries_applied;
        bool is_running;
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
            !should_stop.load(),
            summary_model_ready.load(),
            model_path
        };
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
        
        // Get main model settings to derive summary model config
        int main_context_size = settings_manager->GetInt("Models", "main_context_size", 4096);
        int main_gpu_layers = settings_manager->GetInt("Models", "main_gpu_layers", 999);
        
        // CRITICAL FIX: Proper context size calculation that accounts for all overheads
        // Calculate summary context size with proper overhead accounting:
        // - Safety buffers (SAFETY_BUFFER_TOKENS = 64, GENERATION_BUFFER_TOKENS = 128)
        // - System prompt overhead (~200 tokens)
        // - Template overhead (~100 tokens) 
        // - Response generation space (~300 tokens minimum)
        // Total overhead: ~600 tokens
        //
        // Use 40% of main context size (instead of 20%) to ensure adequate space
        // after all overheads are applied, with minimum 1200 to handle typical summarization workloads
        int raw_summary_size = std::max(1200, (main_context_size * 40) / 100); // 40% with minimum 1200
        int summary_context_size = std::min(raw_summary_size, main_context_size - 200); // Cap to ensure it's smaller than main
        
        LOG_SummarizationPlugin("Loading summary model: " + summary_model_path + 
                               " (context: " + std::to_string(summary_context_size) + 
                               " [40% of main with overhead accounting], gpu_layers: " + std::to_string(main_gpu_layers) + ")");
        
        if (status_callback) status_callback("Loading summary model: " + summary_model_path, false);
        
        try {
            // Load summary model
            ModelConfig config;
            config.model_path = summary_model_path;
            config.context_size = summary_context_size;
            config.gpu_layers = main_gpu_layers;
            
            if (!llama_manager->LoadModel(summary_model_id, config)) {
                LOG_ERROR_SummarizationPlugin("Failed to load summary model: " + summary_model_path);
                if (status_callback) status_callback("Failed to load summary model", true);
                return false;
            }
            
            LOG_SummarizationPlugin("Summary model loaded successfully");
            if (status_callback) status_callback("Creating summary context...", false);
            
            // Create summary context
            auto* context_ptr = llama_manager->GetOrCreateContextInfo(summary_context_id, summary_model_id, "summary");
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
     * Main processing loop - runs in background thread
     */
    void ProcessingLoop() {
        LOG_SummarizationPlugin("Processing loop started");
        
        while (!should_stop.load()) {
            try {
                ProcessPendingSummarizations();
                std::this_thread::sleep_for(check_interval);
            } catch (const std::exception& e) {
                LOG_ERROR_SummarizationPlugin("Exception in processing loop: " + std::string(e.what()));
                std::this_thread::sleep_for(check_interval * 5); // Back off on error
            }
        }
        
        LOG_SummarizationPlugin("Processing loop ended");
    }
    
    /**
     * Check for and process any pending summarization requests
     */
    void ProcessPendingSummarizations() {
        // Skip processing if summary model is not ready
        if (!summary_model_ready.load()) {
            return;
        }
        
        // Check if there are any pruned messages waiting for summarization
        if (!ContextInfo::HasPendingSummarization()) {
            return; // No work to do
        }
        
        LOG_SummarizationPlugin("Found pending summarizations to process");
        
        // Get all pending pruning batches
        auto pruning_batches = ContextInfo::GetAndClearPruningBuffer();
        
        for (const auto& batch : pruning_batches) {
            if (!batch.needs_summarization) {
                continue; // Skip batches that don't need summarization
            }
            
            ProcessSummarizationBatch(batch);
        }
        
        batches_processed += pruning_batches.size();
    }
    
    /**
     * Process a single batch of pruned messages
     */
    void ProcessSummarizationBatch(const PrunedMessageBatch& batch) {
        LOG_SummarizationPlugin("Processing batch for context: " + batch.context_id + 
                               " (" + std::to_string(batch.pruned_messages.size()) + " messages)");
        
        if (!summary_model_ready.load() || !llama_manager) {
            LOG_ERROR_SummarizationPlugin("Summary model not ready for processing batch");
            return;
        }
        
        try {
            // Get the summary context
            auto* summary_ctx = llama_manager->GetContextInfo(summary_context_id);
            if (!summary_ctx) {
                LOG_ERROR_SummarizationPlugin("Summary context not available");
                return;
            }
            
            // Convert message history to a summarization prompt
            std::ostringstream prompt_stream;
            prompt_stream << "Please summarize the following conversation history:\n\n";
            
            for (const auto& [role, content] : batch.pruned_messages) {
                prompt_stream << role << ": " << content << "\n";
            }
            
            prompt_stream << "\nProvide a concise summary that captures the key points and context:";
            
            std::string summarization_prompt = prompt_stream.str();
            
            LOG_SummarizationPlugin("Generating summary for " + std::to_string(batch.pruned_messages.size()) + " messages");
            
            // CRITICAL FIX: Validate content size before processing to prevent context overflow
            // Rough estimation: ~4 characters per token
            size_t estimated_tokens = summarization_prompt.length() / 4;
            auto* model_info = llama_manager->GetModelInfo(summary_model_id);
            if (!model_info) {
                LOG_ERROR_SummarizationPlugin("Summary model info not available");
                return;
            }
            
            // Get actual available space (accounting for safety buffers and overheads)
            int32_t available_space = model_info->n_ctx - 600; // Reserve 600 tokens for overheads and response
            
            if (static_cast<int32_t>(estimated_tokens) > available_space) {
                LOG_ERROR_SummarizationPlugin("Content size (" + std::to_string(estimated_tokens) + 
                    " tokens) exceeds available context space (" + std::to_string(available_space) + 
                    ") - implementing chunked processing");
                
                // CHUNKED PROCESSING: Break large batches into smaller pieces
                std::vector<std::string> chunk_summaries;
                size_t messages_per_chunk = std::max(size_t(1), batch.pruned_messages.size() / 3); // Split into ~3 chunks
                
                for (size_t i = 0; i < batch.pruned_messages.size(); i += messages_per_chunk) {
                    size_t end_idx = std::min(i + messages_per_chunk, batch.pruned_messages.size());
                    
                    std::ostringstream chunk_stream;
                    chunk_stream << "Please summarize this conversation chunk:\n\n";
                    for (size_t j = i; j < end_idx; ++j) {
                        chunk_stream << batch.pruned_messages[j].first << ": " << batch.pruned_messages[j].second << "\n";
                    }
                    chunk_stream << "\nProvide a concise summary:";
                    
                    std::string chunk_prompt = chunk_stream.str();
                    size_t chunk_tokens = chunk_prompt.length() / 4;
                    
                    if (static_cast<int32_t>(chunk_tokens) <= available_space) {
                        // CRITICAL FIX: Reset context state before each chunk to prevent accumulation
                        summary_ctx->ClearContext();
                        
                        std::string chunk_summary = summary_ctx->HandleInput(chunk_prompt, "user");
                        if (!chunk_summary.empty()) {
                            chunk_summaries.push_back(chunk_summary);
                            LOG_SummarizationPlugin("Processed chunk " + std::to_string(chunk_summaries.size()) + 
                                " (" + std::to_string(end_idx - i) + " messages)");
                        }
                    } else {
                        LOG_ERROR_SummarizationPlugin("Even chunked content too large - skipping this chunk");
                    }
                }
                
                // Combine chunk summaries into final summary
                if (!chunk_summaries.empty()) {
                    std::ostringstream final_stream;
                    final_stream << "Please create a consolidated summary from these partial summaries:\n\n";
                    for (size_t i = 0; i < chunk_summaries.size(); ++i) {
                        final_stream << "Summary " << (i + 1) << ": " << chunk_summaries[i] << "\n\n";
                    }
                    final_stream << "Provide a single, coherent summary:";
                    
                    std::string final_prompt = final_stream.str();
                    size_t final_tokens = final_prompt.length() / 4;
                    
                    if (static_cast<int32_t>(final_tokens) <= available_space) {
                        summary_ctx->ClearContext();
                        summarization_prompt = final_prompt; // Use the consolidated prompt
                        LOG_SummarizationPlugin("Using chunked processing - combining " + 
                            std::to_string(chunk_summaries.size()) + " chunk summaries");
                    } else {
                        // Fallback: use the first chunk summary
                        LOG_WARNING_SummarizationPlugin("Final consolidation too large - using first chunk summary");
                        std::string summary = chunk_summaries[0];
                        ApplySummaryToContext(batch.context_id, summary);
                        messages_summarized += batch.pruned_messages.size();
                        summaries_applied++;
                        return;
                    }
                } else {
                    LOG_ERROR_SummarizationPlugin("All chunks failed processing - cannot summarize this batch");
                    return;
                }
            }
            
            // CRITICAL FIX: Validate context state before processing to prevent overflow
            if (!summary_ctx->ValidateAndSyncContextState()) {
                LOG_ERROR_SummarizationPlugin("Summary context state validation failed - resetting context");
                summary_ctx->ClearContext();
            }
            
            // Generate summary using the summary context (synchronous for now)
            // Use "user" role to ensure proper template formatting
            std::string summary = summary_ctx->HandleInput(summarization_prompt, "user");
            
            // CRITICAL FIX: Always reset summary context after use to prevent accumulation
            // Summary contexts should be stateless - each summarization is independent
            summary_ctx->ClearContext();
            
            if (!summary.empty()) {
                LOG_SummarizationPlugin("Generated summary: " + summary.substr(0, 100) + 
                                       (summary.length() > 100 ? "..." : ""));
                
                // Apply the summary back to the original context
                ApplySummaryToContext(batch.context_id, summary);
                
                messages_summarized += batch.pruned_messages.size();
                summaries_applied++;
            } else {
                LOG_ERROR_SummarizationPlugin("Failed to generate summary for batch");
            }
            
        } catch (const std::exception& e) {
            LOG_ERROR_SummarizationPlugin("Exception processing summarization batch: " + std::string(e.what()));
        }
    }
    
    /**
     * Apply a generated summary back to the original context
     */
    void ApplySummaryToContext(const std::string& context_id, const std::string& summary) {
        if (!llama_manager) {
            LOG_ERROR_SummarizationPlugin("LlamaManager not available for applying summary");
            return;
        }
        
        try {
            // Get the original context that requested summarization
            auto* original_context = llama_manager->GetContextInfo(context_id);
            if (!original_context) {
                LOG_WARNING_SummarizationPlugin("Original context not found for summary application: " + context_id);
                return;
            }
            
            // Apply the completed summary to the context
            original_context->ApplyCompletedSummary(summary);
            
            LOG_SummarizationPlugin("Applied summary to context: " + context_id);
            
        } catch (const std::exception& e) {
            LOG_ERROR_SummarizationPlugin("Exception applying summary to context: " + std::string(e.what()));
        }
    }
};

} // namespace LuminaChat
