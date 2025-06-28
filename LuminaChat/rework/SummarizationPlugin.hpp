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
        
        // Get main model context size as hard limit
        int main_context_size = settings_manager->GetInt("Models", "main_context_size", 4096);
        if (new_context_size > static_cast<size_t>(main_context_size)) {
            LOG_WARNING_SummarizationPlugin("Cannot resize summary context to " + std::to_string(new_context_size) + 
                " tokens - would exceed main context limit of " + std::to_string(main_context_size));
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
        
        // Get main model settings to derive summary model config
        int main_context_size = settings_manager->GetInt("Models", "main_context_size", 4096);
        int main_gpu_layers = settings_manager->GetInt("Models", "main_gpu_layers", 999);
        
        // Context size calculation for pairwise summarization
        // Need enough space for: system prompt (~200 tokens) + 2 messages (up to 2000 tokens each) 
        // + formatting overhead (~100 tokens) + response generation (~200 tokens)
        // Total: ~4500 tokens minimum for large message pairs
        // Use 50% of main context size with minimum 4500 tokens to handle large conversations
        int summary_context_size = std::max(4500, (main_context_size * 50) / 100);
        
        LOG_SummarizationPlugin("Loading summary model: " + summary_model_path + 
                               " (context: " + std::to_string(summary_context_size) + 
                               " [50% of main, optimized for large message pairs], gpu_layers: " + std::to_string(main_gpu_layers) + ")");
        
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
     * Process a single batch of pruned messages using pairwise summarization
     * Always summarizes 2 messages at a time (Q->A pairs) for maximum fidelity
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
            
            size_t total_messages = batch.pruned_messages.size();
            
            // Always use pairwise summarization - need at least 2 messages
            if (total_messages < 2) {
                LOG_SummarizationPlugin("Only " + std::to_string(total_messages) + 
                    " message(s) in batch - leaving in context without summarization");
                return;
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
                
                std::string pair_summary = summary_ctx->HandleInput(pair_prompt, "user");
                
                // Check if we got an error (likely context overflow)
                if (pair_summary.empty() || pair_summary.find("Error: Failed to process prompt") == 0) {
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
                            pair_summary = summary_ctx->HandleInput(pair_prompt, "user");
                            
                            if (pair_summary.empty() || pair_summary.find("Error: Failed to process prompt") == 0) {
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
                return;
            } else if (pair_summaries.size() == 1) {
                // Single pair - use directly
                final_summary = "Summary: " + pair_summaries[0];
            } else {
                // Multiple pairs - combine them
                std::ostringstream final_stream;
                final_stream << "Conversation summary:\n";
                for (size_t i = 0; i < pair_summaries.size(); ++i) {
                    final_stream << "• " << pair_summaries[i] << "\n";
                }
                final_summary = final_stream.str();
            }
            
            // Apply summary to original context
            ApplySummaryToContext(batch.context_id, final_summary);
            
            messages_summarized += batch.pruned_messages.size();
            summaries_applied++;
            
            LOG_SummarizationPlugin("Successfully processed batch: " + 
                std::to_string(pair_summaries.size()) + " pairs summarized");
            
        } catch (const std::exception& e) {
            LOG_ERROR_SummarizationPlugin("Exception processing batch: " + std::string(e.what()));
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
