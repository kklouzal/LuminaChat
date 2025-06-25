#pragma once

#include "ContextInfo.hpp"
#include "ProcessingPipeline.hpp"
#include "Orchestrator.hpp"
#include "LogHandler.hpp"
#include <chrono>
#include <thread>

namespace LuminaChat {

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
    
    // Reference to orchestrator for summarization processing
    Orchestrator* orchestrator = nullptr;
    
    // Statistics
    std::atomic<size_t> batches_processed{0};
    std::atomic<size_t> messages_summarized{0};
    std::atomic<size_t> summaries_applied{0};

public:
    explicit SummarizationPlugin(Orchestrator* orch) 
        : orchestrator(orch) {
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
        
        should_stop = false;
        processing_thread = std::make_unique<std::thread>(&SummarizationPlugin::ProcessingLoop, this);
        LOG_SummarizationPlugin("SummarizationPlugin started");
    }
    
    /**
     * Stop the plugin and wait for thread completion
     */
    void Stop() {
        should_stop = true;
        if (processing_thread && processing_thread->joinable()) {
            processing_thread->join();
        }
        LOG_SummarizationPlugin("SummarizationPlugin stopped");
    }
    
    /**
     * Set the check interval for processing pruned messages
     */
    void SetCheckInterval(std::chrono::milliseconds interval) {
        check_interval = interval;
    }
    
    /**
     * Get plugin statistics
     */
    struct PluginStats {
        size_t batches_processed;
        size_t messages_summarized;
        size_t summaries_applied;
        bool is_running;
    };
    
    PluginStats GetStats() const {
        return {
            batches_processed.load(),
            messages_summarized.load(),
            summaries_applied.load(),
            !should_stop.load()
        };
    }
    
private:
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
        
        // Convert message history to content string
        std::ostringstream content_stream;
        for (const auto& [role, content] : batch.pruned_messages) {
            content_stream << role << ": " << content << "\n";
        }
        
        if (orchestrator) {
            // Use orchestrator to handle the summarization request
            orchestrator->RequestSummarization(batch.context_id, content_stream.str());
            messages_summarized += batch.pruned_messages.size();
        } else {
            LOG_ERROR_SummarizationPlugin("No orchestrator available for summarization");
        }
    }
};

} // namespace LuminaChat
