#pragma once

#include "ChatTemplateManager.hpp"
#include "ModelInfo.hpp"
#include "TokenCache.hpp"
#include "Logger.hpp"
// llama.cpp includes
#include "llama-cpp.h"
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <mutex>
#include <chrono>
#include <sstream>
#include <algorithm>
#include <iomanip>

// ContextInfo: Individual conversation context management
// 
// Key Features:
// - Dynamic Template Management: Owns a ChatTemplateManager instance
// - Performance statistics tracking
// - Reference to parent ModelInfo and its TokenCache
// - Message history storage: vector<pair<string, string>> (role, content) - pure conversation only
// - Batch management for generation
// - Smart context rebuilding
// - Plugin Integration: Template-based summarization and context management

// Forward declarations
struct llama_context;

// Helper function declarations
std::string GenerateContextId();

// Forward declaration for ModelInfo
class ModelInfo;

enum class RebuildStrategy {
    FULL,           // Complete rebuild including template re-rendering
    PARTIAL,        // Efficient append-only rebuild for new messages
    TEMPLATE_ONLY   // Only re-render template, keep existing tokens
};

enum class ContextState {
    READY,
    PROCESSING,
    AWAITING_SUMMARIZATION,
    ERROR_STATE
};

struct ContextStats {
    size_t total_tokens_processed = 0;
    size_t current_context_tokens = 0;
    size_t max_context_tokens = 0;
    size_t message_pairs = 0;
    size_t template_renders = 0;
    size_t full_rebuilds = 0;
    size_t partial_rebuilds = 0;
    std::chrono::steady_clock::time_point created_time;
    std::chrono::steady_clock::time_point last_activity;
    
    ContextStats() : created_time(std::chrono::steady_clock::now()), 
                    last_activity(std::chrono::steady_clock::now()) {}
                    
    float GetContextUsageRatio() const {
        return max_context_tokens > 0 ? 
            static_cast<float>(current_context_tokens) / max_context_tokens : 0.0f;
    }
    
    std::chrono::seconds GetAge() const {
        return std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - created_time);
    }
    
    std::chrono::seconds GetTimeSinceActivity() const {
        return std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - last_activity);
    }
};

/**
 * Individual conversation context management with dynamic template system
 * 
 * Revolutionary Processing Flow:
 * 1. Receive input from Orchestrator (pre-sanitized if from Discord)
 * 2. Add pure conversation pair to message history (no system contamination)
 * 3. Render dynamic template with current message history via ChatTemplateManager
 * 4. Process rendered template to tokens via TokenCache
 * 5. Check context size - trigger summarization plugin if needed (updates template sections)
 * 6. Generate LLM response using rendered template
 * 7. Detokenize response via TokenCache
 * 8. Add response pair to message history
 * 9. Return human-readable response
 * 
 * Template Integration Benefits:
 * - Clean Message History: Only actual conversation, no system pollution
 * - Dynamic Context: Summaries, system messages embedded in template sections
 * - Plugin-Driven Updates: Summarization updates template directly, not message history
 * - Per-Context Flexibility: Each context can have completely different template evolution
 */
class ContextInfo {
private:    // Core components
    std::unique_ptr<ChatTemplateManager> template_manager;
    ModelInfo* parent_model;
    TokenCache* token_cache;
    
    // Context state
    std::string context_id;
    ContextState state;
    ContextStats stats;
    mutable std::mutex context_mutex;
    
    // Message history - pure conversation only
    std::vector<std::pair<std::string, std::string>> message_history; // (role, content)
    
    // Llama context management
    llama_context* llama_ctx = nullptr;
    llama_batch batch;
    bool batch_initialized = false;
    std::vector<int32_t> current_tokens;
    int32_t n_past = 0;  // Number of tokens processed in context
    bool context_needs_rebuild = true;
    
    // Generation state
    std::atomic<bool> is_generating{false};
    std::atomic<bool> should_stop_generation{false};
    
    // Callback for summarization requests
    std::function<void(std::string, std::string)> summarization_callback;
    
    // Helper methods (moved to public for testing)
    
    // Context size management
    bool IsNearContextLimit(float threshold = 0.8f) const;
    void RequestSummarization(const std::string& content);
    void UpdateStats();
    
public:    // Constructor overloads
    ContextInfo(const std::string& context_id, ModelInfo* model, const std::string& base_template);
    ContextInfo(ModelInfo* model, const std::string& base_template); // For testing with auto-generated context_id
    
    // Destructor
    ~ContextInfo();
    
    // Core processing interface
    std::string HandleInput(const std::string& input, const std::string& username = "user");
    void AddHistoricalMessage(const std::string& role, const std::string& content);
    
    // Llama.cpp integration methods
    bool InitializeLlamaContext();
    bool ProcessTokensBatch(const std::vector<int32_t>& tokens);
    std::string GenerateResponse(const std::string& prompt);
    
    // Template section management - direct access to ChatTemplateManager
    void UpdateEnvironment(const std::string& env);
    void UpdateIdentity(const std::string& identity);
    void UpdateSystemPrompt(const std::string& system_msg);
    void ApplySummary(const std::string& summary);  // Updates template's summary section
    void UpdateOldChatSummary(const std::string& old_summary); // For context pruning
    void UpdateMotifContext(const std::string& motif);
    void UpdateInternalReflection(const std::string& reflection);
    void AddPastSessionMemory(const std::string& memory);
    
    // Context management
    bool RebuildContext(RebuildStrategy strategy = RebuildStrategy::FULL);
    void ClearContext();
    void ClearMessageHistory();
    
    // State and statistics
    ContextState GetState() const { return state; }
    const ContextStats& GetStats() const { return stats; }
    const std::string& GetContextId() const { return context_id; }
    
    // Message history access
    const std::vector<std::pair<std::string, std::string>>& GetMessageHistory() const { return message_history; }
    size_t GetMessageCount() const { return message_history.size(); }
    std::string GetConversationHistory() const;  // String representation of conversation
    
    // Template access
    ChatTemplateManager& GetTemplateManager() { return *template_manager; }
    const ChatTemplateManager& GetTemplateManager() const { return *template_manager; }
    
    // Callback registration for plugin integration
    void RegisterSummarizationCallback(std::function<void(std::string, std::string)> callback);
    
    // Context validation
    bool ValidateContext() const;
    
    // Advanced features
    void SetMaxContextTokens(size_t max_tokens);
    std::string GetCurrentPrompt() const;
    
    // Debug and testing helpers
    void DumpContextInfo() const;
    std::string GetContextSummary() const;
    
    // Template rendering and tokenization (made public for testing)
    std::string BuildFullPrompt();
    std::vector<int32_t> TokenizePrompt(const std::string& prompt);
    std::string DetokenizeResponse(const std::vector<int32_t>& tokens);
    
    // Batch management methods - context-specific
    void clear_batch();
    
    // Context rebuilding helpers (made public for testing)
    void RebuildContext_Full();
    void RebuildContext_Partial();
    void RebuildContext_TemplateOnly();
    
    // Additional method implementations for testing compatibility
    void AddMessage(const std::string& role, const std::string& content);
    std::vector<std::pair<std::string, std::string>> GetMessages() const;
    int32_t GetCurrentTokenCount() const;
    void UpdateMotif(const std::string& motif);
};

// Helper functions for context management
namespace ContextUtils {
    // Estimate token count for text without full tokenization
    inline size_t EstimateTokenCount(const std::string& text) {
        // Rough estimation: ~4 characters per token on average
        // This is a simplification - actual tokenization varies by model
        return text.length() / 4;
    }
    
    // Calculate optimal batch size for context rebuilding
    inline size_t CalculateOptimalBatchSize(size_t context_size, size_t available_tokens) {
        // Calculate batch size as a percentage of available space
        size_t batch_size = available_tokens / 10; // 10% of available space
        return std::max(batch_size, static_cast<size_t>(512)); // Minimum 512 tokens
    }
    
    // Validate message history integrity
    inline bool ValidateMessageHistory(const std::vector<std::pair<std::string, std::string>>& history) {
        // Check for basic consistency
        for (const auto& msg : history) {
            if (msg.first.empty() || msg.second.empty()) {
                return false;
            }
            
            // Valid roles should be user, assistant, system, etc.
            if (msg.first != "user" && msg.first != "assistant" && msg.first != "system") {
                // Allow other roles but warn they might not be optimal
            }
        }
        
        return true;
    }
    
    // Extract conversation summary for debugging
    inline std::string SummarizeConversation(const std::vector<std::pair<std::string, std::string>>& history, 
                                           size_t max_lines = 10) {
        std::ostringstream oss;
        
        if (history.empty()) {
            return "No conversation history";
        }
        
        oss << "Conversation summary (" << history.size() << " messages):" << std::endl;
        
        size_t start_idx = history.size() > max_lines ? history.size() - max_lines : 0;
        
        for (size_t i = start_idx; i < history.size(); ++i) {
            const auto& msg = history[i];
            std::string preview = msg.second.substr(0, 60);
            if (msg.second.length() > 60) {
                preview += "...";
            }
            
            oss << "[" << i << "] " << msg.first << ": " << preview << std::endl;
        }
        
        if (start_idx > 0) {
            oss << "... (" << start_idx << " earlier messages)" << std::endl;
        }
        
        return oss.str();
    }
}

// Inline implementation of ContextInfo methods
inline ContextInfo::ContextInfo(const std::string& context_id, ModelInfo* model, const std::string& base_template)
    : context_id(context_id)
    , parent_model(model)
    , token_cache(model ? &model->GetTokenCache() : nullptr)
    , state(ContextState::READY)
    , template_manager(std::make_unique<ChatTemplateManager>(base_template))
    , llama_ctx(nullptr)
    , context_needs_rebuild(true)
{
    if (!parent_model) {
        LOG_ERROR_ContextInfo("ContextInfo created with null ModelInfo");
        state = ContextState::ERROR_STATE;
        return;
    }
    
    if (!token_cache) {
        LOG_ERROR_ContextInfo("ContextInfo created with null TokenCache");
        state = ContextState::ERROR_STATE;
        return;
    }
    
    // Initialize llama context from parent model
    // Note: This would need actual llama.cpp integration
    // For testing, we'll skip the actual llama context creation
    
    if (!template_manager->ValidateTemplate()) {
        LOG_ERROR_ContextInfo("Invalid base template provided");
        state = ContextState::ERROR_STATE;
        return;
    }
    
    LOG_DEBUG_ContextInfo("ContextInfo created: " + context_id);
    UpdateStats();
}

inline ContextInfo::ContextInfo(ModelInfo* model, const std::string& base_template)
    : ContextInfo(GenerateContextId(), model, base_template) // Auto-generate context_id
{
    LOG_DEBUG_ContextInfo("ContextInfo created with auto-generated ID: " + context_id);
}

inline ContextInfo::~ContextInfo() {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    LOG_DEBUG_ContextInfo("ContextInfo destructor called for: " + context_id);
    
    // Stop any ongoing generation
    should_stop_generation = true;
    
    // Clean up batch
    if (batch_initialized) {
        llama_batch_free(batch);
        batch_initialized = false;
        LOG_DEBUG_ContextInfo("Batch freed for context: " + context_id);
    }
    
    // Clean up llama context
    if (llama_ctx) {
        llama_free(llama_ctx);
        llama_ctx = nullptr;
        LOG_DEBUG_ContextInfo("Llama context freed for: " + context_id);
    }
    
    LOG_DEBUG_ContextInfo("ContextInfo destroyed: " + context_id);
}

inline bool ContextInfo::InitializeLlamaContext() {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    if (!parent_model || !parent_model->IsLoaded()) {
        LOG_ERROR_ContextInfo("Parent model not loaded for context initialization");
        return false;
    }
    
    if (llama_ctx) {
        LOG_DEBUG_ContextInfo("Llama context already initialized");
        return true;
    }
    
    try {
        // Set up context parameters using model's settings
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = parent_model->GetConfig().context_size;
        ctx_params.n_batch = std::min(512, parent_model->GetConfig().context_size / 8);
        ctx_params.n_threads = parent_model->GetConfig().threads > 0 ? 
                               parent_model->GetConfig().threads : 
                               std::max(1u, std::thread::hardware_concurrency());
        ctx_params.n_threads_batch = ctx_params.n_threads;
        
        // Performance optimizations
        ctx_params.flash_attn = true;
        ctx_params.offload_kqv = true;
        
        LOG_DEBUG_ContextInfo("Creating llama context with params: n_ctx=" + 
                             std::to_string(ctx_params.n_ctx) + 
                             ", n_batch=" + std::to_string(ctx_params.n_batch) + 
                             ", n_threads=" + std::to_string(ctx_params.n_threads));
        
        // Create context
        llama_ctx = llama_init_from_model(parent_model->GetModel(), ctx_params);
        if (!llama_ctx) {
            LOG_ERROR_ContextInfo("Failed to create llama context");
            return false;
        }
        
        // Initialize batch
        batch = llama_batch_init(ctx_params.n_batch, 0, 1);
        if (!batch.token) {
            LOG_ERROR_ContextInfo("Failed to initialize batch");
            llama_free(llama_ctx);
            llama_ctx = nullptr;
            return false;
        }
        
        batch_initialized = true;
        stats.max_context_tokens = ctx_params.n_ctx;
        
        LOG_ContextInfo("Llama context initialized successfully for: " + context_id);
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR_ContextInfo("Exception during llama context initialization: " + std::string(e.what()));
        if (llama_ctx) {
            llama_free(llama_ctx);
            llama_ctx = nullptr;
        }
        return false;
    }
}

inline std::string ContextInfo::HandleInput(const std::string& input, const std::string& username) {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    if (state == ContextState::ERROR_STATE) {
        LOG_ERROR_ContextInfo("Cannot handle input - context in error state");
        return "Error: Context unavailable";
    }
    
    if (state == ContextState::AWAITING_SUMMARIZATION) {
        LOG_ContextInfo("Input queued - context awaiting summarization");
        return "Processing... please wait";
    }
    
    state = ContextState::PROCESSING;
    UpdateStats();
    
    try {
        // 1. Add pure conversation pair to message history
        message_history.emplace_back(username, input);
        LOG_DEBUG_ContextInfo("Added user message to history: " + username + " -> " + 
                             input.substr(0, 50) + (input.length() > 50 ? "..." : ""));
        
        // 2. Check if context needs rebuilding after adding message
        if (context_needs_rebuild || template_manager->IsTemplateDirty()) {
            if (!RebuildContext(RebuildStrategy::FULL)) {
                state = ContextState::ERROR_STATE;
                return "Error: Failed to rebuild context";
            }
        }
        
        // 3. Render dynamic template with current message history
        std::string full_prompt = BuildFullPrompt();
        
        // 4. Check context size - trigger summarization if needed
        std::vector<int32_t> prompt_tokens = TokenizePrompt(full_prompt);
        if (IsNearContextLimit(0.8f)) {
            LOG_ContextInfo("Context approaching limit, requesting summarization");
            
            // Extract content for summarization (recent conversation)
            std::ostringstream content_stream;
            size_t start_idx = message_history.size() > 20 ? message_history.size() - 20 : 0;
            for (size_t i = start_idx; i < message_history.size(); ++i) {
                content_stream << message_history[i].first << ": " << message_history[i].second << "\n";
            }
            
            RequestSummarization(content_stream.str());
            state = ContextState::AWAITING_SUMMARIZATION;
            return "Context full - summarizing recent conversation...";
        }
        
        // 5. Generate LLM response (mock implementation for testing)
        std::string response = "Mock response generated from " + std::to_string(prompt_tokens.size()) + " prompt tokens.";
        
        // 6. Add assistant response to message history
        message_history.emplace_back("assistant", response);
        
        state = ContextState::READY;
        UpdateStats();
        
        LOG_DEBUG_ContextInfo("Generated response: " + response.substr(0, 100) + 
                             (response.length() > 100 ? "..." : ""));
        
        return response;
        
    } catch (const std::exception& e) {
        LOG_ERROR_ContextInfo("Exception in HandleInput: " + std::string(e.what()));
        state = ContextState::ERROR_STATE;
        return "Error: " + std::string(e.what());
    }
}

inline void ContextInfo::AddHistoricalMessage(const std::string& role, const std::string& content) {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    message_history.emplace_back(role, content);
    context_needs_rebuild = true;
    
    LOG_DEBUG_ContextInfo("Added historical message: " + role + " -> " + 
                         content.substr(0, 50) + (content.length() > 50 ? "..." : ""));
    UpdateStats();
}

// Template section management methods
inline void ContextInfo::UpdateEnvironment(const std::string& env) {
    template_manager->UpdateEnvironment(env);
    context_needs_rebuild = true;
    LOG_DEBUG_ContextInfo("Updated environment section");
}

inline void ContextInfo::UpdateIdentity(const std::string& identity) {
    template_manager->UpdateIdentity(identity);
    context_needs_rebuild = true;
    LOG_DEBUG_ContextInfo("Updated identity section");
}

inline void ContextInfo::UpdateSystemPrompt(const std::string& system_msg) {
    template_manager->UpdateSystemPrompt(system_msg);
    context_needs_rebuild = true;
    LOG_DEBUG_ContextInfo("Updated system prompt section");
}

inline void ContextInfo::ApplySummary(const std::string& summary) {
    template_manager->UpdateSummary(summary);
    context_needs_rebuild = true;
    LOG_ContextInfo("Applied summary to template: " + summary.substr(0, 100) + 
                   (summary.length() > 100 ? "..." : ""));
}

inline void ContextInfo::UpdateOldChatSummary(const std::string& old_summary) {
    template_manager->UpdateOldChatSummary(old_summary);
    context_needs_rebuild = true;
    LOG_DEBUG_ContextInfo("Updated old chat summary");
}

inline void ContextInfo::UpdateMotifContext(const std::string& motif) {
    template_manager->UpdateMotifContext(motif);
    context_needs_rebuild = true;
    LOG_DEBUG_ContextInfo("Updated motif context");
}

inline void ContextInfo::UpdateInternalReflection(const std::string& reflection) {
    template_manager->UpdateInternalReflection(reflection);
    context_needs_rebuild = true;
    LOG_DEBUG_ContextInfo("Updated internal reflection");
}

inline void ContextInfo::AddPastSessionMemory(const std::string& memory) {
    template_manager->AddPastSession(memory);
    context_needs_rebuild = true;
    LOG_DEBUG_ContextInfo("Added past session memory");
}

inline std::string ContextInfo::BuildFullPrompt() {
    if (!template_manager) {
        LOG_ERROR_ContextInfo("Template manager not available");
        return "";
    }
    
    // Render template with current message history
    std::string rendered_prompt = template_manager->RenderTemplate(message_history);
    stats.template_renders++;
    
    LOG_DEBUG_ContextInfo("Built full prompt: " + std::to_string(rendered_prompt.length()) + " characters");
    return rendered_prompt;
}

inline std::vector<int32_t> ContextInfo::TokenizePrompt(const std::string& prompt) {
    if (!parent_model || !parent_model->IsLoaded()) {
        LOG_ERROR_ContextInfo("Parent model not available for tokenization");
        return {};
    }
    
    // Use ModelInfo's tokenization
    std::vector<llama_token> llama_tokens = parent_model->TokenizeText(prompt, true);
    
    // Convert to int32_t
    std::vector<int32_t> tokens;
    tokens.reserve(llama_tokens.size());
    for (llama_token token : llama_tokens) {
        tokens.push_back(static_cast<int32_t>(token));
    }
    
    current_tokens = tokens;
    stats.current_context_tokens = tokens.size();
    
    LOG_DEBUG_ContextInfo("Tokenized prompt: " + std::to_string(tokens.size()) + " tokens");
    return tokens;
}

inline std::string ContextInfo::DetokenizeResponse(const std::vector<int32_t>& tokens) {
    if (!parent_model || !parent_model->IsLoaded()) {
        LOG_ERROR_ContextInfo("Parent model not available for detokenization");
        return "";
    }
    
    // Convert to llama_token
    std::vector<llama_token> llama_tokens;
    llama_tokens.reserve(tokens.size());
    for (int32_t token : tokens) {
        llama_tokens.push_back(static_cast<llama_token>(token));
    }
    
    // Use ModelInfo's detokenization
    std::string result = parent_model->DetokenizeTokens(llama_tokens);
    
    LOG_DEBUG_ContextInfo("Detokenized response: " + std::to_string(result.length()) + " characters");
    return result;
}

// Batch management methods - context-specific
inline void ContextInfo::clear_batch() {
    if (!batch_initialized) return;
    batch.n_tokens = 0;
}

inline bool ContextInfo::RebuildContext(RebuildStrategy strategy) {
    if (!InitializeLlamaContext()) {
        LOG_ERROR_ContextInfo("Failed to initialize llama context for rebuild");
        return false;
    }
    
    switch (strategy) {
        case RebuildStrategy::FULL:
            RebuildContext_Full();
            break;
        case RebuildStrategy::PARTIAL:
            RebuildContext_Partial();
            break;
        case RebuildStrategy::TEMPLATE_ONLY:
            RebuildContext_TemplateOnly();
            break;
    }
    
    context_needs_rebuild = false;
    return true;
}

inline void ContextInfo::RebuildContext_Full() {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    LOG_DEBUG_ContextInfo("Performing full context rebuild");
    
    // Clear current context
    if (llama_ctx) {
        llama_memory_clear(llama_get_memory(llama_ctx), true);
        n_past = 0;
    }
    
    // Build and tokenize full prompt
    std::string full_prompt = BuildFullPrompt();
    std::vector<int32_t> tokens = TokenizePrompt(full_prompt);
    
    if (!tokens.empty() && llama_ctx) {
        // Process tokens in batch
        ProcessTokensBatch(tokens);
    }
    
    stats.full_rebuilds++;
    LOG_DEBUG_ContextInfo("Full rebuild completed: " + std::to_string(n_past) + " tokens processed");
}

inline void ContextInfo::RebuildContext_Partial() {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    LOG_DEBUG_ContextInfo("Performing partial context rebuild");
    
    // This is a simplified partial rebuild - in practice, you'd implement
    // incremental token processing based on what changed
    // For now, fall back to full rebuild
    RebuildContext_Full(); // Fallback to full rebuild for now
    
    stats.partial_rebuilds++;
}

inline void ContextInfo::RebuildContext_TemplateOnly() {
    LOG_DEBUG_ContextInfo("Performing template-only rebuild");
    
    // Just re-render template without reprocessing tokens
    BuildFullPrompt();
    stats.template_renders++;
}

inline bool ContextInfo::ProcessTokensBatch(const std::vector<int32_t>& tokens) {
    if (!llama_ctx || !batch_initialized) {
        LOG_ERROR_ContextInfo("Llama context or batch not initialized");
        return false;
    }
    
    if (tokens.empty()) {
        return true;
    }
    
    try {
        // Process tokens in batches
        const int32_t n_batch = llama_n_batch(llama_ctx);
        if (n_batch <= 0) {
            LOG_ERROR_ContextInfo("Invalid batch size from llama context: " + std::to_string(n_batch));
            return false;
        }
        
        const size_t max_batch_size = static_cast<size_t>(n_batch);
        
        for (size_t i = 0; i < tokens.size(); i += max_batch_size) {
            size_t end = std::min(i + max_batch_size, tokens.size());
            size_t chunk_size = end - i;
            
            // Clear batch properly (don't free, just reset)
            clear_batch();
            
            // Add tokens to batch manually (proper API)
            for (size_t j = i; j < end && batch.n_tokens < max_batch_size; ++j) {
                size_t batch_idx = j - i;
                
                batch.token[batch.n_tokens] = static_cast<llama_token>(tokens[j]);
                batch.pos[batch.n_tokens] = n_past + batch_idx;
                batch.n_seq_id[batch.n_tokens] = 1;  // Single sequence
                batch.seq_id[batch.n_tokens][0] = 0; // Sequence ID 0
                batch.logits[batch.n_tokens] = (j == end - 1); // Only last token gets logits
                batch.n_tokens++;
            }
            
            // Decode batch
            int result = llama_decode(llama_ctx, batch);
            if (result != 0) {
                LOG_ERROR_ContextInfo("Batch decode failed with result: " + std::to_string(result));
                return false;
            }
            
            n_past += chunk_size;
        }
        
        LOG_DEBUG_ContextInfo("Processed " + std::to_string(tokens.size()) + " tokens, n_past=" + std::to_string(n_past));
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR_ContextInfo("Exception during token processing: " + std::string(e.what()));
        return false;
    }
}

inline std::string ContextInfo::GenerateResponse(const std::string& prompt) {
    if (!InitializeLlamaContext()) {
        LOG_ERROR_ContextInfo("Failed to initialize context for generation");
        return "Error: Context initialization failed";
    }
    
    is_generating = true;
    should_stop_generation = false;
    
    try {
        // Tokenize the prompt
        std::vector<int32_t> prompt_tokens = TokenizePrompt(prompt);
        
        // Process prompt tokens
        if (!ProcessTokensBatch(prompt_tokens)) {
            is_generating = false;
            return "Error: Failed to process prompt";
        }
        
        // Generate response tokens (simplified - real implementation would use sampling)
        std::vector<int32_t> response_tokens;
        const int max_new_tokens = 512;
        
        for (int i = 0; i < max_new_tokens && !should_stop_generation; ++i) {
            // Get logits for next token
            float* logits = llama_get_logits_ith(llama_ctx, batch.n_tokens - 1);
            if (!logits) {
                break;
            }
            
            // Simple greedy sampling (take highest probability token)
            const llama_vocab* vocab = parent_model->GetVocab();
            int32_t vocab_size = llama_vocab_n_tokens(vocab);
            llama_token next_token = 0;
            float max_logit = logits[0];
            
            for (int32_t j = 1; j < vocab_size; ++j) {
                if (logits[j] > max_logit) {
                    max_logit = logits[j];
                    next_token = j;
                }
            }
            
            // Check for EOS token
            if (next_token == llama_vocab_eos(vocab)) {
                break;
            }
            
            response_tokens.push_back(next_token);
            
            // Process the generated token
            clear_batch();
            
            // Add token to batch manually (proper API)
            batch.token[0] = next_token;
            batch.pos[0] = n_past;
            batch.n_seq_id[0] = 1;  // Single sequence
            batch.seq_id[0][0] = 0; // Sequence ID 0
            batch.logits[0] = true; // Generate logits for next token
            batch.n_tokens = 1;
            
            if (llama_decode(llama_ctx, batch) != 0) {
                LOG_ERROR_ContextInfo("Failed to decode generated token");
                break;
            }
            
            n_past++;
        }
        
        is_generating = false;
        
        // Convert response tokens back to text
        std::string response = DetokenizeResponse(response_tokens);
        
        LOG_DEBUG_ContextInfo("Generated response: " + std::to_string(response_tokens.size()) + 
                             " tokens -> " + response.substr(0, 100) + 
                             (response.length() > 100 ? "..." : ""));
        
        return response;
        
    } catch (const std::exception& e) {
        is_generating = false;
        LOG_ERROR_ContextInfo("Exception during generation: " + std::string(e.what()));
        return "Error: " + std::string(e.what());
    }
}

inline void ContextInfo::ClearContext() {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    if (llama_ctx) {
        llama_memory_clear(llama_get_memory(llama_ctx), true);
        n_past = 0;
    }
    
    current_tokens.clear();
    context_needs_rebuild = true;
    stats.current_context_tokens = 0;
    
    LOG_DEBUG_ContextInfo("Context cleared");
}

inline void ContextInfo::ClearMessageHistory() {
    std::lock_guard<std::mutex> lock(context_mutex);
    
    message_history.clear();
    context_needs_rebuild = true;
    stats.message_pairs = 0;
    
    LOG_DEBUG_ContextInfo("Message history cleared");
}

inline bool ContextInfo::IsNearContextLimit(float threshold) const {
    if (stats.max_context_tokens == 0) return false;
    float usage = static_cast<float>(stats.current_context_tokens) / stats.max_context_tokens;
    return usage >= threshold;
}

inline void ContextInfo::RequestSummarization(const std::string& content) {
    if (summarization_callback) {
        LOG_DEBUG_ContextInfo("Requesting summarization for context: " + context_id);
        summarization_callback(context_id, content);
    } else {
        LOG_ERROR_ContextInfo("No summarization callback available");
    }
}

inline void ContextInfo::UpdateStats() {
    stats.message_pairs = message_history.size();
    stats.current_context_tokens = current_tokens.size();
    stats.total_tokens_processed += current_tokens.size();
    stats.last_activity = std::chrono::steady_clock::now();
}

inline void ContextInfo::RegisterSummarizationCallback(std::function<void(std::string, std::string)> callback) {
    summarization_callback = std::move(callback);
    LOG_ContextInfo("Summarization callback registered for context: " + context_id);
}

// Helper function to generate context IDs
inline std::string GenerateContextId() {
    static std::atomic<size_t> counter{0};
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    return "ctx_" + std::to_string(timestamp) + "_" + std::to_string(counter++);
}
