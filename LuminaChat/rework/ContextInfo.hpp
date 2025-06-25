#pragma once

#include "ChatTemplateManager.hpp"
#include "ModelInfo.hpp"
#include "TokenCache.hpp"
#include "Logger.hpp"
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
    llama_context* llama_ctx;
    std::vector<int32_t> current_tokens;
    bool context_needs_rebuild;
    
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
    
    if (llama_ctx) {
        // llama_free(llama_ctx); // Would need actual llama.cpp integration
        llama_ctx = nullptr;
    }
    
    LOG_DEBUG_ContextInfo("ContextInfo destroyed: " + context_id);
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

inline bool ContextInfo::RebuildContext(RebuildStrategy strategy) {
    try {
        switch (strategy) {
            case RebuildStrategy::FULL:
                RebuildContext_Full();
                stats.full_rebuilds++;
                break;
            case RebuildStrategy::PARTIAL:
                RebuildContext_Partial();
                stats.partial_rebuilds++;
                break;
            case RebuildStrategy::TEMPLATE_ONLY:
                RebuildContext_TemplateOnly();
                stats.template_renders++;
                break;
        }
        
        context_needs_rebuild = false;
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR_ContextInfo("Context rebuild failed: " + std::string(e.what()));
        return false;
    }
}

inline void ContextInfo::RebuildContext_Full() {
    LOG_DEBUG_ContextInfo("Performing full context rebuild");
    
    // Clear existing context
    current_tokens.clear();
    
    // Render template with current message history
    std::string full_prompt = BuildFullPrompt();
    
    // Tokenize the complete prompt
    current_tokens = TokenizePrompt(full_prompt);
    
    // Update statistics
    stats.current_context_tokens = current_tokens.size();
    
    LOG_DEBUG_ContextInfo("Full rebuild complete: " + std::to_string(current_tokens.size()) + " tokens");
}

inline void ContextInfo::RebuildContext_Partial() {
    LOG_DEBUG_ContextInfo("Performing partial context rebuild");
    
    // For partial rebuild, we assume only new messages were added
    // This is an optimization that would require more sophisticated tracking
    // For now, fall back to full rebuild
    RebuildContext_Full();
}

inline void ContextInfo::RebuildContext_TemplateOnly() {
    LOG_DEBUG_ContextInfo("Performing template-only rebuild");
    
    // Re-render template without changing message history
    // This would be used when only template sections change
    RebuildContext_Full(); // Simplified implementation
}

inline std::string ContextInfo::BuildFullPrompt() {
    // Use ChatTemplateManager to render the complete template
    return template_manager->RenderTemplate(message_history);
}

inline std::vector<int32_t> ContextInfo::TokenizePrompt(const std::string& prompt) {
    if (!token_cache) {
        throw std::runtime_error("Token cache not available");
    }
    
    std::vector<int32_t> tokens;
    if (!token_cache->GetTokens(prompt, tokens)) {
        // Cache miss - need to tokenize using actual model
        // For now, create mock tokens for testing
        // In real implementation, this would call llama tokenization
        tokens.clear();
        for (size_t i = 0; i < prompt.length(); i += 4) {
            tokens.push_back(static_cast<int32_t>(i / 4 + 1000)); // Mock token IDs
        }
        
        // Store in cache for future use
        token_cache->StoreTokens(prompt, tokens);
    }
    
    return tokens;
}

inline std::string ContextInfo::DetokenizeResponse(const std::vector<int32_t>& tokens) {
    if (!token_cache) {
        throw std::runtime_error("Token cache not available");
    }
    
    std::string text;
    if (!token_cache->GetText(tokens, text)) {
        // Cache miss - need to detokenize using actual model
        // For now, create mock text for testing
        // In real implementation, this would call llama detokenization
        std::ostringstream oss;
        oss << "Generated text from " << tokens.size() << " tokens: ";
        for (size_t i = 0; i < std::min(tokens.size(), static_cast<size_t>(5)); ++i) {
            oss << "[" << tokens[i] << "] ";
        }
        text = oss.str();
        
        // Store in cache for future use
        token_cache->StoreText(tokens, text);
    }
    
    return text;
}

inline bool ContextInfo::IsNearContextLimit(float threshold) const {
    if (stats.max_context_tokens == 0) {
        return false;
    }
    
    return stats.GetContextUsageRatio() >= threshold;
}

inline void ContextInfo::RequestSummarization(const std::string& content) {
    if (summarization_callback) {
        LOG_ContextInfo("Requesting summarization for context: " + context_id);
        summarization_callback(context_id, content);
    } else {
        LOG_ERROR_ContextInfo("No summarization callback registered");
    }
}

inline void ContextInfo::RegisterSummarizationCallback(std::function<void(std::string, std::string)> callback) {
    summarization_callback = std::move(callback);
    LOG_DEBUG_ContextInfo("Summarization callback registered");
}

inline void ContextInfo::UpdateStats() {
    stats.last_activity = std::chrono::steady_clock::now();
    stats.message_pairs = message_history.size();
    stats.current_context_tokens = current_tokens.size();
}

inline bool ContextInfo::ValidateContext() const {
    if (state == ContextState::ERROR_STATE) {
        return false;
    }
    
    if (!template_manager || !template_manager->ValidateTemplate()) {
        return false;
    }
    
    if (!parent_model || !token_cache) {
        return false;
    }
    
    return true;
}

inline void ContextInfo::ClearContext() {
    std::lock_guard<std::mutex> lock(context_mutex);
    
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

inline void ContextInfo::SetMaxContextTokens(size_t max_tokens) {
    stats.max_context_tokens = max_tokens;
    LOG_DEBUG_ContextInfo("Max context tokens set to: " + std::to_string(max_tokens));
}

inline std::string ContextInfo::GetCurrentPrompt() const {
    if (template_manager) {
        return template_manager->RenderTemplate(message_history);
    }
    return "";
}

inline void ContextInfo::DumpContextInfo() const {
    std::ostringstream oss;
    oss << "=== ContextInfo Debug Dump ===" << std::endl;
    oss << "Context ID: " << context_id << std::endl;
    oss << "State: " << static_cast<int>(state) << std::endl;
    oss << "Message pairs: " << stats.message_pairs << std::endl;
    oss << "Current tokens: " << stats.current_context_tokens << std::endl;
    oss << "Max tokens: " << stats.max_context_tokens << std::endl;
    oss << "Usage ratio: " << std::fixed << std::setprecision(2) << stats.GetContextUsageRatio() << std::endl;
    oss << "Full rebuilds: " << stats.full_rebuilds << std::endl;
    oss << "Partial rebuilds: " << stats.partial_rebuilds << std::endl;
    oss << "Template renders: " << stats.template_renders << std::endl;
    oss << "Age: " << stats.GetAge().count() << " seconds" << std::endl;
    oss << "Time since activity: " << stats.GetTimeSinceActivity().count() << " seconds" << std::endl;
    oss << "Template dirty: " << (template_manager ? template_manager->IsTemplateDirty() : false) << std::endl;
    
    LOG_ContextInfo(oss.str());
}

inline std::string ContextInfo::GetContextSummary() const {
    std::ostringstream oss;
    oss << "Context[" << context_id << "]: ";
    oss << stats.message_pairs << " messages, ";
    oss << stats.current_context_tokens << "/" << stats.max_context_tokens << " tokens ";
    oss << "(" << std::fixed << std::setprecision(1) << stats.GetContextUsageRatio() * 100 << "%)";
    
    return oss.str();
}

inline void ContextInfo::AddMessage(const std::string& role, const std::string& content) {
    std::lock_guard<std::mutex> lock(context_mutex);
    message_history.emplace_back(role, content);
    context_needs_rebuild = true;
    UpdateStats();
    LOG_DEBUG_ContextInfo("Added message: " + role + " -> " + content.substr(0, 50) + 
                         (content.length() > 50 ? "..." : ""));
}

inline std::vector<std::pair<std::string, std::string>> ContextInfo::GetMessages() const {
    std::lock_guard<std::mutex> lock(context_mutex);
    return message_history;
}

inline int32_t ContextInfo::GetCurrentTokenCount() const {
    return static_cast<int32_t>(current_tokens.size());
}

inline void ContextInfo::UpdateMotif(const std::string& motif) {
    UpdateMotifContext(motif);
}

// Helper function to generate context IDs
inline std::string GenerateContextId() {
    static std::atomic<size_t> counter{0};
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    return "ctx_" + std::to_string(timestamp) + "_" + std::to_string(counter++);
}
