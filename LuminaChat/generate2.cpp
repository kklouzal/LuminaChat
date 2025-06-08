// LlamaManager.hpp - Header-only implementation of llama.cpp integration
// Handles core functionality for llama.cpp
//
// Project Settings:
// C++ Language Standard: ISO C++20 Standard (/std:c++20)
// C Language Standard: ISO C17 (2018) Standard (/std:c17)
// Character Set: Use Unicode Character Set
// Whole Program Optimization: Use Link Time Code Generation
// Optimization: Maximum Optimization (Favor Speed) (/O2)
// Enable Intrinsic Functions: Yes (/Oi)
// Favor Size or Speed: Favor fast code (/Ot)
// Whole Program Optimization: Yes (/GL)
// Enable String Pooling: Yes (/GF)
// Runtime Library: Multi-threaded DLL (/MD)
// Enable Run-Time Type Information (RTTI) YES (/GR)
// Link Time Code Generation: Use Link Time Code Generation (/LTCG)
//
// CODING DIRECTIVES:
// 1. Keep the codebase minimalistic and focused on functionality.
// 2. Stay consistent with similar coding styles and patterns throughout the project.
// 3. Comment code thoroughly, where necessary, to explain complex logic or decisions.
// 4. Always eliminate unused code, dead code, and unused includes.
// 5. Use int32_t, uint32_t, etc. for fixed-width integers instead of int, unsigned int, etc. to ensure portability across platforms.
// 6. Ensure there are no logical errors and the execution paths flow as expected.
// 7. Refactor where necessary to maintain clean code, efficient code, and to conform to the above settings and directives.
// 8. NEVER BREAK FUNCTIONALITY THAT IS ALREADY WORKING.

#pragma once
#include <string>
#include <vector>
#include <cstring>
#include <algorithm>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstdint>
#include <sstream>  // Replace format include with sstream for compatibility
#include <unordered_map> // For channel context management

// Assuming the llama-cpp.h header is available
#include "llama-cpp.h"

// Class to handle all llama.cpp interactions
class LlamaManager {
public:
    // Callback type for token-by-token generation updates
    using TokenGenerationCallback = std::function<void(const std::string&)>;
    using ErrorCallback = std::function<void(const std::string&)>;

    inline LlamaManager() {
        primaryFormatted.resize(2048); // Use primaryFormatted instead of formatted
    }

    inline ~LlamaManager() {
        // Free resources
        ClearMessages();

        if (smpl) llama_sampler_free(smpl);
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }

    // Set error callback for logging
    inline void SetErrorCallback(ErrorCallback callback) {
        errorCallback = callback;
    }

    // Initialization
    inline bool Initialize(const std::string& modelPath, int32_t contextSize, int32_t gpuLayers);
    
    // Simplified message processing with callback
    inline std::string ProcessMessage(const std::string& userInput, TokenGenerationCallback tokenCallback = nullptr, const std::string& username = "User");
    
    // Add message to context without generating response
    inline void AddToContext(const std::string& userInput, const std::string& username = "User");
    
    // Add message to context with capacity checking
    inline bool AddToContextWithCapacityCheck(const std::string& userInput, const std::string& username = "User", const std::string& channelContext = "");
    
    // Get current context usage information
    inline double GetContextUsage() const;
    inline int32_t GetContextCapacity() const;
    
    // System message management
    inline void SetSystemMessage(const std::string& newSystemMessage);
    inline std::string GetSystemMessage() const { return systemMessage; }
    
    // Memory management
    inline void ClearMessages();
    inline bool PruneMessagesIfNeeded(bool forceAggressive = false);
    
    // Chat template management
    inline std::string GetDefaultChatTemplate() const;
    inline void SetCustomChatTemplate(const std::string& customTemplate);
    
    // Status
    inline bool IsInitialized() const { return modelInitialized; }
    inline llama_model* GetModel() const { return model; }

    // Configure sampler parameters
    inline void ConfigureSampler(float temperature = 0.8f, float min_p = 0.05f);

    // Tokenization method for accurate token counting
    inline int32_t TokenizeText(const std::string& text) const;

    // Context management for primary/secondary channels
    inline void EnsurePrimaryContext();
    inline void SwitchToChannelContext(const std::string& channelId);

private:
    // llama.cpp components
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_sampler* smpl = nullptr;
    const llama_vocab* vocab = nullptr;
    
    // Primary shared context (used by main application and shared Discord channels)
    std::vector<std::pair<std::string, std::string>> primaryMessageHistory;
    std::vector<char> primaryFormatted;
    int32_t primaryPrevLen = 0;
    
    // Settings
    std::string chatTemplate;
    std::string systemMessage;
    bool modelInitialized = false;
    ErrorCallback errorCallback;

    // User tracking for better context
    std::string lastSpeaker;

    // Secondary segregated channel contexts
    std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> segregatedChannelHistories;
    std::unordered_map<std::string, std::vector<char>> segregatedChannelBuffers;
    std::unordered_map<std::string, int32_t> segregatedChannelPrevLens;
    std::string currentActiveChannel; // Track which channel context is currently loaded ("" = primary)

    // Helper methods for accessing current context
    inline std::vector<std::pair<std::string, std::string>>& GetCurrentMessageHistory() {
        return currentActiveChannel.empty() ? primaryMessageHistory : 
               segregatedChannelHistories[currentActiveChannel];
    }
    
    inline const std::vector<std::pair<std::string, std::string>>& GetCurrentMessageHistory() const {
        return currentActiveChannel.empty() ? primaryMessageHistory : 
               const_cast<LlamaManager*>(this)->segregatedChannelHistories[currentActiveChannel];
    }
    
    inline std::vector<char>& GetCurrentFormatted() {
        return currentActiveChannel.empty() ? primaryFormatted : 
               segregatedChannelBuffers[currentActiveChannel];
    }
    
    inline const std::vector<char>& GetCurrentFormatted() const {
        return currentActiveChannel.empty() ? primaryFormatted : 
               const_cast<LlamaManager*>(this)->segregatedChannelBuffers[currentActiveChannel];
    }
    
    inline int32_t& GetCurrentPrevLen() {
        return currentActiveChannel.empty() ? primaryPrevLen : 
               segregatedChannelPrevLens[currentActiveChannel];
    }
    
    inline const int32_t& GetCurrentPrevLen() const {
        return currentActiveChannel.empty() ? primaryPrevLen : 
               const_cast<LlamaManager*>(this)->segregatedChannelPrevLens[currentActiveChannel];
    }

    // Helper methods
    inline void LogError(const std::string& message);
    inline void LogWarning(const std::string& message);
    inline void LogInfo(const std::string& message);
    inline std::string GenerateWithCallback(const std::string& prompt, TokenGenerationCallback callback = nullptr);
    inline std::vector<llama_chat_message> ConvertToLlamaMessages() const;

    // Helper methods for channel context management
    inline void SaveCurrentChannelContext();
    inline void LoadChannelContext(const std::string& channelId);
};

// Implementation of inline methods
inline void LlamaManager::LogError(const std::string& message) {
    if (errorCallback) {
        errorCallback("[ERROR]: " + message);
    }
    fprintf(stderr, "[ERROR]: %s\n", message.c_str());
}

inline void LlamaManager::LogWarning(const std::string& message) {
    if (errorCallback) {
        errorCallback("[WARNING]: " + message);
    }
    printf("[WARNING]: %s\n", message.c_str());
}

inline void LlamaManager::LogInfo(const std::string& message) {
    if (errorCallback) {
        errorCallback("[INFO]: " + message);
    }
    printf("[INFO]: %s\n", message.c_str());
}

inline bool LlamaManager::Initialize(const std::string& modelPath, int32_t contextSize, int32_t gpuLayers) {
    if (modelPath.empty()) {
        LogError("Model path cannot be empty");
        return false;
    }

    if (contextSize <= 0 || contextSize > 131072) {
        LogError("Invalid context size. Must be between 1 and 131072");
        return false;
    }

    if (gpuLayers < 0 || gpuLayers > 999) {
        LogError("Invalid GPU layers. Must be between 0 and 999");
        return false;
    }

    // Initialize all backends including GPU support
    ggml_backend_load_all();
    
    // Model loading with GPU configuration
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = gpuLayers;
    
    if (gpuLayers > 0) {
        model_params.main_gpu = 0;
        model_params.tensor_split = nullptr;
        model_params.use_mmap = true;
        model_params.use_mlock = false;
    }

    model = llama_model_load_from_file(modelPath.c_str(), model_params);
    if (!model) {
        LogError("Unable to load model from: " + modelPath);
        return false;
    }

    vocab = llama_model_get_vocab(model);

    // Context creation
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = contextSize;
    ctx_params.n_batch = std::min(static_cast<int32_t>(contextSize), static_cast<int32_t>(2048));
    
    if (gpuLayers > 0) {
        ctx_params.n_batch = std::min(static_cast<int32_t>(512), contextSize);
        ctx_params.n_ubatch = ctx_params.n_batch;
        ctx_params.offload_kqv = true;
    }

    ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LogError("Failed to create the llama_context");
        llama_model_free(model);
        model = nullptr;
        return false;
    }

    ConfigureSampler();

    primaryFormatted.resize(llama_n_ctx(ctx));
    modelInitialized = true;
    return true;
}

inline std::vector<llama_chat_message> LlamaManager::ConvertToLlamaMessages() const {
    std::vector<llama_chat_message> messages;
    const auto& messageHistory = const_cast<LlamaManager*>(this)->GetCurrentMessageHistory();
    for (const auto& msg : messageHistory) {
        messages.push_back({ msg.first.c_str(), msg.second.c_str() });
    }
    return messages;
}

inline void LlamaManager::AddToContext(const std::string& userInput, const std::string& username) {
    if (!modelInitialized) {
        LogWarning("Model not initialized, cannot add message to context");
        return;
    }

    // Add system message if this is the first message
    if (GetCurrentMessageHistory().empty() && !systemMessage.empty()) {
        GetCurrentMessageHistory().emplace_back("system", systemMessage);
    }

    // Simplified message formatting with timestamp and username
    std::string messageContent;
    
    // Add timestamp
    std::time_t now = std::time(nullptr);
    char timeBuffer[32];
    std::strftime(timeBuffer, sizeof(timeBuffer), "%H:%M", std::localtime(&now));
    
    // Format as: [TIMESTAMP] [USERNAME]: message
    messageContent = "[" + std::string(timeBuffer) + "] [" + username + "]: " + userInput;
    
    lastSpeaker = username;
    GetCurrentMessageHistory().emplace_back("user", messageContent);

    // Check if we need to prune messages
    PruneMessagesIfNeeded();

    // Update the formatted buffer to include this new message and process tokens
    const char* tmpl = !chatTemplate.empty() ? chatTemplate.c_str()
        : (model ? llama_model_chat_template(model, nullptr) : nullptr);

    if (tmpl) {
        auto llamaMessages = ConvertToLlamaMessages();
        int32_t required_size = llama_chat_apply_template(tmpl, llamaMessages.data(), llamaMessages.size(), false, nullptr, 0);
        
        if (required_size > static_cast<int32_t>(GetCurrentFormatted().size())) {
            GetCurrentFormatted().resize(required_size * 1.2);
        }
        
        int32_t new_len = llama_chat_apply_template(tmpl, llamaMessages.data(), llamaMessages.size(), false, GetCurrentFormatted().data(), GetCurrentFormatted().size());
        
        if (new_len < 0) {
            LogError("Failed to apply chat template for context message");
            return;
        }

        // Process the new tokens through the model context
        if (new_len > GetCurrentPrevLen()) {
            std::string new_content(GetCurrentFormatted().begin() + GetCurrentPrevLen(), GetCurrentFormatted().begin() + new_len);
            
            // Tokenize the new content
            bool is_first = llama_kv_self_seq_pos_max(ctx, 0) == 0;
            const int32_t n_tokens = -llama_tokenize(vocab, new_content.c_str(), new_content.size(), NULL, 0, is_first, true);
            
            if (n_tokens > 0) {
                std::vector<llama_token> tokens(n_tokens);
                if (llama_tokenize(vocab, new_content.c_str(), new_content.size(), tokens.data(), tokens.size(), is_first, true) >= 0) {
                    
                    // Check if we have enough space in context
                    int32_t n_ctx_used = llama_kv_self_seq_pos_max(ctx, 0);
                    if (n_ctx_used + n_tokens <= llama_n_ctx(ctx)) {
                        // Process tokens through the model context
                        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
                        if (llama_decode(ctx, batch) == 0) {
                            GetCurrentPrevLen() = new_len;
                            return;
                        } else {
                            LogWarning("Failed to decode context tokens");
                        }
                    } else {
                        LogWarning("Not enough context space for new message");
                    }
                } else {
                    LogWarning("Failed to tokenize context message");
                }
            }
        }
        
        GetCurrentPrevLen() = new_len;
    }
}

inline bool LlamaManager::AddToContextWithCapacityCheck(const std::string& userInput, const std::string& username, const std::string& channelContext) {
    if (!modelInitialized || !ctx) {
        LogWarning("Model not initialized, cannot add message to context");
        return false;
    }

    // Ensure we're in the correct context BEFORE checking capacity or adding messages
    if (!channelContext.empty()) {
        if (currentActiveChannel != channelContext) {
            SwitchToChannelContext(channelContext);
        }
    } else {
        if (!currentActiveChannel.empty()) {
            SwitchToChannelContext("");
        }
    }

    // Check if adding this message would exceed 50% capacity
    double currentUsage = GetContextUsage();
    if (currentUsage >= 0.5) {
        return false;
    }

    // Estimate token count for this message to avoid processing if it would exceed capacity
    std::string messageContent;
    std::time_t now = std::time(nullptr);
    char timeBuffer[32];
    std::strftime(timeBuffer, sizeof(timeBuffer), "%H:%M", std::localtime(&now));
    messageContent = "[" + std::string(timeBuffer) + "] [" + username + "]: " + userInput;
    
    int32_t estimatedTokens = static_cast<int32_t>(messageContent.length() / 4);
    int32_t n_ctx = llama_n_ctx(ctx);
    int32_t n_ctx_used = llama_kv_self_seq_pos_max(ctx, 0);
    
    if (n_ctx_used + estimatedTokens > n_ctx * 0.6) {
        return false;
    }

    // Add system message if this is the first message
    if (GetCurrentMessageHistory().empty() && !systemMessage.empty()) {
        GetCurrentMessageHistory().emplace_back("system", systemMessage);
    }

    lastSpeaker = username;
    GetCurrentMessageHistory().emplace_back("user", messageContent);

    // Update the formatted buffer and process tokens
    const char* tmpl = !chatTemplate.empty() ? chatTemplate.c_str()
        : (model ? llama_model_chat_template(model, nullptr) : nullptr);

    if (tmpl) {
        auto llamaMessages = ConvertToLlamaMessages();
        int32_t required_size = llama_chat_apply_template(tmpl, llamaMessages.data(), llamaMessages.size(), false, nullptr, 0);
        
        if (required_size > static_cast<int32_t>(GetCurrentFormatted().size())) {
            GetCurrentFormatted().resize(required_size + 512);
        }
        
        int32_t new_len = llama_chat_apply_template(tmpl, llamaMessages.data(), llamaMessages.size(), false, GetCurrentFormatted().data(), GetCurrentFormatted().size());
        
        if (new_len < 0) {
            LogError("Failed to apply chat template for context message");
            GetCurrentMessageHistory().pop_back();
            if (GetCurrentMessageHistory().size() == 1 && GetCurrentMessageHistory()[0].first == "system") {
                GetCurrentMessageHistory().pop_back();
            }
            return false;
        }

        // Process the new tokens through the model context
        if (new_len > GetCurrentPrevLen()) {
            std::string new_content(GetCurrentFormatted().begin() + GetCurrentPrevLen(), GetCurrentFormatted().begin() + new_len);
            
            // Tokenize the new content
            bool is_first = llama_kv_self_seq_pos_max(ctx, 0) == 0;
            const int32_t n_tokens = -llama_tokenize(vocab, new_content.c_str(), new_content.size(), NULL, 0, is_first, true);
            
            if (n_tokens > 0) {
                std::vector<llama_token> tokens(n_tokens);
                if (llama_tokenize(vocab, new_content.c_str(), new_content.size(), tokens.data(), tokens.size(), is_first, true) >= 0) {
                    
                    // Final check if we have enough space in context
                    if (n_ctx_used + n_tokens <= n_ctx) {
                        // Process tokens through the model context
                        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
                        if (llama_decode(ctx, batch) == 0) {
                            GetCurrentPrevLen() = new_len;
                            return true;
                        } else {
                            LogWarning("Failed to decode context tokens");
                        }
                    } else {
                        LogWarning("Not enough context space for new message");
                    }
                } else {
                    LogWarning("Failed to tokenize context message");
                }
            }
        }
        
        GetCurrentPrevLen() = new_len;
    }

    return true;
}

inline void LlamaManager::SwitchToChannelContext(const std::string& channelId) {
    if (currentActiveChannel == channelId) {
        return; // Already on this channel
    }
    
    // Save current context if we have one
    SaveCurrentChannelContext();
    
    // Load the new channel context
    LoadChannelContext(channelId);
    currentActiveChannel = channelId;
}

inline void LlamaManager::SaveCurrentChannelContext() {
    if (currentActiveChannel.empty()) {
        // We're saving the primary context - already stored in primary variables
        return;
    }
    
    // Save segregated channel context
    segregatedChannelHistories[currentActiveChannel] = GetCurrentMessageHistory();
    segregatedChannelBuffers[currentActiveChannel] = GetCurrentFormatted();
    segregatedChannelPrevLens[currentActiveChannel] = GetCurrentPrevLen();
}

inline void LlamaManager::LoadChannelContext(const std::string& channelId) {
    // Clear current model context first
    if (ctx) {
        llama_kv_self_clear(ctx);
    }
    
    if (channelId.empty()) {
        primaryPrevLen = 0;
    } else {
        auto historyIt = segregatedChannelHistories.find(channelId);
        if (historyIt != segregatedChannelHistories.end()) {
            primaryMessageHistory = historyIt->second;
            
            auto bufferIt = segregatedChannelBuffers.find(channelId);
            if (bufferIt != segregatedChannelBuffers.end()) {
                primaryFormatted = bufferIt->second;
            } else {
                primaryFormatted.clear();
                primaryFormatted.resize(llama_n_ctx(ctx));
            }
            
            auto prevLenIt = segregatedChannelPrevLens.find(channelId);
            if (prevLenIt != segregatedChannelPrevLens.end()) {
                primaryPrevLen = prevLenIt->second;
            } else {
                primaryPrevLen = 0;
            }
        } else {
            primaryMessageHistory.clear();
            primaryFormatted.clear();
            primaryFormatted.resize(llama_n_ctx(ctx));
            primaryPrevLen = 0;
        }
    }
    
    // Rebuild model context from current message history
    if (!GetCurrentMessageHistory().empty()) {
        const char* tmpl = !chatTemplate.empty() ? chatTemplate.c_str()
            : (model ? llama_model_chat_template(model, nullptr) : nullptr);
        
        if (tmpl) {
            auto llamaMessages = ConvertToLlamaMessages();
            int32_t required_size = llama_chat_apply_template(tmpl, llamaMessages.data(), llamaMessages.size(), false, nullptr, 0);
            
            if (required_size > static_cast<int32_t>(GetCurrentFormatted().size())) {
                GetCurrentFormatted().resize(required_size * 1.2);
            }
            
            int32_t new_len = llama_chat_apply_template(tmpl, llamaMessages.data(), llamaMessages.size(), false, GetCurrentFormatted().data(), GetCurrentFormatted().size());
            
            if (new_len > 0) {
                std::string full_content(GetCurrentFormatted().begin(), GetCurrentFormatted().begin() + new_len);
                
                if (!full_content.empty()) {
                    bool is_first = true;
                    const int32_t n_tokens = -llama_tokenize(vocab, full_content.c_str(), full_content.size(), NULL, 0, is_first, true);
                    
                    if (n_tokens > 0) {
                        int32_t n_ctx = llama_n_ctx(ctx);
                        
                        if (n_tokens > n_ctx) {
                            LogWarning("Context too large during channel switch, truncating");
                            primaryPrevLen = 0;
                            return;
                        }
                        
                        std::vector<llama_token> tokens(n_tokens);
                        if (llama_tokenize(vocab, full_content.c_str(), full_content.size(), tokens.data(), tokens.size(), is_first, true) >= 0) {
                            
                            const int32_t batch_size = std::min(static_cast<int32_t>(512), n_ctx / 4);
                            int32_t processed = 0;
                            bool decode_success = true;
                            
                            while (processed < n_tokens && decode_success) {
                                int32_t current_batch_size = std::min(batch_size, n_tokens - processed);
                                
                                llama_batch batch = llama_batch_get_one(tokens.data() + processed, current_batch_size);
                                
                                if (batch.n_tokens > batch_size) {
                                    LogWarning("Batch size exceeded during context loading, stopping");
                                    decode_success = false;
                                    break;
                                }
                                
                                if (llama_decode(ctx, batch) != 0) {
                                    LogWarning("Failed to decode context batch during channel switch");
                                    decode_success = false;
                                    break;
                                }
                                
                                processed += current_batch_size;
                            }
                            
                            if (decode_success && processed == n_tokens) {
                                primaryPrevLen = new_len;
                            } else {
                                LogWarning("Context loading incomplete during channel switch");
                                primaryPrevLen = 0;
                                
                                if (ctx) {
                                    llama_kv_self_clear(ctx);
                                }
                            }
                        } else {
                            LogWarning("Failed to tokenize context during channel switch");
                            primaryPrevLen = 0;
                        }
                    } else {
                        LogWarning("No tokens generated from context during channel switch");
                        primaryPrevLen = 0;
                    }
                }
            } else {
                LogWarning("Failed to apply template during channel switch");
                primaryPrevLen = 0;
            }
        }
    } else {
        primaryPrevLen = 0;
    }
}

inline void LlamaManager::EnsurePrimaryContext() {
    if (!currentActiveChannel.empty()) {
        SwitchToChannelContext(""); // Switch to primary context (empty string)
    }
}

inline void LlamaManager::ClearMessages() {
    // Save current context before clearing
    SaveCurrentChannelContext();
    
    // Clear primary context
    primaryMessageHistory.clear();
    primaryPrevLen = 0;
    lastSpeaker.clear();
    
    // Clear all segregated contexts
    segregatedChannelHistories.clear();
    segregatedChannelBuffers.clear();
    segregatedChannelPrevLens.clear();
    
    // Ensure we're on primary context
    currentActiveChannel = "";
    
    // Clear model context
    if (ctx) {
        llama_kv_self_clear(ctx);
    }
}

inline double LlamaManager::GetContextUsage() const {
    if (!ctx) return 0.0;
    
    int32_t n_ctx = llama_n_ctx(ctx);
    int32_t n_ctx_used = llama_kv_self_seq_pos_max(ctx, 0);
    
    // Estimate tokens for current message history
    int32_t estimated_tokens = 0;
    for (const auto& msg : GetCurrentMessageHistory()) {
        estimated_tokens += static_cast<int32_t>(msg.second.length() / 4); // Rough estimate: 4 chars per token
    }
    
    return static_cast<double>(n_ctx_used + estimated_tokens) / n_ctx;
}

inline int32_t LlamaManager::GetContextCapacity() const {
    if (!ctx) return 0;
    return llama_n_ctx(ctx);
}

inline void LlamaManager::SetSystemMessage(const std::string& newSystemMessage) {
    systemMessage = newSystemMessage;

    if (GetCurrentMessageHistory().empty()) {
        return; // No messages yet, nothing to update
    }

    bool hadSystem = !GetCurrentMessageHistory().empty() && GetCurrentMessageHistory()[0].first == "system";

    if (hadSystem) {
        if (systemMessage.empty()) {
            GetCurrentMessageHistory().erase(GetCurrentMessageHistory().begin());
        } else {
            GetCurrentMessageHistory()[0].second = systemMessage;
        }
    } else if (!systemMessage.empty()) {
        GetCurrentMessageHistory().insert(GetCurrentMessageHistory().begin(), {"system", systemMessage});
    }

    // Update the formatted buffer
    const char* tmpl = !chatTemplate.empty() ? chatTemplate.c_str()
        : (model ? llama_model_chat_template(model, nullptr) : nullptr);

    if (tmpl) {
        auto llamaMessages = ConvertToLlamaMessages();
        primaryPrevLen = llama_chat_apply_template(tmpl, llamaMessages.data(), llamaMessages.size(), false, nullptr, 0);

        if (primaryPrevLen > static_cast<int32_t>(GetCurrentFormatted().size())) {
            GetCurrentFormatted().resize(primaryPrevLen * 1.2);
        }
    }
}

inline bool LlamaManager::PruneMessagesIfNeeded(bool forceAggressive) {
    if (!ctx) return false;

    int32_t n_ctx = llama_n_ctx(ctx);
    int32_t n_ctx_used = llama_kv_self_seq_pos_max(ctx, 0);
    
    int32_t recent_tokens = 0;
    if (!GetCurrentMessageHistory().empty()) {
        recent_tokens = static_cast<int32_t>(GetCurrentMessageHistory().back().second.length() / 4); // Rough estimate
    }

    double fill_ratio = static_cast<double>(n_ctx_used + recent_tokens) / n_ctx;
    
    if (fill_ratio <= 0.6 && !forceAggressive) {
        return false;
    }
    
    float keep_ratio;
    bool aggressive_pruning;
    
    if (fill_ratio > 0.9 || forceAggressive) {
        LogWarning("Context nearly full, performing aggressive pruning...");
        keep_ratio = 0.3f;
        aggressive_pruning = true;
    } else if (fill_ratio > 0.75) {
        LogWarning("Context filling up, performing moderate pruning...");
        keep_ratio = 0.5f;
        aggressive_pruning = false;
    } else {
        LogInfo("Context approaching capacity, performing light pruning...");
        keep_ratio = 0.7f;
        aggressive_pruning = false;
    }
    
    // Simple pruning: keep system message and most recent messages
    std::vector<std::pair<std::string, std::string>> pruned;
    
    bool has_system = !GetCurrentMessageHistory().empty() && GetCurrentMessageHistory()[0].first == "system";
    if (has_system) {
        pruned.push_back(GetCurrentMessageHistory()[0]);
    }
    
    size_t non_system_count = GetCurrentMessageHistory().size() - (has_system ? 1 : 0);
    size_t to_keep = std::max(size_t(2), static_cast<size_t>(static_cast<double>(non_system_count) * keep_ratio));
    
    // Keep the most recent messages
    size_t start_idx = GetCurrentMessageHistory().size() - std::min(to_keep, non_system_count);
    if (has_system) start_idx = std::max(start_idx, size_t(1));
    
    for (size_t i = start_idx; i < GetCurrentMessageHistory().size(); i++) {
        pruned.push_back(GetCurrentMessageHistory()[i]);
    }
    
    GetCurrentMessageHistory() = std::move(pruned);
    
    if (aggressive_pruning) {
        llama_kv_self_clear(ctx);
    }
    
    // Update formatted buffer
    const char* tmpl = !chatTemplate.empty() ? chatTemplate.c_str()
        : (model ? llama_model_chat_template(model, nullptr) : nullptr);
    
    if (tmpl) {
        auto llamaMessages = ConvertToLlamaMessages();
        primaryPrevLen = llama_chat_apply_template(tmpl, llamaMessages.data(), llamaMessages.size(), false, nullptr, 0);
        
        if (primaryPrevLen > static_cast<int32_t>(GetCurrentFormatted().size())) {
            GetCurrentFormatted().resize(static_cast<size_t>(primaryPrevLen) * 1.2);
        }
    }
    
    return true;
}

inline std::string LlamaManager::GetDefaultChatTemplate() const {
    if (!model) return "";
    
    const char* tmpl = llama_model_chat_template(model, nullptr);
    return tmpl ? std::string(tmpl) : "";
}

inline void LlamaManager::SetCustomChatTemplate(const std::string& customTemplate) {
    chatTemplate = customTemplate;
}

inline void LlamaManager::ConfigureSampler(float temperature, float min_p) {
    if (smpl) {
        llama_sampler_free(smpl);
    }
    
    smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
}

inline std::string LlamaManager::GenerateWithCallback(const std::string& prompt, TokenGenerationCallback callback) {
    if (prompt.empty()) {
        LogWarning("Empty prompt provided for generation");
        return "";
    }

    std::string response;
    response.reserve(1024); // Pre-allocate to reduce reallocations
    
    bool is_first = llama_kv_self_seq_pos_max(ctx, 0) == 0;
    
    const int32_t n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, is_first, true);
    if (n_prompt_tokens <= 0) {
        LogError("Failed to tokenize the prompt or prompt is empty");
        return response;
    }

    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
        LogError("Failed to tokenize the prompt");
        return response;
    }

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    
    // Calculate safe generation limits
    int32_t n_ctx = llama_n_ctx(ctx);
    int32_t n_ctx_used = llama_kv_self_seq_pos_max(ctx, 0);
    int32_t available_ctx = n_ctx - n_ctx_used - static_cast<int32_t>(prompt_tokens.size());
    int32_t max_tokens = std::min(available_ctx, n_ctx / 2); // Use available space or half context, whichever is smaller
    
    if (max_tokens <= 0) {
        LogError("No context space available for generation");
        return response;
    }
    
    int32_t generated_tokens = 0;
    
    while (generated_tokens < max_tokens) {
        // Check context space before each decode
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            LogWarning("Context size exceeded during generation");
            break;
        }

        if (llama_decode(ctx, batch)) {
            LogError("Failed to decode during generation");
            break;
        }

        llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        // Use larger buffer and proper bounds checking
        constexpr size_t PIECE_BUFFER_SIZE = 512;
        char buf[PIECE_BUFFER_SIZE];
        int32_t n = llama_token_to_piece(vocab, new_token_id, buf, PIECE_BUFFER_SIZE - 1, 0, true);
        
        if (n < 0) {
            LogError("Failed to convert token to piece");
            break;
        }
        
        if (n >= static_cast<int32_t>(PIECE_BUFFER_SIZE)) {
            LogWarning("Token piece truncated due to buffer size");
            n = static_cast<int32_t>(PIECE_BUFFER_SIZE) - 1;
        }

        buf[n] = '\0'; // Ensure null termination
        std::string piece(buf, n);
        response += piece;

        if (callback) {
            callback(piece);
        }

        batch = llama_batch_get_one(&new_token_id, 1);
        generated_tokens++;
        n_ctx_used++; // Track context usage
    }

    if (generated_tokens >= max_tokens) {
        // Use std::to_string instead of problematic formatting
        std::string logMessage = "Generation completed: " + std::to_string(generated_tokens) + 
                                " tokens generated (limit: " + std::to_string(max_tokens) + ")";
        LogInfo(logMessage);
    }

    return response;
}

inline std::string LlamaManager::ProcessMessage(const std::string& userInput, TokenGenerationCallback tokenCallback, const std::string& username) {
    if (!modelInitialized || !ctx) {
        LogError("Model not initialized. Please start the model first.");
        return "";
    }

    if (userInput.empty()) {
        LogWarning("Empty user input received");
        return "";
    }

    // Add system message if this is the first message
    if (GetCurrentMessageHistory().empty() && !systemMessage.empty()) {
        GetCurrentMessageHistory().emplace_back("system", systemMessage);
    }

    // Simplified message formatting with timestamp and username
    std::string messageContent;
    
    // Add timestamp
    std::time_t now = std::time(nullptr);
    char timeBuffer[32];
    std::strftime(timeBuffer, sizeof(timeBuffer), "%H:%M", std::localtime(&now));
    
    // Format as: [TIMESTAMP] [USERNAME]: message
    messageContent = "[" + std::string(timeBuffer) + "] [" + username + "]: " + userInput;
    
    lastSpeaker = username;
    GetCurrentMessageHistory().emplace_back("user", messageContent);

    PruneMessagesIfNeeded();

    auto llamaMessages = ConvertToLlamaMessages();

    const char* tmpl = !chatTemplate.empty() ? chatTemplate.c_str()
        : llama_model_chat_template(model, nullptr);

    if (!tmpl) {
        LogError("No chat template available");
        return "";
    }

    int32_t required_size = llama_chat_apply_template(tmpl, llamaMessages.data(), llamaMessages.size(), true, nullptr, 0);
    if (required_size <= 0) {
        LogError("Failed to calculate template size");
        return "";
    }

    if (required_size > static_cast<int32_t>(GetCurrentFormatted().size())) {
        GetCurrentFormatted().resize(required_size * 1.2);
    }

    int32_t new_len = llama_chat_apply_template(tmpl, llamaMessages.data(), llamaMessages.size(), true, GetCurrentFormatted().data(), GetCurrentFormatted().size());
    if (new_len < 0) {
        LogError("Failed to apply the chat template");
        return "";
    }

    std::string prompt(GetCurrentFormatted().begin() + GetCurrentPrevLen(), GetCurrentFormatted().begin() + new_len);
    std::string response = GenerateWithCallback(prompt, tokenCallback);

    if (!response.empty()) {
        GetCurrentMessageHistory().emplace_back("assistant", response);
        
        auto updatedMessages = ConvertToLlamaMessages();
        GetCurrentPrevLen() = llama_chat_apply_template(tmpl, updatedMessages.data(), updatedMessages.size(), false, nullptr, 0);
        if (GetCurrentPrevLen() < 0) {
            LogWarning("Failed to update template length");
            GetCurrentPrevLen() = 0; // Reset to prevent issues
        }
    }

    return response;
}

inline int32_t LlamaManager::TokenizeText(const std::string& text) const {
    if (!modelInitialized || !vocab || text.empty()) {
        return 0;
    }
    
    // Use llama.cpp tokenization to get accurate count
    const int32_t n_tokens = -llama_tokenize(vocab, text.c_str(), text.size(), NULL, 0, true, true);
    return n_tokens > 0 ? n_tokens : 0;
}