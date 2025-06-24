#pragma once

#include "TokenCache.hpp"
#include "SettingsManager.hpp"
#include "Logger.hpp"
#include <memory>
#include <string>
#include <functional>
#include <atomic>
#include <mutex>

// ModelInfo: Model resource management and vocabulary access
// - Owns TokenCache instance for this model
// - Manages llama vocabulary for tokenization  
// - Handles model loading/unloading lifecycle
// - Settings integration for model paths and parameters
// - Optional resource monitoring callbacks

#define LOG_ModelInfo(message) \
    GetLogger().LogMessage(Logger::LogLevel::INFO, "ModelInfo", message)

#define LOG_DEBUG_ModelInfo(message) \
    GetLogger().LogMessage(Logger::LogLevel::DEBUG, "ModelInfo", message)

#define LOG_ERROR_ModelInfo(message) \
    GetLogger().LogMessage(Logger::LogLevel::ERROR, "ModelInfo", message)

// Forward declarations
struct llama_model;
struct llama_context;

enum class ModelState {
    UNLOADED,
    LOADING,
    LOADED,
    ERROR_STATE
};

enum class ResourceEvent {
    MODEL_LOADING_STARTED,
    MODEL_LOADED,
    MODEL_UNLOADED,
    MEMORY_USAGE_HIGH,
    TOKENIZATION_ERROR,
    CLEANUP_COMPLETED
};

struct ModelConfig {
    std::string model_path;
    int32_t context_size = 8192;
    int32_t gpu_layers = -1;  // -1 = auto-detect
    int32_t batch_size = 512;
    int32_t threads = 0;      // 0 = auto-detect
    bool use_mmap = true;
    bool use_mlock = false;
    float rope_freq_base = 0.0f;  // 0.0 = auto
    float rope_freq_scale = 0.0f; // 0.0 = auto
    
    // Load from settings
    void LoadFromSettings(SettingsManager& settings, const std::string& model_prefix) {
        model_path = settings.GetString("Models", model_prefix + "_model_path", "");
        context_size = settings.GetInt("Models", model_prefix + "_context_size", 8192);
        gpu_layers = settings.GetInt("Models", model_prefix + "_gpu_layers", -1);
        batch_size = settings.GetInt("Models", model_prefix + "_batch_size", 512);
        threads = settings.GetInt("Models", model_prefix + "_threads", 0);
        use_mmap = settings.GetBool("Models", model_prefix + "_use_mmap", true);
        use_mlock = settings.GetBool("Models", model_prefix + "_use_mlock", false);
        rope_freq_base = settings.GetFloat("Models", model_prefix + "_rope_freq_base", 0.0f);
        rope_freq_scale = settings.GetFloat("Models", model_prefix + "_rope_freq_scale", 0.0f);
    }
    
    // Save to settings
    void SaveToSettings(SettingsManager& settings, const std::string& model_prefix) const {
        settings.SetString("Models", model_prefix + "_model_path", model_path);
        settings.SetInt("Models", model_prefix + "_context_size", context_size);
        settings.SetInt("Models", model_prefix + "_gpu_layers", gpu_layers);
        settings.SetInt("Models", model_prefix + "_batch_size", batch_size);
        settings.SetInt("Models", model_prefix + "_threads", threads);
        settings.SetBool("Models", model_prefix + "_use_mmap", use_mmap);
        settings.SetBool("Models", model_prefix + "_use_mlock", use_mlock);
        settings.SetFloat("Models", model_prefix + "_rope_freq_base", rope_freq_base);
        settings.SetFloat("Models", model_prefix + "_rope_freq_scale", rope_freq_scale);
    }
    
    bool IsValid() const {
        return !model_path.empty() && context_size > 0;
    }
};

// Forward declaration of IModelInfo
class IModelInfo {
public:
    virtual ~IModelInfo() = default;
    virtual TokenCache& GetTokenCache() = 0;
    virtual const TokenCache& GetTokenCache() const = 0;
};

class ModelInfo : public IModelInfo {
private:
    // Core components
    std::unique_ptr<TokenCache> token_cache;
    ModelConfig config;
    std::string model_id;
    
    // Model state
    std::atomic<ModelState> state{ModelState::UNLOADED};
    llama_model* model = nullptr;
    llama_context* temp_context = nullptr; // For tokenization operations
    
    // Thread safety
    mutable std::mutex model_mutex;
    
    // Optional resource monitoring callback
    // Orchestrator (higher) can register with ModelInfo (lower) for resource events
    std::function<void(const std::string&, ResourceEvent, const std::string&)> resource_callback;
    
    // Memory tracking
    std::atomic<size_t> estimated_memory_usage{0};
    
    // Helper methods
    void NotifyResourceEvent(ResourceEvent event, const std::string& details = "") {
        if (resource_callback) {
            try {
                resource_callback(model_id, event, details);
            } catch (const std::exception& e) {
                LOG_ERROR_ModelInfo("Exception in resource callback: " + std::string(e.what()));
            }
        }
    }
    
    bool InitializeLlama() {
        // This would initialize the llama.cpp library if needed
        // For now, we'll assume it's initialized elsewhere
        return true;
    }
    
public:
    explicit ModelInfo(const std::string& id) 
        : model_id(id), token_cache(std::make_unique<TokenCache>()) {
        LOG_ModelInfo("Created ModelInfo for: " + model_id);
    }
    
    // Constructor with config
    ModelInfo(const std::string& id, const ModelConfig& initial_config) 
        : model_id(id), config(initial_config), token_cache(std::make_unique<TokenCache>()) {
        LOG_ModelInfo("Created ModelInfo for: " + model_id + " with initial config");
    }
    
    // Static helper method to create config from settings
    static ModelConfig CreateConfigFromSettings(SettingsManager& settings, const std::string& model_prefix) {
        ModelConfig config;
        config.LoadFromSettings(settings, model_prefix);
        return config;
    }
    
    // Method to trigger resource events (for testing)
    void TriggerResourceEvent(ResourceEvent event, const std::string& details = "") {
        NotifyResourceEvent(event, details);
    }

    // Non-copyable, movable
    ModelInfo(const ModelInfo&) = delete;
    ModelInfo& operator=(const ModelInfo&) = delete;
    ModelInfo(ModelInfo&&) = default;
    ModelInfo& operator=(ModelInfo&&) = default;
    
    // Model configuration
    void SetConfig(const ModelConfig& new_config) {
        std::lock_guard<std::mutex> lock(model_mutex);
        config = new_config;
        LOG_ModelInfo("Updated config for model: " + model_id);
    }
    
    const ModelConfig& GetConfig() const {
        std::lock_guard<std::mutex> lock(model_mutex);
        return config;
    }
    
    // Settings integration
    bool LoadConfigFromSettings(SettingsManager& settings, const std::string& model_prefix) {
        std::lock_guard<std::mutex> lock(model_mutex);
        
        config.LoadFromSettings(settings, model_prefix);
        
        if (!config.IsValid()) {
            LOG_ERROR_ModelInfo("Invalid configuration loaded for model: " + model_id);
            return false;
        }
        
        LOG_ModelInfo("Loaded config from settings for model: " + model_id + 
            " (path: " + config.model_path + ", context: " + std::to_string(config.context_size) + ")");
        return true;
    }
    
    void SaveConfigToSettings(SettingsManager& settings, const std::string& model_prefix) const {
        std::lock_guard<std::mutex> lock(model_mutex);
        config.SaveToSettings(settings, model_prefix);
        LOG_ModelInfo("Saved config to settings for model: " + model_id);
    }
    
    // Model lifecycle management
    bool LoadModel() {
        std::lock_guard<std::mutex> lock(model_mutex);
        
        if (state == ModelState::LOADED) {
            LOG_ModelInfo("Model already loaded: " + model_id);
            return true;
        }
        
        if (state == ModelState::LOADING) {
            LOG_ModelInfo("Model already loading: " + model_id);
            return false; // Caller should wait
        }
        
        if (!config.IsValid()) {
            LOG_ERROR_ModelInfo("Cannot load model with invalid config: " + model_id);
            state = ModelState::ERROR_STATE;
            return false;
        }
        
        state = ModelState::LOADING;
        NotifyResourceEvent(ResourceEvent::MODEL_LOADING_STARTED, "Loading model: " + config.model_path);
        
        LOG_ModelInfo("Loading model: " + model_id + " from " + config.model_path);
        
        // TODO: Implement actual llama.cpp model loading
        // For now, we'll simulate the loading process
        try {
            if (!InitializeLlama()) {
                throw std::runtime_error("Failed to initialize llama.cpp");
            }
            
            // Simulated model loading - replace with actual llama.cpp calls
            // model = llama_load_model_from_file(config.model_path.c_str(), model_params);
            // if (!model) { throw std::runtime_error("Failed to load model"); }
            
            // Create temporary context for tokenization
            // llama_context_params ctx_params = llama_context_default_params();
            // ctx_params.n_ctx = 512; // Small context just for tokenization
            // temp_context = llama_new_context_with_model(model, ctx_params);
            
            // Estimate memory usage (placeholder)
            estimated_memory_usage = config.context_size * 1024; // Rough estimate
            
            state = ModelState::LOADED;
            NotifyResourceEvent(ResourceEvent::MODEL_LOADED, 
                "Model loaded successfully, estimated memory: " + std::to_string(estimated_memory_usage / 1024 / 1024) + " MB");
            
            LOG_ModelInfo("Successfully loaded model: " + model_id);
            return true;
            
        } catch (const std::exception& e) {
            LOG_ERROR_ModelInfo("Failed to load model " + model_id + ": " + e.what());
            state = ModelState::ERROR_STATE;
            NotifyResourceEvent(ResourceEvent::TOKENIZATION_ERROR, e.what());
            return false;
        }
    }
    
    bool IsLoaded() const {
        return state == ModelState::LOADED;
    }
    
    ModelState GetState() const {
        return state;
    }
    
    void Cleanup() {
        std::lock_guard<std::mutex> lock(model_mutex);
        
        if (state == ModelState::UNLOADED) {
            return;
        }
        
        LOG_ModelInfo("Cleaning up model: " + model_id);
        
        // Clear token cache first
        if (token_cache) {
            token_cache->ClearAll();
        }
        
        // TODO: Implement actual llama.cpp cleanup
        // if (temp_context) {
        //     llama_free(temp_context);
        //     temp_context = nullptr;
        // }
        // 
        // if (model) {
        //     llama_free_model(model);
        //     model = nullptr;
        // }
        
        estimated_memory_usage = 0;
        state = ModelState::UNLOADED;
        
        NotifyResourceEvent(ResourceEvent::CLEANUP_COMPLETED, "Model cleanup completed");
        LOG_ModelInfo("Cleanup completed for model: " + model_id);
    }
    
    // IModelInfo interface implementation
    TokenCache& GetTokenCache() override {
        return *token_cache;
    }
    
    const TokenCache& GetTokenCache() const override {
        return *token_cache;
    }
    
    // Tokenization interface (requires loaded model)
    bool TokenizeText(const std::string& text, std::vector<int32_t>& tokens) {
        if (state != ModelState::LOADED) {
            LOG_ERROR_ModelInfo("Cannot tokenize - model not loaded: " + model_id);
            return false;
        }
        
        // Check cache first
        if (token_cache->GetTokens(text, tokens)) {
            return true;
        }
        
        std::lock_guard<std::mutex> lock(model_mutex);
        
        try {
            // TODO: Implement actual tokenization with llama.cpp
            // For now, simulate tokenization
            tokens.clear();
            // int32_t* token_array = new int32_t[text.length() + 10];
            // int token_count = llama_tokenize(temp_context, text.c_str(), text.length(), 
            //                                 token_array, text.length() + 10, true, true);
            // if (token_count > 0) {
            //     tokens.assign(token_array, token_array + token_count);
            // }
            // delete[] token_array;
            
            // Placeholder tokenization (1 token per 4 characters, roughly)
            size_t estimated_tokens = (text.length() + 3) / 4;
            tokens.reserve(estimated_tokens);
            for (size_t i = 0; i < estimated_tokens; ++i) {
                tokens.push_back(static_cast<int32_t>(1000 + i)); // Fake token IDs
            }
            
            // Store in cache
            token_cache->StoreTokens(text, tokens);
            
            LOG_DEBUG_ModelInfo("Tokenized text (" + std::to_string(text.length()) + 
                " chars → " + std::to_string(tokens.size()) + " tokens)");
            return true;
            
        } catch (const std::exception& e) {
            LOG_ERROR_ModelInfo("Tokenization error for model " + model_id + ": " + e.what());
            NotifyResourceEvent(ResourceEvent::TOKENIZATION_ERROR, e.what());
            return false;
        }
    }
    
    bool DetokenizeText(const std::vector<int32_t>& tokens, std::string& text) {
        if (state != ModelState::LOADED) {
            LOG_ERROR_ModelInfo("Cannot detokenize - model not loaded: " + model_id);
            return false;
        }
        
        // Check cache first
        if (token_cache->GetText(tokens, text)) {
            return true;
        }
        
        std::lock_guard<std::mutex> lock(model_mutex);
        
        try {
            // TODO: Implement actual detokenization with llama.cpp
            // For now, simulate detokenization
            text.clear();
            // for (int32_t token : tokens) {
            //     const char* piece = llama_token_to_piece(temp_context, token);
            //     if (piece) {
            //         text += piece;
            //     }
            // }
            
            // Placeholder detokenization
            text.reserve(tokens.size() * 4); // Rough estimate
            for (size_t i = 0; i < tokens.size(); ++i) {
                if (i > 0) text += " ";
                text += "tok" + std::to_string(i);
            }
            
            // Store in cache
            token_cache->StoreText(tokens, text);
            
            LOG_DEBUG_ModelInfo("Detokenized tokens (" + std::to_string(tokens.size()) + 
                " tokens → " + std::to_string(text.length()) + " chars)");
            return true;
            
        } catch (const std::exception& e) {
            LOG_ERROR_ModelInfo("Detokenization error for model " + model_id + ": " + e.what());
            NotifyResourceEvent(ResourceEvent::TOKENIZATION_ERROR, e.what());
            return false;
        }
    }
    
    // Model information
    const std::string& GetModelId() const {
        return model_id;
    }
    
    std::string GetModelPath() const {
        std::lock_guard<std::mutex> lock(model_mutex);
        return config.model_path;
    }
    
    int32_t GetContextSize() const {
        std::lock_guard<std::mutex> lock(model_mutex);
        return config.context_size;
    }
    
    size_t GetEstimatedMemoryUsage() const {
        return estimated_memory_usage;
    }
    
    // Resource monitoring callback registration
    // Optional: Orchestrator (higher) registers with ModelInfo (lower) for resource events
    void RegisterResourceCallback(std::function<void(const std::string&, ResourceEvent, const std::string&)> callback) {
        std::lock_guard<std::mutex> lock(model_mutex);
        resource_callback = std::move(callback);
        LOG_ModelInfo("Resource callback registered for model: " + model_id);
    }
    
    // Performance monitoring
    void LogStatistics() const {
        if (token_cache) {
            token_cache->LogStatistics();
        }
        
        LOG_ModelInfo("=== Model Statistics for " + model_id + " ===");
        LOG_ModelInfo("State: " + StateToString(state));
        LOG_ModelInfo("Config: " + config.model_path);
        LOG_ModelInfo("Context Size: " + std::to_string(config.context_size));
        LOG_ModelInfo("GPU Layers: " + std::to_string(config.gpu_layers));
        LOG_ModelInfo("Estimated Memory: " + std::to_string(estimated_memory_usage / 1024 / 1024) + " MB");
    }
    
private:
    std::string StateToString(ModelState s) const {
        switch (s) {
            case ModelState::UNLOADED: return "UNLOADED";
            case ModelState::LOADING: return "LOADING";
            case ModelState::LOADED: return "LOADED";
            case ModelState::ERROR_STATE: return "ERROR";
            default: return "UNKNOWN";
        }
    }
};

// Helper function to convert ResourceEvent to string for logging
inline std::string ResourceEventToString(ResourceEvent event) {
    switch (event) {
        case ResourceEvent::MODEL_LOADING_STARTED: return "MODEL_LOADING_STARTED";
        case ResourceEvent::MODEL_LOADED: return "MODEL_LOADED";
        case ResourceEvent::MODEL_UNLOADED: return "MODEL_UNLOADED";
        case ResourceEvent::MEMORY_USAGE_HIGH: return "MEMORY_USAGE_HIGH";
        case ResourceEvent::TOKENIZATION_ERROR: return "TOKENIZATION_ERROR";
        case ResourceEvent::CLEANUP_COMPLETED: return "CLEANUP_COMPLETED";
        default: return "UNKNOWN_EVENT";
    }
}
