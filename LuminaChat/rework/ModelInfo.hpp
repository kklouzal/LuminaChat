#pragma once

#include "TokenCache.hpp"
#include "SettingsManager.hpp"
#include "Logger.hpp"
// llama.cpp includes
#include "llama.h"
#include <memory>
#include <string>
#include <functional>
#include <atomic>
#include <mutex>
#include <filesystem>
#include <thread>

// ModelInfo: Model resource management and vocabulary access
// - Owns TokenCache instance for this model
// - Manages llama vocabulary for tokenization  
// - Handles model loading/unloading lifecycle
// - Settings integration for model paths and parameters
// - Optional resource monitoring callbacks

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
    
    // Load from settings
    void LoadFromSettings(SettingsManager& settings, const std::string& model_prefix) {
        model_path = settings.GetString("Models", model_prefix + "_model_path", "");
        context_size = settings.GetInt("Models", model_prefix + "_context_size", 8192);
        gpu_layers = settings.GetInt("Models", model_prefix + "_gpu_layers", -1);
        batch_size = settings.GetInt("Models", model_prefix + "_batch_size", 512);
        threads = settings.GetInt("Models", model_prefix + "_threads", 0);
        use_mmap = settings.GetBool("Models", model_prefix + "_use_mmap", true);
        use_mlock = settings.GetBool("Models", model_prefix + "_use_mlock", false);
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
    }
      
    bool IsValid() const {
        return !model_path.empty() && context_size > 0;
    }
};

class ModelInfo {
private:
    // Core components
    std::unique_ptr<TokenCache> token_cache;
    ModelConfig config;
    std::string model_id;
    
    // Model state
    std::atomic<ModelState> state{ModelState::UNLOADED};
    
    // llama.cpp objects
    llama_model* model = nullptr;
    llama_context* temp_context = nullptr; // For tokenization operations only
    
    // Thread safety
    mutable std::mutex model_mutex;
    
    // Optional resource monitoring callback
    // Orchestrator (higher) can register with ModelInfo (lower) for resource events
    std::function<void(const std::string&, ResourceEvent, const std::string&)> resource_callback;
    
    // Memory tracking
    std::atomic<size_t> estimated_memory_usage{0};
    
    // Helper methods
    void NotifyResourceEvent(ResourceEvent event, const std::string& message = "") const {
        if (resource_callback) {
            resource_callback(model_id, event, message);
        }
    }
    
    void UpdateMemoryUsage() {
        if (model) {
            // Rough estimate based on model parameters
            estimated_memory_usage = llama_model_size(model);
        }
    }
    
public:
    // Constructor
    explicit ModelInfo(const std::string& id) 
        : model_id(id)
        , token_cache(std::make_unique<TokenCache>(1024)) // Default cache size
    {
        LOG_ModelInfo("ModelInfo created for: " + model_id);
    }
    
    // Destructor - ensures proper cleanup
    ~ModelInfo() {
        LOG_ModelInfo("ModelInfo destructor called for: " + model_id);
        Cleanup();
    }
    
    // Delete copy constructor and assignment operator (RAII)
    ModelInfo(const ModelInfo&) = delete;
    ModelInfo& operator=(const ModelInfo&) = delete;
    
    // Move constructor and assignment
    ModelInfo(ModelInfo&&) = default;
    ModelInfo& operator=(ModelInfo&&) = default;
    
    // Model loading and management
    bool LoadModel(const ModelConfig& model_config) {
        std::lock_guard<std::mutex> lock(model_mutex);
        
        if (state == ModelState::LOADED) {
            LOG_ModelInfo("Model " + model_id + " already loaded");
            return true;
        }
        
        if (state == ModelState::LOADING) {
            LOG_ERROR_ModelInfo("Model " + model_id + " is already being loaded");
            return false;
        }
        
        LOG_ModelInfo("Loading model: " + model_config.model_path);
        
        if (!std::filesystem::exists(model_config.model_path)) {
            LOG_ERROR_ModelInfo("Model file does not exist: " + model_config.model_path);
            state = ModelState::ERROR_STATE;
            return false;
        }
        
        state = ModelState::LOADING;
        NotifyResourceEvent(ResourceEvent::MODEL_LOADING_STARTED, "Loading " + model_config.model_path);
        
        try {
            // Set up model parameters
            llama_model_params model_params = llama_model_default_params();
            model_params.n_gpu_layers = model_config.gpu_layers;
            model_params.use_mmap = model_config.use_mmap;
            model_params.use_mlock = model_config.use_mlock;
            
            // Load the model
            model = llama_model_load_from_file(model_config.model_path.c_str(), model_params);
            if (!model) {
                LOG_ERROR_ModelInfo("Failed to load model from: " + model_config.model_path);
                state = ModelState::ERROR_STATE;
                return false;
            }
            
            // Create temporary context for tokenization
            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx = 512; // Small context just for tokenization
            ctx_params.n_batch = 512;
            ctx_params.n_threads = std::max(1u, std::thread::hardware_concurrency());
            
            temp_context = llama_init_from_model(model, ctx_params);
            if (!temp_context) {
                LOG_ERROR_ModelInfo("Failed to create tokenization context");
                llama_model_free(model);
                model = nullptr;
                state = ModelState::ERROR_STATE;
                return false;
            }
            
            // Store configuration
            config = model_config;
            state = ModelState::LOADED;
            
            UpdateMemoryUsage();
            NotifyResourceEvent(ResourceEvent::MODEL_LOADED, "Successfully loaded: " + model_config.model_path);
            
            LOG_ModelInfo("Model loaded successfully: " + model_config.model_path + 
                         " (GPU layers: " + std::to_string(model_config.gpu_layers) + ")");
            
            return true;
            
        } catch (const std::exception& e) {
            LOG_ERROR_ModelInfo("Exception during model loading: " + std::string(e.what()));
            state = ModelState::ERROR_STATE;
            if (temp_context) {
                llama_free(temp_context);
                temp_context = nullptr;
            }
            if (model) {
                llama_model_free(model);
                model = nullptr;
            }
            return false;
        }
    }
    
    // Cleanup resources
    void Cleanup() {
        std::lock_guard<std::mutex> lock(model_mutex);
        
        if (temp_context) {
            llama_free(temp_context);
            temp_context = nullptr;
            LOG_ModelInfo("Temp context freed for: " + model_id);
        }
        
        if (model) {
            llama_model_free(model);
            model = nullptr;
            LOG_ModelInfo("Model freed for: " + model_id);
        }
        
        state = ModelState::UNLOADED;
        estimated_memory_usage = 0;
        
        if (token_cache) {
            token_cache->ClearAll();
            LOG_ModelInfo("Token cache cleared for: " + model_id);
        }
        
        NotifyResourceEvent(ResourceEvent::CLEANUP_COMPLETED, "Model resources cleaned up");
    }
    
    // Tokenization methods
    std::vector<llama_token> TokenizeText(const std::string& text, bool add_special = true) {
        std::lock_guard<std::mutex> lock(model_mutex);
        
        if (state != ModelState::LOADED || !model) {
            LOG_ERROR_ModelInfo("Model not loaded for tokenization");
            NotifyResourceEvent(ResourceEvent::TOKENIZATION_ERROR, "Model not loaded");
            return {};
        }
        
        if (text.empty()) {
            return {};
        }
        
        // Check cache first
        std::string cache_key = text + (add_special ? ":s" : ":n");
        std::vector<int32_t> cached_tokens;
        if (token_cache->GetTokens(cache_key, cached_tokens)) {
            // Convert int32_t to llama_token
            std::vector<llama_token> result;
            result.reserve(cached_tokens.size());
            for (int32_t token : cached_tokens) {
                result.push_back(static_cast<llama_token>(token));
            }
            return result;
        }
        
        try {
            // Get required buffer size
            const int32_t n_tokens_required = -llama_tokenize(GetVocab(), text.c_str(), text.size(), nullptr, 0, add_special, true);
            if (n_tokens_required <= 0) {
                return {};
            }
            
            // Tokenize
            std::vector<llama_token> tokens(n_tokens_required);
            const int32_t n_tokens_actual = llama_tokenize(GetVocab(), text.c_str(), text.size(),
                                                          tokens.data(), tokens.size(), add_special, true);
            
            if (n_tokens_actual < 0 || n_tokens_actual != n_tokens_required) {
                LOG_ERROR_ModelInfo("Tokenization failed - expected: " + std::to_string(n_tokens_required) + 
                                   ", got: " + std::to_string(n_tokens_actual));
                NotifyResourceEvent(ResourceEvent::TOKENIZATION_ERROR, "Token count mismatch");
                return {};
            }
            
            // Cache result - convert llama_token to int32_t for storage
            std::vector<int32_t> tokens_for_cache;
            tokens_for_cache.reserve(tokens.size());
            for (llama_token token : tokens) {
                tokens_for_cache.push_back(static_cast<int32_t>(token));
            }
            token_cache->StoreTokens(cache_key, tokens_for_cache);
            
            return tokens;
            
        } catch (const std::exception& e) {
            LOG_ERROR_ModelInfo("Exception during tokenization: " + std::string(e.what()));
            NotifyResourceEvent(ResourceEvent::TOKENIZATION_ERROR, e.what());
            return {};
        }
    }
    
    std::string DetokenizeTokens(const std::vector<llama_token>& tokens) {
        std::lock_guard<std::mutex> lock(model_mutex);
        
        if (state != ModelState::LOADED || !model) {
            LOG_ERROR_ModelInfo("Model not loaded for detokenization");
            return "";
        }
        
        if (tokens.empty()) {
            return "";
        }
        
        // Check cache first - convert llama_token to int32_t for hash lookup
        std::vector<int32_t> tokens_for_cache;
        tokens_for_cache.reserve(tokens.size());
        for (llama_token token : tokens) {
            tokens_for_cache.push_back(static_cast<int32_t>(token));
        }
        
        std::string cached_text;
        if (token_cache->GetText(tokens_for_cache, cached_text)) {
            return cached_text;
        }
        
        try {
            std::string result;
            result.reserve(tokens.size() * 4); // Rough estimate
            
            for (const auto& token : tokens) {
                std::vector<char> buffer(32);
                
                int32_t result_length = llama_token_to_piece(GetVocab(), token, buffer.data(), buffer.size(), 0, true);
                
                if (result_length < 0) {
                    // Buffer too small, resize and retry
                    buffer.resize(-result_length);
                    result_length = llama_token_to_piece(GetVocab(), token, buffer.data(), buffer.size(), 0, true);
                }
                
                if (result_length > 0) {
                    result.append(buffer.data(), result_length);
                }
            }
            
            // Cache result
            token_cache->StoreText(tokens_for_cache, result);
            
            return result;
            
        } catch (const std::exception& e) {
            LOG_ERROR_ModelInfo("Exception during detokenization: " + std::string(e.what()));
            return "";
        }
    }
    
    // Accessors
    TokenCache& GetTokenCache() { return *token_cache; }
    const TokenCache& GetTokenCache() const { return *token_cache; }
    
    ModelState GetState() const { return state; }
    bool IsLoaded() const { return state == ModelState::LOADED; }
    const std::string& GetModelId() const { return model_id; }
    const ModelConfig& GetConfig() const { return config; }
    
    llama_model* GetModel() const { return model; } // For direct access when needed
    
    // Get vocab from model for tokenization functions
    const llama_vocab* GetVocab() const {
        
        return model ? llama_model_get_vocab(model) : nullptr;
    }
    
    size_t GetMemoryUsage() const { return estimated_memory_usage; }
    
    // Callback registration for resource monitoring
    void RegisterResourceCallback(std::function<void(const std::string&, ResourceEvent, const std::string&)> callback) {
        resource_callback = std::move(callback);
    }
    
    // Statistics
    size_t GetTokenCacheHits() const {
        return token_cache ? token_cache->GetStats().GetTotalHits() : 0;
    }
    
    size_t GetTokenCacheRequests() const {
        return token_cache ? token_cache->GetStats().GetTotalRequests() : 0;
    }
    
    float GetTokenCacheHitRatio() const {
        if (!token_cache) return 0.0f;
        auto stats = token_cache->GetStats();
        size_t total = stats.GetTotalRequests();
        return total > 0 ? static_cast<float>(stats.GetTotalHits()) / total : 0.0f;
    }
};
