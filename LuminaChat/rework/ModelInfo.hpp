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
#include <thread>
#include <mutex>
#include <filesystem>
#include <system_error>
#include <unordered_map>

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
    int32_t gpu_layers = 999;  // 999 = full GPU offload when available
    int32_t batch_size = 512;
    int32_t threads = 0;      // 0 = auto-detect
    bool use_mmap = true;
    bool use_mlock = false;
    
    // Load from settings
    void LoadFromSettings(SettingsManager& settings, const std::string& model_prefix) {
        model_path = settings.GetString("Models", model_prefix + "_model_path", "");
        context_size = settings.GetInt("Models", model_prefix + "_context_size", 8192);
        gpu_layers = settings.GetInt("Models", model_prefix + "_gpu_layers", 999);
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
    bool owns_model = true; // Whether this instance owns the model and should free it
    
    // Thread safety
    mutable std::mutex model_mutex;
    
    // Optional resource monitoring callback
    // Orchestrator (higher) can register with ModelInfo (lower) for resource events
    std::function<void(const std::string&, ResourceEvent, const std::string&)> resource_callback;
    
    // Memory tracking
    std::atomic<size_t> estimated_memory_usage{0};
    
    // Static tracking of loaded model files to prevent duplicate loading
    static std::unordered_map<std::string, ModelInfo*> loaded_model_files;
    static std::mutex loaded_files_mutex;
    
    // Helper methods
    void NotifyResourceEvent(ResourceEvent event, const std::string& message = "") const {
        if (resource_callback) {
            // CRITICAL FIX: Run callbacks synchronously during model loading to prevent deadlock
            // Creating detached threads while llama.cpp is loading models can cause resource
            // conflicts on Windows, especially on the very first model load
            try {
                resource_callback(model_id, event, message);
            } catch (...) {
                // Ignore callback exceptions to prevent propagation
                LOG_ERROR_ModelInfo("Exception in resource callback for model: " + model_id);
            }
        }
    }
    
    void UpdateMemoryUsage() {
        if (model) {
            // Rough estimate based on model parameters
            estimated_memory_usage = llama_model_size(model);
        }
    }
    
    // Check if the same model file is already loaded
    static ModelInfo* FindExistingModelByPath(const std::string& model_path) {
        std::lock_guard<std::mutex> lock(loaded_files_mutex);
        auto it = loaded_model_files.find(model_path);
        if (it != loaded_model_files.end()) {
            // Verify the ModelInfo is still valid and loaded
            ModelInfo* existing_model = it->second;
            if (existing_model && existing_model->IsLoaded()) {
                return existing_model;
            } else {
                // Clean up invalid entry
                loaded_model_files.erase(it);
            }
        }
        return nullptr;
    }
    
    // Register this model instance for the given file path
    static void RegisterModelForPath(const std::string& model_path, ModelInfo* model_info) {
        std::lock_guard<std::mutex> lock(loaded_files_mutex);
        loaded_model_files[model_path] = model_info;
    }
    
    // Unregister this model instance from the given file path
    static void UnregisterModelForPath(const std::string& model_path) {
        std::lock_guard<std::mutex> lock(loaded_files_mutex);
        auto it = loaded_model_files.find(model_path);
        if (it != loaded_model_files.end()) {
            loaded_model_files.erase(it);
        }
    }
    
public:
    // Constructor
    explicit ModelInfo(const std::string& id) 
        : model_id(id)
        , token_cache(std::make_unique<TokenCache>(1024, nullptr)) // Default cache size, vocab set after model load
    {
        // NOTE: TokenCache is initialized with null vocab - vocab will be set via SetVocab() 
        // after the model is successfully loaded. This ensures all tokenization operations
        // use the correct vocabulary without requiring redundant vocab parameter passing.
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
    
    // Delete move constructor and assignment - mutex is not movable
    ModelInfo(ModelInfo&&) = delete;
    ModelInfo& operator=(ModelInfo&&) = delete;
    
    // Model loading and management
    bool LoadModel(const ModelConfig& model_config) {
        {
            std::lock_guard<std::mutex> lock(model_mutex);
            
            if (state == ModelState::LOADED) {
                LOG_ModelInfo("Model " + model_id + " already loaded");
                return true;
            }
            
            if (state == ModelState::LOADING) {
                LOG_ERROR_ModelInfo("Model " + model_id + " is already being loaded");
                return false;
            }
            
            // Reset error state if we're trying again
            if (state == ModelState::ERROR_STATE) {
                LOG_ModelInfo("Resetting error state for model " + model_id + " before retry");
                state = ModelState::UNLOADED;
            }
            
            LOG_ModelInfo("Loading model: " + model_config.model_path);
            
            if (!std::filesystem::exists(model_config.model_path)) {
                LOG_ERROR_ModelInfo("Model file does not exist: " + model_config.model_path);
                state = ModelState::ERROR_STATE;
                return false;
            }
            
            // Check if the same model file is already loaded by another ModelInfo instance
            if (auto existing_model = FindExistingModelByPath(model_config.model_path)) {
                if (existing_model != this) {
                    LOG_ModelInfo("Model file " + model_config.model_path + " is already loaded by model " + 
                                 existing_model->GetModelId() + ", sharing the model instance");
                    
                    // Share the model resources from the existing ModelInfo
                    model = existing_model->GetModel();
                    temp_context = nullptr; // We'll create our own tokenization context if needed
                    config = model_config;
                    state = ModelState::LOADED;
                    owns_model = false; // We don't own this model, so don't free it
                    UpdateMemoryUsage();
                    
                    // Set vocab on our TokenCache from the shared model
                    if (model && token_cache) {
                        token_cache->SetVocab(llama_model_get_vocab(model));
                    }
                    
                    // Note: We don't register this path again since the existing model already has it registered
                    NotifyResourceEvent(ResourceEvent::MODEL_LOADED, "Shared model from existing instance: " + model_config.model_path);
                    return true;
                }
            }
            
            state = ModelState::LOADING;
        }
        
        // Notify outside of lock to prevent deadlock
        NotifyResourceEvent(ResourceEvent::MODEL_LOADING_STARTED, "Loading " + model_config.model_path);
        
        // Temporary variables to hold the loaded resources
        llama_model* temp_model = nullptr;
        llama_context* temp_temp_context = nullptr;
        
        try {
            // Set up model parameters
            llama_model_params model_params = llama_model_default_params();
            model_params.n_gpu_layers = model_config.gpu_layers;
            model_params.use_mmap = model_config.use_mmap;
            model_params.use_mlock = model_config.use_mlock;
            
            // Load the model
            temp_model = llama_model_load_from_file(model_config.model_path.c_str(), model_params);
            
            if (!temp_model) {
                LOG_ERROR_ModelInfo("Failed to load model from: " + model_config.model_path);
                {
                    std::lock_guard<std::mutex> lock(model_mutex);
                    state = ModelState::ERROR_STATE;
                }
                return false;
            }
            
            // Create temporary context for tokenization
            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx = 512; // Small context just for tokenization
            ctx_params.n_batch = 512;
            ctx_params.n_threads = std::max(1u, std::thread::hardware_concurrency());
            
            temp_temp_context = llama_init_from_model(temp_model, ctx_params);
            
            if (!temp_temp_context) {
                LOG_ERROR_ModelInfo("Failed to create tokenization context");
                llama_model_free(temp_model);
                {
                    std::lock_guard<std::mutex> lock(model_mutex);
                    state = ModelState::ERROR_STATE;
                }
                return false;
            }
            
            // Atomically assign the loaded resources and update state
            {
                std::lock_guard<std::mutex> lock(model_mutex);
                model = temp_model;
                temp_context = temp_temp_context;
                config = model_config;
                state = ModelState::LOADED;
                UpdateMemoryUsage();
                
                // Set vocab on TokenCache now that model is loaded
                if (token_cache) {
                    token_cache->SetVocab(llama_model_get_vocab(model));
                }
            }
            
            LOG_ModelInfo("Model loaded successfully: " + model_config.model_path + 
                         " (GPU layers: " + std::to_string(model_config.gpu_layers) + ")");
            
            // Register this model instance for the file path to prevent duplicate loading
            RegisterModelForPath(model_config.model_path, this);
            
            // Notify outside of lock to prevent deadlock
            NotifyResourceEvent(ResourceEvent::MODEL_LOADED, "Successfully loaded: " + model_config.model_path);
            
            return true;
            
        } catch (const std::system_error& e) {
            LOG_ERROR_ModelInfo("System error during model loading: " + std::string(e.what()) + 
                               " (error code: " + std::to_string(e.code().value()) + ")");
            {
                std::lock_guard<std::mutex> lock(model_mutex);
                state = ModelState::ERROR_STATE;
            }
            if (temp_temp_context) {
                llama_free(temp_temp_context);
            }
            if (temp_model) {
                llama_model_free(temp_model);
            }
            // Re-throw system errors to be handled by caller
            throw;
        } catch (const std::exception& e) {
            LOG_ERROR_ModelInfo("Exception during model loading: " + std::string(e.what()));
            {
                std::lock_guard<std::mutex> lock(model_mutex);
                state = ModelState::ERROR_STATE;
            }
            if (temp_temp_context) {
                llama_free(temp_temp_context);
            }
            if (temp_model) {
                llama_model_free(temp_model);
            }
            return false;
        } catch (...) {
            LOG_ERROR_ModelInfo("Unknown exception during model loading");
            {
                std::lock_guard<std::mutex> lock(model_mutex);
                state = ModelState::ERROR_STATE;
            }
            if (temp_temp_context) {
                llama_free(temp_temp_context);
            }
            if (temp_model) {
                llama_model_free(temp_model);
            }
            return false;
        }
    }
    
    // Cleanup resources
    void Cleanup() {
        std::string model_path_to_unregister;
        {
            std::lock_guard<std::mutex> lock(model_mutex);
            
            // Save the model path for unregistration
            if (state == ModelState::LOADED && !config.model_path.empty()) {
                model_path_to_unregister = config.model_path;
            }
            
            if (temp_context) {
                llama_free(temp_context);
                temp_context = nullptr;
                LOG_ModelInfo("Temp context freed for: " + model_id);
            }
            
            if (model && owns_model) {
                llama_model_free(model);
                model = nullptr;
                LOG_ModelInfo("Model freed for: " + model_id);
            } else if (model && !owns_model) {
                model = nullptr; // Just clear the pointer, don't free
                LOG_ModelInfo("Shared model pointer cleared for: " + model_id);
            }
            
            state = ModelState::UNLOADED;
            estimated_memory_usage = 0;
            
            if (token_cache) {
                token_cache->ClearAll();
                token_cache->SetVocab(nullptr); // Clear vocab reference when model is unloaded
                LOG_ModelInfo("Token cache cleared for: " + model_id);
            }
        }
        
        // Unregister model path outside of the model_mutex lock (only if we owned the model)
        if (!model_path_to_unregister.empty() && owns_model) {
            UnregisterModelForPath(model_path_to_unregister);
        }
        
        // Don't notify during cleanup to prevent use-after-free in destructor
        // NotifyResourceEvent(ResourceEvent::CLEANUP_COMPLETED, "Model resources cleaned up");
    }
    
    // Accessors
    TokenCache& GetTokenCache() { return *token_cache; }
    const TokenCache& GetTokenCache() const { return *token_cache; }
    
    ModelState GetState() const { return state; }
    bool IsLoaded() const { return state == ModelState::LOADED; }
    const std::string& GetModelId() const { return model_id; }
    const ModelConfig& GetConfig() const { return config; }
    
    llama_model* GetModel() const { 
        std::lock_guard<std::mutex> lock(model_mutex);
        return (state == ModelState::LOADED) ? model : nullptr;
    }
    
    // Get vocab from model for tokenization functions (thread-safe)
    const llama_vocab* GetVocab() const {
        std::lock_guard<std::mutex> lock(model_mutex);
        return (state == ModelState::LOADED && model) ? llama_model_get_vocab(model) : nullptr;
    }
    
    size_t GetMemoryUsage() const { return estimated_memory_usage; }
    
    // Callback registration for resource monitoring
    void RegisterResourceCallback(std::function<void(const std::string&, ResourceEvent, const std::string&)> callback) {
        resource_callback = std::move(callback);
    }
    
    // Model path registration for duplicate prevention
    static void RegisterModelInstance(const std::string& model_path, ModelInfo* model_info) {
        RegisterModelForPath(model_path, model_info);
    }
    
    static void UnregisterModelInstance(const std::string& model_path) {
        UnregisterModelForPath(model_path);
    }
    
    static ModelInfo* FindExistingModel(const std::string& model_path) {
        return FindExistingModelByPath(model_path);
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

// Static member definitions
inline std::unordered_map<std::string, ModelInfo*> ModelInfo::loaded_model_files;
inline std::mutex ModelInfo::loaded_files_mutex;
