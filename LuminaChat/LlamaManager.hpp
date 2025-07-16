#pragma once

#include "ModelInfo.hpp"
#include "Context/ContextInfo.hpp"
#include "SettingsManager.hpp"
#include "Logger.hpp"
#include "ErrorHandling.hpp"
// llama.cpp includes
#include "llama.h"
#include <memory>
#include <unordered_map>
#include <string>
#include <mutex>
#include <thread>
#include <vector>
#include <functional>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <algorithm>

// LlamaManager: Central coordinator for all models and contexts
//
// Resource Management:
// - Container for all ModelInfo instances
// - Container for all ContextInfo instances  
// - Lifecycle management for entire system
// - Settings Integration: Model loading and template configuration from SettingsManager
//
// Key Features:
// - Thread-safe model and context management
// - Automatic cleanup and resource management
// - Template integration via SettingsManager
// - Performance monitoring and statistics
// - Plugin-ready architecture for future extensions

// Forward declarations
class ModelInfo;
class ContextInfo;
class SettingsManager;

enum class LlamaManagerState {
    UNINITIALIZED,
    INITIALIZING,
    READY,
    SHUTTING_DOWN,
    ERROR_STATE
};

struct LlamaManagerStats {
    size_t total_models = 0;
    size_t loaded_models = 0;
    size_t total_contexts = 0;
    size_t active_contexts = 0;
    size_t total_memory_usage = 0; // In bytes
    std::chrono::steady_clock::time_point startup_time;
    
    void Reset() {
        total_models = 0;
        loaded_models = 0;
        total_contexts = 0;
        active_contexts = 0;
        total_memory_usage = 0;
        startup_time = std::chrono::steady_clock::now();
    }
    
    double GetUptimeSeconds() const {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - startup_time);
        return static_cast<double>(duration.count());
    }
};

class LlamaManager {
private:
    // State management
    LlamaManagerState state = LlamaManagerState::UNINITIALIZED;
    mutable std::mutex manager_mutex;
    
    // Resource containers
    std::unordered_map<std::string, std::unique_ptr<ModelInfo>> models;
    std::unordered_map<std::string, std::unique_ptr<ContextInfo>> contexts;
    
    // Settings integration
    SettingsManager* settings_manager = nullptr;
    
    // Statistics and monitoring
    LlamaManagerStats stats;
    
    // Resource monitoring
    std::function<void(std::string, std::string)> resource_callback;
    
    // Optional progress callback for UI updates
    std::function<void(const std::string&, float)> progress_callback;

    // llama.cpp initialization state
    static std::atomic<bool> llama_backend_initialized;
    static std::mutex backend_init_mutex;
    
    // Helper methods
    void UpdateStats();
    void NotifyResourceEvent(const std::string& event, const std::string& details);
    bool ValidateModelId(const std::string& model_id) const;
    bool ValidateContextId(const std::string& context_id) const;
    bool EnsureBackendInitialized();
    
public:
    // Constructors
    LlamaManager() : LlamaManager(nullptr) {}
    explicit LlamaManager(SettingsManager* settings);
    
    // Destructor - ensures proper cleanup
    ~LlamaManager();
    
    // Delete copy constructor and assignment operator (RAII)
    LlamaManager(const LlamaManager&) = delete;
    LlamaManager& operator=(const LlamaManager&) = delete;
    
    // Move constructor and assignment
    LlamaManager(LlamaManager&&) = default;
    LlamaManager& operator=(LlamaManager&&) = default;
    
    // Initialization and lifecycle
    bool Initialize();
    void Cleanup();
    
    // Model management
    ModelInfo* GetOrCreateModelInfo(const std::string& model_id);
    bool LoadModel(const std::string& model_id, const ModelConfig& config);
    bool LoadModelFromSettings(const std::string& model_id, const std::string& settings_prefix);
    bool UnloadModel(const std::string& model_id);
    ModelInfo* GetModelInfo(const std::string& model_id) const;
    std::vector<std::string> GetLoadedModelIds() const;
    
    // Context management - updated to include per-context context_size parameter (architectural fix)
    ContextInfo* GetOrCreateContextInfo(const std::string& context_id, const std::string& model_id, int32_t context_size);
    bool CreateContextWithTemplate(const std::string& context_id, const std::string& model_id, const std::string& template_content, int32_t context_size);
    bool RemoveContext(const std::string& context_id);
    ContextInfo* GetContextInfo(const std::string& context_id) const;
    std::vector<std::string> GetContextIds() const;
    std::vector<ContextInfo*> GetAllActiveContexts() const; // For context size monitoring
    
    // State and statistics
    LlamaManagerState GetState() const { return state; }
    const LlamaManagerStats& GetStats() const { return stats; }
    bool IsReady() const { return state == LlamaManagerState::READY; }
    
    // Settings integration
    void SetSettingsManager(SettingsManager* settings);
    SettingsManager* GetSettingsManager() { return settings_manager; }
    const SettingsManager* GetSettingsManager() const { return settings_manager; }
    bool SaveStatsToSettings() const;
    
    // Resource monitoring
    void RegisterResourceCallback(std::function<void(std::string, std::string)> callback);
    size_t GetTotalMemoryUsage() const;
    std::vector<std::string> GetResourceReport() const;
    
    // Advanced features
    bool OptimizeMemoryUsage();
    void ForceGarbageCollection();
    bool ValidateAllResources() const;
    
    // Debugging and diagnostics
    void DumpManagerState() const;
    std::string GetDetailedReport() const;
    
    // Progress callback registration for model loading UI updates
    void RegisterProgressCallback(std::function<void(const std::string&, float)> callback);
    
    // Clear progress callback
    void ClearProgressCallback();

    // Internal progress callback for forwarding to UI
    bool InternalProgressCallback(const std::string& model_id, float progress);
};

// Helper functions for LlamaManager integration
namespace LlamaManagerHelpers {
    // Generate unique IDs
    std::string GenerateModelId(const std::string& base_name = "model");
    std::string GenerateContextId(const std::string& base_name = "context");
    
    // Resource management utilities
    size_t EstimateModelMemoryUsage(const ModelConfig& config);
    size_t EstimateContextMemoryUsage(const ContextInfo& context);
    
    // Configuration helpers
    bool ValidateModelConfig(const ModelConfig& config);
    ModelConfig CreateDefaultModelConfig();
    
    // Debug helpers
    std::string FormatStats(const LlamaManagerStats& stats);
    std::string FormatModelList(const std::vector<std::string>& model_ids);
    std::string FormatContextList(const std::vector<std::string>& context_ids);
}

// Implementation details
#include <algorithm>
#include <chrono>
#include <sstream>
#include <filesystem>

// Static initialization
std::atomic<bool> LlamaManager::llama_backend_initialized{false};
std::mutex LlamaManager::backend_init_mutex;

// Constructor
inline LlamaManager::LlamaManager(SettingsManager* settings) 
    : settings_manager(settings) {
    LOG_LlamaManager("LlamaManager constructor called");
    stats.Reset();
    state = LlamaManagerState::UNINITIALIZED;
}

// Destructor
inline LlamaManager::~LlamaManager() {
    LOG_LlamaManager("LlamaManager destructor called - performing cleanup");
    try {
        Cleanup();
    } catch (const std::exception& e) {
        // Log the exception but don't let it escape the destructor
        try {
            LOG_ERROR_LlamaManager("Exception during LlamaManager cleanup: " + std::string(e.what()));
        } catch (...) {
            // If logging also fails, there's nothing more we can safely do
        }
    } catch (...) {
        // Catch any non-standard exceptions
        try {
            LOG_ERROR_LlamaManager("Unknown exception during LlamaManager cleanup");
        } catch (...) {
            // If logging fails, silently continue
        }
    }
}

// Initialize the backend and prepare for operation
inline bool LlamaManager::Initialize() {
    std::lock_guard<std::mutex> lock(manager_mutex);
    
    if (state == LlamaManagerState::READY) {
        LOG_LlamaManager("LlamaManager already initialized");
        return true;
    }
    
    LOG_LlamaManager("Initializing LlamaManager");
    state = LlamaManagerState::INITIALIZING;
    
    try {
        // Initialize llama.cpp backend
        if (!EnsureBackendInitialized()) {
            LOG_ERROR_LlamaManager("Failed to initialize llama.cpp backend");
            state = LlamaManagerState::ERROR_STATE;
            return false;
        }
        
        state = LlamaManagerState::READY;
        stats.startup_time = std::chrono::steady_clock::now();
        
        LOG_LlamaManager("LlamaManager initialized successfully");
        NotifyResourceEvent("MANAGER_INITIALIZED", "Ready for operation");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR_LlamaManager("Exception during initialization: " + std::string(e.what()));
        state = LlamaManagerState::ERROR_STATE;
        return false;
    }
}

// Ensure llama.cpp backend is initialized (thread-safe)
inline bool LlamaManager::EnsureBackendInitialized() {
    // Double-checked locking pattern for thread-safe initialization
    if (llama_backend_initialized.load()) {
        return true;
    }
    
    std::lock_guard<std::mutex> lock(backend_init_mutex);
    
    // Check again after acquiring lock
    if (llama_backend_initialized.load()) {
        return true;
    }
    
    LOG_LlamaManager("Initializing llama.cpp backend");
    
    try {
        ggml_backend_load_all();
        llama_backend_initialized.store(true);
        
        LOG_LlamaManager("llama.cpp backend initialized successfully");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR_LlamaManager("Exception in ggml_backend_load_all: " + std::string(e.what()));
        return false;
    }
}

// Model management methods
inline ModelInfo* LlamaManager::GetOrCreateModelInfo(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(manager_mutex);
    
    if (!ValidateModelId(model_id)) {
        LOG_ERROR_LlamaManager("Invalid model ID: " + model_id);
        return nullptr;
    }
    
    auto it = models.find(model_id);
    if (it != models.end()) {
        LOG_DEBUG_LlamaManager("Returning existing ModelInfo for: " + model_id);
        return it->second.get();
    }
    
    LOG_LlamaManager("Creating new ModelInfo for: " + model_id);
    
    try {
        auto model_info = std::make_unique<ModelInfo>(model_id);
        
        // Register resource callback if we have one
        if (resource_callback) {
            model_info->RegisterResourceCallback([this](const std::string& id, ResourceEvent event, const std::string& msg) {
                // Capture values to avoid potential reference issues in async callback
                std::string event_name = "MODEL_" + std::to_string(static_cast<int>(event));
                std::string message = "Model " + id + ": " + msg;
                NotifyResourceEvent(event_name, message);
            });
        }
        
        // Register progress callback if we have one
        if (progress_callback) {
            model_info->RegisterProgressCallback([this, model_id](float progress, void* user_data) -> bool {
                return InternalProgressCallback(model_id, progress);
            });
        }
        
        ModelInfo* model_ptr = model_info.get();
        models[model_id] = std::move(model_info);
        
        UpdateStats();
        LOG_LlamaManager("Created ModelInfo for: " + model_id);
        return model_ptr;
        
    } catch (const std::exception& e) {
        LOG_ERROR_LlamaManager("Failed to create ModelInfo for " + model_id + ": " + std::string(e.what()));
        return nullptr;
    }
}

inline bool LlamaManager::LoadModel(const std::string& model_id, const ModelConfig& config) {
    if (state != LlamaManagerState::READY) {
        LOG_ERROR_LlamaManager("LlamaManager not ready for model loading");
        return false;
    }
    
    if (!config.IsValid()) {
        LOG_ERROR_LlamaManager("Invalid model configuration for: " + model_id);
        return false;
    }
    
    // Check if the same model file is already loaded by another model ID
    {
        std::lock_guard<std::mutex> lock(manager_mutex);
        for (const auto& [existing_id, existing_model] : models) {
            if (existing_model && existing_model->IsLoaded() && 
                existing_model->GetConfig().model_path == config.model_path) {
                
                LOG_LlamaManager("Model file " + config.model_path + " is already loaded as model ID: " + existing_id + 
                               ", creating reference for: " + model_id);
                
                // Create a new ModelInfo that shares the loaded model
                auto shared_model_info = std::make_unique<ModelInfo>(model_id);
                if (shared_model_info->LoadModel(config)) {
                    models[model_id] = std::move(shared_model_info);
                    UpdateStats();
                    NotifyResourceEvent("MODEL_LOADED", "Successfully loaded (shared): " + model_id);
                    LOG_LlamaManager("Model loaded successfully (shared): " + model_id);
                    return true;
                } else {
                    LOG_ERROR_LlamaManager("Failed to share loaded model for: " + model_id);
                    return false;
                }
            }
        }
    }
    
    ModelInfo* model_info = GetOrCreateModelInfo(model_id);
    if (!model_info) {
        LOG_ERROR_LlamaManager("Failed to get/create ModelInfo for: " + model_id);
        return false;
    }
    
    LOG_LlamaManager("Loading model: " + model_id + " from " + config.model_path);
    
    // Call LoadModel without holding manager_mutex to prevent callback deadlock
    if (model_info->LoadModel(config)) {
        // Acquire mutex only for UpdateStats which accesses containers
        {
            std::lock_guard<std::mutex> lock(manager_mutex);
            UpdateStats();
        }
        NotifyResourceEvent("MODEL_LOADED", "Successfully loaded: " + model_id);
        LOG_LlamaManager("Model loaded successfully: " + model_id);
        return true;
    } else {
        LOG_ERROR_LlamaManager("Failed to load model: " + model_id);
        return false;
    }
}

inline bool LlamaManager::LoadModelFromSettings(const std::string& model_id, const std::string& settings_prefix) {
    if (!settings_manager) {
        LOG_ERROR_LlamaManager("No settings manager available for loading model: " + model_id);
        return false;
    }
    
    LOG_LlamaManager("Loading model " + model_id + " from settings with prefix: " + settings_prefix);
    
    ModelConfig config;
    config.LoadFromSettings(*settings_manager, settings_prefix);
    
    if (!config.IsValid()) {
        LOG_ERROR_LlamaManager("Invalid model configuration in settings for: " + model_id);
        return false;
    }
    
    return LoadModel(model_id, config);
}

// Context management methods
inline ContextInfo* LlamaManager::GetOrCreateContextInfo(const std::string& context_id, const std::string& model_id, int32_t context_size) {
    std::lock_guard<std::mutex> lock(manager_mutex);
    
    if (!ValidateContextId(context_id) || !ValidateModelId(model_id)) {
        LOG_ERROR_LlamaManager("Invalid context ID (" + context_id + ") or model ID (" + model_id + ")");
        return nullptr;
    }
    
    auto it = contexts.find(context_id);
    if (it != contexts.end()) {
        LOG_DEBUG_LlamaManager("Returning existing ContextInfo for: " + context_id);
        return it->second.get();
    }
    
    // Get the model - Use internal method that doesn't acquire mutex again
    auto model_it = models.find(model_id);
    ModelInfo* model_info = (model_it != models.end()) ? model_it->second.get() : nullptr;
    if (!model_info) {
        LOG_ERROR_LlamaManager("Model " + model_id + " not found for context creation: " + context_id);
        return nullptr;
    }
    
    if (!model_info->IsLoaded()) {
        LOG_ERROR_LlamaManager("Model " + model_id + " not loaded for context creation: " + context_id);
        return nullptr;
    }
    
    LOG_LlamaManager("Creating new ContextInfo: " + context_id + " with model: " + model_id + " and context size: " + std::to_string(context_size));
    
    try {
        // Determine summary slot count based on context type
        size_t max_summary_slots = 5; // Default
        if (context_id.starts_with("inner_context")) {
            max_summary_slots = 16; // Inner contexts get more summary capacity (set to 16 as requested)
            LOG_LlamaManager("Using enhanced summary capacity (" + std::to_string(max_summary_slots) + " slots) for inner context: " + context_id);
        }
        
        // Create ContextInfo with specific context size and summary slot configuration
        auto context_info = std::make_unique<ContextInfo>(context_id, model_info, context_size, max_summary_slots);
        
        ContextInfo* context_ptr = context_info.get();
        contexts[context_id] = std::move(context_info);
        
        UpdateStats();
        LOG_LlamaManager("Created ContextInfo: " + context_id);
        return context_ptr;
        
    } catch (const std::exception& e) {
        LOG_ERROR_LlamaManager("Failed to create ContextInfo for " + context_id + ": " + std::string(e.what()));
        return nullptr;
    }
}

// Context management methods - continued
inline bool LlamaManager::RemoveContext(const std::string& context_id) {
    std::lock_guard<std::mutex> lock(manager_mutex);
    
    if (!ValidateContextId(context_id)) {
        LOG_ERROR_LlamaManager("Invalid context ID: " + context_id);
        return false;
    }
    
    auto it = contexts.find(context_id);
    if (it == contexts.end()) {
        LOG_WARNING_LlamaManager("Context not found for removal: " + context_id);
        return false;
    }
    
    LOG_LlamaManager("Removing context: " + context_id);
    
    try {
        // The ContextInfo destructor will handle cleanup of llama_ctx and batch
        contexts.erase(it);
        
        UpdateStats();
        LOG_LlamaManager("Successfully removed context: " + context_id);
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR_LlamaManager("Exception removing context " + context_id + ": " + std::string(e.what()));
        return false;
    }
}

// Cleanup
inline void LlamaManager::Cleanup() {
    std::lock_guard<std::mutex> lock(manager_mutex);
    
    LOG_LlamaManager("Cleanup called - cleaning up " + std::to_string(contexts.size()) + 
                    " contexts and " + std::to_string(models.size()) + " models");
    
    state = LlamaManagerState::SHUTTING_DOWN;
    
    try {
        // Clear contexts first (they depend on models)
        contexts.clear();
        LOG_LlamaManager("All contexts cleared");
        
        // Clear models
        models.clear();
        LOG_LlamaManager("All models cleared");
        
        stats.Reset();
        state = LlamaManagerState::UNINITIALIZED;
        
        LOG_LlamaManager("Cleanup completed successfully");
        
    } catch (const std::exception& e) {
        LOG_ERROR_LlamaManager("Exception during cleanup: " + std::string(e.what()));
        state = LlamaManagerState::ERROR_STATE;
    }
}

// Helper methods
inline void LlamaManager::UpdateStats() {
    // Note: This method should be called while holding manager_mutex
    // since it accesses the models and contexts containers
    stats.total_models = models.size();
    stats.total_contexts = contexts.size();
    
    stats.loaded_models = 0;
    stats.active_contexts = 0;
    stats.total_memory_usage = 0;
    
    for (const auto& [id, model] : models) {
        if (model && model->IsLoaded()) {
            stats.loaded_models++;
            stats.total_memory_usage += model->GetMemoryUsage();
        }
    }
    
    for (const auto& [id, context] : contexts) {
        if (context && context->GetState() == ContextState::CONTEXT_IDLE) {
            stats.active_contexts++;
        }
    }
}

inline void LlamaManager::NotifyResourceEvent(const std::string& event, const std::string& details) {
    if (resource_callback) {
        // CRITICAL FIX: Run callbacks synchronously during system initialization to prevent deadlock
        // Creating detached threads while the system is starting up can cause resource conflicts
        // on Windows, especially during the first model load
        try {
            resource_callback(event, details);
        } catch (...) {
            // Ignore callback exceptions to prevent propagation
            LOG_ERROR_LlamaManager("Exception in resource callback: " + event);
        }
    }
}

inline bool LlamaManager::ValidateModelId(const std::string& model_id) const {
    return !model_id.empty() && model_id.length() <= 100 && 
           std::all_of(model_id.begin(), model_id.end(), [](char c) {
               return std::isalnum(c) || c == '_' || c == '-' || c == '.';
           });
}

inline bool LlamaManager::ValidateContextId(const std::string& context_id) const {
    return !context_id.empty() && context_id.length() <= 100 && 
           std::all_of(context_id.begin(), context_id.end(), [](char c) {
               return std::isalnum(c) || c == '_' || c == '-' || c == '.';
           });
}

// Accessors
inline ModelInfo* LlamaManager::GetModelInfo(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(manager_mutex);
    auto it = models.find(model_id);
    return (it != models.end()) ? it->second.get() : nullptr;
}

inline ContextInfo* LlamaManager::GetContextInfo(const std::string& context_id) const {
    std::lock_guard<std::mutex> lock(manager_mutex);
    auto it = contexts.find(context_id);
    return (it != contexts.end()) ? it->second.get() : nullptr;
}

inline std::vector<std::string> LlamaManager::GetLoadedModelIds() const {
    std::lock_guard<std::mutex> lock(manager_mutex);
    std::vector<std::string> result;
    for (const auto& [id, model] : models) {
        if (model && model->IsLoaded()) {
            result.push_back(id);
        }
    }
    return result;
}

inline std::vector<std::string> LlamaManager::GetContextIds() const {
    std::lock_guard<std::mutex> lock(manager_mutex);
    std::vector<std::string> result;
    result.reserve(contexts.size());
    for (const auto& [id, context] : contexts) {
        result.push_back(id);
    }
    return result;
}

inline std::vector<ContextInfo*> LlamaManager::GetAllActiveContexts() const {
    std::lock_guard<std::mutex> lock(manager_mutex);
    std::vector<ContextInfo*> result;
    result.reserve(contexts.size());
    
    for (const auto& [id, context] : contexts) {
        if (context && context->GetState() != ContextState::ERROR_STATE) {
            result.push_back(context.get());
        }
    }
    return result;
}

inline void LlamaManager::RegisterResourceCallback(std::function<void(std::string, std::string)> callback) {
    std::lock_guard<std::mutex> lock(manager_mutex);
    resource_callback = std::move(callback);
    LOG_LlamaManager("Resource callback registered");
}

inline void LlamaManager::RegisterProgressCallback(std::function<void(const std::string&, float)> callback) {
    std::lock_guard<std::mutex> lock(manager_mutex);
    // Register the progress callback
    progress_callback = std::move(callback);
    LOG_LlamaManager("Progress callback registered for UI updates");
}

inline void LlamaManager::ClearProgressCallback() {
    std::lock_guard<std::mutex> lock(manager_mutex);
    // Clear the progress callback
    progress_callback = nullptr;
    LOG_LlamaManager("Progress callback cleared");
}

// Internal progress callback for forwarding to UI
inline bool LlamaManager::InternalProgressCallback(const std::string& model_id, float progress) {
    // Copy the callback to avoid holding lock while calling it
    std::function<void(const std::string&, float)> callback_copy;
    {
        std::lock_guard<std::mutex> lock(manager_mutex);
        callback_copy = progress_callback;
    }
    
    if (callback_copy) {
        try {
            callback_copy(model_id, progress);
            return true;
        } catch (const std::exception& e) {
            LOG_ERROR_LlamaManager("Exception in progress callback: " + std::string(e.what()));
        } catch (...) {
            LOG_ERROR_LlamaManager("Unknown exception in progress callback");
        }
    }
    return true; // Always return true to continue loading
}

