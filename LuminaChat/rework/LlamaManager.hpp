#pragma once

#include "ModelInfo.hpp"
#include "ContextInfo.hpp"
#include "SettingsManager.hpp"
#include "Logger.hpp"
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
    
    // Default templates (loaded from settings)
    std::unordered_map<std::string, std::string> default_templates;
    
    // Resource monitoring
    std::function<void(std::string, std::string)> resource_callback;
    
    // llama.cpp initialization state
    static std::atomic<bool> llama_backend_initialized;
    static std::mutex backend_init_mutex;
    
    // Helper methods
    void UpdateStats();
    void NotifyResourceEvent(const std::string& event, const std::string& details);
    std::string LoadTemplateFromSettings(const std::string& template_name);
    std::string GetTemplateInternal(const std::string& template_name); // Internal version - no mutex
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
    
    // Context management
    ContextInfo* GetOrCreateContextInfo(const std::string& context_id, const std::string& model_id, const std::string& template_name = "default");
    bool CreateContextWithTemplate(const std::string& context_id, const std::string& model_id, const std::string& template_content);
    bool RemoveContext(const std::string& context_id);
    ContextInfo* GetContextInfo(const std::string& context_id) const;
    std::vector<std::string> GetContextIds() const;
    
    // Template management
    bool LoadTemplate(const std::string& template_name, const std::string& template_content);
    bool LoadTemplateFromFile(const std::string& template_name, const std::string& file_path);
    std::string GetTemplate(const std::string& template_name) const;
    std::vector<std::string> GetTemplateNames() const;
    
    // State and statistics
    LlamaManagerState GetState() const { return state; }
    const LlamaManagerStats& GetStats() const { return stats; }
    bool IsReady() const { return state == LlamaManagerState::READY; }
    
    // Settings integration
    void SetSettingsManager(SettingsManager* settings);
    bool LoadDefaultTemplatesFromSettings();
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
};

// Helper functions for LlamaManager integration
namespace LlamaManagerHelpers {
    // Generate unique IDs
    std::string GenerateModelId(const std::string& base_name = "model");
    std::string GenerateContextId(const std::string& base_name = "context");
    
    // Template utilities
    bool IsValidTemplate(const std::string& template_content);
    std::string SanitizeTemplateContent(const std::string& template_content);
    
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
        
        // Load default templates from settings if available
        if (settings_manager) {
            LoadDefaultTemplatesFromSettings();
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
inline ContextInfo* LlamaManager::GetOrCreateContextInfo(const std::string& context_id, const std::string& model_id, const std::string& template_name) {
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
    
    // Get template (internal call - mutex already held)
    std::string template_content = GetTemplateInternal(template_name);
    if (template_content.empty()) {
        LOG_ERROR_LlamaManager("Template '" + template_name + "' not found for context: " + context_id);
        // Use a minimal default template that passes validation
        template_content = R"({% for msg in messages %}{% if msg.role == "user" %}User: {{ msg.content }}
{% else %}Assistant: {{ msg.content }}
{% endif %}{% endfor %}assistant<|end_header_id|>

)";
    }
    
    LOG_LlamaManager("Creating new ContextInfo: " + context_id + " with model: " + model_id);
    
    try {
        auto context_info = std::make_unique<ContextInfo>(context_id, model_info, template_content);
        
        // Register summarization callback if we have resource callback
        if (resource_callback) {
            context_info->RegisterSummarizationCallback([this](const std::string& ctx_id, const std::string& content) {
                NotifyResourceEvent("SUMMARIZATION_REQUEST", "Context " + ctx_id + " requests summarization");
                // Here you would typically forward to Orchestrator
            });
        }
        
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

// Template management
inline bool LlamaManager::LoadTemplate(const std::string& template_name, const std::string& template_content) {
    std::lock_guard<std::mutex> lock(manager_mutex);
    
    if (template_name.empty() || template_content.empty()) {
        LOG_ERROR_LlamaManager("Empty template name or content");
        return false;
    }
    
    default_templates[template_name] = template_content;
    
    // Save to settings if available
    if (settings_manager) {
        settings_manager->SetString("Templates", template_name, template_content);
    }
    
    LOG_LlamaManager("Loaded template: " + template_name + " (" + std::to_string(template_content.length()) + " chars)");
    return true;
}

inline std::string LlamaManager::GetTemplate(const std::string& template_name) const {
    std::lock_guard<std::mutex> lock(manager_mutex);
    
    auto it = default_templates.find(template_name);
    if (it != default_templates.end()) {
        return it->second;
    }
    
    // Try to load from settings if available
    if (settings_manager) {
        std::string template_content = settings_manager->GetString("Templates", template_name, "");
        if (!template_content.empty()) {
            // Cache it for future use
            const_cast<LlamaManager*>(this)->default_templates[template_name] = template_content;
            return template_content;
        }
    }
    
    LOG_DEBUG_LlamaManager("Template not found: " + template_name);
    return "";
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
        
        // Clear templates
        default_templates.clear();
        
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
        if (context && context->GetState() == ContextState::READY) {
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

inline bool LlamaManager::LoadDefaultTemplatesFromSettings() {
    if (!settings_manager) {
        return false;
    }
    
    LOG_LlamaManager("Loading default templates from settings");
    
    // Load basic templates
    std::vector<std::string> template_names = {"default", "chat", "summary", "system"};
    
    for (const std::string& name : template_names) {
        std::string content = settings_manager->GetString("Templates", name, "");
        if (!content.empty()) {
            default_templates[name] = content;
            LOG_LlamaManager("Loaded template from settings: " + name);
        }
    }
    
    return true;
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

inline void LlamaManager::RegisterResourceCallback(std::function<void(std::string, std::string)> callback) {
    std::lock_guard<std::mutex> lock(manager_mutex);
    resource_callback = std::move(callback);
    LOG_LlamaManager("Resource callback registered");
}

// Internal template getter - assumes mutex is already held
inline std::string LlamaManager::GetTemplateInternal(const std::string& template_name) {
    // Note: This method assumes manager_mutex is already acquired by the caller
    auto it = default_templates.find(template_name);
    if (it != default_templates.end()) {
        return it->second;
    }
    
    // Try to load from settings if available
    if (settings_manager) {
        std::string template_content = settings_manager->GetString("Templates", template_name, "");
        if (!template_content.empty()) {
            // Cache it for future use
            default_templates[template_name] = template_content;
            return template_content;
        }
    }
    
    LOG_DEBUG_LlamaManager("Template not found: " + template_name);
    return "";
}
