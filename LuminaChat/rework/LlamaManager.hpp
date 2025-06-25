#pragma once

#include "ModelInfo.hpp"
#include "ContextInfo.hpp"
#include "SettingsManager.hpp"
#include "Logger.hpp"
#include <memory>
#include <unordered_map>
#include <string>
#include <mutex>
#include <vector>
#include <functional>
#include <atomic>
#include <chrono>

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
    
    // Helper methods
    void UpdateStats();
    void NotifyResourceEvent(const std::string& event, const std::string& details);
    std::string LoadTemplateFromSettings(const std::string& template_name);
    bool ValidateModelId(const std::string& model_id) const;
    bool ValidateContextId(const std::string& context_id) const;
    
public:
    // Constructors
    LlamaManager() : LlamaManager(nullptr) {}
    explicit LlamaManager(SettingsManager* settings);
    
    // Destructor - ensures proper cleanup
    ~LlamaManager();
    
    // Non-copyable, movable
    LlamaManager(const LlamaManager&) = delete;
    LlamaManager& operator=(const LlamaManager&) = delete;
    LlamaManager(LlamaManager&&) = default;
    LlamaManager& operator=(LlamaManager&&) = default;
    
    // Initialization and cleanup
    bool Initialize(SettingsManager* settings = nullptr);
    void Cleanup();
    bool IsReady() const;
    
    // Settings integration
    void SetSettingsManager(SettingsManager* settings);
    bool LoadDefaultTemplates();
    bool SaveCurrentState();
    
    // Model management
    ModelInfo* GetOrCreateModelInfo(const std::string& model_id);
    ModelInfo* GetModelInfo(const std::string& model_id);
    bool HasModel(const std::string& model_id) const;
    bool RemoveModel(const std::string& model_id);
    std::vector<std::string> GetLoadedModelIds() const;
    
    // Enhanced model creation with settings integration
    ModelInfo* CreateModelFromSettings(const std::string& model_id, const std::string& settings_prefix);
    bool LoadModelFromSettings(const std::string& model_id, const std::string& settings_prefix);
    
    // Context management  
    ContextInfo* GetOrCreateContextInfo(const std::string& context_id, const std::string& model_id, const std::string& template_name = "default");
    ContextInfo* GetContextInfo(const std::string& context_id);
    bool HasContext(const std::string& context_id) const;
    bool RemoveContext(const std::string& context_id);
    std::vector<std::string> GetActiveContextIds() const;
    
    // Enhanced context creation with template integration
    ContextInfo* CreateContextWithTemplate(const std::string& context_id, const std::string& model_id, const std::string& template_name);
    ContextInfo* CreateContextWithCustomTemplate(const std::string& context_id, const std::string& model_id, const std::string& custom_template);
    
    // Template management
    bool RegisterTemplate(const std::string& template_name, const std::string& template_content);
    std::string GetTemplate(const std::string& template_name) const;
    bool HasTemplate(const std::string& template_name) const;
    std::vector<std::string> GetAvailableTemplates() const;
    
    // Lifecycle management for entire system
    bool LoadAllModelsFromSettings();
    void UnloadAllModels();
    void ClearAllContexts();
    
    // Statistics and monitoring
    const LlamaManagerStats& GetStats() const;
    void RefreshStats();
    std::string GetSystemStatus() const;
    
    // Resource monitoring callback registration
    void RegisterResourceCallback(std::function<void(std::string, std::string)> callback);
    
    // Advanced resource management
    size_t GetTotalMemoryUsage() const;
    void OptimizeMemoryUsage();
    bool PerformMaintenance();
    
    // Context relationship management (for future plugin support)
    std::vector<std::string> GetContextsForModel(const std::string& model_id) const;
    bool MigrateContext(const std::string& context_id, const std::string& new_model_id);
    
    // Batch operations
    bool CreateMultipleContexts(const std::vector<std::string>& context_ids, const std::string& model_id, const std::string& template_name = "default");
    void CleanupInactiveContexts(std::chrono::minutes inactive_threshold = std::chrono::minutes(30));
    
    // Error handling and recovery
    bool RecoverFromError();
    std::vector<std::string> GetErrorMessages() const;
    void ClearErrors();
    
    // Debug and testing support
    void DumpManagerState() const;
    bool ValidateInternalState() const;
    void SetTestMode(bool enabled);
    
    // Plugin integration preparation
    void NotifyContextCreated(const std::string& context_id, const std::string& model_id);
    void NotifyContextDestroyed(const std::string& context_id);
    void NotifyModelLoaded(const std::string& model_id);
    void NotifyModelUnloaded(const std::string& model_id);
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

// Constructor
inline LlamaManager::LlamaManager(SettingsManager* settings) 
    : settings_manager(settings) {
    LOG_LlamaManager("Creating LlamaManager instance");
    stats.Reset();
    
    // Load default templates if settings available
    if (settings_manager) {
        LoadDefaultTemplates();
    }
}

// Destructor
inline LlamaManager::~LlamaManager() {
    if (state != LlamaManagerState::UNINITIALIZED) {
        LOG_LlamaManager("LlamaManager destructor called - performing cleanup");
        Cleanup();
    }
}

// Initialize
inline bool LlamaManager::Initialize(SettingsManager* settings) {
    std::lock_guard<std::mutex> lock(manager_mutex);
    
    if (state == LlamaManagerState::READY) {
        LOG_LlamaManager("LlamaManager already initialized");
        return true;
    }
    
    state = LlamaManagerState::INITIALIZING;
    LOG_LlamaManager("Initializing LlamaManager");
    
    if (settings) {
        settings_manager = settings;
    }
    
    stats.Reset();
    
    // Load default templates from settings
    if (settings_manager && !LoadDefaultTemplates()) {
        LOG_ERROR_LlamaManager("Failed to load default templates");
        state = LlamaManagerState::ERROR_STATE;
        return false;
    }
    
    state = LlamaManagerState::READY;
    LOG_LlamaManager("LlamaManager initialization complete");
    
    NotifyResourceEvent("manager_initialized", "LlamaManager ready for use");
    return true;
}

// Cleanup - cascading cleanup of all resources
inline void LlamaManager::Cleanup() {
    std::lock_guard<std::mutex> lock(manager_mutex);
    
    if (state == LlamaManagerState::UNINITIALIZED) {
        return;
    }
    
    state = LlamaManagerState::SHUTTING_DOWN;
    LOG_LlamaManager("Starting LlamaManager cleanup");
    
    // Clear all contexts first (they depend on models)
    size_t context_count = contexts.size();
    contexts.clear();
    LOG_LlamaManager("Cleared " + std::to_string(context_count) + " contexts");
    
    // Then cleanup all models
    size_t model_count = models.size();
    for (auto& [model_id, model] : models) {
        if (model) {
            model->Cleanup();
        }
    }
    models.clear();
    LOG_LlamaManager("Cleaned up " + std::to_string(model_count) + " models");
    
    // Clear templates
    default_templates.clear();
    
    // Reset statistics
    stats.Reset();
    
    state = LlamaManagerState::UNINITIALIZED;
    LOG_LlamaManager("LlamaManager cleanup complete");
    
    NotifyResourceEvent("manager_shutdown", "LlamaManager cleanup completed");
}

// Check if ready
inline bool LlamaManager::IsReady() const {
    std::lock_guard<std::mutex> lock(manager_mutex);
    return state == LlamaManagerState::READY;
}

// Get or create ModelInfo
inline ModelInfo* LlamaManager::GetOrCreateModelInfo(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(manager_mutex);
    
    if (state != LlamaManagerState::READY) {
        LOG_ERROR_LlamaManager("LlamaManager not ready for GetOrCreateModelInfo");
        return nullptr;
    }
    
    if (!ValidateModelId(model_id)) {
        LOG_ERROR_LlamaManager("Invalid model ID: " + model_id);
        return nullptr;
    }
    
    auto it = models.find(model_id);
    if (it != models.end()) {
        LOG_LlamaManager("Returning existing ModelInfo: " + model_id);
        return it->second.get();
    }
    
    // Create new ModelInfo
    auto model_info = std::make_unique<ModelInfo>(model_id);
    ModelInfo* result = model_info.get();
    
    models[model_id] = std::move(model_info);
    UpdateStats();
    
    LOG_LlamaManager("Created new ModelInfo: " + model_id);
    NotifyResourceEvent("model_created", model_id);
    
    return result;
}

// Get or create ContextInfo with template integration
inline ContextInfo* LlamaManager::GetOrCreateContextInfo(const std::string& context_id, const std::string& model_id, const std::string& template_name) {
    std::lock_guard<std::mutex> lock(manager_mutex);
    
    if (state != LlamaManagerState::READY) {
        LOG_ERROR_LlamaManager("LlamaManager not ready for GetOrCreateContextInfo");
        return nullptr;
    }
    
    if (!ValidateContextId(context_id)) {
        LOG_ERROR_LlamaManager("Invalid context ID: " + context_id);
        return nullptr;
    }
    
    auto it = contexts.find(context_id);
    if (it != contexts.end()) {
        LOG_LlamaManager("Returning existing ContextInfo: " + context_id);
        return it->second.get();
    }
    
    // Get or create the model
    auto model_it = models.find(model_id);
    ModelInfo* model_info = nullptr;
    if (model_it != models.end()) {
        model_info = model_it->second.get();
    } else {
        // Create new ModelInfo directly to avoid recursive locking
        if (!ValidateModelId(model_id)) {
            LOG_ERROR_LlamaManager("Invalid model ID: " + model_id);
            return nullptr;
        }
        
        auto model_info_ptr = std::make_unique<ModelInfo>(model_id);
        model_info = model_info_ptr.get();
        models[model_id] = std::move(model_info_ptr);
        LOG_LlamaManager("Created new ModelInfo: " + model_id);
        NotifyResourceEvent("model_created", model_id);
    }
    
    if (!model_info) {
        LOG_ERROR_LlamaManager("Failed to get ModelInfo for context creation: " + model_id);
        return nullptr;
    }
    
    // Get template
    std::string template_content = GetTemplate(template_name);
    if (template_content.empty()) {
        // Provide default template if not found
        template_content = R"(<|start_header_id|>system_message<|end_header_id|>
{{ system_prompt }}
<|eot_id|>

{%- for msg in messages %}
  {% if msg.role == "assistant" %}
<|start_header_id|>assistant<|end_header_id|>
{{ msg.content | trim }}<|eot_id|>
  {% else %}
<|start_header_id|>user<|end_header_id|>
[{{ msg.role }}] {{ msg.content | trim }}<|eot_id|>
  {% endif %}
{%- endfor %}

<|start_header_id|>assistant<|end_header_id|>)";
        LOG_LlamaManager("Using default template for context: " + context_id);
    }
    
    // Create new ContextInfo
    auto context_info = std::make_unique<ContextInfo>(context_id, model_info, template_content);
    ContextInfo* result = context_info.get();
    
    contexts[context_id] = std::move(context_info);
    UpdateStats();
    
    LOG_LlamaManager("Created new ContextInfo: " + context_id + " with model: " + model_id + " and template: " + template_name);
    NotifyResourceEvent("context_created", context_id + " -> " + model_id);
    
    return result;
}

// Load default templates from settings
inline bool LlamaManager::LoadDefaultTemplates() {
    if (!settings_manager) {
        LOG_ERROR_LlamaManager("No SettingsManager available for loading templates");
        return false;
    }
    
    LOG_LlamaManager("Loading default templates from settings");
    
    // Load default template
    std::string default_template = LoadTemplateFromSettings("default");
    if (!default_template.empty()) {
        default_templates["default"] = default_template;
        LOG_LlamaManager("Loaded default template");
    }
    
    // Load summary template
    std::string summary_template = LoadTemplateFromSettings("summary");
    if (!summary_template.empty()) {
        default_templates["summary"] = summary_template;
        LOG_LlamaManager("Loaded summary template");
    }
    
    // Load any other named templates from settings
    // This would be expanded based on actual settings structure
    
    LOG_LlamaManager("Loaded " + std::to_string(default_templates.size()) + " templates");
    return !default_templates.empty();
}

// Helper method implementations
inline void LlamaManager::UpdateStats() {
    stats.total_models = models.size();
    stats.total_contexts = contexts.size();
    
    stats.loaded_models = 0;
    stats.active_contexts = 0;
    
    for (const auto& [id, model] : models) {
        if (model && model->IsLoaded()) {
            stats.loaded_models++;
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
        resource_callback(event, details);
    }
}

inline std::string LlamaManager::LoadTemplateFromSettings(const std::string& template_name) {
    if (!settings_manager) {
        return "";
    }
    
    // This would load from actual settings structure
    // For now, return a basic template for testing
    if (template_name == "default") {
        return R"(<|start_header_id|>system_message<|end_header_id|>
{{ system_prompt }}
<|eot_id|>

{%- for msg in messages %}
  {% if msg.role == "assistant" %}
<|start_header_id|>assistant<|end_header_id|>
{{ msg.content | trim }}<|eot_id|>
  {% else %}
<|start_header_id|>user<|end_header_id|>
[{{ msg.role }}] {{ msg.content | trim }}<|eot_id|>
  {% endif %}
{%- endfor %}

<|start_header_id|>assistant<|end_header_id|>)";
    }
    
    return "";
}

inline bool LlamaManager::ValidateModelId(const std::string& model_id) const {
    return !model_id.empty() && model_id.length() <= 256 && 
           model_id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") == std::string::npos;
}

inline bool LlamaManager::ValidateContextId(const std::string& context_id) const {
    return !context_id.empty() && context_id.length() <= 256 && 
           context_id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") == std::string::npos;
}

inline std::string LlamaManager::GetTemplate(const std::string& template_name) const {
    auto it = default_templates.find(template_name);
    if (it != default_templates.end()) {
        return it->second;
    }
    
    // If template not found, try to load from settings
    if (settings_manager) {
        return const_cast<LlamaManager*>(this)->LoadTemplateFromSettings(template_name);
    }
    
    return "";
}

// Additional method implementations
inline void LlamaManager::RegisterResourceCallback(std::function<void(std::string, std::string)> callback) {
    std::lock_guard<std::mutex> lock(manager_mutex);
    resource_callback = std::move(callback);
    LOG_LlamaManager("Resource callback registered");
}

inline const LlamaManagerStats& LlamaManager::GetStats() const {
    std::lock_guard<std::mutex> lock(manager_mutex);
    return stats;
}

inline bool LlamaManager::RegisterTemplate(const std::string& template_name, const std::string& template_content) {
    std::lock_guard<std::mutex> lock(manager_mutex);
    
    if (template_name.empty() || template_content.empty()) {
        LOG_ERROR_LlamaManager("Cannot register template with empty name or content");
        return false;
    }
    
    default_templates[template_name] = template_content;
    LOG_LlamaManager("Registered template: " + template_name);
    return true;
}

inline std::vector<std::string> LlamaManager::GetAvailableTemplates() const {
    std::lock_guard<std::mutex> lock(manager_mutex);
    
    std::vector<std::string> templates;
    templates.reserve(default_templates.size());
    
    for (const auto& [name, content] : default_templates) {
        templates.push_back(name);
    }
    
    return templates;
}

inline std::vector<std::string> LlamaManager::GetActiveContextIds() const {
    std::lock_guard<std::mutex> lock(manager_mutex);
    
    std::vector<std::string> context_ids;
    context_ids.reserve(contexts.size());
    
    for (const auto& [id, context] : contexts) {
        if (context) {
            context_ids.push_back(id);
        }
    }
    
    return context_ids;
}

inline std::vector<std::string> LlamaManager::GetLoadedModelIds() const {
    std::lock_guard<std::mutex> lock(manager_mutex);
    
    std::vector<std::string> model_ids;
    model_ids.reserve(models.size());
    
    for (const auto& [id, model] : models) {
        if (model) {
            model_ids.push_back(id);
        }
    }
    
    return model_ids;
}

inline ModelInfo* LlamaManager::GetModelInfo(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(manager_mutex);
    
    auto it = models.find(model_id);
    return (it != models.end()) ? it->second.get() : nullptr;
}

inline ContextInfo* LlamaManager::GetContextInfo(const std::string& context_id) {
    std::lock_guard<std::mutex> lock(manager_mutex);
    
    auto it = contexts.find(context_id);
    return (it != contexts.end()) ? it->second.get() : nullptr;
}

inline bool LlamaManager::HasModel(const std::string& model_id) const {
    std::lock_guard<std::mutex> lock(manager_mutex);
    return models.find(model_id) != models.end();
}

inline bool LlamaManager::HasContext(const std::string& context_id) const {
    std::lock_guard<std::mutex> lock(manager_mutex);
    return contexts.find(context_id) != contexts.end();
}

inline bool LlamaManager::HasTemplate(const std::string& template_name) const {
    std::lock_guard<std::mutex> lock(manager_mutex);
    return default_templates.find(template_name) != default_templates.end();
}

inline bool LlamaManager::RemoveModel(const std::string& model_id) {
    std::lock_guard<std::mutex> lock(manager_mutex);
    
    auto it = models.find(model_id);
    if (it != models.end()) {
        if (it->second) {
            it->second->Cleanup();
        }
        models.erase(it);
        UpdateStats();
        LOG_LlamaManager("Removed model: " + model_id);
        NotifyResourceEvent("model_removed", model_id);
        return true;
    }
    
    return false;
}

inline bool LlamaManager::RemoveContext(const std::string& context_id) {
    std::lock_guard<std::mutex> lock(manager_mutex);
    
    auto it = contexts.find(context_id);
    if (it != contexts.end()) {
        contexts.erase(it);
        UpdateStats();
        LOG_LlamaManager("Removed context: " + context_id);
        NotifyResourceEvent("context_removed", context_id);
        return true;
    }
    
    return false;
}

// Helper function implementations
namespace LlamaManagerHelpers {
    inline std::string GenerateModelId(const std::string& base_name) {
        static std::atomic<int> counter{0};
        return base_name + "_" + std::to_string(++counter);
    }
    
    inline std::string GenerateContextId(const std::string& base_name) {
        static std::atomic<int> counter{0};
        return base_name + "_" + std::to_string(++counter);
    }
    
    inline bool IsValidTemplate(const std::string& template_content) {
        return !template_content.empty() && template_content.find("messages") != std::string::npos;
    }
}
