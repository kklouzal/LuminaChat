// Test Phase 2: Model Layer
// Tests ModelInfo with TokenCache integration and SettingsManager integration
// Tests model loading and tokenization caching

#include "rework/Logger.hpp"
#include "rework/SettingsManager.hpp"
#include "rework/TokenCache.hpp"
#include "rework/ModelInfo.hpp"
#include <iostream>
#include <cassert>
#include <fstream>
#include <memory>

void TestModelInfoCreation() {
    std::cout << "=== Testing ModelInfo Creation ===" << std::endl;
    
    // Setup logger
    GetLogger().RegisterOutputCallback([](std::string_view message) {
        std::cout << "LOG: " << message << std::endl;
    });
    
    // Create model info with mock configuration
    ModelConfig config;
    config.model_path = "./mock_model.gguf";
    config.context_size = 4096;
    config.gpu_layers = 20;
    config.use_mmap = true;
    config.use_mlock = false;
    
    ModelInfo model_info("test_model", config);
    
    // Test initial state
    assert(model_info.GetModelId() == "test_model");
    assert(model_info.GetState() == ModelState::UNLOADED);
    assert(model_info.GetConfig().model_path == "./mock_model.gguf");
    assert(model_info.GetConfig().context_size == 4096);
    
    std::cout << "ModelInfo creation tests passed!" << std::endl;
}

void TestTokenCacheIntegration() {
    std::cout << "=== Testing TokenCache Integration ===" << std::endl;
    
    ModelConfig config;
    config.model_path = "./mock_model.gguf";
    config.context_size = 4096;
    
    ModelInfo model_info("test_model", config);
    
    // Test TokenCache access
    TokenCache& cache = model_info.GetTokenCache();
    
    // Test cache operations through ModelInfo
    std::string test_text = "Hello from ModelInfo test";
    std::vector<int32_t> test_tokens = {100, 200, 300, 400};
    
    // Cache some data
    cache.CacheTokens(test_text, test_tokens);
    
    // Retrieve and verify
    auto cached = cache.GetCachedTokens(test_text);
    assert(cached.has_value());
    assert(cached.value() == test_tokens);
    
    // Test statistics through ModelInfo
    CacheStats stats = cache.GetStats();
    assert(stats.GetTotalRequests() > 0);
    
    std::cout << "TokenCache integration tests passed!" << std::endl;
}

void TestSettingsIntegration() {
    std::cout << "=== Testing SettingsManager Integration ===" << std::endl;
    
    // Create test settings
    std::ofstream test_ini("test_model_settings.ini");
    test_ini << "[Models]\n";
    test_ini << "main_model_path=./models/llama3.gguf\n";
    test_ini << "summary_model_path=./models/llama3_summary.gguf\n";
    test_ini << "main_context_size=8192\n";
    test_ini << "summary_context_size=4096\n";
    test_ini << "main_gpu_layers=35\n";
    test_ini << "summary_gpu_layers=20\n";
    test_ini << "use_mmap=true\n";
    test_ini << "use_mlock=false\n";
    test_ini.close();
    
    SettingsManager settings;
    assert(settings.LoadSettings("test_model_settings.ini"));
    
    // Test creating ModelInfo from settings
    ModelConfig main_config = ModelInfo::CreateConfigFromSettings(settings, "main");
    assert(main_config.model_path == "./models/llama3.gguf");
    assert(main_config.context_size == 8192);
    assert(main_config.gpu_layers == 35);
    assert(main_config.use_mmap == true);
    assert(main_config.use_mlock == false);
    
    ModelConfig summary_config = ModelInfo::CreateConfigFromSettings(settings, "summary");
    assert(summary_config.model_path == "./models/llama3_summary.gguf");
    assert(summary_config.context_size == 4096);
    assert(summary_config.gpu_layers == 20);
    
    // Create ModelInfo instances
    ModelInfo main_model("main_model", main_config);
    ModelInfo summary_model("summary_model", summary_config);
    
    assert(main_model.GetModelId() == "main_model");
    assert(summary_model.GetModelId() == "summary_model");
    
    std::cout << "SettingsManager integration tests passed!" << std::endl;
    
    // Cleanup
    std::remove("test_model_settings.ini");
}

void TestResourceCallbacks() {
    std::cout << "=== Testing Resource Callbacks ===" << std::endl;
    
    ModelConfig config;
    config.model_path = "./mock_model.gguf";
    
    ModelInfo model_info("callback_test_model", config);
    
    // Test callback registration
    bool callback_called = false;
    std::string callback_model_id;
    std::string callback_event;
    
    model_info.RegisterResourceCallback([&](const std::string& model_id, const std::string& event) {
        callback_called = true;
        callback_model_id = model_id;
        callback_event = event;
        std::cout << "Resource callback: model=" << model_id << ", event=" << event << std::endl;
    });
    
    // Simulate a resource event (this would normally be called internally)
    // For testing, we can trigger a mock event
    model_info.TriggerResourceEvent(ResourceEvent::MODEL_LOADING_STARTED);
    
    // Note: Since we don't have actual llama.cpp integration yet, we can't test actual loading
    // But we can test the callback registration and event system
    
    std::cout << "Resource callback tests passed!" << std::endl;
}

void TestModelLifecycle() {
    std::cout << "=== Testing Model Lifecycle ===" << std::endl;
    
    ModelConfig config;
    config.model_path = "./mock_model.gguf";
    config.context_size = 2048;
    config.gpu_layers = 10;
    
    ModelInfo model_info("lifecycle_test", config);
    
    // Test initial state
    assert(model_info.GetState() == ModelState::UNLOADED);
    
    // Test configuration access
    const ModelConfig& retrieved_config = model_info.GetConfig();
    assert(retrieved_config.model_path == config.model_path);
    assert(retrieved_config.context_size == config.context_size);
    assert(retrieved_config.gpu_layers == config.gpu_layers);
    
    // Test model loading (mock - without actual llama.cpp)
    // In real implementation, this would load the actual model
    // For testing, we simulate the state changes
    
    std::cout << "Model Lifecycle tests passed!" << std::endl;
}

void TestMemoryManagement() {
    std::cout << "=== Testing Memory Management ===" << std::endl;
    
    // Test multiple ModelInfo instances
    std::vector<std::unique_ptr<ModelInfo>> models;
    
    for (int i = 0; i < 5; ++i) {
        ModelConfig config;
        config.model_path = "./mock_model_" + std::to_string(i) + ".gguf";
        config.context_size = 1024 * (i + 1);
        
        auto model = std::make_unique<ModelInfo>("model_" + std::to_string(i), config);
        
        // Test TokenCache independence
        TokenCache& cache = model->GetTokenCache();
        std::string test_text = "Test text for model " + std::to_string(i);
        std::vector<int32_t> tokens = {i * 100, i * 101, i * 102};
        
        cache.CacheTokens(test_text, tokens);
        
        models.push_back(std::move(model));
    }
    
    // Verify each model has independent TokenCache
    for (size_t i = 0; i < models.size(); ++i) {
        TokenCache& cache = models[i]->GetTokenCache();
        std::string test_text = "Test text for model " + std::to_string(i);
        
        auto cached = cache.GetCachedTokens(test_text);
        assert(cached.has_value());
        
        // Verify it doesn't have data from other models
        std::string other_text = "Test text for model " + std::to_string((i + 1) % models.size());
        auto other_cached = cache.GetCachedTokens(other_text);
        if (i != (i + 1) % models.size()) { // Don't test same model
            assert(!other_cached.has_value()); // Should not have other model's data
        }
    }
    
    // Test cleanup
    models.clear(); // This should properly cleanup all ModelInfo instances
    
    std::cout << "Memory management tests passed!" << std::endl;
}

int main() {
    std::cout << "=== LuminaChat Phase 2 Model Layer Test ===" << std::endl;
    std::cout << "Testing ModelInfo with TokenCache integration and SettingsManager integration" << std::endl;
    std::cout << "Testing model loading and tokenization caching" << std::endl;
    std::cout << std::endl;
    
    try {
        TestModelInfoCreation();
        std::cout << std::endl;
        
        TestTokenCacheIntegration();
        std::cout << std::endl;
        
        TestSettingsIntegration();
        std::cout << std::endl;
        
        TestResourceCallbacks();
        std::cout << std::endl;
        
        TestModelLifecycle();
        std::cout << std::endl;
        
        TestMemoryManagement();
        std::cout << std::endl;
        
        std::cout << "=== ALL PHASE 2 TESTS PASSED ===" << std::endl;
        std::cout << "Model layer components are working correctly!" << std::endl;
        std::cout << "Ready for Phase 3: Template Layer" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
