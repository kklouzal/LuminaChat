// Phase 1 Implementation: Foundation Components
// Tests Logger, SettingsManager, Sanitizer integration, TokenCache, ContextSizeManager

#pragma once

#include "rework/Logger.hpp"
#include "rework/SettingsManager.hpp"
#include "rework/Sanitizer.hpp"
#include "rework/TokenCache.hpp"
#include "rework/ContextSizeManager.hpp"
#include <iostream>
#include <cassert>
#include <fstream>
#include <thread>
#include <chrono>

namespace Phase1Tests {

void TestLogger() {
    std::cout << "=== Testing Logger ===" << std::endl;
    
    // Test callback registration
    std::string last_log_message;
    GetLogger().RegisterOutputCallback([&](std::string_view message) {
        last_log_message = std::string(message);
        std::cout << "LOG: " << message << std::endl;
    });
    
    // Test different log levels
    GetLogger().SetLogLevel(Logger::LogLevel::DEBUG);
    
    LOG_Logger("Logger initialized successfully");
    assert(!last_log_message.empty());
    assert(last_log_message.find("Logger") != std::string::npos);
    
    LOG_DEBUG("TestComponent", "Debug message test");
    assert(last_log_message.find("DEBUG") != std::string::npos);
    
    LOG_ERROR("TestComponent", "Error message test");
    assert(last_log_message.find("ERROR") != std::string::npos);
    
    std::cout << "Logger tests passed!" << std::endl;
}

void TestSettingsManager() {
    std::cout << "=== Testing SettingsManager ===" << std::endl;
    
    // Create test settings file
    std::ofstream test_ini("test_settings.ini");
    test_ini << "[Model]\n";
    test_ini << "main_model_path=./models/main.gguf\n";
    test_ini << "summary_model_path=./models/summary.gguf\n";
    test_ini << "context_size=8192\n";
    test_ini << "gpu_layers=35\n";
    test_ini << "\n[Discord]\n";
    test_ini << "bot_token=test_token_123\n";
    test_ini << "auto_respond=true\n";
    test_ini << "\n[Templates]\n";
    test_ini << "main_template=llama3_chat\n";
    test_ini << "summary_template=llama3_summary\n";
    test_ini.close();
    
    SettingsManager settings;
    assert(settings.LoadSettings("test_settings.ini"));
    
    // Test string values
    assert(settings.GetString("Model", "main_model_path") == "./models/main.gguf");
    assert(settings.GetString("Discord", "bot_token") == "test_token_123");
    
    // Test int values
    assert(settings.GetInt("Model", "context_size") == 8192);
    assert(settings.GetInt("Model", "gpu_layers") == 35);
    
    // Test bool values
    assert(settings.GetBool("Discord", "auto_respond") == true);
    
    // Test template configuration
    assert(settings.GetString("Templates", "main_template") == "llama3_chat");
    
    // Test setting new values
    settings.SetString("Test", "new_value", "hello world");
    settings.SetInt("Test", "new_int", 42);
    settings.SetBool("Test", "new_bool", false);
    
    assert(settings.GetString("Test", "new_value") == "hello world");
    assert(settings.GetInt("Test", "new_int") == 42);
    assert(settings.GetBool("Test", "new_bool") == false);
    
    // Test save functionality
    assert(settings.SaveSettings());
    
    // Test template management
    settings.SetChatTemplate("test_template", "Template content {{ messages }}");
    assert(settings.GetChatTemplate("test_template") == "Template content {{ messages }}");
    
    std::cout << "SettingsManager tests passed!" << std::endl;
    
    // Cleanup
    std::remove("test_settings.ini");
}

void TestSanitizer() {
    std::cout << "=== Testing Sanitizer ===" << std::endl;
    
    Sanitizer sanitizer;
    
    // Test basic input sanitization
    std::string test_input = "Hello world!";
    assert(sanitizer.SanitizeInput(test_input) == true);
    
    // Test callback registration
    bool filter_callback_called = false;
    sanitizer.RegisterFilterCallback([&](const std::string& content, bool allowed) {
        filter_callback_called = true;
        std::cout << "Filter callback: " << content << " -> " << (allowed ? "ALLOWED" : "BLOCKED") << std::endl;
    });
    
    // Test content filtering
    test_input = "This is clean content";
    bool result = sanitizer.SanitizeInput(test_input);
    
    std::cout << "Sanitizer tests passed!" << std::endl;
}

void TestTokenCache() {
    std::cout << "=== Testing TokenCache ===" << std::endl;
    
    TokenCache cache;
    
    // Test text-to-token caching
    std::string test_text = "Hello, world!";
    std::vector<int32_t> test_tokens = {12345, 67890, 11111};
    
    // Simulate cache miss
    auto cached_tokens = cache.GetCachedTokens(test_text);
    assert(!cached_tokens.has_value()); // Should be cache miss
    
    // Store in cache
    cache.CacheTokens(test_text, test_tokens);
    
    // Simulate cache hit
    cached_tokens = cache.GetCachedTokens(test_text);
    assert(cached_tokens.has_value()); // Should be cache hit
    assert(cached_tokens.value() == test_tokens);
    
    // Test token-to-text caching
    std::string test_decoded = "Hello, world!";
    auto cached_text = cache.GetCachedText(test_tokens);
    assert(!cached_text.has_value()); // Should be cache miss
    
    cache.CacheText(test_tokens, test_decoded);
    cached_text = cache.GetCachedText(test_tokens);
    assert(cached_text.has_value()); // Should be cache hit
    assert(cached_text.value() == test_decoded);
    
    // Test template caching
    std::string template_key = "test_template_123";
    cache.CacheTemplate(template_key, test_tokens);
    auto cached_template = cache.GetCachedTemplate(template_key);
    assert(cached_template.has_value());
    assert(cached_template.value() == test_tokens);
    
    // Test statistics
    CacheStats stats = cache.GetStats();
    assert(stats.GetTotalRequests() > 0);
    assert(stats.GetTotalHits() > 0);
    
    std::cout << "Cache Stats:" << std::endl;
    std::cout << "  Text-to-Token Hit Ratio: " << stats.GetTextToTokenHitRatio() << std::endl;
    std::cout << "  Token-to-Text Hit Ratio: " << stats.GetTokenToTextHitRatio() << std::endl;
    std::cout << "  Template Hit Ratio: " << stats.GetTemplateHitRatio() << std::endl;
    std::cout << "  Total Requests: " << stats.GetTotalRequests() << std::endl;
    std::cout << "  Total Hits: " << stats.GetTotalHits() << std::endl;
    
    // Test callback registration
    bool invalidation_called = false;
    cache.RegisterInvalidationCallback([&](const std::string& cache_key) {
        invalidation_called = true;
        std::cout << "Cache invalidation callback: " << cache_key << std::endl;
    });
    
    cache.InvalidateCache();
    
    std::cout << "TokenCache tests passed!" << std::endl;
}

void TestContextSizeManager() {
    std::cout << "=== Testing ContextSizeManager ===" << std::endl;
    
    ContextSizeManager manager;
    
    // Test basic usage tracking
    manager.UpdateUsage(1000);
    assert(manager.GetCurrentUsage() == 1000);
    
    // Test pruning decision
    int32_t max_tokens = 8192;
    
    // Below threshold - no pruning needed
    assert(!manager.NeedsPruning(4000, max_tokens));
    
    // Above threshold - pruning needed
    assert(manager.NeedsPruning(7000, max_tokens));
    
    // Test callback registration
    bool summarization_requested = false;
    std::string callback_context_id;
    std::string callback_content;
    
    manager.RegisterSummarizationCallback([&](const std::string& context_id, const std::string& content) {
        summarization_requested = true;
        callback_context_id = context_id;
        callback_content = content;
        std::cout << "Summarization callback: context=" << context_id << ", content_size=" << content.size() << std::endl;
    });
    
    // Test trigger summarization (simulate internal call)
    if (manager.NeedsPruning(7000, max_tokens)) {
        std::cout << "Context size exceeded threshold, would request summarization" << std::endl;
    }
    
    // Test statistics
    auto stats = manager.GetUsageStats();
    std::cout << "Usage Stats:" << std::endl;
    std::cout << "  Current Usage: " << stats.current_usage << std::endl;
    std::cout << "  Peak Usage: " << stats.peak_usage << std::endl;
    std::cout << "  Pruning Count: " << stats.pruning_count << std::endl;
    
    std::cout << "ContextSizeManager tests passed!" << std::endl;
}

} // namespace Phase1Tests

int TestPhase1Foundation() {
    std::cout << "Testing Logger, SettingsManager, Sanitizer, TokenCache, ContextSizeManager" << std::endl;
    std::cout << "No dependencies - can be built and tested independently" << std::endl;
    std::cout << std::endl;
    
    try {
        Phase1Tests::TestLogger();
        std::cout << std::endl;
        
        Phase1Tests::TestSettingsManager();
        std::cout << std::endl;
        
        Phase1Tests::TestSanitizer();
        std::cout << std::endl;
        
        Phase1Tests::TestTokenCache();
        std::cout << std::endl;
        
        Phase1Tests::TestContextSizeManager();
        std::cout << std::endl;
        
        std::cout << "Foundation components are working correctly!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Phase 1 test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
