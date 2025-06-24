#pragma once

// Include the rework foundation components for integration testing
#include "rework/Logger.hpp"
#include "rework/SettingsManager.hpp"
#include "rework/Sanitizer.hpp"
// Phase 1.B components
#include "rework/TokenCache.hpp"
#include "rework/ContextSizeManager.hpp"

// This file demonstrates how Phase 1.A components integrate with the existing LuminaChat system
// It can be included in LuminaChat.cpp to test the new architecture alongside the current system

namespace ReworkIntegration {
    
    // Global instances for testing (in real implementation, these would be in LuminaChat.cpp)
    std::unique_ptr<SettingsManager> settings_manager;
    std::unique_ptr<Sanitizer> sanitizer;
    // Phase 1.B components
    std::unique_ptr<TokenCache> token_cache;
    std::unique_ptr<ContextSizeManager> context_size_manager;
    
    void InitializeFoundationComponents() {
        // Initialize Logger with UI callback (simulating LuminaChat.cpp registering with Logger)
        GetLogger().RegisterOutputCallback([](std::string_view log_message) {
            // In real implementation, this would route to wxWidgets UI
            // For testing, we can output to console or existing log system
            std::cout << "[REWORK] " << log_message << std::endl;
        });
        
        GetLogger().SetLogLevel(Logger::LogLevel::INFO);
        LOG_Logger("Foundation components initializing...");
        
        // Initialize SettingsManager with existing .ini file compatibility
        settings_manager = std::make_unique<SettingsManager>();
        
        // Try to load existing settings or create new ones
        std::string ini_path = "LuminaChat.ini"; // Use existing settings file
        if (settings_manager->LoadSettings(ini_path)) {
            LOG_SettingsManager("Loaded existing settings file");
        } else {
            LOG_SettingsManager("Created new settings file with defaults");
        }
        
        // Initialize Sanitizer with settings integration
        sanitizer = std::make_unique<Sanitizer>(settings_manager.get());
        
        // Register filter callback (simulating Orchestrator registering with Sanitizer)
        sanitizer->RegisterFilterCallback([](const std::string& content, bool allowed) {
            if (!allowed) {
                LOG_Sanitizer("Content blocked by filter: " + 
                    (content.length() > 50 ? content.substr(0, 50) + "..." : content));
            }
        });
        
        // Initialize Phase 1.B components
        token_cache = std::make_unique<TokenCache>();
        context_size_manager = std::make_unique<ContextSizeManager>();
        
        // Configure ContextSizeManager with default settings
        context_size_manager->SetMaxTokens(settings_manager->GetInt("Model", "context_size", 4096));
        context_size_manager->SetPruningThreshold(0.80f);
        context_size_manager->SetTargetUsage(0.40f);
        context_size_manager->SetMinBufferTokens(512);
        
        // Register callback (simulating Orchestrator registering with ContextSizeManager)
        context_size_manager->RegisterSummarizationCallback([](const std::string& context_id, const std::string& content) {
            LOG_ContextSizeManager("Summarization requested for context: " + context_id);
            // In real implementation, this would trigger the Orchestrator's summarization pipeline
        });
        
        // Register TokenCache invalidation callback (simulating ContextInfo registering with TokenCache)
        token_cache->RegisterInvalidationCallback([](const std::string& cache_key) {
            LOG_TokenCache("Cache invalidation for key: " + cache_key);
            // In real implementation, this would notify ContextInfo to rebuild
        });
        
        LOG_Logger("✓ Foundation components initialized successfully");
    }
    
    // Demonstration of Discord message processing using new architecture
    std::string ProcessDiscordMessage(const std::string& raw_message, const std::string& username) {
        if (!sanitizer) {
            return raw_message; // Fallback if not initialized
        }
        
        LOG_INFO("ReworkIntegration", "Processing Discord message from: " + username);
        
        // Step 1: Sanitize input (Discord formatting cleanup, Unicode validation, etc.)
        std::string sanitized_message = raw_message;
        bool input_allowed = sanitizer->SanitizeInput(sanitized_message);
        
        if (!input_allowed) {
            LOG_WARNING("ReworkIntegration", "Discord message blocked by sanitizer");
            return ""; // Message was blocked
        }
        
        LOG_INFO("ReworkIntegration", "Discord message sanitized successfully");
        return sanitized_message;
    }
    
    // Demonstration of assistant response filtering using new architecture
    bool FilterAssistantResponse(std::string& response) {
        if (!sanitizer) {
            return true; // Fallback if not initialized
        }
        
        LOG_INFO("ReworkIntegration", "Filtering assistant response");
        
        bool response_allowed = sanitizer->FilterAssistantResponse(response);
        
        if (!response_allowed) {
            LOG_WARNING("ReworkIntegration", "Assistant response blocked by blacklist");
            response = "I apologize, but I cannot provide that response as it contains filtered content.";
            return false;
        }
        
        LOG_INFO("ReworkIntegration", "Assistant response passed filter");
        return true;
    }
    
    // Demonstration of settings integration
    void UpdateConfiguration() {
        if (!settings_manager) return;
        
        // Example: Update sanitization settings
        settings_manager->SetBool("Plugins", "sanitization", true);
        settings_manager->SetBool("Sanitizer", "discord_cleanup", true);
        settings_manager->SetBool("Sanitizer", "blacklist_enabled", true);
        
        // Example: Update default chat template
        std::string default_template = settings_manager->GetChatTemplate("default");
        if (default_template.empty()) {
            // Set a basic template for testing
            settings_manager->SetChatTemplate("default", 
                "{{- bos_token }}\n"
                "<|start_header_id|>system<|end_header_id|>\n"
                "You are a helpful AI assistant.\n"
                "<|eot_id|>\n"
                "{% for msg in messages %}\n"
                "<|start_header_id|>{{ msg.role }}<|end_header_id|>\n"
                "{{ msg.content }}<|eot_id|>\n"
                "{% endfor %}\n"
                "<|start_header_id|>assistant<|end_header_id|>");
        }
        
        // Save settings
        bool saved = settings_manager->SaveSettings();
        LOG_SettingsManager("Settings updated and saved: " + std::string(saved ? "✓" : "✗"));
    }
    
    // Test blacklist management
    void TestBlacklistManagement() {
        if (!sanitizer) return;
        
        LOG_Sanitizer("Testing blacklist management...");
        
        // Add some test patterns
        sanitizer->AddPattern("inappropriate_word");
        sanitizer->AddPattern("blocked_phrase");
        sanitizer->AddPattern("filtered_content");
        
        // Test filtering
        std::vector<std::string> test_messages = {
            "This is a normal message",
            "This contains inappropriate_word",
            "Another blocked_phrase here", 
            "This has filtered_content in it",
            "This is also normal"
        };
        
        for (const auto& message : test_messages) {
            bool allowed = sanitizer->IsContentAllowed(message);
            LOG_INFO("ReworkIntegration", 
                "Message: '" + message + "' → " + (allowed ? "ALLOWED" : "BLOCKED"));
        }
        
        // Show pattern count
        size_t pattern_count = sanitizer->GetPatternCount();
        LOG_Sanitizer("Total blacklist patterns: " + std::to_string(pattern_count));
    }
    
    // Phase 1.B integration demonstrations
    void TestTokenCacheIntegration() {
        if (!token_cache) return;
        
        LOG_INFO("ReworkIntegration", "Testing TokenCache integration...");
        
        // Simulate text tokenization workflow
        std::string sample_text = "This is a sample message that would be tokenized by the model.";
        std::vector<int32_t> tokens;
        
        // First access - should miss
        bool hit = token_cache->GetTokens(sample_text, tokens);
        LOG_INFO("ReworkIntegration", "TokenCache first access: " + std::string(hit ? "HIT" : "MISS"));
        
        // Simulate model tokenization (mock tokens)
        std::vector<int32_t> mock_tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        token_cache->StoreTokens(sample_text, mock_tokens);
        
        // Second access - should hit
        tokens.clear();
        hit = token_cache->GetTokens(sample_text, tokens);
        LOG_INFO("ReworkIntegration", "TokenCache second access: " + std::string(hit ? "HIT" : "MISS"));
        LOG_INFO("ReworkIntegration", "Retrieved " + std::to_string(tokens.size()) + " tokens from cache");
        
        // Test template caching
        std::string template_hash = "chat_template_v1_hash";
        std::vector<int32_t> template_tokens = {100, 101, 102, 103, 104};
        token_cache->StoreTemplateTokens(template_hash, template_tokens);
        
        tokens.clear();
        hit = token_cache->GetTemplateTokens(template_hash, tokens);
        LOG_INFO("ReworkIntegration", "Template cache access: " + std::string(hit ? "HIT" : "MISS"));
        
        // Log statistics
        token_cache->LogStatistics();
    }
    
    void TestContextSizeIntegration() {
        if (!context_size_manager) return;
        
        LOG_INFO("ReworkIntegration", "Testing ContextSizeManager integration...");
        
        // Simulate conversation buildup
        LOG_INFO("ReworkIntegration", "Simulating conversation growth...");
        
        // Start with a reasonable context
        context_size_manager->UpdateUsage(800, 400, 1200); // template, summary, messages
        auto stats = context_size_manager->GetStats();
        LOG_INFO("ReworkIntegration", "Initial usage: " + std::to_string(stats.total_tokens) + "/" + 
            std::to_string(stats.max_tokens) + " (" + std::to_string(static_cast<int>(stats.GetUsagePercentage() * 100)) + "%)");
        
        // Add more messages to approach threshold
        context_size_manager->UpdateUsage(800, 400, 2000); // Approaching 80%
        stats = context_size_manager->GetStats();
        LOG_INFO("ReworkIntegration", "Growing usage: " + std::to_string(stats.total_tokens) + "/" + 
            std::to_string(stats.max_tokens) + " (" + std::to_string(static_cast<int>(stats.GetUsagePercentage() * 100)) + "%)");
        
        if (context_size_manager->NeedsPruning()) {
            LOG_INFO("ReworkIntegration", "Context needs pruning - triggering request");
            bool requested = context_size_manager->RequestPruning("discord_channel_123");
            LOG_INFO("ReworkIntegration", "Pruning requested: " + std::string(requested ? "YES" : "NO"));
            
            // Simulate summarization completion (summarized 1000 tokens down to 200)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            context_size_manager->OnSummarizationCompleted("discord_channel_123", 1000, 200);
        }
        
        // Test emergency scenario
        context_size_manager->UpdateUsage(800, 400, 3500); // 95%+ usage
        if (context_size_manager->IsEmergencyPruningNeeded()) {
            LOG_WARNING("ReworkIntegration", "Emergency pruning needed!");
            bool emergency = context_size_manager->RequestEmergencyPruning("discord_channel_123");
            LOG_INFO("ReworkIntegration", "Emergency pruning requested: " + std::string(emergency ? "YES" : "NO"));
            
            // Simulate emergency cleanup (removed 1500 tokens)
            context_size_manager->OnPruningCompleted("discord_channel_123", 1500);
        }
        
        // Log final statistics
        context_size_manager->LogStatistics();
    }
    
    void TestPhase1BIntegration() {
        LOG_INFO("ReworkIntegration", "=== Phase 1.B Integration Tests ===");
        
        TestTokenCacheIntegration();
        TestContextSizeIntegration();
        
        LOG_INFO("ReworkIntegration", "✓ Phase 1.B integration tests completed");
    }
    
    void CleanupFoundationComponents() {
        LOG_Logger("Cleaning up foundation components...");
        
        if (settings_manager && settings_manager->IsDirty()) {
            settings_manager->SaveSettings();
        }
        
        // Cleanup Phase 1.B components
        if (token_cache) {
            token_cache->LogStatistics();
        }
        if (context_size_manager) {
            context_size_manager->LogStatistics();
        }
        
        token_cache.reset();
        context_size_manager.reset();
        sanitizer.reset();
        settings_manager.reset();
        
        std::cout << "✓ Foundation components cleaned up" << std::endl;
    }
}

// Usage example for integration with existing LuminaChat.cpp:
/*
#include "rework_integration.hpp"

void LuminaChat::Start() {
    // Initialize existing components...
    
    // Initialize new foundation components
    ReworkIntegration::InitializeFoundationComponents();
    ReworkIntegration::UpdateConfiguration();
    ReworkIntegration::TestBlacklistManagement();
    // Phase 1.B integration tests
    ReworkIntegration::TestPhase1BIntegration();
    
    // Continue with existing initialization...
}

void LuminaChat::OnDiscordMessage(const std::string& message, const std::string& username) {
    // Process message with new sanitization system
    std::string sanitized = ReworkIntegration::ProcessDiscordMessage(message, username);
    
    if (!sanitized.empty()) {
        // Continue with existing message processing using sanitized input...
    }
}

void LuminaChat::OnLLMResponse(std::string& response) {
    // Filter response with new blacklist system
    bool allowed = ReworkIntegration::FilterAssistantResponse(response);
    
    // Continue with existing response handling...
}

void LuminaChat::Stop() {
    // Cleanup new foundation components
    ReworkIntegration::CleanupFoundationComponents();
    
    // Continue with existing cleanup...
}
*/
