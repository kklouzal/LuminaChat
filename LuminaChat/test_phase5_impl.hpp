// Phase 5 Implementation: Management Layer
// Tests LlamaManager as resource coordinator with integration testing of model/context/template lifecycle

#pragma once

#include "rework/Logger.hpp"
#include "rework/SettingsManager.hpp"
#include "rework/TokenCache.hpp"
#include "rework/ModelInfo.hpp"
#include "rework/ChatTemplateManager.hpp"
#include "rework/ContextInfo.hpp"
#include "rework/LlamaManager.hpp"
#include <iostream>
#include <cassert>
#include <memory>
#include <chrono>
#include <thread>

namespace Phase5Tests {

void TestLlamaManagerCreation() {
    std::cout << "=== Testing LlamaManager Creation ===" << std::endl;
    
    // Test basic creation
    LlamaManager manager;
    assert(!manager.IsReady()); // Should not be ready until initialized
    
    std::cout << "✓ LlamaManager created successfully" << std::endl;
    
    // Test initialization
    assert(manager.Initialize());
    assert(manager.IsReady());
    
    std::cout << "✓ LlamaManager initialized successfully" << std::endl;
    
    // Test statistics
    const auto& stats = manager.GetStats();
    assert(stats.total_models == 0);
    assert(stats.total_contexts == 0);
    assert(stats.loaded_models == 0);
    assert(stats.active_contexts == 0);
    
    std::cout << "✓ Initial statistics correct" << std::endl;
}

void TestModelManagement() {
    std::cout << "=== Testing Model Management ===" << std::endl;
    
    LlamaManager manager;
    assert(manager.Initialize());
    
    // Test model creation
    ModelInfo* model1 = manager.GetOrCreateModelInfo("test_model_1");
    assert(model1 != nullptr);
    assert(manager.HasModel("test_model_1"));
    
    std::cout << "✓ Model creation successful" << std::endl;
    
    // Test retrieving existing model
    ModelInfo* model1_again = manager.GetOrCreateModelInfo("test_model_1");
    assert(model1_again == model1); // Should be same instance
    
    std::cout << "✓ Model retrieval successful" << std::endl;
    
    // Test multiple models
    ModelInfo* model2 = manager.GetOrCreateModelInfo("test_model_2");
    assert(model2 != nullptr);
    assert(model2 != model1);
    assert(manager.HasModel("test_model_2"));
    
    std::cout << "✓ Multiple model management successful" << std::endl;
    
    // Test statistics update
    const auto& stats = manager.GetStats();
    assert(stats.total_models == 2);
    
    std::cout << "✓ Model statistics tracking successful" << std::endl;
    
    // Test model listing
    auto model_ids = manager.GetLoadedModelIds();
    assert(model_ids.size() == 2);
    assert(std::find(model_ids.begin(), model_ids.end(), "test_model_1") != model_ids.end());
    assert(std::find(model_ids.begin(), model_ids.end(), "test_model_2") != model_ids.end());
    
    std::cout << "✓ Model listing successful" << std::endl;
}

void TestContextManagement() {
    std::cout << "=== Testing Context Management ===" << std::endl;
    
    LlamaManager manager;
    assert(manager.Initialize());
    
    // Register a default template for testing
    std::string test_template = R"(<|start_header_id|>system_message<|end_header_id|>
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
    
    assert(manager.RegisterTemplate("test_template", test_template));
    
    std::cout << "✓ Template registration successful" << std::endl;
    
    // Test context creation - this should automatically create the model too
    ContextInfo* context1 = manager.GetOrCreateContextInfo("test_context_1", "test_model_1", "test_template");
    assert(context1 != nullptr);
    assert(manager.HasContext("test_context_1"));
    assert(manager.HasModel("test_model_1")); // Model should be auto-created
    
    std::cout << "✓ Context creation with auto-model creation successful" << std::endl;
    
    // Test retrieving existing context
    ContextInfo* context1_again = manager.GetOrCreateContextInfo("test_context_1", "test_model_1", "test_template");
    assert(context1_again == context1); // Should be same instance
    
    std::cout << "✓ Context retrieval successful" << std::endl;
    
    // Test multiple contexts with same model
    ContextInfo* context2 = manager.GetOrCreateContextInfo("test_context_2", "test_model_1", "test_template");
    assert(context2 != nullptr);
    assert(context2 != context1);
    assert(manager.HasContext("test_context_2"));
    
    std::cout << "✓ Multiple contexts with same model successful" << std::endl;
    
    // Test context with different model
    ContextInfo* context3 = manager.GetOrCreateContextInfo("test_context_3", "test_model_2", "test_template");
    assert(context3 != nullptr);
    assert(manager.HasModel("test_model_2")); // Second model should be auto-created
    
    std::cout << "✓ Context with different model successful" << std::endl;
    
    // Test statistics
    const auto& stats = manager.GetStats();
    assert(stats.total_models == 2);
    assert(stats.total_contexts == 3);
    
    std::cout << "✓ Context statistics tracking successful" << std::endl;
    
    // Test context listing
    auto context_ids = manager.GetActiveContextIds();
    assert(context_ids.size() == 3);
    assert(std::find(context_ids.begin(), context_ids.end(), "test_context_1") != context_ids.end());
    assert(std::find(context_ids.begin(), context_ids.end(), "test_context_2") != context_ids.end());
    assert(std::find(context_ids.begin(), context_ids.end(), "test_context_3") != context_ids.end());
    
    std::cout << "✓ Context listing successful" << std::endl;
}

void TestTemplateManagement() {
    std::cout << "=== Testing Template Management ===" << std::endl;
    
    LlamaManager manager;
    assert(manager.Initialize());
    
    // Test template registration
    std::string simple_template = "Hello {{ name }}!";
    assert(manager.RegisterTemplate("simple", simple_template));
    assert(manager.HasTemplate("simple"));
    
    std::cout << "✓ Template registration successful" << std::endl;
    
    // Test template retrieval
    std::string retrieved_template = manager.GetTemplate("simple");
    assert(retrieved_template == simple_template);
    
    std::cout << "✓ Template retrieval successful" << std::endl;
    
    // Test multiple templates
    std::string complex_template = R"(
System: {{ system_message }}
{% for msg in messages %}
{{ msg.role }}: {{ msg.content }}
{% endfor %}
)";
    
    assert(manager.RegisterTemplate("complex", complex_template));
    assert(manager.HasTemplate("complex"));
    assert(manager.HasTemplate("simple")); // First template should still exist
    
    std::cout << "✓ Multiple template management successful" << std::endl;
    
    // Test template listing
    auto templates = manager.GetAvailableTemplates();
    assert(templates.size() >= 2); // At least our two + any defaults
    assert(std::find(templates.begin(), templates.end(), "simple") != templates.end());
    assert(std::find(templates.begin(), templates.end(), "complex") != templates.end());
    
    std::cout << "✓ Template listing successful" << std::endl;
    
    // Test using template with context creation
    ContextInfo* context = manager.GetOrCreateContextInfo("template_test_context", "template_test_model", "complex");
    assert(context != nullptr);
    
    std::cout << "✓ Template integration with context creation successful" << std::endl;
}

void TestLifecycleManagement() {
    std::cout << "=== Testing Lifecycle Management ===" << std::endl;
    
    LlamaManager manager;
    assert(manager.Initialize());
    
    // Create some resources
    assert(manager.RegisterTemplate("lifecycle_template", "Test template: {{ content }}"));
    
    ModelInfo* model = manager.GetOrCreateModelInfo("lifecycle_model");
    assert(model != nullptr);
    
    ContextInfo* context = manager.GetOrCreateContextInfo("lifecycle_context", "lifecycle_model", "lifecycle_template");
    assert(context != nullptr);
    
    // Verify they exist
    assert(manager.HasModel("lifecycle_model"));
    assert(manager.HasContext("lifecycle_context"));
    assert(manager.HasTemplate("lifecycle_template"));
    
    std::cout << "✓ Resource creation successful" << std::endl;
    
    // Test individual removal
    assert(manager.RemoveContext("lifecycle_context"));
    assert(!manager.HasContext("lifecycle_context"));
    assert(manager.HasModel("lifecycle_model")); // Model should still exist
    
    std::cout << "✓ Individual context removal successful" << std::endl;
    
    // Recreate context and test full cleanup
    context = manager.GetOrCreateContextInfo("lifecycle_context_2", "lifecycle_model", "lifecycle_template");
    assert(context != nullptr);
    
    // Test full cleanup
    manager.Cleanup();
    assert(!manager.IsReady());
    
    // After cleanup, resources should be gone
    assert(!manager.HasModel("lifecycle_model"));
    assert(!manager.HasContext("lifecycle_context_2"));
    
    std::cout << "✓ Full cleanup successful" << std::endl;
    
    // Test reinitialization
    assert(manager.Initialize());
    assert(manager.IsReady());
    
    const auto& stats = manager.GetStats();
    assert(stats.total_models == 0);
    assert(stats.total_contexts == 0);
    
    std::cout << "✓ Reinitialization after cleanup successful" << std::endl;
}

void TestResourceCallbacks() {
    std::cout << "=== Testing Resource Callbacks ===" << std::endl;
    
    LlamaManager manager;
    assert(manager.Initialize());
    
    std::vector<std::pair<std::string, std::string>> events;
    
    // Register callback to capture events
    manager.RegisterResourceCallback([&events](const std::string& event, const std::string& details) {
        events.emplace_back(event, details);
    });
    
    std::cout << "✓ Callback registration successful" << std::endl;
    
    // Create resources and verify events
    size_t initial_event_count = events.size();
    
    manager.RegisterTemplate("callback_template", "Callback test: {{ data }}");
    
    ModelInfo* model = manager.GetOrCreateModelInfo("callback_model");
    assert(model != nullptr);
    
    ContextInfo* context = manager.GetOrCreateContextInfo("callback_context", "callback_model", "callback_template");
    assert(context != nullptr);
    
    // Should have generated model_created and context_created events
    assert(events.size() > initial_event_count);
    
    bool found_model_event = false;
    bool found_context_event = false;
    
    for (const auto& [event, details] : events) {
        if (event == "model_created" && details.find("callback_model") != std::string::npos) {
            found_model_event = true;
        }
        if (event == "context_created" && details.find("callback_context") != std::string::npos) {
            found_context_event = true;
        }
    }
    
    assert(found_model_event);
    assert(found_context_event);
    
    std::cout << "✓ Resource creation events captured successfully" << std::endl;
}

void TestErrorHandling() {
    std::cout << "=== Testing Error Handling ===" << std::endl;
    
    LlamaManager manager;
    assert(manager.Initialize());
    
    // Test invalid model IDs
    assert(manager.GetOrCreateModelInfo("") == nullptr); // Empty ID
    assert(manager.GetOrCreateModelInfo("invalid/model*id") == nullptr); // Invalid characters
    
    std::cout << "✓ Invalid model ID handling successful" << std::endl;
    
    // Test invalid context IDs
    assert(manager.GetOrCreateContextInfo("", "test_model", "default") == nullptr); // Empty ID
    assert(manager.GetOrCreateContextInfo("invalid/context*id", "test_model", "default") == nullptr); // Invalid characters
    
    std::cout << "✓ Invalid context ID handling successful" << std::endl;
    
    // Test context creation with nonexistent template (should succeed with default template)
    ContextInfo* context_with_default = manager.GetOrCreateContextInfo("test_context", "test_model", "nonexistent_template");
    assert(context_with_default != nullptr); // Should create with default template, not fail
    
    std::cout << "✓ Nonexistent template handling successful (uses default)" << std::endl;
    
    // Test operations on uninitialized manager
    LlamaManager uninitialized_manager;
    assert(!uninitialized_manager.IsReady());
    assert(uninitialized_manager.GetOrCreateModelInfo("test") == nullptr);
    assert(uninitialized_manager.GetOrCreateContextInfo("test", "model", "template") == nullptr);
    
    std::cout << "✓ Uninitialized manager handling successful" << std::endl;
}

void TestIntegrationWorkflow() {
    std::cout << "=== Testing Integration Workflow ===" << std::endl;
    
    // This test simulates a complete workflow using LlamaManager
    LlamaManager manager;
    assert(manager.Initialize());
    
    // Step 1: Register templates
    std::string chat_template = R"(<|start_header_id|>system_message<|end_header_id|>
You are Luna, a helpful AI assistant.
<|eot_id|>

{%- for msg in messages %}
  {% if msg.role == "assistant" %}
<|start_header_id|>assistant<|end_header_id|>
{{ msg.content | trim }}<|eot_id|>
  {% else %}
<|start_header_id|>user<|end_header_id|>
{{ msg.content | trim }}<|eot_id|>
  {% endif %}
{%- endfor %}

<|start_header_id|>assistant<|end_header_id|>)";
    
    std::string summary_template = R"(<|start_header_id|>system_message<|end_header_id|>
Please summarize the following conversation concisely.
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
{{ content_to_summarize }}
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>)";
    
    assert(manager.RegisterTemplate("chat", chat_template));
    assert(manager.RegisterTemplate("summary", summary_template));
    
    std::cout << "✓ Template registration complete" << std::endl;
    
    // Step 2: Create main model and context
    ContextInfo* main_context = manager.GetOrCreateContextInfo("main_chat", "main_model", "chat");
    assert(main_context != nullptr);
    
    std::cout << "✓ Main context creation complete" << std::endl;
    
    // Step 3: Create summarizer model and context
    ContextInfo* summary_context = manager.GetOrCreateContextInfo("summarizer", "summary_model", "summary");
    assert(summary_context != nullptr);
    
    std::cout << "✓ Summary context creation complete" << std::endl;
    
    // Step 4: Verify models were auto-created
    assert(manager.HasModel("main_model"));
    assert(manager.HasModel("summary_model"));
    
    std::cout << "✓ Auto-model creation verified" << std::endl;
    
    // Step 5: Test context functionality
    main_context->UpdateSystemPrompt("You are Luna, a helpful AI assistant in a Discord server.");
    main_context->AddMessage("user", "Hello, Luna!");
    
    // Verify message was added
    assert(main_context->GetMessageCount() == 1);
    const auto& messages = main_context->GetMessageHistory();
    assert(messages.size() == 1);
    assert(messages[0].first == "user");
    assert(messages[0].second == "Hello, Luna!");
    
    std::cout << "✓ Context message handling verified" << std::endl;
    
    // Step 6: Test template integration
    std::string prompt = main_context->BuildFullPrompt();
    assert(!prompt.empty());
    assert(prompt.find("Luna") != std::string::npos);
    assert(prompt.find("Hello, Luna!") != std::string::npos);
    
    std::cout << "✓ Template integration verified" << std::endl;
    
    // Step 7: Verify statistics
    const auto& stats = manager.GetStats();
    assert(stats.total_models == 2);
    assert(stats.total_contexts == 2);
    
    std::cout << "✓ Final statistics verified" << std::endl;
    
    std::cout << "✓ Complete integration workflow successful" << std::endl;
}

} // namespace Phase5Tests

// Main test function for Phase 5
int TestPhase5Management() {
    std::cout << "Starting Phase 5 Tests: Management Layer" << std::endl;
    std::cout << "Testing LlamaManager as resource coordinator" << std::endl;
    std::cout << std::endl;
    
    try {
        Phase5Tests::TestLlamaManagerCreation();
        std::cout << std::endl;
        
        Phase5Tests::TestModelManagement();
        std::cout << std::endl;
        
        Phase5Tests::TestContextManagement();
        std::cout << std::endl;
        
        Phase5Tests::TestTemplateManagement();
        std::cout << std::endl;
        
        Phase5Tests::TestLifecycleManagement();
        std::cout << std::endl;
        
        Phase5Tests::TestResourceCallbacks();
        std::cout << std::endl;
        
        Phase5Tests::TestErrorHandling();
        std::cout << std::endl;
        
        Phase5Tests::TestIntegrationWorkflow();
        std::cout << std::endl;
        
        std::cout << "🎉 All Phase 5 tests passed!" << std::endl;
        std::cout << "LlamaManager implementation is ready for Phase 6 (Plugin Foundation)" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Phase 5 test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Phase 5 test failed with unknown exception" << std::endl;
        return 1;
    }
}
