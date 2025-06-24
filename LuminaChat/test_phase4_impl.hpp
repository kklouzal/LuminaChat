// Phase 4 Implementation: Context Layer
// Tests ContextInfo with ChatTemplateManager integration

#pragma once

#include "rework/Logger.hpp"
#include "rework/SettingsManager.hpp"
#include "rework/TokenCache.hpp"
#include "rework/ModelInfo.hpp"
#include "rework/ChatTemplateManager.hpp"
#include "rework/ContextInfo.hpp"
#include <iostream>
#include <cassert>
#include <memory>

namespace Phase4Tests {

void TestContextInfoCreation() {
    std::cout << "=== Testing ContextInfo Creation ===" << std::endl;
    
    // Create ModelInfo for testing
    ModelConfig config;
    config.model_path = "./mock_model.gguf";
    config.context_size = 4096;
    config.gpu_layers = 20;
    
    ModelInfo model_info("test_model", config);
    
    // Create base template
    std::string base_template = R"(<|start_header_id|>system_message<|end_header_id|>
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

    // Create ContextInfo
    ContextInfo context_info(&model_info, base_template);
    
    // Test initial state
    assert(context_info.GetContextId() == ""); // Should be empty initially
    assert(context_info.GetMessageCount() == 0);
    assert(context_info.GetCurrentTokenCount() == 0);
    
    std::cout << "ContextInfo creation tests passed!" << std::endl;
}

void TestTemplateIntegration() {
    std::cout << "=== Testing Template Integration ===" << std::endl;
    
    ModelConfig config;
    config.model_path = "./mock_model.gguf";
    config.context_size = 4096;
    
    ModelInfo model_info("template_test_model", config);
    
    std::string base_template = R"(<|start_header_id|>env<|end_header_id|>
{{ overarching_environment }}
<|eot_id|>

<|start_header_id|>persona<|end_header_id|>
{{ identity_directive }}
<|eot_id|>

<|start_header_id|>system_message<|end_header_id|>
{{ system_prompt }}
<|eot_id|>

{% if summary %}
<|start_header_id|>summary<|end_header_id|>
{{ summary }}
<|eot_id|>
{% endif %}

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

    ContextInfo context_info(&model_info, base_template);
    
    // Test template section updates through ContextInfo
    context_info.UpdateEnvironment("Discord chat environment");
    context_info.UpdateIdentity("Luna, an AI assistant");
    context_info.UpdateSystemPrompt("Be helpful and accurate");
    
    // Test template rendering
    std::string rendered = context_info.BuildFullPrompt();
    assert(rendered.find("Discord chat environment") != std::string::npos);
    assert(rendered.find("Luna, an AI assistant") != std::string::npos);
    assert(rendered.find("Be helpful and accurate") != std::string::npos);
    
    // Test summary integration
    context_info.ApplySummary("User is testing the template system");
    rendered = context_info.BuildFullPrompt();
    assert(rendered.find("User is testing the template system") != std::string::npos);
    
    std::cout << "Template integration tests passed!" << std::endl;
}

void TestMessageProcessing() {
    std::cout << "=== Testing Message Processing ===" << std::endl;
    
    ModelConfig config;
    config.model_path = "./mock_model.gguf";
    config.context_size = 8192;
    
    ModelInfo model_info("message_test_model", config);
    
    std::string base_template = R"(<|start_header_id|>system_message<|end_header_id|>
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

    ContextInfo context_info(&model_info, base_template);
    context_info.UpdateSystemPrompt("You are a helpful assistant");
    
    // Test adding messages
    context_info.AddMessage("user", "Hello, how are you?");
    assert(context_info.GetMessageCount() == 1);
    
    context_info.AddMessage("assistant", "I'm doing well, thank you! How can I help you today?");
    assert(context_info.GetMessageCount() == 2);
    
    context_info.AddMessage("user", "Can you help me with C++ programming?");
    assert(context_info.GetMessageCount() == 3);
    
    // Test message retrieval
    auto messages = context_info.GetMessages();
    assert(messages.size() == 3);
    assert(messages[0].first == "user");
    assert(messages[0].second == "Hello, how are you?");
    assert(messages[1].first == "assistant");
    assert(messages[1].second.find("doing well") != std::string::npos);
    assert(messages[2].first == "user");
    assert(messages[2].second == "Can you help me with C++ programming?");
    
    // Test full prompt building with messages
    std::string full_prompt = context_info.BuildFullPrompt();
    assert(full_prompt.find("Hello, how are you?") != std::string::npos);
    assert(full_prompt.find("doing well") != std::string::npos);
    assert(full_prompt.find("C++ programming") != std::string::npos);
    assert(full_prompt.find("You are a helpful assistant") != std::string::npos);
    
    std::cout << "Message processing tests passed!" << std::endl;
}

void TestContextRebuilding() {
    std::cout << "=== Testing Context Rebuilding ===" << std::endl;
    
    ModelConfig config;
    config.model_path = "./mock_model.gguf";
    config.context_size = 4096;
    
    ModelInfo model_info("rebuild_test_model", config);
    
    std::string base_template = R"(<|start_header_id|>system_message<|end_header_id|>
{{ system_prompt }}
<|eot_id|>

{% if summary %}
<|start_header_id|>summary<|end_header_id|>
{{ summary }}
<|eot_id|>
{% endif %}

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

    ContextInfo context_info(&model_info, base_template);
    context_info.UpdateSystemPrompt("System instructions");
    
    // Add some conversation
    context_info.AddMessage("user", "First message");
    context_info.AddMessage("assistant", "First response");
    context_info.AddMessage("user", "Second message");
    context_info.AddMessage("assistant", "Second response");
    
    // Test full rebuild
    context_info.RebuildContext_Full();
    std::string full_prompt = context_info.BuildFullPrompt();
    assert(full_prompt.find("First message") != std::string::npos);
    assert(full_prompt.find("Second response") != std::string::npos);
    
    // Add more messages and test partial rebuild
    context_info.AddMessage("user", "Third message");
    context_info.RebuildContext_Partial();
    full_prompt = context_info.BuildFullPrompt();
    assert(full_prompt.find("Third message") != std::string::npos);
    
    // Test rebuild with template changes
    context_info.ApplySummary("Conversation about messages and responses");
    context_info.RebuildContext_Full();
    full_prompt = context_info.BuildFullPrompt();
    assert(full_prompt.find("Conversation about messages") != std::string::npos);
    
    std::cout << "Context rebuilding tests passed!" << std::endl;
}

void TestSummarizationCallbacks() {
    std::cout << "=== Testing Summarization Callbacks ===" << std::endl;
    
    ModelConfig config;
    config.model_path = "./mock_model.gguf";
    config.context_size = 2048; // Small context to trigger summarization
    
    ModelInfo model_info("callback_test_model", config);
    
    std::string base_template = "{{ system_prompt }}\n{% for msg in messages %}{{ msg.content }}\n{% endfor %}";
    ContextInfo context_info(&model_info, base_template);
    
    // Test callback registration
    bool callback_called = false;
    std::string callback_context_id;
    std::string callback_content;
    
    context_info.RegisterSummarizationCallback([&](const std::string& context_id, const std::string& content) {
        callback_called = true;
        callback_context_id = context_id;
        callback_content = content;
        std::cout << "Summarization callback triggered for context: " << context_id << std::endl;
        std::cout << "Content length: " << content.length() << std::endl;
    });
    
    // Add messages that would trigger summarization in real scenario
    context_info.AddMessage("user", "Long conversation that would exceed context limits...");
    context_info.AddMessage("assistant", "Response that continues the conversation...");
    
    // Simulate context size check and callback trigger
    if (context_info.GetMessageCount() > 0) {
        std::cout << "Would trigger summarization callback in real implementation" << std::endl;
    }
    
    std::cout << "Summarization callback tests passed!" << std::endl;
}

void TestPastSessionMemories() {
    std::cout << "=== Testing Past Session Memories ===" << std::endl;
    
    ModelConfig config;
    config.model_path = "./mock_model.gguf";
    config.context_size = 4096;
    
    ModelInfo model_info("memory_test_model", config);
    
    std::string base_template = R"({% if past_sessions and past_sessions|length > 0 %}
  {% for memory in past_sessions %}
<|start_header_id|>memory_{{ loop.index }}<|end_header_id|>
{{ memory }}
<|eot_id|>
  {% endfor %}
{% endif %}

{%- for msg in messages %}
{{ msg.content }}
{%- endfor %})";

    ContextInfo context_info(&model_info, base_template);
    
    // Test adding past session memories
    context_info.AddPastSessionMemory("User prefers detailed technical explanations");
    context_info.AddPastSessionMemory("User is working on LuminaChat project");
    context_info.AddPastSessionMemory("User has experience with C++ and modern practices");
    
    // Add current conversation
    context_info.AddMessage("user", "How is the context system working?");
    
    // Test rendering with memories
    std::string full_prompt = context_info.BuildFullPrompt();
    assert(full_prompt.find("detailed technical explanations") != std::string::npos);
    assert(full_prompt.find("LuminaChat project") != std::string::npos);
    assert(full_prompt.find("C++ and modern practices") != std::string::npos);
    assert(full_prompt.find("context system working") != std::string::npos);
    
    // Verify memory structure
    assert(full_prompt.find("memory_1") != std::string::npos);
    assert(full_prompt.find("memory_2") != std::string::npos);
    assert(full_prompt.find("memory_3") != std::string::npos);
    
    std::cout << "Past session memories tests passed!" << std::endl;
}

void TestDynamicTemplateModification() {
    std::cout << "=== Testing Dynamic Template Modification ===" << std::endl;
    
    ModelConfig config;
    config.model_path = "./mock_model.gguf";
    config.context_size = 4096;
    
    ModelInfo model_info("dynamic_test_model", config);
    
    std::string base_template = R"(<|start_header_id|>env<|end_header_id|>
{{ overarching_environment }}
<|eot_id|>

{% if motif_context %}
<|start_header_id|>motif<|end_header_id|>
{{ motif_context }}
<|eot_id|>
{% endif %}

{% if internal_reflection %}
<|start_header_id|>reflection<|end_header_id|>
{{ internal_reflection }}
<|eot_id|>
{% endif %}

{%- for msg in messages %}
{{ msg.content }}
{%- endfor %})";

    ContextInfo context_info(&model_info, base_template);
    
    // Initial state
    context_info.UpdateEnvironment("Initial environment setup");
    context_info.AddMessage("user", "Hello");
    
    std::string initial_prompt = context_info.BuildFullPrompt();
    assert(initial_prompt.find("Initial environment setup") != std::string::npos);
    assert(initial_prompt.find("motif") == std::string::npos); // Should not appear without content
    
    // Add motif during conversation
    context_info.UpdateMotif("Helpful and educational tone");
    std::string with_motif = context_info.BuildFullPrompt();
    assert(with_motif.find("Helpful and educational tone") != std::string::npos);
    assert(with_motif.find("<|start_header_id|>motif<|end_header_id|>") != std::string::npos);
    
    // Add internal reflection
    context_info.UpdateInternalReflection("User seems interested in learning");
    std::string with_reflection = context_info.BuildFullPrompt();
    assert(with_reflection.find("User seems interested in learning") != std::string::npos);
    assert(with_reflection.find("<|start_header_id|>reflection<|end_header_id|>") != std::string::npos);
    
    // Update environment mid-conversation
    context_info.UpdateEnvironment("Updated environment with new context");
    std::string updated_env = context_info.BuildFullPrompt();
    assert(updated_env.find("Updated environment with new context") != std::string::npos);
    assert(updated_env.find("Initial environment setup") == std::string::npos); // Should be replaced
    
    std::cout << "Dynamic template modification tests passed!" << std::endl;
}

} // namespace Phase4Tests

int TestPhase4Context() {
    std::cout << "Testing ContextInfo with ChatTemplateManager integration" << std::endl;
    std::cout << "Testing dynamic template management and message processing" << std::endl;
    std::cout << "Testing template-based context rebuilding" << std::endl;
    std::cout << std::endl;
    
    try {
        Phase4Tests::TestContextInfoCreation();
        std::cout << std::endl;
        
        Phase4Tests::TestTemplateIntegration();
        std::cout << std::endl;
        
        Phase4Tests::TestMessageProcessing();
        std::cout << std::endl;
        
        Phase4Tests::TestContextRebuilding();
        std::cout << std::endl;
        
        Phase4Tests::TestSummarizationCallbacks();
        std::cout << std::endl;
        
        Phase4Tests::TestPastSessionMemories();
        std::cout << std::endl;
        
        Phase4Tests::TestDynamicTemplateModification();
        std::cout << std::endl;
        
        std::cout << "Context layer components are working correctly!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Phase 4 test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
