// Phase 3 Implementation: Template Layer
// Tests ChatTemplateManager with Jinja2-style rendering

#pragma once

#include "rework/Logger.hpp"
#include "rework/ChatTemplateManager.hpp"
#include <iostream>
#include <cassert>
#include <vector>

namespace Phase3Tests {

void TestBasicTemplateCreation() {
    std::cout << "=== Testing Basic Template Creation ===" << std::endl;
    
    // Create a basic Llama3-style template
    std::string base_template = R"({{- bos_token }}

<|start_header_id|>env<|end_header_id|>
{{ overarching_environment }}
<|eot_id|>

<|start_header_id|>persona<|end_header_id|>
{{ identity_directive }}
<|eot_id|>

<|start_header_id|>system_message<|end_header_id|>
{{ system_prompt }}
<|eot_id|>

{% if old_chat_summary %}
<|start_header_id|>old_chat_summary<|end_header_id|>
{{ old_chat_summary }}
<|eot_id|>
{% endif %}

{% if past_sessions and past_sessions|length > 0 %}
  {% for memory in past_sessions %}
<|start_header_id|>memory_{{ loop.index }}<|end_header_id|>
{{ memory }}
<|eot_id|>
  {% endfor %}
{% endif %}

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

    ChatTemplateManager template_manager(base_template);
    
    // Test basic validation
    assert(template_manager.ValidateTemplate());
    assert(template_manager.IsTemplateDirty()); // Should be dirty initially
    
    std::cout << "Basic template creation tests passed!" << std::endl;
}

void TestSectionManagement() {
    std::cout << "=== Testing Section Management ===" << std::endl;
    
    std::string base_template = "{{ overarching_environment }}\n{{ identity_directive }}\n{{ system_prompt }}";
    ChatTemplateManager template_manager(base_template);
    
    // Test setting sections
    template_manager.SetSection(TemplateSection::OVERARCHING_ENVIRONMENT, 
        "You are in a Discord chat environment with multiple users.");
    template_manager.SetSection(TemplateSection::IDENTITY_DIRECTIVE, 
        "You are Luna, a helpful AI assistant.");
    template_manager.SetSection(TemplateSection::SYSTEM_PROMPT, 
        "Be helpful, harmless, and honest in all responses.");
    
    // Test convenience methods
    template_manager.UpdateEnvironment("Multi-user chat environment");
    template_manager.UpdateIdentity("Luna, an AI assistant");
    template_manager.UpdateSystemPrompt("Core system instructions");
    
    // Test section activation/deactivation
    template_manager.ActivateSection(TemplateSection::SUMMARY);
    template_manager.UpdateSummary("Current conversation summary");
    
    template_manager.DeactivateSection(TemplateSection::SUMMARY);
    
    // Test clearing sections
    template_manager.ClearSection(TemplateSection::MOTIF_CONTEXT);
    
    assert(template_manager.IsTemplateDirty()); // Should be dirty after modifications
    
    std::cout << "Section management tests passed!" << std::endl;
}

void TestPastSessionsArray() {
    std::cout << "=== Testing Past Sessions Array ===" << std::endl;
    
    std::string base_template = "{% for memory in past_sessions %}{{ memory }}\n{% endfor %}";
    ChatTemplateManager template_manager(base_template);
    
    // Test adding past sessions
    template_manager.AddPastSession("Memory from last week: User discussed AI ethics");
    template_manager.AddPastSession("Memory from yesterday: User asked about C++ programming");
    template_manager.AddPastSession("Memory from this morning: User mentioned working on LuminaChat");
    
    // Create test messages
    std::vector<std::pair<std::string, std::string>> messages = {
        {"user", "Hello Luna!"},
        {"assistant", "Hello! How can I help you today?"}
    };
    
    // Test rendering with past sessions
    std::string rendered = template_manager.RenderTemplate(messages);
    assert(rendered.find("AI ethics") != std::string::npos);
    assert(rendered.find("C++ programming") != std::string::npos);
    assert(rendered.find("LuminaChat") != std::string::npos);
    
    // Test clearing past sessions
    template_manager.ClearPastSessions();
    rendered = template_manager.RenderTemplate(messages);
    // After clearing, the memories should not appear
    
    std::cout << "Past sessions array tests passed!" << std::endl;
}

void TestMessageIntegration() {
    std::cout << "=== Testing Message Integration ===" << std::endl;
    
    std::string base_template = R"({%- for msg in messages %}
{% if msg.role == "assistant" %}
ASSISTANT: {{ msg.content | trim }}
{% else %}
USER: {{ msg.content | trim }}
{% endif %}
{%- endfor %})";

    ChatTemplateManager template_manager(base_template);
    
    // Create test conversation
    std::vector<std::pair<std::string, std::string>> messages = {
        {"user", "What is the weather like?"},
        {"assistant", "I don't have access to real-time weather data, but I can help you find weather information."},
        {"user", "That's okay, can you help me with programming instead?"},
        {"assistant", "Absolutely! I'd be happy to help with programming. What specific topic or language are you interested in?"},
        {"user", "I'm working on a C++ project called LuminaChat."}
    };
    
    // Test rendering with messages
    std::string rendered = template_manager.RenderTemplate(messages);
    
    // Verify all messages are present
    assert(rendered.find("What is the weather like?") != std::string::npos);
    assert(rendered.find("real-time weather data") != std::string::npos);
    assert(rendered.find("programming instead") != std::string::npos);
    assert(rendered.find("C++ project called LuminaChat") != std::string::npos);
    
    // Verify role formatting
    assert(rendered.find("USER: What is the weather") != std::string::npos);
    assert(rendered.find("ASSISTANT: I don't have access") != std::string::npos);
    
    std::cout << "Message integration tests passed!" << std::endl;
}

void TestComplexTemplateRendering() {
    std::cout << "=== Testing Complex Template Rendering ===" << std::endl;
    
    // Full Llama3-style template
    std::string base_template = R"(<|start_header_id|>env<|end_header_id|>
{{ overarching_environment }}
<|eot_id|>

<|start_header_id|>persona<|end_header_id|>
{{ identity_directive }}
<|eot_id|>

<|start_header_id|>system_message<|end_header_id|>
{{ system_prompt }}
<|eot_id|>

{% if old_chat_summary %}
<|start_header_id|>old_chat_summary<|end_header_id|>
{{ old_chat_summary }}
<|eot_id|>
{% endif %}

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

    ChatTemplateManager template_manager(base_template);
    
    // Set up all sections
    template_manager.UpdateEnvironment("Discord multi-user chat environment");
    template_manager.UpdateIdentity("You are Luna, a helpful AI assistant specialized in programming and technical support");
    template_manager.UpdateSystemPrompt("Provide helpful, accurate, and concise responses. Be friendly but professional.");
    template_manager.UpdateSummary("User is working on LuminaChat project and discussing template system implementation");
    template_manager.UpdateOldChatSummary("Previous conversation covered initial project setup and architecture planning");
    
    // Add some past sessions
    template_manager.AddPastSession("User has experience with C++ and is familiar with modern practices");
    template_manager.AddPastSession("User values clean architecture and maintainable code");
    
    // Create conversation
    std::vector<std::pair<std::string, std::string>> messages = {
        {"user", "How is the template system working so far?"},
        {"assistant", "The template system is working well! The Jinja2-style rendering allows for dynamic section management."},
        {"user", "Great! Can you show me how the sections are organized?"}
    };
    
    // Render the complete template
    std::string rendered = template_manager.RenderTemplate(messages);
    
    // Verify all components are present
    assert(rendered.find("Discord multi-user chat") != std::string::npos);
    assert(rendered.find("Luna, a helpful AI assistant") != std::string::npos);
    assert(rendered.find("Provide helpful, accurate") != std::string::npos);
    assert(rendered.find("LuminaChat project") != std::string::npos);
    assert(rendered.find("Previous conversation covered") != std::string::npos);
    assert(rendered.find("C++ and is familiar") != std::string::npos);
    assert(rendered.find("template system working") != std::string::npos);
    
    // Verify proper structure
    assert(rendered.find("<|start_header_id|>env<|end_header_id|>") != std::string::npos);
    assert(rendered.find("<|start_header_id|>persona<|end_header_id|>") != std::string::npos);
    assert(rendered.find("<|start_header_id|>system_message<|end_header_id|>") != std::string::npos);
    assert(rendered.find("<|start_header_id|>summary<|end_header_id|>") != std::string::npos);
    assert(rendered.find("<|start_header_id|>old_chat_summary<|end_header_id|>") != std::string::npos);
    assert(rendered.find("<|start_header_id|>assistant<|end_header_id|>") != std::string::npos);
    
    std::cout << "Complex template rendering tests passed!" << std::endl;
}

void TestTemplateCaching() {
    std::cout << "=== Testing Template Caching ===" << std::endl;
    
    std::string base_template = "{{ system_prompt }}\n{% for msg in messages %}{{ msg.content }}\n{% endfor %}";
    ChatTemplateManager template_manager(base_template);
    
    template_manager.UpdateSystemPrompt("System instructions");
    
    std::vector<std::pair<std::string, std::string>> messages = {
        {"user", "Hello"},
        {"assistant", "Hi there!"}
    };
    
    // First render - should create cache
    assert(template_manager.IsTemplateDirty());
    std::string first_render = template_manager.RenderTemplate(messages);
    assert(!template_manager.IsTemplateDirty()); // Should be clean after render
    
    // Second render with same data - should use cache
    std::string second_render = template_manager.RenderTemplate(messages);
    assert(first_render == second_render);
    assert(!template_manager.IsTemplateDirty()); // Should still be clean
    
    // Modify template - should mark dirty
    template_manager.UpdateSystemPrompt("Modified system instructions");
    assert(template_manager.IsTemplateDirty()); // Should be dirty after modification
    
    // Render after modification - should be different
    std::string third_render = template_manager.RenderTemplate(messages);
    assert(first_render != third_render);
    assert(!template_manager.IsTemplateDirty()); // Should be clean after render
    
    std::cout << "Template caching tests passed!" << std::endl;
}

void TestConditionalRendering() {
    std::cout << "=== Testing Conditional Rendering ===" << std::endl;
    
    std::string base_template = R"({% if summary %}
SUMMARY: {{ summary }}
{% endif %}
{% if old_chat_summary %}
OLD_SUMMARY: {{ old_chat_summary }}
{% endif %}
{% if motif_context %}
MOTIF: {{ motif_context }}
{% endif %})";

    ChatTemplateManager template_manager(base_template);
    
    std::vector<std::pair<std::string, std::string>> messages = {
        {"user", "Test message"}
    };
    
    // Test with no sections active
    std::string rendered = template_manager.RenderTemplate(messages);
    assert(rendered.find("SUMMARY:") == std::string::npos);
    assert(rendered.find("OLD_SUMMARY:") == std::string::npos);
    assert(rendered.find("MOTIF:") == std::string::npos);
    
    // Test with summary active
    template_manager.UpdateSummary("Current session summary");
    rendered = template_manager.RenderTemplate(messages);
    assert(rendered.find("SUMMARY: Current session summary") != std::string::npos);
    assert(rendered.find("OLD_SUMMARY:") == std::string::npos);
    assert(rendered.find("MOTIF:") == std::string::npos);
    
    // Test with multiple sections active
    template_manager.UpdateOldChatSummary("Previous chat summary");
    template_manager.UpdateMotifContext("Helpful and educational tone");
    rendered = template_manager.RenderTemplate(messages);
    assert(rendered.find("SUMMARY: Current session summary") != std::string::npos);
    assert(rendered.find("OLD_SUMMARY: Previous chat summary") != std::string::npos);
    assert(rendered.find("MOTIF: Helpful and educational tone") != std::string::npos);
    
    // Test deactivating sections
    template_manager.DeactivateSection(TemplateSection::SUMMARY);
    rendered = template_manager.RenderTemplate(messages);
    assert(rendered.find("SUMMARY:") == std::string::npos);
    assert(rendered.find("OLD_SUMMARY: Previous chat summary") != std::string::npos);
    assert(rendered.find("MOTIF: Helpful and educational tone") != std::string::npos);
    
    std::cout << "Conditional rendering tests passed!" << std::endl;
}

} // namespace Phase3Tests

int TestPhase3Template() {
    std::cout << "Testing ChatTemplateManager with Jinja2-style rendering" << std::endl;
    std::cout << "Testing template section management and caching" << std::endl;
    std::cout << "Testing template rendering with message integration" << std::endl;
    std::cout << std::endl;
    
    try {
        Phase3Tests::TestBasicTemplateCreation();
        std::cout << std::endl;
        
        Phase3Tests::TestSectionManagement();
        std::cout << std::endl;
        
        Phase3Tests::TestPastSessionsArray();
        std::cout << std::endl;
        
        Phase3Tests::TestMessageIntegration();
        std::cout << std::endl;
        
        Phase3Tests::TestComplexTemplateRendering();
        std::cout << std::endl;
        
        Phase3Tests::TestTemplateCaching();
        std::cout << std::endl;
        
        Phase3Tests::TestConditionalRendering();
        std::cout << std::endl;
        
        std::cout << "Template layer components are working correctly!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Phase 3 test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
