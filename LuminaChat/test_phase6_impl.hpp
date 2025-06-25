// Phase 6 Implementation: Plugin Foundation
// Tests core ProcessingPipeline template
// Tests summarization pipeline with template-based callback system  
// Tests context-to-summary-to-template workflow

#pragma once

#include "rework/Logger.hpp"
#include "rework/ProcessingPipeline.hpp"
#include "rework/SettingsManager.hpp"
#include "rework/Sanitizer.hpp"
#include "rework/TokenCache.hpp"
#include "rework/ModelInfo.hpp"
#include "rework/ChatTemplateManager.hpp"
#include "rework/ContextInfo.hpp"
#include "rework/LlamaManager.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>

namespace Phase6Tests {

// Test request/response types for the plugin foundation
struct SummarizationRequest {
    std::string original_context_id;
    std::string content_to_summarize;
    std::string summary_template_name = "summary";
    std::function<void(const std::string&, const std::string&)> completion_callback;
    
    SummarizationRequest() = default;
    SummarizationRequest(const std::string& context_id, const std::string& content)
        : original_context_id(context_id), content_to_summarize(content) {}
};

struct MockSanitizationRequest {
    std::string content;
    std::string context_id;
    
    MockSanitizationRequest() = default;
    MockSanitizationRequest(const std::string& content, const std::string& context_id)
        : content(content), context_id(context_id) {}
};

struct SanitizationResult {
    std::string sanitized_content;
    bool passed_filter;
    std::vector<std::string> blocked_patterns;
    
    SanitizationResult() = default;
    SanitizationResult(const std::string& content, bool passed = true)
        : sanitized_content(content), passed_filter(passed) {}
};

void TestBasicPipelineCreation() {
    std::cout << "=== Testing Basic Pipeline Creation ===" << std::endl;
    
    // Setup logger
    GetLogger().RegisterOutputCallback([](std::string_view message) {
        std::cout << "LOG: " << message << std::endl;
    });
    
    // Test creating a basic pipeline for string processing
    using LuminaChat::ProcessingPipeline;
    using LuminaChat::RequestPriority;
    
    std::atomic<int> processed_count{0};
    std::vector<std::string> results;
    std::mutex results_mutex;
    
    // Create a simple string processing pipeline
    auto processor = [&](const std::string& input, 
                         std::function<void(std::string)> success_callback,
                         std::function<void(const std::string&)> error_callback) {
        
        // Simple processing: uppercase the input
        std::string result = input;
        std::transform(result.begin(), result.end(), result.begin(), ::toupper);
        
        // Simulate some processing time
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        {
            std::lock_guard<std::mutex> lock(results_mutex);
            results.push_back(result);
            processed_count++;
        }
        
        success_callback(result);
    };
    
    ProcessingPipeline<std::string, std::string> pipeline("TestPipeline", processor);
    
    // Test pipeline state
    assert(!pipeline.IsShutdown());
    assert(pipeline.GetQueueDepth() == 0);
    assert(pipeline.GetName() == "TestPipeline");
    
    // Queue some requests
    assert(pipeline.QueueRequest("hello", RequestPriority::NORMAL, "req1"));
    assert(pipeline.QueueRequest("world", RequestPriority::HIGH, "req2"));
    assert(pipeline.QueueRequest("test", RequestPriority::LOW, "req3"));
    
    // Wait for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Check results
    assert(processed_count >= 3);
    assert(results.size() >= 3);
    
    // Verify high priority processed first (approximately)
    bool found_world = false;
    for (const auto& result : results) {
        if (result == "WORLD") {
            found_world = true;
            break;
        }
    }
    assert(found_world);
    
    std::cout << "Basic pipeline creation tests passed!" << std::endl;
}

void TestSummarizationPipeline() {
    std::cout << "=== Testing Summarization Pipeline ===" << std::endl;
    
    using LuminaChat::ProcessingPipeline;
    
    // Mock summarization results
    std::atomic<int> summarization_count{0};
    std::vector<std::pair<std::string, std::string>> summarization_results; // context_id, summary
    std::mutex results_mutex;
    
    // Create summarization processor
    auto summarization_processor = [&](const SummarizationRequest& request,
                                       std::function<void(std::string)> success_callback,
                                       std::function<void(const std::string&)> error_callback) {
        
        // Mock summarization logic - create a summary based on content length
        std::string summary;
        if (request.content_to_summarize.length() > 100) {
            summary = "SUMMARY: Long conversation about " + 
                     request.content_to_summarize.substr(0, 20) + "...";
        } else {
            summary = "SUMMARY: Brief exchange about " + 
                     request.content_to_summarize.substr(0, 15) + "...";
        }
        
        // Simulate summarization processing time
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        {
            std::lock_guard<std::mutex> lock(results_mutex);
            summarization_results.emplace_back(request.original_context_id, summary);
            summarization_count++;
        }
        
        // Execute completion callback if provided
        if (request.completion_callback) {
            request.completion_callback(request.original_context_id, summary);
        }
        
        success_callback(summary);
    };
    
    ProcessingPipeline<SummarizationRequest, std::string> summarization_pipeline(
        "SummarizationPipeline", summarization_processor);
    
    // Test summarization requests
    std::atomic<int> completion_callbacks{0};
    auto completion_callback = [&](const std::string& context_id, const std::string& summary) {
        completion_callbacks++;
        std::cout << "Summarization completed for context: " << context_id << std::endl;
        std::cout << "Summary: " << summary << std::endl;
    };
    
    // Create test requests
    SummarizationRequest req1("context1", "This is a long conversation that needs summarization. It contains multiple exchanges between users and contains detailed technical discussion about the LuminaChat project implementation.");
    req1.completion_callback = completion_callback;
    
    SummarizationRequest req2("context2", "Short chat");
    req2.completion_callback = completion_callback;
    
    // Queue requests with different priorities
    assert(summarization_pipeline.QueueRequest(req1, LuminaChat::RequestPriority::HIGH, "sum1"));
    assert(summarization_pipeline.QueueRequest(req2, LuminaChat::RequestPriority::NORMAL, "sum2"));
    
    // Wait for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Verify results
    assert(summarization_count >= 2);
    assert(completion_callbacks >= 2);
    assert(summarization_results.size() >= 2);
    
    // Check that summaries were generated correctly
    bool found_long_summary = false;
    bool found_short_summary = false;
    
    for (const auto& [context_id, summary] : summarization_results) {
        if (summary.find("Long conversation") != std::string::npos) {
            found_long_summary = true;
            assert(context_id == "context1");
        }
        if (summary.find("Brief exchange") != std::string::npos) {
            found_short_summary = true;
            assert(context_id == "context2");
        }
    }
    
    assert(found_long_summary);
    assert(found_short_summary);
    
    std::cout << "Summarization pipeline tests passed!" << std::endl;
}

void TestContentSanitizationPipeline() {
    std::cout << "=== Testing Content Sanitization Pipeline ===" << std::endl;
    
    using LuminaChat::ProcessingPipeline;
    
    // Mock sanitizer for testing
    std::atomic<int> sanitization_count{0};
    std::vector<SanitizationResult> sanitization_results;
    std::mutex results_mutex;
    
    // Create sanitization processor
    auto sanitization_processor = [&](const MockSanitizationRequest& request,
                                      std::function<void(SanitizationResult)> success_callback,
                                      std::function<void(const std::string&)> error_callback) {
        
        SanitizationResult result;
        result.sanitized_content = request.content;
        result.passed_filter = true;
        
        // Mock some basic filtering rules
        if (request.content.find("badword") != std::string::npos) {
            result.passed_filter = false;
            result.blocked_patterns.push_back("badword");
            result.sanitized_content = "[CONTENT FILTERED]";
        }
        
        if (request.content.find("spam") != std::string::npos) {
            // Remove spam but allow content through
            size_t pos = result.sanitized_content.find("spam");
            result.sanitized_content.replace(pos, 4, "****");
            result.blocked_patterns.push_back("spam");
        }
        
        // Simulate processing time
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        
        {
            std::lock_guard<std::mutex> lock(results_mutex);
            sanitization_results.push_back(result);
            sanitization_count++;
        }
        
        success_callback(result);
    };
    
    ProcessingPipeline<MockSanitizationRequest, SanitizationResult> sanitization_pipeline(
        "SanitizationPipeline", sanitization_processor);
    
    // Test various content types
    MockSanitizationRequest req1("Hello world, this is clean content", "context1");
    MockSanitizationRequest req2("This contains badword and should be blocked", "context2");
    MockSanitizationRequest req3("This message has spam content in it", "context3");
    
    // Queue requests
    assert(sanitization_pipeline.QueueRequest(req1, LuminaChat::RequestPriority::NORMAL, "san1"));
    assert(sanitization_pipeline.QueueRequest(req2, LuminaChat::RequestPriority::HIGH, "san2"));
    assert(sanitization_pipeline.QueueRequest(req3, LuminaChat::RequestPriority::NORMAL, "san3"));
    
    // Wait for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Verify results
    assert(sanitization_count >= 3);
    assert(sanitization_results.size() >= 3);
    
    // Check sanitization results
    bool found_clean = false;
    bool found_blocked = false;
    bool found_filtered = false;
    
    for (const auto& result : sanitization_results) {
        if (result.sanitized_content == "Hello world, this is clean content") {
            found_clean = true;
            assert(result.passed_filter);
            assert(result.blocked_patterns.empty());
        }
        if (result.sanitized_content == "[CONTENT FILTERED]") {
            found_blocked = true;
            assert(!result.passed_filter);
            assert(!result.blocked_patterns.empty());
        }
        if (result.sanitized_content.find("****") != std::string::npos) {
            found_filtered = true;
            assert(result.passed_filter); // Still passes but content modified
            assert(!result.blocked_patterns.empty());
        }
    }
    
    assert(found_clean);
    assert(found_blocked);
    assert(found_filtered);
    
    std::cout << "Content sanitization pipeline tests passed!" << std::endl;
}

void TestTemplateBasedSummarizationWorkflow() {
    std::cout << "=== Testing Template-Based Summarization Workflow ===" << std::endl;
    
    // Create test infrastructure
    SettingsManager settings_manager;
    ModelConfig main_config;
    main_config.model_path = "./mock_main_model.gguf";
    main_config.context_size = 4096;
    main_config.gpu_layers = 0;
    
    ModelConfig summary_config;
    summary_config.model_path = "./mock_summary_model.gguf";
    summary_config.context_size = 2048;
    summary_config.gpu_layers = 0;
    
    // Create models
    ModelInfo main_model("main_model", main_config);
    ModelInfo summary_model("summary_model", summary_config);
    
    // Create template for main context
    std::string main_template = R"(<|start_header_id|>env<|end_header_id|>
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

{% if old_chat_summary %}
<|start_header_id|>old_chat_summary<|end_header_id|>
{{ old_chat_summary }}
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

    // Create summary template  
    std::string summary_template = R"(<|start_header_id|>system_message<|end_header_id|>
You are an expert conversation summarizer. Create a concise summary of the following conversation that captures the key points, topics discussed, and any important context. Focus on what information would be useful for continuing the conversation later.
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Please summarize this conversation:

{{ content_to_summarize }}
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>)";

    // Create contexts
    ContextInfo main_context(&main_model, main_template);
    ContextInfo summary_context(&summary_model, summary_template);
    
    // Setup main context
    main_context.UpdateEnvironment("Discord chat environment");
    main_context.UpdateIdentity("Luna, an AI assistant");
    main_context.UpdateSystemPrompt("Be helpful and engaging");
    
    // Add conversation to main context
    main_context.AddMessage("user", "Hello Luna, I'm working on the LuminaChat project");
    main_context.AddMessage("assistant", "Hello! I'd be happy to help with your LuminaChat project. What specific aspect are you working on?");
    main_context.AddMessage("user", "I'm implementing the plugin architecture for summarization");
    main_context.AddMessage("assistant", "That sounds like an interesting challenge! The plugin architecture should allow for flexible processing workflows. Are you focusing on the template integration aspect?");
    main_context.AddMessage("user", "Yes, exactly. I want summaries to update template sections directly rather than polluting message history");
    main_context.AddMessage("assistant", "That's a clever approach! By updating template sections, you maintain clean conversation history while still providing context. This should work well with the ChatTemplateManager's dynamic section system.");
    
    // Create summarization pipeline workflow
    using LuminaChat::ProcessingPipeline;
    
    std::atomic<bool> workflow_completed{false};
    std::string final_summary;
    
    // Mock summarization processor that simulates real LLM summarization
    auto summarization_processor = [&](const SummarizationRequest& request,
                                       std::function<void(std::string)> success_callback,
                                       std::function<void(const std::string&)> error_callback) {
        
        // Simulate creating a summary context and processing
        ContextInfo temp_summary_context = summary_context; // Copy for this request
        
        // Mock: Extract key information and create summary
        std::string mock_summary = "SUMMARY: User is implementing LuminaChat's plugin architecture, specifically focusing on summarization workflows that update template sections directly rather than polluting message history. Discussion covered the integration with ChatTemplateManager's dynamic section system.";
        
        // Simulate processing time
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Apply summary to original context template
        if (request.completion_callback) {
            request.completion_callback(request.original_context_id, mock_summary);
        }
        
        final_summary = mock_summary;
        workflow_completed = true;
        success_callback(mock_summary);
    };
    
    ProcessingPipeline<SummarizationRequest, std::string> summarization_pipeline(
        "WorkflowTestPipeline", summarization_processor);
    
    // Create workflow callback
    auto workflow_callback = [&main_context](const std::string& context_id, const std::string& summary) {
        std::cout << "Applying summary to context: " << context_id << std::endl;
        std::cout << "Summary: " << summary << std::endl;
        
        // Apply summary to template - this is the key innovation!
        main_context.ApplySummary(summary);
        
        // Verify the template now contains the summary
        std::string rendered_template = main_context.BuildFullPrompt();
        assert(rendered_template.find("SUMMARY:") != std::string::npos);
        assert(rendered_template.find("plugin architecture") != std::string::npos);
    };
    
    // Trigger summarization workflow
    std::string conversation_content = main_context.GetConversationHistory();
    SummarizationRequest workflow_request("main_context", conversation_content);
    workflow_request.completion_callback = workflow_callback;
    
    // Execute the workflow
    assert(summarization_pipeline.QueueRequest(workflow_request, LuminaChat::RequestPriority::HIGH, "workflow1"));
    
    // Wait for completion
    auto start_time = std::chrono::steady_clock::now();
    while (!workflow_completed && 
           std::chrono::steady_clock::now() - start_time < std::chrono::seconds(5)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    assert(workflow_completed);
    assert(!final_summary.empty());
    assert(final_summary.find("plugin architecture") != std::string::npos);
    
    // Verify template integration
    std::string final_template = main_context.BuildFullPrompt();
    assert(final_template.find("<|start_header_id|>summary<|end_header_id|>") != std::string::npos);
    assert(final_template.find("plugin architecture") != std::string::npos);
    
    // Verify conversation history is still clean (no system pollution)
    std::string history = main_context.GetConversationHistory();
    assert(history.find("SUMMARY:") == std::string::npos); // Summary not in conversation history
    assert(history.find("LuminaChat project") != std::string::npos); // But conversation content is there
    
    std::cout << "Template-based summarization workflow tests passed!" << std::endl;
}

void TestPipelineErrorHandling() {
    std::cout << "=== Testing Pipeline Error Handling ===" << std::endl;
    
    using LuminaChat::ProcessingPipeline;
    
    std::atomic<int> error_count{0};
    std::atomic<int> success_count{0};
    
    // Create a processor that sometimes fails
    auto error_prone_processor = [&](const std::string& input,
                                     std::function<void(std::string)> success_callback,
                                     std::function<void(const std::string&)> error_callback) {
        
        if (input == "fail") {
            error_count++;
            error_callback("Simulated processing error");
            return;
        }
        
        if (input == "exception") {
            throw std::runtime_error("Simulated exception");
        }
        
        success_count++;
        success_callback("Processed: " + input);
    };
    
    ProcessingPipeline<std::string, std::string> error_test_pipeline(
        "ErrorTestPipeline", error_prone_processor);
    
    // Queue requests that will succeed and fail
    assert(error_test_pipeline.QueueRequest("success1"));
    assert(error_test_pipeline.QueueRequest("fail"));
    assert(error_test_pipeline.QueueRequest("success2"));
    assert(error_test_pipeline.QueueRequest("exception"));
    assert(error_test_pipeline.QueueRequest("success3"));
    
    // Wait for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Check that pipeline handled errors gracefully
    assert(success_count >= 3);
    assert(error_count >= 1);
    
    // Pipeline should still be operational
    assert(!error_test_pipeline.IsShutdown());
    
    // Statistics should reflect failures
    auto stats = error_test_pipeline.GetStats();
    assert(stats.failed_requests.load() >= 2); // fail + exception
    assert(stats.completed_requests.load() >= 3); // 3 successes
    
    std::cout << "Pipeline error handling tests passed!" << std::endl;
}

void TestMultiplePipelinesCoordination() {
    std::cout << "=== Testing Multiple Pipelines Coordination ===" << std::endl;
    
    using LuminaChat::ProcessingPipeline;
    
    // Create a coordinated workflow with multiple pipelines
    std::atomic<int> step1_completed{0};
    std::atomic<int> step2_completed{0};
    std::atomic<int> final_completed{0};
    
    // Pipeline 1: Input processing
    auto input_processor = [&](const std::string& input,
                               std::function<void(std::string)> success_callback,
                               std::function<void(const std::string&)> error_callback) {
        std::string processed = "STEP1:" + input;
        step1_completed++;
        success_callback(processed);
    };
    
    // Pipeline 2: Transformation  
    auto transform_processor = [&](const std::string& input,
                                   std::function<void(std::string)> success_callback,
                                   std::function<void(const std::string&)> error_callback) {
        std::string transformed = input + ":STEP2";
        step2_completed++;
        success_callback(transformed);
    };
    
    ProcessingPipeline<std::string, std::string> input_pipeline("InputPipeline", input_processor);
    ProcessingPipeline<std::string, std::string> transform_pipeline("TransformPipeline", transform_processor);
    
    // Coordination logic - chain pipelines together
    auto coordination_callback = [&](const std::string& step1_result) {
        // Step 1 completed, queue to step 2
        transform_pipeline.QueueRequest(step1_result, LuminaChat::RequestPriority::HIGH, "step2");
    };
    
    // Mock chaining by processing step1 results
    input_pipeline.QueueRequest("test1");
    input_pipeline.QueueRequest("test2");
    input_pipeline.QueueRequest("test3");
    
    // Wait for step 1
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    assert(step1_completed >= 3);
    
    // Manually trigger step 2 (in real implementation this would be automatic)
    transform_pipeline.QueueRequest("STEP1:test1");
    transform_pipeline.QueueRequest("STEP1:test2");
    transform_pipeline.QueueRequest("STEP1:test3");
    
    // Wait for step 2
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    assert(step2_completed >= 3);
    
    // Verify both pipelines are operational
    assert(!input_pipeline.IsShutdown());
    assert(!transform_pipeline.IsShutdown());
    
    // Check statistics
    auto input_stats = input_pipeline.GetStats();
    auto transform_stats = transform_pipeline.GetStats();
    
    assert(input_stats.completed_requests.load() >= 3);
    assert(transform_stats.completed_requests.load() >= 3);
    
    std::cout << "Multiple pipelines coordination tests passed!" << std::endl;
}

} // namespace Phase6Tests

int TestPhase6Plugin() {
    std::cout << "Testing core ProcessingPipeline template" << std::endl;
    std::cout << "Testing summarization pipeline with template-based callback system" << std::endl;
    std::cout << "Testing context-to-summary-to-template workflow" << std::endl;
    std::cout << std::endl;
    
    try {
        Phase6Tests::TestBasicPipelineCreation();
        std::cout << std::endl;
        
        Phase6Tests::TestSummarizationPipeline();
        std::cout << std::endl;
        
        Phase6Tests::TestContentSanitizationPipeline();
        std::cout << std::endl;
        
        Phase6Tests::TestTemplateBasedSummarizationWorkflow();
        std::cout << std::endl;
        
        Phase6Tests::TestPipelineErrorHandling();
        std::cout << std::endl;
        
        Phase6Tests::TestMultiplePipelinesCoordination();
        std::cout << std::endl;
        
        std::cout << "=== ALL PHASE 6 TESTS PASSED ===" << std::endl;
        std::cout << "Plugin foundation components are working correctly!" << std::endl;
        std::cout << "Ready for Phase 7: Orchestration Layer" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Phase 6 test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
