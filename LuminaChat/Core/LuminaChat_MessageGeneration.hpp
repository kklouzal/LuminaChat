#pragma once

// LuminaChat_MessageGeneration.hpp - Message generation and two-stage reasoning
// This header contains the complex two-stage reasoning logic, generation callbacks,
// and all message processing functionality.

#include <memory>
#include <thread>
#include <string>
#include "../Logger.hpp"
#include "../ErrorHandling.hpp"
#include "../ErrorHandling.hpp"

// === Message Generation and Processing Methods ===

inline void LuminaChatFrame::CaptureTemplateForInspection(ContextInfo* context, const std::string& context_id) {
    try {
        // Build the full prompt to capture the template after variable substitution
        std::string finalized_template = context->BuildFullPrompt();
        
        // Determine which voice UI should display the template
        std::string actual_context_id = context_id.empty() ? current_context_id : context_id;
        
        if (actual_context_id == "inner_context" && inner_voice_ui) {
            inner_voice_ui->UpdateTemplateDisplay(finalized_template);
            LOG_MessageGeneration("Inner voice template inspection updated with finalized template");
        } else if (actual_context_id == "outer_context" && outer_voice_ui) {
            outer_voice_ui->UpdateTemplateDisplay(finalized_template);
            LOG_MessageGeneration("Outer voice template inspection updated with finalized template");
        } else {
            // Fallback: try to determine from current state or log warning
            std::string context_debug_info = "Cannot determine voice UI - actual_context_id: " + actual_context_id + ", current_context_id: " + current_context_id;
            LOG_WARNING_MessageGeneration("Warning: " + context_debug_info);
        }
        
    } catch (const std::exception& template_e) {
        LOG_WARNING_MessageGeneration("Warning: Could not capture template for inspection: " + std::string(template_e.what()));
    }
}

inline GenerationCallbacks LuminaChatFrame::CreateGenerationCallbacks() {
    return GenerationCallbacks(
        // Token callback - called for each token as it's generated
        [this](const std::string& complete_response) {
            // Since we now receive the complete response each time, replace the content
            this->CallAfter([this, complete_response]() {
                ReplaceStreamingMessage(complete_response);
            });
        },
        
        // Completion callback - called when generation is done
        [this](const std::string& full_response, bool success) {
            this->CallAfter([this, full_response, success]() {
                EndStreamingMessage();
                SetGenerationUIState(false);  // Re-enable UI after generation
                UpdateContextStatus();  // Update context usage in status bar
                
                // Ensure focus is set after UI state changes
                if (chat_input) {
                    chat_input->SetFocus();
                }
                
                // PERFORMANCE FIX: Delay context monitoring to avoid interference with generation cleanup
                if (orchestrator) {
                    // Use a timer to delay monitoring by the configured delay after generation completes
                    auto timer = new wxTimer();
                    timer->Bind(wxEVT_TIMER, [this, timer](wxTimerEvent&) {
                        if (orchestrator) {
                            orchestrator->MonitorAllContextSizes();
                        }
                        delete timer;
                    });
                    timer->StartOnce(LuminaChatConstants::CONTEXT_MONITORING_DELAY_MS);
                }
                
                if (success) {
                    LOG_MessageGeneration("Response generation completed successfully");
                    
                    // Request emotional analysis for the context after AI response
                    if (orchestrator) {
                        auto* context = orchestrator->GetLlamaManager()->GetContextInfo(current_context_id);
                        auto* emotag_plugin = orchestrator->GetEmoTagPlugin();
                        if (context && emotag_plugin) {
                            emotag_plugin->RequestEmotionalAnalysis(current_context_id, context->GetMessageHistory());

                            LOG_MessageGeneration("Requested emotional analysis for context: " + current_context_id);
                            
                            // Immediately trigger processing of emotion analysis buffer
                            // instead of waiting for scheduled task
                            std::thread([this]() {
                                std::this_thread::sleep_for(std::chrono::milliseconds(LuminaChatConstants::EMOTION_ANALYSIS_DELAY_MS));
                                if (orchestrator) {
                                    orchestrator->ProcessScheduledTasks();
                                }
                            }).detach();
                        }
                    }
                } else {
                    LOG_MessageGeneration("Response generation was stopped or failed");
                    if (full_response.empty()) {
                        HandleWarning("Response generation was interrupted", "Message Generation", true);
                    }
                }
            });
        },
        
        // Error callback - called if there's an error
        [this](const std::string& error_message) {
            this->CallAfter([this, error_message]() {
                if (is_streaming) {
                    EndStreamingMessage();
                }
                SetGenerationUIState(false);  // Re-enable UI on error
                UpdateContextStatus();  // Update context usage in status bar
                
                // Ensure focus is set after UI state changes (error case)
                if (chat_input) {
                    chat_input->SetFocus();
                }
                
                // Trigger context monitoring even on error to check size
                if (orchestrator) {
                    orchestrator->MonitorAllContextSizes();
                }
                
                LOG_ERROR_MessageGeneration("Error generating response: " + error_message);
                HandleError("Error generating response: " + error_message, "Message Generation", true);
            });
        }
    );
}

inline void LuminaChatFrame::ExecuteMessageGeneration(ContextInfo* context, const std::string& input, const GenerationCallbacks& callbacks) {
    // Start streaming message display
    StartStreamingMessage("Assistant", LuminaChatColors::INFO_BLUE);
    SetGenerationUIState(true);  // Disable UI during generation
    
    // Start async generation
    bool started = context->HandleInputAsync(input, callbacks, "user");
    if (!started) {
        EndStreamingMessage();
        SetGenerationUIState(false);  // Re-enable UI on failure
        
        // Ensure focus is set after UI state changes (failure case)
        if (chat_input) {
            chat_input->SetFocus();
        }
        
        HandleError("Failed to start async generation", "Failed to start response generation");
    }
}

inline void LuminaChatFrame::ExecuteTwoStageReasoning(ContextInfo* inner_context, ContextInfo* outer_context, const std::string& input) {
    // Set UI state for processing
    SetGenerationUIState(true);
    LOG_MessageGeneration("Starting two-stage reasoning process...");
    
    // CRITICAL: Share chat history between contexts so inner voice has the same context as outer voice
    LOG_MessageGeneration("Synchronizing chat history between inner and outer voice contexts...");
    const auto& outer_history = outer_context->GetMessageHistory();
    
    // Clear inner context history first to ensure clean state
    inner_context->ClearMessageHistory();
    
    // Copy all historical messages from outer context to inner context
    for (const auto& [role, content] : outer_history) {
        inner_context->AddHistoricalMessage(role, content);
    }
    
    LOG_MessageGeneration("Chat history synchronized: " + std::to_string(outer_history.size()) + " messages copied to inner voice context");
    
    // Start inner voice streaming display
    StartInnerVoiceStreaming();
    
    // Stage 1: Generate inner voice reasoning (stream to inner voice display)
    LOG_MessageGeneration("Stage 1: Inner voice reasoning...");
    
    // Create callbacks for inner voice (Stage 1) - stream to inner voice display
    auto inner_callbacks = GenerationCallbacks(
        // Token callback - stream to inner voice display
        [this](const std::string& token) {
            this->CallAfter([this, token]() {
                AppendToInnerVoiceStreaming(token);
            });
        },
        
        // Completion callback for inner voice
        [this, outer_context, input](const std::string& inner_response, bool success) {
            this->CallAfter([this, outer_context, input, inner_response, success]() {
                // End inner voice streaming
                EndInnerVoiceStreaming();
                
                if (success && !inner_response.empty()) {
                    LOG_MessageGeneration("Stage 1 Complete: Inner voice generated reasoning");
                    LOG_MessageGeneration("Inner reasoning length: " + std::to_string(inner_response.length()) + " characters");
                    
                    // Stage 2: Update outer voice with reasoning thoughts
                    LOG_MessageGeneration("Stage 2: Updating outer voice with internal reflection...");
                    try {
                        outer_context->UpdateInternalReflection(inner_response);
                        LOG_MessageGeneration("Stage 2 Complete: Internal reflection updated in outer voice");
                        
                        // Stage 3: Generate final response with outer voice
                        LOG_MessageGeneration("Stage 3: Generating final response with outer voice...");
                        
                        // Capture template for inspection (outer voice with reasoning)
                        CaptureTemplateForInspection(outer_context, "outer_context");
                        
                        // Create callbacks for outer voice final response (with streaming)
                        auto outer_callbacks = CreateGenerationCallbacks();
                        
                        // Start streaming message display for final response
                        StartStreamingMessage("Assistant", LuminaChatColors::INFO_BLUE);
                        
                        // Generate final response with outer voice
                        bool started = outer_context->HandleInputAsync(input, outer_callbacks, "user");
                        if (!started) {
                            EndStreamingMessage();
                            SetGenerationUIState(false);
                            
                            // Ensure focus is set after UI state changes (outer voice failure case)
                            if (chat_input) {
                                chat_input->SetFocus();
                            }
                            
                            HandleError("Failed to start outer voice generation", "Failed to generate final response");
                        }
                        
                    } catch (const std::exception& e) {
                        SetGenerationUIState(false);
                        
                        // Ensure focus is set after UI state changes (stage 2 error case)
                        if (chat_input) {
                            chat_input->SetFocus();
                        }
                        
                        HandleError("Error in Stage 2: " + std::string(e.what()), "Error updating reasoning thoughts");
                    }
                    
                } else {
                    // Inner voice failed
                    SetGenerationUIState(false);
                    
                    // Ensure focus is set after UI state changes (inner voice failure case)
                    if (chat_input) {
                        chat_input->SetFocus();
                    }
                    
                    std::string error_msg = success ? "Inner voice generated empty response" : "Inner voice generation failed";
                    HandleError("Stage 1 Failed: " + error_msg, "Inner voice reasoning failed");
                }
            });
        },
        
        // Error callback for inner voice
        [this](const std::string& error_message) {
            this->CallAfter([this, error_message]() {
                // End inner voice streaming on error
                EndInnerVoiceStreaming();
                SetGenerationUIState(false);
                
                // Ensure focus is set after UI state changes (inner voice error case)
                if (chat_input) {
                    chat_input->SetFocus();
                }
                
                HandleError("Stage 1 Error: " + error_message, "Inner voice reasoning error: " + error_message);
            });
        }
    );
    
    // Capture inner voice template for inspection before starting reasoning
    CaptureTemplateForInspection(inner_context, "inner_context");
    
    // Start inner voice reasoning (Stage 1)
    bool started = inner_context->HandleInputAsync(input, inner_callbacks, "user");
    if (!started) {
        EndInnerVoiceStreaming();  // End streaming if generation fails to start
        SetGenerationUIState(false);
        
        // Ensure focus is set after UI state changes (inner voice failure case)
        if (chat_input) {
            chat_input->SetFocus();
        }
        
        HandleError("Failed to start inner voice reasoning", "Failed to start reasoning process");
    }
}

inline void LuminaChatFrame::HandleMessageGenerationError(const std::string& error_message) {
    if (is_streaming) {
        EndStreamingMessage();
    }
    SetGenerationUIState(false);  // Re-enable UI on exception
    
    // Ensure focus is set after UI state changes (general error case)
    if (chat_input) {
        chat_input->SetFocus();
    }
    
    HandleError("Error sending message: " + error_message, "Error: " + error_message);
}
