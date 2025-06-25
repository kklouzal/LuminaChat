# LuminaChat Complete Codebase Rework Plan

## Architecture Overview

This document outlines the complete rewrite of LuminaChat using a clean, hierarchical header-only design that eliminates the current complexity while maintaining all sophisticated features through an extensible plugin architecture.

## Header-Only Hierarchy Design

The new architecture follows a strict top-to-bottom dependency flow within `LuminaChat.cpp`:

```cpp
#include "Logger.hpp"
#include "SettingsManager.hpp"
#include "Sanitizer.hpp"
#include "DiscordManager.hpp"  
#include "ContextSizeManager.hpp"
#include "TokenCache.hpp"
#include "ModelInfo.hpp"
#include "ChatTemplateManager.hpp"
#include "ContextInfo.hpp"
#include "LlamaManager.hpp"
#include "Orchestrator.hpp"
```

**Dependency Principle**: Lower items can reference items above them, but never below. This prevents circular dependencies and creates a clean, maintainable architecture.

## Circular Dependency Management

**8 controlled circular dependencies** exist, all mitigated through lambda callback registrars:

### Callback Registrar Pattern (Corrected)

1. **Higher-level component** (higher in the include hierarchy) defines a registration function (e.g., `RegisterXCallback`).
2. **Lower-level component** (lower in the include hierarchy) calls this registration function during its initialization, passing in a lambda that captures its own logic.
3. **Higher-level component** executes the pre-registered lambda during its normal logic routines, allowing it to communicate DOWN the chain (to the lower-level component) and break the circular dependency. This is necessary because the higher-level component's file is included above the lower-level component, so direct references are not possible.

**Example:**
```cpp
// In HigherComponent.hpp (higher in hierarchy)
class HigherComponent {
public:
    void RegisterDoSomethingCallback(std::function<void(int)> cb) { do_something_cb = std::move(cb); }
    void DoSomething(int x) {
        if (do_something_cb) do_something_cb(x); // Calls into lower component
    }
private:
    std::function<void(int)> do_something_cb;
};

// In LowerComponent.hpp (lower in hierarchy)
// During initialization:
higher_component.RegisterDoSomethingCallback([this](int x) {
    // Lower component's logic here
    this->HandleSomething(x);
});
```

This pattern allows the higher-level component to call into the lower-level component at runtime, even though it cannot reference it directly in code due to the include order.

### UI/Output Callbacks (Lower → Higher)
1. **LuminaChat.cpp ← Orchestrator**: LuminaChat defines registration function, Orchestrator calls it with lambda
2. **LuminaChat.cpp ← Logger**: LuminaChat defines registration function, Logger calls it with lambda

### Input/Communication Callbacks (Lower → Higher)  
3. **Orchestrator ← DiscordManager**: Orchestrator defines registration function, DiscordManager calls it with lambda
4. **Orchestrator ← ContextSizeManager**: Orchestrator defines registration function, ContextSizeManager calls it with lambda
5. **Orchestrator ← ContextInfo**: Orchestrator defines registration function, ContextInfo calls it with lambda
6. **Orchestrator ← ModelInfo**: Orchestrator defines registration function, ModelInfo calls it with lambda (optional)

### Plugin Integration Callbacks (Lower → Higher)
7. **Orchestrator ← Sanitizer**: Orchestrator defines registration function, Sanitizer calls it with lambda (optional)
8. **ContextInfo ← TokenCache**: ContextInfo defines registration function, TokenCache calls it with lambda (optional)

Each uses a `RegisterCallback` pattern where **higher-level components register lambda callbacks with lower-level components**, allowing lower components to call back up to higher ones without creating include dependencies.

### Callback Registration Pattern

The callback registration follows this consistent pattern:

```cpp
// In LuminaChat.cpp Start() function - higher-level component registers callbacks with lower ones
void Start() {
    // Initialize components in dependency order
    settings_manager = std::make_unique<SettingsManager>();
    context_size_manager = std::make_unique<ContextSizeManager>();
    // ... other components ...
    orchestrator = std::make_unique<Orchestrator>();
    
    // Register callbacks: HIGHER components register WITH LOWER components
    
    // UI Output callbacks - LuminaChat (higher) registers with lower components
    orchestrator->RegisterOutputCallback([this](std::string_view output, InputSource source) {
        this->OnOrchestratorOutput(output, source);
    });
    
    logger->RegisterOutputCallback([this](std::string_view log_message) {
        this->OnLogMessage(log_message);
    });
    
    // Communication callbacks - Orchestrator (higher) registers with lower components
    discord_manager->RegisterMessageCallback([this](const std::string& content, const std::string& channel_id, const std::string& username) {
        orchestrator->OnRawDiscordMessage(content, channel_id, username);
    });
    
    context_size_manager->RegisterSummarizationCallback([this](const std::string& context_id, const std::string& content) {
        orchestrator->RequestSummarization(context_id, content);
    });
    
    // Plugin Integration callbacks (registered per context/model as they're created)
    // These are registered in LlamaManager when contexts/models are created
}
```

This approach ensures:
- **Logical Flow**: Higher components define what they want to do when events occur
- **No Include Dependencies**: Lower components never need to include higher ones
- **Clear Ownership**: Callbacks are owned by the component that defines the logic
- **Single Registration Point**: All callbacks registered during initialization
- **Testable**: Components can be unit tested with mock callbacks

## Plugin Architecture Foundation

The Orchestrator implements a **plugin architecture foundation** that handles complex multi-step AI workflows through a consistent **Request → Process → Callback** pattern. This enables:

- **Summarization Workflows**: Context-to-summary-to-context processing (immediate implementation)
- **Content Sanitization**: Input filtering via existing Sanitizer.hpp (immediate implementation)
- **Future Extensions**: Translation, embeddings, external integrations, etc. (architecture ready)
- **Scheduled Tasks**: Timed operations using the same plugin infrastructure

### Core Plugin Pattern

```cpp
template<typename RequestType, typename ResultType>
struct ProcessingPipeline {
    std::queue<RequestType> pending_requests;
    std::function<void(RequestType, std::function<void(ResultType)>)> processor;
    
    void QueueRequest(const RequestType& request);
    void ProcessNext();
};
```

## Component Specifications

### Main Chat UI (LuminaChat.cpp)

**Primary Interface**: wxWidgets-based input/output for user interactions

**Core Functions**:
- `Start()` 
  - Initialize SettingsManager and load configuration
  - Load main_model and create main_context with base chat template
  - Load summary_model and create summary_context with summary template (if configured)
  - Initialize Orchestrator with UI callbacks and plugin pipelines
  - **Register all callback dependencies** (ContextSizeManager, ContextInfo, ModelInfo, etc.)
  - Start timer thread for scheduled operations
- `Stop()`
  - Stop timer thread
  - Save settings via SettingsManager
  - Trigger cleanup process for LlamaManager
  - Clean shutdown of all components and active workflows

**Timer Integration**:
```cpp
void TimerLoop() {
    while (running) {
        std::this_thread::sleep_for(std::chrono::seconds(30));
        if (orchestrator) {
            orchestrator->ProcessScheduledTasks();
        }
    }
}
```

### Settings Management

**Purpose**: Configuration persistence and access

**Features**:
- Uses existing .ini file format for compatibility
- Model paths, context sizes, GPU layer configurations  
- Discord bot tokens and channel settings
- UI preferences and logging levels
- Plugin enable/disable flags
- **Template Configuration**: Base chat templates per model/context type

**Interface**:
```cpp
class SettingsManager {
public:
    bool LoadSettings(const std::string& ini_path);
    bool SaveSettings();
    
    std::string GetString(const std::string& section, const std::string& key, const std::string& default_value = "");
    int GetInt(const std::string& section, const std::string& key, int default_value = 0);
    bool GetBool(const std::string& section, const std::string& key, bool default_value = false);
    
    void SetString(const std::string& section, const std::string& key, const std::string& value);
    void SetInt(const std::string& section, const std::string& key, int value);
    void SetBool(const std::string& section, const std::string& key, bool value);
    
    // Template management
    std::string GetChatTemplate(const std::string& template_name);
    void SetChatTemplate(const std::string& template_name, const std::string& template_content);
};
```

### Content Sanitization

**Purpose**: Input filtering and content moderation

**Implementation**: Direct integration of existing `Sanitizer.hpp` with minimal modifications

**Features**:
- Pattern-based content filtering
- Blacklist management
- Retroactive cleanup capabilities
- **Plugin Integration**: Automatic sanitization of Discord input via Orchestrator
- **Blacklist Integration**: Existing `Blacklist.hpp` carries over with minimal changes - applied to assistant messages before they're added to message history

**Interface**:
```cpp
class Sanitizer {
public:
    bool SanitizeInput(std::string& input);
    void LoadBlacklist(const std::string& blacklist_path);
    bool IsContentAllowed(const std::string& content);
    
    // Blacklist integration - preserved from existing implementation
    bool FilterAssistantResponse(std::string& response);
    
    // Callback registration for plugin integration
    void RegisterFilterCallback(std::function<void(std::string, bool)> callback);
};
```

**Callback Integration**:
```cpp
// Optional: For advanced filtering notifications - Orchestrator (higher) registers with Sanitizer (lower)
sanitizer->RegisterFilterCallback([this](const std::string& content, bool allowed) {
    orchestrator->OnFilterResult(content, allowed);
});
```

**Processing Flow Integration**:
```cpp
void OnDiscordMessage(const std::string& message, const std::string& channel_id) {
    std::string sanitized_message = message;
    if (sanitizer->SanitizeInput(sanitized_message)) {
        // Route sanitized input to Orchestrator
        input_callback(sanitized_message, channel_id, InputSource::DISCORD);
    } else {
        LOG_Discord("Message blocked by content filter");
    }
}

// In ContextInfo::HandleInput after LLM response generation:
std::string assistant_response = /* LLM generated response */;
if (sanitizer->FilterAssistantResponse(assistant_response)) {
    // Add filtered response to message history
    message_history.emplace_back("assistant", assistant_response);
} else {
    LOG_ContextInfo("Assistant response blocked by blacklist filter");
    // Handle blocked response appropriately
}
```

### Discord Interface

**Purpose**: Raw Discord API connection only

**Simplified Responsibilities**:
- Discord bot connection and authentication
- Raw message sending/receiving
- Basic API wrappers for message history fetching
- **No Intelligence**: No channel mapping, context routing, or message filtering

**Interface**:
```cpp
class DiscordManager {
public:
    bool Connect(const std::string& token);
    void Disconnect();
    void SendMessage(const std::string& channel_id, const std::string& content);
    void RegisterMessageCallback(std::function<void(std::string, std::string, std::string)> callback);
    
    // Simple API wrappers for plugins
    std::vector<DiscordMessage> FetchMessageHistory(const std::string& channel_id, 
                                                   const std::string& before_id = "", 
                                                   size_t limit = 50);
};
```

**Processing Flow**:
```cpp
// Raw Discord message → Orchestrator Discord Channel Plugin → Context Processing
void OnRawDiscordMessage(const std::string& content, const std::string& channel_id, const std::string& username) {
    DiscordChannelRequest request{
        .request_type = DiscordChannelRequest::INCOMING_MESSAGE,
        .channel_id = channel_id,
        .username = username,
        .content = content
    };
    
    discord_channel_pipeline.QueueRequest(request);
}
```

### Logger

**Responsibilities**:
- Centralized logging for all components
- Unique macros per file: `LOG_ContextSizeManager`, `LOG_TokenCache`, `LOG_ModelInfo`, etc.
- Thread-safe log message routing to UI

**Interface**:
```cpp
void RegisterOutputCallback(std::function<void(std::string_view)> callback);
```

### ContextSizeManager

**Purpose**: Intelligent context pruning management

**Core Logic**:
- Monitor context usage continuously
- Trigger pruning at 80% capacity, reduce to 40% usage
- Smart analysis of token distribution (summaries, active history, AI space, buffer)
- **Plugin Integration**: Requests summarization via Orchestrator callback
- **Template Integration**: Works with ChatTemplateManager for context size calculations

**Interface**:
```cpp
bool NeedsPruning(int32_t current_tokens, int32_t max_tokens);
void UpdateUsage(int32_t new_usage);
void RegisterSummarizationCallback(std::function<void(std::string, std::string)> callback);
```

**Callback Integration**:
```cpp
// Called during LuminaChat.cpp initialization - Orchestrator (higher) registers with ContextSizeManager (lower)
context_size_manager->RegisterSummarizationCallback([this](const std::string& context_id, const std::string& content) {
    orchestrator->RequestSummarization(context_id, content);
});
```

### TokenCache

**Purpose**: High-performance token caching system

**Features**:
- Text-to-token and token-to-text caching
- Performance statistics tracking
- Single instance per ModelInfo, shared across contexts
- Thread-safe operations
- **Template Caching**: Caches rendered template tokens for performance

**Data Structure**:
```cpp
struct CacheStats {
    size_t hits;
    size_t requests; 
    size_t entries;
    float hit_ratio;
};
```

**Callback Integration**:
```cpp
// Optional: For cache invalidation notifications - ContextInfo (higher) registers with TokenCache (lower)
class TokenCache {
public:
    void RegisterInvalidationCallback(std::function<void(std::string)> callback);
    // ...existing methods...
};

// Usage during initialization
token_cache->RegisterInvalidationCallback([this](const std::string& cache_key) {
    context_info->OnCacheInvalidation(cache_key);
});
```

### ModelInfo

**Purpose**: Model resource management and vocabulary access

**Components**:
- Owns TokenCache instance for this model
- Manages llama vocabulary for tokenization
- Handles model loading/unloading lifecycle
- **Settings Integration**: Model paths and parameters from SettingsManager

**Interface**:
```cpp
bool LoadModel(const std::string& path, int32_t context_size, int32_t gpu_layers);
void Cleanup();
TokenCache& GetTokenCache();
void RegisterResourceCallback(std::function<void(std::string, std::string)> callback);
```

**Callback Integration**:
```cpp
// Optional: For resource monitoring/notifications - Orchestrator (higher) registers with ModelInfo (lower)
model_info->RegisterResourceCallback([this](const std::string& model_id, const std::string& event) {
    orchestrator->OnResourceEvent(model_id, event);
});
```

### ChatTemplateManager

**Purpose**: Dynamic, sophisticated chat template management with Jinja2-style rendering

**Revolutionary Change**: Templates are now per-context and dynamically modified during runtime. System messages, summaries, and contextual information are embedded directly into template sections rather than polluting message history.

**Template Architecture**:
- **Base Template**: Complex Jinja2-style template with multiple specialized sections
- **Dynamic Sections**: Environment, Identity, System Prompts, Summaries, Motifs, Internal Reflection
- **Message Integration**: Conversation history rendered as `messages` array within template
- **Conditional Rendering**: Sections only appear when active and contain content

**Key Features**:
- **Per-Context Templates**: Each ContextInfo has its own template manager
- **Section-Based Management**: Individual control over template components
- **Memory Integration**: Past session memories as array-based template section
- **Summary Integration**: Summarization plugin updates template sections directly
- **Performance Caching**: Template rendering cached until sections change

**Interface**:
```cpp
enum class TemplateSection {
    OVERARCHING_ENVIRONMENT,
    IDENTITY_DIRECTIVE,
    SYSTEM_PROMPT,
    OLD_CHAT_SUMMARY,
    PAST_SESSIONS,
    SUMMARY,
    MOTIF_CONTEXT,
    INTERNAL_REFLECTION
};

class ChatTemplateManager {
private:
    std::string base_template;
    std::unordered_map<TemplateSection, TemplateVariable> sections;
    std::vector<std::string> past_sessions;
    std::string cached_rendered_template;
    bool template_dirty;
    
public:
    ChatTemplateManager(const std::string& base_template_str);
    
    // Section management
    void SetSection(TemplateSection section, const std::string& content, bool active = true);
    void ActivateSection(TemplateSection section);
    void DeactivateSection(TemplateSection section);
    void ClearSection(TemplateSection section);
    
    // Special array section handling
    void AddPastSession(const std::string& memory);
    void ClearPastSessions();
    
    // Template rendering with message history
    std::string RenderTemplate(const std::vector<std::pair<std::string, std::string>>& messages);
    
    // Convenience methods for common operations
    void UpdateEnvironment(const std::string& env);
    void UpdateIdentity(const std::string& identity);
    void UpdateSystemPrompt(const std::string& system_msg);
    void UpdateSummary(const std::string& summary);  // Plugin integration point
    void UpdateOldChatSummary(const std::string& old_summary);
    void UpdateMotifContext(const std::string& motif);
    void UpdateInternalReflection(const std::string& reflection);
    
    // Template validation
    bool ValidateTemplate() const;
    bool IsTemplateDirty() const { return template_dirty; }
};
```

**Template Sections**:
- `OVERARCHING_ENVIRONMENT`: Multi-user chat environment configuration
- `IDENTITY_DIRECTIVE`: AI persona and identity definition  
- `SYSTEM_PROMPT`: Core system instructions and behavior rules
- `OLD_CHAT_SUMMARY`: Summary from context pruning operations
- `PAST_SESSIONS`: Array of memories from previous sessions
- `SUMMARY`: Current session summary (updated by summarization plugin)
- `MOTIF_CONTEXT`: Thematic and emotional context cues
- `INTERNAL_REFLECTION`: Simulated internal AI dialogue and thought processes

**Base Template Structure**:
```jinja2
{{- bos_token }}

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

<|start_header_id|>assistant<|end_header_id>
```

**Plugin Integration**: 
- **Summarization Plugin**: Updates `SUMMARY` and `OLD_CHAT_SUMMARY` sections directly
- **Clean Separation**: No more "fake" system messages in conversation history
- **Dynamic Context**: Template sections can be modified independently throughout conversation

### ContextInfo

**Purpose**: Individual conversation context management

**Key Features**:
- **Dynamic Template Management**: Owns a ChatTemplateManager instance
- Performance statistics tracking
- Reference to parent ModelInfo and its TokenCache
- Message history storage: `std::vector<std::pair<std::string, std::string>>` (human-readable, tokenized)
- Batch management for generation
- Smart context rebuilding
- **Plugin Integration**: Template-based summarization and context management

**Rebuild Strategies**:
- `RebuildContext_Full()` - Complete rebuild including template re-rendering
- `RebuildContext_Partial()` - Efficient append-only rebuild for new messages

**Enhanced Core Functions**:
```cpp
class ContextInfo {
private:
    std::unique_ptr<ChatTemplateManager> template_manager;
    ModelInfo* parent_model;
    TokenCache* token_cache;
    std::vector<std::pair<std::string, std::string>> message_history; // Pure conversation only
    
public:
    ContextInfo(ModelInfo* model, const std::string& base_template);
    
    // Template section management
    void UpdateEnvironment(const std::string& env);
    void UpdateIdentity(const std::string& identity);
    void UpdateSystemPrompt(const std::string& system_msg);
    void ApplySummary(const std::string& summary);  // Updates template's summary section
    void UpdateOldChatSummary(const std::string& old_summary); // For context pruning
    void UpdateMotif(const std::string& motif);
    void UpdateInternalReflection(const std::string& reflection);
    void AddPastSessionMemory(const std::string& memory);
    
    // Core processing with dynamic template rendering
    std::string HandleInput(const std::string& input, const std::string& username);
    void RegisterSummarizationCallback(std::function<void(std::string, std::string)> callback);
    
private:
    std::string BuildFullPrompt();  // Renders complete template with current message history
    void RequestSummarization(const std::string& content); // Triggers plugin workflow
    std::function<void(std::string, std::string)> summarization_callback;
};
```

**Callback Integration**:
```cpp
// Called during ContextInfo creation in LlamaManager - Orchestrator (higher) registers with ContextInfo (lower)
context_info->RegisterSummarizationCallback([this](const std::string& context_id, const std::string& content) {
    orchestrator->RequestSummarization(context_id, content);
});
```

**Revolutionary Processing Flow**:
1. Receive input from Orchestrator (pre-sanitized if from Discord)
2. Add pure conversation pair to message history (no system contamination)
3. **Render dynamic template** with current message history via ChatTemplateManager
4. Process rendered template to tokens via TokenCache
5. **Check context size** - trigger summarization plugin if needed (updates template sections)
6. Generate LLM response using rendered template
7. Detokenize response via TokenCache  
8. Add response pair to message history
9. Return human-readable response

**Template Integration Benefits**:
- **Clean Message History**: Only actual conversation, no system pollution
- **Dynamic Context**: Summaries, system messages embedded in template sections
- **Plugin-Driven Updates**: Summarization updates template directly, not message history
- **Per-Context Flexibility**: Each context can have completely different template evolution

### LlamaManager

**Purpose**: Central coordinator for all models and contexts

**Resource Management**:
- Container for all ModelInfo instances
- Container for all ContextInfo instances  
- Lifecycle management for entire system
- **Settings Integration**: Model loading and template configuration from SettingsManager

**Enhanced Interface**:
```cpp
ModelInfo* GetOrCreateModelInfo(const std::string& model_id);
ContextInfo* GetOrCreateContextInfo(const std::string& context_id, const std::string& model_id, const std::string& template_name = "default");
void Cleanup(); // Cascading cleanup of all resources
```

### Orchestrator

**Purpose**: Central message routing and AI workflow orchestration hub

**Core Responsibilities**:
- **Message Routing**: `UI/Discord → [Plugin Processing] → ContextInfo → [Response] → UI/Discord`
- **Plugin Management**: Coordinates AI processing workflows (summarization, sanitization, Discord channel routing)
- **State Management**: Tracks active workflows and prevents conflicts
- **Scheduled Tasks**: Foundation for timed operations

**Enhanced Implementation**:
```cpp
class Orchestrator {
private:
    // Processing pipelines
    ProcessingPipeline<SummarizationRequest, std::string> summarization_pipeline;
    ProcessingPipeline<DiscordChannelRequest, DiscordChannelResponse> discord_channel_pipeline;
    ProcessingPipeline<HistoryBackfillRequest, HistoryBatch> history_backfill_pipeline;
    
    // Plugin instances
    std::unique_ptr<DiscordChannelPlugin> discord_channel_plugin;
    
    // State tracking
    enum class ProcessingState { NORMAL_PROCESSING, AWAITING_SUMMARIZATION, BACKFILL_IN_PROGRESS };
    std::unordered_map<std::string, ProcessingState> context_states;
    
    // Scheduled task foundation
    std::chrono::steady_clock::time_point last_scheduled_run;
    
    // Component references
    Sanitizer* sanitizer;
    DiscordManager* discord_manager;
    
public:
    // Core message routing
    void RegisterOutputCallback(std::function<void(std::string_view, InputSource)> callback);
    void InputReceived(const std::string& input, const std::string& context_id, InputSource source);
    
    // Enhanced Discord integration
    void OnRawDiscordMessage(const std::string& content, const std::string& channel_id, const std::string& username);
    void OnDiscordChannelResponse(const DiscordChannelResponse& response);
    
    // Plugin workflow coordination
    void RequestSummarization(const std::string& context_id, const std::string& content);
    void RequestHistoryBackfill(const std::string& context_id, const std::string& channel_id);
    
    // Scheduled task foundation
    void ProcessScheduledTasks();
};
```

**Enhanced Summarization Workflow**:
```cpp
struct SummarizationRequest {
    std::string original_context_id;
    std::string summary_context_id;
    std::string content_to_summarize;
    std::function<void(std::string)> completion_callback;
};

void CompleteSummarization(const std::string& context_id, const std::string& summary) {
    auto* original_context = llama_manager->GetOrCreateContextInfo(context_id, "main_model");
    
    // Apply summary directly to template instead of message history
    original_context->ApplySummary(summary);
    
    // Context is now ready for continued conversation with summary embedded in template
    context_states[context_id] = ProcessingState::NORMAL_PROCESSING;
}
```

## Implementation Benefits

### Performance Improvements
- **Smart Caching**: TokenCache eliminates redundant tokenization, including template caching
- **Efficient Context Management**: Partial rebuilds and template caching minimize computational overhead
- **Resource Pooling**: Shared TokenCache across contexts using same model
- **Async Plugin Processing**: Non-blocking workflows maintain UI responsiveness
- **Template Optimization**: Only re-render templates when sections actually change

### Maintainability Gains
- **Clear Dependencies**: Strict hierarchy prevents circular complexity
- **Single Responsibility**: Each component has focused, well-defined purpose
- **Consistent Interface**: All components follow similar patterns
- **Plugin Extensibility**: Foundation ready for future AI workflows without architectural changes
- **Template Separation**: Clean separation between conversation and contextual information

### Reliability Enhancements
- **RAII Compliance**: Proper resource management throughout
- **Thread Safety**: Built-in protection for concurrent access
- **Error Isolation**: Component failures don't cascade
- **State Management**: Plugin workflows prevent conflicts and race conditions
- **Template Validation**: Built-in template validation and error handling

### Feature Preservation & Enhancement
- **Advanced Context Pruning**: Sophisticated summarization system via plugin workflow with template integration
- **Dynamic Templates**: Revolutionary per-context template management with runtime modification
- **Content Filtering**: Existing Sanitizer.hpp integrated for Discord input protection
- **Discord Integration**: Full bot functionality with history backfill and automatic sanitization
- **Settings Management**: Existing .ini format preserved with template configuration support
- **Performance Monitoring**: Statistics tracking preserved in components
- **Multi-Model Support**: Main + summarizer model architecture with different templates
- **wxWidgets UI**: Familiar interface maintained
- **Future-Proof**: Plugin foundation ready for extensions

## Migration Strategy

### Phase 1: Foundation Components
-A Implement Logger, SettingsManager, Sanitizer integration
-B Implement TokenCache, ContextSizeManager
-C No dependencies, can be built and tested independently

### Phase 2: Model Layer  
-A Implement ModelInfo with TokenCache integration
-B Add SettingsManager integration for model configuration
-C Test model loading and tokenization caching

### Phase 3: Template Layer
-A Implement ChatTemplateManager with Jinja2-style rendering
-B Add template section management and caching
-C Test template rendering with message integration

### Phase 4: Context Layer
-A Implement ContextInfo with ChatTemplateManager integration
-B Test dynamic template management and message processing
-C Validate template-based context rebuilding

### Phase 5: Management Layer
-A Implement LlamaManager as resource coordinator
-B Integration testing of model/context/template lifecycle

### Phase 6: Plugin Foundation
-A Implement core ProcessingPipeline template
-B Add summarization pipeline with template-based callback system
-C Test context-to-summary-to-template workflow

### Phase 7: Orchestration Layer
-A Implement Orchestrator with plugin management
-B Integrate Sanitizer for Discord input filtering
-C Add scheduled task foundation

### Phase 8: Orchestration Layer
-A Full system integration with wxWidgets UI and Discord

This phased approach ensures working functionality is maintained throughout the rewrite process, with special attention to template system integration and validation at each stage.

## Future Plugin Extensions

The plugin architecture foundation is designed to easily accommodate future workflows:

### Potential Future Plugins
- **Discord Channel Management**: Intelligent channel-to-context routing with per-channel configuration *(immediate implementation)*
- **Discord History Backfill**: Batched message history loading with rate limiting and context awareness
- **Translation Pipeline**: Auto-detect language, translate content
- **Advanced Content Filtering**: Beyond basic sanitization - AI-powered moderation
- **Embedding Search**: Vector database integration for context retrieval
- **External Integrations**: Web search, API calls, database queries
- **Multi-Model Processing**: Send prompts to multiple models for comparison
- **Template Plugins**: Dynamic template section modification based on conversation analysis

### Adding New Plugins
```cpp
// Discord Channel Management Plugin - implemented as Orchestrator plugin
ProcessingPipeline<DiscordChannelRequest, DiscordChannelResponse> discord_channel_pipeline;

struct DiscordChannelRequest {
    enum Type { INCOMING_MESSAGE, OUTGOING_MESSAGE, CHANNEL_SETUP };
    Type request_type;
    std::string channel_id;
    std::string username;
    std::string content;
};

struct DiscordChannelResponse {
    bool should_respond = true;
    std::string response_content;
    std::string target_channel;
};

void ProcessDiscordChannel(const DiscordChannelRequest& request, 
                          std::function<void(DiscordChannelResponse)> callback) {
    // Channel-to-context mapping
    // Per-channel configuration (templates, auto-respond, user filtering)
    // Integration with sanitization and blacklist filtering
    // Automatic history backfill triggering for new channels
    // Context lifecycle management
}

// Discord History Backfill Plugin - implemented as Orchestrator plugin
ProcessingPipeline<HistoryBackfillRequest, HistoryBatch> history_backfill_pipeline;

struct HistoryBackfillRequest {
    std::string context_id;
    std::string channel_id;
    std::string before_message_id;  // For pagination
    size_t batch_size = 50;
    bool is_initial_request = true;
};

struct HistoryBatch {
    std::vector<DiscordMessage> messages;
    std::string oldest_message_id;  // For next batch
    bool has_more = true;
};

void RequestHistoryBackfill(const std::string& context_id, const std::string& channel_id) {
    // Rate-limited batched loading directly into message_history
    // Stops automatically when context approaches size limit
    // Pure conversation history - no template section updates
    // Non-blocking operation maintains UI responsiveness
    
    if (context_states[context_id] == ProcessingState::NORMAL_PROCESSING) {
        context_states[context_id] = ProcessingState::BACKFILL_IN_PROGRESS;
        
        HistoryBackfillRequest request{
            .context_id = context_id,
            .channel_id = channel_id,
            .is_initial_request = true
        };
        
        history_backfill_pipeline.QueueRequest(request);
    }
}

void OnHistoryBatchComplete(const std::string& context_id, const HistoryBatch& batch) {
    auto* context = llama_manager->GetOrCreateContextInfo(context_id, "main_model");
    
    // Add messages to pure conversation history (oldest first)
    for (auto it = batch.messages.rbegin(); it != batch.messages.rend(); ++it) {
        std::string sanitized_content = it->content;
        if (sanitizer->SanitizeInput(sanitized_content)) {
            context->AddHistoricalMessage(it->username, sanitized_content);
        }
    }
    
    // Context-aware continuation based on usage threshold
    bool should_continue = batch.has_more && 
                          !context->IsNearContextLimit(0.7f); // Stop at 70% to leave room
    
    if (should_continue) {
        // Queue next batch with rate limiting
        HistoryBackfillRequest next_request{
            .context_id = context_id,
            .channel_id = batch.channel_id,
            .before_message_id = batch.oldest_message_id,
            .is_initial_request = false
        };
        
        history_backfill_pipeline.QueueRequest(next_request);
    } else {
        // Backfill complete - return to normal processing
        context_states[context_id] = ProcessingState::NORMAL_PROCESSING;
        
        // Trigger initial summary if context is near limit after backfill
        if (context->IsNearContextLimit(0.8f)) {
            RequestSummarization(context_id, "Initial backfill summary");
        }
    }
}

// Future plugin example - just add to Orchestrator:
ProcessingPipeline<TranslationRequest, std::string> translation_pipeline;

void RequestTranslation(const std::string& context_id, const std::string& content) {
    // Same Request → Process → Callback pattern
    // Can update template sections as needed
}
```

#### Discord Channel Management Benefits
- **Per-Channel Intelligence**: Each Discord channel can have different templates, settings, user restrictions
- **Context Isolation**: Clean separation between different Discord channels/servers
- **Automatic Setup**: New channels automatically get contexts and optional history backfill
- **Flexible Configuration**: Enable/disable features per channel through settings
- **Clean Architecture**: Raw Discord API separated from intelligent routing logic
- **Plugin Coordination**: Seamlessly integrates with History Backfill and Sanitization plugins

#### Discord History Backfill Benefits
- **Context-Aware Stopping**: Automatically stops when approaching context limits
- **Rate Limiting Built-In**: Natural throttling through pipeline processing
- **Non-Blocking Operation**: UI remains responsive during backfill
- **Pure Message History**: Fills conversation history without template interference
- **Error Recovery**: State management handles failures gracefully
- **Resource Efficiency**: Integrates with existing ContextSizeManager and TokenCache
