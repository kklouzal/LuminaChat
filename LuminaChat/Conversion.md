# LuminaChat Codebase Refactoring Guide

## Overview

This document outlines the step-by-step process to refactor the LuminaChat codebase from its current monolithic structure to a clean, modular architecture that eliminates circular dependencies and improves maintainability.

## Current Issues Identified

- **Circular Dependencies**: Complex includes between LlamaContext, LlamaSummarizer, and ContextSizeManager
- **God Classes**: `LlamaContext.hpp` is 1800+ lines with multiple responsibilities
- **Template Implementation in Headers**: Heavy template code mixed with declarations
- **Tight Coupling**: Context management, summarization, and size management are tightly coupled
- **Redundant Code**: Similar functionality scattered across multiple files
- **Constants Fragmentation**: 7+ different constant namespaces scattered across files
- **Threading Inconsistency**: Mixed threading models and inconsistent thread safety contracts
- **GUI-Core Coupling**: SettingsManager mixes wxWidgets GUI code with core logic
- **TokenCache Isolation**: Sophisticated caching system needs integration with new architecture
- **Discord Mini-Architecture**: Complex Discord integration has its own patterns that need unification
- **Performance Tracking Duplication**: Multiple classes implement their own performance metrics
- **Error Handling Inconsistency**: Mixed exception handling patterns across the codebase
- **llama.cpp Integration**: Direct llama.cpp dependencies scattered throughout multiple files

## Target Architecture

```
LuminaChat/
├── Core/                           # Core interfaces and base types
│   ├── Types.hpp                   # Common types, enums, constants
│   ├── Interfaces.hpp              # Pure virtual interfaces
│   └── Events.hpp                  # Event system for loose coupling
│
├── Context/                        # Context management subsystem
│   ├── ContextManager.hpp          # Main context orchestrator
│   ├── ContextState.hpp           # Context state tracking
│   ├── BatchProcessor.hpp         # Token batch processing
│   └── impl/                      # Implementation details
│       ├── ContextManager.cpp
│       ├── BatchProcessor.cpp
│       └── Templates.hpp          # Template implementations
│
├── Memory/                         # Memory and size management
│   ├── SizeAnalyzer.hpp           # Context size analysis
│   ├── SizeStrategy.hpp           # Size management strategies
│   ├── TokenTracker.hpp           # Token usage tracking
│   └── impl/
│       ├── SizeAnalyzer.cpp
│       └── SizeStrategy.cpp
│
├── Conversation/                   # Conversation management
│   ├── MessageHistory.hpp         # Message storage and retrieval
│   ├── ConversationState.hpp     # Conversation state tracking
│   ├── Summarizer.hpp             # Content summarization
│   └── impl/
│       ├── MessageHistory.cpp
│       └── Summarizer.cpp
│
├── LLM/                           # Low-level LLM operations
│   ├── ModelManager.hpp           # Model loading and management
│   ├── TokenProcessor.hpp         # Tokenization operations
│   ├── ResponseGenerator.hpp      # Response generation
│   └── impl/
│       ├── ModelManager.cpp
│       ├── TokenProcessor.cpp
│       └── ResponseGenerator.cpp
│
├── Integration/                    # External integrations
│   ├── DiscordManager.hpp
│   ├── DiscordHistoryLoader.hpp
│   └── impl/
│
├── Utils/                         # Utilities and helpers
│   ├── LogHandler.hpp
│   ├── SettingsManager.hpp
│   ├── TextSanitizer.hpp
│   ├── Blacklist.hpp
│   └── CommonUtils.hpp
│
└── GUI/                           # UI components
    ├── LuminaChatApp.cpp          # Main application
    ├── MainFrame.hpp
    └── Dialogs/
```

---

## Refactoring Plan

### Phase 1: Foundation and Interfaces (Priority: Critical)
**Status: ⏳ Not Started**
**Estimated Time: 2-3 days**

#### Step 1.1: Create Core Infrastructure
- [ ] Create `Core/` directory
- [ ] Create `Core/Types.hpp`
  - Extract common types from `LlamaContext.hpp`
  - **CRITICAL**: Consolidate ALL constants from 7+ namespaces:
    - LlamaConstants (LlamaContext.hpp, LlamaManager.hpp)
    - ResponseConstants (LlamaResponse.hpp)
    - LlamaResponseConstants (LlamaResponse.hpp)
    - SummarizerConstants (LlamaSummarizer.hpp)
    - UIConstants (LuminaChat.cpp)
    - ContextSizeConstants (ContextSizeManager.hpp)
    - SettingsUIConstants (SettingsManager.hpp)
  - Define core enums and structures
  - **CRITICAL**: Define unified threading model and contracts
- [ ] Create `Core/Interfaces.hpp`
  - Define pure virtual interfaces for major components
  - **CRITICAL**: Include llama.cpp abstraction interface to isolate dependencies
- [ ] Create `Core/Events.hpp`
  - Implement event system for loose coupling
  - **CRITICAL**: Design performance tracking event system to unify metrics across classes

**Files to create:**

<details>
<summary>Core/Types.hpp Template</summary>

```cpp
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <optional>

namespace LuminaChat {
    // CONSOLIDATED CONSTANTS - All constants from 7+ namespaces merged here
    namespace Constants {
        // Core System Constants (from LlamaConstants)
        constexpr int32_t DEFAULT_CONTEXT_SIZE = 2048;
        constexpr int32_t DEFAULT_GPU_LAYERS = 0;
        constexpr int32_t DEFAULT_PREDICT_TOKENS = 256;
        constexpr int32_t MAX_SEQ_IDS = 8;
        constexpr size_t MAX_MESSAGE_HISTORY_SIZE = 1000;
        constexpr int32_t MAX_BATCH_SIZE = 8192;
        constexpr float MS_TO_MICROSECONDS = 1000.0f;
        constexpr int32_t MAX_RETRY_ATTEMPTS = 2;
        constexpr int32_t RETRY_BACKOFF_MS = 50;
        
        // Context size management constants (from ContextSizeConstants)
        constexpr float MAX_CONTEXT_USAGE = 0.9f;
        constexpr float TARGET_CONTEXT_USAGE = 0.7f;
        constexpr float EMERGENCY_BUFFER = 0.1f;
        constexpr float AGGRESSIVE_PRUNING_RATIO = 0.3f;
        constexpr float MAX_TOTAL_SUMMARY_ALLOCATION = 0.30f;
        constexpr float MIN_AI_ALLOCATION = 0.15f;
        constexpr float MIN_ACTIVE_CONTENT = 0.25f;
        
        // Response Constants (from ResponseConstants)
        constexpr int32_t MIN_RESPONSE_TOKENS = 1;
        constexpr int32_t MAX_RESPONSE_TOKENS = 2048;
        constexpr std::chrono::milliseconds GENERATION_TIMEOUT{30000};
        
        // Summarizer Constants (from SummarizerConstants)
        constexpr size_t MAX_SUMMARY_SLOTS = 4;
        constexpr int32_t MIN_MESSAGES_FOR_SUMMARY = 6;
        constexpr float SUMMARY_COMPRESSION_RATIO = 0.3f;
        
        // UI Constants (from UIConstants)
        constexpr int32_t DEFAULT_WINDOW_WIDTH = 1200;
        constexpr int32_t DEFAULT_WINDOW_HEIGHT = 800;
        constexpr int32_t CONTEXT_MONITOR_INTERVAL_MS = 1000;
        
        // Threading Constants (NEW - Unified threading model)
        constexpr std::chrono::milliseconds DEFAULT_TIMEOUT{5000};
        constexpr int32_t MAX_WORKER_THREADS = 4;
    }
    
    // UNIFIED THREADING CONTRACTS (NEW - Critical for phase coordination)
    enum class ThreadSafety {
        NONE,           // Not thread-safe, external synchronization required
        READ_ONLY,      // Thread-safe for concurrent reads, exclusive write access needed
        FULL,           // Fully thread-safe for all operations
        INTERNAL_ONLY   // Thread-safe internally, but external access needs synchronization
    };
    
    // Core types
    struct Message {
        std::string role;
        std::string content;
        std::chrono::system_clock::time_point timestamp;
        int32_t token_count = -1; // -1 means not calculated
        
        Message() = default;
        Message(std::string r, std::string c) 
            : role(std::move(r)), content(std::move(c)), 
              timestamp(std::chrono::system_clock::now()) {}
    };
    
    // ENHANCED Context Analysis (includes performance tracking)
    struct ContextAnalysis {
        int32_t total_tokens = 0;
        int32_t available_tokens = 0;
        int32_t context_size = 0;
        int32_t summary_tokens = 0;
        int32_t active_history_tokens = 0;
        int32_t required_ai_space = 0;
        
        bool needs_pruning = false;
        bool emergency_buffer_violated = false;
        bool needs_summary_merge = false;
        
        float usage_percentage = 0.0f;
        float context_utilization_efficiency = 0.0f;
        float prediction_accuracy_score = 0.0f;
        
        // NEW: Unified performance metrics
        std::chrono::microseconds last_operation_time{0};
        int64_t total_operations = 0;
        int64_t cache_hits = 0;
        int64_t cache_misses = 0;
    };
    
    enum class ContextStrategy {
        BALANCED,
        AI_HEAVY,
        SUMMARY_HEAVY,
        MINIMAL
    };
    
    // NEW: Unified performance tracking structure
    struct PerformanceMetrics {
        int64_t total_requests = 0;
        int64_t successful_requests = 0;
        int64_t failed_requests = 0;
        std::chrono::microseconds total_time{0};
        std::chrono::microseconds avg_time{0};
        std::chrono::microseconds peak_time{0};
        int64_t memory_usage_bytes = 0;
        float success_rate = 0.0f;
    };
    
    struct ConversationMetrics {
        size_t total_exchanges = 0;
        size_t avg_user_message_length = 0;
        size_t avg_ai_response_length = 0;
        float conversation_complexity_score = 0.0f;
        PerformanceMetrics performance;  // NEW: Include performance data
    };
}
```
</details>

<details>
<summary>Core/Interfaces.hpp Template</summary>

```cpp
#pragma once
#include "Types.hpp"
#include <memory>
#include <functional>

namespace LuminaChat {
    
    // Forward declarations
    class MessageHistory;
    struct ContextState;
    
    // CRITICAL: llama.cpp abstraction interface to isolate dependencies
    class ILlamaBackend {
    public:
        virtual ~ILlamaBackend() = default;
        virtual bool load_model(const std::string& path, int32_t n_ctx = 0, int32_t n_gpu_layers = 0) = 0;
        virtual void unload_model() = 0;
        virtual bool is_model_loaded() const = 0;
        virtual std::vector<int32_t> tokenize(const std::string& text, bool add_special = true) = 0;
        virtual std::string detokenize(const std::vector<int32_t>& tokens) = 0;
        virtual bool apply_chat_template(const std::vector<Message>& messages, bool add_generation_prompt, std::string& result) = 0;
        virtual int32_t get_context_size() const = 0;
        virtual ThreadSafety get_thread_safety() const = 0;
    };
    
    class IContextSizeAnalyzer {
    public:
        virtual ~IContextSizeAnalyzer() = default;
        virtual ContextAnalysis analyze(const ContextState& state) const = 0;
        virtual bool needs_pruning(int32_t additional_tokens) const = 0;
        virtual float get_optimal_ratio() const = 0;
        virtual void track_ai_response(int32_t tokens, int32_t predicted = 0) = 0;
        virtual void track_user_message() = 0;
        virtual PerformanceMetrics get_performance_metrics() const = 0;  // NEW: Unified metrics
        virtual ThreadSafety get_thread_safety() const = 0;  // NEW: Thread safety contract
    };

    class IConversationSummarizer {
    public:
        virtual ~IConversationSummarizer() = default;
        virtual bool prune_with_summary(MessageHistory& history, float ratio) = 0;
        virtual void refresh_summaries(MessageHistory& history) = 0;
        virtual bool has_summaries_to_inject() const = 0;
        virtual void inject_summaries_into_history(MessageHistory& history) = 0;
        virtual PerformanceMetrics get_performance_metrics() const = 0;  // NEW: Unified metrics
        virtual ThreadSafety get_thread_safety() const = 0;  // NEW: Thread safety contract
    };

    class IContextManager {
    public:
        virtual ~IContextManager() = default;
        virtual bool add_message(const std::string& role, const std::string& content) = 0;
        virtual bool prepare_for_generation() = 0;
        virtual bool generate_response(std::function<void(const std::string&)> callback) = 0;
        virtual void clear_conversation() = 0;
        virtual const MessageHistory& get_message_history() const = 0;
        virtual ContextAnalysis get_current_analysis() const = 0;  // NEW: Expose analysis
        virtual PerformanceMetrics get_performance_metrics() const = 0;  // NEW: Unified metrics
        virtual ThreadSafety get_thread_safety() const = 0;  // NEW: Thread safety contract
    };
    
    class IMessageHistory {
    public:
        virtual ~IMessageHistory() = default;
        virtual void add_message(std::string role, std::string content) = 0;
        virtual void insert_historical(std::string role, std::string content) = 0;
        virtual void prune_to_count(size_t target_count) = 0;
        virtual void clear() = 0;
        
        virtual const std::vector<Message>& get_messages() const = 0;
        virtual size_t size() const = 0;
        virtual bool empty() const = 0;
        
        virtual int32_t get_total_tokens() const = 0;
        virtual void invalidate_token_cache() = 0;
        virtual PerformanceMetrics get_performance_metrics() const = 0;  // NEW: Unified metrics
        virtual ThreadSafety get_thread_safety() const = 0;  // NEW: Thread safety contract
    };
    
    class ITokenProcessor {
    public:
        virtual ~ITokenProcessor() = default;
        virtual std::vector<int32_t> tokenize(const std::string& text, bool add_special = true) = 0;
        virtual std::string detokenize(const std::vector<int32_t>& tokens) = 0;
        virtual bool apply_chat_template(const std::vector<Message>& messages, 
                                       bool add_generation_prompt, 
                                       std::string& result) = 0;
        virtual PerformanceMetrics get_performance_metrics() const = 0;  // NEW: Unified metrics
        virtual ThreadSafety get_thread_safety() const = 0;  // NEW: Thread safety contract
    };
    
    // NEW: Unified Performance Tracker Interface
    class IPerformanceTracker {
    public:
        virtual ~IPerformanceTracker() = default;
        virtual void track_operation_start(const std::string& operation_name) = 0;
        virtual void track_operation_end(const std::string& operation_name, bool success = true) = 0;
        virtual void track_memory_usage(int64_t bytes) = 0;
        virtual void track_cache_hit() = 0;
        virtual void track_cache_miss() = 0;
        virtual PerformanceMetrics get_metrics(const std::string& operation_name = "") const = 0;
        virtual void reset_metrics() = 0;
        virtual ThreadSafety get_thread_safety() const = 0;
    };
    
    // NEW: Configuration Management Interface (to separate from wxWidgets)
    class IConfigurationManager {
    public:
        virtual ~IConfigurationManager() = default;
        virtual bool load_configuration(const std::string& file_path) = 0;
        virtual bool save_configuration(const std::string& file_path) const = 0;
        virtual std::string get_setting(const std::string& section, const std::string& key, const std::string& default_value = "") const = 0;
        virtual void set_setting(const std::string& section, const std::string& key, const std::string& value) = 0;
        virtual ThreadSafety get_thread_safety() const = 0;
    };

} // namespace LuminaChat
```
</details>

<details>
<summary>Core/Events.hpp Template</summary>

```cpp
#pragma once
#include "Types.hpp"
#include <unordered_map>
#include <vector>
#include <functional>
#include <mutex>

namespace LuminaChat::Events {

    enum class EventType {
        CONTEXT_REBUILT,
        CONTEXT_CLEARED,
        SUMMARY_UPDATED,
        PRUNING_NEEDED,
        PRUNING_COMPLETED,
        MODEL_LOADED,
        MODEL_UNLOADED,
        MESSAGE_ADDED,
        TOKEN_COUNT_UPDATED
    };

    struct ContextEvent {
        EventType type;
        int32_t token_count = 0;
        int32_t message_count = 0;
        float usage_percentage = 0.0f;
        std::string details;
        
        ContextEvent(EventType t) : type(t) {}
        ContextEvent(EventType t, const std::string& d) : type(t), details(d) {}
    };

    template<typename T>
    class EventBus {
        std::unordered_map<EventType, std::vector<std::function<void(const T&)>>> handlers_;
        mutable std::mutex mutex_;
        
    public:
        void subscribe(EventType type, std::function<void(const T&)> handler) {
            std::lock_guard<std::mutex> lock(mutex_);
            handlers_[type].push_back(std::move(handler));
        }
        
        void publish(EventType type, const T& data) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (auto it = handlers_.find(type); it != handlers_.end()) {
                for (const auto& handler : it->second) {
                    try {
                        handler(data);
                    } catch (...) {
                        // Log error but don't let one handler break others
                    }
                }
            }
        }
        
        void clear_handlers(EventType type) {
            std::lock_guard<std::mutex> lock(mutex_);
            handlers_[type].clear();
        }
        
        void clear_all_handlers() {
            std::lock_guard<std::mutex> lock(mutex_);
            handlers_.clear();
        }
    };

    // Global event bus instance
    extern EventBus<ContextEvent> g_context_event_bus;

} // namespace LuminaChat::Events
```
</details>

#### Step 1.2: Create Base Interfaces
- [ ] Define `IContextSizeAnalyzer` interface
- [ ] Define `IConversationSummarizer` interface  
- [ ] Define `IContextManager` interface
- [ ] Define `IMessageHistory` interface
- [ ] Define `ITokenProcessor` interface

**Progress Tracking:**
- [ ] Core/Types.hpp created and compiles
- [ ] Core/Interfaces.hpp created and compiles
- [ ] Core/Events.hpp created and compiles
- [ ] All interfaces properly defined
- [ ] Basic event system functional
- [ ] No compilation errors in Phase 1

**Testing Phase 1:**
```cpp
// Test basic compilation
#include "Core/Types.hpp"
#include "Core/Interfaces.hpp" 
#include "Core/Events.hpp"

// Verify all types and interfaces are accessible
```

### Phase 1.5: CRITICAL - Split GUI from Core Logic (Priority: Critical)
**Status: ⏳ Not Started**
**Estimated Time: 2 days**

#### Step 1.5.1: Extract Configuration Management from SettingsManager
- [ ] Create `Config/` directory
- [ ] Create `Config/ConfigurationManager.hpp` - implement `IConfigurationManager`
- [ ] Extract all non-GUI configuration logic from `SettingsManager.hpp`
- [ ] Create `Config/Settings.hpp` - pure data structures for settings
- [ ] **CRITICAL**: Remove wxWidgets dependencies from core configuration logic

#### Step 1.5.2: Create GUI Abstraction Layer
- [ ] Create `GUI/` directory structure
- [ ] Move `SettingsManager.hpp` → `GUI/SettingsDialog.hpp` (GUI-only)
- [ ] Create `GUI/Interfaces.hpp` - abstract GUI interfaces
- [ ] **CRITICAL**: Ensure core logic can work without GUI dependencies

**Progress Tracking:**
- [ ] Core configuration completely separated from GUI
- [ ] Settings can be loaded/saved without wxWidgets
- [ ] GUI is just a view layer over core configuration
- [ ] No wxWidgets includes in core modules

---

### Phase 1.6: CRITICAL - llama.cpp Abstraction (Priority: Critical)
**Status: ⏳ Not Started**
**Estimated Time: 1-2 days**

#### Step 1.6.1: Create llama.cpp Abstraction Layer
- [ ] Create `LLM/Backend/` directory
- [ ] Create `LLM/Backend/LlamaBackend.hpp` - implement `ILlamaBackend`
- [ ] Extract all direct llama.cpp calls from multiple files:
  - `LlamaContext.hpp` (67 line: `#include "llama-cpp.h"`)
  - `LlamaManager.hpp` (73 line: `#include "llama-cpp.h"`)
  - `LlamaResponse.hpp` (52 line: `#include "llama-cpp.h"`)
  - `DiscordHistoryLoader.hpp` (62 line: `#include "llama-cpp.h"`)
  - `TokenCache.hpp` (44 line: `#include "llama-cpp.h"`)
- [ ] **CRITICAL**: Isolate llama.cpp types behind interface

**Progress Tracking:**
- [ ] Only `LLM/Backend/` module directly includes llama.cpp
- [ ] All other modules use `ILlamaBackend` interface
- [ ] llama.cpp types are not exposed outside backend module
- [ ] Easy to swap llama.cpp implementation or versions

---

### Phase 1.7: CRITICAL - Unified Performance System (Priority: High)
**Status: ⏳ Not Started**
**Estimated Time: 1 day**

#### Step 1.7.1: Create Performance Tracking Subsystem
- [ ] Create `Core/Performance/` directory
- [ ] Create `Core/Performance/PerformanceTracker.hpp` - implement `IPerformanceTracker`
- [ ] Extract performance tracking from:
  - `TokenCache.hpp` (cache statistics)
  - `LogHandler.hpp` (performance tracing)
  - `DiscordManager.hpp` (message processing stats)
  - `ContextSizeManager.hpp` (analysis metrics)
- [ ] **CRITICAL**: Unify all performance metrics into single system

**Progress Tracking:**
- [ ] Single performance tracking system across all modules
- [ ] Consistent metrics collection and reporting
- [ ] Performance data can be aggregated and analyzed
- [ ] No duplicate performance tracking code

---
**Status: ⏳ Not Started**
**Estimated Time: 1 day**

#### Step 2.1: Extract Utility Classes
- [ ] Create `Utils/` directory
- [ ] Move `LogHandler.hpp` → `Utils/LogHandler.hpp`
- [ ] Move `SettingsManager.hpp` → `Utils/SettingsManager.hpp`
- [ ] Move `Sanitizer.hpp` → `Utils/TextSanitizer.hpp` (rename for clarity)
- [ ] Move `Blacklist.hpp` → `Utils/Blacklist.hpp`
- [ ] Create `Utils/CommonUtils.hpp` for shared utilities
- [ ] Extract `common_utils.hpp` content into `Utils/CommonUtils.hpp`

#### Step 2.2: Update Include Paths
- [ ] Update all files that include utility headers
- [ ] Search and replace old include paths
- [ ] Update Visual Studio project file (LuminaChat.vcxproj)
- [ ] Verify all utility classes still compile

**Progress Tracking:**
- [ ] All utility files moved to Utils/
- [ ] Include paths updated throughout codebase
- [ ] Visual Studio project updated
- [ ] No compilation errors after utility extraction
- [ ] Utility classes maintain full functionality

**Testing Phase 2:**
```bash
# Build test
msbuild LuminaChat.vcxproj /p:Configuration=Debug /p:Platform=x64
```

---

### Phase 3: Create Memory Management Subsystem (Priority: High)
**Status: ⏳ Not Started**
**Estimated Time: 3-4 days**

#### Step 3.1: Extract Size Management
- [ ] Create `Memory/` directory
- [ ] Create `Memory/SizeAnalyzer.hpp` - extract size analysis logic from ContextSizeManager
- [ ] Create `Memory/SizeStrategy.hpp` - extract strategy logic
- [ ] Create `Memory/TokenTracker.hpp` - extract token tracking
- [ ] Create `Memory/impl/` directory for implementations

#### Step 3.2: Implement New Size Analyzer

<details>
<summary>Memory/SizeAnalyzer.hpp Template</summary>

```cpp
#pragma once
#include "../Core/Interfaces.hpp"
#include "../Core/Types.hpp"
#include "TokenTracker.hpp"
#include <memory>

namespace LuminaChat {
    
    class SizeAnalyzer : public IContextSizeAnalyzer {
    public:
        struct Configuration {
            float max_usage_threshold = Constants::MAX_CONTEXT_USAGE;
            float target_usage = Constants::TARGET_CONTEXT_USAGE;
            float emergency_buffer = Constants::EMERGENCY_BUFFER;
            int32_t min_messages_to_keep = 15;
            float conservative_buffer_multiplier = 1.5f;
        };
        
    private:
        Configuration config_;
        std::unique_ptr<TokenTracker> ai_tracker_;
        std::unique_ptr<TokenTracker> summary_tracker_;
        int32_t context_size_ = Constants::DEFAULT_CONTEXT_SIZE;
        
    public:
        explicit SizeAnalyzer(Configuration config = {});
        ~SizeAnalyzer() override = default;
        
        // IContextSizeAnalyzer implementation
        ContextAnalysis analyze(const ContextState& state) const override;
        bool needs_pruning(int32_t additional_tokens) const override;
        float get_optimal_ratio() const override;
        void track_ai_response(int32_t tokens, int32_t predicted = 0) override;
        void track_user_message() override;
        
        // Configuration
        void set_context_size(int32_t size) { context_size_ = size; }
        int32_t get_context_size() const { return context_size_; }
        
        // Advanced analytics
        ConversationMetrics get_conversation_metrics() const;
        ContextStrategy suggest_optimal_strategy() const;
        std::vector<std::string> get_optimization_recommendations() const;
    };
    
} // namespace LuminaChat
```
</details>

<details>
<summary>Memory/TokenTracker.hpp Template</summary>

```cpp
#pragma once
#include "../Core/Types.hpp"
#include <deque>
#include <chrono>

namespace LuminaChat {
    
    class TokenTracker {
    public:
        struct Statistics {
            int32_t estimated_size = 0;
            int32_t min_size = 0;
            int32_t max_size = 0;
            int32_t total_tracked = 0;
            float prediction_accuracy = 0.0f;
            float average_size = 0.0f;
        };
        
    private:
        struct Entry {
            int32_t actual_tokens;
            int32_t predicted_tokens;
            std::chrono::system_clock::time_point timestamp;
        };
        
        std::deque<Entry> history_;
        size_t max_history_size_;
        Statistics cached_stats_;
        mutable bool stats_dirty_ = true;
        
    public:
        explicit TokenTracker(size_t max_history = 100);
        
        void track_tokens(int32_t actual, int32_t predicted = 0);
        void clear_history();
        
        Statistics get_statistics() const;
        int32_t get_estimated_size() const;
        float get_prediction_accuracy() const;
        
    private:
        void update_statistics() const;
        void cleanup_old_entries();
    };
    
} // namespace LuminaChat
```
</details>

#### Step 3.3: Create Strategy Management
- [ ] Implement `SizeStrategy.hpp` for different sizing approaches
- [ ] Extract strategy logic from existing ContextSizeManager
- [ ] Implement adaptive strategies based on usage patterns

**Progress Tracking:**
- [ ] Memory/ directory structure created
- [ ] SizeAnalyzer class extracted and refactored
- [ ] TokenTracker implementation created
- [ ] SizeStrategy implementation created
- [ ] All memory management interfaces implemented
- [ ] Memory management subsystem compiles without errors
- [ ] Basic functionality tests pass

---

### Phase 4: Create Conversation Management Subsystem (Priority: High)
**Status: ⏳ Not Started**
**Estimated Time: 3-4 days**

#### Step 4.1: Extract Message History Management
- [ ] Create `Conversation/` directory
- [ ] Create `Conversation/MessageHistory.hpp`
- [ ] Extract message storage logic from ContextInfo
- [ ] Implement proper message history interface

<details>
<summary>Conversation/MessageHistory.hpp Template</summary>

```cpp
#pragma once
#include "../Core/Interfaces.hpp"
#include "../Core/Types.hpp"
#include <vector>
#include <optional>
#include <mutex>

namespace LuminaChat {
    
    class MessageHistory : public IMessageHistory {
    private:
        std::vector<Message> messages_;
        mutable std::optional<int32_t> cached_total_tokens_;
        mutable std::mutex mutex_;
        
    public:
        MessageHistory() = default;
        ~MessageHistory() override = default;
        
        // IMessageHistory implementation
        void add_message(std::string role, std::string content) override;
        void insert_historical(std::string role, std::string content) override;
        void prune_to_count(size_t target_count) override;
        void clear() override;
        
        const std::vector<Message>& get_messages() const override;
        size_t size() const override;
        bool empty() const override;
        
        int32_t get_total_tokens() const override;
        void invalidate_token_cache() override;
        
        // Additional functionality
        void prune_to_ratio(float keep_ratio);
        std::vector<Message> get_recent_messages(size_t count) const;
        std::vector<Message> get_system_messages() const;
        size_t count_non_system_messages() const;
        
        // Thread safety
        template<typename Func>
        auto with_lock(Func&& func) const -> decltype(func()) {
            std::lock_guard<std::mutex> lock(mutex_);
            return func();
        }
    };
    
} // namespace LuminaChat
```
</details>

#### Step 4.2: Extract Summarization Logic
- [ ] Move `LlamaSummarizer.hpp` → `Conversation/Summarizer.hpp`
- [ ] Refactor to use new interfaces
- [ ] Remove circular dependencies with ContextInfo
- [ ] Implement `IConversationSummarizer` interface

<details>
<summary>Conversation/Summarizer.hpp Template</summary>

```cpp
#pragma once
#include "../Core/Interfaces.hpp"
#include "../Core/Events.hpp"
#include "MessageHistory.hpp"
#include <memory>
#include <functional>

namespace LuminaChat {
    
    class Summarizer : public IConversationSummarizer {
    public:
        using SummaryModifiedCallback = std::function<void()>;
        
    private:
        // Internal summarizer state and summary slots
        struct SummarySlot {
            std::string content;
            int32_t token_count = 0;
            bool is_active = false;
        };
        
        std::vector<SummarySlot> summary_slots_;
        SummaryModifiedCallback callback_;
        Events::EventBus<Events::ContextEvent>* event_bus_ = nullptr;
        
    public:
        explicit Summarizer(Events::EventBus<Events::ContextEvent>* event_bus = nullptr);
        ~Summarizer() override = default;
        
        // IConversationSummarizer implementation
        bool prune_with_summary(MessageHistory& history, float ratio) override;
        void refresh_summaries(MessageHistory& history) override;
        bool has_summaries_to_inject() const override;
        void inject_summaries_into_history(MessageHistory& history) override;
        
        // Configuration
        void set_summary_modified_callback(SummaryModifiedCallback callback);
        void sync_slots_with_context_manager();
        
        // Summary management
        void clear_summaries();
        size_t get_summary_count() const;
        int32_t get_total_summary_tokens() const;
        
    private:
        bool create_summary_for_messages(const std::vector<Message>& messages, 
                                       std::string& summary);
        void notify_summary_modified();
    };
    
} // namespace LuminaChat
```
</details>

**Progress Tracking:**
- [ ] MessageHistory class created and functional
- [ ] Summarizer moved and refactored
- [ ] All conversation management functionality preserved
- [ ] No circular dependencies remain
- [ ] Conversation subsystem compiles without errors
- [ ] Integration with event system working

---

### Phase 5: Create LLM Operations Subsystem (Priority: Medium)
**Status: ⏳ Not Started**
**Estimated Time: 2-3 days**

#### Step 5.1: Extract Model Management
- [ ] Create `LLM/` directory
- [ ] Create `LLM/ModelManager.hpp` - extract from LlamaManager
- [ ] Create `LLM/TokenProcessor.hpp` - extract tokenization logic
- [ ] Create `LLM/ResponseGenerator.hpp` - extract generation logic

<details>
<summary>LLM/ModelManager.hpp Template</summary>

```cpp
#pragma once
#include "../Core/Interfaces.hpp"
#include "../Core/Types.hpp"
#include <memory>
#include <string>

// Forward declarations for llama.cpp types
struct llama_model;
struct llama_context;

namespace LuminaChat {
    
    struct ModelInfo {
        bool model_loaded = false;
        int32_t n_ctx = 0;
        int32_t n_vocab = 0;
        std::string model_path;
        std::string model_name;
    };
    
    class ModelManager {
    private:
        std::unique_ptr<ModelInfo> model_info_;
        llama_model* model_ = nullptr;
        llama_context* context_ = nullptr;
        
    public:
        ModelManager();
        ~ModelManager();
        
        // Model operations
        bool load_model(const std::string& path, int32_t n_ctx = 0, int32_t n_gpu_layers = 0);
        void unload_model();
        bool is_loaded() const;
        
        // Model information
        const ModelInfo& get_model_info() const;
        llama_context* get_context() const { return context_; }
        
        // Model capabilities
        bool supports_chat_template() const;
        int32_t get_context_size() const;
        int32_t get_vocab_size() const;
    };
    
} // namespace LuminaChat
```
</details>

<details>
<summary>LLM/TokenProcessor.hpp Template</summary>

```cpp
#pragma once
#include "../Core/Interfaces.hpp"
#include "../Core/Types.hpp"
#include "ModelManager.hpp"
#include <vector>

// Forward declarations for llama.cpp types
typedef int32_t llama_token;

namespace LuminaChat {
    
    class TokenProcessor : public ITokenProcessor {
    private:
        const ModelManager* model_manager_;
        
    public:
        explicit TokenProcessor(const ModelManager* model_manager);
        ~TokenProcessor() override = default;
        
        // ITokenProcessor implementation
        std::vector<int32_t> tokenize(const std::string& text, bool add_special = true) override;
        std::string detokenize(const std::vector<int32_t>& tokens) override;
        bool apply_chat_template(const std::vector<Message>& messages, 
                               bool add_generation_prompt, 
                               std::string& result) override;
        
        // Extended functionality
        int32_t count_tokens(const std::string& text) const;
        std::vector<llama_token> tokenize_native(const std::string& text, bool add_special = true) const;
        std::string detokenize_native(const std::vector<llama_token>& tokens) const;
        
        // Batch operations
        std::vector<std::vector<int32_t>> tokenize_batch(const std::vector<std::string>& texts) const;
    };
    
} // namespace LuminaChat
```
</details>

#### Step 5.2: Create Response Generator
- [ ] Extract generation logic from existing context management
- [ ] Implement proper streaming and callback support
- [ ] Handle generation parameters and constraints

**Progress Tracking:**
- [ ] ModelManager extracted and cleaned
- [ ] TokenProcessor implemented
- [ ] ResponseGenerator created
- [ ] All LLM operations properly encapsulated
- [ ] LLM subsystem compiles without errors
- [ ] Integration with existing model loading works

---

### Phase 6: Refactor Context Management (Priority: Critical)
**Status: ⏳ Not Started**
**Estimated Time: 4-5 days**

#### Step 6.1: Create New Context Manager
- [ ] Create `Context/` directory
- [ ] Create `Context/ContextManager.hpp` - new main orchestrator
- [ ] Create `Context/ContextState.hpp` - state tracking
- [ ] Create `Context/BatchProcessor.hpp` - batch processing logic

<details>
<summary>Context/ContextManager.hpp Template</summary>

```cpp
#pragma once
#include "../Core/Interfaces.hpp"
#include "../Core/Events.hpp"
#include "../Memory/SizeAnalyzer.hpp"
#include "../Conversation/Summarizer.hpp"
#include "../Conversation/MessageHistory.hpp"
#include "../LLM/ModelManager.hpp"
#include "../LLM/TokenProcessor.hpp"
#include "ContextState.hpp"
#include "BatchProcessor.hpp"

namespace LuminaChat {
    
    class ContextManager : public IContextManager {
    private:
        std::unique_ptr<SizeAnalyzer> size_analyzer_;
        std::unique_ptr<Summarizer> summarizer_;
        std::unique_ptr<ModelManager> model_manager_;
        std::unique_ptr<TokenProcessor> token_processor_;
        std::unique_ptr<BatchProcessor> batch_processor_;
        std::unique_ptr<MessageHistory> message_history_;
        
        std::unique_ptr<ContextState> state_;
        Events::EventBus<Events::ContextEvent> event_bus_;
        
        mutable std::mutex operations_mutex_;
        
    public:
        ContextManager();
        ~ContextManager() override = default;
        
        // IContextManager implementation
        bool add_message(const std::string& role, const std::string& content) override;
        bool prepare_for_generation() override;
        bool generate_response(std::function<void(const std::string&)> callback) override;
        void clear_conversation() override;
        const MessageHistory& get_message_history() const override;
        
        // Configuration and setup
        bool initialize(const std::string& model_path, int32_t n_ctx = 0);
        void shutdown();
        
        // Event subscription
        void on_context_rebuilt(std::function<void(const Events::ContextEvent&)> handler);
        void on_pruning_needed(std::function<void(const Events::ContextEvent&)> handler);
        void on_message_added(std::function<void(const Events::ContextEvent&)> handler);
        
        // Advanced operations
        bool insert_historical_message(const std::string& role, const std::string& content);
        void begin_historical_loading();
        void end_historical_loading();
        
        // Analytics and monitoring
        ContextAnalysis get_context_analysis() const;
        ConversationMetrics get_conversation_metrics() const;
        
    private:
        bool rebuild_if_needed();
        bool check_pruning_requirements();
        void notify_event(Events::EventType type, const std::string& details = "");
    };
    
} // namespace LuminaChat
```
</details>

<details>
<summary>Context/ContextState.hpp Template</summary>

```cpp
#pragma once
#include "../Core/Types.hpp"
#include <string>
#include <vector>
#include <chrono>

// Forward declarations for llama.cpp types
typedef int32_t llama_token;
struct llama_batch;

namespace LuminaChat {
    
    class ContextState {
    private:
        // Core state
        int32_t n_past_ = 0;
        int32_t prev_len_ = 0;
        bool needs_rebuild_ = false;
        bool is_dirty_ = false;
        
        // Historical loading state
        bool loading_historical_ = false;
        int32_t recent_historical_insertions_ = 0;
        std::chrono::steady_clock::time_point last_historical_insertion_;
        
        // Performance tracking
        std::chrono::steady_clock::time_point last_generation_time_;
        int64_t total_generation_tokens_ = 0;
        int64_t last_decode_time_us_ = 0;
        
    public:
        ContextState() = default;
        
        // Position management
        int32_t get_position() const { return n_past_; }
        void set_position(int32_t pos) { n_past_ = pos; }
        void advance_position(int32_t tokens) { n_past_ += tokens; }
        void reset_position() { n_past_ = 0; }
        
        // Content tracking
        int32_t get_previous_length() const { return prev_len_; }
        void set_previous_length(int32_t len) { prev_len_ = len; }
        
        // State flags
        bool needs_rebuild() const { return needs_rebuild_; }
        void mark_for_rebuild() { needs_rebuild_ = true; }
        void clear_rebuild_flag() { needs_rebuild_ = false; }
        
        bool is_dirty() const { return is_dirty_; }
        void mark_dirty() { is_dirty_ = true; }
        void clear_dirty_flag() { is_dirty_ = false; }
        
        // Historical loading
        bool is_loading_historical() const { return loading_historical_; }
        void begin_historical_loading() { loading_historical_ = true; }
        void end_historical_loading() { loading_historical_ = false; }
        
        void track_historical_insertion();
        bool is_likely_historical_loading() const;
        void clear_historical_state();
        
        // Performance tracking
        void track_generation_time(std::chrono::microseconds duration);
        void track_decode_time(int64_t microseconds) { last_decode_time_us_ = microseconds; }
        void track_generation_tokens(int64_t tokens) { total_generation_tokens_ += tokens; }
        
        // State validation
        bool validate() const;
        void reset();
        
        // Update from external content
        void update(const std::string& formatted_content, const std::vector<llama_token>& tokens);
    };
    
} // namespace LuminaChat
```
</details>

#### Step 6.2: Implement Dependency Injection
- [ ] Create factory classes for component creation
- [ ] Implement proper dependency injection pattern
- [ ] Configure event system integration

<details>
<summary>Context/Factory.hpp Template</summary>

```cpp
#pragma once
#include "../Core/Interfaces.hpp"
#include "ContextManager.hpp"
#include <memory>

namespace LuminaChat {
    
    class ContextFactory {
    public:
        struct Configuration {
            std::string model_path;
            int32_t context_size = Constants::DEFAULT_CONTEXT_SIZE;
            int32_t gpu_layers = Constants::DEFAULT_GPU_LAYERS;
            
            // Size analyzer configuration
            SizeAnalyzer::Configuration size_config;
            
            // Event bus configuration
            bool enable_events = true;
        };
        
        static std::unique_ptr<ContextManager> create_context_manager(const Configuration& config);
        static std::unique_ptr<SizeAnalyzer> create_size_analyzer(const SizeAnalyzer::Configuration& config = {});
        static std::unique_ptr<Summarizer> create_summarizer(Events::EventBus<Events::ContextEvent>* event_bus = nullptr);
        static std::unique_ptr<MessageHistory> create_message_history();
        
    private:
        ContextFactory() = default;
    };
    
} // namespace LuminaChat
```
</details>

**Progress Tracking:**
- [ ] New ContextManager created
- [ ] ContextState properly extracted
- [ ] BatchProcessor implemented
- [ ] Dependency injection properly configured
- [ ] Event system integrated
- [ ] All context operations functional

---

### Phase 7: Extract Existing Logic (Priority: Critical)
**Status: ⏳ Not Started**
**Estimated Time: 3-4 days**

#### Step 7.1: Migrate Core Logic from `LlamaContext.hpp`
- [ ] Extract template processing logic
- [ ] Extract batch processing logic  
- [ ] Extract context rebuilding logic
- [ ] Move implementations to appropriate subsystems

#### Step 7.2: Update Method Signatures
- [ ] Remove template methods from headers where possible
- [ ] Move template implementations to `impl/Templates.hpp`
- [ ] Clean up method signatures and remove redundancy

<details>
<summary>Context/impl/Templates.hpp</summary>

```cpp
#pragma once
#include "../ContextManager.hpp"
#include "../../Core/Types.hpp"

namespace LuminaChat {
    
    // Template method implementations moved from headers
    template<typename TokenProcessor, typename PruningCallback>
    bool ContextManager::prepare_context_for_generation_impl(
        TokenProcessor&& process_text_to_tokens, 
        PruningCallback&& prune_conversation_with_summary) {
        
        std::lock_guard<std::mutex> lock(operations_mutex_);
        
        // Phase 1: Validate state
        if (!state_->validate()) {
            state_->reset();
            state_->mark_for_rebuild();
        }
        
        // Phase 2: Apply template
        std::string formatted_content;
        if (!token_processor_->apply_chat_template(
                message_history_->get_messages(), false, formatted_content)) {
            return false;
        }
        
        // Phase 3: Check if pruning needed
        auto tokens = process_text_to_tokens(formatted_content, true);
        if (size_analyzer_->needs_pruning(static_cast<int32_t>(tokens.size()))) {
            float ratio = size_analyzer_->get_optimal_ratio();
            if (!prune_conversation_with_summary(ratio)) {
                return false;
            }
        }
        
        // Phase 4: Rebuild context
        return rebuild_if_needed();
    }
    
} // namespace LuminaChat
```
</details>

**Progress Tracking:**
- [ ] All template logic extracted from headers
- [ ] Batch processing moved to BatchProcessor
- [ ] Context rebuilding delegated to appropriate classes
- [ ] Header file significantly reduced in size
- [ ] All functionality preserved

---

### Phase 8: Integration and Testing (Priority: Critical)
**Status: ⏳ Not Started**
**Estimated Time: 2-3 days**

#### Step 8.1: Update Integration Points
- [ ] Update `DiscordManager.hpp` to use new interfaces
- [ ] Update `DiscordHistoryLoader.hpp` to use MessageHistory
- [ ] Update main application to use ContextManager
- [ ] Update `LuminaChat.cpp` to use factory pattern

<details>
<summary>Integration Update Example</summary>

```cpp
// Old approach in LuminaChat.cpp
ContextInfo context_info;
// ... complex initialization

// New approach
auto config = ContextFactory::Configuration{
    .model_path = model_path,
    .context_size = 4096,
    .gpu_layers = 32
};

auto context_manager = ContextFactory::create_context_manager(config);
context_manager->initialize(model_path);

// Subscribe to events
context_manager->on_context_rebuilt([](const Events::ContextEvent& event) {
    // Handle context rebuilt
});
```
</details>

#### Step 8.2: Create Factory Classes
- [ ] Implement `ContextFactory` for dependency injection
- [ ] Create configuration structures
- [ ] Set up proper initialization order

**Progress Tracking:**
- [ ] All integration points updated
- [ ] Factory classes created and functional
- [ ] Dependency injection properly configured  
- [ ] All external interfaces maintained
- [ ] Application starts and runs correctly

---

### Phase 9: Legacy Code Removal (Priority: Low)
**Status: ⏳ Not Started**
**Estimated Time: 1-2 days**

#### Step 9.1: Identify Legacy Code
- [ ] Mark deprecated code sections in `LlamaContext.hpp`
- [ ] Identify redundant code in `ContextSizeManager.hpp`
- [ ] List unused utility functions
- [ ] Document code to be removed

#### Step 9.2: Safe Removal Process
- [ ] Remove large portions of `LlamaContext.hpp` (keep minimal interface)
- [ ] Remove redundant `ContextSizeManager.hpp` code
- [ ] Remove duplicate utility functions
- [ ] Clean up unused includes

**SAFETY CHECKLIST before removal:**
- [ ] All functionality verified in new architecture
- [ ] Comprehensive tests pass
- [ ] Performance benchmarks acceptable
- [ ] All integration points working

**Progress Tracking:**
- [ ] Legacy code identified and documented
- [ ] Safe removal plan created
- [ ] Old classes safely removed in stages
- [ ] No functionality lost during removal
- [ ] Codebase significantly cleaner

---

### Phase 10: Validation and Documentation (Priority: Medium)
**Status: ⏳ Not Started**
**Estimated Time: 2-3 days**

#### Step 10.1: Comprehensive Testing
- [ ] Unit tests for all new classes
- [ ] Integration tests for subsystems
- [ ] End-to-end functionality tests
- [ ] Performance benchmarking
- [ ] Memory leak testing

<details>
<summary>Test Plan Example</summary>

```cpp
// Unit test example
TEST(MessageHistoryTest, AddMessage) {
    auto history = ContextFactory::create_message_history();
    history->add_message("user", "Hello");
    
    EXPECT_EQ(history->size(), 1);
    EXPECT_FALSE(history->empty());
    
    const auto& messages = history->get_messages();
    EXPECT_EQ(messages[0].role, "user");
    EXPECT_EQ(messages[0].content, "Hello");
}

// Integration test example
TEST(ContextManagerTest, EndToEndGeneration) {
    auto config = ContextFactory::Configuration{
        .model_path = "test_model.gguf",
        .context_size = 2048
    };
    
    auto manager = ContextFactory::create_context_manager(config);
    ASSERT_TRUE(manager->initialize(config.model_path));
    
    EXPECT_TRUE(manager->add_message("user", "Hello"));
    
    std::string response;
    bool generation_completed = false;
    
    manager->generate_response([&](const std::string& token) {
        response += token;
        if (token.empty()) generation_completed = true;
    });
    
    EXPECT_TRUE(generation_completed);
    EXPECT_FALSE(response.empty());
}
```
</details>

#### Step 10.2: Documentation Updates
- [ ] Update README with new architecture
- [ ] Document new interfaces and their usage
- [ ] Create migration guide for future changes
- [ ] Update build system documentation
- [ ] Create API reference documentation

<details>
<summary>Documentation Structure</summary>

```
docs/
├── README.md                    # Updated overview
├── architecture/
│   ├── overview.md             # High-level architecture
│   ├── subsystems.md           # Detailed subsystem docs
│   └── event-system.md         # Event system documentation
├── api/
│   ├── interfaces.md           # Interface reference
│   ├── core-types.md           # Core types documentation
│   └── factory-pattern.md      # Factory usage guide
├── migration/
│   ├── from-legacy.md          # Migration from old architecture
│   └── breaking-changes.md     # Breaking changes list
└── examples/
    ├── basic-usage.md          # Basic usage examples
    ├── advanced-integration.md # Advanced integration patterns
    └── event-handling.md       # Event handling examples
```
</details>

**Progress Tracking:**
- [ ] All tests passing (unit, integration, end-to-end)
- [ ] Performance maintained or improved
- [ ] Memory usage optimized
- [ ] Documentation complete and accurate
- [ ] API reference generated
- [ ] Migration guide created

---

## Implementation Guidelines

### Code Quality Standards
1. **Single Responsibility**: Each class should have one clear purpose
2. **Interface Segregation**: Keep interfaces small and focused
3. **Dependency Inversion**: Depend on abstractions, not concretions
4. **No Circular Dependencies**: Use forward declarations and event systems
5. **RAII**: Proper resource management throughout
6. **Thread Safety**: Appropriate locking where needed

### Testing Strategy
- **Unit Tests**: Test each class in isolation with mocks
- **Integration Tests**: Test subsystem interactions
- **Regression Tests**: Ensure existing functionality preserved
- **Performance Tests**: Verify no performance degradation
- **Memory Tests**: Check for leaks and proper cleanup

### Migration Safety
- **Incremental Changes**: Complete one phase before starting the next
- **Backward Compatibility**: Maintain during transition phases
- **Rollback Plan**: Keep original code until validation complete
- **Continuous Integration**: Test after each major change
- **Feature Flags**: Enable/disable new architecture during testing

## Risk Mitigation

### High-Risk Areas
1. **Template Method Migration**: Complex template code extraction
2. **Circular Dependency Resolution**: May require significant refactoring
3. **Context State Management**: Critical for application functionality
4. **Performance Impact**: Large-scale changes may affect performance
5. **Memory Management**: Proper cleanup of new object hierarchies

### Contingency Plans
- **Parallel Implementation**: Keep old code alongside new during transition
- **Incremental Rollout**: Migrate one component at a time
- **Performance Monitoring**: Track metrics throughout migration
- **Automated Testing**: Comprehensive test suite to catch regressions
- **Rollback Procedures**: Clear steps to revert if issues arise

## Success Criteria

### Technical Metrics
- [ ] No circular dependencies remain
- [ ] Compilation time improved by >20%
- [ ] Code coverage maintained at >80%
- [ ] All existing functionality preserved
- [ ] Performance within 5% of original
- [ ] Memory usage stable or improved

### Quality Metrics  
- [ ] Header file sizes reduced by >50%
- [ ] Cyclomatic complexity reduced
- [ ] Clear separation of concerns achieved
- [ ] Interfaces properly abstracted
- [ ] Documentation complete and accurate
- [ ] Test coverage for new architecture >90%

### Architecture Metrics
- [ ] Dependencies flow in one direction only
- [ ] Each subsystem can be tested independently
- [ ] New features can be added without modifying existing code
- [ ] Event system enables loose coupling
- [ ] Factory pattern enables easy configuration changes

---

## Tracking Progress

### Current Status Summary
- **Phase 1**: ⏳ Not Started - Foundation and Interfaces
- **Phase 2**: ⏳ Not Started - Extract Utilities
- **Phase 3**: ⏳ Not Started - Memory Management  
- **Phase 4**: ⏳ Not Started - Conversation Management
- **Phase 5**: ⏳ Not Started - LLM Operations
- **Phase 6**: ⏳ Not Started - Context Management
- **Phase 7**: ⏳ Not Started - Extract Existing Logic
- **Phase 8**: ⏳ Not Started - Integration and Testing
- **Phase 9**: ⏳ Not Started - Legacy Code Removal
- **Phase 10**: ⏳ Not Started - Validation and Documentation

### Overall Progress: 0% Complete

---

**Next Steps:** Begin with Phase 1 - Foundation and Interfaces. This provides the groundwork for all subsequent phases and can be implemented with minimal risk to existing functionality.

**Estimated Total Time:** 3-4 weeks of focused development
**Risk Level:** Medium (well-planned incremental approach)
**Benefits:** Significant improvement in maintainability, testability, and extensibility
