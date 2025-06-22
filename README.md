# LuminaChat

A high-performance AI chatbot application built with C++20 and wxWidgets, featuring Discord integration and advanced conversation management capabilities.

## 🚀 Features

### Core Functionality
- **Local AI Model Integration**: Full integration with llama.cpp for running GGUF models locally
- **Cross-Platform GUI**: Native wxWidgets interface with adaptive theming support
- **Real-Time Streaming**: Token-by-token response streaming for smooth user experience
- **Performance Monitoring**: Real-time statistics for token generation speed and cache efficiency

### Discord Integration
- **Multi-Context Support**: Isolated contexts for Discord channels and DMs
- **History Backfill**: Automatic loading of existing Discord conversation history
- **Flexible Channel Configuration**: Support for both shared and isolated channel contexts
- **Rate Limiting & Reconnection**: Robust handling of Discord API limitations

### Advanced Context Management
- **Adaptive Context Sizing**: Intelligent allocation of context space for summaries, history, and AI responses
- **Dynamic Summarization**: Automatic conversation summarization to maintain context within limits
- **Multi-Model Support**: Separate models for main conversation and summarization tasks
- **Context Monitoring**: Real-time visualization of context usage and allocation

### Smart Features
- **Blacklist System**: Content filtering with retroactive cleanup capabilities
- **Token Caching**: Bidirectional text↔token caching for improved performance
- **Message History Management**: Complete conversation tracking with pruning and summarization
- **Settings Persistence**: INI-based configuration with comprehensive options

## 🛠️ Technical Specifications

### Requirements
- **OS**: Windows (primary), cross-platform compatible
- **Compiler**: C++20 compatible (MSVC recommended)
- **Dependencies**: 
  - wxWidgets
  - llama.cpp
  - D++ (Discord library)

### Architecture
- **Language**: C++20 with modern features
- **UI Framework**: wxWidgets for native cross-platform interface
- **AI Backend**: llama.cpp for GGUF model inference
- **Discord**: D++ library for Discord bot functionality
- **Design Pattern**: Header-only implementations with RAII compliance

## 📋 Quick Start

### 1. Initial Setup
1. Launch LuminaChat
2. Click **Settings** to configure:
   - Select your GGUF model file
   - Configure context size and GPU layers
   - Set up system prompts and chat template
   - (Optional) Configure Discord bot token

### 2. Basic Usage
1. Click **Start** to load the model
2. Begin chatting in the main interface
3. Use **Prune & Summarize** when context fills up
4. Monitor context usage in real-time

### 3. Discord Integration
1. Configure Discord bot token in Settings
2. Set up channel IDs for isolated or shared contexts
3. Click **Connect Discord** to start the bot
4. Bot automatically handles DMs and configured channels

## 🎛️ Interface Overview

### Main Tabs
- **Chat**: Primary conversation interface with real-time streaming
- **Summaries**: Monitor summarization operations and outputs
- **Message History**: View complete conversation logs with context switching
- **Logs**: System logs and debug information

### Context Monitoring
Real-time progress bars showing:
- Overall context usage
- Summary allocation
- Active message history
- AI response space
- Emergency buffer

### Performance Stats
- Token generation speed (tokens/second)
- Cache hit ratios and efficiency
- Context utilization percentages

## ⚙️ Configuration Options

### Model Settings
- **Model Path**: Path to GGUF model file
- **Context Size**: Maximum context window (tokens)
- **GPU Layers**: Number of layers to offload to GPU
- **Generation Parameters**: Token limits and sampling settings

### System Prompts
- **Identity Directive**: Core AI personality and behavior
- **Additional Directives**: Supplementary instructions and guidelines
- **Chat Template**: Custom formatting for model input/output

### Discord Configuration
- **Bot Token**: Discord application bot token
- **Channel Management**: Isolated vs shared context channels
- **History Settings**: Automatic backfill configuration
- **DM Support**: Enable/disable direct message handling

### Summarization
- **Separate Model**: Optional dedicated summarization model
- **Context Allocation**: Dynamic slot management system
- **Merge Strategies**: Intelligent summary consolidation

## 🔧 Advanced Features

### Context Size Management
LuminaChat uses a sophisticated context management system that:
- Dynamically allocates space for different content types
- Maintains conversation flow through intelligent summarization
- Prevents context overflow with emergency buffers
- Optimizes token usage for maximum efficiency

### Multi-Context Isolation
Each Discord channel or DM can maintain its own isolated context:
- Independent conversation histories
- Separate context size management
- Isolated summarization processes
- No cross-contamination between conversations

### Performance Optimizations
- **Token Caching**: Reduces redundant tokenization operations
- **Batch Processing**: Efficient handling of multiple operations
- **Branch Prediction**: Optimized code paths for common operations
- **Memory Management**: RAII compliance and smart pointer usage

## 🔍 Monitoring & Debugging

### Real-Time Metrics
- Context usage visualization
- Token generation performance
- Cache efficiency statistics
- Memory allocation tracking

### Logging System
- Component-specific logging (UI, Discord, AI, Summarizer)
- Configurable log levels
- Thread-safe logging operations
- Real-time log viewing in interface

### Summary Slot Viewer
Monitor the summarization system:
- Current slot utilization
- Summary content preview
- Merge operation history
- Performance statistics

## 🚀 Performance Highlights

- **Optimized for Speed**: /O2 optimization with performance-first design
- **Memory Efficient**: Smart caching and minimal allocations
- **Thread-Safe**: Robust multi-threading with atomic operations
- **Responsive UI**: Non-blocking operations with progress feedback
- **Scalable**: Handles multiple Discord contexts simultaneously

## 🎨 UI Features

- **Adaptive Theming**: Automatic light/dark mode switching
- **Real-Time Updates**: Live context and performance monitoring
- **Intuitive Layout**: Organized tabs and progress visualization
- **Responsive Design**: Efficient rendering and smooth interactions

---

**LuminaChat** combines the power of local AI models with modern C++ engineering practices to deliver a robust, feature-rich chatbot platform suitable for both personal use and Discord community management.