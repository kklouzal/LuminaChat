# LlamaManager Lookahead Integration

## Overview
The LlamaManager class now includes full support for lookahead decoding, which can significantly improve text generation performance by using speculative execution and n-gram pattern matching.

## How It Works
The integration is automatic and seamless:
1. **LlamaManager** handles lookahead configuration and status
2. **LlamaResponse** automatically uses lookahead when enabled
3. No changes needed to existing generation code

## Configuration Methods

### 1. Enable with Default Parameters
```cpp
LlamaManager manager;
manager.enable_lookahead();
// Uses: W=15, N=5, G=15 (window=15, ngram_size=5, max_verification=15)
```

### 2. Enable with Custom Parameters
```cpp
manager.enable_lookahead(20, 4, 10);
// W=20, N=4, G=10
```

### 3. Advanced Configuration
```cpp
LookaheadConfig config;
config.enabled = true;
config.window_size = 12;        // W - lookahead window size
config.ngram_size = 6;          // N - n-gram size  
config.max_verification = 8;    // G - max verification n-grams
manager.configure_lookahead(config);
```

### 4. Disable Lookahead
```cpp
manager.disable_lookahead();
```

## Status and Monitoring

### Check if Enabled
```cpp
bool enabled = manager.is_lookahead_enabled();
```

### Get Current Configuration
```cpp
const LookaheadConfig& config = manager.get_lookahead_config();
std::cout << "W=" << config.window_size 
          << ", N=" << config.ngram_size 
          << ", G=" << config.max_verification << std::endl;
```

## Automatic Usage

Once configured, lookahead is used automatically in all text generation:

```cpp
// Load model and create context (standard setup)
manager.load_model("model.gguf", "my_model");
manager.create_context("main_context", "my_model", "You are a helpful assistant.");

// Enable lookahead
manager.enable_lookahead(15, 5, 15);

// Generate response - lookahead is used automatically
auto* context = manager.get_context_info("main_context");
std::string response = manager.generate_response("Hello!", context);
```

## Performance Tuning

### Conservative (Low Memory, Moderate Performance Gain)
```cpp
manager.enable_lookahead(8, 3, 8);
```

### Balanced (Recommended Default)
```cpp
manager.enable_lookahead(); // Uses 15, 5, 15
```

### Aggressive (High Memory, Maximum Performance)
```cpp
manager.enable_lookahead(32, 8, 32);
```

## Parameter Explanation

- **W (window_size)**: Size of lookahead window (3-32)
  - Larger = more speculative tokens, higher memory usage
  - Smaller = less speculation, lower memory usage

- **N (ngram_size)**: N-gram pattern size (2-8)
  - Larger = more complex patterns, better accuracy
  - Smaller = simpler patterns, faster processing

- **G (max_verification)**: Maximum verification n-grams (1-32)
  - Larger = more pattern matching, higher memory usage
  - Smaller = fewer patterns, lower memory usage

## Validation and Error Handling

The system automatically validates configurations:
- Invalid parameters are rejected with log messages
- Out-of-range values are caught and logged
- Graceful fallback to standard generation if lookahead fails

## Logging

The system provides detailed logging:
```
Lookahead configured: enabled=true, W=15, N=5, G=15
Delegating to LlamaResponse for token generation with LOOKAHEAD enabled (W=15, N=5, G=15)
Lookahead generated 45 tokens in 234.5ms (192.1 t/s, 73.3% acceptance)
```

## Migration from Standard Generation

**No code changes required!** Existing generation code works exactly the same:

```cpp
// This code works with or without lookahead
std::string response = manager.generate_response(input, context, username);
```

The only change is enabling lookahead before generation:
```cpp
manager.enable_lookahead(); // Add this line
std::string response = manager.generate_response(input, context, username); // No changes
```

## Best Practices

1. **Enable lookahead after model loading** but before generation
2. **Use default parameters** initially, then tune based on performance needs
3. **Monitor memory usage** with aggressive settings on large models
4. **Check logs** for acceptance rates to validate effectiveness
5. **Disable for debugging** if you need deterministic token-by-token behavior

## Example Integration

```cpp
#include "LlamaManager.hpp"

int main() {
    LlamaManager manager;
    manager.initialize();
    
    // Load model and setup context
    manager.load_model("llama-7b.gguf", "main_model");
    manager.create_context("main_context", "main_model", "You are a helpful AI assistant.");
    
    // Enable lookahead for better performance
    manager.enable_lookahead(); // Default: W=15, N=5, G=15
    
    // Generate responses - lookahead automatically used
    auto* context = manager.get_context_info("main_context");
    while (true) {
        std::string input;
        std::getline(std::cin, input);
        if (input == "quit") break;
        
        std::string response = manager.generate_response(input, context);
        std::cout << response << std::endl;
    }
    
    return 0;
}
```

This integration provides significant performance improvements with minimal code changes and maintains full backward compatibility.
