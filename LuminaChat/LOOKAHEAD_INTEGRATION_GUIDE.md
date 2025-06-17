# LlamaManager Lookahead Integration

## Overview
The LlamaManager class uses **only lookahead decoding** for all text generation. Standard sequential token generation has been completely removed in favor of the superior lookahead decoding approach.

## How It Works
Lookahead decoding is automatic and mandatory:
1. **LlamaManager** always has lookahead enabled by default  
2. **LlamaResponse** uses only lookahead decoding for all generation
3. All existing generation code continues to work with improved performance

## Configuration Methods

### 1. Enabled by Default
```cpp
LlamaManager manager;
// Lookahead is automatically enabled with optimal defaults: W=15, N=5, G=15
```

### 2. Customize Parameters (Optional)
```cpp
manager.enable_lookahead(20, 4, 10);
// W=20, N=4, G=10
```

### 3. Advanced Configuration (Optional)
```cpp
LookaheadConfig config;
config.enabled = true;
config.window_size = 12;        // W - lookahead window size
config.ngram_size = 6;          // N - n-gram size  
config.max_verification = 8;    // G - max verification n-grams
manager.configure_lookahead(config);
```

### 4. Disable Lookahead (Not Recommended)
```cpp
manager.disable_lookahead();
// WARNING: This will cause generation failures since standard generation is removed
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

Lookahead is used automatically in all text generation:

```cpp
// Load model and create context (standard setup)
manager.load_model("model.gguf", "my_model");
manager.create_context("main_context", "my_model", "You are a helpful assistant.");

// Generate response - lookahead is used automatically (no configuration needed)
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
- Generation fails if lookahead cannot be initialized (no fallback)

## Logging

The system provides detailed logging:
```
Lookahead configured: enabled=true, W=15, N=5, G=15
Delegating to LlamaResponse for token generation with LOOKAHEAD enabled (W=15, N=5, G=15)
Lookahead generated 45 tokens in 234.5ms (192.1 t/s, 73.3% acceptance)
```

## Migration from Standard Generation

**No code changes required!** Existing generation code now uses lookahead automatically:

```cpp
// This code automatically uses lookahead (no changes needed)
std::string response = manager.generate_response(input, context, username);
```

All existing generation code receives the performance benefits of lookahead decoding without any modifications.

## Best Practices

1. **Use default parameters** initially - they're optimized for most use cases
2. **Tune parameters** based on performance needs and memory constraints
3. **Monitor memory usage** with aggressive settings on large models
4. **Check logs** for acceptance rates to validate effectiveness
5. **Don't disable** lookahead as there's no fallback generation method

## Example Integration

```cpp
#include "LlamaManager.hpp"

int main() {
    LlamaManager manager;
    manager.initialize();
    
    // Load model and setup context
    manager.load_model("llama-7b.gguf", "main_model");
    manager.create_context("main_context", "main_model", "You are a helpful AI assistant.");
    
    // Lookahead is automatically enabled - no configuration needed!
    
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

This implementation provides significant performance improvements with zero code changes and is the only generation method available.
