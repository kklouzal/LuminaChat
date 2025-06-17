# Debug Logging Control

## Overview
LuminaChat now supports controlling debug logging during generation to improve performance. Debug logging can significantly slow down token generation, especially in the lookahead generation loop.

## Usage

### Method 1: Edit LogHandler.hpp
1. Open `LogHandler.hpp`
2. Find the line: `// #define LUMINA_ENABLE_DEBUG_LOGGING`
3. Uncomment it to: `#define LUMINA_ENABLE_DEBUG_LOGGING`
4. Recompile the project

### Method 2: Compiler Flag
Add `/D LUMINA_ENABLE_DEBUG_LOGGING` to your compiler flags in Visual Studio:
1. Right-click project → Properties
2. Configuration Properties → C/C++ → Preprocessor
3. Add `LUMINA_ENABLE_DEBUG_LOGGING` to Preprocessor Definitions

## What is Controlled
When `LUMINA_ENABLE_DEBUG_LOGGING` is **disabled** (default):
- `LLAMA_LOG()` becomes a no-op (no performance cost)
- `LLAMA_LOG_DEBUG()` becomes a no-op (no performance cost) 
- `LLAMA_LOG_ERROR()` still works (errors always logged)

When `LUMINA_ENABLE_DEBUG_LOGGING` is **enabled**:
- All logging functions work normally
- Verbose debug output during generation loop
- Performance impact during generation

## Performance Impact
With debug logging disabled, generation should be noticeably faster as the following debug statements become no-ops:
- Loop iteration messages
- Token generation messages  
- Verification round messages
- Sequence initialization messages
- Lookahead batch building messages

## Recommendation
- **Development**: Enable debug logging to troubleshoot issues
- **Production**: Keep debug logging disabled for best performance
