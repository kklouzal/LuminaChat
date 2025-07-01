#pragma once

#include <cstdint>

namespace LuminaChat {

/**
 * ContextUsageStats - Statistics structure for context size monitoring
 * 
 * This structure holds comprehensive token usage information for a context,
 * used by the ContextPruningPlugin to make intelligent pruning decisions.
 */
struct ContextUsageStats {
    int32_t total_tokens = 0;
    int32_t max_tokens = 0;
    int32_t template_tokens = 0;      // Base template + system sections
    int32_t summary_tokens = 0;       // Summary sections in template
    int32_t message_tokens = 0;       // Actual conversation history
    int32_t buffer_tokens = 0;        // Reserved space for AI response
    
    float GetUsagePercentage() const {
        return max_tokens > 0 ? static_cast<float>(total_tokens) / max_tokens : 0.0f;
    }
    
    int32_t GetAvailableTokens() const {
        return max_tokens - total_tokens;
    }
    
    bool NeedsPruning(float threshold = 0.80f) const {
        return GetUsagePercentage() >= threshold; // Default 80% threshold
    }
    
    int32_t GetTargetTokensAfterPruning(float target = 0.40f) const {
        return static_cast<int32_t>(max_tokens * target); // Default reduce to 40%
    }
    
    int32_t GetTokensToRemove(float target = 0.40f) const {
        if (!NeedsPruning()) return 0;
        return total_tokens - GetTargetTokensAfterPruning(target);
    }
};

} // namespace LuminaChat
