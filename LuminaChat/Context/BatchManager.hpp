#pragma once

#include "../Logger.hpp"
#include "../ModelInfo.hpp"
#include "../TokenCache.hpp"
#include "llama-cpp.h"
#include <vector>
#include <memory>

// Forward declarations
struct llama_context;

namespace ContextConstants {
    extern const int32_t SAFETY_BUFFER_TOKENS;
    extern const int32_t GENERATION_BUFFER_TOKENS;
}

/**
 * BatchManager: Handles all llama.cpp batch operations for token processing
 * 
 * Responsibilities:
 * - Initialize and manage llama_batch structures
 * - Process token batches for both prompt and generation phases
 * - Validate context bounds and safety margins
 * - Coordinate with llama.cpp decode operations
 * 
 * Thread Safety:
 * - BatchManager methods are NOT thread-safe by design
 * - Synchronization must be handled by the calling class (e.g., ContextInputOutput)
 * - This design assumes exclusive access control by the derived/containing class
 * 
 * Used by ContextInputOutput for:
 * - Prompt token processing (input phase)
 * - Response token generation (output phase)
 */
class BatchManager {
private:
    // Core components
    llama_context* llama_ctx = nullptr;
    llama_batch batch;
    bool batch_initialized = false;
    
    // Context bounds and safety
    size_t max_context_tokens = 0;
    int32_t* n_past_ref = nullptr;  // Reference to parent's n_past for coordination
    
    // Internal validation helpers
    bool ValidateContextBounds(size_t token_count, int32_t current_position) const;
    bool ValidateBatchParameters() const;
    
public:
    // Constructor/Destructor
    BatchManager(llama_context* ctx, size_t max_tokens, int32_t& n_past_reference);
    ~BatchManager();
    
    // Batch lifecycle management
    bool InitializeBatch();
    void ClearBatch();
    void FreeBatch();
    
    // Core batch processing operations
    bool ProcessTokensBatch(const std::vector<int32_t>& tokens);
    bool ProcessSingleToken(int32_t token, bool generate_logits = true);
    
    // Batch building helpers
    bool AddTokenToBatch(int32_t token, int32_t position, bool generate_logits = false);
    bool ExecuteBatch();
    
    // Context state management
    bool ValidateContextState() const;
    void UpdateContextBounds(size_t new_max_tokens);
    
    // Getters for state inspection
    size_t GetMaxContextTokens() const { return max_context_tokens; }
    int32_t GetCurrentPosition() const { return n_past_ref ? *n_past_ref : 0; }
    bool IsBatchInitialized() const { return batch_initialized; }
    size_t GetBatchSize() const { return batch_initialized ? static_cast<size_t>(batch.n_tokens) : 0; }
    
    // Disable copy/move operations (manages llama.cpp resources)
    BatchManager(const BatchManager&) = delete;
    BatchManager& operator=(const BatchManager&) = delete;
    BatchManager(BatchManager&&) = delete;
    BatchManager& operator=(BatchManager&&) = delete;
};

// Inline implementation
inline BatchManager::BatchManager(llama_context* ctx, size_t max_tokens, int32_t& n_past_reference)
    : llama_ctx(ctx)
    , max_context_tokens(max_tokens)
    , n_past_ref(&n_past_reference)
{
    if (!llama_ctx) {
        LOG_ERROR("BatchManager", "BatchManager created with null llama context");
        return;
    }
    
    if (max_tokens == 0) {
        LOG_ERROR("BatchManager", "BatchManager created with zero max tokens");
        return;
    }
    
    LOG_DEBUG("BatchManager", "BatchManager created with max_tokens: " + std::to_string(max_tokens));
}

inline BatchManager::~BatchManager() {
    FreeBatch();
    LOG_DEBUG("BatchManager", "BatchManager destroyed");
}

inline bool BatchManager::InitializeBatch() {
    if (!llama_ctx) {
        LOG_ERROR("BatchManager", "Cannot initialize batch - llama context is null");
        return false;
    }
    
    if (batch_initialized) {
        LOG_DEBUG("BatchManager", "Batch already initialized");
        return true;
    }
    
    try {
        // Get batch size from llama context
        int32_t n_batch = llama_n_batch(llama_ctx);
        if (n_batch <= 0) {
            LOG_ERROR("BatchManager", "Invalid batch size from llama context: " + std::to_string(n_batch));
            return false;
        }
        
        // Initialize batch structure
        batch = llama_batch_init(n_batch, 0, 1);
        if (!batch.token) {
            LOG_ERROR("BatchManager", "Failed to initialize llama_batch structure");
            return false;
        }
        
        batch_initialized = true;
        LOG_DEBUG("BatchManager", "Batch initialized with size: " + std::to_string(n_batch));
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("BatchManager", "Exception during batch initialization: " + std::string(e.what()));
        return false;
    }
}

inline void BatchManager::ClearBatch() {
    if (!batch_initialized) {
        return;
    }
    
    batch.n_tokens = 0;
}

inline void BatchManager::FreeBatch() {
    if (batch_initialized) {
        llama_batch_free(batch);
        batch_initialized = false;
        LOG_DEBUG("BatchManager", "Batch freed");
    }
}

inline bool BatchManager::ValidateContextBounds(size_t token_count, int32_t current_position) const {
    if (max_context_tokens == 0) {
        LOG_ERROR("BatchManager", "Max context tokens not set - cannot validate bounds");
        return false;
    }
    
    if (current_position < 0) {
        LOG_ERROR("BatchManager", "Invalid current position: " + std::to_string(current_position));
        return false;
    }
    
    const int32_t safety_buffer = ContextConstants::SAFETY_BUFFER_TOKENS;
    const int32_t max_safe_position = static_cast<int32_t>(max_context_tokens) - safety_buffer;
    
    if (current_position >= max_safe_position) {
        LOG_ERROR("BatchManager", "Current position (" + std::to_string(current_position) + 
                 ") exceeds safe context limit (" + std::to_string(max_safe_position) + ")");
        return false;
    }
    
    const int32_t available_space = max_safe_position - current_position;
    if (static_cast<int32_t>(token_count) > available_space) {
        LOG_ERROR("BatchManager", "Token count (" + std::to_string(token_count) + 
                 ") exceeds available safe space (" + std::to_string(available_space) + ")");
        return false;
    }
    
    return true;
}

inline bool BatchManager::ValidateBatchParameters() const {
    if (!llama_ctx) {
        LOG_ERROR("BatchManager", "Llama context is null");
        return false;
    }
    
    if (!batch_initialized) {
        LOG_ERROR("BatchManager", "Batch not initialized");
        return false;
    }
    
    if (!n_past_ref) {
        LOG_ERROR("BatchManager", "n_past reference is null");
        return false;
    }
    
    return true;
}

inline bool BatchManager::ProcessTokensBatch(const std::vector<int32_t>& tokens) {
    if (!ValidateBatchParameters()) {
        return false;
    }
    
    if (tokens.empty()) {
        return true;
    }
    
    const int32_t current_position = *n_past_ref;
    
    // Validate context bounds before processing
    if (!ValidateContextBounds(tokens.size(), current_position)) {
        return false;
    }
    
    try {
        // Get maximum batch size from llama context
        const int32_t n_batch = llama_n_batch(llama_ctx);
        const size_t max_batch_size = static_cast<size_t>(n_batch);
        
        // Process tokens in chunks
        for (size_t i = 0; i < tokens.size(); i += max_batch_size) {
            const size_t end = std::min(i + max_batch_size, tokens.size());
            const size_t chunk_size = end - i;
            
            // Final bounds check for this chunk
            const int32_t chunk_start_position = *n_past_ref;
            if (!ValidateContextBounds(chunk_size, chunk_start_position)) {
                LOG_ERROR("BatchManager", "Chunk would exceed context bounds - stopping processing");
                break;
            }
            
            // Clear batch for new chunk
            ClearBatch();
            
            // Add tokens to batch
            for (size_t j = i; j < end && batch.n_tokens < max_batch_size; ++j) {
                const size_t batch_idx = j - i;
                const int32_t token_position = *n_past_ref + static_cast<int32_t>(batch_idx);
                
                batch.token[batch.n_tokens] = static_cast<llama_token>(tokens[j]);
                batch.pos[batch.n_tokens] = token_position;
                batch.n_seq_id[batch.n_tokens] = 1;
                batch.seq_id[batch.n_tokens][0] = 0;
                batch.logits[batch.n_tokens] = (j == end - 1); // Only last token gets logits
                batch.n_tokens++;
            }
            
            // Skip empty batches
            if (batch.n_tokens == 0) {
                LOG_DEBUG("BatchManager", "Skipping empty batch");
                break;
            }
            
            LOG_DEBUG("BatchManager", "Processing batch: " + std::to_string(batch.n_tokens) + 
                     " tokens at positions " + std::to_string(*n_past_ref) + 
                     " to " + std::to_string(*n_past_ref + batch.n_tokens - 1));
            
            // Execute batch decode
            const int result = llama_decode(llama_ctx, batch);
            if (result != 0) {
                LOG_ERROR("BatchManager", "Batch decode failed with result: " + std::to_string(result));
                return false;
            }
            
            // Update position
            *n_past_ref += static_cast<int32_t>(batch.n_tokens);
        }
        
        LOG_DEBUG("BatchManager", "Successfully processed " + std::to_string(tokens.size()) + 
                 " tokens, new position: " + std::to_string(*n_past_ref));
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("BatchManager", "Exception during token processing: " + std::string(e.what()));
        return false;
    }
}

inline bool BatchManager::ProcessSingleToken(int32_t token, bool generate_logits) {
    if (!ValidateBatchParameters()) {
        return false;
    }
    
    const int32_t current_position = *n_past_ref;
    
    // Validate bounds for single token
    if (!ValidateContextBounds(1, current_position)) {
        return false;
    }
    
    try {
        // Clear batch and add single token
        ClearBatch();
        
        batch.token[0] = static_cast<llama_token>(token);
        batch.pos[0] = current_position;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = generate_logits;
        batch.n_tokens = 1;
        
        LOG_DEBUG("BatchManager", "Processing single token at position " + std::to_string(current_position));
        
        // Execute decode
        const int result = llama_decode(llama_ctx, batch);
        if (result != 0) {
            LOG_ERROR("BatchManager", "Single token decode failed with result: " + std::to_string(result));
            return false;
        }
        
        // Update position
        (*n_past_ref)++;
        
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("BatchManager", "Exception during single token processing: " + std::string(e.what()));
        return false;
    }
}

inline bool BatchManager::AddTokenToBatch(int32_t token, int32_t position, bool generate_logits) {
    if (!batch_initialized) {
        LOG_ERROR("BatchManager", "Cannot add token - batch not initialized");
        return false;
    }
    
    const int32_t n_batch = llama_n_batch(llama_ctx);
    if (batch.n_tokens >= n_batch) {
        LOG_ERROR("BatchManager", "Batch is full - cannot add more tokens");
        return false;
    }
    
    batch.token[batch.n_tokens] = static_cast<llama_token>(token);
    batch.pos[batch.n_tokens] = position;
    batch.n_seq_id[batch.n_tokens] = 1;
    batch.seq_id[batch.n_tokens][0] = 0;
    batch.logits[batch.n_tokens] = generate_logits;
    batch.n_tokens++;
    
    return true;
}

inline bool BatchManager::ExecuteBatch() {
    if (!ValidateBatchParameters()) {
        return false;
    }
    
    if (batch.n_tokens == 0) {
        LOG_DEBUG("BatchManager", "No tokens in batch to execute");
        return true;
    }
    
    try {
        LOG_DEBUG("BatchManager", "Executing batch with " + std::to_string(batch.n_tokens) + " tokens");
        
        const int result = llama_decode(llama_ctx, batch);
        if (result != 0) {
            LOG_ERROR("BatchManager", "Batch execution failed with result: " + std::to_string(result));
            return false;
        }
        
        // Update position by batch size
        *n_past_ref += static_cast<int32_t>(batch.n_tokens);
        
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("BatchManager", "Exception during batch execution: " + std::string(e.what()));
        return false;
    }
}

inline bool BatchManager::ValidateContextState() const {
    if (!llama_ctx) {
        LOG_ERROR("BatchManager", "Llama context is null");
        return false;
    }
    
    if (!n_past_ref) {
        LOG_ERROR("BatchManager", "n_past reference is null");
        return false;
    }
    
    const int32_t current_position = *n_past_ref;
    if (current_position < 0) {
        LOG_ERROR("BatchManager", "Invalid current position: " + std::to_string(current_position));
        return false;
    }
    
    if (max_context_tokens > 0 && current_position >= static_cast<int32_t>(max_context_tokens)) {
        LOG_ERROR("BatchManager", "Current position (" + std::to_string(current_position) + 
                 ") exceeds context size (" + std::to_string(max_context_tokens) + ")");
        return false;
    }
    
    return true;
}

inline void BatchManager::UpdateContextBounds(size_t new_max_tokens) {
    max_context_tokens = new_max_tokens;
    LOG_DEBUG("BatchManager", "Updated context bounds to: " + std::to_string(new_max_tokens));
}
