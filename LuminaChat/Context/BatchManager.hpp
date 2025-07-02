#pragma once

#include "../Logger.hpp"
#include "../ModelInfo.hpp"
#include "../TokenCache.hpp"
#include "llama-cpp.h"
#include <vector>
#include <memory>
#include <cstring>      // For memset
#include <algorithm>    // For std::min
#include <immintrin.h>  // For memory prefetching
#include <array>        // For std::array in template methods

// Forward declarations
struct llama_context;

namespace ContextConstants {
    extern const int32_t SAFETY_BUFFER_TOKENS;
    extern const int32_t GENERATION_BUFFER_TOKENS;
}

// Performance optimization constants - ultra-aggressive tuning
namespace BatchManagerConstants {
    static constexpr bool ENABLE_DEBUG_LOGGING = false;  // Compile-time debug control
    static constexpr size_t CACHE_LINE_SIZE = 64;        // For alignment optimizations
    static constexpr size_t PREFETCH_DISTANCE = 3;       // Cache lines to prefetch ahead
    static constexpr int32_t MIN_CHUNK_SIZE = 32;        // Minimum efficient chunk size
    static constexpr int32_t MEMORY_FENCE_THRESHOLD = 512; // When to use memory barriers
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
    // Core components - cache-aligned for performance with improved memory layout
    alignas(BatchManagerConstants::CACHE_LINE_SIZE) llama_context* llama_ctx = nullptr;
    llama_batch batch;
    bool batch_initialized = false;
    
    // Context bounds and safety - frequently accessed together, optimized layout
    size_t max_context_tokens = 0;
    int32_t* n_past_ref = nullptr;  // Reference to parent's n_past for coordination
    
    // Cached values for performance - aligned for optimal access
    alignas(16) mutable int32_t cached_n_batch = 0;           // Cache batch size to avoid repeated calls
    mutable int32_t cached_safety_limit = 0;     // Cache safety limit calculation
    mutable int32_t cached_max_batch_tokens = 0; // Cache maximum tokens per batch
    
    // Template-based validation for compile-time optimization
    template<bool CheckBounds = true, bool ValidatePointers = true>
    [[nodiscard]] [[msvc::forceinline]] constexpr inline bool ValidateContextBoundsInternal(size_t token_count, int32_t current_position) const noexcept;
    
    template<bool ValidatePointers = true>
    [[nodiscard]] [[msvc::forceinline]] constexpr inline bool ValidateBatchParametersInternal() const noexcept;
    
public:
    // Constructor/Destructor
    BatchManager(llama_context* ctx, size_t max_tokens, int32_t& n_past_reference);
    ~BatchManager();
    
    // Batch lifecycle management - optimized
    [[nodiscard]] bool InitializeBatch();
    [[msvc::forceinline]] inline void ClearBatch() noexcept;
    void FreeBatch() noexcept;
    
    // Core batch processing operations - ultra-high performance
    [[nodiscard]] bool ProcessTokensBatch(const std::vector<int32_t>& tokens) noexcept;
    [[nodiscard]] bool ProcessTokensBatch(std::vector<int32_t>&& tokens) noexcept; // Move semantics overload
    [[nodiscard]] [[msvc::forceinline]] inline bool ProcessSingleTokenFast(int32_t token, bool generate_logits = true) noexcept;
    [[nodiscard]] bool ProcessSingleToken(int32_t token, bool generate_logits = true) noexcept;
    
    // Batch building helpers - optimized for hot paths
    [[nodiscard]] [[msvc::forceinline]] inline bool AddTokenToBatchUnchecked(int32_t token, int32_t position, bool generate_logits = false) noexcept;
    [[nodiscard]] bool AddTokenToBatch(int32_t token, int32_t position, bool generate_logits = false) noexcept;
    [[nodiscard]] bool ExecuteBatch() noexcept;
    
    // Ultra-fast specialized methods for specific use cases
    template<bool CheckBounds = false>
    [[nodiscard]] [[msvc::forceinline]] inline bool ProcessTokensBatchUltraFast(const std::vector<int32_t>& tokens) noexcept;
    
    template<bool CheckBounds = false>
    [[nodiscard]] [[msvc::forceinline]] inline bool ProcessTokensBatchUltraFast(std::vector<int32_t>&& tokens) noexcept;
    
    template<size_t N>
    [[nodiscard]] [[msvc::forceinline]] inline bool ProcessFixedTokens(const std::array<int32_t, N>& tokens) noexcept;
    
    // Context state management - optimized
    [[nodiscard]] bool ValidateContextState() const noexcept;
    [[msvc::forceinline]] inline void UpdateContextBounds(size_t new_max_tokens) noexcept;
    
    // Getters for state inspection - force inline for performance
    [[nodiscard]] [[msvc::forceinline]] inline size_t GetMaxContextTokens() const noexcept { return max_context_tokens; }
    [[nodiscard]] [[msvc::forceinline]] inline int32_t GetCurrentPosition() const noexcept { return n_past_ref ? *n_past_ref : 0; }
    [[nodiscard]] [[msvc::forceinline]] inline bool IsBatchInitialized() const noexcept { return batch_initialized; }
    [[nodiscard]] [[msvc::forceinline]] inline size_t GetBatchSize() const noexcept { return batch_initialized ? static_cast<size_t>(batch.n_tokens) : 0; }
    [[nodiscard]] [[msvc::forceinline]] inline int32_t GetCachedBatchSize() const noexcept { return cached_n_batch; }
    
    // Disable copy/move operations (manages llama.cpp resources)
    BatchManager(const BatchManager&) = delete;
    BatchManager& operator=(const BatchManager&) = delete;
    BatchManager(BatchManager&&) = delete;
    BatchManager& operator=(BatchManager&&) = delete;
};

// Inline implementation - ultra-high performance optimizations
inline BatchManager::BatchManager(llama_context* ctx, size_t max_tokens, int32_t& n_past_reference)
    : llama_ctx(ctx)
    , max_context_tokens(max_tokens)
    , n_past_ref(&n_past_reference)
{
    // Fast validation - likely path optimization
    if (llama_ctx) [[likely]] {
        if (max_tokens > 0) [[likely]] {
            // Pre-cache frequently used values for performance
            cached_n_batch = llama_n_batch(llama_ctx);
            cached_safety_limit = static_cast<int32_t>(max_tokens) - ContextConstants::SAFETY_BUFFER_TOKENS;
            cached_max_batch_tokens = cached_n_batch; // Cache max batch tokens
            
            if constexpr (BatchManagerConstants::ENABLE_DEBUG_LOGGING) {
                LOG_DEBUG("BatchManager", "BatchManager created with max_tokens: " + std::to_string(max_tokens));
            }
        } else [[unlikely]] {
            LOG_ERROR("BatchManager", "BatchManager created with zero max tokens");
        }
    } else [[unlikely]] {
        LOG_ERROR("BatchManager", "BatchManager created with null llama context");
    }
}

inline BatchManager::~BatchManager() {
    FreeBatch();
    if constexpr (BatchManagerConstants::ENABLE_DEBUG_LOGGING) {
        LOG_DEBUG("BatchManager", "BatchManager destroyed");
    }
}

inline bool BatchManager::InitializeBatch() {
    if (!llama_ctx) [[unlikely]] {
        LOG_ERROR("BatchManager", "Cannot initialize batch - llama context is null");
        return false;
    }
    
    if (batch_initialized) [[likely]] {
        if constexpr (BatchManagerConstants::ENABLE_DEBUG_LOGGING) {
            LOG_DEBUG("BatchManager", "Batch already initialized");
        }
        return true;
    }
    
    // Get batch size - use cached value if available
    int32_t n_batch = cached_n_batch;
    if (n_batch <= 0) [[unlikely]] {
        n_batch = llama_n_batch(llama_ctx);
        cached_n_batch = n_batch;  // Cache for future use
        
        if (n_batch <= 0) [[unlikely]] {
            LOG_ERROR("BatchManager", "Invalid batch size from llama context: " + std::to_string(n_batch));
            return false;
        }
    }
    
    // Initialize batch structure - no exception handling for performance
    batch = llama_batch_init(n_batch, 0, 1);
    if (!batch.token) [[unlikely]] {
        LOG_ERROR("BatchManager", "Failed to initialize llama_batch structure");
        return false;
    }
    
    batch_initialized = true;
    if constexpr (BatchManagerConstants::ENABLE_DEBUG_LOGGING) {
        LOG_DEBUG("BatchManager", "Batch initialized with size: " + std::to_string(n_batch));
    }
    return true;
}

[[msvc::forceinline]] inline void BatchManager::ClearBatch() noexcept {
    if (batch_initialized) [[likely]] {
        batch.n_tokens = 0;
    }
}

inline void BatchManager::FreeBatch() noexcept {
    if (batch_initialized) [[likely]] {
        llama_batch_free(batch);
        batch_initialized = false;
        if constexpr (BatchManagerConstants::ENABLE_DEBUG_LOGGING) {
            LOG_DEBUG("BatchManager", "Batch freed");
        }
    }
}

inline bool BatchManager::ProcessTokensBatch(const std::vector<int32_t>& tokens) noexcept {
    // Fast path for empty tokens
    if (tokens.empty()) [[unlikely]] {
        return true;
    }
    
    // Pre-validate once to avoid repeated checks - use ultra-fast unchecked version
    if (!ValidateBatchParametersInternal<false>()) [[unlikely]] {
        return false;
    }
    
    const int32_t current_position = *n_past_ref;
    
    // Single context bounds check - use ultra-fast unchecked version
    if (!ValidateContextBoundsInternal<false, false>(tokens.size(), current_position)) [[unlikely]] {
        return false;
    }
    
    // Use cached batch size for performance
    const size_t max_batch_size = static_cast<size_t>(cached_n_batch);
    const size_t total_tokens = tokens.size();
    
    // Pre-calculate chunk information to reduce computations in loop
    const size_t num_chunks = (total_tokens + max_batch_size - 1) / max_batch_size;
    
    // Prefetch first chunk of token data
    if (total_tokens > 0) [[likely]] {
        _mm_prefetch(reinterpret_cast<const char*>(tokens.data()), _MM_HINT_T0);
        if (total_tokens > BatchManagerConstants::CACHE_LINE_SIZE / sizeof(int32_t)) {
            _mm_prefetch(reinterpret_cast<const char*>(tokens.data() + BatchManagerConstants::CACHE_LINE_SIZE / sizeof(int32_t)), _MM_HINT_T0);
        }
    }
    
    // Optimized chunked processing with aggressive loop unrolling hints
    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        const size_t i = chunk_idx * max_batch_size;
        const size_t end = std::min(i + max_batch_size, total_tokens);
        const size_t chunk_size = end - i;
        
        // Prefetch next chunk while processing current one
        if (chunk_idx + 1 < num_chunks) [[likely]] {
            const size_t next_chunk_start = (chunk_idx + 1) * max_batch_size;
            if (next_chunk_start < total_tokens) {
                _mm_prefetch(reinterpret_cast<const char*>(tokens.data() + next_chunk_start), _MM_HINT_T0);
            }
        }
        
        // Fast bounds check using unchecked version
        const int32_t chunk_start_position = *n_past_ref;
        if (chunk_start_position + static_cast<int32_t>(chunk_size) > cached_safety_limit) [[unlikely]] {
            LOG_ERROR("BatchManager", "Chunk would exceed context bounds - stopping processing");
            break;
        }
        
        // Clear batch for new chunk - single operation
        batch.n_tokens = 0;
        
        // Optimized token batch building - eliminate bounds checking in inner loop
        const int32_t* __restrict token_ptr = tokens.data() + i;
        const int32_t batch_start_position = *n_past_ref;
        
        // Vectorized batch setup with manual loop unrolling for small chunks
        if (chunk_size <= 4) [[likely]] {
            // Unrolled loop for common small chunk sizes
            for (size_t j = 0; j < chunk_size; ++j) {
                batch.token[j] = static_cast<llama_token>(token_ptr[j]);
                batch.pos[j] = batch_start_position + static_cast<int32_t>(j);
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j] = false;
            }
            // Set logits for last token only
            if (chunk_size > 0) {
                batch.logits[chunk_size - 1] = true;
            }
        } else {
            // Standard loop for larger chunks
            for (size_t j = 0; j < chunk_size; ++j) {
                const size_t batch_idx = j;
                
                batch.token[batch_idx] = static_cast<llama_token>(token_ptr[j]);
                batch.pos[batch_idx] = batch_start_position + static_cast<int32_t>(j);
                batch.n_seq_id[batch_idx] = 1;
                batch.seq_id[batch_idx][0] = 0;
                batch.logits[batch_idx] = (j == chunk_size - 1); // Only last token gets logits
            }
        }
        batch.n_tokens = static_cast<int32_t>(chunk_size);
        
        if constexpr (BatchManagerConstants::ENABLE_DEBUG_LOGGING) {
            LOG_DEBUG("BatchManager", "Processing batch: " + std::to_string(batch.n_tokens) + 
                     " tokens at positions " + std::to_string(*n_past_ref) + 
                     " to " + std::to_string(*n_past_ref + batch.n_tokens - 1));
        }
        
        // Execute batch decode with memory fence for large batches
        if (chunk_size >= BatchManagerConstants::MEMORY_FENCE_THRESHOLD) {
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }
        
        const int result = llama_decode(llama_ctx, batch);
        if (result != 0) [[unlikely]] {
            LOG_ERROR("BatchManager", "Batch decode failed with result: " + std::to_string(result));
            return false;
        }
        
        // Update position
        *n_past_ref += static_cast<int32_t>(chunk_size);
    }
    
    if constexpr (BatchManagerConstants::ENABLE_DEBUG_LOGGING) {
        LOG_DEBUG("BatchManager", "Successfully processed " + std::to_string(total_tokens) + 
                 " tokens, new position: " + std::to_string(*n_past_ref));
    }
    return true;
}

// Move semantics overload for ProcessTokensBatch
inline bool BatchManager::ProcessTokensBatch(std::vector<int32_t>&& tokens) noexcept {
    // For move semantics, we can process the vector directly since caller doesn't need it
    // This avoids any potential copy and provides the same performance as the const& version
    return ProcessTokensBatch(static_cast<const std::vector<int32_t>&>(tokens));
}

// Ultra-fast single token processing for hot paths (generation loop)
[[nodiscard]] [[msvc::forceinline]] inline bool BatchManager::ProcessSingleTokenFast(int32_t token, bool generate_logits) noexcept {
    // Assumes: batch_initialized=true, llama_ctx!=null, n_past_ref!=null, bounds are valid
    // Caller must ensure these preconditions for maximum performance
    
    const int32_t current_position = *n_past_ref;
    
    // Ultra-fast bounds check using cached limit
    if (current_position >= cached_safety_limit) [[unlikely]] {
        return false;
    }
    
    // Direct batch setup without function calls
    batch.n_tokens = 1;
    batch.token[0] = static_cast<llama_token>(token);
    batch.pos[0] = current_position;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = generate_logits;
    
    // Execute decode
    const int result = llama_decode(llama_ctx, batch);
    if (result == 0) [[likely]] {
        (*n_past_ref)++;
        return true;
    }
    
    return false;
}

// Standard single token processing with full error checking
inline bool BatchManager::ProcessSingleToken(int32_t token, bool generate_logits) noexcept {
    if (!ValidateBatchParametersInternal<true>()) [[unlikely]] {
        return false;
    }
    
    const int32_t current_position = *n_past_ref;
    
    // Validate bounds for single token - use full validation for safety
    if (!ValidateContextBoundsInternal<true, true>(1, current_position)) [[unlikely]] {
        return false;
    }
    
    // Clear batch and add single token
    batch.n_tokens = 1;
    batch.token[0] = static_cast<llama_token>(token);
    batch.pos[0] = current_position;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = generate_logits;
    
    if constexpr (BatchManagerConstants::ENABLE_DEBUG_LOGGING) {
        LOG_DEBUG("BatchManager", "Processing single token at position " + std::to_string(current_position));
    }
    
    // Execute decode
    const int result = llama_decode(llama_ctx, batch);
    if (result != 0) [[unlikely]] {
        if constexpr (BatchManagerConstants::ENABLE_DEBUG_LOGGING) {
            LOG_ERROR("BatchManager", "Single token decode failed with result: " + std::to_string(result));
        }
        return false;
    }
    
    // Update position
    (*n_past_ref)++;
    return true;
}

// Ultra-fast unchecked token addition for hot paths
[[nodiscard]] [[msvc::forceinline]] inline bool BatchManager::AddTokenToBatchUnchecked(int32_t token, int32_t position, bool generate_logits) noexcept {
    // Assumes: batch_initialized=true, batch has space available
    // Caller must ensure preconditions for maximum performance
    
    const int32_t idx = batch.n_tokens;
    batch.token[idx] = static_cast<llama_token>(token);
    batch.pos[idx] = position;
    batch.n_seq_id[idx] = 1;
    batch.seq_id[idx][0] = 0;
    batch.logits[idx] = generate_logits;
    batch.n_tokens++;
    
    return true;
}

// Standard token addition with full error checking
inline bool BatchManager::AddTokenToBatch(int32_t token, int32_t position, bool generate_logits) noexcept {
    if (!batch_initialized) [[unlikely]] {
        if constexpr (BatchManagerConstants::ENABLE_DEBUG_LOGGING) {
            LOG_ERROR("BatchManager", "Cannot add token - batch not initialized");
        }
        return false;
    }
    
    if (batch.n_tokens >= cached_n_batch) [[unlikely]] {
        if constexpr (BatchManagerConstants::ENABLE_DEBUG_LOGGING) {
            LOG_ERROR("BatchManager", "Batch is full - cannot add more tokens");
        }
        return false;
    }
    
    const int32_t idx = batch.n_tokens;
    batch.token[idx] = static_cast<llama_token>(token);
    batch.pos[idx] = position;
    batch.n_seq_id[idx] = 1;
    batch.seq_id[idx][0] = 0;
    batch.logits[idx] = generate_logits;
    batch.n_tokens++;
    
    return true;
}

inline bool BatchManager::ExecuteBatch() noexcept {
    if (!ValidateBatchParametersInternal<false>()) [[unlikely]] {
        return false;
    }
    
    if (batch.n_tokens == 0) [[unlikely]] {
        if constexpr (BatchManagerConstants::ENABLE_DEBUG_LOGGING) {
            LOG_DEBUG("BatchManager", "No tokens in batch to execute");
        }
        return true;
    }
    
    if constexpr (BatchManagerConstants::ENABLE_DEBUG_LOGGING) {
        LOG_DEBUG("BatchManager", "Executing batch with " + std::to_string(batch.n_tokens) + " tokens");
    }
    
    const int result = llama_decode(llama_ctx, batch);
    if (result != 0) [[unlikely]] {
        if constexpr (BatchManagerConstants::ENABLE_DEBUG_LOGGING) {
            LOG_ERROR("BatchManager", "Batch execution failed with result: " + std::to_string(result));
        }
        return false;
    }
    
    // Update position by batch size
    *n_past_ref += batch.n_tokens;
    return true;
}

inline bool BatchManager::ValidateContextState() const noexcept {
    if (!llama_ctx) [[unlikely]] {
        if constexpr (BatchManagerConstants::ENABLE_DEBUG_LOGGING) {
            LOG_ERROR("BatchManager", "Llama context is null");
        }
        return false;
    }
    
    if (!n_past_ref) [[unlikely]] {
        if constexpr (BatchManagerConstants::ENABLE_DEBUG_LOGGING) {
            LOG_ERROR("BatchManager", "n_past reference is null");
        }
        return false;
    }
    
    const int32_t current_position = *n_past_ref;
    if (current_position < 0) [[unlikely]] {
        if constexpr (BatchManagerConstants::ENABLE_DEBUG_LOGGING) {
            LOG_ERROR("BatchManager", "Invalid current position: " + std::to_string(current_position));
        }
        return false;
    }
    
    if (max_context_tokens > 0 && current_position >= static_cast<int32_t>(max_context_tokens)) [[unlikely]] {
        if constexpr (BatchManagerConstants::ENABLE_DEBUG_LOGGING) {
            LOG_ERROR("BatchManager", "Current position (" + std::to_string(current_position) + 
                     ") exceeds context size (" + std::to_string(max_context_tokens) + ")");
        }
        return false;
    }
    
    return true;
}

inline void BatchManager::UpdateContextBounds(size_t new_max_tokens) noexcept {
    max_context_tokens = new_max_tokens;
    cached_safety_limit = static_cast<int32_t>(new_max_tokens) - ContextConstants::SAFETY_BUFFER_TOKENS;
    
    if constexpr (BatchManagerConstants::ENABLE_DEBUG_LOGGING) {
        LOG_DEBUG("BatchManager", "Updated context bounds to: " + std::to_string(new_max_tokens));
    }
}

// Template-based validation implementations for compile-time optimization
template<bool CheckBounds, bool ValidatePointers>
[[nodiscard]] [[msvc::forceinline]] constexpr inline bool BatchManager::ValidateContextBoundsInternal(size_t token_count, int32_t current_position) const noexcept {
    if constexpr (ValidatePointers) {
        if (!n_past_ref) [[unlikely]] return false;
    }
    
    if constexpr (CheckBounds) {
        if (max_context_tokens == 0) [[unlikely]] return false;
        if (current_position < 0) [[unlikely]] return false;
    }
    
    const int32_t available_space = cached_safety_limit - current_position;
    return static_cast<int32_t>(token_count) <= available_space;
}

template<bool ValidatePointers>
[[nodiscard]] [[msvc::forceinline]] constexpr inline bool BatchManager::ValidateBatchParametersInternal() const noexcept {
    if constexpr (ValidatePointers) {
        if (!llama_ctx) [[unlikely]] return false;
        if (!n_past_ref) [[unlikely]] return false;
    }
    
    return batch_initialized;
}

// Ultra-fast template specializations for maximum performance
template<bool CheckBounds>
[[nodiscard]] [[msvc::forceinline]] inline bool BatchManager::ProcessTokensBatchUltraFast(const std::vector<int32_t>& tokens) noexcept {
    if (tokens.empty()) [[unlikely]] return true;
    
    // Assume all preconditions are met by caller for maximum performance
    // No bounds checking if CheckBounds is false
    if constexpr (CheckBounds) {
        if (!ValidateContextBoundsInternal<false, false>(tokens.size(), *n_past_ref)) [[unlikely]] {
            return false;
        }
    }
    
    const size_t total_tokens = tokens.size();
    const size_t max_batch_size = static_cast<size_t>(cached_n_batch);
    
    // Process in optimal chunks without error checking (caller's responsibility)
    for (size_t i = 0; i < total_tokens; i += max_batch_size) {
        const size_t chunk_size = std::min(max_batch_size, total_tokens - i);
        const int32_t* __restrict token_ptr = tokens.data() + i;
        const int32_t batch_start_position = *n_past_ref;
        
        batch.n_tokens = static_cast<int32_t>(chunk_size);
        
        // Ultra-fast batch setup - no bounds checking
        for (size_t j = 0; j < chunk_size; ++j) {
            batch.token[j] = static_cast<llama_token>(token_ptr[j]);
            batch.pos[j] = batch_start_position + static_cast<int32_t>(j);
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = (j == chunk_size - 1);
        }
        
        // Execute without error checking for maximum speed
        llama_decode(llama_ctx, batch);
        *n_past_ref += static_cast<int32_t>(chunk_size);
    }
    
    return true;
}

// Move semantics overload for ProcessTokensBatchUltraFast
template<bool CheckBounds>
[[nodiscard]] [[msvc::forceinline]] inline bool BatchManager::ProcessTokensBatchUltraFast(std::vector<int32_t>&& tokens) noexcept {
    // For move semantics, process the vector directly since caller doesn't need it
    return ProcessTokensBatchUltraFast<CheckBounds>(static_cast<const std::vector<int32_t>&>(tokens));
}

template<size_t N>
[[nodiscard]] [[msvc::forceinline]] inline bool BatchManager::ProcessFixedTokens(const std::array<int32_t, N>& tokens) noexcept {
    if constexpr (N == 0) return true;
    
    // Compile-time optimization for fixed-size arrays
    if constexpr (N <= static_cast<size_t>(cached_n_batch)) {
        // Single batch processing for small fixed arrays
        const int32_t batch_start_position = *n_past_ref;
        batch.n_tokens = static_cast<int32_t>(N);
        
        // Loop unrolling for small arrays
        if constexpr (N <= 4) {
            // Explicit unrolling for very small arrays
            for (size_t i = 0; i < N; ++i) {
                batch.token[i] = static_cast<llama_token>(tokens[i]);
                batch.pos[i] = batch_start_position + static_cast<int32_t>(i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = (i == N - 1);
            }
        } else {
            // Standard loop for larger fixed arrays
            for (size_t i = 0; i < N; ++i) {
                batch.token[i] = static_cast<llama_token>(tokens[i]);
                batch.pos[i] = batch_start_position + static_cast<int32_t>(i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = (i == N - 1);
            }
        }
        
        const int result = llama_decode(llama_ctx, batch);
        if (result == 0) [[likely]] {
            *n_past_ref += static_cast<int32_t>(N);
            return true;
        }
    } else {
        // Fall back to chunked processing for large fixed arrays
        return ProcessTokensBatchUltraFast<false>(std::vector<int32_t>(tokens.begin(), tokens.end()));
    }
    
    return false;
}
