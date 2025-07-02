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
#include <atomic>       // For std::atomic

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
    
    // Compile-time computation helper
    [[nodiscard]] consteval size_t compute_cache_aligned_size(size_t size) noexcept {
        return ((size + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE) * CACHE_LINE_SIZE;
    }
}

// Lock-free atomic statistics for BatchManager performance tracking
struct alignas(BatchManagerConstants::CACHE_LINE_SIZE) BatchStats {
    std::atomic<uint64_t> total_batches_processed{0};
    std::atomic<uint64_t> total_tokens_processed{0};
    std::atomic<uint64_t> batch_failures{0};
    std::atomic<uint64_t> context_bound_violations{0};
    std::atomic<uint64_t> memory_fence_operations{0};
    
    // High-performance atomic updates with relaxed ordering for hot paths
    [[carries_dependency]] inline void RecordBatchProcessed(size_t token_count) noexcept {
        total_batches_processed.fetch_add(1, std::memory_order_relaxed);
        total_tokens_processed.fetch_add(token_count, std::memory_order_relaxed);
    }
    
    [[carries_dependency]] inline void RecordBatchFailure() noexcept {
        batch_failures.fetch_add(1, std::memory_order_relaxed);
    }
    
    [[carries_dependency]] inline void RecordContextBoundViolation() noexcept {
        context_bound_violations.fetch_add(1, std::memory_order_relaxed);
    }
    
    [[carries_dependency]] inline void RecordMemoryFence() noexcept {
        memory_fence_operations.fetch_add(1, std::memory_order_relaxed);
    }
    
    // Lock-free ratio calculations
    [[nodiscard]] double GetSuccessRate() const noexcept {
        const uint64_t processed = total_batches_processed.load(std::memory_order_relaxed);
        const uint64_t failed = batch_failures.load(std::memory_order_relaxed);
        const uint64_t total = processed + failed;
        return total > 0 ? static_cast<double>(processed) / total : 1.0;
    }
};

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
    std::atomic<bool> batch_initialized{false};  // Atomic for lock-free checks
    
    // Context bounds and safety - frequently accessed together, optimized layout
    size_t max_context_tokens = 0;
    int32_t* n_past_ref = nullptr;  // Reference to parent's n_past for coordination
    
    // Cached values for performance - aligned for optimal access with atomic updates
    alignas(16) mutable std::atomic<int32_t> cached_n_batch{0};           // Atomic cache batch size
    mutable std::atomic<int32_t> cached_safety_limit{0};      // Atomic cache safety limit
    mutable std::atomic<int32_t> cached_max_batch_tokens{0};  // Atomic cache maximum tokens per batch
    
    // Lock-free performance statistics - could potentially use [[no_unique_address]] for future optimizations
    mutable BatchStats stats_;
    
    // Atomic error state tracking for lock-free error detection
    mutable std::atomic<bool> has_error_state{false};
    mutable std::atomic<uint32_t> last_error_code{0};
    
    // Template-based validation for compile-time optimization
    template<bool CheckBounds = true, bool ValidatePointers = true>
    [[nodiscard]] [[msvc::forceinline]] constexpr bool ValidateContextBoundsInternal([[maybe_unused]] size_t token_count, [[maybe_unused]] int32_t current_position) const noexcept;
    
    template<bool ValidatePointers = true>
    [[nodiscard]] [[msvc::forceinline]] constexpr bool ValidateBatchParametersInternal() const noexcept;
    
    // Compile-time helpers for bounds checking
    [[nodiscard]] static consteval bool is_valid_chunk_size(int32_t chunk_size) noexcept {
        return chunk_size >= BatchManagerConstants::MIN_CHUNK_SIZE && chunk_size > 0;
    }
    
    [[nodiscard]] static constexpr bool should_use_memory_fence(size_t chunk_size) noexcept {
        return chunk_size >= BatchManagerConstants::MEMORY_FENCE_THRESHOLD;
    }
    
    // Compile-time validation of template parameters
    template<bool CheckBounds, bool ValidatePointers>
    [[nodiscard]] static consteval bool validate_template_params() noexcept {
        return true; // Add specific validation logic if needed
    }
    
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
    [[nodiscard]] [[msvc::forceinline]] bool ProcessSingleTokenFast(int32_t token, bool generate_logits = true) noexcept;
    
    // Context state management - optimized
    [[nodiscard]] bool ValidateContextState() const noexcept;
    [[msvc::forceinline]] inline void UpdateContextBounds(size_t new_max_tokens) noexcept;
    
    // Getters for state inspection - force inline for performance
    [[nodiscard]] [[msvc::forceinline]] constexpr size_t GetMaxContextTokens() const noexcept { return max_context_tokens; }
    [[nodiscard]] [[msvc::forceinline]] inline int32_t GetCurrentPosition() const noexcept { return n_past_ref ? *n_past_ref : 0; }
    [[nodiscard]] [[msvc::forceinline]] inline bool IsBatchInitialized() const noexcept { return batch_initialized.load(std::memory_order_relaxed); }
    [[nodiscard]] [[msvc::forceinline]] inline size_t GetBatchSize() const noexcept { return IsBatchInitialized() ? static_cast<size_t>(batch.n_tokens) : 0; }
    [[nodiscard]] [[msvc::forceinline]] inline int32_t GetCachedBatchSize() const noexcept { return cached_n_batch.load(std::memory_order_relaxed); }
    
    // Lock-free performance statistics access
    [[nodiscard]] inline const BatchStats& GetStats() const noexcept { return stats_; }
    [[nodiscard]] inline bool HasErrorState() const noexcept { return has_error_state.load(std::memory_order_relaxed); }
    [[nodiscard]] inline uint32_t GetLastErrorCode() const noexcept { return last_error_code.load(std::memory_order_relaxed); }
    
    // Atomic state management
    [[carries_dependency]] inline void ClearErrorState() noexcept { 
        has_error_state.store(false, std::memory_order_relaxed);
        last_error_code.store(0, std::memory_order_relaxed);
    }
    
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
            // Pre-cache frequently used values for performance with atomic updates
            cached_n_batch.store(llama_n_batch(llama_ctx), std::memory_order_relaxed);
            cached_safety_limit.store(static_cast<int32_t>(max_tokens) - ContextConstants::SAFETY_BUFFER_TOKENS, std::memory_order_relaxed);
            cached_max_batch_tokens.store(cached_n_batch.load(std::memory_order_relaxed), std::memory_order_relaxed); // Cache max batch tokens
            
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
    
    // Get batch size - use cached atomic value if available
    int32_t n_batch = cached_n_batch.load(std::memory_order_relaxed);
    if (n_batch <= 0) [[unlikely]] {
        n_batch = llama_n_batch(llama_ctx);
        cached_n_batch.store(n_batch, std::memory_order_relaxed);  // Atomic cache for future use
        
        if (n_batch <= 0) [[unlikely]] {
            LOG_ERROR("BatchManager", "Invalid batch size from llama context: " + std::to_string(n_batch));
            return false;
        }
    }
    
    // Initialize batch structure - no exception handling for performance
    batch = llama_batch_init(n_batch, 0, 1);
    if (!batch.token) [[unlikely]] {
        LOG_ERROR("BatchManager", "Failed to initialize llama_batch structure");
        has_error_state.store(true, std::memory_order_relaxed);
        last_error_code.store(1, std::memory_order_relaxed);
        return false;
    }
    
    batch_initialized.store(true, std::memory_order_release);  // Use release ordering to ensure visibility
    if constexpr (BatchManagerConstants::ENABLE_DEBUG_LOGGING) {
        LOG_DEBUG("BatchManager", "Batch initialized with size: " + std::to_string(n_batch));
    }
    return true;
}

[[msvc::forceinline]] inline void BatchManager::ClearBatch() noexcept {
    if (batch_initialized.load(std::memory_order_acquire)) [[likely]] {
        batch.n_tokens = 0;
    }
}

inline void BatchManager::FreeBatch() noexcept {
    if (batch_initialized.load(std::memory_order_acquire)) [[likely]] {
        llama_batch_free(batch);
        batch_initialized.store(false, std::memory_order_release);
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
    
    // Use cached atomic batch size for performance
    const size_t max_batch_size = static_cast<size_t>(cached_n_batch.load(std::memory_order_relaxed));
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
        
        // Fast bounds check using unchecked version with atomic cached limit
        const int32_t chunk_start_position = *n_past_ref;
        const int32_t safety_limit = cached_safety_limit.load(std::memory_order_relaxed);
        if (chunk_start_position + static_cast<int32_t>(chunk_size) > safety_limit) [[unlikely]] {
            LOG_ERROR("BatchManager", "Chunk would exceed context bounds - stopping processing");
            stats_.RecordContextBoundViolation();
            break;
        }
        
        // Clear batch for new chunk - single operation
        batch.n_tokens = 0;
        
        // Optimized token batch building - eliminate bounds checking in inner loop
        const int32_t* __restrict token_ptr = tokens.data() + i;
        const int32_t batch_start_position = *n_past_ref;
        
        // Add compiler hint for loop optimization
        #pragma unroll 4
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
        if (should_use_memory_fence(chunk_size)) {
            std::atomic_thread_fence(std::memory_order_seq_cst);
            stats_.RecordMemoryFence();
        }
        
        const int result = llama_decode(llama_ctx, batch);
        if (result != 0) [[unlikely]] {
            LOG_ERROR("BatchManager", "Batch decode failed with result: " + std::to_string(result));
            stats_.RecordBatchFailure();
            has_error_state.store(true, std::memory_order_relaxed);
            last_error_code.store(static_cast<uint32_t>(result), std::memory_order_relaxed);
            return false;
        }
        
        // Record successful batch processing
        stats_.RecordBatchProcessed(chunk_size);
        
        // Update position
        *n_past_ref += static_cast<int32_t>(chunk_size);
        
        // Record batch processing statistics
        stats_.RecordBatchProcessed(chunk_size);
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
[[nodiscard]] [[msvc::forceinline]] bool BatchManager::ProcessSingleTokenFast(int32_t token, bool generate_logits) noexcept {
    // Assumes: batch_initialized=true, llama_ctx!=null, n_past_ref!=null, bounds are valid
    // Caller must ensure these preconditions for maximum performance
    
    const int32_t current_position = *n_past_ref;
    
    // Add compiler assumptions for better optimization (C++23/compiler extension)
    #ifdef __has_cpp_attribute
        #if __has_cpp_attribute(assume)
            [[assume(llama_ctx != nullptr)]];
            [[assume(n_past_ref != nullptr)]];
            [[assume(batch_initialized.load(std::memory_order_relaxed) == true)]];
        #endif
    #endif
    
    // Ultra-fast bounds check using cached atomic limit
    const int32_t safety_limit = cached_safety_limit.load(std::memory_order_relaxed);
    if (current_position >= safety_limit) [[unlikely]] {
        stats_.RecordContextBoundViolation();
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
        stats_.RecordBatchProcessed(1);
        return true;
    } else {
        stats_.RecordBatchFailure();
        has_error_state.store(true, std::memory_order_relaxed);
        last_error_code.store(static_cast<uint32_t>(result), std::memory_order_relaxed);
    }
    
    return false;
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
    cached_safety_limit.store(static_cast<int32_t>(new_max_tokens) - ContextConstants::SAFETY_BUFFER_TOKENS, std::memory_order_relaxed);
    
    if constexpr (BatchManagerConstants::ENABLE_DEBUG_LOGGING) {
        LOG_DEBUG("BatchManager", "Updated context bounds to: " + std::to_string(new_max_tokens));
    }
}

// Template-based validation implementations for compile-time optimization
template<bool CheckBounds, bool ValidatePointers>
[[nodiscard]] [[msvc::forceinline]] constexpr bool BatchManager::ValidateContextBoundsInternal([[maybe_unused]] size_t token_count, [[maybe_unused]] int32_t current_position) const noexcept {
    if constexpr (ValidatePointers) {
        if (!n_past_ref) [[unlikely]] return false;
    }
    
    if constexpr (CheckBounds) {
        if (max_context_tokens == 0) [[unlikely]] return false;
        if (current_position < 0) [[unlikely]] return false;
    }
    
    const int32_t available_space = cached_safety_limit.load(std::memory_order_relaxed);
    return static_cast<int32_t>(token_count) <= available_space - current_position;
}

template<bool ValidatePointers>
[[nodiscard]] [[msvc::forceinline]] constexpr bool BatchManager::ValidateBatchParametersInternal() const noexcept {
    if constexpr (ValidatePointers) {
        if (!llama_ctx) [[unlikely]] return false;
        if (!n_past_ref) [[unlikely]] return false;
    }
    
    return batch_initialized.load(std::memory_order_acquire);
}
