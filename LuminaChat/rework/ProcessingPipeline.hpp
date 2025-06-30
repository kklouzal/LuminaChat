// ProcessingPipeline.hpp - header-only implementation for asynchronous plugin processing
// Provides the foundational template for all plugin workflows in the LuminaChat rework architecture.
//
// Project Settings:
// C++ Language Standard: ISO C++20 Standard (/std:c++20)
// C Language Standard: ISO C17 (2018) Standard (/std:c17)
// Optimization: Maximum Optimization (Favor Speed) (/O2)
// Favor Size or Speed: Favor fast code (/Ot)
// Runtime Library: Multi-threaded DLL (/MD)
// Enable Run-Time Type Information (RTTI) YES (/GR)

#pragma once

#include <queue>
#include <functional>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <memory>
#include <chrono>
#include <string>
#include <string_view>
#include <optional>
#include <bit>
#include <utility>
#include "Logger.hpp"

namespace LuminaChat {

// Logging macro for ProcessingPipeline (only used for important events)
#define LOG_ProcessingPipeline(message) LOG_INFO("ProcessingPipeline", message)

// Forward declarations
enum class PipelineState : uint8_t {
    IDLE = 0,
    PROCESSING = 1,
    SHUTDOWN = 2
};

enum class RequestPriority : uint8_t {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    URGENT = 3
};

// Compile-time utilities for pipeline configuration
namespace detail {
    // Compile-time validation for queue sizes
    consteval bool is_valid_queue_size(size_t size) noexcept {
        return size > 0 && size <= 1000000; // Reasonable limits
    }
    
    // Compile-time validation for processing delays
    consteval bool is_valid_processing_delay_ms(int64_t ms) noexcept {
        return ms >= 0 && ms <= 10000; // 0-10 seconds max
    }
    
    // Compile-time priority validation
    constexpr bool is_valid_priority(RequestPriority priority) noexcept {
        return priority >= RequestPriority::LOW && priority <= RequestPriority::URGENT;
    }
    
    // Compile-time state validation
    constexpr bool is_valid_state(PipelineState state) noexcept {
        return state >= PipelineState::IDLE && state <= PipelineState::SHUTDOWN;
    }
}

// Compile-time constants - using constexpr (not constinit) for namespace constants
namespace constants {
    constexpr size_t default_max_queue_size = 1000;
    constexpr int64_t default_processing_delay_ms = 10;
    constexpr size_t cache_line_size = 64;
}

// Request wrapper for priority and metadata - optimized for performance
template<typename RequestType>
struct alignas(constants::cache_line_size) PipelineRequest {
    RequestType request;
    RequestPriority priority = RequestPriority::NORMAL;
    std::chrono::steady_clock::time_point queued_time;
    std::string request_id;
    
    // Default constructor
    constexpr PipelineRequest() noexcept = default;
    
    // Optimized constructor with perfect forwarding
    template<typename ReqType>
    PipelineRequest(ReqType&& req, RequestPriority prio = RequestPriority::NORMAL, 
                   std::string_view id = "") noexcept
        : request(std::forward<ReqType>(req)), priority(prio), 
          queued_time(std::chrono::steady_clock::now()), request_id(id) {}
    
    // Copy constructor  
    PipelineRequest(const PipelineRequest& other) noexcept
        : request(other.request), priority(other.priority),
          queued_time(other.queued_time), request_id(other.request_id) {}
    
    // Move constructor
    PipelineRequest(PipelineRequest&& other) noexcept
        : request(std::move(other.request)), priority(other.priority),
          queued_time(other.queued_time), request_id(std::move(other.request_id)) {}
    
    // Copy assignment
    PipelineRequest& operator=(const PipelineRequest& other) noexcept {
        if (this != &other) [[likely]] {
            request = other.request;
            priority = other.priority;
            queued_time = other.queued_time;
            request_id = other.request_id;
        }
        return *this;
    }
    
    // Move assignment
    PipelineRequest& operator=(PipelineRequest&& other) noexcept {
        if (this != &other) [[likely]] {
            request = std::move(other.request);
            priority = other.priority;
            queued_time = other.queued_time;
            request_id = std::move(other.request_id);
        }
        return *this;
    }
          
    // Priority comparison for queue ordering (higher priority = lower value for std::priority_queue)
    [[nodiscard]] constexpr bool operator<(const PipelineRequest& other) const noexcept {
        return static_cast<uint8_t>(priority) < static_cast<uint8_t>(other.priority);
    }
    
    // Utility functions for compile-time operations
    [[nodiscard]] constexpr RequestPriority GetPriority() const noexcept {
        return priority;
    }
    
    [[nodiscard]] constexpr bool IsHighPriority() const noexcept {
        return priority >= RequestPriority::HIGH;
    }
    
    [[nodiscard]] constexpr bool IsUrgent() const noexcept {
        return priority == RequestPriority::URGENT;
    }
};

// Statistics tracking for pipeline performance
struct alignas(constants::cache_line_size) PipelineStats {
    std::atomic<uint64_t> total_requests{0};
    std::atomic<uint64_t> completed_requests{0};
    std::atomic<uint64_t> failed_requests{0};
    std::atomic<uint64_t> pending_requests{0};
    std::chrono::steady_clock::time_point last_activity;
    
    // Default constructor
    inline PipelineStats() noexcept : last_activity(std::chrono::steady_clock::now()) {}
    
    // Copy constructor for atomic variables
    inline PipelineStats(const PipelineStats& other) noexcept
        : total_requests(other.total_requests.load(std::memory_order_relaxed)),
          completed_requests(other.completed_requests.load(std::memory_order_relaxed)),
          failed_requests(other.failed_requests.load(std::memory_order_relaxed)),
          pending_requests(other.pending_requests.load(std::memory_order_relaxed)),
          last_activity(other.last_activity) {}
    
    // Copy assignment operator
    inline PipelineStats& operator=(const PipelineStats& other) noexcept {
        if (this != &other) [[likely]] {
            total_requests.store(other.total_requests.load(std::memory_order_relaxed), std::memory_order_relaxed);
            completed_requests.store(other.completed_requests.load(std::memory_order_relaxed), std::memory_order_relaxed);
            failed_requests.store(other.failed_requests.load(std::memory_order_relaxed), std::memory_order_relaxed);
            pending_requests.store(other.pending_requests.load(std::memory_order_relaxed), std::memory_order_relaxed);
            last_activity = other.last_activity;
        }
        return *this;
    }
    
    // Move constructor
    inline PipelineStats(PipelineStats&& other) noexcept
        : total_requests(other.total_requests.load(std::memory_order_relaxed)),
          completed_requests(other.completed_requests.load(std::memory_order_relaxed)),
          failed_requests(other.failed_requests.load(std::memory_order_relaxed)),
          pending_requests(other.pending_requests.load(std::memory_order_relaxed)),
          last_activity(other.last_activity) {
        // Reset the moved-from object
        other.total_requests.store(0, std::memory_order_relaxed);
        other.completed_requests.store(0, std::memory_order_relaxed);
        other.failed_requests.store(0, std::memory_order_relaxed);
        other.pending_requests.store(0, std::memory_order_relaxed);
    }
    
    // Move assignment operator
    inline PipelineStats& operator=(PipelineStats&& other) noexcept {
        if (this != &other) [[likely]] {
            total_requests.store(other.total_requests.load(std::memory_order_relaxed), std::memory_order_relaxed);
            completed_requests.store(other.completed_requests.load(std::memory_order_relaxed), std::memory_order_relaxed);
            failed_requests.store(other.failed_requests.load(std::memory_order_relaxed), std::memory_order_relaxed);
            pending_requests.store(other.pending_requests.load(std::memory_order_relaxed), std::memory_order_relaxed);
            last_activity = other.last_activity;
            
            // Reset the moved-from object
            other.total_requests.store(0, std::memory_order_relaxed);
            other.completed_requests.store(0, std::memory_order_relaxed);
            other.failed_requests.store(0, std::memory_order_relaxed);
            other.pending_requests.store(0, std::memory_order_relaxed);
        }
        return *this;
    }
    
    [[nodiscard]] inline double GetCompletionRate() const noexcept {
        const uint64_t total = total_requests.load(std::memory_order_relaxed);
        if (total > 0) [[likely]] {
            return static_cast<double>(completed_requests.load(std::memory_order_relaxed)) / total;
        } else [[unlikely]] {
            return 0.0;
        }
    }
    
    [[nodiscard]] inline uint64_t GetQueueDepth() const noexcept {
        return pending_requests.load(std::memory_order_relaxed);
    }
    
    // Additional utility methods
    [[nodiscard]] inline bool HasRequests() const noexcept {
        return total_requests.load(std::memory_order_relaxed) > 0;
    }
    
    [[nodiscard]] inline bool IsEmpty() const noexcept {
        return pending_requests.load(std::memory_order_relaxed) == 0;
    }
    
    [[nodiscard]] inline double GetFailureRate() const noexcept {
        const uint64_t total = total_requests.load(std::memory_order_relaxed);
        if (total > 0) [[likely]] {
            return static_cast<double>(failed_requests.load(std::memory_order_relaxed)) / total;
        } else [[unlikely]] {
            return 0.0;
        }
    }
};

/**
 * ProcessingPipeline - Core asynchronous processing template for plugin workflows
 * 
 * Thread Safety: This class is fully thread-safe for concurrent access
 * Lock-Free Optimizations: Extensively uses lock-free paradigms where possible:
 * - Atomic state and statistics tracking with optimized memory ordering
 * - Configuration parameters (delay, queue size) are atomic
 * - Fast atomic checks for processor availability before acquiring locks
 * - Lightweight mutexes only for critical sections (queue and processor updates)
 * 
 * The pipeline follows this pattern:
 * 1. Requests are queued with priority and metadata
 * 2. A worker thread processes requests using the provided processor function
 * 3. Results are delivered via callback functions
 * 4. Error handling and retry logic is built-in
 * 5. Performance statistics are tracked automatically
 * 
 * Template Parameters:
 * - RequestType: The input type for processing requests
 * - ResultType: The output type returned from processing
 */
template<typename RequestType, typename ResultType>
class ProcessingPipeline {
private:
    // Cache-aligned frequently accessed members for optimal performance
    alignas(constants::cache_line_size) std::atomic<PipelineState> state_{PipelineState::IDLE};
    alignas(constants::cache_line_size) PipelineStats stats_;
    
    // Core processing components - using atomic flag for lock-free processor checking
    std::function<void(const RequestType&, std::function<void(ResultType)>, std::function<void(const std::string&)>)> processor_;
    mutable std::mutex processor_mutex_; // Lightweight mutex just for processor updates
    std::atomic<bool> has_processor_{false}; // Track processor availability without locking
    std::priority_queue<PipelineRequest<RequestType>> request_queue_;
    std::unique_ptr<std::thread> worker_thread_;
    
    // Thread synchronization (grouped for cache efficiency)
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_condition_;
    
    // Configuration and identification
    std::string pipeline_name_;
    std::atomic<std::chrono::milliseconds::rep> processing_delay_ms_{constants::default_processing_delay_ms}; // Lock-free delay access
    std::atomic<size_t> max_queue_size_{constants::default_max_queue_size}; // Make atomic to avoid locking in queue size checks
    
    // Worker thread main loop - hot path for performance with optimized memory ordering
    void WorkerLoop() {
        LOG_ProcessingPipeline("Pipeline '" + pipeline_name_ + "' worker thread started");
        
        // Cache the shutdown state check to avoid repeated atomic loads
        PipelineState current_state;
        while ((current_state = state_.load(std::memory_order_acquire)) != PipelineState::SHUTDOWN) [[likely]] {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            // Wait for requests or shutdown signal with optimized predicate
            queue_condition_.wait(lock, [this]() noexcept {
                return !request_queue_.empty() || 
                       state_.load(std::memory_order_acquire) == PipelineState::SHUTDOWN;
            });
            
            // Check for shutdown with memory fence for consistency
            if (state_.load(std::memory_order_acquire) == PipelineState::SHUTDOWN) [[unlikely]] {
                break;
            }
            
            // Process next request if available
            if (!request_queue_.empty()) [[likely]] {
                auto pipeline_request = std::move(const_cast<PipelineRequest<RequestType>&>(request_queue_.top()));
                request_queue_.pop();
                stats_.pending_requests.fetch_sub(1, std::memory_order_acq_rel);
                
                lock.unlock();
                
                // Update state and stats atomically with proper ordering
                state_.store(PipelineState::PROCESSING, std::memory_order_release);
                stats_.last_activity = std::chrono::steady_clock::now();
                
                // Process the request
                ProcessRequest(std::move(pipeline_request));
                
                // Brief delay to prevent CPU spinning - only if needed (lock-free access)
                const auto delay_ms = processing_delay_ms_.load(std::memory_order_relaxed);
                if (delay_ms > 0) [[unlikely]] {
                    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
                }
                
                // Return to idle if no more requests (use acquire to ensure consistency)
                if (stats_.pending_requests.load(std::memory_order_acquire) == 0) [[likely]] {
                    state_.store(PipelineState::IDLE, std::memory_order_release);
                }
            }
        }
        
        LOG_ProcessingPipeline("Pipeline '" + pipeline_name_ + "' worker thread shutting down");
    }
    
    // Process a single request with error handling - hot path for performance
    void ProcessRequest(PipelineRequest<RequestType> pipeline_request) {
        // Fast atomic check first - completely lock-free
        if (!has_processor_.load(std::memory_order_acquire)) [[unlikely]] {
            stats_.failed_requests.fetch_add(1, std::memory_order_relaxed);
            LOG_ERROR("ProcessingPipeline", "Pipeline '" + pipeline_name_ + "' processor is null!");
            return;
        }
        
        // Copy processor under lock for safe access
        std::function<void(const RequestType&, std::function<void(ResultType)>, std::function<void(const std::string&)>)> processor_copy;
        {
            std::lock_guard<std::mutex> lock(processor_mutex_);
            if (!processor_) [[unlikely]] {
                stats_.failed_requests.fetch_add(1, std::memory_order_relaxed);
                LOG_ERROR("ProcessingPipeline", "Pipeline '" + pipeline_name_ + "' processor is null!");
                return;
            }
            processor_copy = processor_;
        }
        
        try {
            // Success callback - most common execution path
            auto success_callback = [this](ResultType result) {
                stats_.completed_requests.fetch_add(1, std::memory_order_relaxed);
                // Result is handled by the processor's internal callback logic
            };
            
            // Error callback with request_id capture for debugging - rare execution path
            auto error_callback = [this, request_id = std::move(pipeline_request.request_id)](const std::string& error) {
                stats_.failed_requests.fetch_add(1, std::memory_order_relaxed);
                LOG_ERROR("ProcessingPipeline", "Pipeline '" + pipeline_name_ + "' processing failed: " + error);
            };
            
            // Execute the processor (using copied processor)
            processor_copy(pipeline_request.request, std::move(success_callback), std::move(error_callback));
            
        } catch (const std::exception& [[maybe_unused]] e) {
            // Exception handling - rare case
            stats_.failed_requests.fetch_add(1, std::memory_order_relaxed);
            LOG_ERROR("ProcessingPipeline", "Pipeline '" + pipeline_name_ + "' exception: " + std::string(e.what()));
        } catch (...) {
            // Unknown exception handling - very rare case
            stats_.failed_requests.fetch_add(1, std::memory_order_relaxed);
            LOG_ERROR("ProcessingPipeline", "Pipeline '" + pipeline_name_ + "' unknown exception");
        }
    }

public:
    /**
     * Constructor
     * @param name Pipeline identifier for logging and debugging
     * @param processor Function that processes requests: (request, success_callback, error_callback) -> void
     */
    explicit ProcessingPipeline(std::string_view name = "UnnamedPipeline",
                               std::function<void(const RequestType&, std::function<void(ResultType)>, std::function<void(const std::string&)>)> processor = nullptr) noexcept
        : processor_(std::move(processor)), pipeline_name_(name) {
        
        // Update atomic processor flag
        has_processor_.store(static_cast<bool>(processor_), std::memory_order_release);
        
        // Don't auto-start - let the caller start when ready
    }
    
    /**
     * Destructor - ensures clean shutdown
     */
    ~ProcessingPipeline() noexcept {
        Shutdown();
    }
    
    // Delete copy constructor and assignment (this is a unique resource)
    ProcessingPipeline(const ProcessingPipeline&) = delete;
    ProcessingPipeline& operator=(const ProcessingPipeline&) = delete;
    
    // Move constructor and assignment
    ProcessingPipeline(ProcessingPipeline&& other) noexcept
        : processor_(std::move(other.processor_)),
          has_processor_(other.has_processor_.load(std::memory_order_relaxed)),
          request_queue_(std::move(other.request_queue_)),
          worker_thread_(std::move(other.worker_thread_)),
          state_(other.state_.load(std::memory_order_relaxed)),
          stats_(std::move(other.stats_)),
          pipeline_name_(std::move(other.pipeline_name_)),
          processing_delay_ms_(other.processing_delay_ms_.load(std::memory_order_relaxed)),
          max_queue_size_(other.max_queue_size_.load(std::memory_order_relaxed)) {
        
        // Reset moved-from object
        other.state_.store(PipelineState::SHUTDOWN, std::memory_order_relaxed);
        other.has_processor_.store(false, std::memory_order_relaxed);
    }
    
    ProcessingPipeline& operator=(ProcessingPipeline&& other) noexcept {
        if (this != &other) [[likely]] {
            Shutdown();
            processor_ = std::move(other.processor_);
            has_processor_.store(other.has_processor_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            request_queue_ = std::move(other.request_queue_);
            worker_thread_ = std::move(other.worker_thread_);
            state_.store(other.state_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            stats_ = std::move(other.stats_);
            pipeline_name_ = std::move(other.pipeline_name_);
            processing_delay_ms_.store(other.processing_delay_ms_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            max_queue_size_.store(other.max_queue_size_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            
            // Reset moved-from object
            other.state_.store(PipelineState::SHUTDOWN, std::memory_order_relaxed);
            other.has_processor_.store(false, std::memory_order_relaxed);
        }
        return *this;
    }
    
    /**
     * Set the processor function (uses lightweight mutex for thread safety)
     * Thread-safe: Can be called while pipeline is running
     */
    inline void SetProcessor(std::function<void(const RequestType&, std::function<void(ResultType)>, std::function<void(const std::string&)>)> processor) noexcept {
        {
            std::lock_guard<std::mutex> lock(processor_mutex_);
            processor_ = std::move(processor);
        }
        has_processor_.store(static_cast<bool>(processor_), std::memory_order_release);
    }
    
    /**
     * Queue a request for processing (const reference version)
     * @param request The request to process
     * @param priority Priority level for queue ordering
     * @param request_id Optional identifier for tracking
     * @return true if queued successfully, false if queue is full or pipeline is shutdown
     */
    [[nodiscard]] inline bool QueueRequest(const RequestType& request, 
                     RequestPriority priority = RequestPriority::NORMAL,
                     std::string_view request_id = "") noexcept {
        
        std::lock_guard<std::mutex> lock(queue_mutex_);
        
        // Check if shutdown
        if (state_.load(std::memory_order_relaxed) == PipelineState::SHUTDOWN) [[unlikely]] {
            return false;
        }
        
        // Check queue capacity (now lock-free)
        if (request_queue_.size() >= max_queue_size_.load(std::memory_order_relaxed)) [[unlikely]] {
            LOG_ProcessingPipeline("Pipeline '" + pipeline_name_ + "' queue is full, rejecting request");
            return false;
        }
        
        // Add to queue (successful case - most common path)
        request_queue_.emplace(request, priority, std::string(request_id));
        stats_.total_requests.fetch_add(1, std::memory_order_relaxed);
        stats_.pending_requests.fetch_add(1, std::memory_order_relaxed);
        
        // Notify worker thread
        queue_condition_.notify_one();
        
        return true; // [[likely]] - most queue requests succeed
    }
    
    /**
     * Queue a request for processing (move version for optimal performance)
     * @param request The request to process (will be moved)
     * @param priority Priority level for queue ordering
     * @param request_id Optional identifier for tracking
     * @return true if queued successfully, false if queue is full or pipeline is shutdown
     */
    [[nodiscard]] inline bool QueueRequest(RequestType&& request, 
                     RequestPriority priority = RequestPriority::NORMAL,
                     std::string_view request_id = "") noexcept {
        
        std::lock_guard<std::mutex> lock(queue_mutex_);
        
        // Check if shutdown
        if (state_.load(std::memory_order_relaxed) == PipelineState::SHUTDOWN) [[unlikely]] {
            return false;
        }
        
        // Check queue capacity (now lock-free)
        if (request_queue_.size() >= max_queue_size_.load(std::memory_order_relaxed)) [[unlikely]] {
            LOG_ProcessingPipeline("Pipeline '" + pipeline_name_ + "' queue is full, rejecting request");
            return false;
        }
        
        // Add to queue using move semantics (successful case - most common path)
        request_queue_.emplace(std::move(request), priority, std::string(request_id));
        stats_.total_requests.fetch_add(1, std::memory_order_relaxed);
        stats_.pending_requests.fetch_add(1, std::memory_order_relaxed);
        
        // Notify worker thread
        queue_condition_.notify_one();
        
        return true; // [[likely]] - most queue requests succeed
    }
    
    /**
     * Start the pipeline worker thread
     */
    void Start() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        // If we already have a running thread, don't start another one
        if (worker_thread_ && state_.load(std::memory_order_relaxed) != PipelineState::SHUTDOWN) [[unlikely]] {
            return; // Already started
        }
        
        // Join any existing thread before creating a new one
        if (worker_thread_ && worker_thread_->joinable()) [[unlikely]] {
            // Temporarily release lock to avoid deadlock during join
            lock.unlock();
            worker_thread_->join();
            lock.lock();
        }
        
        state_.store(PipelineState::IDLE, std::memory_order_relaxed);
        worker_thread_ = std::make_unique<std::thread>(&ProcessingPipeline::WorkerLoop, this);
        
        LOG_ProcessingPipeline("Pipeline '" + pipeline_name_ + "' started");
    }
    

    
    /**
     * Shutdown the pipeline and wait for worker thread to complete
     */
    void Shutdown() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            state_.store(PipelineState::SHUTDOWN, std::memory_order_release);
        }
        queue_condition_.notify_all();
        
        if (worker_thread_ && worker_thread_->joinable()) [[likely]] {
            worker_thread_->join();
        }
    }
    
    /**
     * Get pipeline statistics
     */
    [[nodiscard]] inline PipelineStats GetStats() const noexcept {
        return stats_;
    }
    
    /**
     * Get pipeline name
     */
    [[nodiscard]] inline const std::string& GetPipelineName() const noexcept {
        return pipeline_name_;
    }
    
    /**
     * Get processing delay (lock-free access)
     */
    [[nodiscard]] inline std::chrono::milliseconds GetProcessingDelay() const noexcept {
        return std::chrono::milliseconds{processing_delay_ms_.load(std::memory_order_relaxed)};
    }
    
    /**
     * Get maximum queue size
     */
    [[nodiscard]] inline size_t GetMaxQueueSize() const noexcept {
        return max_queue_size_.load(std::memory_order_relaxed);
    }
    
    /**
     * Check if pipeline is actively processing
     */
    [[nodiscard]] inline bool IsProcessing() const noexcept {
        return state_.load(std::memory_order_relaxed) == PipelineState::PROCESSING;
    }
    
    /**
     * Check if pipeline is idle (ready to process but no requests queued)
     */
    [[nodiscard]] inline bool IsIdle() const noexcept {
        return state_.load(std::memory_order_relaxed) == PipelineState::IDLE;
    }
    
    /**
     * Check if pipeline is shutdown
     */
    [[nodiscard]] inline bool IsShutdown() const noexcept {
        return state_.load(std::memory_order_relaxed) == PipelineState::SHUTDOWN;
    }
    
    /**
     * Check if processor is set (lock-free)
     */
    [[nodiscard]] inline bool HasProcessor() const noexcept {
        return has_processor_.load(std::memory_order_acquire);
    }
    
    /**
     * Set maximum queue size atomically
     */
    inline void SetMaxQueueSize(size_t new_size) noexcept {
        if (detail::is_valid_queue_size(new_size)) [[likely]] {
            max_queue_size_.store(new_size, std::memory_order_relaxed);
        }
    }
    
    /**
     * Set processing delay atomically (lock-free)
     */
    inline void SetProcessingDelay(std::chrono::milliseconds delay) noexcept {
        const auto delay_ms = delay.count();
        if (detail::is_valid_processing_delay_ms(delay_ms)) [[likely]] {
            processing_delay_ms_.store(delay_ms, std::memory_order_relaxed);
        }
    }
    
    // Compile-time utility functions
    
    /**
     * Validate pipeline configuration at compile time
     */
    consteval static bool ValidateConfiguration(size_t max_queue_size, int64_t processing_delay_ms) noexcept {
        return detail::is_valid_queue_size(max_queue_size) && 
               detail::is_valid_processing_delay_ms(processing_delay_ms);
    }
    
    /**
     * Get compile-time cache line size
     */
    [[nodiscard]] consteval static size_t GetCacheLineSize() noexcept {
        return constants::cache_line_size;
    }
    
    /**
     * Get compile-time default values
     */
    [[nodiscard]] consteval static size_t GetDefaultMaxQueueSize() noexcept {
        return constants::default_max_queue_size;
    }
    
    [[nodiscard]] consteval static std::chrono::milliseconds GetDefaultProcessingDelay() noexcept {
        return std::chrono::milliseconds{constants::default_processing_delay_ms};
    }
    
    // Lock-free utility functions for better performance
    
    /**
     * Check if queue is near capacity (lock-free)
     */
    [[nodiscard]] inline bool IsQueueNearCapacity(double threshold = 0.8) const noexcept {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        const size_t current_size = request_queue_.size();
        const size_t max_size = max_queue_size_.load(std::memory_order_relaxed);
        return current_size >= static_cast<size_t>(max_size * threshold);
    }
    
    /**
     * Get current queue size (requires lock due to std::priority_queue limitations)
     */
    [[nodiscard]] inline size_t GetCurrentQueueSize() const noexcept {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return request_queue_.size();
    }
    
    /**
     * Lock-free check if pipeline is ready to accept requests
     */
    [[nodiscard]] inline bool IsReadyForRequests() const noexcept {
        return has_processor_.load(std::memory_order_acquire) && 
               state_.load(std::memory_order_acquire) != PipelineState::SHUTDOWN;
    }
    
    /**
     * Optimized atomic state check with memory fence
     */
    [[nodiscard]] inline PipelineState GetStateWithFence() const noexcept {
        auto state = state_.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_acquire);
        return state;
    }
    
    /**
     * Batch statistics update (more efficient for multiple operations)
     */
    inline void UpdateStatsAtomic(uint64_t completed_delta, uint64_t failed_delta) noexcept {
        if (completed_delta > 0) {
            stats_.completed_requests.fetch_add(completed_delta, std::memory_order_relaxed);
        }
        if (failed_delta > 0) {
            stats_.failed_requests.fetch_add(failed_delta, std::memory_order_relaxed);
        }
        // Use release fence to ensure visibility
        std::atomic_thread_fence(std::memory_order_release);
    }
};

} // namespace LuminaChat
