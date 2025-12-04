//
// `LongMen` - 'ONNX Model inference in c++'
// Copyright (C) 2019 - present timepi <timepi123@gmail.com>
// LongMen is provided under: GNU Affero General Public License (AGPL3.0)
// https://www.gnu.org/licenses/agpl-3.0.html unless stated otherwise.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//

#ifndef LONGMEN_ARENA_H_
#define LONGMEN_ARENA_H_

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <vector>

#include "common.hpp"
#include "features.hpp"

namespace longmen {

using json = nlohmann::json;

/** @brief Maximum number of cached GraphIO objects per buffer pool */
constexpr size_t BUFFER_SIZE = 32;

/** @brief Number of batch size categories (supports up to 512 batch size) */
constexpr int32_t LABEL_SIZE = 16;

/** @brief Batch size alignment boundary in elements */
constexpr int32_t BATCH_ALIGN = 32;

/** @brief CPU cache line size in bytes for memory alignment */
constexpr size_t CACHE_LINE_SIZE = 64;

/**
 * @brief Allocates cache-aligned memory
 *
 * Allocates memory aligned to CACHE_LINE_SIZE to optimize cache performance
 * and enable efficient SIMD operations.
 *
 * @param size Size in bytes to allocate
 * @return Pointer to aligned memory, or nullptr on allocation failure
 *
 * @note Must be freed using aligned_free_wrapper()
 * @see aligned_free_wrapper()
 */
inline void *aligned_alloc_wrapper(size_t size) {
#ifdef _WIN32
  return _aligned_malloc(size, CACHE_LINE_SIZE);
#else
  void *ptr = nullptr;
  return posix_memalign(&ptr, CACHE_LINE_SIZE, size) == 0 ? ptr : nullptr;
#endif
}

/**
 * @brief Frees memory allocated by aligned_alloc_wrapper()
 *
 * @param ptr Pointer to aligned memory (nullptr is safe)
 * @see aligned_alloc_wrapper()
 */
inline void aligned_free_wrapper(void *ptr) {
  if (!ptr)
    return;
#ifdef _WIN32
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

/**
 * @brief Converts batch size to buffer label index
 *
 * Maps batch size to a buffer category by rounding up to the nearest
 * multiple of BATCH_ALIGN (32).
 *
 * @param batch Batch size
 * @return Label index (0-based), where label N has capacity (N+1)*32
 * @retval -1 if batch <= 0
 *
 * @par Examples:
 * @code
 * batch_to_label(1)   -> 0  (capacity 32)
 * batch_to_label(32)  -> 0  (capacity 32)
 * batch_to_label(33)  -> 1  (capacity 64)
 * batch_to_label(100) -> 3  (capacity 128)
 * @endcode
 */
inline int32_t batch_to_label(int32_t batch) {
  return (batch + BATCH_ALIGN - 1) / BATCH_ALIGN - 1;
}

/**
 * @brief Converts label index to actual capacity
 *
 * @param label Buffer label index (0-based)
 * @return Capacity in number of batch elements
 *
 * @par Examples:
 * @code
 * label_to_capacity(0)  -> 32
 * label_to_capacity(1)  -> 64
 * label_to_capacity(15) -> 512
 * @endcode
 */
inline int32_t label_to_capacity(int32_t label) {
  return (label + 1) * BATCH_ALIGN;
}

/**
 * @brief Input tensor with cache-aligned memory layout
 *
 * Manages input tensor data for model inference with optimized memory
 * allocation. Supports both scalar and array features with efficient
 * broadcasting operations.
 *
 * @note Non-copyable but movable
 * @note Memory is automatically freed in destructor
 */
class Input {
public:
  /**
   * @brief Constructs an input tensor
   *
   * @param index Input node index in the model graph
   * @param capacity Maximum batch size
   * @param width Feature dimension (elements per batch item)
   * @param type ONNX tensor element type (FLOAT or INT64)
   *
   * @note Allocates cache-aligned, zero-initialized memory
   */
  Input(int32_t index, int32_t capacity, int64_t width, minia::DataType type);

  /**
   * @brief Destructor - releases aligned memory
   */
  ~Input();

  /** @brief Copy constructor (deleted) */
  Input(const Input &) = delete;

  /** @brief Copy assignment (deleted) */
  Input &operator=(const Input &) = delete;

  /**
   * @brief Move constructor
   *
   * @param other Source object (left in valid but empty state)
   */
  Input(Input &&other) noexcept;

  /**
   * @brief Broadcasts a single value across all batch items
   *
   * Efficiently replicates one feature value/array to all batch positions
   * using optimized memory operations.
   *
   * @param batch Number of batch items to fill
   * @param feature Feature value to broadcast
   *
   * @note No-op if feature is null or batch <= 0
   * @note For arrays, copies min(array.size(), width_) elements
   */
  void set_value_with_broadcast(int32_t batch, minia::FeaturePtr feature);

  /**
   * @brief Sets value at a specific batch position
   *
   * @param index Batch index (must be < batch_)
   * @param feature Feature value to set
   *
   * @note No-op if index is out of range or feature is null
   */
  void set_value(int32_t index, minia::FeaturePtr feature);

  /**
   * @brief Zeros all tensor data
   *
   * Resets entire tensor memory to zero for reuse.
   */
  void zero();

  /**
   * @brief Gets raw data pointer
   *
   * @return Pointer to tensor data buffer
   * @note Cast to float* or int64_t* based on type_
   */
  void *get_data() { return data_; }

  /**
   * @brief Sets current batch size
   *
   * @param batch Number of valid batch items
   */
  void set_batch(int32_t batch) { batch_ = batch; }

  /**
   * @brief Gets current batch size
   *
   * @return Number of valid batch items
   */
  int32_t get_batch() const { return batch_; }

private:
  /**
   * @brief Broadcasts int64 values internally
   *
   * @param batch Number of batch items
   * @param feature Source feature
   * @param type Data type
   */
  void set_int64_broadcast(int32_t batch, minia::FeaturePtr feature,
                           minia::DataType type);

  /**
   * @brief Broadcasts float values internally
   *
   * @param batch Number of batch items
   * @param feature Source feature
   * @param type Data type
   */
  void set_float_broadcast(int32_t batch, minia::FeaturePtr feature,
                           minia::DataType type);

  int32_t index_;        ///< Input node index
  int32_t batch_;        ///< Current batch size
  int32_t capacity_;     ///< Maximum batch capacity
  int64_t width_;        ///< Feature dimension width
  minia::DataType type_; ///< Element data type
  void *data_;           ///< Tensor data buffer
};

/**
 * @brief Output tensor for model inference results
 *
 * Manages output tensor data with cache-aligned memory.
 * Uses float data type for model outputs.
 *
 * @note Non-copyable but movable
 */
class Output {
public:
  /**
   * @brief Constructs an output tensor
   *
   * @param index Output node index in model graph
   * @param width Output dimension width
   */
  Output(int32_t index, int64_t width);

  /**
   * @brief Destructor
   */
  ~Output() = default;

  /** @brief Copy constructor (deleted) */
  Output(const Output &) = delete;

  /** @brief Copy assignment (deleted) */
  Output &operator=(const Output &) = delete;

  /**
   * @brief Move constructor
   *
   * @param other Source object (left in valid but empty state)
   */
  Output(Output &&other) noexcept;

  /**
   * @brief Zeros all tensor data
   */
  void zero();

  /**
   * @brief Sets data pointer
   *
   * @param data Pointer to output data buffer
   */
  void set_data(float *data) { data_ = data; }

  /**
   * @brief Gets raw data pointer
   *
   * @return Pointer to tensor data buffer (float*)
   */
  void *get_data() { return data_; }

  /**
   * @brief Sets current batch size
   *
   * @param batch Number of valid batch items
   */
  void set_batch(int32_t batch) { batch_ = batch; }

private:
  int32_t index_; ///< Output node index
  int32_t batch_; ///< Current batch size
  int64_t width_; ///< Output dimension width
  void *data_;    ///< Tensor data buffer (float*)
};

/**
 * @brief Container for all model input/output tensors
 *
 * Bundles all input and output tensors required for a single batch
 * inference operation with unified capacity management.
 */
class GraphIO {
public:
  /**
   * @brief Constructs GraphIO from configuration
   *
   * @param config JSON configuration for tensor specifications
   * @param capacity Maximum batch size for all tensors
   */
  GraphIO(const json &config, int32_t capacity);

  /**
   * @brief Zeros all input and output tensors
   *
   * Prepares GraphIO for reuse from memory pool.
   */
  void zero();

  /**
   * @brief Sets batch size for all tensors
   *
   * @param batch Number of valid batch items
   */
  void set_batch(int32_t batch);

  /**
   * @brief Sets output data pointers
   *
   * @param data Array of output data pointers
   */
  void set_outputs(float **data);

  /**
   * @brief Resets GraphIO to initial state
   */
  void reset();

  /**
   * @brief Gets current batch size
   *
   * @return Number of valid batch items
   */
  int32_t get_batch() const { return batch_; }

  /**
   * @brief Gets maximum capacity
   *
   * @return Maximum batch size
   */
  int32_t capacity() const { return capacity_; }

  /**
   * @brief Gets input tensor by index
   *
   * @param idx Input tensor index (must be < input_count())
   * @return Reference to input tensor
   * @warning No bounds checking performed
   */
  Input &get_input(size_t idx) { return inputs_[idx]; }

  /**
   * @brief Gets output tensor by index
   *
   * @param idx Output tensor index (must be < output_count())
   * @return Reference to output tensor
   * @warning No bounds checking performed
   */
  Output &get_output(size_t idx) { return outputs_[idx]; }

  /**
   * @brief Gets number of input tensors
   *
   * @return Total input tensor count
   */
  size_t input_count() const { return inputs_.size(); }

  /**
   * @brief Gets number of output tensors
   *
   * @return Total output tensor count
   */
  size_t output_count() const { return outputs_.size(); }

private:
  /**
   * @brief Parses configuration and initializes tensors
   *
   * @param config JSON configuration
   * @param capacity Maximum batch size
   */
  void parse_config(const json &config, int32_t capacity);

private:
  int32_t batch_;               ///< Current batch size
  int32_t capacity_;            ///< Maximum batch size
  std::vector<Input> inputs_;   ///< Input tensors
  std::vector<Output> outputs_; ///< Output tensors
};

/**
 * @brief Object pool for GraphIO instances of specific capacity
 *
 * Maintains a queue of pre-allocated GraphIO objects to minimize
 * allocation overhead. Each buffer handles a specific capacity range.
 *
 * @note Thread-safe for concurrent operations
 */
class Buffer {
public:
  /**
   * @brief Constructs buffer for a batch size category
   *
   * @param label Batch size label (0-based), capacity = (label+1)*32
   * @param config Shared configuration for creating GraphIO objects
   *
   * @note Buffer starts empty and creates objects on demand
   */
  Buffer(int32_t label, std::shared_ptr<json> config)
      : label_(label), config_(config), max_capacity_(BUFFER_SIZE), size_(0) {}

  /**
   * @brief Gets GraphIO from pool or creates new one
   *
   * Returns cached GraphIO if available, otherwise creates new instance.
   *
   * @return GraphIO object with appropriate capacity
   * @note Thread-safe
   * @note Returned object is NOT zeroed
   */
  std::unique_ptr<GraphIO> get();

  /**
   * @brief Returns GraphIO to pool
   *
   * Zeros object and caches it if pool has space, otherwise destroys it.
   *
   * @param block GraphIO object to return (nullptr is safe)
   * @note Thread-safe
   * @note Object is zeroed before caching
   */
  void put(std::unique_ptr<GraphIO> block);

  /**
   * @brief Gets current pool size
   *
   * @return Number of cached GraphIO objects
   * @note Thread-safe
   */
  size_t size() const;

private:
  int32_t label_;                              ///< Batch size category
  std::shared_ptr<json> config_;               ///< Configuration
  std::queue<std::unique_ptr<GraphIO>> queue_; ///< Cached objects
  mutable std::mutex mutex_;                   ///< Thread synchronization
  size_t max_capacity_;                        ///< Maximum pool size
  size_t size_;                                ///< Current pool size
};

/**
 * @brief Memory arena managing multiple GraphIO pools
 *
 * Provides unified interface for allocating GraphIO objects of various
 * batch sizes. Maintains separate buffer pools for different capacity
 * ranges (up to 512), and directly allocates for larger batches.
 *
 * @par Capacity Ranges:
 * - Buffer 0:  1-32   elements
 * - Buffer 1:  33-64  elements
 * - Buffer 2:  65-96  elements
 * - ...
 * - Buffer 15: 481-512 elements
 *
 * @note Thread-safe for concurrent operations
 */
class Arena {
public:
  /**
   * @brief Constructs arena with model configuration
   *
   * Initializes LABEL_SIZE (16) buffer pools for batch sizes up to 512.
   *
   * @param config Shared configuration for model graph
   */
  explicit Arena(std::shared_ptr<json> config);

  /**
   * @brief Gets GraphIO for specified batch size
   *
   * Returns pooled GraphIO if batch <= 512, otherwise creates new instance.
   * Returned capacity may be larger than requested (rounded up to multiple of
   * 32).
   *
   * @param batch Required batch size (must be > 0)
   * @return GraphIO with capacity >= batch
   *
   * @note Thread-safe
   * @note For batch <= 0, returns capacity 32
   * @note For batch > 512, creates non-pooled object
   *
   * @par Examples:
   * @code
   * get(10)  -> capacity 32  (from Buffer 0)
   * get(50)  -> capacity 64  (from Buffer 1)
   * get(600) -> capacity 600 (not pooled)
   * @endcode
   */
  std::unique_ptr<GraphIO> get(int32_t batch);

  /**
   * @brief Returns GraphIO to arena
   *
   * Routes object to appropriate buffer pool based on capacity.
   * Objects with capacity > 512 are directly destroyed.
   *
   * @param block GraphIO object to return (nullptr is safe)
   * @note Thread-safe
   * @note Object is automatically zeroed before caching
   */
  void put(std::unique_ptr<GraphIO> block);

private:
  std::shared_ptr<json> config_;                 ///< Model configuration
  std::vector<std::unique_ptr<Buffer>> buffers_; ///< Buffer pools
};

} // namespace longmen

#endif // LONGMEN_ARENA_H_
