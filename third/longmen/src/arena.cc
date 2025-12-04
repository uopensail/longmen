#include "arena.h"

#include <iostream>
#include <stdexcept>

namespace longmen {

// ============================================================================
// Input Implementation
// ============================================================================

Input::Input(int32_t index, int32_t capacity, int64_t width,
             minia::DataType type)
    : index_(index), batch_(0), capacity_(capacity), width_(width), type_(type),
      data_(nullptr) {
  // Validate input parameters
  if (capacity <= 0) {
    LOG(ERROR) << "Invalid capacity: " << capacity << " for input " << index;
    throw std::invalid_argument("Capacity must be positive");
  }

  if (width <= 0) {
    LOG(ERROR) << "Invalid width: " << width << " for input " << index;
    throw std::invalid_argument("Width must be positive");
  }

  // Calculate total memory size based on element type
  size_t element_size =
      (type_ == minia::DataType::kFloatValue) ? sizeof(float) : sizeof(int64_t);
  size_t total_size = capacity_ * width_ * element_size;

  // Check for potential overflow
  if (total_size / element_size / width_ != static_cast<size_t>(capacity_)) {
    LOG(ERROR) << "Memory size overflow for input " << index
               << ": capacity=" << capacity_ << ", width=" << width_;
    throw std::overflow_error("Memory allocation size overflow");
  }

  // Allocate cache-aligned memory for better performance
  data_ = aligned_alloc_wrapper(total_size);
  if (!data_) {
    LOG(ERROR) << "Failed to allocate " << total_size << " bytes for input "
               << index;
    throw std::bad_alloc();
  }

  // Zero-initialize memory
  memset(data_, 0, total_size);
}

Input::~Input() {
  if (data_) {
    aligned_free_wrapper(data_);
    data_ = nullptr;
  }
}

Input::Input(Input &&other) noexcept
    : index_(other.index_), batch_(other.batch_), capacity_(other.capacity_),
      width_(other.width_), type_(other.type_), data_(other.data_) {
  // Transfer ownership, leave source in valid empty state
  other.data_ = nullptr;
  other.batch_ = 0;
  other.capacity_ = 0;
}

void Input::set_value_with_broadcast(int32_t batch, minia::FeaturePtr feature) {
  // Validate batch size
  if (batch <= 0) {
    LOG(WARNING) << "Invalid batch size " << batch << " for Input[" << index_
                 << "]";
    return;
  }

  if (batch > capacity_) {
    LOG(ERROR) << "Batch size " << batch << " exceeds capacity " << capacity_
               << " for Input[" << index_ << "]";
    throw std::out_of_range("Batch size exceeds capacity");
  }

  batch_ = batch;

  // Early return for invalid inputs
  if (!feature) {
    LOG(WARNING) << "Null feature pointer for Input[" << index_ << "]";
    return;
  }

  if (!data_) {
    LOG(ERROR) << "Null data pointer for Input[" << index_ << "]";
    throw std::runtime_error("Input data not allocated");
  }

  minia::DataType feature_type = feature->type();

  // Validate type compatibility
  if (type_ == minia::DataType::kInt64Value &&
      feature_type != minia::DataType::kInt64Value &&
      feature_type != minia::DataType::kInt64Array) {
    LOG(ERROR) << "Type mismatch for Input[" << index_
               << "]: expected int64, got " << static_cast<int>(feature_type);
    throw std::invalid_argument("Feature type mismatch");
  }

  if (type_ == minia::DataType::kFloatValue &&
      feature_type != minia::DataType::kFloatValue &&
      feature_type != minia::DataType::kFloatArray) {
    LOG(ERROR) << "Type mismatch for Input[" << index_
               << "]: expected float, got " << static_cast<int>(feature_type);
    throw std::invalid_argument("Feature type mismatch");
  }

  // Dispatch to type-specific broadcast implementation
  if (type_ == minia::DataType::kInt64Value) {
    set_int64_broadcast(batch, feature, feature_type);
  } else if (type_ == minia::DataType::kFloatValue) {
    set_float_broadcast(batch, feature, feature_type);
  }
}

void Input::set_value(int32_t index, minia::FeaturePtr feature) {
  // Validate index bounds
  if (index < 0 || index >= batch_) {
    LOG(WARNING) << "Index " << index << " out of range [0, " << batch_
                 << ") for Input[" << index_ << "]";
    return;
  }

  if (!feature) {
    LOG(WARNING) << "Null feature pointer for Input[" << index_ << "] at index "
                 << index;
    return;
  }

  if (!data_) {
    LOG(ERROR) << "Null data pointer for Input[" << index_ << "]";
    throw std::runtime_error("Input data not allocated");
  }

  minia::DataType feature_type = feature->type();

  if (type_ == minia::DataType::kInt64Value) {
    // Calculate pointer to target row
    int64_t *ptr = static_cast<int64_t *>(data_) + width_ * index;

    if (feature_type == minia::DataType::kInt64Value) {
      // Scalar value: set first element only
      ptr[0] = feature->as_int64();
    } else if (feature_type == minia::DataType::kInt64Array) {
      // Array value: copy with size clamping
      auto array = feature->as_int64_array();
      size_t copy_size = std::min<size_t>(array.size(), width_);

      if (array.size() > static_cast<size_t>(width_)) {
        LOG(WARNING) << "Array size " << array.size() << " exceeds width "
                     << width_ << " for Input[" << index_ << "], truncating";
      }

      memcpy(ptr, array.data(), copy_size * sizeof(int64_t));
    } else {
      LOG(ERROR) << "Invalid feature type " << static_cast<int>(feature_type)
                 << " for int64 Input[" << index_ << "]";
    }
  } else if (type_ == minia::DataType::kFloatValue) {
    // Calculate pointer to target row
    float *ptr = static_cast<float *>(data_) + width_ * index;

    if (feature_type == minia::DataType::kFloatValue) {
      // Scalar value: set first element only
      ptr[0] = feature->as_float();
    } else if (feature_type == minia::DataType::kFloatArray) {
      // Array value: copy with size clamping
      auto array = feature->as_float_array();
      size_t copy_size = std::min<size_t>(array.size(), width_);

      if (array.size() > static_cast<size_t>(width_)) {
        LOG(WARNING) << "Array size " << array.size() << " exceeds width "
                     << width_ << " for Input[" << index_ << "], truncating";
      }

      memcpy(ptr, array.data(), copy_size * sizeof(float));
    } else {
      LOG(ERROR) << "Invalid feature type " << static_cast<int>(feature_type)
                 << " for float Input[" << index_ << "]";
    }
  }
}

void Input::zero() {
  if (!data_) {
    LOG(WARNING) << "Attempting to zero null data for Input[" << index_ << "]";
    return;
  }

  // Calculate total size and zero out entire buffer
  size_t element_size =
      (type_ == minia::DataType::kFloatValue) ? sizeof(float) : sizeof(int64_t);
  size_t total_size = capacity_ * width_ * element_size;
  memset(data_, 0, total_size);
}

void Input::set_int64_broadcast(int32_t batch, minia::FeaturePtr feature,
                                minia::DataType type) {
  int64_t *base_ptr = static_cast<int64_t *>(data_);

  if (type == minia::DataType::kInt64Value) {
    // Broadcast scalar: set first element of each row
    int64_t val = feature->as_int64();
    for (int32_t i = 0; i < batch; ++i) {
      base_ptr[i * width_] = val;
    }
  } else if (type == minia::DataType::kInt64Array) {
    // Broadcast array: copy to first row, then replicate
    auto array = feature->as_int64_array();
    size_t copy_size = std::min<size_t>(array.size(), width_);

    if (array.empty()) {
      LOG(WARNING) << "Empty int64 array for Input[" << index_ << "]";
      return;
    }

    if (array.size() > static_cast<size_t>(width_)) {
      LOG(WARNING) << "Array size " << array.size() << " exceeds width "
                   << width_ << " for Input[" << index_ << "], truncating";
    }

    // Copy source array to first row
    memcpy(base_ptr, array.data(), copy_size * sizeof(int64_t));

    // Replicate first row to remaining rows (better cache locality)
    size_t row_bytes = copy_size * sizeof(int64_t);
    for (int32_t i = 1; i < batch; ++i) {
      memcpy(base_ptr + i * width_, base_ptr, row_bytes);
    }
  }
}

void Input::set_float_broadcast(int32_t batch, minia::FeaturePtr feature,
                                minia::DataType type) {
  float *base_ptr = static_cast<float *>(data_);

  if (type == minia::DataType::kFloatValue) {
    // Broadcast scalar: set first element of each row
    float val = feature->as_float();
    for (int32_t i = 0; i < batch; ++i) {
      base_ptr[i * width_] = val;
    }
  } else if (type == minia::DataType::kFloatArray) {
    // Broadcast array: copy to first row, then replicate
    auto array = feature->as_float_array();
    size_t copy_size = std::min<size_t>(array.size(), width_);

    if (array.empty()) {
      LOG(WARNING) << "Empty float array for Input[" << index_ << "]";
      return;
    }

    if (array.size() > static_cast<size_t>(width_)) {
      LOG(WARNING) << "Array size " << array.size() << " exceeds width "
                   << width_ << " for Input[" << index_ << "], truncating";
    }

    // Copy source array to first row
    memcpy(base_ptr, array.data(), copy_size * sizeof(float));

    // Replicate first row to remaining rows (better cache locality)
    size_t row_bytes = copy_size * sizeof(float);
    for (int32_t i = 1; i < batch; ++i) {
      memcpy(base_ptr + i * width_, base_ptr, row_bytes);
    }
  }
}

// ============================================================================
// Output Implementation
// ============================================================================

Output::Output(int32_t index, int64_t width)
    : index_(index), batch_(0), width_(width), data_(nullptr) {
  // Validate parameters
  if (width <= 0) {
    LOG(ERROR) << "Invalid width: " << width << " for output " << index;
    throw std::invalid_argument("Width must be positive");
  }
}

Output::Output(Output &&other) noexcept
    : index_(other.index_), batch_(other.batch_), width_(other.width_),
      data_(other.data_) {
  // Transfer ownership, leave source in valid empty state
  other.data_ = nullptr;
  other.batch_ = 0;
}

void Output::zero() {
  if (!data_) {
    return;
  }

  if (batch_ <= 0) {
    return;
  }

  size_t total_size = batch_ * width_ * sizeof(float);
  memset(data_, 0, total_size);
}

// ============================================================================
// GraphIO Implementation
// ============================================================================

GraphIO::GraphIO(const json &config, int32_t capacity)
    : batch_(0), capacity_(capacity) {
  // Validate capacity
  if (capacity <= 0) {
    LOG(ERROR) << "Invalid capacity: " << capacity;
    throw std::invalid_argument("Capacity must be positive");
  }

  try {
    parse_config(config, capacity);
    LOG(INFO) << "Created GraphIO: capacity=" << capacity_
              << ", inputs=" << inputs_.size()
              << ", outputs=" << outputs_.size();
  } catch (const std::exception &e) {
    LOG(ERROR) << "Failed to create GraphIO: " << e.what();
    throw;
  }
}

void GraphIO::set_batch(int32_t batch) {
  // Validate batch size
  if (batch < 0) {
    LOG(WARNING) << "Negative batch size " << batch << ", setting to 0";
    batch = 0;
  }

  if (batch > capacity_) {
    LOG(ERROR) << "Batch size " << batch << " exceeds capacity " << capacity_;
    throw std::out_of_range("Batch size exceeds capacity");
  }

  batch_ = batch;

  // Set batch size for all inputs
  for (auto &input : inputs_) {
    input.set_batch(batch);
  }

  // Set batch size for all outputs
  for (auto &output : outputs_) {
    output.set_batch(batch);
  }
}

void GraphIO::parse_config(const json &config, int32_t capacity) {
  try {
    // Parse input specifications
    if (!config.contains("inputs")) {
      throw std::runtime_error("Missing 'inputs' field in config");
    }

    const auto &inputs_array = config.at("inputs");
    if (!inputs_array.is_array()) {
      throw std::runtime_error("'inputs' must be an array");
    }

    if (inputs_array.empty()) {
      LOG(WARNING) << "Empty inputs array in config";
    }

    inputs_.reserve(inputs_array.size());
    int32_t index = 0;

    for (const auto &item : inputs_array) {
      // Validate required fields
      if (!item.contains("shape")) {
        throw std::runtime_error("Missing 'shape' field in input " +
                                 std::to_string(index));
      }
      if (!item.contains("dtype")) {
        throw std::runtime_error("Missing 'dtype' field in input " +
                                 std::to_string(index));
      }

      auto shape = item.at("shape").get<std::vector<int32_t>>();

      // Validate shape
      if (shape.empty()) {
        throw std::runtime_error("Empty shape for input " +
                                 std::to_string(index));
      }

      // Check for negative dimensions
      // first dim = -1 for dynamic batch
      for (size_t i = 1; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
          throw std::runtime_error("Invalid dimension " +
                                   std::to_string(shape[i]) + " at position " +
                                   std::to_string(i) + " for input " +
                                   std::to_string(index));
        }
      }

      // Calculate width (product of all dimensions except batch)
      const int64_t width =
          std::accumulate(shape.begin() + 1, shape.end(), int64_t{1},
                          std::multiplies<int64_t>());

      if (width <= 0) {
        throw std::runtime_error("Invalid width " + std::to_string(width) +
                                 " for input " + std::to_string(index));
      }

      int32_t dtype_value = item.at("dtype").get<int32_t>();
      minia::DataType dtype = static_cast<minia::DataType>(dtype_value);

      // Validate dtype
      if (dtype != minia::DataType::kFloatValue &&
          dtype != minia::DataType::kInt64Value) {
        throw std::runtime_error("Unsupported dtype " +
                                 std::to_string(dtype_value) + " for input " +
                                 std::to_string(index));
      }

      inputs_.emplace_back(index, capacity, width, dtype);
      index++;
    }

    // Parse output specifications
    if (!config.contains("outputs")) {
      throw std::runtime_error("Missing 'outputs' field in config");
    }

    const auto &outputs_array = config.at("outputs");
    if (!outputs_array.is_array()) {
      throw std::runtime_error("'outputs' must be an array");
    }

    if (outputs_array.empty()) {
      LOG(WARNING) << "Empty outputs array in config";
    }

    outputs_.reserve(outputs_array.size());

    for (const auto &item : outputs_array) {
      // Validate required fields
      if (!item.contains("shape")) {
        throw std::runtime_error("Missing 'shape' field in output " +
                                 std::to_string(index));
      }

      auto shape = item.at("shape").get<std::vector<int32_t>>();

      // Validate shape
      if (shape.empty()) {
        throw std::runtime_error("Empty shape for output " +
                                 std::to_string(index));
      }

      // Check for negative dimensions
      // first dim = -1 for dynamic batch
      for (size_t i = 1; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
          throw std::runtime_error("Invalid dimension " +
                                   std::to_string(shape[i]) + " at position " +
                                   std::to_string(i) + " for output " +
                                   std::to_string(index));
        }
      }

      // Calculate width (product of all dimensions except batch)
      const int64_t width =
          std::accumulate(shape.begin() + 1, shape.end(), int64_t{1},
                          std::multiplies<int64_t>());

      if (width <= 0) {
        throw std::runtime_error("Invalid width " + std::to_string(width) +
                                 " for output " + std::to_string(index));
      }

      outputs_.emplace_back(index, width);
      index++;
    }

  } catch (const nlohmann::json::out_of_range &e) {
    LOG(ERROR) << "Missing required field in config: " << e.what();
    throw std::runtime_error("Missing required field in config: " +
                             std::string(e.what()));
  } catch (const nlohmann::json::type_error &e) {
    LOG(ERROR) << "Invalid type in config: " << e.what();
    throw std::runtime_error("Invalid type in config: " +
                             std::string(e.what()));
  } catch (const std::exception &e) {
    LOG(ERROR) << "Error parsing config: " << e.what();
    throw;
  }
}

void GraphIO::reset() {
  // Reset all output data pointers
  for (size_t i = 0; i < outputs_.size(); i++) {
    outputs_[i].set_data(nullptr);
  }
}

void GraphIO::set_outputs(float **data) {
  if (!data) {
    LOG(ERROR) << "Null output data pointer array";
    throw std::invalid_argument("Output data array cannot be null");
  }

  // Set data pointers for all outputs
  for (size_t i = 0; i < outputs_.size(); i++) {
    if (!data[i]) {
      LOG(WARNING) << "Null data pointer for output " << i;
    }
    outputs_[i].set_data(data[i]);
  }
}

void GraphIO::zero() {
  // Zero out all input tensors
  for (auto &input : inputs_) {
    input.zero();
  }

  // Zero out all output tensors
  for (auto &output : outputs_) {
    output.zero();
  }
}

// ============================================================================
// Buffer Implementation
// ============================================================================

std::unique_ptr<GraphIO> Buffer::get() {
  std::unique_ptr<GraphIO> block;

  {
    // Try to get cached object from queue (fast path)
    std::lock_guard<std::mutex> lock(mutex_);
    if (!queue_.empty()) {
      block = std::move(queue_.front());
      queue_.pop();
      --size_;
    }
  }

  // Return cached object if found
  if (block) {
    return block;
  }

  // Slow path: create new object outside lock for better concurrency
  int32_t capacity = label_to_capacity(label_);

  try {
    return std::make_unique<GraphIO>(*config_, capacity);
  } catch (const std::exception &e) {
    LOG(ERROR) << "Failed to create GraphIO for Buffer[" << label_
               << "]: " << e.what();
    throw;
  }
}

void Buffer::put(std::unique_ptr<GraphIO> block) {
  if (!block) {
    return;
  }

  // Validate block capacity matches this buffer
  int32_t expected_capacity = label_to_capacity(label_);
  if (block->capacity() != expected_capacity) {
    LOG(WARNING) << "GraphIO capacity " << block->capacity()
                 << " doesn't match Buffer[" << label_ << "] capacity "
                 << expected_capacity << ", discarding";
    return;
  }

  // Reset object state before caching
  try {
    block->reset();
  } catch (const std::exception &e) {
    LOG(ERROR) << "Failed to reset GraphIO for Buffer[" << label_
               << "]: " << e.what();
    return; // Discard the block
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (size_ < max_capacity_) {
    // Cache object if pool has space
    queue_.push(std::move(block));
    ++size_;
  }
  // If pool is full, block is automatically destroyed when leaving scope
}

size_t Buffer::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return size_;
}

// ============================================================================
// Arena Implementation
// ============================================================================

Arena::Arena(std::shared_ptr<json> config) : config_(config) {
  if (!config) {
    LOG(ERROR) << "Null config pointer for Arena";
    throw std::invalid_argument("Config cannot be null");
  }

  // Initialize buffer pools for different capacity ranges
  // Buffer 0: 32, Buffer 1: 64, ..., Buffer 15: 512
  buffers_.reserve(LABEL_SIZE);

  try {
    for (int32_t i = 0; i < LABEL_SIZE; ++i) {
      buffers_.push_back(std::make_unique<Buffer>(i, config));
    }

    LOG(INFO) << "Created Arena with " << LABEL_SIZE << " buffer pools";
  } catch (const std::exception &e) {
    LOG(ERROR) << "Failed to create Arena: " << e.what();
    throw;
  }
}

std::unique_ptr<GraphIO> Arena::get(int32_t batch) {
  // Validate and clamp batch size
  if (batch <= 0) {
    LOG(WARNING) << "Invalid batch size " << batch << ", using 1";
    batch = 1;
  }

  // Calculate which buffer pool to use
  int32_t label = batch_to_label(batch);

  // Clamp label to valid range
  if (label < 0) {
    LOG(WARNING) << "Negative label " << label << " for batch " << batch
                 << ", using 0";
    label = 0;
  }

  // For very large batches (>512), create directly without pooling
  if (label >= LABEL_SIZE) {
    LOG(INFO) << "Large batch " << batch << " (label=" << label
              << "), creating non-pooled GraphIO";
    try {
      return std::make_unique<GraphIO>(*config_, batch);
    } catch (const std::exception &e) {
      LOG(ERROR) << "Failed to create large GraphIO for batch " << batch << ": "
                 << e.what();
      throw;
    }
  }

  // Get from appropriate buffer pool
  try {
    return buffers_[label]->get();
  } catch (const std::exception &e) {
    LOG(ERROR) << "Failed to get GraphIO from Buffer[" << label
               << "]: " << e.what();
    throw;
  }
}

void Arena::put(std::unique_ptr<GraphIO> block) {
  if (!block) {
    return;
  }

  // Determine which buffer pool this object belongs to
  int32_t capacity = block->capacity();
  int32_t label = batch_to_label(capacity);

  // Objects outside pooling range are automatically destroyed
  if (label < 0 || label >= LABEL_SIZE) {
    return; // block is destroyed when leaving scope
  }

  // Return to appropriate buffer pool
  try {
    buffers_[label]->put(std::move(block));
  } catch (const std::exception &e) {
    LOG(ERROR) << "Failed to return GraphIO to Buffer[" << label
               << "]: " << e.what();
    // Block will be destroyed automatically
  }
}

} // namespace longmen
