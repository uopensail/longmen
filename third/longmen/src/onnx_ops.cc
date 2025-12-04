#include "onnx_ops.h"

#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "embeddings.hpp"

namespace longmen {

// ============================================================================
// SparseEmbeddingLookupOp Implementation
// ============================================================================

SparseEmbeddingLookupOp::SparseEmbeddingLookupOp() {
  LOG(INFO) << "SparseEmbeddingLookupOp instance created";
}

void *SparseEmbeddingLookupOp::CreateKernel(const OrtApi &api,
                                            const OrtKernelInfo *info) const {
  // Validate input
  if (!info) {
    LOG(ERROR) << "CreateKernel called with null OrtKernelInfo pointer";
    throw std::invalid_argument("CreateKernel: OrtKernelInfo pointer is null");
  }

  try {
    auto *kernel = new SparseEmbeddingLookupKernel(api, info);
    LOG(INFO) << "Successfully created SparseEmbeddingLookupKernel at "
              << static_cast<void *>(kernel);
    return kernel;
  } catch (const std::bad_alloc &e) {
    LOG(ERROR) << "Memory allocation failed in CreateKernel: " << e.what();
    throw;
  } catch (const std::exception &e) {
    LOG(ERROR) << "Exception in CreateKernel: " << e.what();
    throw;
  }
}

void SparseEmbeddingLookupOp::KernelDestroy(void *op_kernel) {
  if (!op_kernel) {
    LOG(WARNING) << "KernelDestroy called with null pointer";
    return;
  }

  LOG(INFO) << "Destroying SparseEmbeddingLookupKernel at " << op_kernel;
  delete static_cast<SparseEmbeddingLookupKernel *>(op_kernel);
}

const char *SparseEmbeddingLookupOp::GetName() const {
  return kSparseEmbeddingLookupOpName;
}

const char *SparseEmbeddingLookupOp::GetExecutionProviderType() const {
  return kExecutionProviderType;
}

size_t SparseEmbeddingLookupOp::GetInputTypeCount() const { return 1; }

ONNXTensorElementDataType
SparseEmbeddingLookupOp::GetInputType(size_t index) const {
  if (index != 0) {
    LOG(WARNING) << "GetInputType called with invalid index: " << index;
  }
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

size_t SparseEmbeddingLookupOp::GetOutputTypeCount() const { return 1; }

ONNXTensorElementDataType
SparseEmbeddingLookupOp::GetOutputType(size_t index) const {
  if (index != 0) {
    LOG(WARNING) << "GetOutputType called with invalid index: " << index;
  }
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

OrtMemType SparseEmbeddingLookupOp::GetInputMemoryType(size_t index) const {
  if (index != 0) {
    LOG(WARNING) << "GetInputMemoryType called with invalid index: " << index;
  }
  return OrtMemTypeDefault;
}

OrtMemType SparseEmbeddingLookupOp::GetOutputMemoryType(size_t index) const {
  if (index != 0) {
    LOG(WARNING) << "GetOutputMemoryType called with invalid index: " << index;
  }
  return OrtMemTypeDefault;
}

// ============================================================================
// SparseEmbeddingLookupKernel Implementation
// ============================================================================

SparseEmbeddingLookupKernel::SparseEmbeddingLookupKernel(
    const OrtApi & /*api*/, const OrtKernelInfo *info)
    : group_id_(-1), dim_(0) {
  // Validate input
  if (!info) {
    LOG(ERROR) << "SparseEmbeddingLookupKernel constructor called with null "
                  "OrtKernelInfo pointer";
    throw std::invalid_argument(
        "SparseEmbeddingLookupKernel: OrtKernelInfo pointer is null");
  }

  // Wrap KernelInfo
  Ort::ConstKernelInfo info_wrapper(info);

  // Extract 'group' attribute
  try {
    group_id_ = info_wrapper.GetAttribute<int64_t>("group");
    LOG(INFO) << "Extracted 'group' attribute: " << group_id_;
  } catch (const Ort::Exception &e) {
    LOG(ERROR) << "Failed to get 'group' attribute: " << e.what();
    throw std::runtime_error("Failed to get required 'group' attribute: " +
                             std::string(e.what()));
  } catch (const std::exception &e) {
    LOG(ERROR) << "Unexpected exception getting 'group' attribute: "
               << e.what();
    throw std::runtime_error("Failed to get 'group' attribute: " +
                             std::string(e.what()));
  }

  // Extract 'dim' attribute
  try {
    dim_ = info_wrapper.GetAttribute<int64_t>("dim");
    LOG(INFO) << "Extracted 'dim' attribute: " << dim_;
  } catch (const Ort::Exception &e) {
    LOG(ERROR) << "Failed to get 'dim' attribute: " << e.what();
    throw std::runtime_error("Failed to get required 'dim' attribute: " +
                             std::string(e.what()));
  } catch (const std::exception &e) {
    LOG(ERROR) << "Unexpected exception getting 'dim' attribute: " << e.what();
    throw std::runtime_error("Failed to get 'dim' attribute: " +
                             std::string(e.what()));
  }

  // Validate attribute values
  if (group_id_ < 0) {
    LOG(ERROR) << "Invalid 'group' attribute: " << group_id_
               << " (must be non-negative)";
    throw std::invalid_argument(
        "Attribute 'group' must be non-negative, got: " +
        std::to_string(group_id_));
  }

  if (group_id_ >= kMaxEmbeddingNum) {
    LOG(ERROR) << "Group ID " << group_id_ << " exceeds maximum "
               << kMaxEmbeddingNum;
    throw std::invalid_argument("Attribute 'group' exceeds maximum " +
                                std::to_string(kMaxEmbeddingNum) +
                                ", got: " + std::to_string(group_id_));
  }

  if (dim_ <= 0) {
    LOG(ERROR) << "Invalid 'dim' attribute: " << dim_ << " (must be positive)";
    throw std::invalid_argument("Attribute 'dim' must be positive, got: " +
                                std::to_string(dim_));
  }

  if (dim_ > kMaxEmbeddingDim) {
    LOG(ERROR) << "Dimension " << dim_ << " exceeds maximum "
               << kMaxEmbeddingDim;
    throw std::invalid_argument("Attribute 'dim' exceeds maximum " +
                                std::to_string(kMaxEmbeddingDim) +
                                ", got: " + std::to_string(dim_));
  }

  // Validate embedding table exists and dimension matches
  try {
    auto &manager = Embeddings::get_instance();

    // Check if table is loaded
    if (!manager.is_loaded(group_id_)) {
      LOG(ERROR) << "Embedding table for group " << group_id_
                 << " is not loaded";
      throw std::runtime_error("Embedding table for group " +
                               std::to_string(group_id_) + " is not loaded");
    }

    // Validate dimension
    const int32_t actual_dim = manager.dimension(group_id_);
    if (actual_dim < 0) {
      LOG(ERROR) << "Failed to get dimension for group " << group_id_;
      throw std::runtime_error("Failed to get dimension for group " +
                               std::to_string(group_id_));
    }

    if (actual_dim != static_cast<int32_t>(dim_)) {
      LOG(ERROR) << "Dimension mismatch for group " << group_id_
                 << ": attribute specifies " << dim_ << " but table has "
                 << actual_dim;
      throw std::invalid_argument(
          "Dimension mismatch for group " + std::to_string(group_id_) +
          ": attribute specifies " + std::to_string(dim_) +
          " but table has dimension " + std::to_string(actual_dim));
    }

    // Get table statistics
    const int64_t count = manager.count(group_id_);
    if (count < 0) {
      LOG(WARNING) << "Invalid count for group " << group_id_ << ": " << count;
    }

    LOG(INFO) << "SparseEmbeddingLookupKernel initialized successfully: "
              << "group=" << group_id_ << ", dim=" << dim_
              << ", count=" << count;

  } catch (const std::exception &e) {
    LOG(ERROR) << "Embedding table validation failed for group " << group_id_
               << ": " << e.what();
    throw;
  }
}

SparseEmbeddingLookupKernel::~SparseEmbeddingLookupKernel() {
  LOG(INFO) << "Destroying SparseEmbeddingLookupKernel: group=" << group_id_
            << ", dim=" << dim_;
}

void SparseEmbeddingLookupKernel::Compute(OrtKernelContext *context) {
  // Validate context
  if (!context) {
    LOG(ERROR) << "Compute called with null OrtKernelContext pointer";
    throw std::invalid_argument("Compute: OrtKernelContext pointer is null");
  }

  try {
    // Get input tensor
    const OrtValue *input_ort_value = nullptr;
    Ort::ThrowOnError(
        Ort::GetApi().KernelContext_GetInput(context, 0, &input_ort_value));

    if (!input_ort_value) {
      LOG(ERROR) << "Failed to get input tensor at index 0";
      throw std::runtime_error("Failed to get input tensor at index 0");
    }

    Ort::ConstValue input_tensor(input_ort_value);

    // Validate input is a tensor
    if (!input_tensor.IsTensor()) {
      LOG(ERROR) << "Input at index 0 is not a tensor";
      throw std::runtime_error("Input at index 0 must be a tensor");
    }

    // Get input shape and type
    const auto type_info = input_tensor.GetTensorTypeAndShapeInfo();
    const auto input_shape = type_info.GetShape();
    const auto element_type = type_info.GetElementType();

    // Validate input type
    if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      LOG(ERROR) << "Input tensor has wrong type: "
                 << static_cast<int>(element_type) << " (expected INT64)";
      throw std::runtime_error("Input tensor must be INT64 type, got type: " +
                               std::to_string(static_cast<int>(element_type)));
    }

    // Validate input shape (must be 2D)
    if (input_shape.size() != 2) {
      LOG(ERROR) << "Input tensor has wrong rank: " << input_shape.size()
                 << " (expected 2D)";
      throw std::runtime_error(
          "Input must be 2D tensor with shape [batch_size, seq_len], got " +
          std::to_string(input_shape.size()) + "D tensor");
    }

    const int64_t batch_size = input_shape[0];
    const int64_t seq_len = input_shape[1];

    // Validate dimensions are positive
    if (batch_size <= 0 || seq_len <= 0) {
      LOG(ERROR) << "Invalid input dimensions: batch_size=" << batch_size
                 << ", seq_len=" << seq_len;
      throw std::runtime_error(
          "Input tensor dimensions must be positive, got batch_size=" +
          std::to_string(batch_size) + ", seq_len=" + std::to_string(seq_len));
    }

    // Check for potential overflow in total_elements
    if (batch_size > INT64_MAX / seq_len) {
      LOG(ERROR) << "Total elements overflow: batch_size=" << batch_size
                 << ", seq_len=" << seq_len;
      throw std::runtime_error("Total elements would overflow: batch_size=" +
                               std::to_string(batch_size) +
                               ", seq_len=" + std::to_string(seq_len));
    }

    const int64_t total_elements = batch_size * seq_len;

    // Check for potential overflow in output size
    if (total_elements > INT64_MAX / dim_) {
      LOG(ERROR) << "Output tensor size overflow: total_elements="
                 << total_elements << ", dim=" << dim_;
      throw std::runtime_error(
          "Output tensor size would overflow: batch_size=" +
          std::to_string(batch_size) + ", seq_len=" + std::to_string(seq_len) +
          ", dim=" + std::to_string(dim_));
    }

    // Get input data pointer
    const int64_t *input_data = input_tensor.GetTensorData<int64_t>();
    if (!input_data) {
      LOG(ERROR) << "Failed to get input data pointer";
      throw std::runtime_error("Failed to get input data pointer");
    }

    // Define output shape: [batch_size, seq_len, dim]
    const std::vector<int64_t> output_shape = {batch_size, seq_len, dim_};

    // Allocate output tensor
    OrtValue *output_ort_value = nullptr;
    Ort::ThrowOnError(Ort::GetApi().KernelContext_GetOutput(
        context, 0, output_shape.data(), output_shape.size(),
        &output_ort_value));

    if (!output_ort_value) {
      LOG(ERROR) << "Failed to allocate output tensor with shape ["
                 << batch_size << ", " << seq_len << ", " << dim_ << "]";
      throw std::runtime_error("Failed to allocate output tensor with shape [" +
                               std::to_string(batch_size) + ", " +
                               std::to_string(seq_len) + ", " +
                               std::to_string(dim_) + "]");
    }

    Ort::UnownedValue output_tensor(output_ort_value);

    // Get output data buffer
    float *output_data = output_tensor.GetTensorMutableData<float>();
    if (!output_data) {
      LOG(ERROR) << "Failed to get output data pointer";
      throw std::runtime_error("Failed to get output data pointer");
    }

    // Perform batch embedding lookup
    auto &manager = Embeddings::get_instance();

    // Double-check table is still loaded
    if (!manager.is_loaded(group_id_)) {
      LOG(ERROR) << "Embedding table for group " << group_id_
                 << " is no longer loaded";
      throw std::runtime_error("Embedding table for group " +
                               std::to_string(group_id_) +
                               " is no longer loaded");
    }

    // const size_t found =
    //     manager.batch_lookup(group_id_, input_data,
    //                          static_cast<size_t>(total_elements),
    //                          output_data);

    // Log statistics
    // const double hit_rate =
    //     (total_elements > 0) ? (100.0 * found) / total_elements : 0.0;

    // if (found < static_cast<size_t>(total_elements)) {
    //   LOG(WARNING) << "Embedding lookup incomplete: found=" << found << "/"
    //                << total_elements << " (" << std::fixed
    //                << std::setprecision(1) << hit_rate << "%) for group "
    //                << group_id_;
    // }

    // LOG(INFO) << "Embedding lookup complete: batch_size=" << batch_size
    //           << ", seq_len=" << seq_len << ", found=" << found << "/"
    //           << total_elements << " (" << std::fixed << std::setprecision(1)
    //           << hit_rate << "%), group=" << group_id_;

  } catch (const Ort::Exception &e) {
    LOG(ERROR) << "ONNX Runtime exception in Compute for group " << group_id_
               << ": " << e.what();
    throw std::runtime_error("ONNX Runtime error in Compute for group " +
                             std::to_string(group_id_) + ": " +
                             std::string(e.what()));
  } catch (const std::exception &e) {
    LOG(ERROR) << "Exception in Compute for group " << group_id_ << ": "
               << e.what();
    throw std::runtime_error("Compute failed for group " +
                             std::to_string(group_id_) + ": " +
                             std::string(e.what()));
  }
}

void SparseEmbeddingLookupKernel::validate_input_tensor(
    const Ort::ConstValue &input_tensor,
    const Ort::TensorTypeAndShapeInfo &shape_info) const {
  // Validate tensor type
  if (!input_tensor.IsTensor()) {
    LOG(ERROR) << "Input is not a tensor";
    throw std::runtime_error("Input must be a tensor");
  }

  // Validate element type
  const auto element_type = shape_info.GetElementType();
  if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    LOG(ERROR) << "Invalid input element type: "
               << static_cast<int>(element_type);
    throw std::runtime_error("Input must be INT64 type");
  }

  // Validate shape
  const auto shape = shape_info.GetShape();
  if (shape.size() != 2) {
    LOG(ERROR) << "Invalid input shape rank: " << shape.size();
    throw std::runtime_error("Input must be 2D tensor");
  }

  if (shape[0] <= 0 || shape[1] <= 0) {
    LOG(ERROR) << "Invalid input dimensions: [" << shape[0] << ", " << shape[1]
               << "]";
    throw std::runtime_error("Input dimensions must be positive");
  }
}

size_t SparseEmbeddingLookupKernel::perform_lookup(const int64_t *keys,
                                                   size_t num_keys,
                                                   float *output) const {
  if (!keys || !output) {
    LOG(ERROR) << "Null pointer in perform_lookup";
    throw std::invalid_argument("Keys and output pointers cannot be null");
  }

  if (num_keys == 0) {
    return 0;
  }

  auto &manager = Embeddings::get_instance();
  return manager.batch_lookup(group_id_, keys, num_keys, output);
}

} // namespace longmen
