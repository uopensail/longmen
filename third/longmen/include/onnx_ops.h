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

#ifndef LONGMEN_ONNX_OP_H_
#define LONGMEN_ONNX_OP_H_

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <string>
#include <vector>

#include "common.hpp"

namespace longmen {

// Forward declaration
struct SparseEmbeddingLookupKernel;

/// Custom operator domain name for LongMen operators
constexpr const char *kCustomOpDomain = "custom";

/// Operator name as registered in ONNX Runtime
constexpr const char *kSparseEmbeddingLookupOpName = "SparseEmbeddingLookup";

/// Execution provider type (CPU-based implementation)
constexpr const char *kExecutionProviderType = "CPUExecutionProvider";

/**
 * @class SparseEmbeddingLookupOp
 * @brief Custom ONNX Runtime operator for sparse embedding lookup with batch
 * processing
 *
 * This operator performs efficient batch embedding lookups by receiving a
 * tensor of embedding keys and returning corresponding embedding vectors.
 * Missing keys are automatically filled with zeros.
 *
 * @par Operator Signature
 * - **Operator Name**: SparseEmbeddingLookup
 * - **Domain**: custom
 * - **Opset Version**: 1
 * - **Input**: int64 tensor with shape [batch_size, seq_len]
 * - **Output**: float32 tensor with shape [batch_size, seq_len, dim]
 * - **Attributes**:
 *   - `group` (int64, required): Embedding table group ID (>= 0)
 *   - `dim` (int64, required): Embedding dimension (> 0)
 *
 * @par Thread Safety
 * - Thread-safe for inference (read-only operations)
 * - Multiple instances can execute concurrently
 * - No synchronization required between kernel instances
 *
 * @par Memory Management
 * - Input tensors are read-only (no modification)
 * - Output tensors are allocated by ONNX Runtime
 * - Kernel instances are managed by ONNX Runtime
 *
 * @par Performance Considerations
 * - Batch processing for better cache locality
 * - Zero-copy when possible
 * - Optimized for CPU execution
 *
 * @warning Embedding tables must be loaded before session initialization
 * @warning The 'dim' attribute must match the actual embedding dimension
 * @warning Input tensor must be 2D with shape [batch_size, seq_len]
 *
 * @par ONNX Model Integration (Python)
 * @code{.py}
 * import onnx
 * from onnx import helper, TensorProto
 *
 * # Define custom op node
 * node = helper.make_node(
 *     'SparseEmbeddingLookup',            # Op type
 *     inputs=['input_ids'],               # Input tensor name
 *     outputs=['embeddings'],             # Output tensor name
 *     domain='custom',                    # Custom domain
 *     group=0,                            # Embedding table group ID
 *     dim=128                             # Embedding dimension
 * )
 *
 * # Add to graph with custom opset
 * graph = helper.make_graph(
 *     nodes=[node],
 *     name='embedding_graph',
 *     inputs=[helper.make_tensor_value_info('input_ids', TensorProto.INT64,
 * [None, None])], outputs=[helper.make_tensor_value_info('embeddings',
 * TensorProto.FLOAT, [None, None, 128])]
 * )
 *
 * model = helper.make_model(
 *     graph,
 *     opset_imports=[
 *         helper.make_opsetid("", 13),           # Standard ONNX opset
 *         helper.make_opsetid("custom", 1)       # Custom domain opset
 *     ]
 * )
 *
 * onnx.save(model, 'model.onnx')
 * @endcode
 *
 * @par C++ Usage Example
 * @code{.cpp}
 * #include "onnx_op.h"
 *
 * // Initialize environment
 * Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "LongMen");
 *
 * // Load embeddings first
 * Embeddings::get_instance().load("/path/to/embeddings");
 *
 * // Register custom op
 * Ort::CustomOpDomain domain("custom");
 * static SparseEmbeddingLookupOp op;
 * domain.Add(&op);
 *
 * // Create session with custom op
 * Ort::SessionOptions options;
 * options.Add(domain);
 * Ort::Session session(env, "model.onnx", options);
 *
 * // Run inference
 * std::vector<int64_t> input_ids = {1, 2, 3, 4};
 * // ... (prepare input tensors and run session)
 * @endcode
 *
 * @see SparseEmbeddingLookupKernel
 * @see Embeddings
 */
struct SparseEmbeddingLookupOp
    : Ort::CustomOpBase<SparseEmbeddingLookupOp, SparseEmbeddingLookupKernel> {
  /**
   * @brief Default constructor for operator registration
   *
   * Initializes the operator definition. This constructor is called once
   * during operator registration.
   *
   * @note Lightweight - no heavy initialization
   */
  SparseEmbeddingLookupOp();

  /**
   * @brief Creates kernel instance for each node in the graph
   *
   * ONNX Runtime calls this method for each node in the graph that uses
   * this operator. Each node gets its own kernel instance with independent
   * state (group_id, dim).
   *
   * @param api Reference to ORT API
   * @param info Kernel info containing operator attributes from ONNX model
   *
   * @return Pointer to newly created kernel instance (caller takes ownership)
   *
   * @throws std::invalid_argument If info is null
   * @throws std::runtime_error If kernel creation fails
   * @throws std::bad_alloc If memory allocation fails
   *
   * @note Kernel lifetime is managed by ONNX Runtime
   * @note Each kernel instance is independent and stateful
   *
   * @see KernelDestroy
   */
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const;

  /**
   * @brief Destroys kernel instance
   *
   * ONNX Runtime calls this method to destroy kernel instances when they
   * are no longer needed (e.g., session destruction).
   *
   * @param op_kernel Pointer to kernel instance to destroy (can be nullptr)
   *
   * @note Safe to call with nullptr
   * @note Called automatically by ONNX Runtime
   *
   * @see CreateKernel
   */
  void KernelDestroy(void *op_kernel);

  /**
   * @brief Gets the operator name as defined in ONNX model
   *
   * @return Operator name string ("SparseEmbeddingLookup")
   *
   * @note Must match the op_type in ONNX model
   */
  const char *GetName() const;

  /**
   * @brief Gets the execution provider type for this operator
   *
   * @return Execution provider type string ("CPUExecutionProvider")
   *
   * @note This operator runs on CPU
   * @note GPU support can be added by implementing a GPU kernel
   */
  const char *GetExecutionProviderType() const;

  /**
   * @brief Gets the number of input tensors
   *
   * @return Number of inputs (always 1)
   *
   * @note Input 0: int64 tensor with embedding keys
   */
  size_t GetInputTypeCount() const;

  /**
   * @brief Gets the data type of an input tensor
   *
   * @param index Input tensor index (must be 0)
   *
   * @return Data type (ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
   *
   * @note Only index 0 is valid
   */
  ONNXTensorElementDataType GetInputType(size_t index) const;

  /**
   * @brief Gets the number of output tensors
   *
   * @return Number of outputs (always 1)
   *
   * @note Output 0: float32 tensor with embedding vectors
   */
  size_t GetOutputTypeCount() const;

  /**
   * @brief Gets the data type of an output tensor
   *
   * @param index Output tensor index (must be 0)
   *
   * @return Data type (ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
   *
   * @note Only index 0 is valid
   */
  ONNXTensorElementDataType GetOutputType(size_t index) const;

  /**
   * @brief Gets input memory type
   *
   * @param index Input tensor index
   *
   * @return Memory type (OrtMemTypeDefault)
   *
   * @note Input tensors use default memory allocation
   */
  OrtMemType GetInputMemoryType(size_t index) const;

  /**
   * @brief Gets output memory type
   *
   * @param index Output tensor index
   *
   * @return Memory type (OrtMemTypeDefault)
   *
   * @note Output tensors use default memory allocation
   */
  OrtMemType GetOutputMemoryType(size_t index) const;
};

/**
 * @class SparseEmbeddingLookupKernel
 * @brief Kernel instance for sparse embedding lookup (stateful, one per node)
 *
 * Each node in the ONNX graph gets its own kernel instance with independent
 * state (group_id, dim). This allows multiple embedding lookup nodes with
 * different configurations in the same graph.
 *
 * @par Lifecycle
 * 1. Created by SparseEmbeddingLookupOp::CreateKernel()
 * 2. Compute() called for each inference
 * 3. Destroyed by SparseEmbeddingLookupOp::KernelDestroy()
 *
 * @par State
 * - `group_id_`: Embedding table identifier
 * - `dim_`: Expected embedding dimension
 *
 * @par Thread Safety
 * - Compute() is thread-safe (read-only state)
 * - Multiple threads can call Compute() concurrently
 * - No internal synchronization needed
 *
 * @par Input/Output Specification
 * - **Input 0**: int64 tensor, shape [batch_size, seq_len]
 *   - Contains embedding keys to lookup
 *   - Keys can be any int64 value
 *   - Missing keys result in zero vectors
 *
 * - **Output 0**: float32 tensor, shape [batch_size, seq_len, dim]
 *   - Contains embedding vectors
 *   - Zero-filled for missing keys
 *   - Contiguous memory layout
 *
 * @par Error Handling
 * - Validates input tensor shape (must be 2D)
 * - Validates embedding table is loaded
 * - Validates dimension consistency
 * - Throws exceptions on errors (caught by ONNX Runtime)
 *
 * @see SparseEmbeddingLookupOp
 * @see Embeddings
 */
struct SparseEmbeddingLookupKernel {
  /**
   * @brief Constructor - initializes kernel with attributes from ONNX node
   *
   * Extracts and validates operator attributes (group, dim) from the ONNX
   * node definition.
   *
   * @param api Reference to ORT API
   * @param info Kernel info containing operator attributes from ONNX model
   *
   * @throws std::invalid_argument If info is null
   * @throws std::runtime_error If required attributes are missing
   * @throws std::invalid_argument If attribute values are invalid:
   *         - group_id < 0
   *         - dim <= 0
   *         - dim > kMaxEmbeddingDim
   * @throws std::runtime_error If embedding table is not loaded
   * @throws std::runtime_error If dimension mismatch with loaded table
   *
   * @note Logs INFO on successful initialization
   * @note Logs ERROR on validation failures
   */
  SparseEmbeddingLookupKernel(const OrtApi &api, const OrtKernelInfo *info);

  /**
   * @brief Destructor
   *
   * Cleans up kernel resources. Currently no explicit cleanup needed.
   *
   * @note Logs INFO on destruction
   */
  ~SparseEmbeddingLookupKernel();

  /**
   * @brief Compute function - performs the embedding lookup operation
   *
   * Main computation function called by ONNX Runtime for each inference.
   * Performs the following steps:
   * 1. Validate input tensor (shape, type)
   * 2. Allocate output tensor
   * 3. Perform batch embedding lookup
   * 4. Handle missing keys (zero-fill)
   *
   * @param context ONNX Runtime kernel context containing input/output tensors
   *
   * @throws std::invalid_argument If context is null
   * @throws std::runtime_error If input tensor is invalid:
   *         - Wrong data type (not int64)
   *         - Wrong shape (not 2D)
   *         - Empty tensor
   * @throws std::runtime_error If output allocation fails
   * @throws std::runtime_error If embedding lookup fails
   * @throws std::runtime_error If embedding table not loaded
   *
   * @note Thread-safe - can be called concurrently
   * @note Logs WARNING for missing keys
   * @note Logs ERROR for failures
   *
   * @par Performance
   * - Batch processing for efficiency
   * - Zero-copy when possible
   * - Optimized memory access patterns
   *
   * @par Example Flow
   * @code
   * Input:  [batch=2, seq=3] = [[1, 2, 3],
   *                              [4, 5, 6]]
   * Output: [batch=2, seq=3, dim=128] = [[[emb_1], [emb_2], [emb_3]],
   *                                       [[emb_4], [emb_5], [emb_6]]]
   * @endcode
   */
  void Compute(OrtKernelContext *context);

private:
  /**
   * @brief Validates input tensor shape and type
   *
   * @param input_tensor Input tensor to validate
   * @param shape_info Tensor shape information
   *
   * @throws std::runtime_error If validation fails
   */
  void
  validate_input_tensor(const Ort::ConstValue &input_tensor,
                        const Ort::TensorTypeAndShapeInfo &shape_info) const;

  /**
   * @brief Performs batch embedding lookup
   *
   * @param keys Pointer to input keys array
   * @param num_keys Total number of keys to lookup
   * @param output Pointer to output buffer
   *
   * @return Number of keys found
   *
   * @throws std::runtime_error If lookup fails
   */
  size_t perform_lookup(const int64_t *keys, size_t num_keys,
                        float *output) const;

private:
  int64_t group_id_; ///< Embedding table group ID (>= 0)
  int64_t dim_;      ///< Embedding dimension (> 0)

  /** @brief Copy constructor (deleted - non-copyable) */
  SparseEmbeddingLookupKernel(const SparseEmbeddingLookupKernel &) = delete;

  /** @brief Copy assignment (deleted - non-copyable) */
  SparseEmbeddingLookupKernel &
  operator=(const SparseEmbeddingLookupKernel &) = delete;

  /** @brief Move constructor (deleted - non-movable) */
  SparseEmbeddingLookupKernel(SparseEmbeddingLookupKernel &&) = delete;

  /** @brief Move assignment (deleted - non-movable) */
  SparseEmbeddingLookupKernel &
  operator=(SparseEmbeddingLookupKernel &&) = delete;
};

} // namespace longmen

#endif // LONGMEN_ONNX_OP_H_
