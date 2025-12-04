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

#ifndef LONGMEN_GRAPH_H_
#define LONGMEN_GRAPH_H_

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"
#include "json.hpp"
#include "onnx_ops.h"

namespace longmen {

using json = nlohmann::json;

// Forward declarations
class GraphIO;

/// Maximum number of intra-op threads (0 = auto-detect)
constexpr int32_t kMaxIntraOpThreads = 128;

/// Default number of intra-op threads
constexpr int32_t kDefaultIntraOpThreads = 0;

/**
 * @class OnnxRuntimeEnvSingleton
 * @brief Thread-safe singleton wrapper for ONNX Runtime environment
 *
 * Provides global access to a single Ort::Env instance shared across all
 * sessions in the application. The environment manages global ONNX Runtime
 * state including logging and threading configuration.
 *
 * @par Thread Safety
 * - Thread-safe singleton using Meyer's pattern (C++11 magic statics)
 * - Environment is initialized once on first access
 * - Destroyed automatically at program exit
 *
 * @par Logging Configuration
 * - Default log level: ORT_LOGGING_LEVEL_ERROR
 * - Logger name: "OnnxRuntimeEnvironment"
 * - Can be changed by modifying the initialization in get()
 *
 * @par Usage Example
 * @code
 * // Get environment (creates on first call)
 * Ort::Env& env = OnnxRuntimeEnvSingleton::get();
 *
 * // Use with session
 * Ort::Session session(env, "model.onnx", options);
 * @endcode
 *
 * @note Non-copyable and non-movable
 * @note Environment lifetime spans entire program execution
 *
 * @see Ort::Env
 */
class OnnxRuntimeEnvSingleton {
public:
  /**
   * @brief Gets the singleton ONNX Runtime environment
   *
   * Returns reference to the global Ort::Env instance. On first call,
   * initializes the environment with ERROR logging level.
   *
   * @return Reference to the global Ort::Env instance
   *
   * @note Thread-safe initialization (C++11 guarantees)
   * @note First call may take longer due to initialization
   * @note Subsequent calls return immediately
   *
   * @par Example
   * @code
   * Ort::Env& env = OnnxRuntimeEnvSingleton::get();
   * @endcode
   */
  static Ort::Env &get() noexcept {
    static Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "OnnxRuntimeEnvironment");
    return env;
  }

private:
  /// Private constructor (singleton pattern)
  OnnxRuntimeEnvSingleton() = default;

  /// Private destructor (singleton pattern)
  ~OnnxRuntimeEnvSingleton() = default;

  /** @brief Copy constructor (deleted - singleton) */
  OnnxRuntimeEnvSingleton(const OnnxRuntimeEnvSingleton &) = delete;

  /** @brief Copy assignment (deleted - singleton) */
  OnnxRuntimeEnvSingleton &operator=(const OnnxRuntimeEnvSingleton &) = delete;

  /** @brief Move constructor (deleted - singleton) */
  OnnxRuntimeEnvSingleton(OnnxRuntimeEnvSingleton &&) = delete;

  /** @brief Move assignment (deleted - singleton) */
  OnnxRuntimeEnvSingleton &operator=(OnnxRuntimeEnvSingleton &&) = delete;
};

/**
 * @class CPUGraph
 * @brief CPU-based ONNX model graph wrapper with optimized inference
 *
 * Encapsulates an ONNX Runtime session and provides a simplified interface
 * for model inference on CPU. Handles model loading, session configuration,
 * metadata extraction, and batch inference execution.
 *
 * @par Features
 * - Automatic model loading and validation
 * - Configurable thread pool for intra-op parallelism
 * - Input/output node metadata extraction
 * - Batch inference with dynamic batch sizes
 * - Custom operator support (e.g., SparseEmbeddingLookup)
 * - Graph optimization (level 3 - all optimizations)
 *
 * @par Configuration Format (JSON)
 * @code{.json}
 * {
 *   "model_path": "model.onnx",           // Required: relative to workdir
 *   "threads": 4,                          // Optional: intra-op threads
 * (default: 0 = auto) "inputs": [                            // Required: input
 * node specifications
 *     {
 *       "name": "input_ids",
 *       "shape": [-1, 128],                // -1 for dynamic batch dimension
 *       "dtype": 7                         // ONNX data type (7 = INT64)
 *     }
 *   ],
 *   "outputs": [                           // Required: output node
 * specifications
 *     {
 *       "name": "logits",
 *       "shape": [-1, 10]
 *     }
 *   ]
 * }
 * @endcode
 *
 * @par Thread Safety
 * - Thread-safe for inference operations (const methods)
 * - Multiple threads can call forward() concurrently
 * - Construction is not thread-safe
 * - Session is immutable after construction
 *
 * @par Memory Management
 * - Session uses shared_ptr for safe concurrent access
 * - Input/output tensors managed by GraphIO
 * - Automatic cleanup on destruction
 *
 * @par Performance Considerations
 * - Graph optimization enabled (level 3)
 * - Memory pattern optimization enabled
 * - Configurable thread pool for CPU parallelism
 * - Zero-copy when possible
 *
 * @par Usage Example
 * @code
 * #include "graph.h"
 * #include "arena.h"
 *
 * // Load configuration
 * json config = load_config("model_config.json");
 *
 * // Create graph
 * CPUGraph graph(config, "/path/to/workdir");
 *
 * // Check if ready
 * if (!graph.is_ready()) {
 *   LOG(ERROR) << "Graph initialization failed";
 *   return -1;
 * }
 *
 * // Create arena for memory management
 * Arena arena(graph.get_config());
 *
 * // Get GraphIO for batch size 32
 * auto io = arena.get(32);
 *
 * // Fill input tensors
 * // ... (set input data)
 *
 * // Run inference
 * int ret = graph.forward(*io);
 * if (ret != 0) {
 *   LOG(ERROR) << "Inference failed";
 * }
 *
 * // Process outputs
 * // ... (read output data)
 *
 * // Return GraphIO to arena
 * arena.put(std::move(io));
 * @endcode
 *
 * @note Non-copyable and non-movable
 * @note Always check is_ready() before inference
 * @note Supports custom operators via domain registration
 *
 * @see GraphIO
 * @see Arena
 * @see OnnxRuntimeEnvSingleton
 */
class CPUGraph {
public:
  /**
   * @brief Constructs CPU graph from configuration
   *
   * Loads ONNX model, initializes session with optimizations, and extracts
   * input/output node metadata. The model path in config is resolved relative
   * to workdir.
   *
   * @param config JSON configuration containing model path and node specs
   * @param workdir Working directory for resolving relative paths
   *
   * @throws std::invalid_argument If config is invalid:
   *         - Missing required fields (model_path, inputs, outputs)
   *         - Invalid field types
   *         - Empty node lists
   * @throws std::runtime_error If model loading fails:
   *         - Model file not found
   *         - Invalid ONNX format
   *         - Session creation failure
   * @throws std::runtime_error If node metadata extraction fails:
   *         - Node name mismatch
   *         - Invalid tensor shapes
   *         - Unsupported data types
   *
   * @note Logs INFO on successful initialization
   * @note Logs ERROR on failures
   * @note Sets is_ready_ to false on any error
   *
   * @par Configuration Requirements
   * - model_path: string, non-empty, file must exist
   * - threads: int32, optional, range [0, kMaxIntraOpThreads]
   * - inputs: array, non-empty, each with name/shape/dtype
   * - outputs: array, non-empty, each with name/shape
   */
  explicit CPUGraph(const json &config, const std::string &workdir);

  /**
   * @brief Destructor
   *
   * Automatically releases ONNX Runtime session resources.
   *
   * @note Logs INFO on destruction
   * @note Thread-safe (session is shared_ptr)
   */
  ~CPUGraph();

  /** @brief Copy constructor (deleted - non-copyable) */
  CPUGraph(const CPUGraph &) = delete;

  /** @brief Copy assignment (deleted - non-copyable) */
  CPUGraph &operator=(const CPUGraph &) = delete;

  /** @brief Move constructor (deleted - non-movable) */
  CPUGraph(CPUGraph &&) = delete;

  /** @brief Move assignment (deleted - non-movable) */
  CPUGraph &operator=(CPUGraph &&) = delete;

  /**
   * @brief Performs batch inference
   *
   * Executes the model on input tensors in GraphIO and writes results to
   * output tensors. The GraphIO object must have been properly initialized
   * with input data before calling this method.
   *
   * @param io GraphIO object containing input and output tensors
   *
   * @return 0 on success, -1 on failure
   *
   * @throws std::invalid_argument If io is invalid
   * @throws std::runtime_error If inference fails
   *
   * @note Thread-safe (const method, session is immutable)
   * @note Multiple threads can call concurrently
   * @note Output tensors in io will be overwritten
   * @note Returns -1 if !is_ready()
   *
   * @warning Undefined behavior if batch > io.capacity()
   * @warning Input tensors must be filled before calling
   *
   * @par Performance
   * - Batch processing for efficiency
   * - Parallel execution with configured thread pool
   * - Memory pattern optimization
   *
   * @par Example
   * @code
   * auto io = arena.get(32);
   * // Fill inputs...
   * int ret = graph.forward(*io);
   * if (ret == 0) {
   *   // Process outputs...
   * }
   * @endcode
   */
  int forward(GraphIO &io) const;

  /**
   * @brief Checks if the model is ready for inference
   *
   * Returns true if the model loaded successfully and the session is valid.
   * Always check this before calling forward().
   *
   * @return true if ready for inference, false otherwise
   *
   * @note Thread-safe (const method)
   * @note noexcept guarantee
   *
   * @par Example
   * @code
   * if (graph.is_ready()) {
   *   graph.forward(io);
   * } else {
   *   LOG(ERROR) << "Graph not ready";
   * }
   * @endcode
   */
  bool is_ready() const noexcept { return is_ready_ && session_ != nullptr; }

  /**
   * @brief Gets the number of input nodes
   *
   * @return Number of input tensors
   *
   * @note Thread-safe
   */
  size_t input_count() const noexcept { return input_node_names_.size(); }

  /**
   * @brief Gets the number of output nodes
   *
   * @return Number of output tensors
   *
   * @note Thread-safe
   */
  size_t output_count() const noexcept { return output_node_names_.size(); }

  /**
   * @brief Gets input node name by index
   *
   * @param index Input node index
   * @return Input node name
   *
   * @throws std::out_of_range If index is invalid
   */
  const std::string &input_name(size_t index) const {
    if (index >= input_node_names_.size()) {
      throw std::out_of_range("Input index out of range: " +
                              std::to_string(index));
    }
    return input_node_names_[index];
  }

  /**
   * @brief Gets output node name by index
   *
   * @param index Output node index
   * @return Output node name
   *
   * @throws std::out_of_range If index is invalid
   */
  const std::string &output_name(size_t index) const {
    if (index >= output_node_names_.size()) {
      throw std::out_of_range("Output index out of range: " +
                              std::to_string(index));
    }
    return output_node_names_[index];
  }

private:
  /**
   * @brief Parses JSON configuration and extracts model information
   *
   * Validates the configuration structure and resolves the model path
   * relative to the working directory.
   *
   * @param config JSON configuration object
   * @param workdir Working directory for path resolution
   * @param[out] input_names List of input node names
   * @param[out] output_names List of output node names
   * @param[out] model_path Full path to model file
   *
   * @throws std::runtime_error If required fields are missing
   * @throws std::invalid_argument If field types are incorrect
   * @throws std::runtime_error If model file doesn't exist
   */
  void parse_config(const json &config, const std::string &workdir,
                    std::vector<std::string> &input_names,
                    std::vector<std::string> &output_names,
                    std::string &model_path);

  /**
   * @brief Initializes ONNX Runtime session with optimizations
   *
   * Creates a session with:
   * - Graph optimization level: all optimizations enabled (level 3)
   * - Intra-op parallelism: specified thread count
   * - Memory pattern optimization enabled
   * - Custom operator domain registered
   *
   * @param model_path Path to ONNX model file
   * @param threads Number of intra-op threads (0 = auto-detect)
   *
   * @throws std::invalid_argument If threads < 0 or > kMaxIntraOpThreads
   * @throws std::runtime_error If session creation fails
   * @throws std::runtime_error If model file doesn't exist or is invalid
   */
  void initialize_session(const std::string &model_path, int32_t threads);

  /**
   * @brief Initializes input node metadata from model
   *
   * Extracts and validates input node information including:
   * - Node names and order
   * - Tensor dimensions (first dim must be -1 for batch)
   * - Data types (INT64, FLOAT, etc.)
   * - Feature widths (product of non-batch dimensions)
   *
   * @param input_names List of input node names from config
   *
   * @throws std::invalid_argument If node names don't match model
   * @throws std::runtime_error If node metadata extraction fails
   * @throws std::runtime_error If first dimension is not dynamic (-1)
   * @throws std::runtime_error If unsupported data type
   */
  void initialize_input_nodes(const std::vector<std::string> &input_names);

  /**
   * @brief Initializes output node metadata from model
   *
   * Extracts and validates output node information including:
   * - Node names and order
   * - Tensor dimensions (first dim must be -1 for batch)
   * - Feature widths (product of non-batch dimensions)
   *
   * @param output_names List of output node names from config
   *
   * @throws std::invalid_argument If node names don't match model
   * @throws std::runtime_error If node metadata extraction fails
   * @throws std::runtime_error If first dimension is not dynamic (-1)
   */
  void initialize_output_nodes(const std::vector<std::string> &output_names);

  /**
   * @brief Validates tensor shape for dynamic batching
   *
   * @param shape Tensor shape
   * @param node_name Node name for error messages
   *
   * @throws std::runtime_error If first dimension is not -1
   * @throws std::runtime_error If shape is empty
   */
  void validate_dynamic_shape(const std::vector<int64_t> &shape,
                              const std::string &node_name) const;

  /**
   * @brief Calculates feature width from tensor shape
   *
   * @param shape Tensor shape (first dim is batch)
   * @return Product of all dimensions except first
   *
   * @throws std::runtime_error If overflow detected
   */
  size_t calculate_feature_width(const std::vector<int64_t> &shape) const;

private:
  /// ONNX Runtime session (shared for thread-safety)
  std::shared_ptr<Ort::Session> session_;

  /// Input node names in order
  std::vector<std::string> input_node_names_;

  /// Input feature widths (elements per sample, excluding batch dim)
  std::vector<size_t> input_widths_;

  /// Input tensor dimensions (first dim is batch = -1)
  std::vector<std::vector<int64_t>> input_node_dims_;

  /// Input tensor data types (FLOAT, INT64, etc.)
  std::vector<ONNXTensorElementDataType> input_node_types_;

  /// Output node names in order
  std::vector<std::string> output_node_names_;

  /// Output feature widths (elements per sample, excluding batch dim)
  std::vector<size_t> output_widths_;

  /// Output tensor dimensions (first dim is batch = -1)
  std::vector<std::vector<int64_t>> output_node_dims_;

  /// Sum of all output widths (for convenience)
  size_t total_output_width_;

  /// Memory info for creating CPU tensors
  Ort::MemoryInfo memory_info_;

  /// Model ready status (true if successfully initialized)
  mutable bool is_ready_;

  /// Number of intra-op threads
  int32_t threads_;
};

} // namespace longmen

#endif // LONGMEN_GRAPH_H_
