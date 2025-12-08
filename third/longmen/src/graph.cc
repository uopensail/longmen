#include "graph.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <thread>
#include <unordered_map>

#include "arena.h"
#include "embeddings.hpp"

namespace longmen {

// ============================================================================
// CPUGraph Constructor
// ============================================================================

CPUGraph::CPUGraph(const json &config, const std::string &workdir)
    : total_output_width_(0), memory_info_(Ort::MemoryInfo::CreateCpu(
                                  OrtDeviceAllocator, OrtMemTypeCPU)),
      is_ready_(false), threads_(kDefaultIntraOpThreads) {
  // Validate inputs
  if (workdir.empty()) {
    LOG(ERROR) << "Empty workdir path";
    throw std::invalid_argument("Workdir path cannot be empty");
  }

  try {
    LOG(INFO) << "Initializing CPUGraph from workdir: " << workdir;

    // Load embedding tables from working directory
    try {
      Embeddings::get_instance().load(workdir);
    } catch (const std::exception &e) {
      LOG(ERROR) << "Failed to load embeddings: " << e.what();
      throw std::runtime_error("Failed to load embeddings: " +
                               std::string(e.what()));
    }

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::string model_path;

    // Parse JSON configuration to extract model path and node names
    parse_config(config, workdir, input_names, output_names, model_path);

    // Extract thread count from config if present
    if (config.contains("threads")) {
      try {
        threads_ = config.at("threads").get<int32_t>();
        if (threads_ < 0) {
          LOG(WARNING) << "Negative thread count " << threads_
                       << ", using default";
          threads_ = kDefaultIntraOpThreads;
        } else if (threads_ > kMaxIntraOpThreads) {
          LOG(WARNING) << "Thread count " << threads_ << " exceeds maximum "
                       << kMaxIntraOpThreads << ", clamping";
          threads_ = kMaxIntraOpThreads;
        }
      } catch (const json::type_error &e) {
        LOG(WARNING) << "Invalid 'threads' type, using default: " << e.what();
        threads_ = kDefaultIntraOpThreads;
      }
    }

    // Initialize ONNX Runtime session with threading configuration
    initialize_session(model_path, threads_);

    // Extract and validate input/output node metadata from model
    initialize_input_nodes(input_names);
    initialize_output_nodes(output_names);

    // Calculate total output width for convenience
    total_output_width_ = std::accumulate(output_widths_.begin(),
                                          output_widths_.end(), size_t{0});

    // Check for overflow in total output width
    if (total_output_width_ == 0 && !output_widths_.empty()) {
      LOG(ERROR) << "Total output width overflow or all zeros";
      throw std::runtime_error("Invalid total output width");
    }

    is_ready_ = true;

    LOG(INFO) << "CPUGraph initialized successfully:"
              << " inputs=" << input_node_names_.size()
              << ", outputs=" << output_node_names_.size()
              << ", total_output_width=" << total_output_width_
              << ", threads=" << threads_;

  } catch (const std::exception &e) {
    is_ready_ = false;
    LOG(ERROR) << "Failed to initialize CPUGraph: " << e.what();
    throw;
  }
}

CPUGraph::~CPUGraph() { LOG(INFO) << "Destroying CPUGraph"; }

// ============================================================================
// Inference Implementation
// ============================================================================

int CPUGraph::forward(GraphIO &io) const {
  // Check if graph is ready
  if (!is_ready()) {
    LOG(ERROR) << "Graph is not ready for inference";
    return -1;
  }

  const int32_t batch = io.get_batch();

  // Validate batch size
  if (batch <= 0) {
    LOG(ERROR) << "Invalid batch size: " << batch;
    return -1;
  }

  try {
    const size_t input_node_size = input_node_names_.size();
    const size_t output_node_size = output_node_names_.size();

    // Prepare containers for ONNX Runtime
    std::vector<Ort::Value> input_tensors;
    std::vector<const char *> input_names;
    std::vector<const char *> output_names;

    input_tensors.reserve(input_node_size);
    input_names.reserve(input_node_size);
    output_names.reserve(output_node_size);

    // Create input tensors from GraphIO data
    for (size_t i = 0; i < input_node_size; ++i) {
      input_names.push_back(input_node_names_[i].c_str());

      // Set dynamic batch dimension
      auto dims = input_node_dims_[i];
      dims[0] = static_cast<int64_t>(batch);

      // Calculate total number of elements
      const int64_t total_elements = std::accumulate(
          dims.begin(), dims.end(), int64_t{1}, std::multiplies<int64_t>());

      void *data_ptr = io.get_input(i).get_data();

      // Create tensor based on data type
      if (input_node_types_[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info_, static_cast<int64_t *>(data_ptr),
            static_cast<size_t>(total_elements), dims.data(), dims.size()));

      } else if (input_node_types_[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info_, static_cast<float *>(data_ptr),
            static_cast<size_t>(total_elements), dims.data(), dims.size()));

      } else {
        LOG(ERROR) << "Unsupported input type for node " << i << ": "
                   << static_cast<int>(input_node_types_[i]);
        return -1;
      }
    }

    // Prepare output node names
    for (size_t i = 0; i < output_node_size; ++i) {
      output_names.push_back(output_node_names_[i].c_str());
    }

    // Execute model inference
    Ort::RunOptions run_options;
    auto output_tensors = session_->Run(
        run_options, input_names.data(), input_tensors.data(),
        input_tensors.size(), output_names.data(), output_names.size());

    // Validate output tensor count
    if (output_tensors.size() != output_node_size) {
      LOG(ERROR) << "Output tensor count mismatch: expected "
                 << output_node_size << ", got " << output_tensors.size();
      return -1;
    }

    // Copy output data to GraphIO buffers
    for (size_t i = 0; i < output_node_size; ++i) {
      void *dst_ptr = io.get_output(i).get_data();
      const float *src_ptr = output_tensors[i].GetTensorData<float>();

      const size_t byte_size = output_widths_[i] * sizeof(float) * batch;
      std::memcpy(dst_ptr, src_ptr, byte_size);
    }

    return 0; // Success

  } catch (const Ort::Exception &e) {
    LOG(ERROR) << "ONNX Runtime error: " << e.what();
    return -1;
  } catch (const std::exception &e) {
    LOG(ERROR) << "Inference error: " << e.what();
    return -1;
  }
}

// ============================================================================
// Configuration Parsing
// ============================================================================

void CPUGraph::parse_config(const json &config, const std::string &workdir,
                            std::vector<std::string> &input_names,
                            std::vector<std::string> &output_names,
                            std::string &model_path) {
  try {
    // Validate config is an object
    if (!config.is_object()) {
      throw std::runtime_error("Configuration must be a JSON object");
    }

    // Parse input node names
    if (!config.contains("inputs")) {
      throw std::runtime_error("Missing 'inputs' field in config");
    }

    const auto &inputs_array = config.at("inputs");
    if (!inputs_array.is_array()) {
      throw std::runtime_error("'inputs' must be an array");
    }

    if (inputs_array.empty()) {
      throw std::runtime_error("'inputs' array cannot be empty");
    }

    input_names.reserve(inputs_array.size());
    for (size_t i = 0; i < inputs_array.size(); ++i) {
      const auto &item = inputs_array[i];
      if (!item.contains("name")) {
        throw std::runtime_error("Input " + std::to_string(i) +
                                 " missing 'name' field");
      }

      std::string name = item.at("name").get<std::string>();
      if (name.empty()) {
        throw std::runtime_error("Input " + std::to_string(i) +
                                 " has empty name");
      }

      input_names.emplace_back(std::move(name));
    }

    // Parse output node names
    if (!config.contains("outputs")) {
      throw std::runtime_error("Missing 'outputs' field in config");
    }

    const auto &outputs_array = config.at("outputs");
    if (!outputs_array.is_array()) {
      throw std::runtime_error("'outputs' must be an array");
    }

    if (outputs_array.empty()) {
      throw std::runtime_error("'outputs' array cannot be empty");
    }

    output_names.reserve(outputs_array.size());
    for (size_t i = 0; i < outputs_array.size(); ++i) {
      const auto &item = outputs_array[i];
      if (!item.contains("name")) {
        throw std::runtime_error("Output " + std::to_string(i) +
                                 " missing 'name' field");
      }

      std::string name = item.at("name").get<std::string>();
      if (name.empty()) {
        throw std::runtime_error("Output " + std::to_string(i) +
                                 " has empty name");
      }

      output_names.emplace_back(std::move(name));
    }

    // Parse model path
    if (!config.contains("model")) {
      throw std::runtime_error("Missing 'model' field in config");
    }

    const auto &graph_path = config.at("model");
    if (!graph_path.is_string()) {
      throw std::runtime_error("'model' must be a string");
    }

    std::string relative_path = graph_path.get<std::string>();
    if (relative_path.empty()) {
      throw std::runtime_error("'model' path cannot be empty");
    }

    // Construct full model path
    std::filesystem::path full_path =
        std::filesystem::path(workdir) / relative_path;
    model_path = full_path.string();

    // Validate model file exists
    if (!std::filesystem::exists(model_path)) {
      throw std::runtime_error("Model file not found: " + model_path);
    }

    if (!std::filesystem::is_regular_file(model_path)) {
      throw std::runtime_error("Model path is not a regular file: " +
                               model_path);
    }

    LOG(INFO) << "Configuration parsed: model=" << model_path
              << ", inputs=" << input_names.size()
              << ", outputs=" << output_names.size();

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

// ============================================================================
// Session Initialization
// ============================================================================

void CPUGraph::initialize_session(const std::string &model_path,
                                  int32_t threads) {
  // Validate model path
  if (model_path.empty()) {
    LOG(ERROR) << "Empty model path";
    throw std::invalid_argument("Model path cannot be empty");
  }

  // Validate thread count
  if (threads < 0) {
    LOG(WARNING) << "Negative thread count " << threads << ", using 0 (auto)";
    threads = 0;
  }

  if (threads > kMaxIntraOpThreads) {
    LOG(WARNING) << "Thread count " << threads << " exceeds maximum "
                 << kMaxIntraOpThreads << ", clamping";
    threads = kMaxIntraOpThreads;
  }

  try {
    Ort::SessionOptions session_options;

    // Register custom operators (sparse embedding lookup)
    static Ort::CustomOpDomain domain(kCustomOpDomain);
    static longmen::SparseEmbeddingLookupOp sparse_embedding_op;
    domain.Add(&sparse_embedding_op);
    session_options.Add(domain);

    // Configure threading for optimal performance
    const int num_cpus = static_cast<int>(std::thread::hardware_concurrency());
    if (num_cpus == 0) {
      LOG(WARNING) << "Could not detect CPU count, using threads=" << threads;
    }

    if (threads == 0) {
      // Auto-detect: use all available CPUs
      threads = (num_cpus > 0) ? num_cpus : 1;
      LOG(INFO) << "Auto-detected " << threads << " CPUs";
    } else if (num_cpus > 0) {
      // Clamp to available CPUs
      threads = std::min(threads, num_cpus);
    }

    // Set intra-op parallelism (within a single op)
    session_options.SetIntraOpNumThreads(threads);

    // Set inter-op parallelism (between independent ops)
    session_options.SetInterOpNumThreads(1); // Usually 1 is optimal

    // Enable all graph optimizations
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Enable memory pattern optimization
    session_options.EnableMemPattern();

    // Enable CPU memory arena for better performance
    session_options.EnableCpuMemArena();

    // Create ONNX Runtime session
    session_ = std::make_shared<Ort::Session>(
        OnnxRuntimeEnvSingleton::get(), model_path.c_str(), session_options);

    if (!session_) {
      throw std::runtime_error("Session creation returned null");
    }

    LOG(INFO) << "Model loaded successfully: " << model_path
              << " (threads=" << threads << ")";

  } catch (const Ort::Exception &e) {
    LOG(ERROR) << "Failed to load model from '" << model_path
               << "': " << e.what();
    throw std::runtime_error("Failed to load model from '" + model_path +
                             "': " + std::string(e.what()));
  } catch (const std::exception &e) {
    LOG(ERROR) << "Error initializing session: " << e.what();
    throw;
  }
}

// ============================================================================
// Input Node Initialization
// ============================================================================

void CPUGraph::initialize_input_nodes(
    const std::vector<std::string> &input_names) {
  if (input_names.empty()) {
    LOG(ERROR) << "Empty input names list";
    throw std::invalid_argument("Input names list cannot be empty");
  }

  if (!session_) {
    LOG(ERROR) << "Session is null";
    throw std::runtime_error("Session not initialized");
  }

  try {
    const size_t num_inputs = session_->GetInputCount();

    // Validate input count matches config
    if (num_inputs != input_names.size()) {
      LOG(ERROR) << "Input count mismatch: model has " << num_inputs
                 << " inputs, config specifies " << input_names.size();
      throw std::invalid_argument(
          "Input count mismatch: model has " + std::to_string(num_inputs) +
          " inputs, config specifies " + std::to_string(input_names.size()));
    }

    // Pre-allocate storage
    input_node_names_.resize(num_inputs);
    input_node_dims_.resize(num_inputs);
    input_widths_.resize(num_inputs);
    input_node_types_.resize(num_inputs);

    // Build name-to-index mapping from config order
    std::unordered_map<std::string, size_t> name_to_index;
    name_to_index.reserve(input_names.size());
    for (size_t i = 0; i < input_names.size(); ++i) {
      if (input_names[i].empty()) {
        throw std::invalid_argument("Input name at index " + std::to_string(i) +
                                    " is empty");
      }
      name_to_index[input_names[i]] = i;
    }

    Ort::AllocatorWithDefaultOptions allocator;

    // Extract metadata for each input node
    for (size_t i = 0; i < num_inputs; ++i) {
      // Get node name from model
      auto name_ptr = session_->GetInputNameAllocated(i, allocator);
      if (!name_ptr) {
        throw std::runtime_error("Failed to get input name for index " +
                                 std::to_string(i));
      }

      std::string node_name(name_ptr.get());
      if (node_name.empty()) {
        throw std::runtime_error("Empty input name at index " +
                                 std::to_string(i));
      }

      // Find corresponding index in config
      auto it = name_to_index.find(node_name);
      if (it == name_to_index.end()) {
        LOG(ERROR) << "Input node '" << node_name
                   << "' from model not found in config";
        throw std::invalid_argument("Input node '" + node_name +
                                    "' from model not found in config");
      }

      const size_t config_index = it->second;

      // Get type and shape information
      Ort::TypeInfo type_info = session_->GetInputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      auto element_type = tensor_info.GetElementType();

      // Validate supported types
      if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
          element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        LOG(ERROR) << "Input '" << node_name << "' has unsupported type: "
                   << static_cast<int>(element_type);
        throw std::invalid_argument(
            "Input '" + node_name + "' has unsupported type: " +
            std::to_string(static_cast<int>(element_type)));
      }

      // Get shape
      auto dims = tensor_info.GetShape();
      if (dims.empty()) {
        throw std::invalid_argument("Input '" + node_name +
                                    "' has empty dimensions");
      }

      // Validate first dimension is dynamic (-1)
      if (dims[0] != -1) {
        LOG(WARNING) << "Input '" << node_name
                     << "' first dimension is not dynamic: " << dims[0];
      }

      // Store node information in config order
      input_node_names_[config_index] = std::move(node_name);
      input_node_types_[config_index] = element_type;
      input_node_dims_[config_index] = dims;

      // Calculate feature width (product of all dims except batch)
      input_widths_[config_index] = calculate_feature_width(dims);

      LOG(INFO) << "Input[" << config_index
                << "]: " << input_node_names_[config_index] << ", shape=["
                << dims[0];
      for (size_t d = 1; d < dims.size(); ++d) {
        LOG(INFO) << ", " << dims[d];
      }
      LOG(INFO) << "], width=" << input_widths_[config_index]
                << ", type=" << static_cast<int>(element_type);
    }

  } catch (const Ort::Exception &e) {
    LOG(ERROR) << "ONNX Runtime error initializing input nodes: " << e.what();
    throw std::runtime_error("Failed to initialize input nodes: " +
                             std::string(e.what()));
  } catch (const std::exception &e) {
    LOG(ERROR) << "Error initializing input nodes: " << e.what();
    throw;
  }
}

// ============================================================================
// Output Node Initialization
// ============================================================================

void CPUGraph::initialize_output_nodes(
    const std::vector<std::string> &output_names) {
  if (output_names.empty()) {
    LOG(ERROR) << "Empty output names list";
    throw std::invalid_argument("Output names list cannot be empty");
  }

  if (!session_) {
    LOG(ERROR) << "Session is null";
    throw std::runtime_error("Session not initialized");
  }

  try {
    const size_t num_outputs = session_->GetOutputCount();

    // Validate output count matches config
    if (num_outputs != output_names.size()) {
      LOG(ERROR) << "Output count mismatch: model has " << num_outputs
                 << " outputs, config specifies " << output_names.size();
      throw std::invalid_argument(
          "Output count mismatch: model has " + std::to_string(num_outputs) +
          " outputs, config specifies " + std::to_string(output_names.size()));
    }

    // Pre-allocate storage
    output_node_names_.resize(num_outputs);
    output_node_dims_.resize(num_outputs);
    output_widths_.resize(num_outputs);

    // Build name-to-index mapping from config order
    std::unordered_map<std::string, size_t> name_to_index;
    name_to_index.reserve(output_names.size());
    for (size_t i = 0; i < output_names.size(); ++i) {
      if (output_names[i].empty()) {
        throw std::invalid_argument("Output name at index " +
                                    std::to_string(i) + " is empty");
      }
      name_to_index[output_names[i]] = i;
    }

    Ort::AllocatorWithDefaultOptions allocator;

    // Extract metadata for each output node
    for (size_t i = 0; i < num_outputs; ++i) {
      // Get node name from model
      auto name_ptr = session_->GetOutputNameAllocated(i, allocator);
      if (!name_ptr) {
        throw std::runtime_error("Failed to get output name for index " +
                                 std::to_string(i));
      }

      std::string node_name(name_ptr.get());
      if (node_name.empty()) {
        throw std::runtime_error("Empty output name at index " +
                                 std::to_string(i));
      }

      // Find corresponding index in config
      auto it = name_to_index.find(node_name);
      if (it == name_to_index.end()) {
        LOG(ERROR) << "Output node '" << node_name
                   << "' from model not found in config";
        throw std::invalid_argument("Output node '" + node_name +
                                    "' from model not found in config");
      }

      const size_t config_index = it->second;

      // Get type and shape information
      Ort::TypeInfo type_info = session_->GetOutputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      auto element_type = tensor_info.GetElementType();

      // Validate output type (must be float32)
      if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        LOG(ERROR) << "Output '" << node_name << "' must be float32 type, got: "
                   << static_cast<int>(element_type);
        throw std::invalid_argument(
            "Output '" + node_name + "' must be float32 type, got: " +
            std::to_string(static_cast<int>(element_type)));
      }

      // Get shape
      auto dims = tensor_info.GetShape();
      if (dims.empty()) {
        throw std::invalid_argument("Output '" + node_name +
                                    "' has empty dimensions");
      }

      // Validate first dimension is dynamic (-1)
      if (dims[0] != -1) {
        LOG(WARNING) << "Output '" << node_name
                     << "' first dimension is not dynamic: " << dims[0];
      }

      // Store node information in config order
      output_node_names_[config_index] = std::move(node_name);
      output_node_dims_[config_index] = dims;

      // Calculate feature width (product of all dims except batch)
      output_widths_[config_index] = calculate_feature_width(dims);

      LOG(INFO) << "Output[" << config_index
                << "]: " << output_node_names_[config_index] << ", shape=["
                << dims[0];
      for (size_t d = 1; d < dims.size(); ++d) {
        LOG(INFO) << ", " << dims[d];
      }
      LOG(INFO) << "], width=" << output_widths_[config_index];
    }

  } catch (const Ort::Exception &e) {
    LOG(ERROR) << "ONNX Runtime error initializing output nodes: " << e.what();
    throw std::runtime_error("Failed to initialize output nodes: " +
                             std::string(e.what()));
  } catch (const std::exception &e) {
    LOG(ERROR) << "Error initializing output nodes: " << e.what();
    throw;
  }
}

// ============================================================================
// Helper Methods
// ============================================================================

void CPUGraph::validate_dynamic_shape(const std::vector<int64_t> &shape,
                                      const std::string &node_name) const {
  if (shape.empty()) {
    throw std::runtime_error("Node '" + node_name + "' has empty shape");
  }

  if (shape[0] != -1) {
    throw std::runtime_error("Node '" + node_name +
                             "' first dimension must be dynamic (-1), got: " +
                             std::to_string(shape[0]));
  }
}

size_t
CPUGraph::calculate_feature_width(const std::vector<int64_t> &shape) const {
  if (shape.empty()) {
    throw std::runtime_error("Cannot calculate width from empty shape");
  }

  // Skip first dimension (batch), multiply the rest
  size_t width = 1;
  for (size_t i = 1; i < shape.size(); ++i) {
    if (shape[i] <= 0) {
      throw std::runtime_error("Invalid dimension at index " +
                               std::to_string(i) + ": " +
                               std::to_string(shape[i]));
    }

    // Check for overflow
    if (width > SIZE_MAX / static_cast<size_t>(shape[i])) {
      throw std::runtime_error("Feature width overflow at dimension " +
                               std::to_string(i));
    }

    width *= static_cast<size_t>(shape[i]);
  }

  return width;
}

} // namespace longmen
