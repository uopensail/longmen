#include "onnx.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace longmen {

/**
 * @brief Constructs a KeyMapper from binary mapping file
 * @param file_path Path to binary mapping file containing int64 key-value pairs
 * @throws std::runtime_error If file operations fail
 */
KeyMapper::KeyMapper(const std::string &file_path) {
  std::ifstream file(file_path, std::ios::binary | std::ios::in);

  if (!file.is_open()) {
    throw std::runtime_error("Failed to open mapping file: " + file_path);
  }

  // Get file size using C++17 filesystem for better error messages
  const auto file_size = std::filesystem::file_size(file_path);
  if (file_size % (2 * sizeof(int64_t)) != 0) {
    throw std::runtime_error("Invalid mapping file format: " + file_path);
  }

  const size_t pair_count = file_size / (2 * sizeof(int64_t));
  table.reserve(pair_count);

  int64_t buffer[2];
  for (size_t i = 0; i < pair_count; ++i) {
    if (!file.read(reinterpret_cast<char *>(buffer), 2 * sizeof(int64_t))) {
      throw std::runtime_error("Read error in mapping file: " + file_path);
    }
    table[buffer[0]] = buffer[1];
  }
}

/**
 * @brief Applies key mapping to input tensor data
 * @param input Typed input tensor containing keys to map
 * @throws std::invalid_argument For null input data
 */
void KeyMapper::operator()(TypedInput<int64_t> &input) {
  if (!input.data) {
    throw std::invalid_argument("Null input data in KeyMapper operator");
  }

  int64_t *keys = static_cast<int64_t *>(input.data);
  const int32_t element_count = input.batch * input.width;

  // Parallel mapping using OpenMP
  // TODO: #pragma omp parallel for
  for (int32_t i = 0; i < element_count; ++i) {
    const auto it = table.find(keys[i]);
    keys[i] = (it != table.end()) ? it->second : 0;
  }
}

/**
 * @brief Constructs ONNX graph from model file
 * @param model_path Path to ONNX model file
 * @param threads Number of intra/inter-op threads (0 for default)
 * @throws std::runtime_error For model loading failures
 */
OnnxGraph::OnnxGraph(const std::string &model_path, int threads) {
  Ort::SessionOptions session_options;

  // Thread configuration with hardware awareness
  if (threads > 0) {
    const unsigned hw_concurrency = std::thread::hardware_concurrency();
    const int max_threads = (hw_concurrency > 0) ? hw_concurrency : 1;
    const int safe_threads = std::clamp(threads, 1, max_threads);

    session_options.SetIntraOpNumThreads(safe_threads);
    session_options.SetInterOpNumThreads(safe_threads);
  }

  // Optimization configuration
  session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
  session_options.SetExecutionMode(ORT_SEQUENTIAL);

  try {
    session_ = std::make_shared<Ort::Session>(
        OnnxRuntimeEnv::get(), std::filesystem::canonical(model_path).c_str(),
        session_options);
  } catch (const Ort::Exception &e) {
    throw std::runtime_error("Model initialization failed: " +
                             std::string(e.what()));
  }

  Ort::AllocatorWithDefaultOptions allocator;

  // Process input metadata
  const size_t num_inputs = session_->GetInputCount();
  input_names_.resize(num_inputs);
  input_dims_.resize(num_inputs);
  input_types_.resize(num_inputs);

  for (size_t i = 0; i < num_inputs; ++i) {
    auto name_ptr = session_->GetInputNameAllocated(i, allocator);
    input_names_[i] = std::move(name_ptr);

    const auto type_info = session_->GetInputTypeInfo(i);
    const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    input_types_[i] = tensor_info.GetElementType();
    input_dims_[i] = tensor_info.GetShape();

    // Validate supported types
    if (input_types_[i] != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
        input_types_[i] != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::invalid_argument("Unsupported input tensor type: " +
                                  std::to_string(input_types_[i]));
    }
  }

  // Process output metadata
  const size_t num_outputs = session_->GetOutputCount();
  output_names_.resize(num_outputs);

  for (size_t i = 0; i < num_outputs; ++i) {
    auto name_ptr = session_->GetOutputNameAllocated(i, allocator);
    output_names_[i] = std::move(name_ptr);

    const auto type_info = session_->GetOutputTypeInfo(i);
    const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::invalid_argument("Non-float output tensors not supported");
    }
  }
}

/**
 * @brief Creates input containers for specified batch size
 * @param batch Number of batches to prepare
 * @return InputDict Configured input containers
 * @throws std::logic_error For unsupported tensor types
 */
InputDict OnnxGraph::create_inputs(int32_t batch) const {
  InputDict inputs;
  inputs.reserve(input_names_.size());

  for (size_t i = 0; i < input_names_.size(); ++i) {
    // Calculate elements per batch excluding batch dimension
    const auto &dims = input_dims_[i];
    const int64_t elements =
        std::accumulate(dims.begin() + 1, dims.end(), 1LL, std::multiplies<>());

    switch (input_types_[i]) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      inputs.emplace(input_names_[i],
                     std::make_shared<TypedInput<int64_t>>(batch, elements));
      break;

    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      inputs.emplace(input_names_[i],
                     std::make_shared<TypedInput<float>>(batch, elements));
      break;

    default:
      throw std::logic_error("Unhandled tensor type in create_inputs");
    }
  }
  return inputs;
}

/**
 * @brief Creates output containers for inference results
 * @param batch Number of batches in output
 * @return OutputSlice* Heap-allocated output containers
 * @throws std::bad_alloc For memory allocation failures
 */
OutputSlice *OnnxGraph::create_outputs(int32_t batch) const {
  auto outputs = std::make_unique<OutputSlice>(output_names_.size());

  for (size_t i = 0; i < output_names_.size(); ++i) {
    auto &output = (*outputs)[i];
    output.name = output_names_[i];

    // Handle dynamic batch dimension
    auto dims =
        session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    if (!dims.empty() && dims[0] == -1) {
      dims[0] = batch;
    }

    output.dim = GoSlice<int64_t>(dims);

    const int64_t elements =
        std::accumulate(dims.begin(), dims.end(), 1LL, std::multiplies<>());
    output.value = GoSlice<float>(elements);
  }

  return outputs.release();
}

/**
 * @brief Executes model inference
 * @param inputs Prepared input data
 * @param outputs Output containers to populate
 * @return int32_t Status code (0 = success)
 */
int32_t OnnxGraph::infer(const InputDict &inputs,
                         const OutputSlice &outputs) const noexcept {
  try {
    // Input validation
    if (inputs.size() != input_names_.size()) {
      throw std::invalid_argument("Input count mismatch. Expected: " +
                                  std::to_string(input_names_.size()) +
                                  ", Got: " + std::to_string(inputs.size()));
    }

    if (outputs.len != output_names_.size()) {
      throw std::invalid_argument("Output count mismatch. Expected: " +
                                  std::to_string(output_names_.size()) +
                                  ", Got: " + std::to_string(outputs.len));
    }

    // Prepare tensors
    std::vector<Ort::Value> input_tensors;
    std::vector<const char *> input_names;
    std::vector<const char *> output_names;

    input_tensors.reserve(input_names_.size());
    input_names.reserve(input_names_.size());
    output_names.reserve(output_names_.size());

    // Configure inputs
    for (size_t i = 0; i < input_names_.size(); ++i) {
      const auto &name = input_names_[i];
      const auto &input = inputs.at(name);

      auto dims = input_dims_[i];
      if (!dims.empty() && dims[0] == -1) {
        dims[0] = input->batch;
      }

      input_names.push_back(name.c_str());

      input_tensors.emplace_back(
          Ort::Value::CreateTensor(memory_info_, input->data,
                                   input->batch * input->width * input->stride,
                                   dims.data(), dims.size(), input_types_[i]));
    }

    // Configure outputs
    for (const auto &name : output_names_) {
      output_names.push_back(name.c_str());
    }

    // Execute inference
    Ort::RunOptions run_options;
    auto output_tensors = session_->Run(
        run_options, input_names.data(), input_tensors.data(),
        input_tensors.size(), output_names.data(), output_names.size());

    // Copy outputs
    for (size_t i = 0; i < output_tensors.size(); ++i) {
      const float *src = output_tensors[i].GetTensorData<float>();
      const size_t bytes = outputs[i].value.len * sizeof(float);
      std::memcpy(outputs[i].value.ptr, src, bytes);
    }

    return SUCCESS;

  } catch (const std::exception &e) {
    std::cerr << "Inference failed: " << e.what() << '\n';
    return (dynamic_cast<const Ort::Exception *>(&e)) ? INFERENCE_ERROR
                                                      : SYSTEM_ERROR;
  }
}

/**
 * @brief Constructs ONNX model from workspace directory
 * @param workdir Directory containing model/config/mapping files
 * @throws std::runtime_error For configuration errors
 */
OnnxModel::OnnxModel(const std::string &workdir) {
  const std::string config_path =
      std::filesystem::path(workdir) / "config.toml";

  try {
    // Parse configuration
    auto config = toml::parse_file(config_path);

    user = std::make_shared<minia::Minia>(*config["user"].as_table());
    item = std::make_shared<minia::Minia>(*config["item"].as_table());

    // Initialize graph
    graph = std::make_shared<OnnxGraph>(std::filesystem::path(workdir) /
                                        "model.onnx");

    // Load mappers
    mapper.reserve(graph->input_names_.size());
    for (const auto &input_name : graph->input_names_) {
      const auto mapper_path =
          std::filesystem::path(workdir) / (input_name + ".bin");

      if (std::filesystem::exists(mapper_path)) {
        mapper[input_name] = std::make_shared<KeyMapper>(mapper_path);
      } else {
        mapper[input_name] = nullptr;
      }
    }

  } catch (const std::filesystem::filesystem_error &e) {
    throw std::runtime_error("File system error: " + std::string(e.what()));
  } catch (const toml::parse_error &e) {
    throw std::runtime_error("Config parse error at " +
                             std::string(e.source().path()) + ": " +
                             e.description());
  }
}

/**
 * @brief Executes model inference for a batch of data
 * @param batch Number of items in the batch
 * @param pool Feature pool containing pre-processed item features
 * @param user_features User feature identifier
 * @param items Array of item identifiers to score
 * @return OutputSlice* Pointer to inference results (nullptr on failure)
 * @throws std::invalid_argument For invalid input dimensions or missing
 * features
 */
OutputSlice *OnnxModel::infer(int32_t batch, Pool *pool,
                              const char *user_features, const char **items) {
  // Validate input parameters
  if (batch <= 0)
    throw std::invalid_argument("Invalid batch size: " + std::to_string(batch));
  if (!pool || !items)
    throw std::invalid_argument("Null pointer in input parameters");

  std::unique_ptr<OutputSlice> outputs(graph->create_outputs(batch));
  InputDict inputs = graph->create_inputs(batch);

  try {
    // Process user features
    minia::Features user_feas(user_features);
    user->call(user_feas);

    // Batch-process user features
    for (const auto &name : user->features()) {
      const minia::FeaturePtr fea = user_feas.get(name);
      auto &input = inputs.at(name); // Throws if missing
      for (int32_t b = 0; b < batch; ++b) {
        input->put(b, fea);
      }
    }

    // Process item features in parallel
    // TODO: #pragma omp parallel for
    for (int32_t b = 0; b < batch; ++b) {
      const char *item_id = items[b];
      auto it = pool->find(item_id);
      if (it == pool->end()) {
        continue;
      }

      for (const auto &[key, value] : it->second->values) {
        inputs.at(key)->put(b, value);
      }
    }

    for (auto &[key, value] : inputs) {
      if (mapper[key]) {
        (*mapper[key])(*std::dynamic_pointer_cast<TypedInput<int64_t>>(value));
      }
    }

    // Execute inference
    if (graph->infer(inputs, *outputs) == SUCCESS) {
      return outputs.release();
    }
  } catch (const std::exception &e) {
    std::cerr << "Inference failed: " << e.what() << '\n';
  }

  return nullptr;
}

} // namespace longmen