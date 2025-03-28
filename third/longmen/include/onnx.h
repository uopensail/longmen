#ifndef RANKER_ONNX_H_
#define RANKER_ONNX_H_

#include "feature.h"
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace longmen {

// Error code constants
constexpr int32_t SUCCESS = 0;          ///< Successful operation
constexpr int32_t INFERENCE_ERROR = -1; ///< Inference-related error
constexpr int32_t SYSTEM_ERROR = -2;    ///< System or runtime error

using Pool = std::unordered_map<std::string, std::shared_ptr<minia::Features>>;

/**
 * @struct GoString
 * @brief Mirrors Go's string type layout for C++/Go interoperability
 *
 * Matches Go's runtime.stringHeader memory layout to ensure safe data passing
 * between C++ and Go environments.
 */
struct GoString {
  const char *ptr; ///< Pointer to read-only character data (non-owning)
  size_t len;      ///< Length of string in bytes

  /// @brief Constructs an empty GoString
  GoString() noexcept : ptr(nullptr), len(0) {}

  /**
   * @brief Constructs from std::string
   * @param str Source string to reference (non-owning)
   */
  explicit GoString(const std::string &str) noexcept
      : ptr(str.data()), len(str.size()) {}

  /// @brief Assigns from std::string (shallow copy)
  GoString &operator=(const std::string &str) noexcept {
    ptr = str.data();
    len = str.size();
    return *this;
  }

  ~GoString() = default;
};

/**
 * @struct GoSlice
 * @brief Simulates Go's slice type for type-safe data passing
 * @tparam T Element type of the slice
 *
 * Provides RAII-style memory management for slice data.
 */
template <typename T> struct GoSlice {
  T *ptr;     ///< Pointer to heap-allocated array (owned)
  size_t len; ///< Current element count
  size_t cap; ///< Allocated capacity

  /// @brief Creates an empty slice
  GoSlice() noexcept : ptr(nullptr), len(0), cap(0) {}

  /**
   * @brief Creates slice with pre-allocated capacity
   * @param size Initial capacity
   */
  explicit GoSlice(size_t size) : ptr(new T[size]{0}), len(size), cap(size) {}

  /// @brief Move constructor
  GoSlice(GoSlice &&other) noexcept
      : ptr(other.ptr), len(other.len), cap(other.cap) {
    other.ptr = nullptr;
    other.len = 0;
    other.cap = 0;
  }

  /// @brief Move assignment operator
  GoSlice &operator=(GoSlice &&other) noexcept {
    if (this != &other) {
      delete[] ptr;
      ptr = other.ptr;
      len = other.len;
      cap = other.cap;
      other.ptr = nullptr;
      other.len = 0;
      other.cap = 0;
    }
    return *this;
  }

  /// @brief Assignment from std::vector
  GoSlice &operator=(const std::vector<T> &vec) {
    if (cap < vec.size()) {
      delete[] ptr;
      ptr = new T[vec.size()];
      cap = vec.size();
    }
    len = vec.size();
    std::copy(vec.begin(), vec.end(), ptr);
    return *this;
  }

  /// @brief Element access operator
  T &operator[](int32_t index) noexcept { return ptr[index]; }

  /// @brief Const element access operator
  const T &operator[](int32_t index) const noexcept { return ptr[index]; }

  /**
   * @brief Constructs from std::vector
   * @param vec Source vector to copy
   */
  explicit GoSlice(const std::vector<T> &vec)
      : ptr(new T[vec.size()]), len(vec.size()), cap(vec.size()) {
    std::copy(vec.begin(), vec.end(), ptr);
  }

  /// @brief Destructor (frees owned memory)
  ~GoSlice() { delete[] ptr; }
};

/**
 * @struct Output
 * @brief Container for inference output data
 */
struct Output {
  GoString name;        ///< Output node name
  GoSlice<int64_t> dim; ///< Output dimensions
  GoSlice<float> value; ///< Output values
};

using OutputSlice = GoSlice<Output>;

/**
 * @struct Tensor
 * @brief Base tensor structure for handling batched data
 *
 * Manages memory allocation and basic tensor properties with RAII semantics.
 */
struct Tensor {
  int32_t batch; ///< Number of batches
  int32_t width; ///< Elements per batch
  size_t stride; ///< Stride between elements in bytes
  void *data;    ///< Managed data buffer

  /**
   * @brief Constructs a Tensor
   * @param batch Number of batches
   * @param width Elements per batch
   * @param stride Element size in bytes
   * @throws std::bad_alloc On memory allocation failure
   */
  Tensor(int32_t batch, int32_t width, size_t stride)
      : batch(batch), width(width), stride(stride),
        data(calloc(batch * width, stride)) {
    if (!data) {
      throw std::bad_alloc();
    }
  }

  /// @brief Virtual destructor for proper polymorphic cleanup
  virtual ~Tensor() { free(data); }
};

/**
 * @struct Input
 * @brief Interface for input tensor operations
 */
struct Input : Tensor {
  using Tensor::Tensor;

  /**
   * @brief Inserts feature data at specified batch index
   * @param index Batch position (0-based)
   * @param ptr Feature pointer containing source data
   * @throws std::out_of_range For invalid indices
   * @throws std::runtime_error For type mismatches
   */
  virtual void put(int32_t index, const minia::FeaturePtr &ptr) = 0;
};

/**
 * @struct TypedInput
 * @brief Typed input tensor implementation
 * @tparam T Native data type for elements
 */
template <typename T> struct TypedInput : Input {
  /**
   * @brief Constructs a TypedInput
   * @param batch Number of batches
   * @param width Elements per batch
   */
  TypedInput(int32_t batch, int64_t width) : Input(batch, width, sizeof(T)) {}

  void put(int32_t index, const minia::FeaturePtr &ptr) override {
    if (index < 0 || index >= batch) {
      throw std::out_of_range("Invalid batch index: " + std::to_string(index));
    }

    T *dest = static_cast<T *>(data) + index * width;

    if (ptr->type == minia::TypeID<T>::value) {
      *dest = ptr->get<T>();
    } else {
      const auto &vec = ptr->get<std::vector<T>>();
      const size_t copy_size = std::min(static_cast<size_t>(width), vec.size());
      std::copy_n(vec.data(), copy_size, dest);
    }
  }
};

/// @brief Dictionary mapping input node names to tensor containers
using InputDict = std::map<std::string, std::shared_ptr<Input>>;

/**
 * @struct KeyMapper
 * @brief Handles feature key mapping operations
 */
struct KeyMapper {
  /**
   * @struct IdentityHash
   * @brief Hash function for int64_t keys
   */
  struct IdentityHash {
    size_t operator()(int64_t key) const noexcept {
      return static_cast<size_t>(key);
    }
  };

  std::unordered_map<int64_t, int64_t, IdentityHash> table; ///< Mapping table

  /**
   * @brief Constructs KeyMapper from file
   * @param file_path Path to mapping file
   */
  explicit KeyMapper(const std::string &file_path);

  ~KeyMapper() = default;

  /**
   * @brief Applies mapping to input tensor
   * @param input Target input tensor
   */
  void operator()(TypedInput<int64_t> &input);
};

/**
 * @class OnnxRuntimeEnv
 * @brief Singleton manager for ONNX runtime environment
 *
 * Ensures proper initialization and cleanup of ONNX runtime resources.
 */
class OnnxRuntimeEnv {
public:
  OnnxRuntimeEnv(const OnnxRuntimeEnv &) = delete;
  OnnxRuntimeEnv &operator=(const OnnxRuntimeEnv &) = delete;

  /**
   * @brief Gets singleton instance
   * @return Ort::Env& Reference to runtime environment
   */
  static Ort::Env &get() {
    static Ort::Env instance{ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime"};
    return instance;
  }

private:
  OnnxRuntimeEnv() = default;
};

// Forward declaration
struct OnnxModel;

/**
 * @class OnnxGraph
 * @brief Manages ONNX model graph and inference execution
 */
class OnnxGraph {
public:
  /**
   * @brief Constructs OnnxGraph from model file
   * @param model_path Path to ONNX model file
   * @param threads Number of intra-op threads (0 for default)
   * @throws std::runtime_error If model loading fails
   */
  explicit OnnxGraph(const std::string &model_path, int threads = 0);

  OnnxGraph(const OnnxGraph &) = delete;
  OnnxGraph &operator=(const OnnxGraph &) = delete;
  ~OnnxGraph() = default;

  /**
   * @brief Creates input containers for specified batch size
   * @param batch Number of batches
   * @return InputDict Configured input containers
   */
  InputDict create_inputs(int32_t batch) const;

  /**
   * @brief Creates output containers for specified batch size
   * @param batch Number of batches
   * @return OutputSlice* Configured output containers
   */
  OutputSlice *create_outputs(int32_t batch) const;

  /**
   * @brief Executes model inference
   * @param inputs Prepared input data
   * @param outputs Output containers to fill
   * @return int32_t Status code (0 = success)
   */
  int32_t infer(const InputDict &inputs, const OutputSlice &outputs) const;

private:
  std::shared_ptr<Ort::Session> session_;        ///< ONNX runtime session
  std::vector<std::string> input_names_;         ///< Model input node names
  std::vector<std::vector<int64_t>> input_dims_; ///< Input tensor dimensions
  std::vector<ONNXTensorElementDataType> input_types_; ///< Input data types
  std::vector<std::string> output_names_; ///< Model output node names
  Ort::MemoryInfo memory_info_{
      ///< CPU memory allocator configuration
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)};

  friend struct OnnxModel;
};

/**
 * @struct OnnxModel
 * @brief Complete ONNX model representation with feature processing
 */
struct OnnxModel {
  std::map<std::string, std::shared_ptr<KeyMapper>>
      mapper;                         ///< Feature mapping processors
  std::shared_ptr<minia::Minia> user; ///< User feature extractor
  std::shared_ptr<minia::Minia> item; ///< Item feature extractor
  std::shared_ptr<OnnxGraph> graph;   ///< Inference graph processor

  /**
   * @brief Constructs ONNX model from directory
   * @param model_dir Directory containing model files
   */
  explicit OnnxModel(const std::string &model_dir);

  /**
   * @brief Executes model inference for a batch
   * @param batch_size Number of items in batch
   * @param pool Feature pool containing input data
   * @param user_features User feature identifier
   * @param items Array of item identifiers
   * @return OutputSlice* Pointer to output results
   */
  OutputSlice *infer(int32_t batch_size, Pool *pool, const char *user_features,
                     const char **items);
};

} // namespace longmen

#endif // RANKER_ONNX_H_
