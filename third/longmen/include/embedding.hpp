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

#ifndef LONGMEN_EMBEDDING_HPP_
#define LONGMEN_EMBEDDING_HPP_

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"
#include "fp16_to_fp32.h"

namespace longmen {

/// Maximum number of embedding tables supported
constexpr int kMaxEmbeddingNum = 256;

/// Number of shards for parallel processing (must be power of 2)
constexpr size_t kShardCount = 32;

/// Bitmask for efficient shard index calculation (kShardCount - 1)
constexpr size_t kShardMask = kShardCount - 1;

/// Maximum allowed embedding dimension to prevent memory overflow
constexpr int32_t kMaxEmbeddingDim = 512;

/// Maximum allowed embedding count per shard to prevent memory overflow
constexpr int64_t kMaxShardCount = 100000000; // 100M

/**
 * @class Shard
 * @brief Thread-safe container for FP16-encoded embedding vectors with binary
 * search lookup
 *
 * Stores key-value pairs where keys are sorted 64-bit integers and values are
 * FP16-encoded embedding vectors. Provides O(log n) lookup via binary search.
 *
 * @par File Format
 * Binary format with the following structure:
 * - Header:
 *   - int64_t count: Number of embeddings in this shard
 *   - int32_t dimension: Embedding vector dimension
 * - Records (repeated count times):
 *   - int64_t key: Embedding identifier (sorted ascending)
 *   - uint16_t[dimension]: FP16-encoded embedding values
 *
 * @par Thread Safety
 * Safe for concurrent reads after construction. Not thread-safe during
 * construction or modification.
 *
 * @par Memory Layout
 * Keys and values are stored in separate contiguous arrays for cache
 * efficiency during binary search.
 */
class Shard {
public:
  /**
   * @brief Loads shard data from binary file
   *
   * Reads and validates shard file format, allocates memory, and loads
   * all embedding data. Validates dimension consistency and key ordering.
   *
   * @param dim Expected embedding dimension (must match file header)
   * @param file_path Path to shard binary file
   *
   * @throws std::invalid_argument If dim <= 0 or dim > kMaxEmbeddingDim
   * @throws std::runtime_error If file cannot be opened or read
   * @throws std::runtime_error If header validation fails
   * @throws std::runtime_error If dimension mismatch detected
   * @throws std::runtime_error If keys are not sorted
   * @throws std::bad_alloc If memory allocation fails
   *
   * @note File must exist and be readable
   * @note Keys must be sorted in ascending order
   */
  Shard(int32_t dim, const std::string &file_path) : count_(0), dim_(0) {
    // Validate input parameters
    if (dim <= 0) {
      LOG(ERROR) << "Invalid dimension: " << dim << " for shard " << file_path;
      throw std::invalid_argument("Dimension must be positive");
    }

    if (dim > kMaxEmbeddingDim) {
      LOG(ERROR) << "Dimension " << dim << " exceeds maximum "
                 << kMaxEmbeddingDim << " for shard " << file_path;
      throw std::invalid_argument("Dimension exceeds maximum allowed");
    }

    if (file_path.empty()) {
      LOG(ERROR) << "Shard file path is empty";
      throw std::invalid_argument("File path cannot be empty");
    }

    LOG(INFO) << "Loading shard from: " << file_path;

    try {
      // Open file with error checking
      std::ifstream file(file_path, std::ios::binary);
      if (!file.is_open() || file.fail()) {
        LOG(ERROR) << "Failed to open shard file: " << file_path;
        throw std::runtime_error("Failed to open shard file: " + file_path);
      }

      // Read and validate header
      read_and_validate_header(file, dim, file_path);

      // Handle empty shard
      if (count_ == 0) {
        LOG(INFO) << "Shard is empty: " << file_path;
        file.close();
        return;
      }

      // Allocate memory for keys and values
      allocate_memory(file_path);

      // Read all keys
      read_keys(file, file_path);

      // Read all values
      read_values(file, file_path);

      // Verify file is fully read
      verify_file_end(file, file_path);

      // Close file
      file.close();
      if (file.fail()) {
        LOG(WARNING) << "Error closing file: " << file_path;
      }

      LOG(INFO) << "Successfully loaded shard: " << file_path
                << ", count=" << count_ << ", dim=" << dim_;

    } catch (const std::exception &e) {
      LOG(ERROR) << "Failed to load shard " << file_path << ": " << e.what();
      // Clean up on error
      keys_.clear();
      values_.clear();
      count_ = 0;
      dim_ = 0;
      throw;
    }
  }

  /**
   * @brief Returns embedding dimension
   *
   * @return Embedding vector dimension
   */
  int32_t dimension() const noexcept { return dim_; }

  /**
   * @brief Returns number of embeddings in this shard
   *
   * @return Total embedding count
   */
  int64_t count() const noexcept { return count_; }

  /**
   * @brief Looks up embedding vector by key
   *
   * Performs binary search to find the embedding. If found, converts from
   * FP16 to FP32 and writes to output buffer. If not found, zero-fills
   * the output buffer.
   *
   * @param key Embedding identifier
   * @param[out] data Output buffer (must have size >= dimension)
   *
   * @return true if key exists, false otherwise
   *
   * @note Converts FP16 to FP32 on-the-fly
   * @note Output buffer is always written (either with data or zeros)
   * @note Thread-safe for concurrent calls
   *
   * @warning Caller must ensure data buffer is large enough
   */
  bool lookup(int64_t key, float *data) const {
    if (!data) {
      LOG(ERROR) << "Null output buffer for lookup of key " << key;
      return false;
    }

    // Binary search for key
    auto it = std::lower_bound(keys_.begin(), keys_.end(), key);

    if (it != keys_.end() && *it == key) {
      // Key found - convert FP16 to FP32
      const size_t index = std::distance(keys_.begin(), it);
      const uint16_t *src = &values_[dim_ * index];

      for (int32_t d = 0; d < dim_; ++d) {
        data[d] = fp16_to_fp32(src[d]);
      }
      return true;
    }

    // Key not found - zero fill
    std::memset(data, 0, sizeof(float) * dim_);
    return false;
  }

  /**
   * @brief Batch lookup of multiple embeddings
   *
   * Performs multiple lookups efficiently. Missing keys result in
   * zero-filled vectors.
   *
   * @param keys Array of embedding identifiers
   * @param num_keys Number of keys to lookup
   * @param[out] output Contiguous buffer (size >= num_keys × dimension)
   *
   * @return Number of keys found
   *
   * @note Thread-safe for concurrent calls
   * @warning Caller must ensure buffers are large enough
   */
  size_t batch_lookup(const int64_t *keys, size_t num_keys,
                      float *output) const {
    if (!keys || !output) {
      LOG(ERROR) << "Null buffer in batch_lookup";
      return 0;
    }

    if (num_keys == 0) {
      return 0;
    }

    size_t found = 0;
    for (size_t i = 0; i < num_keys; ++i) {
      if (lookup(keys[i], output + dim_ * i)) {
        ++found;
      }
    }
    return found;
  }

  /** @brief Copy constructor (deleted - non-copyable) */
  Shard(const Shard &) = delete;

  /** @brief Copy assignment (deleted - non-copyable) */
  Shard &operator=(const Shard &) = delete;

  /** @brief Move constructor (deleted - non-movable) */
  Shard(Shard &&) = delete;

  /** @brief Move assignment (deleted - non-movable) */
  Shard &operator=(Shard &&) = delete;

private:
  /**
   * @brief Read and validate shard file header
   *
   * @param file Input file stream
   * @param expected_dim Expected dimension
   * @param file_path File path for error messages
   * @throws std::runtime_error if header is invalid
   */
  void read_and_validate_header(std::ifstream &file, int32_t expected_dim,
                                const std::string &file_path) {
    // Read header
    file.read(reinterpret_cast<char *>(&count_), sizeof(int64_t));
    file.read(reinterpret_cast<char *>(&dim_), sizeof(int32_t));

    if (file.fail()) {
      LOG(ERROR) << "Failed to read header from: " << file_path;
      throw std::runtime_error("Failed to read shard header: " + file_path);
    }

    // Validate count
    if (count_ < 0) {
      LOG(ERROR) << "Invalid count " << count_ << " in shard " << file_path;
      throw std::runtime_error("Invalid embedding count in shard header");
    }

    if (count_ > kMaxShardCount) {
      LOG(ERROR) << "Count " << count_ << " exceeds maximum " << kMaxShardCount
                 << " in shard " << file_path;
      throw std::runtime_error("Embedding count exceeds maximum allowed");
    }

    // Validate dimension
    if (dim_ != expected_dim) {
      LOG(ERROR) << "Dimension mismatch: expected " << expected_dim << ", got "
                 << dim_ << " in shard " << file_path;
      throw std::runtime_error("Dimension mismatch in shard file");
    }

    LOG(INFO) << "Shard header: count=" << count_ << ", dim=" << dim_;
  }

  /**
   * @brief Allocate memory for keys and values
   *
   * @param file_path File path for error messages
   * @throws std::runtime_error if allocation fails
   */
  void allocate_memory(const std::string &file_path) {
    // Check for potential memory overflow
    const size_t values_size = static_cast<size_t>(count_) * dim_;

    if (count_ > 0 && values_size / count_ != static_cast<size_t>(dim_)) {
      LOG(ERROR) << "Memory size overflow: count=" << count_ << ", dim=" << dim_
                 << " in shard " << file_path;
      throw std::runtime_error("Memory allocation size overflow");
    }

    // Calculate total memory needed
    const size_t keys_bytes = count_ * sizeof(int64_t);
    const size_t values_bytes = values_size * sizeof(uint16_t);
    const size_t total_mb = (keys_bytes + values_bytes) / (1024 * 1024);

    LOG(INFO) << "Allocating memory: " << total_mb << " MB for " << count_
              << " embeddings";

    // Allocate memory
    try {
      keys_.resize(count_);
      values_.resize(values_size);
    } catch (const std::bad_alloc &e) {
      LOG(ERROR) << "Failed to allocate " << total_mb << " MB for " << count_
                 << " embeddings of dimension " << dim_ << " in shard "
                 << file_path;
      throw std::runtime_error("Memory allocation failed: " +
                               std::string(e.what()));
    }
  }

  /**
   * @brief Read all keys from file and validate ordering
   *
   * @param file Input file stream
   * @param file_path File path for error messages
   * @throws std::runtime_error if read fails or keys are not sorted
   */
  void read_keys(std::ifstream &file, const std::string &file_path) {
    // Read all keys at once for better performance
    file.read(reinterpret_cast<char *>(keys_.data()), sizeof(int64_t) * count_);

    if (file.fail()) {
      LOG(ERROR) << "Failed to read keys from: " << file_path;
      throw std::runtime_error("Failed to read keys from: " + file_path);
    }

    // Validate key ordering
    validate_key_ordering(file_path);

    LOG(INFO) << "Read " << count_ << " keys from shard";
  }

  /**
   * @brief Validate that keys are sorted in ascending order
   *
   * @param file_path File path for error messages
   * @throws std::runtime_error if keys are not sorted
   */
  void validate_key_ordering(const std::string &file_path) {
    for (int64_t i = 1; i < count_; ++i) {
      if (keys_[i] <= keys_[i - 1]) {
        LOG(ERROR) << "Keys not sorted at index " << i
                   << ": prev=" << keys_[i - 1] << ", current=" << keys_[i]
                   << " in shard " << file_path;
        throw std::runtime_error("Keys not sorted in shard file");
      }
    }

    LOG(INFO) << "Key ordering validated: " << count_ << " keys";
  }

  /**
   * @brief Read all embedding values from file
   *
   * @param file Input file stream
   * @param file_path File path for error messages
   * @throws std::runtime_error if read fails
   */
  void read_values(std::ifstream &file, const std::string &file_path) {
    const size_t values_size = static_cast<size_t>(count_) * dim_;
    const size_t bytes_to_read = values_size * sizeof(uint16_t);

    // Read all values at once for better performance
    file.read(reinterpret_cast<char *>(values_.data()), bytes_to_read);

    if (file.fail()) {
      LOG(ERROR) << "Failed to read values from: " << file_path;
      throw std::runtime_error("Failed to read values from: " + file_path);
    }

    LOG(INFO) << "Read " << values_size << " fp16 values from shard";
  }

  /**
   * @brief Verify that we've reached the end of file
   *
   * @param file Input file stream
   * @param file_path File path for warning messages
   */
  void verify_file_end(std::ifstream &file, const std::string &file_path) {
    // Try to read one more byte
    char extra_byte;
    file.read(&extra_byte, 1);

    if (!file.eof()) {
      LOG(WARNING) << "Extra data found at end of shard file: " << file_path;
      // Not throwing error, just warning
    }
  }

private:
  int64_t count_;                ///< Number of embeddings in this shard
  int32_t dim_;                  ///< Embedding dimension
  std::vector<int64_t> keys_;    ///< Sorted keys for binary search
  std::vector<uint16_t> values_; ///< FP16-encoded embeddings (flattened)
};

/// Smart pointer type for Shard objects
using ShardPtr = std::unique_ptr<Shard>;

/**
 * @class Embedding
 * @brief Distributed embedding table with hash-based sharding
 *
 * Manages a collection of embedding vectors distributed across multiple shards
 * for efficient parallel access. Keys are automatically routed to shards using
 * hash-based partitioning (bitwise AND with mask).
 *
 * @par Directory Structure
 * Expects the following directory layout:
 * @code
 * {path}/
 *   embedding{group}/
 *     shard0.dat
 *     shard1.dat
 *     ...
 *     shard{kShardCount-1}.dat
 * @endcode
 *
 * @par Sharding Strategy
 * Uses low-order bits of key for shard selection:
 * - shard_index = key & kShardMask
 * - Ensures uniform distribution if keys are well-distributed
 * - kShardCount must be power of 2 for efficient bitwise operation
 *
 * @par Thread Safety
 * Safe for concurrent reads after construction. Not thread-safe during
 * construction.
 *
 * @par Performance
 * - Lookup: O(log n) per shard (binary search)
 * - Memory: All embeddings loaded into RAM
 * - Cache: Sharding improves cache locality for parallel access
 */
class Embedding {
public:
  /**
   * @brief Loads embedding table from disk
   *
   * Loads all shards from the specified directory and validates consistency.
   * All shards must have the same dimension and must all exist.
   *
   * @param group Embedding table identifier (used in directory name)
   * @param dim Embedding dimension (must match all shard files)
   * @param path Base directory containing embedding data
   *
   * @throws std::invalid_argument If group < 0 or dim <= 0
   * @throws std::invalid_argument If dim > kMaxEmbeddingDim
   * @throws std::runtime_error If directory doesn't exist
   * @throws std::runtime_error If any shard file is missing
   * @throws std::runtime_error If dimension mismatch in any shard
   * @throws std::bad_alloc If memory allocation fails
   *
   * @note All kShardCount shard files must exist
   * @note Logs INFO message on successful load
   */
  Embedding(int64_t group, int32_t dim, const std::string &path)
      : group_(group), count_(0), dim_(dim) {
    // Validate input parameters
    if (group < 0) {
      LOG(ERROR) << "Invalid group: " << group;
      throw std::invalid_argument("Group must be non-negative");
    }

    if (dim <= 0) {
      LOG(ERROR) << "Invalid dimension: " << dim << " for group " << group;
      throw std::invalid_argument("Dimension must be positive");
    }

    if (dim > kMaxEmbeddingDim) {
      LOG(ERROR) << "Dimension " << dim << " exceeds maximum "
                 << kMaxEmbeddingDim << " for group " << group;
      throw std::invalid_argument("Dimension exceeds maximum allowed");
    }

    if (path.empty()) {
      LOG(ERROR) << "Empty path for embedding group " << group;
      throw std::invalid_argument("Path cannot be empty");
    }

    // Construct embedding directory path
    const std::filesystem::path embedding_dir =
        std::filesystem::path(path) / ("embedding" + std::to_string(group_));

    if (!std::filesystem::exists(embedding_dir)) {
      LOG(ERROR) << "Embedding directory not found: " << embedding_dir.string();
      throw std::runtime_error("Embedding directory not found: " +
                               embedding_dir.string());
    }

    if (!std::filesystem::is_directory(embedding_dir)) {
      LOG(ERROR) << "Path is not a directory: " << embedding_dir.string();
      throw std::runtime_error("Path is not a directory: " +
                               embedding_dir.string());
    }

    // Load all shards
    LOG(INFO) << "Loading embedding group " << group << " from "
              << embedding_dir.string();

    for (size_t i = 0; i < kShardCount; ++i) {
      const auto shard_file =
          embedding_dir / ("shard" + std::to_string(i) + ".dat");

      if (!std::filesystem::exists(shard_file)) {
        LOG(ERROR) << "Shard file not found: " << shard_file.string();
        throw std::runtime_error("Shard file not found: " +
                                 shard_file.string());
      }

      try {
        shards_[i] = std::make_unique<Shard>(dim, shard_file.string());

        // Validate dimension consistency
        if (shards_[i]->dimension() != dim) {
          LOG(ERROR) << "Dimension mismatch in shard " << i << ": expected "
                     << dim << ", got " << shards_[i]->dimension();
          throw std::runtime_error("Dimension mismatch in shard " +
                                   std::to_string(i));
        }

        count_ += shards_[i]->count();
      } catch (const std::exception &e) {
        LOG(ERROR) << "Failed to load shard " << i << " for group " << group
                   << ": " << e.what();
        throw;
      }
    }

    LOG(INFO) << "Loaded embedding group " << group
              << ": total_count=" << count_ << ", dim=" << dim_
              << ", shards=" << kShardCount;
  }

  /**
   * @brief Returns embedding dimension
   *
   * @return Embedding vector dimension
   */
  int32_t dimension() const noexcept { return dim_; }

  /**
   * @brief Returns group identifier
   *
   * @return Embedding table group ID
   */
  int64_t group() const noexcept { return group_; }

  /**
   * @brief Returns total number of embeddings across all shards
   *
   * @return Total embedding count
   */
  int64_t count() const noexcept { return count_; }

  /**
   * @brief Batch lookup with automatic shard routing
   *
   * Looks up multiple embeddings efficiently by routing each key to its
   * corresponding shard. Missing keys result in zero-filled vectors.
   *
   * @param keys Array of embedding identifiers
   * @param num_keys Number of keys to lookup
   * @param[out] output Contiguous buffer (size >= num_keys × dimension)
   *
   * @return Number of keys found
   *
   * @note Thread-safe for concurrent calls
   * @note Missing keys are zero-filled
   * @warning Caller must ensure buffers are large enough
   *
   * @par Example
   * @code
   * int64_t keys[] = {100, 200, 300};
   * std::vector<float> output(3 * embedding.dimension());
   * size_t found = embedding.batch_lookup(keys, 3, output.data());
   * // found <= 3, output contains 3 embedding vectors
   * @endcode
   */
  size_t batch_lookup(const int64_t *keys, size_t num_keys,
                      float *output) const {
    if (!keys || !output) {
      LOG(ERROR) << "Null buffer in batch_lookup for group " << group_;
      return 0;
    }

    if (num_keys == 0) {
      return 0;
    }

    size_t found = 0;
    for (size_t i = 0; i < num_keys; ++i) {
      const size_t shard_idx = get_shard_index(keys[i]);

      // Validate shard index (should never fail if kShardMask is correct)
      if (shard_idx >= kShardCount) {
        LOG(ERROR) << "Invalid shard index " << shard_idx << " for key "
                   << keys[i];
        std::memset(output + dim_ * i, 0, sizeof(float) * dim_);
        continue;
      }

      if (!shards_[shard_idx]) {
        LOG(ERROR) << "Null shard at index " << shard_idx;
        std::memset(output + dim_ * i, 0, sizeof(float) * dim_);
        continue;
      }

      if (shards_[shard_idx]->lookup(keys[i], output + dim_ * i)) {
        ++found;
      }
    }

    return found;
  }

  /** @brief Copy constructor (deleted - non-copyable) */
  Embedding(const Embedding &) = delete;

  /** @brief Copy assignment (deleted - non-copyable) */
  Embedding &operator=(const Embedding &) = delete;

  /** @brief Move constructor (deleted - non-movable) */
  Embedding(Embedding &&) = delete;

  /** @brief Move assignment (deleted - non-movable) */
  Embedding &operator=(Embedding &&) = delete;

private:
  /**
   * @brief Computes shard index using bitwise AND
   *
   * Uses low-order bits of key for efficient shard selection.
   * Assumes kShardCount is power of 2.
   *
   * @param key Embedding identifier
   * @return Shard index in range [0, kShardCount)
   *
   * @note Always returns valid index if kShardCount is power of 2
   */
  size_t get_shard_index(int64_t key) const noexcept {
    return static_cast<size_t>(key) & kShardMask;
  }

  int64_t group_; ///< Embedding table identifier
  int64_t count_; ///< Total embeddings across all shards
  int32_t dim_;   ///< Embedding dimension
  std::array<ShardPtr, kShardCount> shards_; ///< Shard storage array
};

/// Smart pointer type for Embedding objects
using EmbeddingPtr = std::unique_ptr<Embedding>;

} // namespace longmen

#endif // LONGMEN_EMBEDDING_HPP_
