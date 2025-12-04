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

#ifndef LONGMEN_EMBEDDINGS_HPP_
#define LONGMEN_EMBEDDINGS_HPP_

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include "common.hpp"
#include "embedding.hpp"
#include "json.hpp"

namespace longmen {

using json = nlohmann::json;

/**
 * @class Embeddings
 * @brief Thread-safe singleton manager for multiple embedding tables
 *
 * Manages a collection of embedding tables indexed by group ID. Provides
 * centralized loading from configuration and efficient batch lookup operations.
 * Implements the singleton pattern to ensure single global instance.
 *
 * @par Configuration Format (meta.json)
 * The configuration file must be located at `{workdir}/meta.json` with the
 * following structure:
 * @code{.json}
 * {
 *   "embeddings": [
 *     {"group": 0, "dim": 128},
 *     {"group": 1, "dim": 256},
 *     {"group": 5, "dim": 512}
 *   ]
 * }
 * @endcode
 *
 * @par Directory Structure
 * @code
 * {workdir}/
 *   meta.json
 *   embedding0/
 *     shard0.dat
 *     shard1.dat
 *     ...
 *   embedding1/
 *     shard0.dat
 *     ...
 * @endcode
 *
 * @par Thread Safety
 * - Safe for concurrent reads after load() completes
 * - load() must be called once before any other operations
 * - Not thread-safe during load()
 * - Multiple threads can safely call lookup operations concurrently
 *
 * @par Usage Example
 * @code
 * // Initialize (call once during startup)
 * auto& mgr = Embeddings::get_instance();
 * mgr.load("/path/to/workdir");
 *
 * // Lookup embeddings (thread-safe)
 * int64_t keys[] = {1, 2, 3};
 * std::vector<float> output(3 * mgr.dimension(0));
 * size_t found = mgr.batch_lookup(0, keys, 3, output.data());
 * @endcode
 *
 * @note Singleton pattern ensures only one instance exists
 * @note All embedding tables are kept in memory after loading
 */
class Embeddings {
public:
  /**
   * @brief Gets the singleton instance
   *
   * Returns reference to the global Embeddings instance. Thread-safe
   * initialization is guaranteed by C++11 magic statics.
   *
   * @return Reference to the global Embeddings instance
   *
   * @note Thread-safe initialization (C++11 magic statics)
   * @note First call initializes the instance
   */
  static Embeddings &get_instance() {
    static Embeddings instance;
    return instance;
  }

  /**
   * @brief Loads embedding tables from configuration file
   *
   * Reads meta.json from the specified directory and loads all configured
   * embedding tables. Each table is loaded from its corresponding subdirectory.
   *
   * @param workdir Base directory containing meta.json and embedding data
   *
   * @throws std::invalid_argument If workdir is empty
   * @throws std::runtime_error If meta.json is missing or malformed
   * @throws std::runtime_error If 'embeddings' field is missing or not an array
   * @throws std::invalid_argument If group ID is out of range [0,
   * kMaxEmbeddingNum)
   * @throws std::invalid_argument If dimension is invalid (<= 0)
   * @throws std::runtime_error If duplicate group IDs are found
   * @throws std::runtime_error If embedding data files are missing or corrupted
   * @throws std::bad_alloc If memory allocation fails
   *
   * @note Must be called before any lookup operations
   * @note Not thread-safe - call once during initialization
   * @note Logs INFO for each successfully loaded table
   * @note Partially loaded state on exception (some tables may be loaded)
   *
   * @par Example
   * @code
   * try {
   *   Embeddings::get_instance().load("/data/embeddings");
   * } catch (const std::exception& e) {
   *   LOG(ERROR) << "Failed to load embeddings: " << e.what();
   * }
   * @endcode
   */
  void load(const std::string &workdir) {
    // Validate input
    if (workdir.empty()) {
      LOG(ERROR) << "Empty workdir path";
      throw std::invalid_argument("Workdir path cannot be empty");
    }

    LOG(INFO) << "Loading embeddings from: " << workdir;

    // Load and parse configuration
    const json config = load_config(workdir);

    // Validate embeddings array
    if (!config.contains("embeddings")) {
      LOG(ERROR) << "Configuration missing 'embeddings' field in meta.json";
      throw std::runtime_error(
          "Configuration missing 'embeddings' field in meta.json");
    }

    if (!config["embeddings"].is_array()) {
      LOG(ERROR) << "'embeddings' field is not an array in meta.json";
      throw std::runtime_error(
          "'embeddings' field must be an array in meta.json");
    }

    const auto &embeddings_array = config["embeddings"];
    if (embeddings_array.empty()) {
      LOG(WARNING) << "Empty embeddings array in meta.json";
      return;
    }

    LOG(INFO) << "Found " << embeddings_array.size()
              << " embedding table(s) in configuration";

    // Load each embedding table
    size_t loaded_count = 0;
    for (const auto &emb : embeddings_array) {
      try {
        validate_embedding_config(emb);

        const int64_t group = emb["group"].get<int64_t>();
        const int32_t dim = emb["dim"].get<int32_t>();

        LOG(INFO) << "Loading embedding table: group=" << group
                  << ", dim=" << dim;

        tables_[group] = std::make_unique<Embedding>(group, dim, workdir);
        ++loaded_count;

        LOG(INFO) << "Successfully loaded embedding table " << group << " with "
                  << tables_[group]->count() << " embeddings";
      } catch (const std::exception &e) {
        LOG(ERROR) << "Failed to load embedding table: " << e.what();
        throw;
      }
    }

    LOG(INFO) << "Successfully loaded " << loaded_count
              << " embedding table(s)";
  }

  /**
   * @brief Performs batch lookup of embeddings
   *
   * Looks up multiple embedding vectors from the specified table.
   * Missing keys result in zero-filled vectors.
   *
   * @param group_id Embedding table identifier
   * @param keys Array of embedding keys to lookup
   * @param num_keys Number of keys in the array
   * @param[out] output Output buffer (size >= num_keys × dimension(group_id))
   *
   * @return Number of keys successfully found
   *
   * @throws std::invalid_argument If group_id is out of range
   * @throws std::runtime_error If table is not loaded
   * @throws std::invalid_argument If keys or output is nullptr (when num_keys >
   * 0)
   *
   * @note Thread-safe for concurrent calls after load()
   * @note Missing keys result in zero-filled vectors
   * @note Returns 0 if num_keys is 0
   *
   * @warning Caller must ensure output buffer is large enough
   * @warning Undefined behavior if buffer is too small
   *
   * @par Example
   * @code
   * int64_t keys[] = {100, 200, 300};
   * int32_t dim = mgr.dimension(0);
   * std::vector<float> output(3 * dim);
   * size_t found = mgr.batch_lookup(0, keys, 3, output.data());
   * // found <= 3, output contains 3 embedding vectors
   * @endcode
   */
  size_t batch_lookup(int64_t group_id, const int64_t *keys, size_t num_keys,
                      float *output) const {
    // Validate inputs
    if (num_keys > 0) {
      if (!keys) {
        LOG(ERROR) << "Null keys pointer for group " << group_id;
        throw std::invalid_argument("Keys pointer cannot be null");
      }
      if (!output) {
        LOG(ERROR) << "Null output pointer for group " << group_id;
        throw std::invalid_argument("Output pointer cannot be null");
      }
    }

    if (num_keys == 0) {
      return 0;
    }

    // Get table and perform lookup
    const Embedding *table = get_table(group_id);
    return table->batch_lookup(keys, num_keys, output);
  }

  /**
   * @brief Gets the embedding dimension for a table
   *
   * @param group_id Embedding table identifier
   *
   * @return Embedding dimension, or -1 if table doesn't exist or group_id is
   * invalid
   *
   * @note Thread-safe, noexcept
   * @note Returns -1 for invalid group_id or unloaded table
   *
   * @par Example
   * @code
   * int32_t dim = mgr.dimension(0);
   * if (dim > 0) {
   *   // Table exists and is loaded
   * }
   * @endcode
   */
  int32_t dimension(int64_t group_id) const noexcept {
    if (!is_valid_group_id(group_id)) {
      return -1;
    }

    const auto *table = tables_[group_id].get();
    return table ? table->dimension() : -1;
  }

  /**
   * @brief Gets the total number of embeddings in a table
   *
   * @param group_id Embedding table identifier
   *
   * @return Number of embeddings, or -1 if group_id is invalid, or 0 if table
   * not loaded
   *
   * @note Thread-safe, noexcept
   * @note Returns -1 for invalid group_id
   * @note Returns 0 for valid but unloaded table
   */
  int64_t count(int64_t group_id) const noexcept {
    if (!is_valid_group_id(group_id)) {
      return -1;
    }

    const auto *table = tables_[group_id].get();
    return table ? table->count() : 0;
  }

  /**
   * @brief Checks if a table is loaded
   *
   * @param group_id Embedding table identifier
   *
   * @return true if table exists and is loaded, false otherwise
   *
   * @note Thread-safe, noexcept
   * @note Returns false for invalid group_id
   */
  bool is_loaded(int64_t group_id) const noexcept {
    return is_valid_group_id(group_id) && tables_[group_id] != nullptr;
  }

  /**
   * @brief Gets the number of loaded tables
   *
   * @return Number of embedding tables currently loaded
   *
   * @note Thread-safe, noexcept
   */
  size_t loaded_count() const noexcept {
    size_t count = 0;
    for (const auto &table : tables_) {
      if (table) {
        ++count;
      }
    }
    return count;
  }

  /** @brief Copy constructor (deleted - singleton) */
  Embeddings(const Embeddings &) = delete;

  /** @brief Copy assignment (deleted - singleton) */
  Embeddings &operator=(const Embeddings &) = delete;

  /** @brief Move constructor (deleted - singleton) */
  Embeddings(Embeddings &&) = delete;

  /** @brief Move assignment (deleted - singleton) */
  Embeddings &operator=(Embeddings &&) = delete;

private:
  /**
   * @brief Private constructor for singleton pattern
   *
   * Initializes empty tables array. All tables are nullptr until load() is
   * called.
   */
  Embeddings() = default;

  /**
   * @brief Validates group ID range
   *
   * @param group_id Group ID to validate
   * @return true if valid (in range [0, kMaxEmbeddingNum)), false otherwise
   */
  bool is_valid_group_id(int64_t group_id) const noexcept {
    return group_id >= 0 && group_id < kMaxEmbeddingNum;
  }

  /**
   * @brief Gets table pointer with validation
   *
   * @param group_id Embedding table identifier
   * @return Pointer to embedding table (never nullptr)
   *
   * @throws std::invalid_argument If group_id is out of range
   * @throws std::runtime_error If table is not loaded
   */
  const Embedding *get_table(int64_t group_id) const {
    if (!is_valid_group_id(group_id)) {
      LOG(ERROR) << "Group ID " << group_id << " is out of range [0, "
                 << kMaxEmbeddingNum << ")";
      throw std::invalid_argument("Group ID " + std::to_string(group_id) +
                                  " is out of range [0, " +
                                  std::to_string(kMaxEmbeddingNum) + ")");
    }

    const auto *table = tables_[group_id].get();
    if (!table) {
      LOG(ERROR) << "Table for group " << group_id << " is not loaded";
      throw std::runtime_error("Table for group " + std::to_string(group_id) +
                               " is not loaded");
    }

    return table;
  }

  /**
   * @brief Validates embedding configuration entry
   *
   * Checks that required fields exist and have valid values.
   * Also checks for duplicate group IDs.
   *
   * @param emb JSON object containing embedding config
   *
   * @throws std::runtime_error If required fields are missing
   * @throws std::runtime_error If JSON type conversion fails
   * @throws std::invalid_argument If group ID or dimension is invalid
   * @throws std::runtime_error If duplicate group ID is detected
   */
  void validate_embedding_config(const json &emb) const {
    // Check required fields
    if (!emb.contains("group")) {
      LOG(ERROR) << "Embedding config missing 'group' field";
      throw std::runtime_error("Embedding config missing 'group' field");
    }

    if (!emb.contains("dim")) {
      LOG(ERROR) << "Embedding config missing 'dim' field";
      throw std::runtime_error("Embedding config missing 'dim' field");
    }

    // Extract and validate values
    int64_t group;
    int32_t dim;

    try {
      group = emb["group"].get<int64_t>();
    } catch (const json::type_error &e) {
      LOG(ERROR) << "Invalid type for 'group' field: " << e.what();
      throw std::runtime_error("Invalid type for 'group' field");
    }

    try {
      dim = emb["dim"].get<int32_t>();
    } catch (const json::type_error &e) {
      LOG(ERROR) << "Invalid type for 'dim' field: " << e.what();
      throw std::runtime_error("Invalid type for 'dim' field");
    }

    // Validate group ID range
    if (!is_valid_group_id(group)) {
      LOG(ERROR) << "Group ID " << group << " is out of range [0, "
                 << kMaxEmbeddingNum << ")";
      throw std::invalid_argument("Group ID " + std::to_string(group) +
                                  " is out of range [0, " +
                                  std::to_string(kMaxEmbeddingNum) + ")");
    }

    // Validate dimension
    if (dim <= 0) {
      LOG(ERROR) << "Invalid dimension " << dim << " for group " << group;
      throw std::invalid_argument("Invalid dimension " + std::to_string(dim) +
                                  " for group " + std::to_string(group));
    }

    if (dim > kMaxEmbeddingDim) {
      LOG(ERROR) << "Dimension " << dim << " exceeds maximum "
                 << kMaxEmbeddingDim << " for group " << group;
      throw std::invalid_argument("Dimension " + std::to_string(dim) +
                                  " exceeds maximum " +
                                  std::to_string(kMaxEmbeddingDim));
    }

    // Check for duplicate group ID
    if (tables_[group]) {
      LOG(ERROR) << "Duplicate group ID " << group << " in configuration";
      throw std::runtime_error("Duplicate group ID " + std::to_string(group) +
                               " in configuration");
    }
  }

  /**
   * @brief Loads and parses configuration file
   *
   * Reads meta.json from workdir and parses it as JSON.
   *
   * @param workdir Base directory containing meta.json
   * @return Parsed JSON configuration
   *
   * @throws std::runtime_error If file doesn't exist
   * @throws std::runtime_error If file can't be opened
   * @throws std::runtime_error If JSON parsing fails
   */
  json load_config(const std::string &workdir) const {
    const std::filesystem::path config_path =
        std::filesystem::path(workdir) / "meta.json";

    // Check file existence
    if (!std::filesystem::exists(config_path)) {
      LOG(ERROR) << "Configuration file not found: " << config_path.string();
      throw std::runtime_error("Configuration file not found: " +
                               config_path.string());
    }

    // Check if it's a regular file
    if (!std::filesystem::is_regular_file(config_path)) {
      LOG(ERROR) << "Path is not a regular file: " << config_path.string();
      throw std::runtime_error("Path is not a regular file: " +
                               config_path.string());
    }

    // Open file
    std::ifstream file(config_path);
    if (!file) {
      LOG(ERROR) << "Failed to open configuration file: "
                 << config_path.string();
      throw std::runtime_error("Failed to open configuration file: " +
                               config_path.string());
    }

    // Parse JSON
    try {
      json config;
      file >> config;
      LOG(INFO) << "Successfully parsed configuration file: "
                << config_path.string();
      return config;
    } catch (const json::parse_error &e) {
      LOG(ERROR) << "JSON parse error in '" << config_path.string()
                 << "': " << e.what();
      throw std::runtime_error("JSON parse error in '" + config_path.string() +
                               "': " + e.what());
    } catch (const std::exception &e) {
      LOG(ERROR) << "Error reading configuration file '" << config_path.string()
                 << "': " << e.what();
      throw std::runtime_error("Error reading configuration file: " +
                               std::string(e.what()));
    }
  }

private:
  /// Array of embedding tables indexed by group ID (nullptr if not loaded)
  std::array<std::unique_ptr<Embedding>, kMaxEmbeddingNum> tables_;
};

} // namespace longmen

#endif // LONGMEN_EMBEDDINGS_HPP_
