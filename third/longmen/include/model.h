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

#ifndef LONGMEN_MODEL_H_
#define LONGMEN_MODEL_H_

#include <memory>
#include <string>

#include "arena.h"
#include "graph.h"
#include "json.hpp"
#include "placement.h"

namespace longmen {

using json = nlohmann::json;

/**
 * @class Model
 * @brief High-level inference model interface
 *
 * Provides a unified interface for deep learning model inference, integrating:
 * - ONNX model graph execution (CPUGraph)
 * - Feature processing and placement (Placement)
 * - Memory management (Arena)
 * - Hot-swappable feature pools
 *
 * @par Directory Structure
 * The workdir must contain:
 * @code
 * workdir/
 * ├── meta.json          # Model graph configuration
 * ├── features.json      # Feature processing configuration
 * ├── model.onnx         # ONNX model file
 * └── embeddings/        # Embedding tables (optional)
 * @endcode
 *
 * @par Configuration Files
 *
 * **meta.json** (Model Graph Configuration):
 * @code{.json}
 * {
 *   "version": "1764658117",
 *   "model": "model.onnx",
 *   "threads": 4,
 *   "inputs": [
 *     {"name": "user_features", "shape": [-1, 100], "dtype": 0},
 *     {"name": "item_features", "shape": [-1, 50], "dtype": 0}
 *   ],
 *   "outputs": [
 *     {"name": "scores", "shape": [-1, 1], "dtype": 1}
 *   ]
 * }
 * @endcode
 *
 * **features.json** (Feature Processing Configuration):
 * @code{.json}
 * {
 *   "user": [
 *     {"slot": 0, "expr": "hash(user_id)"},
 *     {"slot": 1, "expr": "identity(user_age)"}
 *   ],
 *   "item": [
 *     {"slot": 2, "expr": "hash(item_category)"},
 *     {"slot": 3, "expr": "identity(item_price)"}
 *   ],
 *   "cross": [
 *     {"slot": 4, "expr": "user_age * item_price"}
 *   ]
 * }
 * @endcode
 *
 * @par Thread Safety
 * - forward() is thread-safe (can be called concurrently)
 * - reflush() is thread-safe (atomic pool swap)
 * - Construction is not thread-safe
 *
 * @par Usage Example
 * @code
 * #include "model.h"
 *
 * // Initialize model
 * longmen::Model model("/path/to/workdir");
 *
 * // Load initial feature pool
 * char pool_path[] = "/path/to/items.txt";
 * model.reflush(pool_path, strlen(pool_path), 1);
 *
 * // Prepare input data
 * char user_json[] = "{\"user_id\": value, \"age\": value}";
 * char* items[] = {"item_1", "item_2", "item_3"};
 * size_t lens[] = {6, 6, 6};
 * float* scores[3];
 * int64_t version;
 *
 * // Run inference
 * int32_t ret = model.forward(
 *     user_json, strlen(user_json),
 *     items, lens, 3,
 *     scores, &version
 * );
 *
 * if (ret == 0) {
 *     for (int i = 0; i < 3; i++) {
 *         std::cout << "Item " << i << " score: " << *scores[i] << std::endl;
 *     }
 * }
 *
 * // Hot update feature pool
 * char new_pool[] = "/path/to/new_items.txt";
 * model.reflush(new_pool, strlen(new_pool), 2);
 * @endcode
 *
 * @note Non-copyable and non-movable
 * @note Automatically manages all internal resources
 *
 * @see CPUGraph
 * @see Placement
 * @see Arena
 */
class Model {
public:
  /** @brief Default constructor (deleted) */
  Model() = delete;

  /** @brief Copy constructor (deleted - non-copyable) */
  Model(const Model &) = delete;

  /** @brief Copy assignment (deleted - non-copyable) */
  Model &operator=(const Model &) = delete;

  /** @brief Move constructor (deleted - non-movable) */
  Model(Model &&) = delete;

  /** @brief Move assignment (deleted - non-movable) */
  Model &operator=(Model &&) = delete;

  /**
   * @brief Constructs model from working directory
   *
   * Loads configuration files, initializes ONNX graph, sets up feature
   * processing pipeline, and allocates memory arena.
   *
   * @param workdir Path to working directory containing configuration files
   *
   * @throws std::invalid_argument If workdir is empty or doesn't exist
   * @throws std::runtime_error If configuration files are missing:
   *         - meta.json not found
   *         - features.json not found
   * @throws std::runtime_error If configuration parsing fails:
   *         - Invalid JSON format
   *         - Missing required fields
   *         - Invalid field types
   * @throws std::runtime_error If model initialization fails:
   *         - ONNX model loading error
   *         - Feature handler creation error
   *         - Arena allocation error
   *
   * @note Logs INFO on successful initialization
   * @note Logs ERROR on failures with detailed context
   *
   * @par Directory Validation
   * - Checks workdir exists and is a directory
   * - Validates meta.json exists and is readable
   * - Validates features.json exists and is readable
   * - Validates model.onnx path in meta.json
   *
   * @par Initialization Steps
   * 1. Load and parse meta.json
   * 2. Create Arena from configuration
   * 3. Initialize CPUGraph with ONNX model
   * 4. Load and parse features.json
   * 5. Create Placement with feature handlers
   */
  explicit Model(const std::string &workdir);

  /**
   * @brief Destructor
   *
   * Automatically releases all resources including:
   * - ONNX Runtime session
   * - Feature pools
   * - Memory arena
   *
   * @note Logs INFO on destruction
   */
  ~Model();

  /**
   * @brief Performs batch inference
   *
   * Processes user features, looks up item features, generates cross features,
   * and executes model inference to produce scores.
   *
   * @param user_features JSON-encoded user features (null-terminated or with
   * len)
   * @param len Length of user_features in bytes
   * @param items Array of item IDs (C-strings, not necessarily null-terminated)
   * @param lens Array of item ID lengths (one per item)
   * @param batch Number of items to score (size of items/lens/scores arrays)
   * @param[out] scores Array of pointers to receive output scores
   * @param[out] version Pool version used for this inference
   *
   * @return 0 on success, -1 on failure
   *
   * @note Thread-safe (can be called concurrently)
   * @note GraphIO is automatically managed (allocated and returned to arena)
   * @note Missing items are skipped (not an error)
   * @note Sets *version to -1 if no pool is loaded
   *
   * @warning All pointer parameters must be valid (not null)
   * @warning batch must match the size of items/lens/scores arrays
   * @warning Caller must ensure items[i] points to valid memory of lens[i]
   * bytes
   * @warning scores[i] will be set to point to internal buffers (valid until
   * next call)
   *
   * @par Error Handling
   * - Returns -1 if graph is not ready
   * - Returns -1 if feature processing fails
   * - Returns -1 if inference fails
   * - Logs ERROR with detailed context
   *
   * @par Performance
   * - Uses memory arena for zero-allocation inference
   * - Batch processing for efficiency
   * - Parallel execution with configured threads
   *
   * @par Example
   * @code
   * char user[] = "{\"user_id\": {\"type\":2, \"value\": \"user_123\"}}";
   * char* items[] = {"item_1", "item_2"};
   * size_t lens[] = {6, 6};
   * float* scores[2];
   * int64_t version;
   *
   * int ret = model.forward(user, strlen(user), items, lens, 2, scores,
   * &version); if (ret == 0) { std::cout << "Score 0: " << *scores[0] <<
   * std::endl; std::cout << "Score 1: " << *scores[1] << std::endl; std::cout
   * << "Pool version: " << version << std::endl;
   * }
   * @endcode
   */
  int32_t forward(char *user_features, size_t len, char **items, size_t *lens,
                  int32_t batch, float **scores, int64_t *version);

  /**
   * @brief Hot-swaps the feature pool
   *
   * Atomically replaces the current feature pool with a new one loaded from
   * the specified file. Ongoing inference operations continue using the old
   * pool until completion.
   *
   * @param path Path to new feature file (not necessarily null-terminated)
   * @param plen Length of path in bytes
   * @param version Version number for the new pool
   *
   * @note Thread-safe (atomic pool swap)
   * @note Old pool is kept alive until all references are released
   * @note Does not throw exceptions (logs errors internally)
   * @note Logs INFO on success, ERROR on failure
   *
   * @warning path must point to valid memory of plen bytes
   * @warning File must exist and be readable
   * @warning File format must match expected format
   *
   * @par Behavior on Failure
   * - Old pool remains active
   * - Error is logged but not thrown
   * - Subsequent inference uses old pool
   *
   * @par Example
   * @code
   * char new_pool[] = "/path/to/new_items.txt";
   * model.reflush(new_pool, strlen(new_pool), 2);
   * // Old pool is still used by ongoing inference
   * // New inference will use new pool
   * @endcode
   */
  void reflush(char *path, size_t plen, int64_t version);

  /**
   * @brief Checks if model is ready for inference
   *
   * @return true if model initialized successfully, false otherwise
   *
   * @note Thread-safe
   */
  bool is_ready() const noexcept {
    return graph_ && graph_->is_ready() && placement_ && arena_;
  }

  /**
   * @brief Gets the model configuration
   *
   * @return Shared pointer to configuration JSON
   *
   * @note Thread-safe (shared_ptr)
   * @note Returns nullptr if not initialized
   */
  std::shared_ptr<json> get_config() const noexcept { return config_; }

private:
  /**
   * @brief Loads and parses JSON configuration file
   *
   * @param config_path Path to JSON configuration file
   *
   * @return Parsed JSON object
   *
   * @throws std::runtime_error If file doesn't exist
   * @throws std::runtime_error If file cannot be opened
   * @throws std::runtime_error If JSON parsing fails
   *
   * @note Logs INFO on success
   * @note Logs ERROR on failure with detailed context
   */
  json load_config(const std::string &config_path) const;

private:
  /// Model configuration (shared for thread-safety)
  std::shared_ptr<json> config_;

  /// Memory arena for GraphIO allocation
  std::unique_ptr<Arena> arena_;

  /// ONNX model graph
  std::shared_ptr<CPUGraph> graph_;

  /// Feature processing and placement
  std::unique_ptr<Placement> placement_;
};

} // namespace longmen

#endif // LONGMEN_MODEL_H_
