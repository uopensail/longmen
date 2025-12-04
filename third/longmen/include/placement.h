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

#ifndef LONGMEN_PLACEMENT_H_
#define LONGMEN_PLACEMENT_H_

#include <atomic>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

#include "arena.h"
#include "json.hpp"
#include "minia.h"

namespace longmen {

using json = nlohmann::json;

/**
 * @struct string_hash
 * @brief Transparent hash functor for string types
 *
 * Enables heterogeneous lookup in unordered containers, allowing lookups
 * with string_view without creating temporary std::string objects.
 *
 * @note Requires C++20 or C++14 with transparent comparator support
 */
struct string_hash {
  /// Enable transparent lookup
  using is_transparent = void;

  /**
   * @brief Hash C-string
   * @param str Null-terminated C-string
   * @return Hash value
   */
  size_t operator()(const char *str) const noexcept {
    return std::hash<std::string_view>{}(str);
  }

  /**
   * @brief Hash string_view
   * @param str String view
   * @return Hash value
   */
  size_t operator()(std::string_view str) const noexcept {
    return std::hash<std::string_view>{}(str);
  }

  /**
   * @brief Hash std::string
   * @param str String object
   * @return Hash value
   */
  size_t operator()(const std::string &str) const noexcept {
    return std::hash<std::string>{}(str);
  }
};

/**
 * @struct string_equal
 * @brief Transparent equality comparator for string types
 *
 * Enables heterogeneous comparison in unordered containers.
 */
struct string_equal {
  /// Enable transparent comparison
  using is_transparent = void;

  /**
   * @brief Compare two string views
   * @param lhs Left-hand side
   * @param rhs Right-hand side
   * @return true if equal, false otherwise
   */
  bool operator()(std::string_view lhs, std::string_view rhs) const noexcept {
    return lhs == rhs;
  }
};

/**
 * @class Pool
 * @brief Thread-safe feature pool for item features
 *
 * Manages a collection of pre-computed item features loaded from disk.
 * Supports efficient lookup by item ID using transparent string hashing.
 *
 * @par File Format
 * Each line in the input file should be:
 * @code
 * item_id\t{"feature1": value1, "feature2": value2, ...}
 * @endcode
 *
 * @par Thread Safety
 * - Read operations (operator[], size(), get_version()) are thread-safe
 * - Construction is not thread-safe
 * - Immutable after construction
 *
 * @par Usage Example
 * @code
 * minia::Minia handler(expressions);
 * Pool pool("/path/to/items.txt", 12345, &handler);
 *
 * auto features = pool["item_123"];
 * if (features) {
 *   // Use features...
 * }
 * @endcode
 *
 * @see Placement
 */
class Pool {
public:
  /**
   * @brief Constructs pool by loading features from file
   *
   * Loads item features from a tab-separated file where each line contains
   * an item ID and JSON-encoded features. Features are processed by the
   * provided handler.
   *
   * @param path Path to feature file
   * @param version Version number for this pool
   * @param handler Feature processor (must not be null)
   *
   * @throws std::invalid_argument If handler is null
   * @throws std::runtime_error If file cannot be opened
   * @throws std::runtime_error If file format is invalid
   *
   * @note Logs warnings for malformed lines (skips them)
   * @note Logs INFO with loaded entry count
   */
  Pool(const std::string &path, int64_t version, minia::Minia *handler);

  /**
   * @brief Gets the number of entries in the pool
   *
   * @return Number of loaded item features
   *
   * @note Thread-safe
   * @note O(1) complexity
   */
  size_t size() const noexcept { return entries_.size(); }

  /**
   * @brief Checks if pool is empty
   *
   * @return true if no entries, false otherwise
   */
  bool empty() const noexcept { return entries_.empty(); }

  /**
   * @brief Looks up features by item ID
   *
   * Performs transparent lookup without creating temporary strings.
   *
   * @param id Item identifier (can be string, string_view, or const char*)
   * @return Shared pointer to features, or nullptr if not found
   *
   * @note Thread-safe (const method)
   * @note Returns nullptr for missing items (not an error)
   * @note O(1) average complexity
   *
   * @par Example
   * @code
   * auto features = pool["item_123"];
   * if (features) {
   *   auto value = features->get("feature_name");
   * }
   * @endcode
   */
  std::shared_ptr<minia::Features> operator[](std::string_view id) const;

  /**
   * @brief Gets the version number of this pool
   *
   * @return Version identifier
   *
   * @note Thread-safe
   */
  int64_t get_version() const noexcept { return version_; }

private:
  /**
   * @brief Trims whitespace from string
   *
   * @param str Input string
   * @return Trimmed string
   */
  std::string trim(const std::string &str) const;

private:
  /// Version identifier for this pool
  int64_t version_;

  /// Item ID to features mapping (thread-safe for reads)
  std::unordered_map<std::string, std::shared_ptr<minia::Features>, string_hash,
                     string_equal>
      entries_;
};

/**
 * @class Placement
 * @brief Feature placement and inference orchestrator
 *
 * Manages the complete inference pipeline including:
 * - User feature processing
 * - Item feature lookup from pool
 * - Cross feature generation
 * - Feature placement into model inputs
 * - Hot-swappable feature pool updates
 *
 * @par Configuration Format
 * @code{.json}
 * // features_config
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
 *
 * // graph_config
 * {
 *   "inputs": [
 *     {"name": "user_features", "shape": [-1, 10], "dtype": 0},
 *     {"name": "item_features", "shape": [-1, 20], "dtype": 0}
 *   ],
 *   "outputs": [...]
 * }
 * @endcode
 *
 * @par Thread Safety
 * - put() is thread-safe (uses atomic pool pointer)
 * - reflush() is thread-safe (atomic swap)
 * - Construction is not thread-safe
 *
 * @par Usage Example
 * @code
 * Placement placement(features_config, graph_config);
 *
 * // Initial pool load
 * placement.reflush("/path/to/items.txt", 1);
 *
 * // Inference
 * int64_t version;
 * auto io = placement.put(arena, user_json, user_len,
 *                        item_ids, item_lens, scores, batch, &version);
 *
 * // Hot update pool
 * placement.reflush("/path/to/new_items.txt", 2);
 * @endcode
 *
 * @see Pool
 * @see Arena
 * @see GraphIO
 */
class Placement {
public:
  /**
   * @brief Constructs placement from configuration
   *
   * Parses feature and graph configurations to set up the inference pipeline.
   *
   * @param features_config JSON configuration for feature processing
   * @param graph_config JSON configuration for model graph
   *
   * @throws std::runtime_error If required fields are missing
   * @throws std::invalid_argument If configuration is invalid
   * @throws std::runtime_error If handler creation fails
   *
   * @note Logs INFO on successful initialization
   * @note Logs ERROR on failures
   */
  Placement(const json &features_config, const json &graph_config);

  /**
   * @brief Destructor
   */
  ~Placement() = default;

  /** @brief Copy constructor (deleted - non-copyable) */
  Placement(const Placement &) = delete;

  /** @brief Copy assignment (deleted - non-copyable) */
  Placement &operator=(const Placement &) = delete;

  /** @brief Move constructor (deleted - non-movable) */
  Placement(Placement &&) = delete;

  /** @brief Move assignment (deleted - non-movable) */
  Placement &operator=(Placement &&) = delete;

  /**
   * @brief Gets the current feature pool
   *
   * Returns the currently active pool. May be nullptr if no pool loaded.
   *
   * @return Shared pointer to current pool, or nullptr
   *
   * @note Thread-safe (atomic load)
   * @note Pool may change between calls due to reflush()
   */
  std::shared_ptr<Pool> get_pool() const { return std::atomic_load(&pool_); }

  /**
   * @brief Hot-swaps the feature pool
   *
   * Atomically replaces the current pool with a new one loaded from the
   * specified file. Ongoing inference operations continue using the old pool.
   *
   * @param path Path to new feature file
   * @param version Version number for new pool
   *
   * @return true on success, false on failure
   *
   * @note Thread-safe (atomic swap)
   * @note Old pool is kept alive until all references are released
   * @note Returns false on error (old pool remains active)
   * @note Logs INFO on success, ERROR on failure
   *
   * @par Example
   * @code
   * if (placement.reflush("/path/to/new_items.txt", 2)) {
   *   LOG(INFO) << "Pool updated successfully";
   * }
   * @endcode
   */
  bool reflush(const std::string &path, int64_t version);

  /**
   * @brief Prepares features for inference
   *
   * Processes user features, looks up item features from pool, generates
   * cross features, and places them into GraphIO for inference.
   *
   * @param arena Arena for GraphIO allocation
   * @param user_features JSON-encoded user features
   * @param len Length of user_features
   * @param items Array of item IDs (C-strings)
   * @param lens Array of item ID lengths
   * @param[out] scores Output buffer for scores (will be set in GraphIO)
   * @param batch Batch size (number of items)
   * @param[out] version Pool version used for this batch
   *
   * @return GraphIO ready for inference
   *
   * @note Thread-safe
   * @note Sets *version to -1 if no pool available
   * @note Missing items are skipped (not an error)
   * @note GraphIO is zeroed before filling
   *
   * @warning Caller must ensure pointers are valid
   * @warning arena must not be null
   * @warning batch must match array sizes
   *
   * @par Example
   * @code
   * char* user_json = "{\"user_id\": value, \"age\": value}";
   * char* items[] = {"item_1", "item_2"};
   * size_t lens[] = {6, 6};
   * float* scores[2];
   * int64_t version;
   *
   * auto io = placement.put(arena, user_json, strlen(user_json),
   *                        items, lens, scores, 2, &version);
   * graph.forward(*io);
   * @endcode
   */
  std::unique_ptr<GraphIO> put(Arena *arena, char *user_features, size_t len,
                               char **items, size_t *lens, float **scores,
                               int32_t batch, int64_t *version);

private:
  /**
   * @brief Parses configuration and initializes handlers
   *
   * @param features_config Feature processing configuration
   * @param graph_config Model graph configuration
   *
   * @throws std::runtime_error If parsing fails
   */
  void parse_config(const json &features_config, const json &graph_config);

private:
  /// Current feature pool (atomic for hot-swap)
  std::shared_ptr<Pool> pool_;

  /// Input name to slot index mapping
  std::unordered_map<std::string, int32_t> slots_;

  /// User feature processor
  std::unique_ptr<minia::Minia> user_handler_;

  /// Item feature processor
  std::unique_ptr<minia::Minia> item_handler_;

  /// Cross feature processor
  std::unique_ptr<minia::Minia> cross_handler_;
};

} // namespace longmen

#endif // LONGMEN_PLACEMENT_H_
