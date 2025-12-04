#include "placement.h"

#include <fstream>
#include <sstream>

namespace longmen {

// ============================================================================
// Pool Implementation
// ============================================================================

Pool::Pool(const std::string &path, int64_t version, minia::Minia *handler)
    : version_(version) {
  // Validate inputs
  if (path.empty()) {
    LOG(ERROR) << "Empty file path";
    throw std::invalid_argument("File path cannot be empty");
  }

  if (!handler) {
    LOG(ERROR) << "Null handler pointer";
    throw std::invalid_argument("Handler cannot be null");
  }

  if (version < 0) {
    LOG(WARNING) << "Negative version number: " << version;
  }

  // Open file
  std::ifstream file(path);
  if (!file.is_open()) {
    LOG(ERROR) << "Failed to open file: " << path;
    throw std::runtime_error("Failed to open file: " + path);
  }

  LOG(INFO) << "Loading pool from: " << path << " (version=" << version << ")";

  std::string line;
  int line_number = 0;
  int skipped_lines = 0;
  int error_lines = 0;

  while (std::getline(file, line)) {
    line_number++;

    // Skip empty lines
    if (line.empty()) {
      continue;
    }

    // Find tab separator
    size_t tab_pos = line.find('\t');
    if (tab_pos == std::string::npos) {
      LOG(WARNING) << "Invalid format at line " << line_number
                   << " (missing tab), skipping";
      skipped_lines++;
      continue;
    }

    // Split into ID and JSON
    std::string id = trim(line.substr(0, tab_pos));
    std::string json_value = trim(line.substr(tab_pos + 1));

    // Validate non-empty
    if (id.empty()) {
      LOG(WARNING) << "Empty ID at line " << line_number << ", skipping";
      skipped_lines++;
      continue;
    }

    if (json_value.empty()) {
      LOG(WARNING) << "Empty JSON at line " << line_number << ", skipping";
      skipped_lines++;
      continue;
    }

    // Check for duplicate IDs
    if (entries_.find(id) != entries_.end()) {
      LOG(WARNING) << "Duplicate ID '" << id << "' at line " << line_number
                   << ", overwriting";
    }

    // Parse and process features
    try {
      auto features = std::make_shared<minia::Features>(json_value);
      handler->call(*features);
      entries_[std::move(id)] = std::move(features);

    } catch (const std::exception &e) {
      LOG(ERROR) << "Error parsing JSON at line " << line_number << ": "
                 << e.what();
      error_lines++;
    }
  }

  file.close();

  // Log summary
  LOG(INFO) << "Pool loaded: " << entries_.size() << " entries from " << path
            << " (skipped=" << skipped_lines << ", errors=" << error_lines
            << ")";

  if (entries_.empty()) {
    LOG(WARNING) << "Pool is empty after loading";
  }
}

std::shared_ptr<minia::Features> Pool::operator[](std::string_view id) const {
  auto it = entries_.find(id);
  if (it != entries_.end()) {
    return it->second;
  }
  return nullptr;
}

std::string Pool::trim(const std::string &str) const {
  if (str.empty()) {
    return "";
  }

  size_t first = str.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }

  size_t last = str.find_last_not_of(" \t\r\n");
  return str.substr(first, last - first + 1);
}

// ============================================================================
// Placement Implementation
// ============================================================================

Placement::Placement(const json &features_config, const json &graph_config)
    : pool_(nullptr), user_handler_(nullptr), item_handler_(nullptr),
      cross_handler_(nullptr) {
  try {
    LOG(INFO) << "Initializing Placement";
    parse_config(features_config, graph_config);
    LOG(INFO) << "Placement initialized successfully";
  } catch (const std::exception &e) {
    LOG(ERROR) << "Failed to initialize Placement: " << e.what();
    throw;
  }
}

void Placement::parse_config(const json &features_config,
                             const json &graph_config) {
  // Validate inputs
  if (!features_config.is_object()) {
    throw std::invalid_argument("features_config must be a JSON object");
  }

  if (!graph_config.is_object()) {
    throw std::invalid_argument("graph_config must be a JSON object");
  }

  try {
    // Parse graph inputs to build slot mapping
    if (!graph_config.contains("inputs")) {
      throw std::runtime_error("Missing 'inputs' field in graph_config");
    }

    const auto &inputs = graph_config.at("inputs");
    if (!inputs.is_array() || inputs.empty()) {
      throw std::runtime_error("'inputs' must be a non-empty array");
    }

    std::vector<std::string> input_names;
    input_names.reserve(inputs.size());

    int32_t slot = 0;
    for (const auto &input : inputs) {
      if (!input.contains("name")) {
        throw std::runtime_error("Input at index " + std::to_string(slot) +
                                 " missing 'name' field");
      }

      std::string name = input.at("name").get<std::string>();
      if (name.empty()) {
        throw std::runtime_error("Input at index " + std::to_string(slot) +
                                 " has empty name");
      }

      input_names.push_back(name);
      slots_[name] = slot;
      slot++;
    }

    LOG(INFO) << "Parsed " << input_names.size() << " input slots";

    // Parse user features (optional)
    if (features_config.contains("user")) {
      const auto &user_conf = features_config.at("user");
      if (!user_conf.is_array()) {
        throw std::runtime_error("'user' must be an array");
      }

      if (user_conf.empty()) {
        LOG(WARNING) << "'user' array is empty";
      } else {
        std::vector<std::string> user_expressions;
        user_expressions.reserve(user_conf.size());

        for (size_t i = 0; i < user_conf.size(); ++i) {
          const auto &row = user_conf[i];

          if (!row.contains("slot")) {
            throw std::runtime_error("User feature at index " +
                                     std::to_string(i) + " missing 'slot'");
          }

          if (!row.contains("expr")) {
            throw std::runtime_error("User feature at index " +
                                     std::to_string(i) + " missing 'expr'");
          }

          int32_t slot_idx = row.at("slot").get<int32_t>();
          if (slot_idx < 0 ||
              slot_idx >= static_cast<int32_t>(input_names.size())) {
            throw std::runtime_error(
                "User feature at index " + std::to_string(i) +
                " has invalid slot: " + std::to_string(slot_idx));
          }

          std::string expr =
              input_names[slot_idx] + "=" + row.at("expr").get<std::string>();
          user_expressions.emplace_back(std::move(expr));
        }

        user_handler_ = std::make_unique<minia::Minia>(user_expressions);
        LOG(INFO) << "Initialized user handler with " << user_expressions.size()
                  << " expressions";
      }
    } else {
      LOG(INFO) << "No user features configured";
    }

    // Parse item features (optional)
    if (features_config.contains("item")) {
      const auto &item_conf = features_config.at("item");
      if (!item_conf.is_array()) {
        throw std::runtime_error("'item' must be an array");
      }

      if (item_conf.empty()) {
        LOG(WARNING) << "'item' array is empty";
      } else {
        std::vector<std::string> item_expressions;
        item_expressions.reserve(item_conf.size());

        for (size_t i = 0; i < item_conf.size(); ++i) {
          const auto &row = item_conf[i];

          if (!row.contains("slot")) {
            throw std::runtime_error("Item feature at index " +
                                     std::to_string(i) + " missing 'slot'");
          }

          if (!row.contains("expr")) {
            throw std::runtime_error("Item feature at index " +
                                     std::to_string(i) + " missing 'expr'");
          }

          int32_t slot_idx = row.at("slot").get<int32_t>();
          if (slot_idx < 0 ||
              slot_idx >= static_cast<int32_t>(input_names.size())) {
            throw std::runtime_error(
                "Item feature at index " + std::to_string(i) +
                " has invalid slot: " + std::to_string(slot_idx));
          }

          std::string expr =
              input_names[slot_idx] + "=" + row.at("expr").get<std::string>();
          item_expressions.emplace_back(std::move(expr));
        }

        item_handler_ = std::make_unique<minia::Minia>(item_expressions);
        LOG(INFO) << "Initialized item handler with " << item_expressions.size()
                  << " expressions";
      }
    } else {
      LOG(INFO) << "No item features configured";
    }

    // Parse cross features (optional)
    if (features_config.contains("cross")) {
      const auto &cross_conf = features_config.at("cross");
      if (!cross_conf.is_array()) {
        throw std::runtime_error("'cross' must be an array");
      }

      if (cross_conf.empty()) {
        LOG(WARNING) << "'cross' array is empty";
      } else {
        std::vector<std::string> cross_expressions;
        cross_expressions.reserve(cross_conf.size());

        for (size_t i = 0; i < cross_conf.size(); ++i) {
          const auto &row = cross_conf[i];

          if (!row.contains("slot")) {
            throw std::runtime_error("Cross feature at index " +
                                     std::to_string(i) + " missing 'slot'");
          }

          if (!row.contains("expr")) {
            throw std::runtime_error("Cross feature at index " +
                                     std::to_string(i) + " missing 'expr'");
          }

          int32_t slot_idx = row.at("slot").get<int32_t>();
          if (slot_idx < 0 ||
              slot_idx >= static_cast<int32_t>(input_names.size())) {
            throw std::runtime_error(
                "Cross feature at index " + std::to_string(i) +
                " has invalid slot: " + std::to_string(slot_idx));
          }

          std::string expr =
              input_names[slot_idx] + "=" + row.at("expr").get<std::string>();
          cross_expressions.emplace_back(std::move(expr));
        }

        cross_handler_ = std::make_unique<minia::Minia>(cross_expressions);
        LOG(INFO) << "Initialized cross handler with "
                  << cross_expressions.size() << " expressions";
      }
    } else {
      LOG(INFO) << "No cross features configured";
    }

  } catch (const nlohmann::json::out_of_range &e) {
    LOG(ERROR) << "Missing required field in config: " << e.what();
    throw std::runtime_error("Missing required field in config: " +
                             std::string(e.what()));
  } catch (const nlohmann::json::type_error &e) {
    LOG(ERROR) << "Invalid type in config: " << e.what();
    throw std::runtime_error("Invalid type in config: " +
                             std::string(e.what()));
  }
}

std::unique_ptr<GraphIO> Placement::put(Arena *arena, char *user_features,
                                        size_t len, char **items, size_t *lens,
                                        float **scores, int32_t batch,
                                        int64_t *version) {
  // Validate inputs
  if (!arena) {
    LOG(ERROR) << "Null arena pointer";
    throw std::invalid_argument("Arena cannot be null");
  }

  if (!version) {
    LOG(ERROR) << "Null version pointer";
    throw std::invalid_argument("Version pointer cannot be null");
  }

  if (batch <= 0) {
    LOG(ERROR) << "Invalid batch size: " << batch;
    throw std::invalid_argument("Batch size must be positive");
  }

  // Get GraphIO from arena
  std::unique_ptr<GraphIO> io = arena->get(batch);
  if (!io) {
    LOG(ERROR) << "Failed to get GraphIO from arena";
    throw std::runtime_error("Failed to allocate GraphIO");
  }

  // Get current pool
  std::shared_ptr<Pool> pool = get_pool();
  if (!pool) {
    LOG(WARNING) << "No pool available, returning empty GraphIO";
    *version = -1;
    return io;
  }

  *version = pool->get_version();

  // Setup GraphIO
  io->set_batch(batch);
  io->set_outputs(scores);
  io->zero();

  try {
    // Process user features
    if (user_handler_ && user_features && len > 0) {
      auto user_feas = std::make_unique<minia::Features>(user_features, len);
      user_handler_->call(*user_feas);

      const auto &keys = user_handler_->features();
      for (size_t k = 0; k < keys.size(); ++k) {
        minia::FeaturePtr ptr = user_feas->get(keys[k]);
        io->get_input(slots_[keys[k]]).set_value_with_broadcast(batch, ptr);
      }
    }

    // Process item features
    if (item_handler_ && items && lens) {
      const auto &keys = item_handler_->features();
      int missing_count = 0;

      for (int32_t b = 0; b < batch; ++b) {
        if (!items[b] || lens[b] == 0) {
          LOG(WARNING) << "Empty item ID at batch index " << b;
          missing_count++;
          continue;
        }

        std::string_view id(items[b], lens[b]);
        std::shared_ptr<minia::Features> features = (*pool)[id];

        if (!features) {
          missing_count++;
          continue;
        }

        for (size_t k = 0; k < keys.size(); ++k) {
          minia::FeaturePtr ptr = features->get(keys[k]);
          io->get_input(slots_[keys[k]]).set_value(b, ptr);
        }
      }

      if (missing_count > 0) {
        LOG(WARNING) << "Missing items in pool: " << missing_count << "/"
                     << batch;
      }
    }

    // Process cross features
    if (cross_handler_ && items && lens && user_features && len > 0) {
      auto user_feas = std::make_unique<minia::Features>(user_features, len);
      const auto &keys = cross_handler_->features();
      int missing_count = 0;

      for (int32_t b = 0; b < batch; ++b) {
        if (!items[b] || lens[b] == 0) {
          missing_count++;
          continue;
        }

        std::string_view id(items[b], lens[b]);
        std::shared_ptr<minia::Features> item_features = (*pool)[id];

        if (!item_features) {
          missing_count++;
          continue;
        }

        // Generate cross features
        minia::Features cross;
        cross_handler_->call({&cross, user_feas.get(), item_features.get()});

        for (size_t k = 0; k < keys.size(); ++k) {
          minia::FeaturePtr ptr = cross.get(keys[k]);
          io->get_input(slots_[keys[k]]).set_value(b, ptr);
        }
      }

      if (missing_count > 0) {
        LOG(WARNING) << "Missing items for cross features: " << missing_count
                     << "/" << batch;
      }
    }

  } catch (const std::exception &e) {
    LOG(ERROR) << "Error processing features: " << e.what();
    throw std::runtime_error("Feature processing failed: " +
                             std::string(e.what()));
  }

  return io;
}

bool Placement::reflush(const std::string &path, int64_t version) {
  // Validate inputs
  if (path.empty()) {
    LOG(ERROR) << "Empty path for reflush";
    return false;
  }

  if (!item_handler_) {
    LOG(ERROR) << "Cannot reflush: item handler not initialized";
    return false;
  }

  try {
    LOG(INFO) << "Reflush starting: path=" << path << ", version=" << version;

    // Load new pool
    auto new_pool = std::make_shared<Pool>(path, version, item_handler_.get());

    if (new_pool->empty()) {
      LOG(WARNING) << "New pool is empty, aborting reflush";
      return false;
    }

    // Atomic swap
    std::atomic_store(&pool_, new_pool);

    LOG(INFO) << "Pool updated successfully: version=" << version
              << ", entries=" << new_pool->size();
    return true;

  } catch (const std::exception &e) {
    LOG(ERROR) << "Failed to reflush pool: " << e.what();
    return false;
  }
}

} // namespace longmen
