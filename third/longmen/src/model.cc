
#include "model.h"

#include <filesystem>
#include <fstream>

namespace longmen {

// ============================================================================
// Model Constructor
// ============================================================================

Model::Model(const std::string &workdir)
    : config_(nullptr), arena_(nullptr), graph_(nullptr), placement_(nullptr) {
  // Validate workdir
  if (workdir.empty()) {
    LOG(ERROR) << "Empty workdir path";
    throw std::invalid_argument("Workdir path cannot be empty");
  }

  // Check workdir exists
  if (!std::filesystem::exists(workdir)) {
    LOG(ERROR) << "Workdir does not exist: " << workdir;
    throw std::runtime_error("Workdir does not exist: " + workdir);
  }

  if (!std::filesystem::is_directory(workdir)) {
    LOG(ERROR) << "Workdir is not a directory: " << workdir;
    throw std::runtime_error("Workdir is not a directory: " + workdir);
  }

  LOG(INFO) << "Initializing Model from workdir: " << workdir;

  try {
    // Load model configuration
    const std::string meta_path = workdir + "/meta.json";
    json config = load_config(meta_path);
    config_ = std::make_shared<json>(std::move(config));

    LOG(INFO) << "Model configuration loaded";

    // Create memory arena
    arena_ = std::make_unique<Arena>(config_);
    if (!arena_) {
      throw std::runtime_error("Failed to create Arena");
    }

    LOG(INFO) << "Arena initialized";

    // Initialize ONNX graph
    graph_ = std::make_shared<CPUGraph>(*config_, workdir);
    if (!graph_ || !graph_->is_ready()) {
      throw std::runtime_error("Failed to initialize CPUGraph");
    }

    LOG(INFO) << "CPUGraph initialized";

    // Load feature configuration
    const std::string features_path = workdir + "/features.json";
    json features_config = load_config(features_path);

    // Create placement
    placement_ = std::make_unique<Placement>(features_config, *config_);
    if (!placement_) {
      throw std::runtime_error("Failed to create Placement");
    }

    LOG(INFO) << "Placement initialized";

    LOG(INFO) << "Model initialized successfully from: " << workdir;

  } catch (const std::exception &e) {
    LOG(ERROR) << "Failed to initialize Model: " << e.what();
    // Clean up partial initialization
    placement_.reset();
    graph_.reset();
    arena_.reset();
    config_.reset();
    throw;
  }
}

Model::~Model() { LOG(INFO) << "Destroying Model"; }

// ============================================================================
// Configuration Loading
// ============================================================================

json Model::load_config(const std::string &config_path) const {
  // Validate path
  if (config_path.empty()) {
    LOG(ERROR) << "Empty configuration path";
    throw std::invalid_argument("Configuration path cannot be empty");
  }

  // Check file exists
  if (!std::filesystem::exists(config_path)) {
    LOG(ERROR) << "Configuration file does not exist: " << config_path;
    throw std::runtime_error("Configuration file does not exist: " +
                             config_path);
  }

  // Check is regular file
  if (!std::filesystem::is_regular_file(config_path)) {
    LOG(ERROR) << "Configuration path is not a regular file: " << config_path;
    throw std::runtime_error("Configuration path is not a regular file: " +
                             config_path);
  }

  try {
    // Open file
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
      LOG(ERROR) << "Failed to open configuration file: " << config_path;
      throw std::runtime_error("Failed to open configuration file: " +
                               config_path);
    }

    // Parse JSON
    json config;
    config_file >> config;

    // Validate is object
    if (!config.is_object()) {
      LOG(ERROR) << "Configuration must be a JSON object: " << config_path;
      throw std::runtime_error("Configuration must be a JSON object: " +
                               config_path);
    }

    LOG(INFO) << "Configuration loaded successfully from: " << config_path;
    return config;

  } catch (const json::parse_error &e) {
    LOG(ERROR) << "JSON parse error in file " << config_path << ": " << e.what()
               << " at byte " << e.byte;
    throw std::runtime_error("JSON parse error in file '" + config_path +
                             "': " + std::string(e.what()) + " at byte " +
                             std::to_string(e.byte));
  } catch (const std::exception &e) {
    LOG(ERROR) << "Error reading configuration file " << config_path << ": "
               << e.what();
    throw std::runtime_error("Error reading configuration file '" +
                             config_path + "': " + std::string(e.what()));
  }
}

// ============================================================================
// Inference
// ============================================================================

int32_t Model::forward(char *user_features, size_t len, char **items,
                       size_t *lens, int32_t batch, float **scores,
                       int64_t *version) {
  // Validate inputs
  if (!user_features) {
    LOG(ERROR) << "Null user_features pointer";
    return -1;
  }

  if (len == 0) {
    LOG(ERROR) << "Zero-length user_features";
    return -1;
  }

  if (!items) {
    LOG(ERROR) << "Null items pointer";
    return -1;
  }

  if (!lens) {
    LOG(ERROR) << "Null lens pointer";
    return -1;
  }

  if (!scores) {
    LOG(ERROR) << "Null scores pointer";
    return -1;
  }

  if (!version) {
    LOG(ERROR) << "Null version pointer";
    return -1;
  }

  if (batch <= 0) {
    LOG(ERROR) << "Invalid batch size: " << batch;
    return -1;
  }

  // Check model is ready
  if (!is_ready()) {
    LOG(ERROR) << "Model is not ready for inference";
    return -1;
  }

  try {
    // Prepare features and get GraphIO
    std::unique_ptr<GraphIO> io = placement_->put(
        arena_.get(), user_features, len, items, lens, scores, batch, version);

    if (!io) {
      LOG(ERROR) << "Failed to get GraphIO from placement";
      return -1;
    }

    // Run inference
    const int32_t status = graph_->forward(*io);

    // Return GraphIO to arena
    arena_->put(std::move(io));

    if (status != 0) {
      LOG(ERROR) << "Graph forward failed with status: " << status;
      return -1;
    }

    return 0;

  } catch (const std::exception &e) {
    LOG(ERROR) << "Exception during forward: " << e.what();
    return -1;
  }
}

// ============================================================================
// Pool Reflush
// ============================================================================

void Model::reflush(char *path, size_t plen, int64_t version) {
  // Validate inputs
  if (!path) {
    LOG(ERROR) << "Null path pointer for reflush";
    return;
  }

  if (plen == 0) {
    LOG(ERROR) << "Zero-length path for reflush";
    return;
  }

  if (!placement_) {
    LOG(ERROR) << "Placement not initialized, cannot reflush";
    return;
  }

  // Convert to string
  std::string path_str(path, plen);

  // Validate path
  if (path_str.empty()) {
    LOG(ERROR) << "Empty path string for reflush";
    return;
  }

  try {
    LOG(INFO) << "Reflush requested: path=" << path_str
              << ", version=" << version;

    // Perform reflush
    const bool success = placement_->reflush(path_str, version);

    if (success) {
      LOG(INFO) << "Reflush completed successfully: version=" << version;
    } else {
      LOG(ERROR) << "Reflush failed: version=" << version;
    }

  } catch (const std::exception &e) {
    LOG(ERROR) << "Exception during reflush: " << e.what();
  }
}

} // namespace longmen
