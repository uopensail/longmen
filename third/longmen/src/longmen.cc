#include "longmen.h"

#include <glog/logging.h>

#include <cstdint>
#include <exception>

#include "model.h"

extern "C" {

// ============================================================================
// Model Creation
// ============================================================================

void *longmen_create(char *workdir, int32_t len) {
  // Validate inputs
  if (!workdir) {
    LOG(ERROR) << "longmen_create: NULL workdir pointer";
    return nullptr;
  }

  if (len <= 0) {
    LOG(ERROR) << "longmen_create: Invalid workdir length: " << len;
    return nullptr;
  }

  try {
    // Convert to std::string (handles non-null-terminated strings)
    std::string workdir_str(workdir, static_cast<size_t>(len));

    LOG(INFO) << "longmen_create: Creating model from workdir: " << workdir_str;

    // Create model
    auto *model = new longmen::Model(workdir_str);

    if (!model) {
      LOG(ERROR) << "longmen_create: Model allocation failed";
      return nullptr;
    }

    // Verify model is ready
    if (!model->is_ready()) {
      LOG(ERROR) << "longmen_create: Model initialization failed";
      delete model;
      return nullptr;
    }

    LOG(INFO) << "longmen_create: Model created successfully";
    return static_cast<void *>(model);

  } catch (const std::bad_alloc &e) {
    LOG(ERROR) << "longmen_create: Memory allocation failed: " << e.what();
    return nullptr;
  } catch (const std::exception &e) {
    LOG(ERROR) << "longmen_create: Exception: " << e.what();
    return nullptr;
  } catch (...) {
    LOG(ERROR) << "longmen_create: Unknown exception";
    return nullptr;
  }
}

// ============================================================================
// Model Release
// ============================================================================

void longmen_release(void *model) {
  if (!model) {
    LOG(WARNING) << "longmen_release: NULL model pointer (no-op)";
    return;
  }

  try {
    LOG(INFO) << "longmen_release: Releasing model";

    auto *m = static_cast<longmen::Model *>(model);
    delete m;

    LOG(INFO) << "longmen_release: Model released successfully";

  } catch (const std::exception &e) {
    LOG(ERROR) << "longmen_release: Exception during release: " << e.what();
  } catch (...) {
    LOG(ERROR) << "longmen_release: Unknown exception during release";
  }
}

// ============================================================================
// Pool Reflush
// ============================================================================

void longmen_reflush(void *model, char *path, int32_t len, int64_t version) {
  // Validate inputs
  if (!model) {
    LOG(ERROR) << "longmen_reflush: NULL model pointer";
    return;
  }

  if (!path) {
    LOG(ERROR) << "longmen_reflush: NULL path pointer";
    return;
  }

  if (len <= 0) {
    LOG(ERROR) << "longmen_reflush: Invalid path length: " << len;
    return;
  }

  try {
    auto *m = static_cast<longmen::Model *>(model);

    LOG(INFO) << "longmen_reflush: Reflush requested with version: " << version;

    // Call model reflush (handles conversion and validation internally)
    m->reflush(path, static_cast<size_t>(len), version);

  } catch (const std::exception &e) {
    LOG(ERROR) << "longmen_reflush: Exception: " << e.what();
  } catch (...) {
    LOG(ERROR) << "longmen_reflush: Unknown exception";
  }
}

// ============================================================================
// Inference
// ============================================================================

int32_t longmen_forward(void *model, char *user_features, int32_t len,
                        void *items, void *lens, int32_t size, void *scores,
                        int64_t *version) {
  // Validate model
  if (!model) {
    LOG(ERROR) << "longmen_forward: NULL model pointer";
    return -1;
  }

  // Validate user features
  if (!user_features) {
    LOG(ERROR) << "longmen_forward: NULL user_features pointer";
    return -1;
  }

  if (len <= 0) {
    LOG(ERROR) << "longmen_forward: Invalid user_features length: " << len;
    return -1;
  }

  // Validate items
  if (!items) {
    LOG(ERROR) << "longmen_forward: NULL items pointer";
    return -1;
  }

  // Validate lens
  if (!lens) {
    LOG(ERROR) << "longmen_forward: NULL lens pointer";
    return -1;
  }

  // Validate batch size
  if (size <= 0) {
    LOG(ERROR) << "longmen_forward: Invalid batch size: " << size;
    return -1;
  }

  // Validate scores
  if (!scores) {
    LOG(ERROR) << "longmen_forward: NULL scores pointer";
    return -1;
  }

  // Validate version
  if (!version) {
    LOG(ERROR) << "longmen_forward: NULL version pointer";
    return -1;
  }

  try {
    auto *m = static_cast<longmen::Model *>(model);

    // Cast void pointers to appropriate types
    auto **items_arr = static_cast<char **>(items);
    auto *lens_arr = static_cast<size_t *>(lens);
    auto **scores_arr = static_cast<float **>(scores);

    // Call model forward
    const int32_t ret =
        m->forward(user_features, static_cast<size_t>(len), items_arr, lens_arr,
                   size, scores_arr, version);
    if (ret != 0) {
      LOG(ERROR) << "longmen_forward: Inference failed with status: " << ret;
    }

    return ret;

  } catch (const std::exception &e) {
    LOG(ERROR) << "longmen_forward: Exception: " << e.what();
    return -1;
  } catch (...) {
    LOG(ERROR) << "longmen_forward: Unknown exception";
    return -1;
  }
}

} // extern "C"
