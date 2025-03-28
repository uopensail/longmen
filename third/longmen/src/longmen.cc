#include "longmen.h"
#include "onnx.h"
#include <fstream>
#include <sstream>
#include <vector>

// Internal type forward declarations
namespace longmen {
class OnnxModel;
struct OutputSlice;
Pool;
} // namespace longmen

namespace {

/**
 * @brief Split string using delimiter (optimized version)
 * @param str Input string to split
 * @param delimiter Character to split on
 * @return Vector of tokens
 */
std::vector<std::string> split_string(const std::string &str, char delimiter) {
  std::vector<std::string> tokens;
  tokens.reserve(8); // Pre-allocate for common case

  size_t start = 0;
  size_t end = str.find(delimiter);

  while (end != std::string::npos) {
    tokens.emplace_back(str.substr(start, end - start));
    start = end + 1;
    end = str.find(delimiter, start);
  }

  tokens.emplace_back(str.substr(start));
  return tokens;
}

} // anonymous namespace

LongmenModel longmen_create_model(const char *workdir) {
  try {
    return new longmen::OnnxModel(workdir);
  } catch (const std::exception &e) {
    std::cerr << "Model creation failed: " << e.what() << std::endl;
    return nullptr;
  }
}

void longmen_release_model(LongmenModel model) {
  auto *m = static_cast<longmen::OnnxModel *>(model);
  delete m;
}

LongmenOutputs longmen_serve(LongmenModel model, int batch_size,
                             LongmenPool pool, const char *user_features,
                             const char **items) {
  auto *m = static_cast<longmen::OnnxModel *>(model);
  auto *p = static_cast<longmen::Pool *>(pool);

  try {
    return m->call(batch_size, p, user_features, items);
  } catch (const std::exception &e) {
    std::cerr << "Inference error: " << e.what() << std::endl;
    return nullptr;
  }
}

LongmenPool longmen_create_pool(LongmenModel model, const char *data_path) {
  auto *m = static_cast<longmen::OnnxModel *>(model);

  auto pool = std::make_unique<Pool>();

  std::ifstream reader(data_path);
  if (!reader.is_open()) {
    std::cerr << "Failed to open data file: " << data_path << std::endl;
    return nullptr;
  }

  std::string line;
  while (std::getline(reader, line)) {
    auto columns = split_string(line, '\t');
    if (columns.size() != 2) {
      std::cerr << "Invalid data format in line: " << line << std::endl;
      continue;
    }

    try {
      auto features = std::make_shared<minia::Features>(columns[1]);
      m->item->preprocess(*features);
      pool->emplace(columns[0], std::move(features));
    } catch (const std::exception &e) {
      std::cerr << "Feature processing failed: " << e.what() << std::endl;
    }
  }

  return pool.release();
}

void longmen_release_pool(LongmenPool pool) {
  delete static_cast<Pool *>(pool);
}

void longmen_release_outputs(LongmenOutputs outputs) {
  delete static_cast<longmen::OutputSlice *>(outputs);
}
