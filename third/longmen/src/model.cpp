#include "model.h"

Input::Input(std::shared_ptr<luban::Matrix> matrix) : matrix_(matrix) {}

Input::Input(const Input &input) : matrix_(input.matrix_) {}

Input::Input(const Input &&input) : matrix_(input.matrix_) {}

Input &Input::operator=(const Input &input) {
  if (this == &input) {
    return *this;
  }
  matrix_ = input.matrix_;
  return *this;
}

TorchModel::TorchModel(std::string_view path) {
  try {
    c10::InferenceMode guard;
    this->module_ = torch::jit::load(std::string(path));
  } catch (const c10::Error &e) {
    std::cerr << "loading model from: " << path << " error\n";
    exit(-1);
  }
}

TorchModel::~TorchModel() {}

void TorchModel::forward(std::shared_ptr<luban::Matrices> inputs,
                         float *result) {
  c10::InferenceMode guard;
  std::vector<torch::jit::IValue> values;
  for (size_t i = 0; i < inputs->size(); i++) {
    torch::Dtype type = torch::kInt64;
    if ((*inputs)[i]->m_type == luban::DataType::kFloat32) {
      type = torch::kFloat32;
    }
    torch::Tensor x =
        torch::from_blob((*inputs)[i]->m_data,
                         {(*inputs)[i]->m_rows, (*inputs)[i]->m_cols}, type);
    values.push_back(x);
  }
  torch::Tensor output = this->module_.forward(values).toTensor();
  auto accessor = output.accessor<float, 2>();
  memcpy(result, accessor.data(), sizeof(float) * output.numel());
}

Pool::Pool(std::string_view path, std::string_view key)
    : path_(path), key_(key) {}

void Pool::load() {
  size_t max_size = 4096;
  char *buffer = (char *)malloc(max_size);
  std::ifstream reader(path_, std::ios::in | std::ios::binary);
  if (!reader) {
    std::cerr << "read data file: " << path_ << " error" << std::endl;
    exit(-1);
  }
  std::string line;
  while (std::getline(reader, line)) {
    luban::SharedFeaturesPtr features = std::make_shared<luban::Features>(line);
    auto key = features->operator[](key_);
    if (key == nullptr) {
      continue;
    }
    std::string *value = std::get_if<std::string>(key.get());
    if (value == nullptr) {
      continue;
    }
    items_[*value] = features;
  }
  reader.close();
}

luban::SharedFeaturesPtr Pool::operator[](std::string_view item) {
  auto iter = items_.find(std::string(item));
  if (iter == items_.end()) {
    return nullptr;
  }
  return iter->second;
}

Model::Model(std::string_view toolkit, std::string_view model)
    : toolkit_(std::make_shared<luban::Toolkit>(std::string(toolkit))),
      model_(std::make_shared<TorchModel>(model)) {}

void Model::forward(Pool *pool, char *user_features, size_t len, char **items,
                    int *lens, int size, float *scores) {
  luban::SharedFeaturesPtr user_feas =
      std::make_shared<luban::Features>(std::string_view{user_features, len});

  luban::SharedFeaturesPtr feas = nullptr;
  luban::SharedFeaturesListPtr list = std::make_shared<luban::FeaturesList>();
  for (size_t i = 0; i < size; i++) {
    luban::SharedFeaturesPtr tmp = std::make_shared<luban::Features>();
    tmp->merge(user_feas);
    feas = pool->operator[]({items[i], size_t(lens[i])});
    if (feas != nullptr) {
      tmp->merge(feas);
    }
    list->push(tmp);
  }
  std::shared_ptr<luban::Matrices> matrices = toolkit_->process(list);
  model_->forward(matrices, scores);
}