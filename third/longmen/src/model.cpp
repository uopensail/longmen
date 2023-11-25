#include "model.h"

typedef unsigned char BitMap;

BitMap* new_bitmap(int size) {
    int c_size = (size >> 3) + 1;
    return (BitMap *)calloc(c_size, sizeof(BitMap));
}
void free_bitmap(BitMap *data) {
    free(data);
}
void set_bitmap(BitMap *bitMap, int index) {
    int byteIndex = index >> 3;
    int offset = index & 7;
    bitMap[byteIndex] |= (1 << offset);
}
int check_bitmap(BitMap *bitMap, int index) {
    int byteIndex = index >> 3;
    int offset = index & 7;
    return (bitMap[byteIndex] & (1 << offset)) != 0;
}

Tensor::Tensor(int64_t rows, int64_t cols, int64_t stride, torch::Dtype type)
    : m_rows(rows), m_cols(cols), m_stride(stride), m_type(type) {
  m_data = (char *)calloc(m_rows * m_cols, m_stride);
}

Tensor::~Tensor() {
  if (m_data != nullptr) {
    free(m_data);
    m_data = nullptr;
  }
}

void Tensor::print() {
  std::cout << "[";
  for (int64_t i = 0; i < m_rows; i++) {
    if (i > 0) {
      std::cout << "\n";
    }
    std::cout << "[";
    for (int64_t j = 0; j < m_cols; j++) {
      if (j > 0) {
        std::cout << ",";
      }
      if (m_type == torch::kFloat32) {
        std::cout << ((float *)m_data)[i * m_cols + j];
      } else {
        std::cout << ((int64_t *)m_data)[i * m_cols + j];
      }
    }
    std::cout << "]";
  }
  std::cout << "]" << std::endl;
}

void Tensor::set_row(int64_t row, char *data) {
  memcpy(&m_data[m_cols * m_stride * row], data, m_cols * m_stride);
}

Input::Input(int size) : m_size(size) {
  m_tensors = (Tensor **)calloc(m_size, sizeof(Tensor *));
}

Input::~Input() {
  for (int i = 0; i < m_size; i++) {
    if (m_tensors[i] != nullptr) {
      delete m_tensors[i];
      m_tensors[i] = nullptr;
    }
  }
  free(m_tensors);
  m_tensors = nullptr;
}

Tensor *&Input::operator[](int index) { return m_tensors[index]; }

void Input::print() {
  for (int i = 0; i < m_size; i++) {
    m_tensors[i]->print();
    std::cout << std::endl;
  }
}

TorchModel::TorchModel(std::string_view path) {
  try {
    c10::InferenceMode guard;
    this->module_ = torch::jit::load(std::string(path));
    this->module_.eval();
  } catch (const c10::Error &e) {
    std::cerr << "loading model from: " << path << " error\n";
    exit(-1);
  }
}

TorchModel::~TorchModel() {}

void TorchModel::forward(Input &input, float *result) {
  c10::InferenceMode guard;
  std::vector<torch::jit::IValue> values;
  for (int i = 0; i < input.m_size; i++) {
    torch::Tensor x =
        torch::from_blob(input[i]->m_data, {input[i]->m_rows, input[i]->m_cols},
                         input[i]->m_type);
    values.push_back(x);
  }
  torch::Tensor output = this->module_.forward(values).toTensor();
  auto accessor = output.accessor<float, 2>();
  memcpy(result, accessor.data(), sizeof(float) * output.numel());
}

Model::Model(std::string_view pool, std::string_view key,
             std::string_view toolkit, std::string_view model)
    : m_toolkit(std::make_shared<luban::Toolkit>(std::string(toolkit))),
      m_model(std::make_shared<TorchModel>(model)) {
  std::ifstream reader(std::string(pool), std::ios::in);
  if (!reader) {
    std::cerr << "read pool data file: " << pool << " error" << std::endl;
    exit(-1);
  }
  std::string line;
  std::string m_key(key);
  while (std::getline(reader, line)) {
    luban::SharedFeaturesPtr features = std::make_shared<luban::Features>(line);
    auto key = features->operator[](m_key);
    if (key == nullptr) {
      continue;
    }
    if (std::string *value = std::get_if<std::string>(key.get()); value != nullptr) {
      m_pool[*value] = m_toolkit->process_item(features);
    } else if (int64_t *value = std::get_if<int64_t>(key.get()); value != nullptr) {
      m_pool[std::to_string(*value)] = m_toolkit->process_item(features);
    }
  }
  reader.close();
}

void Model::forward(char *user_features, size_t len, char **items,
                    int64_t *lens, int size, float *scores) {
  auto user_feas =
      std::make_shared<luban::Features>(std::string_view{user_features, len});

  // luban to process user features
  auto user_rows = m_toolkit->process_user(user_feas);

  Input input(m_toolkit->m_groups.size());

  for (auto &group : m_toolkit->m_groups) {
    if (group.type == luban::DataType::kFloat32) {
      input[group.id] =
          new Tensor(size, group.width, group.stride, torch::kFloat32);
    } else {
      input[group.id] =
          new Tensor(size, group.width, group.stride, torch::kInt64);
    }
  }

  char *data = nullptr;
  BitMap* not_found_bitmap = new_bitmap(size);
  for (size_t i = 0; i < size; i++) {
    // copy user processed features
    for (auto &group : m_toolkit->m_user_placer->m_groups) {
      data = (*user_rows)[group.index]->m_data;
      input[group.id]->set_row(i, data);
    }

    // get item processed features
    auto iter = m_pool.find(std::string{items[i], size_t(lens[i])});
    if (iter == m_pool.end()) {
      set_bitmap(not_found_bitmap, i);
      continue;
    }

    for (auto &group : m_toolkit->m_item_placer->m_groups) {
      data = iter->second->m_rows[group.index]->m_data;
      input[group.id]->set_row(i, data);
    }
  }
  
  m_model->forward(input, scores);

  for (int i=0; i< size; i++) {
    if (check_bitmap(not_found_bitmap,i)) {
        scores[i] = -1.0;
    }
  }
  free_bitmap(not_found_bitmap);
}