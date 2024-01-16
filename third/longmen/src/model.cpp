#include "model.h"
#include "sample.h"
#include "json.hpp"
#include <ATen/Version.h>
#include <chrono>
namespace longmen {

std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(str);

    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}


Input::Input(int size) : m_size(size) {
  m_tensors = new torch::Tensor[size];
  m_tensor_sizes.resize(size);
}

Input::~Input() {
  delete []m_tensors;
}

std::map<int,std::shared_ptr<ModelInputEmbeddingMeta>> parse_input_embedding_meta(const std::string& file_path) {
  std::map<int,std::shared_ptr<ModelInputEmbeddingMeta>> ret;
  std::ifstream infile(file_path);
  if (!infile.is_open()) {
    std::cerr << "failed to open file: " << file_path << std::endl;
    return ret;
  }

  infile.seekg(0, std::ios::end);
  std::string buffer(infile.tellg(), ' ');
  infile.seekg(0);
  infile.read(&buffer[0], buffer.size());
  infile.close();

  const json &doc = json::parse(buffer);
  int flag = doc["sparse"].get<int>();

  if (flag == 1) {
    const std::vector<json> &input_metas = doc["meta"];
    for (auto &input_meta : input_metas) {
        int index = input_meta["input"];
        std::shared_ptr<ModelInputEmbeddingMeta> embedding_meta = std::make_shared<ModelInputEmbeddingMeta>();
        embedding_meta->keys = input_meta["keys"].get<std::vector<std::string>>();
        embedding_meta->dims = input_meta["dims"].get<std::vector<int>>();
        int sum = 0;
        for (int j = 0;j<embedding_meta->dims.size();j++) {
            sum+=embedding_meta->dims[j];
        }
        embedding_meta->sum_dims = sum;
        ret.insert({index, embedding_meta});
    }
  }
  return ret;
}
TorchModel::TorchModel(std::string_view path, std::string_view graph_meta) {
  std::cout << torch::show_config() << std::endl;
  try {
    c10::InferenceMode guard;
    this->m_torch_module = torch::jit::load(std::string(path));
    this->m_torch_module.eval();
  } catch (const c10::Error &e) {
    std::cerr << "loading model from: " << path << " error\n";
    exit(-1);
  }
  this->m_input_embedding_meta = parse_input_embedding_meta(std::string(graph_meta));
}

torch::Tensor TorchModel::embedding_forward(std::shared_ptr<ModelInputEmbeddingMeta> input_meta, const torch::Tensor& input_keys) {
    std::vector<torch::Tensor> tensors;
  c10::InferenceMode guard;
  for (int i = 0;i <input_meta->keys.size();i++) {
    auto attr_name = input_meta->keys[i];
    torch::jit::Module submodule = this->m_torch_module.attr(attr_name).toModule();
    auto output = submodule.forward({input_keys});
    tensors.push_back(output.toTensor());
  }
  torch::Tensor result1 = torch::cat(tensors, -1);
  return result1;
}

void TorchModel::torch_forward(Input &input, float *result) {
  c10::InferenceMode guard;
  std::vector<torch::jit::IValue> values;
  for (int i = 0; i < input.m_size; i++) {
    values.push_back(input.m_tensors[i]);
  }
  torch::Tensor output = this->m_torch_module.forward(values).toTensor();
  auto accessor = output.accessor<float, 2>();
  memcpy(result, accessor.data(), sizeof(float) * output.numel());
}
 

}// namespace longmen