//
// `LongMen` - 'Torch Model inference in c++'
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
//

#ifndef LONGMAN_MODEL_H
#define LONGMAN_MODEL_H

#pragma once

#include "toolkit.h"
#include <filesystem>
#include <torch/script.h>
#include <vector>
namespace longmen{

std::vector<std::string> split(const std::string& str, char delimiter);

struct ModelInputEmbeddingMeta {
  std::vector<std::string> keys;
  std::vector<int> dims;
  int sum_dims;
};
 

class Input {
public:
  Input() = delete;
  Input(int size);
  ~Input();

public:
  int m_size;
  torch::Tensor* m_tensors;
  std::vector<std::vector<int64_t>> m_tensor_sizes;
};

class TorchModel {
public:
  TorchModel() = delete;
  TorchModel(const TorchModel &) = delete;
  TorchModel(const TorchModel &&) = delete;
  TorchModel(std::string_view path, std::string_view graph_meta);
  ~TorchModel() = default;

  std::shared_ptr<ModelInputEmbeddingMeta> get_input_embedding_meta(int input_index) {
    auto it = m_input_embedding_meta.find(input_index);
    if (it == m_input_embedding_meta.end()) {
      return nullptr;
    } else {
      return it->second;
    }
  }

  torch::Tensor embedding_forward(std::shared_ptr<ModelInputEmbeddingMeta> input_meta, const torch::Tensor& input_keys);
  void torch_forward(Input &inputs, float *result);
private:
 
  std::map<int, std::shared_ptr<ModelInputEmbeddingMeta>> m_input_embedding_meta;
  torch::jit::Module m_torch_module;
};

}// namespace longmen
#endif // LONGMAN_MODEL_H