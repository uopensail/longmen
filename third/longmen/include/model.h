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

class Tensor {
public:
  Tensor() = delete;
  Tensor(int64_t rows, int64_t cols, int64_t stride, torch::Dtype type);
  ~Tensor();
  void set_row(int64_t row, char *data);
  void print();

public:
  int64_t m_rows;
  int64_t m_cols;
  int64_t m_stride;
  torch::Dtype m_type;
  char *m_data;
};

class Input {
public:
  Input() = delete;
  Input(int size);
  ~Input();
  Tensor *&operator[](int index);
  void print();

public:
  int m_size;
  Tensor **m_tensors;
};

class TorchModel {
public:
  TorchModel() = delete;
  TorchModel(const TorchModel &) = delete;
  TorchModel(const TorchModel &&) = delete;
  TorchModel(std::string_view path);
  ~TorchModel();
  void forward(Input &inputs, float *result);

private:
  torch::jit::Module module_;
};

class Model {
public:
  Model() = delete;
  Model(const Model &) = delete;
  Model(const Model &&) = delete;
  Model(std::string_view pool, std::string_view lua_plugin, std::string_view toolkit,
        std::string_view model);
  ~Model() = default;
  void forward(char *user_features, size_t len, char **items, int64_t *lens,
               int size, float *scores);
  void forward(luban::Rows* row, char **items, int64_t *lens,
               int size, float *scores);
private:
  std::shared_ptr<luban::Toolkit> m_toolkit;
  std::shared_ptr<TorchModel> m_model;
  std::unordered_map<std::string, std::shared_ptr<luban::Rows>> m_pool;
};

#endif // LONGMAN_MODEL_H