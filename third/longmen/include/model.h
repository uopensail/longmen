//
// `LongMen` - 'Torch Model inference in c++'
// Copyright (C) 2019 - present timepi <timepi123@gmail.com>
// LuBan is provided under: GNU Affero General Public License (AGPL3.0)
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

#include <torch/script.h>
#include <vector>

#include <filesystem>

#include "toolkit.h"

class TorchModel;

class Input {
public:
  Input() = delete;
  Input(std::shared_ptr<luban::Matrix> matrix);
  Input(const Input &input);
  Input(const Input &&input);
  Input &operator=(const Input &input);
  ~Input() = default;

private:
  std::shared_ptr<luban::Matrix> matrix_;
  friend class TorchModel;
};

class TorchModel {
public:
  TorchModel() = delete;
  TorchModel(const TorchModel &) = delete;
  TorchModel(const TorchModel &&) = delete;
  TorchModel(std::string_view path);
  ~TorchModel();
  void forward(std::shared_ptr<luban::Matrices> inputs, float *result);

private:
  torch::jit::Module module_;
};

class Pool {
public:
  Pool() = delete;
  Pool(const Pool &) = delete;
  Pool(const Pool &&) = delete;
  Pool(std::string_view path, std::string_view key);
  ~Pool() = default;
  void load();
  luban::SharedFeaturesPtr operator[](std::string_view item);

private:
  std::string path_;
  std::string key_;
  std::unordered_map<std::string, luban::SharedFeaturesPtr> items_;
};

class Model {
public:
  Model() = delete;
  Model(const Model &) = delete;
  Model(const Model &&) = delete;
  Model(std::string_view toolkit, std::string_view model);
  ~Model() = default;
  void forward(Pool *pool, char *user_features, size_t len, char **items,
               int *lens, int size, float *scores);

private:
  std::shared_ptr<luban::Toolkit> toolkit_;
  std::shared_ptr<TorchModel> model_;
};

#endif // LONGMAN_MODEL_H