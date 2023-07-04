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
// This is the sparse model.
//
#ifndef LONGMAN_MODEL
#define LONGMAN_MODEL

#pragma once

#include <torch/script.h>

#include <filesystem>

#include "../third/luban/include/toolkit.hpp"

class TorchModel {
 public:
  TorchModel() = delete;
  TorchModel(const TorchModel &) = delete;
  TorchModel(const TorchModel &&) = delete;
  TorchModel(const std::string &path) {
    try {
      c10::InferenceMode guard;
      this->module_ = torch::jit::load(path);
    } catch (const c10::Error &e) {
      std::cerr << "loading model from: " << path << " error\n";
      exit(-1);
    }
  }
  ~TorchModel() {}
  void forward(void *data, size_t batch, int width, float *result) {
    c10::InferenceMode guard;
    // use blob, don't need to allocate memory
    torch::Tensor x = torch::from_blob(
        data, {static_cast<long long>(batch), width}, torch::kInt64);
    torch::Tensor output = this->module_.forward({x}).toTensor();
    auto result_a = output.accessor<float, 2>();
    for (int i = 0; i < result_a.size(0); i++) {
      if (result[i] != 0.0) {
        result[i] = result_a[i][1];
      }
    }
  }

 private:
  torch::jit::Module module_;
};

class Pool {
 public:
  Pool() = delete;
  Pool(const std::string &pool_file, const std::string &key) {
    this->version_ = std::filesystem::path(pool_file).stem().string();
    size_t max_size = 4096;
    char *buffer = (char *)malloc(max_size);
    std::ifstream reader(pool_file, std::ios::in | std::ios::binary);
    if (!reader) {
      std::cerr << "read data file: " << pool_file << " error" << std::endl;
      exit(-1);
    }
    size_t len;
    while (reader.read((char *)&len, 8)) {
      if (len > max_size) {
        max_size = len + 1024;
        free(buffer);
        buffer = (char *)malloc(max_size);
      }
      reader.ignore(4);
      reader.read(buffer, len);
      reader.ignore(4);
      sample::Example example;
      example.ParseFromArray(buffer, len);

      auto it = example.features().feature().find(key);
      if (it == example.features().feature().end()) {
        continue;
      }
      if (!it->second.has_bytes_list()) {
        continue;
      }
      const std::string &id = it->second.bytes_list().value(0);
      sample::Features *features = new sample::Features();
      features->CopyFrom(example.features());
      this->pool_[id] = features;
    }
    free(buffer);
    reader.close();
  }

  ~Pool() {
    for (auto &kv : this->pool_) {
      delete kv.second;
    }
    this->pool_.clear();
  }

  const std::string &get_version() const { return this->version_; }

  sample::Features *get_features(const std::string &id) {
    auto iter = this->pool_.find(id);
    if (iter == this->pool_.end()) {
      return nullptr;
    }
    return iter->second;
  }

 private:
  std::string version_;
  std::unordered_map<std::string, sample::Features *> pool_;
};

class Model {
 public:
  Model() = delete;
  Model(const std::string &toolkit, const std::string &pool,
        const std::string &model, const std::string &key)
      : key_(key),
        pool_(std::make_shared<Pool>(pool, key)),
        toolkit_(std::make_shared<Toolkit>(toolkit)),
        torch_model_(std::make_shared<TorchModel>(model)) {}
  ~Model() {}

  void reload(const std::string &pool) {
    auto tmp = std::make_shared<Pool>(pool, this->key_);
    this->pool_.swap(tmp);
  }

  char *forward(char *user_features, size_t len, char **items, size_t size,
                float *scores) {
    size_t width = this->toolkit_->width();
    sample::Features *features = new sample::Features();
    features->ParseFromArray(user_features, len);
    int64_t *input = (int64_t *)calloc(size * width, sizeof(int64_t));
    sample::Features *feas;
    int64_t *ptr = nullptr;
    size_t index = 0;

    auto tmp_pool = this->pool_;
    for (size_t i = 0; i < size; i++) {
      ptr = &input[width * i];
      feas = tmp_pool->get_features(items[i]);
      if (feas == nullptr) {
        this->toolkit_->process(features, ptr);
      } else {
        this->toolkit_->process({features, feas}, ptr);
      }
    }
    this->torch_model_->forward(input, size, width, scores);
    free(input);
    delete features;
    const std::string &version = tmp_pool->get_version();
    char *tmp = (char *)malloc(version.size() + 1);
    memcpy(tmp, version.c_str(), version.size());
    return tmp;
  }

 private:
  std::string key_;
  std::shared_ptr<Pool> pool_;
  std::shared_ptr<Toolkit> toolkit_;
  std::shared_ptr<TorchModel> torch_model_;
};

#endif  // LONGMAN_MODEL