#include "pool_rows.h"
#include "sample.h"
#include <fstream>
#include <iostream>

namespace longmen{

PoolRows::PoolRows(const std::string& pool_file, const std::string& lua_cfg_file,
    const std::string& luban_cfg_file, TorchModel* torch_model) {
  m_toolkit = std::make_shared<luban::Toolkit>(luban_cfg_file);
  std::ifstream reader(std::string(pool_file), std::ios::in);
  if (!reader) {
    std::cerr << "read pool_file file: " << pool_file << " error" << std::endl;
    exit(-1);
  }
  std::string line;
  auto s_toolkit = std::make_shared<sample_luban::SamplePreProcessor>(std::string(lua_cfg_file));
  while (std::getline(reader, line)) {
    auto ss = split(line,'\t');
    if (ss.size() != 2) {
        continue;
    }
    auto item_id = ss[0];
    auto features = std::make_shared<luban::Features>(ss[1]);
    if (features == nullptr) {
        continue;
    }
    auto preprocess_feature = s_toolkit->process_item_featrue(features);
    if (preprocess_feature == nullptr) {
        continue;
    }
    auto rows = m_toolkit->process_item(preprocess_feature);
    if (rows == nullptr) {
        continue;
    }

    for (auto &group : m_toolkit->m_item_placer->m_groups) {
        
        auto input_embedding_meta = torch_model->get_input_embedding_meta(group.id);
        if (input_embedding_meta != nullptr) {
            
            auto oldrow = rows->operator[](group.index);
            assert(oldrow->m_type == luban::DataType::kInt64);
            int64_t *ptr = (int64_t *)oldrow->m_data;

            auto row = std::make_shared<luban::Row>(
                luban::DataType::kFloat32, group.width*input_embedding_meta->sum_dims
            );
            //pre embedding forward 
            torch::Tensor input_keys = torch::from_blob(ptr, {1, oldrow->m_cols},
                         torch::kInt64);
            torch::Tensor output_tensor = torch_model->embedding_forward(input_embedding_meta, input_keys);
            assert(output_tensor.numel() == group.width*input_embedding_meta->sum_dims);
            
            float* tensor_data = output_tensor.data_ptr<float>();
            memcpy(row->m_data, tensor_data, sizeof(float)* row->m_cols);
            rows->m_rows[group.index] = row;
        }
    }
    m_pool[item_id] = rows;
    //std::cout << "item_id: " << item_id << "I" <<std::endl;

  }
  reader.close();
}

std::shared_ptr<luban::Rows> PoolRows::get(const std::string& item_id) {
    auto it = m_pool.find(item_id);
    if (it != m_pool.end()) {
        return it->second;
    }
    return nullptr;
}
}