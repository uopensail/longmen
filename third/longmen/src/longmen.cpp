#include "longmen.h"
#include "pool_rows.h"
#include "model.h"
#include "stdint.h"


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

void longmen_user_rows_embedding_preforward_impl(longmen::TorchModel*torch_model, luban::Rows *rows, longmen::PoolRows* pool_cache) {
    auto toolkit = pool_cache->luban_toolkit();
    for (auto &group : toolkit->m_user_placer->m_groups) {

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
}

void model_inference_impl(longmen::TorchModel*torch_model, luban::Rows *user_rows, longmen::PoolRows* pool_cache, char **items, int64_t *lens,
               int size, float *scores) {
   
  auto toolkit = pool_cache->luban_toolkit();
  longmen::Input input(toolkit->m_groups.size());

  for (auto &group : toolkit->m_groups) {

    if (group.type == luban::DataType::kFloat32) {
      input.m_tensors[group.id] = torch::zeros({size, group.width}, torch::kFloat32);
      input.m_tensor_sizes[group.id] = {group.width};
    } else {
      auto input_embedding_meta = torch_model->get_input_embedding_meta(group.id);
      if (input_embedding_meta != nullptr) {
        input.m_tensors[group.id] = torch::zeros({size, group.width,input_embedding_meta->sum_dims}, torch::kFloat32);
        input.m_tensor_sizes[group.id] = {group.width,input_embedding_meta->sum_dims};
      } else {
        input.m_tensors[group.id] = torch::zeros({size, group.width}, torch::kInt64);
        input.m_tensor_sizes[group.id] = {group.width};

      }
      
    }
  }

  char *data = nullptr;
  BitMap* not_found_bitmap = new_bitmap(size);
  for (size_t i = 0; i < size; i++) {
    // copy user processed features
    for (auto &group : toolkit->m_user_placer->m_groups) {
      data = user_rows->operator[](group.index)->m_data;
      auto input_tensor = &input.m_tensors[group.id];
      torch::Tensor tensor = torch::from_blob(data, input.m_tensor_sizes[group.id],
                        input_tensor->scalar_type());
      input_tensor->index_put_({int(i)}, tensor);
    }

    // get item processed features
    auto item_rows = pool_cache->get(std::string{items[i], size_t(lens[i])});
    if (item_rows == nullptr) {
      set_bitmap(not_found_bitmap, i);
      continue;
    }

    for (auto &group : toolkit->m_item_placer->m_groups) {
      data = item_rows->m_rows[group.index]->m_data;
      auto input_tensor = &input.m_tensors[group.id];
      torch::Tensor tensor = torch::from_blob(data, input.m_tensor_sizes[group.id],
                        input_tensor->scalar_type());
      
      input_tensor->index_put_({int(i)}, tensor);
    }
  }
  //auto begin =  std::chrono::high_resolution_clock::now();
  torch_model->torch_forward(input, scores);
  //auto end =  std::chrono::high_resolution_clock::now();
  //std::cout << "torch_forward Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;

  for (int i=0; i< size; i++) {
    if (check_bitmap(not_found_bitmap,i)) {
        scores[i] = -1.0;
    }
  }
  free_bitmap(not_found_bitmap);             
}

void* longmen_new_pool_rows(const char* pool_file,const char* lua_file, const char* luban_file, void * model_ptr){
    auto ptr = new longmen::PoolRows(
        std::string(pool_file), std::string(lua_file),
        std::string(luban_file), (longmen::TorchModel*)model_ptr);
    return ptr;
}
void longmen_delete_pool_rows(void* ptr) {

    longmen::PoolRows* pool_rows = (longmen::PoolRows*)ptr;
    if (pool_rows != nullptr) {
        delete pool_rows;
    }
}
void *new_longmen_torch_model(const char* model_file,const char* model_meta) {
    auto ptr = new longmen::TorchModel(
        std::string(model_file), std::string(model_meta));
    return ptr;
}
void delete_longmen_torch_model(void *ptr) {
   longmen::TorchModel* model = (longmen::TorchModel*)ptr;
    if (model != nullptr) {
        delete model;
    }
}

void longmen_user_rows_embedding_preforward(void *model_ptr, void *user_rows_ptr, void* pool_rows_ptr) {
    longmen::TorchModel* model  = (longmen::TorchModel*)model_ptr;    
    luban::Rows* user_rows = (luban::Rows*)user_rows_ptr;
    longmen::PoolRows* pool_rows = (longmen::PoolRows*)pool_rows_ptr; 
    longmen_user_rows_embedding_preforward_impl(model, user_rows, pool_rows);   
}

void longmen_torch_model_inference(void*model_ptr,void* user_rows_ptr, void* pool_rows_ptr, char *items_ptr,
                     void *lens_ptr, int size, float *scores) {
    longmen::TorchModel* model  = (longmen::TorchModel*)model_ptr;    
    luban::Rows* user_rows = (luban::Rows*)user_rows_ptr;
    longmen::PoolRows* pool_rows = (longmen::PoolRows*)pool_rows_ptr;  
    char** item_arr = (char**)items_ptr;
    int64_t * lens =  (int64_t *)lens_ptr;
  
    model_inference_impl(model, user_rows, pool_rows, item_arr,lens, size, scores);  

}
