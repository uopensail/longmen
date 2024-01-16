#ifndef LONGMAN_POOL_ROWS_H
#define LONGMAN_POOL_ROWS_H

#pragma once

#include "toolkit.h"
#include "model.h"
#include <filesystem>
#include <torch/script.h>
#include <vector>

namespace longmen {

class PoolRows{
public:
    PoolRows() = delete;
    PoolRows(const std::string& pool_file,const std::string& lua_cfg_file, 
        const std::string& luban_cfg_file, TorchModel* torch_model);
    ~PoolRows()=default;

    std::shared_ptr<luban::Rows> get(const std::string& item_id);
    std::shared_ptr<luban::Toolkit> luban_toolkit() {return m_toolkit;}
private:
    std::shared_ptr<luban::Toolkit> m_toolkit;
    std::unordered_map<std::string,  std::shared_ptr<luban::Rows> > m_pool;
};

}//namespace longmen


#endif // LONGMAN_POOL_ROWS_H