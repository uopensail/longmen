#ifndef LONGMEN_CONFIG_H
#define LONGMEN_CONFIG_H

#include "cpptoml.h"

//定义参数服务器的类型
enum PSType
{
    Empty = 0,  //错误
    Memory = 1, //数据量比较小，放在内存
};

//定义模型的类型
enum ModelType
{
    ERRModel = 0,
    LRModel = 1,  // lr模型
    FMModel = 2,  // fm模型
    STFModel = 3, //带有sparse embedding的TensorFlow模型
};

class LoaderConfigure
{
private:
    std::string data_file_;
    std::string luban_config_file_;

public:
    LoaderConfigure() = delete;

    LoaderConfigure(const LoaderConfigure &) = delete;

    LoaderConfigure(const LoaderConfigure &&) = delete;

    LoaderConfigure(const std::shared_ptr<cpptoml::table> &table);

    ~LoaderConfigure() {}

    std::string &get_data_file() { return data_file_; }

    std::string &get_config_file() { return luban_config_file_; }
};

//定义slots的信息
class SlotsConfigure
{
private:
    int slots_;
    int *dims_;
    int *offset_;

public:
    SlotsConfigure() = delete;

    SlotsConfigure(const SlotsConfigure &) = delete;

    SlotsConfigure(const SlotsConfigure &&) = delete;

    ~SlotsConfigure();

    SlotsConfigure(const std::shared_ptr<cpptoml::table> &table);

    int &get_dim(int slot)
    {
        return dims_[slot];
    }

    int &get_slots() { return slots_; }

    inline int &get_offset(int slot) { return offset_[slot]; }
};

class ModelConfigure
{
protected:
    std::shared_ptr<cpptoml::table> table_;
    ModelType type_;
    std::string path_;
    int dim_;

public:
    ModelConfigure() = delete;

    ModelConfigure(const ModelConfigure &) = delete;

    ModelConfigure(const ModelConfigure &&) = delete;

    ModelConfigure(const std::shared_ptr<cpptoml::table> &table);

    virtual ~ModelConfigure() {}

    ModelType &type() { return type_; }

    std::string &get_path() { return path_; }

    int &get_dim() { return dim_; }
};

class STFModelConfigure : public ModelConfigure
{
private:
    std::string input_op_name_;
    std::string output_op_name_;
    std::string sparse_embedding_path_;

public:
    STFModelConfigure() = delete;

    STFModelConfigure(const STFModelConfigure &) = delete;

    STFModelConfigure(const STFModelConfigure &&) = delete;

    STFModelConfigure(const std::shared_ptr<cpptoml::table> &table);

    virtual ~STFModelConfigure() {}

    int &get_dim() { return dim_; }

    std::string &get_input_op() { return input_op_name_; }

    std::string &get_output_op() { return output_op_name_; }

    std::string &get_path() { return path_; }

    std::string &get_sparse_path() { return sparse_embedding_path_; }
};

class GlobalConfigure
{
private:
    std::shared_ptr<SlotsConfigure> slot_conf_;
    std::shared_ptr<LoaderConfigure> loader_conf_;
    std::shared_ptr<ModelConfigure> model_conf_;

public:
    GlobalConfigure() = delete;

    GlobalConfigure(const GlobalConfigure &) = delete;

    GlobalConfigure(const GlobalConfigure &&) = delete;

    GlobalConfigure(std::string config_file);

    ~GlobalConfigure() {}

    std::shared_ptr<SlotsConfigure> &get_slot_conf()
    {
        return slot_conf_;
    }

    std::shared_ptr<LoaderConfigure> &get_loader_conf()
    {
        return loader_conf_;
    }

    std::shared_ptr<ModelConfigure> &get_model_conf()
    {
        return model_conf_;
    }
};

#endif // LONGMEN_CONFIG_H
