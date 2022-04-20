#include "config.h"

ModelConfigure::ModelConfigure(const std::shared_ptr<cpptoml::table> &table) : table_(table),
                                                                               type_((::ModelType) * (table_->get_as<int64_t>("type"))),
                                                                               path_(*(table_->get_as<std::string>("path"))),
                                                                               dim_(*(table_->get_as<int64_t>("dim")))
{
}

LoaderConfigure::LoaderConfigure(const std::shared_ptr<cpptoml::table> &table) : data_file_(*table->get_as<std::string>("data_path")),
                                                                                 luban_config_file_(*table->get_as<std::string>("config_path")) {}

SlotsConfigure::~SlotsConfigure()
{
    delete[] dims_;
    delete[] offset_;
}

SlotsConfigure::SlotsConfigure(const std::shared_ptr<cpptoml::table> &table)
{
    assert(table->contains("slots"));
    auto conf = table->get_array_of<int64_t>("slots");
    slots_ = (*conf).size();
    dims_ = new int[slots_];
    offset_ = new int[slots_];

    for (int i = 0; i < slots_; i++)
    {
        dims_[i] = conf->at(i);
        offset_[i] = (i == 0 ? 0 : offset_[i - 1] + dims_[i - 1]);
    }
}

STFModelConfigure::STFModelConfigure(const std::shared_ptr<cpptoml::table> &table) : ModelConfigure(table),
                                                                                     input_op_name_(*(table_->get_as<std::string>("input_op"))),
                                                                                     output_op_name_(*(table_->get_as<std::string>("output_op"))),
                                                                                     sparse_embedding_path_(*(table_->get_as<std::string>("sparse")))
{

    assert(path_.substr(path_.size() - 4) == ".zip");
}

GlobalConfigure::GlobalConfigure(std::string config_file)
{
    auto global_config = cpptoml::parse_file(config_file);

    // slot的配置
    auto slot_table = global_config->get_table("slot_config");
    assert(slot_table != nullptr);
    std::shared_ptr<SlotsConfigure> slot_conf(new SlotsConfigure(slot_table));
    slot_conf_ = slot_conf;

    // loader配置
    auto loader_table = global_config->get_table("loader_config");
    assert(loader_table != nullptr);
    std::shared_ptr<LoaderConfigure> loader_conf(new LoaderConfigure(loader_table));
    loader_conf_ = loader_conf;

    // model的配置
    auto model_table = global_config->get_table("model_config");
    assert(model_table != nullptr);
    std::shared_ptr<ModelConfigure> model_conf(new ModelConfigure(model_table));
    assert(model_table->contains("type"));
    int model_type = *(model_table->get_as<int>("type"));
    if (model_type == ::ModelType::LRModel)
    {
        model_conf_ = model_conf;
    }
    else if (model_type == ::ModelType::FMModel)
    {
        model_conf_ = model_conf;
    }
    else if (model_type == ::ModelType::STFModel)
    {
        std::shared_ptr<::STFModelConfigure> stf_model_conf(new STFModelConfigure(model_table));
        model_conf_ = stf_model_conf;
    }
    else
    {
        exit(-1);
    }
}