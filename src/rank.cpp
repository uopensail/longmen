#include "rank.h"
#include <zip/zip.h>
#include <filesystem>
#include "stata.h"

model::Rank::Rank(const std::shared_ptr<::GlobalConfigure> &config) :
        global_config_(config),
        extractor_(new loader::Extractor(global_config_->get_slot_conf(),
                                         global_config_->get_loader_conf()->get_data_file(),
                                         global_config_->get_loader_conf()->get_config_file())) {}


void model::Rank::reload_extractor(std::string path) {
    std::shared_ptr<loader::Extractor> new_extractor_(new loader::Extractor(global_config_->get_slot_conf(),
                                                                            path,
                                                                            global_config_->get_loader_conf()->get_config_file()));
    extractor_ = new_extractor_;
}

model::LR::LR(const std::shared_ptr<::GlobalConfigure> &config, std::string path) :
        Rank(config) {
    assert(config->get_slot_conf()->get_slots() > 0);
    for (int i = 0; i < config->get_slot_conf()->get_slots(); i++) {
        assert(config->get_slot_conf()->get_dim(i) == 1);
    }
    std::string model_path = path;
    //没有输入新文件就用默认的配置文件
    if (model_path == "") {
        model_path = config->get_model_conf()->get_path();
    }
    //加载参数
    ps_client_ = std::shared_ptr<ps::Memory>(new ps::Memory(config->get_slot_conf(),
                                                            model_path));
}


model::FM::FM(const std::shared_ptr<::GlobalConfigure> &config, std::string path) : Rank(config) {
    assert(config->get_slot_conf()->get_slots() > 0);
    dim_ = config->get_slot_conf()->get_dim(0);
    for (int i = 0; i < config->get_slot_conf()->get_slots(); i++) {
        assert(config->get_slot_conf()->get_dim(i) == dim_);
    }
    std::string model_path = path;
    //没有输入新文件就用默认的配置文件
    if (model_path == "") {
        model_path = config->get_model_conf()->get_path();
    }
    //加载参数
    ps_client_ = std::shared_ptr<ps::Memory>(new ps::Memory(config->get_slot_conf(),
                                                            model_path));
}

void model::LR::call(tensorflow::Features &user_features, Recalls &recalls, Scores &scores) {
    stata::Unit stata_unit("lr.call");
    stata_unit.SetCount(recalls.size());
    auto batch_kw = extractor_->call(user_features, recalls);
    ps_client_->pull(*batch_kw);
    auto weights = batch_kw->weights();
    int row = 0;
    float *ptr, score;
    for (auto &id: recalls) {
        scores[row].first = id;
        auto &keys = (*batch_kw)[row];
        for (auto &key: keys) {
            ptr = batch_kw->get_weights(key);
            if (ptr != nullptr) {
                score += ptr[0];
            }
        }
        scores[row].second = sigmoid(score);
        row++;
    }
    delete batch_kw;
    stata_unit.End();
}

void model::FM::call(tensorflow::Features &user_features, Recalls &recalls, Scores &scores) {
    stata::Unit stata_unit("fm.call");
    stata_unit.SetCount(recalls.size());
    auto batch_kw = extractor_->call(user_features, recalls);
    ps_client_->pull(*batch_kw);
    auto weights = batch_kw->weights();
    int row = 0;
    float *ptr, score;
    float *vec_sum = (float *) calloc(dim_, sizeof(float));
    for (auto &id: recalls) {
        memset(vec_sum, 0, sizeof(float) * dim_);
        scores[row].first = id;
        auto &keys = (*batch_kw)[row];
        for (auto &key: keys) {
            ptr = batch_kw->get_weights(key);
            if (ptr != nullptr) {
                vec_add(vec_sum, ptr, dim_);
                score -= vec_square(ptr, dim_);
            }
        }
        score += vec_square(vec_sum, dim_);
        scores[row].second = sigmoid(score * 0.5);
        row++;
    }
    free(vec_sum);
    delete batch_kw;
    stata_unit.End();
}

model::STF::STF(
        const std::shared_ptr<::GlobalConfigure> &config, std::string path) :
        Rank(config),
        dims_(std::dynamic_pointer_cast<STFModelConfigure>(config->get_model_conf())->get_dim()),
        input_op_name_(std::dynamic_pointer_cast<STFModelConfigure>(config->get_model_conf())->get_input_op()),
        output_op_name_(std::dynamic_pointer_cast<STFModelConfigure>(config->get_model_conf())->get_output_op()) {

    std::string model_path = path;
    //没有输入新文件就用默认的配置文件
    if (model_path == "") {
        model_path = std::dynamic_pointer_cast<STFModelConfigure>(
                config->get_model_conf())->get_path();
    }

    //这里的文件是zip压缩的
    std::__fs::filesystem::path p(model_path);
    model_dir_ = p.parent_path();
    //解压缩文件
    zip_extract(model_path.c_str(), model_dir_.c_str(), nullptr, nullptr);
    int dim = 0;
    for (int i = 0; i < config->get_slot_conf()->get_slots(); i++) {
        dim += config->get_slot_conf()->get_dim(i);
    }
    //维度检查
    assert(dim == dims_);

    ps_client_ = std::shared_ptr<ps::Memory>(new ps::Memory(config->get_slot_conf(),
                                                            model_dir_ + "/" +
                                                            std::dynamic_pointer_cast<STFModelConfigure>(
                                                                    config->get_model_conf())->get_sparse_path()));
    TF_SessionOptions *options = TF_NewSessionOptions();
    TF_Status *status = TF_NewStatus();
    TF_Buffer *buffer = TF_NewBuffer();
    graph_ = TF_NewGraph();
    session_ = TF_LoadSessionFromSavedModel(options, nullptr, model_dir_.c_str(),
                                            &tags, 1, graph_, buffer, status);
    TF_DeleteSessionOptions(options);
    TF_DeleteStatus(status);
    TF_DeleteBuffer(buffer);

}

model::STF::~STF() {
    if (graph_ != nullptr) {
        TF_DeleteGraph(graph_);
    }
    TF_Status *status = TF_NewStatus();
    if (session_ != nullptr) {
        TF_CloseSession(session_, status);
        TF_DeleteSession(session_, status);
    }
    TF_DeleteStatus(status);
    //删除文件夹
    remove(model_dir_.c_str());
}

void model::STF::call(tensorflow::Features &user_features, Recalls &recalls, Scores &scores) {
    stata::Unit stata_unit("fm.call");
    stata_unit.SetCount(recalls.size());
    auto data = extractor_->call(user_features, recalls);
    ps_client_->pull(*data);

    //生成数据
    float *input = (float *) calloc(1, sizeof(float) * dims_ * recalls.size()), *ptr, *dst;
    int slot, offset, dim;
    for (int i = 0; i < recalls.size(); i++) {
        scores[i].first = recalls[i];
        dst = &(input[i * dims_]);
        auto &keys = (*data)[i];
        for (auto &key: keys) {
            ptr = data->get_weights(key);
            if (ptr != nullptr) {
                slot = get_slot_id(key);
                offset = global_config_->get_slot_conf()->get_offset(slot);
                dim = global_config_->get_slot_conf()->get_dim(slot);
                for (int j = 0; j < dim; j++) {
                    dst[offset + j] += ptr[j];
                }
            }
        }
    }

    int64_t input_dims[] = {int64_t(recalls.size()), dims_};
    TF_Status *status = TF_NewStatus();
    TF_Output *tf_input = new TF_Output{TF_GraphOperationByName(graph_, input_op_name_.c_str()), 0};
    TF_Output *tf_output = new TF_Output{TF_GraphOperationByName(graph_, output_op_name_.c_str()), 0};

    TF_Tensor **input_values = new TF_Tensor *{nullptr};
    TF_Tensor **output_values = new TF_Tensor *{nullptr};

    TF_Tensor *input_tensor = TF_NewTensor(TF_FLOAT, input_dims, 2, input,
                                           input_dims[0] * input_dims[1] * sizeof(float), &NoneOpDeAllocator,
                                           nullptr);

    input_values[0] = input_tensor;
    TF_SessionRun(session_, nullptr,
                  tf_input, input_values, 1,
                  tf_output, output_values, 1,
                  nullptr, 0, nullptr, status);
    if (TF_GetCode(status) == TF_OK) {
        float *buff = (float *) TF_TensorData(output_values[0]);
        assert(TF_NumDims(output_values[0]) == 2);
        assert(TF_Dim(output_values[0], 0) == recalls.size());
        //输出要是2维的，0的概率和1的概率
        assert(TF_Dim(output_values[0], 1) == 2);
        for (int i = 0; i < recalls.size(); i++) {
            scores[i].second = buff[i * 2 + 1];
        }

    }
    //释放数据
    delete data;
    free(input);
    TF_DeleteStatus(status);
    delete tf_input;
    delete tf_output;
    TF_DeleteTensor(input_tensor);
    TF_DeleteTensor(output_values[0]);
    delete input_values;
    delete output_values;
    stata_unit.End();
}