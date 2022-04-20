#include "longmen.h"
#include "rank.h"
#include "stata.h"

//新建模型
void *lm_create_model(char *config_file, char *model_file)
{
    std::shared_ptr<::GlobalConfigure> config(new ::GlobalConfigure(config_file));
    std::string mode_path = model_file == nullptr ? "" : std::string(model_file);
    if (config->get_model_conf()->type() == ::LRModel)
    {
        return new model::LR(config, mode_path);
    }
    else if (config->get_model_conf()->type() == ::FMModel)
    {
        return new model::FM(config, mode_path);
    }
    else if (config->get_model_conf()->type() == ::STFModel)
    {
        return new model::STF(config, mode_path);
    }
    else
    {
        return nullptr;
    }
}

//删除模型
void lm_release_model(void *model)
{
    if (model != nullptr)
    {
        model::Rank *m = (model::Rank *)model;
        delete m;
    }
}

//将TensorFlow::features的二进制数据转化成c++结构
void *lm_create_features(char *data, int len)
{
    tensorflow::Features *f = new tensorflow::Features();
    f->ParseFromArray(data, len);
    return f;
}

//删除TensorFlow::features的c++结构
void lm_release_features(void *features)
{
    delete (tensorflow::Features *)features;
}

//重新加载物料
void lm_reload(void *model, char *data_file)
{
    if (model != nullptr)
    {
        model::Rank *m = (model::Rank *)model;
        m->reload_extractor(data_file);
    }
}

//预测
void lm_predict(void *model, void *features, void *recalls, int len, void *result)
{
    if (model != nullptr)
    {
        Recalls tmp_recalls(len, "");
        Scores scores(len, {"", 0.0});
        float *ret = (float *)result;
        model::Rank *m = (model::Rank *)model;
        char *recalls_ptr = (char *)recalls;
        int size;
        for (int i = 0; i < len; i++)
        {
            size = strlen(recalls_ptr);
            tmp_recalls[i] = std::string(recalls_ptr, size);
            recalls_ptr += size + 1;
        }
        m->call(*((tensorflow::Features *)features), tmp_recalls, scores);
        for (int i = 0; i < len; i++)
        {
            ret[i] = scores[i].second;
        }
    }
}

//获得stat
void *lm_status()
{
    auto status = stata::Stata::getStata()->get_status();
    char *ret = (char *)calloc(status.size() + 1, 1);
    memcpy(ret, status.c_str(), status.size());
    return ret;
}