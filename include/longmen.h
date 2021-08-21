#ifndef LONGMEN_LONGMEN_H
#define LONGMEN_LONGMEN_H
#ifdef __cplusplus
extern "C" {
#endif

//新建模型
void *lm_create_model(char *config_file, char *model_file);

//删除模型
void lm_release_model(void *model);

//将TensorFlow::features的二进制数据转化成c++结构
void *lm_create_features(char *data, int len);

//删除TensorFlow::features的c++结构
void lm_release_features(void *features);

//重新加载物料
void lm_reload(void *model, char *data_file);

//获得stat
void* lm_status();

//预测
void lm_predict(void *model, void *features, void *recalls, int len, void *result);

#ifdef __cplusplus
} /* end extern "C"*/
#endif

#endif //LONGMEN_LONGMEN_H
