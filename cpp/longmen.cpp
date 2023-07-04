#include "longmen.h"

#include "model.hpp"

void *longmen_new(char *config_path, char *data_path, char *model_path,
                  char *key) {
  return new Model(config_path, data_path, model_path, key);
}

void longmen_release(void *ptr) {
  if (ptr == nullptr) {
    return;
  }
  delete (Model *)ptr;
}

void longmen_reload(void *ptr, char *data_path) {
  Model *model = (Model *)ptr;
  model->reload(data_path);
}

char *longmen_forward(void *ptr, char *user_features, int len, void *items,
                      int size, float *scores) {
  Model *model = (Model *)ptr;
  return model->forward(user_features, len, (char **)items, size, scores);
}