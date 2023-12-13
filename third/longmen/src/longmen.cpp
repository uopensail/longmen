#include "longmen.h"

#include "model.h"
#include "stdint.h"

void *longmen_new_model(char *path, int plen, char *key, int klen,
                        char *toolkit, int tlen, char *model, int mlen) {
  return new Model({path, size_t(plen)}, {key, size_t(klen)},
                   {toolkit, size_t(tlen)}, {model, size_t(mlen)});
}

void longmen_del_model(void *model) {
  if (model == nullptr) {
    return;
  }
  delete (Model *)model;
}

void longmen_forward(void *model, char *user_features, int len, char *items,
                     void *lens, int size, float *scores) {
  if (model == nullptr || user_features == nullptr || len == 0 ||
      items == nullptr || lens == nullptr || size == 0 || scores == nullptr) {
    return;
  }
  Model *m = (Model *)model;
  m->forward(user_features, len, (char **)items, (int64_t *)lens, size, scores);
}