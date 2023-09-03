#include "longmen.h"

#include "model.h"

void *longmen_new_model(char *toolkit, int tlen, char *model, int mlen) {
  return new Model({toolkit, size_t(tlen)}, {model, size_t(mlen)});
}

void longmen_del_model(void *model) {
  if (model == nullptr) {
    return;
  }
  delete (Model *)model;
}

void *longmen_new_pool(char *path, int plen, char *key, int klen) {
  Pool *pool = new Pool({path, size_t(plen)}, {key, size_t(klen)});
  pool->load();
  return pool;
}

void longmen_del_pool(void *pool) {
  if (pool == nullptr) {
    return;
  }
  delete (Pool *)pool;
}

void longmen_forward(void *model, void *pool, char *user_features, int len,
                     void *items, int *lens, int size, float *scores) {
  if (model == nullptr || pool == nullptr || user_features == nullptr ||
      len == 0 || items == nullptr || lens == nullptr || size == 0 ||
      scores == nullptr) {
    return;
  }
  Model *m = (Model *)model;
  m->forward((Pool *)pool, user_features, len, (char **)items, lens, size,
             scores);
}