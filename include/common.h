#ifndef LONGMEN_COMMON_H
#define LONGMEN_COMMON_H

#include "config.h"
#include "cpptoml.h"
#include "mutils.h"
#include <assert.h>
#include <string>
#include <unordered_map>
#include <vector>

#define u_int64_t unsigned long long

//这里把uint64中上8位置位slot, 后面是key
#define get_slot_id(x) (x >> 56)

using Keys = std::vector<u_int64_t>;
using Weights = std::vector<float>;
using Score = std::pair<std::string, float>;
using Scores = std::vector<Score>;
using Recalls = std::vector<std::string>;

class KWWrapper
{
private:
    std::shared_ptr<SlotsConfigure> slot_conf_;
    size_t batch_size_;
    int32_t dims_;
    Keys all_keys_;
    std::unordered_map<u_int64_t, int32_t> filter_;
    Weights weighs_;
    std::vector<Keys> keys_;

private:
    void __add__(int index, Keys &keys);

public:
    KWWrapper() = delete;

    KWWrapper(const KWWrapper &) = delete;

    KWWrapper(const KWWrapper &&) = delete;

    KWWrapper(std::shared_ptr<SlotsConfigure> &slot_conf, size_t batch_size);

    ~KWWrapper();

    void add(int row, Keys &user_field_keys, Keys &item_field_keys, Keys &cross_field_keys);

    size_t &size();

    Keys &operator[](int row);

    Keys &get_all_keys();

    Weights &weights();

    float *get_weights(u_int64_t &key);
};

#endif // LONGMEN_COMMON_H
