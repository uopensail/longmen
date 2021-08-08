#ifndef LONGMEN_COMMON_H
#define LONGMEN_COMMON_H


#include <assert.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutils.h>
#include "config.h"
#include "cpptoml.h"
#include "mutils.h"

#define u_int64_t unsigned long long
#define get_slot_id(x) (x >> 56)


using Keys = std::vector<u_int64_t>;
using Weights = std::vector<float>;
using Score = std::pair<std::string, float>;
using Scores = std::vector<Score>;
using Recalls = std::vector<std::string>;

static bool __score_cmp__(const Score &a, const Score &b);

class KWWrapper {
private:
    std::shared_ptr<SlotsConfig> slot_conf_;
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

    KWWrapper(std::shared_ptr<SlotsConfig> &slot_conf, size_t batch_size);

    ~KWWrapper();

    void add(int row, Keys &user_field_keys, Keys &item_field_keys, Keys &cross_field_keys);

    size_t &size();

    inline Keys &operator[](int &row);

    inline Keys &get_all_keys();

    Weights &weights();

    float *get_weights(u_int64_t &key);
};


#endif //LONGMEN_COMMON_H
