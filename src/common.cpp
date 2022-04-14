#include "common.h"

void KWWrapper::__add__(int index, Keys &keys)
{
    assert(index >= 0 and index < batch_size_);
    keys_[index].insert(keys_[index].end(), keys.begin(), keys.end());
    auto iter = filter_.begin();
    for (size_t i = 0; i < keys.size(); i++)
    {
        iter = filter_.find(keys[i]);
        if (iter == filter_.end())
        {
            filter_[keys[i]] = dims_;
            all_keys_.push_back(keys[i]);
            dims_ += slot_conf_->get_dim(get_slot_id(keys[i]));
        }
    }
}

KWWrapper::KWWrapper(std::shared_ptr<SlotsConfigure> &slot_conf, size_t batch_size) : slot_conf_(slot_conf),
                                                                                      batch_size_(batch_size),
                                                                                      dims_(0)
{
    keys_.resize(batch_size_);
}

KWWrapper::~KWWrapper() {}

void KWWrapper::add(int row, Keys &user_field_keys, Keys &item_field_keys, Keys &cross_field_keys)
{
    keys_[row].reserve(user_field_keys.size() + item_field_keys.size() + cross_field_keys.size());
    __add__(row, user_field_keys);
    __add__(row, item_field_keys);
    __add__(row, cross_field_keys);
}

size_t &KWWrapper::size() { return batch_size_; }

Keys &KWWrapper::get_all_keys() { return all_keys_; }

Weights &KWWrapper::weights()
{
    if (weighs_.size() == 0)
    {
        weighs_.resize(dims_, 0.0);
    }
    return weighs_;
}

Keys &KWWrapper::operator[](int row)
{
    return keys_[row];
}

float *KWWrapper::get_weights(u_int64_t &key)
{
    auto iter = filter_.begin();
    iter = filter_.find(key);
    if (iter == filter_.end())
    {
        return nullptr;
    }
    auto ptr = weighs_.data();
    return &ptr[iter->second];
}
