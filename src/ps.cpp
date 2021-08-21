#include "ps.h"
#include "stata.h"

int ps::__sort__(const void *a, const void *b) {
    if (((u_int64_t *) a)[0] > ((u_int64_t *) b)[0]) {
        return 1;
    } else if (((u_int64_t *) a)[0] == ((u_int64_t *) b)[0]) {
        return 0;
    } else {
        return -1;
    }
}

float *ps::binary_search(const char *value, const size_t &n, const size_t &size, u_int64_t &key) {
    stata::Unit stata_unit("ps.binary_search");
    long long low = 0, high = n - 1, middle;
    while (low <= high) {
        middle = (low + high) / 2;

        if (key == ((u_int64_t *) (value + middle * size))[0]) {
            stata_unit.End();
            return (float *) (value + middle * size + sizeof(u_int64_t));
        } else if (key > ((u_int64_t *) (value + middle * size))[0]) {
            low = middle + 1;
        } else {
            high = middle - 1;
        }
    }
    stata_unit.MarkErr().End();
    return nullptr;
}


ps::Memory::Memory(std::shared_ptr<SlotsConfigure> &slot_conf, std::string path) :
        slot_conf_(slot_conf),
        path_(path) {
    stata::Unit stata_unit("Memory.Create");
    std::ifstream reader(path_, std::ios::in | std::ios::binary);
    if (!reader) {
        stata_unit.MarkErr();
        stata_unit.End();
        return;
    }

    reader.read((char *) &slots_, sizeof(int));

    //slot个数的检查
    assert(slots_ == slot_conf_->get_slots());
    dims_ = new int[slots_];
    data_ = new char *[slots_];
    key_count_ = new long[slots_];

    reader.read((char *) dims_, sizeof(int) * slots_);
    reader.read((char *) key_count_, sizeof(long) * slots_);
    size_ = new size_t[slots_];

    size_t size;
    //每一个slot的dim的检查
    for (int i = 0; i < slots_; i++) {
        size_[i] = 0;
        assert(dims_[i] == slot_conf_->get_dim(i));
        size = sizeof(float) * dims_[i] + sizeof(u_int64_t);
        data_[i] = new char[key_count_[i] * size];
    }

    u_int64_t key;
    int slot;

    while (reader.read((char *) &key, sizeof(u_int64_t))) {
        slot = get_slot_id(key);
        size = sizeof(float) * dims_[slot];
        memcpy(data_[slot] + size_[slot], &key, sizeof(u_int64_t));
        reader.read((char *) (data_[slot] + size_[slot] + sizeof(u_int64_t)), size);
        size_[slot] += size + sizeof(u_int64_t);
    }
    reader.close();

    size_t total_count = 0;
    //sort
    for (int i = 0; i < slots_; i++) {
        size = sizeof(float) * dims_[slot] + sizeof(u_int64_t);
        qsort(data_[i], key_count_[i], size, __sort__);
        size_[i] = size;
        total_count += key_count_[i];
    }
    stata_unit.SetCount(total_count).End();

}

ps::Memory::~Memory() {
    for (int i = 0; i < slot_conf_->get_slots(); i++) {
        if (data_[i] != nullptr) {
            delete[] data_[i];
            data_[i] = nullptr;
        }
    }
    delete[] data_;
    delete[] key_count_;
    delete[] dims_;
    delete[] size_;
}


void ps::Memory::pull(KWWrapper &batch_kw) {
    stata::Unit stata_unit("Memory.pull");
    auto &weights = batch_kw.weights();
    auto &all_keys = batch_kw.get_all_keys();
    int slot;
    size_t offset = 0;
    float *ptr;
    for (size_t i = 0; i < all_keys.size(); i++) {
        slot = get_slot_id(all_keys[i]);
        ptr = binary_search(data_[slot], key_count_[slot], size_[slot], all_keys[i]);
        if (ptr == nullptr) {
            offset += slot_conf_->get_dim(slot);
            continue;
        }
        for (size_t j = 0; j < slot_conf_->get_dim(slot); j++) {
            weights[offset] = ptr[j];
            offset++;
        }
    }
    stata_unit.End();
}