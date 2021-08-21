#ifndef LONGMEN_PS_H
#define LONGMEN_PS_H

#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "common.h"

namespace ps {

    int __sort__(const void *a, const void *b);

    float *binary_search(const char *value, const size_t &n, const size_t &size, u_int64_t &key);

    class Memory {
    private:
        std::shared_ptr<SlotsConfigure> slot_conf_;
        std::string path_;
        char **data_;
        long *key_count_;
        int *dims_;
        int slots_;
        size_t *size_;

    public:
        Memory() = delete;

        Memory(const Memory &) = delete;

        Memory(const Memory &&) = delete;

        Memory(std::shared_ptr<SlotsConfigure> &slot_conf, std::string path);

        ~Memory();

        void pull(KWWrapper &batch_kw);
    };

}  // namespace ps
#endif //LONGMEN_PS_H
