#ifndef LONGMEN_LOADER_H
#define LONGMEN_LOADER_H

#include "common.h"
#include "feature.pb.h"
#include "toolkit.h"
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace loader
{
    const std::string item_id_key = "d_s_id";

    class Record
    {
    private:
        tensorflow::Features features_;
        Keys keys_;
        std::string record_id;

    public:
        Record() = delete;

        Record(const Record &) = delete;

        Record(const Record &&) = delete;

        Record(std::shared_ptr<luban::ToolKit> &toolkit, const tensorflow::Features &features);

        std::string &get_id();

        tensorflow::Features &get_features();

        Keys &get_keys();

        ~Record();
    };

    class Store
    {
    private:
        std::shared_ptr<luban::ToolKit> toolkit_;
        int size_;
        std::unordered_map<std::string, std::shared_ptr<Record>> pool_;

    public:
        Store() = delete;

        Store(const Store &) = delete;

        Store(const Store &&) = delete;

        Store(std::string data_file, std::shared_ptr<luban::ToolKit> &toolkit);

        std::shared_ptr<loader::Record> get(std::string &id) const;

        ~Store();
    };

    class Extractor
    {
    private:
        std::shared_ptr<SlotsConfigure> slot_conf_;
        std::shared_ptr<luban::ToolKit> toolkit_;
        std::shared_ptr<Store> store_;

    public:
        Extractor() = delete;

        Extractor(const Extractor &) = delete;

        Extractor(const Extractor &&) = delete;

        Extractor(std::shared_ptr<SlotsConfigure> &slot_conf, std::string data_file, std::string luban_config_file);

        ~Extractor();

        ::KWWrapper *call(tensorflow::Features &user_features, ::Recalls &recalls);
    };

} // namespace loader

#endif // LONGMEN_LOADER_H
