#ifndef LONGMEN_LOADER_H
#define LONGMEN_LOADER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include "feature.pb.h"
#include "toolkit.h"
#include "common.h"


namespace loader {
    const std::string item_id_key = "d_s_id";

    class Record {
    private:
        tensorflow::Features features_;
        Keys keys_;
        std::string record_id;
    public:
        Record() = delete;

        Record(const Record &) = delete;

        Record(const Record &&) = delete;

        Record(std::shared_ptr<luban::ToolKit> &toolkit, char *buffer, int len);

        inline std::string &get_id();

        inline tensorflow::Features &get_features();

        inline Keys &get_keys();

        ~Record();
    };

    class Store {
    private:
        std::shared_ptr<luban::ToolKit> toolkit_;
        int size_;
        std::unordered_map<std::string, Record *> pool_;


    public:
        Store() = delete;

        Store(const Store &) = delete;

        Store(const Store &&) = delete;

        Store(std::string data_file, std::shared_ptr<luban::ToolKit> &toolkit);

        Record *get(std::string &id) const;

        ~Store();
    };

    class Extractor {
    private:
        std::shared_ptr<SlotsConfig> slot_conf_;
        std::shared_ptr<luban::ToolKit> toolkit_;
        std::shared_ptr<Store> store_;
    public:
        Extractor() = delete;

        Extractor(std::shared_ptr<SlotsConfig> &slot_conf, std::string data_file, std::string luban_config_file);

        ~Extractor();

        ::KWWrapper *call(tensorflow::Features &user_features, ::Recalls &recalls);
    };
}//namespace loader

#endif //LONGMEN_LOADER_H
