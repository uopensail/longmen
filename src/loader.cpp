#include "loader.h"
#include "stata.h"

loader::Record::Record(std::shared_ptr<luban::ToolKit> &toolkit, const tensorflow::Features &features) :
        features_(features) {
    toolkit->process(features_, keys_);
    auto features_map = features_.feature();
    auto iter = features_map.find(item_id_key);
    record_id = "";
    if (iter->second.bytes_list().value_size() > 0) {
        record_id = iter->second.bytes_list().value(0);
    }
}

std::string &loader::Record::get_id() {
    return record_id;
}

tensorflow::Features &loader::Record::get_features() {
    return features_;
}

Keys &loader::Record::get_keys() {
    return keys_;
}

loader::Record::~Record() {
}


loader::Store::Store(std::string data_file, std::shared_ptr<luban::ToolKit> &toolkit) : toolkit_(toolkit), size_(0) {
    stata::Unit stata_unit("loader.Store");
    //这里是按照TensorFlow里面tfrecord的结构来读取的，读取的时候不做checksum检查
    u_int64_t max_size = 4096;
    char *buffer = (char *) calloc(max_size, 1);
    std::ifstream reader(data_file, std::ios::in | std::ios::binary);
    if (!reader) {
        stata_unit.MarkErr().End();
        exit(-1);
    }
    u_int64_t len;
    while (!reader.eof()) {
        reader.read((char *) &len, sizeof(u_int64_t));
        if (reader.gcount() == 0) {
            //读不到就说明文件结束了
            break;
        }
        //header的checksum，固定4个长度
        reader.ignore(4);
        if (len > max_size) {
            max_size = len + 1024;
            free(buffer);
            buffer = (char *) calloc(max_size, 1);
        }
        reader.read(buffer, len);

        //footer, 数据的checksum，固定长度是4
        reader.ignore(4);

        tensorflow::Example example;
        example.ParseFromArray(buffer, len);

        auto item = std::shared_ptr<Record>(new Record(toolkit_, example.features()));
        if (item->get_id() == "") {
            continue;
        }
        pool_[item->get_id()] = item;
        size_++;
    }
    free(buffer);
    reader.close();
    stata_unit.SetCount(size_).End();
}

std::shared_ptr<loader::Record> loader::Store::get(std::string &id) const {
    auto iter = pool_.find(id);
    if (iter == pool_.end()) {
        return nullptr;
    }
    return iter->second;
}

loader::Store::~Store() {
}


loader::Extractor::Extractor(std::shared_ptr<SlotsConfigure> &slot_conf, std::string data_file,
                             std::string luban_config_file) :
        slot_conf_(slot_conf),
        toolkit_(new luban::ToolKit(luban_config_file)),
        store_(new loader::Store(data_file, toolkit_)) {
}

loader::Extractor::~Extractor() {}

::KWWrapper *loader::Extractor::call(tensorflow::Features &user_features, ::Recalls &recalls) {
    stata::Unit stata_unit("Extractor.call");
    Keys user_filed_keys;
    toolkit_->process(user_features, user_filed_keys);
    ::KWWrapper *batch_keys = new ::KWWrapper(slot_conf_, recalls.size());
    for (size_t row = 0; row < recalls.size(); row++) {
        auto item = store_->get(recalls[row]);
        if (item == nullptr) {
            continue;
        }
        Keys bi_cross_keys;
        toolkit_->bicross_process(user_features, item->get_features(), bi_cross_keys);
        batch_keys->add(row, user_filed_keys, item->get_keys(), bi_cross_keys);
    }
    stata_unit.SetCount(recalls.size()).End();
    return batch_keys;
}