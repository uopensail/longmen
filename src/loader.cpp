#include "loader.h"


loader::Record::Record(std::shared_ptr<luban::ToolKit> &toolkit, char *buffer, int len) {
    features_.ParseFromArray(buffer, len);
    toolkit->process(features_, keys_);
    auto features_map = features_.feature();
    auto iter = features_map.find(item_id_key);
    record_id = "";
    if (iter->second.bytes_list().value_size() > 0) {
        record_id = iter->second.bytes_list().value(0);
    }
}

inline std::string &loader::Record::get_id() {
    return record_id;
}

inline tensorflow::Features &loader::Record::get_features() {
    return features_;
}

inline Keys &loader::Record::get_keys() {
    return keys_;
}

loader::Record::~Record() {}


loader::Store::Store(std::string data_file, std::shared_ptr<luban::ToolKit> &toolkit) : toolkit_(toolkit), size_(0) {
    //这里是按照TensorFlow里面tfrecord的结构来读取的，读取的时候不做checksum检查
    u_int64_t max_size = 4096;
    char *buffer = (char *) calloc(1, max_size);
    int checksum_len = 4;
    char *tmp = (char *) calloc(1, checksum_len);
    std::ifstream reader(data_file, std::ios::in | std::ios::binary);
    if (!reader) {
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
        reader.read(tmp, checksum_len);
        if (len > max_size) {
            max_size = len + 1024;
            free(buffer);
            buffer = (char *) calloc(1, max_size);
        }
        reader.read(buffer, len);

        //footer, 数据的checksum，固定长度是4
        reader.read(tmp, checksum_len);

        auto item = new Record(toolkit_, buffer, len);
        if (item->get_id() == "") {
            continue;
        }
        pool_[item->get_id()] = item;
        size_++;
    }
    delete[]buffer;
    reader.close();
}

loader::Record *loader::Store::get(std::string &id) const {
    auto iter = pool_.find(id);
    if (iter == pool_.end()) {
        return nullptr;
    }
    return iter->second;
}

loader::Store::~Store() {
    for (auto &v: pool_) {
        delete v.second;
    }
    pool_.clear();
}


loader::Extractor::Extractor(std::shared_ptr<SlotsConfigure> &slot_conf, std::string data_file,
                             std::string luban_config_file) :
        slot_conf_(slot_conf),
        toolkit_(new luban::ToolKit(luban_config_file)),
        store_(new loader::Store(data_file, toolkit_)) {
}

loader::Extractor::~Extractor() {}

::KWWrapper *loader::Extractor::call(tensorflow::Features &user_features, ::Recalls &recalls) {
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

    return batch_keys;
}

std::shared_ptr<loader::Extractor> loader::create_extractor(std::shared_ptr<::GlobalConfigure> &config) {
    std::shared_ptr<loader::Extractor> extractor(new loader::Extractor(config->get_slot_conf(),
                                                                       config->get_loader_conf()->get_data_file(),
                                                                       config->get_loader_conf()->get_config_file()));
    return extractor;
}

