#include "config.h"

PSConfigure::PSConfigure(const std::shared_ptr<cpptoml::table> &table) : table_(table), type_(Empty) {
    assert(table_->contains("type"));
    int ps_type = *(table_->get_as<int>("type"));
    switch (ps_type) {
        case 1:
            type_ = Memory;
            break;
        case 2:
            type_ = RocksDB;
            break;
        case 3:
            type_ = RemoteGRPC;
            break;
        case 4:
            type_ = RemoteGRPCShard;
            break;
        default:
            exit(-1);
    }
}

RemoteGRPCShardPSConfigure::RemoteGRPCShardPSConfigure(const std::shared_ptr<cpptoml::table> &table) :
        PSConfigure(table),
        shards_(*table_->get_as<int>("shards")),
        timeout_(*table_->get_as<int>("timeout")) {
    assert(shards_ > 0);
    hosts_.resize(shards_);
    auto hosts = table->get_array_of<std::string>("hosts");
    assert((*hosts).size() == shards_);
    for (int i = 0; i < shards_; i++) {
        hosts_[i] = hosts->at(i);
    }
}

LoaderConfigure::LoaderConfigure(const std::shared_ptr<cpptoml::table> &table) :
        data_file_(*table->get_as<std::string>("data_path")),
        luban_config_file_(*table->get_as<std::string>("config_path")) {}

SlotsConfigure::~SlotsConfigure() {
    delete[]dims_;
    delete[]space_;
    delete[]value_space_;
    delete[]offset_;
}

SlotsConfigure::SlotsConfigure(const std::shared_ptr<cpptoml::table> &table) {
    assert(table->contains("slots"));
    auto conf = table->get_array_of<int>("slots");
    slots_ = (*conf).size();
    dims_ = new int[slots_];
    value_space_ = new int[slots_];
    space_ = new int[slots_];
    offset_ = new int[slots_];

    for (int i = 0; i < slots_; i++) {
        dims_[i] = conf->at(i);
        value_space_[i] = sizeof(float) * dims_[i];
        space_[i] = sizeof(u_int64_t) + value_space_[i];
        offset_[i] = (i == 0 ? 0 : offset_[i - 1] + dims_[i - 1]);
    }
}

GlobalConfigure::GlobalConfigure(std::string config_file) {
    auto global_config = cpptoml::parse_file(config_file);

    //slot的配置
    auto slot_table = global_config->get_table("slot_config");
    assert(slot_table != nullptr);
    std::shared_ptr<SlotsConfigure> slot_conf(new SlotsConfigure(slot_table));
    slot_conf_ = slot_conf;

    //ps的配置
    auto ps_table = global_config->get_table("ps_config");
    assert(ps_table != nullptr);
    std::shared_ptr<PSConfigure> ps_conf(new PSConfigure(ps_table));
    assert(ps_table->contains("type"));
    int ps_type = *(ps_table->get_as<int>("type"));
    if (ps_type == 1) {
        std::shared_ptr<::MemoryPSConfigure> ps_conf(new MemoryPSConfigure(ps_table));
        ps_conf_ = ps_conf;
    } else if (ps_type == 2) {
        std::shared_ptr<::RocksDBPSConfigure> ps_conf(new RocksDBPSConfigure(ps_table));
        ps_conf_ = ps_conf;
    } else if (ps_type == 3) {
        std::shared_ptr<::RemoteGRPCPSConfigure> ps_conf(new RemoteGRPCPSConfigure(ps_table));
        ps_conf_ = ps_conf;
    } else if (ps_type == 4) {
        std::shared_ptr<::RemoteGRPCShardPSConfigure> ps_conf(new RemoteGRPCShardPSConfigure(ps_table));
        ps_conf_ = ps_conf;
    } else {
        LOG(ERROR) << "ps type: " << ps_type << " error";
        exit(-1);
    }

    //loader配置
    auto loader_table = global_config->get_table("loader_config");
    assert(loader_table != nullptr);
    std::shared_ptr<LoaderConfigure> loader_conf(new LoaderConfigure(loader_table));
    loader_conf_ = loader_conf;

}