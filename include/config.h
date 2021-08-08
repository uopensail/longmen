#ifndef LONGMEN_CONFIG_H
#define LONGMEN_CONFIG_H

#include "cpptoml.h"

//定义参数服务器的类型
enum PSType {
    Empty = 0,                              //错误
    Memory = 1,                             //数据量比较小，放在内存
    RocksDB = 2,                            //数据量中等放在rocksdb
    RemoteGRPC = 3,                         //不需要客户端分片的GRPC
    RemoteGRPCShard = 4                     //客户端分片的GRPC
};

class PSConfigure {
protected:
    std::shared_ptr<cpptoml::table> table_;
    PSType type_;
public:
    PSConfigure() = delete;

    PSConfigure(const PSConfigure &) = delete;

    PSConfigure(const PSConfigure &&) = delete;

    PSConfigure(const std::shared_ptr<cpptoml::table> &table) : table_(table), type_(Empty) {
        assert(table_->contains("ps_type"));
        int ps_type = *(table_->get_as<int>("ps_type"));
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

    ~PSConfigure() {}

    PSType &type() { return type_; }
};

class MemoryPSConfigure final : public PSConfigure {
private:
    std::string path_;
public:
    MemoryPSConfigure() = delete;

    MemoryPSConfigure(const MemoryPSConfigure &) = delete;

    MemoryPSConfigure(const MemoryPSConfigure &&) = delete;

    MemoryPSConfigure(const std::shared_ptr<cpptoml::table> &table) : PSConfigure(table),
                                                                      path_(*table_->get_as<std::string>("path")) {

    }

    std::string &get_path() { return path_; }

    ~MemoryPSConfigure() {}
};

class RocksDBPSConfigure final : public PSConfigure {
private:
    std::string path_;
public:
    RocksDBPSConfigure() = delete;

    RocksDBPSConfigure(const RocksDBPSConfigure &) = delete;

    RocksDBPSConfigure(const RocksDBPSConfigure &&) = delete;

    RocksDBPSConfigure(const std::shared_ptr<cpptoml::table> &table) : PSConfigure(table),
                                                                       path_(*table_->get_as<std::string>("path")) {

    }

    std::string &get_path() { return path_; }

    ~RocksDBPSConfigure() {}
};

class RemoteGRPCPSConfigure final : public PSConfigure {
private:
    std::string host_;
    int timeout_;
public:
    RemoteGRPCPSConfigure() = delete;

    RemoteGRPCPSConfigure(const RemoteGRPCPSConfigure &) = delete;

    RemoteGRPCPSConfigure(const RemoteGRPCPSConfigure &&) = delete;

    RemoteGRPCPSConfigure(const std::shared_ptr<cpptoml::table> &table) : PSConfigure(table),
                                                                          host_(*table_->get_as<std::string>("host")) {

    }

    std::string &get_host() { return host_; }

    int &get_timeout() { return timeout_; }

    ~RemoteGRPCPSConfigure() {}
};


//定义slots的信息
class SlotsConfig {
private:
    int slots_;
    int *dims_;
    int *space_;
    int *value_space_;
    int *offset_;
public:
    SlotsConfig() = delete;

    ~SlotsConfig() {
        delete[]dims_;
        delete[]space_;
        delete[]value_space_;
        delete[]offset_;
    }

    SlotsConfig(const std::shared_ptr<cpptoml::table> &table) {
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

    inline int &get_dim(int slot) {
        return dims_[slot];
    }

    inline int &get_total_space(int slot) {
        return space_[slot];
    }

    inline int &get_value_space(int slot) {
        return value_space_[slot];
    }

    inline int &get_slots() { return slots_; }

};

class GlobalConfig {

};

#endif //LONGMEN_CONFIG_H
