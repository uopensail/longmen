#ifndef LONGMEN_CONFIG_H
#define LONGMEN_CONFIG_H

#include <glog/logging.h>
#include <gflags/gflags.h>
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

    PSConfigure(const std::shared_ptr<cpptoml::table> &table);

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
                                                                          host_(*table_->get_as<std::string>("host")),
                                                                          timeout_(*table_->get_as<int>("timeout")) {

    }

    std::string &get_host() { return host_; }

    int &get_timeout() { return timeout_; }

    ~RemoteGRPCPSConfigure() {}
};

class RemoteGRPCShardPSConfigure final : public PSConfigure {
private:
    int shards_;
    int timeout_;
    std::vector<std::string> hosts_;


public:
    RemoteGRPCShardPSConfigure() = delete;

    RemoteGRPCShardPSConfigure(const RemoteGRPCShardPSConfigure &) = delete;

    RemoteGRPCShardPSConfigure(const RemoteGRPCShardPSConfigure &&) = delete;

    RemoteGRPCShardPSConfigure(const std::shared_ptr<cpptoml::table> &table);

    std::string &get_host(int &index) { return hosts_[index]; }

    int &get_timeout() { return timeout_; }

    ~RemoteGRPCShardPSConfigure() {}
};

class LoaderConfigure {
private:
    std::string data_file_;
    std::string luban_config_file_;
public:
    LoaderConfigure() = delete;

    LoaderConfigure(const LoaderConfigure &) = delete;

    LoaderConfigure(const LoaderConfigure &&) = delete;

    LoaderConfigure(const std::shared_ptr<cpptoml::table> &table);

    ~LoaderConfigure() {}

    std::string &get_data_file() { return data_file_; }

    std::string &get_config_file() { return luban_config_file_; }
};

//定义slots的信息
class SlotsConfigure {
private:
    int slots_;
    int *dims_;
    int *space_;
    int *value_space_;
    int *offset_;
public:
    SlotsConfigure() = delete;

    SlotsConfigure(const SlotsConfigure &) = delete;

    SlotsConfigure(const SlotsConfigure &&) = delete;

    ~SlotsConfigure();

    SlotsConfigure(const std::shared_ptr<cpptoml::table> &table);

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

class GlobalConfigure {
private:
    std::shared_ptr<PSConfigure> ps_conf_;
    std::shared_ptr<SlotsConfigure> slot_conf_;
    std::shared_ptr<LoaderConfigure> loader_conf_;
public:
    GlobalConfigure() = delete;

    GlobalConfigure(const GlobalConfigure &) = delete;

    GlobalConfigure(const GlobalConfigure &&) = delete;


    GlobalConfigure(std::string config_file);

    ~GlobalConfigure() {}

    std::shared_ptr<PSConfigure> &get_ps_conf() {
        return ps_conf_;
    }

    std::shared_ptr<SlotsConfigure> &get_slot_conf() {
        return slot_conf_;
    }

    std::shared_ptr<LoaderConfigure> &get_loader_conf() {
        return loader_conf_;
    }
};

#endif //LONGMEN_CONFIG_H
