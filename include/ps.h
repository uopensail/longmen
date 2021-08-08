#ifndef LONGMEN_PS_H
#define LONGMEN_PS_H

#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <rocksdb/db.h>
#include <rocksdb/write_batch.h>
#include "common.h"
#include "ps.grpc.pb.h"

namespace ps {
#pragma pack(push)
#pragma pack(1)
    struct MetaData {
        u_int64_t key;
        int dim;
        u_int64_t update_time;
        u_int64_t update_num;
        float data[];
    };
#pragma pack(pop)

    class Client {
    public:
        virtual void pull(KWWrapper &batch_kw) = 0;
    };

    int __sort__(const void *a, const void *b);

    float *binary_search(const char *value, const size_t &n, const size_t &size, u_int64_t &key);

    class Memory : public Client {
    private:
        std::shared_ptr<SlotsConfigure> slot_conf_;
        char **data_;
        long *key_count_;
        std::string path_;
    public:
        Memory() = delete;

        Memory(const Memory &) = delete;

        Memory(const Memory &&) = delete;

        Memory(std::shared_ptr<::GlobalConfigure> &global_config);

        ~Memory();

        virtual void pull(KWWrapper &batch_kw);
    };

    class RocksDB : public Client {
    private:
        std::shared_ptr<SlotsConfigure> slot_conf_;
        rocksdb::DB *db_;
        std::string path_;
    public:
        RocksDB() = delete;

        RocksDB(const RocksDB &) = delete;

        RocksDB(const RocksDB &&) = delete;

        RocksDB(std::shared_ptr<::GlobalConfigure> &global_config);

        ~RocksDB();

        virtual void pull(KWWrapper &batch_kw);
    };

    class RemoteGRPC : public Client {
    private:
        std::string host_;
        std::shared_ptr<Service::Stub> stub_;
        int timeout_;
    public:
        RemoteGRPC() = delete;

        RemoteGRPC(const RemoteGRPC &) = delete;

        RemoteGRPC(const RemoteGRPC &&) = delete;

        RemoteGRPC(std::shared_ptr<::GlobalConfigure> &global_config);

        ~RemoteGRPC();

        virtual void pull(KWWrapper &batch_kw);
    };

    class RemoteGRPCShard : public Client {
    private:
        std::string host_;
        std::shared_ptr<Service::Stub> stub_;
        int timeout_;
    public:
        RemoteGRPCShard() = delete;

        RemoteGRPCShard(const RemoteGRPCShard &) = delete;

        RemoteGRPCShard(const RemoteGRPCShard &&) = delete;

        RemoteGRPCShard(std::shared_ptr<::GlobalConfigure> &global_config);

        ~RemoteGRPCShard();

        virtual void pull(KWWrapper &batch_kw);
    };

    std::shared_ptr<Client> create_client(std::shared_ptr<::GlobalConfigure> &global_config);

}  // namespace ps
#endif //LONGMEN_PS_H
