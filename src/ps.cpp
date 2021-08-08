#include "ps.h"


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
    size_t low = 0, high = n, middle;
    while (low < high) {
        middle = (low + high) >> 1;

        if (key == ((u_int64_t *) (value + middle * size))[0]) {
            return (float *) (value + middle * size + sizeof(u_int64_t));
        } else if (key > ((u_int64_t *) (value + middle * size))[0]) {
            low = middle + 1;
        } else {
            high = middle - 1;
        }
    }
    return nullptr;
}


ps::Memory::Memory(std::shared_ptr<::GlobalConfigure> &global_config) :
        slot_conf_(global_config->get_slot_conf()),
        data_((char **) calloc(1, sizeof(char *) * slot_conf_->get_slots())),
        key_count_((long *) calloc(1, sizeof(long) * slot_conf_->get_slots())) {
    auto conf = dynamic_pointer_cast<::MemoryPSConfigure>(global_config->get_ps_conf());
    path_ = conf->get_path();
    std::ifstream reader(path_, std::ios::in | std::ios::binary);
    if (!reader) {
        return;
    }
    int slots_;
    reader.read((char *) &slots_, sizeof(int));

    //slot个数的检查
    assert(slots_ == slot_conf_->get_slots());
    int *dims_ = new int[slots_];
    key_count_ = new long[slots_];
    data_ = new char *[slots_];
    reader.read((char *) dims_, sizeof(int) * slots_);
    reader.read((char *) key_count_, sizeof(long) * slots_);
    size_t *offset = new size_t[slots_];

    //每一个slot的dim的检查
    for (int i = 0; i < slots_; i++) {
        assert(dims_[i] == slot_conf_->get_dim(i));
    }

    u_int64_t key;
    int slot, size;
    while (reader.read((char *) &key, sizeof(u_int64_t))) {
        slot = get_slot_id(key);
        memcpy(data_[slot] + offset[slot], &key, sizeof(u_int64_t));
        reader.read((char *) (&(data_[slot]) + offset[slot] + sizeof(u_int64_t)), slot_conf_->get_value_space(slot));
        offset[slot] += slot_conf_->get_total_space(slot);
    }
    reader.close();
    delete[]offset;
    delete[]dims_;

    //sort
    for (int i = 0; i < slots_; i++) {
        qsort(data_[i], key_count_[i], slot_conf_->get_total_space(i), __sort__);
    }
}

ps::Memory::~Memory() {
    for (int i = 0; i < slot_conf_->get_slots(); i++) {
        delete[]data_[i];
    }
    delete[]data_;
    delete[]key_count_;
}


void ps::Memory::pull(KWWrapper &batch_kw) {
    auto weights = batch_kw.weights();
    auto all_keys = batch_kw.get_all_keys();
    int slot;
    size_t offset = 0;
    float *ptr;
    for (size_t i = 0; i < all_keys.size(); i++) {
        slot = get_slot_id(all_keys[i]);
        ptr = binary_search(data_[slot], key_count_[slot], slot_conf_->get_total_space(slot), all_keys[i]);
        if (ptr == nullptr) {
            offset += slot_conf_->get_dim(slot);
            continue;
        }
        for (size_t j = 0; j < slot_conf_->get_dim(slot); j++) {
            weights[offset] = ptr[j];
            offset++;
        }
    }
}


ps::RocksDB::RocksDB(std::shared_ptr<::GlobalConfigure> &global_config) : slot_conf_(global_config->get_slot_conf()),
                                                                          db_(nullptr) {
    auto conf = dynamic_pointer_cast<::RocksDBPSConfigure>(global_config->get_ps_conf());
    path_ = conf->get_path();
    rocksdb::Options options;
    options.create_if_missing = true;
    rocksdb::Status status = rocksdb::DB::Open(options, path_, &db_);
    if (!status.ok()) {
        std::cerr << "open leveldb error: " << status.ToString() << std::endl;
        exit(-1);
    }
    assert(db_ != nullptr);
    std::cout << "open leveldb: " << path_ << " successfully!" << std::endl;
}

ps::RocksDB::~RocksDB() { delete db_; }

void ps::RocksDB::pull(KWWrapper &batch_kw) {
    auto weights = batch_kw.weights();
    auto all_keys = batch_kw.get_all_keys();
    std::vector<rocksdb::Slice> s_keys;
    std::vector<std::string> result;
    for (size_t i = 0; i < all_keys.size(); i++) {
        s_keys.push_back(rocksdb::Slice((char *) &all_keys[i], sizeof(u_int64_t)));
    }
    rocksdb::ReadOptions get_options;
    auto status = db_->MultiGet(get_options, s_keys, &result);
    size_t offset = 0;
    int slot;
    struct MetaData *ptr;
    for (size_t i = 0; i < all_keys.size(); i++) {
        slot = get_slot_id(all_keys[i]);
        if (!status[i].ok()) {
            offset += slot_conf_->get_dim(slot);
            continue;
        }
        ptr = (struct MetaData *) &(result[i][0]);
        for (size_t j = 0; j < slot_conf_->get_dim(slot); j++) {
            weights[offset] = ptr->data[j];
            offset++;
        }
    }
}


ps::RemoteGRPC::RemoteGRPC(std::shared_ptr<::GlobalConfigure> &global_config) {

}

ps::RemoteGRPC::RemoteGRPC::~RemoteGRPC() {}

void ps::RemoteGRPC::RemoteGRPC::pull(KWWrapper &batch_kw) {
    SingleRequest request;
    SingleResponse response;
    request.set_pack(false);
    auto keys = request.mutable_keys();
    for (auto &key: batch_kw.get_all_keys()) {
        keys->add_data(key);
    }

    grpc::ClientContext context;
    gpr_timespec timespec;
    timespec.tv_sec = 0;
    timespec.tv_nsec = 1000000 * timeout_;
    timespec.clock_type = GPR_TIMESPAN;
    context.set_deadline(timespec);
    auto status = stub_->pull(&context, request, &response);
    if (status.ok()) {
        auto src = response.weights();
        auto dst = batch_kw.weights();
        for (int i = 0; i < src.data_size(); i++) {
            dst[i] = src.data(i);
        }
    }
}

std::shared_ptr<ps::Client> ps::create_client(std::shared_ptr<::GlobalConfigure> &global_config) {
    auto ps_type = global_config->get_ps_conf()->type();
    if (ps_type == ::PSType::Memory) {
        std::shared_ptr<Client> client(new Memory(global_config));
        return client;
    } else if (ps_type == ::PSType::RocksDB) {
        std::shared_ptr<Client> client(new RocksDB(global_config));
        return client;
    } else if (ps_type == ::PSType::RemoteGRPC) {
        std::shared_ptr<Client> client(new RemoteGRPC(global_config));
        return client;
    } else if (ps_type == ::PSType::RemoteGRPCShard) {
        std::shared_ptr<Client> client(new RemoteGRPCShard(global_config));
        return client;
    } else {
        LOG(ERROR) << "ps type: " << ps_type << " error";
        exit(-1);
    }
}