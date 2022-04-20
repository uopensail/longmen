#ifndef LONGMEN_RANK_H
#define LONGMEN_RANK_H

#include "loader.h"
#include "ps.h"
#include <memory>
#include <tensorflow/c/c_api.h>

static const char *tags = "serve";

static void NoneOpDeAllocator(void *, size_t, void *) {}

namespace model
{
    class Rank
    {
    protected:
        std::shared_ptr<::GlobalConfigure> global_config_;
        std::shared_ptr<loader::Extractor> extractor_;
        std::shared_ptr<ps::Memory> ps_client_;

    public:
        Rank() = delete;

        Rank(const Rank &) = delete;

        Rank(const Rank &&) = delete;

        Rank(const std::shared_ptr<::GlobalConfigure> &config);

        virtual ~Rank() {}

        virtual void reload_extractor(std::string path);

        virtual void call(tensorflow::Features &user_features, Recalls &recalls, Scores &scores) = 0;
    };

    class LR : public Rank
    {
    public:
        LR() = delete;

        LR(const LR &) = delete;

        LR(const LR &&) = delete;

        LR(const std::shared_ptr<::GlobalConfigure> &config, std::string path = "");

        virtual ~LR() {}

        virtual void call(tensorflow::Features &user_features, Recalls &recalls, Scores &scores);
    };

    class FM : public Rank
    {
    private:
        int dim_;

    public:
        FM() = delete;

        FM(const FM &) = delete;

        FM(const FM &&) = delete;

        FM(const std::shared_ptr<::GlobalConfigure> &config, std::string path = "");

        virtual ~FM() {}

        virtual void call(tensorflow::Features &user_features, Recalls &recalls, Scores &scores);
    };

    class STF : public Rank
    {
    private:
        int dims_;
        std::string input_op_name_;
        std::string output_op_name_;
        std::string model_dir_;
        TF_Graph *graph_;
        TF_Session *session_;

    public:
        STF() = delete;

        STF(const STF &) = delete;

        STF(const STF &&) = delete;

        STF(const std::shared_ptr<::GlobalConfigure> &config, std::string path = "");

        virtual ~STF();

        virtual void call(tensorflow::Features &user_features, Recalls &recalls, Scores &scores);
    };
}

#endif // LONGMEN_RANK_H
