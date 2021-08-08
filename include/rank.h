#ifndef LONGMEN_RANK_H
#define LONGMEN_RANK_H

#include <memory>
#include "ps.h"
#include "loader.h"

namespace model {
    class Rank {
    protected:
        std::shared_ptr<ps::Client> ps_client_;
        std::shared_ptr<loader::Extractor> extractor_;
    public:
        Rank() = delete;

        Rank(const Rank &) = delete;

        Rank(const Rank &&) = delete;

        Rank(std::shared_ptr<::GlobalConfigure> &config);

        ~Rank() {}

        virtual void call(tensorflow::Features &user_features, Recalls &recalls, Scores &scores) = 0;
    };

    class LR : public Rank {
    public:
        LR() = delete;

        LR(const LR &) = delete;

        LR(const LR &&) = delete;

        LR(std::shared_ptr<::GlobalConfigure> &config);

        ~LR() {}

        virtual void call(tensorflow::Features &user_features, Recalls &recalls, Scores &scores);
    };

    class FM : public Rank {
    private:
        int dim_;
    public:
        FM() = delete;

        FM(const FM &) = delete;

        FM(const FM &&) = delete;

        FM(std::shared_ptr<::GlobalConfigure> &config);

        ~FM() {}

        virtual void call(tensorflow::Features &user_features, Recalls &recalls, Scores &scores);
    };
}


#endif //LONGMEN_RANK_H
