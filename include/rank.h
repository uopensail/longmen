#ifndef LONGMEN_RANK_H
#define LONGMEN_RANK_H

#include <memory>
#include "ps.h"
#include "loader.h"

namespace model {
    class RankModel {
    protected:
        std::shared_ptr<ps::Client> ps_client_;
        std::shared_ptr<loader::Extractor> extractor_;
    public:
        virtual void call(tensorflow::Features &user_features, Recalls &recalls, Scores &scores) = 0;
    };

    class LR : public RankModel {
    public:
        LR() = delete;

        LR(const LR &) = delete;

        LR(const LR &&) = delete;

        LR(std::shared_ptr<::GlobalConfig> &config);

        ~LR();

        virtual void call(tensorflow::Features &user_features, Recalls &recalls, Scores &scores);
    };

    class FM : public RankModel {
    private:
        int dim_;
    public:
        FM() = delete;

        FM(const FM &) = delete;

        FM(const FM &&) = delete;

        ~FM();

        virtual void call(tensorflow::Features &user_features, Recalls &recalls, Scores &scores);
    };
}


#endif //LONGMEN_RANK_H
