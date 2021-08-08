
#include "rank.h"

model::Rank::Rank(std::shared_ptr<::GlobalConfigure> &config) : ps_client_(ps::create_client(config)),
                                                                extractor_(loader::create_extractor(config)) {}

model::LR::LR(std::shared_ptr<::GlobalConfigure> &config) : Rank(config) {
    assert(config->get_slot_conf()->get_slots() > 0);
    for (int i = 0; config->get_slot_conf()->get_slots(); i++) {
        assert(config->get_slot_conf()->get_dim(i) == 1);
    }
}


model::FM::FM(std::shared_ptr<::GlobalConfigure> &config) : Rank(config) {
    assert(config->get_slot_conf()->get_slots() > 0);
    dim_ = config->get_slot_conf()->get_dim(0);
    for (int i = 0; config->get_slot_conf()->get_slots(); i++) {
        assert(config->get_slot_conf()->get_dim(i) == dim_);
    }
}

void model::LR::call(tensorflow::Features &user_features, Recalls &recalls, Scores &scores) {
    auto batch_kw = extractor_->call(user_features, recalls);
    ps_client_->pull(*batch_kw);
    auto weights = batch_kw->weights();
    int row = 0;
    float *ptr, score;
    for (auto &id: recalls) {
        scores[row].first = id;
        for (auto &key: batch_kw->operator[](row)) {
            ptr = batch_kw->get_weights(key);
            if (ptr) {
                score += ptr[0];
            }
        }
        scores[row].second = sigmoid(score);
    }
    std::sort(scores.begin(), scores.end(), ::__score_cmp__);
}

void model::FM::call(tensorflow::Features &user_features, Recalls &recalls, Scores &scores) {
    auto batch_kw = extractor_->call(user_features, recalls);
    ps_client_->pull(*batch_kw);
    auto weights = batch_kw->weights();
    int row = 0;
    float *ptr, score;
    float *vec_sum = (float *) calloc(1, sizeof(float) * dim_);
    for (auto &id: recalls) {
        memset(vec_sum, 0, sizeof(float) * dim_);
        scores[row].first = id;
        for (auto &key: batch_kw->operator[](row)) {
            ptr = batch_kw->get_weights(key);
            if (ptr) {
                vec_add(vec_sum, ptr, dim_);
                score -= vec_square(ptr, dim_);
            }
        }
        score += vec_square(vec_sum, dim_);
        scores[row].second = sigmoid(score * 0.5);
    }
    free(vec_sum);
    std::sort(scores.begin(), scores.end(), ::__score_cmp__);
}