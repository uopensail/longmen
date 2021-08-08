#include "mutils.h"

static float sigmoid(float x) {
    return (1 / (1 + exp(-x)));
}

//把向量src中的值加到dst中
static void vec_add(float *dst, float *src, int dim) {
    for (int i = 0; i < dim; i++) {
        dst[i] += src[i];
    }
}

//向量的平方
static float vec_square(float *src, int dim) {
    float ret = 0.0;
    for (int i = 0; i < dim; i++) {
        ret += src[i] * src[i];
    }
    return ret;
}