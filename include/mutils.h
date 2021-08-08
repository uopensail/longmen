#ifndef LONGMEN_MUTILS_H
#define LONGMEN_MUTILS_H

//这是常用的数学计算

#include <math.h>

static float sigmoid(float x);

//把向量src中的值加到dst中
static void vec_add(float *dst, float *src, int dim);

//向量的平方
static float vec_square(float *src, int dim);

#endif //LONGMEN_MUTILS_H
