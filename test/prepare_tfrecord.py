#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:58:27 2019

@author: random

@description: 生成tfrecord
"""
import random
import argparse
import tensorflow as tf


def transform(value):
    """
    转化成tf-features的格式
    :return:
    """
    if isinstance(value, str):
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[value.encode('utf8', 'ignore')])
        )
    if isinstance(value, int):
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=[value])
        )
    if isinstance(value, float):
        return tf.train.Feature(
            float_list=tf.train.FloatList(value=[value])
        )

    # 下面就只能是list
    assert (isinstance(value, list) and len(value) > 0)
    if isinstance(value[0], str):
        tmp = list(map(lambda _: _.encode('utf8', 'ignore'), value))
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=tmp)
        )
    if isinstance(value[0], int):
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=value)
        )
    if isinstance(value[0], float):
        return tf.train.Feature(
            float_list=tf.train.FloatList(value=value)
        )
    return None


def get_str():
    return ''.join(random.sample(
        ['z', 'y', 'x', 'w', 'v', 'u', 't', 's', 'r', 'q', 'p', 'o', 'n', 'm',
         'l', 'k', 'j', 'i', 'h', 'g', 'f', 'e',
         'd', 'c', 'b', 'a'], 23))


def main():
    """
    生成随机测试
    :return:
    """
    keys = ['d_s_id', 'd_s_1', 'd_s_2', 'd_s_3', 'd_s_4']
    d_s_id = []
    d_s_1 = []
    d_s_2 = [1]
    d_s_3 = []
    d_s_4 = []
    for i in range(10000):
        d_s_id.append(str(i))
        d_s_1.append(random.random())
        d_s_2.append(random.randint(0, 100))
        tmp_3 = []
        tmp_4 = []
        for j in range(20):
            tmp_3.append(random.random())
            tmp_4.append(get_str())
        d_s_3.append(tmp_3)
        d_s_4.append(tmp_4)
    write = tf.io.TFRecordWriter('test.tfrecord')
    for i in range(100):
        dic = {'d_s_id': d_s_id[i], 'd_s_1': d_s_1[i], 'd_s_2': d_s_2[i], 'd_s_3': d_s_3[i], 'd_s_4': d_s_4[i]}
        example = dic2example(dic)
        write.write(example.SerializeToString())
    write.close()


def dic2example(dic):
    """
    字典转tf example
    :param dic:
    :return:
    """
    example = tf.train.Example()
    features = tf.train.Features()
    for (k, v) in dic.items():
        feature = transform(v)
        if feature is not None:
            features.feature[k].CopyFrom(feature)
    example.features.CopyFrom(features)
    # print(example)
    return example


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input', "-input", type=str, required=True, help='输入文件sqlitedb')
    # parser.add_argument('--output', "-output", type=str, required=True, help='输出文件tfrecord')
    # args = parser.parse_args()
    # print(args)
    # process(args.input, args.output)
    main()
