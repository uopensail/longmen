# LongMen(龙门)

这是对参数服务器和item池以及模型预测的封装

这是线上推理的c++封装
支持的配置如下：

config.toml文件

```
[model_config]
# 1:LR
# 2:FM
# 3:STF(sparse embedding with TensorFlow)
type = 1

# 如果是LR/FM，就是参数的保存地址，绝对路径
# 如果是STF，就是模型的压缩包，包含了稠密层和稀疏层的参数，.zip结尾
path = "模型的绝对路径"

# LR: dim必须是1
# FM: 每个slot的dim必须相同，跟slot的配置会检查
# STF: TensorFlow输入层的维度，等于sum(slot dims)
dim = 1


# 如果是STF模型，需要配置如下的参数
# 可以通过tf_tool工具查看所有的operation
# 使用方法: ./tf_tool model_path

input_op = "输入层的operation的名字"
output_op = "输出层的operation的名字"
sparse = "稀疏层在文件夹中的名字"


[slot_config]
# 每一个slot对应的维度
slots = [1, 1, 1]

[loader_config]
config_path = "/绝对路径/特征处理的配置.toml"
data_path = "/绝对路径/候选集.tfrecord"

```


feature_process.toml
```
[[single_features]]
func = "kv"
key = "d_s_id"
name = "d_s_id"

[single_features.params]
slot_id = 0

[[single_features]]
func = "kv"
key = "u_r_click"
name = "u_r_click"

[single_features.params]
slot_id = 1

[[cross_features]]
keyA = "d_s_id"
keyB = "u_r_click"
func = "merge"
name = "test"

[cross_features.params]
slot_id = 2

```

tfrecord文件生成示例
```
参考:
test/prepare_tfrecord.py
```