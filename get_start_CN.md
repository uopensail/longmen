# 基于luban-damo-longmen的模型训练和推理

## 简介

[luban](https://github.com/uopensail/luban)是一个特征处理的工具，[damo-embedding](https://github.com/uopensail/damo-embedding)是基于pytorch的模型训练工具，[longmen](https://github.com/uopensail/longmen)是基于damo导出的模型的线上推理服务。

 

## 特征处理

我们以[Criteo_dataset | Kaggle](https://www.kaggle.com/datasets/mrkmakr/criteo-dataset)数据集为例，进行特征处理。

数据说明：共有39个特征，其中前13个是整型特征，后26个是类别特征，字符串类型。特征数据是以txt形式存储，然后以`\t`作为分割符，第一行是数据，没有给特征命名。第一列是label。

因此我们自己给它们命名，整型特征命名规则是I1,I2,I3,...I13, 类别型特征命名规则是C1,C2,C3,...,C26.

### 

### 编写配置

因为这里是个例子，我们仅仅做示范，不太在用最后模型训练的结果。我们将对特征做如下的处理：

1. 整型特征转成浮点型，默认填充值是0.0

2. 类别型特征做hash处理，默认填充值是0

```json
{
    "features": [
        {
            "name": "I1",
            "type": 1,
            "hash": false,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "I2",
            "type": 1,
            "hash": false,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "I3",
            "type": 1,
            "hash": false,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "I4",
            "type": 1,
            "hash": false,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "I5",
            "type": 1,
            "hash": false,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "I6",
            "type": 1,
            "hash": false,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "I7",
            "type": 1,
            "hash": false,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "I8",
            "type": 1,
            "hash": false,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "I9",
            "type": 1,
            "hash": false,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "I10",
            "type": 1,
            "hash": false,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "I11",
            "type": 1,
            "hash": false,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "I12",
            "type": 1,
            "hash": false,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "I13",
            "type": 1,
            "hash": false,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C1",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C2",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C3",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C4",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C5",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C6",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C7",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C8",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C9",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C10",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C11",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C12",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C13",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C14",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C15",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C16",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C17",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C18",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C19",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C20",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C21",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C22",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C23",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C24",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C25",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        },
        {
            "name": "C26",
            "type": 2,
            "hash": true,
            "padding": 0,
            "dim": 1
        }
    ],
    "groups": [
        [
            "I1",
            "I2",
            "I3",
            "I4",
            "I5",
            "I6",
            "I7",
            "I8",
            "I9",
            "I10",
            "I11",
            "I12",
            "I13"
        ],
        [
            "C1",
            "C2",
            "C3",
            "C4",
            "C5",
            "C6",
            "C7",
            "C8",
            "C9",
            "C10",
            "C11",
            "C12",
            "C13",
            "C14",
            "C15",
            "C16",
            "C17",
            "C18",
            "C19",
            "C20",
            "C21",
            "C22",
            "C23",
            "C24",
            "C25",
            "C26"
        ]
    ]
}
```

### 

### 转换配置文件

我们需要将上述的配置文件，转成程序需要的配置文件，方式如下：

```python
import luban_parser
luban_parser.parse(config_path, new_config_path)
```



### 处理特征

示例代码如下:

```python
def traindata_process(new_config_path: str, train_data: str):
    toolkit = pyluban.Toolkit(new_config_path)
    cols = ["label"] + [f"I{i+1}" for i in range(13)] + [f"C{i+1}" for i in range(26)]

    labels, dense, sparse = [], [], []
    f = open(train_data, "r")
    line = f.readline()

    while line:
        items = line.strip().split("\t")
        values = {}
        for i, (k, v) in enumerate(zip(cols, items)):
            dtype, v = (1, 0 if len(v) == 0 else int(v)) if i < 14 else (2, v)
            values[k] = {"type": dtype, "value": v}
        str = json.dumps(values)
        r = toolkit.process(pyluban.Features(str))
        dense.append(np.asarray(r[0]))
        sparse.append(np.asarray(r[1]))
        labels.append(values["label"]["value"])
        line = f.readline()
    f.close()

    train_dataset = Data.TensorDataset(
        torch.from_numpy(np.array(dense, dtype=np.float32)),
        torch.from_numpy(np.array(sparse, dtype=np.int64)),
        torch.from_numpy(np.array(labels, dtype=np.float32)),
    )
    return Data.DataLoader(dataset=train_dataset, batch_size=4096, shuffle=False)
```

## 模型训练

### 模型定义

我们把类别特征做embedding，size是8，然后把它们和整型特征一起concat起来，然后在后面加几层MLP。具体模型定义如下：

```python
class DNN(torch.nn.Module):
    def __init__(
        self,
        dense_size=13,
        emb_sizes=[8 for _ in range(26)],
        hid_dims=[256, 128],
        num_classes=1,
        dropout=[0.2, 0.2],
        **kwargs,
    ):
        super(DNN, self).__init__()
        self.emb_sizes = emb_sizes

        initializer = {
            "name": "truncate_normal",
            "mean": float(kwargs.get("mean", 0.0)),
            "stddev": float(kwargs.get("stddev", 0.0001)),
        }

        optimizer = {
            "name": "adam",
            "gamma": float(kwargs.get("gamma", 1e-3)),
            "beta1": float(kwargs.get("beta1", 0.9)),
            "beta2": float(kwargs.get("beta2", 0.999)),
            "lambda": float(kwargs.get("lambda", 0.0)),
            "epsilon": float(kwargs.get("epsilon", 1e-8)),
        }

        self.embeddings = nn.ModuleList()
        for emb_size in emb_sizes:
            embedding = Embedding(
                emb_size,
                initializer=initializer,
                optimizer=optimizer,
                **kwargs,
            )
            self.embeddings.append(embedding)

        self.dims = [sum(emb_sizes) + dense_size] + hid_dims
        self.layers = nn.ModuleList()
        for i in range(1, len(self.dims)):
            self.layers.append(nn.Linear(self.dims[i - 1], self.dims[i]))
            self.layers.append(nn.BatchNorm1d(self.dims[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout[i - 1]))
        self.layers.append(nn.Linear(self.dims[-1], num_classes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, dense: torch.Tensor, sparse: torch.Tensor):
        batch_size, _ = sparse.shape
        weights = [dense]
        for i, embedding in enumerate(self.embeddings):
            w = embedding.forward(
                sparse[:][:, i].reshape(batch_size, 1)
            )
            weights.append(torch.sum(w, dim=1))
        dnn_out = torch.concat(weights, dim=1)
        for layer in self.layers:
            dnn_out = layer(dnn_out)
        out = self.sigmoid(dnn_out)
        return out
```

### 模型训练

我们把上面处理过的特征输入进去，然后进行模型训练，具体的代码如下：

```python
def train_model(model: torch.nn.Module, train_loader: Data.DataLoader, epochs=1):
    loss_fcn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    for _ in range(epochs):
        model.train()
        train_loss_sum = 0.0
        start_time = time.time()
        for idx, x in enumerate(train_loader):
            dense, sparse, label = x[0], x[1], x[2]
            pred = model(dense, sparse).view(-1)
            loss = loss_fcn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.cpu().item()
            if (idx + 1) % 10 == 0 or (idx + 1) == len(train_loader):
                print(
                    "Epoch {:04d} | Step {:04d} / {} | Loss {:.4f} | Time {:.4f}".format(
                        _ + 1,
                        idx + 1,
                        len(train_loader),
                        train_loss_sum / (idx + 1),
                        time.time() - start_time,
                    )
                )
```

### 模型保存

为了防止模型训练错误不可恢复，这里需要对模型进行checkpoint保存，同时模型需要导出来做线上推理，具体代码如下：

```python
def dump_model(model: torch.nn.Module):
    Storage.checkpoint("./checkpoint")
    save_model(model, "./model")
```

## 模型推理

在模型推理的时候需要做如下的配置：

1. 物料池相关配置
   
   在该例子中，我们没有用的物料池，所以我们随便写个一个物料池的配置，把所有的特征都当做用户特征。**注: 下面是个例子**
   
   ```textile
   {"id":{"type":2, "value":"A"}}
   {"id":{"type":2, "value":"B"}}
   {"id":{"type":2, "value":"C"}}
   ```

```json
{
    "path":"/tmp/longmen/pool.txt",
    "key": "id",
    "version": "1"
}
```

2. 模型相关的配置
   
   ```json
   {
       "path":"/tmp/longmen/model.pt",
       "pool": "test_pool",
       "kit": "/tmp/longmen/new_config_path.json",
       "version": "1"
   }
   ```

### 设置etcd的值

我们是把所有的配置特征都放到etcd里面去的，因为需要把这些配置写到etcd中，具体的设置代码如下:

```python
import etcd3
def write_config_to_etcd(endpoint,port, pool_config, model_config):
    client = etcd3.client(host=endpoint, port=port)
    pool_name, model_name = "test_pool", "test_model"
    pool_key = "/longmen/pools/%s" % pool_name
    model_key = "/longmen/models/%s" % model_name
    client.put(pool_key, json.dumps(pool_config))
    client.put(model_key, json.dumps(model_config))
    print(client.get(pool_key))
    print(client.get(model_key))
```



### 配置longmen

longmen的配置如下：

```toml
models = ["test_model"]
pools = ["test_pool"]

[server]
project_name = "longmen"
grpc_port = 9527
http_port = 9528
prome_port = 9529
pprof_port = 9530
debug = true
[server.register.etcd]
name = "test"
endpoints = ["http://127.0.0.1:2379"]
```



### 启动longmen

```shell
make clean 
make build-dev
cd build/
./longmen --config=conf/config.toml
```



### 测试服务

我们用python代码进行测试，代码如下：

```python
import requests
def test_longmen():
    user_features = '{"label": {"type": 1, "value": 0}, "I1": {"type": 1, "value": 0}, "I2": {"type": 1, "value": 1}, "I3": {"type": 1, "value": 26}, "I4": {"type": 1, "value": 0}, "I5": {"type": 1, "value": 4566}, "I6": {"type": 1, "value": 0}, "I7": {"type": 1, "value": 0}, "I8": {"type": 1, "value": 1}, "I9": {"type": 1, "value": 32}, "I10": {"type": 1, "value": 0}, "I11": {"type": 1, "value": 0}, "I12": {"type": 1, "value": 0}, "I13": {"type": 1, "value": 1}, "C1": {"type": 2, "value": "68fd1e64"}, "C2": {"type": 2, "value": "e5fb1af3"}, "C3": {"type": 2, "value": "ae150a99"}, "C4": {"type": 2, "value": "c11212fa"}, "C5": {"type": 2, "value": "25c83c98"}, "C6": {"type": 2, "value": "7e0ccccf"}, "C7": {"type": 2, "value": "3fd38f3b"}, "C8": {"type": 2, "value": "0b153874"}, "C9": {"type": 2, "value": "a73ee510"}, "C10": {"type": 2, "value": "fbbf2c95"}, "C11": {"type": 2, "value": "7c430b79"}, "C12": {"type": 2, "value": "6fd8b58f"}, "C13": {"type": 2, "value": "7f0d7407"}, "C14": {"type": 2, "value": "07d13a8f"}, "C15": {"typ^Cue": ""}, "C4": {"type": 2, "value": ""}, "C5": {"type": 2, "value": "4cf72387"}, "C6": {"type": 2, "value": "7e0ccccf"}, "C7": {"type": 2, "value": "7bec6ac1"}, "C8": {"type": 2, "value": "0b153874"}, "C9": {"type": 2, "value": "a73ee510"}, "C10": {"type": 2, "value": "bc67eb65"}, "C11": {"type": 2, "value": "25c007f9"}, "C12": {"type": 2, "value": ""}, "C13": {"type": 2, "value": "d3cef96f"}, "C14": {"type": 2, "value": "b28479f6"}, "C15": {"type": 2, "value": "42b3012c"}, "C16": {"type": 2, "value": ""}, "C17": {"type": 2, "value": "1e88c74f"}, "C18": {"type": 2, "value": "582152eb"}, "C19": {"type": 2, "value": "21ddcdc9"}, "C20": {"type": 2, "value": "5840adea"}, "C21": {"type": 2, "value": ""}, "C22": {"type": 2, "value": ""}, "C23": {"type": 2, "value": "32c7478e"}, "C24": {"type": 2, "value": ""}, "C25": {"type": 2, "value": "001f3601"}, "C26": {"type": 2, "value": "56be3401"}}'

    data = {
        "modelId": "test_model",
        "userId": "123",
        "userFeatures": user_features,
        "records": [
            {
                "id": "A",
            },
            {
                "id": "B",
            },
            {
                "id": "C",
            },
        ],
    }
    r = requests.post("http://localhost:9528/rank", json=data)
    print(r.json())
```

返回的结果如下:

```json
{
    "userId":"123",
    "records":[
        {
            "id":"A",
            "score":0.2977918
        },
        {
            "id":"B",
            "score":0.2977918
        },
        {
            "id":"C",
            "score":0.2977918
        }
    ],
    "extras":{
        "mVer":"1",
        "pVer":"1"
    }
}
```