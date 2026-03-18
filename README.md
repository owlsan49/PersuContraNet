# Persuasion Detection MLP

基于深度学习的虚假新闻检测模型，使用6维说服力特征向量（Attack on reputation, Justification, Simplification, Distraction, Call, Manipulative wording）进行二分类。

## 项目结构

```
Persu-Contra-MLP/
├── data/
│   └── dataset.py       # 数据加载与解析
├── models/
│   └── mlp.py           # MLP模型定义
├── config/
│   └── config.py        # 配置文件
├── train.py             # 训练脚本
└── requirements.txt     # 依赖
```

## 快速开始

### 1. 安装依赖

```bash
pip install torch numpy pandas scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 训练模型

```bash
python train.py
```

### 3. 自定义配置

修改 `config/config.py` 或使用命令行参数：

```bash
# 指定数据集
python train.py --datasets ECTF CoAID

# 指定保存路径
python train.py --save_path models/my_model.pth
```

## 添加新数据集

1. 在 `data/dataset.py` 中创建新的Loader类继承 `BaseDatasetLoader`
2. 在 `DATASET_REGISTRY` 中注册

```python
class MyDatasetLoader(BaseDatasetLoader):
    name = "MyDataset"
    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "my_data.csv")

DATASET_REGISTRY['MyDataset'] = MyDatasetLoader
```

## 添加新模型

1. 在 `models/mlp.py` 中创建新的模型类继承 `BaseModel`
2. 在 `MODEL_REGISTRY` 中注册

```python
MODEL_REGISTRY['my_model'] = MyModel
```

## 数据格式

输入数据应包含 `label` 和 `generated_pred` 列：
- `label`: `real` 或 `fake`
- `generated_pred`: 包含6维说服力特征的JSON字符串

## 输出指标

- Accuracy
- 以 Real(1) 为正类的 Precision/Recall/F1
- 以 Fake(0) 为正类的 Precision/Recall/F1
- Macro F1