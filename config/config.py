"""
训练配置
"""

# 数据配置
DATA_DIR = "pcot_persu_data"
DATASETS = None  # None 表示加载所有数据集

# 模型配置
MODEL_CONFIG = {
    "model_name": "persuasion_mlp",
    "input_dim": 6,
    "hidden_dims": [64, 32],
    "num_classes": 2,
    "dropout": 0.3
}

# 训练配置
TRAIN_CONFIG = {
    "epochs": 50,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "early_stopping_patience": 10,
    "device": "cuda"  # "cuda" or "cpu"
}

# 数据划分
SPLIT_CONFIG = {
    "val_size": 0.15,
    "test_size": 0.15,
    "random_seed": 42
}