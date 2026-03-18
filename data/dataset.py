"""
数据加载模块 - 使用PyTorch框架，支持可扩展性
"""

import os
import re
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split

# 6维特征名称
PERSUASION_FEATURES = [
    'Attack_on_reputation',
    'Justification',
    'Simplification',
    'Distraction',
    'Call',
    'Manipulative_wording'
]


class BaseDatasetLoader:
    """数据集加载基类"""

    name: str = "base"

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load(self) -> pd.DataFrame:
        raise NotImplementedError

    def parse_label(self, label: str) -> int:
        return 1 if label.strip().lower() == 'real' else 0


class ECTFLoader(BaseDatasetLoader):
    name = "ECTF"
    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "deepseek-v3.2_ECTF.csv")


class CoAIDLoader(BaseDatasetLoader):
    name = "CoAID"
    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "deepseek-v3.2_CoAID.csv")


class ISOTFakeNewsLoader(BaseDatasetLoader):
    name = "ISOTFakeNews"
    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "deepseek-v3.2_ISOTFakeNews.csv")


class MultiDisLoader(BaseDatasetLoader):
    name = "MultiDis"
    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "deepseek-v3.2_MultiDis.csv")


class EUDisinfoLoader(BaseDatasetLoader):
    name = "EUDisinfo"
    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "deepseek-v3.2_EUDisinfo.csv")


# 数据集加载器注册表
DATASET_REGISTRY: Dict[str, BaseDatasetLoader] = {
    'ECTF': ECTFLoader,
    'CoAID': CoAIDLoader,
    'ISOTFakeNews': ISOTFakeNewsLoader,
    'MultiDis': MultiDisLoader,
    'EUDisinfo': EUDisinfoLoader,
}


def register_dataset(name: str, loader_class: type):
    """注册新的数据集加载器"""
    DATASET_REGISTRY[name] = loader_class


class PersuasionFeatureParser:
    """解析 persuasion 特征 - 将JSON转换为6维向量"""

    def parse(self, json_str: str) -> np.ndarray:
        """解析JSON字符串，Yes -> 1, No -> 0"""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return self._extract_from_text(json_str)

        if not isinstance(data, dict):
            return np.zeros(6, dtype=np.float32)

        features = []
        for feature in PERSUASION_FEATURES:
            if feature in data:
                is_used = data[feature].get('is_used', 'No')
                features.append(1.0 if is_used.strip().lower() == 'yes' else 0.0)
            else:
                features.append(0.0)
        return np.array(features, dtype=np.float32)

    def _extract_from_text(self, text: str) -> np.ndarray:
        """从文本中提取特征"""
        features = []
        for feature in PERSUASION_FEATURES:
            pattern = rf'"{feature}".*?"is_used"\s*:\s*"([^"]+)"'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match and match.group(1).strip().lower() == 'yes':
                features.append(1.0)
            else:
                features.append(0.0)
        return np.array(features, dtype=np.float32)


class PersuasionDataset(Dataset):
    """PyTorch数据集"""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_datasets(data_dir: str, dataset_names: Optional[List[str]] = None) -> pd.DataFrame:
    """加载数据集"""
    if dataset_names is None:
        dataset_names = list(DATASET_REGISTRY.keys())

    dfs = []
    for name in dataset_names:
        if name in DATASET_REGISTRY:
            loader = DATASET_REGISTRY[name](data_dir)
            df = loader.load()
            df['source'] = name
            dfs.append(df)
            print(f"Loaded {name}: {len(df)} samples")

    return pd.concat(dfs, ignore_index=True) if dfs else None


def prepare_data(data_dir: str, dataset_names: Optional[List[str]] = None,
                 val_size: float = 0.15, test_size: float = 0.15, random_seed: int = 42):
    """
    准备数据并划分数据集

    Args:
        data_dir: 数据目录
        dataset_names: 要加载的数据集名称
        val_size: 验证集比例
        test_size: 测试集比例
        random_seed: 随机种子

    Returns:
        train_loader, val_loader, test_loader
    """
    # 加载数据
    df = load_datasets(data_dir, dataset_names)

    # 解析特征
    parser = PersuasionFeatureParser()
    features = np.array([parser.parse(row['generated_pred'])
                         for _, row in df.iterrows()])

    # 解析标签: real=1, fake=0
    labels = np.array([1 if l.strip().lower() == 'real' else 0
                      for l in df['label']], dtype=np.int64)

    print(f"Total: {len(features)} samples, Features shape: {features.shape}")
    print(f"Label distribution: Real={sum(labels)}, Fake={len(labels)-sum(labels)}")

    # 划分数据集 (train:val:test = 7:1.5:1.5)
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=(val_size + test_size),
        random_state=random_seed, stratify=labels
    )

    # 计算验证集和测试集的比例
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1-val_ratio),
        random_state=random_seed, stratify=y_temp
    )

    # 创建数据集
    train_dataset = PersuasionDataset(X_train, y_train)
    val_dataset = PersuasionDataset(X_val, y_val)
    test_dataset = PersuasionDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader