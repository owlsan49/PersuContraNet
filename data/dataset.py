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

# 6维特征名称
PERSUASION_FEATURES = [
    'Attack_on_reputation',
    'Justification',
    'Simplification',
    'Distraction',
    'Call',
    'Manipulative_wording'
]


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


def prepare_data_from_files(train_files: List[str] = None, val_files: List[str] = None,
                            test_files: List[str] = None,
                            data_dir: str = "pcot_persu_data",
                            model_prefix: str = None):
    """
    从指定文件加载数据

    Args:
        train_files: 训练集文件路径列表 (如 ["gpt-5-mini-2025-08-07_ECTF_train.csv"])
        val_files: 验证集文件路径列表
        test_files: 测试集文件列表
        data_dir: 数据根目录
        model_prefix: 模型前缀，用于构建完整文件名

    Returns:
        train_loader, val_loader, test_loader
    """
    parser = PersuasionFeatureParser()
    dfs = []

    # 加载训练集
    if train_files:
        for train_file in train_files:
            train_path = os.path.join(data_dir, train_file)
            if os.path.exists(train_path):
                df_train = pd.read_csv(train_path)
                df_train['split'] = 'train'
                dfs.append(df_train)
                print(f"Loaded train: {len(df_train)} samples from {train_file}")

    # 加载验证集
    if val_files:
        for val_file in val_files:
            val_path = os.path.join(data_dir, val_file)
            if os.path.exists(val_path):
                df_val = pd.read_csv(val_path)
                df_val['split'] = 'val'
                dfs.append(df_val)
                print(f"Loaded val: {len(df_val)} samples from {val_file}")

    # 加载测试集
    if test_files:
        for test_file in test_files:
            test_path = os.path.join(data_dir, test_file)
            if os.path.exists(test_path):
                df_test = pd.read_csv(test_path)
                df_test['split'] = 'test'
                dfs.append(df_test)
                print(f"Loaded test: {len(df_test)} samples from {test_file}")

    if not dfs:
        raise ValueError("No data loaded!")

    # 合并数据
    df = pd.concat(dfs, ignore_index=True)

    # 解析特征
    features = np.array([parser.parse(row['generated_pred'])
                         for _, row in df.iterrows()])

    # 解析标签
    labels = np.array([1 if l.strip().lower() == 'real' else 0
                      for l in df['label']], dtype=np.int64)

    print(f"Total: {len(features)} samples, Features shape: {features.shape}")
    print(f"Label distribution: Real={sum(labels)}, Fake={len(labels)-sum(labels)}")

    # 按split划分
    train_mask = df['split'] == 'train'
    val_mask = df['split'] == 'val'
    test_mask = df['split'] == 'test'

    X_train = features[train_mask]
    y_train = labels[train_mask]
    X_val = features[val_mask]
    y_val = labels[val_mask]
    X_test = features[test_mask]
    y_test = labels[test_mask]

    # 创建数据集
    train_dataset = PersuasionDataset(X_train, y_train)
    val_dataset = PersuasionDataset(X_val, y_val)
    test_dataset = PersuasionDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader