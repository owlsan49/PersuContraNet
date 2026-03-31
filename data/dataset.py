"""
数据加载模块 - PersuContraGraph 第一阶段
使用文本和6维说服力特征
"""

import os
import json
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List
from transformers import AutoTokenizer

# 6维特征名称
PERSUASION_FEATURES = [
    'Attack_on_reputation',
    'Justification',
    'Simplification',
    'Distraction',
    'Call',
    'Manipulative_wording'
]


def parse_persuasion_features(json_str):
    """解析说服力特征 - 将JSON转换为6维向量"""
    if pd.isna(json_str) or not isinstance(json_str, str):
        return np.zeros(6, dtype=np.float32)
    try:
        data = json.loads(json_str)
    except:
        return np.zeros(6, dtype=np.float32)

    if not isinstance(data, dict):
        return np.zeros(6, dtype=np.float32)

    features = []
    for feature in PERSUASION_FEATURES:
        if feature in data:
            is_used = data[feature].get('is_used', 'No')
            features.append(1.0 if str(is_used).strip().lower() == 'yes' else 0.0)
        else:
            features.append(0.0)
    return np.array(features, dtype=np.float32)


class PersuasionTextDataset(Dataset):
    """文本数据集 - 第一阶段训练"""

    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': self.labels[idx]
        }


def load_persuasion_data(
    train_files: List[str] = None,
    val_files: List[str] = None,
    test_files: List[str] = None,
    data_dir: str = "pcot_persu_data",
    tokenizer_name: str = "roberta-base",
    max_length: int = 512,
    batch_size: int = 16
):
    """
    加载说服力数据

    Args:
        train_files: 训练集文件列表
        val_files: 验证集文件列表
        test_files: 测试集文件列表
        data_dir: 数据目录
        tokenizer_name: tokenizer名称
        max_length: 最大序列长度
        batch_size: batch大小

    Returns:
        train_loader, val_loader, test_loader
    """
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    all_data = []
    splits = {'train': train_files, 'val': val_files, 'test': test_files}

    for split, files in splits.items():
        if not files:
            continue
        for f in files:
            path = os.path.join(data_dir, f)
            if os.path.exists(path):
                df = pd.read_csv(path)
                df['split'] = split
                all_data.append(df)
                print(f"Loaded {split}: {len(df)} from {f}")

    if not all_data:
        raise ValueError("No data loaded!")

    df_all = pd.concat(all_data, ignore_index=True)

    # 解析文本和标签
    texts = df_all['content'].fillna('').tolist()
    labels = np.array([parse_persuasion_features(row['generated_pred'])
                     for _, row in df_all.iterrows()], dtype=np.float32)

    # 按split划分
    train_mask = df_all['split'] == 'train'
    val_mask = df_all['split'] == 'val'
    test_mask = df_all['split'] == 'test'

    # 创建数据集
    train_dataset = PersuasionTextDataset(
        [t for m, t in zip(train_mask, texts) if m],
        labels[train_mask],
        tokenizer,
        max_length
    )
    val_dataset = PersuasionTextDataset(
        [t for m, t in zip(val_mask, texts) if m],
        labels[val_mask],
        tokenizer,
        max_length
    )
    test_dataset = PersuasionTextDataset(
        [t for m, t in zip(test_mask, texts) if m],
        labels[test_mask],
        tokenizer,
        max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Total: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    return train_loader, val_loader, test_loader