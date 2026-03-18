"""
训练脚本
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import prepare_data, DATASET_REGISTRY, register_dataset
from models.mlp import build_model, MODEL_REGISTRY
from config.config import DATA_DIR, MODEL_CONFIG, TRAIN_CONFIG, SPLIT_CONFIG


def train_epoch(model: nn.Module, dataloader: DataLoader,
                criterion: nn.Module, optimizer: optim.Optimizer, device: str):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), 100. * correct / total


def compute_class_metrics(all_labels: np.ndarray, all_preds: np.ndarray):
    """计算每个类别的precision, recall, f1"""
    metrics = {}

    for class_idx, class_name in [(0, 'Fake'), (1, 'Real')]:
        tp = ((all_preds == class_idx) & (all_labels == class_idx)).sum()
        tn = ((all_preds != class_idx) & (all_labels != class_idx)).sum()
        fp = ((all_preds == class_idx) & (all_labels != class_idx)).sum()
        fn = ((all_preds != class_idx) & (all_labels == class_idx)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[class_name] = {'precision': precision, 'recall': recall, 'f1': f1}

    return metrics


def evaluate(model: nn.Module, dataloader: DataLoader,
             criterion: nn.Module, device: str) -> Dict[str, float]:
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total

    # 计算每个类别的指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    class_metrics = compute_class_metrics(all_labels, all_preds)

    # Macro F1
    f1_macro = (class_metrics['Real']['f1'] + class_metrics['Fake']['f1']) / 2

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision_real': class_metrics['Real']['precision'] * 100,
        'recall_real': class_metrics['Real']['recall'] * 100,
        'f1_real': class_metrics['Real']['f1'] * 100,
        'precision_fake': class_metrics['Fake']['precision'] * 100,
        'recall_fake': class_metrics['Fake']['recall'] * 100,
        'f1_fake': class_metrics['Fake']['f1'] * 100,
        'f1_macro': f1_macro * 100
    }


def train(args):
    """训练主函数"""
    # 设置设备
    device = TRAIN_CONFIG['device'] if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 准备数据
    print("\n=== Loading Data ===")
    train_loader, val_loader, test_loader = prepare_data(
        data_dir=args.data_dir or DATA_DIR,
        dataset_names=args.datasets,
        val_size=SPLIT_CONFIG['val_size'],
        test_size=SPLIT_CONFIG['test_size'],
        random_seed=SPLIT_CONFIG['random_seed']
    )

    # 构建模型
    print("\n=== Building Model ===")
    model = build_model(
        MODEL_CONFIG['model_name'],
        input_dim=MODEL_CONFIG['input_dim'],
        hidden_dims=MODEL_CONFIG['hidden_dims'],
        num_classes=MODEL_CONFIG['num_classes'],
        dropout=MODEL_CONFIG['dropout']
    )
    model = model.to(device)
    print(f"Model: {MODEL_CONFIG['model_name']}")
    print(f"Parameters: {model.get_num_params()}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )

    # 训练循环
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None

    print("\n=== Training ===")
    for epoch in range(TRAIN_CONFIG['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%, "
              f"F1(real): {val_metrics['f1_real']:.2f}%, F1(fake): {val_metrics['f1_fake']:.2f}%")

        # 早停，使用macro F1
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_CONFIG['early_stopping_patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # 加载最佳模型并测试
    if best_model_state:
        model.load_state_dict(best_model_state)

    print("\n=== Test Results ===")
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"\n以 Real(1) 为正类:")
    print(f"  Precision: {test_metrics['precision_real']:.2f}%")
    print(f"  Recall: {test_metrics['recall_real']:.2f}%")
    print(f"  F1-score: {test_metrics['f1_real']:.2f}%")
    print(f"\n以 Fake(0) 为正类:")
    print(f"  Precision: {test_metrics['precision_fake']:.2f}%")
    print(f"  Recall: {test_metrics['recall_fake']:.2f}%")
    print(f"  F1-score: {test_metrics['f1_fake']:.2f}%")
    print(f"\nMacro F1: {test_metrics['f1_macro']:.2f}%")

    # 保存模型
    if args.save_path:
        torch.save(model.state_dict(), args.save_path)
        print(f"\nModel saved to {args.save_path}")

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Persuasion Detection Model")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Data directory")
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets to load")
    parser.add_argument("--save_path", type=str, default="models/persuasion_mlp.pth", help="Save path")
    args = parser.parse_args()

    train(args)