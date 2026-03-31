"""
训练脚本 - PersuContraGraph 第一阶段
使用文本编码器和persu_head进行6类多标签分类
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import load_persuasion_data, PERSUASION_FEATURES
from models.persucontra_graph import PersuContraGraph


def load_config(config_path):
    """加载YAML配置"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_epoch(model: nn.Module, dataloader: DataLoader,
                criterion: nn.Module, optimizer: optim.Optimizer, device: str):
    """训练一个epoch"""
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, stage='persu')
        preds = outputs['persu_pred']
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: str):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, stage='persu')
            preds = outputs['persu_pred']
            loss = criterion(preds, labels)
            total_loss += loss.item()

            preds_binary = (preds > 0.5).float()
            all_preds.append(preds_binary.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 计算每类F1
    f1_scores = []
    for i in range(6):
        f1 = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        f1_scores.append(f1)

    macro_f1 = np.mean(f1_scores)
    micro_f1 = f1_score(all_labels.flatten(), all_preds.flatten(), average='micro', zero_division=0)

    return {
        'loss': total_loss / len(dataloader),
        'macro_f1': macro_f1 * 100,
        'micro_f1': micro_f1 * 100,
        'per_class_f1': {PERSUASION_FEATURES[i]: f1_scores[i] * 100 for i in range(6)}
    }


def train(args):
    """训练主函数"""
    # 加载配置
    config = load_config(args.config)

    device = config['train'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    print("\n=== Loading Data ===")
    train_loader, val_loader, test_loader = load_persuasion_data(
        train_files=config['data']['train_files'],
        val_files=config['data']['val_files'],
        test_files=config['data']['test_files'],
        data_dir=config['data']['data_dir'],
        tokenizer_name=config['model']['encoder_name'],
        max_length=config['data']['max_length'],
        batch_size=config['data']['batch_size']
    )

    # 构建模型
    print("\n=== Building Model ===")
    model = PersuContraGraph(
        encoder_name=config['model']['encoder_name'],
        pooling=config['model']['pooling'],
        freeze_encoder=config['model']['freeze_encoder'],
        persu_hidden=config['model']['persu_hidden'],
        cl_hidden=config['model']['cl_hidden'],
        cl_output=config['model']['cl_output'],
        cls_hidden=config['model']['cls_hidden'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']
    )
    model = model.to(device)

    # 设置第一阶段训练
    model.set_trainable(config['train']['stage'])
    print(f"Trainable params: {model.get_trainable_params()} / {model.get_total_params()}")

    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['train']['learning_rate'],
        weight_decay=config['train']['weight_decay']
    )

    # 训练循环
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None

    print("\n=== Training Stage 1: Persuasion Prediction ===")
    for epoch in range(config['train']['epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
              f"Val Macro F1: {val_metrics['macro_f1']:.2f}% | Micro F1: {val_metrics['micro_f1']:.2f}%")

        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['train']['patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # 加载最佳模型并测试
    if best_model_state:
        model.load_state_dict(best_model_state)
        model = model.to(device)

    print("\n=== Test Results ===")
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test Macro F1: {test_metrics['macro_f1']:.2f}%")
    print(f"Test Micro F1: {test_metrics['micro_f1']:.2f}%")
    print("\nPer-class F1:")
    for name, f1 in test_metrics['per_class_f1'].items():
        print(f"  {name}: {f1:.2f}%")

    # 保存模型
    if config['save']['save_path']:
        os.makedirs(os.path.dirname(config['save']['save_path']), exist_ok=True)
        torch.save(model.state_dict(), config['save']['save_path'])
        print(f"\nModel saved to {config['save']['save_path']}")

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PersuContraGraph - Stage 1")
    parser.add_argument("--config", type=str, default="config/stage1.yaml", help="Config file path")
    args = parser.parse_args()

    train(args)