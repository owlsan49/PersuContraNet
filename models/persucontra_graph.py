"""
PersuContraGraph 模型 - 对比学习的说服力检测
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import AutoModel


class PersuHead(nn.Module):
    """Persuasion Head - 说服力预测头 (可训练)"""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        output_dim: int = 6,  # 6维说服特征
        dropout: float = 0.3
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # 输出0-1之间的概率
        )

    def forward(self, x):
        return self.network(x)


class CLHead(nn.Module):
    """Contrastive Head - 对比学习头 (可训练)"""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        output_dim: int = 128,  # 对比学习投影维度
        dropout: float = 0.3
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class ClassificationHead(nn.Module):
    """Classification Head - 分类头"""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.network(x)


class PersuContraGraph(nn.Module):
    """
    PersuContraGraph 模型

    使用预训练的Transformer作为文本编码器，包含:
    - Encoder: 预训练Transformer (如roberta-large)
    - PersuHead: 说服力预测头 (可训练)
    - CLHead: 对比学习头 (可训练)
    - ClassificationHead: 分类头

    支持不同阶段进行前向传播:
    - 'persu': 训练PersuHead
    - 'cl': 训练CLHead
    - 'cls': 训练ClassificationHead
    - 'full': 完整模型训练
    """

    def __init__(
        self,
        encoder_name: str = "roberta-large",
        pooling: str = "mean",
        freeze_encoder: bool = False,
        persu_hidden: int = 256,
        cl_hidden: int = 256,
        cl_output: int = 128,
        cls_hidden: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        # 预训练文本编码器
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.hidden_dim = self.encoder.config.hidden_size
        self.pooling = pooling

        # 冻结编码器
        if freeze_encoder:
            self.freeze_encoder()

        # 说服力预测头 (可训练)
        self.persu_head = PersuHead(
            input_dim=self.hidden_dim,
            hidden_dim=persu_hidden,
            output_dim=6,
            dropout=dropout
        )

        # 对比学习头 (可训练)
        self.cl_head = CLHead(
            input_dim=self.hidden_dim,
            hidden_dim=cl_hidden,
            output_dim=cl_output,
            dropout=dropout
        )

        # 分类头
        self.cls_head = ClassificationHead(
            input_dim=self.hidden_dim,
            hidden_dim=cls_hidden,
            num_classes=num_classes,
            dropout=dropout
        )

    def _pooling(self, outputs, attention_mask=None):
        """池化操作"""
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        elif self.pooling == "max":
            return torch.max(hidden_states, dim=1)[0]
        else:  # "mean"
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask
            return torch.mean(hidden_states, dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        stage: str = 'cls'
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            stage: 当前阶段，可选 'persu', 'cl', 'cls', 'full'

        Returns:
            包含不同输出的字典
        """
        # 编码文本
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self._pooling(outputs, attention_mask)

        results = {'embeddings': embeddings}

        if stage == 'persu':
            # 训练PersuHead阶段
            results['persu_pred'] = self.persu_head(embeddings)

        elif stage == 'cl':
            # 训练CLHead阶段
            results['cl_embedding'] = self.cl_head(embeddings)

        elif stage == 'cls':
            # 训练ClassificationHead阶段
            results['cls_logits'] = self.cls_head(embeddings)

        elif stage == 'full':
            # 完整模型训练
            results['persu_pred'] = self.persu_head(embeddings)
            results['cl_embedding'] = self.cl_head(embeddings)
            results['cls_logits'] = self.cls_head(embeddings)

        else:
            raise ValueError(f"Unknown stage: {stage}. Available: 'persu', 'cl', 'cls', 'full'")

        return results

    def freeze_encoder(self):
        """冻结编码器参数"""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """解冻编码器参数"""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_persu_head(self):
        """冻结PersuHead参数"""
        for param in self.persu_head.parameters():
            param.requires_grad = False

    def unfreeze_persu_head(self):
        """解冻PersuHead参数"""
        for param in self.persu_head.parameters():
            param.requires_grad = True

    def freeze_cl_head(self):
        """冻结CLHead参数"""
        for param in self.cl_head.parameters():
            param.requires_grad = False

    def unfreeze_cl_head(self):
        """解冻CLHead参数"""
        for param in self.cl_head.parameters():
            param.requires_grad = True

    def freeze_cls_head(self):
        """冻结ClassificationHead参数"""
        for param in self.cls_head.parameters():
            param.requires_grad = False

    def unfreeze_cls_head(self):
        """解冻ClassificationHead参数"""
        for param in self.cls_head.parameters():
            param.requires_grad = True

    def set_trainable(self, stage: str):
        """
        根据阶段设置哪些参数可训练

        Args:
            stage: 训练阶段
                - 'pretrain_persu': 预训练PersuHead，只训练persu_head
                - 'pretrain_cl': 预训练CLHead，只训练cl_head
                - 'finetune_cls': 微调分类头，只训练cls_head
                - 'full': 完整模型训练，所有参数都可训练
        """
        # 先冻结所有参数
        for param in self.parameters():
            param.requires_grad = False

        if stage == 'pretrain_persu':
            self.unfreeze_persu_head()
            print(f"Stage: {stage} | Trainable: persu_head")

        elif stage == 'pretrain_cl':
            self.unfreeze_cl_head()
            print(f"Stage: {stage} | Trainable: cl_head")

        elif stage == 'finetune_cls':
            self.unfreeze_cls_head()
            print(f"Stage: {stage} | Trainable: cls_head")

        elif stage == 'full':
            for param in self.parameters():
                param.requires_grad = True
            print(f"Stage: {stage} | Trainable: all")

        else:
            raise ValueError(f"Unknown stage: {stage}")

    def get_trainable_params(self):
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self):
        """获取总参数数量"""
        return sum(p.numel() for p in self.parameters())


# 模型注册
MODEL_REGISTRY = {
    'persu_contra_graph': PersuContraGraph,
}


def build_persucontra_graph(
    encoder_name: str = "roberta-large",
    pooling: str = "mean",
    freeze_encoder: bool = False,
    persu_hidden: int = 256,
    cl_hidden: int = 256,
    cl_output: int = 128,
    cls_hidden: int = 256,
    num_classes: int = 2,
    dropout: float = 0.3
) -> PersuContraGraph:
    """构建PersuContraGraph模型"""
    return PersuContraGraph(
        encoder_name=encoder_name,
        pooling=pooling,
        freeze_encoder=freeze_encoder,
        persu_hidden=persu_hidden,
        cl_hidden=cl_hidden,
        cl_output=cl_output,
        cls_hidden=cls_hidden,
        num_classes=num_classes,
        dropout=dropout
    )