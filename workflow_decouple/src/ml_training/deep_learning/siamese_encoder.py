"""
Siamese Network Encoder: MLP-based embedding network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseEncoder(nn.Module):
    """
    MLP-based Siamese Network Encoder
    
    将checkpoint特征映射到低维embedding空间，
    使得相似的checkpoints在embedding空间中更接近。
    
    Args:
        input_dim: 输入特征维度
        hidden_dims: 隐藏层维度列表
        output_dim: 输出embedding维度
        dropout: Dropout率
    """
    
    def __init__(
        self,
        input_dim: int = 16,
        hidden_dims: list = [64, 128, 64],
        output_dim: int = 32,
        dropout: float = 0.2
    ):
        super(SiameseEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 构建MLP网络
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [batch_size, input_dim] 输入特征
            
        Returns:
            embeddings: [batch_size, output_dim] embedding向量
        """
        embeddings = self.encoder(x)
        
        # L2 normalization（使embedding在单位球面上）
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def get_embedding(self, x):
        """获取embedding（推理模式）"""
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x)
            return self.forward(x)
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
        }, path)
        print(f"✅ 模型已保存到: {path}")
    
    @classmethod
    def load(cls, path, **kwargs):
        """加载模型"""
        checkpoint = torch.load(path)
        model = cls(
            input_dim=checkpoint['input_dim'],
            output_dim=checkpoint['output_dim'],
            **kwargs
        )
        model.load_state_dict(checkpoint['state_dict'])
        print(f"✅ 模型已从 {path} 加载")
        return model


class SiameseContrastiveLoss(nn.Module):
    """
    对比损失（可选）
    用于预训练或辅助损失
    """
    
    def __init__(self, margin: float = 1.0):
        super(SiameseContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1, embedding2, label):
        """
        Args:
            embedding1: [batch_size, embed_dim]
            embedding2: [batch_size, embed_dim]
            label: [batch_size] 0表示相似，1表示不相似
        """
        euclidean_distance = F.pairwise_distance(embedding1, embedding2)
        
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        return loss_contrastive


class TripletLoss(nn.Module):
    """
    Triplet Loss: 让anchor-positive距离小于anchor-negative距离
    
    Loss = max(0, d(a,p)² - d(a,n)² + margin)
    """
    
    def __init__(self, margin: float = 0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: [batch_size, embed_dim]
            positive: [batch_size, embed_dim]
            negative: [batch_size, embed_dim]
        
        Returns:
            loss: scalar tensor
        """
        # 计算距离
        dist_ap = F.pairwise_distance(anchor, positive)
        dist_an = F.pairwise_distance(anchor, negative)
        
        # Triplet loss
        losses = F.relu(dist_ap - dist_an + self.margin)
        
        return losses.mean()
