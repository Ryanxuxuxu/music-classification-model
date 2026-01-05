import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class MusicSegmentCNN(nn.Module):
    """音乐时间分割CNN模型 - 支持one-hot标签"""
    
    def __init__(self, input_shape: Tuple[int, int], num_classes: int, dropout: float = 0.5, output_logits: bool = True):
        """
        初始化模型
        
        Args:
            input_shape: 输入形状 (sequence_length, n_mels)
            num_classes: 类别数
            dropout: Dropout率
            output_logits: 是否输出logits（True用于训练，False用于推理）
        """
        super(MusicSegmentCNN, self).__init__()
        
        self.sequence_length, self.n_mels = input_shape
        self.num_classes = num_classes
        self.dropout = dropout
        self.output_logits = output_logits
        
        # 卷积层
        self.conv1 = nn.Conv1d(self.n_mels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        
        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 计算经过卷积和池化后的序列长度
        conv_output_length = self.sequence_length // 16  # 经过4次池化
        
        # 全连接层
        self.fc1 = nn.Linear(512 * conv_output_length, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, sequence_length, n_mels)
        
        Returns:
            输出张量 (batch_size, num_classes) - logits或概率分布
        """
        # 转换维度: (batch_size, sequence_length, n_mels) -> (batch_size, n_mels, sequence_length)
        x = x.transpose(1, 2)
        
        # 卷积层
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout_layer(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout_layer(x)
        
        x = self.fc3(x)
        
        # 根据设置决定输出logits还是概率分布
        if not self.output_logits:
            x = F.softmax(x, dim=1)
        
        return x
    
    def get_loss_function(self, loss_type: str = 'cross_entropy'):
        """
        获取适合one-hot标签的损失函数
        
        Args:
            loss_type: 损失函数类型 ('cross_entropy', 'bce', 'mse')
        
        Returns:
            损失函数
        """
        if loss_type == 'cross_entropy':
            # 对于one-hot标签，使用CrossEntropyLoss（内部会处理logits）
            return nn.CrossEntropyLoss()
        elif loss_type == 'bce':
            # 二元交叉熵，需要sigmoid输出
            return nn.BCEWithLogitsLoss()
        elif loss_type == 'mse':
            # 均方误差，适用于one-hot标签
            return nn.MSELoss()
        else:
            raise ValueError(f"不支持的损失函数类型: {loss_type}")
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测概率分布（用于推理）
        
        Args:
            x: 输入张量
        
        Returns:
            概率分布 (batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            # 临时设置为输出概率
            original_output_logits = self.output_logits
            self.output_logits = False
            probs = self.forward(x)
            self.output_logits = original_output_logits
        return probs

class MusicSegmentCNNSequence(nn.Module):
    """音乐时间分割CNN模型 - 序列输出版本"""
    
    def __init__(self, input_shape: Tuple[int, int], num_classes: int, dropout: float = 0.5):
        """
        初始化模型
        
        Args:
            input_shape: 输入形状 (sequence_length, n_mels)
            num_classes: 类别数
            dropout: Dropout率
        """
        super(MusicSegmentCNNSequence, self).__init__()
        
        self.sequence_length, self.n_mels = input_shape
        self.num_classes = num_classes
        self.dropout = dropout
        
        # 卷积层
        self.conv1 = nn.Conv1d(self.n_mels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        
        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 计算经过卷积和池化后的序列长度
        conv_output_length = self.sequence_length // 16  # 经过4次池化
        
        # 全连接层
        self.fc1 = nn.Linear(512 * conv_output_length, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, sequence_length, n_mels)
        
        Returns:
            输出张量 (batch_size, sequence_length, num_classes)
        """
        batch_size = x.size(0)
        
        # 转换维度: (batch_size, sequence_length, n_mels) -> (batch_size, n_mels, sequence_length)
        x = x.transpose(1, 2)
        
        # 卷积层
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        # 展平
        x = x.view(batch_size, -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout_layer(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout_layer(x)
        
        x = self.fc3(x)
        
        # 重复输出到序列长度
        x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        return x

def create_model(input_shape: Tuple[int, int], num_classes: int, model_type: str = 'classification') -> nn.Module:
    """
    创建模型的便捷函数
    
    Args:
        input_shape: 输入形状 (sequence_length, n_mels)
        num_classes: 类别数
        model_type: 模型类型 ('classification' 或 'sequence')
    
    Returns:
        模型实例
    """
    if model_type == 'classification':
        return MusicSegmentCNN(input_shape, num_classes)
    elif model_type == 'sequence':
        return MusicSegmentCNNSequence(input_shape, num_classes)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

# 示例用法
if __name__ == "__main__":
    # 创建模型
    input_shape = (128, 256)  # (sequence_length, n_mels)
    num_classes = 15  # 对应我们的15个音乐段落类别
    
    model = create_model(input_shape, num_classes, 'classification')
    print(f"分类模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 测试前向传播
    batch_size = 4
    x = torch.randn(batch_size, *input_shape)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试one-hot标签兼容性
    # 模拟one-hot标签 (batch_size, num_classes)
    one_hot_labels = torch.zeros(batch_size, num_classes)
    one_hot_labels[0, 7] = 1.0   # 第一个样本是intro
    one_hot_labels[1, 3] = 1.0   # 第二个样本是chorus
    one_hot_labels[2, 14] = 1.0  # 第三个样本是verse
    one_hot_labels[3, 2] = 1.0   # 第四个样本是bridge
    
    print(f"One-hot标签形状: {one_hot_labels.shape}")
    
    # 获取适合one-hot标签的损失函数
    loss_fn = model.get_loss_function('cross_entropy')
    loss = loss_fn(output, one_hot_labels.argmax(dim=1))  # CrossEntropyLoss需要类别索引
    print(f"使用CrossEntropyLoss的损失值: {loss.item():.4f}")
    
    # 测试概率预测
    probs = model.predict_proba(x)
    print(f"概率输出形状: {probs.shape}")
    print(f"第一个样本的预测概率: {probs[0]}")
    
    # 序列输出模型
    model_seq = create_model(input_shape, num_classes, 'sequence')
    print(f"序列模型参数数量: {sum(p.numel() for p in model_seq.parameters())}")
    
    output_seq = model_seq(x)
    print(f"序列输出形状: {output_seq.shape}") 
    
    # 添加训练循环
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 80
    
    print("开始训练...")
    for epoch in range(num_epochs):
        # 模拟训练数据
        x = torch.randn(batch_size, *input_shape)
        one_hot_labels = torch.zeros(batch_size, num_classes)
        one_hot_labels[0, 7] = 1.0
        one_hot_labels[1, 3] = 1.0
        one_hot_labels[2, 14] = 1.0
        one_hot_labels[3, 2] = 1.0
        
        # 前向传播
        output = model(x)
        loss = loss_fn(output, one_hot_labels.argmax(dim=1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
    
    print("训练完成!")