import torch
import torch.nn as nn
import torch.nn.functional as F


class VoiceAuthCNN(nn.Module):
    def __init__(self, input_features=20, output_dim=512):
        super().__init__()
        
        # Более глубокая CNN архитектура с residual connections
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Добавляем еще один слой для лучшего извлечения признаков
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Global Average Pooling вместо flatten
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Полносвязные слои с dropout
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_dim)
        
        self.dropout = nn.Dropout(0.3)
        self.dropout_heavy = nn.Dropout(0.5)
        
        # Layer normalization для стабильности обучения
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, x):
        # Добавление канального измерения
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        # Конволюционные блоки с остаточными связями
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Полносвязные слои
        x = F.relu(self.fc1(x))
        x = self.dropout_heavy(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        # L2 нормализация эмбеддингов
        x = F.normalize(x, p=2, dim=1)
        return self.ln(x)


class AngularMarginLoss(nn.Module):
    """Более эффективная loss функция для speaker verification"""
    def __init__(self, margin=0.5, scale=64):
        super().__init__()
        self.margin = margin
        self.scale = scale
        
    def forward(self, embeddings, labels):
        # Косинусное сходство
        cos_sim = F.linear(F.normalize(embeddings), F.normalize(embeddings))
        
        # Создаем маску для позитивных пар
        labels = labels.unsqueeze(1)
        mask = (labels == labels.t()).float()
        
        # Применяем angular margin к позитивным парам
        cos_sim_margin = cos_sim - self.margin * mask
        
        # Масштабирование и softmax
        logits = cos_sim_margin * self.scale
        
        # Центрированный loss
        loss = F.cross_entropy(logits, torch.arange(len(labels)).to(labels.device))
        return loss