import torch
import torch.nn as nn
import torch.nn.functional as F


class VoiceAuthCNN(nn.Module):
    def __init__(self, input_features=20, output_dim=256):
        super().__init__()
        # Конволюционные слои
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Автоматический расчет размера
        self._to_linear = None
        self._dummy_forward(torch.zeros(1, 1, input_features, 128))

        # Полносвязные слои
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.ln = nn.LayerNorm(output_dim)

    def _dummy_forward(self, x):
        # Конволюционные блоки
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)

        # Расчет размерности
        if self._to_linear is None:
            self._to_linear = x.numel() // x.shape[0]
        return x

    def forward(self, x):
        # Добавление канального измерения
        x = x.unsqueeze(1)

        # Конволюционная часть
        x = self._dummy_forward(x)

        # Выравнивание для полносвязных слоев
        x = x.view(x.size(0), -1)

        # Полносвязные слои
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Нормализация выходных эмбеддингов
        return self.ln(F.normalize(x, p=2, dim=1))