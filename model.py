import torch.nn as nn
import torch.nn.functional as F

class VoiceAuthCNN(nn.Module):
    def __init__(self, input_features=20, output_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)

        self.fc1 = nn.Linear(64 * (input_features // 4) * (128 // 4), 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, 20, 128]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # 256-мерный вектор