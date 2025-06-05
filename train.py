import os
import torch
import time
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F

from model import VoiceAuthCNN


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        losses = F.relu(pos_dist - neg_dist + self.margin)
        return torch.mean(losses)


def train_model(train_dataset, val_dataset, num_classes, epochs=100, batch_size=32, device="cpu", save_path="models"):
    os.makedirs(save_path, exist_ok=True)
    print(f"Training parameters:")
    print(f"- Epochs: {epochs}")
    print(f"- Batch size: {batch_size}")
    print(f"- Device: {device}")
    print(f"- Save path: {save_path}")

    model = VoiceAuthCNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = TripletLoss(margin=0.5)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if device == "cuda" else 0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4 if device == "cuda" else 0,
        pin_memory=True
    )

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    print("\nStarting training...")
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        # Обучение
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)

            # Генерация эмбеддингов
            embeddings = model(inputs)

            # Создание триплетов (anchor, positive, negative)
            # Простая стратегия: каждый третий элемент как negative
            anchor = embeddings[::3]
            positive = embeddings[1::3]
            negative = embeddings[2::3]

            # Выравнивание размеров
            min_size = min(anchor.size(0), positive.size(0), negative.size(0))
            if min_size < 1:
                continue

            anchor = anchor[:min_size]
            positive = positive[:min_size]
            negative = negative[:min_size]

            # Вычисление потерь
            loss = criterion(anchor, positive, negative)

            # Оптимизация
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * min_size

        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                embeddings = model(inputs)

                # Создание триплетов для валидации
                anchor = embeddings[::3]
                positive = embeddings[1::3]
                negative = embeddings[2::3]

                min_size = min(anchor.size(0), positive.size(0), negative.size(0))
                if min_size < 1:
                    continue

                loss = criterion(
                    anchor[:min_size],
                    positive[:min_size],
                    negative[:min_size]
                )
                val_loss += loss.item() * min_size

        # Нормализация потерь
        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Обновление learning rate
        scheduler.step(val_loss)

        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))

        # Вывод статистики
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Time: {epoch_time:.1f}s | "
              f"LR: {current_lr:.1e} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Best Val: {best_val_loss:.4f}")

    # Загрузка лучшей модели
    model.load_state_dict(torch.load(os.path.join(save_path, "best_model.pth")))
    return model