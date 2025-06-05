import os
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
from collections import defaultdict
import random

from model import VoiceAuthCNN, AngularMarginLoss


class ImprovedTripletLoss(nn.Module):
    def __init__(self, margin=1.0, mining='hard'):
        super().__init__()
        self.margin = margin
        self.mining = mining
        
    def forward(self, embeddings, labels):
        # Вычисляем все попарные расстояния
        dist_mat = torch.cdist(embeddings, embeddings, p=2)
        
        batch_size = embeddings.size(0)
        losses = []
        
        for i in range(batch_size):
            anchor_label = labels[i]
            
            # Позитивные образцы (тот же класс)
            pos_mask = (labels == anchor_label) & (torch.arange(batch_size).to(labels.device) != i)
            if not pos_mask.any():
                continue
                
            # Негативные образцы (другие классы)  
            neg_mask = labels != anchor_label
            if not neg_mask.any():
                continue
            
            pos_dists = dist_mat[i][pos_mask]
            neg_dists = dist_mat[i][neg_mask]
            
            if self.mining == 'hard':
                # Hard positive: самый дальний из своего класса
                hardest_pos_dist = pos_dists.max()
                # Hard negative: самый близкий из чужого класса
                hardest_neg_dist = neg_dists.min()
                
                loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
                losses.append(loss)
            else:
                # Semi-hard mining
                for pos_dist in pos_dists:
                    for neg_dist in neg_dists:
                        loss = F.relu(pos_dist - neg_dist + self.margin)
                        if loss > 0:
                            losses.append(loss)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, requires_grad=True)


def create_balanced_batches(dataset, batch_size, samples_per_class=4):
    """Создание сбалансированных батчей для лучшего обучения"""
    labels_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        labels_to_indices[label.item()].append(idx)
    
    batches = []
    all_labels = list(labels_to_indices.keys())
    
    while True:
        batch_indices = []
        batch_labels = []
        
        # Случайно выбираем классы для батча
        selected_classes = random.sample(all_labels, min(len(all_labels), batch_size // samples_per_class))
        
        for class_label in selected_classes:
            indices = labels_to_indices[class_label]
            if len(indices) >= samples_per_class:
                selected_indices = random.sample(indices, samples_per_class)
            else:
                selected_indices = random.choices(indices, k=samples_per_class)
            
            batch_indices.extend(selected_indices)
            batch_labels.extend([class_label] * len(selected_indices))
        
        if len(batch_indices) >= batch_size:
            yield batch_indices[:batch_size]
        else:
            yield batch_indices


def train_model(train_dataset, val_dataset, num_classes, epochs=100, batch_size=32, device="cpu", save_path="models"):
    os.makedirs(save_path, exist_ok=True)
    print(f"Training parameters:")
    print(f"- Epochs: {epochs}")
    print(f"- Batch size: {batch_size}")
    print(f"- Device: {device}")
    print(f"- Classes: {num_classes}")

    # Создаем модель с большим размером эмбеддинга
    model = VoiceAuthCNN(output_dim=512).to(device)
    
    # Комбинированная loss функция
    triplet_loss = ImprovedTripletLoss(margin=1.2, mining='hard')
    angular_loss = AngularMarginLoss(margin=0.4, scale=32)
    
    # Оптимизатор с более агрессивными параметрами
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=0.001, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Более агрессивный scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Создаем сбалансированные лоадеры
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if device == "cuda" else 2,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if device == "cuda" else 2,
        pin_memory=True
    )

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    print("\nStarting improved training...")
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Убираем образцы с ошибочными лейблами
            valid_mask = labels >= 0
            if not valid_mask.any():
                continue
                
            inputs = inputs[valid_mask]
            labels = labels[valid_mask]
            
            if len(inputs) < 2:
                continue

            # Получаем эмбеддинги
            embeddings = model(inputs)
            
            # Комбинированная loss
            t_loss = triplet_loss(embeddings, labels)
            a_loss = angular_loss(embeddings, labels)
            loss = 0.7 * t_loss + 0.3 * a_loss
            
            # Оптимизация
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            # Обновляем learning rate каждый батч
            scheduler.step(epoch + batch_idx / len(train_loader))

        # Валидация
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                valid_mask = labels >= 0
                if not valid_mask.any():
                    continue
                    
                inputs = inputs[valid_mask]
                labels = labels[valid_mask]
                
                if len(inputs) < 2:
                    continue

                embeddings = model(inputs)
                t_loss = triplet_loss(embeddings, labels)
                a_loss = angular_loss(embeddings, labels)
                loss = 0.7 * t_loss + 0.3 * a_loss
                
                val_loss += loss.item()
                val_batches += 1

        # Нормализация потерь
        if num_batches > 0:
            train_loss /= num_batches
        if val_batches > 0:
            val_loss /= val_batches
            
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Early stopping и сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': val_loss
            }, os.path.join(save_path, "best_model.pth"))
        else:
            patience_counter += 1

        # Вывод статистики
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Time: {epoch_time:.1f}s | "
              f"LR: {current_lr:.1e} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Best: {best_val_loss:.4f} | "
              f"Patience: {patience_counter}")
        
        # Early stopping
        if patience_counter >= 15:
            print("Early stopping triggered!")
            break

    # Загрузка лучшей модели
    checkpoint = torch.load(os.path.join(save_path, "best_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model