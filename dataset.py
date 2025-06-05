import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
import warnings
import hashlib
import random


class VoiceDataset(Dataset):
    def __init__(self, file_list, labels, n_mfcc=40, max_len=256, cache_dir="mfcc_cache", augment=False):
        if len(file_list) != len(labels):
            raise ValueError("file_list and labels must have the same length")

        self.file_list = file_list
        self.labels = labels
        self.n_mfcc = n_mfcc  # Увеличили количество MFCC коэффициентов
        self.max_len = max_len  # Увеличили длину последовательности
        self.cache_dir = cache_dir
        self.augment = augment

        # Создаем директорию для кэша
        os.makedirs(cache_dir, exist_ok=True)
        
        # Параметры для более качественного извлечения признаков
        self.sr = 16000
        self.n_fft = 1024  # Увеличили для лучшего частотного разрешения
        self.hop_length = 256
        self.win_length = 1024

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]

        # Генерируем уникальный хеш для файла с учетом параметров
        params_str = f"{self.n_mfcc}_{self.max_len}_{self.n_fft}_{self.hop_length}"
        file_hash = hashlib.md5(f"{file_path}_{params_str}".encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{file_hash}.npy")

        # Пытаемся загрузить из кэша (только если не нужна аугментация)
        if not self.augment and os.path.exists(cache_path):
            try:
                features = np.load(cache_path)
                return torch.tensor(features).float(), torch.tensor(label).long()
            except:
                pass

        # Если нет в кэше, обрабатываем файл
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, sr = librosa.load(file_path, sr=self.sr)
                
                # Проверяем минимальную длину аудио
                if len(y) < self.sr * 0.5:  # Минимум 0.5 секунды
                    # Дополняем нулями или повторяем
                    min_len = int(self.sr * 0.5)
                    if len(y) > 0:
                        y = np.tile(y, (min_len // len(y)) + 1)[:min_len]
                    else:
                        y = np.zeros(min_len)

                # Применяем аугментацию только при тренировке
                if self.augment:
                    y = self._apply_augmentation(y)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Возвращаем нулевой тензор при ошибке
            dummy = torch.zeros(self.n_mfcc, self.max_len)
            return dummy, torch.tensor(-1).long()

        # Извлечение улучшенных признаков
        features = self._extract_features(y)
        
        # Сохраняем в кэш только если не используется аугментация
        if not self.augment:
            try:
                np.save(cache_path, features)
            except:
                pass

        return torch.tensor(features).float(), torch.tensor(label).long()
    
    def _apply_augmentation(self, y):
        """Более качественная аугментация данных"""
        augmented = y.copy()
        
        # 1. Добавление шума (30% вероятность)
        if random.random() < 0.3:
            noise_factor = random.uniform(0.001, 0.01)
            noise = np.random.randn(len(y)) * noise_factor * np.std(y)
            augmented = augmented + noise
        
        # 2. Изменение скорости (40% вероятность)
        if random.random() < 0.4:
            speed_factor = random.uniform(0.85, 1.15)
            try:
                augmented = librosa.effects.time_stretch(augmented, rate=speed_factor)
            except:
                pass
        
        # 3. Изменение тональности (30% вероятность)
        if random.random() < 0.3:
            n_steps = random.uniform(-2, 2)
            try:
                augmented = librosa.effects.pitch_shift(augmented, sr=self.sr, n_steps=n_steps)
            except:
                pass
        
        # 4. Добавление эха (20% вероятность)
        if random.random() < 0.2:
            delay = random.randint(int(0.05 * self.sr), int(0.2 * self.sr))
            decay = random.uniform(0.1, 0.3)
            if len(augmented) > delay:
                echo = np.zeros_like(augmented)
                echo[delay:] = augmented[:-delay] * decay
                augmented = augmented + echo
        
        # 5. Случайная обрезка и сдвиг (50% вероятность)
        if random.random() < 0.5 and len(augmented) > self.sr:
            max_offset = min(len(augmented) - self.sr, int(0.1 * self.sr))
            if max_offset > 0:
                offset = random.randint(0, max_offset)
                augmented = augmented[offset:]
        
        return augmented
    
    def _extract_features(self, y):
        """Извлечение улучшенных признаков"""
        # 1. MFCC признаки
        mfcc = librosa.feature.mfcc(
            y=y, sr=self.sr, n_mfcc=self.n_mfcc,
            n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length
        )
        
        # 2. Delta и Delta-Delta признаки для динамики
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # 3. Спектральные признаки
        spectral_centroids = librosa.feature.spectral_centroid(
            y=y, sr=self.sr, hop_length=self.hop_length
        )
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=self.sr, hop_length=self.hop_length
        )
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            y, hop_length=self.hop_length
        )
        
        # 4. Хромограмма
        chroma = librosa.feature.chroma_stft(
            y=y, sr=self.sr, hop_length=self.hop_length, n_fft=self.n_fft
        )
        
        # Объединяем все признаки
        features = np.vstack([
            mfcc,
            delta_mfcc,
            delta2_mfcc,
            spectral_centroids,
            spectral_rolloff, 
            zero_crossing_rate,
            chroma
        ])
        
        # Обработка длины
        if features.shape[1] > self.max_len:
            # Случайная обрезка при аугментации, центральная при валидации
            if self.augment:
                start = np.random.randint(0, features.shape[1] - self.max_len)
            else:
                start = (features.shape[1] - self.max_len) // 2
            features = features[:, start:start + self.max_len]
        else:
            # Дополнение нулями
            pad_width = self.max_len - features.shape[1]
            features = np.pad(features, ((0, 0), (0, pad_width)),
                            mode='constant', constant_values=0)
        
        # Улучшенная нормализация
        # Нормализация по каждому признаку отдельно
        features = (features - np.mean(features, axis=1, keepdims=True))
        std = np.std(features, axis=1, keepdims=True)
        features = features / (std + 1e-8)
        
        # Дополнительная робастная нормализация
        features = np.clip(features, -3, 3)  # Обрезаем выбросы
        
        return features