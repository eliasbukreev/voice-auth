import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
import warnings
import hashlib
import random  # Добавлен импорт для аугментации


class VoiceDataset(Dataset):
    def __init__(self, file_list, labels, n_mfcc=20, max_len=128, cache_dir="mfcc_cache", augment=False):
        if len(file_list) != len(labels):
            raise ValueError("file_list and labels must have the same length")

        self.file_list = file_list
        self.labels = labels
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        self.cache_dir = cache_dir
        self.augment = augment  # Флаг для включения/выключения аугментации

        # Создаем директорию для кэша
        os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]

        # Генерируем уникальный хеш для файла
        file_hash = hashlib.md5(file_path.encode()).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{file_hash}.npy")

        # Пытаемся загрузить из кэша
        if os.path.exists(cache_path):
            try:
                mfcc = np.load(cache_path)
                return torch.tensor(mfcc).float(), torch.tensor(label).long()
            except:
                pass

        # Если нет в кэше, обрабатываем файл
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, sr = librosa.load(file_path, sr=16000)

                # Применяем аугментацию только если включен флаг augment
                if self.augment:
                    # Добавление шума (50% вероятность)
                    if random.random() > 0.5:
                        noise = np.random.randn(len(y)) * 0.005
                        y = y + noise

                    # Изменение скорости (50% вероятность)
                    if random.random() > 0.5:
                        speed_factor = random.uniform(0.9, 1.1)
                        y = librosa.effects.time_stretch(y, rate=speed_factor)

        except Exception as e:
            # Возвращаем нулевой тензор при ошибке
            dummy = torch.zeros(self.n_mfcc, self.max_len)
            return dummy, torch.tensor(-1).long()

        # Извлечение MFCC
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self.n_mfcc,
            n_fft=512, hop_length=256
        )

        # Обработка длины
        if mfcc.shape[1] > self.max_len:
            start = np.random.randint(0, mfcc.shape[1] - self.max_len)
            mfcc = mfcc[:, start:start + self.max_len]
        else:
            pad_width = self.max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)),
                          mode='constant', constant_values=0)

        # Нормализация
        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True))
        mfcc = mfcc / (np.std(mfcc, axis=1, keepdims=True) + 1e-8)

        # Сохраняем в кэш
        try:
            np.save(cache_path, mfcc)
        except:
            pass

        return torch.tensor(mfcc).float(), torch.tensor(label).long()