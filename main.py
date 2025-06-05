import os
import glob
import random
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from dataset import VoiceDataset
from train import train_model
from evaluate import evaluate_model
from utils import convert_common_voice_to_wav
import torch
import shutil

# ======== КОНФИГУРАЦИЯ ========
TSV_PATH = r"C:\Users\Лиза\Desktop\Диплом\voice-auth\en\train.tsv"
AUDIO_DIR = r"C:\Users\Лиза\Desktop\Диплом\voice-auth\en\clips"
PROCESSED_DIR = "full_data"  # Папка для всех сконвертированных файлов
MIN_FILES_PER_USER = 5  # Минимальное количество файлов на пользователя
TEST_SIZE = 0.2  # Размер тестовой выборки
VAL_SIZE = 0.25   # Размер валидационной выборки (от оставшихся)
BATCH_SIZE = 128   # Размер батча для обучения
EPOCHS = 50       # Количество эпох обучения

# ======== ПОДГОТОВКА ========
print("\n" + "="*80)
print("VOICE AUTHENTICATION SYSTEM - FULL DATASET PROCESSING")
print("="*80 + "\n")

# Очистка предыдущих данных (опционально)
if os.path.exists(PROCESSED_DIR):
    print(f"⚠️ Warning: Output directory {PROCESSED_DIR} already exists")
    # shutil.rmtree(PROCESSED_DIR)  # Раскомментировать для очистки

# ======== КОНВЕРТАЦИЯ ========
print("\n===== AUDIO CONVERSION =====")
converted_files = convert_common_voice_to_wav(
    tsv_path=TSV_PATH,
    input_dir=AUDIO_DIR,
    output_dir=PROCESSED_DIR
)

if converted_files == 0:
    print("❌ Conversion failed. Exiting.")
    exit(1)

# ======== ПОДГОТОВКА ДАННЫХ ========
print("\n===== DATA PREPARATION =====")
files = glob.glob(os.path.join(PROCESSED_DIR, "user_*.wav"))
print(f"Found {len(files)} WAV files in {PROCESSED_DIR}")

if len(files) == 0:
    print("❌ No files found! Exiting.")
    exit(1)

# Группировка по пользователям
user_files = defaultdict(list)
for f in files:
    filename = os.path.basename(f)
    parts = filename.split('_')
    if len(parts) >= 3:
        user_id = parts[1]
        user_files[user_id].append(f)

print(f"Total users: {len(user_files)}")

# Фильтрация пользователей
valid_users = [user for user in user_files if len(user_files[user]) >= MIN_FILES_PER_USER]
print(f"Users with ≥{MIN_FILES_PER_USER} recordings: {len(valid_users)}")

if len(valid_users) == 0:
    print("❌ No valid users found! Exiting.")
    exit(1)

# Создание dataset
file_list = []
labels = []
label2id = {user: idx for idx, user in enumerate(valid_users)}
id2label = {idx: user for user, idx in label2id.items()}

for user in valid_users:
    for f in user_files[user]:
        file_list.append(f)
        labels.append(label2id[user])

print(f"Total files selected: {len(file_list)}")
print(f"Unique users: {len(valid_users)}")
print(f"Average files per user: {len(file_list) / len(valid_users):.1f}")

# Разделение данных
print("\n===== DATA SPLITTING =====")
train_files, test_files, train_labels, test_labels = train_test_split(
    file_list, labels,
    test_size=TEST_SIZE,
    stratify=labels,
    random_state=42
)

train_files, val_files, train_labels, val_labels = train_test_split(
    train_files, train_labels,
    test_size=VAL_SIZE,
    stratify=train_labels,
    random_state=42
)

print(f"Train set: {len(train_files)} files")
print(f"Validation set: {len(val_files)} files")
print(f"Test set: {len(test_files)} files")

# Создание датасетов
print("\n===== DATASET CREATION =====")
train_ds = VoiceDataset(train_files, train_labels, cache_dir="train_cache")
val_ds = VoiceDataset(val_files, val_labels, cache_dir="val_cache")
test_ds = VoiceDataset(test_files, test_labels, cache_dir="test_cache")

# ======== ОБУЧЕНИЕ ========
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n===== TRAINING ON: {device.upper()} =====")

model = train_model(
    train_ds,
    val_ds,
    num_classes=len(valid_users),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    device=device
)

# ======== ОЦЕНКА ========
print("\n===== EVALUATION =====")
test_loader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4 if device == "cuda" else 0
)

results = evaluate_model(model, test_loader, device=device)

print("\n===== RESULTS =====")
print(f"EER: {results['eer']:.4f}")
print(f"Accuracy: {results['accuracy']:.4f}")
print(f"AUC: {results['auc']:.4f}")
print(f"FAR: {results['far']:.4f}")
print(f"FRR: {results['frr']:.4f}")

# ======== СОХРАНЕНИЕ МОДЕЛИ ========
print("\n===== SAVING MODEL =====")
model_path = "voice_auth_full_model.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'label2id': label2id,
    'id2label': id2label,
    'config': {
        'n_mfcc': train_ds.n_mfcc,
        'max_len': train_ds.max_len
    }
}, model_path)

print(f"Model saved to {model_path}")
print("\n" + "="*80)
print("PROCESSING COMPLETE!")
print("="*80)