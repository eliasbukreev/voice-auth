import os
import glob
import random
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from dataset import VoiceDataset
from train import train_model
from evaluate import evaluate_model, plot_evaluation_results
from utils import convert_common_voice_to_wav
import torch
import shutil
import csv

# ======== УЛУЧШЕННАЯ КОНФИГУРАЦИЯ ========
TSV_PATH = r"C:\Users\Лиза\Desktop\Диплом\voice-auth\en\train.tsv"
AUDIO_DIR = r"C:\Users\Лиза\Desktop\Диплом\voice-auth\en\clips"
PROCESSED_DIR = "full_data"
MIN_FILES_PER_USER = 8
TEST_SIZE = 0.2
VAL_SIZE = 0.25
BATCH_SIZE = 64
EPOCHS = 100
N_MFCC = 40
MAX_LEN = 256

# Установка seed для воспроизводимости
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# ======== ПОДГОТОВКА ========
print("\n" + "="*80)
print("IMPROVED VOICE AUTHENTICATION SYSTEM - FULL DATASET PROCESSING")
print("="*80 + "\n")

if os.path.exists(PROCESSED_DIR):
    print(f"⚠️ Warning: Output directory {PROCESSED_DIR} already exists")

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
files = glob.glob(os.path.join(PROCESSED_DIR, "*.wav"))
print(f"Found {len(files)} WAV files in {PROCESSED_DIR}")

if len(files) == 0:
    print("❌ No files found! Exiting.")
    exit(1)

# Группировка по пользователям (Common Voice формат)
user_files = defaultdict(list)

with open(TSV_PATH, encoding='utf-8') as tsvfile:
    reader = csv.DictReader(tsvfile, delimiter='\t')
    for row in reader:
        filename = os.path.splitext(os.path.basename(row['path']))[0] + ".wav"
        full_path = os.path.join(PROCESSED_DIR, filename)
        if os.path.exists(full_path):
            user_id = row['client_id']
            user_files[user_id].append(full_path)

print(f"Total users: {len(user_files)}")

# Фильтрация пользователей
valid_users = [user for user in user_files if len(user_files[user]) >= MIN_FILES_PER_USER]
print(f"Users with ≥{MIN_FILES_PER_USER} recordings: {len(valid_users)}")

if len(valid_users) == 0:
    print("❌ No valid users found! Exiting.")
    exit(1)

MAX_USERS = 100
if len(valid_users) > MAX_USERS:
    valid_users = valid_users[:MAX_USERS]
    print(f"Limited to {MAX_USERS} users for this experiment")

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

# ======== РАЗДЕЛЕНИЕ ========
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

from collections import Counter
train_class_counts = Counter(train_labels)
print(f"Train class balance - Min: {min(train_class_counts.values())}, Max: {max(train_class_counts.values())}, Mean: {np.mean(list(train_class_counts.values())):.1f}")

# ======== СОЗДАНИЕ DATASET ========
print("\n===== DATASET CREATION =====")
train_ds = VoiceDataset(train_files, train_labels, n_mfcc=N_MFCC, max_len=MAX_LEN, cache_dir="train_cache", augment=True)
val_ds = VoiceDataset(val_files, val_labels, n_mfcc=N_MFCC, max_len=MAX_LEN, cache_dir="val_cache", augment=False)
test_ds = VoiceDataset(test_files, test_labels, n_mfcc=N_MFCC, max_len=MAX_LEN, cache_dir="test_cache", augment=False)

print(f"Training dataset size: {len(train_ds)}")
print(f"Validation dataset size: {len(val_ds)}")
print(f"Test dataset size: {len(test_ds)}")

sample_input, sample_label = train_ds[0]
print(f"Input shape: {sample_input.shape}")
print(f"Feature dimension: {sample_input.shape[0]}")
print(f"Sequence length: {sample_input.shape[1]}")

# ======== ОБУЧЕНИЕ ========
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n===== TRAINING ON: {device.upper()} =====")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

model = train_model(train_ds, val_ds, num_classes=len(valid_users), epochs=EPOCHS, batch_size=BATCH_SIZE, device=device, save_path="improved_models")

# ======== ОЦЕНКА ========
print("\n===== DETAILED EVALUATION =====")
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4 if device == "cuda" else 2, pin_memory=True)

results = evaluate_model(model, test_loader, device=device, threshold_step=0.001)

if results is not None:
    print("\n===== RESULTS SUMMARY =====")
    print(f"🎯 EER (Equal error rate): {results['eer']:.4f} ({results['eer']*100:.2f}%)")
    print(f"📊 AUC: {results['auc']:.4f}")
    print(f"🎲 Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"❌ FAR: {results['far']:.4f} ({results['far']*100:.2f}%)")
    print(f"🚫 FRR: {results['frr']:.4f} ({results['frr']*100:.2f}%)")
    print(f"📏 d-prime: {results['d_prime']:.4f}")
else:
    print("❌ Evaluation failed!")

print("\n" + "="*80)
print("IMPROVED PROCESSING COMPLETE!")
print("="*80)