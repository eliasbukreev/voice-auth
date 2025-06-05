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

# ======== УЛУЧШЕННАЯ КОНФИГУРАЦИЯ ========
TSV_PATH = r"C:\Users\Лиза\Desktop\Диплом\voice-auth\en\train.tsv"
AUDIO_DIR = r"C:\Users\Лиза\Desktop\Диплом\voice-auth\en\clips"
PROCESSED_DIR = "full_data"
MIN_FILES_PER_USER = 8  # Увеличили минимальное количество файлов
TEST_SIZE = 0.2
VAL_SIZE = 0.25
BATCH_SIZE = 64      # Уменьшили batch size для лучшего обучения
EPOCHS = 100         # Увеличили количество эпох
N_MFCC = 40          # Больше MFCC коэффициентов
MAX_LEN = 256        # Увеличили длину последовательности

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

# Фильтрация пользователей с достаточным количеством файлов
valid_users = [user for user in user_files if len(user_files[user]) >= MIN_FILES_PER_USER]
print(f"Users with ≥{MIN_FILES_PER_USER} recordings: {len(valid_users)}")

if len(valid_users) == 0:
    print("❌ No valid users found! Exiting.")
    exit(1)

# Ограничиваем количество пользователей для управляемого эксперимента
MAX_USERS = 100  # Ограничиваем для начального тестирования
if len(valid_users) > MAX_USERS:
    valid_users = valid_users[:MAX_USERS]
    print(f"Limited to {MAX_USERS} users for this experiment")

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

# Разделение данных с стратификацией
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

# Проверка баланса классов
from collections import Counter
train_class_counts = Counter(train_labels)
print(f"Train class balance - Min: {min(train_class_counts.values())}, "
      f"Max: {max(train_class_counts.values())}, "
      f"Mean: {np.mean(list(train_class_counts.values())):.1f}")

# Создание улучшенных датасетов
print("\n===== DATASET CREATION =====")
train_ds = VoiceDataset(
    train_files, train_labels, 
    n_mfcc=N_MFCC, max_len=MAX_LEN,
    cache_dir="train_cache", 
    augment=True  # Включаем аугментацию для тренировочной выборки
)

val_ds = VoiceDataset(
    val_files, val_labels,
    n_mfcc=N_MFCC, max_len=MAX_LEN,
    cache_dir="val_cache",
    augment=False  # Без аугментации для валидации
)

test_ds = VoiceDataset(
    test_files, test_labels,
    n_mfcc=N_MFCC, max_len=MAX_LEN,
    cache_dir="test_cache",
    augment=False  # Без аугментации для тестирования
)

print(f"Training dataset size: {len(train_ds)}")
print(f"Validation dataset size: {len(val_ds)}")
print(f"Test dataset size: {len(test_ds)}")

# Проверка размерности данных
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

model = train_model(
    train_ds,
    val_ds,
    num_classes=len(valid_users),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    device=device,
    save_path="improved_models"
)

# ======== ДЕТАЛЬНАЯ ОЦЕНКА ========
print("\n===== DETAILED EVALUATION =====")
test_loader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4 if device == "cuda" else 2,
    pin_memory=True
)

results = evaluate_model(model, test_loader, device=device, threshold_step=0.001)

if results is not None:
    print("\n===== RESULTS SUMMARY =====")
    print(f"🎯 EER (Equal Error Rate): {results['eer']:.4f} ({results['eer']*100:.2f}%)")
    print(f"📊 AUC (Area Under Curve): {results['auc']:.4f}")
    print(f"🎲 Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"❌ FAR (False Accept Rate): {results['far']:.4f} ({results['far']*100:.2f}%)")
    print(f"🚫 FRR (False Reject Rate): {results['frr']:.4f} ({results['frr']*100:.2f}%)")
    print(f"🔍 Min DCF: {results['min_dcf']:.4f}")
    print(f"📏 d-prime (separability): {results['d_prime']:.4f}")
    print(f"📐 Metric used: {results['metric_used']}")
    print(f"🔗 Separability: {results['separability']:.4f}")
    
    print(f"\n📈 Genuine scores - Mean: {results['genuine_stats']['mean']:.4f}, "
          f"Std: {results['genuine_stats']['std']:.4f}")
    print(f"📉 Impostor scores - Mean: {results['impostor_stats']['mean']:.4f}, "
          f"Std: {results['impostor_stats']['std']:.4f}")
    
    # Создание графиков
    try:
        plot_evaluation_results(results, "improved_evaluation_plots.png")
        print("📊 Evaluation plots saved to improved_evaluation_plots.png")
    except Exception as e:
        print(f"⚠️ Could not create plots: {e}")

    # ======== СОХРАНЕНИЕ УЛУЧШЕННОЙ МОДЕЛИ ========
    print("\n===== SAVING IMPROVED MODEL =====")
    model_path = "voice_auth_improved_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'label2id': label2id,
        'id2label': id2label,
        'config': {
            'n_mfcc': N_MFCC,
            'max_len': MAX_LEN,
            'output_dim': 512,
            'num_users': len(valid_users)
        },
        'results': results,
        'training_params': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'min_files_per_user': MIN_FILES_PER_USER
        }
    }, model_path)

    print(f"✅ Improved model saved to {model_path}")
    
    # Интерпретация результатов
    print("\n===== PERFORMANCE INTERPRETATION =====")
    if results['eer'] < 0.05:
        print("🟢 EXCELLENT: EER < 5% - Production ready system")
    elif results['eer'] < 0.10:
        print("🟡 GOOD: EER < 10% - Acceptable for most applications")
    elif results['eer'] < 0.20:
        print("🟠 FAIR: EER < 20% - Needs improvement for security applications")
    else:
        print("🔴 POOR: EER > 20% - Significant improvements needed")
    
    if results['auc'] > 0.95:
        print("🟢 EXCELLENT: AUC > 95% - Very good discriminability")
    elif results['auc'] > 0.90:
        print("🟡 GOOD: AUC > 90% - Good discriminability")
    elif results['auc'] > 0.80:
        print("🟠 FAIR: AUC > 80% - Moderate discriminability")
    else:
        print("🔴 POOR: AUC < 80% - Poor discriminability")

else:
    print("❌ Evaluation failed!")

print("\n" + "="*80)
print("IMPROVED PROCESSING COMPLETE!")
print("="*80)