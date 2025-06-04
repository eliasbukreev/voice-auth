import glob
import random
from sklearn.model_selection import train_test_split
from dataset import VoiceDataset
from train import train_model
from evaluate import evaluate_model

# Путь к аудиофайлам
files = glob.glob("data/*.wav")
labels = [f.split("_")[0].split("/")[-1] for f in files]  # предполагаем: user1_000.wav

label2id = {name: i for i, name in enumerate(set(labels))}
ids = [label2id[l] for l in labels]

train_files, test_files, train_labels, test_labels = train_test_split(files, ids, test_size=0.2, stratify=ids)
train_files, val_files, train_labels, val_labels = train_test_split(train_files, train_labels, test_size=0.2)

train_ds = VoiceDataset(train_files, train_labels)
val_ds = VoiceDataset(val_files, val_labels)
test_ds = VoiceDataset(test_files, test_labels)

model = train_model(train_ds, val_ds, num_classes=len(label2id))
eer, acc = evaluate_model(model, torch.utils.data.DataLoader(test_ds, batch_size=1))

print(f"Equal Error Rate (EER): {eer:.4f}")
print(f"Accuracy: {acc:.4f}")