import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score
from scipy.spatial.distance import cosine
from collections import defaultdict


def evaluate_model(model, test_loader, device="cpu", threshold_step=0.01):
    model.eval()
    embeddings_dict = defaultdict(list)

    print("Extracting embeddings...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            emb = model(inputs).cpu().numpy()

            for i, label in enumerate(labels):
                embeddings_dict[label.item()].append(emb[i])

    print("Creating gallery...")
    gallery = {}
    for label, embs in embeddings_dict.items():
        gallery[label] = np.mean(embs, axis=0)

    print("Calculating scores...")
    genuine_scores = []
    impostor_scores = []

    # Подлинные сравнения
    for label, embs in embeddings_dict.items():
        for emb in embs:
            # Косинусное сходство (1 - distance)
            score = 1 - cosine(emb, gallery[label])
            genuine_scores.append(score)

    # Ложные сравнения
    all_labels = list(gallery.keys())
    for label, embs in embeddings_dict.items():
        other_labels = [l for l in all_labels if l != label]

        for other_label in other_labels:
            # Берем первый образец для сравнения
            emb = embs[0]
            score = 1 - cosine(emb, gallery[other_label])
            impostor_scores.append(score)

    print(f"Genuine comparisons: {len(genuine_scores)}")
    print(f"Impostor comparisons: {len(impostor_scores)}")

    # Рассчет метрик
    return calculate_metrics(genuine_scores, impostor_scores, threshold_step)


def calculate_metrics(genuine_scores, impostor_scores, threshold_step=0.01):
    # Объединение всех оценок
    y_true = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])

    y_score = np.concatenate([genuine_scores, impostor_scores])

    # Рассчет ROC и AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Нахождение EER
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]

    # Рассчет FAR и FRR
    far_list = []
    frr_list = []
    thresholds_list = np.arange(0.0, 1.0, threshold_step)

    for threshold in thresholds_list:
        # False Acceptance Rate (FAR)
        far = np.sum(np.array(impostor_scores) >= threshold) / len(impostor_scores)
        # False Rejection Rate (FRR)
        frr = np.sum(np.array(genuine_scores) < threshold) / len(genuine_scores)
        far_list.append(far)
        frr_list.append(frr)

    # Точность при оптимальном пороге
    y_pred = (y_score >= eer_threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)

    # Находим точку, где FAR и FRR наиболее близки
    diff = np.abs(np.array(far_list) - np.array(frr_list))
    min_idx = np.argmin(diff)
    far_at_eer = far_list[min_idx]
    frr_at_eer = frr_list[min_idx]

    return {
        "eer": eer,
        "auc": roc_auc,
        "accuracy": accuracy,
        "threshold": eer_threshold,
        "far": far_at_eer,
        "frr": frr_at_eer,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds
    }