import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, accuracy_score

def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def evaluate_model(model, test_loader, threshold=0.8):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for x, label in test_loader:
            out = model(x)
            embeddings.append(out)
            labels.append(label)

    all_scores = []
    all_labels = []

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            score = cosine_similarity(embeddings[i], embeddings[j])
            same = (labels[i] == labels[j]).item()
            all_scores.append(score)
            all_labels.append(same)

    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    eer = fpr[(1 - tpr).argmin()]
    acc = accuracy_score(all_labels, [s > threshold for s in all_scores])
