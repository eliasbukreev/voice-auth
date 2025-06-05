import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score
from scipy.spatial.distance import cosine, euclidean
from collections import defaultdict
import matplotlib.pyplot as plt


def evaluate_model(model, test_loader, device="cpu", threshold_step=0.001):
    model.eval()
    embeddings_dict = defaultdict(list)
    all_embeddings = []
    all_labels = []

    print("Extracting embeddings...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            # Фильтруем некорректные лейблы
            valid_mask = labels >= 0
            if not valid_mask.any():
                continue
                
            inputs = inputs[valid_mask]
            labels = labels[valid_mask]
            
            emb = model(inputs).cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i, label in enumerate(labels_np):
                embeddings_dict[label].append(emb[i])
                all_embeddings.append(emb[i])
                all_labels.append(label)

    if not embeddings_dict:
        print("❌ No valid embeddings found!")
        return None

    print(f"Extracted embeddings for {len(embeddings_dict)} users")
    print(f"Total embeddings: {len(all_embeddings)}")

    # Создание более качественных шаблонов (галереи)
    print("Creating user templates...")
    templates = {}
    for label, embs in embeddings_dict.items():
        if len(embs) >= 2:  # Минимум 2 образца для создания шаблона
            # Используем медиану вместо среднего для устойчивости к выбросам
            embs_array = np.array(embs)
            template = np.median(embs_array, axis=0)
            # Дополнительная нормализация шаблона
            template = template / (np.linalg.norm(template) + 1e-8)
            templates[label] = template

    print(f"Created templates for {len(templates)} users")

    # Расчет genuine и impostor scores с несколькими метриками
    print("Calculating similarity scores...")
    genuine_scores_cosine = []
    impostor_scores_cosine = []
    genuine_scores_euclidean = []
    impostor_scores_euclidean = []

    # Genuine scores (один пользователь против своего шаблона)
    for label, embs in embeddings_dict.items():
        if label not in templates:
            continue
            
        template = templates[label]
        for emb in embs:
            # Косинусное сходство
            cos_sim = 1 - cosine(emb, template)
            genuine_scores_cosine.append(cos_sim)
            
            # Евклидово расстояние (инвертированное для сходства)
            eucl_dist = euclidean(emb, template)
            genuine_scores_euclidean.append(-eucl_dist)  # Отрицательное для сходства

    # Impostor scores (один пользователь против чужих шаблонов)
    users = list(templates.keys())
    impostor_pairs = 0
    max_impostor_pairs = 10000  # Ограничиваем количество пар для скорости
    
    for i, user1 in enumerate(users):
        for j, user2 in enumerate(users):
            if i != j and impostor_pairs < max_impostor_pairs:
                if user1 in embeddings_dict and len(embeddings_dict[user1]) > 0:
                    # Берем случайный образец пользователя 1
                    emb = embeddings_dict[user1][0]
                    template = templates[user2]
                    
                    # Косинусное сходство
                    cos_sim = 1 - cosine(emb, template)
                    impostor_scores_cosine.append(cos_sim)
                    
                    # Евклидово расстояние
                    eucl_dist = euclidean(emb, template)
                    impostor_scores_euclidean.append(-eucl_dist)
                    
                    impostor_pairs += 1

    print(f"Genuine comparisons: {len(genuine_scores_cosine)}")
    print(f"Impostor comparisons: {len(impostor_scores_cosine)}")

    # Выбираем лучшую метрику на основе разделимости
    cos_separability = abs(np.mean(genuine_scores_cosine) - np.mean(impostor_scores_cosine))
    eucl_separability = abs(np.mean(genuine_scores_euclidean) - np.mean(impostor_scores_euclidean))
    
    print(f"Cosine separability: {cos_separability:.4f}")
    print(f"Euclidean separability: {eucl_separability:.4f}")
    
    if cos_separability > eucl_separability:
        print("Using cosine similarity for evaluation")
        genuine_scores = genuine_scores_cosine
        impostor_scores = impostor_scores_cosine
        metric_name = "cosine"
    else:
        print("Using euclidean distance for evaluation")
        genuine_scores = genuine_scores_euclidean
        impostor_scores = impostor_scores_euclidean
        metric_name = "euclidean"

    # Рассчет улучшенных метрик
    results = calculate_improved_metrics(genuine_scores, impostor_scores, threshold_step)
    results['metric_used'] = metric_name
    results['separability'] = max(cos_separability, eucl_separability)
    
    return results


def calculate_improved_metrics(genuine_scores, impostor_scores, threshold_step=0.001):
    """Расчет улучшенных метрик с более точным определением порогов"""
    
    # Объединение всех оценок
    y_true = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])
    y_score = np.concatenate([genuine_scores, impostor_scores])

    # Рассчет ROC и AUC
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Более точное нахождение EER
    fnr = 1 - tpr
    eer_differences = np.abs(fpr - fnr)
    eer_idx = np.argmin(eer_differences)
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2  # Среднее между FAR и FRR
    eer_threshold = roc_thresholds[eer_idx]

    # Детальный анализ по различным порогам
    min_score = min(min(genuine_scores), min(impostor_scores))
    max_score = max(max(genuine_scores), max(impostor_scores))
    thresholds = np.arange(min_score, max_score, threshold_step)
    
    far_list = []
    frr_list = []
    accuracy_list = []
    
    genuine_array = np.array(genuine_scores)
    impostor_array = np.array(impostor_scores)
    
    for threshold in thresholds:
        # False Acceptance Rate (FAR) - impostor принят как genuine
        far = np.sum(impostor_array >= threshold) / len(impostor_array) if len(impostor_array) > 0 else 0
        
        # False Rejection Rate (FRR) - genuine отклонен
        frr = np.sum(genuine_array < threshold) / len(genuine_array) if len(genuine_array) > 0 else 0
        
        # Accuracy
        true_accepts = np.sum(genuine_array >= threshold)
        true_rejects = np.sum(impostor_array < threshold)
        accuracy = (true_accepts + true_rejects) / (len(genuine_array) + len(impostor_array))
        
        far_list.append(far)
        frr_list.append(frr)
        accuracy_list.append(accuracy)

    # Находим оптимальный порог где FAR ≈ FRR
    differences = np.abs(np.array(far_list) - np.array(frr_list))
    optimal_idx = np.argmin(differences)
    optimal_threshold = thresholds[optimal_idx]
    optimal_far = far_list[optimal_idx]
    optimal_frr = frr_list[optimal_idx]
    optimal_accuracy = accuracy_list[optimal_idx]

    # Дополнительные метрики
    # Detection Cost Function (DCF)
    c_miss = 1.0  # Cost of missing detection
    c_fa = 1.0    # Cost of false alarm
    p_target = 0.01  # Prior probability of target
    
    dcf_list = []
    for far, frr in zip(far_list, frr_list):
        dcf = c_miss * frr * p_target + c_fa * far * (1 - p_target)
        dcf_list.append(dcf)
    
    min_dcf_idx = np.argmin(dcf_list)
    min_dcf = dcf_list[min_dcf_idx]
    min_dcf_threshold = thresholds[min_dcf_idx]
    
    # Статистики распределений
    genuine_stats = {
        'mean': np.mean(genuine_scores),
        'std': np.std(genuine_scores),
        'min': np.min(genuine_scores),
        'max': np.max(genuine_scores)
    }
    
    impostor_stats = {
        'mean': np.mean(impostor_scores),
        'std': np.std(impostor_scores),
        'min': np.min(impostor_scores),
        'max': np.max(impostor_scores)
    }
    
    # d-prime (разделимость распределений)
    d_prime = abs(genuine_stats['mean'] - impostor_stats['mean']) / \
              np.sqrt(0.5 * (genuine_stats['std']**2 + impostor_stats['std']**2))

    return {
        'eer': eer,
        'eer_threshold': eer_threshold,
        'auc': roc_auc,
        'optimal_threshold': optimal_threshold,
        'optimal_far': optimal_far,
        'optimal_frr': optimal_frr,
        'accuracy': optimal_accuracy,
        'far': optimal_far,
        'frr': optimal_frr,
        'min_dcf': min_dcf,
        'min_dcf_threshold': min_dcf_threshold,
        'd_prime': d_prime,
        'genuine_stats': genuine_stats,
        'impostor_stats': impostor_stats,
        'thresholds': thresholds,
        'far_list': far_list,
        'frr_list': frr_list,
        'accuracy_list': accuracy_list,
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': roc_thresholds
    }


def plot_evaluation_results(results, save_path="evaluation_plots.png"):
    """Создание графиков для анализа результатов"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. ROC кривая
    ax1.plot(results['fpr'], results['tpr'], 'b-', linewidth=2, label=f'ROC (AUC = {results["auc"]:.3f})')
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. FAR/FRR кривые
    ax2.plot(results['thresholds'], results['far_list'], 'r-', linewidth=2, label='FAR')
    ax2.plot(results['thresholds'], results['frr_list'], 'b-', linewidth=2, label='FRR')
    ax2.axvline(results['optimal_threshold'], color='g', linestyle='--', 
                label=f'Optimal (EER={results["eer"]:.3f})')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Error Rate')
    ax2.set_title('FAR/FRR vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # 3. Распределения genuine/impostor scores
    ax3.hist(results['genuine_stats'], bins=50, alpha=0.7, color='green', 
             label='Genuine', density=True)
    ax3.hist(results['impostor_stats'], bins=50, alpha=0.7, color='red', 
             label='Impostor', density=True)
    ax3.axvline(results['optimal_threshold'], color='black', linestyle='--', 
                label='Optimal Threshold')
    ax3.set_xlabel('Similarity Score')
    ax3.set_ylabel('Density')
    ax3.set_title('Score Distributions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Accuracy vs Threshold
    ax4.plot(results['thresholds'], results['accuracy_list'], 'purple', linewidth=2)
    ax4.axvline(results['optimal_threshold'], color='g', linestyle='--', 
                label=f'Max Accuracy = {results["accuracy"]:.3f}')
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Accuracy vs Threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig