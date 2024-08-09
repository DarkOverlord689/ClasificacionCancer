from sklearn.metrics import precision_recall_curve, auc, cohen_kappa_score, f1_score
from scipy.stats import hmean
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np

def calculate_metrics(y_true, y_pred, y_pred_proba, class_names):
    metrics = {}
    
    # Accuracy
    #metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Weighted F1-score
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
    
    # Cohen's Kappa
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # Macro-averaged AUPRC
    auprc_scores = []
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_true == i, y_pred_proba[:, i])
        auprc_scores.append(auc(recall, precision))
    metrics['macro_auprc'] = np.mean(auprc_scores)
    
    # Harmonic mean of per-class F1-scores
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    metrics['harmonic_mean_f1'] = hmean(per_class_f1)
    
    return metrics


def get_metrics_per_class(y_true, y_pred, class_names):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    metrics = pd.DataFrame({
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    }, index=class_names)
    return metrics


def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model Accuracy - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'plots/training_history_{model_name}.png')


def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def compare_models(results):
    models = list(results.keys())
    accuracy = [results[m]['accuracy'] for m in models]
    loss = [results[m]['loss'] for m in models]
    
    plt.figure(figsize=(12, 6))
    
    # Comparación de Accuracy
    plt.subplot(1, 2, 1)
    plt.bar(models, accuracy, color='skyblue')
    plt.title('Comparación de Accuracy')
    plt.ylabel('Validation Accuracy')
    plt.xticks(rotation=45)
    
    # Comparación de Loss
    plt.subplot(1, 2, 2)
    plt.bar(models, loss, color='lightcoral')
    plt.title('Comparación de Loss')
    plt.ylabel('Validation Loss')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'plots/comparativa_models.png')

def print_metrics_table(results):
    metrics_df = pd.DataFrame(results).T
    metrics_df.columns = ['Accuracy', 'Loss']
    metrics_df = metrics_df.sort_values('Accuracy', ascending=False)
    print("\nComparación de métricas totales:")
    print(metrics_df.to_string())

def print_metrics_per_class_table(all_class_metrics):
    print("\nComparación de métricas por clase:")
    for model, metrics in all_class_metrics.items():
        print(f"\n{model}:")
        print(metrics.to_string())

def plot_metrics(histories, model_names, final_model_metrics):
    # Graficar Accuracy
    plt.figure(figsize=(14, 6))
    
    # Subplot para accuracy
    plt.subplot(1, 2, 1)
    for i, history in enumerate(histories):
        plt.plot(history.history['accuracy'], label=f'{model_names[i]} Train Accuracy')
        plt.plot(history.history['val_accuracy'], label=f'{model_names[i]} Val Accuracy', linestyle='--')
    
    # Añadir la métrica del modelo final
    plt.axhline(y=final_model_metrics['accuracy'], color='r', linestyle='-', label='Ensemble Final Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Subplot para loss
    plt.subplot(1, 2, 2)
    for i, history in enumerate(histories):
        plt.plot(history.history['loss'], label=f'{model_names[i]} Train Loss')
        plt.plot(history.history['val_loss'], label=f'{model_names[i]} Val Loss', linestyle='--')
    
    # Añadir la métrica del modelo final
    plt.axhline(y=final_model_metrics['loss'], color='r', linestyle='-', label='Ensemble Final Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'plots/grafic_final{model_names}.png')
    