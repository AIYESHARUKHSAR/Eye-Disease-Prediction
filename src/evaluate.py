"""
Evaluation pipeline: ROC-AUC, confusion matrices, classification report,
training history plots, and per-class AUC bar chart.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
                              multilabel_confusion_matrix, average_precision_score)
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

plt.style.use('seaborn-v0_8-whitegrid')
_COLORS = plt.cm.Set2(np.linspace(0, 1, config.NUM_CLASSES))


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------

def get_predictions(model, generator):
    """Collect ground-truth labels and model outputs from a data generator."""
    y_true, y_pred = [], []
    for i in tqdm(range(len(generator)), desc='Predicting'):
        X, y = generator[i]
        y_true.extend(y)
        y_pred.extend(model.predict(X, verbose=0))
    return np.array(y_true), np.array(y_pred)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, threshold=config.THRESHOLD):
    y_bin = (y_pred >= threshold).astype(int)

    per_class_auc = {}
    for i, label in enumerate(config.DISEASE_LABELS):
        if y_true[:, i].sum() > 0:
            per_class_auc[label] = roc_auc_score(y_true[:, i], y_pred[:, i])
        else:
            per_class_auc[label] = float('nan')

    return {
        'per_class_auc':       per_class_auc,
        'micro_auc':           roc_auc_score(y_true, y_pred, average='micro'),
        'macro_auc':           roc_auc_score(y_true, y_pred, average='macro'),
        'weighted_auc':        roc_auc_score(y_true, y_pred, average='weighted'),
        'avg_precision_macro': average_precision_score(y_true, y_pred, average='macro'),
        'classification_report': classification_report(
            y_true, y_bin, target_names=config.DISEASE_LABELS,
            output_dict=True, zero_division=0
        ),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc_curves(y_true, y_pred, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (label, color) in enumerate(zip(config.DISEASE_LABELS, _COLORS)):
        if y_true[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC = {auc:.3f})')

    fpr_m, tpr_m, _ = roc_curve(y_true.ravel(), y_pred.ravel())
    micro_auc = roc_auc_score(y_true, y_pred, average='micro')
    ax.plot(fpr_m, tpr_m, 'k--', lw=2, label=f'Micro-avg (AUC = {micro_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1)

    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curves — ODIR-5K Eye Disease Classification\n(EfficientNetB4)',
                 fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_confusion_matrices(y_true, y_pred, threshold=config.THRESHOLD, save_path=None):
    y_bin = (y_pred >= threshold).astype(int)
    cms = multilabel_confusion_matrix(y_true, y_bin)

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    for label, cm, ax in zip(config.DISEASE_LABELS, cms, axes.ravel()):
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                    xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        ax.set_title(f'{label}\nSens={sens:.2f}  Spec={spec:.2f}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    fig.suptitle('Per-Class Confusion Matrices — EfficientNetB4 on ODIR-5K',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_training_history(history, save_path=None):
    """Plot combined Phase 1 + Phase 2 training history."""
    metrics_to_plot = [('loss', 'Loss'), ('auc', 'AUC'),
                       ('precision', 'Precision'), ('recall', 'Recall')]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for ax, (metric, title) in zip(axes.ravel(), metrics_to_plot):
        p1_tr  = history.get(f'phase1_{metric}',     [])
        p1_val = history.get(f'phase1_val_{metric}', [])
        p2_tr  = history.get(f'phase2_{metric}',     [])
        p2_val = history.get(f'phase2_val_{metric}', [])

        all_tr  = p1_tr + p2_tr
        all_val = p1_val + p2_val
        epochs  = range(1, len(all_tr) + 1)

        ax.plot(epochs, all_tr,  'b-o', ms=3, lw=2, label='Train')
        ax.plot(epochs, all_val, 'r-s', ms=3, lw=2, label='Validation')
        if len(p1_tr) > 0 and len(p2_tr) > 0:
            ax.axvline(len(p1_tr) + 1, color='green', ls='--', alpha=0.7, label='Phase 2 start')

        ax.set_title(f'{title} over Epochs', fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Training History — EfficientNetB4 on ODIR-5K',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_per_class_auc_bar(metrics, save_path=None):
    labels = list(metrics['per_class_auc'].keys())
    aucs   = [v if not np.isnan(v) else 0 for v in metrics['per_class_auc'].values()]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, aucs, color=_COLORS, edgecolor='black', lw=0.5, alpha=0.85)

    ax.axhline(metrics['macro_auc'], color='red',  ls='--', lw=2, label=f"Macro AUC={metrics['macro_auc']:.3f}")
    ax.axhline(metrics['micro_auc'], color='blue', ls='--', lw=2, label=f"Micro AUC={metrics['micro_auc']:.3f}")

    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylim([0, 1.15])
    ax.set_ylabel('AUC', fontsize=13)
    ax.set_title('Per-Class AUC — EfficientNetB4 on ODIR-5K', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    plt.xticks(rotation=25, ha='right', fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Summary print
# ---------------------------------------------------------------------------

def print_summary(metrics):
    report = metrics['classification_report']
    print("\n" + "=" * 65)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 65)
    print(f"{'Macro AUC':<35} {metrics['macro_auc']:>8.4f}")
    print(f"{'Micro AUC':<35} {metrics['micro_auc']:>8.4f}")
    print(f"{'Weighted AUC':<35} {metrics['weighted_auc']:>8.4f}")
    print(f"{'Avg Precision (macro)':<35} {metrics['avg_precision_macro']:>8.4f}")
    print(f"{'Macro F1':<35} {report.get('macro avg',{}).get('f1-score', 0):>8.4f}")
    print(f"{'Weighted F1':<35} {report.get('weighted avg',{}).get('f1-score', 0):>8.4f}")
    print("\n  Per-class AUC:")
    for label, auc in metrics['per_class_auc'].items():
        mark = f"{auc:.4f}" if not np.isnan(auc) else "N/A (no positives)"
        print(f"    {label:<35} {mark}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(model, test_generator, results_dir=None, threshold=config.THRESHOLD):
    results_dir = results_dir or config.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    y_true, y_pred = get_predictions(model, test_generator)
    metrics = compute_metrics(y_true, y_pred, threshold)

    print_summary(metrics)

    # Persist serialisable metrics
    saveable = {k: v for k, v in metrics.items() if k != 'classification_report'}
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(saveable, f, indent=2)

    plot_roc_curves(       y_true, y_pred,          save_path=os.path.join(results_dir, 'roc_curves.png'))
    plot_confusion_matrices(y_true, y_pred, threshold, save_path=os.path.join(results_dir, 'confusion_matrices.png'))
    plot_per_class_auc_bar(metrics,                  save_path=os.path.join(results_dir, 'auc_bar_chart.png'))

    # Training history (if available)
    history_path = os.path.join(results_dir, 'training_history.json')
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        plot_training_history(history, save_path=os.path.join(results_dir, 'training_history.png'))

    print(f"\nAll evaluation artefacts saved to: {results_dir}/")
    return metrics, y_true, y_pred


if __name__ == '__main__':
    import tensorflow as tf
    from src.preprocess import ODIRDataGenerator

    parser = argparse.ArgumentParser(description='Evaluate the eye disease model')
    parser.add_argument('--model',     type=str, default=None)
    parser.add_argument('--test-csv',  type=str, default=None)
    parser.add_argument('--threshold', type=float, default=config.THRESHOLD)
    args = parser.parse_args()

    _model = tf.keras.models.load_model(args.model or config.CHECKPOINT_PATH)
    _csv   = args.test_csv or os.path.join(config.RESULTS_DIR, 'test_set.csv')
    _df    = pd.read_csv(_csv)
    _gen   = ODIRDataGenerator(_df, augment=False, shuffle=False)

    run_evaluation(_model, _gen, threshold=args.threshold)
