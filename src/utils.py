"""
Utility functions: GradCAM, dataset visualisation, and report generation.
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ---------------------------------------------------------------------------
# GradCAM
# ---------------------------------------------------------------------------

class GradCAM:
    """Gradient-weighted Class Activation Mapping for EfficientNetB4.

    Usage:
        gcam = GradCAM(model)
        heatmap = gcam.compute_heatmap(img_array, class_idx=2)
        overlay, _ = gcam.overlay_heatmap(heatmap, original_rgb_image)
    """

    def __init__(self, model, layer_name=None):
        self.model      = model
        self.layer_name = layer_name or self._find_target_layer()
        self.grad_model = self._build_grad_model()

    def _find_target_layer(self):
        """Return the name of the last 4-D conv output layer in the model."""
        for layer in reversed(self.model.layers):
            try:
                if len(layer.output.shape) == 4:
                    return layer.name
            except AttributeError:
                continue
        # Fall back: search inside the EfficientNet sub-model
        for layer in self.model.layers:
            if hasattr(layer, 'layers'):
                for sub in reversed(layer.layers):
                    try:
                        if len(sub.output.shape) == 4:
                            return sub.name
                    except AttributeError:
                        continue
        raise ValueError("Could not find a suitable convolutional layer for GradCAM.")

    def _build_grad_model(self):
        """Build a model that returns (conv_outputs, predictions)."""
        try:
            target = self.model.get_layer(self.layer_name)
            return tf.keras.Model(inputs=self.model.inputs,
                                  outputs=[target.output, self.model.output])
        except ValueError:
            # Layer may live inside an EfficientNet sub-model
            for layer in self.model.layers:
                if hasattr(layer, 'layers'):
                    for sub in layer.layers:
                        if sub.name == self.layer_name:
                            inner_model = tf.keras.Model(
                                inputs=layer.input, outputs=[sub.output, layer.output])
                            return inner_model
            raise ValueError(f"Layer '{self.layer_name}' not found in model.")

    def compute_heatmap(self, image_array, class_idx, eps=1e-8):
        """Return a normalised GradCAM heatmap (H, W) for `class_idx`."""
        with tf.GradientTape() as tape:
            conv_out, preds = self.grad_model(image_array, training=False)
            loss = preds[:, class_idx]

        grads        = tape.gradient(loss, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + eps)
        return heatmap.numpy()

    def overlay_heatmap(self, heatmap, original_image, alpha=0.4,
                        colormap=cv2.COLORMAP_JET):
        """Blend the GradCAM heatmap onto `original_image` (RGB uint8 or [0,1] float)."""
        orig = (original_image * 255).astype(np.uint8) \
               if original_image.max() <= 1.0 else original_image.astype(np.uint8)
        if orig.ndim == 2:
            orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)

        h_resized   = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
        h_uint8     = np.uint8(255 * h_resized)
        h_colored   = cv2.applyColorMap(h_uint8, colormap)
        h_colored   = cv2.cvtColor(h_colored, cv2.COLOR_BGR2RGB)

        superimposed = cv2.addWeighted(orig, 1 - alpha, h_colored, alpha, 0)
        return superimposed, h_colored

    def visualize(self, image_array, original_image, disease_name, class_idx,
                  save_path=None):
        """Generate a 3-panel GradCAM figure: original | heatmap | overlay."""
        heatmap  = self.compute_heatmap(image_array, class_idx)
        overlay, h_colored = self.overlay_heatmap(heatmap, original_image)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        disp = (original_image * 255).astype(np.uint8) \
               if original_image.max() <= 1.0 else original_image.astype(np.uint8)

        axes[0].imshow(disp);       axes[0].set_title('Original',          fontweight='bold'); axes[0].axis('off')
        axes[1].imshow(h_colored);  axes[1].set_title(f'GradCAM\n({disease_name})', fontweight='bold'); axes[1].axis('off')
        axes[2].imshow(overlay);    axes[2].set_title(f'Overlay\n({disease_name})', fontweight='bold'); axes[2].axis('off')

        plt.suptitle(f'Gradient-weighted Class Activation Map — {disease_name}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return fig, heatmap, overlay

    def generate_for_web(self, image_array, original_bgr, class_idx):
        """Return RGB overlay array for direct use in the Flask app."""
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        heatmap = self.compute_heatmap(image_array, class_idx)
        overlay, _ = self.overlay_heatmap(heatmap, original_rgb)
        return overlay


# ---------------------------------------------------------------------------
# Dataset visualisation helpers
# ---------------------------------------------------------------------------

def plot_label_distribution(df, label_columns, save_path=None):
    """Bar + pie chart of ODIR class frequencies."""
    counts = df[label_columns].sum()
    total  = len(df)
    colors = plt.cm.Set2(np.linspace(0, 1, len(label_columns)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    bars = ax1.bar(config.DISEASE_LABELS, counts.values, color=colors, edgecolor='k', alpha=0.85)
    for bar, n in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(int(n)), ha='center', va='bottom', fontweight='bold')
    ax1.set_title('Disease Label Counts', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12)
    plt.setp(ax1.get_xticklabels(), rotation=25, ha='right')

    ax2.pie(counts.values, labels=config.DISEASE_LABELS, autopct='%1.1f%%',
            colors=colors, startangle=90, pctdistance=0.85)
    ax2.set_title('Label Proportions', fontsize=14, fontweight='bold')

    plt.suptitle(f'ODIR-5K Dataset Label Distribution  (n={total})',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def visualize_sample_images(image_df, save_path=None):
    """Display one sample fundus image per disease class."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, (label, code) in zip(axes.ravel(), zip(config.DISEASE_LABELS, config.DISEASE_CODES)):
        subset = image_df[image_df[code] == 1]
        if subset.empty:
            ax.text(0.5, 0.5, 'No samples', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue
        path = subset.sample(1).iloc[0]['image_path']
        img  = cv2.imread(path)
        if img is not None:
            ax.imshow(cv2.cvtColor(cv2.resize(img, (380, 380)), cv2.COLOR_BGR2RGB))
        ax.set_title(f'{label}  [{code}]', fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.suptitle('ODIR-5K — Sample Fundus Images', fontsize=16, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def save_numpy_image(array, output_path):
    """Save a (H, W, 3) float or uint8 array as an image file."""
    arr = (array * 255).astype(np.uint8) if array.max() <= 1.0 else array.astype(np.uint8)
    Image.fromarray(arr).save(output_path)
    return output_path


def create_text_report(metrics, output_path):
    """Write a human-readable evaluation report to disk."""
    lines = [
        "=" * 70,
        "  EYE DISEASE PREDICTION — EVALUATION REPORT",
        "  Model: EfficientNetB4  |  Dataset: ODIR-5K",
        "=" * 70, "",
        "AGGREGATE AUC",
        f"  Macro AUC    : {metrics['macro_auc']:.4f}",
        f"  Micro AUC    : {metrics['micro_auc']:.4f}",
        f"  Weighted AUC : {metrics['weighted_auc']:.4f}", "",
        "PER-CLASS AUC",
    ]
    for label, auc in metrics['per_class_auc'].items():
        lines.append(f"  {label:<35} {auc:.4f}")
    lines += ["", "=" * 70]

    text = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(text)
    print(text)
    return text
