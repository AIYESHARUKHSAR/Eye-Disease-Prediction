"""
Inference module: single-image, dual-eye, and batch prediction.
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocess import preprocess_single_image, load_and_preprocess_image

# Rich descriptions for each disease — used by the Flask UI
DISEASE_INFO = {
    'Normal':                 {'code': 'N', 'color': '#27ae60', 'description': 'No pathological changes detected.'},
    'Diabetic Retinopathy':   {'code': 'D', 'color': '#e74c3c', 'description': 'Retinal vascular damage caused by chronic diabetes.'},
    'Glaucoma':               {'code': 'G', 'color': '#e67e22', 'description': 'Progressive optic nerve damage, often linked to elevated IOP.'},
    'Cataract':               {'code': 'C', 'color': '#f39c12', 'description': 'Clouding of the crystalline lens, impairing vision.'},
    'AMD':                    {'code': 'A', 'color': '#8e44ad', 'description': 'Age-related macular degeneration — central vision loss in elderly.'},
    'Hypertensive Retinopathy': {'code': 'H', 'color': '#c0392b', 'description': 'Retinal changes resulting from persistently high blood pressure.'},
    'Myopia':                 {'code': 'M', 'color': '#2980b9', 'description': 'Pathological myopia with structural elongation of the eye.'},
    'Others':                 {'code': 'O', 'color': '#7f8c8d', 'description': 'Other retinal or ocular conditions not in the above categories.'},
}


def load_model(model_path=None):
    """Load trained model; falls back from checkpoint to final save."""
    paths = [model_path, config.CHECKPOINT_PATH, config.FINAL_MODEL_PATH]
    for p in paths:
        if p and os.path.exists(p):
            print(f"Loading model from {p} …")
            return tf.keras.models.load_model(p)
    raise FileNotFoundError(
        "No trained model found. Run src/train.py first, or supply --model <path>."
    )


def predict_image(image_path_or_array, model,
                  threshold=config.THRESHOLD, apply_ben_graham=True):
    """
    Predict diseases for a single fundus image.

    Returns a dict with probabilities, binary predictions, and metadata.
    """
    img = preprocess_single_image(image_path_or_array, apply_ben_graham)
    probs = model.predict(img, verbose=0)[0]
    preds = (probs >= threshold).astype(int)

    top_idx = int(np.argmax(probs))

    return {
        'probabilities':     {l: float(probs[i]) for i, l in enumerate(config.DISEASE_LABELS)},
        'predictions':       {l: int(preds[i])   for i, l in enumerate(config.DISEASE_LABELS)},
        'detected_diseases': [l for i, l in enumerate(config.DISEASE_LABELS) if preds[i]],
        'confidence_scores': {l: f"{probs[i]*100:.1f}%" for i, l in enumerate(config.DISEASE_LABELS)},
        'top_prediction':    {
            'disease':    config.DISEASE_LABELS[top_idx],
            'code':       config.DISEASE_CODES[top_idx],
            'confidence': float(probs[top_idx]),
        },
        'threshold':    threshold,
        'disease_info': DISEASE_INFO,
    }


def predict_both_eyes(left_path, right_path, model, threshold=config.THRESHOLD):
    """Predict from both left and right eye images; combine by taking the max probability."""
    left  = predict_image(left_path,  model, threshold)
    right = predict_image(right_path, model, threshold)

    combined_probs = {
        l: max(left['probabilities'][l], right['probabilities'][l])
        for l in config.DISEASE_LABELS
    }
    combined_preds = {l: int(p >= threshold) for l, p in combined_probs.items()}
    detected = [l for l, p in combined_preds.items() if p]

    return {
        'left_eye':  left,
        'right_eye': right,
        'combined': {
            'probabilities':     combined_probs,
            'predictions':       combined_preds,
            'detected_diseases': detected,
            'confidence_scores': {l: f"{p*100:.1f}%" for l, p in combined_probs.items()},
        },
        'threshold':    threshold,
        'disease_info': DISEASE_INFO,
    }


def predict_batch(image_paths, model, threshold=config.THRESHOLD, batch_size=32):
    """Run inference on a list of image paths and return per-image result dicts."""
    results = []
    for i in range(0, len(image_paths), batch_size):
        chunk = image_paths[i:i + batch_size]
        imgs = []
        for p in chunk:
            try:
                imgs.append(load_and_preprocess_image(p, config.IMAGE_SIZE))
            except Exception:
                imgs.append(np.zeros((*config.IMAGE_SIZE, 3), dtype=np.float32))

        probs_batch = model.predict(np.array(imgs), verbose=0)

        for path, probs in zip(chunk, probs_batch):
            preds = (probs >= threshold).astype(int)
            results.append({
                'image_path':      path,
                'probabilities':   {l: float(probs[j]) for j, l in enumerate(config.DISEASE_LABELS)},
                'predictions':     {l: int(preds[j])   for j, l in enumerate(config.DISEASE_LABELS)},
                'detected_diseases': [l for j, l in enumerate(config.DISEASE_LABELS) if preds[j]],
            })
    return results


def format_results_cli(results):
    """Pretty-print prediction results to the terminal."""
    probs = results['probabilities']
    preds = results['predictions']

    print("\n" + "=" * 60)
    print("  EYE DISEASE PREDICTION — ODIR / EfficientNetB4")
    print("=" * 60)
    print(f"\n  {'Disease':<30} {'Confidence':>11}  {'Status':>10}")
    print("  " + "-" * 55)
    for label in config.DISEASE_LABELS:
        bar    = "█" * int(probs[label] * 20) + "░" * (20 - int(probs[label] * 20))
        status = "DETECTED" if preds[label] else "—"
        print(f"  {label:<30} {probs[label]*100:>9.1f}%  {status:>10}")

    detected = results.get('detected_diseases', [])
    print("\n  Detected: " + (", ".join(detected) if detected else "None (Normal)"))
    top = results['top_prediction']
    print(f"  Top prediction: {top['disease']}  ({top['confidence']*100:.1f}%)")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict eye diseases from a fundus image')
    parser.add_argument('image',       type=str,   help='Path to fundus image')
    parser.add_argument('--model',     type=str,   default=None)
    parser.add_argument('--threshold', type=float, default=config.THRESHOLD)
    parser.add_argument('--no-ben-graham', action='store_true')
    parser.add_argument('--json',      action='store_true', help='Output as JSON')
    args = parser.parse_args()

    _model   = load_model(args.model)
    _results = predict_image(args.image, _model, args.threshold,
                             apply_ben_graham=not args.no_ben_graham)

    if args.json:
        safe = {k: v for k, v in _results.items() if k != 'disease_info'}
        print(json.dumps(safe, indent=2))
    else:
        format_results_cli(_results)
