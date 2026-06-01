"""
Two-phase training pipeline for EfficientNetB4 on ODIR-5K.

Phase 1 (PHASE1_EPOCHS): Backbone frozen — trains only the custom head.
Phase 2 (PHASE2_EPOCHS): Top FINE_TUNE_LAYERS of backbone unfrozen.
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.model import build_model, compile_model, unfreeze_top_layers, get_callbacks
from src.preprocess import (load_odir_annotations, build_image_df, split_dataset,
                             get_generators)


def setup_dirs():
    for d in [config.MODELS_DIR, config.RESULTS_DIR, 'logs']:
        os.makedirs(d, exist_ok=True)


def train_phase1(model, backbone, train_gen, val_gen):
    print("\n" + "=" * 60)
    print("PHASE 1 — Training classification head (backbone frozen)")
    print("=" * 60)

    model = compile_model(model, learning_rate=config.PHASE1_LR)
    model.summary()

    ckpt = config.CHECKPOINT_PATH.replace('.h5', '_phase1.h5')
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.PHASE1_EPOCHS,
        callbacks=get_callbacks(ckpt),
        verbose=1
    )
    best_auc = max(history.history.get('val_auc', [0]))
    print(f"\nPhase 1 done.  Best val_auc = {best_auc:.4f}")
    return model, history


def train_phase2(model, backbone, train_gen, val_gen):
    print("\n" + "=" * 60)
    print("PHASE 2 — Fine-tuning top backbone layers")
    print("=" * 60)

    backbone = unfreeze_top_layers(backbone, config.FINE_TUNE_LAYERS)
    model = compile_model(model, learning_rate=config.PHASE2_LR)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.PHASE2_EPOCHS,
        callbacks=get_callbacks(config.CHECKPOINT_PATH),
        verbose=1
    )
    best_auc = max(history.history.get('val_auc', [0]))
    print(f"\nPhase 2 done.  Best val_auc = {best_auc:.4f}")
    return model, history


def merge_histories(h1, h2):
    merged = {}
    for key in h1.history:
        merged[f'phase1_{key}'] = h1.history[key]
    for key in h2.history:
        merged[f'phase2_{key}'] = h2.history[key]

    path = os.path.join(config.RESULTS_DIR, 'training_history.json')
    with open(path, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f"Training history saved → {path}")
    return merged


def run_training(data_dir=None, annotations_path=None):
    setup_dirs()
    tf.random.set_seed(config.SEED)
    np.random.seed(config.SEED)

    ann_path  = annotations_path or config.ODIR_ANNOTATIONS
    img_dir   = data_dir or config.ODIR_TRAIN_IMAGES

    print("Loading ODIR-5K annotations …")
    df = load_odir_annotations(ann_path)

    print("Building image DataFrame …")
    image_df = build_image_df(df, img_dir)
    if len(image_df) == 0:
        raise RuntimeError(f"No images found in: {img_dir}\n"
                           "Please check ODIR_TRAIN_IMAGES in config.py")

    train_df, val_df, test_df = split_dataset(image_df)
    test_df.to_csv(os.path.join(config.RESULTS_DIR, 'test_set.csv'), index=False)

    train_gen, val_gen, _ = get_generators(train_df, val_df, test_df)

    print("Building EfficientNetB4 model …")
    model, backbone = build_model()

    model, h1 = train_phase1(model, backbone, train_gen, val_gen)
    model, h2 = train_phase2(model, backbone, train_gen, val_gen)

    model.save(config.FINAL_MODEL_PATH)
    print(f"\nFinal model saved → {config.FINAL_MODEL_PATH}")

    merge_histories(h1, h2)

    print("\nValidation evaluation:")
    model.evaluate(val_gen, verbose=1)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Best checkpoint : {config.CHECKPOINT_PATH}")
    print(f"  Final model     : {config.FINAL_MODEL_PATH}")
    print("=" * 60)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EfficientNetB4 on ODIR-5K')
    parser.add_argument('--data-dir',    type=str, default=None,
                        help='Path to ODIR training images directory')
    parser.add_argument('--annotations', type=str, default=None,
                        help='Path to ODIR Excel annotation file')
    args = parser.parse_args()
    run_training(data_dir=args.data_dir, annotations_path=args.annotations)
