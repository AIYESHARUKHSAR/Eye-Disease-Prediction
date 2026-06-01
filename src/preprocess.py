import os
import sys
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

LABEL_COLUMNS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

# Keywords used to detect each disease from ODIR diagnostic text
_KEYWORD_MAP = {
    'N': ['normal fundus'],
    'D': ['diabetic retinopathy', 'proliferative retinopathy', 'macular edema', 'hard exudate'],
    'G': ['glaucoma', 'glaucomatous', 'optic atrophy'],
    'C': ['cataract', 'lens nuclear', 'cortical cataract', 'posterior subcapsular'],
    'A': ['age-related macular degeneration', 'drusen', 'choroidal neovascularization', 'macular degeneration'],
    'H': ['hypertensive retinopathy', 'hypertension', 'arteriovenous crossing'],
    'M': ['myopia', 'myopic', 'pathological myopia', 'high myopia', 'tessellated fundus'],
    'O': ['others', 'refractive media opacity', 'vitreous degeneration', 'epiretinal membrane',
          'laser photocoagulation', 'branch retinal artery occlusion', 'branch retinal vein occlusion'],
}


def load_odir_annotations(excel_path=config.ODIR_ANNOTATIONS):
    """Load and parse ODIR-5K Excel annotation file into a labelled DataFrame."""
    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()

    if 'N' not in df.columns:
        df = _extract_labels_from_keywords(df)

    for col in LABEL_COLUMNS:
        df[col] = df[col].fillna(0).astype(int)

    print(f"Loaded {len(df)} patient records")
    print("Label distribution:")
    print(df[LABEL_COLUMNS].sum().to_string())
    return df


def _extract_labels_from_keywords(df):
    """Derive binary labels from free-text diagnostic keyword columns."""
    # Try to locate keyword column(s)
    candidate_cols = [c for c in df.columns if 'keyword' in c.lower() or 'diagnostic' in c.lower()]
    if not candidate_cols:
        raise ValueError("Cannot find a diagnostic keywords column in the Excel file. "
                         "Expected a column whose name contains 'keyword' or 'diagnostic'.")

    # Combine left + right keyword columns into one string per row
    combined = df[candidate_cols].fillna('').astype(str).agg(' '.join, axis=1).str.lower()

    for label, keywords in _KEYWORD_MAP.items():
        df[label] = combined.apply(lambda text: int(any(kw in text for kw in keywords)))

    return df


def build_image_df(df, images_dir):
    """Return a flat DataFrame with one row per fundus image, including labels."""
    records = []
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG']

    for _, row in df.iterrows():
        patient_id = row.get('ID', row.name)
        labels = {col: int(row[col]) for col in LABEL_COLUMNS}

        for eye in ('left', 'right'):
            found = None
            for ext in extensions:
                candidate = os.path.join(images_dir, f"{patient_id}_{eye}{ext}")
                if os.path.exists(candidate):
                    found = candidate
                    break

            if found:
                records.append({'image_path': found, 'patient_id': patient_id,
                                'eye': eye, **labels})

    result = pd.DataFrame(records)
    print(f"Found {len(result)} fundus images in {images_dir}")
    return result


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def ben_graham_preprocess(image, sigma_x=10):
    """Ben Graham contrast enhancement with circular crop."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2 - 10, 255, -1)

    blurred = cv2.GaussianBlur(image, (0, 0), sigma_x)
    enhanced = cv2.addWeighted(image, 4, blurred, -4, 128)
    enhanced[mask == 0] = 0
    return enhanced


def load_and_preprocess_image(image_path, target_size=config.IMAGE_SIZE,
                               apply_ben_graham=True):
    """Load a fundus image, optionally apply Ben Graham preprocessing, and normalise."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if apply_ben_graham:
        img = ben_graham_preprocess(img)

    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
    img = img.astype(np.float32) / 255.0
    return img


def preprocess_single_image(image_path_or_array, apply_ben_graham=True):
    """Preprocess one image and return a batch-dimension array ready for model.predict."""
    if isinstance(image_path_or_array, str):
        img = load_and_preprocess_image(image_path_or_array, config.IMAGE_SIZE, apply_ben_graham)
    else:
        img = image_path_or_array.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0
        img = cv2.resize(img, config.IMAGE_SIZE)

    return np.expand_dims(img, axis=0)


# ---------------------------------------------------------------------------
# Dataset split
# ---------------------------------------------------------------------------

def split_dataset(df, train_ratio=config.TRAIN_RATIO,
                  val_ratio=config.VAL_RATIO,
                  random_state=config.SEED):
    """Stratified train / val / test split (stratify on Normal label)."""
    test_ratio = 1.0 - train_ratio - val_ratio
    train_df, temp_df = train_test_split(
        df, test_size=(1.0 - train_ratio), random_state=random_state,
        stratify=df['N']
    )
    val_frac = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1.0 - val_frac), random_state=random_state
    )
    print(f"Split — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------------

class ODIRDataGenerator(tf.keras.utils.Sequence):
    """Keras-compatible generator for the ODIR-5K dataset."""

    def __init__(self, df, batch_size=config.BATCH_SIZE,
                 image_size=config.IMAGE_SIZE,
                 augment=False, shuffle=True,
                 apply_ben_graham=True):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.shuffle = shuffle
        self.apply_ben_graham = apply_ben_graham
        self.indices = np.arange(len(self.df))

        self._aug = ImageDataGenerator(
            rotation_range=config.ROTATION_RANGE,
            zoom_range=config.ZOOM_RANGE,
            brightness_range=config.BRIGHTNESS_RANGE,
            shear_range=config.SHEAR_RANGE,
            horizontal_flip=True,
            fill_mode='reflect'
        )
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = self.df.iloc[batch_idx]

        images, labels = [], []
        for _, row in batch.iterrows():
            try:
                img = load_and_preprocess_image(
                    row['image_path'], self.image_size, self.apply_ben_graham)
                if self.augment:
                    img = self._aug.random_transform(img)
            except Exception:
                img = np.zeros((*self.image_size, 3), dtype=np.float32)

            images.append(img)
            labels.append(row[LABEL_COLUMNS].values.astype(np.float32))

        return np.array(images), np.array(labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def get_generators(train_df, val_df, test_df, batch_size=config.BATCH_SIZE):
    train_gen = ODIRDataGenerator(train_df, batch_size=batch_size, augment=True, shuffle=True)
    val_gen   = ODIRDataGenerator(val_df,   batch_size=batch_size, augment=False, shuffle=False)
    test_gen  = ODIRDataGenerator(test_df,  batch_size=batch_size, augment=False, shuffle=False)
    return train_gen, val_gen, test_gen
