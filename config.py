import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# ODIR-5K dataset paths
ODIR_TRAIN_IMAGES = os.path.join(DATA_DIR, 'Training Images')
ODIR_TEST_IMAGES = os.path.join(DATA_DIR, 'Testing Images')
ODIR_ANNOTATIONS = os.path.join(DATA_DIR, 'data.xlsx')

# Disease labels (8 classes — ODIR)
DISEASE_LABELS = [
    'Normal', 'Diabetic Retinopathy', 'Glaucoma', 'Cataract',
    'AMD', 'Hypertensive Retinopathy', 'Myopia', 'Others'
]
DISEASE_CODES = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
NUM_CLASSES = 8

# Image settings
IMAGE_SIZE = (380, 380)
CHANNELS = 3
INPUT_SHAPE = (380, 380, 3)

# Training hyperparameters
BATCH_SIZE = 32
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 20
PHASE1_LR = 1e-3
PHASE2_LR = 1e-4
THRESHOLD = 0.5

# Fine-tuning: unfreeze top N backbone layers in Phase 2
FINE_TUNE_LAYERS = 30

# Dataset split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Model checkpoint paths
MODEL_NAME = 'efficientnetb4_odir'
CHECKPOINT_PATH = os.path.join(MODELS_DIR, f'{MODEL_NAME}_best.h5')
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, f'{MODEL_NAME}_final.h5')

# Callback settings
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5

# Augmentation
ROTATION_RANGE = 20
ZOOM_RANGE = 0.15
BRIGHTNESS_RANGE = (0.8, 1.2)
SHEAR_RANGE = 0.1

# GradCAM target layer (last conv block of EfficientNetB4)
GRADCAM_LAYER = 'top_conv'

# Flask settings
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = False
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'app', 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Reproducibility
SEED = 42
