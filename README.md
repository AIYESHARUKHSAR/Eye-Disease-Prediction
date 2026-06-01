# Multi-Class Eye Disease Prediction Using EfficientNetB4 and ODIR-5K

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?logo=tensorflow)
![EfficientNet](https://img.shields.io/badge/Model-EfficientNetB4-green)
![Dataset](https://img.shields.io/badge/Dataset-ODIR--5K-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Flask](https://img.shields.io/badge/Web-Flask%203.0-red?logo=flask)

---

## Medical Motivation

Ocular diseases are among the leading causes of preventable blindness worldwide.
Early automated detection from fundus retinal photographs can dramatically improve
outcomes, especially in regions with limited ophthalmology specialists.
This project applies state-of-the-art transfer learning (EfficientNetB4) to
simultaneously detect **eight** ocular conditions from a single fundus image,
providing clinicians with a fast, reproducible AI second opinion.

---

## Disease Categories (ODIR-5K)

| Code | Disease | Description |
|------|---------|-------------|
| **N** | Normal | Healthy fundus — no pathological changes |
| **D** | Diabetic Retinopathy | Retinal vascular damage from chronic diabetes; leading cause of blindness in working-age adults |
| **G** | Glaucoma | Progressive optic nerve damage, often from elevated intraocular pressure |
| **C** | Cataract | Lens opacification — most common cause of treatable blindness globally |
| **A** | AMD | Age-related macular degeneration — central vision loss in elderly |
| **H** | Hypertensive Retinopathy | Retinal vessel changes secondary to high blood pressure |
| **M** | Myopia | Pathological myopia with structural elongation of the globe |
| **O** | Others | Heterogeneous group: epiretinal membranes, vein occlusions, retinitis pigmentosa, etc. |

---

## EfficientNetB4 Architecture

EfficientNetB4 is part of the EfficientNet family, which uses a **compound scaling**
strategy to uniformly scale network depth, width, and resolution. B4 offers an excellent
accuracy/efficiency trade-off at an input resolution of **380×380**.

```
Input (380×380×3)
      │
EfficientNetB4 Backbone (ImageNet pre-trained)
      │  ← Phase 1: Frozen
      │  ← Phase 2: Top 30 layers unfrozen
      │
GlobalAveragePooling2D
      │
Dense(512) → BatchNorm → ReLU → Dropout(0.5)
      │
Dense(256) → BatchNorm → ReLU → Dropout(0.4)
      │
Dense(8, activation='sigmoid')   ← Multi-label output
```

**Multi-label classification**: each output neuron independently models
the probability of one disease. A sigmoid threshold of 0.5 converts
probabilities to binary predictions. A patient can have multiple conditions
simultaneously (e.g., diabetic retinopathy + hypertensive retinopathy).

---

## Folder Structure

```
Eye Disease Prediction/
├── src/
│   ├── model.py          # EfficientNetB4 architecture + callbacks
│   ├── preprocess.py     # ODIR loader, Ben Graham, augmentation, generators
│   ├── train.py          # Two-phase training pipeline (CLI-ready)
│   ├── predict.py        # Single-image & batch inference + CLI
│   ├── evaluate.py       # ROC-AUC, confusion matrix, history plots
│   └── utils.py          # GradCAM, dataset visualisation, report writer
├── app/
│   ├── app.py            # Flask web application
│   ├── templates/
│   │   ├── index.html    # Upload dashboard
│   │   └── about.html    # Disease info & architecture page
│   └── static/
│       ├── style.css     # Professional medical UI
│       ├── script.js     # Dynamic results with bar chart animation
│       └── uploads/      # Temporary image store (git-ignored)
├── models/               # Saved model weights (git-ignored)
├── data/                 # ODIR-5K images & Excel (git-ignored)
├── results/              # Plots, metrics JSON, test CSV
├── notebooks/
│   └── exploration.ipynb # EDA: label distribution, sample images, GradCAM
├── config.py             # All hyperparameters and paths
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/AIYESHARUKHSAR/eye-disease-prediction-efficientnet.git
cd eye-disease-prediction-efficientnet
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ODIR-5K Dataset Download

1. Request access at the official ODIR-5K page:
   https://odir2019.grand-challenge.org/dataset/
2. Download the training package and extract it.
3. Place files so the structure matches `config.py`:

```
data/
├── Training Images/      ← patient_left.jpg / patient_right.jpg
├── Testing Images/
└── data.xlsx             ← official annotation Excel file
```

The `data.xlsx` file contains columns for `ID`, `Patient Age`, `Patient Sex`,
`Left-Diagnostic Keywords`, `Right-Diagnostic Keywords`, and the eight binary
label columns `N, D, G, C, A, H, M, O`.

---

## Data Preprocessing

### Ben Graham Preprocessing

Each fundus image is preprocessed using the method proposed by Ben Graham
(Kaggle Diabetic Retinopathy Detection winner):

1. **Circular mask** — removes non-retinal background pixels.
2. **Contrast enhancement** — local subtraction of a Gaussian-blurred version
   to amplify fine retinal details.

```python
from src.preprocess import load_and_preprocess_image
img = load_and_preprocess_image('fundus.jpg', apply_ben_graham=True)
```

### Augmentation (training only)

| Transform | Value |
|-----------|-------|
| Rotation | ±20° |
| Zoom | 15% |
| Brightness | 0.8 – 1.2× |
| Shear | 10% |
| Horizontal flip | Yes |

### Dataset Split

| Set | Ratio | Stratified on |
|-----|-------|--------------|
| Train | 70% | Normal label |
| Validation | 15% | — |
| Test | 15% | — |

---

## Training

### Phase 1 — Train custom head (backbone frozen)

```bash
python -m src.train --data-dir data/Training\ Images --annotations data/data.xlsx
```

This runs both phases automatically.  To customise, edit `config.py`:

```python
PHASE1_EPOCHS = 10   # head-only training
PHASE2_EPOCHS = 20   # fine-tuning
PHASE1_LR     = 1e-3
PHASE2_LR     = 1e-4
FINE_TUNE_LAYERS = 30
```

Model checkpoints are saved to `models/` after each improvement in val AUC.

---

## Evaluation

```bash
python -m src.evaluate --model models/efficientnetb4_odir_best.h5
```

This generates:

| Artefact | Location |
|----------|----------|
| ROC curves (8 classes + micro-avg) | `results/roc_curves.png` |
| Confusion matrices (2×4 grid) | `results/confusion_matrices.png` |
| Per-class AUC bar chart | `results/auc_bar_chart.png` |
| Training history plot | `results/training_history.png` |
| Metrics JSON | `results/metrics.json` |

---

## Inference (CLI)

```bash
# Single image
python -m src.predict path/to/fundus.jpg

# JSON output
python -m src.predict path/to/fundus.jpg --json

# Custom threshold
python -m src.predict path/to/fundus.jpg --threshold 0.4

# Specify model path
python -m src.predict path/to/fundus.jpg --model models/efficientnetb4_odir_best.h5
```

---

## Flask Web Application

### Start the server

```bash
python app/app.py
```

Navigate to **http://localhost:5000**

### Features

- Upload left eye, right eye, or both fundus images
- Animated confidence bar chart for all 8 disease classes
- Color-coded results: red = detected, green = normal
- GradCAM heatmap highlighting the regions driving the top prediction
- Dual-eye fusion — combines left + right predictions by max probability
- REST API: `POST /predict` returns JSON
- About page with disease descriptions and architecture diagram

### API Example

```bash
curl -X POST http://localhost:5000/predict \
     -F "left_eye=@left_fundus.jpg" \
     -F "right_eye=@right_fundus.jpg"
```

Response:

```json
{
  "success": true,
  "results": {
    "left_eye": {
      "probabilities": {"Normal": 0.05, "Diabetic Retinopathy": 0.91, ...},
      "detected_diseases": ["Diabetic Retinopathy"],
      "top_prediction": {"disease": "Diabetic Retinopathy", "confidence": 0.91}
    }
  }
}
```

---

## Expected Performance Metrics

| Disease | Expected AUC |
|---------|-------------|
| Normal | ≥ 0.92 |
| Diabetic Retinopathy | ≥ 0.90 |
| Glaucoma | ≥ 0.88 |
| Cataract | ≥ 0.91 |
| AMD | ≥ 0.87 |
| Hypertensive Retinopathy | ≥ 0.85 |
| Myopia | ≥ 0.89 |
| Others | ≥ 0.83 |
| **Macro AUC** | **≥ 0.88** |
| **Micro AUC** | **≥ 0.89** |

*Results vary with hardware, exact ODIR split, and augmentation settings.*

---

## Sample Prediction Screenshots

> Add screenshots here after running the Flask app.

| Dashboard | Results |
|-----------|---------|
| *(upload screenshot)* | *(prediction results screenshot)* |

---

## GradCAM Explainability

GradCAM (Gradient-weighted Class Activation Mapping) produces a heatmap showing
which retinal regions the model focused on for each predicted disease class.
This aids clinical trust and model validation.

```python
from src.utils import GradCAM
from src.preprocess import preprocess_single_image

gcam = GradCAM(model)
img  = preprocess_single_image('fundus.jpg')
gcam.visualize(img, original_image, 'Diabetic Retinopathy', class_idx=1)
```

---

## References

1. Tan, M., & Le, Q. V. (2019). **EfficientNet: Rethinking Model Scaling for CNNs**. ICML 2019.
2. Larxel. **ODIR-5K: Ocular Disease Intelligent Recognition**. Kaggle Dataset.
3. Selvaraju, R. R., et al. (2017). **Grad-CAM: Visual Explanations from Deep Networks**. ICCV 2017.
4. Ben Graham (2015). **Diabetic Retinopathy Detection**. Kaggle Competition Report.
5. World Health Organisation (2023). **World Report on Vision**.

---

## Author

**Aiyesha Rukhsar**  
M.Tech Student, National Institute of Technology Delhi (NIT Delhi)  
Email: emailtoaiyesha3@gmail.com  

---

## License

This project is licensed under the **MIT License** — see below.

```
MIT License

Copyright (c) 2024 Aiyesha Rukhsar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
