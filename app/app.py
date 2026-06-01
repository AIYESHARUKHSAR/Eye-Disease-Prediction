"""
Flask web application for ODIR Eye Disease Prediction.
Routes:
  GET  /          — upload page
  POST /predict   — JSON inference endpoint
  GET  /about     — disease info page
  GET  /api/health — health check
"""

import os
import sys
import uuid
import json
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER']      = config.UPLOAD_FOLDER
app.secret_key = 'odir-efficientnetb4-mtech-nitd-2024'

os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)

# Lazy-loaded singletons
_model   = None
_gradcam = None


def _get_model():
    global _model, _gradcam
    if _model is None:
        try:
            from src.predict import load_model
            from src.utils import GradCAM
            _model   = load_model()
            _gradcam = GradCAM(_model)
            app.logger.info("Model loaded successfully.")
        except Exception as exc:
            app.logger.warning(f"Model could not be loaded: {exc}")
    return _model, _gradcam


def _allowed(filename):
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS)


def _save_upload(file, eye_label):
    ext      = file.filename.rsplit('.', 1)[1].lower()
    fname    = f"{uuid.uuid4().hex}_{eye_label}.{ext}"
    filepath = os.path.join(config.UPLOAD_FOLDER, fname)
    file.save(filepath)
    return filepath, fname


def _make_gradcam(gradcam, model, filepath, top_disease):
    """Generate GradCAM overlay; return filename or None on failure."""
    try:
        from src.preprocess import preprocess_single_image
        img_array = preprocess_single_image(filepath)
        class_idx = config.DISEASE_LABELS.index(top_disease)

        orig_bgr  = cv2.imread(filepath)
        overlay   = gradcam.generate_for_web(img_array, orig_bgr, class_idx)

        gc_fname  = f"gradcam_{os.path.basename(filepath)}"
        gc_path   = os.path.join(config.UPLOAD_FOLDER, gc_fname)
        cv2.imwrite(gc_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return gc_fname
    except Exception as exc:
        app.logger.warning(f"GradCAM failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html',
                           disease_labels=config.DISEASE_LABELS,
                           disease_codes=config.DISEASE_CODES)


@app.route('/predict', methods=['POST'])
def predict():
    model, gradcam = _get_model()
    if model is None:
        return jsonify({'error': 'Model not available. Train the model first.'}), 503

    uploaded_any = False
    results      = {}

    for eye in ('left_eye', 'right_eye'):
        file = request.files.get(eye)
        if file is None or file.filename == '' or not _allowed(file.filename):
            continue

        uploaded_any    = True
        filepath, fname = _save_upload(file, eye)

        from src.predict import predict_image
        pred = predict_image(filepath, model)

        gc_fname = _make_gradcam(gradcam, model, filepath,
                                 pred['top_prediction']['disease']) if gradcam else None

        results[eye] = {
            'image_filename':   fname,
            'gradcam_filename': gc_fname,
            'probabilities':    pred['probabilities'],
            'predictions':      pred['predictions'],
            'detected_diseases': pred['detected_diseases'],
            'confidence_scores': pred['confidence_scores'],
            'top_prediction':   pred['top_prediction'],
        }

    if not uploaded_any:
        return jsonify({'error': 'No valid image uploaded.'}), 400

    # Combine both eyes (max-probability fusion)
    if 'left_eye' in results and 'right_eye' in results:
        combined_probs = {
            l: max(results['left_eye']['probabilities'][l],
                   results['right_eye']['probabilities'][l])
            for l in config.DISEASE_LABELS
        }
        results['combined'] = {
            'probabilities':     combined_probs,
            'predictions':       {l: int(p >= config.THRESHOLD) for l, p in combined_probs.items()},
            'detected_diseases': [l for l, p in combined_probs.items() if p >= config.THRESHOLD],
            'confidence_scores': {l: f"{p*100:.1f}%" for l, p in combined_probs.items()},
        }

    return jsonify({'success': True, 'results': results})


@app.route('/about')
def about():
    descriptions = {
        'Normal':
            'Healthy fundus with no visible pathological changes. The optic disc, vessels, and macula appear normal.',
        'Diabetic Retinopathy':
            'A microvascular complication of diabetes mellitus. Characterised by microaneurysms, haemorrhages, and neovascularisation. Leading cause of blindness in working-age adults.',
        'Glaucoma':
            'A group of optic neuropathies defined by progressive loss of retinal ganglion cells. Often associated with elevated intraocular pressure. Detected by cup-to-disc ratio changes.',
        'Cataract':
            'Opacification of the crystalline lens causing blurred vision. The most common cause of treatable blindness worldwide; corrected by surgical lens replacement.',
        'AMD':
            'Age-related macular degeneration — degeneration of the macula in elderly patients. Presents as drusen deposits and choroidal neovascularisation. Impairs central vision.',
        'Hypertensive Retinopathy':
            'Retinal changes secondary to systemic arterial hypertension: arteriovenous nicking, flame haemorrhages, cotton-wool spots, and papilloedema in severe cases.',
        'Myopia':
            'Pathological (degenerative) myopia — axial elongation of the globe leading to lattice degeneration, posterior staphyloma, and increased risk of retinal detachment.',
        'Others':
            'A heterogeneous category including epiretinal membranes, branch vein/artery occlusions, retinitis pigmentosa, and other conditions not captured by the seven labels above.',
    }
    return render_template('about.html', disease_descriptions=descriptions,
                           disease_labels=config.DISEASE_LABELS)


@app.route('/api/health')
def health():
    m, _ = _get_model()
    return jsonify({'status': 'healthy', 'model_loaded': m is not None,
                    'version': '1.0.0', 'diseases': config.DISEASE_LABELS})


if __name__ == '__main__':
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
