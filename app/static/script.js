/* ===================================================================
   OcularAI Frontend — Dynamic results display
   EfficientNetB4 | ODIR-5K | M.Tech NIT Delhi
   =================================================================== */

'use strict';

const DISEASE_COLORS = {
  'Normal':                   '#27ae60',
  'Diabetic Retinopathy':     '#e74c3c',
  'Glaucoma':                 '#e67e22',
  'Cataract':                 '#f39c12',
  'AMD':                      '#8e44ad',
  'Hypertensive Retinopathy': '#c0392b',
  'Myopia':                   '#2980b9',
  'Others':                   '#7f8c8d',
};

// ── File-input wiring ─────────────────────────────────────────────

function wireEyeInput(inputId, previewId, clearId, boxId) {
  const input   = document.getElementById(inputId);
  const preview = document.getElementById(previewId);
  const clearBtn = document.getElementById(clearId);
  const box     = document.getElementById(boxId);
  const label   = box.querySelector('.upload-label');

  input.addEventListener('change', () => {
    const file = input.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.classList.remove('hidden');
    clearBtn.classList.remove('hidden');
    label.classList.add('hidden');
    box.classList.add('has-image');
    updatePredictBtn();
  });

  clearBtn.addEventListener('click', () => {
    input.value = '';
    preview.src = '';
    preview.classList.add('hidden');
    clearBtn.classList.add('hidden');
    label.classList.remove('hidden');
    box.classList.remove('has-image');
    updatePredictBtn();
  });

  // Drag-and-drop
  box.addEventListener('dragover', e => { e.preventDefault(); box.classList.add('drag-over'); });
  box.addEventListener('dragleave', () => box.classList.remove('drag-over'));
  box.addEventListener('drop', e => {
    e.preventDefault();
    box.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
      input.dispatchEvent(new Event('change'));
    }
  });
}

function updatePredictBtn() {
  const leftFile  = document.getElementById('leftEyeInput').files[0];
  const rightFile = document.getElementById('rightEyeInput').files[0];
  document.getElementById('predictBtn').disabled = !(leftFile || rightFile);
}

// ── Form submit / inference ───────────────────────────────────────

document.getElementById('uploadForm').addEventListener('submit', async e => {
  e.preventDefault();

  const form    = new FormData(e.target);
  const loading = document.getElementById('loadingOverlay');
  const results = document.getElementById('resultsSection');

  loading.classList.remove('hidden');
  results.classList.add('hidden');

  try {
    const resp = await fetch('/predict', { method: 'POST', body: form });
    const data = await resp.json();

    if (!resp.ok || data.error) {
      alert(`Error: ${data.error || 'Unknown server error'}`);
      return;
    }
    renderResults(data.results);
    results.classList.remove('hidden');
    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    alert(`Network error: ${err.message}`);
  } finally {
    loading.classList.add('hidden');
  }
});

// ── Render results ────────────────────────────────────────────────

function renderResults(results) {
  const tabsEl   = document.getElementById('resultTabs');
  const panelsEl = document.getElementById('resultPanels');
  const banner   = document.getElementById('statusBanner');

  tabsEl.innerHTML   = '';
  panelsEl.innerHTML = '';

  const keys = Object.keys(results);
  let allDetected = [];

  // Collect all detected diseases across eyes
  keys.forEach(key => {
    const r = results[key];
    if (r.detected_diseases) allDetected.push(...r.detected_diseases);
  });
  allDetected = [...new Set(allDetected)];

  const hasDisease = allDetected.some(d => d !== 'Normal');
  banner.textContent = hasDisease
    ? `Detected: ${allDetected.filter(d => d !== 'Normal').join(', ')}`
    : 'No disease detected — Normal';
  banner.className = `status-banner ${hasDisease ? 'detected' : 'normal'}`;

  // Build tabs + panels
  keys.forEach((key, idx) => {
    const label = key.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
    const tab = document.createElement('button');
    tab.className = `tab-btn${idx === 0 ? ' active' : ''}`;
    tab.textContent = label;
    tab.dataset.target = key;
    tab.addEventListener('click', () => switchTab(key));
    tabsEl.appendChild(tab);

    const panel = document.createElement('div');
    panel.id = `panel-${key}`;
    panel.className = `result-panel${idx !== 0 ? ' hidden' : ''}`;
    panel.innerHTML = buildPanel(key, results[key]);
    panelsEl.appendChild(panel);
  });

  // Animate bars after a short delay
  setTimeout(animateBars, 80);
}

function switchTab(key) {
  document.querySelectorAll('.tab-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.target === key);
  });
  document.querySelectorAll('.result-panel').forEach(p => {
    p.classList.toggle('hidden', p.id !== `panel-${key}`);
  });
}

function buildPanel(key, data) {
  const probs   = data.probabilities   || {};
  const preds   = data.predictions     || {};
  const gradcam = data.gradcam_filename || null;
  const imgFile = data.image_filename  || null;

  const barsHtml = Object.entries(probs).map(([disease, prob]) => {
    const pct     = (prob * 100).toFixed(1);
    const detected = preds[disease] === 1;
    const color   = DISEASE_COLORS[disease] || '#888';
    return `
      <div class="disease-row">
        <div class="disease-row-header">
          <span class="disease-name">${disease}</span>
          <div style="display:flex;gap:8px;align-items:center">
            <span class="disease-conf">${pct}%</span>
            <span class="disease-tag ${detected ? 'detected' : 'normal'}">
              ${detected ? 'DETECTED' : 'Normal'}
            </span>
          </div>
        </div>
        <div class="bar-track">
          <div class="bar-fill" data-width="${pct}"
               style="width:0%;background:${color}"></div>
        </div>
      </div>`;
  }).join('');

  const gradcamHtml = gradcam
    ? `<div class="gradcam-panel">
         <h3>GradCAM — ${data.top_prediction ? data.top_prediction.disease : ''}</h3>
         <img src="/static/uploads/${gradcam}" alt="GradCAM heatmap" loading="lazy" />
         <p class="gradcam-caption">Highlighted regions most influential for the top prediction</p>
       </div>`
    : '';

  const eyeImgHtml = imgFile
    ? `<div class="eye-thumb">
         <img src="/static/uploads/${imgFile}" alt="Uploaded fundus" loading="lazy" />
         <p>${key.replace('_', ' ')}</p>
       </div>`
    : '';

  return `
    <div class="card result-grid">
      <div>
        ${eyeImgHtml}
        <div class="disease-bars">${barsHtml}</div>
      </div>
      <div>${gradcamHtml}</div>
    </div>`;
}

function animateBars() {
  document.querySelectorAll('.bar-fill').forEach(bar => {
    const w = bar.dataset.width;
    bar.style.width = w + '%';
  });
}

// ── Init ──────────────────────────────────────────────────────────

wireEyeInput('leftEyeInput',  'leftPreview',  'clearLeft',  'leftBox');
wireEyeInput('rightEyeInput', 'rightPreview', 'clearRight', 'rightBox');
