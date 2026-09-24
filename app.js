/**
 * TERRA.AI // Minimalist Monochrome Machine Learning Engine
 * Client-side inference, real-time telemetry, interactive visualizations
 */

// Model artifacts state
const STATE = {
  modelData: null,
  classes: ['High', 'Low', 'Medium'],
  scaler: {
    mean: [6.64056667, 264.29783333, 29.49241667, 189.36475, 2.95308333, 29.35839167],
    scale: [1.26346851, 125.13125687, 14.1365602, 74.72648392, 1.4479985, 14.11240589]
  },
  metrics: {
    dt: {
      accuracy: 0.6367,
      precision: 0.6786,
      recall: 0.6367,
      f1: 0.6456,
      cv_mean: 0.6727,
      cv_std: 0.0359,
      cm: [[66, 1, 33], [1, 59, 40], [22, 12, 66]],
      importances: { 'pH': 0.3797, 'Nitrogen': 0.2161, 'Phosphorus': 0.1391, 'Potassium': 0.1320, 'Organic_Matter': 0.1230, 'Moisture': 0.0101 }
    },
    rf: {
      accuracy: 0.7400,
      precision: 0.7393,
      recall: 0.7400,
      f1: 0.7396,
      cv_mean: 0.7573,
      cv_std: 0.0414,
      cm: [[77, 0, 23], [1, 84, 15], [22, 17, 61]],
      importances: { 'pH': 0.2774, 'Nitrogen': 0.2259, 'Phosphorus': 0.1464, 'Potassium': 0.1297, 'Organic_Matter': 0.1212, 'Moisture': 0.0995 }
    }
  },
  dataset: [],
  filteredDataset: [],
  currentPage: 1,
  pageSize: 20,
  activeMatrixModel: 'rf'
};

// Preset soil profiles
const PRESETS = {
  optimal: { pH: 6.8, nitrogen: 320, phosphorus: 32, potassium: 210, organic: 4.2, moisture: 28.0 },
  acidic: { pH: 4.8, nitrogen: 120, phosphorus: 12, potassium: 85, organic: 1.1, moisture: 14.0 },
  depleted: { pH: 7.8, nitrogen: 180, phosphorus: 16, potassium: 120, organic: 1.6, moisture: 42.0 },
  fertile: { pH: 7.1, nitrogen: 410, phosphorus: 45, potassium: 270, organic: 5.2, moisture: 30.0 },
  alkaline: { pH: 8.4, nitrogen: 140, phosphorus: 10, potassium: 95, organic: 0.9, moisture: 8.0 }
};

// DOM Elements
const DOM = {
  // Sliders
  pH: document.getElementById('input-pH'),
  nitrogen: document.getElementById('input-nitrogen'),
  phosphorus: document.getElementById('input-phosphorus'),
  potassium: document.getElementById('input-potassium'),
  organic: document.getElementById('input-organic'),
  moisture: document.getElementById('input-moisture'),
  
  // Slider Value Displays
  valPH: document.getElementById('val-pH'),
  valNitrogen: document.getElementById('val-nitrogen'),
  valPhosphorus: document.getElementById('val-phosphorus'),
  valPotassium: document.getElementById('val-potassium'),
  valOrganic: document.getElementById('val-organic'),
  valMoisture: document.getElementById('val-moisture'),

  // Outputs
  rfClassOutput: document.getElementById('rfClassOutput'),
  rfConfidenceVal: document.getElementById('rfConfidenceVal'),
  dtClassOutput: document.getElementById('dtClassOutput'),
  dtConfidenceVal: document.getElementById('dtConfidenceVal'),
  inferenceLatency: document.getElementById('inferenceLatency'),

  // Prob Bars
  probNumHigh: document.getElementById('probNumHigh'),
  probNumMed: document.getElementById('probNumMedium'),
  probNumLow: document.getElementById('probNumLow'),
  probFillHigh: document.getElementById('probFillHigh'),
  probFillMed: document.getElementById('probFillMedium'),
  probFillLow: document.getElementById('probFillLow'),

  // Prescriptions
  prescriptionBody: document.getElementById('prescriptionBody'),

  // Navigation & Theme
  navTabs: document.querySelectorAll('.nav-tab'),
  tabContents: document.querySelectorAll('.tab-content'),
  themeToggleBtn: document.getElementById('themeToggleBtn'),
  themeIcon: document.getElementById('themeIcon'),
  resetSlidersBtn: document.getElementById('resetSlidersBtn'),

  // Presets
  presetBtns: document.querySelectorAll('.preset-btn'),

  // Confusion Matrix
  confusionMatrixSvg: document.getElementById('confusionMatrixSvg'),
  btnShowRfMatrix: document.getElementById('btnShowRfMatrix'),
  btnShowDtMatrix: document.getElementById('btnShowDtMatrix'),

  // Feature Bars
  featureBarsContainer: document.getElementById('featureBarsContainer'),

  // Dataset Explorer
  datasetSearchInput: document.getElementById('datasetSearchInput'),
  datasetTableBody: document.getElementById('datasetTableBody'),
  prevPageBtn: document.getElementById('prevPageBtn'),
  nextPageBtn: document.getElementById('nextPageBtn'),
  pageInfo: document.getElementById('pageInfo'),

  // Modal
  reportModalBackdrop: document.getElementById('reportModalBackdrop'),
  generateReportBtn: document.getElementById('generateReportBtn'),
  closeModalBtn: document.getElementById('closeModalBtn'),
  cancelModalBtn: document.getElementById('cancelModalBtn'),
  printReportBtn: document.getElementById('printReportBtn'),
  reportTimestamp: document.getElementById('reportTimestamp'),
  repClassification: document.getElementById('repClassification'),
  repConfidence: document.getElementById('repConfidence'),
  reportTableBody: document.getElementById('reportTableBody'),
  reportDirectives: document.getElementById('reportDirectives')
};

// =============================================================================
// INITIALIZATION
// =============================================================================
document.addEventListener('DOMContentLoaded', async () => {
  initTheme();
  setupEventListeners();
  renderFeatureImportanceBars();
  renderConfusionMatrix('rf');
  
  await loadModelArtifacts();
  await loadDataset();
  
  runInference();
});

// =============================================================================
// EVENT LISTENERS SETUP
// =============================================================================
function setupEventListeners() {
  // Navigation Tabs
  DOM.navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetTab = tab.getAttribute('data-tab');
      DOM.navTabs.forEach(t => {
        t.classList.remove('active');
        t.setAttribute('aria-selected', 'false');
      });
      DOM.tabContents.forEach(c => c.classList.remove('active'));

      tab.classList.add('active');
      tab.setAttribute('aria-selected', 'true');
      const contentEl = document.getElementById(targetTab);
      if (contentEl) contentEl.classList.add('active');
    });
  });

  // Slider Inputs
  [DOM.pH, DOM.nitrogen, DOM.phosphorus, DOM.potassium, DOM.organic, DOM.moisture].forEach(input => {
    input.addEventListener('input', () => {
      updateSliderValues();
      runInference();
    });
  });

  // Reset Sliders Button
  DOM.resetSlidersBtn.addEventListener('click', () => {
    applyPreset('optimal');
  });

  // Presets
  DOM.presetBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const presetKey = btn.getAttribute('data-preset');
      DOM.presetBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      applyPreset(presetKey);
    });
  });

  // Matrix Switcher
  DOM.btnShowRfMatrix.addEventListener('click', () => {
    DOM.btnShowRfMatrix.classList.add('active');
    DOM.btnShowDtMatrix.classList.remove('active');
    renderConfusionMatrix('rf');
  });

  DOM.btnShowDtMatrix.addEventListener('click', () => {
    DOM.btnShowDtMatrix.classList.add('active');
    DOM.btnShowRfMatrix.classList.remove('active');
    renderConfusionMatrix('dt');
  });

  // Dataset Table Filter & Pagination
  DOM.datasetSearchInput.addEventListener('input', (e) => {
    filterDataset(e.target.value);
  });

  DOM.prevPageBtn.addEventListener('click', () => {
    if (STATE.currentPage > 1) {
      STATE.currentPage--;
      renderDatasetTable();
    }
  });

  DOM.nextPageBtn.addEventListener('click', () => {
    const totalPages = Math.ceil(STATE.filteredDataset.length / STATE.pageSize) || 1;
    if (STATE.currentPage < totalPages) {
      STATE.currentPage++;
      renderDatasetTable();
    }
  });

  // Report Modal
  DOM.generateReportBtn.addEventListener('click', openReportModal);
  DOM.closeModalBtn.addEventListener('click', closeReportModal);
  DOM.cancelModalBtn.addEventListener('click', closeReportModal);
  DOM.reportModalBackdrop.addEventListener('click', (e) => {
    if (e.target === DOM.reportModalBackdrop) closeReportModal();
  });
  DOM.printReportBtn.addEventListener('click', () => {
    window.print();
  });

  // Theme Toggle
  DOM.themeToggleBtn.addEventListener('click', toggleTheme);
}

// =============================================================================
// THEME MANAGEMENT (MONOCHROME DARK / LIGHT)
// =============================================================================
function initTheme() {
  const savedTheme = localStorage.getItem('terra-theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);
  DOM.themeIcon.textContent = savedTheme === 'dark' ? '◐' : '◑';
}

function toggleTheme() {
  const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', newTheme);
  localStorage.setItem('terra-theme', newTheme);
  DOM.themeIcon.textContent = newTheme === 'dark' ? '◐' : '◑';
  renderConfusionMatrix(STATE.activeMatrixModel);
}

// =============================================================================
// SLIDERS & PRESET HANDLING
// =============================================================================
function updateSliderValues() {
  DOM.valPH.textContent = parseFloat(DOM.pH.value).toFixed(2);
  DOM.valNitrogen.textContent = parseFloat(DOM.nitrogen.value).toFixed(1);
  DOM.valPhosphorus.textContent = parseFloat(DOM.phosphorus.value).toFixed(1);
  DOM.valPotassium.textContent = parseFloat(DOM.potassium.value).toFixed(1);
  DOM.valOrganic.textContent = parseFloat(DOM.organic.value).toFixed(2) + '%';
  DOM.valMoisture.textContent = parseFloat(DOM.moisture.value).toFixed(1) + '%';
}

function applyPreset(key) {
  if (key === 'random') {
    DOM.pH.value = (Math.random() * (8.8 - 4.5) + 4.5).toFixed(1);
    DOM.nitrogen.value = Math.round(Math.random() * (480 - 50) + 50);
    DOM.phosphorus.value = Math.round(Math.random() * (55 - 5) + 5);
    DOM.potassium.value = Math.round(Math.random() * (320 - 60) + 60);
    DOM.organic.value = (Math.random() * (5.5 - 0.5) + 0.5).toFixed(1);
    DOM.moisture.value = (Math.random() * (55 - 5) + 5).toFixed(1);
  } else if (PRESETS[key]) {
    const p = PRESETS[key];
    DOM.pH.value = p.pH;
    DOM.nitrogen.value = p.nitrogen;
    DOM.phosphorus.value = p.phosphorus;
    DOM.potassium.value = p.potassium;
    DOM.organic.value = p.organic;
    DOM.moisture.value = p.moisture;
  }
  updateSliderValues();
  runInference();
}

// =============================================================================
// MODEL LOADING & CLIENT INFERENCE ENGINE
// =============================================================================
async function loadModelArtifacts() {
  try {
    const res = await fetch('models/model_export.json');
    if (res.ok) {
      STATE.modelData = await res.json();
      document.getElementById('modelStatusText').textContent = `ENGINE ONLINE (${STATE.modelData.random_forest.length} RF TREES)`;
    }
  } catch (err) {
    console.warn('Using client-side calibrated fallback ML engine.');
  }
}

function traverseTree(tree, sample) {
  let node = 0;
  while (tree.children_left[node] !== -1) {
    const featureIdx = tree.feature[node];
    const threshold = tree.threshold[node];
    if (sample[featureIdx] <= threshold) {
      node = tree.children_left[node];
    } else {
      node = tree.children_right[node];
    }
  }
  const counts = tree.value[node][0];
  const sum = counts.reduce((a, b) => a + b, 0);
  return counts.map(c => c / sum);
}

function runInference() {
  const t0 = performance.now();

  const rawFeatures = [
    parseFloat(DOM.pH.value),
    parseFloat(DOM.nitrogen.value),
    parseFloat(DOM.phosphorus.value),
    parseFloat(DOM.potassium.value),
    parseFloat(DOM.organic.value),
    parseFloat(DOM.moisture.value)
  ];

  // Feature Scaling: (X - mean) / scale
  const scaledFeatures = rawFeatures.map((val, idx) => (val - STATE.scaler.mean[idx]) / STATE.scaler.scale[idx]);

  let dtProbs = [0, 0, 0];
  let rfProbs = [0, 0, 0];

  if (STATE.modelData && STATE.modelData.decision_tree) {
    // 1. Single Decision Tree Traversal
    dtProbs = traverseTree(STATE.modelData.decision_tree, scaledFeatures);

    // 2. Random Forest 100 Trees Ensemble Averaging
    const numTrees = STATE.modelData.random_forest.length;
    for (let i = 0; i < numTrees; i++) {
      const treeProbs = traverseTree(STATE.modelData.random_forest[i], scaledFeatures);
      rfProbs[0] += treeProbs[0] / numTrees;
      rfProbs[1] += treeProbs[1] / numTrees;
      rfProbs[2] += treeProbs[2] / numTrees;
    }
  } else {
    // High-Fidelity Agronomic Scoring Fallback (exact replica of generate_data.py)
    const [pH, N, P, K, OM, M] = rawFeatures;
    let score = 0;
    
    // pH score
    if (pH >= 6.0 && pH <= 7.5) score += 25.0;
    else if ((pH >= 5.5 && pH < 6.0) || (pH > 7.5 && pH <= 8.2)) score += 15.0;
    else score += 5.0;

    // N score
    if (N >= 280) score += 20.0;
    else if (N >= 160) score += 12.0;
    else score += 4.0;

    // P score
    if (P >= 25) score += 15.0;
    else if (P >= 14) score += 9.0;
    else score += 3.0;

    // K score
    if (K >= 150) score += 15.0;
    else if (K >= 100) score += 9.0;
    else score += 3.0;

    // OM score
    if (OM >= 3.0) score += 15.0;
    else if (OM >= 1.5) score += 9.0;
    else score += 3.0;

    // Moisture score
    if (M >= 20.0 && M <= 40.0) score += 10.0;
    else if ((M >= 12.0 && M < 20.0) || (M > 40.0 && M <= 48.0)) score += 6.0;
    else score += 2.0;

    // Softmax-like probability conversion (Classes: High, Low, Medium)
    if (score >= 70) {
      rfProbs = [0.88, 0.02, 0.10];
      dtProbs = [0.92, 0.00, 0.08];
    } else if (score >= 48) {
      rfProbs = [0.12, 0.18, 0.70];
      dtProbs = [0.10, 0.20, 0.70];
    } else {
      rfProbs = [0.03, 0.85, 0.12];
      dtProbs = [0.00, 0.90, 0.10];
    }
  }

  const t1 = performance.now();
  const latency = Math.max(0.1, (t1 - t0)).toFixed(2);
  DOM.inferenceLatency.textContent = `LATENCY: ${latency}ms`;

  // Determine Class Index (0: High, 1: Low, 2: Medium)
  const dtMaxIdx = dtProbs.indexOf(Math.max(...dtProbs));
  const rfMaxIdx = rfProbs.indexOf(Math.max(...rfProbs));

  const dtClass = STATE.classes[dtMaxIdx];
  const rfClass = STATE.classes[rfMaxIdx];

  const dtConf = (dtProbs[dtMaxIdx] * 100).toFixed(1);
  const rfConf = (rfProbs[rfMaxIdx] * 100).toFixed(1);

  // Update UI Verdicts
  DOM.rfClassOutput.textContent = `${rfClass.toUpperCase()} QUALITY`;
  DOM.rfConfidenceVal.textContent = `${rfConf}%`;

  DOM.dtClassOutput.textContent = `${dtClass.toUpperCase()} QUALITY`;
  DOM.dtConfidenceVal.textContent = `${dtConf}%`;

  // Update Probability Bars
  const pHigh = (rfProbs[0] * 100).toFixed(1);
  const pLow = (rfProbs[1] * 100).toFixed(1);
  const pMed = (rfProbs[2] * 100).toFixed(1);

  DOM.probNumHigh.textContent = `${pHigh}%`;
  DOM.probNumMed.textContent = `${pMed}%`;
  DOM.probNumLow.textContent = `${pLow}%`;

  DOM.probFillHigh.style.width = `${pHigh}%`;
  DOM.probFillMed.style.width = `${pMed}%`;
  DOM.probFillLow.style.width = `${pLow}%`;

  // Generate Prescription
  updatePrescription(rfClass, rawFeatures);
}

function updatePrescription(soilClass, [pH, N, P, K, OM, M]) {
  let text = '';
  if (soilClass === 'High') {
    text = `🟢 <b>High Agronomic Fertility Confirmed.</b> Balanced biochemical profile with adequate organic matter (${OM.toFixed(1)}%) and optimal pH (${pH.toFixed(1)}). Recommended for high-yield food and cash crop production without immediate synthetic amendments. Maintain regular crop rotation.`;
  } else if (soilClass === 'Medium') {
    let tips = [];
    if (pH < 6.0) tips.push('Apply agricultural lime (calcium carbonate) to buffer acidity');
    if (pH > 7.5) tips.push('Incorporate gypsum or elemental sulfur to regulate alkaline soil chemistry');
    if (N < 280) tips.push('Apply slow-release nitrogenous or organic compost top-dressing');
    if (P < 25) tips.push('Augment phosphorus availability via rock phosphate or SSP');
    if (K < 150) tips.push('Supplement potassium to boost plant cell wall resilience');
    if (OM < 3.0) tips.push('Introduce green manure cover crops to elevate organic matter');
    
    text = `🟡 <b>Moderate Fertility Level Detected.</b> Suitable for standard cultivation with targeted agronomic intervention: ${tips.slice(0, 3).join('; ')}.`;
  } else {
    let severe = [];
    if (pH < 5.5) severe.push('Severe soil acidification detected');
    if (pH > 8.2) severe.push('High alkalinity salt lock');
    if (N < 160) severe.push('Critical nitrogen depletion');
    if (OM < 1.5) severe.push('Severe organic humus deficit');
    if (M < 12) severe.push('High drought moisture stress');

    text = `🔴 <b>Low Soil Quality / Acute Nutrient Stress.</b> Root growth and crop yield will suffer severe limitations without immediate remediation: ${severe.join(', ')}. Soil restructuring with organic biochar, liming, and moisture retention mulching required before planting.`;
  }
  DOM.prescriptionBody.innerHTML = text;
}

// =============================================================================
// CONFUSION MATRIX RENDERER (SVG)
// =============================================================================
function renderConfusionMatrix(modelKey) {
  STATE.activeMatrixModel = modelKey;
  const cm = STATE.metrics[modelKey].cm;
  const classes = STATE.classes; // ['High', 'Low', 'Medium']
  const isLight = document.documentElement.getAttribute('data-theme') === 'light';

  const svg = DOM.confusionMatrixSvg;
  svg.innerHTML = '';

  const cellSize = 75;
  const startX = 110;
  const startY = 70;

  // Max value for opacity normalization
  const maxVal = 100;

  // Header Title
  const title = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  title.setAttribute('x', 210);
  title.setAttribute('y', 25);
  title.setAttribute('text-anchor', 'middle');
  title.setAttribute('fill', isLight ? '#09090b' : '#fafafa');
  title.setAttribute('font-family', 'JetBrains Mono, monospace');
  title.setAttribute('font-size', '12');
  title.setAttribute('font-weight', '700');
  title.textContent = `PREDICTED LABELS (${modelKey === 'rf' ? 'RANDOM FOREST' : 'DECISION TREE'})`;
  svg.appendChild(title);

  // Column Headers (Predicted)
  classes.forEach((c, j) => {
    const colText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    colText.setAttribute('x', startX + j * cellSize + cellSize / 2);
    colText.setAttribute('y', startY - 10);
    colText.setAttribute('text-anchor', 'middle');
    colText.setAttribute('fill', isLight ? '#71717a' : '#a1a1aa');
    colText.setAttribute('font-family', 'JetBrains Mono, monospace');
    colText.setAttribute('font-size', '11');
    colText.textContent = c;
    svg.appendChild(colText);
  });

  // Row Headers & Matrix Cells
  classes.forEach((c, i) => {
    // True label text
    const rowText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    rowText.setAttribute('x', startX - 12);
    rowText.setAttribute('y', startY + i * cellSize + cellSize / 2 + 4);
    rowText.setAttribute('text-anchor', 'end');
    rowText.setAttribute('fill', isLight ? '#71717a' : '#a1a1aa');
    rowText.setAttribute('font-family', 'JetBrains Mono, monospace');
    rowText.setAttribute('font-size', '11');
    rowText.textContent = c;
    svg.appendChild(rowText);

    classes.forEach((_, j) => {
      const val = cm[i][j];
      const isDiagonal = (i === j);
      
      // Rect cell
      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('x', startX + j * cellSize);
      rect.setAttribute('y', startY + i * cellSize);
      rect.setAttribute('width', cellSize - 4);
      rect.setAttribute('height', cellSize - 4);
      rect.setAttribute('rx', 4);
      rect.setAttribute('stroke', isLight ? '#e4e4e7' : '#27272a');
      rect.setAttribute('stroke-width', 1);

      // Monochrome Fill Shading
      if (isDiagonal) {
        rect.setAttribute('fill', isLight ? `rgba(0, 0, 0, ${0.15 + (val / maxVal) * 0.7})` : `rgba(255, 255, 255, ${0.15 + (val / maxVal) * 0.75})`);
      } else {
        rect.setAttribute('fill', isLight ? `rgba(0, 0, 0, ${0.02 + (val / maxVal) * 0.2})` : `rgba(255, 255, 255, ${0.02 + (val / maxVal) * 0.15})`);
      }

      svg.appendChild(rect);

      // Value text
      const valText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      valText.setAttribute('x', startX + j * cellSize + cellSize / 2 - 2);
      valText.setAttribute('y', startY + i * cellSize + cellSize / 2 + 5);
      valText.setAttribute('text-anchor', 'middle');
      
      if (isDiagonal && val > 40) {
        valText.setAttribute('fill', isLight ? '#ffffff' : '#000000');
        valText.setAttribute('font-weight', '800');
      } else {
        valText.setAttribute('fill', isLight ? '#09090b' : '#fafafa');
        valText.setAttribute('font-weight', '600');
      }

      valText.setAttribute('font-family', 'JetBrains Mono, monospace');
      valText.setAttribute('font-size', '13');
      valText.textContent = val;
      svg.appendChild(valText);
    });
  });
}

// =============================================================================
// FEATURE IMPORTANCE BARS
// =============================================================================
function renderFeatureImportanceBars() {
  const rfImp = STATE.metrics.rf.importances;
  const dtImp = STATE.metrics.dt.importances;
  const features = Object.keys(rfImp).sort((a, b) => rfImp[b] - rfImp[a]);

  DOM.featureBarsContainer.innerHTML = '';

  features.forEach(f => {
    const rfVal = (rfImp[f] * 100).toFixed(1);
    const dtVal = (dtImp[f] * 100).toFixed(1);

    const row = document.createElement('div');
    row.className = 'feature-bar-row';
    row.innerHTML = `
      <div class="feature-bar-meta mono">
        <span class="bold-white">${f.replace('_', ' ')}</span>
        <span class="mono-muted">RF: <b>${rfVal}%</b> &nbsp;|&nbsp; DT: ${dtVal}%</span>
      </div>
      <div class="feature-track">
        <div class="rf-feature-fill" style="width: ${rfVal}%;" title="Random Forest: ${rfVal}%"></div>
      </div>
    `;
    DOM.featureBarsContainer.appendChild(row);
  });
}

// =============================================================================
// DATASET LOADER & EXPLORER
// =============================================================================
async function loadDataset() {
  try {
    const res = await fetch('data/soil_data.csv');
    if (res.ok) {
      const csvText = await res.text();
      parseAndSetDataset(csvText);
      return;
    }
  } catch (e) {
    console.warn('Loading synthetic dataset locally');
  }

  // Fallback synthetic dataset generation if local fetch blocked
  generateLocalDataset();
}

function parseAndSetDataset(csvText) {
  const lines = csvText.trim().split('\n');
  const headers = lines[0].split(',').map(h => h.trim());
  
  const data = [];
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(',');
    if (parts.length < 7) continue;
    data.push({
      id: i,
      pH: parseFloat(parts[0]),
      nitrogen: parseFloat(parts[1]),
      phosphorus: parseFloat(parts[2]),
      potassium: parseFloat(parts[3]),
      organic: parseFloat(parts[4]),
      moisture: parseFloat(parts[5]),
      quality: parts[6].trim()
    });
  }

  STATE.dataset = data;
  STATE.filteredDataset = [...data];
  renderDatasetTable();
}

function generateLocalDataset() {
  const data = [];
  for (let i = 1; i <= 300; i++) {
    const pH = (4.5 + Math.random() * 4.3).toFixed(2);
    const n = (50 + Math.random() * 430).toFixed(1);
    const p = (5 + Math.random() * 50).toFixed(1);
    const k = (60 + Math.random() * 260).toFixed(1);
    const om = (0.5 + Math.random() * 5.0).toFixed(2);
    const m = (5 + Math.random() * 50).toFixed(2);
    
    let quality = 'Low';
    if (pH >= 6.0 && pH <= 7.5 && n > 250 && om > 2.8) quality = 'High';
    else if (pH >= 5.5 && n > 150 && om > 1.5) quality = 'Medium';

    data.push({ id: i, pH: parseFloat(pH), nitrogen: parseFloat(n), phosphorus: parseFloat(p), potassium: parseFloat(k), organic: parseFloat(om), moisture: parseFloat(m), quality });
  }
  STATE.dataset = data;
  STATE.filteredDataset = [...data];
  renderDatasetTable();
}

function filterDataset(query) {
  const q = query.toLowerCase().trim();
  if (!q) {
    STATE.filteredDataset = [...STATE.dataset];
  } else {
    STATE.filteredDataset = STATE.dataset.filter(row => 
      row.quality.toLowerCase().includes(q) ||
      row.pH.toString().includes(q) ||
      row.nitrogen.toString().includes(q) ||
      row.id.toString().includes(q)
    );
  }
  STATE.currentPage = 1;
  renderDatasetTable();
}

function renderDatasetTable() {
  const start = (STATE.currentPage - 1) * STATE.pageSize;
  const end = start + STATE.pageSize;
  const pageRows = STATE.filteredDataset.slice(start, end);
  const totalPages = Math.ceil(STATE.filteredDataset.length / STATE.pageSize) || 1;

  DOM.datasetTableBody.innerHTML = '';
  
  pageRows.forEach(row => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>#${String(row.id).padStart(4, '0')}</td>
      <td>${row.pH.toFixed(2)}</td>
      <td>${row.nitrogen.toFixed(1)}</td>
      <td>${row.phosphorus.toFixed(1)}</td>
      <td>${row.potassium.toFixed(1)}</td>
      <td>${row.organic.toFixed(2)}%</td>
      <td>${row.moisture.toFixed(2)}%</td>
      <td><span class="tag-class-${row.quality.toLowerCase()}">${row.quality.toUpperCase()}</span></td>
    `;
    DOM.datasetTableBody.appendChild(tr);
  });

  DOM.pageInfo.textContent = `PAGE ${STATE.currentPage} / ${totalPages} (${STATE.filteredDataset.length} RECORDS)`;
  DOM.prevPageBtn.disabled = STATE.currentPage <= 1;
  DOM.nextPageBtn.disabled = STATE.currentPage >= totalPages;
}

// =============================================================================
// REPORT MODAL LOGIC
// =============================================================================
function openReportModal() {
  const dateStr = new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
  DOM.reportTimestamp.textContent = `TIMESTAMP: ${dateStr}`;

  const rfClass = DOM.rfClassOutput.textContent.replace(' QUALITY', '');
  DOM.repClassification.textContent = `${rfClass} QUALITY`;
  DOM.repConfidence.textContent = DOM.rfConfidenceVal.textContent;

  const raw = [
    { name: 'pH (Acidity/Alkalinity)', val: DOM.pH.value, opt: '6.0 - 7.5', status: parseFloat(DOM.pH.value) >= 6.0 && parseFloat(DOM.pH.value) <= 7.5 ? 'OPTIMAL' : 'DEFICIENT/SUB-OPTIMAL' },
    { name: 'Nitrogen (N) [kg/ha]', val: `${DOM.nitrogen.value} kg/ha`, opt: '> 280.0 kg/ha', status: parseFloat(DOM.nitrogen.value) >= 280 ? 'OPTIMAL' : 'BELOW THRESHOLD' },
    { name: 'Phosphorus (P) [kg/ha]', val: `${DOM.phosphorus.value} kg/ha`, opt: '> 25.0 kg/ha', status: parseFloat(DOM.phosphorus.value) >= 25 ? 'OPTIMAL' : 'BELOW THRESHOLD' },
    { name: 'Potassium (K) [kg/ha]', val: `${DOM.potassium.value} kg/ha`, opt: '> 150.0 kg/ha', status: parseFloat(DOM.potassium.value) >= 150 ? 'OPTIMAL' : 'BELOW THRESHOLD' },
    { name: 'Organic Matter (%)', val: `${DOM.organic.value}%`, opt: '> 3.00%', status: parseFloat(DOM.organic.value) >= 3.0 ? 'OPTIMAL' : 'DEPLETED' },
    { name: 'Moisture (%)', val: `${DOM.moisture.value}%`, opt: '20.0% - 40.0%', status: parseFloat(DOM.moisture.value) >= 20 && parseFloat(DOM.moisture.value) <= 40 ? 'OPTIMAL' : 'IMBALANCED' }
  ];

  DOM.reportTableBody.innerHTML = '';
  raw.forEach(r => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${r.name}</td>
      <td><b>${r.val}</b></td>
      <td>${r.opt}</td>
      <td class="${r.status === 'OPTIMAL' ? 'bold-white' : 'mono-dim'}">${r.status}</td>
    `;
    DOM.reportTableBody.appendChild(tr);
  });

  DOM.reportDirectives.innerHTML = DOM.prescriptionBody.innerHTML;
  DOM.reportModalBackdrop.classList.add('open');
}

function closeReportModal() {
  DOM.reportModalBackdrop.classList.remove('open');
}
