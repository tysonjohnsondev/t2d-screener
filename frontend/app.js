/**
 * app.js
 * Handles tab navigation, collapsible sections, form submission,
 * API communication, and Plotly chart rendering.
 */

'use strict';

// ---------------------------------------------------------------------------
// Tab navigation
// ---------------------------------------------------------------------------

const navBtns     = document.querySelectorAll('.nav-btn');
const tabContents = document.querySelectorAll('.tab-content');

navBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;

    navBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    tabContents.forEach(tab => {
      if (tab.id === `tab-${target}`) {
        tab.removeAttribute('hidden');
      } else {
        tab.setAttribute('hidden', '');
      }
    });

    // Lazy-load the global importance chart when About tab is first opened
    if (target === 'about') {
      loadGlobalImportance();
      loadMetrics();
    }
  });
});

// ---------------------------------------------------------------------------
// Collapsible sections
// ---------------------------------------------------------------------------

document.querySelectorAll('.section-toggle').forEach(toggle => {
  toggle.addEventListener('click', () => {
    const expanded = toggle.getAttribute('aria-expanded') === 'true';
    toggle.setAttribute('aria-expanded', String(!expanded));
    const content = toggle.nextElementSibling;
    if (expanded) {
      content.setAttribute('hidden', '');
    } else {
      content.removeAttribute('hidden');
    }
  });
});

// ---------------------------------------------------------------------------
// Form submission
// ---------------------------------------------------------------------------

const form          = document.getElementById('screening-form');
const submitBtn     = document.getElementById('submit-btn');
const resetBtn      = document.getElementById('reset-btn');
const formError     = document.getElementById('form-error');
const resultsPanel  = document.getElementById('results-panel');

/**
 * Parse a form field to a number, or null if empty / non-numeric.
 */
function parseOptionalNumber(value) {
  if (value === '' || value === null || value === undefined) return null;
  const n = Number(value);
  return isNaN(n) ? null : n;
}

/**
 * Build the JSON payload from the form values.
 */
function buildPayload() {
  const fd = new FormData(form);

  // Required integer fields
  const required = {
    age:                 parseInt(fd.get('age'), 10),
    sex_code:            parseInt(fd.get('sex_code'), 10),
    systolic_bp:         parseInt(fd.get('systolic_bp'), 10),
    diastolic_bp:        parseInt(fd.get('diastolic_bp'), 10),
    urine_protein:       parseInt(fd.get('urine_protein'), 10),
    smoking_status:      parseInt(fd.get('smoking_status'), 10),
    alcohol_consumption: parseInt(fd.get('alcohol_consumption'), 10),
  };

  // Required float fields
  const requiredF = {
    height_cm:  parseFloat(fd.get('height_cm')),
    weight_kg:  parseFloat(fd.get('weight_kg')),
    waist_cm:   parseFloat(fd.get('waist_cm')),
  };

  // Optional fields — sent as null when empty
  const optional = {
    hemoglobin:         parseOptionalNumber(fd.get('hemoglobin')),
    serum_creatinine:   parseOptionalNumber(fd.get('serum_creatinine')),
    ast:                parseOptionalNumber(fd.get('ast')),
    alt:                parseOptionalNumber(fd.get('alt')),
    ggt:                parseOptionalNumber(fd.get('ggt')),
    total_cholesterol:  parseOptionalNumber(fd.get('total_cholesterol')),
    triglycerides:      parseOptionalNumber(fd.get('triglycerides')),
    hdl_cholesterol:    parseOptionalNumber(fd.get('hdl_cholesterol')),
    ldl_cholesterol:    parseOptionalNumber(fd.get('ldl_cholesterol')),
  };

  return { ...required, ...requiredF, ...optional };
}

/**
 * Basic client-side validation — returns an error message or null.
 */
function validate(payload) {
  const intFields = [
    'age', 'sex_code', 'systolic_bp', 'diastolic_bp',
    'urine_protein', 'smoking_status', 'alcohol_consumption',
  ];
  const floatFields = ['height_cm', 'weight_kg', 'waist_cm'];

  for (const f of [...intFields, ...floatFields]) {
    if (payload[f] == null || isNaN(payload[f])) {
      return `Please fill in all required fields (missing: ${f.replace(/_/g, ' ')}).`;
    }
  }

  if (payload.systolic_bp <= payload.diastolic_bp) {
    return 'Systolic blood pressure must be greater than diastolic.';
  }

  return null;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  formError.setAttribute('hidden', '');

  const payload = buildPayload();
  const validationError = validate(payload);
  if (validationError) {
    formError.textContent = validationError;
    formError.removeAttribute('hidden');
    return;
  }

  setLoading(true);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Server error (${response.status})`);
    }

    const result = await response.json();
    displayResults(result, payload);

  } catch (err) {
    formError.textContent = `Something went wrong: ${err.message}`;
    formError.removeAttribute('hidden');
  } finally {
    setLoading(false);
  }
});

resetBtn.addEventListener('click', () => {
  form.reset();
  resultsPanel.setAttribute('hidden', '');
  formError.setAttribute('hidden', '');
  form.scrollIntoView({ behavior: 'smooth', block: 'start' });
});

document.getElementById('new-assessment-btn').addEventListener('click', () => {
  resultsPanel.setAttribute('hidden', '');
  form.scrollIntoView({ behavior: 'smooth', block: 'start' });
});

// ---------------------------------------------------------------------------
// Loading state
// ---------------------------------------------------------------------------

function setLoading(loading) {
  if (loading) {
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner"></span>Calculating…';
  } else {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Calculate risk';
  }
}

// ---------------------------------------------------------------------------
// Result display
// ---------------------------------------------------------------------------

function hasNoLabValues(payload) {
  const labFields = [
    'hemoglobin', 'serum_creatinine', 'ast', 'alt',
    'ggt', 'total_cholesterol', 'triglycerides', 'hdl_cholesterol', 'ldl_cholesterol',
  ];
  return labFields.every(f => payload[f] == null);
}

function displayResults(result, payload) {
  // No-labs warning
  const noLabsWarning = document.getElementById('no-labs-warning');
  if (hasNoLabValues(payload)) {
    noLabsWarning.removeAttribute('hidden');
  } else {
    noLabsWarning.setAttribute('hidden', '');
  }

  // Tier badge
  const badge = document.getElementById('tier-badge');
  badge.textContent = result.risk_tier;
  badge.style.background = result.risk_tier_color;

  // Screen status
  const status = document.getElementById('screen-status');
  status.textContent = result.screen_positive
    ? 'Screen positive (above threshold)'
    : 'Screen negative (below threshold)';

  // Probability
  const pct = (result.probability * 100).toFixed(1);
  document.getElementById('probability-display').textContent =
    `Estimated risk probability: ${pct}%`;

  // Recommendation
  document.getElementById('recommendation-text').textContent =
    result.recommendation;

  // Tier card border colour
  document.getElementById('tier-card').style.borderColor = result.risk_tier_color;

  // Waterfall chart
  const fig = result.waterfall_chart;
  Plotly.react('waterfall-chart', fig.data, fig.layout, { responsive: true });

  // Show results, scroll into view
  resultsPanel.removeAttribute('hidden');
  resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---------------------------------------------------------------------------
// About page — global importance chart & metrics
// ---------------------------------------------------------------------------

let globalChartLoaded = false;

async function loadGlobalImportance() {
  if (globalChartLoaded) return;

  const container = document.getElementById('global-importance-chart');
  container.textContent = 'Loading…';

  try {
    const res = await fetch('/global-importance');
    if (!res.ok) throw new Error(`Status ${res.status}`);
    const data = await res.json();
    const fig  = data.chart;
    container.textContent = '';
    Plotly.react('global-importance-chart', fig.data, fig.layout, { responsive: true });
    globalChartLoaded = true;
  } catch (err) {
    container.textContent = `Could not load importance chart: ${err.message}`;
  }
}

let metricsLoaded = false;

async function loadMetrics() {
  if (metricsLoaded) return;

  try {
    const res = await fetch('/metadata');
    if (!res.ok) return;
    const data = await res.json();

    const fmt = (v, digits = 4) =>
      v != null ? Number(v).toFixed(digits) : '—';

    document.getElementById('metric-roc').textContent   = fmt(data.test_roc_auc);
    document.getElementById('metric-ap').textContent    = fmt(data.test_avg_precision);
    document.getElementById('metric-brier').textContent = fmt(data.test_brier);

    metricsLoaded = true;
  } catch (_) {
    // Fail silently — metrics stay as '—'
  }
}
