/**
 * app.js
 * Handles tab navigation, collapsible sections, form submission,
 * API communication, and Plotly chart rendering.
 */

'use strict';

// ===========================================================================
// i18n — client-side EN/KR translation
// ===========================================================================
//
// Every translatable element in index.html carries data-i18n="key" (text) or
// data-i18n-title="key" (the title attribute). Runtime-generated strings
// (results panel, loading state) are pulled from I18N[lang].rt.* below.
//
// Korean: professional 존댓말 (-습니다/-입니다). English abbreviations kept
// where Korean clinicians / ML practitioners use them (FPG, SHAP, HbA1c, BMI,
// AST/ALT/GGT, ROC-AUC, CatBoost). Terms flagged

const I18N = {
  en: {
    back_link: '← Portfolio',
    app_title: 'T2D Risk Screener',
    app_subtitle: 'Undiagnosed Type 2 Diabetes · Korean NHIS Data',
    tab_screener: 'Screening Tool',
    tab_about: 'About the Model',

    disclaimer_lead: 'Screening tool only.',
    disclaimer_body: 'This tool does not provide a diagnosis and is not a substitute for professional medical advice. Results are based on statistical patterns in Korean population data and are intended to support, not replace, clinical judgement.',

    sec_basic: 'Basic Information',
    lbl_age: 'Age',
    unit_years: '(years)',
    lbl_sex: 'Sex',
    opt_select: 'Select…',
    opt_male: 'Male',
    opt_female: 'Female',
    lbl_smoking: 'Smoking status',
    opt_smoke_never: 'Never smoked',
    opt_smoke_former: 'Former smoker',
    opt_smoke_current: 'Current smoker',
    lbl_alcohol: 'Alcohol consumption',
    opt_no: 'No',
    opt_yes: 'Yes',

    sec_body: 'Body Measurements',
    lbl_height: 'Height',
    lbl_weight: 'Weight',
    lbl_waist: 'Waist circumference',
    tip_waist: 'Measure at the midpoint between the lowest rib and the top of the hip bone.',

    sec_bp: 'Blood Pressure',
    lbl_systolic: 'Systolic',
    lbl_diastolic: 'Diastolic',

    sec_urine: 'Urinalysis',
    lbl_urine_protein: 'Urine protein',
    tip_urine: 'Dipstick result: 1=Negative, 2=Trace, 3=(+1), 4=(+2), 5=(+3), 6=(+4)',
    opt_urine_1: '1 — Negative',
    opt_urine_2: '2 — Trace',
    opt_urine_3: '3 — +1',
    opt_urine_4: '4 — +2',
    opt_urine_5: '5 — +3',
    opt_urine_6: '6 — +4',

    sec_labs: 'Additional Lab Values',
    badge_optional: 'Optional',
    hint_labs: 'Including these values improves prediction accuracy but is not required.',
    lbl_hemoglobin: 'Hemoglobin',
    lbl_creatinine: 'Serum creatinine',
    lbl_ast: 'AST (GOT)',
    lbl_alt: 'ALT (GPT)',
    lbl_ggt: 'GGT (γ-GTP)',

    sec_lipids: 'Lipid Panel',
    badge_lipids: 'Optional — ~34% of check-ups include this',
    hint_lipids: 'The model handles missing lipid values gracefully — it was trained on data where approximately 66% of records had no lipid panel results.',
    lbl_total_chol: 'Total cholesterol',
    lbl_triglycerides: 'Triglycerides',
    lbl_hdl: 'HDL cholesterol',
    lbl_ldl: 'LDL cholesterol',

    btn_calculate: 'Calculate risk',
    btn_reset: 'Reset',

    results_title: 'Screening Result',
    btn_new: 'New assessment',
    nolabs_lead: '⚠ No laboratory values were provided.',
    nolabs_body: 'This model is significantly less accurate without lab data. Missing lab values are treated as a signal in themselves — learned from patients in the training data who skipped those tests — and can underestimate risk even when anthropometric factors (BMI, waist circumference) suggest otherwise. For a meaningful result, please provide at least hemoglobin, serum creatinine, and one liver enzyme (AST, ALT, or GGT).',
    why_title: 'Why this result?',
    why_hint: 'The chart below shows which of your measurements most influenced this prediction. Red bars increased your risk score; blue bars decreased it. Values are shown on a log-odds scale used internally by the model.',
    shap_note: "SHAP values reflect the contribution of each input to the model's raw score, not to the final probability directly. Calibrated probability is derived from the CatBoost model's raw output via Platt scaling.",

    about_h_background: 'Background',
    about_p_background: 'Type 2 diabetes is a significant and growing health burden in South Korea. A substantial proportion of people living with T2D are unaware of their diagnosis because the condition develops gradually and is often asymptomatic in its early stages. Early identification allows for lifestyle interventions that can delay or prevent progression to overt diabetes.',
    about_h_data: 'Data',
    about_p_data1_pre: 'This model was trained on the',
    about_p_data1_link: 'Korean National Health Insurance Service (NHIS) General Health Examination dataset',
    about_p_data1_post: ' — one million randomly selected adult subscribers who received a health check-up in 2024. The dataset is representative of the general insured adult population in South Korea.',
    about_p_data2: 'Fasting plasma glucose (FPG) was used to construct the binary label: records with FPG ≥ 126 mg/dL were labelled as likely undiagnosed T2D. Patients whose records were flagged as likely known diabetics (on treatment) were not removed, as the NHIS dataset does not include treatment history; this is a known limitation.',
    about_p_data3: "Crucially, FPG itself was excluded from all model features. Because FPG directly measures blood glucose, including it would make the model nearly circular — essentially just applying a threshold to the answer it is supposed to predict. Instead, the model learns from indirect metabolic indicators: body composition, blood pressure, kidney function, liver enzymes, and lipid levels. This defines the tool's intended role: identify which patients are likely to benefit from a fasting glucose test, not to replace one. A positive screen should prompt an FPG or HbA1c measurement for confirmation.",
    about_h_model: 'Model',
    about_p_model1: 'A CatBoost gradient-boosted decision tree was selected after comparing against a PyTorch tabular network and an sklearn MLP classifier. CatBoost was chosen for its strong performance, native support for categorical features without preprocessing, built-in missing-value handling, and CPU efficiency — making it practical to deploy without GPU infrastructure.',
    about_p_model2: 'Raw model probabilities were calibrated using Platt scaling (logistic regression on validation-set logits). The screening threshold was selected to achieve a sensitivity (recall) of at least 85% on the validation set, reflecting the asymmetric cost of a missed case versus a false positive in a population screening context.',
    about_h_perf: 'Performance (test set)',
    metric_col_metric: 'Metric',
    metric_col_value: 'Value',
    metric_ap: 'Average precision',
    metric_brier: 'Brier score',
    about_h_importance: 'Feature importance',
    about_p_importance: "The chart below shows the mean absolute SHAP value for each feature across a 5,000-row sample of the held-out test set — a measure of how much, on average, each feature contributes to the model's predictions.",
    about_h_limits: 'Limitations',
    about_li_1: 'The model identifies statistical risk patterns and does not diagnose diabetes. A fasting plasma glucose test or HbA1c measurement is required for diagnosis.',
    about_li_2: 'Training data is from South Korea (2024 cohort). Generalisability to other populations may be limited.',
    about_li_3: "In the NHIS dataset, laboratory tests (hemoglobin, creatinine, liver enzymes, lipids) were ordered at a clinician's discretion, so patients who had them measured tended to be sicker on average. The model learned this pattern: absent lab values act as a weak signal of lower risk. As a result, predictions without any laboratory input may substantially underestimate risk in patients who simply haven't been tested.",
    about_li_4: 'The FPG-based label does not distinguish between newly detected and previously known (but poorly controlled) diabetes.',
    about_li_5: 'This is a research portfolio project and has not been clinically validated.',
    about_h_source: 'Source code',
    about_p_source_pre: 'All code, training scripts, and the analysis notebook are available on',
    about_p_source_post: '.',

    footer_pre: 'Built by Tyson Johnson · George Mason University, Computational & Data Sciences · Data:',
    footer_data: 'Korean NHIS 2024',

    // Runtime-generated strings (results panel & loading state)
    rt: {
      calculating: 'Calculating…',
      screen_positive: 'Screen positive (above threshold)',
      screen_negative: 'Screen negative (below threshold)',
      probability: (pct) => `Estimated risk probability: ${pct}%`,
      validation_missing: (fields) => `Please fill in all required fields: ${fields}.`,
      validation_bp: 'Systolic blood pressure must be greater than diastolic.',
      error_server: (status) => `Server error (${status})`,
      error_generic: (msg) => `Something went wrong: ${msg}`,
      loading: 'Loading…',
      importance_fail: (msg) => `Could not load importance chart: ${msg}`,
    },
  },

  ko: {
    back_link: '← 포트폴리오',
    app_title: '제2형 당뇨병 위험 선별 도구',
    app_subtitle: '미진단 제2형 당뇨병 · 한국 NHIS(국민건강보험공단) 데이터',
    tab_screener: '선별 도구',
    tab_about: '모델 소개',

    disclaimer_lead: '선별 목적 전용 도구입니다.',
    disclaimer_body: '본 도구는 진단을 제공하지 않으며 전문적인 의학적 조언을 대체할 수 없습니다. 결과는 한국 인구 데이터의 통계적 패턴에 기반하며, 임상적 판단을 대체하는 것이 아니라 보조하기 위한 것입니다.',

    sec_basic: '기본 정보',
    lbl_age: '나이',
    unit_years: '(세)',
    lbl_sex: '성별',
    opt_select: '선택…',
    opt_male: '남성',
    opt_female: '여성',
    lbl_smoking: '흡연 상태',
    opt_smoke_never: '비흡연',
    opt_smoke_former: '과거 흡연',
    opt_smoke_current: '현재 흡연',
    lbl_alcohol: '음주 여부',
    opt_no: '아니요',
    opt_yes: '예',

    sec_body: '신체 계측',
    lbl_height: '키',
    lbl_weight: '체중',
    lbl_waist: '허리둘레',
    tip_waist: '가장 아래쪽 갈비뼈와 골반뼈 상단의 중간 지점에서 측정하십시오.',

    sec_bp: '혈압',
    lbl_systolic: '수축기',
    lbl_diastolic: '이완기',

    sec_urine: '소변 검사',
    lbl_urine_protein: '요단백',
    tip_urine: '시험지봉 결과: 1=음성, 2=미량, 3=(+1), 4=(+2), 5=(+3), 6=(+4)',
    opt_urine_1: '1 — 음성',
    opt_urine_2: '2 — 미량',
    opt_urine_3: '3 — +1',
    opt_urine_4: '4 — +2',
    opt_urine_5: '5 — +3',
    opt_urine_6: '6 — +4',

    sec_labs: '추가 검사 수치',
    badge_optional: '선택 사항',
    hint_labs: '이 수치들을 입력하면 예측 정확도가 향상되지만 필수는 아닙니다.',
    lbl_hemoglobin: '헤모글로빈',
    lbl_creatinine: '혈청 크레아티닌',
    lbl_ast: 'AST (GOT)',
    lbl_alt: 'ALT (GPT)',
    lbl_ggt: 'GGT (γ-GTP)',

    sec_lipids: '지질 검사',
    badge_lipids: '선택 사항 — 검진의 약 34%만 포함',
    hint_lipids: '본 모델은 누락된 지질 수치를 무리 없이 처리합니다. 학습 데이터의 약 66%가 지질 검사 결과 없이 구성되어 있었습니다.',
    lbl_total_chol: '총콜레스테롤',
    lbl_triglycerides: '중성지방',
    lbl_hdl: 'HDL 콜레스테롤',
    lbl_ldl: 'LDL 콜레스테롤',

    btn_calculate: '위험도 계산',
    btn_reset: '초기화',

    results_title: '선별 결과',
    btn_new: '새로 평가하기',
    nolabs_lead: '⚠ 검사 수치가 입력되지 않았습니다.',
    nolabs_body: '본 모델은 검사 데이터가 없으면 정확도가 상당히 떨어집니다. 누락된 검사 수치 자체가 하나의 신호로 처리되는데, 이는 학습 데이터에서 해당 검사를 받지 않은 환자들로부터 학습된 것입니다. 따라서 신체 계측 요인(BMI, 허리둘레)이 위험을 시사하더라도 위험도를 과소평가할 수 있습니다. 유의미한 결과를 얻으려면 최소한 헤모글로빈, 혈청 크레아티닌, 그리고 간 효소(AST, ALT 또는 GGT) 중 하나 이상을 입력해 주십시오.',
    why_title: '이 결과가 나온 이유는?',
    why_hint: '아래 차트는 입력하신 수치 중 어떤 것이 이 예측에 가장 큰 영향을 주었는지 보여줍니다. 빨간색 막대는 위험 점수를 높였고, 파란색 막대는 낮췄습니다. 값은 모델 내부에서 사용하는 로그-오즈(log-odds) 척도로 표시됩니다.',
    shap_note: 'SHAP 값은 각 입력값이 최종 확률에 직접 기여한 정도가 아니라 모델의 원시 점수(raw score)에 기여한 정도를 나타냅니다. 보정된 확률은 CatBoost 모델의 원시 출력값에 Platt 스케일링을 적용하여 산출됩니다.',

    about_h_background: '배경',
    about_p_background: '제2형 당뇨병은 한국에서 심각하고 지속적으로 증가하는 보건 부담입니다. 제2형 당뇨병을 가진 사람들 중 상당수는 자신의 진단 사실을 인지하지 못하는데, 이는 질환이 서서히 진행되고 초기 단계에서는 무증상인 경우가 많기 때문입니다. 조기 발견은 명백한 당뇨병으로의 진행을 늦추거나 예방할 수 있는 생활습관 개선의 기회를 제공합니다.',
    about_h_data: '데이터',
    about_p_data1_pre: '본 모델은',
    about_p_data1_link: '한국 국민건강보험공단(NHIS) 일반건강검진 데이터셋',
    about_p_data1_post: '으로 학습되었습니다. 이는 2024년에 건강검진을 받은 성인 가입자 100만 명을 무작위로 추출한 것으로, 한국의 일반 성인 가입자 인구를 대표합니다.',
    about_p_data2: '공복혈당(FPG)을 사용하여 이진 라벨을 구성하였으며, FPG ≥ 126 mg/dL인 경우 미진단 제2형 당뇨병 가능성이 높은 것으로 표시하였습니다. 이미 알려진(치료 중인) 당뇨병 환자로 추정되는 기록은 제거하지 않았는데, NHIS 데이터셋에 치료 이력이 포함되어 있지 않기 때문입니다. 이는 알려진 한계점입니다.',
    about_p_data3: '중요한 점은 FPG 자체를 모든 모델 변수에서 제외했다는 것입니다. FPG는 혈당을 직접 측정하므로 이를 포함하면 모델이 거의 순환 논리에 빠지게 됩니다 — 즉, 예측해야 할 정답에 단순히 임계값을 적용하는 것과 다름없게 됩니다. 대신 모델은 간접적인 대사 지표인 체성분, 혈압, 신장 기능, 간 효소, 지질 수치로부터 학습합니다. 이는 본 도구의 의도된 역할을 정의합니다 — 공복혈당 검사를 대체하는 것이 아니라, 어떤 환자가 해당 검사를 통해 이익을 얻을 가능성이 높은지를 식별하는 것입니다. 선별 양성 결과는 확진을 위해 FPG 또는 HbA1c 측정을 시행하는 계기가 되어야 합니다.',
    about_h_model: '모델',
    about_p_model1: 'PyTorch 표 형식(tabular) 신경망 및 sklearn MLP 분류기와 비교한 끝에 CatBoost 그래디언트 부스팅 의사결정나무를 선택하였습니다. CatBoost는 우수한 성능, 전처리 없이 범주형 변수를 기본 지원하는 점, 내장된 결측치 처리, 그리고 CPU 효율성 덕분에 선택되었으며, 이로써 GPU 인프라 없이도 실용적으로 배포할 수 있습니다.',
    about_p_model2: '모델의 원시 확률은 Platt 스케일링(검증 세트 로짓에 대한 로지스틱 회귀)을 사용하여 보정되었습니다. 선별 임계값은 검증 세트에서 최소 85%의 민감도(재현율)를 달성하도록 선택되었으며, 이는 인구 선별 상황에서 놓친 사례와 위양성 사이의 비대칭적 비용을 반영한 것입니다.',
    about_h_perf: '성능 (테스트 세트)',
    metric_col_metric: '지표',
    metric_col_value: '값',
    metric_ap: '평균 정밀도(Average precision)',
    metric_brier: 'Brier 점수',
    about_h_importance: '변수 중요도',
    about_p_importance: '아래 차트는 보류된 테스트 세트에서 추출한 5,000개 행 표본에 대해 각 변수의 평균 절대 SHAP 값을 보여줍니다. 이는 각 변수가 모델의 예측에 평균적으로 얼마나 기여하는지를 나타내는 척도입니다.',
    about_h_limits: '한계점',
    about_li_1: '본 모델은 통계적 위험 패턴을 식별할 뿐 당뇨병을 진단하지 않습니다. 진단을 위해서는 공복혈당 검사 또는 HbA1c 측정이 필요합니다.',
    about_li_2: '학습 데이터는 한국(2024년 코호트)의 것입니다. 다른 인구 집단에 대한 일반화 가능성은 제한적일 수 있습니다.',
    about_li_3: 'NHIS 데이터셋에서 검사(헤모글로빈, 크레아티닌, 간 효소, 지질)는 임상의의 재량으로 시행되었기 때문에, 검사를 받은 환자들은 평균적으로 더 건강이 좋지 않은 경향이 있었습니다. 모델은 이 패턴을 학습하여, 검사 수치의 부재를 위험이 더 낮다는 약한 신호로 받아들입니다. 그 결과, 검사 입력값이 전혀 없는 예측은 단지 검사를 받지 않았을 뿐인 환자의 위험을 상당히 과소평가할 수 있습니다.',
    about_li_4: 'FPG 기반 라벨은 새로 발견된 당뇨병과 기존에 알려진(그러나 잘 관리되지 않는) 당뇨병을 구분하지 못합니다.',
    about_li_5: '본 프로젝트는 연구용 포트폴리오 프로젝트이며 임상적으로 검증되지 않았습니다.',
    about_h_source: '소스 코드',
    about_p_source_pre: '모든 코드, 학습 스크립트, 분석 노트북은 다음에서 확인하실 수 있습니다:',
    about_p_source_post: '.',

    footer_pre: '제작: Tyson Johnson · George Mason University 전산·데이터과학 · 데이터:',
    footer_data: '한국 NHIS 2024',

    // Runtime-generated strings (results panel & loading state)
    rt: {
      calculating: '계산 중…',
      screen_positive: '선별 양성 (임계값 초과)',
      screen_negative: '선별 음성 (임계값 미만)',
      probability: (pct) => `추정 위험 확률: ${pct}%`,
      validation_missing: (fields) => `필수 항목을 모두 입력해 주세요: ${fields}.`,
      validation_bp: '수축기 혈압은 이완기 혈압보다 커야 합니다.',
      error_server: (status) => `서버 오류 (${status})`,
      error_generic: (msg) => `오류가 발생했습니다: ${msg}`,
      loading: '불러오는 중…',
      importance_fail: (msg) => `중요도 차트를 불러오지 못했습니다: ${msg}`,
    },
  },
};

// Korean labels for required-field validation messages (keyed by field name).
const FIELD_LABELS = {
  en: {
    age: 'Age', sex_code: 'Sex', systolic_bp: 'Systolic',
    diastolic_bp: 'Diastolic', urine_protein: 'Urine protein',
    smoking_status: 'Smoking status', alcohol_consumption: 'Alcohol consumption',
    height_cm: 'Height', weight_kg: 'Weight', waist_cm: 'Waist circumference',
  },
  ko: {
    age: '나이', sex_code: '성별', systolic_bp: '수축기혈압',
    diastolic_bp: '이완기혈압', urine_protein: '요단백',
    smoking_status: '흡연 상태', alcohol_consumption: '음주 여부',
    height_cm: '키', weight_kg: '체중', waist_cm: '허리둘레',
  },
};

let currentLang = 'en';

function t(key) {
  const dict = I18N[currentLang] || I18N.en;
  return (key in dict) ? dict[key] : (I18N.en[key] !== undefined ? I18N.en[key] : key);
}

function rt() {
  return (I18N[currentLang] || I18N.en).rt;
}

function getCookie(name) {
  const m = document.cookie.match('(?:^|; )' + name + '=([^;]*)');
  return m ? decodeURIComponent(m[1]) : null;
}

function setLangCookie(lang) {
  document.cookie = `lang=${lang};path=/;max-age=31536000;samesite=lax`;
}

function applyLanguage(lang) {
  if (lang !== 'en' && lang !== 'ko') lang = 'en';
  currentLang = lang;
  document.documentElement.lang = lang;

  // Text content
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const val = t(el.getAttribute('data-i18n'));
    if (typeof val === 'string') el.textContent = val;
  });
  // Title attributes (tooltips)
  document.querySelectorAll('[data-i18n-title]').forEach(el => {
    const val = t(el.getAttribute('data-i18n-title'));
    if (typeof val === 'string') el.setAttribute('title', val);
  });

  // Slide the language switch to the active side
  const sw = document.querySelector('.lang-switch');
  if (sw) sw.dataset.lang = lang;

  // Re-render any visible runtime strings so they follow the language.
  refreshDynamicStrings();
}

// Re-applies runtime strings that app.js writes imperatively, so a live
// language switch updates the results panel and the submit button too.
function refreshDynamicStrings() {
  // Submit button: restore its label unless mid-calculation.
  if (!submitBtn.disabled) {
    submitBtn.textContent = t('btn_calculate');
  } else {
    submitBtn.innerHTML = `<span class="spinner"></span>${rt().calculating}`;
  }

  // Results panel — only if currently shown.
  if (lastResult && !resultsPanel.hasAttribute('hidden')) {
    renderResultStrings(lastResult);
  }
}

function setupLanguageToggle() {
  document.querySelectorAll('.lang-switch button').forEach(btn => {
    btn.addEventListener('click', () => {
      const lang = btn.dataset.lang;
      setLangCookie(lang);
      applyLanguage(lang);
    });
  });
}

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

// Ordered list of every required field (matches the form section order).
const REQUIRED_FIELDS = [
  'age', 'sex_code', 'smoking_status', 'alcohol_consumption',
  'height_cm', 'weight_kg', 'waist_cm',
  'systolic_bp', 'diastolic_bp', 'urine_protein',
];

// Remove the .field-error highlight from every required input/select.
function clearFieldErrors() {
  REQUIRED_FIELDS.forEach(name => {
    const el = form.elements[name];
    if (el) el.classList.remove('field-error');
  });
}

// Add the .field-error highlight to the listed fields.
function markFieldErrors(names) {
  names.forEach(name => {
    const el = form.elements[name];
    if (el) el.classList.add('field-error');
  });
}

/**
 * Basic client-side validation. Returns { message, fields } where `fields`
 * is the list of every missing required field name (empty when valid).
 */
function validate(payload) {
  const missing = REQUIRED_FIELDS.filter(
    f => payload[f] == null || isNaN(payload[f])
  );

  if (missing.length) {
    const labels = (FIELD_LABELS[currentLang] || FIELD_LABELS.en);
    const list = missing
      .map(f => labels[f] || f.replace(/_/g, ' '))
      .join(', ');
    return { message: rt().validation_missing(list), fields: missing };
  }

  if (payload.systolic_bp <= payload.diastolic_bp) {
    return { message: rt().validation_bp, fields: ['systolic_bp', 'diastolic_bp'] };
  }

  return { message: null, fields: [] };
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  formError.setAttribute('hidden', '');

  // Clear any prior field highlights at the start of every submit attempt.
  clearFieldErrors();

  const payload = buildPayload();
  const validation = validate(payload);
  if (validation.message) {
    markFieldErrors(validation.fields);
    formError.textContent = validation.message;
    formError.removeAttribute('hidden');
    return;
  }

  setLoading(true);

  try {
    const response = await fetch('predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || rt().error_server(response.status));
    }

    const result = await response.json();
    displayResults(result, payload);

  } catch (err) {
    formError.textContent = rt().error_generic(err.message);
    formError.removeAttribute('hidden');
  } finally {
    setLoading(false);
  }
});

resetBtn.addEventListener('click', () => {
  form.reset();
  clearFieldErrors();
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
    submitBtn.innerHTML = `<span class="spinner"></span>${rt().calculating}`;
  } else {
    submitBtn.disabled = false;
    submitBtn.textContent = t('btn_calculate');
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

// Holds the most recent API result so a live language switch can re-render
// the localized portions of the results panel.
let lastResult = null;

// Sets the localized text inside the tier card (screen status + probability
// prefix). The tier name and recommendation text are server-provided and
// shown as returned. Safe to call repeatedly (used on language switch).
function renderResultStrings(result) {
  // Tier badge
  const badge = document.getElementById('tier-badge');
  badge.textContent = result.risk_tier;
  badge.style.background = result.risk_tier_color;

  // Screen status
  const status = document.getElementById('screen-status');
  status.textContent = result.screen_positive
    ? rt().screen_positive
    : rt().screen_negative;

  // Probability
  const pct = (result.probability * 100).toFixed(1);
  document.getElementById('probability-display').textContent = rt().probability(pct);

  // Recommendation (server-provided)
  document.getElementById('recommendation-text').textContent = result.recommendation;

  // Tier card border colour
  document.getElementById('tier-card').style.borderColor = result.risk_tier_color;
}

function displayResults(result, payload) {
  lastResult = result;

  // No-labs warning
  const noLabsWarning = document.getElementById('no-labs-warning');
  if (hasNoLabValues(payload)) {
    noLabsWarning.removeAttribute('hidden');
  } else {
    noLabsWarning.setAttribute('hidden', '');
  }

  // Localized tier card strings
  renderResultStrings(result);

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
  container.textContent = rt().loading;

  try {
    const res = await fetch('global-importance');
    if (!res.ok) throw new Error(`Status ${res.status}`);
    const data = await res.json();
    const fig  = data.chart;
    container.textContent = '';
    Plotly.react('global-importance-chart', fig.data, fig.layout, { responsive: true });
    globalChartLoaded = true;
  } catch (err) {
    container.textContent = rt().importance_fail(err.message);
  }
}

let metricsLoaded = false;

async function loadMetrics() {
  if (metricsLoaded) return;

  try {
    const res = await fetch('metadata');
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

// ---------------------------------------------------------------------------
// Initialize language: read `lang` cookie (set by the main portfolio),
// default to 'en'. Then wire up the toggle. This runs after all the elements
// and helper variables above are defined.
// ---------------------------------------------------------------------------

(function initLanguage() {
  const cookieLang = getCookie('lang');
  const lang = (cookieLang === 'en' || cookieLang === 'ko') ? cookieLang : 'en';
  setupLanguageToggle();
  applyLanguage(lang);
})();
