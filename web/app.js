/**
 * CardioLAC-SSM™ Clinical Workstation JavaScript Engine
 * Real-time 12-Lead ECG Canvas Renderer, Incomplete Lead Simulator,
 * Live PyTorch Inference Dispatcher, Dual Uncertainty Dials, and Visualizer Gallery.
 */

// Global Application State
const STATE = {
    currentCaseIndex: 3,
    activeLeads: new Set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    leadNames: ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
    signalData: null,
    curatedCases: [],
    viewMode: 'multi', // 'multi' or 'rhythm'
    isSweeping: false,
    sweepProgress: 0,
    sweepAnimId: null,
    leaderboardData: [],
    sortColumn: 'Accuracy (%)',
    sortAsc: false,
    galleryData: null
};

// DOM References
const DOM = {
    // Tabs
    tabs: document.querySelectorAll('.nav-tab'),
    panels: document.querySelectorAll('.tab-panel'),

    // Patient selection
    caseChipsContainer: document.getElementById('case-chips-container'),
    directCaseInput: document.getElementById('direct-case-input'),
    btnLoadDirect: document.getElementById('btn-load-direct'),
    btnRandomPatient: document.getElementById('btn-random-patient'),
    metaPid: document.getElementById('meta-pid'),
    metaDemo: document.getElementById('meta-demo'),
    metaHr: document.getElementById('meta-hr'),
    metaGtBadges: document.getElementById('meta-gt-badges'),

    // Lead simulator
    activeLeadCounter: document.getElementById('active-lead-counter'),
    leadReductionPct: document.getElementById('lead-reduction-pct'),
    leadChips: document.querySelectorAll('.lead-chip'),
    btnRunInference: document.getElementById('btn-run-inference'),
    inferenceSpinner: document.getElementById('inference-spinner'),

    // Presets
    presetAll12: document.getElementById('preset-all-12'),
    presetLimbOnly: document.getElementById('preset-limb-only'),
    presetPrecordialOnly: document.getElementById('preset-precordial-only'),
    presetEinthoven: document.getElementById('preset-einthoven'),
    presetHolter: document.getElementById('preset-holter'),
    presetSingleLead: document.getElementById('preset-single-lead'),
    presetRandomDropout: document.getElementById('preset-random-dropout'),

    // Canvas
    canvas: document.getElementById('ecgCanvas'),
    canvasViewport: document.getElementById('canvas-viewport'),
    sweepCursor: document.getElementById('sweep-cursor'),
    canvasTooltip: document.getElementById('canvas-tooltip'),
    btnViewMulti: document.getElementById('btn-view-multi'),
    btnViewRhythm: document.getElementById('btn-view-rhythm'),
    btnPlaySweep: document.getElementById('btn-play-sweep'),
    btnResetZoom: document.getElementById('btn-reset-zoom'),

    // Predictions & Triage
    predictionsList: document.getElementById('predictions-list'),
    triageBanner: document.getElementById('triage-banner'),
    triageBannerIcon: document.getElementById('triage-banner-icon'),
    triageBannerTitle: document.getElementById('triage-banner-title'),
    triageBannerDesc: document.getElementById('triage-banner-desc'),
    triageStatusPill: document.getElementById('triage-status-pill'),
    epistemicVal: document.getElementById('epistemic-val'),
    epistemicTag: document.getElementById('epistemic-tag'),
    epistemicFill: document.getElementById('epistemic-fill'),
    aleatoricVal: document.getElementById('aleatoric-val'),
    aleatoricTag: document.getElementById('aleatoric-tag'),
    aleatoricFill: document.getElementById('aleatoric-fill'),
    leadAttentionBars: document.getElementById('lead-attention-bars'),
    resLatency: document.getElementById('res-latency'),
    resConfidence: document.getElementById('res-confidence'),
    teleLatency: document.getElementById('tele-latency'),

    // Benchmark Tab
    leaderboardSearch: document.getElementById('leaderboard-search'),
    leaderboardFamilyFilter: document.getElementById('leaderboard-family-filter'),
    benchmarkTbody: document.getElementById('benchmark-tbody'),

    // Gallery Tab
    galleryModelSelect: document.getElementById('gallery-model-select'),
    displayCmModelName: document.getElementById('display-cm-model-name'),
    displayCmImg: document.getElementById('display-cm-img'),
    displayRocModelName: document.getElementById('display-roc-model-name'),
    displayRocImg: document.getElementById('display-roc-img'),
    masterFiguresGrid: document.getElementById('master-figures-grid'),

    // Lightbox
    lightboxModal: document.getElementById('lightbox-modal'),
    lightboxImg: document.getElementById('lightbox-img'),
    lightboxCaption: document.getElementById('lightbox-caption')
};

// -------------------------------------------------------------
// 1. Initialization
// -------------------------------------------------------------
document.addEventListener('DOMContentLoaded', async () => {
    initTabs();
    initLeadChips();
    initPresets();
    initCanvasEvents();
    initBenchmarkSort();

    await loadCases();
    await loadCaseSignal(STATE.currentCaseIndex);
    await runInference();

    loadLeaderboard();
    loadGallery();
});

// -------------------------------------------------------------
// 2. Navigation Tabs
// -------------------------------------------------------------
function initTabs() {
    DOM.tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;
            DOM.tabs.forEach(t => t.classList.remove('active'));
            DOM.panels.forEach(p => p.classList.remove('active'));

            tab.classList.add('active');
            const targetPanel = document.getElementById(target);
            if (targetPanel) targetPanel.classList.add('active');

            if (target === 'tab-studio') {
                requestAnimationFrame(drawECG);
            }
        });
    });
}

// -------------------------------------------------------------
// 3. Clinical Cases & Signal Loading
// -------------------------------------------------------------
async function loadCases() {
    try {
        const res = await fetch('/api/cases');
        const data = await res.json();
        STATE.curatedCases = data.cases || [];

        renderCaseChips();
    } catch (err) {
        console.error('Failed to load clinical cases:', err);
    }
}

function renderCaseChips() {
    DOM.caseChipsContainer.innerHTML = '';
    STATE.curatedCases.forEach(c => {
        const card = document.createElement('div');
        card.className = `case-chip-card ${c.index === STATE.currentCaseIndex ? 'active' : ''}`;
        card.dataset.index = c.index;

        const diagLabels = c.ground_truth.join(', ');
        card.innerHTML = `
            <div class="chip-header">
                <span class="chip-icon">${c.icon}</span>
                <span class="chip-diag-tag">${diagLabels}</span>
            </div>
            <div class="chip-title">${c.title}</div>
            <div class="chip-sub">Patient #${c.index} | ${c.age}y ${c.gender}</div>
        `;

        card.addEventListener('click', () => {
            document.querySelectorAll('.case-chip-card').forEach(el => el.classList.remove('active'));
            card.classList.add('active');
            selectCase(c.index, c);
        });

        DOM.caseChipsContainer.appendChild(card);
    });
}

async function selectCase(index, metadata) {
    STATE.currentCaseIndex = index;
    DOM.directCaseInput.value = index;

    if (metadata) {
        updatePatientMetadata(metadata);
    }

    await loadCaseSignal(index);
    await runInference();
}

function updatePatientMetadata(data) {
    DOM.metaPid.textContent = `PAT-${String(data.index).padStart(3, '0')}`;
    DOM.metaDemo.textContent = `${data.age} yrs | ${data.gender}`;
    DOM.metaHr.textContent = `${data.hr_bpm} bpm`;

    DOM.metaGtBadges.innerHTML = '';
    data.ground_truth.forEach(lbl => {
        const span = document.createElement('span');
        span.className = `badge ${lbl === 'SR' ? 'badge-normal' : 'badge-abnormal'}`;
        span.textContent = lbl === 'SR' ? 'SR (Sinus Rhythm)' : lbl;
        DOM.metaGtBadges.appendChild(span);
    });
}

async function loadCaseSignal(index) {
    try {
        const res = await fetch(`/api/case_signal/${index}`);
        const data = await res.json();
        STATE.signalData = data;

        // If direct load without metadata
        if (!STATE.curatedCases.find(c => c.index === index)) {
            DOM.metaPid.textContent = `PAT-${String(index).padStart(3, '0')}`;
            DOM.metaDemo.textContent = `Cohort Patient #${index}`;
            DOM.metaHr.textContent = `75 bpm (est)`;
            DOM.metaGtBadges.innerHTML = '';
            data.ground_truth.forEach(lbl => {
                const span = document.createElement('span');
                span.className = `badge ${lbl === 'SR' ? 'badge-normal' : 'badge-abnormal'}`;
                span.textContent = lbl;
                DOM.metaGtBadges.appendChild(span);
            });
        }

        drawECG();
    } catch (err) {
        console.error('Failed to load ECG signal:', err);
    }
}

// -------------------------------------------------------------
// 4. Incomplete Lead Simulator & Toggles
// -------------------------------------------------------------
function initLeadChips() {
    DOM.leadChips.forEach(chip => {
        chip.addEventListener('click', () => {
            const leadIdx = parseInt(chip.dataset.lead, 10);
            if (STATE.activeLeads.has(leadIdx)) {
                if (STATE.activeLeads.size > 1) {
                    STATE.activeLeads.delete(leadIdx);
                    chip.classList.remove('active');
                    chip.querySelector('.chip-status').textContent = 'OFF';
                }
            } else {
                STATE.activeLeads.add(leadIdx);
                chip.classList.add('active');
                chip.querySelector('.chip-status').textContent = 'ON';
            }

            updateLeadCounters();
            drawECG();
            runInference();
        });
    });

    DOM.btnRunInference.addEventListener('click', runInference);

    // Direct case load
    DOM.btnLoadDirect.addEventListener('click', () => {
        const val = parseInt(DOM.directCaseInput.value, 10);
        if (!isNaN(val) && val >= 0 && val <= 524) {
            selectCase(val, STATE.curatedCases.find(c => c.index === val));
        }
    });

    // Random patient
    DOM.btnRandomPatient.addEventListener('click', () => {
        const randIdx = Math.floor(Math.random() * 525);
        selectCase(randIdx, STATE.curatedCases.find(c => c.index === randIdx));
    });
}

function updateLeadCounters() {
    const activeCount = STATE.activeLeads.size;
    const lossPct = Math.round(((12 - activeCount) / 12) * 100);

    DOM.activeLeadCounter.textContent = `${activeCount} / 12 Leads Active`;
    DOM.leadReductionPct.textContent = `${lossPct}% Data Loss`;

    DOM.leadReductionPct.className = `stat-badge ${lossPct > 50 ? 'danger' : lossPct > 0 ? 'secondary' : ''}`;
}

function setLeadsState(leadIndices) {
    STATE.activeLeads = new Set(leadIndices);
    DOM.leadChips.forEach(chip => {
        const idx = parseInt(chip.dataset.lead, 10);
        const isActive = STATE.activeLeads.has(idx);
        chip.classList.toggle('active', isActive);
        chip.querySelector('.chip-status').textContent = isActive ? 'ON' : 'OFF';
    });

    updateLeadCounters();
    drawECG();
    runInference();
}

function initPresets() {
    DOM.presetAll12.addEventListener('click', () => setLeadsState([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]));
    DOM.presetLimbOnly.addEventListener('click', () => setLeadsState([0, 1, 2, 3, 4, 5]));
    DOM.presetPrecordialOnly.addEventListener('click', () => setLeadsState([6, 7, 8, 9, 10, 11]));
    DOM.presetEinthoven.addEventListener('click', () => setLeadsState([0, 1, 2]));
    DOM.presetHolter.addEventListener('click', () => setLeadsState([0, 1]));
    DOM.presetSingleLead.addEventListener('click', () => setLeadsState([1])); // Lead II
    DOM.presetRandomDropout.addEventListener('click', () => {
        // Randomly select 4 to 8 leads
        const all = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        all.sort(() => Math.random() - 0.5);
        const count = Math.floor(Math.random() * 5) + 4; // 4 to 8 leads
        setLeadsState(all.slice(0, count));
    });
}

// -------------------------------------------------------------
// 5. Hospital-Grade Canvas ECG Renderer
// -------------------------------------------------------------
function drawECG() {
    const canvas = DOM.canvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // 1. Clear background
    ctx.fillStyle = '#060c14';
    ctx.fillRect(0, 0, width, height);

    // 2. Draw Medical Millimeter Grid
    drawMedicalGrid(ctx, width, height);

    if (!STATE.signalData || !STATE.signalData.leads) return;

    if (STATE.viewMode === 'multi') {
        drawMultiLeadGrid(ctx, width, height);
    } else {
        drawRhythmStrip(ctx, width, height);
    }
}

function drawMedicalGrid(ctx, width, height) {
    const minorStep = 10; // 1mm equivalent
    const majorStep = 50; // 5mm equivalent

    // Minor lines
    ctx.beginPath();
    ctx.strokeStyle = '#0d1e2e';
    ctx.lineWidth = 0.6;
    for (let x = 0; x <= width; x += minorStep) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
    }
    for (let y = 0; y <= height; y += minorStep) {
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
    }
    ctx.stroke();

    // Major lines
    ctx.beginPath();
    ctx.strokeStyle = '#16334d';
    ctx.lineWidth = 1.1;
    for (let x = 0; x <= width; x += majorStep) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
    }
    for (let y = 0; y <= height; y += majorStep) {
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
    }
    ctx.stroke();
}

/**
 * Standard 12-Lead Clinical Layout:
 * 4 Columns x 3 Rows for the 12 leads (first 2.5s of each group)
 * + 1 Full 10-Second Continuous Rhythm Strip of Lead II across the bottom!
 */
function drawMultiLeadGrid(ctx, width, height) {
    const leads = STATE.signalData.leads;
    const cols = 4;
    const rows = 3;
    const topHeight = height * 0.74;
    const bottomHeight = height * 0.26;

    const cellW = width / cols;
    const cellH = topHeight / rows;

    // Standard clinical lead matrix
    const matrix = [
        ["I", "aVR", "V1", "V4"],
        ["II", "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"]
    ];

    // Draw top 12 boxes
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const leadName = matrix[r][c];
            const leadIdx = STATE.leadNames.indexOf(leadName);
            const isActive = STATE.activeLeads.has(leadIdx);

            const x0 = c * cellW;
            const y0 = r * cellH;

            // Draw bounding subtle divider
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
            ctx.lineWidth = 1;
            ctx.strokeRect(x0, y0, cellW, cellH);

            // Lead Name Label
            ctx.font = 'bold 12px "JetBrains Mono", monospace';
            ctx.fillStyle = isActive ? '#00f2fe' : '#64748b';
            ctx.fillText(leadName, x0 + 12, y0 + 20);

            const leadValues = leads[leadName];
            if (!leadValues) continue;

            const midY = y0 + cellH / 2;
            const scaleY = (cellH * 0.38); // amplitude scaling

            if (isActive) {
                // Draw active waveform
                ctx.beginPath();
                ctx.strokeStyle = '#00f2fe';
                ctx.lineWidth = 1.6;
                ctx.lineJoin = 'round';

                // Display 250 points corresponding to 2.5s quadrant
                const ptsCount = 250;
                const startPt = c * ptsCount;

                for (let i = 0; i < ptsCount; i++) {
                    const sampleIdx = Math.min(startPt + i, leadValues.length - 1);
                    const val = leadValues[sampleIdx];
                    const px = x0 + (i / (ptsCount - 1)) * cellW;
                    const py = midY - val * scaleY;

                    if (i === 0) ctx.moveTo(px, py);
                    else ctx.lineTo(px, py);
                }
                ctx.stroke();
            } else {
                // Draw Masked Flatline with Warning
                ctx.beginPath();
                ctx.strokeStyle = '#475569';
                ctx.lineWidth = 1.2;
                ctx.setLineDash([4, 4]);
                ctx.moveTo(x0 + 10, midY);
                ctx.lineTo(x0 + cellW - 10, midY);
                ctx.stroke();
                ctx.setLineDash([]); // reset

                // Warning badge text
                ctx.font = '10px "Inter", sans-serif';
                ctx.fillStyle = '#f59e0b';
                ctx.fillText('[MASKED / NO SIGNAL]', x0 + cellW / 2 - 58, midY - 6);
            }
        }
    }

    // Bottom continuous rhythm strip (Lead II)
    const rhythmY0 = topHeight;
    const rhythmH = bottomHeight;
    const rhythmLeadName = "II";
    const rhythmLeadIdx = 1;
    const isRhythmActive = STATE.activeLeads.has(rhythmLeadIdx);

    // Section border
    ctx.strokeStyle = 'rgba(0, 242, 254, 0.3)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(0, rhythmY0, width, rhythmH);

    // Rhythm Label
    ctx.font = 'bold 12px "JetBrains Mono", monospace';
    ctx.fillStyle = isRhythmActive ? '#00f2fe' : '#64748b';
    ctx.fillText('LEAD II (CONTINUOUS 10.0s RHYTHM STRIP)', 14, rhythmY0 + 20);

    const rValues = leads[rhythmLeadName];
    if (rValues) {
        const midY = rhythmY0 + rhythmH / 2;
        const scaleY = (rhythmH * 0.40);

        if (isRhythmActive) {
            ctx.beginPath();
            ctx.strokeStyle = '#10b981'; // Emerald for continuous rhythm
            ctx.lineWidth = 1.8;
            ctx.lineJoin = 'round';

            const len = rValues.length;
            for (let i = 0; i < len; i++) {
                const px = (i / (len - 1)) * width;
                const py = midY - rValues[i] * scaleY;
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }
            ctx.stroke();
        } else {
            ctx.beginPath();
            ctx.strokeStyle = '#475569';
            ctx.setLineDash([6, 6]);
            ctx.moveTo(10, midY);
            ctx.lineTo(width - 10, midY);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.font = '12px "Inter", sans-serif';
            ctx.fillStyle = '#f59e0b';
            ctx.fillText('[LEAD II DISCONNECTED / STATE-SPACE IMPUTED]', width / 2 - 130, midY - 8);
        }
    }
}

/**
 * Single Lead Expanded Rhythm View
 */
function drawRhythmStrip(ctx, width, height) {
    const leads = STATE.signalData.leads;
    const leadName = "II";
    const leadIdx = 1;
    const isActive = STATE.activeLeads.has(leadIdx);

    ctx.font = 'bold 14px "JetBrains Mono", monospace';
    ctx.fillStyle = isActive ? '#00f2fe' : '#64748b';
    ctx.fillText('EXPANDED 10-SECOND HIGH-RESOLUTION STRIP (LEAD II)', 20, 30);

    const values = leads[leadName];
    if (!values) return;

    const midY = height / 2;
    const scaleY = (height * 0.35);

    if (isActive) {
        ctx.beginPath();
        ctx.strokeStyle = '#00f2fe';
        ctx.lineWidth = 2.2;
        ctx.lineJoin = 'round';

        const len = values.length;
        for (let i = 0; i < len; i++) {
            const px = (i / (len - 1)) * width;
            const py = midY - values[i] * scaleY;
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        ctx.stroke();
    } else {
        ctx.beginPath();
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([8, 8]);
        ctx.moveTo(20, midY);
        ctx.lineTo(width - 20, midY);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.font = '14px "Inter", sans-serif';
        ctx.fillStyle = '#f59e0b';
        ctx.fillText('[LEAD II MASKED - RE-ENABLE LEAD II TO OBSERVE WAVEFORM]', width / 2 - 180, midY - 10);
    }
}

// -------------------------------------------------------------
// 6. Interactive Canvas Tools & Sweep Animation
// -------------------------------------------------------------
function initCanvasEvents() {
    DOM.btnViewMulti.addEventListener('click', () => {
        STATE.viewMode = 'multi';
        DOM.btnViewMulti.classList.add('active');
        DOM.btnViewRhythm.classList.remove('active');
        drawECG();
    });

    DOM.btnViewRhythm.addEventListener('click', () => {
        STATE.viewMode = 'rhythm';
        DOM.btnViewRhythm.classList.add('active');
        DOM.btnViewMulti.classList.remove('active');
        drawECG();
    });

    // Play sweep
    DOM.btnPlaySweep.addEventListener('click', toggleSweepAnimation);

    // Canvas Hover
    DOM.canvas.addEventListener('mousemove', (e) => {
        const rect = DOM.canvas.getBoundingClientRect();
        const scaleX = DOM.canvas.width / rect.width;
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top);

        const timeSec = ((x / DOM.canvas.width) * 10.0).toFixed(2);
        DOM.canvasTooltip.style.display = 'block';
        DOM.canvasTooltip.style.left = `${e.clientX - rect.left + 15}px`;
        DOM.canvasTooltip.style.top = `${e.clientY - rect.top + 10}px`;
        DOM.canvasTooltip.textContent = `t = ${timeSec}s`;
    });

    DOM.canvas.addEventListener('mouseleave', () => {
        DOM.canvasTooltip.style.display = 'none';
    });
}

function toggleSweepAnimation() {
    STATE.isSweeping = !STATE.isSweeping;
    if (STATE.isSweeping) {
        DOM.btnPlaySweep.textContent = '⏹ Stop Sweep';
        DOM.sweepCursor.style.display = 'block';
        runSweepLoop();
    } else {
        DOM.btnPlaySweep.textContent = '▶ Play Sweep';
        DOM.sweepCursor.style.display = 'none';
        if (STATE.sweepAnimId) cancelAnimationFrame(STATE.sweepAnimId);
    }
}

function runSweepLoop() {
    if (!STATE.isSweeping) return;

    STATE.sweepProgress += 0.0035; // 10s cycle
    if (STATE.sweepProgress > 1.0) STATE.sweepProgress = 0;

    const viewportW = DOM.canvasViewport.clientWidth;
    DOM.sweepCursor.style.left = `${STATE.sweepProgress * viewportW}px`;

    STATE.sweepAnimId = requestAnimationFrame(runSweepLoop);
}

// -------------------------------------------------------------
// 7. Live Model Inference & Prediction Rendering
// -------------------------------------------------------------
async function runInference() {
    DOM.inferenceSpinner.style.display = 'inline-block';
    DOM.btnRunInference.disabled = true;

    try {
        const payload = {
            case_index: STATE.currentCaseIndex,
            active_leads: Array.from(STATE.activeLeads),
            mc_runs: 10
        };

        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        renderPredictionResults(data);
    } catch (err) {
        console.error('Inference error:', err);
    } finally {
        DOM.inferenceSpinner.style.display = 'none';
        DOM.btnRunInference.disabled = false;
    }
}

function renderPredictionResults(data) {
    if (!data || !data.predictions) return;

    // 1. Render Multi-label Predictions List
    DOM.predictionsList.innerHTML = '';
    data.predictions.forEach(p => {
        const row = document.createElement('div');
        const isNormal = p.class_code === 'SR';
        let rowClass = 'pred-row';
        if (p.is_positive) {
            rowClass += isNormal ? ' positive-normal' : ' positive-abnormal';
        }
        row.className = rowClass;

        // Progress bar color
        let barColor = 'dim';
        if (p.is_positive) {
            barColor = isNormal ? 'normal' : 'danger';
        } else if (p.probability >= 0.35) {
            barColor = 'warning';
        }

        // Pill
        let pillHtml = '';
        if (p.is_positive) {
            pillHtml = isNormal 
                ? '<span class="pred-pill normal-detected">NORMAL RHYTHM</span>'
                : '<span class="pred-pill detected">ABNORMALITY DETECTED</span>';
        } else {
            pillHtml = '<span class="pred-pill safe">NON-DETECTED</span>';
        }

        // Match icon
        let matchHtml = '';
        if (p.match_status === 'TRUE_POSITIVE' || p.match_status === 'TRUE_NEGATIVE') {
            matchHtml = '<span class="match-icon tp">✓ Confirmed</span>';
        } else if (p.match_status === 'FALSE_POSITIVE') {
            matchHtml = '<span class="match-icon fp">⚠ Elevated (Low Lead)</span>';
        } else {
            matchHtml = '<span class="match-icon fn">✗ Attenuated</span>';
        }

        row.innerHTML = `
            <div class="pred-row-top">
                <div class="pred-class-info">
                    <span class="pred-code">${p.class_code}</span>
                    <span class="pred-name">${p.class_name}</span>
                </div>
                <div class="pred-tag-group">
                    ${pillHtml}
                    ${matchHtml}
                </div>
            </div>

            <div class="pred-bar-wrapper">
                <div class="threshold-marker" title="Decision Threshold τ = 0.50"></div>
                <div class="pred-bar-fill ${barColor}" style="width: ${Math.min(100, Math.max(3, p.probability_pct))}%;"></div>
            </div>

            <div class="pred-stats-bottom">
                <span>Probability: <strong>${p.probability_pct}%</strong></span>
                <span>Epistemic Var: σ=${p.epistemic_uncertainty}</span>
                <span>Aleatoric: s=${p.aleatoric_uncertainty}</span>
            </div>
        `;

        DOM.predictionsList.appendChild(row);
    });

    // 2. Render Triage Decision
    const triage = data.triage;
    DOM.triageStatusPill.className = `triage-status-pill ${triage.badge}`;
    DOM.triageStatusPill.textContent = triage.title.toUpperCase();

    DOM.triageBanner.className = `triage-banner ${triage.badge}`;
    DOM.triageBannerIcon.textContent = triage.badge === 'success' ? '🟢' : triage.badge === 'warning' ? '🟡' : '🔴';
    DOM.triageBannerTitle.textContent = triage.title;
    DOM.triageBannerDesc.textContent = triage.description;

    // 3. Uncertainty Values & Dials
    DOM.epistemicVal.textContent = triage.mean_epistemic.toFixed(3);
    const epPct = Math.min(100, (triage.mean_epistemic / 0.3) * 100);
    DOM.epistemicFill.style.width = `${epPct}%`;
    DOM.epistemicFill.className = `u-fill ${triage.mean_epistemic < 0.08 ? 'green' : triage.mean_epistemic < 0.16 ? 'amber' : 'red'}`;
    DOM.epistemicTag.textContent = triage.mean_epistemic < 0.08 ? 'LOW (STABLE)' : triage.mean_epistemic < 0.16 ? 'MODERATE' : 'ELEVATED';

    DOM.aleatoricVal.textContent = triage.mean_aleatoric.toFixed(3);
    const alPct = Math.min(100, (triage.mean_aleatoric / 0.2) * 100);
    DOM.aleatoricFill.style.width = `${alPct}%`;

    // 4. Lead Attention Bars (XAI)
    DOM.leadAttentionBars.innerHTML = '';
    const maxAttn = Math.max(...data.lead_attentions.map(a => a.attention_weight), 0.01);
    data.lead_attentions.forEach(a => {
        const col = document.createElement('div');
        col.className = 'attn-bar-col';
        const heightPct = Math.round((a.attention_weight / maxAttn) * 100);

        col.innerHTML = `
            <div class="attn-bar-box" title="${a.lead_name}: ${a.attention_pct}% weight">
                <div class="attn-bar-fill" style="height: ${a.is_active ? Math.max(8, heightPct) : 2}%; opacity: ${a.is_active ? 1 : 0.2};"></div>
            </div>
            <span class="attn-bar-lbl">${a.lead_name}</span>
        `;
        DOM.leadAttentionBars.appendChild(col);
    });

    // 5. Telemetry
    DOM.resLatency.textContent = `${data.inference_time_ms} ms`;
    DOM.teleLatency.textContent = `~${data.inference_time_ms} ms`;
    DOM.resConfidence.textContent = `${triage.confidence_score}%`;
}

// -------------------------------------------------------------
// 8. Multi-Model Benchmark Arena Leaderboard
// -------------------------------------------------------------
async function loadLeaderboard() {
    try {
        const res = await fetch('/api/leaderboard');
        const data = await res.json();
        STATE.leaderboardData = data.leaderboard || [];
        renderLeaderboard();
    } catch (err) {
        console.error('Failed to load leaderboard:', err);
    }
}

function renderLeaderboard() {
    const tbody = DOM.benchmarkTbody;
    tbody.innerHTML = '';

    const query = (DOM.leaderboardSearch.value || '').toLowerCase();
    const family = DOM.leaderboardFamilyFilter.value;

    let filtered = STATE.leaderboardData.filter(m => {
        const name = (m.Model || '').toLowerCase();
        const matchesQuery = name.includes(query);
        if (!matchesQuery) return false;

        if (family === 'ALL') return true;
        if (family === 'State-Space') return name.includes('lac-ssm') || name.includes('state space');
        if (family === 'Deep Learning') return name.includes('1d') || name.includes('resnet') || name.includes('tcn') || name.includes('lstm') || name.includes('gru') || name.includes('vgg');
        if (family === 'Classical ML') return name.includes('forest') || name.includes('stacking') || name.includes('boost') || name.includes('svm') || name.includes('trees') || name.includes('regression') || name.includes('knn') || name.includes('perceptron');
        if (family === 'One-Class') return name.includes('one-class');
        return true;
    });

    // Sort
    filtered.sort((a, b) => {
        let va = a[STATE.sortColumn] || 0;
        let vb = b[STATE.sortColumn] || 0;
        return STATE.sortAsc ? va - vb : vb - va;
    });

    filtered.forEach((m, idx) => {
        const tr = document.createElement('tr');
        const isSSM = m.Model.includes('LAC-SSM') || m.Model.includes('Proposed');
        const isWinner = m.Model.includes('Stacking') || m.Model.includes('TCN') || m.Model.includes('ResNet');

        if (isSSM) tr.className = 'row-ssm';
        else if (isWinner) tr.className = 'row-winner';

        // Categorize family
        let familyName = 'Deep Learning';
        let leadResilience = 'Low (Fails on missing)';
        if (isSSM) {
            familyName = 'State-Space (S4D)';
            leadResilience = '★★★★★ High (0.7240 on 1-lead)';
        } else if (m.Model.includes('Stacking') || m.Model.includes('Forest') || m.Model.includes('XGB') || m.Model.includes('Trees') || m.Model.includes('Logistic')) {
            familyName = 'Classical ML';
            leadResilience = '★☆☆☆☆ Imputation Dependent';
        } else if (m.Model.includes('One-Class')) {
            familyName = 'Unsupervised Outlier';
            leadResilience = '★★☆☆☆ Baseline';
        }

        tr.innerHTML = `
            <td>#${idx + 1}</td>
            <td class="model-name-cell">
                ${isSSM ? '👑' : ''} ${m.Model}
            </td>
            <td><span class="mtag">${familyName}</span></td>
            <td class="mono-metric ${m['Macro F1'] > 0.65 ? 'top-score' : ''}">${Number(m['Macro F1']).toFixed(4)}</td>
            <td class="mono-metric">${Number(m['Micro F1']).toFixed(4)}</td>
            <td class="mono-metric ${m['Macro AUROC'] > 0.90 ? 'top-score' : ''}">${Number(m['Macro AUROC']).toFixed(4)}</td>
            <td class="mono-metric ${m['Micro AUROC'] > 0.93 ? 'top-score' : ''}">${Number(m['Micro AUROC']).toFixed(4)}</td>
            <td class="mono-metric">${Number(m['Hamming Loss']).toFixed(4)}</td>
            <td class="mono-metric ${m['Accuracy (%)'] > 90 ? 'top-score' : ''}"><strong>${Number(m['Accuracy (%)']).toFixed(2)}%</strong></td>
            <td><small>${leadResilience}</small></td>
        `;

        tbody.appendChild(tr);
    });
}

function initBenchmarkSort() {
    DOM.leaderboardSearch.addEventListener('input', renderLeaderboard);
    DOM.leaderboardFamilyFilter.addEventListener('change', renderLeaderboard);

    document.querySelectorAll('.benchmark-table th.sortable').forEach(th => {
        th.addEventListener('click', () => {
            const col = th.dataset.sort;
            if (STATE.sortColumn === col) {
                STATE.sortAsc = !STATE.sortAsc;
            } else {
                STATE.sortColumn = col;
                STATE.sortAsc = false;
            }
            renderLeaderboard();
        });
    });
}

// -------------------------------------------------------------
// 9. Diagnostic Plot Gallery & Lightbox
// -------------------------------------------------------------
async function loadGallery() {
    try {
        const res = await fetch('/api/gallery');
        const data = await res.json();
        STATE.galleryData = data;

        // Populate dropdown
        DOM.galleryModelSelect.innerHTML = '';
        data.models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.id;
            opt.textContent = `${m.name} (${m.family})`;
            DOM.galleryModelSelect.appendChild(opt);
        });

        DOM.galleryModelSelect.addEventListener('change', (e) => {
            updateGallerySelection(e.target.value);
        });

        // Populate master figures
        DOM.masterFiguresGrid.innerHTML = '';
        data.key_figures.forEach(fig => {
            const card = document.createElement('div');
            card.className = 'showcase-card';
            card.innerHTML = `
                <div class="showcase-title">${fig.title}</div>
                <img src="${fig.url}" alt="${fig.title}" class="showcase-img" onclick="openLightbox('${fig.url}', '${fig.title}')">
                <div class="showcase-caption">${fig.caption}</div>
            `;
            DOM.masterFiguresGrid.appendChild(card);
        });

    } catch (err) {
        console.error('Failed to load gallery:', err);
    }
}

function updateGallerySelection(modelId) {
    if (!STATE.galleryData) return;
    const model = STATE.galleryData.models.find(m => m.id === modelId);
    if (!model) return;

    DOM.displayCmModelName.textContent = model.name;
    DOM.displayRocModelName.textContent = model.name;

    if (model.cm_image) {
        DOM.displayCmImg.src = model.cm_image;
        DOM.displayCmImg.alt = `${model.name} Confusion Matrix`;
    }
    if (model.roc_image) {
        DOM.displayRocImg.src = model.roc_image;
        DOM.displayRocImg.alt = `${model.name} ROC Curve`;
    }
}

// Lightbox Handlers
window.openLightbox = function(src, caption) {
    DOM.lightboxImg.src = src;
    DOM.lightboxCaption.textContent = caption || '';
    DOM.lightboxModal.classList.add('active');
};

window.closeLightbox = function() {
    DOM.lightboxModal.classList.remove('active');
};
