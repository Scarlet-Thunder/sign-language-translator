import { createHandTracker, drawHand } from './hand.js';
import { normalize } from './features.js';
import { classifyHeuristic } from './heuristic.js';
import { MlpClassifier } from './classifier.js';
import { Committer } from './committer.js';
import { Dataset } from './dataset.js';
import * as store from './storage.js';

// J and Z are left out: they are drawn in the air and need a sequence model.
const LABELS = [...'ABCDEFGHIKLMNOPQRSTUVWXY', 'SPACE', 'DEL'];
const HINTS = {
  A: 'Fist, thumb resting against the side of the index finger.',
  B: 'Flat hand, fingers together pointing up, thumb folded across the palm.',
  C: 'Curve fingers and thumb into a C shape.',
  D: 'Index finger up; the other fingertips touch the thumb.',
  E: 'Fingertips bent down to rest on the thumb, which is tucked across the palm.',
  F: 'Index and thumb touch in a circle; the other three fingers up and spread.',
  G: 'Index and thumb point sideways, parallel; other fingers curled.',
  H: 'Index and middle fingers extended sideways, together.',
  I: 'Pinky up, other fingers in a fist with the thumb across.',
  K: 'Index and middle up in a V, thumb touching between them.',
  L: 'Index up and thumb out to make an L.',
  M: 'Thumb tucked under the first three fingers.',
  N: 'Thumb tucked under the first two fingers.',
  O: 'All fingertips touch the thumb to form an O.',
  P: 'Like K, but pointing down.',
  Q: 'Like G, but pointing down.',
  R: 'Index and middle fingers crossed.',
  S: 'Fist with the thumb wrapped across the front of the fingers.',
  T: 'Thumb tucked between the index and middle fingers.',
  U: 'Index and middle fingers up, held together.',
  V: 'Index and middle fingers up, spread apart.',
  W: 'Index, middle and ring fingers up and spread.',
  X: 'Index finger hooked, other fingers in a fist.',
  Y: 'Thumb and pinky out, other fingers curled.',
  SPACE: 'Open hand with all five fingers spread. Inserts a word break.',
  DEL: 'Any distinct sign you like (e.g. thumbs-down). Deletes the last letter.',
};
const DISPLAY = { SPACE: '␣', DEL: '⌫' };
const RING_LEN = 326.7;
const RECORD_TARGET = 150;
const AUTO_SPACE_MS = 1500;

const $ = (id) => document.getElementById(id);
const els = {
  video: $('video'), overlay: $('overlay'), videoWrap: $('video-wrap'),
  startOverlay: $('start-overlay'), startBtn: $('start-btn'), loadStatus: $('load-status'),
  cameraSelect: $('camera-select'), handHint: $('hand-hint'),
  engineBadge: $('engine-badge'), fps: $('fps'),
  ring: $('ring-fg'), letter: $('current-letter'), top3: $('top3'),
  transcript: $('transcript'),
  labelGrid: $('label-grid'), labelHint: $('label-hint'), datasetTotal: $('dataset-total'),
  recordBtn: $('record-btn'), recordProgress: $('record-progress'), recordFlash: $('record-flash'),
  trainBtn: $('train-btn'), trainProgress: $('train-progress'), trainLog: $('train-log'),
  toast: $('toast'),
};
const ctx = els.overlay.getContext('2d');

const prefs = store.loadPrefs({
  holdMs: 800, minConf: 0.7, autoSpace: true, speakWords: false,
  showLandmarks: true, mirror: true, engine: 'auto',
});

const state = {
  tracker: null,
  stream: null,
  running: false,
  mode: 'translate',
  userModel: null,
  shippedModel: null,
  dataset: new Dataset(),
  selectedLabel: 'A',
  recording: null,
  training: false,
  lastHandAt: 0,
  lastVideoTime: -1,
  frames: 0,
  fpsAt: performance.now(),
  lastShown: '',
};
const committer = new Committer({ holdMs: prefs.holdMs, minConfidence: prefs.minConf });

/* ---------------- Startup ---------------- */

init();

async function init() {
  bindUI();
  applyPrefs();
  try {
    els.loadStatus.textContent = 'Downloading hand-tracking model (≈8 MB, cached after first visit)…';
    await waitForTf();
    const [tracker] = await Promise.all([createHandTracker(), loadModels(), state.dataset.load()]);
    state.tracker = tracker;
    renderLabelGrid();
    updateEngineBadge();
    els.loadStatus.textContent = 'Ready. Your browser will ask for camera permission.';
    els.startBtn.disabled = false;
    els.startBtn.textContent = 'Start camera';
  } catch (err) {
    console.error(err);
    els.loadStatus.textContent = 'Could not load the models. Check your connection and reload. (' + err.message + ')';
    els.startBtn.textContent = 'Load failed';
  }
}

function waitForTf() {
  return new Promise((resolve, reject) => {
    const t0 = performance.now();
    (function check() {
      if (window.tf) return resolve();
      if (performance.now() - t0 > 30000) return reject(new Error('TensorFlow.js failed to load'));
      setTimeout(check, 50);
    })();
  });
}

async function loadModels() {
  const saved = await store.get('user-model');
  if (saved) {
    try { state.userModel = MlpClassifier.fromJSON(saved); } catch (e) { console.warn('Ignoring saved model:', e); }
  }
  try {
    const res = await fetch('model/weights.json', { cache: 'no-cache' });
    if (res.ok) state.shippedModel = MlpClassifier.fromJSON(await res.json());
  } catch { /* no bundled model, that's fine */ }
  if (state.userModel) logTrain(describeModel(state.userModel, 'Saved model loaded'));
}

/* ---------------- Camera ---------------- */

async function startCamera(deviceId) {
  if (!window.isSecureContext) {
    toast('Camera needs HTTPS (or localhost).');
    return;
  }
  try {
    if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: deviceId
        ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
        : { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
    });
    els.video.srcObject = state.stream;
    await els.video.play();
    els.startOverlay.hidden = true;
    await populateCameras();
    if (!state.running) {
      state.running = true;
      requestAnimationFrame(loop);
    }
  } catch (err) {
    console.error(err);
    const msg = err.name === 'NotAllowedError'
      ? 'Camera permission was denied. Allow it in your browser’s site settings and try again.'
      : err.name === 'NotFoundError' ? 'No camera found.' : 'Could not start camera: ' + err.message;
    els.loadStatus.textContent = msg;
    toast(msg);
  }
}

async function populateCameras() {
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
  const current = state.stream?.getVideoTracks()[0]?.getSettings().deviceId;
  els.cameraSelect.innerHTML = '';
  devices.forEach((d, i) => {
    const o = document.createElement('option');
    o.value = d.deviceId;
    o.textContent = d.label || `Camera ${i + 1}`;
    if (d.deviceId === current) o.selected = true;
    els.cameraSelect.appendChild(o);
  });
  els.cameraSelect.disabled = devices.length < 2;
}

/* ---------------- Main loop ---------------- */

function loop() {
  requestAnimationFrame(loop);
  const v = els.video;
  if (v.readyState < 2 || v.currentTime === state.lastVideoTime) return;
  state.lastVideoTime = v.currentTime;

  const now = performance.now();
  const result = state.tracker.detectForVideo(v, now);
  tickFps(now);

  const w = v.videoWidth, h = v.videoHeight;
  if (els.overlay.width !== w || els.overlay.height !== h) {
    els.overlay.width = w;
    els.overlay.height = h;
  }
  ctx.clearRect(0, 0, w, h);

  const lm = result.landmarks && result.landmarks[0];
  const handedList = result.handedness || result.handednesses;
  let pred = null;
  if (lm) {
    const handed = handedList?.[0]?.[0]?.categoryName || 'Right';
    const aspect = w / h;
    state.lastHandAt = now;
    if (prefs.showLandmarks) drawHand(ctx, lm, w, h, state.recording?.active ? '#ff5d5d' : '#6d86ff');
    if (state.mode === 'train') {
      if (state.recording?.active) recordSample(normalize(lm, handed, aspect), now);
    } else {
      pred = predict(lm, handed, aspect);
    }
  }
  setHandHint(!!lm);
  if (state.mode === 'translate') handlePrediction(pred, now);
}

function predict(lm, handed, aspect) {
  const engine = activeEngine();
  if (engine.kind === 'model') return engine.model.predict(normalize(lm, handed, aspect));
  return classifyHeuristic(lm, aspect);
}

function activeEngine() {
  const model = state.userModel || state.shippedModel;
  if (prefs.engine !== 'rules' && model) return { kind: 'model', model };
  return { kind: 'rules' };
}

function handlePrediction(pred, now) {
  const { stable, progress, commit } = committer.update(pred, now);

  const confident = pred && pred.label && pred.confidence >= committer.minConfidence;
  const shown = stable || (confident ? pred.label : null);
  const text = shown ? (DISPLAY[shown] || shown) : '–';
  if (text !== state.lastShown) {
    els.letter.textContent = text;
    state.lastShown = text;
  }
  els.ring.style.strokeDashoffset = String(RING_LEN * (1 - progress));
  els.ring.classList.toggle('done', !!stable && committer.committed);
  renderTop3(pred);

  if (commit) applyCommit(commit);

  const t = els.transcript.value;
  if (!pred && prefs.autoSpace && t && !t.endsWith(' ') && now - state.lastHandAt > AUTO_SPACE_MS) {
    addSpace();
  }
}

function renderTop3(pred) {
  const items = pred ? pred.top : [];
  els.top3.innerHTML = items.map(({ label, p }) => `
    <li><span>${label}</span>
      <span class="bar"><span style="width:${(p * 100).toFixed(0)}%"></span></span>
      <span>${(p * 100).toFixed(0)}%</span></li>`).join('') || '<li class="muted">Show your hand…</li>';
}

function tickFps(now) {
  state.frames++;
  if (now - state.fpsAt >= 1000) {
    els.fps.textContent = Math.round((state.frames * 1000) / (now - state.fpsAt)) + ' fps';
    state.frames = 0;
    state.fpsAt = now;
  }
}

function setHandHint(on) {
  const text = on ? 'Hand detected' : 'No hand detected';
  if (els.handHint.textContent !== text) {
    els.handHint.textContent = text;
    els.handHint.classList.toggle('on', on);
  }
}

/* ---------------- Transcript ---------------- */

function applyCommit(label) {
  if (label === 'SPACE') addSpace();
  else if (label === 'DEL') backspace();
  else appendText(label);
}

function appendText(s) {
  els.transcript.value += s;
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

function addSpace() {
  const t = els.transcript.value;
  if (!t || t.endsWith(' ')) return;
  appendText(' ');
  if (prefs.speakWords) speak(t.split(' ').pop());
}

function backspace() {
  els.transcript.value = els.transcript.value.slice(0, -1);
}

function speak(text) {
  if (!text.trim() || !('speechSynthesis' in window)) return;
  const u = new SpeechSynthesisUtterance(text.toLowerCase());
  u.rate = 0.95;
  speechSynthesis.speak(u);
}

/* ---------------- Training ---------------- */

function renderLabelGrid() {
  els.labelGrid.innerHTML = '';
  for (const label of LABELS) {
    const b = document.createElement('button');
    const n = state.dataset.count(label);
    b.className = (label.length > 1 ? 'wide ' : '') + (n >= 10 ? 'has' : '');
    b.setAttribute('aria-pressed', String(label === state.selectedLabel));
    b.innerHTML = `${label}<small>${n}</small>`;
    b.addEventListener('click', () => {
      if (state.recording) return;
      state.selectedLabel = label;
      renderLabelGrid();
    });
    els.labelGrid.appendChild(b);
  }
  els.labelHint.innerHTML = `<strong>${state.selectedLabel}</strong>: ${HINTS[state.selectedLabel]}`;
  els.datasetTotal.textContent = `${state.dataset.total()} samples`;
  els.recordBtn.textContent = state.recording ? 'Stop' : `Record ${RECORD_TARGET} “${state.selectedLabel}”`;
}

async function toggleRecording() {
  if (state.recording) return stopRecording();
  if (!state.running) return toast('Start the camera first.');
  const rec = { label: state.selectedLabel, got: 0, lastAt: 0, active: false };
  state.recording = rec;
  renderLabelGrid();
  els.recordFlash.hidden = false;
  for (let i = 3; i > 0; i--) {
    if (state.recording !== rec) return;
    els.recordFlash.textContent = `Get ready: “${rec.label}” in ${i}…`;
    await new Promise((r) => setTimeout(r, 700));
  }
  if (state.recording !== rec) return;
  rec.active = true;
  els.recordFlash.textContent = `● Recording “${rec.label}”`;
}

function recordSample(vec, now) {
  const rec = state.recording;
  if (now - rec.lastAt < 30) return;
  rec.lastAt = now;
  state.dataset.add(rec.label, vec);
  rec.got++;
  els.recordProgress.style.width = `${(rec.got / RECORD_TARGET) * 100}%`;
  if (rec.got % 10 === 0) renderLabelGrid();
  if (rec.got >= RECORD_TARGET) stopRecording();
}

async function stopRecording() {
  const rec = state.recording;
  state.recording = null;
  els.recordFlash.hidden = true;
  els.recordProgress.style.width = '0';
  renderLabelGrid();
  if (rec && rec.got) {
    await state.dataset.save();
    toast(`Saved ${rec.got} samples of “${rec.label}”.`);
  }
}

async function trainModel() {
  if (state.training) return;
  state.training = true;
  els.trainBtn.disabled = true;
  els.trainLog.textContent = 'Training…\n';
  try {
    const model = await MlpClassifier.train(state.dataset.samples, {
      epochs: 40,
      onEpoch: (e, total, logs) => {
        els.trainProgress.style.width = `${(e / total) * 100}%`;
        if (e % 5 === 0 || e === total) {
          const acc = logs.acc ?? logs.accuracy, vacc = logs.val_acc ?? logs.val_accuracy;
          logTrain(`epoch ${String(e).padStart(2)}  loss ${logs.loss.toFixed(3)}  acc ${(acc * 100).toFixed(1)}%  val ${(vacc * 100).toFixed(1)}%`, true);
        }
      },
    });
    if (state.userModel) state.userModel.dispose();
    state.userModel = model;
    await store.set('user-model', model.toJSON());
    logTrain('\n' + describeModel(model, 'Done. Model saved to this browser'), true);
    updateEngineBadge();
    toast('Model trained. Switch to Translate to try it.');
  } catch (err) {
    logTrain('Error: ' + err.message, true);
  } finally {
    state.training = false;
    els.trainBtn.disabled = false;
    setTimeout(() => { els.trainProgress.style.width = '0'; }, 800);
  }
}

function describeModel(model, prefix) {
  const m = model.meta.metrics || {};
  const acc = m.valAccuracy != null ? `${(m.valAccuracy * 100).toFixed(1)}% validation accuracy` : 'accuracy n/a';
  return `${prefix}: ${model.labels.length} signs (${model.labels.join(' ')}), ${m.samples ?? '?'} samples, ${acc}.`;
}

function logTrain(line, append = false) {
  els.trainLog.textContent = append ? els.trainLog.textContent + line + '\n' : line + '\n';
  els.trainLog.scrollTop = els.trainLog.scrollHeight;
}

/* ---------------- UI wiring ---------------- */

function applyPrefs() {
  $('hold').value = prefs.holdMs;
  $('conf').value = prefs.minConf;
  $('auto-space').checked = prefs.autoSpace;
  $('speak-words').checked = prefs.speakWords;
  $('show-landmarks').checked = prefs.showLandmarks;
  $('mirror').checked = prefs.mirror;
  $('engine-select').value = prefs.engine;
  $('hold-out').textContent = `${prefs.holdMs} ms`;
  $('conf-out').textContent = `${Math.round(prefs.minConf * 100)}%`;
  els.videoWrap.classList.toggle('mirrored', prefs.mirror);
  committer.holdMs = prefs.holdMs;
  committer.minConfidence = effectiveMinConf();
}

// Starter-rule scores are fuzzy match strengths, not probabilities, so they get a lower bar.
function effectiveMinConf() {
  return activeEngine().kind === 'rules' ? prefs.minConf * 0.75 : prefs.minConf;
}

function setPref(key, value) {
  prefs[key] = value;
  store.savePrefs(prefs);
  applyPrefs();
  updateEngineBadge();
}

function updateEngineBadge() {
  const e = activeEngine();
  let text;
  if (e.kind === 'rules') {
    text = prefs.engine === 'model' ? 'No model yet: starter rules' : 'Starter rules';
  } else {
    const acc = e.model.meta.metrics?.valAccuracy;
    text = `${e.model === state.userModel ? 'Your model' : 'Bundled model'} · ${e.model.labels.length} signs`
      + (acc != null ? ` · ${(acc * 100).toFixed(0)}%` : '');
  }
  els.engineBadge.textContent = text;
  committer.minConfidence = effectiveMinConf();
  committer.reset();
}

function setMode(mode) {
  if (state.recording) stopRecording();
  state.mode = mode;
  $('tab-translate').setAttribute('aria-selected', String(mode === 'translate'));
  $('tab-train').setAttribute('aria-selected', String(mode === 'train'));
  $('panel-translate').hidden = mode !== 'translate';
  $('panel-train').hidden = mode !== 'train';
  committer.reset();
}

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement('a'), { href: url, download: name });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

let toastTimer;
function toast(msg) {
  els.toast.textContent = msg;
  els.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { els.toast.hidden = true; }, 2800);
}

function bindUI() {
  els.startBtn.addEventListener('click', () => startCamera());
  els.cameraSelect.addEventListener('change', (e) => startCamera(e.target.value));
  $('tab-translate').addEventListener('click', () => setMode('translate'));
  $('tab-train').addEventListener('click', () => setMode('train'));

  $('space-btn').addEventListener('click', addSpace);
  $('back-btn').addEventListener('click', backspace);
  $('clear-btn').addEventListener('click', () => { els.transcript.value = ''; });
  $('speak-btn').addEventListener('click', () => speak(els.transcript.value));
  $('copy-btn').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(els.transcript.value); toast('Copied.'); }
    catch { els.transcript.select(); toast('Press Ctrl+C to copy.'); }
  });

  $('hold').addEventListener('input', (e) => setPref('holdMs', Number(e.target.value)));
  $('conf').addEventListener('input', (e) => setPref('minConf', Number(e.target.value)));
  $('auto-space').addEventListener('change', (e) => setPref('autoSpace', e.target.checked));
  $('speak-words').addEventListener('change', (e) => setPref('speakWords', e.target.checked));
  $('show-landmarks').addEventListener('change', (e) => setPref('showLandmarks', e.target.checked));
  $('mirror').addEventListener('change', (e) => setPref('mirror', e.target.checked));
  $('engine-select').addEventListener('change', (e) => setPref('engine', e.target.value));

  els.recordBtn.addEventListener('click', toggleRecording);
  $('clear-label-btn').addEventListener('click', async () => {
    if (!confirm(`Delete all samples for “${state.selectedLabel}”?`)) return;
    state.dataset.clear(state.selectedLabel);
    await state.dataset.save();
    renderLabelGrid();
  });
  $('clear-all-btn').addEventListener('click', async () => {
    if (!confirm('Delete ALL recorded samples? This cannot be undone.')) return;
    state.dataset.clear();
    await state.dataset.save();
    renderLabelGrid();
  });
  els.trainBtn.addEventListener('click', trainModel);
  $('delete-model-btn').addEventListener('click', async () => {
    if (!state.userModel || !confirm('Delete your trained model?')) return;
    state.userModel.dispose();
    state.userModel = null;
    await store.del('user-model');
    logTrain('Trained model deleted.');
    updateEngineBadge();
  });

  $('export-csv-btn').addEventListener('click', () => {
    if (!state.dataset.total()) return toast('No samples recorded yet.');
    download('signlens-dataset.csv', state.dataset.toCSV(), 'text/csv');
  });
  $('import-csv').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const n = state.dataset.importCSV(await file.text());
    await state.dataset.save();
    renderLabelGrid();
    toast(`Imported ${n} samples.`);
    e.target.value = '';
  });
  $('export-model-btn').addEventListener('click', () => {
    const model = state.userModel || state.shippedModel;
    if (!model) return toast('Train a model first.');
    download('weights.json', JSON.stringify(model.toJSON()), 'application/json');
  });
  $('import-model').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      const json = JSON.parse(await file.text());
      const model = MlpClassifier.fromJSON(json);
      if (state.userModel) state.userModel.dispose();
      state.userModel = model;
      await store.set('user-model', json);
      logTrain(describeModel(model, 'Imported model'));
      updateEngineBadge();
      toast('Model imported.');
    } catch (err) {
      toast('Invalid model file: ' + err.message);
    }
    e.target.value = '';
  });

  document.addEventListener('keydown', (e) => {
    if (state.mode !== 'translate') return;
    const tag = document.activeElement?.tagName;
    if (tag === 'TEXTAREA' || tag === 'INPUT' || tag === 'SELECT' || tag === 'BUTTON') return;
    if (e.key === ' ') { e.preventDefault(); addSpace(); }
    else if (e.key === 'Backspace') { e.preventDefault(); backspace(); }
  });
}
