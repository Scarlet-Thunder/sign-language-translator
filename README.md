# SignLens: real-time sign language translator

A browser app that turns ASL fingerspelling from your webcam into text, live.
It runs entirely on-device: MediaPipe tracks 21 hand landmarks, and a small
TensorFlow.js neural network classifies them. No video or images ever leave the browser.

**Stack:** MediaPipe Hand Landmarker (WASM/WebGL) · TensorFlow.js · vanilla JS (no build step) ·
Python + TensorFlow/Keras for offline training.

## Features

- **Live translation**: letters commit after you hold a sign steady (progress ring), with auto-space when your hand leaves the frame, an open-hand "space" gesture, text-to-speech, and copy.
- **Works on first load**: built-in starter rules recognise most static letters with zero training.
- **Train in the browser**: record your own samples per letter, then train a neural net in seconds. It saves to IndexedDB and shows validation accuracy.
- **Offline training pipeline**: extract landmarks from a public image dataset, train with Keras, and export one `model/weights.json` that every visitor gets.
- **Private by design**: the camera feed is processed locally. Models and samples live in the browser's own storage.
- Left or right hand, mirrored or raw video, camera picker, adjustable hold time and confidence, light/dark theme, responsive layout.

## Run locally

The camera API requires `https://` or `localhost`. Serve the folder with any static server:

```bash
npx serve .
```

```bash
python -m http.server 8080
```

On Windows without Python/Node: `powershell -ExecutionPolicy Bypass -File serve.ps1`, then open http://localhost:8080.

Self-test (plumbing checks with synthetic hands): http://localhost:8080/tests/selftest.html

## Deploy (static hosting, free, HTTPS included)

- **Netlify:** drag the folder onto https://app.netlify.com/drop, or connect the repo (`netlify.toml` included).
- **Vercel:** `npx vercel` in this folder (`vercel.json` included).
- **GitHub Pages:** in the repo, go to Settings → Pages → Build and deployment → Source: **Deploy from a branch**, Branch: `main`, folder `/ (root)`. The site goes live at `https://<user>.github.io/<repo>/`.

## Getting real accuracy

The starter rules are a heuristic demo. For a reliable translator, use a trained model:

### Option A: train in the browser (5–10 min)
1. Open the **Train** tab and start the camera.
2. For each letter, press **Record 150** while holding the sign. Vary distance and angle slightly.
3. Press **Train model**. The model is saved automatically and the Translate tab uses it.
4. Optional: **Export model (JSON)**, save it as `model/weights.json`, and redeploy so all visitors get it.

### Option B: train offline on a public dataset (more signers, so it generalises better)
```bash
cd training
pip install -r requirements.txt
# e.g. Kaggle "ASL Alphabet" (folders A..Z, space, del, nothing)
python extract_landmarks.py path/to/asl_alphabet_train out/landmarks.csv --per-class 600
# optionally add your own webcam recordings exported from the Train tab
python train.py out/landmarks.csv signlens-dataset.csv --out ../model/weights.json
```
`train.py` prints a held-out **test accuracy**, a per-class report, and saves `model/confusion_matrix.png`.
Mixing a public dataset with a few hundred of your own webcam samples closes the gap between dataset photos and live video.

> **Resume tip:** quote the numbers your own run prints (test accuracy, number of classes, and the fps shown in the app's header on your machine), not placeholder figures.

## How it works

```
webcam ─► MediaPipe Hand Landmarker ─► 21 × (x,y,z) ─► normalise ─► classifier ─► smoothing ─► text
                                                       (wrist-centred,        (MLP 63→128→64→N   (majority vote
                                                        palm-scaled,           or starter rules)  over 10 frames
                                                        left hand mirrored)                       + hold timer)
```

- `js/features.js` and `training/common.py` implement the **same** normalisation, so models trained in either place are interchangeable.
- Model files use a tiny JSON format (`signlens-mlp`: labels + Dense layer weights), so there's no `tensorflowjs` converter dependency.
- J and Z are drawn in the air (motion), so this static-pose classifier excludes them. Supporting them needs a sequence model (e.g. an LSTM over landmark sequences), which is a natural next step.

## Project structure

```
index.html            App shell
css/styles.css        Styles (light/dark)
js/app.js             UI, camera, main loop, train mode
js/hand.js            MediaPipe wrapper + landmark drawing
js/features.js        Landmark normalisation + augmentation
js/heuristic.js       Zero-training starter rules
js/classifier.js      TF.js MLP: train / predict / save / load
js/committer.js       Temporal smoothing → committed letters
js/dataset.js         Recorded samples (IndexedDB, CSV import/export)
js/storage.js         IndexedDB + preferences helpers
model/                Optional bundled weights.json
training/             Python landmark extraction + Keras training
tests/                Browser self-test
```

## Limitations

- ASL fingerspelling only (static letters + space/delete), not full signed sentences or grammar.
- Lighting, camera angle and occlusion (thumb hidden in M/N/S/T) affect landmark quality.
- Models and recordings are per-browser unless you export them.
