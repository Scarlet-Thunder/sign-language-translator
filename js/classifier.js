/* global tf */
// Small MLP over the 63-d landmark feature vector, built with TensorFlow.js.
//
// Model file format ("signlens-mlp", shared with training/train.py):
// {
//   "format": "signlens-mlp", "version": 1, "featureVersion": 1,
//   "labels": ["A", "B", ...],
//   "layers": [{ "units": 128, "activation": "relu", "kernel": [[...]], "bias": [...] }, ...],
//   "metrics": { "valAccuracy": 0.95, "samples": 5200 }
// }
// kernel has shape [inputs, units], same as Keras / TF.js Dense layers.

import { FEATURE_VERSION, NUM_FEATURES, augment } from './features.js';

export const MODEL_FORMAT = 'signlens-mlp';

export class MlpClassifier {
  constructor(model, labels, meta = {}) {
    this.model = model;
    this.labels = labels;
    this.meta = meta;
  }

  static fromJSON(json) {
    if (!json || json.format !== MODEL_FORMAT) throw new Error('Not a SignLens model file');
    if (json.featureVersion !== FEATURE_VERSION) {
      throw new Error(`Model uses feature version ${json.featureVersion}, app expects ${FEATURE_VERSION}`);
    }
    const model = tf.sequential();
    json.layers.forEach((l, i) => {
      model.add(tf.layers.dense({
        units: l.units,
        activation: l.activation,
        inputShape: i === 0 ? [NUM_FEATURES] : undefined,
      }));
    });
    const weights = [];
    json.layers.forEach((l) => {
      weights.push(tf.tensor2d(l.kernel), tf.tensor1d(l.bias));
    });
    model.setWeights(weights);
    weights.forEach((w) => w.dispose());
    return new MlpClassifier(model, json.labels, { metrics: json.metrics || {}, createdAt: json.createdAt });
  }

  toJSON() {
    const layers = this.model.layers
      .filter((l) => l.getClassName() === 'Dense')
      .map((l) => {
        const [k, b] = l.getWeights();
        return {
          units: l.units,
          activation: l.getConfig().activation,
          kernel: k.arraySync(),
          bias: Array.from(b.dataSync()),
        };
      });
    return {
      format: MODEL_FORMAT,
      version: 1,
      featureVersion: FEATURE_VERSION,
      labels: this.labels,
      layers,
      metrics: this.meta.metrics || {},
      createdAt: this.meta.createdAt || new Date().toISOString(),
    };
  }

  predict(vec) {
    const probs = tf.tidy(() => this.model.predict(tf.tensor2d([vec], [1, NUM_FEATURES])).dataSync());
    const ranked = Array.from(probs, (p, i) => ({ label: this.labels[i], p })).sort((a, b) => b.p - a.p);
    return { label: ranked[0].label, confidence: ranked[0].p, top: ranked.slice(0, 3) };
  }

  dispose() {
    this.model.dispose();
  }

  /**
   * Train from { label: [vec, ...] }. Labels with fewer than `minPerClass`
   * samples are skipped. Returns a new MlpClassifier.
   */
  static async train(samples, { epochs = 40, minPerClass = 10, onEpoch } = {}) {
    const labels = Object.keys(samples).filter((k) => samples[k].length >= minPerClass).sort();
    if (labels.length < 2) throw new Error(`Need at least 2 signs with ${minPerClass}+ samples each`);

    const trainX = [], trainY = [], valX = [], valY = [];
    labels.forEach((label, li) => {
      const vecs = shuffle(samples[label].slice());
      const nVal = Math.max(1, Math.round(vecs.length * 0.15));
      vecs.forEach((v, i) => {
        if (i < nVal) { valX.push(v); valY.push(li); return; }
        trainX.push(v); trainY.push(li);
        // Two augmented copies of each training sample.
        trainX.push(augment(v)); trainY.push(li);
        trainX.push(augment(v)); trainY.push(li);
      });
    });
    const order = shuffle([...trainX.keys()]);

    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [NUM_FEATURES] }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: labels.length, activation: 'softmax' }));
    model.compile({ optimizer: tf.train.adam(0.003), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    const xs = tf.tensor2d(order.map((i) => trainX[i]), [order.length, NUM_FEATURES]);
    const ys = tf.oneHot(tf.tensor1d(order.map((i) => trainY[i]), 'int32'), labels.length);
    const vxs = tf.tensor2d(valX, [valX.length, NUM_FEATURES]);
    const vys = tf.oneHot(tf.tensor1d(valY, 'int32'), labels.length);

    let last = {};
    try {
      await model.fit(xs, ys, {
        epochs,
        batchSize: 32,
        shuffle: true,
        yieldEvery: 'never', // built-in yielding uses rAF, which stalls in background tabs
        validationData: [vxs, vys],
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            last = logs;
            if (onEpoch) onEpoch(epoch + 1, epochs, logs);
            // Yield so the UI can repaint. setTimeout (unlike rAF) keeps running in background tabs.
            await new Promise((r) => setTimeout(r, 0));
          },
        },
      });
    } finally {
      tf.dispose([xs, ys, vxs, vys]);
    }

    const metrics = {
      valAccuracy: last.val_acc ?? last.val_accuracy ?? null,
      trainAccuracy: last.acc ?? last.accuracy ?? null,
      samples: labels.reduce((n, l) => n + samples[l].length, 0),
      classes: labels.length,
    };
    return new MlpClassifier(model, labels, { metrics, createdAt: new Date().toISOString() });
  }
}

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}
