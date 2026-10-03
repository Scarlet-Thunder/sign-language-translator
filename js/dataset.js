// Recorded landmark samples, persisted in IndexedDB.
// CSV layout (shared with training/): label,f0,f1,...,f62

import { FEATURE_VERSION, NUM_FEATURES } from './features.js';
import * as store from './storage.js';

const KEY = 'dataset-v' + FEATURE_VERSION;

export class Dataset {
  constructor() {
    this.samples = {};
  }

  async load() {
    const saved = await store.get(KEY);
    this.samples = (saved && saved.samples) || {};
    return this;
  }

  save() {
    return store.set(KEY, { featureVersion: FEATURE_VERSION, samples: this.samples });
  }

  add(label, vec) {
    (this.samples[label] ||= []).push(vec.map((x) => Math.round(x * 1e4) / 1e4));
  }

  count(label) {
    return (this.samples[label] || []).length;
  }

  total() {
    return Object.values(this.samples).reduce((n, v) => n + v.length, 0);
  }

  clear(label) {
    if (label) delete this.samples[label];
    else this.samples = {};
  }

  toCSV() {
    const header = ['label', ...Array.from({ length: NUM_FEATURES }, (_, i) => 'f' + i)].join(',');
    const rows = [header];
    for (const [label, vecs] of Object.entries(this.samples)) {
      for (const v of vecs) rows.push(label + ',' + v.join(','));
    }
    return rows.join('\n');
  }

  /** Merge samples from CSV text. Returns number of rows imported. */
  importCSV(text) {
    let n = 0;
    for (const line of text.split(/\r?\n/)) {
      const parts = line.trim().split(',');
      if (parts.length !== NUM_FEATURES + 1 || parts[0] === 'label') continue;
      const vec = parts.slice(1).map(Number);
      if (vec.some((x) => !Number.isFinite(x))) continue;
      (this.samples[parts[0]] ||= []).push(vec);
      n++;
    }
    return n;
  }
}
