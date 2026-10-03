// Browser self-test: open tests/selftest.html via a local server.
// Uses synthetic hand landmarks, so it checks the plumbing (features, rules,
// training, model round-trip, CSV, commit logic), not real-world accuracy.
import { classifyHeuristic } from '../js/heuristic.js';
import { normalize, augment } from '../js/features.js';
import { MlpClassifier } from '../js/classifier.js';
import { Dataset } from '../js/dataset.js';
import { Committer } from '../js/committer.js';

const E = [5, 5, 5], C = [80, 100, 60];
const THUMB = {
  out: [[0.39, 0.76], [0.34, 0.75], [0.29, 0.75]],
  side: [[0.42, 0.72], [0.415, 0.66], [0.42, 0.61]],
  across: [[0.45, 0.72], [0.49, 0.68], [0.53, 0.66]],
};

/** Synthetic hand in image coords. bends: [mcp, pip, dip] degrees per finger, curling toward the camera. */
export function hand({ bends, spread = 0, thumb = 'across' }) {
  const lm = new Array(21);
  const P = (x, y, z = 0) => ({ x, y, z });
  lm[0] = P(0.5, 0.85);
  lm[1] = P(0.44, 0.78);
  const bases = [[0.45, 0.62], [0.49, 0.6], [0.53, 0.61], [0.565, 0.635]];
  const lens = [[0.075, 0.045, 0.035], [0.08, 0.05, 0.038], [0.075, 0.047, 0.035], [0.06, 0.035, 0.03]];
  for (let f = 0; f < 4; f++) {
    let p = [...bases[f], 0];
    lm[5 + f * 4] = P(...p);
    const phi = (f - 1.5) * spread;
    let th = 0;
    for (let j = 0; j < 3; j++) {
      th += (bends[f][j] * Math.PI) / 180;
      const d = [Math.sin(phi) * Math.cos(th), -Math.cos(phi) * Math.cos(th), -Math.sin(th)];
      p = p.map((v, i) => v + d[i] * lens[f][j]);
      lm[6 + f * 4 + j] = P(...p);
    }
  }
  THUMB[thumb].forEach(([x, y], i) => { lm[2 + i] = P(x, y, thumb === 'across' ? -0.02 * i : 0); });
  return lm;
}

export const SHAPES = {
  B: { bends: [E, E, E, E] },
  SPACE: { bends: [E, E, E, E], spread: 0.18, thumb: 'out' },
  A: { bends: [C, C, C, C], thumb: 'side' },
  L: { bends: [E, C, C, C], thumb: 'out' },
  U: { bends: [E, E, C, C] },
  V: { bends: [E, E, C, C], spread: 0.2 },
  W: { bends: [E, E, E, C], spread: 0.15 },
  I: { bends: [C, C, C, E] },
  Y: { bends: [C, C, C, E], thumb: 'out' },
  X: { bends: [[10, 90, 70], C, C, C] },
};

export async function run(log) {
  let pass = 0, fail = 0;
  const check = (name, ok, detail = '') => { ok ? pass++ : fail++; log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  ' + detail : ''}`); };

  for (const [label, shape] of Object.entries(SHAPES)) {
    const r = classifyHeuristic(hand(shape), 4 / 3);
    check(`rules: ${label}`, r.label === label, `got ${r.label} (${r.confidence.toFixed(2)})`);
  }

  const v = normalize(hand(SHAPES.B), 'Right', 4 / 3);
  const vl = normalize(hand(SHAPES.B), 'Left', 4 / 3);
  check('features: 63 dims, finite', v.length === 63 && v.every(Number.isFinite));
  check('features: Left mirrors x', Math.abs(v[24] + vl[24]) < 1e-9 && v[24] !== 0);
  check('features: middle MCP at unit distance', Math.abs(Math.hypot(v[27], v[28], v[29]) - 1) < 1e-9);

  const ds = new Dataset();
  const labels = ['A', 'B', 'V', 'Y'];
  for (const l of labels) {
    const base = normalize(hand(SHAPES[l]), 'Right', 4 / 3);
    for (let i = 0; i < 60; i++) ds.add(l, augment(base));
  }
  const ds2 = new Dataset();
  check('dataset: CSV round trip', ds2.importCSV(ds.toCSV()) === 240 && ds2.count('V') === 60);

  const t0 = performance.now();
  const model = await MlpClassifier.train(ds2.samples, { epochs: 15 });
  check('model: trains', model.labels.join() === labels.join(), `${Math.round(performance.now() - t0)} ms, val acc ${model.meta.metrics.valAccuracy?.toFixed(3)}`);
  const model2 = MlpClassifier.fromJSON(JSON.parse(JSON.stringify(model.toJSON())));
  for (const l of labels) {
    const vec = normalize(hand(SHAPES[l]), 'Right', 4 / 3);
    const a = model.predict(vec), b = model2.predict(vec);
    check(`model: predicts ${l}`, a.label === l && b.label === l && Math.abs(a.confidence - b.confidence) < 1e-4,
      `${a.label} ${a.confidence.toFixed(3)} / round-trip ${b.label} ${b.confidence.toFixed(3)}`);
  }
  model.dispose();
  model2.dispose();

  const c = new Committer({ holdMs: 500, minConfidence: 0.6 });
  const commits = [];
  for (let t = 0; t < 3000; t += 33) {
    const pred = t < 1000 ? { label: 'L', confidence: 0.9 } : t < 1300 ? null : { label: 'L', confidence: 0.9 };
    const r = c.update(pred, t);
    if (r.commit) commits.push(r.commit);
  }
  check('committer: commits once per hold, repeats after release', commits.join('') === 'LL', commits.join(''));

  log(`\n${pass} passed, ${fail} failed`);
  return { pass, fail };
}
