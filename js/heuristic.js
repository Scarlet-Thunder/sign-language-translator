// Rule-based "starter" recogniser for static ASL fingerspelling.
// It works with zero training data so the app is usable on first load, but it
// is approximate. Record your own samples in Train mode (or ship a model from
// training/) for real accuracy. J and Z involve motion and are not covered.

import { toPoints } from './features.js';

const FINGERS = {
  index: [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring: [13, 14, 15, 16],
  pinky: [17, 18, 19, 20],
};

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const norm = (v) => Math.hypot(v[0], v[1], v[2]);
const clamp01 = (x) => Math.max(0, Math.min(1, x));
function angleDeg(u, v) {
  const d = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / ((norm(u) * norm(v)) || 1e-6);
  return (Math.acos(Math.max(-1, Math.min(1, d))) * 180) / Math.PI;
}
// Fuzzy comparisons: 1 when clearly true, 0 when clearly false, linear in between.
const lt = (x, t, w) => clamp01(0.5 - (x - t) / w);
const gt = (x, t, w) => clamp01(0.5 + (x - t) / w);
const all = (...xs) => Math.min(...xs);

function measure(landmarks, aspect) {
  // Image-space 3D points: y grows downwards. We avoid left/right specific
  // rules so this works for either hand and with mirrored or raw video.
  const p = toPoints(landmarks, aspect);
  const palm = norm(sub(p[9], p[0])) || 1e-6;
  const d = (i, j) => norm(sub(p[i], p[j])) / palm;

  const bend = {}, hookBend = {}, mcpBend = {}, dir = {};
  for (const [name, [mcp, pip, dip, tip]] of Object.entries(FINGERS)) {
    const a = angleDeg(sub(p[mcp], p[0]), sub(p[pip], p[mcp]));
    const b = angleDeg(sub(p[pip], p[mcp]), sub(p[dip], p[pip]));
    const c = angleDeg(sub(p[dip], p[pip]), sub(p[tip], p[dip]));
    bend[name] = a + b + c;
    mcpBend[name] = a;
    hookBend[name] = b + c;
    const v = sub(p[tip], p[mcp]);
    const l = Math.hypot(v[0], v[1]) || 1e-6;
    dir[name] = [v[0] / l, v[1] / l];
  }
  const tv = sub(p[4], p[2]);
  const tl = Math.hypot(tv[0], tv[1]) || 1e-6;
  const thumbDir = [tv[0] / tl, tv[1] / tl];
  const thumbStraight = angleDeg(sub(p[3], p[2]), sub(p[4], p[3]));
  const thumbVsIndex = angleDeg(sub(p[4], p[2]), sub(p[8], p[5]));
  return { p, d, bend, hookBend, mcpBend, dir, thumbDir, thumbStraight, thumbVsIndex };
}

export function classifyHeuristic(landmarks, aspect) {
  const m = measure(landmarks, aspect);
  const { d, bend, hookBend, mcpBend, dir, p } = m;

  const E = (f) => lt(bend[f], 60, 40);           // extended
  const C = (f) => gt(bend[f], 120, 60);          // curled
  const H = (f) => all(gt(bend[f], 50, 40), lt(bend[f], 175, 60)); // half bent
  const up = (f) => lt(dir[f][1], -0.6, 0.4);
  const down = (f) => gt(dir[f][1], 0.45, 0.4);
  const side = (f) => gt(Math.abs(dir[f][0]), 0.75, 0.3);
  const thumbOut = all(gt(d(4, 5), 0.55, 0.3), lt(m.thumbStraight, 40, 30));
  const thumbIn = lt(d(4, 5), 0.6, 0.3);
  const spread = d(8, 12) / (d(5, 9) || 1e-6);
  const crossed = Math.sign(p[8][0] - p[12][0]) !== Math.sign(p[5][0] - p[9][0]) && d(8, 12) < 0.5;
  // Fist letters bend the index at the knuckle; X keeps the knuckle straight.
  const fist = all(C('index'), C('middle'), C('ring'), C('pinky'), gt(mcpBend.index, 35, 30));
  const tipReach = (d(8, 5) + d(12, 9) + d(16, 13) + d(20, 17)) / 4;

  const s = {
    A: all(fist, lt(d(4, 6), 0.5, 0.25), gt(d(4, 10) - d(4, 6), 0.12, 0.15)),
    B: all(E('index'), E('middle'), E('ring'), E('pinky'), up('middle'), lt(spread, 1.5, 0.5),
      lt(Math.min(d(4, 9), d(4, 13)), 0.6, 0.3)),
    C: all(H('index'), H('middle'), H('ring'), H('pinky'), gt(d(4, 8), 0.4, 0.2), lt(d(4, 8), 1.1, 0.3),
      gt(tipReach, 0.5, 0.25)),
    D: all(E('index'), up('index'), C('middle'), C('ring'), C('pinky'), lt(d(4, 12), 0.4, 0.25)),
    E: all(fist, lt(Math.min(d(4, 12), d(4, 16), d(4, 8)), 0.4, 0.25), lt(tipReach, 0.5, 0.25),
      gt(d(4, 10), 0.25, 0.2)),
    F: all(lt(d(4, 8), 0.3, 0.2), E('middle'), E('ring'), E('pinky'), gt(bend.index, 60, 40)),
    G: all(E('index'), side('index'), C('middle'), C('ring'), C('pinky'),
      gt(Math.abs(m.thumbDir[0]), 0.6, 0.3)),
    H: all(E('index'), E('middle'), side('index'), side('middle'), C('ring'), C('pinky')),
    I: all(E('pinky'), C('index'), C('middle'), C('ring'), 1 - thumbOut),
    K: all(E('index'), E('middle'), up('index'), gt(spread, 1.3, 0.4), C('ring'), C('pinky'),
      lt(d(4, 10), 0.4, 0.25)),
    L: all(E('index'), up('index'), C('middle'), C('ring'), C('pinky'), thumbOut,
      gt(m.thumbVsIndex, 60, 30)),
    M: all(fist, lt(d(4, 18), 0.45, 0.25), lt(d(4, 18), d(4, 10), 0.1)),
    N: all(fist, lt(d(4, 14), 0.4, 0.25), gt(d(4, 18), 0.45, 0.2), gt(d(4, 6), 0.4, 0.2)),
    O: all(H('index'), H('middle'), H('ring'), lt(d(4, 8), 0.3, 0.2), lt(d(4, 12), 0.45, 0.25),
      gt(tipReach, 0.45, 0.2)),
    P: all(E('index'), down('index'), gt(bend.middle, 20, 40), C('ring'), C('pinky'),
      lt(d(4, 10), 0.45, 0.25)),
    Q: all(E('index'), down('index'), C('middle'), C('ring'), C('pinky'), gt(m.thumbDir[1], 0.3, 0.4)),
    R: all(E('index'), E('middle'), up('index'), crossed ? 1 : 0, C('ring'), C('pinky')),
    S: all(fist, lt(d(4, 10), 0.5, 0.25), lt(d(4, 10) - d(4, 6), 0, 0.15), gt(d(4, 18), 0.45, 0.2)),
    T: all(fist, lt(d(4, 6), 0.3, 0.2), lt(d(4, 10), 0.42, 0.2), lt(Math.abs(d(4, 6) - d(4, 10)), 0.15, 0.15)),
    U: all(E('index'), E('middle'), up('index'), up('middle'), lt(spread, 1.25, 0.35), crossed ? 0 : 1,
      C('ring'), C('pinky')),
    V: all(E('index'), E('middle'), up('index'), gt(spread, 1.7, 0.5), C('ring'), C('pinky'),
      gt(d(4, 10), 0.4, 0.25)),
    W: all(E('index'), E('middle'), E('ring'), C('pinky'), up('middle')),
    X: all(lt(mcpBend.index, 45, 40), gt(hookBend.index, 100, 50), C('middle'), C('ring'), C('pinky')),
    Y: all(E('pinky'), C('index'), C('middle'), C('ring'), thumbOut),
    // Open hand with all five fingers spread = word break.
    SPACE: all(E('index'), E('middle'), E('ring'), E('pinky'), thumbOut, gt(spread, 1.6, 0.5)),
  };
  // Thumb tucked inside a fist only counts for A if it is not hidden.
  s.A = Math.min(s.A, 1 - thumbIn * 0.3);

  const ranked = Object.entries(s).sort((a, b) => b[1] - a[1]);
  return {
    label: ranked[0][1] >= 0.45 ? ranked[0][0] : null,
    confidence: ranked[0][1],
    top: ranked.slice(0, 3).map(([label, v]) => ({ label, p: v })),
  };
}
