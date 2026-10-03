// Landmark -> feature vector conversion.
// IMPORTANT: training/common.py implements the exact same transform. If you
// change anything here, change it there too and bump FEATURE_VERSION.

export const FEATURE_VERSION = 1;
export const NUM_LANDMARKS = 21;
export const NUM_FEATURES = NUM_LANDMARKS * 3;

/**
 * Convert MediaPipe landmarks into aspect-corrected 3D points.
 * MediaPipe normalises x by image width and y by image height, so we scale x
 * (and z, which shares x's scale) by width/height to get square units.
 */
export function toPoints(landmarks, aspect) {
  return landmarks.map((p) => [p.x * aspect, p.y, p.z * aspect]);
}

/**
 * Canonical feature vector: wrist-centred, scaled by palm size (wrist -> middle
 * MCP), mirrored so left and right hands share one representation.
 * Returns a plain Array of length 63.
 */
export function normalize(landmarks, handedness, aspect) {
  const pts = toPoints(landmarks, aspect);
  const flip = handedness === 'Left' ? -1 : 1;
  const [wx, wy, wz] = pts[0];
  const mx = pts[9][0] - wx, my = pts[9][1] - wy, mz = pts[9][2] - wz;
  const scale = Math.hypot(mx, my, mz) || 1e-6;
  const out = new Array(NUM_FEATURES);
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    out[i * 3] = (flip * (pts[i][0] - wx)) / scale;
    out[i * 3 + 1] = (pts[i][1] - wy) / scale;
    out[i * 3 + 2] = (pts[i][2] - wz) / scale;
  }
  return out;
}

/** Random rotation / scale / jitter, used to augment training samples. */
export function augment(vec, rand = Math.random) {
  const theta = (rand() - 0.5) * 0.5; // about +/-14 degrees in the image plane
  const c = Math.cos(theta), s = Math.sin(theta);
  const k = 0.9 + rand() * 0.2;
  const out = new Array(vec.length);
  for (let i = 0; i < vec.length; i += 3) {
    const x = vec[i], y = vec[i + 1];
    out[i] = (c * x - s * y) * k + (rand() - 0.5) * 0.04;
    out[i + 1] = (s * x + c * y) * k + (rand() - 0.5) * 0.04;
    out[i + 2] = vec[i + 2] * k + (rand() - 0.5) * 0.04;
  }
  return out;
}
