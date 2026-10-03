"""Feature transform shared with the web app.

Must stay identical to js/features.js. If you change it, change both and bump
FEATURE_VERSION.
"""
import csv
import math
from pathlib import Path

import numpy as np

FEATURE_VERSION = 1
NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3


def normalize(landmarks, handedness: str, aspect: float) -> list[float]:
    """landmarks: 21 objects with .x .y .z (MediaPipe normalized coords)."""
    pts = [(p.x * aspect, p.y, p.z * aspect) for p in landmarks]
    flip = -1.0 if handedness == "Left" else 1.0
    wx, wy, wz = pts[0]
    mx, my, mz = pts[9][0] - wx, pts[9][1] - wy, pts[9][2] - wz
    scale = math.sqrt(mx * mx + my * my + mz * mz) or 1e-6
    out = []
    for x, y, z in pts:
        out += [flip * (x - wx) / scale, (y - wy) / scale, (z - wz) / scale]
    return out


def augment(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random in-plane rotation, scale and jitter. Mirrors augment() in features.js."""
    n = X.shape[0]
    pts = X.reshape(n, NUM_LANDMARKS, 3).copy()
    theta = (rng.random(n) - 0.5) * 0.5
    k = 0.9 + rng.random(n) * 0.2
    c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
    x, y = pts[:, :, 0].copy(), pts[:, :, 1].copy()
    pts[:, :, 0] = (c * x - s * y) * k[:, None]
    pts[:, :, 1] = (s * x + c * y) * k[:, None]
    pts[:, :, 2] *= k[:, None]
    pts += (rng.random(pts.shape) - 0.5) * 0.04
    return pts.reshape(n, NUM_FEATURES)


def read_csvs(paths: list[Path]) -> tuple[np.ndarray, list[str]]:
    X, y = [], []
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.reader(f):
                if not row or row[0] == "label" or len(row) != NUM_FEATURES + 1:
                    continue
                X.append([float(v) for v in row[1:]])
                y.append(row[0])
    return np.asarray(X, dtype=np.float32), y


def write_csv(path: Path, rows: list[tuple[str, list[float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label"] + [f"f{i}" for i in range(NUM_FEATURES)])
        for label, vec in rows:
            w.writerow([label] + [f"{v:.4f}" for v in vec])
