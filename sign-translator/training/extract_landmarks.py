"""Run MediaPipe Hand Landmarker over an image dataset and write a landmark CSV.

Expected layout (one folder per sign, e.g. the Kaggle "ASL Alphabet" dataset):

    data/asl_alphabet_train/A/*.jpg
    data/asl_alphabet_train/B/*.jpg
    ...
    data/asl_alphabet_train/space/*.jpg   -> label SPACE
    data/asl_alphabet_train/del/*.jpg     -> label DEL

Usage:
    python extract_landmarks.py data/asl_alphabet_train out/landmarks.csv --per-class 600
"""
import argparse
import random
import urllib.request
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from common import normalize, write_csv

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RENAME = {"space": "SPACE", "del": "DEL", "delete": "DEL"}
SKIP = {"nothing"}
MOTION = {"J", "Z"}


def label_for(folder: str) -> str:
    return RENAME.get(folder.lower(), folder.upper())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image_root", type=Path)
    ap.add_argument("out_csv", type=Path)
    ap.add_argument("--per-class", type=int, default=0, help="max images per class (0 = all)")
    ap.add_argument("--include-motion", action="store_true", help="keep J and Z (static frames only)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        print("Downloading hand_landmarker.task ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
    rng = random.Random(args.seed)
    rows, stats = [], {}
    with vision.HandLandmarker.create_from_options(options) as detector:
        for folder in sorted(p for p in args.image_root.iterdir() if p.is_dir()):
            if folder.name.lower() in SKIP:
                continue
            label = label_for(folder.name)
            if label in MOTION and not args.include_motion:
                continue
            files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            rng.shuffle(files)
            if args.per_class:
                files = files[: args.per_class]
            ok = 0
            for path in files:
                try:
                    image = mp.Image.create_from_file(str(path))
                except Exception:
                    continue
                result = detector.detect(image)
                if not result.hand_landmarks:
                    continue
                handed = result.handedness[0][0].category_name
                rows.append((label, normalize(result.hand_landmarks[0], handed, image.width / image.height)))
                ok += 1
            stats[label] = (ok, len(files))
            print(f"{label:>6}: {ok}/{len(files)} images had a detectable hand")

    write_csv(args.out_csv, rows)
    print(f"\nWrote {len(rows)} samples across {len(stats)} classes to {args.out_csv}")


if __name__ == "__main__":
    main()
