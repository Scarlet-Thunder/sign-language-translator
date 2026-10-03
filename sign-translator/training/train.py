"""Train the landmark classifier and export it for the web app.

Usage:
    python train.py out/landmarks.csv [more.csv ...] --out ../model/weights.json

Inputs are CSVs from extract_landmarks.py and/or "Export dataset (CSV)" in the
app's Train tab. Mixing both (public dataset + your own webcam recordings) gives
the best real-world results.

Prints held-out test accuracy and a per-class report. Use THAT number, measured
on your data, for any accuracy claims.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from common import FEATURE_VERSION, NUM_FEATURES, augment, read_csvs


def build_model(num_classes: int) -> tf.keras.Model:
    # Same architecture as MlpClassifier.train in js/classifier.js.
    model = tf.keras.Sequential([
        tf.keras.layers.Input((NUM_FEATURES,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.003),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def export_signlens(model: tf.keras.Model, labels: list[str], metrics: dict, path: Path) -> None:
    layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            kernel, bias = layer.get_weights()
            layers.append({
                "units": int(layer.units),
                "activation": layer.get_config()["activation"],
                "kernel": np.round(kernel, 6).tolist(),
                "bias": np.round(bias, 6).tolist(),
            })
    payload = {
        "format": "signlens-mlp",
        "version": 1,
        "featureVersion": FEATURE_VERSION,
        "labels": labels,
        "layers": layers,
        "metrics": metrics,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "model" / "weights.json")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--min-per-class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    X, y_raw = read_csvs(args.csv)
    counts = {l: y_raw.count(l) for l in set(y_raw)}
    labels = sorted(l for l, c in counts.items() if c >= args.min_per_class)
    keep = np.array([l in labels for l in y_raw])
    X = X[keep]
    y = np.array([labels.index(l) for l, k in zip(y_raw, keep) if k])
    print(f"{len(X)} samples, {len(labels)} classes: {' '.join(labels)}")
    dropped = sorted(set(counts) - set(labels))
    if dropped:
        print(f"Dropped (fewer than {args.min_per_class} samples): {dropped}")

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=args.seed)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=args.seed)

    # Two augmented copies of the training set.
    X_aug = np.concatenate([X_train, augment(X_train, rng), augment(X_train, rng)])
    y_aug = np.concatenate([y_train, y_train, y_train])

    model = build_model(len(labels))
    model.fit(
        X_aug, y_aug,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=64,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6),
        ],
        verbose=2,
    )

    pred = model.predict(X_test, verbose=0).argmax(axis=1)
    test_acc = float((pred == y_test).mean())
    val_acc = float((model.predict(X_val, verbose=0).argmax(axis=1) == y_val).mean())
    print("\n" + classification_report(y_test, pred, target_names=labels, digits=3))
    print(f"Held-out test accuracy: {test_acc:.4f}  ({len(y_test)} samples)")

    cm = confusion_matrix(y_test, pred)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=90)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix (test acc {test_acc:.1%})")
        fig.tight_layout()
        cm_path = args.out.with_name("confusion_matrix.png")
        fig.savefig(cm_path, dpi=120)
        print(f"Saved {cm_path}")
    except ImportError:
        pass

    metrics = {"valAccuracy": val_acc, "testAccuracy": test_acc, "samples": int(len(X)), "classes": len(labels)}
    export_signlens(model, labels, metrics, args.out)
    print(f"Exported model to {args.out}")


if __name__ == "__main__":
    main()
