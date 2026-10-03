# Bundled model (optional)

Put a `weights.json` here and every visitor gets it automatically. It is used
whenever a visitor hasn't trained their own model in the Train tab.

Two ways to produce it:

1. **In the browser:** Train tab → Train model → Import / export → *Export model (JSON)*.
2. **Offline, on a full dataset:** `python training/train.py ...` (see `training/README.md`).

If this file is missing, the app falls back to the built-in starter rules.
