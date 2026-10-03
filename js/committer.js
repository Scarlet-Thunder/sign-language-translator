// Turns a noisy per-frame prediction stream into committed letters.
// A letter is committed once it has been the clear majority of recent frames
// for `holdMs`. To repeat a letter (the "LL" in HELLO), relax the hand or
// change shape briefly, then sign it again.

export class Committer {
  constructor({ holdMs = 800, minConfidence = 0.7, windowSize = 10 } = {}) {
    this.holdMs = holdMs;
    this.minConfidence = minConfidence;
    this.windowSize = windowSize;
    this.reset();
  }

  reset() {
    this.window = [];
    this.current = null;
    this.since = 0;
    this.committed = false;
  }

  /** pred: {label, confidence} or null when no hand. Returns {stable, progress, commit}. */
  update(pred, now) {
    const label = pred && pred.label && pred.confidence >= this.minConfidence ? pred.label : null;
    this.window.push(label);
    if (this.window.length > this.windowSize) this.window.shift();

    const counts = new Map();
    for (const l of this.window) counts.set(l, (counts.get(l) || 0) + 1);
    let stable = null, best = 0;
    for (const [l, c] of counts) if (c > best) { best = c; stable = l; }
    if (best / this.windowSize < 0.7) stable = null;

    if (stable !== this.current) {
      this.current = stable;
      this.since = now;
      this.committed = false;
    }
    const progress = stable && !this.committed ? Math.min(1, (now - this.since) / this.holdMs) : 0;
    let commit = null;
    if (stable && !this.committed && progress >= 1) {
      this.committed = true;
      commit = stable;
    }
    return { stable, progress, commit };
  }
}
