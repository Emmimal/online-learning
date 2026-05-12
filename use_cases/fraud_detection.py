"""
use_cases/fraud_detection.py
============================
Online learning for fraud detection — H2 section in Article 06.

This module demonstrates why online learning is the correct architecture
for fraud detection:

    1. Fraud patterns change continuously. A batch model retrained monthly
       has a one-month blind spot between the moment a new fraud pattern
       emerges and the moment the next retraining run captures it.

    2. Fraud data is severely class-imbalanced. Online learners can update
       their internal statistics on each transaction as it arrives —
       including rare fraud events — without waiting for a full retraining
       run to accumulate enough fraud examples.

    3. Labels arrive with a delay. A transaction flagged as fraud may not
       have a confirmed label for days (waiting for the customer dispute
       process). Online learners can be updated the moment a label arrives,
       regardless of when the transaction occurred.

Architecture used
-----------------
    RiverAdaptiveRF (primary)  — drift-aware ensemble, handles the class
                                 imbalance shift that occurs at the drift point
    RiverHoeffdingTree          — lightweight baseline
    ADWIN                       — drift detector
    PrequentialEvaluator        — test-then-train evaluation

Metrics reported
----------------
Because fraud datasets are highly imbalanced (1–3% fraud), raw accuracy
is misleading. This module reports:
  - Window precision, recall, and F1 for the fraud class (y=1)
  - Cumulative accuracy (for comparison with the benchmark)
  - Detection lag: how many samples after the drift the detector fires

Usage
-----
    python use_cases/fraud_detection.py

    # Or import and run programmatically:
    from use_cases.fraud_detection import run_fraud_demo
    results = run_fraud_demo(n_samples=20_000, verbose=True)
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

from data.generators import FraudStream
from methods.drift_detector import ADWIN
from methods.river_learner import RiverAdaptiveRF, RiverHoeffdingTree
from evaluation.prequential import OnlineMetrics


# ---------------------------------------------------------------------------
# Imbalance-aware prequential loop for fraud
# ---------------------------------------------------------------------------

def _prequential_fraud(
    model,
    stream: FraudStream,
    detector: ADWIN,
    window_size: int = 1_000,
    verbose: bool = True,
) -> Dict:
    """
    Prequential evaluation loop specialised for imbalanced fraud data.

    Reports per-window precision, recall, and F1 for the fraud class
    rather than raw accuracy (which is inflated by the majority non-fraud
    class).

    Returns
    -------
    dict with keys:
      window_accuracies, window_f1s, window_precisions, window_recalls,
      drift_detections, cumulative_acc, runtime_s, n_samples
    """
    window_tp = window_fp = window_fn = window_tn = 0
    window_correct = 0
    total_correct = total_n = 0
    drift_log: List[Tuple[int, str]] = []
    window_f1s: List[float] = []
    window_precs: List[float] = []
    window_recs: List[float] = []
    window_accs: List[float] = []

    t_start = time.perf_counter()

    for i, (x, y) in enumerate(stream):
        # Step 1: Predict
        y_pred = model.predict_one(x)

        # Step 2: Accumulate per-class counts
        correct = int(y_pred == y)
        window_correct += correct
        total_correct += correct
        total_n += 1

        if y == 1 and y_pred == 1:
            window_tp += 1
        elif y == 0 and y_pred == 1:
            window_fp += 1
        elif y == 1 and y_pred == 0:
            window_fn += 1
        else:
            window_tn += 1

        # Step 3: Drift detection
        detector.update(y, y_pred)
        if detector.drift_detected:
            drift_log.append((i, "ADWIN"))
            if verbose:
                print(f"    ⚠ Fraud pattern shift detected at sample {i:,}")

        # Step 4: Learn
        model.learn_one(x, y)

        # Step 5: Window boundary metrics
        if (i + 1) % window_size == 0:
            precision = window_tp / (window_tp + window_fp) if (window_tp + window_fp) > 0 else 0.0
            recall = window_tp / (window_tp + window_fn) if (window_tp + window_fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0
            )
            acc = window_correct / window_size

            window_accs.append(acc)
            window_f1s.append(f1)
            window_precs.append(precision)
            window_recs.append(recall)

            if verbose:
                print(
                    f"  [n={i+1:>7,}]  "
                    f"acc={acc:.4f}  "
                    f"precision={precision:.3f}  "
                    f"recall={recall:.3f}  "
                    f"f1={f1:.3f}"
                )

            # Reset window
            window_tp = window_fp = window_fn = window_tn = 0
            window_correct = 0

    return {
        "window_accuracies": window_accs,
        "window_f1s": window_f1s,
        "window_precisions": window_precs,
        "window_recalls": window_recs,
        "drift_detections": drift_log,
        "cumulative_acc": total_correct / total_n if total_n > 0 else 0.0,
        "runtime_s": time.perf_counter() - t_start,
        "n_samples": total_n,
    }


# ---------------------------------------------------------------------------
# Main demo function
# ---------------------------------------------------------------------------

def run_fraud_demo(
    n_samples: int = 20_000,
    drift_at: int = 10_000,
    fraud_rate_before: float = 0.01,
    fraud_rate_after: float = 0.03,
    window_size: int = 1_000,
    verbose: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Demonstrate online fraud detection with concept drift.

    Two models are compared:
      - RiverHoeffdingTree  (single tree, no built-in drift handling)
      - RiverAdaptiveRF     (ensemble with per-tree ADWIN drift detection)

    A separate ADWIN detector is attached to the Hoeffding Tree to show
    how explicit drift monitoring works alongside a model that does not
    handle drift internally.

    Parameters
    ----------
    n_samples         : Total stream length.
    drift_at          : Sample index where the fraud pattern shifts.
    fraud_rate_before : Fraud prevalence before drift (e.g. 0.01 = 1%).
    fraud_rate_after  : Fraud prevalence after drift (e.g. 0.03 = 3%).
    window_size       : Prequential evaluation window.
    verbose           : Print per-window metrics.
    seed              : Random seed.

    Returns
    -------
    dict with results for both models.
    """
    print("=" * 70)
    print("  FRAUD DETECTION: Online Learning with Concept Drift")
    print(f"  Stream: {n_samples:,} transactions | "
          f"Drift at sample {drift_at:,}")
    print(f"  Fraud rate: {fraud_rate_before:.0%} → {fraud_rate_after:.0%} "
          f"at drift point")
    print("=" * 70)

    results = {}

    # ── Model 1: Hoeffding Tree + explicit ADWIN detector ────────────────
    print("\n[1] Hoeffding Tree + ADWIN Detector")
    print("-" * 40)

    stream_ht = FraudStream(
        n_samples=n_samples,
        fraud_rate=fraud_rate_before,
        fraud_rate_after=fraud_rate_after,
        drift_at=drift_at,
        seed=seed,
    )
    ht_model = RiverHoeffdingTree(grace_period=100)
    ht_detector = ADWIN(delta=0.002)

    ht_results = _prequential_fraud(
        model=ht_model,
        stream=stream_ht,
        detector=ht_detector,
        window_size=window_size,
        verbose=verbose,
    )
    results["HoeffdingTree"] = ht_results

    # ── Model 2: Adaptive Random Forest (internal drift handling) ─────────
    print(f"\n[2] Adaptive Random Forest (internal ADWIN per tree)")
    print("-" * 40)

    stream_arf = FraudStream(
        n_samples=n_samples,
        fraud_rate=fraud_rate_before,
        fraud_rate_after=fraud_rate_after,
        drift_at=drift_at,
        seed=seed,
    )
    arf_model = RiverAdaptiveRF(n_models=10, seed=seed)
    arf_detector = ADWIN(delta=0.002)

    arf_results = _prequential_fraud(
        model=arf_model,
        stream=stream_arf,
        detector=arf_detector,
        window_size=window_size,
        verbose=verbose,
    )
    results["AdaptiveRF"] = arf_results

    # ── Summary ───────────────────────────────────────────────────────────
    _print_fraud_summary(results, drift_at, window_size)

    return results


def _print_fraud_summary(
    results: Dict,
    drift_at: int,
    window_size: int,
) -> None:
    """Print a formatted comparison table."""
    import numpy as np

    print("\n" + "=" * 70)
    print("  FRAUD DETECTION RESULTS SUMMARY")
    print("=" * 70)
    header = f"{'Method':<22}  {'Cum Acc':>8}  {'Mean F1':>8}  "
    header += f"{'Post-drift F1':>13}  {'Detections':>10}"
    print(header)
    print("-" * 70)

    for name, r in results.items():
        f1s = r["window_f1s"]
        # Determine which windows are post-drift
        post_drift_start = drift_at // window_size
        post_drift_f1s = f1s[post_drift_start:] if post_drift_start < len(f1s) else f1s

        cum_acc = r["cumulative_acc"]
        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        post_f1 = float(np.mean(post_drift_f1s)) if post_drift_f1s else 0.0
        n_det = len(r["drift_detections"])

        print(
            f"  {name:<20}  {cum_acc:>8.4f}  {mean_f1:>8.4f}  "
            f"{post_f1:>13.4f}  {n_det:>10}"
        )

    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_fraud_demo(n_samples=20_000, verbose=True)
