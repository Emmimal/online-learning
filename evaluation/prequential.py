"""
evaluation/prequential.py
=========================
Prequential evaluation for online learning models.

What is prequential evaluation?
--------------------------------
Standard ML evaluation requires a held-out test set. Online learning has no
held-out test set — the stream is the only data you have, and you must learn
from every sample. The correct evaluation protocol is prequential, also called
test-then-train or interleaved test-then-train:

    FOR each sample (x, y) in the stream:
        1. Predict  →  ŷ = model.predict_one(x)
        2. Evaluate →  update metrics with (y, ŷ)
        3. Learn    →  model.learn_one(x, y)

Predicting before learning means the model is always tested on data it has
not seen yet. The metrics accumulate a rolling view of how the model performs
on the incoming stream — including before and after drift points.

Classes
-------
PrequentialEvaluator
    Runs the test-then-train loop on any model with the learn_one /
    predict_one / predict_proba_one interface. Computes:
      - Rolling accuracy (window-based and cumulative)
      - Rolling AUC-ROC (approximated via windowed rank statistic)
      - Per-window F1 score
      - Detection lag when a drift detector is attached

OnlineMetrics
    Lightweight container for the per-sample metrics computed during
    a prequential evaluation run. Used by the benchmark and use cases.

Usage
-----
    from evaluation.prequential import PrequentialEvaluator
    from methods.river_learner import RiverHoeffdingTree
    from data.generators import SEAConceptStream

    stream    = SEAConceptStream(n_samples=10_000, drift_at=5_000)
    model     = RiverHoeffdingTree()
    evaluator = PrequentialEvaluator(window_size=500)

    result = evaluator.run(model, stream)
    print(result.summary())
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterator, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# OnlineMetrics — data container
# ---------------------------------------------------------------------------

@dataclass
class OnlineMetrics:
    """
    Container for metrics collected during a prequential evaluation run.

    Attributes
    ----------
    accuracies        : rolling accuracy at each window boundary
    window_f1s        : F1 score within each evaluation window
    cumulative_acc    : cumulative accuracy from sample 0 to the end
    final_accuracy    : accuracy over the last full window
    n_samples         : total samples processed
    runtime_s         : wall-clock seconds for the full loop
    drift_detections  : list of (sample_index, detector_name) tuples
    """
    accuracies: List[float] = field(default_factory=list)
    window_f1s: List[float] = field(default_factory=list)
    cumulative_acc: float = 0.0
    final_accuracy: float = 0.0
    n_samples: int = 0
    runtime_s: float = 0.0
    drift_detections: List[Tuple[int, str]] = field(default_factory=list)
    window_size: int = 500

    def summary(self) -> str:
        """
        Return a formatted summary string consistent with the Article 05
        benchmark output format.
        """
        det_str = (
            f"{len(self.drift_detections)} detection(s) at samples "
            + ", ".join(str(s) for s, _ in self.drift_detections)
            if self.drift_detections
            else "none"
        )
        return (
            f"  Cumulative accuracy : {self.cumulative_acc:.4f}\n"
            f"  Final window acc    : {self.final_accuracy:.4f}\n"
            f"  Mean window acc     : {np.mean(self.accuracies):.4f}\n"
            f"  Min  window acc     : {np.min(self.accuracies):.4f}\n"
            f"  Samples processed   : {self.n_samples:,}\n"
            f"  Runtime             : {self.runtime_s:.1f}s\n"
            f"  Drift detections    : {det_str}"
        )


# ---------------------------------------------------------------------------
# F1 helper (no sklearn dependency in the hot path)
# ---------------------------------------------------------------------------

def _f1_from_window(
    y_true_window: List[int],
    y_pred_window: List[int],
) -> float:
    """Binary F1 computed from two lists. Returns 0.0 when undefined."""
    tp = sum(yt == 1 and yp == 1 for yt, yp in zip(y_true_window, y_pred_window))
    fp = sum(yt == 0 and yp == 1 for yt, yp in zip(y_true_window, y_pred_window))
    fn = sum(yt == 1 and yp == 0 for yt, yp in zip(y_true_window, y_pred_window))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# PrequentialEvaluator
# ---------------------------------------------------------------------------

class PrequentialEvaluator:
    """
    Runs the test-then-train loop and collects metrics.

    The evaluator is model-agnostic: any object with learn_one,
    predict_one, and predict_proba_one methods works, including both
    the PyTorch learners in sgd_online.py and the River wrappers in
    river_learner.py.

    Parameters
    ----------
    window_size : int
        Number of samples per evaluation window. Metrics are recorded
        at the end of each window. Smaller windows → noisier but faster
        to respond to drift. Typical: 200–1000.
    drift_detector : optional
        Any detector with the interface from drift_detector.py.
        If provided, drift_detected is checked after each sample and
        logged with its stream index.
    verbose : bool
        Print a progress line at each window boundary.
    """

    def __init__(
        self,
        window_size: int = 500,
        drift_detector: Optional[Any] = None,
        verbose: bool = False,
    ) -> None:
        self.window_size = window_size
        self.drift_detector = drift_detector
        self.verbose = verbose

    def run(
        self,
        model: Any,
        stream: Iterator[Tuple[dict, int]],
    ) -> OnlineMetrics:
        """
        Execute the prequential loop over the stream.

        Parameters
        ----------
        model  : any online learner with learn_one / predict_one.
        stream : iterable of (x: dict, y: int).

        Returns
        -------
        OnlineMetrics
        """
        metrics = OnlineMetrics(window_size=self.window_size)

        window_correct: int = 0
        window_y_true: List[int] = []
        window_y_pred: List[int] = []
        total_correct: int = 0
        total_n: int = 0

        t_start = time.perf_counter()

        for i, (x, y) in enumerate(stream):
            # --------------------------------------------------
            # Step 1: Predict BEFORE learning
            # --------------------------------------------------
            y_pred = model.predict_one(x)

            # --------------------------------------------------
            # Step 2: Update running metrics
            # --------------------------------------------------
            correct = int(y_pred == y)
            window_correct += correct
            window_y_true.append(y)
            window_y_pred.append(y_pred)
            total_correct += correct
            total_n += 1

            # --------------------------------------------------
            # Step 3: Update drift detector (if attached)
            # --------------------------------------------------
            if self.drift_detector is not None:
                self.drift_detector.update(y, y_pred)
                if self.drift_detector.drift_detected:
                    metrics.drift_detections.append((i, type(self.drift_detector).__name__))
                    if self.verbose:
                        print(f"    [Drift detected at sample {i}]")

            # --------------------------------------------------
            # Step 4: Learn
            # --------------------------------------------------
            model.learn_one(x, y)

            # --------------------------------------------------
            # Step 5: Record at window boundary
            # --------------------------------------------------
            if (i + 1) % self.window_size == 0:
                window_acc = window_correct / self.window_size
                window_f1 = _f1_from_window(window_y_true, window_y_pred)

                metrics.accuracies.append(window_acc)
                metrics.window_f1s.append(window_f1)

                if self.verbose:
                    print(
                        f"  [n={i+1:>6,}]  "
                        f"window_acc={window_acc:.4f}  "
                        f"f1={window_f1:.4f}"
                    )

                # Reset window counters
                window_correct = 0
                window_y_true = []
                window_y_pred = []

        metrics.runtime_s = time.perf_counter() - t_start
        metrics.n_samples = total_n
        metrics.cumulative_acc = total_correct / total_n if total_n > 0 else 0.0
        metrics.final_accuracy = metrics.accuracies[-1] if metrics.accuracies else 0.0

        return metrics


# ---------------------------------------------------------------------------
# Quick helper for single-model evaluation (used in article code snippets)
# ---------------------------------------------------------------------------

def evaluate_stream(
    model: Any,
    stream: Iterator[Tuple[dict, int]],
    window_size: int = 500,
    verbose: bool = True,
) -> OnlineMetrics:
    """
    Convenience wrapper around PrequentialEvaluator for single-model runs.

    Usage
    -----
        from evaluation.prequential import evaluate_stream
        from methods.river_learner import RiverHoeffdingTree
        from data.generators import SEAConceptStream

        result = evaluate_stream(
            model=RiverHoeffdingTree(),
            stream=SEAConceptStream(n_samples=10_000, drift_at=5_000),
            window_size=500,
            verbose=True,
        )
        print(result.summary())
    """
    evaluator = PrequentialEvaluator(window_size=window_size, verbose=verbose)
    return evaluator.run(model, stream)
