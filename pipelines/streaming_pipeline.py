"""
pipelines/streaming_pipeline.py
================================
Production streaming pipeline for online learning.

This module is the article's central code example for:
  H2 — Implementing Online Learning with River in Python (Step-by-Step)
  H3 — Setting Up a Streaming Data Pipeline
  H3 — Updating Model Weights on Each Incoming Sample

The StreamingPipeline class wires together:
  1. A data source (any iterable of (x, y) dicts)
  2. An online learner (learn_one / predict_one interface)
  3. A drift detector (optional — any detector from drift_detector.py)
  4. A prequential evaluator
  5. A drift response strategy: retrain from scratch or adapt in place

This is the code you would extend to run against a real Kafka or Kinesis
stream. The only change needed is to swap out the generator in step (1) for
an actual stream consumer — the rest of the pipeline is stream-agnostic.

Usage (article code snippet)
-----------------------------
    from pipelines.streaming_pipeline import StreamingPipeline
    from methods.river_learner import RiverAdaptiveRF
    from methods.drift_detector import ADWIN
    from data.generators import SEAConceptStream

    pipeline = StreamingPipeline(
        model=RiverAdaptiveRF(n_models=10),
        drift_detector=ADWIN(delta=0.002),
        window_size=500,
        verbose=True,
    )

    stream = SEAConceptStream(n_samples=10_000, drift_at=5_000)
    result = pipeline.run(stream)
    print(result.summary())
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterator, List, Optional, Tuple

from evaluation.prequential import OnlineMetrics, PrequentialEvaluator


# ---------------------------------------------------------------------------
# DriftResponse — what to do when drift is detected
# ---------------------------------------------------------------------------

class DriftResponse:
    """
    Strategy object called by the pipeline when drift is detected.

    Two built-in strategies are provided as class methods:
    - DriftResponse.log_only()    — record the detection, do nothing else
    - DriftResponse.reset_model() — replace the model with a fresh copy

    You can also pass a custom callable:
        DriftResponse(fn=my_custom_handler)

    The handler receives (model, sample_index) and must return the model
    to use going forward (the same object or a replacement).
    """

    def __init__(self, fn: Callable[[Any, int], Any]) -> None:
        self._fn = fn

    def __call__(self, model: Any, sample_index: int) -> Any:
        return self._fn(model, sample_index)

    @classmethod
    def log_only(cls) -> "DriftResponse":
        """Record drift but continue with the existing model unchanged."""
        return cls(fn=lambda model, _: model)

    @classmethod
    def reset_model(cls, model_factory: Callable[[], Any]) -> "DriftResponse":
        """
        Replace the current model with a fresh instance.

        Parameters
        ----------
        model_factory : callable
            A zero-argument factory that returns a new model instance,
            e.g. lambda: RiverHoeffdingTree(grace_period=100).

        This strategy is aggressive — it discards all accumulated knowledge.
        Use it only when the post-drift distribution is known to be completely
        unrelated to the pre-drift distribution. For partial concept drift
        (where some prior knowledge remains valid), use an adaptive model
        like RiverAdaptiveRF instead.
        """
        return cls(fn=lambda _, idx: model_factory())


# ---------------------------------------------------------------------------
# StreamingPipeline
# ---------------------------------------------------------------------------

class StreamingPipeline:
    """
    End-to-end streaming pipeline: ingest → predict → evaluate → learn.

    H3 — Setting Up a Streaming Data Pipeline
    -----------------------------------------
    The pipeline is initialised with:
      - A model (any learn_one / predict_one compatible object)
      - An optional drift detector
      - An optional drift response strategy
      - Evaluation window size

    H3 — Updating Model Weights on Each Incoming Sample
    ----------------------------------------------------
    For each incoming sample, the pipeline runs in this exact order:
      1. Predict   — compute ŷ without updating any weights
      2. Evaluate  — record whether ŷ == y in the current window
      3. Detect    — pass the error signal to the drift detector
      4. Respond   — if drift detected, invoke the drift response
      5. Learn     — call model.learn_one(x, y)

    This order is non-negotiable. Swapping steps 1 and 5 (learn before
    predict) would give the model access to the label before it predicts,
    producing optimistic accuracy estimates that do not reflect real
    deployment performance.

    Parameters
    ----------
    model          : online learner with learn_one / predict_one.
    drift_detector : optional — from drift_detector.py.
    drift_response : optional DriftResponse — defaults to log_only.
    window_size    : int — evaluation window size.
    verbose        : bool — print window-level progress.
    """

    def __init__(
        self,
        model: Any,
        drift_detector: Optional[Any] = None,
        drift_response: Optional[DriftResponse] = None,
        window_size: int = 500,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.drift_detector = drift_detector
        self.drift_response = drift_response or DriftResponse.log_only()
        self.window_size = window_size
        self.verbose = verbose

        self._drift_log: List[Tuple[int, str]] = []

    def run(self, stream: Iterator[Tuple[dict, int]]) -> OnlineMetrics:
        """
        Run the pipeline over the entire stream.

        Parameters
        ----------
        stream : iterable of (x: dict, y: int)
            Any generator from data/generators.py or a real stream consumer.

        Returns
        -------
        OnlineMetrics with full per-window accuracy and drift detections.
        """
        metrics = OnlineMetrics(window_size=self.window_size)

        window_correct = 0
        window_y_true: List[int] = []
        window_y_pred: List[int] = []
        total_correct = 0
        total_n = 0
        t_start = time.perf_counter()

        for i, (x, y) in enumerate(stream):

            # ── Step 1: Predict ──────────────────────────────────────────
            y_pred = self.model.predict_one(x)

            # ── Step 2: Evaluate ─────────────────────────────────────────
            correct = int(y_pred == y)
            window_correct += correct
            window_y_true.append(y)
            window_y_pred.append(y_pred)
            total_correct += correct
            total_n += 1

            # ── Step 3: Detect drift ──────────────────────────────────────
            if self.drift_detector is not None:
                self.drift_detector.update(y, y_pred)

                if self.drift_detector.drift_detected:
                    detector_name = type(self.drift_detector).__name__
                    self._drift_log.append((i, detector_name))
                    metrics.drift_detections.append((i, detector_name))

                    if self.verbose:
                        print(f"    ⚠ Drift detected at sample {i:,} "
                              f"({detector_name})")

                    # ── Step 4: Respond ───────────────────────────────────
                    self.model = self.drift_response(self.model, i)

            # ── Step 5: Learn ─────────────────────────────────────────────
            self.model.learn_one(x, y)

            # ── Window boundary ───────────────────────────────────────────
            if (i + 1) % self.window_size == 0:
                window_acc = window_correct / self.window_size
                metrics.accuracies.append(window_acc)

                if self.verbose:
                    print(
                        f"  [n={i+1:>7,}]  "
                        f"window_acc={window_acc:.4f}"
                    )

                window_correct = 0
                window_y_true = []
                window_y_pred = []

        metrics.runtime_s = time.perf_counter() - t_start
        metrics.n_samples = total_n
        metrics.cumulative_acc = total_correct / total_n if total_n > 0 else 0.0
        metrics.final_accuracy = metrics.accuracies[-1] if metrics.accuracies else 0.0

        return metrics

    @property
    def drift_log(self) -> List[Tuple[int, str]]:
        """List of (sample_index, detector_name) for all detected drifts."""
        return list(self._drift_log)
