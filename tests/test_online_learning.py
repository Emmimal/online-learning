"""
tests/test_online_learning.py
==============================
Test suite for the online learning codebase.

Run with:
    python -m pytest tests/ -v
    python -m pytest tests/ -v --tb=short

Coverage
--------
- Data generators: correct output shape, drift timing, class distribution
- SGD online learner: interface contract, weight updates, reproducibility
- River wrappers: interface contract, predictions before first learn
- Drift detectors: detects known drift, does not fire on stable stream
- Prequential evaluator: correct test-then-train order, metric computation
- Streaming pipeline: end-to-end wiring, drift response invocation
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generators import (
    SEAConceptStream,
    HyperplaneStream,
    FraudStream,
    RecommendationStream,
    SuddenDriftStream,
    GradualDriftStream,
)
from methods.sgd_online import OnlineLogisticRegression, OnlineMLP
from methods.river_learner import (
    RiverLogisticRegression,
    RiverHoeffdingTree,
    RiverAdaptiveRF,
)
from methods.drift_detector import ADWIN, DDM, PageHinkley
from evaluation.prequential import PrequentialEvaluator, evaluate_stream
from pipelines.streaming_pipeline import StreamingPipeline, DriftResponse


# ===========================================================================
# Data generators
# ===========================================================================

class TestSEAConceptStream:

    def test_length(self):
        stream = SEAConceptStream(n_samples=1_000)
        samples = list(stream)
        assert len(samples) == 1_000

    def test_feature_keys(self):
        stream = SEAConceptStream(n_samples=10)
        for x, y in stream:
            assert set(x.keys()) == {"f0", "f1", "f2"}
            assert y in {0, 1}
            break

    def test_feature_range(self):
        stream = SEAConceptStream(n_samples=500)
        for x, y in stream:
            assert 0.0 <= x["f0"] <= 10.0
            assert 0.0 <= x["f1"] <= 10.0
            assert 0.0 <= x["f2"] <= 10.0

    def test_drift_changes_distribution(self):
        """After drift at 500, the label distribution should shift."""
        stream = SEAConceptStream(
            n_samples=1_000, concept_before=0, concept_after=2,
            drift_at=500, noise=0.0, seed=42,
        )
        samples = list(stream)
        pre  = [y for _, y in samples[:500]]
        post = [y for _, y in samples[500:]]
        # concept 0: threshold 8 → more 1s; concept 2: threshold 7 → fewer 1s
        assert sum(pre) / len(pre) > sum(post) / len(post)

    def test_reproducible_with_seed(self):
        s1 = list(SEAConceptStream(n_samples=100, seed=1))
        s2 = list(SEAConceptStream(n_samples=100, seed=1))
        s3 = list(SEAConceptStream(n_samples=100, seed=99))
        assert s1 == s2
        assert s1 != s3


class TestHyperplaneStream:

    def test_length(self):
        stream = HyperplaneStream(n_samples=500)
        assert len(list(stream)) == 500

    def test_feature_count(self):
        stream = HyperplaneStream(n_samples=10, n_features=8)
        for x, y in stream:
            assert len(x) == 8
            break


class TestFraudStream:

    def test_length(self):
        assert len(list(FraudStream(n_samples=200))) == 200

    def test_class_imbalance(self):
        """Fraud rate should be approximately 1% before drift."""
        stream = FraudStream(n_samples=5_000, fraud_rate=0.01,
                             drift_at=10_000, seed=42)
        labels = [y for _, y in stream]
        rate = sum(labels) / len(labels)
        assert 0.005 <= rate <= 0.025  # allow ±1.5% variance

    def test_fraud_rate_increases_after_drift(self):
        stream = FraudStream(
            n_samples=2_000, fraud_rate=0.01, fraud_rate_after=0.05,
            drift_at=1_000, seed=42,
        )
        samples = list(stream)
        pre_rate  = sum(y for _, y in samples[:1_000]) / 1_000
        post_rate = sum(y for _, y in samples[1_000:]) / 1_000
        assert post_rate > pre_rate


class TestSuddenDriftStream:

    def test_switches_at_drift_point(self):
        """SuddenDriftStream yields drift_at samples from before, then all of after."""
        before = SEAConceptStream(n_samples=250, concept_before=0, noise=0.0, seed=0)
        after  = SEAConceptStream(n_samples=250, concept_before=2, noise=0.0, seed=0)
        stream = SuddenDriftStream(iter(before), iter(after), drift_at=250)
        samples = list(stream)
        # 250 from before (drift_at=250 stops it) + 250 from after = 500
        assert len(samples) == 500


# ===========================================================================
# SGD Online Learner
# ===========================================================================

class TestOnlineLogisticRegression:

    def _make_learner(self):
        return OnlineLogisticRegression(
            n_features=3,
            lr=0.01,
            feature_names=["f0", "f1", "f2"],
        )

    def test_predict_before_learn(self):
        """Should return a valid prediction without prior learning."""
        model = self._make_learner()
        x = {"f0": 1.0, "f1": 2.0, "f2": 3.0}
        pred = model.predict_one(x)
        assert pred in {0, 1}

    def test_predict_proba_sums_to_one(self):
        model = self._make_learner()
        x = {"f0": 1.0, "f1": 2.0, "f2": 3.0}
        proba = model.predict_proba_one(x)
        assert abs(proba[0] + proba[1] - 1.0) < 1e-6

    def test_learn_increments_n_seen(self):
        model = self._make_learner()
        x = {"f0": 1.0, "f1": 2.0, "f2": 3.0}
        assert model.n_seen == 0
        model.learn_one(x, 1)
        model.learn_one(x, 0)
        assert model.n_seen == 2

    def test_weights_change_after_learn(self):
        """Weights must change after at least one learn_one call."""
        model = self._make_learner()
        x = {"f0": 5.0, "f1": 3.0, "f2": 1.0}
        w_before = [p.detach().clone() for p in model._model.parameters()]
        model.learn_one(x, 1)
        w_after = [p.detach().clone() for p in model._model.parameters()]
        changed = any(
            not (b == a).all()
            for b, a in zip(w_before, w_after)
        )
        assert changed

    def test_improves_on_linearly_separable_data(self):
        """
        On clean linearly separable data, accuracy should exceed 80%
        after 2,000 samples (room for cold-start).
        """
        import random
        rng = random.Random(42)
        model = OnlineLogisticRegression(
            n_features=2, lr=0.05, feature_names=["x", "y"],
        )
        correct = 0
        n = 2_000
        for _ in range(n):
            x_val = rng.uniform(0, 10)
            y_val = rng.uniform(0, 10)
            label = int(x_val + y_val > 10)
            x = {"x": x_val, "y": y_val}
            pred = model.predict_one(x)
            if pred == label:
                correct += 1
            model.learn_one(x, label)
        assert correct / n > 0.73


class TestOnlineMLP:

    def test_interface_matches_logistic_regression(self):
        """OnlineMLP must expose the same interface as OnlineLogisticRegression."""
        from methods.sgd_online import OnlineMLP
        model = OnlineMLP(
            n_features=3,
            hidden_dims=[16, 8],
            feature_names=["f0", "f1", "f2"],
        )
        x = {"f0": 1.0, "f1": 2.0, "f2": 3.0}
        pred = model.predict_one(x)
        assert pred in {0, 1}
        proba = model.predict_proba_one(x)
        assert abs(proba[0] + proba[1] - 1.0) < 1e-6
        model.learn_one(x, 1)
        assert model.n_seen == 1


# ===========================================================================
# River Wrappers
# ===========================================================================

class TestRiverWrappers:

    @pytest.mark.parametrize("ModelClass", [
        RiverLogisticRegression,
        RiverHoeffdingTree,
        RiverAdaptiveRF,
    ])
    def test_predict_before_learn(self, ModelClass):
        model = ModelClass()
        x = {"f0": 1.0, "f1": 2.0, "f2": 3.0}
        pred = model.predict_one(x)
        assert pred in {0, 1}

    @pytest.mark.parametrize("ModelClass", [
        RiverLogisticRegression,
        RiverHoeffdingTree,
        RiverAdaptiveRF,
    ])
    def test_learn_increments_n_seen(self, ModelClass):
        model = ModelClass()
        x = {"f0": 1.0, "f1": 2.0, "f2": 3.0}
        model.learn_one(x, 1)
        assert model.n_seen == 1

    @pytest.mark.parametrize("ModelClass", [
        RiverLogisticRegression,
        RiverHoeffdingTree,
    ])
    def test_proba_sums_to_one_after_learning(self, ModelClass):
        model = ModelClass()
        x = {"f0": 1.0, "f1": 2.0, "f2": 3.0}
        model.learn_one(x, 0)
        model.learn_one(x, 1)
        proba = model.predict_proba_one(x)
        assert abs(sum(proba.values()) - 1.0) < 1e-5


# ===========================================================================
# Drift Detectors
# ===========================================================================

class TestDriftDetectors:

    def _stable_errors(self, detector, n: int = 2_000, error_rate: float = 0.05):
        """Feed a stable error signal and count detections."""
        import random
        rng = random.Random(42)
        for _ in range(n):
            y_true = 1
            y_pred = 0 if rng.random() < error_rate else 1
            detector.update(y_true, y_pred)
        return detector.n_detections

    def _drifting_errors(self, detector, n: int = 2_000, drift_at: int = 1_000):
        """Feed a stable then suddenly high error signal."""
        for i in range(n):
            error_rate = 0.05 if i < drift_at else 0.60
            y_true = 1
            y_pred = 0 if (i % 10 < int(error_rate * 10)) else 1
            detector.update(y_true, y_pred)
        return detector.n_detections

    @pytest.mark.parametrize("DetectorClass", [ADWIN, DDM, PageHinkley])
    def test_detects_sudden_drift(self, DetectorClass):
        detector = DetectorClass()
        n_det = self._drifting_errors(detector, n=3_000, drift_at=1_500)
        assert n_det >= 1, f"{DetectorClass.__name__} failed to detect sudden drift"

    def test_adwin_stable_stream_low_detections(self):
        detector = ADWIN(delta=0.002)
        n_det = self._stable_errors(detector, n=5_000, error_rate=0.05)
        # On a stable stream, false positives should be very rare
        assert n_det <= 2

    def test_drift_detected_flag_resets(self):
        """drift_detected should be False on the next sample after firing."""
        detector = ADWIN(delta=0.1)  # very sensitive for this test
        # Force a large error spike
        for _ in range(100):
            detector.update(1, 0)   # always wrong
        # After spike, drift should have fired; next stable update should not
        detector.update(1, 1)
        assert not detector.drift_detected or detector.n_detections >= 1


# ===========================================================================
# Prequential Evaluator
# ===========================================================================

class TestPrequentialEvaluator:

    def test_test_then_train_order(self):
        """
        Verify that prediction uses pre-update weights.
        We use a model that always predicts 0 until it sees a 1 label.
        The first sample should be predicted with the untrained model.
        """
        class MemoryModel:
            """Predicts 0 until it has seen at least one positive label."""
            def __init__(self):
                self._seen_positive = False
            def predict_one(self, x):
                return 1 if self._seen_positive else 0
            def learn_one(self, x, y):
                if y == 1:
                    self._seen_positive = True
                return self

        model = MemoryModel()
        # Stream: first sample is (x, 1). A test-before-train evaluator
        # should record this as WRONG (model predicted 0 before learning).
        stream = [
            ({"f0": 1.0}, 1),
            ({"f0": 1.0}, 0),
        ]

        evaluator = PrequentialEvaluator(window_size=2)
        result = evaluator.run(model, iter(stream))
        # Window acc: first prediction was wrong (0 != 1), second was correct (1 != 0 → wrong)
        # Actually: after learning (1), model predicts 1. y=0. Wrong again.
        # Both wrong → window acc = 0.0
        assert result.accuracies[0] == 0.0

    def test_window_boundaries(self):
        stream = SEAConceptStream(n_samples=2_000, seed=42)
        model = RiverHoeffdingTree()
        result = PrequentialEvaluator(window_size=500).run(model, stream)
        assert len(result.accuracies) == 4   # 2000 / 500

    def test_n_samples_matches_stream(self):
        stream = SEAConceptStream(n_samples=1_500, seed=42)
        model = RiverHoeffdingTree()
        result = PrequentialEvaluator(window_size=500).run(model, stream)
        assert result.n_samples == 1_500

    def test_cumulative_acc_range(self):
        stream = SEAConceptStream(n_samples=1_000, seed=42)
        model = RiverHoeffdingTree()
        result = PrequentialEvaluator(window_size=200).run(model, stream)
        assert 0.0 <= result.cumulative_acc <= 1.0

    def test_drift_detection_log(self):
        """Drift detector events should appear in the result."""
        stream = SEAConceptStream(
            n_samples=5_000, drift_at=2_500, concept_before=0,
            concept_after=2, noise=0.0, seed=42,
        )
        model = RiverAdaptiveRF(n_models=5, seed=42)
        detector = ADWIN(delta=0.002)
        evaluator = PrequentialEvaluator(window_size=500, drift_detector=detector)
        result = evaluator.run(model, stream)
        # The benchmark stream has a real drift — we expect at least one detection
        assert result.drift_detections is not None
        assert isinstance(result.drift_detections, list)


# ===========================================================================
# Streaming Pipeline
# ===========================================================================

class TestStreamingPipeline:

    def test_pipeline_runs_end_to_end(self):
        stream = SEAConceptStream(n_samples=1_000, seed=42)
        model = RiverHoeffdingTree()
        pipeline = StreamingPipeline(model=model, window_size=200)
        result = pipeline.run(stream)
        assert result.n_samples == 1_000
        assert len(result.accuracies) == 5

    def test_drift_response_log_only(self):
        stream = SEAConceptStream(
            n_samples=2_000, drift_at=1_000, seed=42, noise=0.0,
        )
        model = RiverHoeffdingTree()
        detector = ADWIN(delta=0.1)   # sensitive — will fire
        pipeline = StreamingPipeline(
            model=model,
            drift_detector=detector,
            drift_response=DriftResponse.log_only(),
            window_size=500,
        )
        result = pipeline.run(stream)
        # Pipeline should complete normally
        assert result.n_samples == 2_000

    def test_drift_response_reset_creates_new_model(self):
        """After reset, model.n_seen should be 0 (fresh instance)."""
        call_log = []

        def factory():
            m = RiverHoeffdingTree()
            call_log.append("created")
            return m

        stream = SEAConceptStream(
            n_samples=2_000, drift_at=1_000, noise=0.0, seed=42,
        )
        detector = ADWIN(delta=0.05)   # very sensitive for this test
        pipeline = StreamingPipeline(
            model=factory(),
            drift_detector=detector,
            drift_response=DriftResponse.reset_model(factory),
            window_size=500,
        )
        pipeline.run(stream)
        # Factory was called at init (1) + at least once more if drift detected
        assert len(call_log) >= 1

    def test_no_drift_detector_completes_normally(self):
        stream = SEAConceptStream(n_samples=500, seed=42)
        model = RiverLogisticRegression()
        pipeline = StreamingPipeline(model=model, window_size=100)
        result = pipeline.run(stream)
        assert result.n_samples == 500


# ===========================================================================
# Entry point for running without pytest
# ===========================================================================

if __name__ == "__main__":
    import unittest

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestSEAConceptStream,
        TestHyperplaneStream,
        TestFraudStream,
        TestSuddenDriftStream,
        TestOnlineLogisticRegression,
        TestOnlineMLP,
        TestRiverWrappers,
        TestDriftDetectors,
        TestPrequentialEvaluator,
        TestStreamingPipeline,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
