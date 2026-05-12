"""
use_cases/recommendation.py
============================
Online learning for recommendation systems — H2 section in Article 06.

Why online learning fits recommendations better than batch training
-------------------------------------------------------------------
Recommendation systems face three properties that make batch training
a poor fit and online learning the natural choice:

    1. Volume. A production recommendation system may serve millions of
       impressions per day. Retraining a batch model on all historical
       interactions is expensive and introduces lag between a user's
       behaviour and the model's response.

    2. Cold start. New users and new items appear continuously. A batch
       model trained yesterday has no representation for a user who
       signed up this morning. An online learner updates its internal
       state from the very first interaction.

    3. Preference drift. User interests evolve. A batch model trained on
       six months of data reflects six-month-old preferences. An online
       model weighted toward recent interactions reflects current ones.

Architecture used here
----------------------
This module implements a lightweight online click-through-rate (CTR)
predictor: given a (user_id, item_id, context_features) tuple, predict
whether the user will click.

Two online models are compared:
  - OnlineLogisticRegression (PyTorch SGD) — linear, fast, interpretable
  - RiverHoeffdingTree                     — non-linear, no batch gradient

The stream is from data/generators.py RecommendationStream, which models
user–item interactions with a latent dot-product and realistic 2–8% CTR.

Usage
-----
    python use_cases/recommendation.py

    from use_cases.recommendation import run_recommendation_demo
    results = run_recommendation_demo(n_samples=15_000, verbose=True)
"""

from __future__ import annotations

import time
from typing import Dict, List

from data.generators import RecommendationStream
from methods.sgd_online import OnlineLogisticRegression
from methods.river_learner import RiverHoeffdingTree
from evaluation.prequential import PrequentialEvaluator


# ---------------------------------------------------------------------------
# Feature scaler for PyTorch learner
# ---------------------------------------------------------------------------

class _OnlineMinMaxScaler:
    """
    Online min-max scaler that updates bounds incrementally.

    Scales each feature to [0, 1] using the running min and max seen so far.
    This is the online equivalent of sklearn's MinMaxScaler.fit_transform().

    The scaler is applied in the pipeline before features reach the PyTorch
    model — necessary because OnlineLogisticRegression is sensitive to
    feature scale.
    """

    def __init__(self) -> None:
        self._min: Dict[str, float] = {}
        self._max: Dict[str, float] = {}

    def transform(self, x: dict) -> dict:
        scaled = {}
        for k, v in x.items():
            lo = self._min.get(k, v)
            hi = self._max.get(k, v)
            denom = hi - lo
            scaled[k] = (v - lo) / denom if denom > 1e-8 else 0.0
        return scaled

    def update(self, x: dict) -> None:
        for k, v in x.items():
            if k not in self._min:
                self._min[k] = v
                self._max[k] = v
            else:
                self._min[k] = min(self._min[k], v)
                self._max[k] = max(self._max[k], v)


class ScaledOnlineLogisticRegression:
    """
    OnlineLogisticRegression with online min-max feature scaling.

    Wraps the base learner with a scaler that updates its bounds on each
    sample before scaling. The update order is:

        1. Transform x with current scaler bounds  (for prediction)
        2. Update scaler bounds with raw x          (update bounds)
        3. Predict using scaled x
        4. Learn using scaled x + true label

    This mirrors how a streaming scaler works in River's pipeline.
    """

    def __init__(self, n_features: int, lr: float = 0.01) -> None:
        self._scaler = _OnlineMinMaxScaler()
        self._model = OnlineLogisticRegression(n_features=n_features, lr=lr)

    def predict_one(self, x: dict) -> int:
        x_scaled = self._scaler.transform(x)
        return self._model.predict_one(x_scaled)

    def predict_proba_one(self, x: dict) -> Dict[str, float]:
        x_scaled = self._scaler.transform(x)
        return self._model.predict_proba_one(x_scaled)

    def learn_one(self, x: dict, y: int) -> "ScaledOnlineLogisticRegression":
        self._scaler.update(x)
        x_scaled = self._scaler.transform(x)
        self._model.learn_one(x_scaled, y)
        return self

    @property
    def n_seen(self) -> int:
        return self._model.n_seen


# ---------------------------------------------------------------------------
# Main demo function
# ---------------------------------------------------------------------------

def run_recommendation_demo(
    n_samples: int = 15_000,
    n_users: int = 500,
    n_items: int = 200,
    window_size: int = 500,
    verbose: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Compare online CTR prediction models on a synthetic recommendation stream.

    Parameters
    ----------
    n_samples   : Total number of user–item interactions.
    n_users     : Number of unique users in the simulation.
    n_items     : Number of unique items.
    window_size : Prequential evaluation window.
    verbose     : Print per-window progress.
    seed        : Random seed.

    Returns
    -------
    dict with results for each model.
    """
    print("=" * 70)
    print("  RECOMMENDATION: Online CTR Prediction")
    print(f"  Stream: {n_samples:,} interactions | "
          f"{n_users} users | {n_items} items")
    print("=" * 70)

    results = {}

    # ── Model 1: Online Logistic Regression (PyTorch) ────────────────────
    print("\n[1] Online Logistic Regression (PyTorch SGD)")
    print("-" * 40)

    stream_lr = RecommendationStream(
        n_samples=n_samples,
        n_users=n_users,
        n_items=n_items,
        seed=seed,
    )
    lr_model = ScaledOnlineLogisticRegression(n_features=6, lr=0.05)
    evaluator_lr = PrequentialEvaluator(window_size=window_size, verbose=verbose)
    lr_result = evaluator_lr.run(lr_model, stream_lr)
    results["LogisticRegression"] = lr_result

    # ── Model 2: Hoeffding Tree (River) ─────────────────────────────────
    print(f"\n[2] Hoeffding Tree (River)")
    print("-" * 40)

    stream_ht = RecommendationStream(
        n_samples=n_samples,
        n_users=n_users,
        n_items=n_items,
        seed=seed,
    )
    ht_model = RiverHoeffdingTree(grace_period=100)
    evaluator_ht = PrequentialEvaluator(window_size=window_size, verbose=verbose)
    ht_result = evaluator_ht.run(ht_model, stream_ht)
    results["HoeffdingTree"] = ht_result

    # ── Summary ───────────────────────────────────────────────────────────
    _print_rec_summary(results)
    return results


def _print_rec_summary(results: Dict) -> None:
    import numpy as np

    print("\n" + "=" * 70)
    print("  RECOMMENDATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Method':<28}  {'Cum Acc':>9}  {'Mean Window Acc':>15}  {'Runtime':>8}")
    print("-" * 70)
    for name, r in results.items():
        print(
            f"  {name:<28}  {r.cumulative_acc:>9.4f}  "
            f"{float(np.mean(r.accuracies)):>15.4f}  "
            f"{r.runtime_s:>7.1f}s"
        )
    print("=" * 70)
    print()
    print("  Notes:")
    print("  - CTR ~5% means a naive 'always predict 0' baseline scores ~95%.")
    print("  - The gap between cumulative and mean-window accuracy shows")
    print("    how the model improves as it accumulates more interactions.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_recommendation_demo(n_samples=15_000, verbose=True)
