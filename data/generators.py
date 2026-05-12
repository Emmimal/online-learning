"""
data/generators.py
==================
Streaming data generators for online learning experiments.

Generators implemented
----------------------
SEAConceptStream      — classic SEA benchmark with sudden concept drift
HyperplaneStream      — rotating hyperplane (gradual drift)
SuddenDriftStream     — wraps any two generators with an abrupt drift point
GradualDriftStream    — blends two generators over a transition window
FraudStream           — synthetic fraud detection stream with class imbalance
RecommendationStream  — synthetic user–item click stream

Each generator is an iterator that yields (x, y) pairs one sample at a time.
This mirrors the interface River expects and what the streaming pipeline
in pipelines/streaming_pipeline.py consumes.

Usage
-----
    from data.generators import SEAConceptStream

    stream = SEAConceptStream(n_samples=10_000, drift_at=5_000, seed=42)
    for x, y in stream:
        # x is a dict  {'f0': ..., 'f1': ..., 'f2': ...}
        # y is an int  0 or 1
        model.learn_one(x, y)
"""

from __future__ import annotations

import math
import random
from typing import Generator, Iterator, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Sample = Tuple[dict, int]


# ---------------------------------------------------------------------------
# SEA Concept Stream
# ---------------------------------------------------------------------------

class SEAConceptStream:
    """
    SEA concepts benchmark stream [1].

    Three numerical features f0, f1, f2 in [0, 10].
    Label rule depends on the active concept (0–3):

        concept 0 : f0 + f1 <= 8
        concept 1 : f0 + f1 <= 9
        concept 2 : f0 + f1 <= 7
        concept 3 : f0 + f1 <= 9.5

    A sudden drift swaps the concept at `drift_at` samples.

    Parameters
    ----------
    n_samples : int
        Total number of samples to generate.
    concept_before : int
        SEA concept index (0–3) before the drift.
    concept_after : int
        SEA concept index (0–3) after the drift.
    drift_at : int
        Sample index at which the drift occurs.
    noise : float
        Probability of flipping the true label (label noise).
    seed : int
        Random seed for reproducibility.

    References
    ----------
    [1] Street, W. N., & Kim, Y. (2001). A streaming ensemble algorithm (SEA)
        for large-scale classification. KDD '01.
    """

    _THRESHOLDS = {0: 8.0, 1: 9.0, 2: 7.0, 3: 9.5}

    def __init__(
        self,
        n_samples: int = 10_000,
        concept_before: int = 0,
        concept_after: int = 2,
        drift_at: int = 5_000,
        noise: float = 0.10,
        seed: int = 42,
    ) -> None:
        self.n_samples = n_samples
        self.concept_before = concept_before
        self.concept_after = concept_after
        self.drift_at = drift_at
        self.noise = noise
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[Sample]:
        for i in range(self.n_samples):
            concept = self.concept_before if i < self.drift_at else self.concept_after
            threshold = self._THRESHOLDS[concept]

            f0, f1, f2 = self._np_rng.uniform(0, 10, 3)
            y = int(f0 + f1 <= threshold)

            if self._rng.random() < self.noise:
                y = 1 - y

            yield {"f0": f0, "f1": f1, "f2": f2}, y

    def __len__(self) -> int:
        return self.n_samples


# ---------------------------------------------------------------------------
# Rotating Hyperplane Stream (gradual drift)
# ---------------------------------------------------------------------------

class HyperplaneStream:
    """
    Rotating hyperplane stream — a standard gradual drift benchmark [2].

    Generates samples from a d-dimensional hyperplane whose normal vector
    rotates slowly over time. The rotation speed controls how fast the
    concept drifts.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    n_features : int
        Number of input features.
    mag_change : float
        Magnitude of weight change per sample (controls drift speed).
        Typical values: 0.0 (no drift) to 0.001 (fast drift).
    noise : float
        Label noise probability.
    seed : int
        Random seed.

    References
    ----------
    [2] Hulten, G., Spencer, L., & Domingos, P. (2001). Mining time-changing
        data streams. KDD '01.
    """

    def __init__(
        self,
        n_samples: int = 10_000,
        n_features: int = 10,
        mag_change: float = 0.0004,
        noise: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.n_samples = n_samples
        self.n_features = n_features
        self.mag_change = mag_change
        self.noise = noise
        self._rng = np.random.default_rng(seed)
        self._label_rng = random.Random(seed + 1)

        # Initialise weights and their direction of change
        self._weights = self._rng.uniform(0, 1, n_features)
        self._direction = self._rng.choice([-1, 1], n_features)

    def __iter__(self) -> Iterator[Sample]:
        weights = self._weights.copy()
        direction = self._direction.copy()

        for _ in range(self.n_samples):
            x_arr = self._rng.uniform(0, 1, self.n_features)
            threshold = weights.sum() / self.n_features
            y = int(np.dot(x_arr, weights) >= threshold)

            if self._label_rng.random() < self.noise:
                y = 1 - y

            x = {f"f{i}": float(x_arr[i]) for i in range(self.n_features)}
            yield x, y

            # Rotate weights
            weights += self.mag_change * direction
            weights = np.clip(weights, 0, 1)

            # Flip direction when a weight hits a boundary
            for j in range(self.n_features):
                if weights[j] <= 0.0 or weights[j] >= 1.0:
                    direction[j] *= -1

    def __len__(self) -> int:
        return self.n_samples


# ---------------------------------------------------------------------------
# Sudden and Gradual Drift Wrappers
# ---------------------------------------------------------------------------

class SuddenDriftStream:
    """
    Compose two generators with an abrupt drift at a given sample index.

    Parameters
    ----------
    stream_before : iterable of (x, y)
        Stream active before the drift point.
    stream_after : iterable of (x, y)
        Stream active from the drift point onward.
    drift_at : int
        Sample index at which the switch occurs (0-indexed).
    """

    def __init__(
        self,
        stream_before: Iterator[Sample],
        stream_after: Iterator[Sample],
        drift_at: int,
    ) -> None:
        self._before = stream_before
        self._after = stream_after
        self.drift_at = drift_at

    def __iter__(self) -> Iterator[Sample]:
        for i, sample in enumerate(self._before):
            if i >= self.drift_at:
                break
            yield sample
        yield from self._after


class GradualDriftStream:
    """
    Compose two generators with a gradual drift over a transition window.

    During the window [drift_start, drift_start + width], samples are drawn
    from stream_before with decreasing probability and from stream_after with
    increasing probability. Outside the window the respective stream is used
    exclusively.

    Parameters
    ----------
    stream_before : iterable of (x, y)
    stream_after  : iterable of (x, y)
    n_samples     : int   Total stream length.
    drift_start   : int   Sample index where the transition window begins.
    width         : int   Number of samples over which the blend occurs.
    seed          : int
    """

    def __init__(
        self,
        stream_before: Iterator[Sample],
        stream_after: Iterator[Sample],
        n_samples: int = 10_000,
        drift_start: int = 4_000,
        width: int = 2_000,
        seed: int = 42,
    ) -> None:
        self._before = iter(stream_before)
        self._after = iter(stream_after)
        self.n_samples = n_samples
        self.drift_start = drift_start
        self.width = width
        self._rng = random.Random(seed)

    def __iter__(self) -> Iterator[Sample]:
        drift_end = self.drift_start + self.width

        for i in range(self.n_samples):
            if i < self.drift_start:
                yield next(self._before)
            elif i >= drift_end:
                yield next(self._after)
            else:
                # Linearly increasing probability of sampling from after-stream
                p_after = (i - self.drift_start) / self.width
                if self._rng.random() < p_after:
                    yield next(self._after)
                else:
                    yield next(self._before)

    def __len__(self) -> int:
        return self.n_samples


# ---------------------------------------------------------------------------
# Fraud Detection Stream
# ---------------------------------------------------------------------------

class FraudStream:
    """
    Synthetic fraud detection stream with realistic class imbalance and drift.

    Feature layout
    --------------
    amount       : transaction amount (log-normal, shifts up after drift)
    hour_of_day  : hour 0–23 (uniform)
    days_since_first_tx : tenure (exponential)
    velocity_1h  : number of transactions in the past hour (Poisson)
    foreign_tx   : binary flag
    card_present : binary flag

    Drift mechanism
    ---------------
    After `drift_at` samples the fraud distribution shifts:
    - Mean transaction amount increases by `drift_amount_shift`
    - Fraud rate increases from `fraud_rate` to `fraud_rate_after`

    This simulates a new fraud pattern emerging mid-stream — a realistic
    scenario for any production fraud model.

    Parameters
    ----------
    n_samples        : int    Total stream length.
    fraud_rate       : float  Fraud prevalence before drift (e.g. 0.01 = 1%).
    fraud_rate_after : float  Fraud prevalence after drift.
    drift_at         : int    Sample index of the drift.
    drift_amount_shift: float Log-scale shift in fraudulent amount post-drift.
    seed             : int
    """

    def __init__(
        self,
        n_samples: int = 20_000,
        fraud_rate: float = 0.01,
        fraud_rate_after: float = 0.03,
        drift_at: int = 10_000,
        drift_amount_shift: float = 1.5,
        seed: int = 42,
    ) -> None:
        self.n_samples = n_samples
        self.fraud_rate = fraud_rate
        self.fraud_rate_after = fraud_rate_after
        self.drift_at = drift_at
        self.drift_amount_shift = drift_amount_shift
        self._rng = np.random.default_rng(seed)

    def _make_sample(self, is_fraud: bool, post_drift: bool) -> dict:
        amount_mean = 5.0 if not is_fraud else (
            6.5 + (self.drift_amount_shift if post_drift else 0.0)
        )
        amount = float(np.exp(self._rng.normal(amount_mean, 0.8)))

        return {
            "amount": amount,
            "hour_of_day": float(self._rng.integers(0, 24)),
            "days_since_first_tx": float(self._rng.exponential(180)),
            "velocity_1h": float(self._rng.poisson(1.5 if not is_fraud else 4.0)),
            "foreign_tx": float(self._rng.binomial(1, 0.05 if not is_fraud else 0.40)),
            "card_present": float(self._rng.binomial(1, 0.85 if not is_fraud else 0.30)),
        }

    def __iter__(self) -> Iterator[Sample]:
        for i in range(self.n_samples):
            post_drift = i >= self.drift_at
            rate = self.fraud_rate_after if post_drift else self.fraud_rate
            is_fraud = bool(self._rng.binomial(1, rate))
            x = self._make_sample(is_fraud, post_drift)
            yield x, int(is_fraud)

    def __len__(self) -> int:
        return self.n_samples


# ---------------------------------------------------------------------------
# Recommendation Stream (implicit feedback)
# ---------------------------------------------------------------------------

class RecommendationStream:
    """
    Synthetic user–item click stream for online recommendation experiments.

    Models `n_users` users and `n_items` items. Each sample represents
    one user seeing one item and either clicking (y=1) or not (y=0).

    Features
    --------
    user_id         : int   Encoded as float for compatibility.
    item_id         : int
    item_popularity : float Log-normalised historical click rate.
    user_activity   : float Number of prior clicks by this user (running count).
    hour_of_day     : float
    is_new_user     : float Binary flag (1 if first 5 interactions for that user).

    Click probability is a sigmoid over a latent dot-product with injected
    noise, giving a realistic 2–8% average click-through rate.

    Parameters
    ----------
    n_samples : int
    n_users   : int
    n_items   : int
    seed      : int
    """

    def __init__(
        self,
        n_samples: int = 15_000,
        n_users: int = 500,
        n_items: int = 200,
        seed: int = 42,
    ) -> None:
        self.n_samples = n_samples
        self.n_users = n_users
        self.n_items = n_items
        self._rng = np.random.default_rng(seed)

        # Latent factors
        self._user_factors = self._rng.normal(0, 0.5, (n_users, 8))
        self._item_factors = self._rng.normal(0, 0.5, (n_items, 8))
        self._item_popularity = np.abs(self._rng.normal(0, 1, n_items))
        self._user_activity: dict[int, int] = {}

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def __iter__(self) -> Iterator[Sample]:
        for _ in range(self.n_samples):
            user_id = int(self._rng.integers(0, self.n_users))
            item_id = int(self._rng.integers(0, self.n_items))

            activity = self._user_activity.get(user_id, 0)
            score = float(np.dot(self._user_factors[user_id], self._item_factors[item_id]))
            p_click = self._sigmoid(score - 2.0)  # offset keeps CTR realistic
            y = int(self._rng.binomial(1, p_click))

            x = {
                "user_id": float(user_id),
                "item_id": float(item_id),
                "item_popularity": float(self._item_popularity[item_id]),
                "user_activity": float(activity),
                "hour_of_day": float(self._rng.integers(0, 24)),
                "is_new_user": float(activity < 5),
            }

            self._user_activity[user_id] = activity + 1
            yield x, y

    def __len__(self) -> int:
        return self.n_samples
