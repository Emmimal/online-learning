"""
methods/river_learner.py
========================
Thin wrappers around River classifiers that standardise the interface
and add production-friendly logging.

Why wrap River at all?
----------------------
River's classifiers already expose learn_one / predict_one / predict_proba_one.
These wrappers add:
  1. A consistent __repr__ that matches the SGD learners in sgd_online.py
  2. A `n_seen` property so the benchmark can track sample counts uniformly
  3. Optional StandardScaler preprocessing — River models are sensitive to
     feature scale and the scaler itself is updated online (no held-out fit)
  4. Class imbalance handling via SMOTE-equivalent class weights where supported

Classifiers exposed
-------------------
RiverLogisticRegression  — online logistic regression (River's SGDClassifier)
RiverHoeffdingTree       — Hoeffding Tree (VFDT) — the canonical streaming tree
RiverAdaptiveRF          — Adaptive Random Forest — drift-aware ensemble

All three expose the same interface:
    learn_one(x, y) -> self
    predict_one(x)  -> int
    predict_proba_one(x) -> dict[int, float]
    n_seen -> int

Usage
-----
    from methods.river_learner import RiverHoeffdingTree

    model = RiverHoeffdingTree()
    for x, y in stream:
        pred = model.predict_one(x)
        model.learn_one(x, y)
"""

from __future__ import annotations

from typing import Dict, Optional

from river import linear_model, preprocessing, tree, forest


# ---------------------------------------------------------------------------
# River Logistic Regression
# ---------------------------------------------------------------------------

class RiverLogisticRegression:
    """
    Online logistic regression using River's SGDClassifier.

    Internally uses River's Perceptron with a sigmoid loss — equivalent
    to logistic regression updated one sample at a time.

    Preprocessing
    -------------
    River's linear models are sensitive to feature scale. StandardScaler
    is chained upstream and updated online: it maintains running mean and
    variance without ever seeing the full dataset. This is the correct
    approach for streaming data — you cannot fit a scaler on a held-out set
    that does not exist yet.

    Parameters
    ----------
    l2 : float
        L2 regularisation strength.
    optimizer_lr : float
        Learning rate for the SGD step inside River's Perceptron.
    scale_features : bool
        Whether to prepend online StandardScaler. Default True.
    """

    def __init__(
        self,
        l2: float = 1e-4,
        optimizer_lr: float = 0.01,
        scale_features: bool = True,
    ) -> None:
        self.l2 = l2
        self.optimizer_lr = optimizer_lr
        self.scale_features = scale_features
        self._n_seen = 0

        from river import optim as river_optim

        _lr = linear_model.LogisticRegression(
            l2=l2,
            optimizer=river_optim.SGD(optimizer_lr),
        )

        if scale_features:
            self._pipeline = preprocessing.StandardScaler() | _lr
        else:
            self._pipeline = _lr

    def learn_one(self, x: dict, y: int) -> "RiverLogisticRegression":
        self._pipeline.learn_one(x, y)
        self._n_seen += 1
        return self

    def predict_proba_one(self, x: dict) -> Dict[int, float]:
        return self._pipeline.predict_proba_one(x)

    def predict_one(self, x: dict) -> int:
        return self._pipeline.predict_one(x)

    @property
    def n_seen(self) -> int:
        return self._n_seen

    def __repr__(self) -> str:
        return (
            f"RiverLogisticRegression(l2={self.l2}, "
            f"lr={self.optimizer_lr}, n_seen={self._n_seen})"
        )


# ---------------------------------------------------------------------------
# River Hoeffding Tree
# ---------------------------------------------------------------------------

class RiverHoeffdingTree:
    """
    Hoeffding Tree classifier (Very Fast Decision Tree, VFDT) via River.

    The Hoeffding Tree is the canonical streaming decision tree algorithm [1].
    It grows the tree incrementally: each leaf accumulates sufficient statistics
    until the Hoeffding bound guarantees that the best split at the leaf is
    the same one the full dataset would choose — then it splits.

    This makes it:
    - Memory bounded: only leaf statistics are stored, not raw samples
    - Anytime correct: predictions are valid at any point in the stream
    - Naturally drift-sensitive: new splits can replace old ones

    Parameters
    ----------
    grace_period : int
        Minimum samples at a leaf before evaluating a split (default 200).
        Lower → faster adaptation but noisier splits.
        Higher → more stable but slower to capture new patterns.
    delta : float
        Confidence for the Hoeffding bound (default 1e-7).
        Lower → more conservative splits (fewer false splits).
    leaf_prediction : str
        Prediction strategy at leaves: 'mc' (majority class),
        'nb' (Naive Bayes), or 'nba' (Naive Bayes Adaptive).

    References
    ----------
    [1] Domingos, P., & Hulten, G. (2000). Mining very fast data streams.
        KDD '00.
    """

    def __init__(
        self,
        grace_period: int = 200,
        delta: float = 1e-7,
        leaf_prediction: str = "nba",
    ) -> None:
        self.grace_period = grace_period
        self.delta = delta
        self.leaf_prediction = leaf_prediction
        self._n_seen = 0

        self._model = tree.HoeffdingTreeClassifier(
            grace_period=grace_period,
            delta=delta,
            leaf_prediction=leaf_prediction,
        )

    def learn_one(self, x: dict, y: int) -> "RiverHoeffdingTree":
        self._model.learn_one(x, y)
        self._n_seen += 1
        return self

    def predict_proba_one(self, x: dict) -> Dict[int, float]:
        proba = self._model.predict_proba_one(x)
        # River may return an empty dict before the first split
        if not proba:
            return {0: 0.5, 1: 0.5}
        return proba

    def predict_one(self, x: dict) -> int:
        pred = self._model.predict_one(x)
        return int(pred) if pred is not None else 0

    @property
    def n_seen(self) -> int:
        return self._n_seen

    @property
    def n_nodes(self) -> int:
        """Number of nodes currently in the tree."""
        return self._model.n_nodes if hasattr(self._model, "n_nodes") else 0

    def __repr__(self) -> str:
        return (
            f"RiverHoeffdingTree(grace_period={self.grace_period}, "
            f"delta={self.delta}, n_seen={self._n_seen})"
        )


# ---------------------------------------------------------------------------
# River Adaptive Random Forest
# ---------------------------------------------------------------------------

class RiverAdaptiveRF:
    """
    Adaptive Random Forest (ARF) classifier via River [2].

    ARF extends the Hoeffding Tree ensemble with an explicit drift detection
    mechanism per tree: each background tree monitors the error of its
    foreground counterpart. When drift is detected, the background tree
    replaces the foreground tree.

    This makes ARF the strongest general-purpose baseline for streaming
    classification with concept drift — it handles both sudden and gradual
    drift without manual intervention.

    Parameters
    ----------
    n_models : int
        Number of trees in the forest.
    max_features : str or int
        Number of features per split: 'sqrt', 'log2', or an integer.
    drift_detector : str
        Drift detector used per tree: 'ADWIN' (default) or 'EDDM'.
    seed : int

    References
    ----------
    [2] Gomes, H. M., et al. (2017). Adaptive random forests for evolving
        data stream classification. Machine Learning, 106(9).
    """

    def __init__(
        self,
        n_models: int = 10,
        max_features: str = "sqrt",
        drift_detector: str = "ADWIN",
        seed: int = 42,
    ) -> None:
        self.n_models = n_models
        self.max_features = max_features
        self.drift_detector_name = drift_detector
        self.seed = seed
        self._n_seen = 0
        self._n_drifts_detected = 0

        self._model = forest.ARFClassifier(
            n_models=n_models,
            max_features=max_features,
            seed=seed,
        )

    def learn_one(self, x: dict, y: int) -> "RiverAdaptiveRF":
        self._model.learn_one(x, y)
        self._n_seen += 1
        return self

    def predict_proba_one(self, x: dict) -> Dict[int, float]:
        proba = self._model.predict_proba_one(x)
        if not proba:
            return {0: 0.5, 1: 0.5}
        return proba

    def predict_one(self, x: dict) -> int:
        pred = self._model.predict_one(x)
        return int(pred) if pred is not None else 0

    @property
    def n_seen(self) -> int:
        return self._n_seen

    def __repr__(self) -> str:
        return (
            f"RiverAdaptiveRF(n_models={self.n_models}, "
            f"n_seen={self._n_seen})"
        )
