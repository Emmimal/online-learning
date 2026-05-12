"""
methods/sgd_online.py
=====================
PyTorch SGD-based online learner for streaming classification.

This module shows how to implement online learning using raw PyTorch —
without any specialised online learning library. The key insight is simple:
process one sample at a time, call loss.backward(), and call optimiser.step().
No batching, no epoch loops, no held-out validation set.

Classes
-------
OnlineLogisticRegression
    Linear model with sigmoid output. Equivalent to River's SGDClassifier
    but written in PyTorch so you can swap in any architecture.

OnlineMLP
    Two-layer MLP for non-linear streaming classification. Shares the same
    learn_one / predict_one interface as OnlineLogisticRegression so they
    are drop-in replacements.

Both classes expose the River-compatible interface:
    learn_one(x: dict, y: int)    -> self
    predict_one(x: dict)          -> int
    predict_proba_one(x: dict)    -> dict[int, float]

Usage
-----
    from methods.sgd_online import OnlineLogisticRegression

    model = OnlineLogisticRegression(n_features=3, lr=0.01)

    for x, y in stream:
        p = model.predict_proba_one(x)   # predict BEFORE learning
        model.learn_one(x, y)            # then update weights
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _dict_to_tensor(x: dict, feature_names: List[str]) -> torch.Tensor:
    """
    Convert a feature dict to a 1-D float tensor using a fixed feature order.

    Feature names that are absent in `x` are filled with 0.0. This makes
    the learner robust to sparse or variable-schema streams.
    """
    values = [float(x.get(name, 0.0)) for name in feature_names]
    return torch.tensor(values, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Online Logistic Regression
# ---------------------------------------------------------------------------

class OnlineLogisticRegression:
    """
    Online logistic regression via mini-batch-size-1 SGD in PyTorch.

    The model processes exactly one sample per learn_one() call. There is
    no replay buffer, no Fisher regularisation — just a single forward pass,
    a binary cross-entropy loss, and one optimiser step.

    Parameters
    ----------
    n_features : int
        Number of input features. Must be set at initialisation because
        PyTorch allocates the weight tensor up front.
    lr : float
        Learning rate for SGD. Typical range for streaming data: 0.001–0.05.
        Too high → unstable; too low → slow adaptation to drift.
    weight_decay : float
        L2 regularisation coefficient. Prevents weight explosion on long streams.
    feature_names : list of str, optional
        Ordered list of feature keys to extract from the input dict.
        If None, names are inferred from the first call to learn_one().
    """

    def __init__(
        self,
        n_features: int,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        self.n_features = n_features
        self.lr = lr
        self.weight_decay = weight_decay
        self.feature_names: Optional[List[str]] = feature_names

        self._model = nn.Linear(n_features, 1)
        self._criterion = nn.BCEWithLogitsLoss()
        self._optimiser = torch.optim.SGD(
            self._model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self._n_seen = 0

    def _infer_feature_names(self, x: dict) -> None:
        """Set feature_names from first sample if not provided at init."""
        if self.feature_names is None:
            self.feature_names = sorted(x.keys())

    def learn_one(self, x: dict, y: int) -> "OnlineLogisticRegression":
        """
        Update model weights on a single (x, y) pair.

        The update order — predict first, then learn — is enforced by the
        prequential evaluator. Calling learn_one before predict_one on the
        same sample would leak the label into the evaluation metric.

        Parameters
        ----------
        x : dict   Feature dict for this sample.
        y : int    True label (0 or 1).

        Returns
        -------
        self  (allows chaining)
        """
        self._infer_feature_names(x)

        x_tensor = _dict_to_tensor(x, self.feature_names).unsqueeze(0)
        y_tensor = torch.tensor([[float(y)]])

        self._model.train()
        self._optimiser.zero_grad()
        logit = self._model(x_tensor)
        loss = self._criterion(logit, y_tensor)
        loss.backward()
        self._optimiser.step()

        self._n_seen += 1
        return self

    def predict_proba_one(self, x: dict) -> Dict[int, float]:
        """
        Return class probabilities for a single sample without updating weights.

        Parameters
        ----------
        x : dict   Feature dict.

        Returns
        -------
        dict  {0: p_negative, 1: p_positive}
        """
        self._infer_feature_names(x)

        x_tensor = _dict_to_tensor(x, self.feature_names).unsqueeze(0)

        self._model.eval()
        with torch.no_grad():
            logit = self._model(x_tensor)
            p_pos = float(torch.sigmoid(logit).item())

        return {0: 1.0 - p_pos, 1: p_pos}

    def predict_one(self, x: dict) -> int:
        """Predict the most probable class label for a single sample."""
        proba = self.predict_proba_one(x)
        return int(proba[1] >= 0.5)

    @property
    def n_seen(self) -> int:
        """Total number of samples seen during training."""
        return self._n_seen

    def __repr__(self) -> str:
        return (
            f"OnlineLogisticRegression("
            f"n_features={self.n_features}, lr={self.lr}, "
            f"n_seen={self._n_seen})"
        )


# ---------------------------------------------------------------------------
# Online MLP
# ---------------------------------------------------------------------------

class _MLPNet(nn.Module):
    """Internal PyTorch module used by OnlineMLP."""

    def __init__(self, n_features: int, hidden_dims: List[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = n_features
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OnlineMLP:
    """
    Online two-layer MLP updated one sample at a time via SGD.

    Shares the learn_one / predict_one / predict_proba_one interface with
    OnlineLogisticRegression so either class can be swapped into the
    streaming pipeline without other changes.

    Parameters
    ----------
    n_features  : int
    hidden_dims : list of int   Width of each hidden layer, e.g. [64, 32].
    lr          : float
    weight_decay: float
    feature_names: list of str, optional

    Notes
    -----
    For simple or low-dimensional streams, OnlineLogisticRegression often
    matches or outperforms OnlineMLP because the non-linearity adds parameters
    that are harder to fit one sample at a time. Prefer OnlineMLP when:
      - Features are dense and high-dimensional (>20 features)
      - The decision boundary is known to be non-linear
      - You have enough stream volume to fit the extra parameters
    """

    def __init__(
        self,
        n_features: int,
        hidden_dims: Optional[List[int]] = None,
        lr: float = 0.005,
        weight_decay: float = 1e-4,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        self.n_features = n_features
        self.hidden_dims = hidden_dims or [64, 32]
        self.lr = lr
        self.weight_decay = weight_decay
        self.feature_names: Optional[List[str]] = feature_names

        self._model = _MLPNet(n_features, self.hidden_dims)
        self._criterion = nn.BCEWithLogitsLoss()
        self._optimiser = torch.optim.Adam(
            self._model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self._n_seen = 0

    def _infer_feature_names(self, x: dict) -> None:
        if self.feature_names is None:
            self.feature_names = sorted(x.keys())

    def learn_one(self, x: dict, y: int) -> "OnlineMLP":
        self._infer_feature_names(x)

        x_tensor = _dict_to_tensor(x, self.feature_names).unsqueeze(0)
        y_tensor = torch.tensor([[float(y)]])

        self._model.train()
        self._optimiser.zero_grad()
        logit = self._model(x_tensor)
        loss = self._criterion(logit, y_tensor)
        loss.backward()
        self._optimiser.step()

        self._n_seen += 1
        return self

    def predict_proba_one(self, x: dict) -> Dict[int, float]:
        self._infer_feature_names(x)

        x_tensor = _dict_to_tensor(x, self.feature_names).unsqueeze(0)

        self._model.eval()
        with torch.no_grad():
            logit = self._model(x_tensor)
            p_pos = float(torch.sigmoid(logit).item())

        return {0: 1.0 - p_pos, 1: p_pos}

    def predict_one(self, x: dict) -> int:
        proba = self.predict_proba_one(x)
        return int(proba[1] >= 0.5)

    @property
    def n_seen(self) -> int:
        return self._n_seen

    def __repr__(self) -> str:
        return (
            f"OnlineMLP(n_features={self.n_features}, "
            f"hidden_dims={self.hidden_dims}, lr={self.lr}, "
            f"n_seen={self._n_seen})"
        )
