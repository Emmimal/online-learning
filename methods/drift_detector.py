"""
methods/drift_detector.py
=========================
Concept drift detectors for online learning pipelines.

Three detectors are implemented
--------------------------------
ADWIN (ADaptive WINdowing)
    Error-rate-based detector. Maintains a variable-length window of binary
    outcomes (correct/incorrect). Signals drift when the mean error of
    the most recent sub-window differs significantly from the rest.
    Strong for sudden drift. River's ADWIN is used internally.

DDM (Drift Detection Method)
    Monitors the running mean and standard deviation of the binary error
    signal. Raises a WARNING when error + std exceeds the warning threshold,
    and DRIFT when it exceeds the drift threshold.
    Lightweight — constant memory regardless of stream length.

PageHinkley
    Cumulative-sum test on the error signal. Fires when the cumulative
    deviation from the running mean exceeds a threshold δ.
    Fastest to respond but requires threshold tuning.

All detectors expose a unified interface:
    update(y_true, y_pred) -> None
    drift_detected         -> bool   (True if drift was signalled this step)
    warning_detected       -> bool   (True if in warning zone, where supported)
    n_detections           -> int    (cumulative drift count)
    reset()                -> None   (clear state for next window)

Usage
-----
    from methods.drift_detector import ADWIN, DDM, PageHinkley
    from data.generators import SEAConceptStream

    stream   = SEAConceptStream(n_samples=10_000, drift_at=5_000)
    detector = ADWIN(delta=0.002)

    errors = []
    for i, (x, y_true) in enumerate(stream):
        y_pred = model.predict_one(x)
        model.learn_one(x, y_true)

        detector.update(y_true, y_pred)
        if detector.drift_detected:
            print(f"Drift detected at sample {i}")
"""

from __future__ import annotations

from river import drift as river_drift  # used by ADWIN and PageHinkley


# ---------------------------------------------------------------------------
# ADWIN
# ---------------------------------------------------------------------------

class ADWIN:
    """
    ADaptive WINdowing drift detector (Bifet & Gavalda, 2007).

    ADWIN maintains a variable-length window over the binary error signal
    (1 = wrong prediction, 0 = correct). It tests all pairs of sub-windows
    within the window; if any pair shows a statistically significant
    difference in mean, drift is detected and the window is cut.

    Parameters
    ----------
    delta : float
        Confidence parameter. Lower delta → more conservative (fewer false
        positives but slower detection). Typical values: 0.002 – 0.1.

    Notes
    -----
    ADWIN is the default detector in River's AdaptiveRandomForest.
    It is well-suited for sudden drift but can lag on gradual drift.
    """

    def __init__(self, delta: float = 0.002) -> None:
        self.delta = delta
        self._detector = river_drift.ADWIN(delta=delta)
        self._n_detections = 0
        self._drift_detected = False

    def update(self, y_true: int, y_pred: int) -> None:
        """
        Feed one binary error signal into the detector.

        Call this AFTER both predict_one and learn_one for the same sample.
        The error is 1 if the prediction was wrong, 0 if it was correct.
        """
        error = int(y_true != y_pred)
        self._detector.update(error)
        self._drift_detected = self._detector.drift_detected
        if self._drift_detected:
            self._n_detections += 1

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected

    @property
    def warning_detected(self) -> bool:
        # ADWIN does not have a separate warning zone — drift is binary
        return False

    @property
    def n_detections(self) -> int:
        return self._n_detections

    def reset(self) -> None:
        self._detector = river_drift.ADWIN(delta=self.delta)
        self._drift_detected = False

    def __repr__(self) -> str:
        return f"ADWIN(delta={self.delta}, n_detections={self._n_detections})"


# ---------------------------------------------------------------------------
# DDM — Drift Detection Method
# ---------------------------------------------------------------------------

class DDM:
    """
    Drift Detection Method — pure Python implementation (Gama et al., 2004).

    DDM tracks the running mean (p) and standard deviation (s) of the
    binary error signal. It maintains two thresholds:

        warning zone : p + s > p_min + warning_level * s_min
        drift zone   : p + s > p_min + drift_level  * s_min

    Where p_min and s_min are the minimum values of (p + s) observed
    since the last detected drift or the start of the stream.

    This is a pure Python implementation — it does not depend on River's
    drift module, making it portable to any environment.

    Advantages over ADWIN
    ---------------------
    - Constant memory (O(1)) — stores only p, s, p_min, s_min, n
    - No window scanning
    - Separate warning zone alerts you before full drift is confirmed

    Limitation
    ----------
    DDM assumes the error rate is stationary between drift points. Long
    gradual drifts keep p+s elevated, which prevents p_min / s_min from
    updating, causing DDM to miss slow drift. Use PageHinkley for gradual
    drift and ADWIN for sudden drift.

    Parameters
    ----------
    warning_level   : float  Deviations above minimum for warning zone.
    drift_level     : float  Deviations above minimum for drift signal.
    min_n_instances : int    Minimum samples before detection activates.
    """

    def __init__(
        self,
        warning_level: float = 2.0,
        drift_level: float = 3.0,
        min_n_instances: int = 30,
    ) -> None:
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_n_instances = min_n_instances
        self._reset_stats()

    def _reset_stats(self) -> None:
        self._n: int = 0
        self._p: float = 1.0          # running error mean
        self._p_min: float = float("inf")
        self._s_min: float = float("inf")
        self._drift_detected: bool = False
        self._warning_detected: bool = False
        self._n_detections: int = getattr(self, "_n_detections", 0)

    def update(self, y_true: int, y_pred: int) -> None:
        """
        Feed one binary error signal (1 = wrong, 0 = correct) and update state.
        """
        import math

        error = int(y_true != y_pred)
        self._n += 1

        if self._n < self.min_n_instances:
            self._drift_detected = False
            self._warning_detected = False
            # Welford online mean
            self._p += (error - self._p) / self._n
            return

        # Online mean update
        self._p += (error - self._p) / self._n
        s = math.sqrt(self._p * (1.0 - self._p) / self._n)

        # Update minimum reference level
        if self._p + s <= self._p_min + self._s_min:
            self._p_min = self._p
            self._s_min = s

        # Check thresholds
        level = self._p + s
        ref_drift   = self._p_min + self.drift_level   * self._s_min
        ref_warning = self._p_min + self.warning_level * self._s_min

        if level > ref_drift:
            self._drift_detected = True
            self._warning_detected = False
            self._n_detections += 1
            self._reset_stats()   # reset after drift
        elif level > ref_warning:
            self._warning_detected = True
            self._drift_detected = False
        else:
            self._drift_detected = False
            self._warning_detected = False

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected

    @property
    def warning_detected(self) -> bool:
        return self._warning_detected

    @property
    def n_detections(self) -> int:
        return self._n_detections

    def reset(self) -> None:
        self._reset_stats()

    def __repr__(self) -> str:
        return (
            f"DDM(warning_level={self.warning_level}, "
            f"drift_level={self.drift_level}, "
            f"n_detections={self._n_detections})"
        )


# ---------------------------------------------------------------------------
# Page-Hinkley Test
# ---------------------------------------------------------------------------

class PageHinkley:
    """
    Page-Hinkley cumulative-sum test for drift detection.

    Tracks the cumulative deviation of the error signal from a running mean.
    Signals drift when the maximum minus current cumulative sum exceeds
    the threshold δ.

    Formula
    -------
        U_t = U_{t-1} + (x_t - x̄_t - ε)
        PH_t = max(U) - U_t
        Drift if PH_t > δ

    Where ε is a small allowance for gradual increase that should not be
    flagged (prevents hypersensitivity to small upward trends).

    Best used for
    -------------
    Gradual drift over long streams. Page-Hinkley is faster to respond
    than DDM in this scenario because it does not require the error rate
    to first reach a minimum before monitoring begins.

    Parameters
    ----------
    min_instances  : int   Warmup period before detection activates.
    delta          : float Allowance for natural error increase (ε).
    threshold      : float Detection threshold (δ). Higher → less sensitive.
    alpha          : float Forgetting factor for the mean estimate (0–1).
                    1.0 = standard cumulative mean; < 1.0 = exponential mean.
    """

    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 1 - 0.0001,
    ) -> None:
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha

        self._detector = river_drift.PageHinkley(
            min_instances=min_instances,
            delta=delta,
            threshold=threshold,
            alpha=alpha,
        )
        self._n_detections = 0
        self._drift_detected = False

    def update(self, y_true: int, y_pred: int) -> None:
        error = int(y_true != y_pred)
        self._detector.update(error)
        self._drift_detected = self._detector.drift_detected
        if self._drift_detected:
            self._n_detections += 1

    @property
    def drift_detected(self) -> bool:
        return self._drift_detected

    @property
    def warning_detected(self) -> bool:
        return False  # Page-Hinkley has no warning zone

    @property
    def n_detections(self) -> int:
        return self._n_detections

    def reset(self) -> None:
        self._detector = river_drift.PageHinkley(
            min_instances=self.min_instances,
            delta=self.delta,
            threshold=self.threshold,
            alpha=self.alpha,
        )
        self._drift_detected = False

    def __repr__(self) -> str:
        return (
            f"PageHinkley(threshold={self.threshold}, "
            f"n_detections={self._n_detections})"
        )
