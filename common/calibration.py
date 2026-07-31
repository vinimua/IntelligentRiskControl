"""Serialization-compatible calibrators for the frozen Champion V1 bundles.

From: risk_inquiry_agent/common/calibration.py
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=0)

    def fit(self, scores, labels):
        self.model.fit(np.asarray(scores).reshape(-1, 1), labels)
        return self

    def predict(self, scores):
        return self.model.predict_proba(np.asarray(scores).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    def __init__(self) -> None:
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, scores, labels):
        self.model.fit(np.asarray(scores), labels)
        return self

    def predict(self, scores):
        return self.model.predict(np.asarray(scores))
