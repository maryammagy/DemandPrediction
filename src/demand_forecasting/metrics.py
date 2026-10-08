"""Evaluation metric."""

from __future__ import annotations

import numpy as np
import pandas as pd


def wape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Weighted absolute percentage error: ``sum|y_pred - y_true| / sum|y_true|``.

    Unlike MAPE it is demand-weighted, so a handful of tiny cities with huge relative errors
    barely move it - which is exactly what a volume-driven business wants. Computed over all
    (cluster, check-in) cells at once, not averaged per cluster.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true).sum()
    if denominator == 0:
        return float("nan")
    return float(np.abs(y_pred - y_true).sum() / denominator)
