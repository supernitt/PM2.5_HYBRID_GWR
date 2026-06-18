"""Performance metrics: R2, RMSE, MAE, mean bias.

These follow the equations in the Model Validation subsection of the
manuscript (Eq. R2, RMSE, MAE, MB).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MB = mean(pred - obs). Positive => over-prediction on average."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(y_pred - y_true))


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return a dict of all four metrics."""
    return {
        "R2": r2(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MB": mean_bias(y_true, y_pred),
        "n": int(len(y_true)),
    }


def metrics_table(records: list[dict]) -> pd.DataFrame:
    """Convert a list of metric dicts (with 'model' and 'scheme' keys) to df."""
    return pd.DataFrame(records)
