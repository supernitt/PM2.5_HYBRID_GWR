"""Benchmark regressors used as counterparts to the hybrid LME-GWR model.

Two families are implemented, all sharing the same predictor set:

  * Component-based regressors:
      - PCR  (PCA + linear regression)
      - PLSR (partial least squares)

  * Tree-based ensembles:
      - RF      (RandomForestRegressor)
      - GB      (GradientBoostingRegressor)
      - XGBoost (xgboost.XGBRegressor)

Each model is wrapped in a small object with .fit(X, y), .predict(X),
and tune_*(...) helpers for grid-search 10-fold CV. Hyper-parameters
are loaded from configs/default.yaml.
"""
from __future__ import annotations

import logging
from itertools import product

import subprocess

import numpy as np
import pandas as pd
try:
    from sklearn.utils.parallel import Parallel, delayed
except ImportError:          # sklearn < 1.1
    from joblib import Parallel, delayed
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import (HistGradientBoostingRegressor,
                               RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("pm25_hybrid")


def _detect_gpu() -> bool:
    """Return True if a CUDA GPU is visible AND XGBoost can actually use it."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return False
    except Exception:
        return False

    # nvidia-smi found a GPU — verify this XGBoost build can actually reach it.
    try:
        from xgboost import XGBRegressor
        import numpy as _np
        XGBRegressor(tree_method="hist", device="cuda",
                     n_estimators=1, verbosity=0).fit(
            _np.zeros((10, 2)), _np.zeros(10))
        return True
    except Exception:
        return False


_GPU_AVAILABLE: bool = _detect_gpu()
if _GPU_AVAILABLE:
    logger.info("CUDA GPU detected and verified — XGBoost will use device='cuda'.")
else:
    logger.debug("No usable CUDA GPU found — XGBoost will run on CPU.")


# =====================================================================
# PCR
# =====================================================================
class PCRModel:
    def __init__(self, n_components: int = 5):
        self.n_components = n_components
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("ols", LinearRegression()),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PCRModel":
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)


def tune_pcr(X: np.ndarray, y: np.ndarray, candidates: list[int],
             n_splits: int = 10, seed: int = 42) -> int:
    """Pick the n_components value that minimizes 10-fold CV RMSE."""
    best, best_rmse = candidates[0], np.inf
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for n in candidates:
        rmses = []
        for tr, te in kf.split(X):
            m = PCRModel(n_components=n).fit(X[tr], y[tr])
            yhat = m.predict(X[te])
            rmses.append(np.sqrt(np.mean((y[te] - yhat) ** 2)))
        rmse = float(np.mean(rmses))
        logger.debug("PCR n=%d -> RMSE=%.4f", n, rmse)
        if rmse < best_rmse:
            best, best_rmse = n, rmse
    logger.info("PCR best n_components=%d (CV RMSE=%.4f)", best, best_rmse)
    return best


# =====================================================================
# PLSR
# =====================================================================
class PLSRModel:
    def __init__(self, n_components: int = 5):
        self.n_components = n_components
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("plsr", PLSRegression(n_components=n_components)),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSRModel":
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X).ravel()


def tune_plsr(X: np.ndarray, y: np.ndarray, candidates: list[int],
              n_splits: int = 10, seed: int = 42) -> int:
    best, best_rmse = candidates[0], np.inf
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for n in candidates:
        if n >= X.shape[1]:
            continue
        rmses = []
        for tr, te in kf.split(X):
            m = PLSRModel(n_components=n).fit(X[tr], y[tr])
            yhat = m.predict(X[te])
            rmses.append(np.sqrt(np.mean((y[te] - yhat) ** 2)))
        rmse = float(np.mean(rmses))
        logger.debug("PLSR n=%d -> RMSE=%.4f", n, rmse)
        if rmse < best_rmse:
            best, best_rmse = n, rmse
    logger.info("PLSR best n_components=%d (CV RMSE=%.4f)", best, best_rmse)
    return best


# =====================================================================
# RF
# =====================================================================
def _eval_combo(combo, keys, X, y, factory, kf):
    params = dict(zip(keys, combo))
    rmses = []
    for tr, te in kf.split(X):
        m = factory(**params).fit(X[tr], y[tr])
        rmses.append(np.sqrt(np.mean((y[te] - m.predict(X[te])) ** 2)))
    return params, float(np.mean(rmses))


def _grid_search_tree(X: np.ndarray, y: np.ndarray, factory,
                      grid: dict, n_splits: int = 10, seed: int = 42,
                      n_jobs: int = -1) -> dict:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    combos = list(product(*values))
    n_total_fits = len(combos) * n_splits
    logger.info("Grid search: %d combinations × %d folds = %d model fits",
                len(combos), n_splits, n_total_fits)
    # prefer="threads": avoids spawning nested loky processes since the inner
    # tree models (RF, XGB) already use their own process pools.
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_eval_combo)(combo, keys, X, y, factory, kf)
        for combo in combos
    )
    best_params, best_rmse = min(results, key=lambda r: r[1])
    logger.info("Grid search done — best params: %s (CV RMSE=%.4f)", best_params, best_rmse)
    return best_params


class RFModel:
    def __init__(self, n_jobs: int = -1, **params):
        self.params = params
        # oob_score=True: computes out-of-bag predictions at no extra cost.
        # HybridTreeGWR uses oob_prediction_ to get unbiased residuals for GWR.
        self.model = RandomForestRegressor(random_state=42, n_jobs=n_jobs,
                                           oob_score=True, **params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


def tune_rf(X, y, grid: dict, n_splits=10, seed=42, n_jobs=-1) -> dict:
    # n_jobs=1 for inner model: outer Parallel already provides concurrency
    return _grid_search_tree(X, y, lambda **p: RFModel(n_jobs=1, **p), grid, n_splits, seed, n_jobs)


# =====================================================================
# GB  (uses HistGradientBoostingRegressor — same algorithm as LightGBM/XGBoost,
#      much faster than the classic GradientBoostingRegressor)
# =====================================================================
class GBModel:
    def __init__(self, **params):
        # Config uses 'n_estimators'; HistGBR calls the same thing 'max_iter'.
        hist_params = dict(params)
        if "n_estimators" in hist_params:
            hist_params["max_iter"] = hist_params.pop("n_estimators")
        self.params = hist_params
        self.model = HistGradientBoostingRegressor(random_state=42, **hist_params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


def tune_gb(X, y, grid: dict, n_splits=10, seed=42, n_jobs=-1) -> dict:
    return _grid_search_tree(X, y, lambda **p: GBModel(**p), grid, n_splits, seed, n_jobs)


# =====================================================================
# XGBoost
# =====================================================================
class XGBModel:
    def __init__(self, n_jobs: int = -1, force_cpu: bool = False, **params):
        """
        Parameters
        ----------
        force_cpu : if True, always run on CPU even when a GPU is available.
            Used during hyperparameter search so many parallel workers can run
            without competing for VRAM.  The final full-data fit should use
            force_cpu=False to take advantage of the GPU.
        """
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError("xgboost is not installed. `pip install xgboost`") from e
        self.params = params
        self._uses_gpu = False
        xgb_kwargs: dict = {"tree_method": "hist"}
        if _GPU_AVAILABLE and not force_cpu:
            xgb_kwargs["device"] = "cuda"
            n_jobs = 1          # GPU owns all parallelism; CPU threads are wasted
            self._uses_gpu = True
        xgb_kwargs.update(params)   # caller params take precedence
        self.model = XGBRegressor(random_state=42, n_jobs=n_jobs,
                                  objective="reg:squarederror", **xgb_kwargs)

    def fit(self, X, y):
        try:
            self.model.fit(X, y)
        except Exception as e:
            # GPU out-of-memory: rebuild without GPU and retry once.
            if self._uses_gpu and "cudaErrorMemoryAllocation" in str(e):
                logger.warning(
                    "XGBoost GPU out-of-memory — retrying on CPU (free VRAM too low)."
                )
                from xgboost import XGBRegressor
                cpu_params = {k: v for k, v in self.model.get_params().items()
                              if k != "device"}
                cpu_params["n_jobs"] = -1
                self.model = XGBRegressor(**cpu_params)
                self._uses_gpu = False
                self.model.fit(X, y)
            else:
                raise
        return self

    def predict(self, X):
        return self.model.predict(X)


def tune_xgb(X, y, grid: dict, n_splits=10, seed=42, n_jobs=-1) -> dict:
    # Hyperparameter search runs on CPU in parallel — no VRAM competition.
    # Each search model gets n_jobs=1 because the outer Parallel provides
    # concurrency across combos.  The final full-data fit (called separately
    # by the caller with XGBModel()) uses GPU via force_cpu=False (default).
    logger.info("XGBoost grid search: using CPU (parallel) to avoid GPU VRAM "
                "contention; final model will use GPU.")
    return _grid_search_tree(
        X, y, lambda **p: XGBModel(n_jobs=1, force_cpu=True, **p),
        grid, n_splits, seed, n_jobs,   # restore full outer parallelism
    )
