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


def _detect_xgboost() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


# Real availability check — `XGBModel  # probe import` at a call site does
# NOT raise ImportError (the class is defined unconditionally; the error only
# fires inside XGBModel.__init__). That illusory guard let every CV fold fail
# silently (caught by the fold-level try/except) instead of skipping cleanly.
XGB_AVAILABLE: bool = _detect_xgboost()

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


def tune_rf(X, y, grid: dict, n_splits=10, seed=42,
            n_jobs=8, inner_n_jobs=6) -> dict:
    """RF grid search with OOM-safe widths (2026-08-08 scheme, re-applied).

    outer 8 x inner 6: each thread in the outer pool holds a fully grown RF
    (measured 4.4 GB pickled at ~35k rows).  A 48- or 96-wide outer pool
    OOM-killed this machine at 122.7 GB RSS — do NOT raise these defaults
    without going through the compute-resource gate.
    """
    return _grid_search_tree(X, y, lambda **p: RFModel(n_jobs=inner_n_jobs, **p),
                             grid, n_splits, seed, n_jobs)


# =====================================================================
# GB  (uses HistGradientBoostingRegressor — same algorithm as LightGBM/XGBoost,
#      much faster than the classic GradientBoostingRegressor)
# =====================================================================
class GBModel:
    def __init__(self, n_threads: int | None = None, **params):
        # Config uses 'n_estimators'; HistGBR calls the same thing 'max_iter'.
        # n_threads: cap on HistGBR's OpenMP pool during fit/predict.  HistGBR
        # has no n_jobs parameter and sizes its pool to ALL cores per instance;
        # inside a parallel grid search that is pool_width x 48 threads.
        hist_params = dict(params)
        if "n_estimators" in hist_params:
            hist_params["max_iter"] = hist_params.pop("n_estimators")
        self.params = hist_params
        self.n_threads = n_threads
        self.model = HistGradientBoostingRegressor(random_state=42, **hist_params)

    def _limit(self):
        from contextlib import nullcontext
        if self.n_threads is None:
            return nullcontext()
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=self.n_threads, user_api="openmp")

    def fit(self, X, y):
        with self._limit():
            self.model.fit(X, y)
        return self

    def predict(self, X):
        with self._limit():
            return self.model.predict(X)


def tune_gb(X, y, grid: dict, n_splits=10, seed=42,
            n_jobs=16, inner_threads=3) -> dict:
    """GB grid search, OOM/thread-safe widths (2026-08-08 scheme, re-applied):
    outer 16 x OpenMP 3.  See tune_rf docstring before raising."""
    return _grid_search_tree(X, y, lambda **p: GBModel(n_threads=inner_threads, **p),
                             grid, n_splits, seed, n_jobs)


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


def tune_xgb(X, y, grid: dict, n_splits=10, seed=42,
             n_jobs=12, inner_n_jobs=4) -> dict:
    # Hyperparameter search runs on CPU in parallel — no VRAM competition.
    # OOM-safe widths (2026-08-08 scheme, re-applied): outer 12 x inner 4.
    # The final full-data fit (called separately by the caller with
    # XGBModel()) uses GPU via force_cpu=False (default).
    logger.info("XGBoost grid search: using CPU (outer %d x inner %d) to avoid "
                "GPU VRAM contention; final model will use GPU.",
                n_jobs, inner_n_jobs)
    return _grid_search_tree(
        X, y, lambda **p: XGBModel(n_jobs=inner_n_jobs, force_cpu=True, **p),
        grid, n_splits, seed, n_jobs,
    )
