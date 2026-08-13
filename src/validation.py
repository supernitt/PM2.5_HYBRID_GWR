"""Cross-validation routines used by the manuscript.

Two schemes:
  * Random 10-fold CV — splits station-day records randomly.
  * Site-based  CV  — splits by station (leave-stations-out).

Both schemes report R2, RMSE, MAE, and MB overall and stratified by season
and (optionally) Thai region.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd
try:
    from sklearn.utils.parallel import Parallel, delayed
except ImportError:          # sklearn < 1.1
    from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from . import metrics

logger = logging.getLogger("pm25_hybrid")


# ---------------------------------------------------------------------
@dataclass
class FoldResult:
    fold: int
    y_true: np.ndarray
    y_pred: np.ndarray
    season: np.ndarray
    region: np.ndarray
    station_id: np.ndarray
    date: np.ndarray | None = None   # per-row dates; enables day-level
                                     # diagnostics (dRMSE vs Moran's I)


# ---------------------------------------------------------------------
def random_kfold_indices(n: int, n_splits: int = 10, seed: int = 42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n)))


def site_kfold_indices(stations: np.ndarray, n_splits: int = 10, seed: int = 42):
    """Yield (train_idx, test_idx) where each test fold holds out whole stations."""
    unique_sids = np.array(sorted(np.unique(stations)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_sids)
    folds = np.array_split(unique_sids, n_splits)
    splits = []
    for fold in folds:
        test_mask = np.isin(stations, fold)
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        splits.append((train_idx, test_idx))
    return splits


# ---------------------------------------------------------------------
def _run_fold(k: int, tr: np.ndarray, te: np.ndarray,
              df: pd.DataFrame,
              train_fn: Callable, predict_fn: Callable,
              target_col: str, model_name: str,
              n_folds: int = 0,
              scale_predictors: list[str] | None = None) -> FoldResult | None:
    fold_label = f"fold {k + 1}/{n_folds}" if n_folds else f"fold {k + 1}"
    logger.info("[%s] %s — training on %d rows, testing on %d rows ...",
                model_name, fold_label, len(tr), len(te))
    import time as _time
    _t0 = _time.time()
    train_df = df.iloc[tr].reset_index(drop=True)
    test_df = df.iloc[te].reset_index(drop=True)

    # Per-fold standardization: fit the scaler on the training fold ONLY and
    # apply it to both.  A scaler fitted on the full panel leaks test-fold
    # moments into training — especially under site-based CV, where held-out
    # stations' climate must stay unseen.
    if scale_predictors:
        from sklearn.preprocessing import StandardScaler
        _scaler = StandardScaler().fit(train_df[scale_predictors].values)
        train_df[scale_predictors] = _scaler.transform(
            train_df[scale_predictors].values)
        test_df[scale_predictors] = _scaler.transform(
            test_df[scale_predictors].values)
    try:
        model = train_fn(train_df)
        y_pred = predict_fn(model, test_df)
    except Exception as e:
        logger.error("Fold %d failed for %s: %s", k, model_name, e)
        return None
    logger.info("[%s] %s done (%.1f min)", model_name, fold_label, (_time.time() - _t0) / 60)
    return FoldResult(
        fold=k,
        y_true=test_df[target_col].to_numpy(),
        y_pred=np.asarray(y_pred, dtype=float),
        season=test_df.get("season", pd.Series([None] * len(test_df))).to_numpy(),
        region=test_df.get("region", pd.Series([None] * len(test_df))).to_numpy(),
        station_id=test_df.get("station_id",
                               pd.Series([None] * len(test_df))).to_numpy(),
        date=test_df.get("date", pd.Series([pd.NaT] * len(test_df))).to_numpy(),
    )


def run_cv(df: pd.DataFrame,
           splits: list[tuple[np.ndarray, np.ndarray]],
           train_fn: Callable[[pd.DataFrame], object],
           predict_fn: Callable[[object, pd.DataFrame], np.ndarray],
           target_col: str,
           scheme_name: str,
           model_name: str,
           n_jobs: int = 1,
           scale_predictors: list[str] | None = None
           ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run cross-validation.

    Returns
    -------
    overall_df   : one-row DataFrame with pooled R2, RMSE, MAE, MB, n
    strat_df     : stratified metrics by season / region
    cv_preds_df  : raw fold predictions — columns:
                   [model, scheme, fold, y_true, y_pred, season, region, station_id]
                   Use this for scatter plots, residual maps, and per-fold error bars.

    Parameters
    ----------
    n_jobs : parallel folds. Use 1 when the model already parallelises
             internally (e.g. HybridLMEGWR with GWR n_jobs=-1) to avoid
             nested parallelism. Use -1 for pure sklearn benchmarks.
    scale_predictors : if given, z-score these columns per fold (scaler fitted
             on the training fold only, applied to both).  Pass the RAW,
             unstandardized panel when using this.
    """
    n_folds = len(splits)
    logger.info("Starting %d-fold CV for %s (scheme: %s) ...", n_folds, model_name, scheme_name)
    raw = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_fold)(k, tr, te, df, train_fn, predict_fn, target_col, model_name, n_folds,
                           scale_predictors=scale_predictors)
        for k, (tr, te) in enumerate(splits)
    )
    fold_results: list[FoldResult] = [r for r in raw if r is not None]

    if not fold_results:
        empty = pd.DataFrame()
        return empty, empty, empty

    y_true_all = np.concatenate([r.y_true for r in fold_results])
    y_pred_all = np.concatenate([r.y_pred for r in fold_results])
    fold_ids   = np.concatenate([[r.fold] * len(r.y_true) for r in fold_results])
    season_all = np.concatenate([r.season     for r in fold_results])
    region_all = np.concatenate([r.region     for r in fold_results])
    sid_all    = np.concatenate([r.station_id for r in fold_results])
    date_all   = np.concatenate([
        r.date if r.date is not None else np.array([None] * len(r.y_true))
        for r in fold_results])

    # Drop any predictions that came back as NaN
    mask = ~np.isnan(y_pred_all) & ~np.isnan(y_true_all)
    y_true_all = y_true_all[mask]
    y_pred_all = y_pred_all[mask]
    season_all = season_all[mask]
    region_all = region_all[mask]
    sid_all    = sid_all[mask]
    fold_ids   = fold_ids[mask]
    date_all   = date_all[mask]

    # ── Raw predictions DataFrame ─────────────────────────────────────
    cv_preds_df = pd.DataFrame({
        "model":      model_name,
        "scheme":     scheme_name,
        "fold":       fold_ids,
        "date":       date_all,
        "y_true":     y_true_all,
        "y_pred":     y_pred_all,
        "season":     season_all,
        "region":     region_all,
        "station_id": sid_all,
    })

    # ── Pooled overall metrics ────────────────────────────────────────
    overall = {
        "model": model_name,
        "scheme": scheme_name,
        **metrics.all_metrics(y_true_all, y_pred_all),
    }

    # ── Stratified metrics ────────────────────────────────────────────
    strat_rows: list[dict] = []
    for season in pd.unique(season_all):
        if pd.isna(season):
            continue
        sm = season_all == season
        if sm.sum() < 5:
            continue
        strat_rows.append({
            "model": model_name, "scheme": scheme_name,
            "stratum": "season", "value": str(season),
            **metrics.all_metrics(y_true_all[sm], y_pred_all[sm]),
        })
    for region in pd.unique(region_all):
        if pd.isna(region):
            continue
        rm = region_all == region
        if rm.sum() < 5:
            continue
        strat_rows.append({
            "model": model_name, "scheme": scheme_name,
            "stratum": "region", "value": str(region),
            **metrics.all_metrics(y_true_all[rm], y_pred_all[rm]),
        })

    return pd.DataFrame([overall]), pd.DataFrame(strat_rows), cv_preds_df
