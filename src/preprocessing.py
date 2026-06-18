"""Preprocessing utilities: missing-value filtering, VIF, standardization."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("pm25_hybrid")


def filter_missing(df: pd.DataFrame, target: str, aod: str,
                   drop_target: bool = True, drop_aod: bool = True) -> pd.DataFrame:
    n0 = len(df)
    if drop_target:
        df = df.dropna(subset=[target])
    if drop_aod:
        df = df.dropna(subset=[aod])
    n1 = len(df)
    logger.info("Filtered missing values: %d -> %d rows (%d removed)", n0, n1, n0 - n1)
    return df.reset_index(drop=True)


def compute_vif(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """Variance inflation factor for each predictor.

    A row is dropped if any predictor in the row is NaN.
    """
    X = df[predictors].dropna().astype(float).values
    if X.shape[0] < X.shape[1] + 1:
        logger.warning("Not enough rows for VIF calculation.")
        return pd.DataFrame({"predictor": predictors, "VIF": np.nan})

    vifs = []
    for j, name in enumerate(predictors):
        y_j = X[:, j]
        X_j = np.delete(X, j, axis=1)
        # Add intercept
        X_j = np.column_stack([np.ones(len(X_j)), X_j])
        # Solve OLS: beta = (X'X)^-1 X'y
        try:
            beta, *_ = np.linalg.lstsq(X_j, y_j, rcond=None)
            y_pred = X_j @ beta
            ss_res = np.sum((y_j - y_pred) ** 2)
            ss_tot = np.sum((y_j - y_j.mean()) ** 2)
            r2_j = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            vif = 1.0 / (1.0 - r2_j) if r2_j < 1 else np.inf
        except np.linalg.LinAlgError:
            vif = np.inf
        vifs.append(vif)

    return pd.DataFrame({"predictor": predictors, "VIF": vifs})


def compute_vif_within_station(df: pd.DataFrame, predictors: list[str],
                               station_id: str) -> pd.DataFrame:
    """VIF computed on within-station demeaned predictors.

    In an LME with random intercepts, fixed effects are estimated from
    within-station temporal variation.  Pooled VIF is dominated by
    between-station differences (altitude, climate zone) and gives
    pathologically large values for ERA5-Land fields that all share a
    monsoon seasonal cycle.  Demeaning each predictor by its per-station
    mean removes the between-station component and gives the collinearity
    that actually matters for the LME fixed effects.
    """
    cols_needed = [station_id] + predictors
    sub = df[cols_needed].dropna().copy()

    for p in predictors:
        sub[p] = sub[p] - sub.groupby(station_id)[p].transform("mean")

    return compute_vif(sub, predictors)


def standardize(df: pd.DataFrame, predictors: list[str],
                fit_on: pd.DataFrame | None = None) -> tuple[pd.DataFrame, StandardScaler]:
    """Z-score standardization of predictors.

    If fit_on is provided, the scaler is fitted on that frame (training)
    and applied to df (testing). Otherwise, fit_transform is done on df.
    """
    scaler = StandardScaler()
    out = df.copy()
    if fit_on is not None:
        scaler.fit(fit_on[predictors].values)
    else:
        scaler.fit(out[predictors].values)
    out[predictors] = scaler.transform(out[predictors].values)
    return out, scaler


def smooth_predictors(df: pd.DataFrame,
                      predictors: list[str],
                      station_id: str,
                      date_col: str,
                      window_length: int = 11,
                      polyorder: int = 3,
                      static_predictors: list[str] | None = None) -> pd.DataFrame:
    """Per-station gap-fill then Savitzky-Golay smooth for temporal predictors.

    Steps applied column-by-column, station-by-station:
      1. Sort by date.
      2. Linear interpolation (time-based) to fill interior NaN gaps; forward/
         backward fill for any remaining edge NaN.
      3. Savitzky-Golay filter to smooth the now-complete series.

    Columns listed in *static_predictors* (e.g. ``elev``, ``pop``) are skipped
    because they carry a single constant value per station and smoothing is
    meaningless.  Columns that are entirely NaN for a station are also skipped
    (they will be removed later by :func:`drop_all_nan_predictors`).
    """
    static_set = set(static_predictors or [])
    temporal = [p for p in predictors if p not in static_set and p in df.columns]

    # window_length must be odd and > polyorder
    wl = window_length if window_length % 2 == 1 else window_length + 1

    parts = []
    for sid, grp in df.groupby(station_id, sort=False):
        grp = grp.sort_values(date_col).copy()
        for col in temporal:
            s = grp[col].copy()
            if s.isna().all():
                continue  # nothing to smooth; leave as NaN

            # Step 1: interpolate gaps
            s = s.interpolate(method="linear", limit_direction="both")

            # Step 2: SG filter — need at least wl non-NaN points
            n_valid = int(s.notna().sum())
            effective_wl = wl
            while effective_wl > n_valid and effective_wl > polyorder + 2:
                effective_wl -= 2  # shrink to next smaller odd number
            if n_valid >= effective_wl > polyorder:
                smoothed = savgol_filter(s.to_numpy(dtype=float),
                                         effective_wl, polyorder)
                s = pd.Series(smoothed, index=s.index)

            grp[col] = s
        parts.append(grp)

    result = pd.concat(parts).loc[df.index]   # restore original row order
    logger.info("Smoothed %d temporal predictor(s) with SG(window=%d, poly=%d)",
                len(temporal), wl, polyorder)
    return result


def drop_all_nan_predictors(df: pd.DataFrame,
                            predictors: list[str]) -> list[str]:
    """Return *predictors* with any entirely-NaN column removed.

    Columns that are all-NaN (e.g. ``elev``/``pop`` before GEE static
    extraction) would make statsmodels drop all rows from endog/exog while
    the separately-passed ``groups`` array keeps its full length, causing a
    shape-mismatch error.  Removing them here prevents that crash.
    """
    valid, dropped = [], []
    for p in predictors:
        if p in df.columns and df[p].notna().any():
            valid.append(p)
        else:
            dropped.append(p)
    if dropped:
        logger.warning(
            "Predictor(s) dropped from model (all-NaN — not yet extracted?): %s",
            dropped,
        )
    return valid


def assemble_predictors(cfg_columns: dict) -> list[str]:
    """Combine AOD + meteo + ancillary into a single predictor list."""
    return [cfg_columns["aod"]] + list(cfg_columns["meteo"]) + list(cfg_columns["ancillary"])
