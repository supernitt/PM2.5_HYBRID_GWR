"""Preprocessing utilities: missing-value filtering, VIF, standardization."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("pm25_hybrid")


def validate_ranges(df: pd.DataFrame, valid_ranges: dict[str, list],
                    max_masked_frac: float = 0.10) -> pd.DataFrame:
    """Mask physically impossible values (GEE sentinels, unit errors) to NaN.

    valid_ranges maps column -> [min, max] in the panel's native units.
    Values outside the range are set to NaN and counted; a column losing
    more than *max_masked_frac* of its non-null values raises, because that
    indicates systemic corruption (e.g. -9999 sentinels propagated through
    a derived variable) rather than a few bad records.

    This guard exists because run0 trained on a panel where -9999 u/v wind
    components produced ws = 14,140 m/s in 1,427 rows and inflated pooled
    VIFs to 10^5-10^6.  It must stay first in the preprocess order.
    """
    df = df.copy()
    for col, (lo, hi) in valid_ranges.items():
        if col not in df.columns:
            continue
        vals = df[col]
        n_valid = int(vals.notna().sum())
        if n_valid == 0:
            continue
        bad = vals.notna() & ((vals < lo) | (vals > hi))
        n_bad = int(bad.sum())
        if n_bad == 0:
            continue
        frac = n_bad / n_valid
        logger.warning("Range guard [%s]: %d value(s) outside [%g, %g] "
                       "(%.2f%% of non-null) masked to NaN. Sample: %s",
                       col, n_bad, lo, hi, 100 * frac,
                       vals[bad].head(3).tolist())
        if frac > max_masked_frac:
            raise ValueError(
                f"Column '{col}': {100 * frac:.1f}% of values outside "
                f"[{lo}, {hi}] — systemic corruption, refusing to continue. "
                f"Re-derive the panel instead of masking this much data."
            )
        df.loc[bad, col] = np.nan
    return df


def impute_aod(df: pd.DataFrame, aod: str, station_id: str, date_col: str,
               max_gap_days: int = 7) -> pd.DataFrame:
    """Disclosed AOD gap-filling for the sensitivity arm ONLY.

    Per-station linear interpolation in time, restricted to interior gaps of
    at most *max_gap_days* consecutive days.  Edge gaps are never filled and
    long monsoon gaps (wet-season coverage is ~11%) are left as NaN so they
    are dropped by filter_missing rather than fabricated.

    Adds a boolean column ``aod_imputed`` so downstream analyses can compare
    observed-vs-imputed subsets, and logs the imputed fraction — both are
    required for the manuscript's disclosure statement.
    """
    df = df.copy()
    observed = df[aod].notna()
    parts = []
    for _, grp in df.groupby(station_id, sort=False):
        grp = grp.sort_values(date_col).copy()
        s = grp.set_index(date_col)[aod]
        # Reindex to a DAILY grid before interpolating: pandas' `limit` counts
        # consecutive NaN *entries*, not calendar days, and station rows cover
        # only ~31% of calendar days — without the reindex a 21-day gap holding
        # 2 NaN rows would be bridged and miscounted as "<= 7 d".
        s = s[~s.index.duplicated()].asfreq("D")
        filled = s.interpolate(method="time", limit=max_gap_days,
                               limit_area="inside")
        grp[aod] = filled.reindex(grp[date_col]).to_numpy()
        parts.append(grp)
    out = pd.concat(parts).loc[df.index]
    out["aod_imputed"] = out[aod].notna() & ~observed
    n_obs = int(observed.sum())
    n_imp = int(out["aod_imputed"].sum())
    logger.info("AOD imputation (sensitivity arm): %d observed + %d imputed "
                "(gap <= %d d) = %d usable of %d rows (%.1f%% imputed share)",
                n_obs, n_imp, max_gap_days, n_obs + n_imp, len(out),
                100 * n_imp / max(n_obs + n_imp, 1))
    return out


def drop_missing_predictors(df: pd.DataFrame,
                            predictors: list[str]) -> pd.DataFrame:
    """Drop rows with NaN in any predictor, logging the per-column cause.

    Replaces the old approach of interpolating+smoothing every predictor to
    completeness, which silently fabricated 66% of AOD and manufactured
    collinearity.  Honest rows only; the log shows what each column cost.
    """
    present = [p for p in predictors if p in df.columns]
    n0 = len(df)
    per_col = {p: int(df[p].isna().sum()) for p in present}
    culprits = {p: n for p, n in per_col.items() if n > 0}
    if culprits:
        logger.info("Predictor NaN counts before row drop: %s", culprits)
    out = df.dropna(subset=present).reset_index(drop=True)
    logger.info("Dropped rows with missing predictors: %d -> %d (%d removed)",
                n0, len(out), n0 - len(out))
    return out


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
