"""Hybrid LME-GWR estimator.

Implements Eq. (hybrid) from the manuscript:

    PM2.5_hat^hybrid = PM2.5_hat^LME  +  r_hat

where r_hat is the GWR-corrected LME residual at the same location-day.
On days when the LME residuals show no significant spatial autocorrelation,
the hybrid prediction reduces to the LME fitted value.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .gwr_model import GWRStage2, GWRStage2Output
from .lme_model import LMEModel

logger = logging.getLogger("pm25_hybrid")


class HybridLMEGWR:
    """Two-stage hybrid LME-GWR predictor."""

    def __init__(self, lme: LMEModel, gwr: GWRStage2,
                 station_id: str = "station_id",
                 date_col: str = "date"):
        self.lme = lme
        self.gwr = gwr
        self.station_id = station_id
        self.date_col = date_col
        self.gwr_output_: GWRStage2Output | None = None

    # -----------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "HybridLMEGWR":
        # Stage 1
        self.lme.fit(df)

        # Compute Stage-2 residuals using the CONDITIONAL LME prediction.
        # The conditional prediction absorbs each training station's random effect,
        # leaving only the spatially autocorrelated component for GWR to correct.
        # Using predict_fixed here (marginal) would mix random-effect noise into
        # the GWR targets, causing IDW interpolation to transmit that noise to
        # out-of-sample test stations and degrade CV performance.
        df = df.copy()
        df["lme_pred"] = self.lme.predict(df)        # conditional (RE included)
        df["lme_residual"] = df[self.lme.target] - df["lme_pred"]

        # Stage 2
        self.gwr_output_ = self.gwr.fit_all(df, residual_col="lme_residual")
        return self

    # -----------------------------------------------------------------
    def predict(self, df: pd.DataFrame, k_neighbors: int = 8) -> np.ndarray:
        """Predict PM2.5 at station-day records.

        For in-sample stations (present in the GWR training day), the stored
        GWR correction is applied directly.  For out-of-sample stations
        (e.g. held-out CV folds), the GWR coefficients are interpolated from
        neighbouring training stations via IDW — the same approach used in
        predict_grid().  Days with no fitted GWR result fall back to LME.
        """
        if self.gwr_output_ is None:
            raise RuntimeError("Hybrid model has not been fitted yet.")

        # Use the CONDITIONAL LME prediction as the baseline.
        # GWR was trained on y − conditional_lme (pure spatial residuals after
        # removing random effects), so conditional_lme + GWR_correction is the
        # correct two-stage combination — no double-counting.
        # For unknown stations (site-based CV), predict() automatically falls back
        # to the marginal (fixed-effects only) prediction, so no RE leakage occurs.
        lme_pred = self.lme.predict(df)
        out = lme_pred.copy()

        for date, day_idx in df.groupby(self.date_col).indices.items():
            day_idx = np.asarray(day_idx)
            res = self.gwr_output_.daily_results.get(pd.Timestamp(date))
            if res is None or not res.fitted or res.coefficients is None:
                continue

            coef_df = res.coefficients
            coef_cols = [c for c in coef_df.columns
                         if c not in ("lon", "lat", "station_id")]
            active_predictors = [c for c in coef_cols if c != "intercept"]
            station_coords = coef_df[["lon", "lat"]].to_numpy()
            station_coefs = coef_df[coef_cols].to_numpy()
            sid_to_correction = dict(zip(coef_df["station_id"].values,
                                         res.residual_correction))

            # Split day records into exact (in-sample) and out-of-sample
            exact, oos = [], []
            for i in day_idx:
                sid = df.iloc[i][self.station_id]
                if sid in sid_to_correction:
                    exact.append(i)
                else:
                    oos.append(i)

            # In-sample: apply stored GWR correction directly
            for i in exact:
                out[i] = lme_pred[i] + float(sid_to_correction[df.iloc[i][self.station_id]])

            # Out-of-sample: interpolate GWR coefficients via IDW
            if oos:
                oos = np.asarray(oos)
                oos_coords = df.iloc[oos][[self.gwr.lon_col,
                                           self.gwr.lat_col]].to_numpy()
                interp = _idw_interpolate(oos_coords, station_coords,
                                          station_coefs, k=k_neighbors)
                X_oos = np.column_stack([
                    np.ones(len(oos)),
                    df.iloc[oos][active_predictors].to_numpy(),
                ])
                out[oos] = lme_pred[oos] + np.einsum("ij,ij->i", X_oos, interp)

        return out

    # -----------------------------------------------------------------
    def predict_grid(self, grid_df: pd.DataFrame,
                     k_neighbors: int = 8) -> np.ndarray:
        """Predict PM2.5 on grid cells.

        For each day in grid_df:
          1. Compute LME baseline at every grid cell using fixed-effects only
             (no station random effect available off-network).
          2. If Stage-2 was fitted that day, interpolate the station-level GWR
             coefficients to each grid cell using inverse-distance weights of
             the k nearest stations, then apply them to the predictors.
        """
        if self.gwr_output_ is None:
            raise RuntimeError("Hybrid model has not been fitted yet.")

        # Stage-1: marginal LME baseline
        lme_pred = self.lme.predict_fixed(grid_df)
        out = lme_pred.copy()

        for date, day_idx in grid_df.groupby(self.date_col).indices.items():
            day_idx = np.asarray(day_idx)
            res = self.gwr_output_.daily_results.get(pd.Timestamp(date))
            if res is None or not res.fitted or res.coefficients is None:
                continue

            # Use only the predictors that were active on this day (zero-variance
            # columns are excluded from coef_df at fit time).
            coef_cols = [c for c in res.coefficients.columns
                         if c not in ("lon", "lat", "station_id")]
            active_predictors = [c for c in coef_cols if c != "intercept"]

            station_coords = res.coefficients[["lon", "lat"]].to_numpy()
            station_coefs = res.coefficients[coef_cols].to_numpy()

            grid_coords = grid_df.iloc[day_idx][[self.gwr.lon_col,
                                                 self.gwr.lat_col]].to_numpy()

            # IDW interpolation of coefficients to grid cells
            interp = _idw_interpolate(grid_coords, station_coords,
                                      station_coefs, k=k_neighbors)

            # Apply: delta = intercept + sum_p coef_p * X_p
            X_grid = np.column_stack([
                np.ones(len(day_idx)),
                grid_df.iloc[day_idx][active_predictors].to_numpy(),
            ])
            delta = np.einsum("ij,ij->i", X_grid, interp)
            out[day_idx] = lme_pred[day_idx] + delta

        return out


class HybridTreeGWR:
    """Two-stage hybrid: sklearn tree model (RF/GB/XGB) → GWR on residuals.

    Mirrors HybridLMEGWR but replaces Stage-1 LME with any sklearn-compatible
    regressor.  The tree model provides a global PM2.5 estimate; GWR then
    absorbs the spatially autocorrelated residuals day by day.

        PM2.5_hat = tree_pred  +  GWR_correction(residual)

    Parameters
    ----------
    tree_model  : fitted or unfitted sklearn-style regressor with .fit(X, y)
                  and .predict(X).
    gwr         : GWRStage2 instance (shared configuration with HybridLMEGWR).
    target      : name of the target column in the panel DataFrame.
    predictors  : list of predictor column names.
    station_id  : column identifying monitoring stations.
    date_col    : column identifying observation dates.
    """

    def __init__(self, tree_model, gwr: GWRStage2,
                 target: str, predictors: list[str],
                 station_id: str = "station_id",
                 date_col: str = "date"):
        self.tree_model = tree_model
        self.gwr = gwr
        self.target = target
        self.predictors = list(predictors)
        self.station_id = station_id
        self.date_col = date_col
        self.gwr_output_: GWRStage2Output | None = None

    # -----------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "HybridTreeGWR":
        X = df[self.predictors].to_numpy()
        y = df[self.target].to_numpy()

        # Stage 1: fit tree on full training data.
        self.tree_model.fit(X, y)

        # Compute UNBIASED (out-of-bag / out-of-fold) residuals for the GWR stage.
        # Using in-sample predictions would give near-zero residuals for deep trees
        # (RF perfectly memorises training data), making the GWR fit ill-conditioned.
        oof_pred = _oof_predictions(self.tree_model, X, y)

        df = df.copy()
        df["tree_pred"] = oof_pred
        df["tree_residual"] = df[self.target] - df["tree_pred"]

        # GWR is trained only on rows with valid OOF predictions (NaN can appear
        # for RF when a sample is never left out, which is very rare).
        valid = ~np.isnan(df["tree_residual"])
        self.gwr_output_ = self.gwr.fit_all(
            df[valid].reset_index(drop=True), residual_col="tree_residual"
        )
        return self

    # -----------------------------------------------------------------
    def predict(self, df: pd.DataFrame, k_neighbors: int = 8) -> np.ndarray:
        if self.gwr_output_ is None:
            raise RuntimeError("HybridTreeGWR has not been fitted yet.")

        X = df[self.predictors].to_numpy()
        tree_pred = self.tree_model.predict(X)
        out = tree_pred.copy()

        for date, day_idx in df.groupby(self.date_col).indices.items():
            day_idx = np.asarray(day_idx)
            res = self.gwr_output_.daily_results.get(pd.Timestamp(date))
            if res is None or not res.fitted or res.coefficients is None:
                continue

            coef_df = res.coefficients
            coef_cols = [c for c in coef_df.columns
                         if c not in ("lon", "lat", "station_id")]
            active_predictors = [c for c in coef_cols if c != "intercept"]
            station_coords = coef_df[["lon", "lat"]].to_numpy()
            station_coefs = coef_df[coef_cols].to_numpy()
            sid_to_correction = dict(zip(coef_df["station_id"].values,
                                         res.residual_correction))

            exact, oos = [], []
            for i in day_idx:
                sid = df.iloc[i][self.station_id]
                if sid in sid_to_correction:
                    exact.append(i)
                else:
                    oos.append(i)

            for i in exact:
                out[i] = tree_pred[i] + float(sid_to_correction[df.iloc[i][self.station_id]])

            if oos:
                oos = np.asarray(oos)
                oos_coords = df.iloc[oos][[self.gwr.lon_col,
                                           self.gwr.lat_col]].to_numpy()
                interp = _idw_interpolate(oos_coords, station_coords,
                                          station_coefs, k=k_neighbors)
                X_oos = np.column_stack([
                    np.ones(len(oos)),
                    df.iloc[oos][active_predictors].to_numpy(),
                ])
                out[oos] = tree_pred[oos] + np.einsum("ij,ij->i", X_oos, interp)

        return out

    # -----------------------------------------------------------------
    def predict_grid(self, grid_df: pd.DataFrame,
                     k_neighbors: int = 8) -> np.ndarray:
        """Predict PM2.5 on grid cells.

        Tree model provides the global baseline; per-day GWR coefficients are
        spatially interpolated to grid cells via IDW and applied as a delta.
        """
        if self.gwr_output_ is None:
            raise RuntimeError("HybridTreeGWR has not been fitted yet.")

        X_grid = grid_df[self.predictors].to_numpy()
        tree_pred = self.tree_model.predict(X_grid)
        out = tree_pred.copy()

        for date, day_idx in grid_df.groupby(self.date_col).indices.items():
            day_idx = np.asarray(day_idx)
            res = self.gwr_output_.daily_results.get(pd.Timestamp(date))
            if res is None or not res.fitted or res.coefficients is None:
                continue

            coef_cols = [c for c in res.coefficients.columns
                         if c not in ("lon", "lat", "station_id")]
            active_predictors = [c for c in coef_cols if c != "intercept"]

            station_coords = res.coefficients[["lon", "lat"]].to_numpy()
            station_coefs = res.coefficients[coef_cols].to_numpy()

            grid_coords = grid_df.iloc[day_idx][[self.gwr.lon_col,
                                                  self.gwr.lat_col]].to_numpy()

            interp = _idw_interpolate(grid_coords, station_coords,
                                      station_coefs, k=k_neighbors)

            X_day = np.column_stack([
                np.ones(len(day_idx)),
                grid_df.iloc[day_idx][active_predictors].to_numpy(),
            ])
            delta = np.einsum("ij,ij->i", X_day, interp)
            out[day_idx] = tree_pred[day_idx] + delta

        return out


# ---------------------------------------------------------------------
def _idw_interpolate(target_xy: np.ndarray, source_xy: np.ndarray,
                     source_values: np.ndarray, k: int = 8,
                     power: float = 2.0, eps: float = 1e-9) -> np.ndarray:
    """Inverse-distance-weighted interpolation for k nearest sources.

    Parameters
    ----------
    target_xy     : (N, 2) target coordinates (lon, lat)
    source_xy     : (M, 2) source coordinates
    source_values : (M, P) values to interpolate at each source
    k             : number of nearest sources used per target
    power         : IDW power (typically 2)

    Returns
    -------
    (N, P) interpolated values.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(source_xy)
    k_use = min(k, len(source_xy))
    dists, idxs = tree.query(target_xy, k=k_use)
    # cKDTree returns a 1-D array for k=1
    if k_use == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]
    weights = 1.0 / (dists ** power + eps)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return np.einsum("nk,nkp->np", weights, source_values[idxs])


# ---------------------------------------------------------------------
def _oof_predictions(tree_model, X: np.ndarray,
                     y: np.ndarray) -> np.ndarray:
    """Return unbiased out-of-fold predictions for GWR residual training.

    Tree ensembles memorise training data, so in-sample residuals are near zero.
    GWR fitted on near-zero residuals with many predictors becomes ill-conditioned
    and produces huge extrapolated corrections.  Using OOB/OOF predictions avoids
    this by giving residuals on the scale of true generalisation error.

    Strategy
    --------
    * Random Forest  : free OOB predictions (oob_score=True in RFModel).
    * Gradient Boost / XGBoost : 3-fold cross-validated predictions.
    """
    inner = getattr(tree_model, "model", None)

    # --- RF: use the free OOB predictions ---
    if inner is not None and hasattr(inner, "oob_prediction_"):
        oof = inner.oob_prediction_.copy()
        # Very rare NaN (sample never left out) → fall back to in-sample
        nan_mask = np.isnan(oof)
        if nan_mask.any():
            oof[nan_mask] = tree_model.predict(X[nan_mask])
        return oof

    # --- GB / XGB: 3-fold cross-validated predictions ---
    import copy
    from sklearn.model_selection import KFold

    logger.info("Computing 3-fold OOF predictions for HybridTreeGWR residuals ...")
    oof = np.full(len(y), np.nan)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        try:
            # Clone: rebuild from params to avoid GPU state issues in deepcopy
            ModelClass = type(tree_model)
            saved_params = getattr(tree_model, "params", {})
            if hasattr(tree_model, "_uses_gpu"):
                # XGBModel: force CPU for OOF folds (fast enough, avoids VRAM)
                clone = ModelClass(n_jobs=-1, force_cpu=True, **saved_params)
            else:
                clone = copy.deepcopy(tree_model)
            clone.fit(X[tr_idx], y[tr_idx])
            oof[val_idx] = clone.predict(X[val_idx])
            logger.info("  OOF fold %d/3 done", fold_i)
        except Exception as e:
            logger.warning("OOF fold %d failed (%s) — filling with in-sample.", fold_i, e)
            oof[val_idx] = tree_model.predict(X[val_idx])

    return oof
