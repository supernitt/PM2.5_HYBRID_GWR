"""Stage-2 geographically weighted regression on LME residuals.

For each day, the LME residuals at the monitoring stations are regressed
against AOD and selected predictors using GWR. The bandwidth is selected
by minimizing AICc (or the user-specified criterion). When the residuals
on a given day show no significant spatial autocorrelation (Moran's I,
p > alpha), Stage 2 is skipped and the LME prediction is used directly.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
try:
    from sklearn.utils.parallel import Parallel, delayed
except ImportError:          # sklearn < 1.1
    from joblib import Parallel, delayed

logger = logging.getLogger("pm25_hybrid")


# ---------------------------------------------------------------------
# Optional imports — only required when the user actually runs GWR.
# ---------------------------------------------------------------------
def _import_mgwr():
    try:
        from mgwr.gwr import GWR
        from mgwr.sel_bw import Sel_BW
        return GWR, Sel_BW
    except ImportError as e:
        raise ImportError(
            "mgwr is not installed. Run `pip install mgwr libpysal esda` to enable GWR."
        ) from e


def _import_morans():
    try:
        from libpysal.weights import KNN
        from esda.moran import Moran
        return KNN, Moran
    except ImportError as e:
        raise ImportError(
            "libpysal and esda are required for Moran's I. Run `pip install libpysal esda`."
        ) from e


@dataclass
class GWRDailyResult:
    date: pd.Timestamp
    fitted: bool                         # True if Stage-2 was actually fit
    morans_p: float | None
    bandwidth: float | None
    coefficients: pd.DataFrame | None    # one row per station, cols=predictors
    residual_correction: np.ndarray | None  # delta added to LME pred at each station


@dataclass
class GWRStage2Output:
    daily_results: dict[pd.Timestamp, GWRDailyResult] = field(default_factory=dict)
    fitted_days: int = 0
    skipped_days: int = 0


class GWRStage2:
    """Apply daily GWR to LME residuals."""

    def __init__(self, predictors: list[str],
                 lat_col: str = "lat", lon_col: str = "lon",
                 kernel: str = "gaussian", adaptive: bool = True,
                 bw_search: str = "AICc",
                 morans_alpha: float = 0.05,
                 min_stations_per_day: int = 20,
                 n_jobs: int = -1,
                 bw_cache: dict | None = None):
        self.predictors = list(predictors)
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.kernel = kernel
        self.adaptive = adaptive
        self.bw_search = bw_search
        self.morans_alpha = morans_alpha
        self.min_stations_per_day = min_stations_per_day
        self.n_jobs = n_jobs
        # Shared mutable dict {date -> bandwidth}. Populated on first fit_all and
        # reused in subsequent calls (e.g. CV folds) to skip Sel_BW.search.
        # Excluded from pickling so loky workers don't carry a huge copy.
        self.bw_cache = bw_cache

    # Exclude bw_cache from the pickled state sent to loky worker processes.
    # Workers receive fixed_bw as a plain float argument instead.
    def __getstate__(self):
        state = self.__dict__.copy()
        state["bw_cache"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # -----------------------------------------------------------------
    def _morans_p(self, residuals: np.ndarray, coords: np.ndarray) -> float:
        KNN, Moran = _import_morans()
        k = min(5, len(coords) - 1)
        if k < 2:
            return 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # suppress disconnected-graph warning
            w = KNN.from_array(coords, k=k)
            w.transform = "r"
            # permutations=0 → use the analytical normal-approximation p-value
            # (p_norm) instead of a Monte-Carlo simulation.  With ≥20 stations
            # per day the normal approximation is valid and ~200× faster.
            mi = Moran(residuals, w, permutations=0)
        return float(mi.p_norm)

    # -----------------------------------------------------------------
    def fit_day(self, day_df: pd.DataFrame, residuals: np.ndarray,
               fixed_bw: float | None = None) -> GWRDailyResult:
        """Fit one day's GWR.

        Parameters
        ----------
        fixed_bw : pre-computed bandwidth to reuse (skips Sel_BW.search).
            Passed explicitly from fit_all so workers don't need the bw_cache dict.
        """
        date = day_df["date"].iloc[0] if "date" in day_df.columns else pd.NaT
        n = len(day_df)
        if n < self.min_stations_per_day:
            return GWRDailyResult(date=date, fitted=False,
                                  morans_p=None, bandwidth=None,
                                  coefficients=None, residual_correction=None)

        coords = day_df[[self.lon_col, self.lat_col]].to_numpy()
        morans_p = self._morans_p(residuals, coords)

        if morans_p > self.morans_alpha:
            return GWRDailyResult(date=date, fitted=False,
                                  morans_p=morans_p, bandwidth=None,
                                  coefficients=None, residual_correction=None)

        # Spatial structure exists → fit GWR
        GWR, Sel_BW = _import_mgwr()

        # check_constant (called internally by mgwr/spglm) silently removes any
        # column with zero variance before adding the intercept.  On days where
        # all stations share the same value for a predictor (e.g. FRP == 0 when
        # there are no fires), that column is dropped, making params.shape[1]
        # one less than expected and causing a shape mismatch in the DataFrame
        # constructor below.  Strip zero-variance columns here first so the
        # label list stays in sync with the actual params array.
        X_raw = day_df[self.predictors].to_numpy(dtype=float)
        zero_var = np.ptp(X_raw, axis=0) == 0          # peak-to-peak == 0
        active_predictors = [p for p, z in zip(self.predictors, zero_var) if not z]
        if zero_var.any():
            dropped = [p for p, z in zip(self.predictors, zero_var) if z]
            logger.debug("Dropped zero-variance predictor(s) on %s: %s", date, dropped)
        X = X_raw[:, ~zero_var]
        y = residuals.reshape(-1, 1)

        # Minimum adaptive bandwidth for a numerically stable local regression.
        # A well-conditioned system needs observations >> parameters.  With
        # n_params = n_active + 1 (intercept), require at least 3 × n_params
        # neighbours so the local WLS is comfortably overdetermined.
        n_params = X.shape[1] + 1
        if self.adaptive:
            min_stable_bw = max(self.min_stations_per_day, 3 * n_params)
            if n < min_stable_bw:
                # Too few stations to run GWR stably on this day.
                logger.debug(
                    "Skipping GWR on %s: n=%d < min_stable_bw=%d",
                    date, n, min_stable_bw,
                )
                return GWRDailyResult(date=date, fitted=False,
                                      morans_p=morans_p, bandwidth=None,
                                      coefficients=None, residual_correction=None)

        try:
            if fixed_bw is not None:
                bw = fixed_bw
            else:
                bw_selector = Sel_BW(coords, y, X, kernel=self.kernel,
                                     fixed=not self.adaptive)
                bw = bw_selector.search(criterion=self.bw_search)

            # Clamp the adaptive bandwidth into [lower, n-1].
            #   lower = stable minimum (>= 3 * n_params) so the local system is
            #           comfortably overdetermined and not near-singular.
            #   upper = n-1 because an adaptive bandwidth is a *neighbour count*
            #           and mgwr indexes the bw-th nearest neighbour; bw >= n
            #           raises "kth(=bw) out of bounds" — which happens when a
            #           cached bandwidth from a denser day is reused in a sparser
            #           CV fold.
            if self.adaptive:
                lower = min(n - 1, min_stable_bw)
                bw = int(min(max(bw, lower), n - 1))

            gwr_model = GWR(coords, y, X, bw=bw,
                            kernel=self.kernel, fixed=not self.adaptive)
            gwr_res = gwr_model.fit()
        except Exception as e:
            logger.warning("GWR fit failed on %s (%s) — falling back to LME.", date, e)
            return GWRDailyResult(date=date, fitted=False,
                                  morans_p=morans_p, bandwidth=None,
                                  coefficients=None, residual_correction=None)

        # Coefficient sanity check: if local regression produced extreme
        # coefficients (sign of near-singular system), discard this day's GWR.
        # With standardised predictors, |coef| > 100 is physically impossible
        # (would require a 100 μg/m³ residual change per 1-SD predictor shift).
        max_coef = float(np.max(np.abs(gwr_res.params)))
        if max_coef > 100.0:
            logger.debug(
                "GWR coefficients unstable on %s (max |coef|=%.1f) — "
                "falling back to LME.", date, max_coef,
            )
            return GWRDailyResult(date=date, fitted=False,
                                  morans_p=morans_p, bandwidth=None,
                                  coefficients=None, residual_correction=None)

        coef_df = pd.DataFrame(
            gwr_res.params,
            columns=["intercept"] + active_predictors,
        )
        coef_df.insert(0, "lon", coords[:, 0])
        coef_df.insert(1, "lat", coords[:, 1])
        coef_df["station_id"] = day_df["station_id"].values

        residual_correction = gwr_res.predy.flatten()

        return GWRDailyResult(date=date, fitted=True,
                              morans_p=morans_p, bandwidth=float(bw),
                              coefficients=coef_df,
                              residual_correction=residual_correction)

    # -----------------------------------------------------------------
    def fit_all(self, df_with_residuals: pd.DataFrame,
                residual_col: str = "lme_residual") -> GWRStage2Output:
        grouped = [
            (date, day_df.reset_index(drop=True))
            for date, day_df in df_with_residuals.groupby("date")
        ]
        # Pre-extract cached bandwidths as plain floats so loky workers don't
        # need to pickle/unpickle the entire bw_cache dict (which can be large
        # on subsequent CV calls).  Workers receive None on first (training) run.
        fixed_bws = [
            self.bw_cache.get(date) if self.bw_cache is not None else None
            for date, _ in grouped
        ]
        n_days = len(grouped)
        n_cached = sum(bw is not None for bw in fixed_bws)
        logger.info("GWR Stage-2 starting: %d days total, %d with cached bandwidth "
                    "(%d will run Sel_BW.search)", n_days, n_cached, n_days - n_cached)

        # Process in chunks so we can log progress rather than waiting silently
        # for all days to finish.
        chunk_size = max(10, min(50, n_days // 10 + 1))
        results: list = []
        for chunk_start in range(0, n_days, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_days)
            chunk_grouped = grouped[chunk_start:chunk_end]
            chunk_bws = fixed_bws[chunk_start:chunk_end]

            chunk_results = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=0)(
                delayed(self.fit_day)(day_df, day_df[residual_col].to_numpy(), fixed_bw)
                for (_, day_df), fixed_bw in zip(chunk_grouped, chunk_bws)
            )
            results.extend(chunk_results)
            logger.info("GWR Stage-2 progress: %d / %d days processed",
                        chunk_end, n_days)

            # Write bandwidths back to cache after each chunk
            if self.bw_cache is not None:
                for (date, _), res in zip(chunk_grouped, chunk_results):
                    if res.fitted and res.bandwidth is not None and date not in self.bw_cache:
                        self.bw_cache[date] = res.bandwidth

        out = GWRStage2Output()
        for (date, _), res in zip(grouped, results):
            out.daily_results[date] = res
            if res.fitted:
                out.fitted_days += 1
            else:
                out.skipped_days += 1
        logger.info("GWR Stage-2 finished: %d days fitted, %d days skipped (of %d total)",
                    out.fitted_days, out.skipped_days, n_days)
        # Canary: if Stage-2 fit *zero* days the hybrid silently collapses to
        # the Stage-1 model (LME/tree).  This must never pass unnoticed again —
        # surface it loudly so a long run is not wasted on a no-op correction.
        if out.fitted_days == 0 and n_days > 0:
            logger.warning(
                "GWR Stage-2 fitted 0 / %d days — the hybrid reduces to its "
                "Stage-1 model (no spatial correction applied). Check gwr.predictors "
                "(multicollinearity?), morans_alpha, and min_stations_per_day.",
                n_days,
            )
        return out
