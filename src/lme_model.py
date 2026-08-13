"""Stage-1 linear mixed-effects model.

Fits Equation (1) of the manuscript:

    PM2.5_{it} = (alpha + a_i) + (beta + b_i)*AOD_{it}
                 + sum_k gamma_k * X_{kit} + eps_{it}

where a_i and b_i are station-specific random intercept and random slope
on AOD, both assumed to follow a bivariate normal with mean zero.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLMResults

logger = logging.getLogger("pm25_hybrid")


class LMEModel:
    """Random-intercept + (optional) random-slope-on-AOD mixed-effects model."""

    def __init__(self, target: str, aod: str, predictors: list[str],
                 station_id: str = "station_id",
                 random_intercept: bool = True,
                 random_slope_on_aod: bool = True,
                 reml: bool = True):
        self.target = target
        self.aod = aod
        self.predictors = list(predictors)            # full predictor list (incl. AOD)
        self.other_predictors = [p for p in predictors if p != aod]
        self.station_id = station_id
        self.random_intercept = random_intercept
        self.random_slope_on_aod = random_slope_on_aod
        self.reml = reml
        self.result_: MixedLMResults | None = None

    # -----------------------------------------------------------------
    # Fitting
    # -----------------------------------------------------------------
    def _formula(self) -> str:
        rhs_terms = [self.aod] + self.other_predictors
        return f"{self.target} ~ {' + '.join(rhs_terms)}"

    def fit(self, df: pd.DataFrame) -> "LMEModel":
        formula = self._formula()
        re_formula = "~AOD" if False else None  # placeholder, replaced below
        re_formula = f"~{self.aod}" if self.random_slope_on_aod else "~1"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = smf.mixedlm(formula, df, groups=df[self.station_id],
                             re_formula=re_formula)
            self.result_ = md.fit(reml=self.reml, method="lbfgs")
        logger.info("LME fitted: log-likelihood=%.3f", self.result_.llf)
        return self

    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------
    def predict_fixed(self, df: pd.DataFrame) -> np.ndarray:
        """Population-level (fixed-effects only) prediction."""
        if self.result_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.result_.predict(df).to_numpy()

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Conditional prediction including station random effects when known.

        For stations seen during training, the station random intercept (and
        slope) is added back. For unseen stations, falls back to the fixed-
        effects prediction.
        """
        if self.result_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        fixed_pred = self.predict_fixed(df)
        re_dict = self.result_.random_effects   # {station: pandas Series}
        out = fixed_pred.copy()

        sid_col = df[self.station_id].values
        aod_col = df[self.aod].values
        for i, sid in enumerate(sid_col):
            re = re_dict.get(sid)
            if re is None:
                continue
            # Random intercept term
            out[i] += float(re.get("Group", 0.0))
            # Random slope on AOD
            if self.random_slope_on_aod and self.aod in re.index:
                out[i] += float(re.get(self.aod, 0.0)) * float(aod_col[i])
        return out

    def residuals(self, df: pd.DataFrame, mode: str = "marginal") -> np.ndarray:
        """Residuals.

        mode='marginal' uses fixed-effects prediction (used for the GWR stage,
                       which is the manuscript's intended pipeline).
        mode='conditional' uses fixed + random effects.
        """
        y = df[self.target].to_numpy()
        if mode == "marginal":
            return y - self.predict_fixed(df)
        return y - self.predict(df)

    def summary(self) -> str:
        if self.result_ is None:
            return "<not fitted>"
        return str(self.result_.summary())
