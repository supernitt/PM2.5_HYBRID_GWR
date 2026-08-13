"""Main entry point for the PM2.5 hybrid LME-GWR pipeline.

Run with::

    python main.py --config configs/default.yaml

Stages are controlled by --stages (default = all):

    preprocess   load + clean + standardize the panel
    train        fit hybrid LME-GWR and benchmarks on full data
    validate     run random 10-fold and site-based CV for every model
    map          predict daily PM2.5 on the grid and aggregate to monthly/seasonal

Outputs (CSVs, model summaries, prediction grids) go to <output_dir>/.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import (
    benchmarks,
    data_loader,
    gwr_model,
    hybrid_model,
    lme_model,
    mapping,
    preprocessing,
    utils,
    validation,
)
from src.hybrid_model import HybridTreeGWR


# =====================================================================
# CV output checkpointing (crash safety + subset-run safety)
# =====================================================================
def _upsert_cv_csv(path: Path, new_df: pd.DataFrame, key_cols: list[str]) -> None:
    if new_df.empty:
        return
    path = Path(path)
    if path.exists():
        old_df = pd.read_csv(path)
        if not old_df.empty and all(c in old_df.columns for c in key_cols):
            new_keys = set(map(tuple, new_df[key_cols].astype(str).values))
            old_keys = old_df[key_cols].astype(str).apply(tuple, axis=1)
            old_df = old_df[~old_keys.isin(new_keys)]
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(path, index=False)


def _checkpoint_cv(output_dir: Path, overall: pd.DataFrame, strat: pd.DataFrame,
                   preds: pd.DataFrame) -> None:
    """Persist one model x scheme's CV results to disk immediately."""
    _upsert_cv_csv(output_dir / "cv_overall.csv", overall, ["model", "scheme"])
    if not strat.empty:
        _upsert_cv_csv(output_dir / "cv_stratified.csv", strat, ["model", "scheme"])
    _upsert_cv_csv(output_dir / "cv_predictions.csv", preds, ["model", "scheme"])


# =====================================================================
# CLI
# =====================================================================
_ALL_ALGORITHMS = "hybrid,lme,pcr,plsr,rf,gb,xgboost,hybrid_rf,hybrid_gb,hybrid_xgb"


def parse_args():
    p = argparse.ArgumentParser(description="Hybrid LME-GWR PM2.5 estimation pipeline")
    p.add_argument("--config", type=str, default="configs/default.yaml",
                   help="Path to YAML config file.")
    p.add_argument("--stages", type=str, default="preprocess,train,validate,map",
                   help=("Comma-separated list of stages to run. "
                         "Available: preprocess, train, validate, map, map_raster."))
    p.add_argument("--algorithms", type=str, default=_ALL_ALGORITHMS,
                   help=("Comma-separated list of algorithms to train/validate. "
                         f"Available: {_ALL_ALGORITHMS}. "
                         "Applies to the train and validate stages only."))
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


# =====================================================================
def main():
    args = parse_args()
    cfg = utils.load_config(args.config)
    output_dir = utils.ensure_dir(cfg["paths"]["output_dir"])
    log_path = output_dir / "run.log"

    logger = utils.setup_logger(log_path, level=getattr(logging, args.log_level))
    utils.set_global_seed(cfg["validation"]["random_seed"])

    _pipeline_start = time.time()

    def _elapsed() -> str:
        """Return wall-clock time since pipeline start as a human-readable string."""
        s = int(time.time() - _pipeline_start)
        h, m = divmod(s, 3600)
        m, s = divmod(m, 60)
        return f"{h:02d}h {m:02d}m {s:02d}s"

    def _banner(msg: str) -> None:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  %s", msg)
        logger.info("  Elapsed: %s", _elapsed())
        logger.info("=" * 60)

    stages = set(s.strip() for s in args.stages.split(","))
    algorithms = set(s.strip().lower() for s in args.algorithms.split(","))
    n_jobs = cfg.get("compute", {}).get("n_jobs", -1)
    logger.info("Pipeline stages: %s", sorted(stages))
    logger.info("Algorithms selected: %s", sorted(algorithms))
    logger.info("n_jobs: %s", n_jobs)

    # Shared bandwidth cache: {date -> float}.  Populated during full-data GWR
    # training and reused in all CV folds to skip per-day Sel_BW.search.
    _gwr_bw_cache: dict = {}

    # =================================================================
    # Stage: preprocess
    # =================================================================
    cols = cfg["columns"]
    target = cols["target"]
    aod = cols["aod"]
    predictors = preprocessing.assemble_predictors(cols)

    df = data_loader.load_panel(cfg["paths"]["panel_file"], cols)

    # ── Study-period restriction ─────────────────────────────────────
    # Everything downstream (training, CV, exceedance statistics) must stay
    # inside the manuscript's stated period; run0 leaked 139 days of 2026
    # predictions into the headline exceedance numbers.
    period = cfg.get("study_period") or {}
    if period.get("start") or period.get("end"):
        t0 = pd.Timestamp(period.get("start", df[cols["date"]].min()))
        t1 = pd.Timestamp(period.get("end", df[cols["date"]].max()))
        n_before = len(df)
        df = df[(df[cols["date"]] >= t0) & (df[cols["date"]] <= t1)]
        df = df.reset_index(drop=True)
        logger.info("Study period %s..%s: %d -> %d rows (%d outside removed)",
                    t0.date(), t1.date(), n_before, len(df), n_before - len(df))

    # ── Physical range guard (GEE sentinels, unit errors) ────────────
    # Must run FIRST: run0's panel carried -9999-derived wind speeds of
    # 14,140 m/s into training and manufactured VIFs of 10^5-10^6.
    valid_ranges = cfg["preprocessing"].get("valid_ranges") or {}
    if valid_ranges:
        df = preprocessing.validate_ranges(df, valid_ranges)

    # ── AOD policy ───────────────────────────────────────────────────
    # "observed": MAIAC retrievals only (main analysis) — missing AOD rows
    #             are dropped below by filter_missing.
    # "imputed":  disclosed sensitivity arm — interior gaps <= max_gap_days
    #             linearly interpolated per station, flagged in aod_imputed.
    aod_policy = cfg["preprocessing"].get("aod_policy", "observed").lower()
    if aod_policy == "imputed":
        df = preprocessing.impute_aod(
            df, aod=aod, station_id=cols["station_id"], date_col=cols["date"],
            max_gap_days=int(cfg["preprocessing"].get("aod_impute_max_gap_days", 7)),
        )
    elif aod_policy != "observed":
        raise ValueError(f"Unknown preprocessing.aod_policy: {aod_policy!r} "
                         "(expected 'observed' or 'imputed')")
    logger.info("AOD policy: %s", aod_policy)

    # ── Optional legacy smoothing (sensitivity only — OFF by default) ─
    # run0 smoothed+interpolated every predictor to completeness, which
    # fabricated 66% of AOD and defeated the missing-AOD filter entirely.
    if cfg["preprocessing"].get("smooth_predictors", False):
        logger.warning("Savitzky-Golay smoothing ENABLED — sensitivity runs "
                       "only; never use for headline results.")
        df = preprocessing.smooth_predictors(
            df,
            predictors=predictors,
            station_id=cols["station_id"],
            date_col=cols["date"],
            window_length=cfg["preprocessing"]["savgol_window"],
            polyorder=cfg["preprocessing"]["savgol_polyorder"],
            static_predictors=cfg["preprocessing"]["static_predictors"],
        )

    df = preprocessing.filter_missing(
        df,
        target=target,
        aod=aod,
        drop_target=cfg["preprocessing"]["drop_missing_target"],
        drop_aod=cfg["preprocessing"]["drop_missing_aod"],
    )


    predictors = preprocessing.drop_all_nan_predictors(df, predictors)
    df = preprocessing.drop_missing_predictors(df, predictors)
    vif = preprocessing.compute_vif(df, predictors)
    vif.to_csv(output_dir / "vif_pooled.csv", index=False)
    vif_ws = preprocessing.compute_vif_within_station(df, predictors, cols["station_id"])
    vif_ws.to_csv(output_dir / "vif_within_station.csv", index=False)
    logger.info("Within-station VIF:\n%s", vif_ws.to_string(index=False))

    too_high = vif_ws[vif_ws["VIF"] > cfg["preprocessing"]["vif_threshold"]]
    if not too_high.empty:
        logger.warning("Predictors with within-station VIF > %d:\n%s",
                       cfg["preprocessing"]["vif_threshold"],
                       too_high.to_string(index=False))

    df_raw = df.copy()

    scaler = None
    if cfg["preprocessing"]["standardize_predictors"]:
        df, scaler = preprocessing.standardize(df, predictors)
        # Persist the scaler so raster-mode prediction can reuse it.
        with open(output_dir / "predictor_scaler.pkl", "wb") as f:
            pickle.dump({"scaler": scaler, "predictors": predictors}, f)

    if "preprocess" in stages and "train" not in stages and "validate" not in stages:
        df.to_parquet(output_dir / "preprocessed_panel.parquet", index=False)
        logger.info("Preprocessed panel saved. Exiting.")
        return

    # =================================================================
    # Helpers — model factories
    # =================================================================
    def build_lme() -> lme_model.LMEModel:
        return lme_model.LMEModel(
            target=target, aod=aod, predictors=predictors,
            station_id=cols["station_id"],
            random_intercept=cfg["lme"]["random_intercept"],
            random_slope_on_aod=cfg["lme"]["random_slope_on_aod"],
            reml=cfg["lme"]["reml"],
        )

    _gwr_pred_cfg = cfg["gwr"].get("predictors") or predictors
    gwr_predictors = [p for p in _gwr_pred_cfg if p in predictors]
    if not gwr_predictors:
        raise ValueError(
            f"gwr.predictors={_gwr_pred_cfg} has no overlap with the available "
            f"predictors {predictors}."
        )
    logger.info("GWR Stage-2 predictors: %s", gwr_predictors)

    def build_gwr(use_bw_cache: bool = True) -> gwr_model.GWRStage2:
        return gwr_model.GWRStage2(
            predictors=gwr_predictors,
            lat_col=cols["lat"], lon_col=cols["lon"],
            kernel=cfg["gwr"]["kernel"],
            adaptive=cfg["gwr"]["adaptive"],
            bw_search=cfg["gwr"]["bw_search"],
            morans_alpha=cfg["gwr"]["morans_alpha"],
            min_stations_per_day=cfg["gwr"]["min_stations_per_day"],
            n_jobs=n_jobs,
            bw_cache=_gwr_bw_cache if use_bw_cache else None,
            bw_min=cfg["gwr"].get("bw_min"),
            bw_max=cfg["gwr"].get("bw_max"),
        )

    def build_hybrid(use_bw_cache: bool = True) -> hybrid_model.HybridLMEGWR:
        return hybrid_model.HybridLMEGWR(
            lme=build_lme(), gwr=build_gwr(use_bw_cache),
            station_id=cols["station_id"], date_col=cols["date"],
        )

    def build_hybrid_rf(rf_params: dict, use_bw_cache: bool = True) -> HybridTreeGWR:
        return HybridTreeGWR(
            tree_model=benchmarks.RFModel(**rf_params),
            gwr=build_gwr(use_bw_cache),
            target=target, predictors=predictors,
            station_id=cols["station_id"], date_col=cols["date"],
        )

    def build_hybrid_gb(gb_params: dict, use_bw_cache: bool = True) -> HybridTreeGWR:
        return HybridTreeGWR(
            tree_model=benchmarks.GBModel(**gb_params),
            gwr=build_gwr(use_bw_cache),
            target=target, predictors=predictors,
            station_id=cols["station_id"], date_col=cols["date"],
        )

    def build_hybrid_xgb(xgb_params: dict, use_bw_cache: bool = True) -> HybridTreeGWR:
        return HybridTreeGWR(
            tree_model=benchmarks.XGBModel(**xgb_params),
            gwr=build_gwr(use_bw_cache),
            target=target, predictors=predictors,
            station_id=cols["station_id"], date_col=cols["date"],
        )

    # =================================================================
    # Stage: train (fit on full data, save model artifacts)
    # =================================================================
    fitted_hybrid: hybrid_model.HybridLMEGWR | None = None
    fitted_lme_only: lme_model.LMEModel | None = None
    fitted_hybrid_rf: HybridTreeGWR | None = None
    fitted_hybrid_gb: HybridTreeGWR | None = None
    fitted_hybrid_xgb: HybridTreeGWR | None = None
    fitted_benchmarks: dict[str, object] = {}

    _tuned_rf_params: dict | None = None
    _tuned_gb_params: dict | None = None
    _tuned_xgb_params: dict | None = None
    _tuned_bench: dict[str, object] = {}   # name -> params dict (trees) or int n_components

    if "train" in stages:
        _banner("STAGE: train — fitting models on full data")
        if "hybrid" in algorithms:
            _banner("Training hybrid LME-GWR on full data")
            _t0 = time.time()
            fitted_hybrid = build_hybrid().fit(df)
            logger.info("Hybrid LME-GWR training done (%.1f min)", (time.time() - _t0) / 60)
            with open(output_dir / "hybrid_model.pkl", "wb") as f:
                pickle.dump(fitted_hybrid, f)
            with open(output_dir / "lme_summary.txt", "w") as f:
                f.write(fitted_hybrid.lme.summary())
        elif "lme" in algorithms:
            logger.info("=== Training LME on full data ===")
            fitted_lme_only = build_lme().fit(df)
            with open(output_dir / "lme_summary.txt", "w") as f:
                f.write(fitted_lme_only.summary())

        # Benchmarks (full-data tuning + fit)
        bench_algo = {"pcr", "plsr", "rf", "gb", "xgboost"} & algorithms
        if bench_algo:
            _banner(f"Tuning + fitting benchmark models: {sorted(bench_algo)}")
            X_full = df[predictors].to_numpy()
            y_full = df[target].to_numpy()

            if "pcr" in algorithms:
                n_pcr = benchmarks.tune_pcr(
                    X_full, y_full,
                    cfg["benchmarks"]["pcr"]["n_components_search"],
                    n_splits=cfg["validation"]["random_kfold"],
                    seed=cfg["validation"]["random_seed"],
                )
                fitted_benchmarks["PCR"] = benchmarks.PCRModel(n_components=n_pcr).fit(X_full, y_full)
                _tuned_bench["PCR"] = n_pcr

            if "plsr" in algorithms:
                n_pls = benchmarks.tune_plsr(
                    X_full, y_full,
                    cfg["benchmarks"]["plsr"]["n_components_search"],
                    n_splits=cfg["validation"]["random_kfold"],
                    seed=cfg["validation"]["random_seed"],
                )
                fitted_benchmarks["PLSR"] = benchmarks.PLSRModel(n_components=n_pls).fit(X_full, y_full)
                _tuned_bench["PLSR"] = n_pls

            if "rf" in algorithms:
                rf_params = benchmarks.tune_rf(
                    X_full, y_full, cfg["benchmarks"]["rf"],
                    n_splits=cfg["validation"]["random_kfold"],
                    seed=cfg["validation"]["random_seed"],
                )
                fitted_benchmarks["RF"] = benchmarks.RFModel(**rf_params).fit(X_full, y_full)
                _tuned_bench["RF"] = rf_params

            if "gb" in algorithms:
                gb_params = benchmarks.tune_gb(
                    X_full, y_full, cfg["benchmarks"]["gb"],
                    n_splits=cfg["validation"]["random_kfold"],
                    seed=cfg["validation"]["random_seed"],
                )
                fitted_benchmarks["GB"] = benchmarks.GBModel(**gb_params).fit(X_full, y_full)
                _tuned_bench["GB"] = gb_params

            if "xgboost" in algorithms:
                try:
                    xgb_params = benchmarks.tune_xgb(
                        X_full, y_full, cfg["benchmarks"]["xgboost"],
                        n_splits=cfg["validation"]["random_kfold"],
                        seed=cfg["validation"]["random_seed"],
                    )
                    fitted_benchmarks["XGBoost"] = benchmarks.XGBModel(**xgb_params).fit(X_full, y_full)
                    _tuned_bench["XGBoost"] = xgb_params
                except ImportError:
                    logger.warning("xgboost not installed — skipping XGBoost benchmark.")

            with open(output_dir / "benchmarks.pkl", "wb") as f:
                pickle.dump(fitted_benchmarks, f)

        # Hybrid tree-GWR models (tune tree stage, then fit two-stage model)
        X_full = df[predictors].to_numpy()
        y_full = df[target].to_numpy()

        if "hybrid_rf" in algorithms:
            _banner("Training Hybrid RF-GWR on full data")
            _t0 = time.time()
            # Dedup: the standalone RF benchmark tunes on identical data, grid,
            # folds, and seed — reuse its params instead of a second ~2 h search.
            rf_params = _tuned_bench.get("RF")
            if rf_params is None:
                rf_params = benchmarks.tune_rf(
                    X_full, y_full, cfg["benchmarks"]["rf"],
                    n_splits=cfg["validation"]["random_kfold"],
                    seed=cfg["validation"]["random_seed"],
                )
                _tuned_bench["RF"] = rf_params
            logger.info("RF tuning done (%.1f min) — best params: %s",
                        (time.time() - _t0) / 60, rf_params)
            logger.info("Fitting Hybrid RF-GWR on full data ...")
            _t0 = time.time()
            _tuned_rf_params = rf_params  # reused in CV to skip inner tuning
            fitted_hybrid_rf = build_hybrid_rf(rf_params).fit(df)
            logger.info("Hybrid RF-GWR training done (%.1f min)", (time.time() - _t0) / 60)
            with open(output_dir / "hybrid_rf_model.pkl", "wb") as f:
                pickle.dump(fitted_hybrid_rf, f)

        if "hybrid_gb" in algorithms:
            _banner("Training Hybrid GB-GWR on full data")
            _t0 = time.time()
            gb_params = _tuned_bench.get("GB")
            if gb_params is None:
                gb_params = benchmarks.tune_gb(
                    X_full, y_full, cfg["benchmarks"]["gb"],
                    n_splits=cfg["validation"]["random_kfold"],
                    seed=cfg["validation"]["random_seed"],
                )
                _tuned_bench["GB"] = gb_params
            logger.info("GB tuning done (%.1f min) — best params: %s",
                        (time.time() - _t0) / 60, gb_params)
            logger.info("Fitting Hybrid GB-GWR on full data ...")
            _t0 = time.time()
            _tuned_gb_params = gb_params  # reused in CV to skip inner tuning
            fitted_hybrid_gb = build_hybrid_gb(gb_params).fit(df)
            logger.info("Hybrid GB-GWR training done (%.1f min)", (time.time() - _t0) / 60)
            with open(output_dir / "hybrid_gb_model.pkl", "wb") as f:
                pickle.dump(fitted_hybrid_gb, f)

        if "hybrid_xgb" in algorithms:
            try:
                _banner("Training Hybrid XGB-GWR on full data")
                _t0 = time.time()
                xgb_params = _tuned_bench.get("XGBoost")
                if xgb_params is None:
                    xgb_params = benchmarks.tune_xgb(
                        X_full, y_full, cfg["benchmarks"]["xgboost"],
                        n_splits=cfg["validation"]["random_kfold"],
                        seed=cfg["validation"]["random_seed"],
                    )
                    _tuned_bench["XGBoost"] = xgb_params
                logger.info("XGB tuning done (%.1f min) — best params: %s",
                            (time.time() - _t0) / 60, xgb_params)
                logger.info("Fitting Hybrid XGB-GWR on full data ...")
                _t0 = time.time()
                _tuned_xgb_params = xgb_params  # reused in CV to skip inner tuning
                fitted_hybrid_xgb = build_hybrid_xgb(xgb_params).fit(df)
                logger.info("Hybrid XGB-GWR training done (%.1f min)", (time.time() - _t0) / 60)
                with open(output_dir / "hybrid_xgb_model.pkl", "wb") as f:
                    pickle.dump(fitted_hybrid_xgb, f)
            except ImportError:
                logger.warning("xgboost not installed — skipping Hybrid XGB-GWR.")

        # Persist tuned hyperparameters: a crash between train and validate —
        # or running the stages as separate invocations — must not silently
        # fall back to 20 grid searches per tree model in CV.
        if _tuned_bench:
            import json
            with open(output_dir / "tuned_params.json", "w") as f:
                json.dump(_tuned_bench, f, indent=2)
            logger.info("Persisted tuned hyperparameters -> tuned_params.json: %s",
                        sorted(_tuned_bench.keys()))

    # =================================================================
    # Post-train: save analysis artifacts for the report
    # =================================================================
    # ── GWR daily statistics (bandwidth, Moran's p, % days fitted) ───
    for gwr_source, label in [
        (fitted_hybrid,     "hybrid"),
        (fitted_hybrid_rf,  "hybrid_rf"),
        (fitted_hybrid_gb,  "hybrid_gb"),
        (fitted_hybrid_xgb, "hybrid_xgb"),
    ]:
        if gwr_source is None or not hasattr(gwr_source, "gwr_output_"):
            continue
        if gwr_source.gwr_output_ is None:
            continue
        gwr_rows = []
        for date, res in gwr_source.gwr_output_.daily_results.items():
            gwr_rows.append({
                "date":            date,
                "gwr_fitted":      res.fitted,
                "morans_p":        res.morans_p,
                "morans_p_after":  getattr(res, "morans_p_after", None),
                "bandwidth":       res.bandwidth,
                "bw_at_floor":     getattr(res, "bw_at_search_floor", None),
                "n_stations": (len(res.coefficients)
                               if res.coefficients is not None else 0),
            })
        if gwr_rows:
            gwr_stats_df = pd.DataFrame(gwr_rows).sort_values("date")
            out_path = output_dir / f"gwr_daily_stats_{label}.csv"
            gwr_stats_df.to_csv(out_path, index=False)
            pct = 100 * gwr_stats_df["gwr_fitted"].mean()
            logger.info("Saved GWR daily stats for %s (%d days, %.1f%% fitted) -> %s",
                        label, len(gwr_rows), pct, out_path.name)

            # GWR spatial coefficient summary (mean ± std across stations per day)
            coef_summary_rows = []
            for date, res in gwr_source.gwr_output_.daily_results.items():
                if not res.fitted or res.coefficients is None:
                    continue
                coef_cols = [c for c in res.coefficients.columns
                             if c not in ("lon", "lat", "station_id")]
                for col in coef_cols:
                    coef_summary_rows.append({
                        "date":      date,
                        "predictor": col,
                        "mean":      float(res.coefficients[col].mean()),
                        "std":       float(res.coefficients[col].std()),
                        "min":       float(res.coefficients[col].min()),
                        "max":       float(res.coefficients[col].max()),
                    })
            if coef_summary_rows:
                coef_sum_df = pd.DataFrame(coef_summary_rows).sort_values(["date", "predictor"])
                coef_out = output_dir / f"gwr_coef_summary_{label}.csv"
                coef_sum_df.to_csv(coef_out, index=False)
                logger.info("Saved GWR coefficient spatial summary -> %s", coef_out.name)


    lme_obj = fitted_hybrid.lme if fitted_hybrid is not None else fitted_lme_only
    if lme_obj is None and "train" not in stages and "validate" in stages:
        logger.info("No LME in memory (train stage skipped) — fitting on full "
                    "data once to populate lme_coefficients.csv.")
        lme_obj = build_lme().fit(df)
    if lme_obj is not None and lme_obj.result_ is not None:
        res = lme_obj.result_
        ci = res.conf_int()
        lme_coef_df = pd.DataFrame({
            "predictor":  res.fe_params.index,
            "coef":       res.fe_params.values,
            "se":         res.bse[res.fe_params.index].values,
            "z":          res.tvalues[res.fe_params.index].values,
            "p_value":    res.pvalues[res.fe_params.index].values,
            "ci_lower":   ci.loc[res.fe_params.index, 0].values,
            "ci_upper":   ci.loc[res.fe_params.index, 1].values,
        })
        lme_coef_df.to_csv(output_dir / "lme_coefficients.csv", index=False)
        logger.info("Saved LME coefficient table -> lme_coefficients.csv")

    # ── Feature importance (RF, GB, XGB standalone benchmarks) ───────
    if fitted_benchmarks:
        fi_rows = []
        for model_key, bm in fitted_benchmarks.items():
            inner = getattr(bm, "model", None)
            if inner is not None and hasattr(inner, "feature_importances_"):
                for pred, imp in zip(predictors, inner.feature_importances_):
                    fi_rows.append({"model": model_key, "predictor": pred,
                                    "importance": float(imp)})
        if fi_rows:
            fi_df = pd.DataFrame(fi_rows).sort_values(
                ["model", "importance"], ascending=[True, False])
            fi_df.to_csv(output_dir / "feature_importance.csv", index=False)
            logger.info("Saved feature importance -> feature_importance.csv")

    # ── Feature importance for hybrid tree-GWR models ────────────────
    for ht_model, ht_label in [
        (fitted_hybrid_rf,  "HybridRF_GWR"),
        (fitted_hybrid_gb,  "HybridGB_GWR"),
        (fitted_hybrid_xgb, "HybridXGB_GWR"),
    ]:
        if ht_model is None:
            continue
        inner = getattr(getattr(ht_model, "tree_model", None), "model", None)
        if inner is not None and hasattr(inner, "feature_importances_"):
            fi_ht = pd.DataFrame({
                "predictor":  predictors,
                "importance": inner.feature_importances_,
            }).sort_values("importance", ascending=False)
            fi_ht.to_csv(output_dir / f"feature_importance_{ht_label}.csv", index=False)
            logger.info("Saved feature importance for %s -> feature_importance_%s.csv",
                        ht_label, ht_label)

    # ── In-sample training metrics (overfitting check) ────────────────
    from src import metrics as _metrics
    insample_rows = []
    for model_obj, model_label in [
        (fitted_hybrid,     "HybridLMEGWR"),
        (fitted_hybrid_rf,  "HybridRF_GWR"),
        (fitted_hybrid_gb,  "HybridGB_GWR"),
        (fitted_hybrid_xgb, "HybridXGB_GWR"),
    ]:
        if model_obj is None:
            continue
        try:
            y_insample = model_obj.predict(df)
            y_obs = df[target].to_numpy()
            mask_is = ~np.isnan(y_insample) & ~np.isnan(y_obs)
            insample_rows.append({
                "model": model_label, "split": "train",
                **_metrics.all_metrics(y_obs[mask_is], y_insample[mask_is]),
            })
        except Exception as e:
            logger.warning("Could not compute in-sample metrics for %s: %s", model_label, e)
    if insample_rows:
        pd.DataFrame(insample_rows).to_csv(
            output_dir / "insample_metrics.csv", index=False)
        logger.info("Saved in-sample training metrics -> insample_metrics.csv")

    # =================================================================
    # Stage: validate
    # =================================================================
    if "validate" in stages:
        _banner("STAGE: validate — cross-validation")

        _tp_path = output_dir / "tuned_params.json"
        if not _tuned_bench and _tp_path.exists():
            import json
            with open(_tp_path) as f:
                _tuned_bench = json.load(f)
            logger.info("Loaded tuned hyperparameters from %s: %s",
                        _tp_path.name, sorted(_tuned_bench.keys()))
        if _tuned_rf_params is None:
            _tuned_rf_params = _tuned_bench.get("RF")
        if _tuned_gb_params is None:
            _tuned_gb_params = _tuned_bench.get("GB")
        if _tuned_xgb_params is None:
            _tuned_xgb_params = _tuned_bench.get("XGBoost")

        # CV runs on the RAW panel; each fold standardizes with a scaler
        # fitted on its own training rows (see validation._run_fold).
        df_cv = df_raw
        cv_scale = predictors if cfg["preprocessing"]["standardize_predictors"] else None
        random_splits = validation.random_kfold_indices(
            n=len(df_cv),
            n_splits=cfg["validation"]["random_kfold"],
            seed=cfg["validation"]["random_seed"],
        )
        site_splits = validation.site_kfold_indices(
            stations=df_cv[cols["station_id"]].to_numpy(),
            n_splits=cfg["validation"]["site_kfold"],
            seed=cfg["validation"]["random_seed"],
        )


        _n_cv_runs = 0

        # ----- Hybrid LME-GWR -----
        if "hybrid" in algorithms:
            def _train_hybrid(train_df):
                # No full-data bandwidth cache inside folds (leakage).
                return build_hybrid(use_bw_cache=False).fit(train_df)

            def _predict_hybrid(model, test_df):
                return model.predict(test_df)

            for splits, scheme in [(random_splits, "random_10fold"),
                                   (site_splits, "site_based")]:
                _banner(f"CV  HybridLMEGWR  |  scheme: {scheme}")
                _t0 = time.time()
                o, s, preds = validation.run_cv(
                    df_cv, splits, _train_hybrid, _predict_hybrid,
                    target_col=target, scheme_name=scheme, model_name="HybridLMEGWR",
                    n_jobs=1,  # GWR already uses n_jobs internally per fold
                    scale_predictors=cv_scale,
                )
                logger.info("HybridLMEGWR / %s CV done (%.1f min)", scheme, (time.time() - _t0) / 60)
                _checkpoint_cv(output_dir, o, s, preds)
                _n_cv_runs += 1

        # ----- LME alone -----
        if "lme" in algorithms:
            def _train_lme(train_df):
                return build_lme().fit(train_df)

            def _predict_lme(model, test_df):
                return model.predict(test_df)

            for splits, scheme in [(random_splits, "random_10fold"),
                                   (site_splits, "site_based")]:
                _banner(f"CV  LME  |  scheme: {scheme}")
                _t0 = time.time()
                o, s, preds = validation.run_cv(
                    df_cv, splits, _train_lme, _predict_lme,
                    target_col=target, scheme_name=scheme, model_name="LME",
                    n_jobs=n_jobs,
                    scale_predictors=cv_scale,
                )
                logger.info("LME / %s CV done (%.1f min)", scheme, (time.time() - _t0) / 60)
                _checkpoint_cv(output_dir, o, s, preds)
                _n_cv_runs += 1

        # ----- Benchmarks -----
        if "pcr" in algorithms:
            bench_specs.append((
                "PCR",
                lambda p: benchmarks.PCRModel(n_components=p),
                lambda X, y: benchmarks.tune_pcr(
                    X, y, cfg["benchmarks"]["pcr"]["n_components_search"],
                    n_splits=cfg["validation"]["random_kfold"],
                    seed=cfg["validation"]["random_seed"]),
            ))
        if "plsr" in algorithms:
            bench_specs.append((
                "PLSR",
                lambda p: benchmarks.PLSRModel(n_components=p),
                lambda X, y: benchmarks.tune_plsr(
                    X, y, cfg["benchmarks"]["plsr"]["n_components_search"],
                    n_splits=cfg["validation"]["random_kfold"],
                    seed=cfg["validation"]["random_seed"]),
            ))

        if "rf" in algorithms:
            bench_specs.append((
                "RF",
                lambda p: benchmarks.RFModel(n_jobs=4, **p),
                lambda X, y: benchmarks.tune_rf(
                    X, y, cfg["benchmarks"]["rf"],
                    n_splits=cfg["validation"]["random_kfold"],
                    seed=cfg["validation"]["random_seed"]),
            ))
        if "gb" in algorithms:
            bench_specs.append((
                "GB",
                lambda p: benchmarks.GBModel(n_threads=4, **p),
                lambda X, y: benchmarks.tune_gb(
                    X, y, cfg["benchmarks"]["gb"],
                    n_splits=cfg["validation"]["random_kfold"],
                    seed=cfg["validation"]["random_seed"]),
            ))
        if "xgboost" in algorithms:
            if benchmarks.XGB_AVAILABLE:
                bench_specs.append((
                    "XGBoost",
                    lambda p: benchmarks.XGBModel(n_jobs=4, force_cpu=True, **p),
                    lambda X, y: benchmarks.tune_xgb(
                        X, y, cfg["benchmarks"]["xgboost"],
                        n_splits=cfg["validation"]["random_kfold"],
                        seed=cfg["validation"]["random_seed"]),
                ))
            else:
                logger.warning("xgboost not installed — skipping XGBoost CV.")

        for name, make_fn, tune_fn in bench_specs:
            pre_params = _tuned_bench.get(name)
            if pre_params is None:
                logger.info("CV %s: no train-stage params — tuning inside each "
                            "training fold.", name)

            def _train(train_df, _make=make_fn, _tune=tune_fn, _pre=pre_params):
                X_tr = train_df[predictors].to_numpy()
                y_tr = train_df[target].to_numpy()
                params = _pre if _pre is not None else _tune(X_tr, y_tr)
                return _make(params).fit(X_tr, y_tr)

            def _predict(model, test_df):
                X_te = test_df[predictors].to_numpy()
                return model.predict(X_te)

            for splits, scheme in [(random_splits, "random_10fold"),
                                   (site_splits, "site_based")]:
                _banner(f"CV  {name}  |  scheme: {scheme}")
                _t0 = time.time()
                o, s, preds = validation.run_cv(
                    df_cv, splits, _train, _predict,
                    target_col=target, scheme_name=scheme, model_name=name,
                    n_jobs=n_jobs,
                    scale_predictors=cv_scale,
                )
                logger.info("%s / %s CV done (%.1f min)", name, scheme, (time.time() - _t0) / 60)
                _checkpoint_cv(output_dir, o, s, preds)
                _n_cv_runs += 1

        # ----- Hybrid tree-GWR -----
        hybrid_tree_specs = []
        if "hybrid_rf" in algorithms:
            hybrid_tree_specs.append(("HybridRF_GWR", cfg["benchmarks"]["rf"],
                                      benchmarks.tune_rf, build_hybrid_rf,
                                      _tuned_rf_params))
        if "hybrid_gb" in algorithms:
            hybrid_tree_specs.append(("HybridGB_GWR", cfg["benchmarks"]["gb"],
                                      benchmarks.tune_gb, build_hybrid_gb,
                                      _tuned_gb_params))
        if "hybrid_xgb" in algorithms:
            if benchmarks.XGB_AVAILABLE:
                hybrid_tree_specs.append(("HybridXGB_GWR", cfg["benchmarks"]["xgboost"],
                                          benchmarks.tune_xgb, build_hybrid_xgb,
                                          _tuned_xgb_params))
            else:
                logger.warning("xgboost not installed — skipping Hybrid XGB-GWR CV.")

        for model_name, grid, tune_fn, builder_fn, pretrained_params in hybrid_tree_specs:
            def _train_htgwr(train_df, _grid=grid, _tune=tune_fn, _build=builder_fn,
                             _pre=pretrained_params):
                # Reuse params tuned on full data when available — avoids
                # repeating the expensive inner grid-search 10× per CV scheme.
                # (Same protocol as the standalone benchmarks above — parity.)
                if _pre is not None:
                    params = _pre
                else:
                    X_tr = train_df[predictors].to_numpy()
                    y_tr = train_df[target].to_numpy()
                    params = _tune(X_tr, y_tr, _grid,
                                   n_splits=cfg["validation"]["random_kfold"],
                                   seed=cfg["validation"]["random_seed"])
                return _build(params, use_bw_cache=False).fit(train_df)

            def _predict_htgwr(model, test_df):
                return model.predict(test_df)

            for splits, scheme in [(random_splits, "random_10fold"),
                                   (site_splits, "site_based")]:
                _banner(f"CV  {model_name}  |  scheme: {scheme}")
                _t0 = time.time()
                o, s, preds = validation.run_cv(
                    df_cv, splits, _train_htgwr, _predict_htgwr,
                    target_col=target, scheme_name=scheme, model_name=model_name,
                    n_jobs=1,  # GWR already uses n_jobs internally per fold
                    scale_predictors=cv_scale,
                )
                logger.info("%s / %s CV done (%.1f min)", model_name, scheme, (time.time() - _t0) / 60)
                _checkpoint_cv(output_dir, o, s, preds)
                _n_cv_runs += 1

        logger.info("This invocation completed %d model x scheme CV run(s).", _n_cv_runs)
        overall_df = pd.read_csv(output_dir / "cv_overall.csv")
        logger.info("Saved CV metrics:\n%s", overall_df.to_string(index=False))

        cv_preds_path = output_dir / "cv_predictions.csv"
        cv_preds_df = pd.read_csv(cv_preds_path) if cv_preds_path.exists() else pd.DataFrame()
        logger.info("CV raw predictions on disk: %d rows -> %s",
                    len(cv_preds_df), cv_preds_path.name)

        # ── Per-fold metrics (for error-bar / box plots) ───────────────
        fold_metric_rows = []
        if not cv_preds_df.empty:
            for (mdl, scheme, fold), grp in cv_preds_df.groupby(
                    ["model", "scheme", "fold"]):
                mask_f = ~np.isnan(grp["y_pred"].values) & ~np.isnan(grp["y_true"].values)
                if mask_f.sum() < 2:
                    continue
                fold_metric_rows.append({
                    "model": mdl, "scheme": scheme, "fold": fold,
                    **validation.metrics.all_metrics(
                        grp["y_true"].values[mask_f],
                        grp["y_pred"].values[mask_f]),
                })
        if fold_metric_rows:
            pd.DataFrame(fold_metric_rows).to_csv(
                output_dir / "cv_fold_metrics.csv", index=False)
            logger.info("Saved per-fold CV metrics -> cv_fold_metrics.csv")

    # =================================================================
    # Stage: map
    # =================================================================
    if "map" in stages and cfg["mapping"]["enable"]:
        _banner("STAGE: map — predicting on grid")
        grid_df = data_loader.load_grid(cfg["paths"].get("grid_file"), cols)
        if grid_df is None:
            logger.info("No grid file — skipping mapping stage.")
        else:
            if fitted_hybrid is None:
                # Reload from disk if previously trained
                model_path = output_dir / "hybrid_model.pkl"
                if model_path.exists():
                    with open(model_path, "rb") as f:
                        fitted_hybrid = pickle.load(f)
                else:
                    logger.info("Hybrid model not in memory or on disk — fitting now.")
                    fitted_hybrid = build_hybrid().fit(df)

            if cfg["preprocessing"]["standardize_predictors"]:
                grid_df, _ = preprocessing.standardize(grid_df, predictors, fit_on=df_raw)

            logger.info("Predicting on %d grid rows", len(grid_df))
            y_grid = fitted_hybrid.predict_grid(grid_df)

            grids = mapping.aggregate_grid_predictions(
                grid_df, y_grid,
                min_valid_days_per_month=cfg["mapping"]["min_valid_days_per_month"],
                grid_id_col="grid_id" if "grid_id" in grid_df.columns else cols["lat"],
                date_col=cols["date"],
            )
            mapping.save_grids(grids, output_dir,
                               save_daily=cfg["mapping"]["save_daily_grids"],
                               save_monthly=cfg["mapping"]["save_monthly_grids"],
                               save_seasonal=cfg["mapping"]["save_seasonal_grids"])

    # =================================================================
    # Stage: map_raster (recommended for country-scale, multi-year prediction)
    # =================================================================
    if "map_raster" in stages:
        from src import raster_prediction

        rcfg = cfg["raster_mapping"]
        kind = rcfg.get("model_kind", "hybrid_xgb").lower()


        _HYBRID_TREE_PKLS = {
            "hybrid_xgb": output_dir / "hybrid_xgb_model.pkl",
            "hybrid_rf":  output_dir / "hybrid_rf_model.pkl",
            "hybrid_gb":  output_dir / "hybrid_gb_model.pkl",
        }

        # Resolve which fitted model to use
        if kind in _HYBRID_TREE_PKLS:
            pkl_path = _HYBRID_TREE_PKLS[kind]
            if not pkl_path.exists():
                raise FileNotFoundError(
                    f"{pkl_path.name} not found — run --stages train first."
                )
            with open(pkl_path, "rb") as f:
                model_obj = pickle.load(f)
            model_kind = "hybrid"
            logger.info("Raster prediction model: %s (loaded from %s)",
                        kind.upper(), pkl_path.name)

        elif kind == "hybrid":
            if fitted_hybrid is None:
                model_path = output_dir / "hybrid_model.pkl"
                if not model_path.exists():
                    logger.info("Hybrid model not on disk — fitting now.")
                    fitted_hybrid = build_hybrid().fit(df)
                    with open(model_path, "wb") as f:
                        pickle.dump(fitted_hybrid, f)
                else:
                    with open(model_path, "rb") as f:
                        fitted_hybrid = pickle.load(f)
            model_obj = fitted_hybrid
            model_kind = "hybrid"
            logger.info("Raster prediction model: HybridLMEGWR (hybrid_model.pkl)")

        elif kind == "lme":
            lme_obj = build_lme().fit(df)
            model_obj = lme_obj
            model_kind = "lme"
            logger.info("Raster prediction model: LME (fixed-effects only)")

        else:
            # Standalone benchmark by name (PCR, PLSR, RF, GB, XGBoost)
            bench_path = output_dir / "benchmarks.pkl"
            if not bench_path.exists():
                raise FileNotFoundError(
                    f"benchmarks.pkl not found — run --stages train first to fit "
                    f"the {kind.upper()} benchmark."
                )
            with open(bench_path, "rb") as f:
                bench_dict = pickle.load(f)
            key = kind.upper()
            if key not in bench_dict:
                raise KeyError(f"Benchmark '{key}' not in fitted benchmarks: "
                               f"{list(bench_dict.keys())}")
            model_obj = bench_dict[key]
            model_kind = "sklearn"
            logger.info("Raster prediction model: %s (benchmark)", key)

        # Load the persisted scaler (raster predictors arrive raw)
        scaler_obj = None
        scaler_path = output_dir / "predictor_scaler.pkl"
        if cfg["preprocessing"]["standardize_predictors"]:
            if scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    scaler_obj = pickle.load(f)["scaler"]
            else:
                logger.warning("Scaler file missing — raster predictors will "
                               "NOT be standardized. This will degrade results "
                               "if training used standardization.")


        _raster_dates = raster_prediction.discover_dates(
            Path(rcfg["grid_dir"]), rcfg["daily_predictors"])
        period = cfg.get("study_period") or {}
        if period.get("start") or period.get("end"):
            _t0p = pd.Timestamp(period.get("start", min(_raster_dates)))
            _t1p = pd.Timestamp(period.get("end", max(_raster_dates)))
            n_all = len(_raster_dates)
            _raster_dates = [d for d in _raster_dates if _t0p <= d <= _t1p]
            logger.info("Raster dates restricted to study period %s..%s: "
                        "%d -> %d days", _t0p.date(), _t1p.date(),
                        n_all, len(_raster_dates))

        _banner("Raster prediction — generating daily PM2.5 surfaces")
        _written_rasters, _day_log_df = raster_prediction.predict_all_days_raster(
            model=model_obj,
            grid_dir=rcfg["grid_dir"],
            output_dir=rcfg["output_dir"],
            predictors=predictors,
            daily_predictors=rcfg["daily_predictors"],
            static_predictors=rcfg["static_predictors"],
            model_kind=model_kind,
            scaler=scaler_obj,
            dates=_raster_dates,
            k_neighbors=rcfg.get("k_neighbors_idw", 8),
            skip_existing=rcfg.get("skip_existing", True),
            n_jobs=n_jobs,
        )
        if not _day_log_df.empty:
            _day_log_path = Path(rcfg["output_dir"]) / "raster_prediction_log.csv"
            _day_log_df.to_csv(_day_log_path, index=False)
            logger.info("Saved per-day prediction log (%d days) -> %s",
                        len(_day_log_df), _day_log_path.name)

        # Aggregate daily rasters into monthly, seasonal, and annual means
        # The elev.tif land mask ensures every Thailand land pixel has a value.
        _elev_path = Path(rcfg["grid_dir"]) / "elev.tif"
        _land_mask_path = _elev_path if _elev_path.exists() else None
        if _land_mask_path is None:
            logger.warning("elev.tif not found in %s — gap-filling will be skipped "
                           "for aggregated maps.", rcfg["grid_dir"])

        agg = rcfg.get("aggregate", {})
        if agg.get("enable", True):
            _banner("Raster aggregation — monthly / seasonal / annual means")
            raster_prediction.aggregate_daily_rasters(
                daily_dir=rcfg["output_dir"],
                output_dir=rcfg["output_dir"],
                min_valid_days_per_month=agg.get("min_valid_days_per_month", 7),
                save_monthly=agg.get("save_monthly", True),
                save_seasonal=agg.get("save_seasonal", True),
                land_mask_path=_land_mask_path,
            )
            raster_prediction.aggregate_annual_rasters(
                daily_dir=rcfg["output_dir"],
                output_dir=rcfg["output_dir"],
                min_valid_days_per_year=agg.get("min_valid_days_per_year", 180),
                land_mask_path=_land_mask_path,
            )

        # ── Report-ready outputs ─────────────────────────────────────────
        # WHO / Thailand NAAQS 24-hr exceedance fraction maps
        _banner("Raster outputs — exceedance maps + spatial statistics")
        raster_prediction.compute_exceedance_fraction_maps(
            daily_dir=rcfg["output_dir"],
            output_dir=rcfg["output_dir"],
        )

        # Spatial summary statistics CSV (used for paper tables)
        # Includes optional population-weighted PM2.5 exposure statistics
        _pop_raster = Path(rcfg["grid_dir"]) / "pop.tif"
        raster_prediction.compute_spatial_stats_csv(
            raster_dir=rcfg["output_dir"],
            output_csv=Path(rcfg["output_dir"]) / "pm25_spatial_stats.csv",
            pop_raster_path=_pop_raster if _pop_raster.exists() else None,
        )

    _banner(f"Pipeline finished. Total time: {_elapsed()}")
    logger.info("Outputs in %s", output_dir)


# =====================================================================
if __name__ == "__main__":
    main()
