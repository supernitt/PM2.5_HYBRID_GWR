#!/usr/bin/env python
"""Block C — statistical analyses that carry the manuscript's novelty claim.

Purpose
-------
1. Fold-level paired tests + bootstrap CIs on the site-based CV ranking:
   is the top-3 hybrid ordering statistically distinguishable, and is the
   hybrid-vs-standalone gain significant per estimator family?
   (Protects the model-selection-reversal claim from the one-sentence
   reviewer kill identified in output/reports/01_novelty_diagnosis.md.)
2. Rank-stability bootstrap: P(model is #1 | site-based CV) per arm.
3. Spine demonstration: per-day Moran's I p BEFORE vs AFTER the Stage-2
   correction (gwr_daily_stats_*.csv, columns morans_p / morans_p_after),
   plus bandwidth distribution and %-at-search-floor.
4. Day-level dRMSE (standalone - hybrid) vs Stage-1 residual-autocorrelation
   intensity (-log10 morans_p): the pre-fit diagnostic that turns the
   correction gain into a predictable, transferable quantity.

Inputs (per arm)
----------------
outputs{,_aod_imputed}/cv_fold_metrics.csv        (model,scheme,fold,R2,RMSE,...)
outputs{,_aod_imputed}/cv_predictions.csv         (row-level CV predictions)
outputs{,_aod_imputed}/cv_stratified.csv          (season strata)
outputs{,_aod_imputed}/gwr_daily_stats_hybrid*.csv
./data/panel.csv                                 (date recovery, see note)

"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED = 42
N_BOOT = 10_000
ROOT = Path(__file__).resolve().parents[1]          # .../pm25_hybrid
PANEL = ROOT / "data" / "panel.csv"
OUT_ROOT = ROOT / "output" / "analysis" / "block_c"

ARMS = {
    "observed": ROOT / "outputs",
    "imputed":  ROOT / "outputs_aod_imputed",
}
HYBRID_OF = {"LME": "HybridLMEGWR", "RF": "HybridRF_GWR",
             "GB": "HybridGB_GWR", "XGBoost": "HybridXGB_GWR"}
TOP3 = ["HybridXGB_GWR", "HybridRF_GWR", "HybridGB_GWR"]
GWR_STATS_FILES = {"HybridLMEGWR": "gwr_daily_stats_hybrid.csv",
                   "HybridRF_GWR": "gwr_daily_stats_hybrid_rf.csv",
                   "HybridGB_GWR": "gwr_daily_stats_hybrid_gb.csv",
                   "HybridXGB_GWR": "gwr_daily_stats_hybrid_xgb.csv"}


def _req(path: Path) -> Path:
    assert path.exists(), f"Required input missing: {path}"
    return path


def paired_tests(fold: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """Pairwise fold-level paired t + Wilcoxon on R2 and RMSE."""
    rows = []
    sub = fold[fold.scheme == scheme]
    models = sorted(sub.model.unique())
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            fa = sub[sub.model == a].set_index("fold").sort_index()
            fb = sub[sub.model == b].set_index("fold").sort_index()
            common = fa.index.intersection(fb.index)
            if len(common) < 5:
                continue
            for metric in ("R2", "RMSE"):
                da = fa.loc[common, metric].to_numpy()
                db = fb.loc[common, metric].to_numpy()
                diff = da - db
                t_p = stats.ttest_rel(da, db).pvalue
                try:
                    w_p = stats.wilcoxon(da, db).pvalue
                except ValueError:
                    w_p = np.nan
                rng = np.random.default_rng(SEED)
                boots = np.array([
                    diff[rng.integers(0, len(diff), len(diff))].mean()
                    for _ in range(N_BOOT)])
                rows.append({
                    "model_a": a, "model_b": b, "metric": metric,
                    "mean_diff_a_minus_b": diff.mean(),
                    "ci95_lo": np.percentile(boots, 2.5),
                    "ci95_hi": np.percentile(boots, 97.5),
                    "paired_t_p": t_p, "wilcoxon_p": w_p, "n_folds": len(common),
                })
    return pd.DataFrame(rows)


def hybrid_gain(fold: pd.DataFrame) -> pd.DataFrame:
    """Hybrid minus its own standalone, per family and scheme."""
    rows = []
    for scheme in fold.scheme.unique():
        sub = fold[fold.scheme == scheme]
        for base, hyb in HYBRID_OF.items():
            fb = sub[sub.model == base].set_index("fold").sort_index()
            fh = sub[sub.model == hyb].set_index("fold").sort_index()
            common = fb.index.intersection(fh.index)
            if len(common) < 5:
                continue
            d_r2 = (fh.loc[common, "R2"] - fb.loc[common, "R2"]).to_numpy()
            d_rmse = (fb.loc[common, "RMSE"] - fh.loc[common, "RMSE"]).to_numpy()
            rows.append({
                "scheme": scheme, "family": base, "hybrid": hyb,
                "mean_dR2": d_r2.mean(),
                "dR2_t_p": stats.ttest_rel(fh.loc[common, "R2"],
                                           fb.loc[common, "R2"]).pvalue,
                "mean_dRMSE_improvement": d_rmse.mean(),
                "frac_folds_improved": float((d_r2 > 0).mean()),
                "n_folds": len(common),
            })
    return pd.DataFrame(rows)


def rank_bootstrap(fold: pd.DataFrame, scheme: str = "site_based") -> pd.DataFrame:
    """P(model ranks #1 on mean R2) under fold resampling."""
    sub = fold[fold.scheme == scheme]
    models = sorted(sub.model.unique())
    mat = pd.DataFrame({
        m: sub[sub.model == m].set_index("fold")["R2"] for m in models
    }).dropna()
    rng = np.random.default_rng(SEED)
    n = len(mat)
    wins = dict.fromkeys(models, 0)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        wins[mat.iloc[idx].mean().idxmax()] += 1
    return (pd.DataFrame({"model": list(wins), "p_rank1": [wins[m] / N_BOOT for m in wins]})
            .sort_values("p_rank1", ascending=False))


def morans_spine(arm_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Before/after Moran's I and bandwidth summaries per hybrid."""
    spine, bw_rows = [], []
    for model, fname in GWR_STATS_FILES.items():
        p = arm_dir / fname
        if not p.exists():
            continue
        d = pd.read_csv(p)
        fitted = d[d.gwr_fitted == True]  # noqa: E712
        if fitted.empty:
            continue
        has_after = "morans_p_after" in fitted.columns and fitted["morans_p_after"].notna().any()
        spine.append({
            "model": model,
            "days_fitted": len(fitted),
            "pct_sig_before": 100 * (fitted.morans_p < 0.05).mean(),
            "pct_sig_after": (100 * (fitted.morans_p_after < 0.05).mean()
                              if has_after else np.nan),
            "median_p_before": fitted.morans_p.median(),
            "median_p_after": (fitted.morans_p_after.median() if has_after else np.nan),
        })
        bw = fitted.bandwidth.dropna()
        bw_rows.append({
            "model": model, "bw_median": bw.median(),
            "bw_p10": bw.quantile(.1), "bw_p90": bw.quantile(.9),
            "pct_at_floor": (100 * fitted.bw_at_floor.fillna(False).mean()
                             if "bw_at_floor" in fitted.columns else np.nan),
        })
    return pd.DataFrame(spine), pd.DataFrame(bw_rows)


def rebuild_arm_panel(arm: str) -> pd.DataFrame:
    import yaml
    sys.path.insert(0, str(ROOT))
    from src import data_loader, preprocessing as prep

    cfg_file = ROOT / "configs" / ("default.yaml" if arm == "observed"
                                   else "aod_imputed.yaml")
    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)
    cols = cfg["columns"]
    df = data_loader.load_panel(PANEL, cols)
    period = cfg.get("study_period") or {}
    if period:
        t0, t1 = pd.Timestamp(period["start"]), pd.Timestamp(period["end"])
        df = df[(df[cols["date"]] >= t0) & (df[cols["date"]] <= t1)]
        df = df.reset_index(drop=True)
    vr = cfg["preprocessing"].get("valid_ranges") or {}
    if vr:
        df = prep.validate_ranges(df, vr)
    if cfg["preprocessing"].get("aod_policy", "observed") == "imputed":
        df = prep.impute_aod(df, aod=cols["aod"], station_id=cols["station_id"],
                             date_col=cols["date"],
                             max_gap_days=int(cfg["preprocessing"]
                                              .get("aod_impute_max_gap_days", 7)))
    df = prep.filter_missing(df, target=cols["target"], aod=cols["aod"],
                             drop_target=cfg["preprocessing"]["drop_missing_target"],
                             drop_aod=cfg["preprocessing"]["drop_missing_aod"])
    predictors = prep.assemble_predictors(cols)
    predictors = prep.drop_all_nan_predictors(df, predictors)
    df = prep.drop_missing_predictors(df, predictors)
    return df[[cols["station_id"], cols["date"], cols["target"]]].rename(
        columns={cols["station_id"]: "station_id", cols["date"]: "date",
                 cols["target"]: "pm25"})


def recover_dates(preds: pd.DataFrame, arm: str) -> pd.DataFrame | None:
    panel = rebuild_arm_panel(arm)
    by_station = {sid: g.reset_index(drop=True)
                  for sid, g in panel.groupby("station_id")}
    site = preds[preds.scheme == "site_based"].copy()
    out_parts, n_skip = [], 0
    for (model, sid), g in site.groupby(["model", "station_id"], sort=False):
        ref = by_station.get(sid)
        g = g.reset_index(drop=True)
        if ref is None or len(ref) != len(g) or \
                not np.allclose(ref.pm25.to_numpy(), g.y_true.to_numpy(),
                                rtol=0, atol=1e-9):
            n_skip += 1
            continue
        g["date"] = ref.date.to_numpy()
        out_parts.append(g)
    if not out_parts:
        print("    positional recovery failed for every station — skipping.")
        return None
    out = pd.concat(out_parts, ignore_index=True)
    total_groups = site.groupby(["model", "station_id"]).ngroups
    rate = out.shape[0] / site.shape[0]
    print(f"    positional date recovery (site-based): {100 * rate:.1f}% of rows"
          f" ({total_groups - n_skip}/{total_groups} model-station groups verified)")
    if rate < 0.95:
        print("    < 95% — skipping day-level regression (biased subset risk).")
        return None
    return out


def drmse_vs_moran(preds: pd.DataFrame, arm_dir: Path) -> pd.DataFrame:
    """Day-level dRMSE (standalone - hybrid) ~ -log10(stage-1 Moran p)."""
    rows = []
    site = preds[preds.scheme == "site_based"]
    for base, hyb in HYBRID_OF.items():
        stats_file = arm_dir / GWR_STATS_FILES[hyb]
        if not stats_file.exists():
            continue
        gd = pd.read_csv(stats_file, parse_dates=["date"])
        gd = gd[gd.morans_p.notna()][["date", "morans_p", "gwr_fitted"]]

        def day_rmse(model):
            s = site[site.model == model]
            return (s.assign(sq=(s.y_true - s.y_pred) ** 2)
                    .groupby("date")["sq"].mean() ** 0.5)

        rb, rh = day_rmse(base), day_rmse(hyb)
        common = rb.index.intersection(rh.index)
        d = pd.DataFrame({"dRMSE": rb[common] - rh[common]}).reset_index()
        d["date"] = pd.to_datetime(d["date"])
        d = d.merge(gd, on="date", how="inner")
        if len(d) < 50:
            continue
        d["intensity"] = -np.log10(d.morans_p.clip(lower=1e-300))
        ols = stats.linregress(d.intensity, d.dRMSE)
        rho = stats.spearmanr(d.intensity, d.dRMSE)
        gated = d[d.gwr_fitted == True]  # noqa: E712
        ungated = d[d.gwr_fitted == False]  # noqa: E712
        rows.append({
            "family": base, "hybrid": hyb, "n_days": len(d),
            "ols_slope": ols.slope, "ols_p": ols.pvalue, "ols_r2": ols.rvalue ** 2,
            "spearman_rho": rho.statistic, "spearman_p": rho.pvalue,
            "mean_dRMSE_gated_days": gated.dRMSE.mean() if len(gated) else np.nan,
            "mean_dRMSE_ungated_days": ungated.dRMSE.mean() if len(ungated) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> int:
    print("Block C analysis  |  seed", SEED, "|", N_BOOT, "bootstrap draws")
    _req(PANEL)   # date recovery rebuilds the filtered panel on demand

    for arm, arm_dir in ARMS.items():
        if not arm_dir.exists():
            print(f"[{arm}] {arm_dir} missing — skipped")
            continue
        print(f"\n=== ARM: {arm} ({arm_dir.name}) ===")
        out = OUT_ROOT / arm
        out.mkdir(parents=True, exist_ok=True)

        fold = pd.read_csv(_req(arm_dir / "cv_fold_metrics.csv"))
        assert {"model", "scheme", "fold", "R2", "RMSE"} <= set(fold.columns)

        print("  [1/5] paired fold-level tests (site-based, all pairs) ...")
        pt = paired_tests(fold, "site_based")
        pt.to_csv(out / "paired_tests_site.csv", index=False)

        print("  [2/5] hybrid-vs-standalone gains ...")
        hg = hybrid_gain(fold)
        hg.to_csv(out / "hybrid_gain_tests.csv", index=False)

        print("  [3/5] rank-stability bootstrap ...")
        rb = rank_bootstrap(fold)
        rb.to_csv(out / "rank_bootstrap.csv", index=False)

        print("  [4/5] Moran spine + bandwidths ...")
        spine, bw = morans_spine(arm_dir)
        spine.to_csv(out / "morans_before_after.csv", index=False)
        bw.to_csv(out / "bandwidth_summary.csv", index=False)

        print("  [5/5] day-level dRMSE vs Moran intensity ...")
        preds = pd.read_csv(_req(arm_dir / "cv_predictions.csv"))
        if "date" in preds.columns and preds["date"].notna().any():
            preds["date"] = pd.to_datetime(preds["date"])
            dated = preds
        else:
            dated = recover_dates(preds, arm)
        if dated is not None:
            dm = drmse_vs_moran(dated, arm_dir)
            dm.to_csv(out / "drmse_vs_moran.csv", index=False)
        else:
            dm = pd.DataFrame()

        strat = pd.read_csv(_req(arm_dir / "cv_stratified.csv"))
        wet = strat[(strat.stratum == "season") & (strat.value == "wet_monsoon")
                    & (strat.scheme == "site_based")].sort_values("R2", ascending=False)
        wet.to_csv(out / "wet_season_site.csv", index=False)

        # ---- summary.md -------------------------------------------------
        top3 = pt[(pt.model_a.isin(TOP3)) & (pt.model_b.isin(TOP3))
                  & (pt.metric == "R2")]
        with open(out / "summary.md", "w") as f:
            f.write(f"# Block C summary — {arm} arm\n\n")
            f.write("## Top-3 site-based distinguishability (R2, paired)\n\n")
            f.write(top3.to_string(index=False) + "\n\n")
            f.write("## P(rank #1 | site-based), fold bootstrap\n\n")
            f.write(rb.to_string(index=False) + "\n\n")
            f.write("## Hybrid gain per family\n\n")
            f.write(hg.to_string(index=False) + "\n\n")
            f.write("## Moran's I before/after Stage-2 (spine)\n\n")
            f.write(spine.to_string(index=False) + "\n\n")
            f.write("## Bandwidths\n\n" + bw.to_string(index=False) + "\n\n")
            if not dm.empty:
                f.write("## dRMSE vs Moran intensity (site-based, day level)\n\n")
                f.write(dm.to_string(index=False) + "\n\n")
            f.write("## Wet-monsoon site-based honesty table\n\n")
            f.write(wet.to_string(index=False) + "\n")
        print(f"  -> {out / 'summary.md'}")

    print("\nDone. All outputs under", OUT_ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
