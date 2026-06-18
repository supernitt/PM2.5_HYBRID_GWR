"""Generate manuscript-ready tables and figures from the pipeline outputs.

Run this **after** ``main.py`` has finished (it reads the CSVs that the
train/validate stages write to ``<output_dir>/``).  It does not re-fit any
model — it only summarises the existing results into publication-quality
assets for the Scopus manuscript.

Usage::

    python scripts/make_report.py --config configs/default.yaml

Outputs go to ``<output_dir>/report/``::

    report/
        figures/   *.png and *.pdf (300 dpi, vector PDF for the camera-ready)
        tables/    *.csv and *.tex  (LaTeX booktabs for direct \\input{})
        REPORT_MANIFEST.md   list of every asset with a draft caption

Every table/figure is produced independently and wrapped in error handling,
so a single missing CSV only skips that one asset instead of aborting the run.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------------
# Optional: reuse the project's config loader so --config "just works".
# Falls back to a tiny YAML read if the package import is unavailable
# (e.g. when this script is copied elsewhere).
# ---------------------------------------------------------------------
try:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import utils as _utils

    def _load_config(path):
        return _utils.load_config(path)
except Exception:  # pragma: no cover - fallback path
    import yaml

    def _load_config(path):
        with open(path, "r", encoding="utf-8-sig") as f:
            return yaml.safe_load(f)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | report | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("report")


# =====================================================================
# Publication style + display-name maps
# =====================================================================
def _set_elsevier_font():
    """Force the Elsevier/CAS look: a Times-family serif for text and maths.

    Elsevier journals (including the cas-dc class used by this manuscript) are
    typeset in a Times-like serif.  We prefer an installed Times-compatible face
    --- Times New Roman, or the metric-compatible Tinos / Nimbus Roman /
    Liberation Serif commonly found on Linux --- and fall back to matplotlib's
    bundled STIX family, which is Times-metric-compatible, so the figures render
    the same on any machine even when Times New Roman is not installed.  Maths is
    set to the matching STIX glyph set so equations and axis labels share the
    Times look.
    """
    preferred = ["Times New Roman", "Times", "Tinos", "Nimbus Roman",
                 "Nimbus Roman No9 L", "Liberation Serif", "FreeSerif"]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((name for name in preferred if name in available), None)
    if chosen is not None:
        serif_list = ([chosen] + [n for n in preferred if n != chosen]
                      + ["STIX Two Text", "STIXGeneral", "DejaVu Serif"])
        plt.rcParams["font.serif"] = serif_list
        logger.info("Report font: '%s' (Times-compatible serif).", chosen)
    else:
        plt.rcParams["font.serif"] = ["STIX Two Text", "STIXGeneral", "DejaVu Serif"]
        logger.warning(
            "No Times-family font installed; using matplotlib's bundled STIX "
            "(Times-metric-compatible). For an exact match install 'Times New "
            "Roman' or the metric-compatible 'Tinos' / 'Liberation Serif'."
        )
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "stix"   # Times-compatible maths
    plt.rcParams["mathtext.rm"] = "serif"


plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 600,          # > 400 dpi for camera-ready raster figures
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})
_set_elsevier_font()

# Single-column / double-column widths (inches) following Elsevier guidance.
COL1 = 3.54
COL2 = 7.48

# Canonical model display order + pretty labels.
MODEL_ORDER = [
    "LME", "PCR", "PLSR", "RF", "GB", "XGBoost",
    "HybridLMEGWR", "HybridRF_GWR", "HybridGB_GWR", "HybridXGB_GWR",
]
MODEL_LABEL = {
    "LME": "LME",
    "PCR": "PCR",
    "PLSR": "PLSR",
    "RF": "RF",
    "GB": "GB",
    "XGBoost": "XGBoost",
    "HybridLMEGWR": "Hybrid LME-GWR",
    "HybridRF_GWR": "Hybrid RF-GWR",
    "HybridGB_GWR": "Hybrid GB-GWR",
    "HybridXGB_GWR": "Hybrid XGB-GWR",
}
SCHEME_LABEL = {
    "random_10fold": "Random 10-fold CV",
    "site_based": "Site-based CV",
}
SEASON_ORDER = ["cool_dry", "hot_dry", "wet_monsoon"]
SEASON_LABEL = {
    "cool_dry": "Cool-dry",
    "hot_dry": "Hot-dry",
    "wet_monsoon": "Wet-monsoon",
}
PRED_LABEL = {
    "intercept": "Intercept", "Intercept": "Intercept",
    "aod": "AOD", "t2m": "2 m temperature", "rh": "Relative humidity",
    "ws": "Wind speed", "blh": "Boundary-layer height", "prec": "Precipitation",
    "ssr": "Surface solar radiation", "sp": "Surface pressure", "frp": "Fire radiative power",
    "ndvi": "NDVI", "elev": "Elevation", "pop": "Population density",
}

# Colour-blind-safe palette (Wong, 2011).
C_PRIMARY = "#0072B2"   # blue
C_SECOND = "#D55E00"    # vermillion
C_THIRD = "#009E73"     # green
C_GREY = "#888888"


def _model_sort_key(m: str) -> int:
    return MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER)


def _pretty_model(m: str) -> str:
    return MODEL_LABEL.get(m, m)


def _stars(p) -> str:
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        logger.warning("Missing input: %s — skipping dependent assets.", path.name)
        return None
    return pd.read_csv(path)


def _savefig(fig, fig_dir: Path, name: str, manifest: list, caption: str):
    png = fig_dir / f"{name}.png"
    pdf = fig_dir / f"{name}.pdf"
    fig.savefig(png, dpi=600)    # raster: 600 dpi (> 400 dpi requirement)
    fig.savefig(pdf)             # vector PDF: resolution-independent
    plt.close(fig)
    manifest.append(("figure", f"figures/{name}.png", caption))
    logger.info("Saved figure -> %s (+ .pdf)", png.name)


def _save_table(df: pd.DataFrame, tab_dir: Path, name: str, manifest: list,
                caption: str, index: bool = False, float_format: str = "%.3f"):
    csv = tab_dir / f"{name}.csv"
    df.to_csv(csv, index=index)
    tex = tab_dir / f"{name}.tex"
    try:
        df.to_latex(
            tex, index=index, escape=True, float_format=float_format,
            caption=caption, label=f"tab:{name}", longtable=False,
            column_format="l" + "r" * (df.shape[1] - 1),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("LaTeX export failed for %s (%s) — CSV still written.", name, e)
    manifest.append(("table", f"tables/{name}.csv", caption))
    logger.info("Saved table -> %s (+ .tex)", csv.name)


# =====================================================================
# TABLES
# =====================================================================
def table_cv_performance(out: Path, tab_dir: Path, manifest: list) -> pd.DataFrame | None:
    """Table 1 — headline CV metrics, every model × CV scheme."""
    df = _read_csv(out / "cv_overall.csv")
    if df is None:
        return None
    df = df.copy()
    df["__k"] = df["model"].map(_model_sort_key)
    df = df.sort_values(["__k", "scheme"]).drop(columns="__k")

    wide = df.pivot(index="model", columns="scheme",
                    values=["R2", "RMSE", "MAE", "MB"])
    # Order columns scheme-major: (Random R2, RMSE, MAE, MB), (Site ...)
    schemes = [s for s in ["random_10fold", "site_based"] if s in df["scheme"].unique()]
    metrics = ["R2", "RMSE", "MAE", "MB"]
    wide = wide.reindex(columns=[(m, s) for s in schemes for m in metrics])
    wide = wide.reindex([m for m in MODEL_ORDER if m in wide.index])

    flat = wide.copy()
    flat.columns = [f"{SCHEME_LABEL.get(s, s)} — {m}" for (m, s) in wide.columns]
    flat.insert(0, "Model", [_pretty_model(m) for m in flat.index])
    flat = flat.reset_index(drop=True)

    _save_table(
        flat, tab_dir, "table1_cv_performance", manifest,
        caption=("Ten-fold cross-validation performance of all candidate models "
                 "under random and site-based hold-out schemes. R\\textsuperscript{2}, "
                 "coefficient of determination; RMSE and MAE in \\textmu g m\\textsuperscript{-3}; "
                 "MB, mean bias."),
    )
    return df


def table_seasonal(out: Path, tab_dir: Path, manifest: list) -> pd.DataFrame | None:
    """Table 2 — seasonal stratified performance."""
    df = _read_csv(out / "cv_stratified.csv")
    if df is None:
        return None
    df = df[df["stratum"] == "season"].copy()
    if df.empty:
        logger.warning("No season strata in cv_stratified.csv — skipping seasonal table.")
        return None

    df["season_label"] = df["value"].map(SEASON_LABEL).fillna(df["value"])
    df["__sk"] = df["value"].map(lambda v: SEASON_ORDER.index(v) if v in SEASON_ORDER else 9)
    df["__mk"] = df["model"].map(_model_sort_key)
    df = df.sort_values(["scheme", "__mk", "__sk"])

    tidy = df[["model", "scheme", "season_label", "R2", "RMSE", "MAE", "MB", "n"]].copy()
    tidy["model"] = tidy["model"].map(_pretty_model)
    tidy["scheme"] = tidy["scheme"].map(lambda s: SCHEME_LABEL.get(s, s))
    tidy.columns = ["Model", "CV scheme", "Season", "R2", "RMSE", "MAE", "MB", "n"]

    _save_table(
        tidy, tab_dir, "table2_seasonal_performance", manifest,
        caption=("Season-stratified cross-validation performance. Seasons follow the "
                 "Thai meteorological calendar (cool-dry, hot-dry, wet-monsoon)."),
    )
    return df


def table_lme_coefficients(out: Path, tab_dir: Path, manifest: list):
    """Table 3 — LME fixed-effect coefficients with significance."""
    df = _read_csv(out / "lme_coefficients.csv")
    if df is None:
        return
    df = df.copy()
    df["Predictor"] = df["predictor"].map(lambda p: PRED_LABEL.get(p, p))
    df["Sig."] = df["p_value"].map(_stars)
    df["95% CI"] = df.apply(lambda r: f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]", axis=1)

    def _fmt_p(p):
        p = float(p)
        return "<0.001" if p < 1e-3 else f"{p:.3f}"

    tidy = pd.DataFrame({
        "Predictor": df["Predictor"],
        "Coefficient": df["coef"].round(3),
        "Std. error": df["se"].round(3),
        "z": df["z"].round(2),
        "p-value": df["p_value"].map(_fmt_p),
        "95% CI": df["95% CI"],
        "Sig.": df["Sig."],
    })
    _save_table(
        tidy, tab_dir, "table3_lme_coefficients", manifest,
        caption=("Stage-1 linear mixed-effects fixed-effect estimates on standardised "
                 "predictors. Significance: $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$."),
        float_format="%.3f",
    )


def table_generalization(out: Path, tab_dir: Path, manifest: list):
    """Table 4 — in-sample vs CV R2 (overfitting / generalization gap)."""
    ins = _read_csv(out / "insample_metrics.csv")
    cv = _read_csv(out / "cv_overall.csv")
    if ins is None or cv is None:
        return
    cv_r = cv[cv["scheme"] == "random_10fold"].set_index("model")
    rows = []
    for _, r in ins.iterrows():
        m = r["model"]
        cv_r2 = cv_r.loc[m, "R2"] if m in cv_r.index else np.nan
        cv_rmse = cv_r.loc[m, "RMSE"] if m in cv_r.index else np.nan
        rows.append({
            "Model": _pretty_model(m),
            "Train R2": r["R2"],
            "CV R2": cv_r2,
            "Gap (Train-CV)": (r["R2"] - cv_r2) if pd.notna(cv_r2) else np.nan,
            "Train RMSE": r["RMSE"],
            "CV RMSE": cv_rmse,
        })
    tidy = pd.DataFrame(rows)
    tidy = tidy.sort_values("CV R2", ascending=False, na_position="last")
    _save_table(
        tidy, tab_dir, "table4_generalization_gap", manifest,
        caption=("In-sample (training) versus cross-validated performance. A large "
                 "Train$-$CV R\\textsuperscript{2} gap indicates over-fitting."),
    )


def table_vif(out: Path, tab_dir: Path, manifest: list):
    """Table 5 — within-station VIF (the reportable multicollinearity diagnostic)."""
    df = _read_csv(out / "vif_within_station.csv")
    if df is None:
        return
    df = df.copy()
    df["Predictor"] = df["predictor"].map(lambda p: PRED_LABEL.get(p, p))
    tidy = df[["Predictor", "VIF"]].copy()
    tidy["VIF"] = tidy["VIF"].round(2)
    _save_table(
        tidy, tab_dir, "table5_vif_within_station", manifest,
        caption=("Within-station variance inflation factors. Demeaning by the "
                 "per-station mean isolates the temporal variation that drives the "
                 "fixed-effect estimation; all values are well below the conventional "
                 "threshold of 10."),
        float_format="%.2f",
    )


def table_feature_importance(out: Path, tab_dir: Path, manifest: list):
    """Table 6 — tree-model feature importances (standalone benchmarks)."""
    df = _read_csv(out / "feature_importance.csv")
    if df is None:
        return
    df = df.copy()
    df["Predictor"] = df["predictor"].map(lambda p: PRED_LABEL.get(p, p))
    wide = df.pivot(index="Predictor", columns="model", values="importance")
    # Order rows by mean importance descending.
    wide = wide.reindex(wide.mean(axis=1).sort_values(ascending=False).index)
    wide = wide.round(4).reset_index()
    _save_table(
        wide, tab_dir, "table6_feature_importance", manifest,
        caption=("Gini / gain-based feature importances from the tree benchmarks "
                 "(normalised to sum to one within each model)."),
        float_format="%.4f",
    )


# =====================================================================
# FIGURES
# =====================================================================
def fig_model_comparison(cv_df: pd.DataFrame | None, fig_dir: Path, manifest: list):
    """Fig 1 — grouped bars of R2 and RMSE by model and CV scheme."""
    if cv_df is None:
        return
    schemes = [s for s in ["random_10fold", "site_based"] if s in cv_df["scheme"].unique()]
    models = [m for m in MODEL_ORDER if m in cv_df["model"].unique()]
    x = np.arange(len(models))
    width = 0.8 / max(len(schemes), 1)
    colours = [C_PRIMARY, C_SECOND, C_THIRD]

    fig, axes = plt.subplots(2, 1, figsize=(COL2, 6.4), sharex=True)
    for ax, metric, ylab in zip(axes, ["R2", "RMSE"],
                                ["R$^2$", r"RMSE ($\mu$g m$^{-3}$)"]):
        for j, sch in enumerate(schemes):
            sub = cv_df[cv_df["scheme"] == sch].set_index("model")
            vals = [sub.loc[m, metric] if m in sub.index else np.nan for m in models]
            ax.bar(x + (j - (len(schemes) - 1) / 2) * width, vals, width,
                   label=SCHEME_LABEL.get(sch, sch), color=colours[j % len(colours)],
                   edgecolor="white", linewidth=0.5)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        if metric == "R2":
            ax.set_ylim(0, 1.0)
            ax.legend(loc="upper left", ncol=len(schemes))
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([_pretty_model(m) for m in models],
                             rotation=35, ha="right")
    fig.suptitle("Cross-validated performance by model", y=0.995)
    _savefig(fig, fig_dir, "fig1_model_comparison", manifest,
             "Cross-validated R² (top) and RMSE (bottom) for every candidate "
             "model under random 10-fold and site-based hold-out.")


def _stream_predictions(path: Path, models: set[str] | None = None,
                        chunksize: int = 1_000_000) -> pd.DataFrame:
    """Read the (large) cv_predictions.csv in chunks.

    Pass ``models=None`` to read every model; pass a set to keep only those.
    Only the columns needed for the obs-vs-pred figures are loaded.
    """
    cols = ["model", "scheme", "y_true", "y_pred", "season"]
    parts = []
    for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize):
        parts.append(chunk if models is None else chunk[chunk["model"].isin(models)])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def _metrics(y_true, y_pred) -> dict:
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[m], y_pred[m]
    if len(yt) < 2:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "MB": np.nan, "n": len(yt)}
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return {
        "R2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "RMSE": float(np.sqrt(np.mean((yt - yp) ** 2))),
        "MAE": float(np.mean(np.abs(yt - yp))),
        "MB": float(np.mean(yp - yt)),
        "n": int(len(yt)),
    }


def _plot_obs_vs_pred_model(df_model: pd.DataFrame, model: str,
                            lim: tuple[float, float],
                            fig_dir: Path, manifest: list):
    """Render one model's observed-vs-predicted hexbin (one panel per CV scheme)."""
    schemes = [s for s in ["random_10fold", "site_based"]
               if s in df_model["scheme"].unique()]
    if not schemes:
        logger.warning("No CV-scheme rows for %s — skipping its obs-vs-pred.", model)
        return

    fig, axes = plt.subplots(1, len(schemes), figsize=(COL2, 3.7), squeeze=False)
    axes = axes[0]
    hb = None
    for ax, sch in zip(axes, schemes):
        sub = df_model[df_model["scheme"] == sch]
        yt = sub["y_true"].to_numpy(float)
        yp = sub["y_pred"].to_numpy(float)
        if len(yt):
            hb = ax.hexbin(yt, yp, gridsize=60, bins="log", extent=(*lim, *lim),
                           cmap="viridis", mincnt=1, linewidths=0.0)
        ax.plot(lim, lim, "--", color="k", linewidth=1, label="1:1")
        # Least-squares fit line.
        mask = ~(np.isnan(yt) | np.isnan(yp))
        if mask.sum() > 2:
            b, a = np.polyfit(yt[mask], yp[mask], 1)
            xs = np.array(lim)
            ax.plot(xs, b * xs + a, "-", color=C_SECOND, linewidth=1.2, label="fit")
        mt = _metrics(yt, yp)
        ax.set_title(SCHEME_LABEL.get(sch, sch))
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect("equal", "box")
        ax.set_xlabel(r"Observed PM$_{2.5}$ ($\mu$g m$^{-3}$)")
        ax.text(0.04, 0.96,
                f"R$^2$ = {mt['R2']:.3f}\nRMSE = {mt['RMSE']:.2f}\n"
                f"MAE = {mt['MAE']:.2f}\nN = {mt['n']:,}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85))
        ax.legend(loc="lower right")
    axes[0].set_ylabel(r"Predicted PM$_{2.5}$ ($\mu$g m$^{-3}$)")
    if hb is not None:
        cb = fig.colorbar(hb, ax=axes, fraction=0.046, pad=0.02)
        cb.set_label("count (log)")
    fig.suptitle(f"Observed vs predicted — {_pretty_model(model)}", y=1.02)
    safe = model.replace("/", "-").replace(" ", "_")
    _savefig(fig, fig_dir, f"fig2_obs_vs_pred_{safe}", manifest,
             f"Observed versus predicted daily PM₂.₅ for {_pretty_model(model)}; "
             "colour shows point density (log scale). Dashed line is 1:1, solid "
             "line is the least-squares fit. Axes share a common scale across all "
             "models for direct comparison.")


def fig_obs_vs_pred_all(out: Path, fig_dir: Path, manifest: list):
    """Fig 2 — observed vs predicted hexbin, one figure per model.

    The 121 MB cv_predictions.csv is read once; a single axis scale is shared
    across all models so the panels are directly comparable.
    """
    path = out / "cv_predictions.csv"
    if not path.exists():
        logger.warning("Missing cv_predictions.csv — skipping obs-vs-pred figures.")
        return
    df = _stream_predictions(path, models=None)   # read once, every model
    if df.empty:
        logger.warning("cv_predictions.csv has no rows — skipping obs-vs-pred.")
        return

    # Shared axis limit (99.5th percentile of all obs+pred) for comparability.
    vmax = float(np.nanpercentile(
        np.concatenate([df["y_true"].values, df["y_pred"].values]), 99.5))
    lim = (0, max(vmax, 1))

    present = [m for m in MODEL_ORDER if m in df["model"].unique()]
    present += [m for m in df["model"].unique() if m not in MODEL_ORDER]
    logger.info("Obs-vs-pred: generating %d model figures: %s", len(present), present)
    for model in present:
        _plot_obs_vs_pred_model(df[df["model"] == model], model, lim, fig_dir, manifest)


def fig_seasonal(out: Path, fig_dir: Path, manifest: list):
    """Fig 3 — seasonal R2 by model (site-based scheme)."""
    df = _read_csv(out / "cv_stratified.csv")
    if df is None:
        return
    df = df[df["stratum"] == "season"].copy()
    if df.empty:
        return
    scheme = "site_based" if "site_based" in df["scheme"].unique() else df["scheme"].iloc[0]
    df = df[df["scheme"] == scheme]
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    seasons = [s for s in SEASON_ORDER if s in df["value"].unique()]
    x = np.arange(len(models))
    width = 0.8 / max(len(seasons), 1)
    colours = [C_PRIMARY, C_SECOND, C_THIRD]

    fig, ax = plt.subplots(figsize=(COL2, 3.8))
    for j, s in enumerate(seasons):
        sub = df[df["value"] == s].set_index("model")
        vals = [sub.loc[m, "R2"] if m in sub.index else np.nan for m in models]
        ax.bar(x + (j - (len(seasons) - 1) / 2) * width, vals, width,
               label=SEASON_LABEL.get(s, s), color=colours[j % len(colours)],
               edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_model(m) for m in models], rotation=35, ha="right")
    ax.set_ylabel("R$^2$")
    ax.set_title(f"Season-stratified R$^2$ ({SCHEME_LABEL.get(scheme, scheme)})")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(ncol=len(seasons), loc="lower left")
    _savefig(fig, fig_dir, "fig3_seasonal_r2", manifest,
             "Season-stratified cross-validated R² per model under the "
             "site-based scheme.")


def fig_feature_importance(out: Path, fig_dir: Path, manifest: list):
    """Fig 4 — horizontal bar of feature importance (RF vs XGBoost)."""
    df = _read_csv(out / "feature_importance.csv")
    if df is None:
        return
    df = df.copy()
    df["Predictor"] = df["predictor"].map(lambda p: PRED_LABEL.get(p, p))
    wide = df.pivot(index="Predictor", columns="model", values="importance")
    wide = wide.reindex(wide.mean(axis=1).sort_values().index)  # ascending for barh
    models = list(wide.columns)
    y = np.arange(len(wide))
    height = 0.8 / max(len(models), 1)
    colours = [C_PRIMARY, C_SECOND, C_THIRD]

    fig, ax = plt.subplots(figsize=(COL1 * 1.7, 4.4))
    for j, m in enumerate(models):
        ax.barh(y + (j - (len(models) - 1) / 2) * height, wide[m].values, height,
                label=m, color=colours[j % len(colours)], edgecolor="white", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(wide.index)
    ax.set_xlabel("Relative importance")
    ax.set_title("Predictor importance")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    _savefig(fig, fig_dir, "fig4_feature_importance", manifest,
             "Relative predictor importance from the Random Forest and XGBoost "
             "benchmarks.")


def fig_lme_forest(out: Path, fig_dir: Path, manifest: list):
    """Fig 5 — forest plot of LME fixed-effect coefficients with 95% CI."""
    df = _read_csv(out / "lme_coefficients.csv")
    if df is None:
        return
    df = df[df["predictor"].str.lower() != "intercept"].copy()
    df["Predictor"] = df["predictor"].map(lambda p: PRED_LABEL.get(p, p))
    # Order by coefficient magnitude for readability.
    df = df.reindex(df["coef"].abs().sort_values().index).reset_index(drop=True)
    df["y"] = np.arange(len(df))
    df["sig"] = df["p_value"].astype(float) < 0.05

    fig, ax = plt.subplots(figsize=(COL1 * 1.7, 4.6))
    # errorbar's `ecolor` takes a single colour, so plot significant and
    # non-significant predictors as two separate groups.
    for is_sig, colour in [(True, C_PRIMARY), (False, C_GREY)]:
        sub = df[df["sig"] == is_sig]
        if sub.empty:
            continue
        ax.errorbar(sub["coef"], sub["y"],
                    xerr=[sub["coef"] - sub["ci_lower"], sub["ci_upper"] - sub["coef"]],
                    fmt="o", color=colour, ecolor=colour, elinewidth=1.4,
                    capsize=3, markersize=5, zorder=3)
    ax.axvline(0, color="k", linewidth=0.8, linestyle="--")
    y = df["y"].to_numpy()
    ax.set_yticks(y)
    ax.set_yticklabels(df["Predictor"])
    ax.set_xlabel("Standardised coefficient (95% CI)")
    ax.set_title("LME fixed effects")
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    # Legend proxy for significance colouring.
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_PRIMARY,
               markersize=7, label="p < 0.05"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_GREY,
               markersize=7, label="n.s."),
    ], loc="lower right")
    _savefig(fig, fig_dir, "fig5_lme_forest", manifest,
             "Forest plot of Stage-1 LME fixed-effect coefficients (standardised "
             "predictors) with 95% confidence intervals; grey markers are not "
             "significant at α = 0.05.")


def fig_fold_box(out: Path, fig_dir: Path, manifest: list):
    """Fig 6 — per-fold R2 distribution (random scheme) as boxplots."""
    df = _read_csv(out / "cv_fold_metrics.csv")
    if df is None:
        return
    scheme = "random_10fold" if "random_10fold" in df["scheme"].unique() else df["scheme"].iloc[0]
    df = df[df["scheme"] == scheme]
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    data = [df[df["model"] == m]["R2"].dropna().values for m in models]

    fig, ax = plt.subplots(figsize=(COL2, 3.8))
    bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.6,
                    medianprops=dict(color="k", linewidth=1.2),
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))
    for patch in bp["boxes"]:
        patch.set_facecolor(C_PRIMARY)
        patch.set_alpha(0.6)
    ax.set_xticklabels([_pretty_model(m) for m in models], rotation=35, ha="right")
    ax.set_ylabel("Per-fold R$^2$")
    ax.set_title(f"Fold-level R$^2$ spread ({SCHEME_LABEL.get(scheme, scheme)})")
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    _savefig(fig, fig_dir, "fig6_fold_r2_box", manifest,
             "Distribution of per-fold R² across the cross-validation folds; "
             "boxes span the inter-quartile range, whiskers the 1.5×IQR.")


def fig_gwr_timeseries(out: Path, fig_dir: Path, manifest: list):
    """Fig 7 — GWR Stage-2 daily diagnostics (if any days were fitted)."""
    # Prefer the hybrid tree-GWR stats with the most fitted days.
    candidates = sorted(out.glob("gwr_daily_stats_*.csv"))
    if not candidates:
        return
    best_df, best_label, best_fit = None, None, -1
    for c in candidates:
        d = pd.read_csv(c, parse_dates=["date"])
        n_fit = int(d["gwr_fitted"].sum()) if "gwr_fitted" in d else 0
        if n_fit > best_fit:
            best_df, best_label, best_fit = d, c.stem.replace("gwr_daily_stats_", ""), n_fit
    if best_df is None or best_fit == 0:
        logger.info("GWR Stage-2 fitted 0 days in all variants — skipping fig7 "
                    "(GWR correction inactive on this run).")
        return

    d = best_df.sort_values("date")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL2, 4.6), sharex=True)
    fit = d[d["gwr_fitted"] == True]  # noqa: E712
    a1.plot(fit["date"], fit["bandwidth"], ".", color=C_PRIMARY, markersize=3)
    a1.set_ylabel("Bandwidth\n(neighbours)")
    a1.set_title(f"GWR Stage-2 daily diagnostics ({best_label})")
    a2.plot(d["date"], d["morans_p"], ".", color=C_GREY, markersize=2, alpha=0.5)
    a2.axhline(0.05, color=C_SECOND, linestyle="--", linewidth=1, label="α = 0.05")
    a2.set_ylabel("Moran's I\np-value")
    a2.set_yscale("log")
    a2.legend(loc="upper right")
    a2.set_xlabel("Date")
    _savefig(fig, fig_dir, "fig7_gwr_diagnostics", manifest,
             "Daily Stage-2 GWR diagnostics: adaptive bandwidth on fitted days (top) "
             "and the Moran's I residual-autocorrelation p-value (bottom).")


# =====================================================================
# Moran's I — spatial autocorrelation of stage-one residuals
# =====================================================================
# Map each gwr_daily_stats_<key>.csv to its Stage-1 estimator label.
_STAGE1_LABEL = {"hybrid": "LME", "hybrid_rf": "RF",
                 "hybrid_gb": "GB", "hybrid_xgb": "XGBoost"}
_STAGE1_ORDER = ["LME", "RF", "GB", "XGBoost"]
_MORANS_ALPHA = 0.05


def _season_from_month(m: int) -> str:
    """Thai meteorological season from calendar month (matches the manuscript)."""
    if m in (11, 12, 1, 2):
        return "cool_dry"
    if m in (3, 4, 5):
        return "hot_dry"
    return "wet_monsoon"


def _load_morans(out: Path) -> pd.DataFrame | None:
    """Combine every gwr_daily_stats_*.csv into one long Moran's-I frame.

    Columns: stage1 (LME/RF/GB/XGBoost), date, morans_p, gwr_fitted, season.
    Days with too few stations have a missing morans_p and are kept as NaN so
    they can be excluded from the 'days assessed' denominator.
    """
    frames = []
    for path in sorted(out.glob("gwr_daily_stats_*.csv")):
        key = path.stem.replace("gwr_daily_stats_", "")
        d = pd.read_csv(path, parse_dates=["date"])
        if "morans_p" not in d.columns:
            continue
        d = d[["date", "morans_p", "gwr_fitted"]].copy()
        d["gwr_fitted"] = d["gwr_fitted"].astype(str).str.lower().isin(["true", "1"])
        d["stage1"] = _STAGE1_LABEL.get(key, key)
        frames.append(d)
    if not frames:
        logger.warning("No gwr_daily_stats_*.csv found — skipping Moran's I assets.")
        return None
    df = pd.concat(frames, ignore_index=True)
    df["season"] = df["date"].dt.month.map(_season_from_month)
    return df


def _stage1_present(df: pd.DataFrame) -> list[str]:
    present = [s for s in _STAGE1_ORDER if s in df["stage1"].unique()]
    present += [s for s in df["stage1"].unique() if s not in _STAGE1_ORDER]
    return present


def table_morans_summary(out: Path, tab_dir: Path, manifest: list):
    """Table — per Stage-1 model: how often residuals are spatially autocorrelated."""
    df = _load_morans(out)
    if df is None:
        return
    rows = []
    for s in _stage1_present(df):
        sub = df[df["stage1"] == s]
        assessed = sub["morans_p"].notna()
        n_assessed = int(assessed.sum())
        n_sig = int((sub["morans_p"] <= _MORANS_ALPHA).sum())  # NaN compares False
        n_fit = int(sub["gwr_fitted"].sum())
        med_p = float(sub.loc[assessed, "morans_p"].median()) if n_assessed else np.nan
        rows.append({
            "Stage-1 model": s,
            "Days assessed": n_assessed,
            "Significant days": n_sig,
            "% significant": 100 * n_sig / n_assessed if n_assessed else np.nan,
            "Days GWR-applied": n_fit,
            "Median p": med_p,
        })
    tidy = pd.DataFrame(rows)
    _save_table(
        tidy, tab_dir, "table7_morans_summary", manifest,
        caption=("Spatial autocorrelation of Stage-1 residuals (Moran's I). "
                 "``Days assessed'' counts days with enough stations for a valid "
                 "test; ``Significant days'' are those with $p \\leq 0.05$, on which "
                 "the Stage-2 GWR correction is warranted; ``Days GWR-applied'' are "
                 "those that additionally passed the stability checks and were "
                 "actually corrected."),
        float_format="%.3f",
    )


def fig_morans(out: Path, fig_dir: Path, manifest: list):
    """Fig — Moran's I distribution (ECDF) and seasonal share of significant days."""
    df = _load_morans(out)
    if df is None:
        return
    present = _stage1_present(df)
    palette = [C_PRIMARY, C_SECOND, C_THIRD, C_GREY]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(COL2, 3.8))

    # (a) ECDF of Moran's p per Stage-1 model.
    for i, s in enumerate(present):
        p = np.sort(df.loc[df["stage1"] == s, "morans_p"].dropna().to_numpy())
        if p.size == 0:
            continue
        ecdf = np.arange(1, p.size + 1) / p.size
        frac_sig = float((p <= _MORANS_ALPHA).mean())
        axL.plot(p, ecdf, label=f"{s} ({100*frac_sig:.0f}%)",
                 color=palette[i % len(palette)], linewidth=1.6)
    axL.axvline(_MORANS_ALPHA, color="k", linestyle="--", linewidth=1)
    axL.set_xlim(0, 1)
    axL.set_ylim(0, 1)
    axL.set_xlabel("Moran's I $p$-value")
    axL.set_ylabel("Cumulative fraction of days")
    axL.set_title("(a) Residual Moran's $p$ distribution")
    axL.legend(title="(% with $p\\leq0.05$)", loc="lower right")
    axL.grid(alpha=0.3, linewidth=0.5)

    # (b) % of days with significant autocorrelation, by season.
    seasons = [s for s in SEASON_ORDER if s in df["season"].unique()]
    x = np.arange(len(seasons))
    width = 0.8 / max(len(present), 1)
    for i, s in enumerate(present):
        sub = df[df["stage1"] == s]
        fracs = []
        for sea in seasons:
            d = sub[(sub["season"] == sea) & sub["morans_p"].notna()]
            fracs.append(100 * (d["morans_p"] <= _MORANS_ALPHA).mean()
                         if len(d) else np.nan)
        axR.bar(x + (i - (len(present) - 1) / 2) * width, fracs, width,
                label=s, color=palette[i % len(palette)],
                edgecolor="white", linewidth=0.4)
    axR.set_xticks(x)
    axR.set_xticklabels([SEASON_LABEL.get(s, s) for s in seasons])
    axR.set_ylim(0, 120)
    axR.set_ylabel("Days significant (%)")
    axR.set_title("(b) Significant days by season")
    axR.legend(loc="upper right", ncol=2)
    axR.grid(axis="y", alpha=0.3, linewidth=0.5)

    _savefig(fig, fig_dir, "fig8_morans", manifest,
             "Spatial autocorrelation of Stage-1 residuals. (a) Empirical "
             "cumulative distribution of the daily Moran's I $p$-value for each "
             "Stage-1 estimator (legend gives the share of days with $p\\leq0.05$); "
             "the dashed line marks $\\alpha=0.05$. (b) Percentage of days with "
             "significant residual spatial autocorrelation, stratified by season.")


# =====================================================================
# Manifest
# =====================================================================
def write_manifest(report_dir: Path, manifest: list):
    lines = ["# Manuscript report assets\n",
             "Auto-generated by `scripts/make_report.py`. "
             "Each entry lists the relative path and a draft caption.\n"]
    figs = [m for m in manifest if m[0] == "figure"]
    tabs = [m for m in manifest if m[0] == "table"]
    if tabs:
        lines.append("\n## Tables\n")
        for i, (_, path, cap) in enumerate(tabs, 1):
            lines.append(f"- **Table {i}** — `{path}`\n  - {cap}\n")
    if figs:
        lines.append("\n## Figures\n")
        for i, (_, path, cap) in enumerate(figs, 1):
            lines.append(f"- **Figure {i}** — `{path}`\n  - {cap}\n")
    (report_dir / "REPORT_MANIFEST.md").write_text("".join(lines), encoding="utf-8")
    logger.info("Wrote manifest -> REPORT_MANIFEST.md (%d tables, %d figures)",
                len(tabs), len(figs))


# =====================================================================
def main():
    ap = argparse.ArgumentParser(description="Build manuscript tables & figures "
                                             "from pipeline CSV outputs.")
    ap.add_argument("--config", default="configs/default.yaml",
                    help="YAML config (used only to locate output_dir).")
    ap.add_argument("--outputs-dir", default=None,
                    help="Override the pipeline output dir (where the CSVs live).")
    ap.add_argument("--report-dir", default=None,
                    help="Where to write report/ assets (default: <outputs>/report).")
    args = ap.parse_args()

    if args.outputs_dir:
        out = Path(args.outputs_dir)
    else:
        cfg = _load_config(args.config)
        out = Path(cfg["paths"]["output_dir"])
    out = out.resolve()
    if not out.exists():
        raise FileNotFoundError(f"Output dir not found: {out} — run main.py first.")

    report_dir = Path(args.report_dir).resolve() if args.report_dir else out / "report"
    fig_dir = report_dir / "figures"
    tab_dir = report_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Reading CSVs from: %s", out)
    logger.info("Writing report to: %s", report_dir)

    manifest: list = []

    # --- Tables (do these first; fig1 reuses the cv frame) ---
    tasks = [
        ("Table 1 CV performance", lambda: table_cv_performance(out, tab_dir, manifest)),
        ("Table 2 seasonal", lambda: table_seasonal(out, tab_dir, manifest)),
        ("Table 3 LME coefficients", lambda: table_lme_coefficients(out, tab_dir, manifest)),
        ("Table 4 generalization", lambda: table_generalization(out, tab_dir, manifest)),
        ("Table 5 VIF", lambda: table_vif(out, tab_dir, manifest)),
        ("Table 6 feature importance", lambda: table_feature_importance(out, tab_dir, manifest)),
        ("Table 7 Moran's I summary", lambda: table_morans_summary(out, tab_dir, manifest)),
    ]
    cv_df = None
    for name, fn in tasks:
        try:
            res = fn()
            if name.startswith("Table 1"):
                cv_df = res
        except Exception as e:  # noqa: BLE001
            logger.exception("Failed: %s (%s)", name, e)

    # --- Figures ---
    fig_tasks = [
        ("Fig 1 model comparison", lambda: fig_model_comparison(cv_df, fig_dir, manifest)),
        ("Fig 2 obs-vs-pred (all models)", lambda: fig_obs_vs_pred_all(out, fig_dir, manifest)),
        ("Fig 3 seasonal", lambda: fig_seasonal(out, fig_dir, manifest)),
        ("Fig 4 feature importance", lambda: fig_feature_importance(out, fig_dir, manifest)),
        ("Fig 5 LME forest", lambda: fig_lme_forest(out, fig_dir, manifest)),
        ("Fig 6 fold boxplot", lambda: fig_fold_box(out, fig_dir, manifest)),
        ("Fig 7 GWR diagnostics", lambda: fig_gwr_timeseries(out, fig_dir, manifest)),
        ("Fig 8 Moran's I", lambda: fig_morans(out, fig_dir, manifest)),
    ]
    for name, fn in fig_tasks:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            logger.exception("Failed: %s (%s)", name, e)

    write_manifest(report_dir, manifest)
    logger.info("Done. %d assets generated.", len(manifest))


if __name__ == "__main__":
    main()
