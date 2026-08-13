# PM2.5_HYBRID_GWR

A single daily GWR residual correction, applied only when a Moran's I gate
detects spatial structure in the stage-one residuals, is held fixed while the
stage-one estimator varies (LME, RF, GB, XGBoost), evaluated under random
10-fold and site-based (leave-stations-out) cross-validation on observed-only
and gap-filled MAIAC AOD panels.

## Layout

```
main.py            pipeline driver (stages: preprocess, train, validate, map_raster)
src/               pipeline modules (preprocessing, LME, GWR, hybrids, benchmarks, CV, rasters)
configs/           default.yaml (observed arm, main analysis)
                   aod_imputed.yaml (gap-filled sensitivity arm)
                   gate_ablation.yaml (Moran gate held open, alpha = 1)
scripts/           post-run analyses and figures
data/panel.csv     station-day panel (98,695 records, 101 PCD stations, 2023-01-01..2025-12-31)
```

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Main analysis (observed-AOD arm): train + both CV schemes  (~3 h, 24 cores)
python main.py --config configs/default.yaml --stages train,validate

# Sensitivity arm (gap-filled AOD)                            (~5 h)
python main.py --config configs/aod_imputed.yaml --stages train,validate

# Gate ablation (always-on correction; copy tuned params first)
mkdir -p outputs_gate_ablation && cp outputs/tuned_params.json outputs_gate_ablation/
python main.py --config configs/gate_ablation.yaml --stages preprocess,validate \
    --algorithms hybrid,hybrid_rf,hybrid_gb,hybrid_xgb

# Statistical analyses (paired tests, rank bootstrap, dose-response) and figures
python scripts/block_c_analysis.py
python scripts/full_panel_stats.py
python scripts/make_figures.py
```

Hardware: 64 GB RAM minimum (128 GB recommended); `compute.n_jobs` in the
configs is set for a 24-core budget.

## Data

`data/panel.csv` is the co-located station-day panel used for all model
fitting and validation: daily PCD PM2.5 paired with MAIAC AOD (MCD19A2 C6.1),
ERA5/ERA5-Land meteorology (t2m, rh, ws, blh, prec, ssr, sp), VNP14A1 FRP,
MOD13A2 NDVI, SRTM elevation, and GPWv4.11 population density, extracted via
Google Earth Engine. All upstream products are publicly available; sources,
GEE asset IDs, and resolutions are listed in Table 1 of the manuscript.
Physically impossible values (e.g. -9999 sentinels) are masked by the
pipeline's valid-range guard at load time, not in this file.

**Not included** (size / format): the ~0.5 GB of daily gridded predictor
GeoTIFFs required by the country-wide mapping stage (`--stages map_raster`) —
regenerate them from the same GEE assets — and two small boundary GeoJSONs
used only by the figure scripts (Thailand outline and provinces; derive from
Natural Earth 110m/10m admin layers).

## Reproducibility notes

- CV folds are seeded (seed 42); site-based folds hold out whole stations.
- Tuned hyperparameters are persisted to `outputs/tuned_params.json` and
  reused so standalone and hybrid models share identical stage-one settings.
- The Moran gate (k = 5 row-standardized KNN, alpha = 0.05) and its always-on
  ablation (alpha = 1) differ only in `gwr.morans_alpha`.
