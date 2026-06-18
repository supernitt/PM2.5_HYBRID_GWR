# PM2.5 Hybrid Model Estimation
# Powered by Capybara Geo Lab

Python pipeline for satellite-based daily PM2.5 estimation over Thailand (2023–2025)
using two families of two-stage hybrid models:

- **LME-GWR** — Stage-1 Linear Mixed-Effects model + Stage-2 GWR on LME residuals
- **Tree-GWR** — Stage-1 tree-based model (Random Forest, Gradient Boosting, or XGBoost)
  + Stage-2 GWR on tree residuals *(recommended — best accuracy)*

Both share the same GWR correction layer; they differ only in how the global PM2.5
estimate is produced in Stage 1.  Benchmark regressors (PCR, PLSR, RF, GB, XGBoost)
without GWR correction are also available for comparison.

## Project structure

```
pm25_hybrid/
├── main.py                      # Pipeline entry point
├── configs/
│   └── default.yaml             # All settings (paths, predictors, model params)
├── scripts/
│   ├── prepare_pm25_pcd.py      # Reshape PCD Excel → long-format CSV
│   ├── gee_extract_stations.py  # GEE: sample predictors at PCD station points
│   ├── gee_export_grids.py      # GEE: export country-wide daily predictor rasters
│   └── merge_gee_exports.py     # Merge per-predictor CSVs → panel.csv
├── src/
│   ├── data_loader.py           # Read station-day panel (CSV/Parquet)
│   ├── preprocessing.py         # Clean, filter, standardise predictors
│   ├── lme_model.py             # Stage-1 LME (statsmodels MixedLM)
│   ├── gwr_model.py             # Stage-2 GWR on LME residuals (mgwr)
│   ├── hybrid_model.py          # HybridLMEGWR (LME+GWR) and HybridTreeGWR (tree+GWR)
│   ├── benchmarks.py            # PCR, PLSR, RF, GB, XGBoost
│   ├── validation.py            # 10-fold random + site-based CV
│   ├── mapping.py               # CSV-based grid prediction (small domains)
│   ├── raster_prediction.py     # Raster-based prediction (country-scale, recommended)
│   ├── metrics.py               # R², RMSE, MAE, MB
│   └── utils.py                 # Logging, seeds, IO helpers
├── data/                        # Input data (not versioned)
│   ├── panel.csv                # Station-day training panel (built by merge_gee_exports.py)
│   └── grids/                   # Daily predictor rasters (built by gee_export_grids.py)
├── outputs/                     # All generated results
└── requirements.txt
```

---

## Step-by-step workflow

### Step 0 — Environment

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.10. Key packages: `statsmodels`, `mgwr`, `scikit-learn`,
`rasterio`, `scipy`, `geemap`, `earthengine-api`, `xgboost`.

---

### Step 1 — Prepare PCD PM2.5 measurements

Convert the annual PCD Excel files (wide format, one column per station) into
the long-format CSV expected by the pipeline:

```bash
python scripts/prepare_pm25_pcd.py \
    --pm-dir   /path/to/PM_station \
    --stations /path/to/Air4Thai_station.csv \
    --out      data/pcd_pm25_daily.csv
```

**Output:** `data/pcd_pm25_daily.csv` — columns: `station_id, date, pm25, lat, lon`.

**This step must be run once before Step 2.** It only needs to be re-run if new
PCD data (e.g. a new year) are added.

---

### Step 2 — Extract predictor data at station locations (GEE)

Samples all predictor collections at the PCD monitoring stations for
2023-01-01 → 2025-12-31 and downloads one CSV per predictor:

```bash
python scripts/gee_extract_stations.py
```

Requires an authenticated Earth Engine account (`earthengine authenticate`).
The script uses chunked monthly exports with retries. Outputs land in
`data/gee_exports/` (one CSV per predictor, e.g. `pcd_aod_maiac.csv`,
`pcd_t2m.csv`, …).

**This step must be run once.** It can be safely interrupted and re-run; each
monthly chunk is skipped if it already exists.

---

### Step 3 — Build the station-day training panel

Merge the per-predictor CSVs with the PCD PM2.5 file, derive RH and WS,
convert T2m from Kelvin to Celsius, and linearly interpolate NDVI to daily:

```bash
python scripts/merge_gee_exports.py \
    --gee-dir  data/gee_exports \
    --pm25-file data/pcd_pm25_daily.csv \
    --out       data/panel.csv
```

**Output:** `data/panel.csv` — one row per station-day with all 12 predictors
plus PM2.5. This is the training dataset.

**This step must be run once (after Steps 1 and 2).** Re-run only if the raw
GEE CSVs or the PCD data change.

> **Important — unit conventions in panel.csv:**
> `t2m` is stored in **Celsius** (converted from ERA5-Land Kelvin).
> All other predictors are in their native GEE units:
> `prec` (m/day), `ssr` (J/m²), `sp` (Pa), `rh` (%), `ws` (m/s), `blh` (m),
> `frp` (MW), `ndvi` (–), `elev` (m), `pop` (persons/km²).

---

### Step 4 — Download country-wide predictor rasters (GEE)

Exports one GeoTIFF per predictor per day to `data/grids/` for country-wide
PM2.5 mapping. Study period: 2023-01-01 → 2025-12-31 (~10,950 files total).

```bash
python scripts/gee_export_grids.py
```

Expected folder layout after download:

```
data/grids/
├── aod/    aod_2023-01-01.tif  aod_2023-01-02.tif  ...   # clear-sky days only
├── t2m/    t2m_2023-01-01.tif  ...                        # daily (Celsius)
├── rh/     rh_2023-01-01.tif   ...
├── ws/     ws_2023-01-01.tif   ...
├── blh/    blh_2023-01-01.tif  ...
├── prec/   prec_2023-01-01.tif ...
├── ssr/    ssr_2023-01-01.tif  ...
├── sp/     sp_2023-01-01.tif   ...
├── frp/    frp_2023-01-01.tif  ...
├── ndvi/   ndvi_2023-01-01.tif  ndvi_2023-01-17.tif  ...  # 16-day composites only
├── elev.tif   (static — SRTM)
└── pop.tif    (static — GPWv4 2020)
```

**Notes:**
- AOD only has files on clear-sky days (cloud-masked). Expect ~1,100–1,250 files
  out of 1,096 calendar days (MAIAC sometimes provides Terra + Aqua coverage
  on the same day, raising the count above the number of calendar days).
- NDVI has ~78 files only (16-day MOD13A2 composites). The pipeline
  **forward-fills** NDVI between composites automatically — do not worry
  about the sparse file count.
- The script is idempotent (`SKIP_EXISTING = True`); re-run to resume after
  interruption or to fill any failed downloads.

> **Unit note:** `t2m` rasters are exported in **Celsius** by this script
> (consistent with `panel.csv`). If you have older t2m rasters downloaded in
> Kelvin, the pipeline automatically detects and corrects them at read time
> (median > 200 → subtract 273.15).

---

### Step 5 — Train and validate the model

```bash
python main.py --config configs/default.yaml --stages preprocess,train,validate
```

This runs:
1. **preprocess** — clean, standardise, apply Savitzky-Golay smoothing to predictors.
2. **train** — fit Stage-1 LME + Stage-2 GWR (LME-GWR) and Stage-1 tree models
   (RF, GB, XGBoost) + Stage-2 GWR (Tree-GWR); fit all benchmark models.
3. **validate** — 10-fold random CV + 10-fold site-based CV for all models.

Outputs written to `outputs/`:
- `metrics_cv_random.csv`, `metrics_cv_site.csv` — CV performance per model
- `metrics_train.csv` — in-sample performance

**Training must complete before `map_raster`** (the hybrid model object and
the StandardScaler are serialised and reloaded by the mapping stage).

**Re-run this step if:**
- The training panel (`panel.csv`) changes.
- Config parameters (predictors, LME structure, GWR settings) change.
- You want fresh benchmark hyperparameter tuning results.

---

### Step 6 — Generate country-wide PM2.5 prediction maps

```bash
python main.py --config configs/default.yaml --stages map_raster
```

What this stage does:

| Sub-task | Output |
|---|---|
| Daily prediction | `outputs/pm25_predictions/pm25_YYYY-MM-DD.tif` — one file per clear-sky day |
| Monthly means | `outputs/pm25_predictions/pm25_monthly_YYYY-MM.tif` |
| Seasonal means | `outputs/pm25_predictions/pm25_seasonal_YYYY_<season>.tif` (`cool_dry`, `hot_dry`, `wet_monsoon`) |
| Annual means | `outputs/pm25_predictions/pm25_annual_YYYY.tif` |
| Exceedance maps | `outputs/pm25_predictions/pm25_exceedance_who_24hr.tif` (fraction of days > 15 μg/m³) |
| | `outputs/pm25_predictions/pm25_exceedance_thai_24hr.tif` (fraction of days > 37.5 μg/m³) |
| Spatial statistics | `outputs/pm25_predictions/pm25_spatial_stats.csv` — mean, std, percentiles, pop-weighted mean per raster |
| Day log | `outputs/raster_prediction_log.csv` — per-day status and GWR usage flag |

**Boundary masking:** Output pixels outside Thailand's national boundary are
set to NaN.  The mask is built in two steps: (1) Thailand's polygon is
rasterized from the Natural Earth countries dataset (via `geopandas`);
(2) the polygon mask is intersected with the SRTM elevation mask so coastal
sea pixels inside Thailand's convex hull are also excluded.  Neighbouring
countries (Myanmar, Laos, Cambodia, Malaysia) are masked out.

**Gap-filling:** Cloud-masked NaN pixels within the Thailand land mask are
filled using **Inverse Distance Weighting (IDW, power=2, k=10 neighbours)**
— the standard spatial gap-filling method in regional PM2.5 remote sensing
(He et al. 2019, *Atmos. Environ.*; Wei et al. 2021, *STOTEN*).

**Physical bounds:** PM2.5 predictions are clipped to [0, 500] μg/m³ after
model application and before gap-filling to suppress GWR extrapolation
artefacts.

**Skip-existing:** By default `skip_existing: true` in the config — already
written daily TIFs are not reprocessed. Set to `false` (or delete the outputs)
if you need to regenerate everything, e.g. after a code fix.

> **If you need to regenerate all outputs** (e.g. after a bug fix), SSH into
> the server and run:
> ```bash
> rm /mnt/DataHub/Projects/2026_luktarn/pm25_hybrid/outputs/pm25_predictions/*.tif
> rm /mnt/DataHub/Projects/2026_luktarn/pm25_hybrid/outputs/pm25_predictions/*.csv
> python main.py --config configs/default.yaml --stages map_raster
> ```

---

## Running individual stages

```bash
# Full pipeline
python main.py --config configs/default.yaml

# Data prep only
python main.py --config configs/default.yaml --stages preprocess

# Train + validate (no mapping)
python main.py --config configs/default.yaml --stages preprocess,train,validate

# Map only (model must already be trained)
python main.py --config configs/default.yaml --stages map_raster
```

Available stage names: `preprocess`, `train`, `validate`, `map`, `map_raster`.

---

## Input data format

The `panel.csv` training file must have one row per station-day with:

| Column | Description | Units |
|--------|-------------|-------|
| `station_id` | Unique station ID (string) | — |
| `date` | Date (YYYY-MM-DD) | — |
| `lat`, `lon` | Station coordinates | decimal degrees |
| `pm25` | Daily mean PM2.5 (target variable) | μg/m³ |
| `aod` | MAIAC AOD at 470 nm (MCD19A2, QA-masked) | unitless |
| `t2m` | 2 m air temperature | **°C** (Celsius) |
| `rh` | 2 m relative humidity | % |
| `ws` | 10 m wind speed | m/s |
| `blh` | Planetary boundary layer height | m |
| `prec` | Total precipitation (daily sum) | m/day |
| `ssr` | Surface solar radiation downwards (daily sum) | J/m² |
| `sp` | Surface pressure | Pa |
| `frp` | VIIRS fire radiative power (daily sum, 1 km buffer) | MW |
| `ndvi` | MOD13A2 NDVI, linearly interpolated to daily | unitless |
| `elev` | SRTM elevation | m |
| `pop` | GPWv4 population density (2020 epoch) | persons/km² |

---

## Key configuration options

Edit `configs/default.yaml` to change behaviour:

```yaml
raster_mapping:
  grid_dir: "./data/grids"           # root of daily predictor GeoTIFFs
  output_dir: "./outputs/pm25_predictions"
  model_kind: "hybrid_xgb"          # recommended (see table below)
  k_neighbors_idw: 8                 # neighbours for GWR coefficient interpolation
  skip_existing: true                # set false to force regeneration

  aggregate:
    min_valid_days_per_month: 7      # min clear-sky days for a valid monthly mean pixel
    min_valid_days_per_year: 180     # min clear-sky days for a valid annual mean pixel
    save_monthly: true
    save_seasonal: true
```

**Model choice for `model_kind`:**

| Value | Description | R² random CV | R² site CV | Notes |
|-------|-------------|:---:|:---:|-------|
| `hybrid_xgb` | XGBoost + GWR correction | 0.874 | 0.723 | **Recommended** — best accuracy, immune to multicollinearity |
| `hybrid_rf` | Random Forest + GWR | 0.844 | 0.727 | Robust alternative |
| `hybrid_gb` | Gradient Boosting + GWR | 0.836 | 0.724 | Similar to RF |
| `hybrid` | LME + GWR | 0.707 | 0.593 | Not recommended for raster — LME has VIF up to 6.5M among meteorological predictors, causing extreme predictions outside training station locations |
| `lme` | LME fixed-effects only | 0.707 | 0.593 | Same issue as `hybrid` |
| `rf`, `gb`, `xgboost` | Standalone benchmarks (no GWR) | varies | varies | Available for comparison |

---

## Citation

If you use this code, please cite the manuscript:
Please wait for publications ^^.