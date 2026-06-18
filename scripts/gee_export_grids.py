"""
Export daily predictor rasters over Thailand (geemap)

Downloads one GeoTIFF per predictor per day to data/grids/ for use
in the raster-based country-wide PM2.5 prediction stage.

Study period: 1 January 2023 – 31 December 2025  (3 years)
Output layout:
    data/grids/
    ├── aod/    aod_2023-01-01.tif  aod_2023-01-02.tif  ...
    ├── t2m/ rh/ ws/ blh/ prec/ ssr/ sp/ frp/ ndvi/
    ├── elev.tif   (static — SRTM)
    └── pop.tif    (static — GPWv4 2020 epoch)

Runtime: 3 years x 10 daily predictors = 10,950 GeoTIFFs.
Safe to interrupt and re-run; SKIP_EXISTING=True skips completed files.
"""

import time
from pathlib import Path

import ee
import geemap
import pandas as pd
from tqdm.auto import tqdm

EE_PROJECT = "ee-intaratt"
try:
    ee.Initialize(project=EE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT)

# ── User settings ─────────────────────────────────────────────────────────────

START_DATE    = "2023-01-01"
END_DATE      = "2026-01-01"    # inclusive (exclusive form: 2026-01-01)
SCALE_M       = 3000             # output resolution in metres (3 km)
GRID_DIR      = Path("./data/grids")
SKIP_EXISTING = True
REGION        = ee.Geometry.Rectangle([97.0, 5.5, 106.0, 21.0])  # Thailand bbox
MAX_PIXELS    = int(1e10)
MAX_RETRIES   = 3
RETRY_BACKOFF = 15               # seconds; doubled each retry
THROTTLE_S    = 1.5              # seconds between consecutive download calls

GRID_DIR.mkdir(parents=True, exist_ok=True)

# ── Core helper — export_daily ────────────────────────────────────────────────


def _with_retry(fn, label=""):
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as exc:
            wait = RETRY_BACKOFF * (2 ** attempt)
            if attempt < MAX_RETRIES - 1:
                tqdm.write(f"  [{label}] attempt {attempt+1} failed: {exc}"
                           f" — retrying in {wait}s")
                time.sleep(wait)
            else:
                tqdm.write(f"  [{label}] gave up after {MAX_RETRIES} attempts: {exc}")
                raise


def export_daily(collection, band_out, prefix, scale, out_subdir, dates=None):
    """
    Export one GeoTIFF per day.

    Parameters
    ----------
    collection : ee.ImageCollection — export_daily applies filterDate per day.
    band_out   : str  — band name after rename.
    prefix     : str  — filename prefix (e.g. 'aod').
    scale      : int  — export resolution in metres.
    out_subdir : Path — destination folder.
    dates      : DatetimeIndex | None — explicit dates for sparse collections
                 (e.g. NDVI composites). None -> every day in [START, END].
    """
    out_subdir = Path(out_subdir)
    out_subdir.mkdir(parents=True, exist_ok=True)

    if dates is None:
        dates = pd.date_range(START_DATE, END_DATE, inclusive="both", freq="D")

    skipped = 0
    for d in tqdm(dates, desc=prefix, leave=False):
        date_str = d.strftime("%Y-%m-%d")
        out_path = out_subdir / f"{prefix}_{date_str}.tif"

        if SKIP_EXISTING and out_path.exists() and out_path.stat().st_size > 0:
            continue

        d0  = ee.Date(date_str)
        d1  = d0.advance(1, "day")
        col = collection.filterDate(d0, d1)

        # Empty-collection guard: prevents writing a zero-value raster (the
        # result of .mean() on an empty ImageCollection) for cloud-covered days.
        try:
            n_imgs = col.size().getInfo()
        except Exception as exc:
            tqdm.write(f"  [{prefix} {date_str}] size() failed: {exc} — skipping")
            skipped += 1
            continue
        if n_imgs == 0:
            skipped += 1
            continue

        img = col.mean().rename(band_out).clip(REGION).toFloat()

        try:
            _with_retry(
                lambda: geemap.ee_export_image(
                    img, str(out_path),
                    region=REGION, scale=scale,
                    crs="EPSG:4326", file_per_band=False,
                ),
                label=f"{prefix} {date_str}",
            )
        except Exception:
            pass

        time.sleep(THROTTLE_S)

    if skipped:
        tqdm.write(f"  [{prefix}] {skipped} days skipped (no data / cloud)")


# ── 1. MAIAC AOD (470 nm, QA-masked) ─────────────────────────────────────────


def mask_maiac_qa(img):
    qa = img.select("AOD_QA")
    cloud_clear = qa.bitwiseAnd(7).eq(1)   # bits 0-2 == 1: clear sky
    return img.updateMask(cloud_clear)


def scale_maiac(img):
    aod = img.select("Optical_Depth_047").multiply(0.001)
    aod = aod.updateMask(aod.gt(0)).updateMask(aod.lt(3.0))
    return aod.copyProperties(img, ["system:time_start"])


maiac = (ee.ImageCollection("MODIS/061/MCD19A2_GRANULES")
         .filterBounds(REGION)
         .select(["Optical_Depth_047", "AOD_QA"])
         .map(mask_maiac_qa)
         .map(scale_maiac))

export_daily(maiac, "aod", "aod", SCALE_M, GRID_DIR / "aod")

# ── 2. ERA5-Land daily variables ──────────────────────────────────────────────

era5land = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")

# t2m: convert Kelvin → Celsius to match merge_gee_exports.py (line 167),
# which does the same conversion before writing the training panel.
# Both the training predictors and raster predictors must share the same scale
# so the fitted StandardScaler applies correctly at prediction time.
era5land_t2m = era5land.map(
    lambda img: img.select("temperature_2m")
                   .subtract(273.15)
                   .rename("t2m")
                   .copyProperties(img, ["system:time_start"])
)
export_daily(era5land_t2m, "t2m", "t2m", SCALE_M, GRID_DIR / "t2m")

# Other ERA5-Land variables exported in native units (same units as training panel)
ERA5L_RAW_VARS = [
    ("total_precipitation_sum",               "prec", "prec"),
    ("surface_solar_radiation_downwards_sum", "ssr",  "ssr"),
    ("surface_pressure",                      "sp",   "sp"),
]
for band_in, band_out, sub in ERA5L_RAW_VARS:
    export_daily(
        era5land.select([band_in]),
        band_out=band_out, prefix=band_out,
        scale=SCALE_M, out_subdir=GRID_DIR / sub,
    )

# ── 3. Derived RH and WS (computed inside Earth Engine) ──────────────────────


def rh_image(img):
    t = img.select("temperature_2m").subtract(273.15)
    d = img.select("dewpoint_temperature_2m").subtract(273.15)
    a, b = 17.625, 243.04
    es_t = t.multiply(a).divide(t.add(b)).exp()
    es_d = d.multiply(a).divide(d.add(b)).exp()
    return (es_d.divide(es_t).multiply(100).clamp(0, 100)
                .rename("rh")
                .copyProperties(img, ["system:time_start"]))


def ws_image(img):
    u = img.select("u_component_of_wind_10m")
    v = img.select("v_component_of_wind_10m")
    return (u.pow(2).add(v.pow(2)).sqrt()
             .rename("ws")
             .copyProperties(img, ["system:time_start"]))


export_daily(era5land.map(rh_image), "rh", "rh", SCALE_M, GRID_DIR / "rh")
export_daily(era5land.map(ws_image), "ws", "ws", SCALE_M, GRID_DIR / "ws")

# ── 4. Boundary layer height — ERA5 hourly → daily mean ──────────────────────
#
# Passing era5_hourly_blh (a plain ee.ImageCollection) directly to
# export_daily is correct and safe: export_daily applies a 1-day filterDate
# for each date, then .mean() averages the 24 hourly images.
# This produces a flat, single-level EE compute graph — no nested
# ee.List.sequence or ee.Algorithms.If nodes that could blow the graph limit.

era5_hourly_blh = ee.ImageCollection("ECMWF/ERA5/HOURLY").select("boundary_layer_height")
export_daily(era5_hourly_blh, "blh", "blh", SCALE_M, GRID_DIR / "blh")

# ── 5. VIIRS active fire — daily FRP (MW) ────────────────────────────────────

fire = (ee.ImageCollection("NASA/VIIRS/002/VNP14A1")
        .map(lambda img: (img.select("MaxFRP")
                             .updateMask(img.select("MaxFRP").gte(0))
                             .multiply(0.1)
                             .unmask(0)
                             .copyProperties(img, ["system:time_start"]))))
export_daily(fire, "frp", "frp", SCALE_M, GRID_DIR / "frp")

# ── 6. NDVI — MOD13A2 composite dates only ───────────────────────────────────
#
# Exporting every calendar day would write ~1,095 files for ~69 actual composites.
# Instead retrieve the actual composite dates from EE and export only those.
# raster_prediction.py forward-fills NDVI between composite dates.

ndvi_col = (ee.ImageCollection("MODIS/061/MOD13A2")
            .select("NDVI")
            .map(lambda img: img.multiply(0.0001)
                                .copyProperties(img, ["system:time_start"])))

ndvi_ts_raw = (
    ndvi_col
    .filterDate(START_DATE,
                (pd.Timestamp(END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    .aggregate_array("system:time_start")
    .getInfo()
)
ndvi_dates = pd.to_datetime(ndvi_ts_raw, unit="ms").floor("D")
print(f"NDVI: {len(ndvi_dates)} composite dates between {START_DATE} and {END_DATE}")

export_daily(ndvi_col, "ndvi", "ndvi", SCALE_M, GRID_DIR / "ndvi", dates=ndvi_dates)

# ── 7. Static layers — elevation and population ───────────────────────────────


def export_static_image(image, filename, scale):
    out_path = GRID_DIR / filename
    if SKIP_EXISTING and out_path.exists() and out_path.stat().st_size > 0:
        print(f"  skip (exists): {filename}")
        return
    _with_retry(
        lambda: geemap.ee_export_image(
            image, str(out_path),
            region=REGION, scale=scale,
            crs="EPSG:4326", file_per_band=False,
        ),
        label=filename,
    )
    print(f"  wrote -> {out_path}")


elev = (ee.Image("USGS/SRTMGL1_003")
          .select("elevation").rename("elev")
          .clip(REGION).toFloat())
export_static_image(elev, "elev.tif", SCALE_M)

# GPWv4 2020-epoch image — most recent available, held constant 2023-2025.
pop = (ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Density")
         .filterDate("2020-01-01", "2021-01-01")
         .first()
         .select("population_density").rename("pop")
         .clip(REGION).toFloat())
export_static_image(pop, "pop.tif", SCALE_M)

print("\nDone. data/grids/ is ready for the raster prediction stage.")
print("Run: python main.py --config configs/default.yaml --stages map_raster")

# ── Optional — Drive batch export fallback ────────────────────────────────────
# Use when direct downloads are impractical. Tasks run server-side; download
# GeoTIFFs from Google Drive when complete.
# EE task-queue limit: 3,000 concurrent tasks.
# 10 predictors x 1,095 days = 10,950 total tasks.
# Submissions are batched with a confirmation prompt between each batch.

DRIVE_FOLDER  = "pm25_thailand_grids"
TASK_BATCH_SZ = 500


def submit_drive_daily(collection, band_out, prefix, scale,
                       dates=None, folder=DRIVE_FOLDER):
    """Queue one Export.image.toDrive task per date, in batches."""
    if dates is None:
        dates = pd.date_range(START_DATE, END_DATE, inclusive="both", freq="D")
    tasks = []
    for i, d in enumerate(dates):
        date_str = d.strftime("%Y-%m-%d")
        d0 = ee.Date(date_str)
        d1 = d0.advance(1, "day")
        img = (collection.filterDate(d0, d1)
                         .mean().rename(band_out)
                         .clip(REGION).toFloat())
        desc = f"{prefix}_{date_str}"
        task = ee.batch.Export.image.toDrive(
            image=img, description=desc, folder=folder,
            fileNamePrefix=desc, region=REGION,
            scale=scale, maxPixels=MAX_PIXELS,
        )
        task.start()
        tasks.append(task)
        if (i + 1) % TASK_BATCH_SZ == 0:
            print(f"  submitted {i+1} tasks for {prefix}")
            input("  Press Enter to submit next batch...")
    print(f"  queued {len(tasks)} Drive tasks for prefix={prefix}")
    return tasks


# Examples (uncomment to use):
# submit_drive_daily(maiac,                              "aod",  "aod",  SCALE_M)
# submit_drive_daily(era5land.select(["temperature_2m"]), "t2m", "t2m", SCALE_M)
# submit_drive_daily(era5_hourly_blh,                    "blh",  "blh",  SCALE_M)
