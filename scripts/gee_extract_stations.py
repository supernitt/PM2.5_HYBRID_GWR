"""
Extract daily predictors at PCD stations (geemap) — v4

Samples all predictor collections at the PCD monitoring stations and
downloads one CSV per predictor to local disk via geemap.ee_export_vector.

Study period: 1 January 2023 – 31 December 2025
Inputs:
  - Earth Engine account (earthengine authenticate once per machine).
  - data/pcd_stations.csv with columns station_ID, Latitude, Longitude.
Outputs in data/gee_exports/:
  pcd_aod_maiac.csv, pcd_t2m.csv, pcd_d2m.csv, pcd_u10.csv, pcd_v10.csv,
  pcd_blh.csv, pcd_prec.csv, pcd_ssr.csv, pcd_sp.csv, pcd_frp_viirs.csv,
  pcd_ndvi.csv, pcd_elev_srtm.csv, pcd_pop_gpw.csv
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

STATIONS_CSV  = Path("/data/DataHub/Projects/2026_luktarn/data/Air4Thai_station.csv")
OUT_DIR       = Path("/data/DataHub/Projects/2026_luktarn/data/gee_exports")
START_DATE    = "2023-01-01"
END_DATE      = "2026-01-01"   # exclusive upper bound → covers through 2025-12-31
BUFFER_M      = 1000            # radius (m) for FRP spatial sum
MAX_RETRIES   = 3
RETRY_BACKOFF = 10              # seconds; doubled each retry

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "_chunks").mkdir(parents=True, exist_ok=True)

# ── Load stations and build EE FeatureCollection ──────────────────────────────

stations_df = pd.read_csv(STATIONS_CSV)
required = {"station_ID", "Latitude", "Longitude"}
missing = required - set(stations_df.columns)
assert not missing, f"{STATIONS_CSV} is missing columns: {missing}"
print(f"Loaded {len(stations_df)} stations from {STATIONS_CSV}")

stations_fc = ee.FeatureCollection([
    ee.Feature(
        ee.Geometry.Point([float(r.Longitude), float(r.Latitude)]),
        {"station_id": str(r.station_ID)},
    )
    for r in stations_df.itertuples()
])

# ── Core helpers ──────────────────────────────────────────────────────────────

def _month_chunks(start, end):
    """Yield (a, b) YYYY-MM-DD string pairs, one per calendar month."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    for ts in pd.date_range(s, e - pd.Timedelta(days=1), freq="MS"):
        a = ts
        b = min(e, ts + pd.offsets.MonthBegin(1))
        yield a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")


def _chunk_ok(path):
    """True only for a non-empty, parseable CSV with at least one data row."""
    try:
        return len(pd.read_csv(path, nrows=1)) > 0
    except Exception:
        return False


def sample_daily(collection_or_fn, band_in, band_out, scale, out_csv,
                 reducer_kind="mean", buffer_m=None, dedup=False,
                 min_val=None, max_val=None):
    """
    Sample an ImageCollection at PCD stations, month-by-month, and save
    one merged CSV to out_csv.

    Parameters
    ----------
    collection_or_fn : ee.ImageCollection or callable(start_str, end_str)
    band_in   : str   — band name inside each image.
    band_out  : str   — output column / EE property name.
    scale     : int   — reduceRegions scale in metres.
    out_csv   : Path  — final merged output.
    reducer_kind : 'mean' | 'sum'
    buffer_m  : int | None — spatial buffer radius for FRP.
    dedup     : bool  — if True, average multiple same-day rows (Terra+Aqua).
    min_val   : float | None — drop rows where band_out < min_val.
    max_val   : float | None — drop rows where band_out > max_val.
    """
    out_csv = Path(out_csv)
    if out_csv.exists():
        print(f"  skip (exists): {out_csv.name}")
        return

    sample_fc = (stations_fc if buffer_m is None
                 else stations_fc.map(lambda f: f.buffer(buffer_m)))
    reducer = (ee.Reducer.mean() if reducer_kind == "mean" else ee.Reducer.sum()
               ).setOutputs([band_out])

    chunks_dir = out_csv.parent / "_chunks"
    chunk_paths = []

    for a, b in tqdm(list(_month_chunks(START_DATE, END_DATE)),
                     desc=out_csv.stem, leave=False):
        chunk_csv = chunks_dir / f"{out_csv.stem}__{a}__{b}.csv"

        if chunk_csv.exists():
            if _chunk_ok(chunk_csv):
                chunk_paths.append(chunk_csv)
                continue
            chunk_csv.unlink()

        col = (collection_or_fn(a, b) if callable(collection_or_fn)
               else collection_or_fn.filterDate(a, b))

        def per_image(img):
            date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
            return (
                img.select([band_in], [band_out])
                   .reduceRegions(
                       collection=sample_fc,
                       reducer=reducer,
                       scale=scale,
                       tileScale=4,
                   )
                   # KEY FIX: ee.Element.get() has no default-value argument in
                   # Python. Use ee.Algorithms.If + propertyNames() to supply
                   # -9999 when reduceRegions omits the property entirely for
                   # pixels with no granule coverage. The sentinel is filtered
                   # out by min_val in Python.
                   .map(lambda f: f.set(
                       "date", date,
                       band_out, ee.Algorithms.If(
                           f.propertyNames().contains(band_out),
                           f.get(band_out),
                           -9999,
                       ),
                   ))
            )

        flat = col.map(per_image).flatten()

        try:
            n_feat = flat.size().getInfo()
        except Exception as exc:
            tqdm.write(f"  [{out_csv.stem} {a}] size() failed: {exc} — skipping")
            continue
        if n_feat == 0:
            tqdm.write(f"  [{out_csv.stem} {a}] 0 features — skipping")
            continue

        for attempt in range(MAX_RETRIES):
            try:
                geemap.ee_export_vector(flat, str(chunk_csv), verbose=False)
                if _chunk_ok(chunk_csv):
                    chunk_paths.append(chunk_csv)
                    break
                raise ValueError("downloaded CSV is empty")
            except Exception as exc:
                wait = RETRY_BACKOFF * (2 ** attempt)
                if attempt < MAX_RETRIES - 1:
                    tqdm.write(f"  [{out_csv.stem} {a}] attempt {attempt+1} "
                               f"failed: {exc} — retrying in {wait}s")
                    time.sleep(wait)
                else:
                    tqdm.write(f"  [{out_csv.stem} {a}] gave up after "
                               f"{MAX_RETRIES} attempts: {exc}")

    dfs = []
    for p in chunk_paths:
        if p.exists() and p.stat().st_size > 0:
            try:
                dfs.append(pd.read_csv(p))
            except Exception as exc:
                tqdm.write(f"  warning: could not read {p.name}: {exc}")
    if not dfs:
        print(f"  no chunks produced for {out_csv.name}")
        return

    out = pd.concat(dfs, ignore_index=True)
    keep = [c for c in ["station_id", "date", band_out] if c in out.columns]
    out = out[keep]

    if band_out in out.columns:
        if min_val is not None:
            out = out[out[band_out] >= min_val]
        if max_val is not None:
            out = out[out[band_out] <= max_val]

    if dedup and band_out in out.columns:
        out = (out.dropna(subset=[band_out])
                  .groupby(["station_id", "date"])[band_out]
                  .mean().reset_index())

    out.to_csv(out_csv, index=False)
    print(f"  wrote {len(out):,} rows -> {out_csv.name}")


# ── 1. MAIAC AOD (470 nm, raw scaled, Terra + Aqua averaged) ─────────────────

maiac = (ee.ImageCollection("MODIS/061/MCD19A2_GRANULES")
         .filterBounds(stations_fc.geometry())
         .select("Optical_Depth_047")
         .map(lambda img: (img.multiply(0.001)
                              .copyProperties(img, ["system:time_start"]))))

sample_daily(
    maiac,
    band_in="Optical_Depth_047",
    band_out="aod",
    scale=1000,
    out_csv=OUT_DIR / "pcd_aod_maiac.csv",
    dedup=True,
    min_val=0.0,
    max_val=3.0,
)

# ── 2. ERA5-Land daily aggregates (t2m, d2m, u10, v10, prec, ssr, sp) ────────

era5land = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")

ERA5L_VARS = [
    ("temperature_2m",                        "t2m",  "pcd_t2m.csv"),
    ("dewpoint_temperature_2m",               "d2m",  "pcd_d2m.csv"),
    ("u_component_of_wind_10m",               "u10",  "pcd_u10.csv"),
    ("v_component_of_wind_10m",               "v10",  "pcd_v10.csv"),
    ("total_precipitation_sum",               "prec", "pcd_prec.csv"),
    ("surface_solar_radiation_downwards_sum", "ssr",  "pcd_ssr.csv"),
    ("surface_pressure",                      "sp",   "pcd_sp.csv"),
]
for band_in, band_out, fname in ERA5L_VARS:
    sample_daily(
        era5land,
        band_in=band_in,
        band_out=band_out,
        scale=11132,
        out_csv=OUT_DIR / fname,
    )

# ── 3. ERA5 hourly BLH → daily mean ──────────────────────────────────────────
#
# Building with ee.List.sequence + ee.Algorithms.If inside a yearly chunk
# creates ~365 nested nodes that exceed EE's expression-graph memory limit.
# Fix: iterate over days in Python and build a small ee.ImageCollection of
# day-mean images. Each monthly chunk contains at most 31 images — a flat,
# shallow EE graph within the compute budget.

era5_hourly = ee.ImageCollection("ECMWF/ERA5/HOURLY").select("boundary_layer_height")


def blh_monthly_builder(start_str, end_str):
    """
    Return an ee.ImageCollection of daily-mean BLH images for [start_str, end_str).
    Built day-by-day in Python to avoid deep nested ee.Algorithms.If trees.
    """
    dates = pd.date_range(start_str, end_str, freq="D", inclusive="left")
    images = []
    for d in dates:
        d0 = ee.Date(d.strftime("%Y-%m-%d"))
        d1 = d0.advance(1, "day")
        daily_mean = (era5_hourly
                      .filterDate(d0, d1)
                      .mean()
                      .rename("blh")
                      .set("system:time_start", d0.millis()))
        images.append(daily_mean)
    return ee.ImageCollection(images)


sample_daily(
    blh_monthly_builder,
    band_in="blh",
    band_out="blh",
    scale=27830,
    out_csv=OUT_DIR / "pcd_blh.csv",
)

# ── 4. VIIRS active fire — daily MaxFRP summed within 1 km buffer ─────────────

viirs_fire = (ee.ImageCollection("NASA/VIIRS/002/VNP14A1")
              .select("MaxFRP")
              .map(lambda img: (img.updateMask(img.gte(0))
                                   .multiply(0.1)
                                   .copyProperties(img, ["system:time_start"]))))

sample_daily(
    viirs_fire,
    band_in="MaxFRP",
    band_out="frp",
    scale=1000,
    out_csv=OUT_DIR / "pcd_frp_viirs.csv",
    reducer_kind="sum",
    buffer_m=BUFFER_M,
)

# ── 5. NDVI — MOD13A2 16-day composites ──────────────────────────────────────

ndvi = (ee.ImageCollection("MODIS/061/MOD13A2")
        .select("NDVI")
        .map(lambda img: img.multiply(0.0001)
                            .copyProperties(img, ["system:time_start"])))

sample_daily(
    ndvi,
    band_in="NDVI",
    band_out="ndvi",
    scale=1000,
    out_csv=OUT_DIR / "pcd_ndvi.csv",
)

# ── 6. Static layers — SRTM elevation and GPWv4 population density ────────────


def export_static(image, band_in, band_out, scale, out_csv):
    """Sample a static (single) EE Image at all PCD stations."""
    out_csv = Path(out_csv)
    if out_csv.exists():
        print(f"  skip (exists): {out_csv.name}")
        return

    reducer = ee.Reducer.mean().setOutputs([band_out])
    feats = (image.select([band_in], [band_out])
                  .reduceRegions(collection=stations_fc, reducer=reducer,
                                 scale=scale, tileScale=4))

    n = feats.size().getInfo()
    if n == 0:
        print(f"  warning: 0 features returned for {out_csv.name} — skipping")
        return

    for attempt in range(MAX_RETRIES):
        try:
            geemap.ee_export_vector(feats, str(out_csv), verbose=False)
            break
        except Exception as exc:
            wait = RETRY_BACKOFF * (2 ** attempt)
            if attempt < MAX_RETRIES - 1:
                print(f"  retry {attempt+1}/{MAX_RETRIES} for {out_csv.name}: {exc}")
                time.sleep(wait)
            else:
                print(f"  gave up on {out_csv.name}: {exc}")
                return

    if out_csv.exists():
        df = pd.read_csv(out_csv)
        keep = [c for c in ["station_id", band_out] if c in df.columns]
        df[keep].to_csv(out_csv, index=False)
        print(f"  wrote {len(df):,} rows -> {out_csv.name}")


srtm = ee.Image("USGS/SRTMGL1_003").select("elevation")
export_static(srtm, "elevation", "elev", 30, OUT_DIR / "pcd_elev_srtm.csv")

# GPWv4: 2020-epoch image (most recent available, held constant 2023-2025).
pop = (ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Density")
       .filterDate("2020-01-01", "2021-01-01")
       .first()
       .select("population_density"))
export_static(pop, "population_density", "pop", 1000, OUT_DIR / "pcd_pop_gpw.csv")

print("\nDone. All CSVs written to", OUT_DIR)

# ── Optional — Drive batch export fallback ────────────────────────────────────
# Use only if direct downloads keep timing out. Submits one task per year per
# predictor to avoid EE's per-task memory limit.

def _year_chunks(start, end):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    for y in range(s.year, e.year + 1):
        a = max(s, pd.Timestamp(f"{y}-01-01"))
        b = min(e, pd.Timestamp(f"{y+1}-01-01"))
        if a < b:
            yield a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")


def submit_drive_export(collection_or_fn, band_in, band_out, scale,
                        description_prefix, reducer_kind="mean", buffer_m=None,
                        drive_folder="pm25_gee_exports"):
    """Submit one Drive export task per year. Avoids 3-year single-task timeout."""
    sample_fc = (stations_fc if buffer_m is None
                 else stations_fc.map(lambda f: f.buffer(buffer_m)))
    reducer = (ee.Reducer.mean() if reducer_kind == "mean" else ee.Reducer.sum()
               ).setOutputs([band_out])
    tasks = []
    for a, b in _year_chunks(START_DATE, END_DATE):
        col = (collection_or_fn(a, b) if callable(collection_or_fn)
               else collection_or_fn.filterDate(a, b))

        def per_image(img):
            date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
            return (img.select([band_in], [band_out])
                       .reduceRegions(collection=sample_fc, reducer=reducer,
                                      scale=scale, tileScale=4)
                       .map(lambda f: f.set("date", date)))

        flat = col.map(per_image).flatten()
        desc = f"{description_prefix}_{a[:4]}"
        task = ee.batch.Export.table.toDrive(
            collection=flat, description=desc,
            folder=drive_folder, fileNamePrefix=desc, fileFormat="CSV",
        )
        task.start()
        tasks.append(task)
        print(f"  queued Drive task: {desc}  (id {task.id})")
    return tasks


# Examples (uncomment to use):
# submit_drive_export(maiac, "Optical_Depth_047", "aod", 1000, "pcd_aod_maiac")
# submit_drive_export(era5land, "temperature_2m", "t2m", 11132, "pcd_t2m")
