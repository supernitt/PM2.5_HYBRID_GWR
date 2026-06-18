"""Merge per-predictor CSVs exported from Google Earth Engine into the
single station-day panel that the PM2.5 hybrid pipeline consumes.

This script also derives:
  - RH from T2m and dewpoint (August-Roche-Magnus approximation, returning
    relative humidity in percent).
  - WS from u10 and v10 (m/s).
  - Converts T2m from Kelvin to Celsius for easier reading.

It also linearly interpolates the 16-day NDVI series to daily values per
station, so the panel ends up with one NDVI value per station-day.

Usage
-----
    python merge_gee_exports.py \
        --gee-dir ./data/gee_exports \
        --pm25-file ./data/pcd_pm25_daily.csv \
        --out ./data/panel.csv

The PCD PM2.5 file must have columns: station_id, date, pm25, lat, lon.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("merge")


# ---------------------------------------------------------------------
# Predictor -> exported filename mapping (filenames as produced by
# the gee_extract_stations.js script in this folder).
# ---------------------------------------------------------------------
DAILY_FILES = {
    "aod":  "pcd_aod_maiac.csv",
    "t2m":  "pcd_t2m.csv",
    "d2m":  "pcd_d2m.csv",
    "u10":  "pcd_u10.csv",
    "v10":  "pcd_v10.csv",
    "blh":  "pcd_blh.csv",
    "prec": "pcd_prec.csv",
    "ssr":  "pcd_ssr.csv",
    "sp":   "pcd_sp.csv",
    "frp":  "pcd_frp_viirs.csv",
    "ndvi": "pcd_ndvi.csv",
}
STATIC_FILES = {
    "elev": "pcd_elev_srtm.csv",
    "pop":  "pcd_pop_gpw.csv",
}


# ---------------------------------------------------------------------
def _read_daily(gee_dir: Path, predictor: str) -> pd.DataFrame:
    path = gee_dir / DAILY_FILES[predictor]
    df = pd.read_csv(path)
    # GEE exports may include extra columns like 'system:index'; keep what we need
    keep = ["station_id", "date", predictor]
    df = df[[c for c in keep if c in df.columns]]
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["station_id", "date"])
    return df


def _read_static(gee_dir: Path, predictor: str) -> pd.DataFrame | None:
    path = gee_dir / STATIC_FILES[predictor]
    if not path.exists():
        logger.warning("Static file not found, skipping %s: %s", predictor, path)
        return None
    df = pd.read_csv(path)
    keep = ["station_id", predictor]
    return df[[c for c in keep if c in df.columns]]


# ---------------------------------------------------------------------
def compute_rh(t2m_k: pd.Series, d2m_k: pd.Series) -> pd.Series:
    """Relative humidity (%) from 2m temperature and dewpoint, both in K.

    August-Roche-Magnus approximation:
        e_s(T) = 6.1094 * exp(17.625 * T_C / (T_C + 243.04))
    """
    t_c = t2m_k - 273.15
    d_c = d2m_k - 273.15
    a, b = 17.625, 243.04
    es_t = np.exp(a * t_c / (b + t_c))
    es_d = np.exp(a * d_c / (b + d_c))
    rh = 100.0 * es_d / es_t
    return rh.clip(lower=0, upper=100)


def compute_ws(u10: pd.Series, v10: pd.Series) -> pd.Series:
    """10 m wind speed (m/s) from u and v components."""
    return np.sqrt(u10.astype(float) ** 2 + v10.astype(float) ** 2)


# ---------------------------------------------------------------------
def interpolate_ndvi_to_daily(ndvi_df: pd.DataFrame,
                              full_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Linearly interpolate the 16-day NDVI series to daily values per station."""
    out = []
    for sid, g in ndvi_df.groupby("station_id"):
        g = g.set_index("date").sort_index()
        # Reindex to the full daily range and interpolate in time
        g = g.reindex(full_dates)
        g["station_id"] = sid
        g["ndvi"] = g["ndvi"].interpolate(method="time", limit_direction="both")
        g.index.name = "date"
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    _GEE = Path("/data/DataHub/Projects/2026_luktarn/data/gee_exports")
    _DATA = Path("/data/DataHub/Projects/2026_luktarn/data")
    p.add_argument("--gee-dir", type=Path, default=_GEE,
                   help="Folder with the per-predictor CSVs exported from GEE.")
    p.add_argument("--pm25-file", type=Path, default=_DATA / "pcd_pm25_daily.csv",
                   help="PCD PM2.5 daily file (station_id, date, pm25, lat, lon).")
    p.add_argument("--out", type=Path, default=_DATA / "panel.csv",
                   help="Output merged panel CSV.")
    args = p.parse_args()

    # ---- 1. Read PM2.5 + station coords ----
    pm25 = pd.read_csv(args.pm25_file)
    pm25["date"] = pd.to_datetime(pm25["date"])
    logger.info("PM2.5 records: %d  | stations: %d",
                len(pm25), pm25["station_id"].nunique())

    # ---- 2. Read all daily predictors and merge ----
    panel = pm25.copy()
    for pred in ["aod", "t2m", "d2m", "u10", "v10",
                 "blh", "prec", "ssr", "sp", "frp"]:
        df = _read_daily(args.gee_dir, pred)
        panel = panel.merge(df, on=["station_id", "date"], how="left")
        logger.info("Merged %s — non-null rate: %.1f%%",
                    pred, 100 * panel[pred].notna().mean())

    # ---- 3. Read 16-day NDVI and interpolate to daily ----
    ndvi_raw = _read_daily(args.gee_dir, "ndvi")
    full_dates = pd.date_range(panel["date"].min(),
                               panel["date"].max(), freq="D")
    ndvi_daily = interpolate_ndvi_to_daily(ndvi_raw, full_dates)
    panel = panel.merge(ndvi_daily, on=["station_id", "date"], how="left")
    logger.info("Merged ndvi — non-null rate: %.1f%%",
                100 * panel["ndvi"].notna().mean())

    # ---- 4. Static predictors (optional — skip if not yet extracted) ----
    for pred in ["elev", "pop"]:
        df = _read_static(args.gee_dir, pred)
        if df is not None:
            panel = panel.merge(df, on="station_id", how="left")
            logger.info("Merged static %s — non-null rate: %.1f%%",
                        pred, 100 * panel[pred].notna().mean())

    # ---- 5. Derived variables ----
    panel["rh"] = compute_rh(panel["t2m"], panel["d2m"])
    panel["ws"] = compute_ws(panel["u10"], panel["v10"])
    # Convert T2m K -> C for readability (model is scale-invariant)
    panel["t2m"] = panel["t2m"] - 273.15

    # ---- 6. Drop intermediate columns we no longer need ----
    panel = panel.drop(columns=["d2m", "u10", "v10"])

    # ---- 7. Final column order ----
    cols = ["station_id", "date", "lat", "lon", "pm25",
            "aod", "t2m", "rh", "ws", "blh", "prec", "ssr", "sp",
            "frp", "ndvi", "elev", "pop"]
    panel = panel[[c for c in cols if c in panel.columns]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out, index=False)
    logger.info("Wrote merged panel to %s (%d rows, %d columns)",
                args.out, len(panel), panel.shape[1])


if __name__ == "__main__":
    main()
