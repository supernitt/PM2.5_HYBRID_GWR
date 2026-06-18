"""Fetch static predictor layers (SRTM elevation, WorldPop population) for PCD stations.

Produces:
  data/gee_exports/pcd_elev_srtm.csv   — columns: station_id, elev  (metres, SRTM30m)
  data/gee_exports/pcd_pop_gpw.csv     — columns: station_id, pop   (persons/km², WorldPop 2020)

No GEE account required.  Elevation is fetched from the OpenTopoData public API
(SRTM 30m, 100 locations per request).  Population is sampled from the WorldPop
Thailand 2020 unconstrained 1-km GeoTIFF (publicly available, ~15 MB download).

Run:
    /mnt/Data1/geoENV/bin/python3 scripts/fetch_static_layers.py
"""
from __future__ import annotations

import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT.parent / "data"
GEE_DIR      = DATA_DIR / "gee_exports"
STATIONS_CSV = DATA_DIR / "pcd_pm25_daily.csv"

OUT_ELEV = GEE_DIR / "pcd_elev_srtm.csv"
OUT_POP  = GEE_DIR / "pcd_pop_gpw.csv"

WORLDPOP_URL = (
    "https://data.worldpop.org/GIS/Population/"
    "Global_2000_2020/2020/THA/tha_ppp_2020.tif"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_stations(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    stations = (df[["station_id", "lat", "lon"]]
                .drop_duplicates("station_id")
                .reset_index(drop=True))
    logger.info("Loaded %d unique stations from %s", len(stations), csv.name)
    return stations


def fetch_elevation_opentopodata(stations: pd.DataFrame,
                                 batch_size: int = 100,
                                 pause: float = 1.0) -> pd.DataFrame:
    """Query OpenTopoData SRTM30m in batches of up to 100 locations."""
    BASE = "https://api.opentopodata.org/v1/srtm30m"
    rows = []
    n = len(stations)
    for start in range(0, n, batch_size):
        batch = stations.iloc[start:start + batch_size]
        locs = "|".join(f"{r.lat},{r.lon}" for r in batch.itertuples())
        logger.info("Elevation request: stations %d–%d / %d",
                    start + 1, min(start + batch_size, n), n)
        resp = requests.get(BASE, params={"locations": locs}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if data["status"] != "OK":
            raise RuntimeError(f"OpenTopoData error: {data}")
        for sid, result in zip(batch["station_id"], data["results"]):
            rows.append({"station_id": sid, "elev": result["elevation"]})
        if start + batch_size < n:
            time.sleep(pause)   # be polite to the public API
    return pd.DataFrame(rows)


def download_worldpop(url: str, dest: Path) -> None:
    """Stream-download the WorldPop GeoTIFF with a progress log."""
    logger.info("Downloading WorldPop raster from %s", url)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):   # 1 MB chunks
                f.write(chunk)
                downloaded += len(chunk)
        if total:
            logger.info("Downloaded %.1f MB", downloaded / 1e6)


def sample_raster(tif_path: Path, lats: np.ndarray,
                  lons: np.ndarray) -> np.ndarray:
    """Sample a GeoTIFF at (lat, lon) points using rasterio."""
    import rasterio
    with rasterio.open(tif_path) as src:
        nodata = src.nodata
        coords = list(zip(lons, lats))
        values = np.array([v[0] for v in src.sample(coords)], dtype=float)
        if nodata is not None:
            values[values == nodata] = np.nan
    return values


def fetch_population_worldpop(stations: pd.DataFrame) -> pd.DataFrame:
    """Download WorldPop Thailand 2020 raster and sample at station points."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tif_path = Path(tmpdir) / "tha_ppp_2020.tif"
        download_worldpop(WORLDPOP_URL, tif_path)
        logger.info("Sampling %d stations from raster…", len(stations))
        values = sample_raster(tif_path,
                               stations["lat"].to_numpy(),
                               stations["lon"].to_numpy())
    df = stations[["station_id"]].copy()
    df["pop"] = values
    # WorldPop counts persons per pixel; 1-km grid → persons/km² ≈ count/pixel
    # Fill coastal/missing with 0 (no residents at sea-edge stations)
    df["pop"] = df["pop"].fillna(0.0).clip(lower=0.0)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    GEE_DIR.mkdir(parents=True, exist_ok=True)
    stations = load_stations(STATIONS_CSV)

    # --- Elevation ---
    if OUT_ELEV.exists():
        logger.info("skip (exists): %s", OUT_ELEV.name)
    else:
        logger.info("=== Fetching SRTM elevation ===")
        elev_df = fetch_elevation_opentopodata(stations)
        elev_df.to_csv(OUT_ELEV, index=False)
        logger.info("Saved %s  (%d rows)", OUT_ELEV, len(elev_df))
        logger.info("Elevation range: %.1f – %.1f m",
                    elev_df["elev"].min(), elev_df["elev"].max())

    # --- Population ---
    if OUT_POP.exists():
        logger.info("skip (exists): %s", OUT_POP.name)
    else:
        logger.info("=== Fetching WorldPop population ===")
        pop_df = fetch_population_worldpop(stations)
        pop_df.to_csv(OUT_POP, index=False)
        logger.info("Saved %s  (%d rows)", OUT_POP, len(pop_df))
        logger.info("Population range: %.0f – %.0f persons/km²",
                    pop_df["pop"].min(), pop_df["pop"].max())

    logger.info("Done. Files written to %s", GEE_DIR)


if __name__ == "__main__":
    main()
