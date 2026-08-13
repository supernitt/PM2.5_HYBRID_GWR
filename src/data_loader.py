"""Load the station-day PM2.5 panel and (optionally) the grid prediction file."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("pm25_hybrid")


def _read(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if p.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(p)
    raise ValueError(f"Unsupported file extension: {p.suffix}")


def load_panel(path: str | Path, columns: dict) -> pd.DataFrame:
    """Load the station-day panel and parse the date column.

    Returns a dataframe sorted by station and date.
    """
    df = _read(path)

    # Core columns must be present — pipeline cannot run without them.
    core = (
        [columns["station_id"], columns["date"], columns["lat"], columns["lon"],
         columns["target"], columns["aod"]]
        + columns["meteo"]
    )
    missing_core = [c for c in core if c not in df.columns]
    if missing_core:
        raise KeyError(f"Panel file is missing required columns: {missing_core}")

    # Ancillary columns (elev, pop, …) are optional — fill absent ones with NaN
    # so the pipeline can run before static GEE layers are extracted.
    missing_ancillary = [c for c in columns["ancillary"] if c not in df.columns]
    if missing_ancillary:
        logger.warning(
            "Ancillary column(s) not in panel, filled with NaN: %s",
            missing_ancillary,
        )
        for col in missing_ancillary:
            df[col] = float("nan")

    df[columns["date"]] = pd.to_datetime(df[columns["date"]])
    df = df.sort_values([columns["station_id"], columns["date"]]).reset_index(drop=True)

    # Tag season and Thai region if not already present (used for stratified metrics).
    if "season" not in df.columns:
        df["season"] = df[columns["date"]].apply(_thai_season)
    if "region" not in df.columns and "region" not in df.columns:
        # If user has not provided region, leave as NaN; stratification skips NaNs.
        df["region"] = pd.NA

    logger.info("Loaded panel: %d rows, %d stations, %s -> %s",
                len(df), df[columns["station_id"]].nunique(),
                df[columns["date"]].min().date(),
                df[columns["date"]].max().date())
    return df


def load_grid(path: str | Path | None, columns: dict) -> pd.DataFrame | None:
    """Load grid-day predictors for spatial mapping. Optional."""
    if path is None or not Path(path).exists():
        logger.warning("Grid file not found at %s — mapping stage will be skipped.", path)
        return None
    df = _read(path)
    df[columns["date"]] = pd.to_datetime(df[columns["date"]])
    logger.info("Loaded grid file: %d rows, %d unique grid cells",
                len(df),
                df[["lat", "lon"]].drop_duplicates().shape[0]
                if {"lat", "lon"}.issubset(df.columns) else -1)
    return df


def _thai_season(date) -> str:
    """Tag a date as cool-dry / hot-dry / wet-monsoon (Thailand)."""
    m = date.month
    if m in (11, 12, 1, 2):
        return "cool_dry"
    if m in (3, 4, 5):
        return "hot_dry"
    return "wet_monsoon"
