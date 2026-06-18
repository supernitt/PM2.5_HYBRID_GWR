#!/usr/bin/env python3
"""
prepare_pm25_pcd.py

Reshape PCD PM2.5 Excel files (wide-format, one column per station) into the
long-format CSV that merge_gee_exporttt.py / merge_gee_exports.py consumes
as --pm25-file.

Input
-----
  PM_station/2023.xlsx   sheet "PM2.5"
  PM_station/2024.xlsx   sheet "Data"
  PM_station/2025.xlsx   sheet "DATA"
  data/Air4Thai_station.csv   columns: station_ID, Latitude, Longitude

Output
------
  data/pcd_pm25_daily.csv   columns: station_id, date, pm25, lat, lon

  • station_id : lowercase (e.g. "02t") — matches GEE CSV and merge script
  • date       : YYYY-MM-DD string
  • pm25       : daily-average PM2.5 (µg/m³), valid range [0, 500]
  • lat / lon  : from Air4Thai_station.csv; stations absent from that file
                 (10T, 11T) are silently dropped with a warning.

Usage
-----
    python prepare_pm25_pcd.py                        # uses default paths
    python prepare_pm25_pcd.py \\
        --pm-dir   /mnt/Data1/2026_luktarn/PM_station \\
        --stations /mnt/Data1/2026_luktarn/data/Air4Thai_station.csv \\
        --out      /mnt/Data1/2026_luktarn/data/pcd_pm25_daily.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("prepare_pm25")

# ── Default paths ─────────────────────────────────────────────────────────────
_BASE            = Path("/data/DataHub/Projects/2026_luktarn")
DEFAULT_PM_DIR   = _BASE / "PM_station"
DEFAULT_STATIONS = _BASE / "data" / "Air4Thai_station.csv"
DEFAULT_OUT      = _BASE / "data" / "pcd_pm25_daily.csv"

# Sheet names per year (PCD changes these across annual releases)
YEAR_SHEETS: dict[int, str] = {
    2023: "PM2.5",
    2024: "Data",
    2025: "DATA",
}

PM25_MIN: float = 0.0    # µg/m³ — physical lower bound
PM25_MAX: float = 500.0  # µg/m³ — implausibly large values


# ── Per-file loader ───────────────────────────────────────────────────────────

def _sheet_for_year(xl: pd.ExcelFile, year: int) -> str:
    """Return the data sheet name, falling back to the first sheet."""
    preferred = YEAR_SHEETS.get(year)
    if preferred and preferred in xl.sheet_names:
        return preferred
    # Heuristic fallback: pick the first sheet that is not a station-detail sheet
    for name in xl.sheet_names:
        if "รายละเอียด" not in name:
            logger.warning("Year %d: expected sheet %r not found; using %r",
                           year, preferred, name)
            return name
    return xl.sheet_names[0]


def _read_one_year(path: Path, year: int) -> pd.DataFrame:
    """Read one annual Excel file and return a long-format DataFrame."""
    xl = pd.ExcelFile(path)
    sheet = _sheet_for_year(xl, year)
    df = pd.read_excel(xl, sheet_name=sheet, header=0)

    # First column is always the date; rename it
    df = df.rename(columns={df.columns[0]: "date"})

    # Drop trailing remark rows (NaT date or non-date text)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # Melt wide → long (one row per station-day)
    df_long = df.melt(id_vars="date", var_name="station_id", value_name="pm25")

    # Lowercase station IDs to match Air4Thai_station.csv and GEE outputs
    df_long["station_id"] = (
        df_long["station_id"].astype(str).str.strip().str.lower()
    )

    return df_long


# ── Multi-year aggregation ────────────────────────────────────────────────────

def load_pm25(pm_dir: Path) -> pd.DataFrame:
    """
    Load all annual Excel files in *pm_dir*, concatenate, clean, and return
    a long-format DataFrame with columns [station_id, date, pm25].
    """
    dfs: list[pd.DataFrame] = []

    # Discover .xlsx files and infer year from filename
    for path in sorted(pm_dir.glob("*.xlsx")):
        try:
            year = int(path.stem)
        except ValueError:
            logger.warning("Skipping unrecognised file: %s", path.name)
            continue

        df = _read_one_year(path, year)
        logger.info("  %d  %s  →  %d raw rows", year, path.name, len(df))
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No *.xlsx files found in {pm_dir}")

    combined = pd.concat(dfs, ignore_index=True)

    # Coerce pm25 to float; drop rows with no measurement
    combined["pm25"] = pd.to_numeric(combined["pm25"], errors="coerce")
    combined = combined.dropna(subset=["pm25"])

    # Range filter: remove physically implausible values and -9999 fill artefacts
    n_before = len(combined)
    combined = combined[
        combined["pm25"].between(PM25_MIN, PM25_MAX, inclusive="both")
    ]
    n_drop = n_before - len(combined)
    if n_drop:
        logger.warning(
            "Removed %d out-of-range rows (outside [%.0f, %.0f] µg/m³)",
            n_drop, PM25_MIN, PM25_MAX,
        )

    # Deduplicate: average duplicate (station_id, date) pairs if any arise
    combined = (
        combined
        .groupby(["station_id", "date"], sort=False)["pm25"]
        .mean()
        .reset_index()
    )

    return combined.sort_values(["station_id", "date"]).reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert PCD PM2.5 Excel files to long-format CSV "
            "for use with merge_gee_exporttt.py."
        )
    )
    parser.add_argument(
        "--pm-dir", type=Path, default=DEFAULT_PM_DIR,
        help="Directory containing YYYY.xlsx PM2.5 files (default: %(default)s)",
    )
    parser.add_argument(
        "--stations", type=Path, default=DEFAULT_STATIONS,
        help="Air4Thai_station.csv with station_ID, Latitude, Longitude (default: %(default)s)",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help="Output CSV path (default: %(default)s)",
    )
    args = parser.parse_args()

    # ── 1. Load and clean PM2.5 from Excel ───────────────────────────────
    logger.info("Reading PM2.5 Excel files from %s", args.pm_dir)
    pm25 = load_pm25(args.pm_dir)
    logger.info(
        "After cleaning: %d rows  |  %d stations  |  %d unique dates",
        len(pm25), pm25["station_id"].nunique(), pm25["date"].nunique(),
    )

    # ── 2. Attach lat/lon from station registry ───────────────────────────
    stations = pd.read_csv(args.stations)
    stations = (
        stations
        .rename(columns={"station_ID": "station_id",
                         "Latitude": "lat",
                         "Longitude": "lon"})
        [["station_id", "lat", "lon"]]
    )
    stations["station_id"] = stations["station_id"].str.strip().str.lower()

    ids_before = set(pm25["station_id"].unique())
    pm25 = pm25.merge(stations, on="station_id", how="inner")
    ids_after = set(pm25["station_id"].unique())

    dropped_ids = ids_before - ids_after
    if dropped_ids:
        logger.warning(
            "%d station(s) not found in %s and dropped (no lat/lon): %s",
            len(dropped_ids), args.stations.name,
            ", ".join(sorted(dropped_ids)),
        )

    # ── 3. Final column order expected by merge_gee_exporttt.py ──────────
    pm25 = pm25[["station_id", "date", "pm25", "lat", "lon"]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pm25.to_csv(args.out, index=False)
    logger.info(
        "Wrote %d rows  (%d stations, %d dates)  →  %s",
        len(pm25), pm25["station_id"].nunique(),
        pm25["date"].nunique(), args.out,
    )


if __name__ == "__main__":
    main()
