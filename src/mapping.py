"""Spatial mapping: aggregate daily PM2.5 grid predictions to monthly/seasonal."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("pm25_hybrid")


def aggregate_grid_predictions(grid_df: pd.DataFrame,
                               y_pred: np.ndarray,
                               min_valid_days_per_month: int = 7,
                               grid_id_col: str = "grid_id",
                               date_col: str = "date") -> dict[str, pd.DataFrame]:
    """Aggregate daily grid predictions to monthly and seasonal means.

    Returns a dict with keys 'daily', 'monthly', 'seasonal', each holding
    a long-format DataFrame.
    """
    df = grid_df.copy()
    df["pm25_pred"] = y_pred
    df = df.dropna(subset=["pm25_pred"])

    df["year_month"] = df[date_col].dt.to_period("M").astype(str)
    df["season"] = df[date_col].apply(_thai_season)

    # Monthly
    monthly = (df.groupby([grid_id_col, "year_month"])
                 .agg(pm25_pred=("pm25_pred", "mean"),
                      n_days=("pm25_pred", "count"),
                      lat=("lat", "first"),
                      lon=("lon", "first"))
                 .reset_index())
    monthly = monthly[monthly["n_days"] >= min_valid_days_per_month]

    # Seasonal — average over months within the same season-year
    df["season_year"] = df[date_col].dt.year.astype(str) + "_" + df["season"]
    seasonal = (df.groupby([grid_id_col, "season_year"])
                  .agg(pm25_pred=("pm25_pred", "mean"),
                       n_days=("pm25_pred", "count"),
                       lat=("lat", "first"),
                       lon=("lon", "first"))
                  .reset_index())

    return {"daily": df, "monthly": monthly, "seasonal": seasonal}


def save_grids(grids: dict[str, pd.DataFrame], output_dir: str | Path,
               save_daily: bool = False, save_monthly: bool = True,
               save_seasonal: bool = True) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if save_daily:
        grids["daily"].to_parquet(out / "pm25_daily_grid.parquet", index=False)
        logger.info("Saved daily grid -> %s", out / "pm25_daily_grid.parquet")
    if save_monthly:
        grids["monthly"].to_csv(out / "pm25_monthly_grid.csv", index=False)
        logger.info("Saved monthly grid -> %s", out / "pm25_monthly_grid.csv")
    if save_seasonal:
        grids["seasonal"].to_csv(out / "pm25_seasonal_grid.csv", index=False)
        logger.info("Saved seasonal grid -> %s", out / "pm25_seasonal_grid.csv")


def _thai_season(date) -> str:
    m = date.month
    if m in (11, 12, 1, 2):
        return "cool_dry"
    if m in (3, 4, 5):
        return "hot_dry"
    return "wet_monsoon"
