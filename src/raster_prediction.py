"""Raster-based prediction for the PM2.5 hybrid LME-GWR pipeline.

This module is the operational alternative to predicting from a long-format
grid CSV. It reads daily predictor GeoTIFFs (one per predictor, or one stack
with multiple bands), applies a trained model to every valid pixel, and
writes a prediction GeoTIFF per day. NaN pixels (cloud-masked AOD, missing
predictors) propagate through and the corresponding output pixels are NaN.

Expected input layout
---------------------
data/grids/
├── aod/
│   ├── aod_2023-01-01.tif
│   ├── aod_2023-01-02.tif
│   └── ...
├── temp/
│   ├── temp_2023-01-01.tif
│   └── ...
├── rh/, ws/, blh/, prec/, ndvi/, fire_count/
├── elev.tif              # static (single file)
└── road_density.tif      # static (single file)

Filename convention for daily rasters:
    <predictor>_<YYYY-MM-DD>.tif     (default — overridable per predictor)

Outputs
-------
outputs/pm25_predictions/
└── pm25_<YYYY-MM-DD>.tif

Model support
-------------
- Linear & sklearn-style models (PCR, PLSR, RF, GB, XGBoost): apply .predict
  to the flattened pixel matrix.
- LME alone: applies fixed-effect coefficients via vectorized matmul.
- Hybrid LME-GWR: LME baseline + IDW-interpolated GWR coefficient correction
  per day.
"""
from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("pm25_hybrid")

# Module-level cache for static predictor arrays.  Static rasters (elev, pop)
# never change between days; caching them avoids repeated reads over SFTP for
# every predicted day.  Key: absolute path string.
_STATIC_RASTER_CACHE: dict[str, np.ndarray] = {}

# Land mask derived from the SRTM elevation raster.
# Key: "<elev_path>|<width>x<height>".  Value: bool array (True = land pixel).
_LAND_MASK_CACHE: dict[str, np.ndarray | None] = {}


# ---------------------------------------------------------------------
# Optional rasterio import — only required when raster mode is used.
# ---------------------------------------------------------------------
def _import_rasterio():
    try:
        import rasterio
        return rasterio
    except ImportError as e:
        raise ImportError(
            "rasterio is required for raster mode. "
            "Run `pip install rasterio` and retry."
        ) from e


# =====================================================================
# File discovery
# =====================================================================
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")


@dataclass
class RasterPaths:
    """Resolved file paths for one date."""
    date: pd.Timestamp
    daily_paths: dict[str, Path]   # predictor -> file path (daily rasters only)
    static_paths: dict[str, Path]  # predictor -> file path (static rasters)


def discover_dates(grid_dir: Path, daily_predictors: list[str]) -> list[pd.Timestamp]:
    """Return dates where daily predictors have coverage.

    Composite/sparse predictors (e.g. NDVI from MOD13A2 16-day composites) are
    automatically excluded from the intersection — they contain far fewer files
    than truly daily predictors and are forward-filled by resolve_paths_for_date.
    The intersection is computed only over predictors whose file count is at least
    50% of the most-populated predictor folder.
    """
    per_pred_dates: dict[str, set[str]] = {}
    for pred in daily_predictors:
        pred_dir = grid_dir / pred
        if not pred_dir.exists():
            raise FileNotFoundError(f"Missing predictor folder: {pred_dir}")
        dates = set()
        for f in pred_dir.glob(f"{pred}_*.tif"):
            m = DATE_PATTERN.search(f.name)
            if m:
                dates.add(m.group(1))
        per_pred_dates[pred] = dates

    if not per_pred_dates:
        return []

    # Auto-detect composite/sparse predictors: those with < 50% of the max file
    # count.  Sparse predictors are forward-filled by resolve_paths_for_date and
    # must NOT limit the intersection (NDVI has ~75 files vs ~900 for AOD/ERA5).
    max_n = max(len(d) for d in per_pred_dates.values())
    primary_preds = {p for p, d in per_pred_dates.items() if len(d) >= 0.5 * max_n}
    sparse_preds = set(per_pred_dates) - primary_preds
    if sparse_preds:
        logger.info(
            "Sparse predictors detected (forward-filled, excluded from date intersection): %s",
            sorted(sparse_preds),
        )

    primary_dates = {p: per_pred_dates[p] for p in primary_preds}
    common = set.intersection(*primary_dates.values()) if primary_dates else set()
    out = sorted(pd.Timestamp(d) for d in common)

    # Warn on any primary predictors missing some dates
    union = set.union(*primary_dates.values()) if primary_dates else set()
    missing_summary = {
        p: len(union - per_pred_dates[p]) for p in daily_predictors
        if union - per_pred_dates[p]
    }
    if missing_summary:
        logger.warning("Predictors with missing dates (count): %s", missing_summary)

    logger.info("Found %d dates with complete predictor coverage.", len(out))
    return out


def _find_latest_raster(pred_dir: Path, pred: str,
                        date: pd.Timestamp) -> Optional[Path]:
    """Return the most recent raster for `pred` on or before `date`.

    Used to forward-fill sparse/composite predictors (e.g. NDVI from
    MOD13A2 16-day composites).  Returns None if no file exists at all.
    """
    target = date.strftime("%Y-%m-%d")
    candidates = sorted(pred_dir.glob(f"{pred}_*.tif"))
    # Walk backwards to find the newest file whose date <= target
    for path in reversed(candidates):
        m = DATE_PATTERN.search(path.name)
        if m and m.group(1) <= target:
            return path
    return None


def resolve_paths_for_date(date: pd.Timestamp,
                           grid_dir: Path,
                           daily_predictors: list[str],
                           static_predictors: list[str]) -> RasterPaths:
    """Look up file paths for one date.

    For predictors that only have composite/sparse files (e.g. NDVI at
    16-day MOD13A2 cadence), the most recent available raster on or before
    `date` is used as a forward-fill.  A warning is logged the first time
    a forward-fill is applied for a predictor.
    """
    date_str = date.strftime("%Y-%m-%d")
    daily = {}
    for pred in daily_predictors:
        exact = grid_dir / pred / f"{pred}_{date_str}.tif"
        if exact.exists():
            daily[pred] = exact
        else:
            # Forward-fill: use latest composite available on or before date
            filled = _find_latest_raster(grid_dir / pred, pred, date)
            if filled is None:
                raise FileNotFoundError(
                    f"Missing daily raster (and no earlier file to forward-fill): {exact}"
                )
            # Only log the first forward-fill per predictor to avoid log spam
            fill_date = DATE_PATTERN.search(filled.name).group(1)
            if fill_date != date_str:
                logger.debug(
                    "Forward-filling %s on %s with composite from %s",
                    pred, date_str, fill_date,
                )
            daily[pred] = filled
    static = {}
    for pred in static_predictors:
        path = grid_dir / f"{pred}.tif"
        if not path.exists():
            raise FileNotFoundError(f"Missing static raster: {path}")
        static[pred] = path
    return RasterPaths(date=date, daily_paths=daily, static_paths=static)


# =====================================================================
# Reading & stacking
# =====================================================================
def _read_raster(path: Path) -> tuple[np.ndarray, dict]:
    """Read first band of a raster as float32, treating nodata as NaN."""
    rasterio = _import_rasterio()
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).astype(np.float32)
        # masked array → NaN for invalid pixels
        arr = arr.filled(np.nan)
        profile = src.profile.copy()
    return arr, profile


def _read_raster_resampled(path: Path, ref_profile: dict) -> np.ndarray:
    """Read a raster and (if needed) resample/align to reference profile.

    Static predictors are commonly at a different resolution than AOD.
    This function reprojects them onto the reference grid using bilinear
    resampling, returning an array of the same shape as the reference.
    """
    rasterio = _import_rasterio()
    from rasterio.warp import Resampling, reproject

    with rasterio.open(path) as src:
        same_grid = (
            src.crs == ref_profile["crs"]
            and src.transform == ref_profile["transform"]
            and src.width == ref_profile["width"]
            and src.height == ref_profile["height"]
        )
        if same_grid:
            arr = src.read(1, masked=True).astype(np.float32).filled(np.nan)
            return arr

        dst = np.full((ref_profile["height"], ref_profile["width"]),
                      np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=Resampling.bilinear,
        )
        return dst


# =====================================================================
# Land-mask creation and spatial gap-filling
# =====================================================================
def _rasterize_thailand_boundary(ref_profile: dict) -> "np.ndarray | None":
    """Rasterize the project's Thailand boundary shapefile onto the prediction grid.

    Reads data/shp/th_bndry.shp (UTM Zone 47N), reprojects to the raster CRS
    (WGS84/EPSG:4326), and burns all features onto a (H, W) boolean array.
    The result is intersected with the SRTM land mask in _get_or_create_land_mask
    to exclude coastal sea pixels.

    Returns a bool (H, W) array (True = inside Thailand) or None on failure.
    """
    try:
        import geopandas as gpd
        from rasterio.features import rasterize as _rasterize

        shp_path = Path(__file__).resolve().parent.parent / "data" / "shp" / "th_bndry.shp"
        if not shp_path.exists():
            logger.warning("Thailand boundary shapefile not found at %s — "
                           "falling back to SRTM land mask.", shp_path)
            return None

        gdf = gpd.read_file(str(shp_path))

        # Reproject from UTM Zone 47N to raster CRS (WGS84 geographic).
        # Convert rasterio CRS object → EPSG string so geopandas accepts it.
        raster_crs = ref_profile.get("crs")
        if raster_crs is None:
            target_crs = "EPSG:4326"
        else:
            try:
                epsg = raster_crs.to_epsg()
                target_crs = f"EPSG:{epsg}" if epsg else raster_crs.to_wkt()
            except Exception:
                target_crs = "EPSG:4326"
        gdf = gdf.to_crs(target_crs)

        H = ref_profile["height"]
        W = ref_profile["width"]
        shapes = [(geom, 1) for geom in gdf.geometry
                  if geom is not None and geom.is_valid]
        if not shapes:
            logger.warning("Thailand boundary shapefile has no valid geometries.")
            return None

        mask = _rasterize(
            shapes,
            out_shape=(H, W),
            transform=ref_profile["transform"],
            fill=0,
            dtype=np.uint8,
            all_touched=True,   # keep boundary pixels (important for thin Kra Isthmus)
        )
        logger.info("Thailand boundary rasterized from %s — %d pixels (%.1f%% of grid)",
                    shp_path.name, int(mask.sum()), 100.0 * mask.sum() / mask.size)
        return mask.astype(bool)

    except Exception as exc:
        logger.warning("Thailand boundary rasterization failed (%s) — "
                       "falling back to SRTM land mask.", exc)
        return None


def _get_or_create_land_mask(elev_path: Path,
                              ref_profile: dict) -> "np.ndarray | None":
    """Return a boolean (H, W) mask covering Thailand's land territory only.

    Two-step approach:
      1. Rasterize Thailand's national boundary polygon (geopandas Natural Earth).
      2. Intersect with the SRTM elevation mask (NaN = ocean / no data).
         This removes sea pixels that lie within Thailand's convex hull
         (e.g. the Gulf of Thailand, Andaman Sea inlets).

    If geopandas is unavailable the SRTM mask alone is used, which includes
    all land in the prediction bounding box (neighbouring countries).
    The mask is cached in memory after the first call.
    """
    cache_key = f"{elev_path}|{ref_profile.get('width')}x{ref_profile.get('height')}"
    if cache_key not in _LAND_MASK_CACHE:
        # --- Step 1: SRTM-derived land mask (ocean = NaN) ---
        srtm_mask: "np.ndarray | None" = None
        if Path(elev_path).exists():
            try:
                arr = _read_raster_resampled(Path(elev_path), ref_profile)
                srtm_mask = ~np.isnan(arr)
            except Exception as exc:
                logger.warning("Could not read SRTM elev for land mask: %s", exc)

        # --- Step 2: Thailand boundary polygon mask ---
        thai_mask = _rasterize_thailand_boundary(ref_profile)

        # --- Combine: Thailand boundary ∩ SRTM land ---
        if thai_mask is not None and srtm_mask is not None:
            mask = thai_mask & srtm_mask
            logger.info(
                "Thailand land mask: %d pixels (boundary ∩ SRTM, %.1f%% of grid)",
                int(mask.sum()), 100.0 * mask.sum() / mask.size,
            )
        elif thai_mask is not None:
            mask = thai_mask
            logger.info(
                "Thailand land mask: %d pixels (boundary only, %.1f%% of grid)",
                int(mask.sum()), 100.0 * mask.sum() / mask.size,
            )
        elif srtm_mask is not None:
            mask = srtm_mask
            logger.warning(
                "Thailand boundary unavailable — using SRTM mask (%d pixels, "
                "includes neighbouring countries).", int(mask.sum()),
            )
        else:
            mask = None
            logger.warning("No land mask available — output will not be clipped "
                           "to Thailand territory.")

        _LAND_MASK_CACHE[cache_key] = mask

    return _LAND_MASK_CACHE[cache_key]


def fill_gaps_in_raster(arr: np.ndarray,
                         land_mask: "np.ndarray | None" = None,
                         k: int = 10,
                         power: float = 2.0) -> np.ndarray:
    """Fill NaN land pixels using Inverse Distance Weighting (IDW).

    IDW is the standard spatial gap-filling method for cloud-masked PM2.5
    and AOD rasters (He et al. 2019 Atmos. Environ.; Wei et al. 2021 STOTEN;
    Stafoggia et al. 2019 Environ. Int.).  Each NaN land pixel is assigned the
    distance-weighted mean of its k spatially nearest valid pixels:

        ŷ = Σ w_i · y_i / Σ w_i,  w_i = 1 / (d_i^power + ε)

    where d_i is the Euclidean pixel distance to valid source i.

    Parameters
    ----------
    arr       : (H, W) float32 prediction array; NaN where predictors were missing.
    land_mask : (H, W) bool array from SRTM — True = land pixel.
                If None, fills all NaN pixels without a domain constraint.
    k         : number of nearest valid pixels to use for IDW (default 10).
    power     : distance decay exponent — 2 gives the classic IDW² formulation.

    Returns
    -------
    Gap-filled float32 array of the same shape.
    """
    from scipy.spatial import cKDTree

    valid = ~np.isnan(arr)
    fill_target = (~valid & land_mask) if land_mask is not None else ~valid

    if not fill_target.any():
        result = arr.copy()
        if land_mask is not None:
            result[~land_mask] = np.nan
        return result.astype(np.float32)

    if not valid.any():
        logger.debug("fill_gaps_in_raster: no valid source pixels — returning unchanged.")
        return arr

    H, W = arr.shape
    rows, cols = np.indices((H, W))

    # Source pixels: locations with valid predictions
    src_pts = np.column_stack([rows[valid].ravel(), cols[valid].ravel()]).astype(np.float32)
    src_vals = arr[valid].ravel().astype(np.float32)

    # Target pixels: NaN within the land boundary
    tgt_pts = np.column_stack(
        [rows[fill_target].ravel(), cols[fill_target].ravel()]
    ).astype(np.float32)

    tree = cKDTree(src_pts)
    k_use = min(k, len(src_pts))
    dists, idxs = tree.query(tgt_pts, k=k_use)

    if k_use == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    # IDW weights and weighted sum
    weights = np.float32(1.0) / (dists.astype(np.float32) ** power + np.float32(1e-9))
    weights /= weights.sum(axis=1, keepdims=True)
    filled_vals = (weights * src_vals[idxs]).sum(axis=1)

    filled = arr.copy()
    filled[fill_target] = filled_vals

    if land_mask is not None:
        filled[~land_mask] = np.nan

    return filled.astype(np.float32)


def stack_predictors(paths: RasterPaths,
                     predictor_order: list[str]) -> tuple[np.ndarray, dict]:
    """Read all predictor rasters for one day and stack into (H, W, P).

    The first daily predictor in ``predictor_order`` defines the reference
    grid. Subsequent daily and static rasters are resampled to that grid
    if they don't already match.

    Static predictors are served from ``_STATIC_RASTER_CACHE`` after the
    first read so subsequent days skip the SFTP round-trip entirely.
    """
    if not predictor_order:
        raise ValueError("predictor_order is empty.")

    layers: list[np.ndarray] = []
    ref_profile: dict | None = None

    for pred in predictor_order:
        is_static = pred in paths.static_paths
        if pred in paths.daily_paths:
            path = paths.daily_paths[pred]
        elif is_static:
            path = paths.static_paths[pred]
        else:
            raise KeyError(f"Predictor '{pred}' has no resolved path.")

        if ref_profile is None:
            arr, ref_profile = _read_raster(path)
        elif is_static:
            # Serve static rasters from cache (keyed by path + ref profile dims)
            cache_key = f"{path}|{ref_profile.get('width')}x{ref_profile.get('height')}"
            if cache_key in _STATIC_RASTER_CACHE:
                arr = _STATIC_RASTER_CACHE[cache_key]
            else:
                arr = _read_raster_resampled(path, ref_profile)
                _STATIC_RASTER_CACHE[cache_key] = arr
        else:
            arr = _read_raster_resampled(path, ref_profile)

        # ERA5-Land temperature_2m is exported in Kelvin by gee_export_grids.py,
        # but merge_gee_exports.py converts to Celsius (subtract 273.15) for the
        # training panel.  Auto-detect Kelvin rasters (median > 200 K) and convert
        # so the scaler applied below sees the same Celsius scale as training.
        if pred == "t2m":
            finite = arr[np.isfinite(arr)]
            if finite.size > 0 and float(np.median(finite)) > 200.0:
                arr = arr - np.float32(273.15)

        layers.append(arr)

    X = np.stack(layers, axis=-1)  # (H, W, P)
    return X, ref_profile


# =====================================================================
# Writing
# =====================================================================
def write_prediction_raster(y_grid: np.ndarray, ref_profile: dict,
                            out_path: Path) -> None:
    """Write a 2-D prediction array as a single-band float32 GeoTIFF."""
    rasterio = _import_rasterio()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = ref_profile.copy()
    profile.update(dtype="float32", count=1, nodata=np.float32(-9999),
                   compress="deflate")
    # Tiled output requires height/width to be multiples of 16; fall back
    # to stripped output when the grid is smaller than that.
    H, W = y_grid.shape
    if H >= 16 and W >= 16:
        profile.update(tiled=True, blockxsize=16, blockysize=16)
    arr = np.where(np.isnan(y_grid), -9999.0, y_grid).astype(np.float32)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)


# =====================================================================
# Per-pixel prediction core
# =====================================================================
def _flatten_and_mask(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """(H, W, P) -> ((N, P), valid_mask, (H, W))."""
    H, W, P = X.shape
    X_flat = X.reshape(-1, P)
    valid = ~np.isnan(X_flat).any(axis=1)
    return X_flat, valid, (H, W)


# Maximum allowed standardised predictor value.  Raster predictors can be far
# outside the training distribution — GPWv4 population density at Bangkok's
# dense urban core is >19,000 persons/km² while training-panel station values
# top out at ~250 persons/km² (~555 SD out of range).  Without this guard,
# any positive LME/GWR coefficient for pop produces unrealistically high
# predictions exactly over Bangkok.  Clipping to ±_SIGMA_CLIP retains all
# physically realistic variation while preventing linear extrapolation artefacts.
_SIGMA_CLIP: float = 5.0


def _scale_and_clip(scaler, Xv: np.ndarray) -> np.ndarray:
    """Standardise Xv with scaler then clip each predictor to ±_SIGMA_CLIP.

    Clipping after scaling prevents out-of-training-distribution raster
    predictors (population density, elevation extremes) from producing
    implausible PM2.5 predictions through linear extrapolation.
    """
    Xv = scaler.transform(Xv).astype(np.float32)
    return np.clip(Xv, -_SIGMA_CLIP, _SIGMA_CLIP)


def _predict_sklearn(model, X: np.ndarray, scaler=None) -> np.ndarray:
    """Predict on a (H, W, P) stack with any sklearn-style model."""
    X_flat, valid, (H, W) = _flatten_and_mask(X)
    y = np.full(X_flat.shape[0], np.nan, dtype=np.float32)
    if valid.any():
        Xv = X_flat[valid]
        if scaler is not None:
            Xv = _scale_and_clip(scaler, Xv)
        y[valid] = model.predict(Xv).astype(np.float32)
    return y.reshape(H, W)


# =====================================================================
# LME raster prediction (fixed-effects only)
# =====================================================================
def predict_lme_raster(lme_model_obj, X: np.ndarray,
                       predictors: list[str],
                       scaler=None) -> np.ndarray:
    """Apply LME fixed-effect coefficients to a predictor stack.

    Off-network grid cells have no station random effect, so the marginal
    (population-level) prediction is the appropriate baseline. We extract
    the fixed-effect coefficients directly from the fitted statsmodels
    result and do the matrix multiplication ourselves — much faster than
    constructing a DataFrame for every pixel and calling .predict().
    """
    if lme_model_obj.result_ is None:
        raise RuntimeError("LME model has not been fitted yet.")
    fe_params = lme_model_obj.result_.fe_params  # pandas Series

    intercept = float(fe_params.get("Intercept", 0.0))
    coefs = np.array([float(fe_params.get(p, 0.0)) for p in predictors],
                     dtype=np.float32)

    X_flat, valid, (H, W) = _flatten_and_mask(X)
    y = np.full(X_flat.shape[0], np.nan, dtype=np.float32)
    if valid.any():
        Xv = X_flat[valid]
        if scaler is not None:
            Xv = _scale_and_clip(scaler, Xv)
        y[valid] = (intercept + Xv.astype(np.float32) @ coefs).astype(np.float32)
    return y.reshape(H, W)


# =====================================================================
# Hybrid LME-GWR raster prediction
# =====================================================================
def predict_hybrid_raster_day(hybrid_obj,
                              date: pd.Timestamp,
                              X: np.ndarray,
                              predictors: list[str],
                              ref_profile: dict,
                              scaler=None,
                              k_neighbors: int = 8,
                              power: float = 2.0) -> tuple[np.ndarray, bool]:
    """Predict one day's PM2.5 surface with the hybrid LME-GWR model.

    Supports both HybridLMEGWR (Stage-1 = LME) and HybridTreeGWR
    (Stage-1 = sklearn tree model such as RF/GB/XGBoost).

    Returns
    -------
    y_grid    : (H, W) prediction array
    used_gwr  : True if the GWR correction was applied that day
    """
    # Stage-1: branch on model type
    if hasattr(hybrid_obj, "tree_model"):
        # HybridTreeGWR — tree model was trained on standardised predictors
        y_stage1 = _predict_sklearn(hybrid_obj.tree_model, X, scaler=scaler)
    else:
        # HybridLMEGWR — fixed-effect LME baseline
        y_stage1 = predict_lme_raster(hybrid_obj.lme, X, predictors, scaler=scaler)
    y_lme = y_stage1  # used below; name kept for clarity in the GWR delta section

    daily_results = hybrid_obj.gwr_output_.daily_results if hybrid_obj.gwr_output_ else {}
    res = daily_results.get(pd.Timestamp(date))
    if res is None or not res.fitted or res.coefficients is None:
        return y_lme, False

    # Zero-variance predictors are dropped during GWR fitting (see gwr_model.py),
    # so res.coefficients may have fewer columns than the full predictor set.
    # Use only the columns actually present in this day's result.
    _meta = {"lon", "lat", "station_id"}
    actual_coef_cols = [c for c in res.coefficients.columns if c not in _meta]
    active_preds = actual_coef_cols[1:]  # strip leading "intercept"

    station_xy = res.coefficients[["lon", "lat"]].to_numpy()
    station_coefs = res.coefficients[actual_coef_cols].to_numpy(dtype=np.float32)

    grid_xy = _pixel_coordinates(X.shape[:2], ref_profile)
    coef_grid = _idw_interpolate(grid_xy, station_xy, station_coefs,
                                 k=k_neighbors, power=power)

    X_flat, valid, (H, W) = _flatten_and_mask(X)
    delta = np.full(X_flat.shape[0], np.nan, dtype=np.float32)
    if valid.any():
        Xv = X_flat[valid]
        if scaler is not None:
            Xv = _scale_and_clip(scaler, Xv)
        # Build augmented matrix for active predictors only (matches coef_grid columns)
        pred_idx = [predictors.index(p) for p in active_preds]
        Xv_aug = np.column_stack(
            [np.ones(len(Xv), dtype=np.float32), Xv[:, pred_idx].astype(np.float32)]
        )
        delta[valid] = np.einsum("ij,ij->i", coef_grid[valid].astype(np.float32), Xv_aug)

    return (y_lme + delta.reshape(H, W)).astype(np.float32), True


# =====================================================================
# Geometry helpers
# =====================================================================
def _pixel_coordinates(shape: tuple[int, int], profile: dict) -> np.ndarray:
    """Return an (N, 2) array of (lon, lat) for every pixel center.

    Pixel coordinates are computed from the affine transform stored in
    the rasterio profile. If the CRS is not geographic (lon/lat), the
    coordinates are reprojected so they match the station coordinate
    system used for GWR fitting (assumed lon/lat / EPSG:4326).
    """
    rasterio = _import_rasterio()
    from rasterio.transform import xy
    from rasterio.warp import transform as warp_transform

    H, W = shape
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    xs, ys = xy(profile["transform"], rows.ravel(), cols.ravel(), offset="center")
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    crs = profile.get("crs")
    if crs is not None and crs.to_epsg() != 4326:
        xs, ys = warp_transform(crs, "EPSG:4326",
                                xs.tolist(), ys.tolist())
        xs = np.asarray(xs)
        ys = np.asarray(ys)
    return np.column_stack([xs, ys])


def _idw_interpolate(target_xy: np.ndarray, source_xy: np.ndarray,
                     source_values: np.ndarray, k: int = 8,
                     power: float = 2.0, eps: float = 1e-9) -> np.ndarray:
    """Inverse-distance interpolation of (M, P) values at M sources to N targets.

    Uses k nearest sources per target. Returns an (N, P) array.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(source_xy)
    k_use = min(k, len(source_xy))
    dists, idxs = tree.query(target_xy, k=k_use)
    if k_use == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]
    weights = 1.0 / (dists ** power + eps)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return np.einsum("nk,nkp->np", weights.astype(np.float32),
                     source_values[idxs].astype(np.float32))


# =====================================================================
# Per-day worker (runs in a thread — model is shared by reference)
# =====================================================================
def _predict_day_task(date: pd.Timestamp,
                      out_path: Path,
                      grid_dir: Path,
                      daily_predictors: list[str],
                      static_predictors: list[str],
                      predictors: list[str],
                      model_kind: str,
                      model,
                      scaler,
                      k_neighbors: int,
                      skip_existing: bool) -> tuple[pd.Timestamp, str, bool]:
    """Predict one day and write result. Returns (date, status, used_gwr)."""
    if skip_existing and out_path.exists():
        return date, "skipped", False
    try:
        paths = resolve_paths_for_date(date, grid_dir, daily_predictors, static_predictors)
        X, ref_profile = stack_predictors(paths, predictors)
    except FileNotFoundError as exc:
        logger.warning("Skipping %s: %s", date.date(), exc)
        return date, "missing", False

    used_gwr = False
    if model_kind == "lme":
        y_grid = predict_lme_raster(model, X, predictors, scaler=scaler)
    elif model_kind == "hybrid":
        y_grid, used_gwr = predict_hybrid_raster_day(
            model, date, X, predictors, ref_profile, scaler=scaler,
            k_neighbors=k_neighbors)
    else:
        y_grid = _predict_sklearn(model, X, scaler=scaler)

    # Physical bounds: PM2.5 cannot be negative; values > 500 µg/m³ are
    # implausible for Thailand and indicate GWR extrapolation artefacts.
    y_grid = np.clip(y_grid, np.float32(0.0), np.float32(500.0))

    # IDW gap-fill: fill cloud-masked NaN land pixels from surrounding valid
    # pixels.  Land mask from SRTM is cached after the first day.
    elev_path = paths.static_paths.get("elev")
    if elev_path is not None:
        land_mask = _get_or_create_land_mask(Path(elev_path), ref_profile)
        if land_mask is not None:
            y_grid = fill_gaps_in_raster(y_grid, land_mask)

    write_prediction_raster(y_grid, ref_profile, out_path)
    return date, "written", used_gwr


# =====================================================================
# Driver: predict every available day
# =====================================================================
def predict_all_days_raster(model,
                            grid_dir: str | Path,
                            output_dir: str | Path,
                            predictors: list[str],
                            daily_predictors: list[str],
                            static_predictors: list[str],
                            model_kind: str = "sklearn",
                            scaler=None,
                            dates: Optional[Iterable[pd.Timestamp]] = None,
                            skip_existing: bool = True,
                            k_neighbors: int = 8,
                            n_jobs: int = -1) -> tuple[list[Path], pd.DataFrame]:
    """Loop over dates, predict each day's PM2.5 surface, and save GeoTIFFs.

    Parameters
    ----------
    model            : fitted model (LME, hybrid, or any sklearn-like benchmark)
    grid_dir         : root folder containing predictor sub-folders
    output_dir       : where to write pm25_<date>.tif files
    predictors       : full ordered list of predictor names matching training
    daily_predictors : subset of `predictors` that vary per day
    static_predictors: subset of `predictors` that are static rasters
    model_kind       : 'sklearn', 'lme', or 'hybrid'
    scaler           : optional sklearn StandardScaler fitted on the training panel
    dates            : optional iterable of dates to process (default = all available)
    skip_existing    : skip dates whose output already exists

    Returns
    -------
    Tuple of (written_paths, day_log_df).
    day_log_df columns: date, status ('written'|'skipped'|'missing'), used_gwr (bool).
    """
    grid_dir = Path(grid_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dates is None:
        dates = discover_dates(grid_dir, daily_predictors)
    else:
        dates = sorted(pd.Timestamp(d) for d in dates)

    # k_neighbors is passed in as a parameter (config: k_neighbors_idw)
    n_workers = os.cpu_count() if n_jobs < 0 else max(1, n_jobs)
    logger.info("Raster prediction: %d dates, %d worker threads", len(dates), n_workers)

    written: list[Path] = []
    day_log_rows: list[dict] = []
    n_used_gwr = 0
    n_lme_only = 0
    n_skipped = 0
    n_total = len(dates)

    tasks = {
        date: output_dir / f"pm25_{date.strftime('%Y-%m-%d')}.tif"
        for date in dates
    }

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_map = {
            executor.submit(
                _predict_day_task,
                date, out_path, grid_dir, daily_predictors, static_predictors,
                predictors, model_kind, model, scaler, k_neighbors, skip_existing,
            ): date
            for date, out_path in tasks.items()
        }

        for i, future in enumerate(as_completed(future_map), start=1):
            try:
                date, status, used_gwr = future.result()
            except Exception as exc:
                logger.error("Day %s failed: %s", future_map[future].date(), exc)
                continue

            day_log_rows.append({"date": date, "status": status, "used_gwr": used_gwr})
            if status == "written":
                written.append(tasks[date])
                if used_gwr:
                    n_used_gwr += 1
                else:
                    n_lme_only += 1
            elif status == "skipped":
                n_skipped += 1

            if i % 50 == 0 or i == n_total:
                logger.info("Processed %d / %d days", i, n_total)

    if model_kind == "hybrid":
        logger.info("Hybrid prediction summary: %d days with GWR correction, "
                    "%d days LME-only.", n_used_gwr, n_lme_only)
    if n_skipped:
        logger.info("Skipped %d days (output already existed).", n_skipped)
    logger.info("Wrote %d prediction rasters to %s", len(written), output_dir)
    day_log_df = (pd.DataFrame(day_log_rows)
                  .sort_values("date")
                  .reset_index(drop=True))
    return written, day_log_df


# =====================================================================
# Aggregation: daily GeoTIFFs -> monthly / seasonal mean GeoTIFFs
# =====================================================================
def aggregate_daily_rasters(daily_dir: str | Path,
                            output_dir: str | Path,
                            min_valid_days_per_month: int = 7,
                            save_monthly: bool = True,
                            save_seasonal: bool = True,
                            land_mask_path: "str | Path | None" = None,
                            ) -> dict[str, list[Path]]:
    """Average daily prediction GeoTIFFs into monthly and seasonal means.

    For each month, only pixels with at least ``min_valid_days_per_month``
    valid (non-NaN) days contribute a monthly mean.  Any remaining NaN pixels
    within Thailand's land boundary are then filled by nearest-neighbor
    interpolation from surrounding valid pixels (``land_mask_path`` required).
    """
    rasterio = _import_rasterio()
    daily_dir = Path(daily_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(daily_dir.glob("pm25_*.tif"))
    if not files:
        logger.warning("No daily prediction rasters found in %s", daily_dir)
        return {"monthly": [], "seasonal": []}

    by_month: dict[str, list[Path]] = {}
    by_season: dict[str, list[Path]] = {}
    for f in files:
        m = DATE_PATTERN.search(f.name)
        if not m:
            continue
        date = pd.Timestamp(m.group(1))
        ym = f"{date.year}-{date.month:02d}"
        season = _thai_season(date)
        sy = f"{date.year}_{season}"
        by_month.setdefault(ym, []).append(f)
        by_season.setdefault(sy, []).append(f)

    monthly_paths: list[Path] = []
    seasonal_paths: list[Path] = []

    def _apply_gap_fill(arr, profile):
        if land_mask_path is None:
            return arr
        lm = _get_or_create_land_mask(Path(land_mask_path), profile)
        return fill_gaps_in_raster(arr, lm) if lm is not None else arr

    if save_monthly:
        for ym, paths in sorted(by_month.items()):
            mean_arr, profile = _stack_and_mean(paths, min_valid=min_valid_days_per_month)
            mean_arr = _apply_gap_fill(mean_arr, profile)
            out = output_dir / f"pm25_monthly_{ym}.tif"
            write_prediction_raster(mean_arr, profile, out)
            monthly_paths.append(out)
        logger.info("Wrote %d monthly mean rasters.", len(monthly_paths))

    if save_seasonal:
        for sy, paths in sorted(by_season.items()):
            mean_arr, profile = _stack_and_mean(paths, min_valid=min_valid_days_per_month)
            mean_arr = _apply_gap_fill(mean_arr, profile)
            out = output_dir / f"pm25_seasonal_{sy}.tif"
            write_prediction_raster(mean_arr, profile, out)
            seasonal_paths.append(out)
        logger.info("Wrote %d seasonal mean rasters.", len(seasonal_paths))

    return {"monthly": monthly_paths, "seasonal": seasonal_paths}


def _stack_and_mean(paths: list[Path], min_valid: int) -> tuple[np.ndarray, dict]:
    rasterio = _import_rasterio()
    arrs = []
    profile = None
    for p in paths:
        with rasterio.open(p) as src:
            a = src.read(1, masked=True).astype(np.float32).filled(np.nan)
            if profile is None:
                profile = src.profile.copy()
        arrs.append(a)
    stack = np.stack(arrs, axis=0)  # (D, H, W)
    valid_count = np.sum(~np.isnan(stack), axis=0)
    # Pixels that are NaN on every day produce a harmless 'all-NaN slice'
    # warning from np.nanmean — silence it since we mask those pixels below.
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(stack, axis=0)
    mean[valid_count < min_valid] = np.nan
    return mean, profile


def _thai_season(date) -> str:
    m = date.month
    if m in (11, 12, 1, 2):
        return "cool_dry"
    if m in (3, 4, 5):
        return "hot_dry"
    return "wet_monsoon"


# =====================================================================
# Annual aggregation
# =====================================================================
def aggregate_annual_rasters(
    daily_dir: str | Path,
    output_dir: str | Path,
    min_valid_days_per_year: int = 180,
    land_mask_path: "str | Path | None" = None,
) -> list[Path]:
    """Average daily prediction GeoTIFFs into annual mean rasters (one per year).

    Pixels with fewer than ``min_valid_days_per_year`` valid days are set to NaN;
    remaining NaN land pixels are then gap-filled from surrounding valid pixels.
    """
    daily_dir = Path(daily_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_year: dict[int, list[Path]] = {}
    for f in sorted(daily_dir.glob("pm25_*.tif")):
        m = DATE_PATTERN.search(f.name)   # only matches YYYY-MM-DD filenames
        if m:
            by_year.setdefault(pd.Timestamp(m.group(1)).year, []).append(f)

    if not by_year:
        logger.warning("No daily prediction rasters found in %s for annual aggregation.",
                       daily_dir)
        return []

    annual_paths: list[Path] = []
    for year, paths in sorted(by_year.items()):
        mean_arr, profile = _stack_and_mean(paths, min_valid=min_valid_days_per_year)
        if land_mask_path is not None:
            lm = _get_or_create_land_mask(Path(land_mask_path), profile)
            if lm is not None:
                mean_arr = fill_gaps_in_raster(mean_arr, lm)
        out = output_dir / f"pm25_annual_{year}.tif"
        write_prediction_raster(mean_arr, profile, out)
        annual_paths.append(out)
        valid_pct = float(np.mean(~np.isnan(mean_arr))) * 100
        logger.info(
            "Wrote annual mean %d (%d days used, %.1f%% valid pixels) -> %s",
            year, len(paths), valid_pct, out.name,
        )
    logger.info("Wrote %d annual mean rasters.", len(annual_paths))
    return annual_paths


# =====================================================================
# WHO / Thailand NAAQS exceedance fraction maps
# =====================================================================
# Reference thresholds (μg/m³).
# WHO AQG 2021: https://www.who.int/news-room/feature-stories/detail/what-are-the-who-air-quality-guidelines
# Thailand NAAQS (Pollution Control Department, revised 2021)
EXCEEDANCE_THRESHOLDS_24HR: dict[str, float] = {
    "who_24hr":   15.0,   # WHO 2021 24-hr PM2.5 guideline
    "thai_24hr":  37.5,   # Thailand 24-hr NAAQS
}


def compute_exceedance_fraction_maps(
    daily_dir: str | Path,
    output_dir: str | Path,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Path]:
    """For each threshold, compute the fraction of days where daily PM2.5 exceeds it.

    Streams all daily pm25_YYYY-MM-DD.tif files one at a time to avoid loading
    all ~1,095 rasters into memory at once.

    Output: one GeoTIFF per threshold → pm25_exceedance_<name>.tif
    Pixel values are fractions in [0, 1]; NaN where all days are cloud-masked.
    These maps are used in the paper to show spatial distribution of pollution
    days relative to WHO and national air-quality standards.
    """
    rasterio = _import_rasterio()
    daily_dir = Path(daily_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if thresholds is None:
        thresholds = EXCEEDANCE_THRESHOLDS_24HR

    daily_files = [f for f in sorted(daily_dir.glob("pm25_*.tif"))
                   if DATE_PATTERN.search(f.name)]
    if not daily_files:
        logger.warning("No daily rasters in %s — skipping exceedance maps.", daily_dir)
        return {}

    logger.info("Computing exceedance fraction maps from %d daily rasters ...",
                len(daily_files))

    exceedance_counts: dict[str, np.ndarray | None] = {n: None for n in thresholds}
    valid_count: np.ndarray | None = None
    profile: dict | None = None

    for f in daily_files:
        with rasterio.open(f) as src:
            arr = src.read(1, masked=True).astype(np.float32).filled(np.nan)
            if profile is None:
                profile = src.profile.copy()

        valid_mask = ~np.isnan(arr)
        if valid_count is None:
            valid_count = valid_mask.astype(np.int32)
            for name in thresholds:
                exceedance_counts[name] = np.zeros(arr.shape, dtype=np.int32)
        else:
            valid_count += valid_mask.astype(np.int32)

        for name, thr in thresholds.items():
            exceedance_counts[name] += ((arr > thr) & valid_mask).astype(np.int32)

    if valid_count is None or profile is None:
        return {}

    out_paths: dict[str, Path] = {}
    for name, exc_arr in exceedance_counts.items():
        with np.errstate(invalid="ignore"):
            fraction = np.where(
                valid_count > 0,
                exc_arr.astype(np.float32) / valid_count.clip(min=1),
                np.nan,
            ).astype(np.float32)
        fraction[valid_count == 0] = np.nan

        out = output_dir / f"pm25_exceedance_{name}.tif"
        write_prediction_raster(fraction, profile, out)
        out_paths[name] = out

        valid_vals = fraction[~np.isnan(fraction)]
        if valid_vals.size > 0:
            logger.info(
                "Exceedance [%s > %.1f μg/m³]: mean %.1f%% of days, "
                "%.1f%% of land area exceeds on ≥1 day",
                name, thresholds[name],
                float(np.mean(valid_vals)) * 100,
                float(np.mean(valid_vals > 0)) * 100,
            )
        logger.info("Wrote exceedance map -> %s", out.name)

    return out_paths


# =====================================================================
# Spatial summary statistics CSV
# =====================================================================
def _parse_raster_period(path: Path) -> tuple[str, str] | None:
    """Parse (period_type, period_label) from aggregated raster filename."""
    name = path.stem
    m = re.search(r"pm25_annual_(\d{4})$", name)
    if m:
        return "annual", m.group(1)
    m = re.search(r"pm25_monthly_(\d{4}-\d{2})$", name)
    if m:
        return "monthly", m.group(1)
    m = re.search(r"pm25_seasonal_(.+)$", name)
    if m:
        return "seasonal", m.group(1)
    m = re.search(r"pm25_exceedance_(.+)$", name)
    if m:
        return "exceedance", m.group(1)
    return None


def compute_spatial_stats_csv(
    raster_dir: str | Path,
    output_csv: str | Path,
    pop_raster_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compute spatial summary statistics for all aggregated prediction rasters.

    Processes pm25_annual_*, pm25_monthly_*, pm25_seasonal_*, and
    pm25_exceedance_* GeoTIFFs in ``raster_dir``.

    For each raster the following statistics are computed over all valid pixels:
        mean, std, min, p5, p25, p50 (median), p75, p95, max,
        valid_pixels, total_pixels, valid_fraction

    If ``pop_raster_path`` is provided (GPWv4 population density GeoTIFF),
    a population-weighted mean is also computed per raster.
    For PM2.5 rasters this is the exposure-weighted concentration (μg/m³).
    For exceedance rasters (values 0-1) it is the population-weighted fraction
    of days on which residents are exposed above the threshold.

    The resulting CSV forms the basis for paper tables and is referenced in the
    Results section for annual/seasonal mean PM2.5 and health exposure statistics.
    """
    rasterio = _import_rasterio()
    raster_dir = Path(raster_dir)
    output_csv = Path(output_csv)

    files: list[Path] = []
    for pat in ("pm25_annual_*.tif", "pm25_monthly_*.tif",
                "pm25_seasonal_*.tif", "pm25_exceedance_*.tif"):
        files.extend(raster_dir.glob(pat))
    files = sorted(set(files))

    if not files:
        logger.warning("No aggregated rasters found in %s — skipping stats CSV.", raster_dir)
        return pd.DataFrame()

    # Population array cached after first load (resampled to prediction grid)
    _pop_cache: dict[str, np.ndarray | None] = {}

    def _get_pop(ref_profile: dict) -> np.ndarray | None:
        if pop_raster_path is None:
            return None
        key = f"{ref_profile.get('width')}x{ref_profile.get('height')}"
        if key not in _pop_cache:
            try:
                raw = _read_raster_resampled(Path(pop_raster_path), ref_profile)
                _pop_cache[key] = np.where(
                    np.isnan(raw) | (raw < 0), 0.0, raw,
                ).astype(np.float32)
                logger.info("Loaded population raster for population-weighted statistics.")
            except Exception as exc:
                logger.warning("Could not load population raster (%s) — "
                               "pop_weighted_mean will be NaN.", exc)
                _pop_cache[key] = None
        return _pop_cache[key]

    rows = []
    for f in files:
        parsed = _parse_raster_period(f)
        if parsed is None:
            continue
        period_type, period_label = parsed

        with rasterio.open(f) as src:
            arr = src.read(1, masked=True).astype(np.float32).filled(np.nan)
            ref_profile = src.profile.copy()

        flat = arr.ravel()
        valid_mask_1d = ~np.isnan(flat)
        valid = flat[valid_mask_1d]
        total_pixels = int(flat.size)
        valid_pixels = int(valid.size)

        row: dict = {
            "period_type":    period_type,
            "period":         period_label,
            "mean":           float(np.mean(valid))            if valid.size else np.nan,
            "std":            float(np.std(valid))             if valid.size else np.nan,
            "min":            float(np.min(valid))             if valid.size else np.nan,
            "p5":             float(np.percentile(valid,  5))  if valid.size else np.nan,
            "p25":            float(np.percentile(valid, 25))  if valid.size else np.nan,
            "p50":            float(np.percentile(valid, 50))  if valid.size else np.nan,
            "p75":            float(np.percentile(valid, 75))  if valid.size else np.nan,
            "p95":            float(np.percentile(valid, 95))  if valid.size else np.nan,
            "max":            float(np.max(valid))             if valid.size else np.nan,
            "valid_pixels":   valid_pixels,
            "total_pixels":   total_pixels,
            "valid_fraction": valid_pixels / total_pixels if total_pixels > 0 else 0.0,
            "pop_weighted_mean": np.nan,
        }

        pop_arr = _get_pop(ref_profile)
        if pop_arr is not None:
            pop_flat = pop_arr.ravel()
            pop_valid = pop_flat[valid_mask_1d]
            pm25_valid = flat[valid_mask_1d]
            pop_sum = float(pop_valid.sum())
            if pop_sum > 0:
                row["pop_weighted_mean"] = float(
                    np.sum(pop_valid * pm25_valid) / pop_sum
                )

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        sort_order = {"annual": 0, "seasonal": 1, "monthly": 2, "exceedance": 3}
        df["_sort"] = df["period_type"].map(sort_order).fillna(4)
        df = (df.sort_values(["_sort", "period"])
                .drop(columns="_sort")
                .reset_index(drop=True))

    df.to_csv(output_csv, index=False)
    logger.info("Saved spatial stats for %d rasters -> %s", len(df), output_csv.name)
    return df
