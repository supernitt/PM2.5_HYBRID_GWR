from __future__ import annotations

import re
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "outputs" / "pm25_predictions"
FIGS = ROOT / "output" / "manuscript" / "figs"
AN = ROOT / "output" / "analysis"

NOV_MAY = {11, 12, 1, 2, 3, 4, 5}
WHO, THAI = 15.0, 37.5

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5,
                     "figure.dpi": 150, "savefig.dpi": 300,
                     # MDPI body typeface is Palatino; P052 is the URW clone
                     "font.family": "serif",
                     "font.serif": ["P052", "Palatino", "TeX Gyre Pagella"],
                     "mathtext.fontset": "custom",
                     "mathtext.rm": "P052", "mathtext.it": "P052:italic",
                     "mathtext.bf": "P052:bold"})


def read(path):
    with rasterio.open(path) as src:
        a = src.read(1).astype("float64")
        a[a == src.nodata] = np.nan
        return a, src.profile


# ---------------------------------------------------------------- pass 1
daily = sorted(PRED.glob("pm25_2???-??-??.tif"))
print(f"{len(daily)} daily surfaces")
_, prof = read(daily[0])
shape = (prof["height"], prof["width"])

n_valid = np.zeros(shape); n_who = np.zeros(shape); n_thai = np.zeros(shape)
for p in daily:
    m = int(p.stem.split("-")[1])
    if m not in NOV_MAY:
        continue
    a, _ = read(p)
    v = np.isfinite(a)
    n_valid += v
    n_who += v & (a > WHO)
    n_thai += v & (a > THAI)
nm_days = int(sum(1 for p in daily if int(p.stem.split("-")[1]) in NOV_MAY))

pop, _ = read(ROOT / "data" / "grids" / "pop.tif")
pop = np.where(np.isfinite(pop) & (pop > 0), pop, 0.0)

area_who = n_who.sum() / n_valid.sum() * 100
area_thai = n_thai.sum() / n_valid.sum() * 100
pw_who = (pop * n_who).sum() / (pop * n_valid).sum() * 100
pw_thai = (pop * n_thai).sum() / (pop * n_valid).sum() * 100

# per-pixel exceedance-fraction maps (>= 30 valid Nov-May days over 3 y)
MIN_D = 30
frac_who = np.where(n_valid >= MIN_D, n_who / n_valid, np.nan)
frac_thai = np.where(n_valid >= MIN_D, n_thai / n_valid, np.nan)
out_prof = dict(prof, dtype="float32", nodata=-9999.0)
for name, arr in [("who", frac_who), ("thai", frac_thai)]:
    with rasterio.open(PRED / f"pm25_exceedance_{name}_novmay.tif", "w",
                       **out_prof) as dst:
        dst.write(np.where(np.isfinite(arr), arr, -9999.0).astype("float32"), 1)

# population share living where the Nov-May-mean exceeds thresholds
mean_sum = np.zeros(shape);
for p in daily:
    m = int(p.stem.split("-")[1])
    if m not in NOV_MAY:
        continue
    a, _ = read(p)
    mean_sum += np.where(np.isfinite(a), a, 0.0)
nm_mean = np.where(n_valid >= MIN_D, mean_sum / n_valid, np.nan)
popshare_who = pop[np.isfinite(nm_mean) & (nm_mean > WHO)].sum() / \
    pop[np.isfinite(nm_mean)].sum() * 100
popshare_thai = pop[np.isfinite(nm_mean) & (nm_mean > THAI)].sum() / \
    pop[np.isfinite(nm_mean)].sum() * 100

# ---------------------------------------------------------------- regions
prov = gpd.read_file(AN / "thailand_provinces.geojson")
REGIONS = ["Northern", "Northeastern", "Central", "Eastern", "Western", "Southern"]
rmap = {r: i + 1 for i, r in enumerate(REGIONS)}
with rasterio.open(daily[0]) as src:
    tfm = src.transform
region_ras = rasterize(
    [(geom, rmap[r]) for geom, r in zip(prov.geometry, prov.region)],
    out_shape=shape, transform=tfm, fill=0, dtype="int16")

def comp_mean(pattern):
    arrs = [read(p)[0] for p in sorted(PRED.glob(pattern))]
    return np.nanmean(np.stack(arrs), axis=0)

annual = comp_mean("pm25_annual_20??.tif")
hot = comp_mean("pm25_seasonal_20??_hot_dry.tif")
cool = comp_mean("pm25_seasonal_20??_cool_dry.tif")
wet = comp_mean("pm25_seasonal_20??_wet_monsoon.tif")

rows = []
for r, code in rmap.items():
    m = region_ras == code
    rows.append({
        "region": r,
        "annual": np.nanmean(annual[m]),
        "hot_dry": np.nanmean(hot[m]),
        "cool_dry": np.nanmean(cool[m]),
        "wet": np.nanmean(wet[m]),
        "novmay_who_pct": 100 * np.nansum(n_who[m]) / np.nansum(n_valid[m]),
        "novmay_thai_pct": 100 * np.nansum(n_thai[m]) / np.nansum(n_valid[m]),
    })
reg = pd.DataFrame(rows).round(1)
reg.to_csv(AN / "map_region_stats.csv", index=False)

# ---------------------------------------------------------------- figures
th = gpd.read_file(FIGS / "thailand_boundary.geojson")
extent = (prof["transform"].c,
          prof["transform"].c + prof["transform"].a * prof["width"],
          prof["transform"].f + prof["transform"].e * prof["height"],
          prof["transform"].f)

def draw(ax, arr, vmin, vmax, cmap="YlOrRd"):
    im = ax.imshow(arr, extent=extent, origin="upper", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    th.boundary.plot(ax=ax, color="0.35", lw=0.5)
    ax.set_xlim(96.9, 106.1); ax.set_ylim(5.4, 21.1)
    ax.set_xticks([]); ax.set_yticks([])
    return im

# Figure 6 (fig:ss_pm): seasonal composites, 3 years x 3 seasons
fig, axes = plt.subplots(3, 3, figsize=(8.6, 12.6))
vmax5 = float(np.nanpercentile(hot, 99))
for i, year in enumerate([2023, 2024, 2025]):
    for j, (stub, lab) in enumerate([("cool_dry", "Cool-dry"),
                                     ("hot_dry", "Hot-dry"),
                                     ("wet_monsoon", "Wet-monsoon")]):
        a, _ = read(PRED / f"pm25_seasonal_{year}_{stub}.tif")
        ax = axes[i, j]
        im = draw(ax, a, 0, vmax5)
        if i == 0:
            ax.set_title(lab + ("\n(background estimate)" if j == 2 else ""))
        if j == 0:
            ax.set_ylabel(str(year), fontsize=11)
        ax.axis("on")
fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02,
             label="Seasonal mean PM$_{2.5}$ (µg m$^{-3}$)")
fig.savefig(FIGS / "Figure6.png", bbox_inches="tight")
plt.close(fig)

# Figure 7 (fig:monthly_pm): monthly climatology (2023-2025 mean per month)
fig, axes = plt.subplots(3, 4, figsize=(11.2, 12.4))
monthly = {}
for mm in range(1, 13):
    monthly[mm] = comp_mean(f"pm25_monthly_20??-{mm:02d}.tif")
vmax6 = float(np.nanpercentile(np.stack(list(monthly.values())), 99))
names = ["January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"]
for mm in range(1, 13):
    ax = axes[(mm - 1) // 4, (mm - 1) % 4]
    im = draw(ax, monthly[mm], 0, vmax6)
    ax.set_title(names[mm - 1] + (" (bg)" if mm in (6, 7, 8, 9, 10) else ""))
    ax.axis("on")
fig.colorbar(im, ax=axes, shrink=0.55, pad=0.02,
             label="Monthly mean PM$_{2.5}$ (µg m$^{-3}$), 2023--2025")
fig.savefig(FIGS / "Figure7.png", bbox_inches="tight")
plt.close(fig)

# Figure 8 (fig:exceed_pm): Nov-May exceedance fractions
fig, axes = plt.subplots(1, 2, figsize=(9.4, 6.4))
for ax, arr, ttl in [
        (axes[0], frac_who * 100,
         f"Days above WHO 24-h guideline (15 µg m$^{{-3}}$)"),
        (axes[1], frac_thai * 100,
         f"Days above Thai 24-h standard (37.5 µg m$^{{-3}}$)")]:
    im = draw(ax, arr, 0, 100, cmap="magma_r")
    ax.set_title(ttl, fontsize=9)
    ax.axis("on")
    fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02,
                 label="Share of valid November--May days (%)")
fig.savefig(FIGS / "Figure8.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- report
s = pd.read_csv(PRED / "pm25_spatial_stats.csv")
ann = s[s.period_type == "annual"].set_index("period")
mon = s[s.period_type == "monthly"].copy()
mon["year"] = mon.period.str[:4]
peaks = mon.loc[mon.groupby("year")["mean"].idxmax()]

L = []
L.append("# Map-derived statistics (repaired GB-GWR surfaces, 2023-2025)")
L.append("")
L.append(f"Daily surfaces: {len(daily)} of 1,096 study days (288 with GWR "
         f"correction, rest stage-1 only per the gate). Grid 578x335 at ~3 km; "
         f"61,925 Thailand pixels; valid-pixel coverage tracks daily AOD.")
L.append("")
L.append("## Annual national means (area / population-weighted, ug/m3)")
for y in ["2023", "2024", "2025"]:
    L.append(f"- {y}: {ann.loc[y,'mean']:.1f} / {ann.loc[y,'pop_weighted_mean']:.1f}")
L.append("")
L.append("## Seasonal composite national means (area-weighted, ug/m3)")
for _, r in s[s.period_type == "seasonal"].iterrows():
    L.append(f"- {r.period}: {r['mean']:.1f} (pop-weighted {r.pop_weighted_mean:.1f}, "
             f"p95 {r.p95:.1f}, max {r['max']:.1f})")
L.append("")
L.append("## Peak months (national monthly mean)")
for _, r in peaks.iterrows():
    L.append(f"- {r.period}: {r['mean']:.1f} (pop-weighted {r.pop_weighted_mean:.1f})")
L.append("")
L.append(f"## November--May exceedance (pixel-day basis, {nm_days} Nov-May days)")
L.append(f"- WHO 24-h guideline 15 ug/m3: {area_who:.1f}% of valid pixel-days; "
         f"population-weighted {pw_who:.1f}% of person-days")
L.append(f"- Thai 24-h standard 37.5 ug/m3: {area_thai:.1f}% of valid pixel-days; "
         f"population-weighted {pw_thai:.1f}% of person-days")
L.append(f"- Population living where the Nov--May mean exceeds 15: "
         f"{popshare_who:.1f}%; exceeds 37.5: {popshare_thai:.1f}%")
L.append("")
L.append("## Regional statistics (composite means, ug/m3; Nov-May exceedance %)")
L.append(reg.to_string(index=False))
L.append("")
L.append("## Trend caution inputs")
L.append(f"- annual (area): 2023 {ann.loc['2023','mean']:.1f} -> 2024 "
         f"{ann.loc['2024','mean']:.1f} -> 2025 {ann.loc['2025','mean']:.1f}")
hd = s[(s.period_type == 'seasonal') & s.period.str.contains('hot_dry')]
L.append(f"- hot-dry seasonal: " + " -> ".join(
    f"{r['mean']:.1f}" for _, r in hd.iterrows()) +
    " (2023 El Nino year highest; interpret per ENSO caution)")

(AN / "map_stats.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
