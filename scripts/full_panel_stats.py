from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_loader, preprocessing  # noqa: E402

OUT = ROOT / "output" / "analysis" / "full_panel_stats.md"

cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
cols = cfg["columns"]

df = data_loader.load_panel(ROOT / cfg["paths"]["panel_file"], cols)

period = cfg["study_period"]
t0, t1 = pd.Timestamp(period["start"]), pd.Timestamp(period["end"])
df = df[(df["date"] >= t0) & (df["date"] <= t1)].reset_index(drop=True)
n_period = len(df)

df = preprocessing.validate_ranges(df, cfg["preprocessing"]["valid_ranges"])

# Display-unit conversions (panel native units per default.yaml comments)
disp = df.copy()
disp["prec"] = disp["prec"] * 1000.0        # m/day  -> mm/day
disp["ssr"] = disp["ssr"] / 1.0e6           # J/m2   -> MJ/m2
disp["sp"] = disp["sp"] / 100.0             # Pa     -> hPa

SEASON = {11: "cool-dry", 12: "cool-dry", 1: "cool-dry", 2: "cool-dry",
          3: "hot-dry", 4: "hot-dry", 5: "hot-dry",
          6: "wet", 7: "wet", 8: "wet", 9: "wet", 10: "wet"}
disp["season"] = disp["date"].dt.month.map(SEASON)

pm = disp.loc[disp["pm25"].notna(), ["date", "station_id", "season", "pm25"]]

lines = []
lines.append("# Full co-location panel — descriptive statistics")
lines.append("")
lines.append(f"Panel: `{cfg['paths']['panel_file']}` -> study period "
             f"{t0.date()}..{t1.date()} -> valid-range guard.")
lines.append(f"- Records in study period: **{n_period:,}**")
lines.append(f"- Stations: **{disp['station_id'].nunique()}**")
lines.append(f"- Records with valid PM2.5: **{len(pm):,}**")
lines.append(f"- Records with observed AOD (and valid PM2.5): "
             f"**{int((disp['aod'].notna() & disp['pm25'].notna()).sum()):,}**")
lines.append("")

q = pm["pm25"].quantile
lines.append("## PM2.5 distribution (all valid-PM2.5 records)")
lines.append(f"- mean {pm['pm25'].mean():.2f} | median {q(0.5):.2f} | "
             f"IQR {q(0.25):.2f}-{q(0.75):.2f} | p99 {q(0.99):.2f} | "
             f"max {pm['pm25'].max():.2f}")
lines.append(f"- WHO 24-h guideline 15 ug/m3 exceeded: "
             f"{(pm['pm25'] > 15).mean() * 100:.1f}% of records")
lines.append(f"- Thai 24-h standard 37.5 ug/m3 exceeded: "
             f"{(pm['pm25'] > 37.5).mean() * 100:.1f}% of records")
lines.append("")

lines.append("## Seasonal PM2.5 (record-level mean / SD / n; then mean of daily national means)")
for s in ["cool-dry", "hot-dry", "wet"]:
    g = pm[pm["season"] == s]
    daily = g.groupby("date")["pm25"].mean()
    lines.append(f"- {s}: mean {g['pm25'].mean():.2f} (SD {g['pm25'].std():.2f}, "
                 f"n={len(g):,}); mean of daily national means {daily.mean():.2f}")
lines.append("")

lines.append("## Table 1 cells (display units; mean / SD / min / max)")
lines.append("")
lines.append("| Variable | Mean | SD | Min | Max |")
lines.append("|---|---|---|---|---|")
spec = [("pm25", "PM2.5 (ug/m3)", 2), ("aod", "AOD (-)", 2),
        ("t2m", "Temperature (degC)", 2), ("rh", "RH (%)", 2),
        ("ws", "Wind speed (m/s)", 2), ("blh", "PBLH (m)", 1),
        ("prec", "Precipitation (mm/day)", 1), ("ssr", "SSR (MJ/m2/day)", 2),
        ("sp", "Surface pressure (hPa)", 1), ("frp", "FRP (MW)", 2),
        ("ndvi", "NDVI (-)", 2), ("elev", "Elevation (m)", 1),
        ("pop", "Pop. density (per km2)", 1)]
for col, label, nd in spec:
    v = disp[col].dropna()
    lines.append(f"| {label} | {v.mean():.{nd}f} | {v.std():.{nd}f} | "
                 f"{v.min():.{nd}f} | {v.max():.{nd}f} |")
lines.append("")
lines.append("FRP high quantiles (for the 'zero most days' sentence): "
             f"share exactly 0 = {(disp['frp'].dropna() == 0).mean() * 100:.1f}%, "
             f"p99 = {disp['frp'].dropna().quantile(0.99):.3f} MW.")
lines.append("")
lines.append("## Missingness (share of study-period records)")
for col in ["pm25", "aod"] + cfg["columns"]["meteo"] + cfg["columns"]["ancillary"]:
    lines.append(f"- {col}: {disp[col].isna().mean() * 100:.1f}% missing")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nwritten -> {OUT}")
