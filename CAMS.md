# CAMS — Copernicus Atmosphere Monitoring Service

> **What CAMS is, what it outputs, and why we use it as the physics backbone for Delhi NCR 72-hour AQI forecasting.**
> Part of the [Delhi NCR Coupled AQI Forecasting System](./README.md) — CAMS provides real coupled chemistry transport; our ML-hybrid sharpens it to station level.

---

## Table of Contents
- [What is CAMS?](#what-is-cams)
- [The 5 Main Categories of Outputs](#the-5-main-categories-of-outputs)
  - [1. Air Quality and Reactive Gases](#1-air-quality-and-reactive-gases-hourly--5-day-forecasts)
  - [2. Greenhouse Gas Tracking](#2-greenhouse-gas-tracking)
  - [3. Aerosols and Tracers](#3-aerosols-and-tracers)
  - [4. Solar Radiation and UV Indices](#4-solar-radiation-and-uv-indices)
  - [5. Live Biomass Burning and Emissions Alerts](#5-live-biomass-burning-and-emissions-alerts)
- [Cons / Limitations of CAMS](#cons--limitations-of-cams)
  - [1. The Pre-Designed Model Resolution (The Native Grid)](#1-the-pre-designed-model-resolution-the-native-grid)
  - [Other Limitations](#other-limitations)
- [How to Get a Custom Grid](#how-to-get-a-custom-grid)
  - [Option A: Request Re-gridding via the API (Server-Side)](#option-a-request-re-gridding-via-the-api-server-side)
  - [Option B: Downscale using Python (Client-Side)](#option-b-downscale-using-python-client-side)
- [How We Use CAMS in This Project](#how-we-use-cams-in-this-project)
- [Key Links](#key-links)

---

## What is CAMS?

**CAMS (Copernicus Atmosphere Monitoring Service)** is the European Centre for Medium-Range Weather Forecasts (**ECMWF**) atmosphere service, part of the EU **Copernicus** programme. It is:

- **A coupled chemistry-transport model** — it solves meteorology + chemistry together (advection, diffusion, chemical reactions, aerosol physics) on a global 3D grid, every hour.
- **Operational, not research-only** — it assimilates satellite + in-situ observations daily and publishes **hourly forecasts up to 5 days ahead** (00/12 UTC cycles).
- **Free and open** via the **Atmosphere Data Store (ADS) / CDS API** — anyone with an ECMWF account can pull NetCDF/GRIB.
- **The reason we don’t need WRF-Chem on Day 1** — CAMS already embeds emission inventories (CAMS-GLOB-ANT, GFAS fires, MEGAN biogenics), long-range transport of stubble-burning plumes, and aerosol–radiation feedbacks. We blend its physics prior with local ML correction (see [README: CAMS+ML-Hybrid](./README.md)).

**In this project CAMS is the physics backbone:** regional PM₂.₅/PM₁₀/O₃/NOₓ fields that our `data/fetch_cams.py` pulls, then `ml/features.py` aligns to CPCB stations and `ml/infer.py` bias-corrects.

---

## The 5 Main Categories of Outputs

### 1. Air Quality and Reactive Gases (Hourly & 5-Day Forecasts)

The system outputs the exact concentrations of specific toxic gases and particulate matter at different grid layers.

-   **Particulate Matter:** PM2.5 and PM10 mass concentration (µg/m³) — surface + column.
-   **Regulated Gases:** Nitrogen Dioxide (NO₂), Ground-level Ozone (O₃), Sulfur Dioxide (SO₂), Carbon Monoxide (CO), and Formaldehyde (HCHO).
-   **Atmospheric Chemistry:** Total columns and surface-level mixing ratios for **dozens of secondary chemical species** (e.g., PAN, HNO₃, isoprene products) — useful for diagnosing ozone chemistry.

> **We use:** `pm2p5`, `pm10`, `o3`, `no2`, `co`, `so2` at surface level for Delhi NCR bbox. These are the training targets for `ml/features.py`.

### 2. Greenhouse Gas Tracking

CAMS monitors long-lived gases responsible for climate change.

-   **Carbon Dioxide (CO₂):** Global high-resolution 5-day forecasts of atmospheric CO₂ (ppm) — column-averaged and 3D.
-   **Methane (CH₄):** High-resolution tracking and localized hotspot maps indicating anthropogenic leaks (ppb).
-   **Surface Fluxes:** Net estimates of how much greenhouse gas is absorbed or emitted by land, oceans, and vegetation (flux inversions via IFS).

> **Project note:** Not used for AQI, but demonstrates CAMS is a full greenhouse-gas reanalysis — relevant if you extend the system to carbon monitoring.

### 3. Aerosols and Tracers

The system categorizes microscopic solid or liquid particles suspended in the air to track **where they came from**.

-   **Aerosol Optical Depth (AOD):** A measure of how much direct sunlight is blocked by airborne particles (dimensionless, at 550 nm). High AOD = dense haze → reduced surface radiation → PBL feedback we model.
-   **Aerosol Tracers:** Separate tracking maps showing how much of the pollution is **Dust**, **Sea Salt**, **Biomass Burning (Smoke)**, **Organic Matter**, or **Volcanic Ash** — each as AOD contribution and mass.

> **We use:** Total AOD and `aod_biomass_burning` to flag stubble-burning episodes vs dust events before FIRMS even fires. Helps explain *why* PM₂.₅ spikes.

### 4. Solar Radiation and UV Indices

Useful for green energy grids and public health notices, CAMS provides:

-   **UV Index:** Projections of daily solar ultraviolet radiation levels reaching the earth's surface to help manage sunburn and skin cancer risks (clear-sky + all-sky).
-   **Solar Energy Availability:** Precise calculations of direct and diffuse solar radiation hitting the ground (W/m²) to help solar power plants estimate grid generation.
-   **Total Stratospheric Ozone:** Monitoring the health and holes in the higher-altitude protective ozone layer (Dobson Units).

> **Coupling relevance:** Solar radiation ↔ O₃ and aerosols → temperature. CAMS radiation fields let us derive inversion/ventilation proxies without a separate weather model.

### 5. Live Biomass Burning and Emissions Alerts

-   **Fire Radiative Power (FRP):** Measurements of heat emitted by active wildfires and agricultural burning (MW) — assimilated from satellite FRP via GFAS (Global Fire Assimilation System).
-   **Custom Plume Charts & Alerts:** Immediate alerts sent out when massive plumes of smoke or dust are forecasted to drift over densely populated cities — CAMS publishes biomass-burning plume forecasts (CO, AOD, PM) as forward trajectories.

> **We use:** FRP-adjacent — we prefer **NASA FIRMS VIIRS** for hotspot lat/lon, but CAMS `aod_biomass_burning` and `co` plumes cross-validate whether a Punjabi stubble plume will reach Delhi in 24–48h under current winds.

**All outputs are available as:** hourly NetCDF/GRIB on pressure levels + surface, via ADS. Forecast cycle: 00 UTC and 12 UTC, lead 0–120h.

---

## Cons / Limitations of CAMS

Understanding these is why our system adds an **ML-hybrid correction** and a **WRF-Chem upgrade path**.

### 1. The Pre-Designed Model Resolution (The Native Grid)

When CAMS calculates atmospheric changes, it divides the entire Earth into a fixed grid. The resolution depends on which dataset you choose:

| Dataset | Native Horizontal Grid | Approx. Resolution | Notes |
|---|---|---|---|
| **Global Forecasts** (`cams-global-...`, `cams-global-atmospheric-composition-forecasts`) | ~0.4° × 0.4° | **~40 km × 40 km** | Operational forecast, 5-day lead. **This is what we use.** |
| **Global Reanalysis (EAC4 — Historical Data)** | ~0.75° × 0.75° | **~80 km × 80 km** | 2003–present, for training history |
| **Regional European Models** (`cams-europe-air-quality-forecasts`, ensemble) | 0.1° × 0.1° | **~10 km × 10 km** | European window only — not available for Delhi |

**What this means for Delhi NCR:** One CAMS grid cell covers ~40 km — roughly all of South Delhi. It cannot resolve:
- Road-level NOₓ hotspots (ITO, Anand Vihar)
- PBL trapping in narrow winter inversion layers
- Neighbourhood-scale stubble plume edges

→ **Mitigation in this project:** `ml/features.py` interpolates CAMS to station point + `ml/train.py` learns per-station bias. See [phase.md Day 1–2](./phase.md).

### Other Limitations

| Limitation | Impact | Our Mitigation |
|---|---|---|
| **Coarse time for emissions** | Direct emission inventories (CAMS-GLOB-ANT) are monthly climatology + scaling, not real-time stack emissions | ML learns recent CPCB obs lag; FIRMS injects real-time fire signal |
| **No street-canyon chemistry** | Misses sub-grid NOx–O3 titration | Per-station correction captures systematic offset |
| **5-day lead is global model** | Boundary conditions from IFS; regional biases persist | Verification panel tracks RMSE per lead; ensemble bands widen at 72h |
| **Reanalysis is coarse (80 km)** | Historical training data lower res than forecast | Train on forecast archives when possible, not just EAC4 |
| **European high-res not for India** | Tempting to assume 10 km exists globally — it doesn’t | Documented above; WRF-Chem Phase 6 is the true high-res path |
| **API latency + ADS queue** | Downloads can queue 5–30 min at 00/12 UTC peaks | Cache last forecast; retry + `data/samples/` fallback (see phase.md P1) |
| **No emission-scenario experiments** | CAMS can’t answer “what if crop burning halved?” | WRF-Chem in Phase 6 enables emission sensitivity runs |

> **Bottom line:** CAMS is the best free physics prior, but it is **not** a Delhi street-level model. Treat it as a skilled regional background that needs local sharpening.

---

## How to Get a Custom Grid

Native 0.4° is too coarse for station maps. You have two ways to request or create a finer mesh.

### Option A: Request Re-gridding via the API (Server-Side)

You can pass a custom `grid` parameter directly inside your Python ADS download request. The **API server will automatically use bilinear or nearest-neighbor interpolation** to map the native 0.4° data onto your requested grid size — no local math needed.

You can modify your script by adding the `grid` key:

```python
import cdsapi

c = cdsapi.Client()

# Example: CAMS global forecast, Delhi NCR bbox, re-gridded to 0.1° (~10 km)
c.retrieve(
    'cams-global-atmospheric-composition-forecasts',
    {
        'variable': ['particulate_matter_2.5um', 'particulate_matter_10um',
                     'ozone', 'nitrogen_dioxide'],
        'date': '2026-09-02/2026-09-05',
        'time': '00:00',
        'leadtime_hour': ['0', '24', '48', '72'],
        'type': 'forecast',
        # --- Custom grid (server-side regridding) ---
        'grid': [0.1, 0.1],          # [lat, lon] degrees → ~10 km
        'area': [29.0, 76.8, 28.3, 77.5],  # N, W, S, E (Delhi NCR)
        'format': 'netcdf',
    },
    'cams_delhi_0.1deg.nc'
)

# For coarser/faster download, use 0.25° (~25 km):
# 'grid': [0.25, 0.25],

# For native resolution, omit 'grid' entirely or set 0.4:
# 'grid': [0.4, 0.4],
```

**Pros:** Zero local dependencies, consistent with ECMWF interpolation, smaller file if you also clip `area`.
**Cons:** Still interpolated — not true higher physics; server queue time slightly longer for fine grids; max fine ~0.1° before file bloats.

**Tip for this project:** Use `grid: [0.1, 0.1]` + `area: Delhi NCR` for Phase 2 — 10 km is enough to map to 10 stations smoothly. Don’t request 0.01° server-side (will timeout).

---

### Option B: Downscale using Python (Client-Side)

If you want **hyper-local resolution** (e.g., a **1 km × 1 km grid** to map individual sectors of Delhi NCR), you can pull the data at its **native resolution** and downscale it locally using Python’s `xarray` + `scipy`/`xesmf`/`rioxarray`. This creates a smoother, finer mesh by calculating values between the model’s standard data points — you control the method.

```python
import xarray as xr
import numpy as np
from scipy.interpolate import griddata

# 1. Load native 0.4° CAMS file
ds = xr.open_dataset('cams_delhi_native.nc')  # dims: latitude, longitude, leadtime
pm25 = ds['pm2p5'].isel(leadtime=24)  # example: 24h lead, surface level

# 2. Define native points and values
lats_native = ds.latitude.values
lons_native = ds.longitude.values
lon2d, lat2d = np.meshgrid(lons_native, lats_native)
points = np.column_stack([lat2d.ravel(), lon2d.ravel()])
values = pm25.values.ravel()
# mask NaNs
mask = ~np.isnan(values)
points, values = points[mask], values[mask]

# 3. Define Delhi hyper-local grid (1 km ≈ 0.009°)
lat_fine = np.arange(28.3, 29.0, 0.009)   # ~78 points
lon_fine = np.arange(76.8, 77.5, 0.009)   # ~77 points
lon_fine2d, lat_fine2d = np.meshgrid(lon_fine, lat_fine)

# 4. Interpolate (cubic = smoother, linear = faster)
pm25_fine = griddata(points, values, (lat_fine2d, lon_fine2d), method='cubic')

# 5. Wrap back to xarray for plotting / station extraction
da_fine = xr.DataArray(
    pm25_fine,
    dims=['latitude', 'longitude'],
    coords={'latitude': lat_fine, 'longitude': lon_fine},
    attrs=pm25.attrs
)
da_fine.to_netcdf('cams_delhi_1km.nc')

# For smoother meteorology, try xesmf (conservative) or rioxarray:
# import xesmf as xe
# regridder = xe.Regridder(ds, ds_fine_template, method='bilinear')
# da_fine = regridder(ds['pm2p5'])
```

**More advanced (recommended for maps):**

```python
# Using xarray's interp (fast, bilinear) — good enough for 10 stations
ds_fine = ds.interp(latitude=lat_fine, longitude=lon_fine, method='linear')

# Extract to stations (nearest 1 km cell, no extra file)
import pandas as pd
stations = pd.read_csv('config/stations.csv')  # lat, lon cols
for _, s in stations.iterrows():
    pm25_at_station = da_fine.sel(latitude=s.lat, longitude=s.lon, method='nearest').values
```

**Pros:** Full control, can go to **1 km** for Leaflet contour maps; use `cubic`/`nearest` per variable; no server wait; easy to blend with ML features.
**Cons:** Interpolated values are **not new physics** — still 40 km information smoothed; can create false precision if over-interpreted; heavier RAM for 72h × 1 km grid.

**Project guidance (phase.md sprint):**

| Use Case | Recommended Method | Grid | File |
|---|---|---|---|
| `ml/features.py` station alignment | Client-side `interp(method='nearest')` | Station points directly, no grid | — |
| Leaflet PM2.5 contour map | Client-side `interp` or `griddata(cubic)` | 0.02° (~2 km) | `cams_delhi_2km.nc` |
| Download for EDA | Server-side `grid: [0.1, 0.1]` | ~10 km | `cams_delhi_0.1deg.nc` |
| Don’t do in 3-day sprint | Server-side 0.009° | ~1 km via API | Too slow — do client-side |

> **Honesty note for jury:** Always label maps as “CAMS 0.4° native, interpolated to 2 km for visualization” — don’t claim 1 km physics. True 1–4 km physics requires **WRF-Chem (Phase 6)**.

---

## How We Use CAMS in This Project

```
STAGE 1: data/fetch_cams.py  →  ADS API + grid [0.1,0.1] + area Delhi NCR
STAGE 2: ml/features.py      →  xarray interp to station lat/lon (nearest)
                                + lag obs + fire + inversion/VC features
STAGE 3: ml/train.py/infer.py →  bias-correct CAMS → 72h station forecast
STAGE 4: postprocess/aqi.py  →  PM → AQI (Indian formula)
STAGE 5: backend/api         →  serve /api/forecast {cams, corrected, bands}
STAGE 6: docs & dashboard    →  show CAMS vs corrected, reveal value of ML
         Future: replace fetch_cams.py with fetch_wrfchem.py, same schema
```

-   **Config:** `config/domain.yaml` sets `cams.grid: [0.1, 0.1]`, `cams.area: [29.0, 76.8, 28.3, 77.5]`
-   **Samples:** `data/samples/cams_sample.nc` committed so `ml/features.py --mock` works offline
-   **Verification:** `postprocess/metrics.py` logs `RMSE_cams` vs `RMSE_corrected` per lead

---

## Key Links

-   CAMS Global Forecasts: `cams-global-atmospheric-composition-forecasts` on ADS
-   CAMS EAC4 Reanalysis: `cams-global-reanalysis-eac4`
-   ADS (Atmosphere Data Store): https://ads.atmosphere.copernicus.eu
-   ECMWF `cdsapi` Python client: `pip install cdsapi`
-   Copernicus CAMS docs: https://atmosphere.copernicus.eu
-   GFAS fires: https://atmosphere.copernicus.eu/gfas
-   FIRMS (for cross-validation): https://firms.modaps.eosdis.nasa.gov

---

*Last updated: 2026-09-02 | Related: [README.md](./README.md) · [phase.md](./phase.md) · `docs/architecture-flow.mmd`*
