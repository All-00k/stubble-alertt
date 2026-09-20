# New Implementation — Coupled CNN-LSTM AQI Forecasting System for Delhi NCR
### 72-Hour High-Resolution Forecast on a Common 5×5 km Grid (CPCB + CAMS + Weather + FIRMS)

> Complete implementation blueprint: from raw data sources → common grid → CNN spatial learning → LSTM temporal learning → 72h pollutant forecast → CPCB AQI. This is the detailed build guide for the 3-day sprint and beyond.

---

## Table of Contents
- [1. Overall System Architecture](#1-overall-system-architecture)
- [2. First Important Decision: What Exactly Are We Predicting?](#2-first-important-decision-what-exactly-are-we-predicting)
- [3. Geographic Domain](#3-geographic-domain)
- [4. Data We Need (7 Groups)](#4-data-we-need)
- [5. Dataset 1 — CPCB Ground Stations](#5-dataset-1--cpcb-ground-stations)
- [6. Don't Throw Away Multiple Stations in One Grid Cell](#6-dont-throw-away-multiple-stations-in-one-grid-cell)
- [7. Dataset 2 — Weather](#7-dataset-2--weather)
- [8. Atmospheric Stability / Inversion Features](#8-atmospheric-stability--inversion-features)
- [9. Other Meteorological Features](#9-other-meteorological-features)
- [10. Dataset 3 — FIRMS Fire Data](#10-dataset-3--firms-fire-data)
- [11. FIRMS → Grid Conversion](#11-firms--grid-conversion)
- [12. FIRMS Temporal Features](#12-firms-temporal-features)
- [13. Very Important: Transport-Aware Fire Features](#13-very-important-transport-aware-fire-features)
- [14. Transport-Weighted Fire Feature](#14-transport-weighted-fire-feature)
- [15. Should We Use All FIRMS Data?](#15-should-we-use-all-firms-data)
- [16. Dataset 4 — CAMS](#16-dataset-4--cams)
- [17. CAMS + CPCB](#17-cams--cpcb)
- [18. Dataset 5 — Land Use / Land Cover](#18-dataset-5--land-use--land-cover)
- [19. Population / Emission Proxies](#19-population--emission-proxies)
- [20. Road Features](#20-road-features)
- [21. Industrial Features](#21-industrial-features)
- [22. Dataset 6 — Elevation / Topography](#22-dataset-6--elevation--topography)
- [23. Dataset 7 — Time Features](#23-dataset-7--time-features)
- [24. Complete Feature List — A. Pollution Features](#24-complete-feature-list--a-pollution-features)
- [25. B. Weather Features](#25-b-weather-features)
- [26. C. FIRMS Features](#26-c-firms-features)
- [27. D. CAMS Features](#27-d-cams-features)
- [28. E. Geography Features](#28-e-geography-features)
- [29. F. Human Activity / Emission Proxy Features](#29-f-human-activity--emission-proxy-features)
- [30. G. Time Features](#30-g-time-features)
- [31. What Does One Grid Cell Actually Look Like?](#31-what-does-one-grid-cell-actually-look-like)
- [32. The Actual CNN Input](#32-the-actual-cnn-input)
- [33. Why CNN?](#33-why-cnn)
- [34. Then LSTM](#34-then-lstm)
- [35. Recommended Input Window](#35-recommended-input-window)
- [36. Forecast Output](#36-forecast-output)
- [37. Better Output Design](#37-better-output-design)
- [38. Full Data Pipeline](#38-full-data-pipeline)
- [39. Training Dataset Structure](#39-training-dataset-structure)
- [40. Example Training Sample](#40-example-training-sample)
- [41. Important Improvement: Don't Predict Every Cell Initially](#41-important-improvement-dont-predict-every-cell-initially)
- [42. Even Better: Station-Aware Training](#42-even-better-station-aware-training)
- [43. Loss Function](#43-loss-function)
- [44. Don't Randomly Split the Data](#44-dont-randomly-split-the-data)
- [45. Evaluation](#45-evaluation)
- [46. Baselines You Absolutely Need](#46-baselines-you-absolutely-need)
- [47. Where XGBoost Fits](#47-where-xgboost-fits)
- [48. The Most Important Features](#48-the-most-important-features)
- [49. Final Recommended Feature Set (Core 30 Channels)](#49-final-recommended-feature-set-core-30-channels)
- [50. How the "Punjab Fire → Delhi AQI" Mechanism Works](#50-how-the-punjab-fire--delhi-aqi-mechanism-works)
- [51. What Each Dataset Contributes](#51-what-each-dataset-contributes)
- [52. The Final System I Recommend + Build Order](#52-the-final-system-i-recommend--build-order)

---

## 1. Overall System Architecture

```
                  ┌──────────────────────────────┐
                  │       DATA SOURCES           │
                  └──────────────┬───────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
   CPCB CAAQMS                CAMS                  Weather
   Ground stations        Atmospheric data       ERA5/GFS/etc.
        │                        │                        │
        ▼                        ▼                        ▼
   FIRMS Fire Data        Land / Emission          Elevation
   VIIRS / MODIS           Source Data             / Geography
        │                        │                        │
        └────────────────────────┼────────────────────────┘
                                 ▼
                    ┌────────────────────────┐
                    │   DATA PREPROCESSING   │
                    │                        │
                    │ • Time alignment       │
                    │ • Spatial alignment    │
                    │ • Missing values       │
                    │ • Quality filtering    │
                    │ • Feature engineering  │
                    └────────────┬───────────┘
                                 ▼
                 ┌──────────────────────────────┐
                 │     COMMON 5×5 km GRID       │
                 │                              │
                 │  Punjab → Haryana → Delhi   │
                 │  NCR → Western UP → Rajasthan│
                 └──────────────┬───────────────┘
                                ▼
                 Hourly Grid Tensor
                 H × W × Features
                                │
                                ▼
             ┌────────────────────────────────┐
             │          CNN MODULE             │
             │                                │
             │ Learns spatial relationships   │
             │                                │
             │ Fire → Wind → Pollution → Delhi│
             └───────────────┬────────────────┘
                             ▼
                    Spatial Embedding
                             │
                             ▼
             ┌────────────────────────────────┐
             │          LSTM MODULE            │
             │                                │
             │ Learns temporal evolution      │
             │                                │
             │ t-48 → ... → t-1 → t           │
             └───────────────┬────────────────┘
                             ▼
                 ┌──────────────────────┐
                 │ 72-HOUR FORECAST     │
                 │                      │
                 │ PM2.5 / O3 / PM10   │
                 │ NO2 / etc.           │
                 └──────────┬───────────┘
                            ▼
                  ┌─────────────────────┐
                  │    CPCB AQI ENGINE  │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │  AQI 1h → 72h       │
                  │ + Category          │
                  │ + Uncertainty       │
                  └─────────────────────┘
```

---

## 2. First Important Decision: What Exactly Are We Predicting?

There are actually two possible targets.

### Option A — Predict PM2.5 → PM2.5 AQI (Simpler — Best MVP)

```
Inputs
  ↓
CNN + LSTM
  ↓
Future PM2.5
  ↓
CPCB PM2.5 sub-index
  ↓
AQI
```

### Option B — Predict multiple pollutants → Full CPCB AQI (More scientifically correct)

```
CNN + LSTM
      │
      ├── PM2.5
      ├── PM10
      ├── NO2
      ├── SO2
      ├── CO
      ├── O3
      └── NH3
              ↓
       CPCB sub-indices
              ↓
       Maximum sub-index
              ↓
            AQI
```

### Recommended phasing

-   **Phase 1:** Predict `PM2.5 for the next 72 hours` only
-   **Phase 2:** Predict `PM2.5 + PM10 + O3 + NO2`
-   **Phase 3:** Full pollutant set if data quality supports it

This prevents the project from becoming unnecessarily complicated at the beginning.

---

## 3. Geographic Domain

Do not create a grid only around Delhi.

Your domain should capture the transport pathway:

```
              PUNJAB
     ┌─────────────────────┐
     │                     │
     │    Crop-burning     │
     │      regions        │
     └──────────┬──────────┘
                │
                │ wind transport
                ▼
            HARYANA
                │
                ▼
       ┌────────────────┐
       │   DELHI NCR    │
       │                │
       │ Delhi          │
       │ Gurugram       │
       │ Noida          │
       │ Ghaziabad      │
       │ Faridabad      │
       └───────┬────────┘
               │
               ▼
          WESTERN UP

       + parts of Rajasthan
       + surrounding buffer
```

### Recommended grid: 5 km × 5 km

```
     5 km
┌───────┬───────┬───────┬───────┐
│ Cell  │ Cell  │ Cell  │ Cell  │
├───────┼───────┼───────┼───────┤
│ Cell  │ Delhi │ Delhi │ Cell  │
├───────┼───────┼───────┼───────┤
│ Cell  │ Delhi │ Delhi │ NCR   │
├───────┼───────┼───────┼───────┤
│ Cell  │ Cell  │ Cell  │ Cell  │
└───────┴───────┴───────┴───────┘
```

**Why 5 km?**

-   Fine enough to capture regional differences
-   Compatible with CNN (image-like tensor)
-   Much easier computationally than 1 km
-   Better spatial detail than CAMS global data (0.4° ≈ 40 km)
-   FIRMS can be aggregated into it
-   CPCB stations can be mapped into it
-   Weather fields can be interpolated/regridded into it

---

## 4. Data We Need

Divide all data into 7 major groups.

| Data | Purpose | Importance |
|---|---|---|
| **CPCB** | Ground truth | ⭐⭐⭐⭐⭐ |
| **Weather** | Transport + dispersion | ⭐⭐⭐⭐⭐ |
| **FIRMS** | Fire emissions | ⭐⭐⭐⭐⭐ |
| **CAMS** | Regional chemical background | ⭐⭐⭐⭐⭐ |
| **Land-use** | Local emission characteristics | ⭐⭐⭐ |
| **Topography** | Dispersion effects | ⭐⭐ |
| **Calendar/time** | Seasonal/temporal patterns | ⭐⭐⭐ |

---

## 5. Dataset 1 — CPCB Ground Stations

This is your most important observation dataset. CPCB provides station-level measurements.

For each station and hour:

```
station_id
latitude
longitude
timestamp

PM2.5
PM10
NO2
SO2
CO
O3
NH3
```

Depending on station availability, not every pollutant will always be available.

**CPCB features (lag engineering):**

```
PM2.5
PM2.5_t
PM2.5_t-1
PM2.5_t-3
PM2.5_t-6
PM2.5_t-12
PM2.5_t-24
PM2.5_t-48
PM10 (same lags)

Other pollutants: NO2, SO2, O3, CO, NH3 (same)
```

---

## 6. Don't Throw Away Multiple Stations in One Grid Cell

Suppose:

```
5 km × 5 km cell
       │
       ├── CPCB Station A
       ├── CPCB Station B
       └── CPCB Station C
```

Don't force them into one value.

Keep:

```
Station A → observation
Station B → observation
Station C → observation
```

during training.

You can additionally calculate:

```
cell_mean_PM25
cell_max_PM25
cell_min_PM25
cell_std_PM25
station_count
```

This is useful because one grid cell may have substantial local variability.

---

## 7. Dataset 2 — Weather

Weather is critical. Pollution is not just determined by emissions — a major part of Delhi pollution is **how efficiently the atmosphere transports and disperses** those emissions.

Use **hourly** meteorological variables.

### Essential weather features

**Temperature**
```
temperature_2m
```

**Wind — Do not primarily feed only wind direction in degrees. Convert it into:**

```
u_wind  (east-west wind component)
v_wind  (north-south wind component)

u = -wind_speed * sin(wind_dir)
v = -wind_speed * cos(wind_dir)
```

Also derivable:

```
wind_speed
wind_direction
```

**Relative humidity**

```
RH  — Very important because aerosol properties and atmospheric chemistry depend strongly on humidity.
```

**Pressure**

```
surface_pressure
```

**Planetary Boundary Layer Height — One of the most important features:**

```
PBLH
```

Low PBL:

```
      Atmosphere
────────────────────
       ↓ ↓ ↓
    pollution
████████████████████
      Delhi
```

High PBL:

```
      Atmosphere
        ↑
        ↑
        ↑
pollution dispersed
────────────────────
      Delhi
```

> **Low PBLH → pollution accumulation** — something the model should learn.

---

## 8. Atmospheric Stability / Inversion Features

This is particularly important for Delhi winter.

Create:

```
temperature_2m
temperature_850hPa
temperature_925hPa
```

and derive:

```
temperature_gradient
```

For example:

```
Surface = 10°C
925 hPa = 14°C

warmer air above cooler surface
          ↓
       inversion
          ↓
pollution trapped
```

Useful engineered features:

```
inversion_flag
inversion_strength
stability_index
```

---

## 9. Other Meteorological Features

Include where available:

```
precipitation
cloud_cover
surface_radiation
solar_radiation
dew_point
visibility
```

Especially:

```
Rain

Rain ↑
   ↓
PM2.5 ↓  (wet deposition removes particulate matter)
```

---

## 10. Dataset 3 — FIRMS Fire Data

This is your regional biomass-burning information.

Use primarily **VIIRS active fire detections** (MODIS as backup).

For each detection:

```
latitude
longitude
timestamp
confidence (l/n/h)
FRP  (Fire Radiative Power — fire intensity in MW)
satellite (VIIRS / MODIS)
```

---

## 11. FIRMS → Grid Conversion

Map every fire to the `5 km × 5 km` grid, then aggregate hourly.

For each grid cell per hour:

```
fire_count
total_FRP
mean_FRP
max_FRP
high_confidence_fire_count
```

Example:

```
Punjab

🔥       🔥
     🔥
          🔥

→ 5 km Grid → per-cell counts
```

---

## 12. FIRMS Temporal Features

Don't just use `fire_count_now`. Create historical windows:

```
fire_count_1h
fire_count_3h
fire_count_6h
fire_count_12h
fire_count_24h
fire_count_48h

FRP_1h
FRP_3h
FRP_6h
FRP_12h
FRP_24h
FRP_48h
```

This is important because smoke doesn't teleport from Punjab to Delhi — it takes time to travel.

---

## 13. Very Important: Transport-Aware Fire Features

This is one of the strongest parts of your proposed system.

```
                 Wind
                   ↓
Punjab 🔥🔥🔥 ───────────────→ Delhi   (HIGH impact)
```

vs

```
Punjab 🔥
       ← wind
Delhi                                     (LOW impact)
```

-   A fire **200 km away upwind** can matter
-   A fire **20 km away downwind** may have much less influence

Therefore calculate:

```
Distance: distance_fire_to_Delhi

Wind alignment:
  wind direction
        ↓
        ↓
  🔥 ───→ ───→ Delhi  → high alignment

  🔥
   ↑
   │ wind
  Delhi → low alignment
```

---

## 14. Transport-Weighted Fire Feature

Create something like:

```
transport_fire_influence
  = FRP × wind_alignment × distance_decay × atmospheric_transport_factor
```

Conceptually:

```
FIRMS
  +
Wind
  +
Distance
  +
PBL
  +
Time
       ↓
Transport Influence Map  → additional CNN channel
```

This is much better than simply saying `"There are 500 fires in Punjab"`.

---

## 15. Should We Use All FIRMS Data?

No. You don't need every fire on Earth.

Use a **source + transport domain.** For Delhi forecasting, focus on:

```
Punjab
Haryana
Delhi NCR
Western UP
Rajasthan
surrounding northwest/north India + buffer
```

The model should learn:

```
fire + favorable wind + suitable PBL → possible Delhi impact
```

rather than:

```
every Punjab fire = Delhi pollution (wrong)
```

---

## 16. Dataset 4 — CAMS

CAMS gives you a **regional atmospheric composition background.**

Useful variables include:

```
PM2.5
PM10
O3
NO2
SO2
CO
NH3
aerosol optical depth
aerosol species
```

depending on the selected CAMS product (global forecasts 0.4° or EAC4 reanalysis 0.75°). See [CAMS.md](./CAMS.md).

The key concept is:

> **CAMS provides the broader atmospheric chemical state; CPCB provides local surface observations.**

---

## 17. CAMS + CPCB

Think of them like this:

```
              CAMS
       Regional background
               │
               ▼
        ┌─────────────┐
        │  5 km grid  │
        └─────────────┘
               ▲
               │
          CPCB stations
         Local observations
```

CAMS may say:

```
Regional PM2.5 = 80
```

while CPCB says:

```
Delhi station = 145
```

The model learns:

> *"Under these meteorological, fire, and local conditions, the local concentration tends to deviate from the coarse regional background in this way."*

That is essentially **spatial downscaling + observational bias correction / data fusion**.

---

## 18. Dataset 5 — Land Use / Land Cover

Add relatively static features.

For each grid cell:

```
urban_fraction
agricultural_fraction
forest_fraction
water_fraction
built_up_fraction
```

Why?

```
urban area      → traffic + industry + construction
agricultural area → potential crop-residue burning
```

---

## 19. Population / Emission Proxies

If available, add:

```
population_density
road_density
industrial_density
built-up density
```

Example:

```
High road density → traffic emissions → NOx / PM
```

---

## 20. Road Features

You don't necessarily need detailed road networks in the first MVP, but useful engineered features are:

```
major_road_density
highway_density
road_length_per_cell
```

These become static CNN channels.

---

## 21. Industrial Features

If data is available:

```
industrial_area
industrial_facility_count
emission_source_density
power_plant_distance
```

Again, these are optional for MVP.

---

## 22. Dataset 6 — Elevation / Topography

Use:

```
elevation
slope
```

Elevation isn't as important as wind/PBL/fire/CPCB, but it provides geographic context.

---

## 23. Dataset 7 — Time Features

The model needs to understand temporal patterns.

Use:

```
hour
day_of_week
day_of_year
month
```

But don't simply feed `hour = 23`. Use **cyclical encoding:**

```
sin(hour)
cos(hour)

sin(day_of_year)
cos(day_of_year)
```

This helps the model understand `23:00 → 00:00` are close together.

---

## 24. Complete Feature List — A. Pollution Features

```
PM2.5, PM10, NO2, SO2, O3, CO, NH3

Historical values:
  t-1, t-3, t-6, t-12, t-24, t-48

Derived:
  PM2.5 rolling mean 3h, 6h, 12h, 24h
  PM2.5 trend, PM2.5 acceleration
```

## 25. B. Weather Features

```
temperature_2m
dew_point
relative_humidity
surface_pressure

u_wind, v_wind, wind_speed, wind_direction

PBLH

temperature_925hPa, temperature_850hPa
precipitation, cloud_cover, solar_radiation

Derived: inversion_strength, inversion_flag, stability_index
```

## 26. C. FIRMS Features

```
fire_count, total_FRP, mean_FRP, max_FRP, high_confidence_fire_count

Historical:
  fire_count_1h, _3h, _6h, _12h, _24h, _48h
  FRP_1h, _3h, _6h, _12h, _24h, _48h

Transport:
  distance_to_Delhi, wind_alignment, transport_time,
  transport_weighted_FRP, upwind_fire_count, upwind_FRP
```

## 27. D. CAMS Features

```
CAMS_PM2.5, CAMS_PM10, CAMS_O3, CAMS_NO2, CAMS_SO2, CAMS_CO, CAMS_NH3
+ aerosol_optical_depth, aerosol_species (product dependent)
```

## 28. E. Geography Features

```
latitude, longitude, elevation,
urban_fraction, agriculture_fraction, forest_fraction, water_fraction, built_up_fraction
```

## 29. F. Human Activity / Emission Proxy Features

```
population_density, road_density, highway_density, industrial_density, power_plant_distance
```

## 30. G. Time Features

```
hour_sin, hour_cos,
day_of_week_sin, day_of_week_cos,
day_of_year_sin, day_of_year_cos,
month
```

---

## 31. What Does One Grid Cell Actually Look Like?

Suppose `Grid Cell ID = 5234`, `Time = 2026-01-15 18:00`:

```
PM2.5                 = 142
PM10                  = 235
NO2                   = 78
O3                    = 32

CAMS_PM2.5            = 95

Temperature           = 12°C
Humidity              = 78%
Wind_U                = -1.4
Wind_V                = 0.8
PBLH                  = 180m

Rain                  = 0
Inversion_strength    = 3.2

Fire_count            = 21
FRP                   = 480 MW
Upwind_FRP            = 350 MW
Transport_FRP         = 290 MW

Urban_fraction        = 0.72
Agriculture_fraction  = 0.12

Population_density    = 8500
Road_density          = 3.5

Hour_sin              = ...
Hour_cos              = ...
```

So one cell becomes:

```
X[cell, time, features]   →  e.g., X[5234, 2026-01-15 18:00, 30]
```

---

## 32. The Actual CNN Input

Suppose your grid is `100 × 100` and you have `30 features`, then at one hour:

```
100 × 100 × 30
```

This is essentially an **image with 30 channels**.

```
Channel 1  → PM2.5
Channel 2  → PM10
Channel 3  → CAMS PM2.5
Channel 4  → temperature
Channel 5  → wind U
Channel 6  → wind V
Channel 7  → PBLH
Channel 8  → humidity
Channel 9  → fire count
Channel 10 → FRP
Channel 11 → transport FRP
...
```

CNN processes this.

---

## 33. Why CNN?

CNN learns spatial relationships.

For example:

```
🔥 🔥 🔥
 ↓ ↓ ↓
→ → → →→ Delhi
```

The CNN can learn that a spatial pattern involving `fire + wind + pollution + distance` is associated with higher pollution downstream.

It doesn't need you to manually write:

```
IF Punjab fire AND wind from northwest AND PBL < 500m THEN Delhi PM2.5 increases
```

The network learns these patterns from historical examples.

---

## 34. Then LSTM

CNN processes each hour sequentially:

```
t-24 → CNN → spatial embedding
t-23 → CNN → spatial embedding
t-22 → CNN → spatial embedding
...
t-1  → CNN → spatial embedding
t    → CNN → spatial embedding
```

Then:

```
        CNN
         ↓
24 hourly spatial embeddings
         ↓
       LSTM
         ↓
temporal representation
         ↓
72-hour forecast
```

---

## 35. Recommended Input Window

For your problem: **48–72 hours** historical window.

```
Past 72 hours → CNN + LSTM → Next 72 hours
```

Why 72?

Because pollution transport and atmospheric buildup have memory:

```
Day 1: Punjab fires
      ↓
Day 1 night: wind transport
      ↓
Day 2: Delhi pollution increases
      ↓
Day 3: accumulation continues
```

---

## 36. Forecast Output

The model should ideally output `t+1` through `t+72`.

For each target grid cell:

```
PM2.5(t+1), PM2.5(t+2), ..., PM2.5(t+72)
```

For Delhi specifically:

```
Delhi PM2.5 → CPCB PM2.5 sub-index → Delhi AQI
```

---

## 37. Better Output Design

Eventually make the model multi-pollutant:

```
                    CNN
                     ↓
                   LSTM
                     ↓
          ┌──────────┴──────────┐
          │                     │
       PM2.5                  O3
          │                     │
       PM10                   NO2
          │                     │
       SO2                    CO
          │                     │
          └──────────┬──────────┘
                     ↓
               AQI Engine
                     ↓
            72-hour AQI forecast
```

This gives you a genuine multi-pollutant AQI rather than merely calling PM2.5-derived AQI "CPCB AQI."

---

## 38. Full Data Pipeline

```
                    RAW DATA
                       │
       ┌───────────────┼────────────────┐
       │               │                │
      CPCB           CAMS             Weather
       │               │                │
       └───────────────┼────────────────┘
                       │
                     FIRMS
                       │
                       ▼
              DATA CLEANING
                       │
          ┌────────────┼────────────┐
          │            │            │
      Time Align   Spatial Map   QC Filter
          │            │            │
          └────────────┼────────────┘
                       ▼
                  5 km GRID
                       │
                       ▼
             FEATURE ENGINEERING
                       │
        ┌──────────────┼──────────────┐
        │              │              │
    Pollution       Weather         Fires
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
              TRANSPORT FEATURES
                       │
                       ▼
               NORMALIZATION
                       │
                       ▼
             SEQUENCE GENERATION
                       │
                       ▼
               CNN + LSTM
                       │
                       ▼
             1–72 HOUR PM2.5
                       │
                       ▼
                CPCB AQI
                       │
                       ▼
           ┌─────────────────────┐
           │ Forecast Dashboard  │
           │                     │
           │ AQI: 312            │
           │ PM2.5: 185 µg/m³    │
           │ Trend: ↑            │
           │ Confidence: 82%     │
           └─────────────────────┘
```

---

## 39. Training Dataset Structure

Organize training data like:

```
timestamp, grid_id, latitude, longitude,
PM25, PM10, NO2, SO2, O3, CO,
CAMS_PM25, CAMS_PM10, CAMS_O3,
temperature, humidity, pressure, wind_u, wind_v, PBLH, rain, radiation,
fire_count, FRP, upwind_FRP, transport_FRP,
urban_fraction, agriculture_fraction, population_density, road_density, elevation,
hour_sin, hour_cos, day_sin, day_cos
```

Then create sequences: `X = [t-71 ... t]`, `y = [t+1 ... t+72]`.

---

## 40. Example Training Sample

Suppose `Prediction time = 15 Jan 18:00`:

```
Input:  13 Jan 18:00 → ... → 15 Jan 17:00  (72 hours)
Target: 15 Jan 19:00 → ... → 18 Jan 18:00  (72 hours)

INPUT   72 × 100 × 100 × 30
          ↓
         CNN
          ↓
   Spatial embeddings
          ↓
        LSTM
          ↓
OUTPUT  72 × target_cells
```

---

## 41. Important Improvement: Don't Predict Every Cell Initially

A full `100 × 100 × 72` output is computationally expensive.

For your **first implementation**, I recommend:

```
Regional input (100×100)
     ↓
CNN
     ↓
Regional spatial representation
     ↓
LSTM
     ↓
Delhi-NCR decoder
     ↓
AQI for Delhi-NCR cells only
```

This keeps the model manageable. Predict Delhi NCR cells, not every cell in Punjab/Rajasthan.

---

## 42. Even Better: Station-Aware Training

Your ground truth exists at CPCB stations.

So during training:

```
Regional grid → CNN → LSTM → target location embedding → predicted PM2.5
Target: CPCB station PM2.5
```

This allows multiple stations in the same grid. Later, inference can produce a full Delhi NCR grid map.

---

## 43. Loss Function

For PM2.5 forecasting, **MSE** is a basic choice, but pollution has extreme spikes. A better starting point is **Huber Loss** (less sensitive to outliers).

You can also weight severe pollution periods more:

```
Normal AQI    → weight 1
Poor AQI      → weight 1.2
Very Poor     → weight 1.5
Severe        → weight 2
```

This makes the model care more about the cases that matter most operationally.

---

## 44. Don't Randomly Split the Data

Very important — **don't do random 80/20.**

Future information can leak into the past.

Instead:

```
2019 ───────── 2023    TRAIN
2024                   VALIDATION
2025                   TEST
```

or equivalent **chronological splits.** Also note: 72-hour sequences overlap — ensure no overlapping windows leak between splits.

---

## 45. Evaluation

Evaluate separately for each forecast horizon:

```
1h, 6h, 12h, 24h, 48h, 72h
```

**Metrics:**

```
MAE, RMSE, R², MAPE / sMAPE
```

For AQI categories:

```
Accuracy, F1, Confusion Matrix
```

Also evaluate: **How well does the model predict severe pollution episodes?** — more meaningful than only average RMSE.

---

## 46. Baselines You Absolutely Need

Before claiming CNN-LSTM works, compare against:

```
Baseline 1: Persistence          PM2.5(t+1) = PM2.5(t)
Baseline 2: Historical average   (hour-of-day, month)
Baseline 3: XGBoost              (same tabular features)
Baseline 4: LSTM without CNN     (no spatial)
Baseline 5: CNN-LSTM             (full)
```

Then demonstrate:

```
Persistence → XGBoost → LSTM → CNN-LSTM  (should improve at each step)
```

This makes your model evaluation much stronger.

---

## 47. Where XGBoost Fits

You don't need XGBoost inside the CNN-LSTM. Use it as a **benchmark:**

```
Same engineered features
       │
       ├────────→ XGBoost
       │
       └────────→ CNN + LSTM
                 ↓ compare
```

This answers: *Does spatial-temporal deep learning actually provide an advantage over a strong tabular model?*

---

## 48. The Most Important Features

If you need to prioritize because collecting everything is difficult:

### Tier 1 — MUST HAVE

```
CPCB PM2.5, CPCB PM10
Temperature, Wind U, Wind V, Humidity, PBLH, Pressure, Rain
FIRMS fire count, FIRMS FRP
CAMS PM2.5, CAMS PM10
```

### Tier 2 — HIGH VALUE

```
NO2, O3, SO2, CO
temperature at 925/850 hPa, inversion strength
transport-weighted FRP, upwind FRP, distance, wind alignment
```

### Tier 3 — IMPROVEMENT

```
population density, road density, industrial density,
land cover, elevation, solar radiation, cloud cover, AOD
```

---

## 49. Final Recommended Feature Set (Core 30 Channels)

For your first serious CNN-LSTM model, target **25–35 channels**, not 100+:

```
┌─────────────────────────────┐
│       CORE 30 CHANNELS      │
├─────────────────────────────┤
│ 1  PM2.5                    │
│ 2  PM10                     │
│ 3  NO2                      │
│ 4  O3                       │
│ 5  SO2                      │
│ 6  CO                       │
│                             │
│ 7  CAMS PM2.5               │
│ 8  CAMS PM10                │
│ 9  CAMS O3                  │
│                             │
│ 10 Temperature              │
│ 11 Humidity                 │
│ 12 Pressure                 │
│ 13 Wind U                   │
│ 14 Wind V                   │
│ 15 PBLH                     │
│ 16 Rain                     │
│ 17 Radiation                │
│ 18 Inversion strength       │
│                             │
│ 19 Fire count               │
│ 20 Total FRP                │
│ 21 Max FRP                  │
│ 22 Upwind FRP               │
│ 23 Transport-weighted FRP   │
│                             │
│ 24 Urban fraction           │
│ 25 Agriculture fraction     │
│ 26 Road density             │
│ 27 Population density       │
│ 28 Elevation                │
│                             │
│ 29 Time encoding sin        │
│ 30 Time encoding cos        │
└─────────────────────────────┘
```

This is a very reasonable starting architecture.

---

## 50. How the "Punjab Fire → Delhi AQI" Mechanism Works

This is the key story you should be able to explain during your presentation.

Imagine:

```
                    WIND
                     ↓
        🔥 🔥 🔥 ───────────────→
        Punjab                   Delhi
                                  │
                                  ▼
                              CPCB stations
```

At `t=0`: FIRMS detects fires.

```
Punjab cell: fire_count ↑, FRP ↑
Weather: wind → southeast, PBLH = low
Transport feature: transport_FRP ↑
CNN sees: 🔥🔥🔥 + wind → + distance → Delhi
```

At the next hours: `t+6 → t+12 → t+18 → t+24`

LSTM learns the temporal evolution. Eventually:

```
Regional fires → transport → Delhi PM2.5 increases → CPCB PM2.5 sub-index increases → AQI increases
```

That is the **core spatiotemporal forecasting mechanism.**

---

## 51. What Each Dataset Contributes

This is a good presentation slide:

| Dataset | Answers |
|---|---|
| **CPCB** | What pollution is actually observed? |
| **CAMS** | What is the regional atmospheric chemical background? |
| **Weather** | Where will pollution move and disperse? |
| **FIRMS** | Where are active biomass-burning sources? |
| **Land use** | What type of emission environment exists here? |
| **Population/Roads** | Where are local anthropogenic sources likely? |
| **Topography** | What geographic constraints exist? |
| **Time features** | What seasonal/diurnal patterns exist? |

---

## 52. The Final System I Recommend + Build Order

Your complete system should therefore be:

```
                    ┌─────────────────┐
                    │      FIRMS      │
                    │ Fires + FRP     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │    WEATHER      │
                    │ Wind/PBL/Temp   │
                    │ RH/Rain/etc.    │
                    └────────┬────────┘
                             │
        ┌────────────────────▼──────────────────┐
        │             5 km GRID                 │
        │                                       │
        │ Punjab → Haryana → Delhi → W. UP     │
        └────────────────────┬──────────────────┘
                             │
       ┌─────────────────────┼──────────────────────┐
       │                     │                      │
       ▼                     ▼                      ▼
    CPCB                  CAMS                 Static
  Observations         Regional chemistry       Features
       │                     │                      │
       └─────────────────────┼──────────────────────┘
                             ▼
                    FEATURE ENGINEERING
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
     Pollution           Transport          Temporal
       history            features           features
          │                  │                  │
          └──────────────────┼──────────────────┘
                             ▼
                  ┌─────────────────────┐
                  │   CNN ENCODER       │
                  │                     │
                  │ Spatial patterns    │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │   LSTM ENCODER      │
                  │                     │
                  │ Temporal patterns   │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │   72h FORECAST      │
                  │                     │
                  │ PM2.5               │
                  │ PM10                │
                  │ O3                  │
                  │ NO2                 │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │    CPCB AQI         │
                  │    CALCULATOR       │
                  └──────────┬──────────┘
                             ▼
              ┌─────────────────────────────┐
              │       FINAL OUTPUT          │
              │                             │
              │ 1h ───────── AQI            │
              │ 6h ───────── AQI            │
              │ 12h ──────── AQI            │
              │ 24h ──────── AQI            │
              │ 48h ──────── AQI            │
              │ 72h ──────── AQI            │
              │                             │
              │ Spatial map                 │
              │ Confidence / uncertainty    │
              └─────────────────────────────┘
```

### The most important takeaway

> Your model is **not simply "CPCB + FIRMS → LSTM."** It is: *Regional atmospheric state + ground observations + fire emissions + meteorology + transport physics + spatial context → CNN spatial learning → LSTM temporal learning → 72-hour pollutant forecast → CPCB AQI.* And the **5 km grid is the common language** that allows all these datasets to come together.

### Build order I recommend

Don't try to implement all 30 features on day one. Build it incrementally:

```
PHASE 1: CPCB + Weather → LSTM → PM2.5 forecast
PHASE 2: CPCB + Weather + CAMS → CNN + LSTM
PHASE 3: Add FIRMS + FRP → CNN + LSTM
PHASE 4: Add transport-weighted FIRMS → CNN + LSTM
PHASE 5: Add land use / roads / population → Full model
PHASE 6: Multi-pollutant prediction → Full CPCB AQI
```

---

*Last updated: 2026-09-02 | System: CNN-LSTM on 5×5 km grid | Related: [README.md](./README.md) · [CAMS.md](./CAMS.md) · [phase.md](./phase.md)*
