# Delhi NCR 72-Hour AQI Forecasting System --- Data Specification

## 1. Project Objective

Build a real-time, high-resolution **72-hour air-quality forecasting
system for Delhi-NCR** using a common **5 km × 5 km square grid** and a
**CNN + LSTM** model.

The system combines:

-   Ground observations from CPCB
-   Atmospheric composition from CAMS
-   Historical/forecast meteorology
-   Satellite active-fire observations from NASA FIRMS
-   Land-use, population, road and industrial-source proxies
-   Static geographic/topographic information
-   Time and seasonal features

### Core pipeline

``` text
CPCB + CAMS + Weather + FIRMS + Static Data
                    ↓
            Quality Control
                    ↓
       Time + Spatial Alignment
                    ↓
             5 km × 5 km Grid
                    ↓
          Feature Engineering
                    ↓
             CNN Spatial Encoder
                    ↓
              LSTM Temporal Encoder
                    ↓
        1–72 hour pollutant forecast
                    ↓
              CPCB AQI Engine
                    ↓
       Delhi-NCR AQI + uncertainty
```

------------------------------------------------------------------------

# 2. Geographic Domain

Do **not** use Delhi alone.

The model needs a regional source-and-transport domain that can capture
pollution entering Delhi-NCR.

### Primary domain

-   Delhi
-   Haryana
-   Punjab
-   Western Uttar Pradesh
-   Relevant parts of Rajasthan
-   Surrounding buffer sufficient to capture major transport pathways

### Common spatial representation

**Grid:** 5 km × 5 km square cells.

Every gridded dataset will eventually be transformed into:

``` text
timestamp × grid_y × grid_x × feature
```

CPCB stations remain station-level observations during training and are
mapped to grid cells for spatial modeling.

------------------------------------------------------------------------

# 3. Data Sources --- Summary

  --------------------------------------------------------------------------------------------------
  Dataset         Main source                Role                   Resolution / form Priority
  --------------- -------------------------- ---------------------- ----------------- --------------
  CPCB CAAQMS     CPCB                       Ground truth + current 15-min            MUST
                                             state                  observations →    
                                                                    hourly            

  CAMS            Copernicus Atmosphere      Regional chemical      Gridded,          MUST
                  Monitoring Service         background + forecast  hourly/3-hourly   
                                                                    depending         
                                                                    variable          

  Weather         ERA5 for history;          Transport, dispersion, Hourly gridded    MUST
                  operational forecast       inversion              history; forecast 
                  source for deployment                             for future        

  FIRMS           NASA                       Active fires + fire    Satellite         MUST
                                             intensity              detections;       
                                                                    aggregate to      
                                                                    hourly/grid       

  Land Cover      ESA / Copernicus / other   Emission/environment   Static raster     HIGH
                  authoritative source       context                                  

  Population      WorldPop / GHSL            Local emission proxy   Static raster     MEDIUM

  Roads           OpenStreetMap /            Traffic-emission proxy Vector → grid     MEDIUM
                  authoritative road data                                             

  Industry        Government/authoritative   Point/area emission    Vector/grid       MEDIUM
                  inventories where          proxy                                    
                  available                                                           

  Elevation       SRTM / Copernicus DEM      Topographic context    Static raster     MEDIUM

  Calendar/time   Derived locally            Diurnal/seasonal       Hourly            MUST
                                             patterns                                 
  --------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

# 4. Dataset A --- CPCB CAAQMS

## Purpose

CPCB is the **ground-truth observation layer**.

It tells the model what pollutant concentrations are actually being
observed at monitoring stations.

CPCB's CAAQM transmission protocol specifies 15-minute average values
and updates the station data every 15 minutes.

**Official source:**\
https://cpcb.nic.in/

**CAAQM data-transmission protocol:**\
https://cpcb.nic.in/upload/caaqm/Protocol_CAAQM.pdf

## Data to collect

### Raw temporal data

Prefer the highest-quality historical and current observations
available.

Target:

-   Historical: ideally multiple years, preferably \~2021--present or
    the longest consistent period available
-   Real-time: latest available observations
-   Raw resolution: 15-minute where available
-   Modeling resolution: hourly

### Station metadata

``` text
station_id
station_name
latitude
longitude
city
state
agency
station_type (if available)
```

### Pollutants

``` text
PM2.5
PM10
NO2
SO2
O3
CO
NH3
```

Keep additional available parameters in the raw archive rather than
deleting them.

## CPCB-derived features

### Current concentration

``` text
PM2.5
PM10
NO2
SO2
O3
CO
NH3
```

### Historical lags

``` text
PM2.5_t-1
PM2.5_t-3
PM2.5_t-6
PM2.5_t-12
PM2.5_t-24
PM2.5_t-48
PM2.5_t-72
```

Equivalent lags can be generated for other pollutants.

### Rolling statistics

``` text
PM2.5_mean_3h
PM2.5_mean_6h
PM2.5_mean_12h
PM2.5_mean_24h
PM2.5_std_6h
PM2.5_std_24h
```

### Trend features

``` text
PM2.5_change_1h
PM2.5_change_3h
PM2.5_change_6h
PM2.5_slope_6h
PM2.5_slope_24h
```

### Data-quality features

``` text
observation_available
missing_fraction
station_data_quality
```

## Important rule

Do **not** use AQI as the primary model target.

Preferred:

``` text
CPCB pollutant observations
        ↓
CNN + LSTM predicts pollutant concentration
        ↓
CPCB AQI calculation
```

AQI is a derived output.

------------------------------------------------------------------------

# 5. Dataset B --- CAMS Atmospheric Composition

## Purpose

CAMS provides a regional atmospheric-composition background and
forecast.

It is particularly useful where CPCB stations are sparse.

CAMS global atmospheric-composition forecasts provide many chemical
species and aerosol types, with hourly single-level and 3-hourly
multi-level data depending on variable.

**Official source:**\
https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts

## Main variables

Prioritize:

``` text
PM2.5
PM10
O3
NO2
SO2
CO
NH3
```

Useful additional aerosol variables where available:

``` text
black carbon
organic matter
sulphate
nitrate
ammonium
dust
sea salt
aerosol optical depth
```

## Derived features

``` text
CAMS_PM2.5
CAMS_PM10
CAMS_O3
CAMS_NO2
CAMS_SO2
CAMS_CO
CAMS_NH3
```

### CPCB--CAMS relationship features

``` text
CPCB_PM2.5 - CAMS_PM2.5
CPCB_PM2.5 / CAMS_PM2.5
CPCB_PM10 - CAMS_PM10
```

These can help the model learn local bias/downscaling relationships.

## Important limitation

Regridding CAMS onto a 5 km grid does **not** create genuine 5 km
chemical information.

CAMS supplies the **regional background**; local spatial detail comes
from CPCB, FIRMS, weather and static/source features.

------------------------------------------------------------------------

# 6. Dataset C --- Meteorology / Weather

## Purpose

Weather determines how pollutants:

-   move
-   disperse
-   accumulate
-   chemically transform
-   are removed by precipitation

### Historical weather

Use **ERA5** for consistent historical training data.

Official ERA5 single levels:
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels

Official ERA5 pressure levels:
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels

ERA5 provides hourly global reanalysis. Single-level variables include
10 m wind components, 2 m temperature, pressure and precipitation;
pressure-level data can provide upper-air temperature and wind.

## Essential surface variables

``` text
2m_temperature
2m_dewpoint_temperature
surface_pressure
10m_u_wind
10m_v_wind
total_precipitation
boundary_layer_height
surface_solar_radiation
cloud_cover
```

Derived:

``` text
relative_humidity
wind_speed
wind_direction
```

## Upper-air variables

At minimum:

``` text
temperature_925hPa
temperature_850hPa
u_wind_925hPa
v_wind_925hPa
u_wind_850hPa
v_wind_850hPa
```

## Inversion features

Derive:

``` text
inversion_strength_925
inversion_strength_850
inversion_flag
```

Example:

``` text
T_925 - T_surface > threshold
        ↓
possible inversion
        ↓
pollution trapping
```

## PBL features

``` text
PBLH
PBLH_3h_mean
PBLH_6h_mean
PBLH_12h_mean
```

Low boundary-layer height is particularly important for winter pollution
accumulation.

------------------------------------------------------------------------

# 7. Weather for Real-Time 72-Hour Prediction

This is an important distinction.

### Training

Use historical reanalysis:

``` text
ERA5
```

### Deployment

We cannot use future ERA5 observations because they are not known at
forecast time.

The real-time system therefore needs a **future weather forecast
source**, such as an operational numerical weather prediction dataset.

Conceptually:

``` text
Historical:
ERA5 → training weather

Real-time:
Weather forecast → next 72h meteorology
```

The future-weather input must include, as far as possible:

``` text
temperature
wind_u
wind_v
humidity
PBLH
pressure
precipitation
radiation/cloud variables
upper-air temperature/wind
```

This prevents future-information leakage.

------------------------------------------------------------------------

# 8. Dataset D --- NASA FIRMS Active Fires

## Purpose

FIRMS identifies active fires and provides information about fire
location and intensity.

Official source: https://firms.modaps.eosdis.nasa.gov/active_fire/

FIRMS provides MODIS, VIIRS and Landsat active-fire products. VIIRS 375
m products are especially useful for this project. FIRMS also provides
archive data for historical analysis and near-real-time services.

## Preferred products

Use:

``` text
VIIRS 375m / S-NPP
VIIRS 375m / NOAA-20
VIIRS 375m / NOAA-21
```

Use MODIS as an additional/legacy source if useful.

## Raw fire features

``` text
latitude
longitude
acquisition_date
acquisition_time
satellite
confidence
FRP
brightness_temperature (where available)
scan
track
```

## Grid aggregation

Map every fire detection to the 5 km × 5 km grid.

For each cell/hour:

``` text
fire_count
total_FRP
mean_FRP
max_FRP
high_confidence_fire_count
```

## Temporal fire features

``` text
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

## Transport-aware fire features

These are critical.

``` text
distance_to_Delhi
distance_to_target_cell
wind_alignment
upwind_fire_count
upwind_FRP
transport_weighted_FRP
estimated_transport_time
```

Conceptual transport score:

``` text
transport_influence
    =
FRP
× wind_alignment
× distance_decay
× atmospheric_transport_factor
```

A more advanced version can use HYSPLIT or another trajectory/plume
model to construct transport influence maps.

## Important FIRMS limitation

FIRMS active-fire observations are satellite detections, not continuous
hourly measurements.

Therefore:

``` text
satellite detections
        ↓
hourly aggregation
```

does not mean the satellite observes every location every hour.

The model must account for this observation sparsity.

FIRMS official documentation recommends standard science-quality
products for scientific analysis once they replace near-real-time data
in the archive.

------------------------------------------------------------------------

# 9. Dataset E --- Land Cover / Land Use

## Purpose

Provides relatively static spatial context and helps distinguish:

-   urban areas
-   agricultural regions
-   forests
-   water
-   other land types

## Features

``` text
urban_fraction
agriculture_fraction
forest_fraction
water_fraction
built_up_fraction
```

Optional:

``` text
cropland_fraction
grassland_fraction
bare_land_fraction
```

## Transformation

Rasterize/regrid to the common 5 km grid.

------------------------------------------------------------------------

# 10. Dataset F --- Population

## Purpose

Population density acts as a proxy for potential local anthropogenic
activity.

Possible sources:

-   WorldPop
-   GHSL
-   Other authoritative population datasets

## Features

``` text
population_density
population_total
urban_population_fraction
```

Aggregate to each 5 km grid cell.

------------------------------------------------------------------------

# 11. Dataset G --- Roads / Traffic Proxies

## Purpose

Traffic is an important urban emission source.

Possible source:

-   OpenStreetMap
-   Government road datasets where available

## Features

``` text
road_length
road_density
major_road_length
highway_length
highway_density
```

Optional:

``` text
distance_to_major_road
```

Do not attempt detailed traffic emissions in the first MVP unless
reliable traffic/emission data is available.

------------------------------------------------------------------------

# 12. Dataset H --- Industrial / Emission Sources

## Purpose

Represent persistent local emission sources.

Use government or authoritative inventories wherever available.

## Features

``` text
industrial_facility_count
industrial_area
point_source_count
emission_source_density
distance_to_major_source
power_plant_distance
```

If actual emissions inventories are available:

``` text
NOx_emission
SO2_emission
PM_emission
VOC_emission
NH3_emission
```

These are higher-value than simple facility counts.

------------------------------------------------------------------------

# 13. Dataset I --- Elevation / Topography

## Purpose

Provide geographic context that can influence atmospheric flow and
dispersion.

Possible sources:

-   SRTM
-   Copernicus DEM

## Features

``` text
elevation
mean_elevation
elevation_std
slope
```

Usually static.

------------------------------------------------------------------------

# 14. Dataset J --- Time / Calendar Features

These are generated locally; no external data source is required.

## Raw time

``` text
hour
day_of_week
day_of_year
month
```

## Cyclical encoding

``` text
hour_sin
hour_cos

day_of_year_sin
day_of_year_cos

day_of_week_sin
day_of_week_cos
```

These capture:

-   daily cycles
-   weekly patterns
-   seasonal behavior

------------------------------------------------------------------------

# 15. Final Feature Inventory

## A. Pollution

``` text
PM2.5
PM10
NO2
SO2
O3
CO
NH3

pollution_lags
rolling_means
rolling_std
pollution_trends
```

## B. CAMS

``` text
CAMS_PM2.5
CAMS_PM10
CAMS_O3
CAMS_NO2
CAMS_SO2
CAMS_CO
CAMS_NH3

aerosol_species
AOD
```

## C. Weather

``` text
temperature
dewpoint
relative_humidity
surface_pressure

wind_u
wind_v
wind_speed
wind_direction

PBLH
precipitation
radiation
cloud_cover

temperature_925
temperature_850
wind_u_925
wind_v_925
wind_u_850
wind_v_850

inversion_strength
inversion_flag
```

## D. FIRMS

``` text
fire_count
total_FRP
mean_FRP
max_FRP
high_confidence_fire_count

fire_count_3h
fire_count_6h
fire_count_12h
fire_count_24h
fire_count_48h

FRP_3h
FRP_6h
FRP_12h
FRP_24h
FRP_48h

upwind_fire_count
upwind_FRP
wind_alignment
distance_to_target
estimated_transport_time
transport_weighted_FRP
```

## E. Static geography

``` text
latitude
longitude
elevation
slope
urban_fraction
agriculture_fraction
forest_fraction
water_fraction
built_up_fraction
```

## F. Human activity

``` text
population_density
road_density
highway_density
industrial_density
power_plant_distance
```

## G. Time

``` text
hour_sin
hour_cos
day_of_week_sin
day_of_week_cos
day_of_year_sin
day_of_year_cos
```

------------------------------------------------------------------------

# 16. Priority Classification

## Tier 1 --- Required for MVP

``` text
CPCB PM2.5
CPCB PM10

Temperature
Wind U
Wind V
Humidity
PBLH
Pressure
Precipitation

CAMS PM2.5
CAMS PM10

FIRMS fire_count
FIRMS FRP
```

## Tier 2 --- Strongly Recommended

``` text
NO2
O3
SO2
CO

925/850 hPa temperature
925/850 hPa wind

inversion strength

upwind FRP
transport-weighted FRP
wind alignment
distance
```

## Tier 3 --- Model improvement

``` text
AOD
aerosol species
land cover
population
road density
industrial sources
elevation
radiation
cloud cover
```

------------------------------------------------------------------------

# 17. Target Variables

## Phase 1

Primary target:

``` text
PM2.5
```

Output:

``` text
PM2.5_t+1
PM2.5_t+2
...
PM2.5_t+72
```

Then calculate:

``` text
PM2.5 forecast
      ↓
CPCB PM2.5 sub-index
      ↓
AQI
```

## Phase 2

Multi-pollutant:

``` text
PM2.5
PM10
O3
NO2
```

## Phase 3

Potential full target:

``` text
PM2.5
PM10
NO2
SO2
CO
O3
NH3
```

Then:

``` text
pollutant predictions
        ↓
CPCB sub-indices
        ↓
maximum valid sub-index
        ↓
AQI
```

------------------------------------------------------------------------

# 18. Real-Time Data Architecture

The system must distinguish between **observed/past data** and **future
forecast data**.

``` text
                    NOW
                     │
       ┌─────────────┼─────────────┐
       │             │             │
   CPCB latest    CAMS latest   Latest weather
   observations    analysis       observations
       │             │             │
       └─────────────┼─────────────┘
                     │
              HISTORICAL STATE
                     │
                     ▼
                  CNN-LSTM
                     ▲
                     │
       FUTURE WEATHER + CAMS FORECAST
                     │
                     │
              +1h ... +72h
                     │
                     ▼
              72h PM2.5/AQI
```

### Important

CPCB does **not** provide the future 72 hours.

CPCB supplies:

-   current observations
-   historical observations

Future atmospheric conditions must come from weather/CAMS forecasts and
the model.

------------------------------------------------------------------------

# 19. Common Time Standard

Store all raw timestamps in:

``` text
UTC
```

Then create:

``` text
local_time = Asia/Kolkata
```

for presentation and local feature generation.

Internally, all datasets should be aligned to one consistent timezone.

Final modeling timestep:

``` text
1 hour
```

------------------------------------------------------------------------

# 20. Common Spatial Standard

All gridded datasets should eventually use:

``` text
5 km × 5 km square grid
```

Each cell should have:

``` text
grid_id
row
column
center_lat
center_lon
geometry
```

Example:

``` text
grid_id = DELHI_042_051
row = 42
column = 51
```

------------------------------------------------------------------------

# 21. Dataset Storage Structure

``` text
data/
│
├── raw/
│   ├── cpcb/
│   │   ├── station_metadata/
│   │   └── observations_15min/
│   │
│   ├── cams/
│   │
│   ├── weather/
│   │
│   ├── firms/
│   │
│   ├── landcover/
│   ├── population/
│   ├── roads/
│   ├── industry/
│   └── elevation/
│
├── processed/
│   ├── cpcb_hourly/
│   ├── cams_hourly/
│   ├── weather_hourly/
│   ├── firms_hourly/
│   └── static_grid/
│
├── grid/
│   └── delhi_ncr_5km.geojson
│
└── features/
    ├── hourly_grid/
    └── training_sequences/
```

------------------------------------------------------------------------

# 22. Recommended Feature Tensor

For one hour:

``` text
X_t ∈ R^(H × W × F)
```

where:

``` text
H = number of grid rows
W = number of grid columns
F = number of features/channels
```

Example:

``` text
100 × 100 × 30
```

For 72 hours of historical input:

``` text
72 × 100 × 100 × 30
```

Conceptually:

``` text
Hour -71 ──→ CNN ──→ spatial embedding
Hour -70 ──→ CNN ──→ spatial embedding
...
Hour -1  ──→ CNN ──→ spatial embedding
Hour  0  ──→ CNN ──→ spatial embedding
                              │
                              ▼
                            LSTM
                              │
                              ▼
                       72-hour forecast
```

------------------------------------------------------------------------

# 23. Training Window

Recommended initial configuration:

``` text
Historical input = 72 hours
Forecast horizon = 72 hours
```

So each training example is:

``` text
Input:
t-71 ... t

Target:
t+1 ... t+72
```

A shorter input window such as 24 or 48 hours can be tested as an
ablation.

------------------------------------------------------------------------

# 24. Data Leakage Rules

These rules are mandatory.

### Never use future CPCB observations as input.

For prediction at time `t`:

``` text
Allowed:
CPCB ≤ t

Not allowed:
CPCB > t
```

### Never use future ERA5 observations.

For real-time forecasting:

``` text
Historical:
ERA5

Future:
operational weather forecast
```

### FIRMS

Only use fire detections that would actually have been available by
forecast issue time.

### CAMS

Use the forecast/analysis information that would have been available at
the corresponding historical forecast initialization time whenever
possible.

This is essential for a realistic evaluation.

------------------------------------------------------------------------

# 25. Evaluation Dataset

Use chronological splitting.

Example:

``` text
2021–2023 → TRAIN
2024       → VALIDATION
2025       → TEST
```

Avoid random train/test splitting because it can leak temporal
information.

Evaluate separately for:

``` text
+1h
+6h
+12h
+24h
+48h
+72h
```

Metrics:

``` text
MAE
RMSE
R²
```

For AQI categories:

``` text
Accuracy
Precision
Recall
F1
Confusion Matrix
```

Also specifically evaluate:

``` text
severe pollution episodes
winter inversion episodes
post-monsoon fire episodes
high-FRP transport episodes
```

------------------------------------------------------------------------

# 26. Baseline Models

Before claiming CNN-LSTM is effective, compare against:

``` text
1. Persistence
2. Historical/seasonal baseline
3. XGBoost
4. LSTM only
5. CNN only
6. CNN + LSTM
```

The same leakage-safe data should be used for comparison.

------------------------------------------------------------------------

# 27. Data Acquisition Order

Do not download everything simultaneously.

Recommended order:

``` text
STEP 1
CPCB
   ↓
Station observations + metadata

STEP 2
ERA5
   ↓
Historical weather

STEP 3
CAMS
   ↓
Historical atmospheric composition

STEP 4
FIRMS
   ↓
Historical active fires + FRP

STEP 5
Static datasets
   ↓
Land cover + population + roads + elevation

STEP 6
Common 5 km grid
   ↓
Spatial alignment

STEP 7
Hourly alignment
   ↓
Feature engineering

STEP 8
Training tensors
   ↓
CNN + LSTM
```

------------------------------------------------------------------------

# 28. Minimum Viable Dataset

If time/storage is limited, start with:

``` text
CPCB:
    PM2.5
    PM10

Weather:
    temperature
    wind_u
    wind_v
    humidity
    PBLH
    pressure
    precipitation

CAMS:
    PM2.5
    PM10

FIRMS:
    fire_count
    total_FRP
```

This is enough to build the first meaningful prototype.

Then add:

``` text
NO2
O3
inversion
transport-weighted FRP
land use
population
roads
```

incrementally.

------------------------------------------------------------------------

# 29. Final Data-to-Model Mapping

  --------------------------------------------------------------------------
  Source                  Raw data                Final model features
  ----------------------- ----------------------- --------------------------
  CPCB                    Station pollutant       PM2.5, PM10, NO2, O3, SO2,
                          observations            CO, NH3 + lags/trends

  CAMS                    Atmospheric composition Regional
                                                  PM2.5/PM10/O3/NO2/etc.
                                                  background

  ERA5                    Historical weather      Wind U/V, temperature, RH,
                                                  PBLH, pressure, rain,
                                                  inversion

  Weather forecast        Future weather          +1 to +72h meteorological
                                                  inputs

  FIRMS                   Fire detections         Fire count, FRP, upwind
                                                  FRP, transport-weighted
                                                  FRP

  Land cover              Raster                  Urban/agriculture/forest
                                                  fractions

  Population              Raster                  Population density

  Roads                   Vector                  Road/highway density

  Industry                Point/vector            Source density and
                                                  distances

  DEM                     Raster                  Elevation/slope

  Time                    Timestamp               Cyclical hour/day/season
                                                  features
  --------------------------------------------------------------------------

------------------------------------------------------------------------

# 30. Recommended Final Architecture

``` text
                         DATA SOURCES
                              │
       ┌──────────┬───────────┼───────────┬──────────┐
       ▼          ▼           ▼           ▼          ▼
     CPCB       CAMS       Weather      FIRMS     Static
   Ground      Chemistry   Historical    Fires     Data
   Truth                    + Forecast
       │          │           │           │          │
       └──────────┴───────────┼───────────┴──────────┘
                              ▼
                     QUALITY CONTROL
                              ▼
                    TIME SYNCHRONIZATION
                              ▼
                    SPATIAL REGRIDDING
                              ▼
                       5 km × 5 km
                              ▼
                    FEATURE ENGINEERING
                              ▼
                 TRANSPORT-AWARE FEATURES
                              ▼
                    NORMALIZATION/MASKS
                              ▼
                   72h HISTORICAL WINDOW
                              ▼
                         CNN ENCODER
                              ▼
                      SPATIAL EMBEDDING
                              ▼
                         LSTM ENCODER
                              ▼
                     72h MULTI-STEP OUTPUT
                              ▼
                 PM2.5 / PM10 / O3 / NO2
                              ▼
                     CPCB AQI ENGINE
                              ▼
                 ┌────────────────────────┐
                 │ Delhi-NCR AQI Forecast │
                 │ +1h ... +72h           │
                 │ Spatial map             │
                 │ Uncertainty             │
                 └────────────────────────┘
```

------------------------------------------------------------------------

# 31. First Implementation Milestone

The first milestone is **not the CNN-LSTM**.

It is:

> **Create a clean, hourly CPCB observation dataset with station
> metadata.**

Expected output:

``` text
processed/cpcb_hourly/
    cpcb_hourly.parquet
    stations.parquet
```

with:

``` text
timestamp
station_id
station_name
latitude
longitude

PM2.5
PM10
NO2
SO2
O3
CO
NH3

quality/missingness fields
```

Once this is reliable, we add ERA5 and build the common time axis.

------------------------------------------------------------------------

## Official source references

-   CPCB: https://cpcb.nic.in/
-   CPCB CAAQM transmission protocol:
    https://cpcb.nic.in/upload/caaqm/Protocol_CAAQM.pdf
-   CAMS global atmospheric-composition forecasts:
    https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts
-   ERA5 hourly single levels:
    https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
-   ERA5 hourly pressure levels:
    https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels
-   NASA FIRMS active fire:
    https://firms.modaps.eosdis.nasa.gov/active_fire/
-   NASA FIRMS archive/API: https://firms.modaps.eosdis.nasa.gov/
