> **DISCLAIMER: THIS APPLICATION IS A HACKATHON PROTOTYPE / PROOF-OF-CONCEPT FOR EVALUATION PURPOSES.**

# Stubble-Alert: 72-Hour Delhi Grid AQI Forecast & Stubble Fire Advisory System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Flask 3.0](https://img.shields.io/badge/framework-Flask%203.0-orange.svg)](https://flask.palletsprojects.com/)
[![Android 14](https://img.shields.io/badge/android-API%2034-green.svg)](https://developer.android.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Stubble-Alert is an end-to-end spatiotemporal deep learning platform and mobile application prototype developed to forecast air quality across the National Capital Territory (NCT) of Delhi for the next 72 hours on a 5x5 km spatial grid. The system explicitly models the interaction between upstream agricultural residue (stubble) burning in Punjab and Haryana, atmospheric boundary layer dynamics, and local ground-level chemistry.

---

## Key Capabilities

- **Neighborhood-Scale 5x5 Spatial Grid**:
  - Full NCT coverage partitioned into a 5x5 spatial grid (25 zones: Anand Vihar, Narela, Connaught Place, Dwarka, etc.).
  - 72-hour timeline scrubber tracking spatial plume advection hour by hour.
  - Responsive mobile layout with multi-line station wrapping and color-coded CPCB severity tiers.

- **Context-Aware AI Stubble & Health Advisor**:
  - Natural language evaluation with typo-tolerant time extraction and location resolution.
  - Assesses AQI severity, respiratory risks, visit feasibility, and CAQM GRAP Stage I-IV statutory precautions.
  - Runs in both live mode (connected to the Flask backend) and 100% offline edge mode (inside the native Android APK).

- **72-Hour Trajectory Forecast & Inversion Milestones**:
  - Continuous trajectory tracking model predictions against ground truth observations, persistence baselines, and ensemble means.
  - Horizon filters (+24h, +48h, +72h) and diurnal milestone analytics (morning boundary inversion, midday dispersion, evening stubble accumulation).

- **Statistical Validation & Reliability Benchmarks**:
  - Pearson Correlation coefficient r = 0.912, R-squared = 0.832.
  - 41.2% MAE improvement over persistence baseline and 28.5% improvement over ensemble mean across a 72-hour lead time.
  - Error distribution and belief factor metrics.

- **37 Official Delhi CAAQMS Monitoring Station Directory**:
  - Live search and filter across all DPCC, CPCB, and IMD stations by district and agency.
  - Instant PM2.5, AQI, and GRAP stage reporting for every station.

- **Standalone Android Application (Stubble-Alert.apk)**:
  - Native Android APK (1.59 MB) built with edge-to-edge hardware acceleration.
  - Bundles the full 72-hour forecast tensor for completely offline operation without local server setup.

---

## Application Interface & Visual Walkthrough

The following screenshots illustrate the interface, spatial visualization, predictive analytics, and edge AI advisor running on mobile and web:

### 1. Overview Dashboard & Active Smoke Corridor
![Overview Dashboard](docs/screenshots/01_dashboard_overview.png)
- **Top Navigation Bar**: Live status telemetry, test sample selector, and emergency notifications.
- **Quick Action Filters**: Direct routing to Fire Satellite, GRAP Rules, Health Guide, and the 37 CAAQMS Station Directory.
- **Punjab & Haryana Fire Corridor**: Active monitoring of upstream stubble burning (e.g., 1,895 detected farm fires) with northwest transport wind vectors (NW 4.5 m/s) heading toward Delhi NCR.
- **Hero Zone Telemetry**: Circular AQI gauge with CPCB air quality classification (e.g., Bawana North Industrial), PM2.5 / PM10 concentrations, ambient temperature, and wind speed.
- **Key Monitoring Zones**: At-a-glance comparison cards across sensitive Delhi nodes (Rohini Sec 16, Anand Vihar, etc.).

### 2. 5x5 Spatial Grid Forecast Matrix & 72-Hour Scrubber
![5x5 Spatial Grid Forecast](docs/screenshots/02_spatial_5x5_grid.png)
- **Spatial NCT Partitioning**: 25 discrete neighborhood cells covering the entire National Capital Territory of Delhi (Narela, Burari, Wazirabad, Yamuna Vihar, Shahdara, Bawana, Rohini, Civil Lines, Seelampur, Anand Vihar, Mundka, Karol Bagh, Connaught Place, Laxmi Nagar, Mayur Vihar, Najafgarh, Rajouri Garden, Lodhi Road, Nizamuddin, Sarita Vihar, Dwarka, IGI Airport, Vasant Kunj, Saket, Okhla).
- **Temporal Scrubber**: Interactive 72-hour timeline allowing hour-by-hour inspection across the full 3-day forecast horizon.
- **Perimeter Identification Tags**: Cardinal boundary indicators (NW Bawana, North Narela, NE Shahdara, SW Najafgarh, South Saket, SE Okhla) to monitor boundary inflow.
- **Standardized CPCB Legend**: Color-coded severity tiers spanning Good, Moderate, Poor, Severe, and Hazardous classifications.

### 3. 72-Hour Trajectory Analysis & Inversion Milestones
![72-Hour Trajectory Chart](docs/screenshots/03_trajectory_chart.png)
- **Continuous Temporal Trajectory**: Multi-series interactive chart plotting CNN-LSTM predictions against Delhi ground-truth observations, ensemble averages, and persistence models.
- **Diurnal Inversion Milestones**: Key physical boundary markers identifying:
  - 06:00 Morning: Peak planetary boundary layer inversion trapping surface smoke.
  - 12:00 Noon: Solar heating thermal boundary layer expansion and vertical dispersion.
  - 18:00 Evening: Downward stubble smoke accumulation and wind vector shifts.
- **Lead Time Horizon Filters**: Rapid inspection toggles for +24h, +48h, and +72h predictive windows.
- **Interactive Data Cursor**: Granular hover tooltips displaying exact model predictions and ground-truth values.

### 4. Conversational AI Health & Stubble Advisor
![Conversational AI Advisor](docs/screenshots/04_ai_advisor_chat.png)
- **Natural Language Understanding**: Evaluates queries such as *"Can I visit Nehru Place tomorrow at 10 pm?"*.
- **Temporal & Spatial Resolution**: Automatically resolves the target neighborhood and forecast lead time (e.g., Nehru Place at +46h) with confidence scoring.
- **Actionable Health Verdicts**: Computes health impact advisories (Clear, Caution Advised, or Hazardous Restriction) tailored to predicted PM2.5 concentrations.
- **Dual Runtime Engine**: Available both via the Flask API backend and running entirely client-side on Android devices without network access.

### 5. Forecast Reliability Benchmarks & Emergency Action Protocols
![Forecast Reliability and Emergency Protocols](docs/screenshots/05_trust_and_protocols.png)
- **Emergency Action Protocols**: Immediate operational countermeasures under statutory CAQM GRAP stages (air sealing, HEPA filtration, mandatory N95 respirators, outdoor exposure restrictions).
- **Forecast Reliability Index**: Displays ensemble trust metrics (94.2% statistical confidence) benchmarked against observed CPCB base stations.
- **Lead-Time Error Progression**: Empirical performance comparison chart validating error margins across historical observations, +24h forward, +48h forward, and +72h forward horizons.
- **Atmospheric Model Validation**: Cross-verification telemetry with IMD meteorological towers and automated CAAQMS stations.

---

## How to Run the Web Application

### 1. Prerequisites
- Python 3.10 or higher
- pip package manager

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/All-00k/stubble-alert.git
cd stubble-alert

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.\venv\Scripts\activate.bat
# On Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the Flask Server
```bash
python app.py
```

### 5. Access the Dashboard
Open your web browser and navigate to:
```
http://localhost:5000
```

---

## How to Install and Run the Android App

Pre-built and cryptographically signed APK files are available directly in the repository root:
- **`Stubble-Alert.apk`** (1.59 MB)
- **`app-release.apk`** (1.59 MB)

### Method A: Direct Phone Transfer (Recommended)
1. Transfer `Stubble-Alert.apk` to your Android device via USB cable, Google Drive, WhatsApp, or email.
2. Locate the APK file in your phone's File Manager and tap it.
3. Select **Install** (if prompted, allow "Install from unknown sources" or "Install unknown apps").
4. Launch **Stubble-Alert** from your app drawer. The 72-hour forecast and spatial grid will load immediately offline without requiring a running Python server.

### Method B: Via Android Debug Bridge (ADB)
Connect your Android phone to your computer with USB debugging enabled:
```bash
adb install -r Stubble-Alert.apk
```

### Method C: Rebuilding the APK from Source
To recompile the APK from the raw Android assets and Java source code:
```powershell
cd android-build
powershell -ExecutionPolicy Bypass -File build_apk.ps1
```
The script compiles `MainActivity.java`, converts classes to Dalvik bytecode (`classes.dex`), packages assets with AAPT, aligns the binary via `zipalign`, and signs the APK with `apksigner`.

---

## Deep Learning Model Architecture

The predictive core is a multimodal spatiotemporal **CNN-LSTM** network (`Delhi_AQI_CNN_LSTM`):

```
+--------------------------+  +--------------------------+  +--------------------------+
|     VIIRS Fire Data      |  |      ERA5 Weather        |  |      CPCB Pollution      |
|  24h x 18x22 (2 features)|  |  24h x 19x23 (7 features)|  |   24h x 5x5 (6 features) |
+------------+-------------+  +------------+-------------+  +------------+-------------+
             |                             |                             |
    TimeDistributed Conv2D        TimeDistributed Conv2D        TimeDistributed Conv2D
     (32 filters, 3x3)             (32 filters, 3x3)             (32 filters, 3x3)
             |                             |                             |
          MaxPool                       MaxPool                       MaxPool
             |                             |                             |
    TimeDistributed Conv2D        TimeDistributed Conv2D        TimeDistributed Conv2D
     (64 filters, 3x3)             (64 filters, 3x3)             (64 filters, 3x3)
             |                             |                             |
      GlobalAvgPool                 GlobalAvgPool                 GlobalAvgPool
             +-----------------------------+-----------------------------+
                                           |
                                Concatenate Feature Tensors
                                           |
                                LSTM Layer (128 units)
                                           |
                                Dense Layer (1,800 units)
                                           |
                                Reshape Layer (72, 5, 5)
                                           |
                       72-Hour PM2.5 Forecast Tensor for Delhi 5x5 Grid
```

### Data Specifications:
| Data Branch | Grid Dimensions | Features Included | Physical Role |
| :--- | :--- | :--- | :--- |
| **VIIRS Satellite** | 24h x 18 x 22 x 2 | `fire_count`, `total_FRP` (Fire Radiative Power) | Quantifies upstream stubble burning emissions in Punjab and Haryana. |
| **ERA5 Meteorology** | 24h x 19 x 23 x 7 | `u10_ms`, `v10_ms`, `boundary_layer_height_m`, `temperature_C`, `dewpoint_C`, `surface_pressure_hPa`, `precipitation_mm` | Models atmospheric boundary layer collapse, temperature inversions, and transport wind vectors. |
| **CPCB Pollution** | 24h x 5 x 5 x 6 | `PM2.5`, `PM10`, `NO2`, `NOx`, `CO`, `Ozone` | Captures ground-level urban pollution concentrations and baseline air chemistry. |

### Artifact Files:
| File Name | Shape / Type | Description |
| :--- | :--- | :--- |
| `predictions.npy` | `(15, 72, 5, 5)` | CNN-LSTM test run predictions (PM2.5 in ug/m3) |
| `target.npy` | `(15, 72, 5, 5)` | Ground-truth observations from CAAQMS stations |
| `mean_predictions.npy` | `(15, 72, 5, 5)` | Multi-model ensemble mean baseline |
| `persistence_predictions.npy` | `(15, 72, 5, 5)` | Persistence benchmark (PM2.5(t+h) = PM2.5(t)) |
| `scalers.pkl` | Pickle dictionary | StandardScalers for VIIRS, ERA5, CPCB, and Target features |
| `model.keras` | Keras HDF5 (5.58 MB) | Trained deep neural network weights |

---

## REST API Documentation

| HTTP Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the main Stubble-Alert web application |
| `GET` | `/api/forecast/<sample_id>?zone=<name>` | Returns the 72-hour forecast tensor, belief factors, and selected zone metrics |
| `POST` | `/api/chat` | AI stubble query processor (body: `query`, `sample_id`, `selected_zone`) |
| `GET` | `/api/delhi-stations?sample_id=<id>` | Returns all 37 Delhi CAAQMS stations with coordinates and telemetry |
| `GET` | `/api/trust-stats` | Model benchmark metrics (Pearson r, R-squared, MAE, comparisons) |
| `GET` | `/api/health` | Service status, sample count, and active configuration |

---

## Repository Structure

```
stubble-alert/
|-- Stubble-Alert.apk              # Signed standalone Android APK (release)
|-- app-release.apk                # Backup signed release APK
|-- app.py                         # Flask backend & REST API server
|-- index.html                     # Responsive web application frontend
|-- requirements.txt               # Python package dependencies
|-- README.md                      # Project documentation
|-- LICENSE                        # MIT License
|-- .gitignore                     # Git ignore rules (virtualenvs, cache, PPT, notebooks)
|-- predictions.npy                # Model forecast tensor
|-- target.npy                     # Ground truth observations
|-- mean_predictions.npy           # Ensemble baseline tensor
|-- persistence_predictions.npy    # Persistence baseline tensor
|-- scalers.pkl                    # Feature normalization scalers
|-- model.keras                    # Trained CNN-LSTM model file
|-- static/
|   `-- delhi_banner.jpg           # Atmospheric skyline banner image
|-- docs/
|   `-- screenshots/               # Application UI captures
|-- android-build/                 # Native Android SDK build directory
|   |-- AndroidManifest.xml        # Android application manifest
|   |-- build_apk.ps1              # Automated PowerShell build script
|   |-- debug.keystore             # Cryptographic debug keystore
|   |-- res/                       # App icons, layouts, colors, and themes
|   |-- src/                       # Native Java source (MainActivity.java)
|   `-- assets/                    # Offline bundled assets (Chart.js, HTML, JS tensor)
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
