The problem we're solving

Delhi NCR has some of the worst air pollution in the world, especially every winter. Right now, when people check "AQI forecasts," they're relying on tools that have a fundamental flaw: they treat weather and pollution as two separate problems, when in reality they're locked in a feedback loop.

Here's the physical phenomenon our project is built around:

In winter, cold nights create a temperature inversion — a layer of warm air sits above a layer of cold air near the ground, acting like a lid. Pollution gets trapped underneath instead of dispersing upward.
Thick pollution (PM2.5) then blocks sunlight, which cools the ground further, which strengthens the inversion, which traps even more pollution. It's a vicious cycle.
On top of this, every October–November, farmers in Punjab/Haryana burn crop stubble, and smoke plumes drift into Delhi depending on wind direction.

Standard forecasting tools miss this coupling, and even when they use real physics, they're too coarse (~40km grid) to say anything useful about a specific neighborhood in Delhi.

Why existing solutions fall short (our "Problem" slide)

we identified four real gaps:

Decoupled modeling — weather and pollution modeled independently, so inversion trapping isn't captured.
Coarse resolution — CAMS (a real, free, physics-based European forecasting service) only gives ~40km grid cells. That's the size of half of Delhi in one pixel — useless for telling Anand Vihar from Dwarka.
Heavy alternatives exist (WRF-Chem — a fully coupled physics+chemistry model) but need supercomputers, hours per run, weeks of setup. Not something a small team or a city agency can just spin up.
Limited actionability — even where forecasts exist, they don't tell we why a spike is happening (is it stubble smoke? is it an inversion trapping local traffic pollution?) or how confident the forecast is.



our core idea — the "aha" of the whole project

Instead of picking one extreme (cheap-but-coarse physics, or heavy-but-slow full simulation), we're doing a hybrid:

Real physics forecast (CAMS, free, ~40km, already accounts for large-scale
transport, chemistry, and weather trends)
        +
Local ground-truth observations (CPCB station readings — actual measured
AQI at specific Delhi locations)
        +
Local weather/fire signals (temperature, wind, boundary layer height, 
farm-fire hotspots)
        ↓
A machine learning model that LEARNS how to correct CAMS's coarse
40km estimate down to what a specific station actually experiences
        ↓
A station-level 72-hour forecast, with an honest uncertainty range,
that's actually useful to a person deciding whether to send their kid
outside tomorrow.

Think of it like this: CAMS is a weather forecaster who's excellent at big-picture physics but has never been to Delhi. our ML model is a local guide who's watched years of "CAMS says X, but this particular station in Anand Vihar actually reads Y because of local traffic/inversion/wind patterns" and learns to correct for that gap.

Why this beats the alternatives (our "Why We Stand Out" slide)
vs. CAMS alone: we get local accuracy CAMS can't give.
vs. a pure ML model with no physics: pure ML tends to fall apart on rare, extreme events (like a sudden severe smog episode) because it hasn't seen enough examples. Anchoring to CAMS's physics gives we a sane baseline even in unusual conditions.
vs. WRF-Chem: we get 90% of the benefit (coupled, locally accurate forecasting) without needing an HPC cluster — this runs on a normal cloud VM or laptop.
And: our architecture is designed so that later, if we ever do get access to WRF-Chem-grade compute, we just swap out the CAMS data source — the ML correction layer, database, and dashboard don't need to change. That's the "Phase 6" WRF-Chem upgrade path in our roadmap.
How it actually works, step by step (our Technical Approach slide)
Ingest four data streams on a schedule (every 6–12 hours): CAMS forecast, CPCB station observations, meteorology (Open-Meteo), and fire hotspots (NASA FIRMS, for stubble burning).
Feature engineering: align the coarse CAMS grid to each specific Delhi monitoring station's coordinates; build "lag features" (what was PM2.5 6/12/24 hours ago — pollution has momentum); compute inversion/ventilation proxies from temperature and wind.
ML correction engine: an XGBoost (and optionally LSTM) model, trained offline on historical data, learns the systematic bias between what CAMS predicted and what the station actually measured, and corrects live forecasts accordingly. It also outputs a 10–90% confidence band using quantile regression, so a forecast isn't just one number — it's a range, and that range should widen the further out we forecast (72 hours out is less certain than 6 hours out).
Post-processing: convert corrected pollutant concentrations into the actual Indian AQI number (there's an official CPCB formula for this), compute the inversion-strength and ventilation-coefficient gauges, and trace fire-plume trajectories using wind direction.
Serve: a backend (FastAPI) exposes this as an API; a dashboard (React + Leaflet map + charts) displays it live — the 72h forecast, the map, the gauges, and a "here's what CAMS said vs. what we corrected it to" comparison, so users/judges can literally see the value our ML layer adds.
Verify: we also track how good our forecasts actually were against real CPCB readings afterward (RMSE/MAE), which builds trust and shows scientific rigor.
Why it matters (Impact slide)
Public health: 72-hour lead time lets people, schools, and vulnerable groups (elderly, asthma/cardiac patients) actually plan around bad air days instead of reacting after the fact.
Policy: pinpointing plume sources gives regulators evidence to act on stubble burning specifically, rather than blanket restrictions.
Low cost: runs on ~$50–100/month of cloud infrastructure, not a national supercomputer — realistic for a municipal body or even a hackathon team to actually deploy.
The honest limits (which we should be upfront about, not hide)
It's still fundamentally bounded by CAMS's physics quality — if CAMS gets a big-picture pattern wrong, our local correction can only do so much.
ML correction is weaker on rare/extreme events by nature — we mitigate this by keeping the CAMS physics as a backbone rather than replacing it entirely.
CAMS updates only 1–2 times a day, plus we refresh every 6–12h — not truly real-time down to the minute, but that's fine for a 72h outlook.



Roles

1. CAMS Data

Register on ADS, accept license on cams-global-atmospheric-composition-forecasts.
Write fetch_cams.py: pull live 5-day forecast for the Delhi NCR bounding box (~28.3–28.9°N, 76.8–77.5°E), and the 60–90 day historical archive for the same box.
Output: CSV in the shared schema, one row per grid-cell/timestamp.
Deliverable due: forecast pull working in ~2h, historical pull running in background by ~4h (these downloads can be slow — start the historical pull early and let it run while doing other things).

2. CPCB Data

Register on data.gov.in, get API key.
Write fetch_cpcb.py: live pull (resource ID 3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69, filtered to Delhi) + download 60–90 days of historical hourly data from CPCB's CCR portal for 5–6 chosen stations (Anand Vihar, RK Puram, Punjabi Bagh, ITO, Dwarka — pick ones with clean data).
Output: same shared schema.
This is your training labels — highest priority alongside CAMS historical.

3. Meteorology + Fire Data

Write fetch_met.py (Open-Meteo forecast + historical archive: temp, wind speed/direction, boundary layer height) and fetch_fires.py (NASA FIRMS, MAP_KEY, live + historical hotspots over Punjab/Haryana, filtered to fires upwind of Delhi).
Both are no-key or instant-key, so this person can move fast and should finish first — once done, help whichever of #1/#2 is behind (CAMS/CPCB historical pulls are the likely bottleneck).
Output: same shared schema, merged in later by lat/lon proximity to CPCB stations.

4. ML Lead

Starts immediately on synthetic/sample data matching the schema — don't wait for real data to start writing code.
features.py: align CAMS grid to nearest station, build lag features (t-6/-12/-24h), inversion proxy (temp-based), ventilation proxy (wind × boundary layer height), time-of-day/season features.
train.py (in a Kaggle Notebook): XGBoost per pollutant, with quantile objectives (0.1/0.5/0.9) for confidence bands, chronological train/val split.
Once real data lands (~hour 5–6 from data leads), swap it in, retrain, export model_*.pkl, and produce the RMSE/MAE-per-lead-time numbers for your verification panel.
This is on the critical path — everything downstream depends on this person's output, so give them the least side-work.

5. Backend 

Starts on postprocess/aqi.py (Indian AQI sub-index formula — this needs no data at all, code it from the public spec immediately) and postprocess/metrics.py (inversion strength, ventilation coefficient formulas) in parallel with everyone else.
Once models land: ml/infer.py (load models, run on live fetched data) and the FastAPI app (/forecast/{station}, /stations, /verification).
Owns SQLite schema and wiring all the pieces together — effectively your integration point.

6. Frontend

Starts immediately on the React + Leaflet map + Plotly/ECharts dashboard shell using hardcoded mock JSON matching the eventual API response shape (agree this shape with the Backend Lead in the same kickoff conversation).
Builds: station map, 72h band chart, inversion/ventilation gauges, CAMS-vs-corrected comparison view, alert banner.
Swaps mock data for the real FastAPI endpoint the moment it's live — should be a small change if the response shape was agreed upfront.



