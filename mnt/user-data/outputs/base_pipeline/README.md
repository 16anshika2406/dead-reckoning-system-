# Dead-Reckoning Rebuild — Progress So Far

**👉 First time here? Read `SETUP_GUIDE.md`. New to data collection?
Read `DATA_COLLECTION_GUIDE.md` before you go record anything.**

## Step 1: Base Pipeline (`dr_pipeline.py`)
Fixed the critical bug from the old codebase: heading was hardcoded to 0°,
so the trajectory could only ever be a straight line. Now every sample
uses the phone's live orientation to track real heading changes.
- Proof: `trajectory_comparison.png`

## Step 2: Calibration Engine (`calibration.py`, `dr_pipeline_v2.py`)
Removed the assumption that the phone is mounted exactly aligned with the
vehicle. Estimates the phone's actual mounting orientation from its own
sensor data. Detected a real ~34° mounting offset in our test recording.
- Proof: `v1_vs_v2_comparison.png`

## Step 3: GNSS + INS Fusion (`gnss_fusion.py`, `gnss_simulator.py`, `demo_fusion.py`)
A lightweight per-axis Kalman filter that blends DR position/heading with
GNSS fixes when available, and coasts on pure DR during outages — no
hard mode-switch branch, the math naturally degrades gracefully.

**Honest note on how this was validated:** we don't have a real paired
IMU+GNSS recording with a real tunnel yet, so this was proven against a
fully synthetic scenario with independent ground truth: a 90-second
straight→turn→straight path, with realistic injected accelerometer +
gyro bias (so the "raw DR" actually drifts, the way a real phone would),
a noisy simulated 1Hz GNSS stream, and an 8-second dropout during the
turn. Results (`fusion_demo.png`):
- Pure DR-only: drifted to **98m** error by the end of the 90s trip
- Fused (GNSS+INS): stayed around **10-15m** throughout, including
  through the outage, and re-converged within 5s of GNSS returning

This proves the fusion **algorithm** is correct. It does NOT yet prove
real-world performance — that requires real data (see below).

## 🎯 Immediate next action: go collect real data
Follow `DATA_COLLECTION_GUIDE.md` — install Sensor Logger, record a real
trip including a real tunnel/underpass/parking garage, export it, and run
`merge_sensorlogger_export.py` to turn it into the format everything here
already expects. Once you have that, `gnss_fusion.py` runs on it exactly
as-is (feed it the real merged `_gnss.csv` instead of the simulator's
output) — no code changes needed to switch from synthetic to real data.

## Known limitations (being upfront, not hiding these)
- Steps 1-2 only validated on one real 15-second, low-speed clip.
- Step 3 only validated synthetically so far.
- Small t=0 spike from an uninitialized first quaternion sample (cosmetic).
- Calibration's forward/backward sign resolution is a heuristic — GNSS
  fusion (once real) gives us a way to cross-check and fully resolve this.

## What's built so far vs. what's next
| Step | What | Status |
|---|---|---|
| 1 | Base pipeline (fixes heading/turning bug) | ✅ Done |
| 2 | Calibration/alignment engine | ✅ Done |
| 3 | GNSS + INS fusion | ✅ Done (synthetic validation; needs real data next) |
| 4 | Seamless GNSS-outage handler | 🟡 Mostly covered by Step 3's mode-blend — needs real-outage tuning |
| 5 | Map-matching upgrade | ✅ Done (HMM-based, validated on real OSM data) |
| 6 | Mobile app shell | Not started |
| 7 | ML speed/vibration model | Not started (needs real data first) |
