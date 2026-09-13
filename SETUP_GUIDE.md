# 🛠️ SETUP GUIDE — Dead Reckoning Rebuild

This covers everything built so far (Step 1: base pipeline, Step 2:
calibration engine) and how to run it from a completely fresh machine.
Every teammate should be able to follow this top to bottom with zero
prior context.

---

## 1. One-time machine setup

You need **Python 3.9+**. Check what you have:

```bash
python3 --version
```

### Windows
1. Install Python from https://python.org/downloads (tick "Add to PATH" during install).
2. Open **Command Prompt** or **PowerShell**.

### Mac / Linux
Python 3 is usually pre-installed. If not: `brew install python3` (Mac) or
`sudo apt install python3 python3-pip` (Linux/Ubuntu).

---

## 2. Get the project files

Put all these files in **one folder** (e.g. `dead-reckoning-rebuild/`):

```
dead-reckoning-rebuild/
├── imu_data_1788285144311.csv     ← the real test recording
├── dr_pipeline.py                  ← Step 1: base pipeline
├── calibration.py                  ← Step 2: calibration engine
├── dr_pipeline_v2.py                ← Step 1 + 2 combined
├── make_plots.py                   ← generates v1 proof plots
├── compare_v1_v2.py                 ← generates v1-vs-v2 comparison plot
├── requirements.txt
└── README.md
```

(If you're on the team's shared drive/GitHub, just `git clone` or download
the folder — same idea.)

---

## 3. Create an isolated environment (do this once per machine)

This keeps this project's packages separate from anything else on your
computer — avoids version conflicts, and means everyone on the team runs
the exact same setup.

```bash
cd dead-reckoning-rebuild

# create the environment (creates a "venv" folder)
python3 -m venv venv

# activate it — you'll need to do this every time you open a new terminal
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows (Command Prompt)
venv\Scripts\Activate.ps1       # Windows (PowerShell)
```

You'll know it worked because your terminal prompt now shows `(venv)` at
the start of the line.

---

## 4. Install dependencies

With the venv activated:

```bash
pip install -r requirements.txt
```

This installs `numpy`, `pandas`, `scipy`, `matplotlib` — everything the
pipeline needs. Takes under a minute.

---

## 5. Run Step 1: the base pipeline

```bash
python3 dr_pipeline.py imu_data_1788285144311.csv
```

**What you should see:**
```
[load] 754 samples, ~49.9 Hz, duration 15.08s
[load] gyro bias (rad/s): [...]
[orientation] heading range: -20.1 deg to 23.6 deg (span 43.8 deg)
[zupt] stationary samples: 146 / 754 (19.4%)

Saved: base_pipeline_output.csv
Final position (fixed): x=... m, y=... m
Final position (buggy): x=... m, y=0.000 m   <- proves the old bug
```

Then generate the proof plots:
```bash
python3 make_plots.py
```
This creates 3 PNGs: `trajectory_comparison.png`, `heading_over_time.png`,
`zupt_detection.png`. Open them with any image viewer.

---

## 6. Run Step 2: the calibrated pipeline

```bash
python3 dr_pipeline_v2.py imu_data_1788285144311.csv
```

**What you should see:** a calibration report (detected mounting offset in
degrees), followed by the same kind of trajectory output as Step 1, but
now correctly accounting for however the phone happens to be mounted.

Then compare it side-by-side against Step 1:
```bash
python3 compare_v1_v2.py
```
This creates `v1_vs_v2_comparison.png`.

---

## 7. Running on YOUR OWN new recording

Whenever someone on the team records a fresh IMU log, as long as the CSV
has these exact column headers, it'll just work with the same commands:

```
time_seconds, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, quat_w, quat_x, quat_y, quat_z
```

```bash
python3 dr_pipeline_v2.py your_new_recording.csv
```

**Important for calibration to work:** the recording needs (a) a few
seconds where the phone/vehicle is completely still at the start, and
(b) some driving in a reasonably straight line at some point (ideally
including pulling away from a stop) — that's what lets the calibration
engine figure out "down" and "forward."

---

## 8. Common issues

| Problem | Fix |
|---|---|
| `python3: command not found` | Use `python` instead of `python3` (common on Windows) |
| `pip: command not found` | Use `python3 -m pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'scipy'` | You forgot to activate the venv, or forgot step 4 |
| Calibration raises `"Could not find a stationary window"` | Your recording doesn't have a still period at the start — re-record with 2-3 seconds of no movement first |
| Calibration raises `"Not enough low-turning-rate moving samples"` | Recording is too short or too twisty — needs some straight-line driving |

---

## 9. Recommended team workflow from here

- **One person owns running the pipeline** on any new recording and
  committing the output — avoids "works on my machine" confusion.
- Put the project in a shared **git repo** (GitHub/GitLab) as soon as
  possible instead of passing zip files around — much easier to track
  who changed what as we add Steps 3-7.
- Every new stage should get its own file (`dr_pipeline_v3.py`, etc.) the
  way we've done it so far, so earlier stages stay runnable and it's easy
  to show judges the progression.

---

## What's built so far vs. what's next

| Step | What | Status |
|---|---|---|
| 1 | Base pipeline (fixes heading/turning bug) | ✅ Done |
| 2 | Calibration/alignment engine | ✅ Done |
| 3 | GNSS + INS fusion | ⏭️ Next |
| 4 | Seamless GNSS-outage handler | Not started |
| 5 | Map-matching upgrade | Not started |
| 6 | Mobile app shell | Not started |
| 7 | ML speed/vibration model | Not started (needs more real data first) |

---

## 10. Getting and using REAL data (Step 3 onward)

Read `DATA_COLLECTION_GUIDE.md` in full before recording anything — it
covers the app to use, what sensors to enable, and the exact recording
protocol (stationary window, straight pull-away, a real tunnel/parking
garage, etc.).

Once you've recorded and exported a trip from the Sensor Logger app:

```bash
# unzip the app's export first, then:
python3 merge_sensorlogger_export.py /path/to/unzipped_folder trip01
```

This produces `trip01_imu.csv` and `trip01_gnss.csv` in exactly the
format the rest of the pipeline expects. Run the full pipeline on it:

```bash
python3 dr_pipeline_v2.py trip01_imu.csv
```

## 11. Running the GNSS fusion demo (Step 3)

Right now this runs on a synthetic scenario (see README.md for why):

```bash
python3 demo_fusion.py
```

Generates `fusion_demo.png` and prints drift numbers comparing fused vs.
pure-DR performance. Once you have a real `trip01_gnss.csv` from a real
recording with a real GNSS outage, swap the synthetic simulator call in
`demo_fusion.py` for your real merged files — `gnss_fusion.py`'s `fuse()`
function itself needs no changes, it just takes IMU arrays + a GNSS
DataFrame in the same shape either way.
