# 📱 DATA COLLECTION GUIDE — Starting From Zero

You don't need any special hardware. Your phone already has everything:
accelerometer, gyroscope, and GPS. You just need the right app and a
recording protocol that gives every downstream stage (calibration, ZUPT,
fusion, map-matching) what it needs.

---

## 1. The app: **Sensor Logger** (free)

- **Android:** Google Play → search "Sensor Logger" (publisher: Kelvin Choi / tszheichoi)
- **iPhone:** App Store → search "Sensor Logger" (same publisher)
- Official page: https://www.tszheichoi.com/sensorlogger

This is very likely the exact app your team used before — your old CSV's
columns (`time_seconds`, `acc_x/y/z`, `gyro_x/y/z`, `quat_w/x/y/z`) match
this app's export format almost exactly. It's free, well-documented, and
logs everything we need **simultaneously and time-synced**: accelerometer,
gyroscope, orientation (quaternion), and GPS.

## 2. What to enable before recording

Open the app → tap the sensor list → make sure these are ON:
- ✅ **Accelerometer** (Android: this one excludes gravity — that's fine, see step 4)
- ✅ **TotalAcceleration** (Android only — if available, turn this on too; it includes gravity, which is what we actually want to use)
- ✅ **Gravity** (turn this on regardless — iOS needs it, and it's a useful cross-check on Android)
- ✅ **Gyroscope**
- ✅ **Orientation** (this gives you the quaternion)
- ✅ **Location** (this is your GPS/GNSS log)

Set the motion sensor sampling rate to **~50 Hz** if there's an option
(Settings → Sensor Configuration). GPS stays at its default rate (usually
~1 Hz) — that's normal and expected; our fusion code will handle the
different rates.

## 3. Mounting the phone

**It genuinely does not matter where you mount it** — dashboard, cupholder,
stuck to the window, flat on the seat — because Step 2's calibration
engine figures out the mounting angle automatically. The only two rules:

1. **Keep it rigidly fixed for the whole recording.** Don't pick it up,
   rotate it, or hold it in your hand mid-recording — the calibration
   assumes one fixed mounting for the whole session.
2. Screen doesn't need to face any particular way.

## 4. The recording protocol (follow this every time)

This exact sequence is what makes every downstream module work:

| Phase | Duration | What to do | Why |
|---|---|---|---|
| 1. Stationary | 5–10 sec | Sit still in the parked car before moving | Calibration engine needs this to find "down" and to zero the gyro bias |
| 2. Straight pull-away | 10–15 sec | Drive straight, ideally from a stop | Calibration engine needs a genuine "speeding up" event to find "forward" and resolve its direction |
| 3. Normal driving | As long as possible (aim for 10+ min) | Mix of turns, straight roads, stop-and-go | This is your real validation data — the more, the better |
| 4. **GNSS-denied stretch** | At least once | Drive through a **tunnel, underpass, multi-level parking garage, or dense urban canyon** | **This is the actual scenario the whole PS is about.** Without this, you cannot demonstrate the core ask to judges |
| 5. Stationary | 5–10 sec | Park and sit still before stopping the recording | Gives a clean end-of-trip reference |

**Do this at least 2–3 times** on different routes if you can — one good
recording is a demo, three is a dataset.

## 5. Exporting

In the app: Recording → tap the recording → **Export → Zipped CSV**.
Unzip it — you'll get a folder with files like:
```
Accelerometer.csv       (or TotalAcceleration.csv on Android)
Gravity.csv
Gyroscope.csv
Orientation.csv
Location.csv
Metadata.csv
```

## 6. Turning that into what our pipeline needs

Don't manually merge these — use `merge_sensorlogger_export.py` (next to
this guide). It automatically:
- Picks the right acceleration source (TotalAcceleration if present,
  otherwise reconstructs it from Accelerometer + Gravity)
- Time-aligns accelerometer, gyroscope, and orientation into one CSV in
  **exactly the format `dr_pipeline.py` / `dr_pipeline_v2.py` already expect**
- Produces a separate, time-aligned GNSS log for the fusion stage

```bash
python3 merge_sensorlogger_export.py /path/to/unzipped_folder my_trip_01
```

This creates:
- `my_trip_01_imu.csv` → feed straight into `dr_pipeline_v2.py`
- `my_trip_01_gnss.csv` → will be used by the GNSS fusion stage (Step 3)

---

## Quick checklist before you drive off

- [ ] Accelerometer/TotalAcceleration, Gravity, Gyroscope, Orientation, Location all enabled
- [ ] Phone mounted somewhere fixed, not in your hand
- [ ] You'll sit still 5-10 sec before moving
- [ ] You'll pull away in a straight line at the start
- [ ] Your route includes at least one tunnel/underpass/parking garage/urban canyon
- [ ] You'll sit still 5-10 sec before stopping the recording
