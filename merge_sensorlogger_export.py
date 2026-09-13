"""
merge_sensorlogger_export.py
==============================

Converts a raw, unzipped "Sensor Logger" app export folder into:
  1. <name>_imu.csv  -- exactly the format dr_pipeline.py / dr_pipeline_v2.py
     already expect: time_seconds, acc_x/y/z, gyro_x/y/z, quat_w/x/y/z
  2. <name>_gnss.csv -- time-aligned GPS log for the fusion stage:
     time_seconds, latitude, longitude, speed, bearing, horizontal_accuracy

USAGE
-----
    python3 merge_sensorlogger_export.py /path/to/unzipped_folder my_trip_01

NOTES ON SENSOR LOGGER'S FILE FORMAT (from the app's own documentation:
https://github.com/tszheichoi/awesome-sensor-logger/blob/main/UNITS.md)
--------------------------------------------------------------------------
- Every file has `time` (nanoseconds since epoch) and `seconds_elapsed`
  (seconds since the recording started) -- we use seconds_elapsed.
- Android: `Accelerometer.csv` excludes gravity; `TotalAcceleration.csv`
  (if present) includes it. We NEED gravity included, because our own
  pipeline does its own gravity removal using the quaternion. So we
  prefer TotalAcceleration.csv when it exists.
- iOS: only `Accelerometer.csv` exists and it EXCLUDES gravity. We
  reconstruct raw (gravity-included) acceleration by adding `Gravity.csv`
  back in, sample-by-sample (nearest-time match).
- `Orientation.csv` has qw, qx, qy, qz already.
- `Location.csv` has latitude, longitude, speed, bearing, horizontalAccuracy.

If your export folder structure differs slightly (app updates do happen),
check the printed column names when this script errors -- it will tell
you what it found vs. what it expected.
"""

import sys
import os
import pandas as pd
import numpy as np


def _load(folder, filename):
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df


def _nearest_merge(base, other, other_cols, tolerance_s=0.05):
    """Merge `other`'s columns onto `base`'s timestamps using nearest-time
    matching (within tolerance_s). Both must have 'seconds_elapsed'."""
    base_sorted = base.sort_values("seconds_elapsed").reset_index(drop=True)
    other_sorted = other.sort_values("seconds_elapsed").reset_index(drop=True)

    merged = pd.merge_asof(
        base_sorted, other_sorted[["seconds_elapsed"] + other_cols],
        on="seconds_elapsed", direction="nearest",
        tolerance=tolerance_s,
    )
    return merged


def build_imu_csv(folder):
    accel_total = _load(folder, "TotalAcceleration.csv")
    accel_plain = _load(folder, "Accelerometer.csv")
    gravity = _load(folder, "Gravity.csv")
    gyro = _load(folder, "Gyroscope.csv")
    orientation = _load(folder, "Orientation.csv")

    if gyro is None or orientation is None:
        raise FileNotFoundError(
            "Missing Gyroscope.csv or Orientation.csv in the export folder. "
            "Check that these sensors were enabled while recording."
        )

    # --- figure out the raw (gravity-included) acceleration source ---
    if accel_total is not None:
        print("[merge] using TotalAcceleration.csv (includes gravity)")
        acc = accel_total.rename(columns={"x": "acc_x", "y": "acc_y", "z": "acc_z"})
        acc = acc[["seconds_elapsed", "acc_x", "acc_y", "acc_z"]]
    elif accel_plain is not None and gravity is not None:
        print("[merge] reconstructing raw acceleration from "
              "Accelerometer.csv + Gravity.csv (iOS-style export)")
        acc_only = accel_plain.rename(
            columns={"x": "a_x", "y": "a_y", "z": "a_z"}
        )[["seconds_elapsed", "a_x", "a_y", "a_z"]]
        grav = gravity.rename(
            columns={"x": "g_x", "y": "g_y", "z": "g_z"}
        )[["seconds_elapsed", "g_x", "g_y", "g_z"]]
        merged = _nearest_merge(acc_only, grav, ["g_x", "g_y", "g_z"])
        merged["acc_x"] = merged["a_x"] + merged["g_x"]
        merged["acc_y"] = merged["a_y"] + merged["g_y"]
        merged["acc_z"] = merged["a_z"] + merged["g_z"]
        acc = merged[["seconds_elapsed", "acc_x", "acc_y", "acc_z"]]
    elif accel_plain is not None:
        print("[merge] WARNING: only Accelerometer.csv found (no Gravity.csv). "
              "This likely EXCLUDES gravity, which will break gravity removal "
              "downstream. Re-record with 'Gravity' or 'TotalAcceleration' "
              "enabled if results look wrong.")
        acc = accel_plain.rename(
            columns={"x": "acc_x", "y": "acc_y", "z": "acc_z"}
        )[["seconds_elapsed", "acc_x", "acc_y", "acc_z"]]
    else:
        raise FileNotFoundError(
            "No acceleration file found (need TotalAcceleration.csv, or "
            "Accelerometer.csv + Gravity.csv, or at minimum Accelerometer.csv)."
        )

    gyro = gyro.rename(columns={"x": "gyro_x", "y": "gyro_y", "z": "gyro_z"})
    gyro = gyro[["seconds_elapsed", "gyro_x", "gyro_y", "gyro_z"]]

    orient_cols = ["qw", "qx", "qy", "qz"]
    missing = [c for c in orient_cols if c not in orientation.columns]
    if missing:
        raise ValueError(f"Orientation.csv is missing columns: {missing}. "
                          f"Found columns: {list(orientation.columns)}")

    orientation = orientation.rename(
        columns={"qw": "quat_w", "qx": "quat_x", "qy": "quat_y", "qz": "quat_z"}
    )[["seconds_elapsed", "quat_w", "quat_x", "quat_y", "quat_z"]]

    # merge everything onto the accelerometer's timestamp grid
    result = _nearest_merge(acc, gyro,
                             ["gyro_x", "gyro_y", "gyro_z"])
    result = _nearest_merge(result, orientation,
                             ["quat_w", "quat_x", "quat_y", "quat_z"])

    result = result.dropna().reset_index(drop=True)
    result = result.rename(columns={"seconds_elapsed": "time_seconds"})

    print(f"[merge] IMU: {len(result)} synchronized samples, "
          f"~{1/np.mean(np.diff(result['time_seconds'])):.1f} Hz")

    return result[["time_seconds", "acc_x", "acc_y", "acc_z",
                    "gyro_x", "gyro_y", "gyro_z",
                    "quat_w", "quat_x", "quat_y", "quat_z"]]


def build_gnss_csv(folder):
    location = _load(folder, "Location.csv")
    if location is None:
        print("[merge] WARNING: no Location.csv found -- no GNSS log will "
              "be produced. GNSS fusion (Step 3) needs this.")
        return None

    keep = ["seconds_elapsed", "latitude", "longitude", "speed", "bearing",
            "horizontalAccuracy"]
    missing = [c for c in keep if c not in location.columns]
    if missing:
        raise ValueError(f"Location.csv is missing columns: {missing}. "
                          f"Found columns: {list(location.columns)}")

    gnss = location[keep].rename(columns={
        "seconds_elapsed": "time_seconds",
        "horizontalAccuracy": "horizontal_accuracy",
    })
    gnss = gnss.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    print(f"[merge] GNSS: {len(gnss)} fixes, "
          f"~{1/np.mean(np.diff(gnss['time_seconds'])):.2f} Hz average")

    return gnss


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 merge_sensorlogger_export.py <unzipped_folder> <output_name>")
        sys.exit(1)

    folder = sys.argv[1]
    out_name = sys.argv[2]

    print(f"[merge] reading from: {folder}")
    print(f"[merge] files found: {os.listdir(folder)}")

    imu_df = build_imu_csv(folder)
    imu_path = f"{out_name}_imu.csv"
    imu_df.to_csv(imu_path, index=False)
    print(f"[merge] saved {imu_path}")

    gnss_df = build_gnss_csv(folder)
    if gnss_df is not None:
        gnss_path = f"{out_name}_gnss.csv"
        gnss_df.to_csv(gnss_path, index=False)
        print(f"[merge] saved {gnss_path}")

    print("\nDone. You can now run:")
    print(f"    python3 dr_pipeline_v2.py {imu_path}")
