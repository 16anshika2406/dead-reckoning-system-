"""
gnss_simulator.py
===================

We don't have a real recording with paired GNSS + a real tunnel/outage yet
(that's what DATA_COLLECTION_GUIDE.md is for). To build and PROVE the
fusion logic right now, we simulate a realistic GNSS stream from a known
ground-truth trajectory:

  1. Take a ground-truth (x, y) path (for now: the DR output from
     dr_pipeline_v2.py, treated as "truth" for this synthetic exercise --
     clearly a placeholder for a real GPS-surveyed ground truth later).
  2. Convert to fake lat/lon around a real-world origin.
  3. Add realistic GNSS position noise (a few meters, matches typical
     smartphone horizontal accuracy).
  4. Downsample to ~1 Hz (typical consumer GNSS fix rate).
  5. Cut a chunk out entirely -- this is the simulated "tunnel" GNSS outage.

BE HONEST WHEN PRESENTING THIS: this is a synthetic proof that the fusion
MATH works, not a real-world validation. That still requires a real
recording through a real tunnel/parking garage (see DATA_COLLECTION_GUIDE.md).
"""

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6371000.0


def xy_to_latlon(x, y, origin_lat_deg, origin_lon_deg):
    """Small-area flat-earth approximation -- fine for a few hundred meters."""
    lat = origin_lat_deg + np.degrees(y / EARTH_RADIUS_M)
    lon = origin_lon_deg + np.degrees(
        x / (EARTH_RADIUS_M * np.cos(np.radians(origin_lat_deg)))
    )
    return lat, lon


def simulate_gnss(dr_result, origin_lat=28.7041, origin_lon=77.1025,
                   fix_rate_hz=1.0, position_noise_std_m=3.0,
                   outage_start_s=None, outage_duration_s=5.0,
                   random_seed=42):
    """
    dr_result : DataFrame with 'time', 'world_x', 'world_y' columns
                (output of dr_pipeline_v2.run_pipeline_v2)
    outage_start_s : when the simulated GNSS blackout starts (seconds into
                      the trip). Defaults to the midpoint of the recording.

    Returns a DataFrame: time_seconds, latitude, longitude, speed, bearing,
    horizontal_accuracy -- same shape as a real Location.csv-derived GNSS log,
    PLUS an 'is_outage' column marking the simulated blackout (for plotting
    only -- the fusion code must NOT be given this column, it has to detect
    the outage itself from fix availability/accuracy).
    """
    rng = np.random.default_rng(random_seed)

    t_full = dr_result["time"].to_numpy()
    x_full = dr_result["world_x"].to_numpy()
    y_full = dr_result["world_y"].to_numpy()

    if outage_start_s is None:
        outage_start_s = t_full[len(t_full) // 2]

    # sample at the GNSS fix rate
    t_gnss = np.arange(t_full[0], t_full[-1], 1.0 / fix_rate_hz)
    x_gnss = np.interp(t_gnss, t_full, x_full)
    y_gnss = np.interp(t_gnss, t_full, y_full)

    # add realistic position noise
    x_noisy = x_gnss + rng.normal(0, position_noise_std_m, size=len(t_gnss))
    y_noisy = y_gnss + rng.normal(0, position_noise_std_m, size=len(t_gnss))

    lat, lon = xy_to_latlon(x_noisy, y_noisy, origin_lat, origin_lon)

    # crude speed/bearing from consecutive noisy fixes
    dx = np.diff(x_noisy, prepend=x_noisy[0])
    dy = np.diff(y_noisy, prepend=y_noisy[0])
    dt = np.diff(t_gnss, prepend=t_gnss[0] - 1.0 / fix_rate_hz)
    dt[dt <= 0] = 1.0 / fix_rate_hz
    speed = np.sqrt(dx**2 + dy**2) / dt
    bearing = (np.degrees(np.arctan2(dx, dy))) % 360

    horizontal_accuracy = np.full(len(t_gnss), position_noise_std_m)

    is_outage = (t_gnss >= outage_start_s) & (t_gnss < outage_start_s + outage_duration_s)

    gnss = pd.DataFrame({
        "time_seconds": t_gnss,
        "latitude": lat,
        "longitude": lon,
        "speed": speed,
        "bearing": bearing,
        "horizontal_accuracy": horizontal_accuracy,
        "is_outage": is_outage,
    })

    # DROP the fixes during the simulated outage entirely (that's what a
    # real GNSS blackout looks like -- no fixes, not just bad ones)
    gnss_with_outage = gnss[~gnss["is_outage"]].reset_index(drop=True)

    print(f"[gnss_sim] generated {len(gnss)} fixes at {fix_rate_hz} Hz, "
          f"{position_noise_std_m}m noise std")
    print(f"[gnss_sim] simulated outage: {outage_start_s:.1f}s to "
          f"{outage_start_s + outage_duration_s:.1f}s "
          f"({gnss['is_outage'].sum()} fixes dropped)")

    return gnss_with_outage, gnss  # (fixes with outage removed, full reference incl. outage flag)
