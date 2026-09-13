"""
gnss_fusion.py
================

STEP 3: GNSS + INS Fusion Engine (+ a first cut at Step 4, the seamless
mode switch, since they're naturally the same code).

APPROACH: a lightweight, per-axis scalar Kalman filter (NOT a full
Unscented Kalman Filter yet -- this is a deliberate, honest MVP choice
for a hackathon timeline; a UKF is a natural later upgrade of this same
architecture, not a rewrite).

HOW IT WORKS
-------------
- Every IMU sample: propagate position using the calibrated DR pipeline
  from Step 1+2 (forward velocity -> NHC -> rotate by heading -> integrate),
  exactly like dr_pipeline_v2.py. This is the "predict" step.
- Track a scalar uncertainty (variance) per axis that GROWS over time
  while coasting on DR alone (process noise Q) -- this represents "we
  trust the DR less the longer we go without a GNSS check-in".
- Whenever a GNSS fix arrives: blend it in with a Kalman gain
  K = P / (P + R), where R is the GNSS fix's own reported accuracy.
  Good GNSS (small R) pulls the estimate strongly toward the fix.
  Poor/old DR confidence (large P) also pulls harder toward the fix.
  This IS the seamless part: there's no hard "if GNSS available: use
  GNSS, else: use DR" branch anywhere -- the blend naturally goes to
  100% DR when there are no fixes (outage), and gradually re-trusts
  GNSS as fixes resume.
- Heading is corrected the same way using GNSS-derived bearing, but only
  when GNSS speed is high enough that bearing is actually meaningful.

WHAT MAKES THIS "SEAMLESS" (PS deliverable #5)
-------------------------------------------------
Because correction is a continuous blend (not a mode switch), there's no
visible "teleport" when GNSS returns after a short outage -- the position
snaps back gradually over a few fixes, proportional to how uncertain we'd
become. Longer outages == bigger (but still gradual) correction. We track
a `mode` label per sample purely for the UI/diagnostics, not for the math.
"""

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6371000.0


def latlon_to_xy(lat, lon, origin_lat_deg, origin_lon_deg):
    y = np.radians(lat - origin_lat_deg) * EARTH_RADIUS_M
    x = np.radians(lon - origin_lon_deg) * EARTH_RADIUS_M * np.cos(np.radians(origin_lat_deg))
    return x, y


def fuse(d, vehicle_acc, heading, stationary, gnss_df,
         process_noise_pos=0.5,       # (m/s) how fast position uncertainty grows per second of pure DR
         process_noise_heading_deg=2.0,  # (deg/s) how fast heading uncertainty grows
         min_speed_for_bearing=1.0,   # m/s, below this GNSS bearing is unreliable, skip heading correction
         max_velocity=15.0):

    t = d["t"]
    n = len(t)

    origin_lat = gnss_df["latitude"].iloc[0]
    origin_lon = gnss_df["longitude"].iloc[0]
    gnss_x, gnss_y = latlon_to_xy(gnss_df["latitude"].to_numpy(),
                                   gnss_df["longitude"].to_numpy(),
                                   origin_lat, origin_lon)
    gnss_t = gnss_df["time_seconds"].to_numpy()
    gnss_r = gnss_df["horizontal_accuracy"].to_numpy() ** 2  # measurement variance
    gnss_speed = gnss_df["speed"].to_numpy()
    gnss_bearing_rad = np.radians(gnss_df["bearing"].to_numpy())

    fix_idx = 0
    n_fixes = len(gnss_t)

    forward_v = np.zeros(n)
    right_v = np.zeros(n)

    fused_x = np.zeros(n)
    fused_y = np.zeros(n)

    # IMPORTANT: fused_heading is propagated using the INCREMENT
    # (delta) of the raw orientation source, added onto the previous
    # CORRECTED heading -- not reset to the raw absolute value every
    # step. That's what makes a GNSS heading correction actually
    # persist forward in time instead of evaporating on the very next
    # sample (a real bug caught during testing -- see gnss_fusion.py
    # module docstring / commit notes).
    fused_heading = np.zeros(n)
    fused_heading[0] = heading[0]

    dr_only_x = np.zeros(n)   # pure DR, no correction -- for comparison plots
    dr_only_y = np.zeros(n)

    P_x = np.zeros(n)
    P_y = np.zeros(n)
    P_heading = np.zeros(n)

    mode = np.array(["gnss_aided"] * n, dtype=object)
    time_since_fix = np.zeros(n)

    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if dt <= 0 or dt > 0.2:
            dt = 1.0 / d["fs"]

        # ---- PREDICT: same DR propagation as dr_pipeline_v2 ----
        forward_v[i] = forward_v[i - 1] + vehicle_acc[i, 0] * dt
        right_v[i] = 0.0  # NHC

        if stationary[i]:
            forward_v[i] = 0.0
        forward_v[i] = np.clip(forward_v[i], -max_velocity, max_velocity)

        # propagate heading by the INCREMENT of the raw source, on top
        # of the previous CORRECTED heading (see note above)
        heading_delta = np.arctan2(
            np.sin(heading[i] - heading[i - 1]), np.cos(heading[i] - heading[i - 1])
        )
        fused_heading[i] = fused_heading[i - 1] + heading_delta

        theta = fused_heading[i]
        world_vx = forward_v[i] * np.cos(theta) - right_v[i] * np.sin(theta)
        world_vy = forward_v[i] * np.sin(theta) + right_v[i] * np.cos(theta)

        fused_x[i] = fused_x[i - 1] + world_vx * dt
        fused_y[i] = fused_y[i - 1] + world_vy * dt

        dr_only_x[i] = dr_only_x[i - 1] + world_vx * dt
        dr_only_y[i] = dr_only_y[i - 1] + world_vy * dt

        # uncertainty grows with time spent coasting on DR alone
        P_x[i] = P_x[i - 1] + process_noise_pos * dt
        P_y[i] = P_y[i - 1] + process_noise_pos * dt
        P_heading[i] = P_heading[i - 1] + np.radians(process_noise_heading_deg) * dt

        time_since_fix[i] = time_since_fix[i - 1] + dt
        mode[i] = "dead_reckoning" if time_since_fix[i] > 1.5 else "gnss_aided"

        # ---- UPDATE: apply any GNSS fix(es) that occurred by now ----
        while fix_idx < n_fixes and gnss_t[fix_idx] <= t[i]:
            R_pos = gnss_r[fix_idx]

            Kx = P_x[i] / (P_x[i] + R_pos)
            Ky = P_y[i] / (P_y[i] + R_pos)

            fused_x[i] = fused_x[i] + Kx * (gnss_x[fix_idx] - fused_x[i])
            fused_y[i] = fused_y[i] + Ky * (gnss_y[fix_idx] - fused_y[i])

            P_x[i] = (1 - Kx) * P_x[i]
            P_y[i] = (1 - Ky) * P_y[i]

            if gnss_speed[fix_idx] > min_speed_for_bearing:
                R_heading = np.radians(10.0) ** 2  # assume ~10 deg GNSS bearing noise at speed
                Kh = P_heading[i] / (P_heading[i] + R_heading)
                diff = np.arctan2(
                    np.sin(gnss_bearing_rad[fix_idx] - fused_heading[i]),
                    np.cos(gnss_bearing_rad[fix_idx] - fused_heading[i]),
                )
                fused_heading[i] = fused_heading[i] + Kh * diff
                P_heading[i] = (1 - Kh) * P_heading[i]

            time_since_fix[i] = 0.0
            mode[i] = "gnss_aided"
            fix_idx += 1

    return pd.DataFrame({
        "time": t,
        "fused_x": fused_x,
        "fused_y": fused_y,
        "fused_heading_deg": np.degrees(fused_heading),
        "dr_only_x": dr_only_x,
        "dr_only_y": dr_only_y,
        "position_uncertainty_x": np.sqrt(P_x),
        "position_uncertainty_y": np.sqrt(P_y),
        "mode": mode,
        "time_since_last_fix": time_since_fix,
    })
