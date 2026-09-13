"""
demo_fusion.py  (v2 — corrected)
==================================

The first version of this demo made a mistake worth documenting rather
than hiding: it generated the "GNSS" stream FROM the same DR output it
then tested against, so there was no real drift for fusion to correct
-- the comparison was circular and made fusion look pointless. Fixing
that here.

THIS VERSION uses a genuinely independent synthetic ground truth:
  - A hand-specified path (straight line -> turn -> straight line) with
    known, exact position/heading at every instant.
  - A simulated "raw IMU/DR" trajectory built by injecting REALISTIC
    drift sources onto that ground truth:
      * constant accelerometer bias -> velocity error growing linearly
        -> position error growing QUADRATICALLY over time (this is the
        textbook behaviour of uncorrected INS, and worth saying out loud
        in a judge Q&A)
      * constant gyro bias -> heading error growing linearly over time,
        which then ALSO corrupts position (because velocity gets rotated
        by the wrong heading) -- heading drift is the nastier of the two
  - A simulated noisy, 1Hz GNSS stream (from the TRUE path, not the
    drifting one), with a dropout window standing in for a tunnel.

This is still fully synthetic (labelled honestly) -- it validates the
FUSION ALGORITHM's math and behaviour before we have a real paired
IMU+GNSS recording with a real tunnel. Once DATA_COLLECTION_GUIDE.md
data comes in, re-run this same fusion engine on the real merged data.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gnss_simulator import xy_to_latlon
from gnss_fusion import fuse

# ------------------------------------------------------------
# 1. TRUE ground-truth path (independent of any DR computation)
# ------------------------------------------------------------
dt = 0.02  # 50 Hz, matches our real IMU rate
duration = 90.0  # seconds
t = np.arange(0, duration, dt)
n = len(t)

true_speed = 12.0  # m/s (~43 km/h), constant cruising speed after ramp-up
ramp_up_time = 5.0

speed_true = np.clip(true_speed * (t / ramp_up_time), 0, true_speed)

# path: straight for 35s, then a smooth 90 deg turn over 10s, then straight
heading_true = np.zeros(n)
turn_start, turn_duration = 35.0, 10.0
turn_rate = np.radians(90) / turn_duration
for i in range(n):
    if t[i] < turn_start:
        heading_true[i] = 0.0
    elif t[i] < turn_start + turn_duration:
        heading_true[i] = turn_rate * (t[i] - turn_start)
    else:
        heading_true[i] = np.radians(90)

vx_true = speed_true * np.cos(heading_true)
vy_true = speed_true * np.sin(heading_true)
x_true = np.cumsum(vx_true) * dt
y_true = np.cumsum(vy_true) * dt

forward_accel_true = np.gradient(speed_true, dt)  # what the accelerometer "should" read (forward axis)

# ------------------------------------------------------------
# 2. Simulate a DRIFTING raw IMU/DR track: inject realistic sensor bias
# ------------------------------------------------------------
ACCEL_BIAS = 0.03       # m/s^2 constant forward-axis accelerometer bias
GYRO_BIAS_DEG_S = 0.4   # deg/s constant heading-rate (gyro) bias
ACCEL_NOISE_STD = 0.05
GYRO_NOISE_STD_DEG = 0.05

rng = np.random.default_rng(7)

# vehicle_acc fed to fuse(): axis 0 = forward. This is what a real
# accelerometer WOULD read given the bias + noise on top of true motion.
vehicle_acc = np.zeros((n, 3))
vehicle_acc[:, 0] = forward_accel_true + ACCEL_BIAS + rng.normal(0, ACCEL_NOISE_STD, n)

# raw "orientation source" fed to fuse(): true heading corrupted by a
# constant gyro bias integrated over time, plus noise
gyro_bias_rad_s = np.radians(GYRO_BIAS_DEG_S)
heading_drifted = heading_true + gyro_bias_rad_s * t + np.radians(
    rng.normal(0, GYRO_NOISE_STD_DEG, n)
)

stationary = np.zeros(n, dtype=bool)  # no stops in this synthetic drive

d = {"t": t, "fs": 1.0 / dt}

# ------------------------------------------------------------
# 3. Simulate GNSS from the TRUE path (not the drifting one!)
# ------------------------------------------------------------
origin_lat, origin_lon = 28.7041, 77.1025
fix_rate_hz = 1.0
position_noise_std_m = 3.0
outage_start, outage_duration = 45.0, 8.0  # deliberately spans part of the turn

t_gnss = np.arange(0, duration, 1.0 / fix_rate_hz)
x_gnss_true = np.interp(t_gnss, t, x_true)
y_gnss_true = np.interp(t_gnss, t, y_true)
speed_gnss_true = np.interp(t_gnss, t, speed_true)
heading_gnss_true = np.interp(t_gnss, t, heading_true)

x_gnss_noisy = x_gnss_true + rng.normal(0, position_noise_std_m, len(t_gnss))
y_gnss_noisy = y_gnss_true + rng.normal(0, position_noise_std_m, len(t_gnss))
lat, lon = xy_to_latlon(x_gnss_noisy, y_gnss_noisy, origin_lat, origin_lon)

is_outage = (t_gnss >= outage_start) & (t_gnss < outage_start + outage_duration)

gnss_df = pd.DataFrame({
    "time_seconds": t_gnss,
    "latitude": lat,
    "longitude": lon,
    "speed": speed_gnss_true,
    "bearing": (np.degrees(heading_gnss_true)) % 360,
    "horizontal_accuracy": np.full(len(t_gnss), position_noise_std_m),
})
gnss_with_outage = gnss_df[~is_outage].reset_index(drop=True)

print(f"[demo] {len(gnss_df)} true GNSS fixes generated, "
      f"{is_outage.sum()} dropped in simulated {outage_duration:.0f}s outage "
      f"starting at t={outage_start:.0f}s")

# ------------------------------------------------------------
# 4. Run fusion (dr_only_x/y come back from the same call for comparison)
# ------------------------------------------------------------
fusion_result = fuse(d, vehicle_acc, heading_drifted, stationary, gnss_with_outage)

fused_x, fused_y = fusion_result["fused_x"].to_numpy(), fusion_result["fused_y"].to_numpy()
dr_only_x, dr_only_y = fusion_result["dr_only_x"].to_numpy(), fusion_result["dr_only_y"].to_numpy()

fused_err = np.sqrt((fused_x - x_true) ** 2 + (fused_y - y_true) ** 2)
dr_only_err = np.sqrt((dr_only_x - x_true) ** 2 + (dr_only_y - y_true) ** 2)

print("\n=== Final position error at end of 90s trip ===")
print(f"Fused (GNSS+INS): {fused_err[-1]:.2f} m")
print(f"Pure DR-only:      {dr_only_err[-1]:.2f} m   <- unbounded growth without GNSS")

outage_mask = (t >= outage_start) & (t < outage_start + outage_duration)
print(f"\n=== Error growth DURING the {outage_duration:.0f}s outage ===")
print(f"Fused:   {fused_err[outage_mask][0]:.2f} m -> {fused_err[outage_mask][-1]:.2f} m")
print(f"Pure DR: {dr_only_err[outage_mask][0]:.2f} m -> {dr_only_err[outage_mask][-1]:.2f} m")

post_outage_mask = (t >= outage_start + outage_duration) & (t < outage_start + outage_duration + 5.0)
print(f"\n=== How fast fusion re-converges after GNSS returns ===")
print(f"Fused error 5s after GNSS returns: {fused_err[post_outage_mask][-1]:.2f} m")

# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

axes[0].plot(x_true, y_true, color="black", linewidth=2.5, label="Ground truth")
axes[0].plot(fused_x, fused_y, color="darkgreen", linewidth=1.6, label="Fused (GNSS+INS)")
axes[0].plot(dr_only_x, dr_only_y, color="crimson", linewidth=1.4, linestyle="--", label="Pure DR (no GNSS)")
axes[0].scatter(x_gnss_noisy, y_gnss_noisy, s=10, color="royalblue", alpha=0.5, label="Noisy GNSS fixes")
axes[0].set_title("Trajectory: straight -> turn -> straight,\nwith a simulated GNSS outage during the turn")
axes[0].set_xlabel("X (m)"); axes[0].set_ylabel("Y (m)")
axes[0].axis("equal"); axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=8)

axes[1].plot(t, fused_err, color="darkgreen", label="Fused error")
axes[1].plot(t, dr_only_err, color="crimson", linestyle="--", label="Pure DR error")
axes[1].axvspan(outage_start, outage_start + outage_duration, color="gray", alpha=0.2, label="GNSS outage")
axes[1].set_title("Position error vs. ground truth over time")
axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Error (m)")
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

mode_numeric = (fusion_result["mode"] == "dead_reckoning").astype(int)
axes[2].fill_between(t, 0, mode_numeric, step="mid", color="orange", alpha=0.6,
                      label="Dead-reckoning mode")
axes[2].axvspan(outage_start, outage_start + outage_duration, color="gray", alpha=0.15)
axes[2].set_title("Mode indicator (0=GNSS-aided, 1=pure DR)")
axes[2].set_xlabel("Time (s)"); axes[2].set_ylim(-0.1, 1.1)
axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fusion_demo.png", dpi=150)
print("\nSaved fusion_demo.png")
