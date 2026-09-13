"""
dr_pipeline.py
==============

BASE dead-reckoning pipeline (Step 1 of the rebuild).

WHAT THIS FIXES vs. the old codebase:
--------------------------------------
The old pipeline hardcoded a single, fixed heading (heading_degrees = 0.0)
to convert between "world frame" and "vehicle frame". That only works if
the vehicle never turns. The moment it turns a corner, the whole
downstream velocity/position math becomes wrong because the forward/right
axes silently stop pointing the way the vehicle is actually pointing.

This version fixes that by using the phone's own rotation-vector
quaternion (already logged in the CSV) at EVERY timestep, not once at
the start. That gives us a continuously-updating heading theta(t), so
turns are represented correctly all the way through.

PIPELINE STAGES
----------------
1. Load raw IMU log (time, acc_xyz, gyro_xyz, quaternion)
2. Gyro bias correction (assumes first ~1s is stationary)
3. Per-sample orientation (from quaternion) -> extract yaw theta(t)
4. Remove gravity IN THE PHONE FRAME (rotate gravity vector into phone
   frame using the quaternion, subtract) -> body-frame linear acceleration
   (this doubles as "vehicle frame" under the assumption phone-forward =
   vehicle-forward; a real calibration engine replacing this assumption
   is Step 2 of the rebuild)
5. Filter (spike removal + smoothing)
6. ZUPT: zero-velocity update using acc-magnitude + gyro-magnitude with a
   consecutive-sample debounce
7. NHC: zero out lateral + vertical velocity IN BODY FRAME (this is valid
   because the body frame here is genuinely vehicle-aligned at every
   instant, unlike the old code which zeroed things in an already-wrong
   frame)
8. THE FIX: rotate the corrected body-frame (forward, right) velocity
   into a FIXED WORLD FRAME using the current heading theta(t) -- this
   is the step that was completely missing before, and it's what lets
   the trajectory curve when the vehicle turns
9. Integrate world-frame velocity -> world-frame position (x, y)

This is intentionally a monolithic, readable script (not yet split into
a package) so the whole team can read it top to bottom in one sitting.
We'll modularize once the architecture is validated.
"""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

G = 9.80665  # standard gravity, m/s^2


# ============================================================
# STAGE 1-2: LOAD + GYRO BIAS CORRECTION
# ============================================================

def load_and_prepare(csv_path, stationary_calib_seconds=1.0):
    """Load the raw IMU csv and correct gyro bias using the first
    `stationary_calib_seconds` of data (assumed stationary)."""

    data = pd.read_csv(csv_path)
    data = data.dropna().reset_index(drop=True)

    t = data["time_seconds"].to_numpy(dtype=float)
    acc = data[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=float)
    gyro = data[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(dtype=float)
    quat = data[["quat_w", "quat_x", "quat_y", "quat_z"]].to_numpy(dtype=float)

    # normalize quaternions (avoid divide-by-zero on any dropped rows)
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    quat = quat / norm

    fs = 1.0 / np.mean(np.diff(t))
    n_calib = max(1, int(stationary_calib_seconds * fs))

    gyro_bias = gyro[:n_calib].mean(axis=0)
    gyro_corrected = gyro - gyro_bias

    print(f"[load] {len(t)} samples, ~{fs:.1f} Hz, duration {t[-1]-t[0]:.2f}s")
    print(f"[load] gyro bias (rad/s): {gyro_bias}")

    return {
        "t": t,
        "acc": acc,
        "gyro": gyro_corrected,
        "quat": quat,  # [w, x, y, z] order as stored in CSV
        "fs": fs,
    }


# ============================================================
# STAGE 3-4: ORIENTATION + GRAVITY REMOVAL IN BODY FRAME
# ============================================================

def compute_body_frame_acceleration(d):
    """
    For every sample:
      - build rotation R(t): phone-frame -> world-frame from quaternion
      - rotate the WORLD gravity vector [0,0,g] into the phone frame
        using R(t)^-1
      - subtract it from the raw accelerometer reading
        -> body-frame linear acceleration (gravity-free, still expressed
           in the phone's own rotating axes = "vehicle frame" under the
           rigid-mount assumption)
      - also extract yaw (heading) from R(t) for later use in Stage 8

    NOTE ON QUATERNION ORDER: scipy expects [x, y, z, w]; our CSV stores
    [w, x, y, z], so we reorder before constructing Rotation objects.
    """

    quat_wxyz = d["quat"]
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]

    rotations = R.from_quat(quat_xyzw)

    gravity_world = np.array([0.0, 0.0, G])

    n = len(d["t"])
    body_linear_acc = np.zeros((n, 3))
    heading = np.zeros(n)  # yaw, radians, world-referenced

    for i in range(n):
        Ri = rotations[i]

        # gravity expressed in the phone's current orientation
        gravity_in_phone = Ri.inv().apply(gravity_world)

        body_linear_acc[i] = d["acc"][i] - gravity_in_phone

        # yaw of the phone in the world frame ('ZYX' Euler -> first angle
        # is yaw about world Z)
        yaw, pitch, roll = Ri.as_euler("ZYX")
        heading[i] = yaw

    print(f"[orientation] heading range: "
          f"{np.degrees(heading.min()):.1f} deg to "
          f"{np.degrees(heading.max()):.1f} deg "
          f"(span {np.degrees(heading.max()-heading.min()):.1f} deg)")

    return body_linear_acc, heading


# ============================================================
# STAGE 5: FILTER (spike removal + smoothing)
# ============================================================

def filter_signal(acc, spike_threshold=15.0, smooth_window=5):

    def remove_spikes(x):
        out = x.copy()
        bad = np.where(np.abs(out) > spike_threshold)[0]
        for i in bad:
            if i == 0:
                out[i] = out[i + 1]
            elif i == len(out) - 1:
                out[i] = out[i - 1]
            else:
                out[i] = 0.5 * (out[i - 1] + out[i + 1])
        return out

    def smooth(x):
        return pd.Series(x).rolling(
            smooth_window, center=True, min_periods=1
        ).mean().to_numpy()

    filtered = np.zeros_like(acc)
    for axis in range(3):
        filtered[:, axis] = smooth(remove_spikes(acc[:, axis]))

    return filtered


# ============================================================
# STAGE 6: ZUPT DETECTION
# ============================================================

def detect_stationary(acc_body, gyro, acc_threshold=0.20,
                       gyro_threshold=0.06, min_consecutive=8):

    acc_mag = np.linalg.norm(acc_body, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    raw = (acc_mag < acc_threshold) & (gyro_mag < gyro_threshold)

    stationary = np.zeros(len(raw), dtype=bool)
    count = 0
    for i in range(len(raw)):
        count = count + 1 if raw[i] else 0
        if count >= min_consecutive:
            stationary[i - min_consecutive + 1: i + 1] = True

    print(f"[zupt] stationary samples: {stationary.sum()} / {len(stationary)} "
          f"({100*stationary.mean():.1f}%)")

    return stationary, acc_mag, gyro_mag


# ============================================================
# STAGE 7-9: NHC + WORLD-FRAME ROTATION (THE FIX) + INTEGRATION
# ============================================================

def integrate_trajectory(d, acc_body, heading, stationary,
                          max_velocity=15.0):
    """
    Body frame convention (matches original codebase):
        axis 0 = forward
        axis 1 = right
        axis 2 = down

    Steps per sample:
      a) integrate body-frame acceleration -> body-frame velocity
      b) NHC: zero the right/down velocity components (valid in body
         frame at every instant)
      c) ZUPT: zero all velocity components if stationary
      d) *** THE FIX ***: rotate (forward, right) velocity into world
         (x, y) using the CURRENT heading theta(t), not a fixed value
      e) integrate world-frame velocity -> world-frame position
    """

    t = d["t"]
    n = len(t)

    forward_v = np.zeros(n)
    right_v = np.zeros(n)
    down_v = np.zeros(n)

    world_x = np.zeros(n)   # e.g. "East-ish", meters
    world_y = np.zeros(n)   # e.g. "North-ish", meters

    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if dt <= 0 or dt > 0.2:
            dt = 1.0 / d["fs"]

        # (a) integrate acceleration -> body velocity
        forward_v[i] = forward_v[i - 1] + acc_body[i, 0] * dt
        right_v[i] = right_v[i - 1] + acc_body[i, 1] * dt
        down_v[i] = down_v[i - 1] + acc_body[i, 2] * dt

        # (b) NHC
        right_v[i] = 0.0
        down_v[i] = 0.0

        # (c) ZUPT
        if stationary[i]:
            forward_v[i] = 0.0
            right_v[i] = 0.0
            down_v[i] = 0.0

        # clip runaway velocity (sensor glitches)
        forward_v[i] = np.clip(forward_v[i], -max_velocity, max_velocity)

        # (d) *** THE FIX: rotate into world frame using CURRENT heading ***
        theta = heading[i]
        world_vx = forward_v[i] * np.cos(theta) - right_v[i] * np.sin(theta)
        world_vy = forward_v[i] * np.sin(theta) + right_v[i] * np.cos(theta)

        # (e) integrate to position
        world_x[i] = world_x[i - 1] + world_vx * dt
        world_y[i] = world_y[i - 1] + world_vy * dt

    return pd.DataFrame({
        "time": t,
        "heading_deg": np.degrees(heading),
        "forward_velocity": forward_v,
        "right_velocity": right_v,
        "stationary": stationary,
        "world_x": world_x,
        "world_y": world_y,
    })


# ============================================================
# STRAIGHT-LINE-ONLY BASELINE (reproduces the OLD bug, for comparison)
# ============================================================

def integrate_trajectory_old_buggy(d, acc_body, stationary, max_velocity=15.0):
    """Reproduces the OLD behaviour: heading fixed at 0 forever, so the
    trajectory can only ever go in a straight line. Used only to visually
    prove the difference the fix makes."""

    t = d["t"]
    n = len(t)

    forward_v = np.zeros(n)
    right_v = np.zeros(n)

    world_x = np.zeros(n)
    world_y = np.zeros(n)

    fixed_heading = 0.0  # <-- the bug

    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if dt <= 0 or dt > 0.2:
            dt = 1.0 / d["fs"]

        forward_v[i] = forward_v[i - 1] + acc_body[i, 0] * dt
        right_v[i] = 0.0  # old NHC hack

        if stationary[i]:
            forward_v[i] = 0.0

        forward_v[i] = np.clip(forward_v[i], -max_velocity, max_velocity)

        world_vx = forward_v[i] * np.cos(fixed_heading) - right_v[i] * np.sin(fixed_heading)
        world_vy = forward_v[i] * np.sin(fixed_heading) + right_v[i] * np.cos(fixed_heading)

        world_x[i] = world_x[i - 1] + world_vx * dt
        world_y[i] = world_y[i - 1] + world_vy * dt

    return world_x, world_y


# ============================================================
# ORCHESTRATOR
# ============================================================

def run_pipeline(csv_path):
    d = load_and_prepare(csv_path)
    body_acc_raw, heading = compute_body_frame_acceleration(d)
    body_acc = filter_signal(body_acc_raw)
    stationary, acc_mag, gyro_mag = detect_stationary(body_acc, d["gyro"])
    result = integrate_trajectory(d, body_acc, heading, stationary)

    old_x, old_y = integrate_trajectory_old_buggy(d, body_acc, stationary)
    result["old_buggy_world_x"] = old_x
    result["old_buggy_world_y"] = old_y

    return result, d, acc_mag, gyro_mag


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "imu_data_1788285144311.csv"
    result, d, acc_mag, gyro_mag = run_pipeline(path)
    result.to_csv("base_pipeline_output.csv", index=False)
    print("\nSaved: base_pipeline_output.csv")
    print(f"Final position (fixed):  x={result['world_x'].iloc[-1]:.3f} m, "
          f"y={result['world_y'].iloc[-1]:.3f} m")
    print(f"Final position (buggy):  x={result['old_buggy_world_x'].iloc[-1]:.3f} m, "
          f"y={result['old_buggy_world_y'].iloc[-1]:.3f} m")
