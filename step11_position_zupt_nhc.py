# ============================================================
# STEP 11: POSITION FROM ZUPT + NHC CORRECTED VELOCITY
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. LOAD CSV
# ------------------------------------------------------------

file_path = r"E:\SIH\DeadReckoning\vehicle_frame_acceleration.csv"

data = pd.read_csv(file_path)

print("CSV loaded successfully")
print()
print("Columns Found:")
print(data.columns.tolist())
print()


# ------------------------------------------------------------
# 2. FIND TIME COLUMN
# ------------------------------------------------------------

time_candidates = [
    "time_seconds",
    "time",
    "timestamp",
    "Time"
]

time_col = None

for col in time_candidates:
    if col in data.columns:
        time_col = col
        break

if time_col is None:
    print("ERROR: Could not find time column.")
    print("Available columns:")
    print(data.columns.tolist())
    raise SystemExit

time = data[time_col].to_numpy(dtype=float)


# ------------------------------------------------------------
# 3. READ VEHICLE-FRAME ACCELERATION
# ------------------------------------------------------------

forward_acc = data["vehicle_forward"].to_numpy(dtype=float)
right_acc   = data["vehicle_right"].to_numpy(dtype=float)
down_acc    = data["vehicle_down"].to_numpy(dtype=float)


# ------------------------------------------------------------
# 4. REMOVE INVALID VALUES
# ------------------------------------------------------------

forward_acc = np.nan_to_num(forward_acc)
right_acc   = np.nan_to_num(right_acc)
down_acc    = np.nan_to_num(down_acc)


# ------------------------------------------------------------
# 5. CREATE VELOCITY BY INTEGRATING ACCELERATION
# ------------------------------------------------------------

forward_velocity = np.zeros(len(time))
right_velocity   = np.zeros(len(time))
down_velocity    = np.zeros(len(time))

for i in range(1, len(time)):

    dt = time[i] - time[i - 1]

    # Ignore invalid time differences
    if dt <= 0:
        dt = 0

    forward_velocity[i] = (
        forward_velocity[i - 1]
        + forward_acc[i] * dt
    )

    right_velocity[i] = (
        right_velocity[i - 1]
        + right_acc[i] * dt
    )

    down_velocity[i] = (
        down_velocity[i - 1]
        + down_acc[i] * dt
    )


# ------------------------------------------------------------
# 6. ZUPT
# ------------------------------------------------------------
# When the vehicle is stationary, velocity should be zero.
#
# We detect stationary periods using small acceleration.
# ------------------------------------------------------------

acc_magnitude = np.sqrt(
    forward_acc**2 +
    right_acc**2 +
    down_acc**2
)

stationary = acc_magnitude < 0.15


# ------------------------------------------------------------
# 7. APPLY ZUPT
# ------------------------------------------------------------

for i in range(len(time)):

    if stationary[i]:

        forward_velocity[i] = 0.0
        right_velocity[i] = 0.0
        down_velocity[i] = 0.0


# ------------------------------------------------------------
# 8. NON-HOLONOMIC CONSTRAINT (NHC)
# ------------------------------------------------------------
# For a normal ground vehicle:
#
# Forward velocity -> allowed
# Right velocity   -> approximately zero
# Down velocity    -> approximately zero
#
# Therefore:
#
# V_right = 0
# V_down  = 0
# ------------------------------------------------------------

right_velocity[:] = 0.0
down_velocity[:] = 0.0


# ------------------------------------------------------------
# 9. CREATE POSITION ARRAYS
# ------------------------------------------------------------

forward_position = np.zeros(len(time))
right_position   = np.zeros(len(time))
down_position    = np.zeros(len(time))


# ------------------------------------------------------------
# 10. INTEGRATE VELOCITY TO POSITION
# ------------------------------------------------------------

for i in range(1, len(time)):

    dt = time[i] - time[i - 1]

    if dt <= 0:
        dt = 0

    forward_position[i] = (
        forward_position[i - 1]
        + forward_velocity[i] * dt
    )

    right_position[i] = (
        right_position[i - 1]
        + right_velocity[i] * dt
    )

    down_position[i] = (
        down_position[i - 1]
        + down_velocity[i] * dt
    )


# ------------------------------------------------------------
# 11. APPLY POSITION HOLD DURING ZUPT
# ------------------------------------------------------------

for i in range(len(time)):

    if stationary[i]:

        if i > 0:
            forward_position[i] = forward_position[i - 1]
            right_position[i] = right_position[i - 1]
            down_position[i] = down_position[i - 1]


# ------------------------------------------------------------
# 12. PRINT FINAL POSITION
# ------------------------------------------------------------

print("======================================")
print("FINAL POSITION")
print("======================================")

print(
    f"Forward position : {forward_position[-1]:.4f} m"
)

print(
    f"Right position   : {right_position[-1]:.4f} m"
)

print(
    f"Down position    : {down_position[-1]:.4f} m"
)

print()


# ------------------------------------------------------------
# 13. PLOT POSITION
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    time,
    forward_position,
    label="Forward position"
)

plt.plot(
    time,
    right_position,
    label="Right position"
)

plt.plot(
    time,
    down_position,
    label="Down position"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Position (m)")
plt.title("Vehicle Position - ZUPT + NHC")

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 14. 2D VEHICLE TRAJECTORY
# ------------------------------------------------------------
# X = Right
# Y = Forward
# ------------------------------------------------------------

plt.figure(figsize=(10, 7))

plt.plot(
    right_position,
    forward_position,
    label="Vehicle trajectory"
)

# Start point
plt.scatter(
    right_position[0],
    forward_position[0],
    s=100,
    label="Start"
)

# End point
plt.scatter(
    right_position[-1],
    forward_position[-1],
    s=100,
    marker="X",
    label="End"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Right position (m)")
plt.ylabel("Forward position (m)")

plt.title("Vehicle 2D Trajectory - ZUPT + NHC")

plt.grid(True)
plt.axis("equal")
plt.legend()

plt.tight_layout()
plt.show()