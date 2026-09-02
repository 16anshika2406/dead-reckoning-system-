# ============================================================
# STEP 9 - 2D VEHICLE TRAJECTORY
# Forward vs Right Position
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# 1. LOAD POSITION DATA
# ------------------------------------------------------------

FILE = "vehicle_position.csv"

data = pd.read_csv(FILE)

print("CSV loaded successfully")
print("Columns:")
print(data.columns.tolist())


# ------------------------------------------------------------
# 2. READ TIME
# ------------------------------------------------------------

time = data["time"].to_numpy(dtype=float)


# ------------------------------------------------------------
# 3. READ VEHICLE POSITION
# ------------------------------------------------------------

forward = data["forward_position"].to_numpy(dtype=float)
right = data["right_position"].to_numpy(dtype=float)
down = data["down_position"].to_numpy(dtype=float)


# ------------------------------------------------------------
# 4. REMOVE NaN / INVALID VALUES
# ------------------------------------------------------------

valid = (
    np.isfinite(time)
    & np.isfinite(forward)
    & np.isfinite(right)
    & np.isfinite(down)
)

time = time[valid]
forward = forward[valid]
right = right[valid]
down = down[valid]


# ------------------------------------------------------------
# 5. PRINT FINAL POSITION
# ------------------------------------------------------------

print()
print("FINAL POSITION")
print("============================")

print(f"Forward = {forward[-1]:.3f} m")
print(f"Right   = {right[-1]:.3f} m")
print(f"Down    = {down[-1]:.3f} m")


# ------------------------------------------------------------
# 6. TOTAL HORIZONTAL DISTANCE FROM ORIGIN
# ------------------------------------------------------------

horizontal_distance = np.sqrt(
    forward[-1]**2 +
    right[-1]**2
)

print()
print(f"Horizontal displacement = {horizontal_distance:.3f} m")


# ------------------------------------------------------------
# 7. 2D TRAJECTORY
# ------------------------------------------------------------

plt.figure(figsize=(10, 7))

plt.plot(
    right,
    forward,
    linewidth=2,
    label="Vehicle trajectory"
)

# Starting point
plt.scatter(
    right[0],
    forward[0],
    s=100,
    marker="o",
    label="Start"
)

# Ending point
plt.scatter(
    right[-1],
    forward[-1],
    s=100,
    marker="X",
    label="End"
)

# Direction reference
plt.axhline(0, linestyle="--", linewidth=1)
plt.axvline(0, linestyle="--", linewidth=1)

plt.xlabel("Right position (m)")
plt.ylabel("Forward position (m)")

plt.title("Vehicle 2D Trajectory from IMU")

plt.grid(True)
plt.axis("equal")
plt.legend()

plt.show()