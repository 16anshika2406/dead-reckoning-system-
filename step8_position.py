import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STEP 8 - POSITION ESTIMATION
# Velocity -> Position
# ============================================================

INPUT_FILE = "vehicle_velocity_zupt.csv"
OUTPUT_FILE = "vehicle_position.csv"


# ============================================================
# 1. LOAD CSV
# ============================================================

data = pd.read_csv(INPUT_FILE)

print("======================================")
print("STEP 8 - POSITION ESTIMATION")
print("======================================")

print()
print("CSV loaded successfully")
print("Columns:")
print(data.columns.tolist())

print()
print("Samples:", len(data))


# ============================================================
# 2. READ TIME
# ============================================================

if "time" in data.columns:

    t = data["time"].to_numpy(dtype=float)

else:

    print("ERROR: time column not found.")
    exit()


# ============================================================
# 3. READ VELOCITY
# ============================================================

vx = data["forward_velocity"].to_numpy(dtype=float)

vy = data["right_velocity"].to_numpy(dtype=float)

vz = data["down_velocity"].to_numpy(dtype=float)


# ============================================================
# 4. REMOVE NaN / INVALID VALUES
# ============================================================

vx = np.nan_to_num(vx)

vy = np.nan_to_num(vy)

vz = np.nan_to_num(vz)


# ============================================================
# 5. INITIAL POSITION
# ============================================================

px = np.zeros(len(t))

py = np.zeros(len(t))

pz = np.zeros(len(t))


# ============================================================
# 6. INTEGRATE VELOCITY -> POSITION
#
# Trapezoidal integration:
#
# position_new =
# position_old +
# (velocity_old + velocity_new)/2 * dt
# ============================================================

for i in range(1, len(t)):

    dt = t[i] - t[i - 1]

    # Protect against bad timestamps

    if dt <= 0 or dt > 0.1:

        dt = 0.01


    # Forward position

    px[i] = px[i - 1] + (
        (vx[i - 1] + vx[i]) / 2
    ) * dt


    # Right position

    py[i] = py[i - 1] + (
        (vy[i - 1] + vy[i]) / 2
    ) * dt


    # Down position

    pz[i] = pz[i - 1] + (
        (vz[i - 1] + vz[i]) / 2
    ) * dt


# ============================================================
# 7. 3D DISTANCE FROM START
# ============================================================

displacement = np.sqrt(
    px**2 +
    py**2 +
    pz**2
)


# ============================================================
# 8. HORIZONTAL DISTANCE
# Forward + Right
# ============================================================

horizontal_distance = np.sqrt(
    px**2 +
    py**2
)


# ============================================================
# 9. TOTAL PATH DISTANCE
# ============================================================

path_distance = np.zeros(len(t))

for i in range(1, len(t)):

    dx = px[i] - px[i - 1]

    dy = py[i] - py[i - 1]

    dz = pz[i] - pz[i - 1]

    segment = np.sqrt(
        dx**2 +
        dy**2 +
        dz**2
    )

    path_distance[i] = (
        path_distance[i - 1] + segment
    )


# ============================================================
# 10. SAVE POSITION CSV
# ============================================================

result = pd.DataFrame({

    "time": t,

    "forward_velocity": vx,

    "right_velocity": vy,

    "down_velocity": vz,

    "forward_position": px,

    "right_position": py,

    "down_position": pz,

    "horizontal_distance": horizontal_distance,

    "3d_displacement": displacement,

    "path_distance": path_distance

})


result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 11. PRINT FINAL RESULTS
# ============================================================

print()
print("======================================")
print("POSITION RESULTS")
print("======================================")

print()
print(
    "Final Forward Position =",
    px[-1],
    "m"
)

print(
    "Final Right Position   =",
    py[-1],
    "m"
)

print(
    "Final Down Position    =",
    pz[-1],
    "m"
)

print()
print(
    "Final Horizontal Distance =",
    horizontal_distance[-1],
    "m"
)

print(
    "Final 3D Displacement =",
    displacement[-1],
    "m"
)

print(
    "Total Path Distance =",
    path_distance[-1],
    "m"
)

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("======================================")
print("STEP 8 COMPLETE")
print("======================================")


# ============================================================
# 12. POSITION vs TIME
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    t,
    px,
    label="Forward position"
)

plt.plot(
    t,
    py,
    label="Right position"
)

plt.plot(
    t,
    pz,
    label="Down position"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Time (seconds)")

plt.ylabel("Position (m)")

plt.title(
    "Vehicle Position from IMU"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 13. TOP-DOWN TRAJECTORY
# ============================================================

plt.figure(figsize=(8, 8))

plt.plot(
    px,
    py,
    label="Trajectory"
)

plt.scatter(
    px[0],
    py[0],
    label="START"
)

plt.scatter(
    px[-1],
    py[-1],
    label="END"
)

plt.xlabel("Forward Position (m)")

plt.ylabel("Right Position (m)")

plt.title(
    "Vehicle Trajectory - Top View"
)

plt.grid(True)

plt.axis("equal")

plt.legend()

plt.tight_layout()

plt.show()