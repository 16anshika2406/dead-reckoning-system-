import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STEP 5 - IMPROVED ZUPT
# ============================================================

INPUT_FILE = "vehicle_frame_acceleration.csv"
OUTPUT_FILE = "vehicle_velocity_zupt.csv"


# ============================================================
# 1. LOAD CSV
# ============================================================

data = pd.read_csv(INPUT_FILE)

print("CSV loaded successfully")
print("Columns:")
print(data.columns.tolist())
print("Samples:", len(data))


# ============================================================
# 2. FIND TIME COLUMN
# ============================================================

if "time" in data.columns:
    t = data["time"].to_numpy(dtype=float)

elif "timestamp" in data.columns:
    timestamp = data["timestamp"].to_numpy(dtype=float)

    if np.mean(np.diff(timestamp)) > 1:
        t = (timestamp - timestamp[0]) / 1000.0
    else:
        t = timestamp - timestamp[0]

else:
    print("WARNING: No time column found.")
    print("Assuming 100 Hz sampling rate.")

    t = np.arange(len(data)) / 100.0


# ============================================================
# 3. FIND VEHICLE ACCELERATION COLUMNS
# ============================================================

def find_column(possible_names):

    for name in possible_names:

        if name in data.columns:
            return name

    return None


forward_col = find_column([
    "forward",
    "vehicle_forward",
    "Forward",
    "vehicle_forward_acceleration"
])

right_col = find_column([
    "right",
    "vehicle_right",
    "Right",
    "vehicle_right_acceleration"
])

down_col = find_column([
    "down",
    "vehicle_down",
    "Down",
    "vehicle_down_acceleration"
])


print()
print("Detected columns:")
print("Forward:", forward_col)
print("Right:", right_col)
print("Down:", down_col)


if forward_col is None or right_col is None or down_col is None:

    print()
    print("ERROR: Acceleration columns not found.")
    print(data.columns.tolist())

    exit()


ax = data[forward_col].to_numpy(dtype=float)
ay = data[right_col].to_numpy(dtype=float)
az = data[down_col].to_numpy(dtype=float)


# ============================================================
# 4. REMOVE NaN
# ============================================================

ax = np.nan_to_num(ax)
ay = np.nan_to_num(ay)
az = np.nan_to_num(az)


# ============================================================
# 5. MEDIAN FILTER
# ============================================================

ax = pd.Series(ax).rolling(
    5,
    center=True,
    min_periods=1
).median().to_numpy()

ay = pd.Series(ay).rolling(
    5,
    center=True,
    min_periods=1
).median().to_numpy()

az = pd.Series(az).rolling(
    5,
    center=True,
    min_periods=1
).median().to_numpy()


# ============================================================
# 6. LOW PASS FILTER
# ============================================================

alpha = 0.12

fx = np.zeros(len(ax))
fy = np.zeros(len(ay))
fz = np.zeros(len(az))

fx[0] = ax[0]
fy[0] = ay[0]
fz[0] = az[0]

for i in range(1, len(ax)):

    fx[i] = alpha * ax[i] + (1 - alpha) * fx[i - 1]

    fy[i] = alpha * ay[i] + (1 - alpha) * fy[i - 1]

    fz[i] = alpha * az[i] + (1 - alpha) * fz[i - 1]


# ============================================================
# 7. ACCELERATION MAGNITUDE
# ============================================================

acc_mag = np.sqrt(
    fx**2 +
    fy**2 +
    fz**2
)


# ============================================================
# 8. ACCELERATION CHANGE
# ============================================================

acc_change = np.zeros(len(acc_mag))

for i in range(1, len(acc_mag)):

    acc_change[i] = abs(
        acc_mag[i] -
        acc_mag[i - 1]
    )


# ============================================================
# 9. ROLLING VARIATION
# ============================================================

window = 20

std_x = pd.Series(fx).rolling(
    window,
    center=True,
    min_periods=1
).std().fillna(0).to_numpy()

std_y = pd.Series(fy).rolling(
    window,
    center=True,
    min_periods=1
).std().fillna(0).to_numpy()

std_z = pd.Series(fz).rolling(
    window,
    center=True,
    min_periods=1
).std().fillna(0).to_numpy()


acc_std = np.sqrt(
    std_x**2 +
    std_y**2 +
    std_z**2
)


# ============================================================
# 10. IMPROVED STATIONARY DETECTION
# ============================================================

# These are deliberately less aggressive than before.

ACC_THRESHOLD = 0.20
STD_THRESHOLD = 0.08
CHANGE_THRESHOLD = 0.08


stationary = (

    (acc_mag < ACC_THRESHOLD)

    &

    (acc_std < STD_THRESHOLD)

    &

    (acc_change < CHANGE_THRESHOLD)
)


# ============================================================
# 11. REQUIRE A FEW CONSECUTIVE STATIONARY SAMPLES
# ============================================================

# Prevents isolated samples from suddenly resetting velocity.

required_samples = 8

stationary_final = np.zeros(len(stationary), dtype=bool)

count = 0

for i in range(len(stationary)):

    if stationary[i]:

        count += 1

    else:

        count = 0

    if count >= required_samples:

        stationary_final[
            i - required_samples + 1 : i + 1
        ] = True


stationary = stationary_final


print()
print("Stationary samples:", np.sum(stationary))
print("Moving samples:", np.sum(~stationary))


# ============================================================
# 12. VELOCITY INTEGRATION
# ============================================================

vx = np.zeros(len(t))
vy = np.zeros(len(t))
vz = np.zeros(len(t))


for i in range(1, len(t)):

    dt = t[i] - t[i - 1]

    # Protect against bad timestamps

    if dt <= 0 or dt > 0.1:

        dt = 0.01


    # Integrate acceleration

    vx[i] = vx[i - 1] + fx[i] * dt

    vy[i] = vy[i - 1] + fy[i] * dt

    vz[i] = vz[i - 1] + fz[i] * dt


    # ========================================================
    # ZUPT
    # ========================================================

    if stationary[i]:

        vx[i] = 0.0
        vy[i] = 0.0
        vz[i] = 0.0


# ============================================================
# 13. VELOCITY MAGNITUDE
# ============================================================

velocity_mag = np.sqrt(
    vx**2 +
    vy**2 +
    vz**2
)


# ============================================================
# 14. SAVE CSV
# ============================================================

result = pd.DataFrame({

    "time": t,

    "forward_acceleration": fx,

    "right_acceleration": fy,

    "down_acceleration": fz,

    "stationary": stationary,

    "forward_velocity": vx,

    "right_velocity": vy,

    "down_velocity": vz,

    "velocity_magnitude": velocity_mag

})


result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 15. PRINT RESULTS
# ============================================================

print()
print("======================================")
print("STEP 5 COMPLETE")
print("======================================")

print()
print("Maximum velocities:")

print(
    "Forward:",
    np.max(np.abs(vx)),
    "m/s"
)

print(
    "Right:",
    np.max(np.abs(vy)),
    "m/s"
)

print(
    "Down:",
    np.max(np.abs(vz)),
    "m/s"
)

print()
print("Saved:")
print(OUTPUT_FILE)


# ============================================================
# 16. VELOCITY GRAPH
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    t,
    vx,
    label="Forward velocity"
)

plt.plot(
    t,
    vy,
    label="Right velocity"
)

plt.plot(
    t,
    vz,
    label="Down velocity"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Time (seconds)")

plt.ylabel("Velocity (m/s)")

plt.title(
    "Vehicle Velocity - Improved ZUPT"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 17. STATIONARY DETECTION GRAPH
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    t,
    acc_mag,
    label="Acceleration magnitude"
)

plt.axhline(
    ACC_THRESHOLD,
    linestyle="--",
    label="Stationary threshold"
)

plt.xlabel("Time (seconds)")

plt.ylabel("Acceleration (m/s²)")

plt.title(
    "Improved Stationary Detection"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()