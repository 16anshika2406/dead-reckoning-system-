import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# STEP 10 / 11
# IMPROVED ZUPT
# ACCELERATION + GYROSCOPE STATIONARY DETECTION
# ============================================================


# ============================================================
# 1. FILES
# ============================================================

# Original Android IMU data
IMU_FILE = r"E:\SIH\DeadReckoning\imu_data_1788285144311.csv"

# Vehicle-frame acceleration from previous step
VEHICLE_FILE = r"E:\SIH\DeadReckoning\vehicle_frame_acceleration.csv"


# ============================================================
# 2. LOAD FILES
# ============================================================

imu = pd.read_csv(IMU_FILE)
vehicle = pd.read_csv(VEHICLE_FILE)

print("==========================================")
print("FILES LOADED")
print("==========================================")

print("\nIMU columns:")
print(imu.columns.tolist())

print("\nVehicle columns:")
print(vehicle.columns.tolist())


# ============================================================
# 3. REQUIRED COLUMNS
# ============================================================

imu_required = [
    "time_seconds",
    "gyro_x",
    "gyro_y",
    "gyro_z"
]

vehicle_required = [
    "time_seconds",
    "vehicle_forward",
    "vehicle_right",
    "vehicle_down"
]


for column in imu_required:
    if column not in imu.columns:
        raise ValueError(
            f"Missing IMU column: {column}"
        )


for column in vehicle_required:
    if column not in vehicle.columns:
        raise ValueError(
            f"Missing vehicle column: {column}"
        )


# ============================================================
# 4. READ DATA
# ============================================================

time = vehicle["time_seconds"].to_numpy(dtype=float)

forward_acc = vehicle[
    "vehicle_forward"
].to_numpy(dtype=float)

right_acc = vehicle[
    "vehicle_right"
].to_numpy(dtype=float)

down_acc = vehicle[
    "vehicle_down"
].to_numpy(dtype=float)


# ============================================================
# 5. READ GYROSCOPE
# ============================================================

gyro_time = imu[
    "time_seconds"
].to_numpy(dtype=float)

gyro_x = imu[
    "gyro_x"
].to_numpy(dtype=float)

gyro_y = imu[
    "gyro_y"
].to_numpy(dtype=float)

gyro_z = imu[
    "gyro_z"
].to_numpy(dtype=float)


# ============================================================
# 6. CLEAN NaN / INFINITE VALUES
# ============================================================

forward_acc = np.nan_to_num(
    forward_acc,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

right_acc = np.nan_to_num(
    right_acc,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

down_acc = np.nan_to_num(
    down_acc,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

gyro_x = np.nan_to_num(
    gyro_x,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

gyro_y = np.nan_to_num(
    gyro_y,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

gyro_z = np.nan_to_num(
    gyro_z,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


# ============================================================
# 7. ACCELERATION MAGNITUDE
# ============================================================

acc_magnitude = np.sqrt(
    forward_acc ** 2 +
    right_acc ** 2 +
    down_acc ** 2
)


# ============================================================
# 8. GYROSCOPE MAGNITUDE
# ============================================================

gyro_magnitude = np.sqrt(
    gyro_x ** 2 +
    gyro_y ** 2 +
    gyro_z ** 2
)


# ============================================================
# 9. ALIGN GYROSCOPE WITH VEHICLE TIME
# ============================================================

# The IMU and vehicle CSV may not have exactly
# the same timestamps.
#
# For every vehicle timestamp, find the closest
# gyroscope measurement.

gyro_aligned = np.interp(
    time,
    gyro_time,
    gyro_magnitude
)


# ============================================================
# 10. THRESHOLDS
# ============================================================

# Vehicle acceleration threshold
ACC_THRESHOLD = 0.15

# Gyroscope threshold
GYRO_THRESHOLD = 0.05


print("\n==========================================")
print("ZUPT PARAMETERS")
print("==========================================")

print(
    "Acceleration threshold:",
    ACC_THRESHOLD,
    "m/s²"
)

print(
    "Gyroscope threshold:",
    GYRO_THRESHOLD,
    "rad/s"
)


# ============================================================
# 11. INITIAL STATIONARY DETECTION
# ============================================================

stationary_raw = (
    (acc_magnitude < ACC_THRESHOLD)
    &
    (gyro_aligned < GYRO_THRESHOLD)
)


# ============================================================
# 12. REMOVE VERY SHORT STATIONARY EVENTS
# ============================================================

# A single low-acceleration sample should not
# immediately trigger ZUPT.
#
# We require several consecutive samples.

MIN_STATIONARY_SAMPLES = 8

stationary = np.zeros(
    len(time),
    dtype=bool
)

count = 0

for i in range(len(time)):

    if stationary_raw[i]:

        count += 1

    else:

        count = 0

    if count >= MIN_STATIONARY_SAMPLES:

        start = i - MIN_STATIONARY_SAMPLES + 1

        stationary[start:i + 1] = True


# ============================================================
# 13. PRINT DETECTION INFORMATION
# ============================================================

print("\n==========================================")
print("STATIONARY DETECTION")
print("==========================================")

print(
    "Total samples:",
    len(time)
)

print(
    "Stationary samples:",
    np.sum(stationary)
)

print(
    "Moving samples:",
    np.sum(~stationary)
)


# ============================================================
# 14. VELOCITY ARRAYS
# ============================================================

forward_velocity = np.zeros(
    len(time)
)

right_velocity = np.zeros(
    len(time)
)

down_velocity = np.zeros(
    len(time)
)


# ============================================================
# 15. VELOCITY INTEGRATION + ZUPT
# ============================================================

for i in range(1, len(time)):

    dt = time[i] - time[i - 1]

    # Protect against bad timestamps
    if dt <= 0 or dt > 0.1:

        dt = 0.01


    # --------------------------------------------------------
    # INTEGRATE ACCELERATION
    # --------------------------------------------------------

    forward_velocity[i] = (
        forward_velocity[i - 1]
        +
        forward_acc[i] * dt
    )

    right_velocity[i] = (
        right_velocity[i - 1]
        +
        right_acc[i] * dt
    )

    down_velocity[i] = (
        down_velocity[i - 1]
        +
        down_acc[i] * dt
    )


    # --------------------------------------------------------
    # ZERO VELOCITY UPDATE
    # --------------------------------------------------------

    if stationary[i]:

        forward_velocity[i] = 0.0
        right_velocity[i] = 0.0
        down_velocity[i] = 0.0


# ============================================================
# 16. SAVE RESULT
# ============================================================

output = pd.DataFrame({

    "time_seconds": time,

    "vehicle_forward_acc": forward_acc,

    "vehicle_right_acc": right_acc,

    "vehicle_down_acc": down_acc,

    "acceleration_magnitude": acc_magnitude,

    "gyro_magnitude": gyro_aligned,

    "stationary": stationary,

    "forward_velocity": forward_velocity,

    "right_velocity": right_velocity,

    "down_velocity": down_velocity
})


OUTPUT_FILE = (
    r"E:\SIH\DeadReckoning"
    r"\step10_improved_zupt.csv"
)

output.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n==========================================")
print("RESULT SAVED")
print("==========================================")

print(OUTPUT_FILE)


# ============================================================
# 17. VELOCITY GRAPH
# ============================================================

plt.figure(figsize=(14, 7))

plt.plot(
    time,
    forward_velocity,
    label="Forward velocity"
)

plt.plot(
    time,
    right_velocity,
    label="Right velocity"
)

plt.plot(
    time,
    down_velocity,
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

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 18. STATIONARY DETECTION GRAPH
# ============================================================

plt.figure(figsize=(14, 7))

plt.plot(
    time,
    acc_magnitude,
    label="Acceleration magnitude"
)

plt.plot(
    time,
    gyro_aligned,
    label="Gyroscope magnitude"
)

plt.axhline(
    ACC_THRESHOLD,
    linestyle="--",
    label="Acceleration threshold"
)

plt.axhline(
    GYRO_THRESHOLD,
    linestyle="--",
    label="Gyroscope threshold"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Magnitude")

plt.title(
    "ZUPT Stationary Detection"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 19. STATIONARY REGIONS
# ============================================================

plt.figure(figsize=(14, 4))

plt.plot(
    time,
    stationary.astype(int)
)

plt.xlabel("Time (seconds)")

plt.ylabel(
    "Stationary\n0 = Moving, 1 = Stationary"
)

plt.title(
    "ZUPT Detection Result"
)

plt.grid(True)

plt.tight_layout()

plt.show()


print("\n==========================================")
print("STEP 10 IMPROVED ZUPT COMPLETE")
print("==========================================")