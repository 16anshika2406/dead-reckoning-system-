import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# STEP 11
# ZUPT + NON-HOLONOMIC CONSTRAINTS (NHC)
# ============================================================


# ============================================================
# 1. FILE PATHS
# ============================================================

VEHICLE_FILE = r"E:\SIH\DeadReckoning\vehicle_frame_acceleration.csv"

IMU_FILE = r"E:\SIH\DeadReckoning\imu_data_1788285144311.csv"

OUTPUT_FILE = r"E:\SIH\DeadReckoning\step11_nhc_velocity.csv"


# ============================================================
# 2. LOAD CSV FILES
# ============================================================

vehicle = pd.read_csv(VEHICLE_FILE)
imu = pd.read_csv(IMU_FILE)

print("==========================================")
print("STEP 11 - NHC + ZUPT")
print("==========================================")

print("\nVehicle CSV columns:")
print(vehicle.columns.tolist())

print("\nIMU CSV columns:")
print(imu.columns.tolist())


# ============================================================
# 3. CHECK VEHICLE COLUMNS
# ============================================================

required_vehicle = [
    "time_seconds",
    "vehicle_forward",
    "vehicle_right",
    "vehicle_down"
]

for col in required_vehicle:

    if col not in vehicle.columns:

        raise ValueError(
            "Missing vehicle column: " + col
        )


# ============================================================
# 4. FIND GYROSCOPE COLUMNS
# ============================================================

# Your Android file should contain gyro_x, gyro_y, gyro_z.
# This section also checks alternative names.

gyro_x_name = None
gyro_y_name = None
gyro_z_name = None


possible_x = [
    "gyro_x",
    "gyroscope_x",
    "gyroX"
]

possible_y = [
    "gyro_y",
    "gyroscope_y",
    "gyroY"
]

possible_z = [
    "gyro_z",
    "gyroscope_z",
    "gyroZ"
]


for name in possible_x:

    if name in imu.columns:
        gyro_x_name = name
        break


for name in possible_y:

    if name in imu.columns:
        gyro_y_name = name
        break


for name in possible_z:

    if name in imu.columns:
        gyro_z_name = name
        break


if (
    gyro_x_name is None
    or gyro_y_name is None
    or gyro_z_name is None
):

    raise ValueError(
        "Could not find gyroscope columns.\n"
        "Available columns:\n"
        + str(imu.columns.tolist())
    )


print("\nGyroscope columns used:")

print(gyro_x_name)
print(gyro_y_name)
print(gyro_z_name)


# ============================================================
# 5. READ VEHICLE ACCELERATION
# ============================================================

time = vehicle[
    "time_seconds"
].to_numpy(dtype=float)

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
# 6. READ GYROSCOPE
# ============================================================

gyro_time = imu[
    "time_seconds"
].to_numpy(dtype=float)

gyro_x = imu[
    gyro_x_name
].to_numpy(dtype=float)

gyro_y = imu[
    gyro_y_name
].to_numpy(dtype=float)

gyro_z = imu[
    gyro_z_name
].to_numpy(dtype=float)


# ============================================================
# 7. CLEAN DATA
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
# 8. ACCELERATION MAGNITUDE
# ============================================================

acc_magnitude = np.sqrt(
    forward_acc ** 2
    +
    right_acc ** 2
    +
    down_acc ** 2
)


# ============================================================
# 9. GYROSCOPE MAGNITUDE
# ============================================================

gyro_magnitude = np.sqrt(
    gyro_x ** 2
    +
    gyro_y ** 2
    +
    gyro_z ** 2
)


# ============================================================
# 10. ALIGN GYRO WITH VEHICLE TIMESTAMP
# ============================================================

gyro_aligned = np.interp(
    time,
    gyro_time,
    gyro_magnitude
)


# ============================================================
# 11. PARAMETERS
# ============================================================

# Stationary detector thresholds

ACC_THRESHOLD = 0.15

GYRO_THRESHOLD = 0.05


# Number of consecutive samples required
# before declaring stationary.

MIN_STATIONARY_SAMPLES = 8


print("\n==========================================")
print("PARAMETERS")
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

print(
    "Minimum stationary samples:",
    MIN_STATIONARY_SAMPLES
)


# ============================================================
# 12. RAW STATIONARY DETECTION
# ============================================================

stationary_raw = (

    (acc_magnitude < ACC_THRESHOLD)

    &

    (gyro_aligned < GYRO_THRESHOLD)

)


# ============================================================
# 13. CONSECUTIVE-SAMPLE FILTER
# ============================================================

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

        start = (
            i
            -
            MIN_STATIONARY_SAMPLES
            +
            1
        )

        stationary[start:i + 1] = True


# ============================================================
# 14. DISPLAY STATIONARY INFORMATION
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
    int(np.sum(stationary))
)

print(
    "Moving samples:",
    int(np.sum(~stationary))
)


# ============================================================
# 15. VELOCITY ARRAYS
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
# 16. VELOCITY INTEGRATION
# ============================================================

for i in range(1, len(time)):

    dt = (
        time[i]
        -
        time[i - 1]
    )


    # Protect against bad timestamp

    if dt <= 0 or dt > 0.1:

        dt = 0.01


    # --------------------------------------------------------
    # INTEGRATE FORWARD ACCELERATION
    # --------------------------------------------------------

    forward_velocity[i] = (

        forward_velocity[i - 1]

        +

        forward_acc[i] * dt

    )


    # --------------------------------------------------------
    # INTEGRATE RIGHT ACCELERATION
    # --------------------------------------------------------

    right_velocity[i] = (

        right_velocity[i - 1]

        +

        right_acc[i] * dt

    )


    # --------------------------------------------------------
    # INTEGRATE DOWN ACCELERATION
    # --------------------------------------------------------

    down_velocity[i] = (

        down_velocity[i - 1]

        +

        down_acc[i] * dt

    )


    # ========================================================
    # NHC
    # ========================================================

    # A normal ground vehicle has approximately:
    #
    # lateral velocity  = 0
    # vertical velocity = 0
    #
    # Therefore we constrain these components.

    right_velocity[i] = 0.0

    down_velocity[i] = 0.0


    # ========================================================
    # ZUPT
    # ========================================================

    if stationary[i]:

        forward_velocity[i] = 0.0

        right_velocity[i] = 0.0

        down_velocity[i] = 0.0


# ============================================================
# 17. VELOCITY LIMIT
# ============================================================

MAX_VELOCITY = 10.0


forward_velocity = np.clip(
    forward_velocity,
    -MAX_VELOCITY,
    MAX_VELOCITY
)

right_velocity = np.clip(
    right_velocity,
    -MAX_VELOCITY,
    MAX_VELOCITY
)

down_velocity = np.clip(
    down_velocity,
    -MAX_VELOCITY,
    MAX_VELOCITY
)


# ============================================================
# 18. SAVE RESULT
# ============================================================

result = pd.DataFrame({

    "time_seconds": time,

    "vehicle_forward_acc":
        forward_acc,

    "vehicle_right_acc":
        right_acc,

    "vehicle_down_acc":
        down_acc,

    "acceleration_magnitude":
        acc_magnitude,

    "gyro_magnitude":
        gyro_aligned,

    "stationary":
        stationary,

    "forward_velocity":
        forward_velocity,

    "right_velocity":
        right_velocity,

    "down_velocity":
        down_velocity

})


result.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n==========================================")
print("CSV SAVED")
print("==========================================")

print(OUTPUT_FILE)


# ============================================================
# 19. VELOCITY GRAPH
# ============================================================

plt.figure(
    figsize=(14, 7)
)


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


plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Velocity (m/s)"
)


plt.title(
    "Vehicle Velocity - ZUPT + NHC"
)


plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 20. ACCELERATION + GYRO GRAPH
# ============================================================

plt.figure(
    figsize=(14, 7)
)


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


plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Magnitude"
)


plt.title(
    "Stationary Detection"
)


plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 21. STATIONARY / MOVING GRAPH
# ============================================================

plt.figure(
    figsize=(14, 4)
)


plt.plot(
    time,
    stationary.astype(int)
)


plt.xlabel(
    "Time (seconds)"
)


plt.ylabel(
    "Stationary"
)


plt.title(
    "ZUPT Detection"
)


plt.yticks(
    [0, 1],
    ["Moving", "Stationary"]
)


plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# COMPLETE
# ============================================================

print("\n==========================================")
print("STEP 11 COMPLETE")
print("==========================================")