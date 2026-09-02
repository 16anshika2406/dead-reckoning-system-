import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


# ============================================================
# 1. LOAD CSV
# ============================================================

filename = "imu_data_1788285144311.csv"

data = pd.read_csv(filename)

time = data["time_seconds"].to_numpy()

ax = data["acc_x"].to_numpy()
ay = data["acc_y"].to_numpy()
az = data["acc_z"].to_numpy()

gx = data["gyro_x"].to_numpy()
gy = data["gyro_y"].to_numpy()
gz = data["gyro_z"].to_numpy()

N = len(time)


# ============================================================
# 2. SAMPLING TIME
# ============================================================

dt = np.diff(time)

print("==========================================")
print("COMPLEMENTARY FILTER DEAD RECKONING")
print("==========================================")

print("Samples:", N)
print("Average sampling rate:",
      1 / np.mean(dt), "Hz")


# ============================================================
# 3. GYROSCOPE BIAS
# ============================================================
#
# First 2 seconds are assumed stationary.
#

fs = 1 / np.mean(dt)

calibration_samples = min(
    int(2 * fs),
    N
)

gyro_bias = np.array([
    np.mean(gx[:calibration_samples]),
    np.mean(gy[:calibration_samples]),
    np.mean(gz[:calibration_samples])
])

print("\nGyroscope bias:")
print("X:", gyro_bias[0])
print("Y:", gyro_bias[1])
print("Z:", gyro_bias[2])


gx = gx - gyro_bias[0]
gy = gy - gyro_bias[1]
gz = gz - gyro_bias[2]


# ============================================================
# 4. INITIAL ORIENTATION
# ============================================================
#
# Phone assumption:
#
# Screen UP
# Phone TOP -> Vehicle FRONT
#
# Android:
# X = right
# Y = top
# Z = screen-out
#
# Vehicle:
# X = forward
# Y = right
# Z = down
#

initial_acc = np.array([
    np.mean(ax[:calibration_samples]),
    np.mean(ay[:calibration_samples]),
    np.mean(az[:calibration_samples])
])

initial_acc = initial_acc / np.linalg.norm(initial_acc)

print("\nInitial acceleration direction:")
print(initial_acc)


# ============================================================
# 5. COMPLEMENTARY FILTER
# ============================================================
#
# Gyroscope:
#   good for fast movement
#
# Accelerometer:
#   good for gravity direction
#
# alpha close to 1 means more gyro.
#

alpha = 0.98


# Rotation from PHONE frame to WORLD/VEHICLE frame
orientation = Rotation.identity()


# Store orientation
forward_acc = np.zeros(N)
right_acc = np.zeros(N)
down_acc = np.zeros(N)

gravity_world = np.zeros((N, 3))


# ============================================================
# 6. INITIAL GRAVITY
# ============================================================

gravity_phone = initial_acc * 9.81


# ============================================================
# 7. ORIENTATION LOOP
# ============================================================

for i in range(N):

    if i == 0:
        dt_i = np.mean(dt)
    else:
        dt_i = time[i] - time[i - 1]

    # ------------------------------------------
    # Gyroscope rotation
    # ------------------------------------------

    gyro = np.array([
        gx[i],
        gy[i],
        gz[i]
    ])

    rotation_delta = Rotation.from_rotvec(
        gyro * dt_i
    )

    orientation = orientation * rotation_delta


    # ------------------------------------------
    # Accelerometer gravity direction
    # ------------------------------------------

    acc = np.array([
        ax[i],
        ay[i],
        az[i]
    ])

    acc_norm = np.linalg.norm(acc)

    if acc_norm > 0:

        acc_direction = acc / acc_norm

        # Current estimated gravity direction
        estimated_gravity_phone = orientation.inv().apply(
            np.array([0, 0, 1])
        )

        # Cross product gives correction axis
        correction_axis = np.cross(
            estimated_gravity_phone,
            acc_direction
        )

        correction_magnitude = np.linalg.norm(
            correction_axis
        )

        if correction_magnitude > 1e-8:

            correction_axis = (
                correction_axis /
                correction_magnitude
            )

            angle = np.arcsin(
                min(
                    correction_magnitude,
                    1.0
                )
            )

            # Accelerometer correction
            correction = Rotation.from_rotvec(
                correction_axis *
                angle *
                (1 - alpha)
            )

            orientation = (
                orientation *
                correction
            )


    # ========================================================
    # 8. REMOVE GRAVITY
    # ========================================================

    # Convert acceleration to vehicle/world frame

    acc_world = orientation.apply(acc)

    # Gravity points upward in this coordinate convention
    # because Android accelerometer reports +g when flat.

    linear_acc_world = (
        acc_world -
        np.array([0, 0, 9.81])
    )


    # ========================================================
    # 9. STORE VEHICLE ACCELERATION
    # ========================================================

    #
    # Vehicle:
    # X = FORWARD
    # Y = RIGHT
    # Z = DOWN
    #

    forward_acc[i] = linear_acc_world[1]

    right_acc[i] = linear_acc_world[0]

    down_acc[i] = -linear_acc_world[2]


# ============================================================
# 10. REMOVE SMALL SENSOR NOISE
# ============================================================

# Deadband around zero.
#
# This prevents tiny sensor noise from becoming
# large velocity errors later.

deadband = 0.08

forward_acc[
    np.abs(forward_acc) < deadband
] = 0

right_acc[
    np.abs(right_acc) < deadband
] = 0

down_acc[
    np.abs(down_acc) < deadband
] = 0


# ============================================================
# 11. PRINT RESULTS
# ============================================================

print("\n==========================================")
print("FINAL VEHICLE FRAME")
print("==========================================")

print("\nFORWARD:")
print("Mean =", np.mean(forward_acc))
print("Maximum =", np.max(np.abs(forward_acc)))

print("\nRIGHT:")
print("Mean =", np.mean(right_acc))
print("Maximum =", np.max(np.abs(right_acc)))

print("\nDOWN:")
print("Mean =", np.mean(down_acc))
print("Maximum =", np.max(np.abs(down_acc)))


# ============================================================
# 12. SAVE DATA
# ============================================================

output = pd.DataFrame({

    "time_seconds": time,

    "forward_acceleration": forward_acc,

    "right_acceleration": right_acc,

    "down_acceleration": down_acc

})

output.to_csv(
    "vehicle_frame_acceleration.csv",
    index=False
)

print("\nSaved:")
print("vehicle_frame_acceleration.csv")


# ============================================================
# 13. GRAPH — VEHICLE ACCELERATION
# ============================================================

plt.figure()

plt.plot(
    time,
    forward_acc,
    label="Forward"
)

plt.plot(
    time,
    right_acc,
    label="Right"
)

plt.plot(
    time,
    down_acc,
    label="Down"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (m/s²)")

plt.title(
    "Vehicle Frame Acceleration - Complementary Filter"
)

plt.legend()
plt.grid()

plt.show()


# ============================================================
# 14. FORWARD ACCELERATION
# ============================================================

plt.figure()

plt.plot(
    time,
    forward_acc
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Forward acceleration (m/s²)")

plt.title(
    "Vehicle Forward Acceleration"
)

plt.grid()

plt.show()


print("\n==========================================")
print("STEP 3 COMPLETE")
print("==========================================")