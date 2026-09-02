import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STEP 4
# PHONE FRAME -> WORLD FRAME -> GRAVITY REMOVAL
# ============================================================

# CHANGE THIS TO YOUR NEW CSV FILE NAME
filename = "imu_data_1788285144311.csv"

# ------------------------------------------------------------
# 1. LOAD CSV
# ------------------------------------------------------------

data = pd.read_csv(filename)

print("CSV loaded successfully")
print()
print("Columns:")
print(data.columns.tolist())
print()

# ------------------------------------------------------------
# 2. READ ACCELEROMETER
# ------------------------------------------------------------

ax = data["acc_x"].to_numpy()
ay = data["acc_y"].to_numpy()
az = data["acc_z"].to_numpy()

# ------------------------------------------------------------
# 3. READ QUATERNION
# ------------------------------------------------------------

qw = data["quat_w"].to_numpy()
qx = data["quat_x"].to_numpy()
qy = data["quat_y"].to_numpy()
qz = data["quat_z"].to_numpy()

# ------------------------------------------------------------
# 4. NORMALIZE QUATERNION
# ------------------------------------------------------------

q_norm = np.sqrt(
    qw**2 +
    qx**2 +
    qy**2 +
    qz**2
)

qw = qw / q_norm
qx = qx / q_norm
qy = qy / q_norm
qz = qz / q_norm

# ------------------------------------------------------------
# 5. PHONE FRAME -> WORLD FRAME
#
# Android rotation quaternion is used to construct
# the rotation matrix.
# ------------------------------------------------------------

world_x = np.zeros(len(data))
world_y = np.zeros(len(data))
world_z = np.zeros(len(data))

for i in range(len(data)):

    w = qw[i]
    x = qx[i]
    y = qy[i]
    z = qz[i]

    # Rotation matrix
    R = np.array([

        [
            1 - 2*(y*y + z*z),
            2*(x*y - z*w),
            2*(x*z + y*w)
        ],

        [
            2*(x*y + z*w),
            1 - 2*(x*x + z*z),
            2*(y*z - x*w)
        ],

        [
            2*(x*z - y*w),
            2*(y*z + x*w),
            1 - 2*(x*x + y*y)
        ]

    ])

    # Acceleration in PHONE frame
    phone_acc = np.array([
        ax[i],
        ay[i],
        az[i]
    ])

    # Convert PHONE -> WORLD
    world_acc = R @ phone_acc

    world_x[i] = world_acc[0]
    world_y[i] = world_acc[1]
    world_z[i] = world_acc[2]


# ------------------------------------------------------------
# 6. REMOVE GRAVITY
#
# World Z is vertical.
# Android accelerometer includes gravity.
# ------------------------------------------------------------

g = 9.80665

linear_x = world_x
linear_y = world_y
linear_z = world_z - g


# ------------------------------------------------------------
# 7. SAVE RESULT
# ------------------------------------------------------------

result = pd.DataFrame({

    "time_seconds": data["time_seconds"],

    "world_acc_x": world_x,
    "world_acc_y": world_y,
    "world_acc_z": world_z,

    "linear_acc_x": linear_x,
    "linear_acc_y": linear_y,
    "linear_acc_z": linear_z

})

result.to_csv(
    "world_frame_acceleration.csv",
    index=False
)


# ------------------------------------------------------------
# 8. PRINT RESULTS
# ------------------------------------------------------------

print("========================================")
print("WORLD FRAME ACCELERATION")
print("========================================")

print()
print("World X:")
print("Mean =", np.mean(world_x))
print("Maximum =", np.max(np.abs(world_x)))

print()
print("World Y:")
print("Mean =", np.mean(world_y))
print("Maximum =", np.max(np.abs(world_y)))

print()
print("World Z:")
print("Mean =", np.mean(world_z))
print("Maximum =", np.max(np.abs(world_z)))


print()
print("========================================")
print("LINEAR ACCELERATION")
print("========================================")

print()
print("Linear X:")
print("Mean =", np.mean(linear_x))
print("Maximum =", np.max(np.abs(linear_x)))

print()
print("Linear Y:")
print("Mean =", np.mean(linear_y))
print("Maximum =", np.max(np.abs(linear_y)))

print()
print("Linear Z:")
print("Mean =", np.mean(linear_z))
print("Maximum =", np.max(np.abs(linear_z)))


print()
print("========================================")
print("STEP 4 COMPLETE")
print("========================================")

print()
print("Saved:")
print("world_frame_acceleration.csv")


# ------------------------------------------------------------
# 9. PLOT LINEAR ACCELERATION
# ------------------------------------------------------------

time = data["time_seconds"]

plt.figure(figsize=(12, 6))

plt.plot(
    time,
    linear_x,
    label="World X"
)

plt.plot(
    time,
    linear_y,
    label="World Y"
)

plt.plot(
    time,
    linear_z,
    label="World Z"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Linear acceleration (m/s²)")

plt.title(
    "World Frame Linear Acceleration"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()