import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STEP 5
# WORLD FRAME -> VEHICLE FRAME
# ============================================================

# Input created by Step 4
input_file = "world_frame_acceleration.csv"

# Output
output_file = "vehicle_frame_acceleration.csv"


# ============================================================
# 1. VEHICLE HEADING
# ============================================================

# IMPORTANT:
# This is only a prototype.
#
# 0 degrees  = vehicle forward points toward World X
# 90 degrees = vehicle forward points toward World Y
#
# Later this will come from GNSS / vehicle heading.

heading_degrees = 0.0

heading = np.deg2rad(heading_degrees)


# ============================================================
# 2. LOAD DATA
# ============================================================

data = pd.read_csv(input_file)

print("CSV loaded successfully")
print()
print("Columns:")
print(data.columns.tolist())


# ============================================================
# 3. READ WORLD FRAME ACCELERATION
# ============================================================

world_x = data["linear_acc_x"].to_numpy()
world_y = data["linear_acc_y"].to_numpy()
world_z = data["linear_acc_z"].to_numpy()

time = data["time_seconds"].to_numpy()


# ============================================================
# 4. WORLD -> VEHICLE TRANSFORMATION
# ============================================================

# Vehicle forward unit vector
forward_x = np.cos(heading)
forward_y = np.sin(heading)

# Vehicle right unit vector
right_x = -np.sin(heading)
right_y = np.cos(heading)


# Forward acceleration
vehicle_forward = (
    world_x * forward_x +
    world_y * forward_y
)


# Right acceleration
vehicle_right = (
    world_x * right_x +
    world_y * right_y
)


# Down acceleration
#
# World Z is UP.
# Therefore DOWN = -World Z.

vehicle_down = -world_z


# ============================================================
# 5. CREATE OUTPUT DATAFRAME
# ============================================================

result = pd.DataFrame({

    "time_seconds": time,

    "world_acc_x": world_x,
    "world_acc_y": world_y,
    "world_acc_z": world_z,

    "vehicle_forward": vehicle_forward,
    "vehicle_right": vehicle_right,
    "vehicle_down": vehicle_down
})


# ============================================================
# 6. SAVE
# ============================================================

result.to_csv(
    output_file,
    index=False
)


# ============================================================
# 7. PRINT RESULTS
# ============================================================

print()
print("========================================")
print("VEHICLE FRAME RESULTS")
print("========================================")

print()
print("Vehicle Forward:")
print("Mean =", np.mean(vehicle_forward))
print(
    "Maximum =",
    np.max(np.abs(vehicle_forward))
)

print()
print("Vehicle Right:")
print("Mean =", np.mean(vehicle_right))
print(
    "Maximum =",
    np.max(np.abs(vehicle_right))
)

print()
print("Vehicle Down:")
print("Mean =", np.mean(vehicle_down))
print(
    "Maximum =",
    np.max(np.abs(vehicle_down))
)


print()
print("========================================")
print("SAVED")
print("========================================")

print(output_file)


# ============================================================
# 8. PLOT VEHICLE FRAME
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    time,
    vehicle_forward,
    label="Forward"
)

plt.plot(
    time,
    vehicle_right,
    label="Right"
)

plt.plot(
    time,
    vehicle_down,
    label="Down"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (m/s²)")

plt.title(
    "Vehicle Frame Acceleration"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()