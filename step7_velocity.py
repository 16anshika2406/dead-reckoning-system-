import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STEP 7 - ACCELERATION TO VELOCITY
# Improved version with bias correction and ZUPT
# ============================================================

input_file = "filtered_vehicle_acceleration.csv"
output_file = "vehicle_velocity.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

data = pd.read_csv(input_file)

time = data["time_seconds"].to_numpy()

forward_acc = data["forward_acceleration"].to_numpy()
right_acc = data["right_acceleration"].to_numpy()
down_acc = data["down_acceleration"].to_numpy()

print("CSV loaded successfully")
print("Samples:", len(time))


# ============================================================
# 2. CALCULATE ACCELERATION MAGNITUDE
# ============================================================

acc_magnitude = np.sqrt(
    forward_acc**2 +
    right_acc**2 +
    down_acc**2
)


# ============================================================
# 3. DETECT STATIONARY SAMPLES
# ============================================================

# When acceleration is close to zero,
# assume vehicle is stationary.

stationary_threshold = 0.25

stationary = acc_magnitude < stationary_threshold


# ============================================================
# 4. ESTIMATE SENSOR BIAS
# ============================================================

# Use stationary samples to estimate remaining
# acceleration bias.

if np.sum(stationary) > 10:

    forward_bias = np.mean(
        forward_acc[stationary]
    )

    right_bias = np.mean(
        right_acc[stationary]
    )

    down_bias = np.mean(
        down_acc[stationary]
    )

else:

    forward_bias = 0.0
    right_bias = 0.0
    down_bias = 0.0


print()
print("Estimated acceleration bias:")

print("Forward bias =", forward_bias)
print("Right bias   =", right_bias)
print("Down bias    =", down_bias)


# ============================================================
# 5. REMOVE BIAS
# ============================================================

forward_acc = forward_acc - forward_bias
right_acc = right_acc - right_bias
down_acc = down_acc - down_bias


# ============================================================
# 6. VELOCITY ARRAYS
# ============================================================

forward_velocity = np.zeros(len(time))
right_velocity = np.zeros(len(time))
down_velocity = np.zeros(len(time))


# ============================================================
# 7. INTEGRATE ACCELERATION
# ============================================================

for i in range(1, len(time)):

    dt = time[i] - time[i - 1]

    # Protect against invalid timestamps
    if dt <= 0 or dt > 0.2:
        dt = 0.01

    # Trapezoidal integration
    forward_velocity[i] = (
        forward_velocity[i - 1]
        + 0.5 *
        (forward_acc[i] + forward_acc[i - 1])
        * dt
    )

    right_velocity[i] = (
        right_velocity[i - 1]
        + 0.5 *
        (right_acc[i] + right_acc[i - 1])
        * dt
    )

    down_velocity[i] = (
        down_velocity[i - 1]
        + 0.5 *
        (down_acc[i] + down_acc[i - 1])
        * dt
    )


# ============================================================
# 8. ZERO VELOCITY UPDATE
# ============================================================

# If the vehicle is stationary,
# its velocity must be zero.

for i in range(len(time)):

    if stationary[i]:

        forward_velocity[i] = 0.0
        right_velocity[i] = 0.0
        down_velocity[i] = 0.0


# ============================================================
# 9. SAVE
# ============================================================

result = pd.DataFrame({

    "time_seconds": time,

    "forward_velocity": forward_velocity,

    "right_velocity": right_velocity,

    "down_velocity": down_velocity
})

result.to_csv(
    output_file,
    index=False
)


# ============================================================
# 10. RESULTS
# ============================================================

print()
print("========================================")
print("VELOCITY RESULTS")
print("========================================")

print()
print("FORWARD:")
print(
    "Maximum =",
    np.max(np.abs(forward_velocity)),
    "m/s"
)

print()
print("RIGHT:")
print(
    "Maximum =",
    np.max(np.abs(right_velocity)),
    "m/s"
)

print()
print("DOWN:")
print(
    "Maximum =",
    np.max(np.abs(down_velocity)),
    "m/s"
)

print()
print("Stationary samples:")
print(np.sum(stationary))

print()
print("Saved:")
print(output_file)


# ============================================================
# 11. PLOT
# ============================================================

plt.figure(figsize=(12, 6))

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
    "Vehicle Velocity - Bias Corrected + ZUPT"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()


print()
print("========================================")
print("STEP 7 COMPLETE")
print("========================================")