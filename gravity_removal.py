import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ==========================================
# 1. LOAD CSV
# ==========================================

filename = "imu_data_1788285144311.csv"

data = pd.read_csv(filename)

time = data["time_seconds"].to_numpy()

ax = data["acc_x"].to_numpy()
ay = data["acc_y"].to_numpy()
az = data["acc_z"].to_numpy()


# ==========================================
# 2. CALCULATE SAMPLING RATE
# ==========================================

dt = np.mean(np.diff(time))
fs = 1.0 / dt

print("Sampling rate:", fs, "Hz")


# ==========================================
# 3. ESTIMATE GRAVITY
# ==========================================

# Gravity changes slowly.
# Motion changes faster.
#
# Therefore, use a low-pass filter
# to estimate the gravity component.

cutoff = 0.5  # Hz

b, a = butter(
    2,
    cutoff / (fs / 2),
    btype="low"
)

gravity_x = filtfilt(b, a, ax)
gravity_y = filtfilt(b, a, ay)
gravity_z = filtfilt(b, a, az)


# ==========================================
# 4. REMOVE GRAVITY
# ==========================================

linear_x = ax - gravity_x
linear_y = ay - gravity_y
linear_z = az - gravity_z


# ==========================================
# 5. LINEAR ACCELERATION MAGNITUDE
# ==========================================

linear_magnitude = np.sqrt(
    linear_x**2 +
    linear_y**2 +
    linear_z**2
)


# ==========================================
# 6. PRINT RESULTS
# ==========================================

print("\n==============================")
print("GRAVITY REMOVAL RESULTS")
print("==============================")

print("\nEstimated gravity:")

print("Gravity X mean:",
      np.mean(gravity_x))

print("Gravity Y mean:",
      np.mean(gravity_y))

print("Gravity Z mean:",
      np.mean(gravity_z))


print("\nLinear acceleration:")

print("Linear X mean:",
      np.mean(linear_x))

print("Linear Y mean:",
      np.mean(linear_y))

print("Linear Z mean:",
      np.mean(linear_z))


print("\nLinear acceleration magnitude:")

print("Mean:",
      np.mean(linear_magnitude))

print("Maximum:",
      np.max(linear_magnitude))


# ==========================================
# 7. GRAPH — ESTIMATED GRAVITY
# ==========================================

plt.figure()

plt.plot(
    time,
    gravity_x,
    label="Gravity X"
)

plt.plot(
    time,
    gravity_y,
    label="Gravity Y"
)

plt.plot(
    time,
    gravity_z,
    label="Gravity Z"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Estimated Gravity")

plt.legend()
plt.grid()

plt.show()


# ==========================================
# 8. GRAPH — LINEAR ACCELERATION
# ==========================================

plt.figure()

plt.plot(
    time,
    linear_x,
    label="Linear X"
)

plt.plot(
    time,
    linear_y,
    label="Linear Y"
)

plt.plot(
    time,
    linear_z,
    label="Linear Z"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Acceleration After Gravity Removal")

plt.legend()
plt.grid()

plt.show()


# ==========================================
# 9. GRAPH — LINEAR ACCELERATION MAGNITUDE
# ==========================================

plt.figure()

plt.plot(
    time,
    linear_magnitude,
    label="Linear Acceleration"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Linear Acceleration Magnitude")

plt.legend()
plt.grid()

plt.show()


print("\n==============================")
print("STEP 2 COMPLETE")
print("==============================")