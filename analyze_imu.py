import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# LOAD IMU DATA
# ==============================

filename = "imu_data_1788285144311.csv"

data = pd.read_csv(filename)

print("\n========== IMU DATA ==========\n")

print("First 5 samples:")
print(data.head())

print("\nNumber of samples:", len(data))

print("\nColumns:")
print(data.columns.tolist())


# ==============================
# TIMESTAMP ANALYSIS
# ==============================

time = data["time_seconds"].to_numpy()

dt = np.diff(time)

print("\n========== TIMING ==========\n")

print("Average time interval:",
      np.mean(dt), "seconds")

print("Minimum time interval:",
      np.min(dt), "seconds")

print("Maximum time interval:",
      np.max(dt), "seconds")

print("Approximate sampling rate:",
      1 / np.mean(dt), "Hz")


# ==============================
# ACCELEROMETER
# ==============================

ax = data["acc_x"].to_numpy()
ay = data["acc_y"].to_numpy()
az = data["acc_z"].to_numpy()

acc_magnitude = np.sqrt(
    ax**2 + ay**2 + az**2
)

print("\n========== ACCELEROMETER ==========\n")

print("Ax mean:", np.mean(ax))
print("Ay mean:", np.mean(ay))
print("Az mean:", np.mean(az))

print("\nAcceleration magnitude:")
print("Mean:", np.mean(acc_magnitude))
print("Minimum:", np.min(acc_magnitude))
print("Maximum:", np.max(acc_magnitude))


# ==============================
# GYROSCOPE
# ==============================

gx = data["gyro_x"].to_numpy()
gy = data["gyro_y"].to_numpy()
gz = data["gyro_z"].to_numpy()

gyro_magnitude = np.sqrt(
    gx**2 + gy**2 + gz**2
)

print("\n========== GYROSCOPE ==========\n")

print("Gx mean:", np.mean(gx))
print("Gy mean:", np.mean(gy))
print("Gz mean:", np.mean(gz))

print("\nGyroscope magnitude:")
print("Mean:", np.mean(gyro_magnitude))
print("Maximum:", np.max(gyro_magnitude))


# ==============================
# ACCELEROMETER GRAPH
# ==============================

plt.figure()

plt.plot(time, ax, label="Ax")
plt.plot(time, ay, label="Ay")
plt.plot(time, az, label="Az")

plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Raw Accelerometer Data")

plt.legend()
plt.grid()

plt.show()


# ==============================
# GYROSCOPE GRAPH
# ==============================

plt.figure()

plt.plot(time, gx, label="Gx")
plt.plot(time, gy, label="Gy")
plt.plot(time, gz, label="Gz")

plt.xlabel("Time (seconds)")
plt.ylabel("Angular velocity (rad/s)")
plt.title("Raw Gyroscope Data")

plt.legend()
plt.grid()

plt.show()


# ==============================
# ACCELERATION MAGNITUDE
# ==============================

plt.figure()

plt.plot(
    time,
    acc_magnitude,
    label="Acceleration magnitude"
)

plt.axhline(
    9.81,
    linestyle="--",
    label="Gravity = 9.81 m/s²"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Acceleration Magnitude")

plt.legend()
plt.grid()

plt.show()

print("\n========== ANALYSIS COMPLETE ==========\n")