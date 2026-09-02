import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STEP 6
# FILTER VEHICLE FRAME ACCELERATION
# ============================================================

input_file = "vehicle_frame_acceleration.csv"
output_file = "filtered_vehicle_acceleration.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

data = pd.read_csv(input_file)

print("CSV loaded successfully")
print()
print("Columns:")
print(data.columns.tolist())


# ============================================================
# 2. READ DATA
# ============================================================

time = data["time_seconds"].to_numpy()

forward = data["vehicle_forward"].to_numpy()
right = data["vehicle_right"].to_numpy()
down = data["vehicle_down"].to_numpy()


# ============================================================
# 3. REMOVE EXTREME SENSOR SPIKES
# ============================================================

def remove_spikes(signal, threshold=12.0):
    """
    Replace unrealistic acceleration spikes with
    an interpolated value from neighboring samples.
    """

    filtered = signal.copy()

    spike_indices = np.where(np.abs(filtered) > threshold)[0]

    for i in spike_indices:

        if i == 0:
            filtered[i] = filtered[i + 1]

        elif i == len(filtered) - 1:
            filtered[i] = filtered[i - 1]

        else:
            filtered[i] = (
                filtered[i - 1] +
                filtered[i + 1]
            ) / 2.0

    return filtered


forward_clean = remove_spikes(forward)
right_clean = remove_spikes(right)
down_clean = remove_spikes(down)


# ============================================================
# 4. LOW-PASS FILTER
# ============================================================

def moving_average(signal, window=5):

    return (
        pd.Series(signal)
        .rolling(
            window=window,
            center=True,
            min_periods=1
        )
        .mean()
        .to_numpy()
    )


forward_filtered = moving_average(
    forward_clean,
    window=5
)

right_filtered = moving_average(
    right_clean,
    window=5
)

down_filtered = moving_average(
    down_clean,
    window=5
)


# ============================================================
# 5. SAVE FILTERED DATA
# ============================================================

result = pd.DataFrame({

    "time_seconds": time,

    "forward_acceleration": forward_filtered,

    "right_acceleration": right_filtered,

    "down_acceleration": down_filtered
})


result.to_csv(
    output_file,
    index=False
)


# ============================================================
# 6. PRINT RESULTS
# ============================================================

print()
print("========================================")
print("FILTERED VEHICLE ACCELERATION")
print("========================================")

print()
print("FORWARD:")
print(
    "Mean =",
    np.mean(forward_filtered)
)
print(
    "Maximum =",
    np.max(np.abs(forward_filtered))
)

print()
print("RIGHT:")
print(
    "Mean =",
    np.mean(right_filtered)
)
print(
    "Maximum =",
    np.max(np.abs(right_filtered))
)

print()
print("DOWN:")
print(
    "Mean =",
    np.mean(down_filtered)
)
print(
    "Maximum =",
    np.max(np.abs(down_filtered))
)

print()
print("Saved:")
print(output_file)


# ============================================================
# 7. PLOT
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    time,
    forward_filtered,
    label="Forward"
)

plt.plot(
    time,
    right_filtered,
    label="Right"
)

plt.plot(
    time,
    down_filtered,
    label="Down"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration (m/s²)")

plt.title(
    "Filtered Vehicle Frame Acceleration"
)

plt.legend()

plt.grid()

plt.tight_layout()

plt.show()


print()
print("========================================")
print("STEP 6 COMPLETE")
print("========================================")