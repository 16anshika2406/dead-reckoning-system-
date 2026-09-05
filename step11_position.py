import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# STEP 11
# VEHICLE POSITION FROM ZUPT + NHC VELOCITY
# ============================================================

# ------------------------------------------------------------
# CHANGE THIS TO YOUR ACTUAL INPUT CSV
# ------------------------------------------------------------

CSV_FILE = r"E:\SIH\DeadReckoning\step11_nhc_velocity.csv"

# Output file MUST be different from input file
OUTPUT_FILE = r"E:\SIH\DeadReckoning\step11_position.csv"


# ============================================================
# 1. CHECK INPUT FILE
# ============================================================

if not os.path.isfile(CSV_FILE):
    print("ERROR: CSV file not found!")
    print(CSV_FILE)
    raise SystemExit

print("Input CSV:")
print(CSV_FILE)


# ============================================================
# 2. LOAD CSV
# ============================================================

try:
    data = pd.read_csv(CSV_FILE)

except PermissionError:
    print("\nERROR: Permission denied.")
    print("Close the CSV file in Excel/Notepad/another program")
    print("and run the code again.")
    raise SystemExit

except Exception as e:
    print("\nERROR while reading CSV:")
    print(e)
    raise SystemExit


print("\nCSV loaded successfully!")


# ============================================================
# 3. SHOW COLUMNS
# ============================================================

print("\nColumns found:")
print(list(data.columns))


# ============================================================
# 4. REQUIRED COLUMNS
# ============================================================

required = [
    "time_seconds",
    "vehicle_forward",
    "vehicle_right",
    "vehicle_down"
]

missing = []

for column in required:
    if column not in data.columns:
        missing.append(column)


if len(missing) > 0:

    print("\nERROR: Missing columns:")
    print(missing)

    print("\nYour CSV contains:")
    print(list(data.columns))

    raise SystemExit


# ============================================================
# 5. EXTRACT DATA
# ============================================================

time = data["time_seconds"].astype(float).to_numpy()

forward_velocity = (
    data["vehicle_forward"]
    .astype(float)
    .to_numpy()
)

right_velocity = (
    data["vehicle_right"]
    .astype(float)
    .to_numpy()
)

down_velocity = (
    data["vehicle_down"]
    .astype(float)
    .to_numpy()
)


# ============================================================
# 6. REMOVE INVALID DATA
# ============================================================

valid = (
    np.isfinite(time)
    & np.isfinite(forward_velocity)
    & np.isfinite(right_velocity)
    & np.isfinite(down_velocity)
)

time = time[valid]

forward_velocity = forward_velocity[valid]
right_velocity = right_velocity[valid]
down_velocity = down_velocity[valid]


# ============================================================
# 7. SORT BY TIME
# ============================================================

order = np.argsort(time)

time = time[order]

forward_velocity = forward_velocity[order]
right_velocity = right_velocity[order]
down_velocity = down_velocity[order]


# ============================================================
# 8. REMOVE DUPLICATE TIME VALUES
# ============================================================

unique_time, unique_indices = np.unique(
    time,
    return_index=True
)

time = unique_time

forward_velocity = forward_velocity[unique_indices]
right_velocity = right_velocity[unique_indices]
down_velocity = down_velocity[unique_indices]


# ============================================================
# 9. INITIALIZE POSITION
# ============================================================

forward_position = np.zeros(len(time))
right_position = np.zeros(len(time))
down_position = np.zeros(len(time))


# ============================================================
# 10. VELOCITY → POSITION
#
# Trapezoidal integration:
#
# p(k) = p(k-1)
#        + 0.5 * (v(k-1) + v(k)) * dt
# ============================================================

for i in range(1, len(time)):

    dt = time[i] - time[i - 1]

    # Ignore abnormal time gaps
    if dt <= 0:
        continue

    if dt > 1.0:
        print(
            "Large time gap at:",
            time[i],
            "dt =",
            dt
        )

        forward_position[i] = forward_position[i - 1]
        right_position[i] = right_position[i - 1]
        down_position[i] = down_position[i - 1]

        continue


    # Forward position
    forward_position[i] = (
        forward_position[i - 1]
        + 0.5
        * (
            forward_velocity[i - 1]
            + forward_velocity[i]
        )
        * dt
    )


    # Right position
    right_position[i] = (
        right_position[i - 1]
        + 0.5
        * (
            right_velocity[i - 1]
            + right_velocity[i]
        )
        * dt
    )


    # Down position
    down_position[i] = (
        down_position[i - 1]
        + 0.5
        * (
            down_velocity[i - 1]
            + down_velocity[i]
        )
        * dt
    )


# ============================================================
# 11. SAVE RESULT
# ============================================================

result = pd.DataFrame({

    "time_seconds": time,

    "vehicle_forward": forward_velocity,
    "vehicle_right": right_velocity,
    "vehicle_down": down_velocity,

    "forward_position": forward_position,
    "right_position": right_position,
    "down_position": down_position
})


try:

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nPosition CSV saved successfully:")
    print(OUTPUT_FILE)

except PermissionError:

    print("\nERROR: Cannot save output CSV.")
    print("Close step11_position.csv if it is open.")
    raise SystemExit


# ============================================================
# 12. FINAL POSITION
# ============================================================

print("\n===================================")
print("STEP 11 FINAL POSITION")
print("===================================")

print(
    f"Forward position : "
    f"{forward_position[-1]:.4f} m"
)

print(
    f"Right position   : "
    f"{right_position[-1]:.4f} m"
)

print(
    f"Down position    : "
    f"{down_position[-1]:.4f} m"
)


# ============================================================
# 13. POSITION VS TIME
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    time,
    forward_position,
    label="Forward position"
)

plt.plot(
    time,
    right_position,
    label="Right position"
)

plt.plot(
    time,
    down_position,
    label="Down position"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Position (m)")

plt.title(
    "Step 11 - Vehicle Position from ZUPT + NHC"
)

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 14. 2D VEHICLE TRAJECTORY
# ============================================================

plt.figure(figsize=(10, 8))

plt.plot(
    right_position,
    forward_position,
    label="Vehicle trajectory"
)

# START
plt.scatter(
    right_position[0],
    forward_position[0],
    s=100,
    label="Start"
)

# END
plt.scatter(
    right_position[-1],
    forward_position[-1],
    s=100,
    marker="X",
    label="End"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Right position (m)"
)

plt.ylabel(
    "Forward position (m)"
)

plt.title(
    "Step 11 - Vehicle 2D Trajectory"
)

plt.grid(True)

plt.axis("equal")

plt.legend()

plt.tight_layout()

plt.show()


print("\n===================================")
print("STEP 11 COMPLETE")
print("===================================")