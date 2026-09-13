import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from dr_pipeline import run_pipeline

result, d, acc_mag, gyro_mag = run_pipeline("imu_data_1788285144311.csv")

# ------------------------------------------------------------
# PLOT 1: trajectory comparison (old buggy vs fixed)
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

axes[0].plot(result["old_buggy_world_x"], result["old_buggy_world_y"],
             color="crimson", linewidth=2)
axes[0].scatter(result["old_buggy_world_x"].iloc[0], result["old_buggy_world_y"].iloc[0],
                color="green", s=80, zorder=5, label="Start")
axes[0].scatter(result["old_buggy_world_x"].iloc[-1], result["old_buggy_world_y"].iloc[-1],
                color="black", s=80, marker="X", zorder=5, label="End")
axes[0].set_title("OLD PIPELINE (fixed heading=0)\n\u2192 can only draw a straight line")
axes[0].set_xlabel("X (m)")
axes[0].set_ylabel("Y (m)")
axes[0].axhline(0, linestyle="--", color="gray", linewidth=0.8)
axes[0].axvline(0, linestyle="--", color="gray", linewidth=0.8)
axes[0].axis("equal")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(result["world_x"], result["world_y"], color="royalblue", linewidth=2)
axes[1].scatter(result["world_x"].iloc[0], result["world_y"].iloc[0],
                color="green", s=80, zorder=5, label="Start")
axes[1].scatter(result["world_x"].iloc[-1], result["world_y"].iloc[-1],
                color="black", s=80, marker="X", zorder=5, label="End")
axes[1].set_title("FIXED PIPELINE (real-time heading)\n\u2192 trajectory actually curves")
axes[1].set_xlabel("X (m)")
axes[1].set_ylabel("Y (m)")
axes[1].axhline(0, linestyle="--", color="gray", linewidth=0.8)
axes[1].axvline(0, linestyle="--", color="gray", linewidth=0.8)
axes[1].axis("equal")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig("trajectory_comparison.png", dpi=150)
print("Saved trajectory_comparison.png")

# ------------------------------------------------------------
# PLOT 2: heading over time (proves we're tracking real orientation change)
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(result["time"], result["heading_deg"], color="darkorange", linewidth=1.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Heading (degrees)")
ax.set_title("Tracked Heading Over Time (this is what the old code hardcoded to 0)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("heading_over_time.png", dpi=150)
print("Saved heading_over_time.png")

# ------------------------------------------------------------
# PLOT 3: ZUPT stationary detection sanity check
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(d["t"], acc_mag, label="Linear acc magnitude (body frame)", linewidth=1)
ax.plot(d["t"], gyro_mag, label="Gyro magnitude", linewidth=1)
ax.fill_between(d["t"], 0, 1, where=result["stationary"], color="gray", alpha=0.25,
                transform=ax.get_xaxis_transform(), label="Detected stationary")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Magnitude")
ax.set_title("ZUPT Stationary Detection")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("zupt_detection.png", dpi=150)
print("Saved zupt_detection.png")
