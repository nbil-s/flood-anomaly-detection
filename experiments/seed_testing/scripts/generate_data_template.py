
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# DATASET CONFIGURATION
# ============================================================

RANDOM_SEED = 42
NUM_RECORDS = 5000
INTERVAL_SECONDS = 30

START_TIME = "2026-01-01 00:00:00"

OUTPUT_PATH = Path(
    "experiments/improved_simulation_24/data/raw/"
    "simulated_environmental_data.csv"
)

FEATURES = [
    "water_level",
    "rainfall",
    "temperature",
    "humidity",
]

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# TIMESTAMPS
# ============================================================

timestamps = pd.date_range(
    start=START_TIME,
    periods=NUM_RECORDS,
    freq=f"{INTERVAL_SECONDS}s",
)

time_seconds = np.arange(NUM_RECORDS) * INTERVAL_SECONDS
time_in_days = time_seconds / 86400


# ============================================================
# RAINFALL EPISODES
# ============================================================

rainfall = np.zeros(NUM_RECORDS)

RAIN_START_PROBABILITY = 0.0012
RAIN_CONTINUE_PROBABILITY = 0.90
RAIN_SHAPE = 1.8
RAIN_SCALE = 0.20

is_raining = False

for i in range(NUM_RECORDS):

    if is_raining:
        if rng.random() >= RAIN_CONTINUE_PROBABILITY:
            is_raining = False
    else:
        if rng.random() < RAIN_START_PROBABILITY:
            is_raining = True

    if is_raining:
        intensity = rng.gamma(
            shape=RAIN_SHAPE,
            scale=RAIN_SCALE,
        )

        if rng.random() < 0.05:
            intensity *= rng.uniform(1.5, 3.0)

        rainfall[i] = intensity


# ============================================================
# TEMPERATURE
# ============================================================


temperature = (
    25
    + 4.0 * np.sin(
        2 * np.pi * time_in_days - np.pi / 2
    )
    + 0.5 * np.sin(
        2 * np.pi * time_in_days / 7
    )
    + rng.normal(0, 0.35, NUM_RECORDS)
)


# ============================================================
# HUMIDITY
# ============================================================


humidity = (
    68
    - 0.9 * (temperature - 25)
    + 2.5 * np.minimum(rainfall, 3)
    + rng.normal(0, 1.2, NUM_RECORDS)
)

humidity = np.clip(humidity, 30, 100)


# ============================================================
# DELAYED RUNOFF AND WATER-LEVEL RESPONSE
# ============================================================


# ============================================================
# DELAYED RUNOFF AND WATER-LEVEL RESPONSE
# ============================================================

water_level = np.zeros(NUM_RECORDS)
runoff = np.zeros(NUM_RECORDS)

BASE_WATER_LEVEL = 30.0

RUNOFF_RETENTION = 0.995
RUNOFF_RESPONSE = 0.65
WATER_LEVEL_NOISE = 0.025

# Distribute each rainfall observation across
RAINFALL_ROUTING_WEIGHTS = np.array([
    0.03,
    0.07,
    0.12,
    0.18,
    0.22,
    0.18,
    0.12,
    0.08,
])

# immediately enter the runoff calculation.
routing_kernel = np.concatenate((
    [0.0],
    RAINFALL_ROUTING_WEIGHTS,
))

routed_rainfall = np.convolve(
    rainfall,
    routing_kernel,
    mode="full",
)[:NUM_RECORDS]

water_level[0] = BASE_WATER_LEVEL

for i in range(1, NUM_RECORDS):

    # Accumulate delayed rainfall into runoff.
    runoff[i] = (
        RUNOFF_RETENTION * runoff[i - 1]
        + routed_rainfall[i]
    )

    expected_level = (
        BASE_WATER_LEVEL
        + RUNOFF_RESPONSE * runoff[i]
    )

    noise = (
        0.90 * (
            water_level[i - 1]
            - (
                BASE_WATER_LEVEL
                + RUNOFF_RESPONSE * runoff[i - 1]
            )
        )
        + rng.normal(0, WATER_LEVEL_NOISE)
    )

    water_level[i] = expected_level + noise

water_level = np.clip(water_level, 5, 150)

# ============================================================
# BUILD DATAFRAME
# ============================================================

df = pd.DataFrame({
    "timestamp": timestamps,
    "water_level": water_level,
    "rainfall": rainfall,
    "temperature": temperature,
    "humidity": humidity,
})


# ============================================================
# VALIDATION
# ============================================================

if len(df) != NUM_RECORDS:
    raise ValueError("Unexpected number of generated records.")

if df[FEATURES].isna().any().any():
    raise ValueError("Missing values detected.")

if not np.isfinite(df[FEATURES].to_numpy()).all():
    raise ValueError("Non-finite values detected.")

if (df["rainfall"] < 0).any():
    raise ValueError("Negative rainfall detected.")

intervals = df["timestamp"].diff().dropna()

if not (intervals == pd.Timedelta(seconds=INTERVAL_SECONDS)).all():
    raise ValueError("Incorrect timestamp intervals.")


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(OUTPUT_PATH, index=False)


print("\nIMPROVED SYNTHETIC DATASET GENERATED")
print("-----------------------------------")

print(f"Records: {len(df)}")
print(f"Sampling interval: {INTERVAL_SECONDS} seconds")
print(f"Start: {df['timestamp'].iloc[0]}")
print(f"End: {df['timestamp'].iloc[-1]}")

print(f"\nSaved to: {OUTPUT_PATH}")

print("\nFeature statistics:")
print(df[FEATURES].describe().round(4))

wet_readings = int((df["rainfall"] > 0).sum())

print("\nRainfall summary:")
print(f"Rainy readings: {wet_readings}")
print(f"Dry readings: {NUM_RECORDS - wet_readings}")
print(f"Rainy reading percentage: {wet_readings / NUM_RECORDS * 100:.2f}%")
print(f"Total simulated rainfall: {df['rainfall'].sum():.3f} mm")

print("\nWater-level summary:")
print(f"Minimum: {df['water_level'].min():.3f}")
print(f"Maximum: {df['water_level'].max():.3f}")
print(f"Mean: {df['water_level'].mean():.3f}")
print(f"Standard deviation: {df['water_level'].std():.3f}")

print("\nTimestamp interval verification:")
print(intervals.value_counts())

print("\nFirst five records:")
print(df.head().to_string(index=False))

print("\nDataset validation: PASSED")
