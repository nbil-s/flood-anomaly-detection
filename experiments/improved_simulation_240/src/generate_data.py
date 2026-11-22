
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# DATASET CONFIGURATION
# ============================================================

RANDOM_SEED = 42

NUM_RECORDS = 100_000
OUTPUT_INTERVAL_SECONDS = 60
INTERNAL_INTERVAL_SECONDS = 30

INTERNAL_RECORDS = NUM_RECORDS * 2

START_TIME = "2026-01-01 00:00:00"

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "simulated_environmental_data.csv"
)

FEATURES = [
    "water_level",
    "rainfall",
    "temperature",
    "humidity",
]

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# INTERNAL TIMESTAMPS
# ============================================================

internal_timestamps = pd.date_range(
    start=START_TIME,
    periods=INTERNAL_RECORDS,
    freq=f"{INTERNAL_INTERVAL_SECONDS}s",
)

time_seconds = (
    np.arange(INTERNAL_RECORDS)
    * INTERNAL_INTERVAL_SECONDS
)

time_in_days = time_seconds / 86400


# ============================================================
# RAINFALL EPISODES
# ============================================================

rainfall_30s = np.zeros(INTERNAL_RECORDS)

RAIN_START_PROBABILITY = 0.0012
RAIN_CONTINUE_PROBABILITY = 0.90

RAIN_SHAPE = 1.8
RAIN_SCALE = 0.20

is_raining = False

for i in range(INTERNAL_RECORDS):

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

        rainfall_30s[i] = intensity


# ============================================================
# TEMPERATURE
# ============================================================

temperature_30s = (
    25
    + 4.0 * np.sin(
        2 * np.pi * time_in_days - np.pi / 2
    )
    + 0.5 * np.sin(
        2 * np.pi * time_in_days / 7
    )
    + rng.normal(0, 0.35, INTERNAL_RECORDS)
)


# ============================================================
# HUMIDITY
# ============================================================

humidity_30s = (
    68
    - 0.9 * (temperature_30s - 25)
    + 2.5 * np.minimum(rainfall_30s, 3)
    + rng.normal(0, 1.2, INTERNAL_RECORDS)
)

humidity_30s = np.clip(
    humidity_30s,
    30,
    100,
)


# ============================================================
# DELAYED RUNOFF AND WATER-LEVEL RESPONSE
# ============================================================

water_level_30s = np.zeros(INTERNAL_RECORDS)
runoff = np.zeros(INTERNAL_RECORDS)

BASE_WATER_LEVEL = 30.0

RUNOFF_RETENTION = 0.995
RUNOFF_RESPONSE = 0.65
WATER_LEVEL_NOISE = 0.025

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

routing_kernel = np.concatenate((
    [0.0],
    RAINFALL_ROUTING_WEIGHTS,
))

routed_rainfall = np.convolve(
    rainfall_30s,
    routing_kernel,
    mode="full",
)[:INTERNAL_RECORDS]

water_level_30s[0] = BASE_WATER_LEVEL

for i in range(1, INTERNAL_RECORDS):

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
            water_level_30s[i - 1]
            - (
                BASE_WATER_LEVEL
                + RUNOFF_RESPONSE * runoff[i - 1]
            )
        )
        + rng.normal(0, WATER_LEVEL_NOISE)
    )

    water_level_30s[i] = expected_level + noise

water_level_30s = np.clip(
    water_level_30s,
    5,
    150,
)


# ============================================================
# AGGREGATE TWO 30-SECOND SAMPLES INTO ONE MINUTE
# ============================================================

rainfall = rainfall_30s.reshape(NUM_RECORDS, 2).sum(axis=1)

temperature = temperature_30s.reshape(
    NUM_RECORDS, 2
).mean(axis=1)

humidity = humidity_30s.reshape(
    NUM_RECORDS, 2
).mean(axis=1)

# Use the final water-level measurement in each minute.
water_level = water_level_30s[1::2]


# ============================================================
# BUILD FINAL DATAFRAME
# ============================================================

timestamps = pd.date_range(
    start=START_TIME,
    periods=NUM_RECORDS,
    freq="1min",
)

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
    raise ValueError("Incorrect number of observations.")

if df[FEATURES].isna().any().any():
    raise ValueError("Missing values detected.")

if not np.isfinite(
    df[FEATURES].to_numpy(dtype=float)
).all():
    raise ValueError("Non-finite values detected.")

if (df["rainfall"] < 0).any():
    raise ValueError("Negative rainfall detected.")

if not df["timestamp"].is_monotonic_increasing:
    raise ValueError("Timestamps are not ordered.")

if df["timestamp"].duplicated().any():
    raise ValueError("Duplicate timestamps detected.")

intervals = df["timestamp"].diff().dropna()

if not (
    intervals == pd.Timedelta(minutes=1)
).all():
    raise ValueError("Incorrect one-minute intervals.")

if not df["water_level"].between(5, 150).all():
    raise ValueError("Water-level values outside configured bounds.")

if not df["humidity"].between(30, 100).all():
    raise ValueError("Humidity values outside configured bounds.")


# ============================================================
# SAVE DATASET
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(OUTPUT_PATH, index=False)


# ============================================================
# GENERATION REPORT
# ============================================================

wet_readings = int((df["rainfall"] > 0).sum())

print("\n100,000-OBSERVATION DATASET GENERATED")
print("=" * 55)

print(f"Observations: {len(df):,}")
print(f"Internal simulation interval: {INTERNAL_INTERVAL_SECONDS} seconds")
print(f"Final sampling interval: {OUTPUT_INTERVAL_SECONDS} seconds")
print(f"Internal observations: {INTERNAL_RECORDS:,}")

print(f"\nStart: {df['timestamp'].iloc[0]}")
print(f"End:   {df['timestamp'].iloc[-1]}")

print(f"\nSaved to: {OUTPUT_PATH}")

print("\nFEATURE STATISTICS")
print(df[FEATURES].describe().round(4))

print("\nRAINFALL SUMMARY")
print(f"Wet observations: {wet_readings:,}")
print(f"Dry observations: {NUM_RECORDS - wet_readings:,}")
print(f"Wet percentage: {wet_readings / NUM_RECORDS * 100:.2f}%")
print(f"Total simulated rainfall: {df['rainfall'].sum():.3f}")

print("\nWATER-LEVEL SUMMARY")
print(f"Minimum: {df['water_level'].min():.3f}")
print(f"Maximum: {df['water_level'].max():.3f}")
print(f"Mean: {df['water_level'].mean():.3f}")
print(f"Standard deviation: {df['water_level'].std():.3f}")

print("\nSAMPLING INTERVAL VERIFICATION")
print(intervals.value_counts())

print("\nFIRST FIVE OBSERVATIONS")
print(df.head().to_string(index=False))

print("\nLAST FIVE OBSERVATIONS")
print(df.tail().to_string(index=False))

print("\nDATASET VALIDATION: PASSED")
