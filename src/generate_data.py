
from pathlib import Path

import numpy as np
import pandas as pd


# Reproducible simulated data
rng = np.random.default_rng(42)

# Dataset settings
NUM_RECORDS = 5000
INTERVAL_SECONDS = 30

# Reference interval used by the original simulation
REFERENCE_INTERVAL_SECONDS = 300

# Time scaling relative to the original five-minute simulation
TIME_SCALE = INTERVAL_SECONDS / REFERENCE_INTERVAL_SECONDS

timestamps = pd.date_range(
    start="2026-01-01 00:00:00",
    periods=NUM_RECORDS,
    freq=f"{INTERVAL_SECONDS}s",
)

# Adjust rainfall occurrence probability to the shorter interval.
# This approximately preserves the probability of rain over five minutes.
REFERENCE_RAIN_PROBABILITY = 0.12

rain_probability = 1 - (
    1 - REFERENCE_RAIN_PROBABILITY
) ** TIME_SCALE

rain_occurs = rng.random(NUM_RECORDS) < rain_probability

# Rainfall amount per 30-second interval
rainfall = np.zeros(NUM_RECORDS)

rainfall[rain_occurs] = rng.gamma(
    shape=1.5,
    scale=1.2 * TIME_SCALE,
    size=rain_occurs.sum(),
)

# Simulate daily temperature variation
time_in_days = (
    np.arange(NUM_RECORDS) * INTERVAL_SECONDS / 86400
)

temperature = (
    25
    + 4 * np.sin(2 * np.pi * time_in_days - np.pi / 2)
    + rng.normal(0, 0.4, NUM_RECORDS)
)

# Simulate humidity
humidity = (
    68
    - 0.8 * (temperature - 25)
    + 1.5 * np.minimum(rainfall, 5)
    + rng.normal(0, 1.5, NUM_RECORDS)
)

humidity = np.clip(humidity, 30, 100)

# Simulate water-level changes
water_level = np.zeros(NUM_RECORDS)
water_level[0] = 30.0

# Scale drainage with elapsed time and random variation
# with the square root of elapsed time.
drainage_coefficient = 0.015 * TIME_SCALE
noise_standard_deviation = 0.08 * np.sqrt(TIME_SCALE)

for i in range(1, NUM_RECORDS):
    water_level[i] = (
        water_level[i - 1]
        + 0.12 * rainfall[i]
        - drainage_coefficient * (water_level[i - 1] - 30)
        + rng.normal(0, noise_standard_deviation)
    )

water_level = np.clip(water_level, 5, 150)

# Assemble the time-series dataset
df = pd.DataFrame({
    "timestamp": timestamps,
    "water_level": water_level,
    "rainfall": rainfall,
    "temperature": temperature,
    "humidity": humidity,
})

# Save the generated dataset
output_path = Path("data/raw/simulated_environmental_data.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

print("Dataset generated successfully.")
print(f"Rows: {len(df)}")
print(f"Sampling interval: {INTERVAL_SECONDS} seconds")
print(f"Rain probability per reading: {rain_probability:.5f}")
print(f"Saved to: {output_path}")

print("\nFirst five records:")
print(df.head().to_string(index=False))

print("\nSummary statistics:")
print(
    df[["water_level", "rainfall", "temperature", "humidity"]]
    .describe()
)

print("\nTimestamp interval verification:")
print(df["timestamp"].diff().dropna().value_counts())
