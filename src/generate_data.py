
from pathlib import Path

import numpy as np
import pandas as pd


# Reproducible simulated data
rng = np.random.default_rng(42)

# Dataset settings
NUM_RECORDS = 5000
INTERVAL_MINUTES = 5

timestamps = pd.date_range(
    start="2026-01-01 00:00:00",
    periods=NUM_RECORDS,
    freq=f"{INTERVAL_MINUTES}min",
)

# Simulate rainfall measurements in each five-minute interval.
# Most intervals are dry; some contain light or moderate rain.
rain_occurs = rng.random(NUM_RECORDS) < 0.12

rainfall = np.zeros(NUM_RECORDS)
rainfall[rain_occurs] = rng.gamma(
    shape=1.5,
    scale=1.2,
    size=rain_occurs.sum(),
)

# Simulate daily temperature variation.
time_in_days = np.arange(NUM_RECORDS) * INTERVAL_MINUTES / 1440

temperature = (
    25
    + 4 * np.sin(2 * np.pi * time_in_days - np.pi / 2)
    + rng.normal(0, 0.4, NUM_RECORDS)
)

# Simulate humidity with daily variation and a modest rain response.
humidity = (
    68
    - 0.8 * (temperature - 25)
    + 1.5 * np.minimum(rainfall, 5)
    + rng.normal(0, 1.5, NUM_RECORDS)
)

humidity = np.clip(humidity, 30, 100)

# Simulate water-level changes.
# Rainfall can raise the level, while gradual drainage lowers it.
water_level = np.zeros(NUM_RECORDS)
water_level[0] = 30.0

for i in range(1, NUM_RECORDS):
    water_level[i] = (
        water_level[i - 1]
        + 0.12 * rainfall[i]
        - 0.015 * (water_level[i - 1] - 30)
        + rng.normal(0, 0.08)
    )

water_level = np.clip(water_level, 5, 150)

# Assemble the time-series dataset.
df = pd.DataFrame(
    {
        "timestamp": timestamps,
        "water_level": water_level,
        "rainfall": rainfall,
        "temperature": temperature,
        "humidity": humidity,
    }
)

# Save the generated dataset.
output_path = Path("data/raw/simulated_environmental_data.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

print("Dataset generated successfully.")
print(f"Rows: {len(df)}")
print(f"Sampling interval: {INTERVAL_MINUTES} minutes")
print(f"Saved to: {output_path}")

print("\nFirst five records:")
print(df.head().to_string(index=False))

print("\nSummary statistics:")
print(df[["water_level", "rainfall", "temperature", "humidity"]].describe())

