from pathlib import Path
import pandas as pd

results_dir = Path("results")
test_path = results_dir / "controlled_test_data.csv"
evaluation_path = results_dir / "controlled_anomaly_evaluation.csv"

for path in [test_path, evaluation_path]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

test = pd.read_csv(test_path, parse_dates=["timestamp"])
evaluation = pd.read_csv(evaluation_path)

window_size = 12
features = ["water_level", "rainfall", "temperature", "humidity"]

missed = evaluation[
    (evaluation["actual_anomaly"] == 1)
    & (evaluation["predicted_anomaly"] == 0)
]

false_positives = evaluation[
    (evaluation["actual_anomaly"] == 0)
    & (evaluation["predicted_anomaly"] == 1)
]

selected = pd.concat(
    [
        missed.assign(diagnostic_type="missed_anomaly"),
        false_positives.assign(diagnostic_type="false_positive"),
    ],
    ignore_index=True,
)

diagnostic_rows = []

for _, result in selected.iterrows():
    index = int(result["window_index"])
    window = test.iloc[index:index + window_size]

    if len(window) != window_size:
        print(f"Skipping incomplete window {index}")
        continue

    injected_count = int(window["injected_anomaly"].sum())

    row = {
        "diagnostic_type": result["diagnostic_type"],
        "window_index": index,
        "window_start": window["timestamp"].iloc[0],
        "window_end": window["timestamp"].iloc[-1],
        "reconstruction_error": result["reconstruction_error"],
        "threshold": result["threshold"],
        "injected_readings_in_window": injected_count,
    }

    for feature in features:
        row[f"{feature}_first"] = window[feature].iloc[0]
        row[f"{feature}_last"] = window[feature].iloc[-1]
        row[f"{feature}_min"] = window[feature].min()
        row[f"{feature}_max"] = window[feature].max()
        row[f"{feature}_mean"] = window[feature].mean()

    diagnostic_rows.append(row)

diagnostics = pd.DataFrame(diagnostic_rows)
output_path = results_dir / "window_reading_diagnostics.csv"
diagnostics.to_csv(output_path, index=False)

print("\nMISSED WINDOWS AND FALSE POSITIVES")
print("=" * 75)

if diagnostics.empty:
    print("No diagnostic windows found.")
else:
    columns = [
        "diagnostic_type",
        "window_index",
        "window_start",
        "window_end",
        "reconstruction_error",
        "threshold",
        "injected_readings_in_window",
        "water_level_min",
        "water_level_max",
        "rainfall_max",
        "temperature_min",
        "temperature_max",
    ]
    print(diagnostics[columns].to_string(index=False))

print(f"\nSaved to: {output_path}")