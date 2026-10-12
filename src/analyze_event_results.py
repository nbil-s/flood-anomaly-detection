
from pathlib import Path
import pandas as pd

path = Path("results/controlled_anomaly_evaluation.csv")
df = pd.read_csv(path)

events = {
    "A - Rapid water-level rise": (89, 105),
    "B - Intense rainfall": (239, 255),
    "C - Combined conditions": (389, 405),
    "D - Simulated sensor fault": (539, 550),
}

print("\nPER-EVENT DETECTION RESULTS")
print("=" * 60)

for name, (start, end) in events.items():
    group = df[df["window_index"].between(start, end)]

    actual = group["actual_anomaly"].astype(int)
    predicted = group["predicted_anomaly"].astype(int)

    tp = int(((actual == 1) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    positive = int((actual == 1).sum())

    rate = tp / positive if positive else 0

    print(f"\n{name}")
    print(f"  Labelled positive windows: {positive}")
    print(f"  Detected positive windows: {tp}")
    print(f"  Missed positive windows: {fn}")
    print(f"  Detection rate: {rate:.2%}")

print("\nNote: Windows overlap, and the events are simulated.")
