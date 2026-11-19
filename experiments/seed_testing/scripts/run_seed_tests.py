from pathlib import Path
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
TEMPLATE = BASE_DIR / "scripts" / "generate_data_template.py"

SEEDS = [42, 123, 456, 789, 2026]

for seed in SEEDS:
    output_dir = BASE_DIR / f"seed_{seed}" / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "simulated_environmental_data.csv"

    script_text = TEMPLATE.read_text(encoding="utf-8")

    script_text = script_text.replace(
        "RANDOM_SEED = 42",
        f"RANDOM_SEED = {seed}",
        1,
    )

    old_path = (
        'OUTPUT_PATH = Path(\n'
        '    "experiments/improved_simulation_24/data/raw/"\n'
        '    "simulated_environmental_data.csv"\n'
        ')'
    )

    new_path = (
        'OUTPUT_PATH = Path(r"'
        + str(output_file.resolve())
        + '")'
    )

    if old_path not in script_text:
        raise RuntimeError(
            f"Could not locate OUTPUT_PATH configuration for seed {seed}."
        )

    script_text = script_text.replace(old_path, new_path, 1)

    temporary_script = BASE_DIR / f"generate_seed_{seed}.py"
    temporary_script.write_text(script_text, encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"GENERATING DATASET WITH SEED {seed}")
    print(f"{'=' * 60}")

    try:
        subprocess.run(
            [sys.executable, str(temporary_script)],
            check=True,
        )
    finally:
        temporary_script.unlink(missing_ok=True)

print("\nALL FIVE SEED DATASETS GENERATED.")
