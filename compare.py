"""Compare PyTorch and Keras across multiple seeds."""

from pathlib import Path

import pandas as pd

from train import main as pytorch_main
from train_keras import main as keras_main

SEEDS = [
    42,
    100,
    777,
]


def extract_metrics(results_frame):
    neural_network_row = results_frame.iloc[2]

    return {
        "validation_mae": float(neural_network_row["validation_mae"]),
        "test_mae": float(neural_network_row["test_mae"]),
    }


def run_pytorch(seed):
    print(f"\nRunning PyTorch (seed={seed})")

    result = pytorch_main(seed=seed)

    metrics = extract_metrics(result["results"])

    return {
        "framework": "PyTorch",
        "seed": seed,
        "validation_mae": metrics["validation_mae"],
        "test_mae": metrics["test_mae"],
    }


def run_keras(seed):
    print(f"\nRunning Keras (seed={seed})")

    result = keras_main(seed=seed)

    metrics = extract_metrics(result["results"])

    return {
        "framework": "Keras",
        "seed": seed,
        "validation_mae": metrics["validation_mae"],
        "test_mae": metrics["test_mae"],
    }


def main():
    Path("reports").mkdir(
        exist_ok=True,
    )

    rows = []

    for seed in SEEDS:
        rows.append(run_pytorch(seed))

    for seed in SEEDS:
        rows.append(run_keras(seed))

    comparison = pd.DataFrame(rows)

    comparison.to_csv(
        "reports/comparison.csv",
        index=False,
    )

    print("\nComparison Results")

    print(comparison)

    print("\nSaved reports/comparison.csv")

    return comparison


if __name__ == "__main__":
    main()
