"""
Usage:
    python generate_cycle15_summary.py
"""

import os
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_RESULTS = _SCRIPT_DIR.parent / "results"

MAX_CYCLE = 15

METRICS = [
    "test_precision",
    "test_recall",
    "test_specificity",
    "test_auroc",
    "test_auprc",
    "test_mcc",
    "test_ef5",
    "hits_discovered",
]

METRIC_NAMES = {
    "test_precision": "Test Precision",
    "test_recall": "Test Recall",
    "test_specificity": "Test Specificity",
    "test_auroc": "Test AUROC",
    "test_auprc": "Test AUPRC",
    "test_mcc": "Test MCC",
    "test_ef5": "Test EF@5",
    "hits_discovered": "Hits Discovered",
}


def read_results_csvs(root_dir: Path) -> pd.DataFrame:
    """Read and concatenate all *_results.csv files in a directory."""
    csv_paths = sorted(root_dir.glob("*_results.csv"))
    if not csv_paths:
        print(f"  WARNING: No *_results.csv found in {root_dir}")
        return pd.DataFrame()

    frames = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        frame["source_file"] = csv_path.name
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_cycle15_summary(
    df: pd.DataFrame,
    dataset: str,
    methods: list[str],
    output_dir: Path,
) -> Optional[Path]:
    """Generate cycle15_summary.csv for one dataset."""
    dataset_df = df[df["dataset"] == dataset]
    summary_rows: list[list[str]] = []

    for method in methods:
        subset = dataset_df[
            (dataset_df["acquisition_method"] == method)
            & (dataset_df["test_cycle"] == MAX_CYCLE)
        ]
        if subset.empty:
            continue

        row = [method]
        for metric in METRICS:
            if metric not in subset.columns:
                row.append("NA")
                continue
            per_seed = subset.groupby("seed")[metric].mean()
            mean = float(per_seed.mean())
            std = float(per_seed.std()) if len(per_seed) > 1 else 0.0
            if metric == "hits_discovered":
                row.append(f"{int(round(mean))}±{int(round(std))}")
            else:
                row.append(f"{mean:.2f}±{std:.2f}")
        summary_rows.append(row)

    if not summary_rows:
        return None

    summary_df = pd.DataFrame(
        summary_rows,
        columns=["Method"] + [METRIC_NAMES.get(m, m) for m in METRICS],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{dataset}_cycle15_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    return csv_path


def process_experiment_dir(exp_dir_name: str) -> None:
    """Process one experiment directory: read CSVs, generate summaries."""
    root = _RESULTS / exp_dir_name
    if not root.is_dir():
        print(f"SKIP: {root} does not exist")
        return

    print(f"\n{'='*60}")
    print(f"Processing: {exp_dir_name}")
    print(f"{'='*60}")

    df = read_results_csvs(root)
    if df.empty:
        print("  No data loaded.")
        return

    # Discover available datasets and methods
    datasets = sorted(df["dataset"].dropna().astype(str).unique())
    available_methods = sorted(df["acquisition_method"].dropna().astype(str).unique())

    print(f"  Datasets: {datasets}")
    print(f"  Methods:  {available_methods}")

    for dataset in datasets:
        output_dir = root / f"{dataset}_metrics_results"
        csv_path = save_cycle15_summary(df, dataset, available_methods, output_dir)
        if csv_path is not None:
            print(f"  ✓ {csv_path.relative_to(_RESULTS)}")
        else:
            print(f"  ✗ {dataset}: no cycle {MAX_CYCLE} data")


def main() -> None:
    os.chdir(_SCRIPT_DIR)

    experiment_dirs = [
        "20251014_strategy_experiments",
        "260323_dynamiccomb_ratio_experiments",
        "260526_custom_dynamic_experiments_p10of64",
    ]

    for exp_dir in experiment_dirs:
        process_experiment_dir(exp_dir)

    print(f"\nDone. Summary files are in {_RESULTS}")


if __name__ == "__main__":
    main()
