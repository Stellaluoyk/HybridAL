"""
Batch experiment runner for HybridAL.

Runs multiple active learning experiments across datasets, strategies, and seeds
by invoking main.py with the appropriate arguments.

Usage:
    # Dry-run: print all commands without executing
    python run_experiments.py

    # Execute all commands sequentially
    python run_experiments.py --execute

    # Override defaults
    python run_experiments.py --datasets CYP3A4 PKM2 --strategies bald policy1 --seed 3 --batch_size 32

    # Run on specific architectures only
    python run_experiments.py --architectures mlp gcn --execute

Configuration:
    Edit the EXPERIMENT_CONFIG dictionary below to define which experiments to run.
    Each key is an experiment group name, and each group specifies:
        - datasets: list of dataset names
        - strategies: list of acquisition strategy keys
        - architectures: (optional) list of model architectures
        - seed: (optional) number of seeds, generates 0..N-1
        - bias: (optional) initial dataset bias
        - init_pos_count: (optional) number of positive samples in starting set

    See README.md for available strategy keys.

Author: Yukun Luo
"""

from __future__ import annotations

import subprocess
from pathlib import Path
import argparse
from datetime import datetime

# ---------------------------------------------------------------------------
# Default experiment configuration
# ---------------------------------------------------------------------------

N_START = 64
BATCH_SIZE = 64
BASE_DIR = Path(__file__).resolve().parent

EXPERIMENT_CONFIG = {
    # --- Benchmark datasets with pure strategies ---
    "benchmark_pure": {
        "datasets": ["CYP3A4", "PKM2", "TP53"],
        "strategies": ["random", "exploitation", "uncertainty", "bald", "bala", "similarity"],
    },
    # --- Hybrid combination strategies ---
    "benchmark_hybridcombo": {
        "datasets": ["CYP3A4", "PKM2", "TP53"],
        "strategies": ["Hybridcombo9010", "Hybridcombo8020", "Hybridcombo955"],
    },
    # --- Time-triggered switching strategies ---
    "benchmark_policy": {
        "datasets": ["CYP3A4", "PKM2", "TP53"],
        "strategies": ["policy1", "policy2", "policy3", "policy4"],
    },
    # --- New datasets (ESR1_ant, MTORC1, PPARG) ---
    "newdata": {
        "datasets": ["ESR1_ant", "MTORC1", "PPARG"],
        "strategies": ["random", "exploitation", "uncertainty", "bald", "Hybridcombo9010", "policy1"],
    },
}

# Default settings applied to all experiment groups unless overridden in the group config
DEFAULT_ARCHITECTURES = ["mlp"]
DEFAULT_SEED = 5
DEFAULT_BIAS = "real"
DEFAULT_INIT_POS_COUNT = 10

# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------


def _output_path(experiment_name: str, pos_count: int, n_start: int, date: str) -> Path:
    token = f"p{pos_count}of{n_start}"
    return BASE_DIR / f"{date}_{experiment_name}_{token}"


def build_commands(
    experiment_config: dict | None = None,
    architectures: list[str] | None = None,
    seed: int | None = None,
    n_start: int = N_START,
    batch_size: int = BATCH_SIZE,
    bias: str = DEFAULT_BIAS,
    init_pos_count: int = DEFAULT_INIT_POS_COUNT,
    date: str | None = None,
) -> list[list[str]]:
    if experiment_config is None:
        experiment_config = EXPERIMENT_CONFIG
    if architectures is None:
        architectures = DEFAULT_ARCHITECTURES
    if seed is None:
        seed = DEFAULT_SEED
    if date is None:
        date = datetime.now().strftime("%Y%m%d")

    commands: list[list[str]] = []

    for group_name, cfg in experiment_config.items():
        group_datasets = cfg["datasets"]
        group_strategies = cfg["strategies"]
        group_archs = cfg.get("architectures", architectures)
        group_seed = cfg.get("seed", seed)
        group_bias = cfg.get("bias", bias)
        group_pos = cfg.get("init_pos_count", init_pos_count)
        group_n_start = cfg.get("n_start", n_start)
        group_batch = cfg.get("batch_size", batch_size)

        for dataset in group_datasets:
            for strategy in group_strategies:
                for arch in group_archs:
                    cmd = [
                        "python", "main.py",
                        "-o", str(_output_path(group_name, group_pos, group_n_start, date)),
                        "-acq", strategy,
                        "-dataset", dataset,
                        "-arch", arch,
                        "-batch_size", str(group_batch),
                        "-n_start", str(group_n_start),
                        "-bias", group_bias,
                        "-init_pos_count", str(group_pos),
                        "--seed", str(group_seed),
                    ]
                    commands.append(cmd)

    return commands


def run_commands(
    commands: list[list[str]],
    cwd: Path | None = None,
    dry_run: bool = True,
) -> int:
    cwd = cwd or BASE_DIR
    failures = 0

    for cmd in commands:
        cmd_str = " ".join(cmd)
        if dry_run:
            print(f"[DRY RUN] {cmd_str}")
            continue

        print(f"[RUN] {cmd_str}")
        result = subprocess.run(cmd, cwd=cwd)
        if result.returncode == 0:
            print(f"  -> OK\n")
        else:
            print(f"  -> FAILED (exit={result.returncode})\n")
            failures += 1

    return failures


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HybridAL batch experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview all configured experiments (dry run)
  python run_experiments.py

  # Execute everything
  python run_experiments.py --execute

  # Run only specific datasets and strategies
  python run_experiments.py --datasets CYP3A4 --strategies bald Hybridcombo9010 --execute

  # Override experiment config entirely via CLI (ignores EXPERIMENT_CONFIG in file)
  python run_experiments.py --datasets PKM2 TP53 --strategies exploitation policy1 --architectures mlp --seed 2 --execute
        """,
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually execute commands (default: dry-run, print only)",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        help="Override datasets (e.g. --datasets CYP3A4 PKM2)",
    )
    parser.add_argument(
        "--strategies", nargs="+",
        help="Override strategies (e.g. --strategies bald Hybridcombo9010 policy1)",
    )
    parser.add_argument(
        "--architectures", nargs="+",
        default=DEFAULT_ARCHITECTURES,
        help=f"Model architectures (default: {DEFAULT_ARCHITECTURES})",
    )
    parser.add_argument(
        "--seed", type=int,
        default=DEFAULT_SEED,
        help=f"Number of seeds, passes --seed N to main.py (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help=f"Molecules per cycle (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--n_start", type=int, default=N_START,
        help=f"Initial training set size (default: {N_START})",
    )
    parser.add_argument(
        "--bias", default=DEFAULT_BIAS,
        help=f"Initial bias type (default: {DEFAULT_BIAS})",
    )
    parser.add_argument(
        "--init_pos_count", type=int, default=DEFAULT_INIT_POS_COUNT,
        help=f"Initial positive count (default: {DEFAULT_INIT_POS_COUNT})",
    )
    args = parser.parse_args()

    if args.datasets and args.strategies:
        config = {
            "cli_experiment": {
                "datasets": args.datasets,
                "strategies": args.strategies,
                "architectures": args.architectures,
                "seed": args.seed,
                "bias": args.bias,
                "init_pos_count": args.init_pos_count,
                "n_start": args.n_start,
                "batch_size": args.batch_size,
            }
        }
    else:
        config = EXPERIMENT_CONFIG

    commands = build_commands(
        experiment_config=config,
        architectures=args.architectures,
        seed=args.seed,
        n_start=args.n_start,
        batch_size=args.batch_size,
        bias=args.bias,
        init_pos_count=args.init_pos_count,
    )

    print(f"Total commands: {len(commands)}")
    print(f"Mode: {'DRY RUN' if not args.execute else 'EXECUTE'}")
    print("-" * 60)

    failures = run_commands(commands, dry_run=not args.execute)

    if failures:
        print(f"\n{failures} command(s) failed.")
        raise SystemExit(1)
    elif not args.execute:
        print("\nDry run complete. Add --execute to run.")
    else:
        print("\nAll experiments completed successfully.")


if __name__ == "__main__":
    main()
