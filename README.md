![python version](https://img.shields.io/badge/python-3.9_|_3.10_|_3.11-blue)
![license](https://img.shields.io/badge/license-MIT-orange)


# HybridAL: Hybrid active learning improves robustness in low-data molecular screening

## Description

It supports 16 acquisition strategies including pure uncertainty-based methods,
fixed-ratio exploitation-uncertainty mixing (Hybridcombo series), and
time-triggered strategy switching (policy series).

## Requirements

Python 3.9 with:
- PyTorch 1.12.1 / PyTorch Geometric 2.3.1
- RDKit 2023.3.2
- Scikit-learn 1.3.0

## Installation

```bash
conda env create -f env.yaml
conda activate traversing_chem
```

## Data

This repository includes preprocessed data for 6 datasets:
`CYP3A4`, `PKM2`, `TP53`, `ESR1_ant`, `MTORC1`, `PPARG`.

Original data from [Wang et al.](https://github.com/wangwrx/An-uncertainty-guided-deep-learning-method-facilitates-rapid-screening-of-CYP3A4-inhibitors) and [LIT-PCBA](https://drugdesign.unistra.fr/LIT-PCBA/).

To preprocess new data:
```bash
python preprocess_data.py
```

## Usage

### Basic active learning run
```bash
python main.py -acq exploitation -dataset CYP3A4 -arch mlp -batch_size 64 -n_start 64
```

### Command-line arguments

| Argument | Options | Default |
|---|---|---|
| `-acq` | Strategy key (see below) | `exploitation` |
| `-dataset` | `CYP3A4`, `PKM2`, `TP53`, `ESR1_ant`, `MTORC1`, `PPARG` | `CYP3A4` |
| `-batch_size` | int | `64` |
| `-n_start` | int | `64` |
| `-init_pos_count` | int | `10` |
| `-seed` | int, number of seeds | `5` |

### Batch experiments

Use `run_experiments.py` to run multiple experiments in sequence across
datasets, strategies, architectures, and seeds.

```bash
# Preview all configured experiments (dry run)
python run_experiments.py

# Execute all configured experiments
python run_experiments.py --execute

# Run specific experiments via CLI
python run_experiments.py --datasets CYP3A4 PKM2 --strategies bald Hybridcombo9010 --seed 3 --execute

# Run on specific architectures only
python run_experiments.py --architectures gcn --execute
```

Edit the `EXPERIMENT_CONFIG` dictionary in `run_experiments.py` to customize
which experiment groups to run.

## Available Acquisition Strategies

### Pure Strategies

| Strategy | Key | Description |
|---|---|---|
| Random | `random` | Uniform random selection from screening pool |
| Exploitation | `exploitation` | Select molecules with highest predicted hit probability |
| Uncertainty | `uncertainty` | Select molecules with highest prediction entropy |
| BALD | `bald` | Select molecules with highest mutual information (Bayesian Active Learning by Disagreement) |
| BALA | `bala` | Select molecules with lowest mutual information (Bayesian Active Learning by Agreement) |
| Similarity | `similarity` | Select molecules with highest Tanimoto similarity to known hits |

### Hybrid Combination Strategies (Combo)

Fixed-ratio mixing of exploitation and uncertainty. Each batch of `n` molecules
splits into exploitation and uncertainty picks at a fixed ratio, with
de-duplication to avoid selecting the same molecule twice.

| Strategy | Key | Ratio (exploitation:uncertainty) |
|---|---|---|
| Hybridcombo9010 | `Hybridcombo9010` | 90:10 |
| Hybridcombo8020 | `Hybridcombo8020` | 80:20 |
| Hybridcombo955 | `Hybridcombo955` | 95:5 |

### Time-Triggered Switching Strategies (Policy)

Phase-based strategies that switch acquisition methods depending on the
active learning cycle (0-indexed).

| Strategy | Key | Cycles 0–4 | Cycles 5–7 | Cycles 8+ |
|---|---|---|---|---|
| policy1 | `policy1` | Exploitation | Uncertainty | Uncertainty |
| policy2 | `policy2` | Exploitation | BALD | BALD |
| policy3 | `policy3` | Exploitation | Uncertainty | Exploitation |
| policy4 | `policy4` | Exploitation | BALD | Exploitation |

## Sample Data Tracking

All acquisition strategies automatically save per-batch CSV files containing:
`smiles`, `probability`, `mutual_information`, `uncertainty`, Tanimoto similarity
to hits/non-hits, and similarity difference.

Output path: `{output_dir}/obtained_data/seed{seed}_obtained_data_{dataset}/`

## How to Cite

If you use this code, please cite both:

1. The original framework:
   > van Tilborg, D. & Grisoni, F. 
   > Traversing Chemical Space with Active Deep Learning for Low-data Drug Discovery. *Nat. Comput. Sci.* **4**, 786–796 (2024).
   > DOI: [10.1038/s43588-024-00697-2](https://doi.org/10.1038/s43588-024-00697-2)

2. This repository.
   > Luo, Y., Hu, D. & Guan, X. Hybrid active learning improves robustness in low-data molecular screening. (2026).

## License

All code is under MIT license. Original code copyright 2023 Derek van Tilborg.
