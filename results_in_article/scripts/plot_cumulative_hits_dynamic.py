import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_SCRIPT_DIR, "..", "results")
_FIGURES = os.path.join(_SCRIPT_DIR, "..", "figures")

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 20, "axes.labelsize": 24, "axes.titlesize": 26,
    "xtick.labelsize": 22, "ytick.labelsize": 20,
    "legend.fontsize": 18, "axes.linewidth": 1.0,
})

datasets = ["CYP3A4", "PKM2", "TP53"]
strategies = ["exploitation", "dynamicexp", "dynamicexp2", "dynamicexphit", "dynamicexphit2"]
strategy_labels = ["Exploitation", "Policy1", "Policy2", "Policy3", "Policy4"]
colors = ["#5A5A5A", "#457B9D", "#9B5DE5", "#F4A261", "#E76F51"]
cycles = np.arange(0, 16)
data_dir = os.path.join(_RESULTS, "260526_custom_dynamic_experiments_p10of64")

output_dir = os.path.join(_FIGURES, "cumulative_hits_dynamic_output")
os.makedirs(output_dir, exist_ok=True)

for dataset in datasets:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor("white")
    ax.set_box_aspect(1)

    y_max_all = 0
    for strategy, label, color in zip(strategies, strategy_labels, colors):
        pattern = os.path.join(data_dir, f"{dataset}_{strategy}_64_p10of64_results.csv")
        files = glob.glob(pattern)
        if not files:
            continue

        df = pd.read_csv(files[0])
        grouped = df.groupby("train_cycle")["hits_discovered"]
        means = []
        stds = []
        valid_cycles = []
        for cycle in cycles:
            if cycle in grouped.groups:
                g = grouped.get_group(cycle)
                means.append(g.mean())
                stds.append(g.std())
                valid_cycles.append(cycle)

        means = np.array(means)
        stds = np.array(stds)
        valid_cycles = np.array(valid_cycles)

        ax.plot(valid_cycles, means, color=color, lw=2.5, label=label, marker="o", markersize=6)
        ax.fill_between(valid_cycles, means - stds, means + stds, color=color, alpha=0.12)
        if len(means) > 0:
            y_max_all = max(y_max_all, (means + stds).max())

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Cumulative Hits")
    ax.set_title(dataset, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(cycles)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, max(y_max_all * 1.15, y_max_all + 60))

    ax.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Cumulative_Hits_Dynamic_{dataset}.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"{output_dir}/Cumulative_Hits_Dynamic_{dataset}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved Cumulative_Hits_Dynamic_{dataset}.png/pdf")

print("Done.")
