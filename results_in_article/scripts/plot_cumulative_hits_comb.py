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
strategy_order = ["exploitation", "dynamiccomb955", "dynamiccomb", "dynamiccomb8020"]
strategy_labels = ["Exploitation", "Combo (95:5)", "Combo (90:10)", "Combo (80:20)"]
colors = ["#E64B35", "#4DBBD5", "#00A087", "#E89242"]
cycles = np.arange(0, 16)

EXP_DIR = os.path.join(_RESULTS, "260323_dynamiccomb_ratio_experiments")
FALLBACK_DIR = os.path.join(_RESULTS, "20251014_strategy_experiments")

output_dir = os.path.join(_FIGURES, "cumulative_hits_comb_output")
os.makedirs(output_dir, exist_ok=True)


def load_strategy_data(dataset):
    """Load means, stds, valid_cycles for all strategies on a dataset."""
    all_data = {}
    y_max_all = 0.0
    for strategy, label, color in zip(strategy_order, strategy_labels, colors):
        pattern = os.path.join(EXP_DIR, f"{dataset}_{strategy}_64_results.csv")
        files = glob.glob(pattern)
        if not files and strategy == "exploitation":
            files = glob.glob(os.path.join(FALLBACK_DIR, f"{dataset}_exploitation_64_results.csv"))
        if not files:
            continue

        df = pd.read_csv(files[0])
        grouped = df.groupby("train_cycle")["hits_discovered"]
        means, stds, valid_cycles = [], [], []
        for cycle in cycles:
            if cycle in grouped.groups:
                g = grouped.get_group(cycle)
                means.append(g.mean())
                stds.append(g.std())
                valid_cycles.append(cycle)

        means = np.array(means)
        stds = np.array(stds)
        valid_cycles = np.array(valid_cycles)
        all_data[strategy] = {
            "means": means, "stds": stds, "valid_cycles": valid_cycles,
            "label": label, "color": color,
        }
        if len(means) > 0:
            y_max_all = max(y_max_all, (means + stds).max())
    return all_data, y_max_all


def plot_on_ax(ax, all_data):
    """Plot all strategies onto a single axes. Returns the axes."""
    for strategy, d in all_data.items():
        ax.plot(d["valid_cycles"], d["means"], color=d["color"], lw=2.5,
                label=d["label"], marker="o", markersize=6)
        ax.fill_between(d["valid_cycles"],
                        d["means"] - d["stds"],
                        d["means"] + d["stds"],
                        color=d["color"], alpha=0.12)
    return ax


def style_ax(ax):
    """Common axis styling."""
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(cycles)
    ax.set_xlim(0, 15)


# margin per dataset
margin_map = {"CYP3A4": 1.15, "PKM2": 1.16, "TP53": 1.14}

# broken-axis datasets: (split_y, height_ratios top:bottom)
BROKEN_CONFIG = {
    "PKM2": {"split_y": 10, "ratios": [9, 1]},
    "TP53": {"split_y": 10, "ratios": [9, 1]},
}

for dataset in datasets:
    all_data, y_max_all = load_strategy_data(dataset)
    y_margin = margin_map.get(dataset, 1.15)

    if dataset in BROKEN_CONFIG:
        cfg = BROKEN_CONFIG[dataset]
        split_y = cfg["split_y"]
        y_top = y_max_all * y_margin

        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, sharex=True,
            gridspec_kw={'height_ratios': cfg["ratios"]},
            figsize=(7, 7),
        )
        fig.subplots_adjust(hspace=0.05)

        # Plot on both axes
        plot_on_ax(ax_top, all_data)
        plot_on_ax(ax_bottom, all_data)

        # Styling
        style_ax(ax_top)
        style_ax(ax_bottom)

        ax_top.set_ylim(split_y, y_top)
        ax_bottom.set_ylim(0, split_y)
        top_ticks = ax_top.get_yticks()
        ax_top.set_yticks(top_ticks[top_ticks > split_y])
        ax_bottom.set_yticks([0, split_y])

        # Hide spines at the break
        ax_top.spines['bottom'].set_visible(False)
        ax_bottom.spines['top'].set_visible(False)
        ax_top.tick_params(axis="x", top=False, bottom=False,
                           labeltop=False, labelbottom=False)
        ax_bottom.xaxis.tick_bottom()

        # Diagonal break marks
        d = 0.02
        kw = dict(transform=ax_top.transAxes, color='k', clip_on=False, lw=1.2)
        ax_top.plot((-d, +d), (-d, +d), **kw)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kw)
        kw['transform'] = ax_bottom.transAxes
        ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kw)
        ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)

        # Labels and title
        ax_bottom.set_xlabel("Cycle")
        fig.text(0.02, 0.5, "Cumulative Hits", va='center',
                 rotation='vertical', fontsize=24)
        ax_top.set_title(dataset, pad=12)

        # Legend on top subplot only
        ax_top.legend(frameon=False, loc="upper left")

        plt.savefig(os.path.join(output_dir, f"Cumulative_Hits_Combo_{dataset}.png"),
                    dpi=600, bbox_inches="tight")
        plt.savefig(os.path.join(output_dir, f"Cumulative_Hits_Combo_{dataset}.pdf"),
                    bbox_inches="tight")
        plt.close()
        print(f"Saved Cumulative_Hits_Combo_{dataset}.png/pdf (broken axis)")

    else:
        # normal single-axis plot (CYP3A4)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_box_aspect(1)

        plot_on_ax(ax, all_data)
        style_ax(ax)

        ax.set_xlabel("Cycle")
        ax.set_ylabel("Cumulative Hits")
        ax.set_title(dataset, pad=12)
        ax.set_ylim(0, y_max_all * y_margin)

        ax.legend(frameon=False, loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Cumulative_Hits_Combo_{dataset}.png"),
                    dpi=600, bbox_inches="tight")
        plt.savefig(os.path.join(output_dir, f"Cumulative_Hits_Combo_{dataset}.pdf"),
                    dpi=600, bbox_inches="tight")
        plt.close()
        print(f"Saved Cumulative_Hits_Combo_{dataset}.png/pdf")

print("Done.")
