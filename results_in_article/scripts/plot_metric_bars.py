import csv
import os
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
    "legend.fontsize": 20, "axes.linewidth": 1.0,
})

datasets = ["CYP3A4", "PKM2", "TP53"]
strategy_order = ["exploitation", "bald", "uncertainty", "bala", "similarity", "random"]
strategy_labels = ["Exploitation", "Exploration-BALD", "Exploration-Uncertainty", "BALA", "Similarity", "Random"]

metric_map = {
    "Recall": "Test Recall",
    "Specificity": "Test Specificity",
    "AUROC": "Test AUROC",
    "AUPRC": "Test AUPRC",
    "Cumulative Hits": "Hits Discovered",
}

metric_colors = {
    "Recall":           "#3D9F3C",
    "Specificity":      "#9ED17B",
    "AUROC":            "#367DB0",
    "AUPRC":            "#9DC7DD",
    "Cumulative Hits":  "#7872B9",
}

output_root = os.path.join(_FIGURES, "metric_bars_output")


def parse_csv(filepath):
    with open(filepath) as f:
        reader = csv.reader(f)
        headers = next(reader)

        col_idx = {}
        for i, h in enumerate(headers):
            h = h.strip()
            if h == "Method":
                col_idx["Method"] = i
            elif h == "Test Recall" and "Test Recall" not in col_idx:
                col_idx["Test Recall"] = i
            elif h == "Test Specificity" and "Test Specificity" not in col_idx:
                col_idx["Test Specificity"] = i
            elif h == "Test AUROC" and "Test AUROC" not in col_idx:
                col_idx["Test AUROC"] = i
            elif h == "Test AUPRC" and "Test AUPRC" not in col_idx:
                col_idx["Test AUPRC"] = i
            elif h == "Hits Discovered" and "Hits Discovered" not in col_idx:
                col_idx["Hits Discovered"] = i

        data = {}
        for row in reader:
            if len(row) <= col_idx["Method"]:
                continue
            method = row[col_idx["Method"]].strip().lower()
            if method not in strategy_order:
                continue

            data[method] = {}
            for display_metric, csv_col in metric_map.items():
                idx = col_idx[csv_col]
                val_str = row[idx].strip()
                mean_str, std_str = val_str.split("±")
                data[method][display_metric] = (float(mean_str), float(std_str))

        return data


for dataset in datasets:
    csv_path = os.path.join(
        _RESULTS, "20251014_strategy_experiments",
        f"{dataset}_metrics_results", f"{dataset}_cycle15_summary.csv")
    raw_data = parse_csv(csv_path)

    means = {m: [] for m in metric_map}
    stds = {m: [] for m in metric_map}
    for strategy in strategy_order:
        if strategy in raw_data:
            for metric in metric_map:
                m, s = raw_data[strategy][metric]
                means[metric].append(m)
                stds[metric].append(s)
        else:
            for metric in metric_map:
                means[metric].append(0)
                stds[metric].append(0)

    out_dir = os.path.join(output_root, dataset)
    os.makedirs(out_dir, exist_ok=True)

    x = np.arange(len(strategy_order))
    metric_order = ["Recall", "Specificity", "AUROC", "AUPRC", "Cumulative Hits"]

    for metric in metric_order:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.bar(x, means[metric], color=metric_colors[metric], edgecolor="white",
               linewidth=1.2, yerr=stds[metric], capsize=6,
               error_kw={"linewidth": 1.5, "ecolor": "black"})

        ax.set_xticks(x)
        ax.set_xticklabels(strategy_labels, rotation=45, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(dataset, pad=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        all_vals = np.array(means[metric])
        all_errs = np.array(stds[metric])
        if metric == "Cumulative Hits":
            y_min = 0
            y_max = (all_vals + all_errs).max() + 100
        else:
            y_min = 0
            y_max = (all_vals + all_errs).max() + 0.04
        ax.set_ylim(y_min, y_max)

        plt.tight_layout()
        plt.savefig(f"{out_dir}/Bar_{metric}.png", dpi=600, bbox_inches="tight")
        plt.savefig(f"{out_dir}/Bar_{metric}.pdf", bbox_inches="tight")
        plt.close()
        print(f"Saved {out_dir}/Bar_{metric}.png/pdf")

print("Done.")
