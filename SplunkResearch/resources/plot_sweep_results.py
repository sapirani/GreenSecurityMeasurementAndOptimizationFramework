#!/usr/bin/env python3
"""Bar chart for the 2026-05-06 50-episode sweep results.

Groups:
  1. Manual policy: only_4662 (host 184) — diversity 0.00 → 1.00
  2. Manual policy: all_relevant (host 159) — diversity 0.00 → 1.00
  3. Trained agent sweep (hosts 159 + 184) — various hosts_num × add_pct combos
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

BASE = Path("/home/shouei/GreenSecurityMeasurementAndOptimizationFramework/SplunkResearch")

# ---------------------------------------------------------------------------
# Data definitions  (run dir → label metadata)
# ---------------------------------------------------------------------------

ONLY_4662 = [
    ("host_132.72.81.184_experiments/runs/manual_only_4662_20260506123136", 0.00),
    ("host_132.72.81.184_experiments/runs/manual_only_4662_20260506133015", 0.25),
    ("host_132.72.81.184_experiments/runs/manual_only_4662_20260506142857", 0.50),
    ("host_132.72.81.184_experiments/runs/manual_only_4662_20260506153202", 0.75),
    ("host_132.72.81.184_experiments/runs/manual_only_4662_20260506163503", 1.00),
]

ALL_RELEVANT = [
    ("host_132.72.80.159_experiments/runs/manual_all_relevant_d0.000_20260506122555", 0.00),
    ("host_132.72.80.159_experiments/runs/manual_all_relevant_d0.250_20260506130759", 0.25),
    ("host_132.72.80.159_experiments/runs/manual_all_relevant_d0.500_20260506135258", 0.50),
    ("host_132.72.80.159_experiments/runs/manual_all_relevant_d0.750_20260506143740", 0.75),
    ("host_132.72.80.159_experiments/runs/manual_all_relevant_d1.000_20260506152345", 1.00),
]

# (run dir, hosts_num, add_pct, host_id)
AGENT_RUNS = [
    # host 159
    ("host_132.72.80.159_experiments/runs/eval_post_training_20260506202137",  10, 0.1, 159),
    ("host_132.72.80.159_experiments/runs/eval_post_training_20260506204304",  10, 0.5, 159),
    ("host_132.72.80.159_experiments/runs/eval_post_training_20260506211229",  10, 1.0, 159),
    ("host_132.72.80.159_experiments/runs/eval_post_training_20260506215317",  50, 0.1, 159),
    ("host_132.72.80.159_experiments/runs/eval_post_training_20260506221629",  50, 0.5, 159),
    # host 184
    ("host_132.72.81.184_experiments/runs/eval_post_training_20260506202158",  50, 1.0, 184),
    ("host_132.72.81.184_experiments/runs/eval_post_training_20260506211722", 100, 0.1, 184),
    ("host_132.72.81.184_experiments/runs/eval_post_training_20260506214711", 100, 0.5, 184),
    ("host_132.72.81.184_experiments/runs/eval_post_training_20260506223421", 100, 1.0, 184),
]


def load_mean(run_dir):
    f = BASE / run_dir / "results" / "full_eval_results_random_False.csv"
    df = pd.read_csv(f)
    return df[["cpu", "baseline_cpu", "alert", "baseline_alert", "full_ac_distribution_value"]].mean()


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def build_rows():
    rows = []

    for path, div in ONLY_4662:
        m = load_mean(path)
        rows.append(dict(group="only_4662", label=f"d={div:.2f}", sort=(0, div),
                         cpu=m.cpu, base_cpu=m.baseline_cpu,
                         alert=m.alert, base_alert=m.baseline_alert,
                         dist=m.full_ac_distribution_value))

    for path, div in ALL_RELEVANT:
        m = load_mean(path)
        rows.append(dict(group="all_relevant", label=f"d={div:.2f}", sort=(1, div),
                         cpu=m.cpu, base_cpu=m.baseline_cpu,
                         alert=m.alert, base_alert=m.baseline_alert,
                         dist=m.full_ac_distribution_value))

    for path, hosts, add, host_id in AGENT_RUNS:
        m = load_mean(path)
        rows.append(dict(group="agent", label=f"h{hosts}\n+{add}", sort=(2, hosts + add),
                         cpu=m.cpu, base_cpu=m.baseline_cpu,
                         alert=m.alert, base_alert=m.baseline_alert,
                         dist=m.full_ac_distribution_value))

    return sorted(rows, key=lambda r: r["sort"])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_chart(rows):
    labels = [r["label"] for r in rows]
    groups = [r["group"] for r in rows]
    n = len(rows)
    x = np.arange(n)

    # Color palette per group
    COLOR = {
        "only_4662":   "#e07b39",   # orange family
        "all_relevant":"#3a87c8",   # blue family
        "agent":       "#4caf50",   # green family
    }
    # Within a group lighten from first (darkest) to last (lightest)
    def group_shades(group_name, base_hex, count):
        import matplotlib.colors as mc
        base = mc.to_rgb(base_hex)
        shades = []
        for i in range(count):
            factor = 0.55 + 0.45 * (i / max(count - 1, 1))
            shades.append(tuple(min(1.0, c * factor + (1 - factor)) for c in base))
        return shades

    counts = {}
    for g in ["only_4662", "all_relevant", "agent"]:
        counts[g] = sum(1 for r in rows if r["group"] == g)

    group_shade_map = {}
    for g, base in COLOR.items():
        shades = group_shades(g, base, counts[g])
        idx = 0
        for i, r in enumerate(rows):
            if r["group"] == g:
                group_shade_map[i] = shades[idx]
                idx += 1

    bar_colors = [group_shade_map[i] for i in range(n)]

    fig, axes = plt.subplots(3, 1, figsize=(18, 13), sharex=True)
    fig.suptitle("50-Episode Sweep — 2026-05-06  (only_4662 | all_relevant | trained agent)",
                 fontsize=14, fontweight="bold", y=0.98)

    width = 0.7

    def add_group_spans(ax):
        prev_g, start = groups[0], 0
        for i, g in enumerate(groups):
            if g != prev_g or i == n - 1:
                end = i if g != prev_g else i + 1
                color = {"only_4662": "#ffe8d6", "all_relevant": "#d6e8ff", "agent": "#d6f5d6"}[prev_g]
                ax.axvspan(start - 0.5, end - 0.5, alpha=0.25, color=color, zorder=0)
                prev_g, start = g, i

    # — CPU ratio —
    ax = axes[0]
    cpu_ratio = [r["cpu"] / r["base_cpu"] if r["base_cpu"] > 0 else 0 for r in rows]
    bars = ax.bar(x, cpu_ratio, width=width, color=bar_colors, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Baseline (1×)")
    add_group_spans(ax)
    ax.set_ylabel("CPU ratio (attack / baseline)", fontsize=10)
    ax.set_title("CPU Overhead Ratio", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.4, zorder=0)
    for bar, v in zip(bars, cpu_ratio):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}×", ha="center", va="bottom", fontsize=7.5)

    # — Mean alerts —
    ax = axes[1]
    alert_vals = [r["alert"] for r in rows]
    bars = ax.bar(x, alert_vals, width=width, color=bar_colors, edgecolor="white", linewidth=0.5, zorder=3)
    add_group_spans(ax)
    ax.set_ylabel("Mean alert count / episode", fontsize=10)
    ax.set_title("Mean Alerts Triggered", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.4, zorder=0)
    for bar, v in zip(bars, alert_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)

    # — Distribution value —
    ax = axes[2]
    dist_vals = [r["dist"] for r in rows]
    bars = ax.bar(x, dist_vals, width=width, color=bar_colors, edgecolor="white", linewidth=0.5, zorder=3)
    add_group_spans(ax)
    ax.set_ylabel("Mean AC distribution value", fontsize=10)
    ax.set_title("Full AC Distribution Value (1.0 = perfect spread)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.2)
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1)
    ax.grid(axis="y", alpha=0.4, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    for bar, v in zip(bars, dist_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

    # Group separators + x-axis group labels
    group_bounds = {}
    for i, g in enumerate(groups):
        group_bounds.setdefault(g, []).append(i)
    for g, idxs in group_bounds.items():
        mid = np.mean(idxs)
        color_map = {"only_4662": "#c04000", "all_relevant": "#1a4f8a", "agent": "#1a6b1a"}
        axes[2].text(mid, -0.42, {"only_4662": "only_4662", "all_relevant": "all_relevant",
                                   "agent": "trained agent"}[g],
                     ha="center", va="top", fontsize=10, fontweight="bold",
                     color=color_map[g], transform=axes[2].get_xaxis_transform())

    # Legend patches
    patches = [
        mpatches.Patch(color=COLOR["only_4662"],   label="only_4662 (diversity sweep)"),
        mpatches.Patch(color=COLOR["all_relevant"], label="all_relevant (diversity sweep)"),
        mpatches.Patch(color=COLOR["agent"],        label="Trained agent (hosts × add% grid)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.005))

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    out = BASE / "resources" / "sweep_results_20260506.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    return out


if __name__ == "__main__":
    rows = build_rows()
    print(f"Loaded {len(rows)} experiments")
    out = make_chart(rows)
