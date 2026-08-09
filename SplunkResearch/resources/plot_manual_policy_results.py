#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results():
    """Load all manual policy evaluation results from both hosts."""
    base_path = Path("/home/shouei/GreenSecurityMeasurementAndOptimizationFramework/SplunkResearch")

    results = []

    # Load from both hosts
    for host_dir in ["host_132.72.81.184_experiments", "host_132.72.80.159_experiments"]:
        host_path = base_path / host_dir / "runs"

        # Find all manual_* directories
        for run_dir in sorted(host_path.glob("manual_*")):
            results_file = run_dir / "results" / "full_eval_results_random_False.csv"

            if results_file.exists():
                df = pd.read_csv(results_file)

                # Extract policy and diversity from run name
                run_name = run_dir.name
                if "only_4662" in run_name:
                    policy = "only_4662"
                    diversity = 1.0  # baseline
                else:
                    policy = "all_relevant"
                    # Extract diversity: manual_all_relevant_d0.000_...
                    parts = run_name.split("_d")
                    if len(parts) > 1:
                        diversity = float(parts[1].split("_")[0])
                    else:
                        diversity = None

                # Add metadata columns
                df["policy"] = policy
                df["diversity"] = diversity
                df["host"] = host_dir.split("_")[1]  # Extract IP

                results.append(df)

    return pd.concat(results, ignore_index=True)

def aggregate_results(df):
    """Aggregate results by policy and diversity."""
    agg_dict = {
        "cpu": "mean",
        "baseline_cpu": "mean",
        "duration": "mean",
        "baseline_duration": "mean",
        "alert": "mean",
        "baseline_alert": "mean",
        "memory_mb": "mean",
        "baseline_memory_mb": "mean",
        "read_bytes": "mean",
        "baseline_read_bytes": "mean",
        "write_bytes": "mean",
        "baseline_write_bytes": "mean",
    }

    # Group by policy and diversity (combining both hosts)
    grouped = df.groupby(["policy", "diversity"]).agg(agg_dict).reset_index()

    return grouped

def plot_results(grouped_df):
    """Create bar charts for key metrics."""
    # Sort by policy and diversity
    grouped_df = grouped_df.sort_values(["policy", "diversity"])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Manual Policy Evaluation Results (Aggregated across both hosts)", fontsize=14, fontweight="bold")

    # Prepare data for plotting
    labels = []
    cpu_vals = []
    cpu_baseline = []
    duration_vals = []
    duration_baseline = []
    alert_vals = []
    alert_baseline = []

    for _, row in grouped_df.iterrows():
        if row["policy"] == "only_4662":
            label = "only_4662\n(baseline)"
        else:
            label = f"all_relevant\nd={row['diversity']:.3f}"
        labels.append(label)

        cpu_vals.append(row["cpu"])
        cpu_baseline.append(row["baseline_cpu"])
        duration_vals.append(row["duration"])
        duration_baseline.append(row["baseline_duration"])
        alert_vals.append(row["alert"])
        alert_baseline.append(row["baseline_alert"])

    x = np.arange(len(labels))
    width = 0.35

    # CPU Usage
    axes[0, 0].bar(x - width/2, cpu_vals, width, label="Attack (avg CPU%)", alpha=0.8)
    axes[0, 0].bar(x + width/2, cpu_baseline, width, label="Baseline (avg CPU%)", alpha=0.8)
    axes[0, 0].set_ylabel("CPU %")
    axes[0, 0].set_title("CPU Usage")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, fontsize=9)
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.3)

    # Duration
    axes[0, 1].bar(x - width/2, duration_vals, width, label="Attack (avg seconds)", alpha=0.8)
    axes[0, 1].bar(x + width/2, duration_baseline, width, label="Baseline (avg seconds)", alpha=0.8)
    axes[0, 1].set_ylabel("Seconds")
    axes[0, 1].set_title("Execution Duration")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, fontsize=9)
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.3)

    # Alerts
    axes[1, 0].bar(x - width/2, alert_vals, width, label="Attack (avg alerts)", alpha=0.8)
    axes[1, 0].bar(x + width/2, alert_baseline, width, label="Baseline (avg alerts)", alpha=0.8)
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Detection Alerts")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, fontsize=9)
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.3)

    # CPU overhead ratio (attack / baseline)
    cpu_ratio = [c / b if b > 0 else 0 for c, b in zip(cpu_vals, cpu_baseline)]
    axes[1, 1].bar(x, cpu_ratio, alpha=0.8, color="orange")
    axes[1, 1].axhline(y=1, color="red", linestyle="--", label="Baseline (1x)")
    axes[1, 1].set_ylabel("Ratio (Attack / Baseline)")
    axes[1, 1].set_title("CPU Overhead Ratio")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, fontsize=9)
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_path = Path("/home/shouei/GreenSecurityMeasurementAndOptimizationFramework/SplunkResearch/resources/manual_policy_results.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {output_path}")

    # Also print summary table
    print("\n" + "="*80)
    print("AGGREGATED RESULTS SUMMARY")
    print("="*80)
    summary = grouped_df[["policy", "diversity", "cpu", "baseline_cpu", "duration", "alert", "baseline_alert"]].copy()
    summary["cpu_ratio"] = summary["cpu"] / summary["baseline_cpu"]
    print(summary.to_string(index=False))

    return grouped_df

if __name__ == "__main__":
    print("Loading results from both hosts...")
    df = load_results()
    print(f"Loaded {len(df)} episodes")

    print("Aggregating results...")
    grouped = aggregate_results(df)
    print(f"Aggregated into {len(grouped)} configurations")

    print("Creating visualizations...")
    plot_results(grouped)
    print("\nDone!")
