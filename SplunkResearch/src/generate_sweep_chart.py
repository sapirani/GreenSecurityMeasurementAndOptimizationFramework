#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import re
from datetime import datetime

# Configuration
base_dir = '/home/shouei/GreenSecurityMeasurementAndOptimizationFramework/SplunkResearch/host_132.72.80.159_experiments/runs'

# Load only_4662 results (with diversity suffix)
only_4662_files = sorted(glob.glob(os.path.join(base_dir, 'manual_only_4662_d*/results/full_eval_results_random_False.csv')))
only_4662_data = []

for f in only_4662_files:
    df = pd.read_csv(f)
    match = re.search(r'manual_only_4662_d([0-9.]+)', f)
    if match:
        diversity = float(match.group(1))
        df['policy'] = 'only_4662'
        df['diversity'] = diversity
        only_4662_data.append(df)

# Load all_relevant results
all_relevant_files = sorted(glob.glob(os.path.join(base_dir, 'manual_all_relevant_d*/results/full_eval_results_random_False.csv')))
all_relevant_data = []

for f in all_relevant_files:
    df = pd.read_csv(f)
    match = re.search(r'manual_all_relevant_d([0-9.]+)', f)
    if match:
        diversity = float(match.group(1))
        df['policy'] = 'all_relevant'
        df['diversity'] = diversity
        all_relevant_data.append(df)

# Load eval_post_training (trained agent) results
eval_files = sorted(glob.glob(os.path.join(base_dir, 'eval_post_training_*/results/full_eval_results_random_False.csv')))
eval_data = []

for i, f in enumerate(eval_files):
    df = pd.read_csv(f)
    df['policy'] = 'trained_agent'
    # Extract hosts_number and additional_percentage from first row
    hosts = int(df.iloc[0]['hosts_number'])
    addit = df.iloc[0]['additional_percentage']
    # Use tuple as diversity key for sorting/grouping (will be converted to string for display)
    df['diversity'] = i
    df['config_label'] = f"h{hosts}a{addit:.1f}"
    eval_data.append(df)

# Combine all data
all_results = pd.concat(only_4662_data + all_relevant_data + eval_data, ignore_index=True)

# Calculate metrics
all_results['cpu_overhead_ratio'] = all_results['cpu'] / all_results['baseline_cpu']
all_results['mean_alerts'] = all_results['alert']
all_results['ac_distribution'] = all_results['full_ac_distribution_value']

# Aggregate by policy and diversity
summary = all_results.groupby(['policy', 'diversity']).agg({
    'cpu_overhead_ratio': 'mean',
    'mean_alerts': 'mean',
    'ac_distribution': 'mean'
}).reset_index()

print("Summary Statistics:")
print(summary)
print()

# Create figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(16, 12))
fig.suptitle(f'50-Episode Sweep — 2026-08-10/11 (only_4662 | all_relevant)', fontsize=16, fontweight='bold')

# Color mapping
colors = {
    'only_4662': '#E8A868',      # Orange
    'all_relevant': '#7BA3D6',   # Blue
    'trained_agent': '#7CB342'   # Green
}

x_labels_4662 = [f'd={d:.2f}' for d in sorted(summary[summary['policy']=='only_4662']['diversity'].unique())]
x_labels_relevant = [f'd={d:.3f}' for d in sorted(summary[summary['policy']=='all_relevant']['diversity'].unique())]
# Extract config labels from eval data (h{hosts}a{additional})
eval_configs = []
for df in eval_data:
    if not df.empty:
        label = df.iloc[0].get('config_label', f'eval_{len(eval_configs)}')
        eval_configs.append(label)
x_labels_agent = eval_configs if eval_configs else [f'eval_{i}' for i in range(len(summary[summary['policy']=='trained_agent']))]

# ===== Panel 1: CPU Overhead Ratio =====
ax = axes[0]
only_4662_summary = summary[summary['policy']=='only_4662'].sort_values('diversity')
all_relevant_summary = summary[summary['policy']=='all_relevant'].sort_values('diversity')
agent_summary = summary[summary['policy']=='trained_agent'].sort_values('diversity')

x_pos = np.arange(len(x_labels_4662) + len(x_labels_relevant) + len(x_labels_agent) + 2)  # +2 for gaps
width = 0.7

# Plot only_4662
x_4662 = np.arange(len(x_labels_4662))
bars1 = ax.bar(x_4662, only_4662_summary['cpu_overhead_ratio'].values, width,
               label='only_4662 (diversity sweep)', color=colors['only_4662'], alpha=0.9)

# Plot all_relevant
x_relevant = np.arange(len(x_labels_4662) + 1, len(x_labels_4662) + 1 + len(x_labels_relevant))
bars2 = ax.bar(x_relevant, all_relevant_summary['cpu_overhead_ratio'].values, width,
               label='all_relevant (diversity sweep)', color=colors['all_relevant'], alpha=0.9)

# Plot trained agent
x_agent = np.arange(len(x_labels_4662) + len(x_labels_relevant) + 2,
                    len(x_labels_4662) + len(x_labels_relevant) + 2 + len(x_labels_agent))
bars3 = ax.bar(x_agent, agent_summary['cpu_overhead_ratio'].values, width,
               label='Trained agent (eval_post_training)', color=colors['trained_agent'], alpha=0.9)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}x',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add baseline line
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (1x)', alpha=0.7)

ax.set_ylabel('CPU Ratio (injected / baseline)', fontsize=11, fontweight='bold')
ax.set_title('CPU Overhead Ratio', fontsize=12, fontweight='bold')
all_x = list(x_4662) + list(x_relevant) + list(x_agent)
all_labels = x_labels_4662 + x_labels_relevant + x_labels_agent
ax.set_xticks(all_x)
ax.set_xticklabels(all_labels, fontsize=9)
ax.set_ylim(0, max(only_4662_summary['cpu_overhead_ratio'].max(),
                   all_relevant_summary['cpu_overhead_ratio'].max(),
                   agent_summary['cpu_overhead_ratio'].max()) * 1.15)
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Add background colors for policy regions
ax.axvspan(-0.5, len(x_labels_4662)-0.5, alpha=0.05, color='orange')
ax.axvspan(len(x_labels_4662)+0.5, len(x_labels_4662)+len(x_labels_relevant)+0.5, alpha=0.05, color='blue')
ax.axvspan(len(x_labels_4662)+len(x_labels_relevant)+1.5, len(x_labels_4662)+len(x_labels_relevant)+1.5+len(x_labels_agent), alpha=0.05, color='green')

# ===== Panel 2: Mean Alerts Triggered =====
ax = axes[1]
bars1 = ax.bar(x_4662, only_4662_summary['mean_alerts'].values, width,
               label='only_4662 (diversity sweep)', color=colors['only_4662'], alpha=0.9)
bars2 = ax.bar(x_relevant, all_relevant_summary['mean_alerts'].values, width,
               label='all_relevant (diversity sweep)', color=colors['all_relevant'], alpha=0.9)
bars3 = ax.bar(x_agent, agent_summary['mean_alerts'].values, width,
               label='Trained agent (eval_post_training)', color=colors['trained_agent'], alpha=0.9)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Mean alert count / episode', fontsize=11, fontweight='bold')
ax.set_title('Mean Alerts Triggered', fontsize=12, fontweight='bold')
ax.set_xticks(all_x)
ax.set_xticklabels(all_labels, fontsize=9)
ax.set_ylim(0, max(only_4662_summary['mean_alerts'].max(),
                   all_relevant_summary['mean_alerts'].max(),
                   agent_summary['mean_alerts'].max()) * 1.15)
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Add background colors for policy regions
ax.axvspan(-0.5, len(x_labels_4662)-0.5, alpha=0.05, color='orange')
ax.axvspan(len(x_labels_4662)+0.5, len(x_labels_4662)+len(x_labels_relevant)+0.5, alpha=0.05, color='blue')
ax.axvspan(len(x_labels_4662)+len(x_labels_relevant)+1.5, len(x_labels_4662)+len(x_labels_relevant)+1.5+len(x_labels_agent), alpha=0.05, color='green')

# ===== Panel 3: Full AC Distribution Value =====
ax = axes[2]
bars1 = ax.bar(x_4662, only_4662_summary['ac_distribution'].values, width,
               label='only_4662 (diversity sweep)', color=colors['only_4662'], alpha=0.9)
bars2 = ax.bar(x_relevant, all_relevant_summary['ac_distribution'].values, width,
               label='all_relevant (diversity sweep)', color=colors['all_relevant'], alpha=0.9)
bars3 = ax.bar(x_agent, agent_summary['ac_distribution'].values, width,
               label='Trained agent (eval_post_training)', color=colors['trained_agent'], alpha=0.9)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Mean AC distribution value', fontsize=11, fontweight='bold')
ax.set_title('Full AC Distribution Value (1.0 = perfect spread)', fontsize=12, fontweight='bold')
ax.set_xticks(all_x)
ax.set_xticklabels(all_labels, fontsize=9)
ax.set_ylim(0, 1.2)
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Add background colors for policy regions
ax.axvspan(-0.5, len(x_labels_4662)-0.5, alpha=0.05, color='orange')
ax.axvspan(len(x_labels_4662)+0.5, len(x_labels_4662)+len(x_labels_relevant)+0.5, alpha=0.05, color='blue')
ax.axvspan(len(x_labels_4662)+len(x_labels_relevant)+1.5, len(x_labels_4662)+len(x_labels_relevant)+1.5+len(x_labels_agent), alpha=0.05, color='green')

plt.tight_layout()
plt.savefig('/tmp/claude-343819017/-home-shouei-GreenSecurityMeasurementAndOptimizationFramework/b687be0a-9f4b-4f9d-923d-bf5ee2a286f4/scratchpad/sweep_comparison_chart.png', dpi=150, bbox_inches='tight')
print("\n✓ Chart saved to sweep_comparison_chart.png")

plt.show()

# Print detailed statistics
print("\n=== DETAILED STATISTICS ===\n")
print("Only 4662:")
for _, row in only_4662_summary.iterrows():
    print(f"  d={row['diversity']:.2f}: CPU={row['cpu_overhead_ratio']:.2f}x, Alerts={row['mean_alerts']:.0f}, AC={row['ac_distribution']:.2f}")

print("\nAll Relevant:")
for _, row in all_relevant_summary.iterrows():
    print(f"  d={row['diversity']:.3f}: CPU={row['cpu_overhead_ratio']:.2f}x, Alerts={row['mean_alerts']:.0f}, AC={row['ac_distribution']:.2f}")
