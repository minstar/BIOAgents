#!/usr/bin/env python3
"""Plot training dynamics for all 4 GRPO-family algorithms.

Generates publication-quality figures for the NeurIPS paper showing:
1. Reward curves (accuracy + format)
2. KL divergence
3. Entropy
4. Loss
5. Learning rate schedule
"""
import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "/data/project/private/minstar/workspace/BIOAgents/results/training_dynamics"
OUTPUT_DIR = "/data/project/private/minstar/workspace/BIOAgents/Neurips_2026_paper_writing/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALGORITHMS = {
    'grpo_baseline_from_state.csv': ('GRPO', '#1f77b4', '-'),
    'dapo_only_from_state.csv': ('DAPO', '#ff7f0e', '--'),
    'gspo_only_from_state.csv': ('GSPO', '#2ca02c', '-.'),
    'drgrpo_from_state.csv': ('Dr.GRPO', '#d62728', ':'),
    'gspo_dapo_v2_from_state.csv': ('GSPO+DAPO', '#9467bd', '-'),
}


def load_csv(filepath):
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v) if v != '' else None
                except (ValueError, TypeError):
                    parsed[k] = None
            rows.append(parsed)
    return rows


def plot_metric(ax, all_data, metric, ylabel, title=None, ylim=None):
    for fname, (label, color, ls) in ALGORITHMS.items():
        if fname not in all_data:
            continue
        data = all_data[fname]
        epochs = [r['epoch'] for r in data if r.get(metric) is not None]
        values = [r[metric] for r in data if r.get(metric) is not None]
        if epochs:
            ax.plot(epochs, values, label=label, color=color, linestyle=ls,
                    linewidth=1.5, alpha=0.85)
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold')
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)


def main():
    # Load all data
    all_data = {}
    for fname in ALGORITHMS:
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            all_data[fname] = load_csv(path)
            print(f"Loaded {fname}: {len(all_data[fname])} entries")
        else:
            print(f"Missing: {fname}")

    # Figure 1: 2x2 grid of key training metrics
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle('Training Dynamics: GRPO-Family Algorithms on Medical QA',
                 fontsize=13, fontweight='bold', y=0.98)

    plot_metric(axes[0, 0], all_data, 'reward', 'Mean Reward',
                'Total Reward')
    plot_metric(axes[0, 1], all_data, 'kl', 'KL Divergence',
                'KL Divergence from Reference')
    plot_metric(axes[1, 0], all_data, 'entropy', 'Entropy',
                'Policy Entropy')
    plot_metric(axes[1, 1], all_data, 'loss', 'Loss',
                'Policy Gradient Loss')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    outpath = os.path.join(OUTPUT_DIR, 'training_dynamics_4panel.pdf')
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()

    # Figure 2: Accuracy reward + format reward (if available)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_metric(axes[0], all_data, 'accuracy_reward', 'Accuracy Reward',
                'Accuracy Reward')
    plot_metric(axes[1], all_data, 'format_reward', 'Format Reward',
                'Format Reward')
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, 'reward_decomposition.pdf')
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()

    # Figure 3: Grad norm comparison
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plot_metric(ax, all_data, 'grad_norm', 'Gradient Norm',
                'Gradient Norm During Training')
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, 'grad_norm.pdf')
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()

    # Figure 4: Learning rate schedule
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plot_metric(ax, all_data, 'learning_rate', 'Learning Rate',
                'Cosine Learning Rate Schedule')
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, 'lr_schedule.pdf')
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()

    # Figure 5: Combined evaluation results over training
    # This uses the actual eval scores from the paper
    plot_eval_trajectory()


def plot_eval_trajectory():
    """Plot the MedQA trajectory showing sub-epoch phase transition and catastrophic forgetting."""
    # Corrected mapping: 400 steps = 1 epoch (800 training tasks, batch_size=2)
    steps = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 800]
    epochs = [s / 400.0 for s in steps]  # correct epoch mapping

    # Updated with full eval data including collapse trajectories
    grpo_medqa = [39.1, 39.4, 57.3, 57.3, 57.2, 57.2, 38.7, 57.2, 57.4, 57.4, 57.3, 30.2]
    dapo_medqa = [39.1, 38.6, 57.5, 57.3, 57.4, 57.2, 38.5, 57.0, 57.0, 57.2, None, None]
    gspo_medqa = [39.1, 39.7, 57.3, 57.3, 57.1, 57.4, 37.8, 57.3, 40.2, 30.2, 30.2, None]
    drgrpo_medqa = [39.1, 39.9, 57.2, 57.4, 57.3, 57.3, 37.3, 38.6, 30.1, 30.0, 30.2, None]

    # Combined GSPO+DAPO (same 400 steps/epoch mapping)
    combined_steps = [0, 50, 100, 150, 200, 250, 300, 350, 450, 500, 550, 600, 650]
    combined_epochs = [s / 400.0 for s in combined_steps]
    combined_medqa = [39.1, None, 39.4, 39.5, 38.4, 57.4, 38.4, 39.4, 38.5, 37.9, 37.6, 38.0, 38.9]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Individual algorithms
    for data, label, color, ls in [
        (grpo_medqa, 'GRPO', '#1f77b4', '-'),
        (dapo_medqa, 'DAPO', '#ff7f0e', '--'),
        (gspo_medqa, 'GSPO', '#2ca02c', '-.'),
        (drgrpo_medqa, 'Dr.GRPO', '#d62728', ':'),
    ]:
        valid_epochs = [e for e, v in zip(epochs, data) if v is not None]
        valid_vals = [v for v in data if v is not None]
        ax1.plot(valid_epochs, valid_vals, label=label, color=color,
                 linestyle=ls, linewidth=2, marker='o', markersize=4)

    ax1.axhline(y=39.1, color='gray', linestyle='--', alpha=0.5, label='SFT-only')
    ax1.axvspan(0.7, 0.8, alpha=0.1, color='red', label='Step-300 dip')
    ax1.axvspan(1.8, 2.1, alpha=0.1, color='darkred', label='Catastrophic forgetting')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('MedQA Accuracy (%)', fontsize=11)
    ax1.set_title('Sub-Epoch Phase Transition & Algorithm-Dependent Collapse',
                   fontsize=11, fontweight='bold')
    ax1.legend(fontsize=7, loc='center right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(25, 65)
    ax1.set_xlim(-0.05, 2.15)

    # Right: GSPO+DAPO combined (permanent failure)
    valid_ce = [e for e, v in zip(combined_epochs, combined_medqa) if v is not None]
    valid_cv = [v for v in combined_medqa if v is not None]
    ax2.plot(valid_ce, valid_cv, label='GSPO+DAPO', color='#9467bd',
             linewidth=2, marker='s', markersize=4)
    ax2.axhline(y=39.1, color='gray', linestyle='--', alpha=0.5, label='SFT-only')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('MedQA Accuracy (%)', fontsize=11)
    ax2.set_title('GSPO+DAPO Combined: Permanent Failure',
                   fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(25, 65)

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, 'eval_trajectory.pdf')
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()


if __name__ == '__main__':
    main()
