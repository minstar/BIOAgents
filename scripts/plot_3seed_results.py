#!/usr/bin/env python3
"""Plot 3-seed GRPO replication results with error bars.

Run after all 3 seeds (42, 123, 456) complete training and evaluation.
Generates publication-quality figure showing phase transition reproducibility.
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_BASE = "/data/project/private/minstar/workspace/BIOAgents/results/algorithm_comparison"
OUTPUT_DIR = "/data/project/private/minstar/workspace/BIOAgents/Neurips_2026_paper_writing/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Known results from seed 42 (the original GRPO baseline)
SEED42_MEDQA = {
    50: 39.4, 100: 57.3, 150: 57.3, 200: 57.2, 250: 57.2,
    300: 38.7, 350: 57.2, 400: 57.4, 450: 57.4, 500: 57.3,
}
SEED42_MEDMCQA = {
    50: 43.9, 100: 61.0, 150: 61.0, 200: 61.1, 250: 60.9,
    300: 42.9, 350: 61.0, 400: 61.1, 450: 61.0, 500: 61.2,
}
SEED42_MMLU = {
    50: 48.2, 100: 66.7, 150: 66.6, 200: 66.8, 250: 66.7,
    300: 47.1, 350: 66.9, 400: 66.8, 450: 66.6, 500: 66.9,
}


def load_seed_results(seed_name, steps):
    """Load textqa results for a given seed across checkpoint steps."""
    results = {}
    for step in steps:
        # Try different directory naming conventions
        patterns = [
            f"{seed_name}_step{step}",
            f"{seed_name}_step{step}_textqa",
            f"{seed_name}_checkpoint-{step}_textqa",
        ]
        for pattern in patterns:
            result_dir = os.path.join(RESULTS_BASE, pattern)
            if os.path.exists(result_dir):
                json_files = [f for f in os.listdir(result_dir)
                              if f.endswith('.json') and 'detail' not in f]
                for jf in json_files:
                    try:
                        with open(os.path.join(result_dir, jf)) as f:
                            d = json.load(f)
                        benchmarks = d.get('benchmarks', d)
                        if 'medqa' in benchmarks:
                            results[step] = {
                                'medqa': benchmarks['medqa']['accuracy'] * 100,
                                'medmcqa': benchmarks.get('medmcqa', {}).get('accuracy', 0) * 100,
                                'mmlu': benchmarks.get('mmlu_clinical', {}).get('accuracy', 0) * 100,
                            }
                            break
                    except Exception:
                        pass
    return results


def main():
    steps = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    epochs = [s / 400.0 for s in steps]  # Corrected: 400 steps = 1 epoch

    # Load seed results
    seed123 = load_seed_results("grpo_seed123", steps)
    seed456 = load_seed_results("grpo_seed456", steps)

    print(f"Seed 42 data: {len(SEED42_MEDQA)} checkpoints")
    print(f"Seed 123 data: {len(seed123)} checkpoints")
    print(f"Seed 456 data: {len(seed456)} checkpoints")

    if not seed123 and not seed456:
        print("\nNo seed 123/456 results found yet. Run evaluations first.")
        print("Expected result dirs: grpo_seed123_step{50..500}_textqa/")
        return

    # Compute means and stds
    medqa_means, medqa_stds = [], []
    medmcqa_means, medmcqa_stds = [], []
    mmlu_means, mmlu_stds = [], []

    for step in steps:
        vals_medqa = [SEED42_MEDQA.get(step)]
        vals_medmcqa = [SEED42_MEDMCQA.get(step)]
        vals_mmlu = [SEED42_MMLU.get(step)]

        if step in seed123:
            vals_medqa.append(seed123[step]['medqa'])
            vals_medmcqa.append(seed123[step]['medmcqa'])
            vals_mmlu.append(seed123[step]['mmlu'])

        if step in seed456:
            vals_medqa.append(seed456[step]['medqa'])
            vals_medmcqa.append(seed456[step]['medmcqa'])
            vals_mmlu.append(seed456[step]['mmlu'])

        vals_medqa = [v for v in vals_medqa if v is not None]
        vals_medmcqa = [v for v in vals_medmcqa if v is not None]
        vals_mmlu = [v for v in vals_mmlu if v is not None]

        medqa_means.append(np.mean(vals_medqa))
        medqa_stds.append(np.std(vals_medqa) if len(vals_medqa) > 1 else 0)
        medmcqa_means.append(np.mean(vals_medmcqa))
        medmcqa_stds.append(np.std(vals_medmcqa) if len(vals_medmcqa) > 1 else 0)
        mmlu_means.append(np.mean(vals_mmlu))
        mmlu_stds.append(np.std(vals_mmlu) if len(vals_mmlu) > 1 else 0)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, means, stds, title in [
        (axes[0], medqa_means, medqa_stds, 'MedQA'),
        (axes[1], medmcqa_means, medmcqa_stds, 'MedMCQA'),
        (axes[2], mmlu_means, mmlu_stds, 'MMLU-Med'),
    ]:
        ax.errorbar(epochs, means, yerr=stds, fmt='o-', color='#1f77b4',
                    linewidth=2, markersize=5, capsize=3, capthick=1.5,
                    label=f'GRPO (n={max(len([v for v in [True, bool(seed123), bool(seed456)] if v]), 1)} seeds)')
        ax.axhline(y=39.1, color='gray', linestyle='--', alpha=0.5, label='SFT-only')
        ax.axvspan(5.5, 6.5, alpha=0.1, color='red')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(30, 75)
        ax.set_xticks([0.125, 0.25, 0.5, 0.75, 1.0, 1.25])

    plt.suptitle('3-Seed GRPO Replication: Sub-Epoch Phase Transition',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, '3seed_replication.pdf')
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {outpath}")
    plt.close()

    # Print summary table
    print("\n=== 3-Seed Summary ===")
    print(f"{'Step':>5} {'Epoch':>5} {'MedQA':>15} {'MedMCQA':>15} {'MMLU-Med':>15}")
    for i, step in enumerate(steps):
        print(f"{step:>5} {epochs[i]:>5} "
              f"{medqa_means[i]:>7.1f}±{medqa_stds[i]:.1f} "
              f"{medmcqa_means[i]:>7.1f}±{medmcqa_stds[i]:.1f} "
              f"{mmlu_means[i]:>7.1f}±{mmlu_stds[i]:.1f}")


if __name__ == '__main__':
    main()
