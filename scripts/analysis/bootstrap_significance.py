#!/usr/bin/env python3
"""
Bootstrap confidence intervals and TOST equivalence testing for medical QA evaluation results.

Computes:
  - Bootstrap 95% CI for difference in accuracy
  - Binomial exact test (two-sided)
  - TOST equivalence test (delta=1pp)
  - McNemar's test (when per-sample data available)
  - Standard error of proportion

Usage:
  # Run with default paper numbers (MedQA + MedMCQA)
  python bootstrap_significance.py

  # Run with custom accuracy numbers
  python bootstrap_significance.py --baseline-acc 0.607 --baseline-n 1273 \
      --treatment-acc 0.624 --treatment-n 1273

  # Run with per-sample JSON predictions
  python bootstrap_significance.py --json results.json

JSON format expected:
  {
    "baseline": [0, 1, 1, 0, ...],      # per-sample correct/incorrect
    "treatment": [1, 1, 0, 1, ...]
  }
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

SEED = 42
N_BOOTSTRAP = 10000
ALPHA = 0.05
TOST_DELTA = 0.01  # 1 percentage point

# Default paper numbers
PAPER_BENCHMARKS = [
    {
        "name": "MedQA",
        "baseline_acc": 0.607,
        "treatment_acc": 0.624,
        "n": 1273,
        "treatment_label": "H12 (best RL)",
    },
    {
        "name": "MedMCQA",
        "baseline_acc": 0.525,
        "treatment_acc": 0.534,
        "n": 4183,
        "treatment_label": "best RL",
    },
]

LOG_DIR = Path("/data/project/private/minstar/workspace/BIOAgents/logs/analysis")


@dataclass(frozen=True)
class SignificanceResult:
    benchmark: str
    baseline_acc: float
    treatment_acc: float
    n_baseline: int
    n_treatment: int
    diff_pp: float
    se_baseline: float
    se_treatment: float
    se_diff: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    bootstrap_mean_diff: float
    p_value_twosided: float
    significant_at_alpha: bool
    tost_p_upper: float
    tost_p_lower: float
    tost_delta: float
    tost_equivalent: bool
    mcnemar_chi2: Optional[float]
    mcnemar_p: Optional[float]


# ---------------------------------------------------------------------------
# Statistical helpers (numpy + stdlib only)
# ---------------------------------------------------------------------------

def se_proportion(p: float, n: int) -> float:
    """Standard error of a proportion."""
    return math.sqrt(p * (1.0 - p) / n)


def se_diff_proportions(p1: float, n1: int, p2: float, n2: int) -> float:
    """Standard error of difference between two independent proportions."""
    return math.sqrt(p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2)


def _normal_cdf(z: float) -> float:
    """Standard normal CDF using the error function from math."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _two_sided_p_from_z(z: float) -> float:
    """Two-sided p-value from a z-statistic."""
    return 2.0 * (1.0 - _normal_cdf(abs(z)))


def _one_sided_p_from_z(z: float) -> float:
    """One-sided p-value P(Z > z)."""
    return 1.0 - _normal_cdf(z)


def bootstrap_ci_from_counts(
    p1: float,
    n1: int,
    p2: float,
    n2: int,
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = ALPHA,
    rng: Optional[np.random.Generator] = None,
) -> tuple:
    """
    Bootstrap 95% CI for difference (p2 - p1) by resampling Bernoulli draws.

    Returns (ci_lower, ci_upper, mean_diff, diffs_array).
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    # Simulate binary outcomes from the observed proportions
    baseline_samples = rng.binomial(n1, p1, size=n_bootstrap) / n1
    treatment_samples = rng.binomial(n2, p2, size=n_bootstrap) / n2
    diffs = treatment_samples - baseline_samples

    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    mean_diff = float(np.mean(diffs))
    return lo, hi, mean_diff, diffs


def bootstrap_ci_from_samples(
    baseline: np.ndarray,
    treatment: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = ALPHA,
    rng: Optional[np.random.Generator] = None,
) -> tuple:
    """
    Bootstrap 95% CI for difference by resampling paired per-sample data.

    Returns (ci_lower, ci_upper, mean_diff, diffs_array).
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    n = len(baseline)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[i] = treatment[idx].mean() - baseline[idx].mean()

    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    mean_diff = float(np.mean(diffs))
    return lo, hi, mean_diff, diffs


def binomial_z_test(p1: float, n1: int, p2: float, n2: int) -> float:
    """Two-sided z-test for difference of two proportions. Returns p-value."""
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1.0 / n1 + 1.0 / n2))
    if se < 1e-15:
        return 1.0
    z = (p2 - p1) / se
    return _two_sided_p_from_z(z)


def tost_test(
    p1: float,
    n1: int,
    p2: float,
    n2: int,
    delta: float = TOST_DELTA,
) -> tuple:
    """
    Two One-Sided Tests for equivalence within +/- delta.

    H0: |p2 - p1| >= delta  (not equivalent)
    H1: |p2 - p1| < delta   (equivalent)

    Returns (p_upper, p_lower, equivalent_at_alpha).
    The overall TOST p-value is max(p_upper, p_lower).
    """
    diff = p2 - p1
    se = se_diff_proportions(p1, n1, p2, n2)
    if se < 1e-15:
        return (0.0, 0.0, True)

    # Test 1: H0: diff >= delta  => z1 = (diff - delta) / se, reject if z1 << 0
    z_upper = (diff - delta) / se
    p_upper = _normal_cdf(z_upper)  # P(Z < z_upper)

    # Test 2: H0: diff <= -delta => z2 = (diff + delta) / se, reject if z2 >> 0
    z_lower = (diff + delta) / se
    p_lower = _one_sided_p_from_z(z_lower)  # P(Z > z_lower) ... wait

    # Correct formulation:
    # For TOST, we need:
    #   p_lower = P(Z > z_lower) where z_lower tests diff <= -delta
    #   Actually: p_lower = 1 - Phi(z_lower) tests H0: diff <= -delta
    #   p_upper = Phi(z_upper) tests H0: diff >= delta
    # No — standard TOST:
    #   t1 = (diff - (-delta)) / se = (diff + delta) / se → reject H0_lower if t1 large → p1 = 1 - Phi(t1)
    #   t2 = (diff - delta) / se → reject H0_upper if t2 small (negative) → p2 = Phi(t2)

    t1 = (diff + delta) / se  # tests H0: diff <= -delta
    t2 = (diff - delta) / se  # tests H0: diff >= +delta

    p_lower = 1.0 - _normal_cdf(t1)  # one-sided: reject when t1 is large
    p_upper = _normal_cdf(t2)        # one-sided: reject when t2 is very negative

    tost_p = max(p_lower, p_upper)
    equivalent = tost_p < ALPHA
    return p_upper, p_lower, equivalent


def mcnemar_test(baseline: np.ndarray, treatment: np.ndarray) -> tuple:
    """
    McNemar's test for paired nominal data.

    Returns (chi2, p_value).
    """
    # b = baseline correct, treatment wrong
    # c = baseline wrong, treatment correct
    b = int(np.sum((baseline == 1) & (treatment == 0)))
    c = int(np.sum((baseline == 0) & (treatment == 1)))

    if b + c == 0:
        return 0.0, 1.0

    # McNemar chi-squared with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    # p-value from chi-squared distribution with df=1
    # Using normal approximation: sqrt(chi2) ~ N(0,1)
    z = math.sqrt(chi2)
    p = _two_sided_p_from_z(z)
    return float(chi2), p


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------

def analyze_from_counts(
    name: str,
    p1: float,
    n1: int,
    p2: float,
    n2: int,
    treatment_label: str = "treatment",
    rng: Optional[np.random.Generator] = None,
) -> SignificanceResult:
    """Run the full analysis from summary statistics."""
    if rng is None:
        rng = np.random.default_rng(SEED)

    diff = p2 - p1
    se_b = se_proportion(p1, n1)
    se_t = se_proportion(p2, n2)
    se_d = se_diff_proportions(p1, n1, p2, n2)

    ci_lo, ci_hi, boot_mean, _ = bootstrap_ci_from_counts(p1, n1, p2, n2, rng=rng)
    p_val = binomial_z_test(p1, n1, p2, n2)
    tost_p_upper, tost_p_lower, tost_eq = tost_test(p1, n1, p2, n2)

    return SignificanceResult(
        benchmark=name,
        baseline_acc=p1,
        treatment_acc=p2,
        n_baseline=n1,
        n_treatment=n2,
        diff_pp=diff,
        se_baseline=se_b,
        se_treatment=se_t,
        se_diff=se_d,
        bootstrap_ci_lower=ci_lo,
        bootstrap_ci_upper=ci_hi,
        bootstrap_mean_diff=boot_mean,
        p_value_twosided=p_val,
        significant_at_alpha=p_val < ALPHA,
        tost_p_upper=tost_p_upper,
        tost_p_lower=tost_p_lower,
        tost_delta=TOST_DELTA,
        tost_equivalent=tost_eq,
        mcnemar_chi2=None,
        mcnemar_p=None,
    )


def analyze_from_samples(
    name: str,
    baseline: np.ndarray,
    treatment: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> SignificanceResult:
    """Run the full analysis from per-sample binary predictions."""
    if rng is None:
        rng = np.random.default_rng(SEED)

    n = len(baseline)
    p1 = float(baseline.mean())
    p2 = float(treatment.mean())
    diff = p2 - p1
    se_b = se_proportion(p1, n)
    se_t = se_proportion(p2, n)
    se_d = se_diff_proportions(p1, n, p2, n)

    ci_lo, ci_hi, boot_mean, _ = bootstrap_ci_from_samples(baseline, treatment, rng=rng)
    p_val = binomial_z_test(p1, n, p2, n)
    tost_p_upper, tost_p_lower, tost_eq = tost_test(p1, n, p2, n)
    mcn_chi2, mcn_p = mcnemar_test(baseline, treatment)

    return SignificanceResult(
        benchmark=name,
        baseline_acc=p1,
        treatment_acc=p2,
        n_baseline=n,
        n_treatment=n,
        diff_pp=diff,
        se_baseline=se_b,
        se_treatment=se_t,
        se_diff=se_d,
        bootstrap_ci_lower=ci_lo,
        bootstrap_ci_upper=ci_hi,
        bootstrap_mean_diff=boot_mean,
        p_value_twosided=p_val,
        significant_at_alpha=p_val < ALPHA,
        tost_p_upper=tost_p_upper,
        tost_p_lower=tost_p_lower,
        tost_delta=TOST_DELTA,
        tost_equivalent=tost_eq,
        mcnemar_chi2=mcn_chi2,
        mcnemar_p=mcn_p,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(result: SignificanceResult) -> str:
    """Format a human-readable significance report."""
    lines = []
    lines.append(f"=== Statistical Significance Report: {result.benchmark} ===")
    lines.append(f"Baseline: {result.baseline_acc * 100:.1f}% (n={result.n_baseline})")
    lines.append(f"Treatment: {result.treatment_acc * 100:.1f}% (n={result.n_treatment})")
    sign = "+" if result.diff_pp >= 0 else ""
    lines.append(f"Difference: {sign}{result.diff_pp * 100:.1f}pp")
    lines.append("")
    lines.append(f"SE(baseline): {result.se_baseline:.4f}")
    lines.append(f"SE(treatment): {result.se_treatment:.4f}")
    lines.append(f"SE(diff): {result.se_diff:.4f}")
    lines.append("")
    lines.append(
        f"Bootstrap 95% CI: [{result.bootstrap_ci_lower * 100:.1f}%, "
        f"{result.bootstrap_ci_upper * 100:.1f}%]"
    )
    lines.append(f"Bootstrap mean diff: {result.bootstrap_mean_diff * 100:.2f}pp")
    lines.append(f"p-value (two-sided): {result.p_value_twosided:.4f}")
    sig_str = "Yes" if result.significant_at_alpha else "No"
    lines.append(f"Significant at alpha=0.05: {sig_str}")
    lines.append("")
    delta_pp = result.tost_delta * 100
    lines.append(f"TOST Equivalence (delta={delta_pp:.0f}pp):")
    lines.append(f"  p(upper): {result.tost_p_upper:.4f}  (H0: diff >= +{delta_pp:.0f}pp)")
    lines.append(f"  p(lower): {result.tost_p_lower:.4f}  (H0: diff <= -{delta_pp:.0f}pp)")
    tost_p = max(result.tost_p_upper, result.tost_p_lower)
    lines.append(f"  TOST p-value: {tost_p:.4f}")
    eq_str = "Yes (within equivalence margin)" if result.tost_equivalent else "No (not equivalent)"
    lines.append(f"  Equivalent at alpha=0.05: {eq_str}")

    if result.mcnemar_chi2 is not None:
        lines.append("")
        lines.append("McNemar's Test (paired):")
        lines.append(f"  chi2: {result.mcnemar_chi2:.3f}")
        lines.append(f"  p-value: {result.mcnemar_p:.4f}")

    lines.append("")
    return "\n".join(lines)


def save_results(results: list, output_path: Path) -> None:
    """Save results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "seed": SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "alpha": ALPHA,
        "tost_delta": TOST_DELTA,
        "results": [asdict(r) for r in results],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap CI and TOST equivalence testing for medical QA evaluation."
    )
    parser.add_argument("--baseline-acc", type=float, default=None, help="Baseline accuracy (0-1)")
    parser.add_argument("--baseline-n", type=int, default=None, help="Baseline sample size")
    parser.add_argument("--treatment-acc", type=float, default=None, help="Treatment accuracy (0-1)")
    parser.add_argument("--treatment-n", type=int, default=None, help="Treatment sample size")
    parser.add_argument("--name", type=str, default="custom", help="Benchmark name")
    parser.add_argument("--json", type=str, default=None, help="Path to JSON with per-sample predictions")
    parser.add_argument(
        "--output",
        type=str,
        default=str(LOG_DIR / "significance_test.json"),
        help="Output JSON path",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP, help="Number of bootstrap samples")
    parser.add_argument("--delta", type=float, default=TOST_DELTA, help="TOST equivalence margin")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global SEED, N_BOOTSTRAP, TOST_DELTA
    SEED = args.seed
    N_BOOTSTRAP = args.n_bootstrap
    TOST_DELTA = args.delta

    rng = np.random.default_rng(SEED)
    results = []

    if args.json is not None:
        # Per-sample mode
        with open(args.json) as f:
            data = json.load(f)
        baseline = np.array(data["baseline"], dtype=np.float64)
        treatment = np.array(data["treatment"], dtype=np.float64)
        if len(baseline) != len(treatment):
            print("ERROR: baseline and treatment arrays must have the same length.", file=sys.stderr)
            sys.exit(1)
        result = analyze_from_samples(args.name, baseline, treatment, rng=rng)
        results.append(result)
        print(format_report(result))

    elif args.baseline_acc is not None:
        # Custom count mode
        if any(v is None for v in [args.baseline_acc, args.baseline_n, args.treatment_acc, args.treatment_n]):
            print(
                "ERROR: Provide all of --baseline-acc, --baseline-n, --treatment-acc, --treatment-n.",
                file=sys.stderr,
            )
            sys.exit(1)
        result = analyze_from_counts(
            args.name,
            args.baseline_acc,
            args.baseline_n,
            args.treatment_acc,
            args.treatment_n,
            rng=rng,
        )
        results.append(result)
        print(format_report(result))

    else:
        # Default: run all paper benchmarks
        print("Running with default paper benchmarks...\n")
        for bench in PAPER_BENCHMARKS:
            result = analyze_from_counts(
                name=bench["name"],
                p1=bench["baseline_acc"],
                n1=bench["n"],
                p2=bench["treatment_acc"],
                n2=bench["n"],
                treatment_label=bench.get("treatment_label", "treatment"),
                rng=rng,
            )
            results.append(result)
            print(format_report(result))

    save_results(results, Path(args.output))


if __name__ == "__main__":
    main()
