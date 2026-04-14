#!/usr/bin/env python3
"""
Bayesian changepoint detection on GRPO training reward time series.

Run periodically during training. Meaningful results require at least 100+
task completions.

Usage:
    python phase_transition_detection.py [--log PATH] [--out PATH]

Detects phase transitions using two methods:
  1. PELT (Pruned Exact Linear Time) with L2 cost
  2. Segmented regression (2-segment vs 1-segment, BIC comparison)

Outputs results to JSON and prints an ASCII reward curve with changepoints.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_LOG = (
    "/data/project/private/minstar/workspace/BIOAgents/logs/training_fast_vllm_v3.log"
)
DEFAULT_OUT = (
    "/data/project/private/minstar/workspace/BIOAgents/logs/analysis/phase_transition.json"
)
TOTAL_TASKS = 3338
QUARTER_EPOCH_TASK = 835  # ~0.25 epochs
CHANGEPOINT_PROXIMITY = 50  # tasks within this range count as "near"


# ---------------------------------------------------------------------------
# 1. Log parsing
# ---------------------------------------------------------------------------
_TASK_RE = re.compile(
    r"Task\s+(\d+)/(\d+)\s+\[batch=\d+x\]:\s+mean_R=([\d.]+),\s+"
    r"best_R=([\d.]+),\s+worst_R=([\d.]+)"
)


def parse_log(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse training log and return (task_numbers, mean_rewards).

    When the same task appears multiple times (e.g. different batches),
    we keep all observations in chronological order — the time series
    reflects the *sequence* of training steps, not task identity.
    """
    tasks: list[int] = []
    rewards: list[float] = []

    with open(path, "r") as fh:
        for line in fh:
            m = _TASK_RE.search(line)
            if m:
                tasks.append(int(m.group(1)))
                rewards.append(float(m.group(3)))

    if not tasks:
        print("[WARN] No task lines found in log.", file=sys.stderr)
        return np.array([]), np.array([])

    return np.array(tasks, dtype=np.int64), np.array(rewards, dtype=np.float64)


# ---------------------------------------------------------------------------
# 2a. PELT changepoint detection (L2 cost, from scratch)
# ---------------------------------------------------------------------------
def _segment_cost_l2(data: np.ndarray, start: int, end: int) -> float:
    """Sum of squared deviations from the segment mean."""
    seg = data[start:end]
    if len(seg) <= 1:
        return 0.0
    return float(np.sum((seg - seg.mean()) ** 2))


def pelt_l2(
    data: np.ndarray, penalty: Optional[float] = None, min_size: int = 5
) -> List[int]:
    """Pruned Exact Linear Time changepoint detection with L2 cost.

    Returns a list of changepoint indices (positions where a new segment
    begins). Uses the standard PELT recursion with pruning.

    Parameters
    ----------
    data : 1-D array of observations
    penalty : BIC-style penalty per changepoint. Default = 2 * log(n) * var.
    min_size : minimum segment length
    """
    n = len(data)
    if n < 2 * min_size:
        return []

    if penalty is None:
        penalty = 2.0 * np.log(n) * float(np.var(data))

    # F[t] = optimal cost for data[0:t]
    INF = float("inf")
    cost_to = np.full(n + 1, INF)
    cost_to[0] = -penalty  # offset so first segment doesn't pay double
    prev = [0] * (n + 1)  # backtrack pointers
    candidates: list[int] = [0]

    # Pre-compute cumulative sums for O(1) segment cost
    cum_sum = np.zeros(n + 1)
    cum_sq = np.zeros(n + 1)
    cum_sum[1:] = np.cumsum(data)
    cum_sq[1:] = np.cumsum(data ** 2)

    def _cost(s: int, e: int) -> float:
        """L2 segment cost for data[s:e] using prefix sums."""
        length = e - s
        if length <= 1:
            return 0.0
        sm = cum_sum[e] - cum_sum[s]
        sq = cum_sq[e] - cum_sq[s]
        return float(sq - sm * sm / length)

    for t in range(min_size, n + 1):
        best_cost = INF
        best_prev = 0
        new_candidates: list[int] = []
        for s in candidates:
            if t - s < min_size:
                new_candidates.append(s)
                continue
            c = cost_to[s] + _cost(s, t) + penalty
            if c < best_cost:
                best_cost = c
                best_prev = s
            # PELT pruning: keep candidate only if it could still be optimal
            if cost_to[s] + _cost(s, t) <= best_cost:
                new_candidates.append(s)

        cost_to[t] = best_cost
        prev[t] = best_prev
        new_candidates.append(t)
        candidates = new_candidates

    # Backtrack to recover changepoints
    changepoints: list[int] = []
    idx = n
    while idx > 0:
        p = prev[idx]
        if p > 0:
            changepoints.append(p)
        idx = p

    changepoints.sort()
    return changepoints


# ---------------------------------------------------------------------------
# 2b. Segmented regression (2-segment piecewise linear vs single linear)
# ---------------------------------------------------------------------------
def _fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit y = a + b*x. Return (a, b, residual_sum_of_squares)."""
    n = len(x)
    if n < 2:
        return 0.0, 0.0, 0.0
    xm, ym = x.mean(), y.mean()
    ss_xx = np.sum((x - xm) ** 2)
    if ss_xx < 1e-15:
        return float(ym), 0.0, float(np.sum((y - ym) ** 2))
    b = float(np.sum((x - xm) * (y - ym)) / ss_xx)
    a = float(ym - b * xm)
    rss = float(np.sum((y - (a + b * x)) ** 2))
    return a, b, rss


def segmented_regression(
    x: np.ndarray, y: np.ndarray, min_size: int = 10
) -> dict:
    """Find the best single changepoint using piecewise linear regression.

    Compares 2-segment model (4 params) vs 1-segment model (2 params)
    using BIC and F-test.
    """
    n = len(x)
    result: dict = {
        "changepoint_idx": None,
        "changepoint_task": None,
        "bic_single": None,
        "bic_two": None,
        "delta_bic": None,
        "f_statistic": None,
        "p_value": None,
        "slopes": [],
    }

    if n < 2 * min_size:
        return result

    # Single-segment fit
    a0, b0, rss0 = _fit_linear(x.astype(float), y)
    k0 = 2  # parameters
    sigma2_0 = rss0 / n if rss0 > 0 else 1e-15
    bic_single = n * np.log(max(sigma2_0, 1e-15)) + k0 * np.log(n)

    # Sweep all possible changepoints
    best_bic2 = float("inf")
    best_cp = min_size
    best_rss = rss0
    best_slopes: list[float] = [b0]

    for cp in range(min_size, n - min_size + 1):
        a1, b1, rss1 = _fit_linear(x[:cp].astype(float), y[:cp])
        a2, b2, rss2 = _fit_linear(x[cp:].astype(float), y[cp:])
        rss_total = rss1 + rss2
        k2 = 4  # 2 intercepts + 2 slopes
        sigma2_2 = rss_total / n if rss_total > 0 else 1e-15
        bic2 = n * np.log(max(sigma2_2, 1e-15)) + k2 * np.log(n)
        if bic2 < best_bic2:
            best_bic2 = bic2
            best_cp = cp
            best_rss = rss_total
            best_slopes = [b1, b2]

    # F-test: (improvement in RSS / extra params) / (RSS_full / df_full)
    df_extra = 2  # additional parameters in 2-segment model
    df_full = n - 4
    if df_full > 0 and best_rss > 0:
        f_stat = ((rss0 - best_rss) / df_extra) / (best_rss / df_full)
        # Approximate p-value using F-distribution survival function
        # Use the incomplete beta function relationship:
        # P(F > f) = I_{x}(d1/2, d2/2) where x = d1*f / (d1*f + d2)
        p_value = _f_survival(f_stat, df_extra, df_full)
    else:
        f_stat = 0.0
        p_value = 1.0

    result["changepoint_idx"] = int(best_cp)
    result["changepoint_task"] = int(x[best_cp]) if best_cp < n else None
    result["bic_single"] = float(bic_single)
    result["bic_two"] = float(best_bic2)
    result["delta_bic"] = float(bic_single - best_bic2)
    result["f_statistic"] = float(f_stat)
    result["p_value"] = float(p_value)
    result["slopes"] = [float(s) for s in best_slopes]

    return result


def _f_survival(f: float, d1: int, d2: int) -> float:
    """Approximate survival function P(F > f) for F(d1, d2).

    Uses the regularized incomplete beta function via continued fraction.
    """
    if f <= 0:
        return 1.0
    x = d1 * f / (d1 * f + d2)
    # P(F > f) = 1 - I_x(d1/2, d2/2) = I_{1-x}(d2/2, d1/2)
    return _reg_incomplete_beta(1.0 - x, d2 / 2.0, d1 / 2.0)


def _reg_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I_x(a, b) via continued fraction.

    Good enough for our significance testing purposes.
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use symmetry if needed for convergence
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _reg_incomplete_beta(1.0 - x, b, a)

    # Log of the prefactor
    ln_prefactor = (
        a * math.log(x) + b * math.log(1.0 - x)
        - math.log(a)
        - _log_beta(a, b)
    )

    # Lentz continued fraction
    TINY = 1e-30
    MAX_ITER = 200
    f_cf = 1.0 + TINY
    C = f_cf
    D = 0.0

    for m in range(1, MAX_ITER + 1):
        # Even step
        m2 = 2 * m
        # a_{2m}
        num = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        D = 1.0 + num * D
        if abs(D) < TINY:
            D = TINY
        D = 1.0 / D
        C = 1.0 + num / C
        if abs(C) < TINY:
            C = TINY
        f_cf *= C * D

        # Odd step: a_{2m+1}
        num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        D = 1.0 + num * D
        if abs(D) < TINY:
            D = TINY
        D = 1.0 / D
        C = 1.0 + num / C
        if abs(C) < TINY:
            C = TINY
        delta = C * D
        f_cf *= delta

        if abs(delta - 1.0) < 1e-10:
            break

    return math.exp(ln_prefactor) * f_cf


def _log_beta(a: float, b: float) -> float:
    """Log of the beta function using lgamma."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


# ---------------------------------------------------------------------------
# 3. Reward acceleration (second derivative)
# ---------------------------------------------------------------------------
def compute_acceleration(
    tasks: np.ndarray, rewards: np.ndarray, window: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate d²R/dt² using centered finite differences on smoothed data.

    Returns (task_centers, acceleration) arrays.
    """
    if len(rewards) < 2 * window + 2:
        return np.array([]), np.array([])

    # Smooth with rolling mean
    kernel = np.ones(window) / window
    smoothed = np.convolve(rewards, kernel, mode="valid")
    t = tasks[window - 1 :][:len(smoothed)].astype(float)

    # First derivative (central differences)
    if len(smoothed) < 3:
        return np.array([]), np.array([])

    dt = np.diff(t)
    dt[dt == 0] = 1.0
    dr = np.diff(smoothed) / dt

    # Second derivative
    if len(dr) < 2:
        return np.array([]), np.array([])

    t2 = t[1:-1]
    dt2 = (t[2:] - t[:-2]) / 2.0
    dt2[dt2 == 0] = 1.0
    accel = np.diff(dr) / dt2[:len(np.diff(dr))]

    return t2[:len(accel)], accel


# ---------------------------------------------------------------------------
# 4. ASCII plotting
# ---------------------------------------------------------------------------
def ascii_plot(
    tasks: np.ndarray,
    rewards: np.ndarray,
    changepoints_pelt: list[int],
    changepoint_seg: Optional[int],
    width: int = 80,
    height: int = 24,
) -> str:
    """Render an ASCII reward curve with changepoints marked."""
    if len(tasks) == 0:
        return "[No data to plot]"

    lines: list[str] = []
    lines.append("=" * width)
    lines.append("  REWARD CURVE  |  P = PELT changepoint  |  S = segmented regression")
    lines.append("=" * width)

    r_min = float(rewards.min())
    r_max = float(rewards.max())
    r_range = r_max - r_min if r_max > r_min else 1.0
    t_min = int(tasks.min())
    t_max = int(tasks.max())
    t_range = t_max - t_min if t_max > t_min else 1

    # Bin tasks into columns
    grid = np.full((height, width), " ")

    # Smooth for plotting
    if len(rewards) > width:
        bin_size = len(rewards) / width
        plot_r = np.array(
            [rewards[int(i * bin_size):int((i + 1) * bin_size)].mean()
             for i in range(width)]
        )
        plot_t = np.array(
            [tasks[int(i * bin_size):int((i + 1) * bin_size)].mean()
             for i in range(width)]
        )
    else:
        plot_r = rewards
        plot_t = tasks.astype(float)
        # Map to columns
        cols = np.clip(
            ((plot_t - t_min) / t_range * (width - 1)).astype(int), 0, width - 1
        )
        grid_r = np.full(width, np.nan)
        for c, r in zip(cols, plot_r):
            grid_r[c] = r
        # Interpolate gaps
        valid = ~np.isnan(grid_r)
        if valid.sum() > 1:
            grid_r = np.interp(np.arange(width), np.where(valid)[0], grid_r[valid])
        plot_r = grid_r

    # Mark changepoints
    cp_cols_pelt: set[int] = set()
    for cp_idx in changepoints_pelt:
        if cp_idx < len(tasks):
            col = int((tasks[cp_idx] - t_min) / t_range * (width - 1))
            col = max(0, min(width - 1, col))
            cp_cols_pelt.add(col)

    cp_col_seg: Optional[int] = None
    if changepoint_seg is not None and changepoint_seg < len(tasks):
        cp_col_seg = int((tasks[changepoint_seg] - t_min) / t_range * (width - 1))
        cp_col_seg = max(0, min(width - 1, cp_col_seg))

    # Fill grid
    for col in range(min(width, len(plot_r))):
        row = int((plot_r[col] - r_min) / r_range * (height - 1))
        row = max(0, min(height - 1, row))
        row = height - 1 - row  # invert y-axis
        if col in cp_cols_pelt:
            grid[row][col] = "P"
        elif cp_col_seg is not None and col == cp_col_seg:
            grid[row][col] = "S"
        else:
            grid[row][col] = "·"

    # Draw changepoint vertical lines
    for col in cp_cols_pelt:
        for row in range(height):
            if grid[row][col] == " ":
                grid[row][col] = "|"
    if cp_col_seg is not None:
        for row in range(height):
            if grid[row][cp_col_seg] == " ":
                grid[row][cp_col_seg] = ":"

    # Render
    for row_idx in range(height):
        r_val = r_max - (row_idx / (height - 1)) * r_range
        label = f"{r_val:.3f} |"
        lines.append(label + "".join(grid[row_idx]))

    # X-axis
    lines.append(" " * 8 + "-" * width)
    x_label = " " * 8 + f"task {t_min}" + " " * (width - len(str(t_min)) - len(str(t_max)) - 9) + f"task {t_max}"
    lines.append(x_label)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian changepoint detection on GRPO reward time series"
    )
    parser.add_argument("--log", default=DEFAULT_LOG, help="Training log path")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output JSON path")
    args = parser.parse_args()

    print(f"[INFO] Parsing log: {args.log}")
    tasks, rewards = parse_log(args.log)
    n = len(tasks)
    print(f"[INFO] Found {n} task completions")

    if n < 10:
        print("[WARN] Too few data points for analysis. Need at least 10.")
        results = {
            "timestamp": datetime.now().isoformat(),
            "n_observations": n,
            "status": "insufficient_data",
        }
        _save_results(results, args.out)
        return

    if n < 100:
        print(
            "[WARN] Fewer than 100 observations — results may be unreliable. "
            "Run again later as training progresses."
        )

    # Sort by chronological order (line order in log) — already in order
    order = np.arange(n)
    seq_tasks = tasks  # task IDs in sequence order
    seq_rewards = rewards

    # --- PELT ---
    print("[INFO] Running PELT changepoint detection...")
    pelt_cps = pelt_l2(seq_rewards, min_size=max(5, n // 20))
    pelt_task_ids = [int(seq_tasks[i]) for i in pelt_cps if i < n]
    print(f"[INFO] PELT found {len(pelt_cps)} changepoint(s): task indices {pelt_cps}")

    # --- Segmented Regression ---
    print("[INFO] Running segmented regression...")
    seg_result = segmented_regression(
        seq_tasks, seq_rewards, min_size=max(3, n // 10)
    )
    seg_cp_idx = seg_result.get("changepoint_idx")
    print(
        f"[INFO] Segmented regression changepoint: "
        f"idx={seg_cp_idx}, task={seg_result.get('changepoint_task')}, "
        f"delta_BIC={seg_result.get('delta_bic', 0):.2f}, "
        f"p={seg_result.get('p_value', 1):.4f}"
    )

    # --- Acceleration ---
    print("[INFO] Computing reward acceleration...")
    accel_t, accel_v = compute_acceleration(
        seq_tasks, seq_rewards, window=max(3, n // 10)
    )

    # Acceleration near detected changepoints
    accel_at_cps: dict[str, float] = {}
    all_cp_tasks = set(pelt_task_ids)
    if seg_result.get("changepoint_task") is not None:
        all_cp_tasks.add(seg_result["changepoint_task"])

    for cp_task in all_cp_tasks:
        if len(accel_t) > 0:
            closest = int(np.argmin(np.abs(accel_t - cp_task)))
            accel_at_cps[str(cp_task)] = float(accel_v[closest])

    # --- Near 0.25 epochs? ---
    near_quarter = any(
        abs(t - QUARTER_EPOCH_TASK) <= CHANGEPOINT_PROXIMITY for t in all_cp_tasks
    )
    quarter_info = {
        "expected_task": QUARTER_EPOCH_TASK,
        "proximity_threshold": CHANGEPOINT_PROXIMITY,
        "changepoint_detected_nearby": near_quarter,
        "nearby_changepoints": [
            t for t in sorted(all_cp_tasks)
            if abs(t - QUARTER_EPOCH_TASK) <= CHANGEPOINT_PROXIMITY
        ],
    }

    # --- Significance summary ---
    seg_significant = (
        seg_result.get("p_value") is not None
        and seg_result["p_value"] < 0.05
        and seg_result.get("delta_bic", 0) > 2
    )

    # --- Assemble results ---
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_observations": n,
        "status": "ok",
        "reward_stats": {
            "mean": float(seq_rewards.mean()),
            "std": float(seq_rewards.std()),
            "min": float(seq_rewards.min()),
            "max": float(seq_rewards.max()),
        },
        "pelt": {
            "changepoint_indices": pelt_cps,
            "changepoint_tasks": pelt_task_ids,
            "n_changepoints": len(pelt_cps),
        },
        "segmented_regression": {
            "changepoint_idx": seg_result.get("changepoint_idx"),
            "changepoint_task": seg_result.get("changepoint_task"),
            "bic_single_segment": seg_result.get("bic_single"),
            "bic_two_segment": seg_result.get("bic_two"),
            "delta_bic": seg_result.get("delta_bic"),
            "f_statistic": seg_result.get("f_statistic"),
            "p_value": seg_result.get("p_value"),
            "significant": seg_significant,
            "slopes_per_segment": seg_result.get("slopes", []),
        },
        "acceleration_at_changepoints": accel_at_cps,
        "quarter_epoch_analysis": quarter_info,
    }

    # --- ASCII Plot ---
    plot = ascii_plot(seq_tasks, seq_rewards, pelt_cps, seg_cp_idx)
    print("\n" + plot)

    # --- Summary ---
    print("=" * 60)
    print("PHASE TRANSITION ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Observations:          {n}")
    print(f"  PELT changepoints:     {len(pelt_cps)} at tasks {pelt_task_ids}")
    if seg_cp_idx is not None:
        sig_str = "YES (p<0.05, dBIC>2)" if seg_significant else "no"
        print(
            f"  Segmented regression:  task {seg_result['changepoint_task']} "
            f"(significant: {sig_str})"
        )
        print(
            f"    F={seg_result['f_statistic']:.2f}, "
            f"p={seg_result['p_value']:.4f}, "
            f"dBIC={seg_result['delta_bic']:.2f}"
        )
        if seg_result.get("slopes"):
            print(
                f"    Slopes: before={seg_result['slopes'][0]:.6f}, "
                f"after={seg_result['slopes'][1]:.6f}"
            )
    print(f"  Near 0.25 epochs ({QUARTER_EPOCH_TASK})? {near_quarter}")
    if accel_at_cps:
        print(f"  Acceleration at CPs:   {accel_at_cps}")
    print("=" * 60)

    # --- Save ---
    _save_results(results, args.out)
    print(f"\n[INFO] Results saved to {args.out}")


def _save_results(results: dict, path: str) -> None:
    """Save results JSON, creating directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
