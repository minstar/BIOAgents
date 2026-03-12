"""
compare_pre_post_eval.py
========================
Dry-run / Phase-2 GYM 1-cycle 전후 eval 비교 스크립트.

사용법:
  python scripts/compare_pre_post_eval.py \
      --agent_id dryrun_qwen25vl \
      --log_dir  logs/autonomous/dryrun_qwen25vl \
      --out      logs/eval_delta/dryrun_cycle1_delta.json

동작:
  1. log_dir/*/eval/ 하위 폴더를 시간순 정렬
  2. 첫 번째 eval 폴더(pre-training) vs 마지막 eval 폴더(post-training) 비교
  3. 도메인별 action_score, reward, error_types 집계
  4. Delta 출력 + JSON 저장 + PLANNING.md 업데이트 제안
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def load_task_jsons(eval_dir: Path) -> list[dict]:
    """eval 디렉토리의 모든 task JSON 파일을 로드."""
    results = []
    for jf in sorted(eval_dir.glob("task_*.json")):
        try:
            with open(jf) as f:
                data = json.load(f)
            data["_file"] = jf.name
            results.append(data)
        except Exception as e:
            print(f"  [WARN] Failed to load {jf.name}: {e}", file=sys.stderr)
    return results


def aggregate_eval(tasks: list[dict]) -> dict:
    """task 목록에서 집계 지표 산출."""
    if not tasks:
        return {"n": 0, "avg_action_score": None, "avg_reward": None, "error_types": {}}

    action_scores = [t.get("action_score", t.get("composite_score", 0.0)) for t in tasks]
    rewards = [t.get("final_reward", t.get("reward", t.get("action_score", 0.0))) for t in tasks]

    error_counts: dict[str, int] = defaultdict(int)
    for t in tasks:
        for et in t.get("error_types", []):
            error_counts[et] += 1

    return {
        "n": len(tasks),
        "avg_action_score": round(sum(action_scores) / len(action_scores), 4),
        "avg_reward": round(sum(rewards) / len(rewards), 4),
        "min_reward": round(min(rewards), 4),
        "max_reward": round(max(rewards), 4),
        "pct_above_0.5": round(sum(1 for r in rewards if r >= 0.5) / len(rewards), 4),
        "error_types": dict(sorted(error_counts.items(), key=lambda x: -x[1])),
    }


def find_eval_dirs(log_dir: Path, agent_id: str) -> list[tuple[str, Path]]:
    """
    log_dir/<agent_id>/eval/<model>_<domain>_<timestamp> 패턴의 폴더를 찾아
    (timestamp, path) 목록으로 반환 (시간순 정렬).
    """
    base = log_dir / agent_id / "eval"
    if not base.exists():
        base = log_dir / "eval"
    if not base.exists():
        print(f"[ERROR] Eval directory not found: {base}", file=sys.stderr)
        sys.exit(1)

    entries = []
    for d in base.iterdir():
        if d.is_dir():
            # 폴더명에서 timestamp 추출: 마지막 두 언더스코어 파트 (날짜_시각)
            parts = d.name.rsplit("_", 2)
            if len(parts) >= 3:
                ts = f"{parts[-2]}_{parts[-1]}"
            else:
                ts = d.name
            entries.append((ts, d))

    entries.sort(key=lambda x: x[0])
    return entries


def main():
    parser = argparse.ArgumentParser(description="Compare pre/post-training GYM eval results")
    parser.add_argument("--agent_id", default="dryrun_qwen25vl")
    parser.add_argument("--log_dir", default="logs/autonomous/dryrun_qwen25vl")
    parser.add_argument("--out", default="logs/eval_delta/dryrun_cycle1_delta.json")
    parser.add_argument("--all_cycles", action="store_true",
                        help="Show all cycles (not just first vs last)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    eval_dirs = find_eval_dirs(log_dir, args.agent_id)
    if not eval_dirs:
        print("[ERROR] No eval directories found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(eval_dirs)} eval run(s) in {log_dir}:")
    for ts, d in eval_dirs:
        print(f"  [{ts}] {d.name}")

    # 도메인별로 분리
    domain_runs: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for ts, d in eval_dirs:
        # 도메인 추출: <model>_<domain>_<date>_<time>
        # e.g. Qwen2.5-VL-7B-Instruct_clinical_diagnosis_20260309_215006
        # Split off the model name prefix (contains dashes)
        name = d.name
        for marker in ["Qwen2.5-VL-7B-Instruct_", "Lingshu-7B_", "Step3-VL-10B_",
                       "Qwen3-7B_", "Qwen3-14B_"]:
            if marker in name:
                rest = name.split(marker, 1)[1]  # clinical_diagnosis_20260309_215006
                parts = rest.rsplit("_", 2)
                domain = parts[0] if len(parts) >= 3 else rest
                domain_runs[domain].append((ts, d))
                break
        else:
            domain_runs["unknown"].append((ts, d))

    all_results = {
        "agent_id": args.agent_id,
        "generated_at": datetime.now().isoformat(),
        "log_dir": str(log_dir),
        "domains": {},
    }

    print("\n" + "=" * 70)
    print("PRE vs POST Training Eval Delta")
    print("=" * 70)

    for domain, runs in sorted(domain_runs.items()):
        print(f"\n[Domain: {domain}]  ({len(runs)} eval run(s))")

        if len(runs) == 1:
            ts, d = runs[0]
            tasks = load_task_jsons(d)
            agg = aggregate_eval(tasks)
            print(f"  Only 1 run (pre-training):  n={agg['n']}, "
                  f"action={agg['avg_action_score']}, reward={agg['avg_reward']}")
            all_results["domains"][domain] = {
                "pre": {"timestamp": ts, **agg},
                "post": None,
                "delta": None,
                "status": "only_pre",
            }
            continue

        # Pre = 첫 번째, Post = 마지막
        pre_ts, pre_dir = runs[0]
        post_ts, post_dir = runs[-1]

        pre_tasks = load_task_jsons(pre_dir)
        post_tasks = load_task_jsons(post_dir)

        pre_agg = aggregate_eval(pre_tasks)
        post_agg = aggregate_eval(post_tasks)

        if pre_agg["avg_action_score"] is not None and post_agg["avg_action_score"] is not None:
            delta_action = round(post_agg["avg_action_score"] - pre_agg["avg_action_score"], 4)
            delta_reward = round(post_agg["avg_reward"] - pre_agg["avg_reward"], 4)
        else:
            delta_action = delta_reward = None

        print(f"  PRE  [{pre_ts}]: n={pre_agg['n']}, "
              f"action={pre_agg['avg_action_score']}, reward={pre_agg['avg_reward']}")
        print(f"  POST [{post_ts}]: n={post_agg['n']}, "
              f"action={post_agg['avg_action_score']}, reward={post_agg['avg_reward']}")

        if delta_action is not None:
            sign_a = "+" if delta_action >= 0 else ""
            sign_r = "+" if delta_reward >= 0 else ""
            print(f"  DELTA:  action={sign_a}{delta_action}  reward={sign_r}{delta_reward}")
            verdict = "✅ IMPROVED" if delta_reward > 0.01 else \
                      "⚠️  MARGINAL" if delta_reward >= -0.01 else "❌ DEGRADED"
            print(f"  → {verdict}")
        else:
            print("  DELTA: N/A")

        # Intermediate cycles if requested
        if args.all_cycles and len(runs) > 2:
            for i, (ts, d) in enumerate(runs[1:-1], 1):
                tasks = load_task_jsons(d)
                agg = aggregate_eval(tasks)
                print(f"  Cycle {i+1} [{ts}]: n={agg['n']}, "
                      f"action={agg['avg_action_score']}, reward={agg['avg_reward']}")

        all_results["domains"][domain] = {
            "pre": {"timestamp": pre_ts, **pre_agg},
            "post": {"timestamp": post_ts, **post_agg},
            "delta": {
                "action_score": delta_action,
                "reward": delta_reward,
            },
            "num_cycles": len(runs),
            "status": "improved" if (delta_reward or 0) > 0.01 else
                      "marginal" if (delta_reward or 0) >= -0.01 else "degraded",
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    improved = []
    degraded = []
    marginal = []
    for domain, res in all_results["domains"].items():
        d = res.get("delta")
        if d is None:
            marginal.append(domain)
        elif d.get("reward", 0) > 0.01:
            improved.append(f"{domain}(Δreward={d['reward']:+.4f})")
        elif d.get("reward", 0) < -0.01:
            degraded.append(f"{domain}(Δreward={d['reward']:+.4f})")
        else:
            marginal.append(domain)

    if improved:
        print(f"  ✅ Improved:  {', '.join(improved)}")
    if marginal:
        print(f"  ⚠️  Marginal:  {', '.join(marginal)}")
    if degraded:
        print(f"  ❌ Degraded:  {', '.join(degraded)}")

    # Save
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {out_path}")

    # PLANNING.md update suggestion
    print("\n[Next] Update PLANNING.md with delta results:")
    print("  python scripts/compare_pre_post_eval.py --agent_id dryrun_qwen25vl")
    print("  → then paste summary into PLANNING.md §[2026-03-09~10] dry-run post-eval")


if __name__ == "__main__":
    main()
