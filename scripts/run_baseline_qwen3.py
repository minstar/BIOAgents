#!/usr/bin/env python3
"""Quick baseline experiment with Qwen3-8B-Base (already available locally).

Run while Qwen2.5-VL and Lingshu are downloading.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bioagents.evaluation.agent_runner import AgentRunner, RunConfig, parse_tool_call, build_system_prompt
from bioagents.gym.agent_env import BioAgentGymEnv
from loguru import logger


def run_baseline():
    """Run Qwen3-8B-Base on all clinical diagnosis tasks using transformers."""
    
    model_path = "/data/project/private/minstar/models/Qwen3-8B-Base"
    
    config = RunConfig(
        model_name_or_path=model_path,
        backend="transformers",
        domain="clinical_diagnosis",
        task_ids=None,  # all tasks
        max_turns=15,
        temperature=0.1,
        top_p=0.95,
        max_new_tokens=1024,
        log_dir=str(Path(__file__).parent.parent / "logs" / "runs"),
    )
    
    print("=" * 70)
    print("  BIOAgents Baseline Experiment")
    print(f"  Model: Qwen3-8B-Base")
    print(f"  Domain: clinical_diagnosis")
    print(f"  Time: {datetime.now().isoformat()}")
    print("=" * 70)
    
    runner = AgentRunner(config)
    
    print("\n[1/2] Loading model...")
    runner.load_model()
    print("  ✓ Model loaded")
    
    print("\n[2/2] Running tasks...")
    results = runner.run_all_tasks()
    
    # Detailed per-task output
    print("\n\n" + "=" * 70)
    print("  DETAILED RESULTS")
    print("=" * 70)
    
    for result in results:
        print(f"\n--- Task: {result.task_id} ---")
        print(f"  Action Score: {result.action_score:.3f}")
        print(f"  Final Reward: {result.final_reward:.3f}")
        print(f"  Turns: {result.total_turns}")
        print(f"  Latency: {result.total_latency:.1f}s")
        if result.error:
            print(f"  ERROR: {result.error[:200]}")
        
        print(f"  Turn-by-turn:")
        for turn in result.turns:
            if turn.parsed_tool_call:
                tc = turn.parsed_tool_call
                print(f"    [{turn.turn_idx}] TOOL: {tc['name']}({json.dumps(tc.get('arguments', {}))[:80]})")
                if turn.tool_response:
                    print(f"         → {turn.tool_response[:100]}...")
            elif turn.is_final_answer:
                print(f"    [{turn.turn_idx}] ANSWER: {turn.raw_output[:150]}...")
            else:
                print(f"    [{turn.turn_idx}] TEXT: {turn.raw_output[:100]}...")
    
    return results


if __name__ == "__main__":
    results = run_baseline()
