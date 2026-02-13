#!/usr/bin/env python3
"""P2 Parallel Evaluation: Run checkpoint evaluation on a specific GPU + domains.

Usage:
    # Evaluate SFT merged model on clinical_diagnosis + visual_diagnosis (GPU 1)
    CUDA_VISIBLE_DEVICES=1 python scripts/run_p2_eval_parallel.py \
        --model checkpoints/sft_p2_aggressive_lingshu/merged \
        --model-name P2-SFT-Aggressive \
        --domains clinical_diagnosis visual_diagnosis

    # Evaluate GRPO checkpoint on drug_interaction + ehr_management (GPU 2)
    CUDA_VISIBLE_DEVICES=2 python scripts/run_p2_eval_parallel.py \
        --model checkpoints/grpo_p2_qwen3_8b/final \
        --base-model /data/project/private/minstar/models/Qwen3-8B-Base \
        --model-name P2-GRPO-Qwen3 \
        --domains drug_interaction ehr_management
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bioagents.evaluation.agent_runner import AgentRunner, RunConfig
from bioagents.gym.agent_env import BioAgentGymEnv
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent


def load_model_with_adapter(model_path: str, base_model_path: str = None):
    """Check if model_path is a LoRA adapter and needs a base model."""
    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists() and base_model_path:
        # This is a LoRA adapter — merge first
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        logger.info(f"Merging LoRA adapter: {model_path} + {base_model_path}")
        merged_path = str(Path(model_path).parent / "eval_merged")

        if Path(merged_path).exists() and (Path(merged_path) / "config.json").exists():
            logger.info(f"Using existing merged model: {merged_path}")
            return merged_path

        config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
        model_type = getattr(config, "model_type", "")

        if model_type in ("qwen2_5_vl", "qwen2_vl"):
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
            )

        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        model.save_pretrained(merged_path)

        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        tokenizer.save_pretrained(merged_path)
        logger.info(f"Merged to: {merged_path}")
        return merged_path

    return model_path


def evaluate_model(model_path: str, model_name: str, domains: list[str], base_model: str = None):
    """Evaluate a model on specified domains."""
    # Handle LoRA adapters
    effective_path = load_model_with_adapter(model_path, base_model)

    results = {}
    for domain in domains:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {model_name} on {domain}")
        logger.info(f"{'='*60}")

        try:
            config = RunConfig(
                model_name_or_path=effective_path,
                backend="transformers",
                domain=domain,
                max_turns=15,
                temperature=0.1,
                log_dir=str(PROJECT_ROOT / "logs" / "p2_eval"),
            )
            runner = AgentRunner(config)
            runner.load_model()
            task_results = runner.run_all_tasks()

            avg_action = sum(r.action_score for r in task_results) / max(len(task_results), 1)
            avg_reward = sum(r.final_reward for r in task_results) / max(len(task_results), 1)
            avg_turns = sum(r.total_turns for r in task_results) / max(len(task_results), 1)

            results[domain] = {
                "action_score": round(avg_action, 3),
                "reward": round(avg_reward, 3),
                "avg_turns": round(avg_turns, 1),
                "tasks": len(task_results),
                "per_task": [
                    {
                        "task_id": r.task_id,
                        "action_score": r.action_score,
                        "reward": r.final_reward,
                        "turns": r.total_turns,
                    }
                    for r in task_results
                ],
            }

            logger.info(f"  {domain}: action={avg_action:.3f}, reward={avg_reward:.3f}, turns={avg_turns:.1f}")

            # Free model memory between domains
            del runner
            import torch
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"  Error evaluating {domain}: {e}")
            import traceback
            traceback.print_exc()
            results[domain] = {"error": str(e)}

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = PROJECT_ROOT / "logs" / "p2_eval" / f"{model_name}_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "model_name": model_name,
        "model_path": model_path,
        "domains": results,
        "timestamp": datetime.now().isoformat(),
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  EVAL RESULTS: {model_name}")
    print(f"{'='*60}")
    for domain, res in results.items():
        if "error" in res:
            print(f"  {domain}: ERROR - {res['error'][:80]}")
        else:
            print(f"  {domain}: action={res['action_score']:.3f}, reward={res['reward']:.3f}, turns={res['avg_turns']:.1f}")

    overall_scores = [res["action_score"] for res in results.values() if "action_score" in res]
    if overall_scores:
        print(f"  OVERALL action_score: {sum(overall_scores)/len(overall_scores):.3f}")
    print(f"  Saved to: {output_path}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="P2 Parallel Evaluation")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--model-name", required=True, help="Model name for logging")
    parser.add_argument("--base-model", default=None, help="Base model (for LoRA adapters)")
    parser.add_argument("--domains", nargs="+", required=True,
                        choices=["clinical_diagnosis", "visual_diagnosis",
                                 "drug_interaction", "ehr_management", "medical_qa"])
    args = parser.parse_args()

    evaluate_model(args.model, args.model_name, args.domains, args.base_model)


if __name__ == "__main__":
    main()
