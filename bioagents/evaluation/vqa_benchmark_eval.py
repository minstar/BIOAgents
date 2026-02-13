"""VQA Benchmark Evaluator for BIOAgents Healthcare AI GYM.

Evaluates Vision-Language models on 6 Visual Medical QA benchmarks:
1. VQA-RAD     — Radiology VQA
2. SLAKE       — Multilingual Medical VQA
3. PathVQA     — Pathology VQA
4. PMC-VQA     — PubMedCentral VQA (multiple-choice)
5. VQA-Med-2021 — Medical VQA Challenge
6. Quilt-VQA   — Histopathology VQA

Supports:
- Qwen2.5-VL / Qwen2-VL models (native multimodal)
- Lingshu-7B (medical MLLM)
- Text-only fallback (image description in prompt)
- Multiple metrics: Exact Match, Token F1, BLEU, BERTScore
- Per-dataset and aggregate reporting

Usage:
    evaluator = VQABenchmarkEvaluator(config)
    results = evaluator.evaluate_all()

CLI:
    python -m bioagents.evaluation.vqa_benchmark_eval \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --benchmarks vqa_rad slake pathvqa \
        --max-samples 200
"""

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import torch
from loguru import logger

from bioagents.data_pipeline.vqa_loader import (
    VQA_DATASET_REGISTRY,
    get_vqa_stats,
    load_all_vqa_datasets,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent


# ══════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════

@dataclass
class VQABenchmarkConfig:
    """Configuration for VQA benchmark evaluation."""
    model_name_or_path: str
    model_name: str = "BIOAgent-VL"
    backend: Literal["transformers", "vllm"] = "transformers"
    benchmarks: list[str] = field(
        default_factory=lambda: ["vqa_rad", "slake", "pathvqa"]
    )
    max_samples: int = 0  # 0 = all
    batch_size: int = 4
    output_dir: str = "logs/benchmarks/vqa"
    temperature: float = 0.0  # greedy for benchmarks
    max_new_tokens: int = 256
    # VQA-specific
    use_images: bool = True  # False = text-only (image description in prompt)
    metrics: list[str] = field(
        default_factory=lambda: ["exact_match", "token_f1", "bleu", "contains"]
    )


# ══════════════════════════════════════════════════════════════
#  VQA Metrics
# ══════════════════════════════════════════════════════════════

class VQAMetrics:
    """Compute VQA evaluation metrics."""

    @staticmethod
    def exact_match(prediction: str, reference: str) -> float:
        """Exact match after normalization."""
        pred = _normalize_answer(prediction)
        ref = _normalize_answer(reference)
        return 1.0 if pred == ref else 0.0

    @staticmethod
    def contains_match(prediction: str, reference: str) -> float:
        """Check if reference answer is contained in prediction."""
        pred = _normalize_answer(prediction)
        ref = _normalize_answer(reference)
        if not ref:
            return 0.0
        return 1.0 if ref in pred else 0.0

    @staticmethod
    def token_f1(prediction: str, reference: str) -> float:
        """Token-level F1 score (precision-recall of word overlap)."""
        pred_tokens = set(_normalize_answer(prediction).split())
        ref_tokens = set(_normalize_answer(reference).split())

        if not pred_tokens or not ref_tokens:
            return 1.0 if pred_tokens == ref_tokens else 0.0

        common = pred_tokens & ref_tokens
        if not common:
            return 0.0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def bleu_score(prediction: str, reference: str) -> float:
        """Simple BLEU-1 score (unigram precision with brevity penalty)."""
        pred_tokens = _normalize_answer(prediction).split()
        ref_tokens = _normalize_answer(reference).split()

        if not pred_tokens or not ref_tokens:
            return 1.0 if pred_tokens == ref_tokens else 0.0

        ref_counts = Counter(ref_tokens)
        clipped = 0
        for token in pred_tokens:
            if ref_counts.get(token, 0) > 0:
                clipped += 1
                ref_counts[token] -= 1

        precision = clipped / len(pred_tokens) if pred_tokens else 0

        # Brevity penalty
        bp = min(1.0, len(pred_tokens) / len(ref_tokens)) if ref_tokens else 0

        return bp * precision

    @staticmethod
    def bertscore(prediction: str, reference: str) -> float:
        """BERTScore using BiomedBERT (optional — returns 0 if unavailable)."""
        try:
            from bioagents.evaluation.rewards import accuracy_reward_bertscore
            return accuracy_reward_bertscore(prediction, reference)
        except Exception:
            return 0.0

    @classmethod
    def compute_all(cls, prediction: str, reference: str,
                    metrics: list[str] = None) -> dict[str, float]:
        """Compute all requested metrics."""
        if metrics is None:
            metrics = ["exact_match", "token_f1", "bleu", "contains"]

        results = {}
        metric_fns = {
            "exact_match": cls.exact_match,
            "token_f1": cls.token_f1,
            "bleu": cls.bleu_score,
            "contains": cls.contains_match,
            "bertscore": cls.bertscore,
        }

        for m in metrics:
            fn = metric_fns.get(m)
            if fn:
                results[m] = fn(prediction, reference)

        return results


def _normalize_answer(s: str) -> str:
    """Normalize answer string for comparison."""
    s = s.strip().lower()
    # Remove punctuation
    s = re.sub(r"[^\w\s]", " ", s)
    # Collapse whitespace
    s = " ".join(s.split())
    return s


# ══════════════════════════════════════════════════════════════
#  VQA Benchmark Evaluator
# ══════════════════════════════════════════════════════════════

class VQABenchmarkEvaluator:
    """Evaluate VL models on Visual Medical QA benchmarks."""

    def __init__(self, config: VQABenchmarkConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.processor = None
        self._is_vl_model = False
        self._model_type = ""
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def load_model(self):
        """Load the Vision-Language model."""
        from transformers import AutoConfig, AutoTokenizer

        logger.info(f"Loading VL model: {self.config.model_name_or_path}")

        model_config = AutoConfig.from_pretrained(
            self.config.model_name_or_path, trust_remote_code=True
        )
        self._model_type = getattr(model_config, "model_type", "")
        self._is_vl_model = self._model_type in (
            "qwen2_5_vl", "qwen2_vl", "llava", "llava_next",
            "internvl", "lingshu",
        )

        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

        if self._model_type in ("qwen2_5_vl", "qwen2_vl"):
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model_name_or_path, **load_kwargs
            )
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name_or_path, trust_remote_code=True
            )
            self.tokenizer = self.processor.tokenizer
            logger.info("Loaded Qwen2.5-VL with processor")
        else:
            # Generic CausalLM (text-only fallback or Lingshu)
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path, **load_kwargs
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name_or_path, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info(f"Model loaded: type={self._model_type}, VL={self._is_vl_model}, params={param_count:.0f}M")

    def evaluate_all(self) -> dict:
        """Evaluate on all configured VQA benchmarks."""
        if self.model is None:
            self.load_model()

        all_results = {}
        for benchmark in self.config.benchmarks:
            logger.info(f"\n{'='*60}")
            logger.info(f"  VQA Benchmark: {benchmark}")
            logger.info(f"{'='*60}")

            try:
                results = self.evaluate_benchmark(benchmark)
                all_results[benchmark] = results
                self._log_benchmark_result(benchmark, results)
            except Exception as e:
                logger.error(f"  Error evaluating {benchmark}: {e}")
                import traceback
                traceback.print_exc()
                all_results[benchmark] = {"error": str(e)}

        # Save and print
        self._save_results(all_results)
        self._print_summary(all_results)

        return all_results

    def evaluate_benchmark(self, benchmark: str) -> dict:
        """Evaluate on a single VQA benchmark."""
        # Load VQA data
        if benchmark not in VQA_DATASET_REGISTRY:
            return {"error": f"Unknown VQA benchmark: {benchmark}"}

        info = VQA_DATASET_REGISTRY[benchmark]
        loader = info["loader"]

        # Load dataset
        if benchmark in ("vqa_med_2021", "quilt_vqa"):
            data = loader(max_samples=self.config.max_samples or None)
        else:
            data = loader(
                max_samples=self.config.max_samples or None,
                split="test",
            )

        if not data:
            return {"error": f"No data found for {benchmark}"}

        logger.info(f"  Loaded {len(data)} samples for {benchmark}")

        # Evaluate each sample
        per_sample_results = []
        metric_sums = Counter()

        for i, item in enumerate(data):
            question = item["question"]
            reference = item["answer"]
            image_path = item.get("image_path")
            answer_type = item.get("answer_type", "open_ended")
            options = item.get("options")

            if not question or not reference:
                continue

            # Build prompt
            prompt = self._build_vqa_prompt(
                question=question,
                answer_type=answer_type,
                options=options,
                image_path=image_path,
                modality=item.get("modality", ""),
            )

            # Generate answer
            if self._is_vl_model and image_path and self.config.use_images:
                prediction = self._generate_vl_answer(prompt, image_path)
            else:
                prediction = self._generate_text_answer(prompt)

            # Compute metrics
            metrics = VQAMetrics.compute_all(
                prediction, reference, self.config.metrics
            )

            per_sample_results.append({
                "id": item["id"],
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "answer_type": answer_type,
                "metrics": metrics,
            })

            for m, v in metrics.items():
                metric_sums[m] += v

            if (i + 1) % 50 == 0:
                n = len(per_sample_results)
                em = metric_sums.get("exact_match", 0) / n
                f1 = metric_sums.get("token_f1", 0) / n
                logger.info(
                    f"  Progress: {i+1}/{len(data)} | "
                    f"EM={em:.3f} F1={f1:.3f}"
                )

        total = len(per_sample_results)
        if total == 0:
            return {"error": "No valid samples evaluated"}

        # Aggregate metrics
        avg_metrics = {m: v / total for m, v in metric_sums.items()}

        # Per answer-type breakdown
        by_type = {}
        for r in per_sample_results:
            at = r["answer_type"]
            if at not in by_type:
                by_type[at] = {"count": 0, "metrics": Counter()}
            by_type[at]["count"] += 1
            for m, v in r["metrics"].items():
                by_type[at]["metrics"][m] += v

        for at, info in by_type.items():
            info["metrics"] = {m: v / info["count"] for m, v in info["metrics"].items()}

        return {
            "benchmark": benchmark,
            "total": total,
            "metrics": avg_metrics,
            "by_answer_type": by_type,
            "per_sample": per_sample_results,
        }

    def _build_vqa_prompt(
        self,
        question: str,
        answer_type: str = "open_ended",
        options: Optional[list] = None,
        image_path: Optional[str] = None,
        modality: str = "",
    ) -> str:
        """Build a VQA prompt optimized for medical VQA."""
        parts = []

        # System context
        if modality:
            parts.append(f"You are a medical imaging expert specializing in {modality}.")
        else:
            parts.append("You are a medical imaging expert.")

        # Image context (for text-only models)
        if not (self._is_vl_model and image_path and self.config.use_images):
            parts.append("[Medical image provided for analysis]")

        # Question
        parts.append(f"\nQuestion: {question}")

        # Multiple choice options
        if options:
            parts.append("\nOptions:")
            for i, opt in enumerate(options):
                letter = chr(65 + i)  # A, B, C, D
                parts.append(f"  {letter}) {opt}")

        # Answer format instruction
        if answer_type == "yes_no":
            parts.append("\nAnswer with 'yes' or 'no':")
        elif answer_type == "choice" and options:
            parts.append("\nAnswer with only the correct option letter (A, B, C, or D):")
        elif answer_type == "number":
            parts.append("\nAnswer with a number:")
        else:
            parts.append("\nProvide a concise answer:")

        return "\n".join(parts)

    def _generate_vl_answer(self, prompt: str, image_path: str) -> str:
        """Generate answer using a Vision-Language model with image input."""
        if self._model_type in ("qwen2_5_vl", "qwen2_vl"):
            return self._generate_qwen_vl(prompt, image_path)
        else:
            # Fallback to text-only
            return self._generate_text_answer(prompt)

    def _generate_qwen_vl(self, prompt: str, image_path: str) -> str:
        """Generate using Qwen2.5-VL with image."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                do_sample=self.config.temperature > 0,
            )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        response = self.processor.decode(generated, skip_special_tokens=True)
        return response.strip()

    def _generate_text_answer(self, prompt: str) -> str:
        """Generate answer using text-only model."""
        messages = [
            {
                "role": "system",
                "content": "You are a medical imaging expert. Answer medical visual questions concisely and accurately.",
            },
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=4096
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        return response.strip()

    def _log_benchmark_result(self, benchmark: str, results: dict):
        """Log a single benchmark result."""
        if "error" in results:
            logger.error(f"  {benchmark}: ERROR - {results['error']}")
            return

        metrics = results.get("metrics", {})
        total = results.get("total", 0)
        logger.info(f"  {benchmark} ({total} samples):")
        for m, v in metrics.items():
            logger.info(f"    {m}: {v:.4f}")

    def _save_results(self, all_results: dict):
        """Save evaluation results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            Path(self.config.output_dir)
            / f"vqa_{self.config.model_name}_{timestamp}.json"
        )

        # Build report (without per-sample details for main report)
        report = {
            "model_name": self.config.model_name,
            "model_path": self.config.model_name_or_path,
            "is_vl_model": self._is_vl_model,
            "use_images": self.config.use_images,
            "timestamp": datetime.now().isoformat(),
            "benchmarks": {},
        }

        for bench, results in all_results.items():
            if "error" in results:
                report["benchmarks"][bench] = {"error": results["error"]}
            else:
                report["benchmarks"][bench] = {
                    "total": results["total"],
                    "metrics": results.get("metrics", {}),
                    "by_answer_type": {
                        at: {"count": info["count"], "metrics": dict(info["metrics"])}
                        for at, info in results.get("by_answer_type", {}).items()
                    },
                }

        # Save main report
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Results saved to {output_path}")

        # Save detailed per-sample results separately
        detail_path = output_path.with_suffix(".detail.json")
        detail_report = {}
        for bench, results in all_results.items():
            if "error" not in results:
                detail_report[bench] = results.get("per_sample", [])

        with open(detail_path, "w") as f:
            json.dump(detail_report, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed results saved to {detail_path}")

    def _print_summary(self, all_results: dict):
        """Print evaluation summary table."""
        print(f"\n{'='*70}")
        print(f"  VQA BENCHMARK RESULTS: {self.config.model_name}")
        print(f"  Model: {self.config.model_name_or_path}")
        print(f"  VL Model: {self._is_vl_model} | Images: {self.config.use_images}")
        print(f"{'='*70}")

        # Header
        metric_names = self.config.metrics[:4]  # Show up to 4 metrics
        header = f"  {'Benchmark':<20} {'N':>6}"
        for m in metric_names:
            header += f" {m:>12}"
        print(header)
        print(f"  {'─'*66}")

        # Per-benchmark rows
        all_metrics = {m: [] for m in metric_names}
        for bench, results in all_results.items():
            if "error" in results:
                print(f"  {bench:<20} {'ERROR':>6}  {results['error'][:40]}")
                continue

            total = results.get("total", 0)
            metrics = results.get("metrics", {})
            row = f"  {bench:<20} {total:>6}"
            for m in metric_names:
                v = metrics.get(m, 0.0)
                row += f" {v:>12.4f}"
                all_metrics[m].append(v)
            print(row)

        # Average row
        print(f"  {'─'*66}")
        row = f"  {'AVERAGE':<20} {'':>6}"
        for m in metric_names:
            vals = all_metrics[m]
            avg = sum(vals) / len(vals) if vals else 0
            row += f" {avg:>12.4f}"
        print(row)
        print(f"{'='*70}")


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="VQA Medical Benchmark Evaluation")
    parser.add_argument("--model", required=True, help="Model path or name")
    parser.add_argument("--model-name", default="BIOAgent-VL", help="Display name")
    parser.add_argument(
        "--benchmarks", nargs="+",
        default=["vqa_rad", "slake", "pathvqa"],
        choices=list(VQA_DATASET_REGISTRY.keys()),
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output-dir", default="logs/benchmarks/vqa")
    parser.add_argument("--no-images", action="store_true",
                        help="Disable image input (text-only)")
    parser.add_argument(
        "--metrics", nargs="+",
        default=["exact_match", "token_f1", "bleu", "contains"],
        choices=["exact_match", "token_f1", "bleu", "contains", "bertscore"],
    )
    parser.add_argument("--backend", default="transformers",
                        choices=["transformers", "vllm"])

    args = parser.parse_args()

    config = VQABenchmarkConfig(
        model_name_or_path=args.model,
        model_name=args.model_name,
        backend=args.backend,
        benchmarks=args.benchmarks,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        use_images=not args.no_images,
        metrics=args.metrics,
    )

    evaluator = VQABenchmarkEvaluator(config)
    evaluator.evaluate_all()


if __name__ == "__main__":
    main()
