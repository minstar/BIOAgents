#!/usr/bin/env python3
"""Merge OPD LoRA adapters into full models for evaluation."""
import sys
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoTokenizer, AutoConfig
from transformers import Qwen2_5_VLForConditionalGeneration

BASE_PATH = "checkpoints/drgrpo_lingshu7b/checkpoint-950-merged"

MERGE_QUEUE = [
    ("checkpoints/opd_cross_stage_lingshu7b/best", "checkpoints/opd_cross_stage_lingshu7b/best_merged", "OPD_alpha05"),
    ("checkpoints/opd_cross_stage_low_alpha_lingshu7b/best", "checkpoints/opd_cross_stage_low_alpha_lingshu7b/best_merged", "OPD_alpha03"),
]


def merge_adapter(adapter_path, output_path, name):
    output = Path(output_path)
    if output.exists() and (output / "config.json").exists():
        print(f"[SKIP] {name} already merged at {output_path}")
        return

    print(f"[{name}] Loading base model from {BASE_PATH}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_PATH, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    print(f"[{name}] Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print(f"[{name}] Merging...")
    model = model.merge_and_unload()

    print(f"[{name}] Saving to {output_path}...")
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"[{name}] Done!")


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    if idx >= 0:
        adapter_path, output_path, name = MERGE_QUEUE[idx]
        merge_adapter(adapter_path, output_path, name)
    else:
        for adapter_path, output_path, name in MERGE_QUEUE:
            merge_adapter(adapter_path, output_path, name)
