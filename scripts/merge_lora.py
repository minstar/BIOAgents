#!/usr/bin/env python3
"""Merge LoRA adapter into base model."""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

base_path = "checkpoints/models/Lingshu-7B"
adapter_path = "checkpoints/sft_p2_aggressive_lingshu/final"
merged_path = "checkpoints/sft_p2_aggressive_lingshu/merged"

print("Loading base model...")
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel

config = AutoConfig.from_pretrained(base_path, trust_remote_code=True)
model_type = getattr(config, "model_type", "")
is_qwen_vl = model_type in ("qwen2_5_vl", "qwen2_vl")

if is_qwen_vl:
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

print("Loading adapter...")
model = PeftModel.from_pretrained(model, adapter_path)
print("Merging...")
model = model.merge_and_unload()
print("Saving merged model...")
model.save_pretrained(merged_path)

tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
tokenizer.save_pretrained(merged_path)
print(f"Merged model saved to {merged_path}")
