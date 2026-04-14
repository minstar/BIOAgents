#!/usr/bin/env python3
"""Merge LoRA adapter into base model."""
import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

parser = argparse.ArgumentParser()
parser.add_argument("--base-model", default="checkpoints/models/Lingshu-7B")
parser.add_argument("--adapter", default="checkpoints/sft_p2_aggressive_lingshu/final")
parser.add_argument("--output", default="checkpoints/sft_p2_aggressive_lingshu/merged")
args = parser.parse_args()

base_path = args.base_model
adapter_path = args.adapter
merged_path = args.output

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

# Copy processor files for VL models (preprocessor_config.json, etc.)
if is_qwen_vl:
    import shutil
    for fname in ("preprocessor_config.json", "chat_template.json"):
        src = Path(base_path) / fname
        if src.exists():
            shutil.copy2(src, Path(merged_path) / fname)
            print(f"Copied {fname} from base model")

print(f"Merged model saved to {merged_path}")
