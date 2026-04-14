#!/usr/bin/env python3
"""Quick TPS benchmark for Qwen3.5-9B with different optimization techniques.
Uses GPU 0 alongside training (shares memory).
"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/data/project/private/minstar/workspace/BIOAgents/checkpoints/models/Qwen3.5-9B"
DEVICE = torch.device("cuda:0")
PROMPT = """You are a medical AI assistant. A patient presents with chest pain, shortness of breath, and elevated troponin levels. What is the most likely diagnosis and what diagnostic workup would you recommend? Please think step by step."""
MAX_NEW_TOKENS = 256


def benchmark(label, model, tokenizer, warmup=1, runs=3):
    """Run benchmark and return avg tok/s."""
    inputs = tokenizer(PROMPT, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=32, do_sample=False,
                           pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)

    # Benchmark
    times = []
    gen_tokens = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                                 temperature=0.9, top_p=0.95,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        n_gen = out.shape[1] - inputs["input_ids"].shape[1]
        times.append(elapsed)
        gen_tokens.append(n_gen)

    avg_time = sum(times) / len(times)
    avg_tokens = sum(gen_tokens) / len(gen_tokens)
    tps = avg_tokens / avg_time
    print(f"[{label}] avg={avg_time:.1f}s, tokens={avg_tokens:.0f}, TPS={tps:.1f}")
    return tps


def main():
    print(f"Loading model from {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === Test 1: Baseline (eager, no SDPA) ===
    print("\n--- Test 1: Baseline (eager) ---")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(DEVICE).eval()
    tps_baseline = benchmark("baseline", model, tokenizer)
    del model; torch.cuda.empty_cache()

    # === Test 2: SDPA ===
    print("\n--- Test 2: SDPA ---")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(DEVICE).eval()
    tps_sdpa = benchmark("sdpa", model, tokenizer)
    del model; torch.cuda.empty_cache()

    # === Test 3: Static cache + torch.compile ===
    print("\n--- Test 3: Static cache + torch.compile ---")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(DEVICE).eval()
    model.generation_config.cache_implementation = "static"
    try:
        model.forward = torch.compile(model.forward, mode="reduce-overhead")
        # Need extra warmup for compile
        tps_compile = benchmark("static+compile", model, tokenizer, warmup=3, runs=3)
    except Exception as e:
        print(f"  compile failed: {e}")
        tps_compile = 0
    del model; torch.cuda.empty_cache()

    # === Summary ===
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Baseline:        {tps_baseline:.1f} tok/s")
    print(f"SDPA:            {tps_sdpa:.1f} tok/s ({tps_sdpa/tps_baseline:.2f}x)")
    if tps_compile > 0:
        print(f"Static+Compile:  {tps_compile:.1f} tok/s ({tps_compile/tps_baseline:.2f}x)")


if __name__ == "__main__":
    main()
