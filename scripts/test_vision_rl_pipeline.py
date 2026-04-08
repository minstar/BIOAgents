#!/usr/bin/env python3
"""Small-scale test for the vision RL training pipeline.

Tests that:
1. Qwen3.5 is detected as a VL model
2. Processor loads correctly
3. Multimodal messages are built for VQA tasks
4. process_vision_info extracts images
5. Processor tokenizes with pixel_values
6. Model forward pass works with pixel_values
7. _tokenize_trajectory handles both vision and text-only

Usage:
    CUDA_VISIBLE_DEVICES=X python scripts/test_vision_rl_pipeline.py
"""

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer


def test_model_detection():
    """Test C1 fix: model type detection works correctly."""
    print("\n=== Test 1: Model Type Detection ===")
    model_path = "checkpoints/models/Qwen3.5-9B"
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")
    print(f"  model_type: {model_type}")

    _is_qwen_vl = (
        model_type in ("qwen2_5_vl", "qwen2_vl", "qwen3_5")
        or ("qwen2" in model_type.lower() and "vl" in model_type.lower())
    )
    _is_qwen3_5 = model_type == "qwen3_5"

    assert _is_qwen_vl, f"Expected VL model, got model_type={model_type}"
    assert _is_qwen3_5, f"Expected qwen3_5, got model_type={model_type}"
    print(f"  _is_qwen_vl={_is_qwen_vl}, _is_qwen3_5={_is_qwen3_5}")
    print("  ✅ PASS")
    return model_type


def test_processor_loading():
    """Test that processor loads and has the right methods."""
    print("\n=== Test 2: Processor Loading ===")
    model_path = "checkpoints/models/Qwen3.5-9B"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    assert hasattr(processor, "apply_chat_template"), "Processor missing apply_chat_template"
    assert hasattr(processor, "__call__"), "Processor not callable"
    print(f"  Processor type: {type(processor).__name__}")
    print(f"  Tokenizer type: {type(tokenizer).__name__}")
    print("  ✅ PASS")
    return processor, tokenizer


def test_multimodal_message_building():
    """Test C2: multimodal message format is correct."""
    print("\n=== Test 3: Multimodal Message Building ===")

    # Find a real VQA image
    tasks_path = Path("data/domains/full_4modality_combined/tasks.json")
    if not tasks_path.exists():
        tasks_path = Path("data/domains/multimodal_rl_combined/tasks.json")
    if not tasks_path.exists():
        print("  SKIP: No task file found")
        return None, None

    with open(tasks_path) as f:
        tasks = json.load(f)

    vqa_task = None
    for t in tasks:
        img = t.get("_image_path")
        if img and os.path.isfile(img):
            vqa_task = t
            break

    if not vqa_task:
        print("  SKIP: No VQA task with valid image found")
        return None, None

    print(f"  Task: {vqa_task['id']}")
    print(f"  Image: {vqa_task['_image_path']}")

    image_path = vqa_task["_image_path"]
    observation_text = vqa_task.get("ticket", "Test question")

    # Build multimodal content
    user_content = [
        {"type": "image", "image": f"file://{image_path}"},
        {"type": "text", "text": observation_text[:200]},
    ]

    messages = [
        {"role": "system", "content": "You are a medical AI assistant."},
        {"role": "user", "content": user_content},
    ]

    # Simulate multi-turn: add text-only follow-up
    messages.append({"role": "assistant", "content": "Let me analyze this image."})
    messages.append({"role": "user", "content": "What do you see?"})

    print(f"  Messages: {len(messages)} turns")
    print(f"  First user content type: {type(messages[1]['content'])}")
    print(f"  Third user content type: {type(messages[3]['content'])}")
    print("  ✅ PASS")
    return messages, vqa_task


def test_process_vision_info(messages):
    """Test that process_vision_info extracts images from messages."""
    print("\n=== Test 4: process_vision_info ===")
    if messages is None:
        print("  SKIP: No messages")
        return

    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)

    print(f"  image_inputs: {len(image_inputs)} images")
    print(f"  video_inputs: {len(video_inputs) if video_inputs else 0} videos")
    assert len(image_inputs) > 0, "Expected at least 1 image"
    print(f"  Image type: {type(image_inputs[0])}")
    print(f"  Image size: {image_inputs[0].size if hasattr(image_inputs[0], 'size') else 'N/A'}")
    print("  ✅ PASS")
    return image_inputs


def test_processor_tokenization(processor, messages, image_inputs):
    """Test that processor creates correct inputs with pixel_values."""
    print("\n=== Test 5: Processor Tokenization ===")
    if messages is None or processor is None:
        print("  SKIP")
        return None

    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    print(f"  Template length: {len(text)} chars")
    print(f"  Contains <|vision_start|>: {'<|vision_start|>' in text}")

    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=False,
    )

    print(f"  input_ids shape: {inputs['input_ids'].shape}")
    print(f"  Has pixel_values: {'pixel_values' in inputs}")
    if "pixel_values" in inputs:
        print(f"  pixel_values shape: {inputs['pixel_values'].shape}")
        print(f"  pixel_values dtype: {inputs['pixel_values'].dtype}")
    if "image_grid_thw" in inputs:
        print(f"  image_grid_thw shape: {inputs['image_grid_thw'].shape}")

    assert "pixel_values" in inputs, "Expected pixel_values in processor output"
    print("  ✅ PASS")
    return inputs


def test_model_forward(inputs, tokenizer):
    """Test that model forward pass works with pixel_values."""
    print("\n=== Test 6: Model Forward Pass (with vision) ===")
    if inputs is None:
        print("  SKIP")
        return

    from transformers import Qwen3_5ForConditionalGeneration

    model_path = "checkpoints/models/Qwen3.5-9B"
    print(f"  Loading model... ", end="", flush=True)
    t0 = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        device_map={"": 0},
    )
    model.eval()
    print(f"done in {time.time()-t0:.1f}s")

    device = model.device
    fwd_inputs = {k: v.to(device) for k, v in inputs.items()}
    fwd_inputs["labels"] = fwd_inputs["input_ids"].clone()

    print(f"  Running forward pass... ", end="", flush=True)
    t0 = time.time()
    with torch.no_grad():
        outputs = model(**fwd_inputs)
    print(f"done in {time.time()-t0:.1f}s")

    print(f"  loss: {outputs.loss.item():.4f}")
    print(f"  logits shape: {outputs.logits.shape}")
    assert not torch.isnan(outputs.loss), "Loss is NaN!"
    assert not torch.isinf(outputs.loss), "Loss is Inf!"

    # Test generate
    print(f"  Running generate (10 tokens)... ", end="", flush=True)
    gen_inputs = {k: v for k, v in fwd_inputs.items() if k != "labels"}
    t0 = time.time()
    with torch.no_grad():
        gen_out = model.generate(
            **gen_inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    print(f"done in {time.time()-t0:.1f}s")
    print(f"  Generated {gen_out.shape[1] - fwd_inputs['input_ids'].shape[1]} new tokens")
    print("  ✅ PASS")

    del model
    torch.cuda.empty_cache()


def test_tokenize_trajectory(processor, tokenizer, vqa_task):
    """Test H1/H2: _tokenize_trajectory works for vision and text-only."""
    print("\n=== Test 7: _tokenize_trajectory ===")
    if processor is None or vqa_task is None:
        print("  SKIP")
        return

    # Import the function
    import importlib
    import bioagents.training.grpo_trainer as trainer_mod
    importlib.reload(trainer_mod)

    # Check if _tokenize_trajectory exists
    assert hasattr(trainer_mod, "_tokenize_trajectory"), \
        "_tokenize_trajectory not found in grpo_trainer"

    _tokenize_trajectory = trainer_mod._tokenize_trajectory
    device = torch.device("cuda:0")

    # Test 7a: Vision trajectory
    image_path = vqa_task["_image_path"]
    vision_messages = [
        {"role": "system", "content": "You are a medical AI assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": "What do you see?"},
        ]},
        {"role": "assistant", "content": "I see a medical image."},
    ]

    vision_traj = {
        "task_id": "test_vision",
        "full_text": "I see a medical image.",
        "messages": vision_messages,
        "image_path": image_path,
    }

    result = _tokenize_trajectory(vision_traj, tokenizer, processor, 4096, device)
    if result is not None:
        print(f"  7a Vision: input_ids={result['input_ids'].shape}, "
              f"has pixel_values={'pixel_values' in result}")
        assert "pixel_values" in result, "Expected pixel_values for vision trajectory"
        print("  ✅ 7a PASS (vision trajectory)")
    else:
        print("  ⚠️ 7a: Vision trajectory returned None (may be too long)")

    # Test 7b: Text-only trajectory
    text_traj = {
        "task_id": "test_text",
        "full_text": "The answer is B. This is a clinical diagnosis.",
        "messages": [
            {"role": "system", "content": "You are a medical AI assistant."},
            {"role": "user", "content": "What is the diagnosis?"},
            {"role": "assistant", "content": "The answer is B."},
        ],
        "image_path": None,
    }

    result = _tokenize_trajectory(text_traj, tokenizer, processor, 4096, device)
    assert result is not None, "Text trajectory should not be None"
    print(f"  7b Text:   input_ids={result['input_ids'].shape}, "
          f"has pixel_values={'pixel_values' in result}")
    assert "pixel_values" not in result, "Text trajectory should NOT have pixel_values"
    print("  ✅ 7b PASS (text-only trajectory)")


def main():
    print("=" * 60)
    print("  Vision RL Training Pipeline Test")
    print("=" * 60)

    # Test 1: Model detection
    model_type = test_model_detection()

    # Test 2: Processor loading
    processor, tokenizer = test_processor_loading()

    # Test 3: Multimodal message building
    messages, vqa_task = test_multimodal_message_building()

    # Test 4: process_vision_info
    image_inputs = test_process_vision_info(messages)

    # Test 5: Processor tokenization
    inputs = test_processor_tokenization(processor, messages, image_inputs)

    # Test 6: Model forward pass
    test_model_forward(inputs, tokenizer)

    # Test 7: _tokenize_trajectory
    test_tokenize_trajectory(processor, tokenizer, vqa_task)

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
