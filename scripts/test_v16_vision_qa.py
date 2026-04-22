"""
Quick test: Load v16 VLM checkpoint and run vision QA inference.
Verifies that the full VLM (vision + text) forward pass works with image input.
"""

import sys
import torch

# cuDNN conv3d fails on this GPU config; disable and fall back to native impl
torch.backends.cudnn.enabled = False

from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

MODEL_PATH = (
    "/data/project/private/minstar/workspace/verl/checkpoints/"
    "bioagents-verl-grpo/qwen3_5_9b_self_distill_v16_ema_cosine/"
    "global_step_60/actor/merged_hf"
)

def make_dummy_image() -> Image.Image:
    """Create a simple test image with colored blocks and text."""
    img = Image.new("RGB", (224, 224), color=(200, 220, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 100, 100], fill=(255, 80, 80))
    draw.rectangle([120, 20, 200, 100], fill=(80, 255, 80))
    draw.rectangle([20, 120, 100, 200], fill=(80, 80, 255))
    draw.rectangle([120, 120, 200, 200], fill=(255, 255, 80))
    draw.text((70, 105), "TEST IMAGE", fill=(0, 0, 0))
    return img


def main() -> None:
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    print(f"\nLoading processor from: {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("Processor loaded OK")

    print(f"\nLoading model from: {MODEL_PATH}")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="cuda:0",  # CUDA_VISIBLE_DEVICES=1 → logical device 0
        trust_remote_code=True,
    )
    model.eval()

    # Count and report parameters
    total_params = sum(p.numel() for p in model.parameters())
    vision_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if "visual" in name or "vision" in name
    )
    print(f"Model loaded OK  (total params: {total_params/1e9:.2f}B, "
          f"vision params: {vision_params/1e6:.1f}M)")

    # Build test image
    image = make_dummy_image()
    print(f"\nDummy image created: {image.size} {image.mode}")

    # Build chat messages with image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What do you see in this image? Describe the colors and shapes briefly."},
            ],
        }
    ]

    # Apply chat template
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process inputs
    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    # Move all tensors to the model device; cast pixel_values to bfloat16
    processed_inputs = {}
    for k, v in inputs.items():
        if hasattr(v, "to"):
            if k == "pixel_values":
                processed_inputs[k] = v.to(model.device, dtype=torch.bfloat16)
            else:
                processed_inputs[k] = v.to(model.device)
        else:
            processed_inputs[k] = v
    inputs = processed_inputs

    print(f"\nInput shapes:")
    for k, v in inputs.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {v.shape} dtype={v.dtype}")

    # Run inference
    print("\nRunning forward pass (generate)...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][input_len:]
    response = processor.decode(new_tokens, skip_special_tokens=True)

    print("\n" + "=" * 60)
    print("VISION QA RESULT")
    print("=" * 60)
    print(f"Prompt: What do you see in this image?")
    print(f"Response: {response}")
    print("=" * 60)
    print("\nSUCCESS: v16 VLM vision forward pass works correctly!")


if __name__ == "__main__":
    main()
