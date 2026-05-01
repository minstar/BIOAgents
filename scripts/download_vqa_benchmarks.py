"""
Download and prepare 3 VQA medical benchmarks for evaluation.
Each dataset gets: datasets/vqa/<name>/test.json + datasets/vqa/<name>/images/

Sources:
- PMC-VQA: jspetrisko/PMC-VQA-Test-Clean (2000 test examples with images)
- VQA-Med-2021: bangthe2222/vqa_med (425 test examples)
- Quilt-VQA: wisdomik/Quilt_VQA (gated, requires HF auth)
"""

import io
import json
import os
import sys
from pathlib import Path

import datasets.features.image as img_mod
from PIL import Image as PILImage

# Monkey-patch to fix PIL.Image.ExifTags AttributeError on older Pillow versions
_orig_decode = img_mod.Image.decode_example


def _patched_decode(self, value, token_per_repo_id=None):
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return PILImage.open(io.BytesIO(value["bytes"]))
        if value.get("path") is not None:
            return PILImage.open(value["path"])
    return _orig_decode(self, value, token_per_repo_id=token_per_repo_id)


img_mod.Image.decode_example = _patched_decode

from datasets import load_dataset  # noqa: E402

BASE_DIR = Path("/data/project/private/minstar/workspace/BIOAgents/datasets/vqa")


def save_dataset(name: str, dataset, question_key: str, answer_key: str, image_key: str = "image"):
    """Save a HuggingFace dataset to test.json + images/."""
    out_dir = BASE_DIR / name
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total = len(dataset)
    skipped = 0
    for idx, row in enumerate(dataset):
        if idx % 200 == 0:
            print(f"  [{name}] Processing {idx}/{total}...")

        question = str(row[question_key]).strip()
        answer = str(row[answer_key]).strip()

        img_filename = f"{idx:06d}.jpg"
        img_path = img_dir / img_filename

        if not img_path.exists():
            img = row.get(image_key)
            if img is None:
                skipped += 1
                continue
            try:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(str(img_path), "JPEG", quality=95)
            except Exception as e:
                print(f"  [{name}] WARNING: Failed to save image {idx}: {e}")
                skipped += 1
                continue

        records.append({
            "question": question,
            "answer": answer,
            "image_path": f"images/{img_filename}",
        })

    test_json_path = out_dir / "test.json"
    with open(test_json_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"  [{name}] Saved {len(records)} records ({skipped} skipped) -> {test_json_path}")
    return len(records)


def download_pmc_vqa():
    """PMC-VQA test set (2000 examples) from jspetrisko/PMC-VQA-Test-Clean."""
    print("\n=== PMC-VQA ===")
    try:
        # jspetrisko/PMC-VQA-Test-Clean has images embedded, only 'train' split
        ds = load_dataset("jspetrisko/PMC-VQA-Test-Clean", split="train")
        print(f"  Loaded {len(ds)} examples from jspetrisko/PMC-VQA-Test-Clean")
        print(f"  Columns: {ds.column_names}")
        # Columns: query, answer, image
        return save_dataset("pmc_vqa", ds, "query", "answer", "image")
    except Exception as e:
        print(f"  ERROR: {e}")
        return 0


def download_vqa_med_2021():
    """VQA-Med-2021 test set (425 examples) from bangthe2222/vqa_med."""
    print("\n=== VQA-Med-2021 ===")
    try:
        ds = load_dataset("bangthe2222/vqa_med", split="test")
        print(f"  Loaded {len(ds)} examples from bangthe2222/vqa_med")
        print(f"  Columns: {ds.column_names}")
        # Columns: image, question, answer
        return save_dataset("vqa_med_2021", ds, "question", "answer", "image")
    except Exception as e:
        print(f"  ERROR: {e}")
        return 0


def download_quilt_vqa():
    """Quilt-VQA from wisdomik/Quilt_VQA (gated - requires HF auth)."""
    print("\n=== Quilt-VQA ===")
    try:
        ds = load_dataset("wisdomik/Quilt_VQA", split="test")
        print(f"  Loaded {len(ds)} examples")
        print(f"  Columns: {ds.column_names}")

        q_key = "question" if "question" in ds.column_names else "Question"
        a_key = "answer" if "answer" in ds.column_names else "Answer"
        img_key = "image" if "image" in ds.column_names else "Image"
        return save_dataset("quilt_vqa", ds, q_key, a_key, img_key)
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  NOTE: wisdomik/Quilt_VQA is a gated dataset.")
        print("  You need to: 1) Request access at https://huggingface.co/datasets/wisdomik/Quilt_VQA")
        print("               2) Run 'huggingface-cli login' with your token")
        print("               3) Re-run this script with: python3 scripts/download_vqa_benchmarks.py quilt_vqa")
        return 0


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["pmc_vqa", "vqa_med_2021", "quilt_vqa"]
    results = {}

    if "pmc_vqa" in targets:
        results["pmc_vqa"] = download_pmc_vqa()
    if "vqa_med_2021" in targets:
        results["vqa_med_2021"] = download_vqa_med_2021()
    if "quilt_vqa" in targets:
        results["quilt_vqa"] = download_quilt_vqa()

    print("\n=== Summary ===")
    for name, count in results.items():
        status = "OK" if count > 0 else "FAILED"
        print(f"  {name}: {count} records [{status}]")
