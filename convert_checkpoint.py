"""
convert_checkpoint.py — Convert train_state.pt into HuggingFace ./model format
=============================================================================
Loads the 6,000-step checkpoint from member2_outputs_v2/checkpoint/train_state.pt,
exports model.safetensors, config.json, and tokenizer files into ./model,
and sets thresholds.json.
"""

import json
import os
import sys
from pathlib import Path

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

LABELS = ["toxic", "obscene", "threat", "insult", "identity_hate", "sexual_explicit"]
ID2LABEL = {i: lab for i, lab in enumerate(LABELS)}
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}

# Pre-tuned optimal thresholds from Member 2's validation search
TUNED_THRESHOLDS = {
    "toxic": 0.720,
    "obscene": 0.900,
    "threat": 0.863,
    "insult": 0.789,
    "identity_hate": 0.821,
    "sexual_explicit": 0.920,
}

def main():
    ckpt_path = Path("member2_outputs_v2/checkpoint/train_state.pt")
    out_dir = Path("model")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CONVERTING train_state.pt TO ./model FORMAT")
    print("=" * 80)
    print(f"Loading checkpoint from: {ckpt_path.resolve()}")

    state = torch.load(ckpt_path, map_location="cpu")
    print(f"Checkpoint keys: {list(state.keys())}")
    meta = state.get("state", {})
    print(f"Global training steps: {meta.get('global_step', 'unknown')}")

    # Config
    print("\n[1/4] Configuring DistilBERT (distilbert-base-cased)...")
    config = AutoConfig.from_pretrained(
        "distilbert-base-cased",
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        problem_type="multi_label_classification",
    )

    # Model
    print("[2/4] Initializing model and loading state_dict...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-cased",
        config=config,
    )
    res = model.load_state_dict(state["model"], strict=True)
    print(f"  Load result: {res}")

    # Save Model
    print(f"[3/4] Exporting model.safetensors and config.json to {out_dir.resolve()}...")
    model.save_pretrained(out_dir, safe_serialization=True)

    # Tokenizer
    print("[4/4] Exporting tokenizer files (tokenizer.json, tokenizer_config.json, vocab.txt)...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    tokenizer.save_pretrained(out_dir)

    # Save vocab.txt
    vocab_dict = tokenizer.get_vocab()
    with open(out_dir / "vocab.txt", "w", encoding="utf-8") as f:
        for word, idx in sorted(vocab_dict.items(), key=lambda x: x[1]):
            f.write(word + "\n")

    # Update thresholds.json
    thr_path = Path("thresholds.json")
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump(TUNED_THRESHOLDS, f, indent=4)
    print(f"\n[Saved] Updated thresholds in {thr_path.resolve()}:")
    for lab, thr in TUNED_THRESHOLDS.items():
        print(f"  * {lab:<16}: {thr:.3f}")

    # Verify all expected files
    expected = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt"]
    print("\nVerification of ./model/ files:")
    all_ok = True
    for fname in expected:
        fp = out_dir / fname
        if fp.exists():
            print(f"  [OK] {fname:<24} ({fp.stat().st_size:,} bytes)")
        else:
            print(f"  [MISSING] {fname}")
            all_ok = False

    if all_ok:
        print("\nAll files exported successfully from train_state.pt!")

if __name__ == "__main__":
    main()
