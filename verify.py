"""
verify.py — Verify the fine-tuned ToxiGuard model on required test sentences.
=============================================================================
Loads the trained model from ./model and evaluates the 5 verification sentences,
displaying probability, threshold, and trigger status for all 6 labels.
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from predictor import ToxicityPredictor, LABELS

TEST_SENTENCES = [
    {
        "id": 1,
        "text": "This article is very informative, thank you!",
        "expected": "All labels LOW / Safe",
    },
    {
        "id": 2,
        "text": "You are a complete idiot and nobody wants you here.",
        "expected": "TOXIC and INSULT HIGH",
    },
    {
        "id": 3,
        "text": "I will find you and make you regret saying that.",
        "expected": "THREAT HIGH",
    },
    {
        "id": 4,
        "text": "I hate all people from that religion, they should not be allowed here.",
        "expected": "IDENTITY_HATE HIGH",
    },
    {
        "id": 5,
        "text": "I am a Muslim woman.",
        "expected": "All labels LOW (identity mention without toxicity / bias check)",
    },
]

def main():
    print("=" * 80)
    print("TOXIGUARD MODEL VERIFICATION")
    print("=" * 80)

    # Initialize predictor
    predictor = ToxicityPredictor(model_dir="./model")

    print(f"\nModel loaded from: ./model")
    print(f"Device: {predictor.device}")
    print(f"Demo Mode: {predictor.is_demo} (Expected: False)\n")

    if predictor.is_demo:
        print("[FAIL] Warning: Model is still running in DEMO mode! Ensure ./model/config.json exists.")
    else:
        print("[OK] Fine-tuned model detected successfully!")

    print("\nCurrent Decision Thresholds:")
    for lab in LABELS:
        thr = predictor.thresholds.get(lab, 0.5)
        print(f"  * {lab:<16}: {thr:.3f}")

    print("\n" + "=" * 80)
    print("RUNNING 5 BENCHMARK SENTENCES")
    print("=" * 80)

    for item in TEST_SENTENCES:
        idx = item["id"]
        text = item["text"]
        expected = item["expected"]

        res = predictor.predict(text)
        triggered = res["labels_triggered"]

        print(f"\n[{idx}] \"{text}\"")
        print(f"    Expected: {expected}")
        print(f"    Triggered: {triggered if triggered else 'None (Safe)'}")
        print(f"    Uncertain: {res['is_uncertain']}")
        print("    Label Probabilities:")
        for lab in LABELS:
            p = res["probabilities"][lab]
            thr = predictor.thresholds.get(lab, 0.5)
            flag = "[TRIGGER]" if p >= thr else "[OK]     "
            print(f"      {flag} | {lab:<16}: {p:.4f} (threshold: {thr:.3f})")

    print("\n" + "=" * 80)
    print("Verification completed successfully.")
    print("=" * 80)

if __name__ == "__main__":
    main()
