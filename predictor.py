"""
predictor.py — Model loading, inference, and confidence logic.
Wraps the DistilBERT multi-label toxicity classifier so the Streamlit
app has a single, clean interface.
"""

import json, os, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

warnings.filterwarnings("ignore")

# ── project imports ──────────────────────────────────────────────────
from toxic_preprocessing import clean_text

# ── constants ────────────────────────────────────────────────────────
LABELS = ["toxic", "obscene", "threat", "insult", "identity_hate", "sexual_explicit"]
LABEL_DISPLAY = {
    "toxic": "☣️ Toxic",
    "obscene": "🤬 Obscene",
    "threat": "⚠️ Threat",
    "insult": "💢 Insult",
    "identity_hate": "🚫 Identity Hate",
    "sexual_explicit": "🔞 Sexually Explicit",
}

# ── helpers ──────────────────────────────────────────────────────────

def _load_thresholds(path: Optional[str] = None) -> Dict[str, float]:
    """Load per-label decision thresholds (from Member 2's tuning)."""
    base_dir = Path(__file__).resolve().parent
    search = [
        path,
        "thresholds.json",
        str(base_dir / "thresholds.json"),
        "member2_outputs/thresholds.json",
        str(base_dir / "member2_outputs" / "thresholds.json"),
    ]
    for p in search:
        if p and Path(p).exists():
            with open(p) as f:
                return json.load(f)
    # hard-coded fallback from the notebook output
    return {
        "toxic": 0.720, "obscene": 0.900, "threat": 0.863,
        "insult": 0.789, "identity_hate": 0.821, "sexual_explicit": 0.920,
    }


def _load_calibration(path: Optional[str] = None) -> Optional[Dict]:
    """Load temperature-scaling config (from Member 4's notebook)."""
    search = [path, "calibration_config.json", "member4_outputs/calibration_config.json"]
    for p in search:
        if p and Path(p).exists():
            with open(p) as f:
                return json.load(f)
    return None


# ── Predictor class ──────────────────────────────────────────────────

class ToxicityPredictor:
    """
    Wraps DistilBERT for multi-label toxicity prediction.

    Usage:
        pred = ToxicityPredictor()
        result = pred.predict("You are terrible!")
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        threshold_path: Optional[str] = None,
        calibration_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.threshold_path = threshold_path
        self.thresholds = _load_thresholds(threshold_path)
        self.calibration = _load_calibration(calibration_path)
        self.model = None
        self.tokenizer = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model(model_dir)

    # ── model loading ────────────────────────────────────────────────

    def _load_model(self, model_dir: Optional[str]):
        """Try local model dir, then fall back to HuggingFace hub."""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        base_dir = Path(__file__).resolve().parent
        search_dirs = [
            model_dir,
            "model",
            str(base_dir / "model"),
            "member2_outputs/model",
            str(base_dir / "member2_outputs" / "model"),
            "distilbert_toxicity",
        ]

        for d in search_dirs:
            if d and Path(d).exists() and (Path(d) / "config.json").exists():
                print(f"[Predictor] Loading fine-tuned model from {d}")
                self.tokenizer = AutoTokenizer.from_pretrained(d)
                self.model = AutoModelForSequenceClassification.from_pretrained(d)
                self.model.to(self.device).eval()
                self._demo_mode = False
                return

        # Fallback: use base distilbert (demo mode — will give random-ish outputs)
        print("[Predictor] No fine-tuned model found. Loading base distilbert-base-cased for DEMO mode.")
        print("[Predictor] To use your trained model, place the model/ and tokenizer/ folders here.")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-cased", num_labels=len(LABELS), problem_type="multi_label_classification"
        )
        self.model.to(self.device).eval()
        self._demo_mode = True

    # ── calibration ──────────────────────────────────────────────────

    def _calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling if calibration config is available."""
        if self.calibration is None:
            return self._sigmoid(logits)
        temps = self.calibration.get("temperatures", {})
        probs = np.zeros_like(logits)
        for j, lab in enumerate(LABELS):
            t = temps.get(lab, 1.0)
            probs[j] = self._sigmoid(logits[j] / t)
        return probs

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    # ── prediction ───────────────────────────────────────────────────

    def predict(self, text: str) -> Dict:
        """
        Run the full pipeline on a single comment.

        Returns dict with keys:
            cleaned_text, probabilities, predictions, confidence_scores,
            is_uncertain, labels_triggered, max_confidence, review_reason
        """
        self.thresholds = _load_thresholds(self.threshold_path)
        cleaned = clean_text(text)

        inputs = self.tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=240,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits.cpu().numpy()[0]

        probs = self._calibrate(logits)

        # apply per-label thresholds
        preds = {}
        confs = {}
        triggered = []
        for i, lab in enumerate(LABELS):
            thr = self.thresholds.get(lab, 0.5)
            p = float(probs[i])
            preds[lab] = int(p >= thr)
            # confidence = distance from threshold, normalised
            confs[lab] = round(abs(p - thr) / max(thr, 1 - thr), 4)
            if preds[lab]:
                triggered.append(lab)

        # uncertainty detection
        prob_dict = {lab: round(float(probs[i]), 6) for i, lab in enumerate(LABELS)}
        uncertain_labels = []
        review_reasons = []

        for lab in LABELS:
            thr = self.thresholds.get(lab, 0.5)
            p = prob_dict[lab]
            margin = abs(p - thr)
            if margin < 0.10:
                uncertain_labels.append(lab)
                review_reasons.append(
                    f"{LABEL_DISPLAY[lab]} probability ({p:.3f}) is close to threshold ({thr:.3f})"
                )

        is_uncertain = len(uncertain_labels) > 0

        return {
            "original_text": text,
            "cleaned_text": cleaned,
            "probabilities": prob_dict,
            "predictions": preds,
            "confidence_scores": confs,
            "is_uncertain": is_uncertain,
            "uncertain_labels": uncertain_labels,
            "labels_triggered": triggered,
            "max_probability": max(prob_dict.values()),
            "review_reasons": review_reasons,
        }

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """Run prediction on a list of texts."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for t in batch:
                results.append(self.predict(t))
        return results

    @property
    def is_demo(self) -> bool:
        return getattr(self, "_demo_mode", False)
