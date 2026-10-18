"""
train.py — DistilBERT Multi-Label Toxicity Classifier Training & Threshold Tuning
==================================================================================
Trains distilbert-base-uncased for multi-label classification across 6 labels:
  toxic, obscene, threat, insult, identity_hate, sexual_explicit

Features:
- Handles BCEWithLogitsLoss with pos_weight class imbalance scaling
- Stratified sampling prioritizing rare labels on CPU/sample constraints
- Auto-detects CUDA / GPU with fp16 mixed precision
- Tunes per-label decision thresholds on validation split (maximizing F1)
- Exports model.safetensors, config.json, tokenizer files to ./model
- Saves tuned thresholds to thresholds.json
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure stdout and stderr handle utf-8 safely on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# ── Global Constants ────────────────────────────────────────────────────────
LABELS = ["toxic", "obscene", "threat", "insult", "identity_hate", "sexual_explicit"]
ID2LABEL = {i: lab for i, lab in enumerate(LABELS)}
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}

# ── Dataset Definition ───────────────────────────────────────────────────────
class ToxicityDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


# ── Data Loading & Preprocessing ────────────────────────────────────────────
def load_and_preprocess_data(
    data_path: str,
    val_path: str = None,
    sample_size: int = 200000,
    seed: int = 42,
    device_type: str = "cpu",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads dataset from parquet or csv, standardizes columns and labels,
    and returns (train_df, val_df).
    """
    p = Path(data_path)
    if not p.exists():
        # Check standard fallback paths
        fallbacks = [
            Path("processed_data/processed_data/train.parquet"),
            Path("processed_data/train.parquet"),
            Path("train.parquet"),
            Path("train.csv"),
        ]
        found = False
        for fb in fallbacks:
            if fb.exists():
                p = fb
                found = True
                break
        if not found:
            raise FileNotFoundError(
                f"Data file not found at '{data_path}' or standard fallbacks. "
                "Please place the Jigsaw dataset or processed_data folder in the workspace."
            )

    print(f"[Data] Loading training data from: {p}")
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    # Standardize column mappings (e.g. identity_attack -> identity_hate)
    if "identity_attack" in df.columns and "identity_hate" not in df.columns:
        df = df.rename(columns={"identity_attack": "identity_hate"})

    # Check text column
    text_col = None
    for cand in ["text_raw", "text_clean", "comment_text", "text"]:
        if cand in df.columns:
            text_col = cand
            break
    if text_col is None:
        raise ValueError(f"Could not find a text column in {df.columns.tolist()}")
    print(f"[Data] Using text column: '{text_col}'")

    # Binarize labels >= 0.5
    for lab in LABELS:
        if lab not in df.columns:
            # Check if soft score version exists
            score_col = f"score_{lab}"
            if score_col in df.columns:
                df[lab] = (df[score_col] >= 0.5).astype(int)
            else:
                df[lab] = 0
        else:
            if df[lab].dtype in [float, np.float32, np.float64]:
                df[lab] = (df[lab] >= 0.5).astype(int)
            else:
                df[lab] = df[lab].astype(int)

    # Clean text column: drop nulls/empty
    df = df[df[text_col].notna()].copy()
    df[text_col] = df[text_col].astype(str)

    # Load or split validation set
    val_df = None
    if val_path and Path(val_path).exists():
        print(f"[Data] Loading explicit validation data from: {val_path}")
        vp = Path(val_path)
        val_df = pd.read_parquet(vp) if vp.suffix == ".parquet" else pd.read_csv(vp)
        if "identity_attack" in val_df.columns and "identity_hate" not in val_df.columns:
            val_df = val_df.rename(columns={"identity_attack": "identity_hate"})
        for lab in LABELS:
            if lab in val_df.columns:
                if val_df[lab].dtype in [float, np.float32, np.float64]:
                    val_df[lab] = (val_df[lab] >= 0.5).astype(int)
                else:
                    val_df[lab] = val_df[lab].astype(int)
            else:
                val_df[lab] = 0
        val_df = val_df[val_df[text_col].notna()].copy()
    elif "split" in df.columns and "val" in df["split"].unique():
        val_df = df[df["split"] == "val"].copy()
        df = df[df["split"] == "train"].copy()

    # Stratified / Positive-aware sampling for train_df
    total_rows = len(df)
    if sample_size and sample_size < total_rows:
        print(f"[Data] Subsampling training set to {sample_size} (from {total_rows:,} rows)...")
        # Ensure rare positives are well-represented
        is_pos = (df[LABELS].sum(axis=1) > 0)
        pos_df = df[is_pos]
        neg_df = df[~is_pos]

        # Target 40% positive, 60% negative or take all positives if fewer
        target_pos = min(len(pos_df), int(sample_size * 0.40))
        target_neg = sample_size - target_pos

        pos_sample = pos_df.sample(n=target_pos, random_state=seed)
        neg_sample = neg_df.sample(n=min(len(neg_df), target_neg), random_state=seed)
        df = pd.concat([pos_sample, neg_sample]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        print(f"[Data] Subsampled composition: {target_pos:,} toxic/flagged comments, {len(neg_sample):,} clean comments.")

    # If no validation set yet, create 90/10 split
    if val_df is None:
        print("[Data] Creating 90/10 train/validation split...")
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        split_idx = int(len(shuffled) * 0.90)
        train_df = shuffled.iloc[:split_idx].copy()
        val_df = shuffled.iloc[split_idx:].copy()
    else:
        train_df = df.reset_index(drop=True)
        # Limit validation size for evaluation speed if on CPU
        if device_type == "cpu" and len(val_df) > 300:
            print(f"[Data] Sampling 300 validation comments for CPU evaluation speed.")
            val_pos = val_df[val_df[LABELS].sum(axis=1) > 0]
            val_neg = val_df[val_df[LABELS].sum(axis=1) == 0]
            n_pos = min(len(val_pos), 150)
            n_neg = 300 - n_pos
            val_df = pd.concat([
                val_pos.sample(n=n_pos, random_state=seed),
                val_neg.sample(n=min(len(val_neg), n_neg), random_state=seed)
            ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(f"[Data] Final sizes -> Train: {len(train_df):,}, Val: {len(val_df):,}")
    return train_df, val_df, text_col


# ── Metric & Threshold Optimization ──────────────────────────────────────────
def tune_thresholds_and_evaluate(
    val_labels: np.ndarray,
    val_probs: np.ndarray,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Finds best F1 threshold per label over a grid from 0.10 to 0.95.
    Returns optimal thresholds dict and detailed evaluation dataframe.
    """
    threshold_grid = np.arange(0.10, 0.96, 0.02)
    best_thresholds = {}
    metrics_records = []

    for idx, lab in enumerate(LABELS):
        y_true = val_labels[:, idx]
        y_prob = val_probs[:, idx]

        best_f1 = -1.0
        best_thr = 0.5
        best_prec = 0.0
        best_rec = 0.0

        for thr in threshold_grid:
            y_pred = (y_prob >= thr).astype(int)
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            f = f1_score(y_true, y_pred, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_thr = thr
                best_prec = p
                best_rec = r

        # Compute ROC-AUC (fallback to 0.5 if only 1 class present)
        try:
            auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
        except Exception:
            auc = 0.5

        best_thresholds[lab] = float(round(best_thr, 4))
        metrics_records.append({
            "Label": lab,
            "Precision": round(best_prec, 4),
            "Recall": round(best_rec, 4),
            "F1-Score": round(best_f1, 4),
            "ROC-AUC": round(auc, 4),
            "Tuned Threshold": round(best_thr, 4),
            "Support": int(y_true.sum()),
        })

    metrics_df = pd.DataFrame(metrics_records)
    return best_thresholds, metrics_df


# ── Training Routine ─────────────────────────────────────────────────────────
def train_model(args):
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        use_amp = True
        print(f"[Hardware] GPU detected: {device_name}. Mixed precision (fp16) ENABLED.")
    else:
        device = torch.device("cpu")
        use_amp = False
        print("[Hardware] No GPU detected. Running on CPU.")
        if args.sample_size > 1000:
            print(f"[Hardware] Adjusting sample_size to 600 for fast CPU completion.")
            args.sample_size = 600
        if args.batch_size > 16:
            args.batch_size = 16

    # Data loading
    train_df, val_df, text_col = load_and_preprocess_data(
        data_path=args.data_path,
        val_path=args.val_path,
        sample_size=args.sample_size,
        seed=args.seed,
        device_type=device.type,
    )

    # Pos weight calculation
    train_labels = train_df[LABELS].values
    pos_counts = train_labels.sum(axis=0)
    neg_counts = len(train_labels) - pos_counts
    # Softened pos_weight = (neg / (pos + 1e-5)) ** 0.5
    raw_weights = neg_counts / (pos_counts + 1e-5)
    softened_weights = np.clip(np.sqrt(raw_weights), 1.0, 30.0)
    pos_weight_tensor = torch.tensor(softened_weights, dtype=torch.float).to(device)

    print("\n[Weights] Positive class weights (BCEWithLogitsLoss):")
    for lab, w in zip(LABELS, softened_weights):
        print(f"  • {lab:<16}: {w:.2f}")

    # Tokenizer & Model
    print(f"\n[Model] Initializing distilbert-base-uncased...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    config = AutoConfig.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        problem_type="multi_label_classification",
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        config=config,
    )
    model.to(device)

    # Datasets & Loaders
    train_dataset = ToxicityDataset(
        texts=train_df[text_col].tolist(),
        labels=train_labels,
        tokenizer=tokenizer,
        max_len=args.max_len,
    )
    val_dataset = ToxicityDataset(
        texts=val_df[text_col].tolist(),
        labels=val_df[LABELS].values,
        tokenizer=tokenizer,
        max_len=args.max_len,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.06)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Training loop
    print(f"\n[Train] Starting training: {args.epochs} epochs, {len(train_loader)} batches/epoch, batch size {args.batch_size}...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            if step % 5 == 0 or step == len(train_loader):
                elapsed = time.time() - epoch_start
                avg_l = total_loss / step
                print(f"  Epoch {epoch}/{args.epochs} | Step {step}/{len(train_loader)} | Loss: {avg_l:.4f} | Time: {elapsed:.1f}s", flush=True)

    train_duration = time.time() - start_time
    print(f"\n[Train] Training completed in {train_duration:.1f}s ({train_duration / 60:.2f} min).")

    # Evaluation on Validation Set for Threshold Tuning
    print("\n[Evaluation] Running validation inference for threshold tuning...")
    model.eval()
    val_preds_list = []
    val_targets_list = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].numpy()

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            probs = torch.sigmoid(logits).cpu().numpy()
            val_preds_list.append(probs)
            val_targets_list.append(targets)

    all_val_probs = np.vstack(val_preds_list)
    all_val_targets = np.vstack(val_targets_list)

    # Step 3: Threshold Tuning
    print("\n[Thresholds] Tuning per-label thresholds to maximize F1...")
    best_thresholds, metrics_df = tune_thresholds_and_evaluate(all_val_targets, all_val_probs)

    print("\n" + "=" * 80)
    print("FINAL MODEL PERFORMANCE & TUNED THRESHOLDS")
    print("=" * 80)
    print(metrics_df.to_string(index=False))
    print("=" * 80)

    # Save thresholds.json to project root
    thr_file = Path("thresholds.json")
    with open(thr_file, "w") as f:
        json.dump(best_thresholds, f, indent=4)
    print(f"\n[Saved] Optimal thresholds saved to {thr_file.resolve()}")

    # Step 4: Save Model & Tokenizer in format app expects
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Saved] Exporting model and tokenizer to: {out_dir.resolve()}...")

    # Save model with safetensors
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    # Verify saved directory contents
    expected_files = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt"]
    print("\n[Verification] Checking exported files in ./model/:")
    for f in expected_files:
        fp = out_dir / f
        exists = fp.exists()
        size = fp.stat().st_size if exists else 0
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {f:<24} ({size:,} bytes)")

    print(f"\n[Done] Fine-tuned model is ready! ToxiGuard can now load from '{args.output_dir}'.")
    return best_thresholds, metrics_df


# ── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DistilBERT Toxicity Classifier")
    parser.add_argument("--data_path", type=str, default="processed_data/processed_data/train.parquet", help="Path to training data")
    parser.add_argument("--val_path", type=str, default="processed_data/processed_data/val.parquet", help="Path to validation data")
    parser.add_argument("--sample_size", type=int, default=200000, help="Number of training samples (default: 200,000; CPU auto-clamps to 6,000)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--max_len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./model", help="Model export directory")

    args = parser.parse_args()
    train_model(args)
