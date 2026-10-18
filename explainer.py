"""
explainer.py — Token-level explanation using attention weights.
Provides word-importance highlighting for the Streamlit app.

Uses DistilBERT's attention heads as a lightweight proxy for SHAP
(full SHAP on a transformer is too slow for interactive use).
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


LABELS = ["toxic", "obscene", "threat", "insult", "identity_hate", "sexual_explicit"]


def get_attention_explanation(
    model,
    tokenizer,
    text: str,
    device: str = "cpu",
    top_k: int = 15,
) -> Dict:
    """
    Extract token-level importance from the last attention layer.

    Returns:
        {
            "tokens": [...],
            "scores": [...],       # normalised importance per token
            "html": "...",         # styled HTML for display
            "top_tokens": [...]    # most important tokens
        }
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=240,
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Average attention from the last layer, all heads → [seq_len, seq_len]
    last_attn = outputs.attentions[-1]  # (1, heads, seq, seq)
    avg_attn = last_attn.mean(dim=1)[0]  # (seq, seq)

    # CLS token attention to all other tokens
    cls_attn = avg_attn[0].cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Merge subword tokens and aggregate attention
    merged_tokens, merged_scores = _merge_subwords(tokens, cls_attn)

    # Remove special tokens
    clean_tokens = []
    clean_scores = []
    for t, s in zip(merged_tokens, merged_scores):
        if t in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>"):
            continue
        clean_tokens.append(t)
        clean_scores.append(s)

    # Normalise
    scores = np.array(clean_scores)
    if scores.max() > 0:
        scores = scores / scores.max()

    # Build highlighted HTML
    html = _build_highlight_html(clean_tokens, scores)

    # Top-K most important
    top_idx = np.argsort(scores)[::-1][:top_k]
    top_tokens = [(clean_tokens[i], round(float(scores[i]), 4)) for i in top_idx]

    return {
        "tokens": clean_tokens,
        "scores": scores.tolist(),
        "html": html,
        "top_tokens": top_tokens,
    }


def _merge_subwords(tokens: List[str], scores: np.ndarray) -> Tuple[List[str], List[float]]:
    """Merge WordPiece subwords (##prefix) back into whole words."""
    merged_tokens = []
    merged_scores = []

    for token, score in zip(tokens, scores):
        if token.startswith("##") and merged_tokens:
            merged_tokens[-1] += token[2:]
            merged_scores[-1] = max(merged_scores[-1], score)
        else:
            merged_tokens.append(token)
            merged_scores.append(float(score))

    return merged_tokens, merged_scores


def _build_highlight_html(tokens: List[str], scores: np.ndarray) -> str:
    """Create HTML with background-colour intensity proportional to importance."""
    parts = []
    for token, score in zip(tokens, scores):
        # Gradient from transparent to red
        intensity = max(0.0, min(1.0, float(score)))
        if intensity > 0.6:
            bg = f"rgba(239, 68, 68, {intensity * 0.7})"
            color = "white"
        elif intensity > 0.3:
            bg = f"rgba(251, 146, 60, {intensity * 0.6})"
            color = "#1a1a2e"
        else:
            bg = f"rgba(148, 163, 184, {intensity * 0.3})"
            color = "#e2e8f0"

        parts.append(
            f'<span style="background:{bg}; color:{color}; '
            f'padding:2px 5px; margin:1px 2px; border-radius:4px; '
            f'display:inline-block; font-size:0.95rem; '
            f'font-family: \'Inter\', sans-serif;">{token}</span>'
        )

    return " ".join(parts)


def get_simple_word_importance(text: str, probabilities: Dict[str, float]) -> str:
    """
    Fallback: simple keyword-based highlighting when attention isn't available.
    Highlights known toxic keywords with intensity based on overall toxicity score.
    """
    toxic_keywords = {
        "kill", "die", "hate", "stupid", "idiot", "fool", "ugly",
        "dumb", "shut up", "loser", "moron", "trash", "worthless",
        "disgusting", "pathetic", "terrible", "horrible", "awful",
        "damn", "hell", "crap", "suck", "fat", "freak",
    }

    max_prob = max(probabilities.values()) if probabilities else 0

    words = text.split()
    parts = []
    for word in words:
        clean_word = word.lower().strip(".,!?;:'\"()-")
        if clean_word in toxic_keywords:
            intensity = max(0.4, min(1.0, max_prob))
            parts.append(
                f'<span style="background:rgba(239, 68, 68, {intensity * 0.7}); '
                f'color:white; padding:2px 5px; margin:1px 2px; '
                f'border-radius:4px; display:inline-block; '
                f'font-weight:600; font-size:0.95rem;">{word}</span>'
            )
        else:
            parts.append(
                f'<span style="padding:2px 3px; display:inline-block; '
                f'color:#e2e8f0; font-size:0.95rem;">{word}</span>'
            )

    return " ".join(parts)
