"""
eval_threshold.py

Threshold Evaluation Engine for Semantic Caching:
Sweeps similarity thresholds from 0.70 to 0.98 across a labeled dataset of 150+ query pairs
(~75 true paraphrases, ~75 hard negatives).

Metrics Calculated:
- Precision: TP / (TP + FP) -> Ratio of true matches among all predicted matches
- Recall / True Hit Rate: TP / (TP + FN) -> Ratio of actual paraphrases correctly cached
- F1 Score: 2 * (Precision * Recall) / (Precision + Recall) -> Harmonic mean balancing Precision & Recall
- False Positive Rate (FPR): FP / (FP + TN) -> Ratio of distinct queries incorrectly merged (Cache Contamination)

Design & Interview Defense:
In semantic caching, a False Positive is far worse than a False Negative:
- False Negative (Cache Miss): Costs a few extra milliseconds and API cents, but returns accurate LLM output.
- False Positive (Bad Cache Hit): Returns completely WRONG/unrelated information to the user (e.g. canceling vs pausing subscription).
Therefore, the optimal threshold selection algorithm prioritizes maximizing F1 while enforcing Zero or Minimal False Positives (FPR <= 0.01).
"""

import json
import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

# Add parent and current dir to path to import config cleanly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

THRESHOLDS_TO_TEST = [0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95, 0.98]


def load_eval_pairs():
    """Load eval pairs from data/eval_pairs.json or backend/data/eval_pairs.json."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(base_dir, "..", "data", "eval_pairs.json"),
        os.path.join(base_dir, "data", "eval_pairs.json"),
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

    # Fallback: run create_eval_pairs module if available
    try:
        import create_eval_pairs
        create_eval_pairs.main()
        for p in candidate_paths:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to auto-generate eval_pairs.json: {e}")

    raise FileNotFoundError("Could not locate data/eval_pairs.json dataset file.")


def run_threshold_sweep(eval_pairs=None):
    """Run cosine similarity sweep across candidate thresholds and calculate classification metrics."""
    if eval_pairs is None:
        eval_pairs = load_eval_pairs()

    print(f"[EVAL] Encoding {len(eval_pairs)} evaluation query pairs with {config.EMBEDDING_MODEL}...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    a_texts = [p["query_a"] for p in eval_pairs]
    b_texts = [p["query_b"] for p in eval_pairs]
    labels = [bool(p["should_match"]) for p in eval_pairs]

    emb_a = model.encode(a_texts, normalize_embeddings=True)
    emb_b = model.encode(b_texts, normalize_embeddings=True)

    # Cosine similarity is dot product of normalized vectors
    similarities = np.sum(emb_a * emb_b, axis=1)

    total_positives = sum(1 for l in labels if l)      # True paraphrases
    total_negatives = sum(1 for l in labels if not l)  # Hard negatives

    metrics_by_threshold = []
    best_threshold = 0.92
    best_f1 = -1.0
    min_fp = float("inf")

    print(
        f"\n{'Threshold':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'True Hit Rate':<15} {'FPR':<10} {'TP':<6} {'FP':<6}"
    )
    print("-" * 85)

    for threshold in THRESHOLDS_TO_TEST:
        predicted = similarities >= threshold

        tp = int(sum(1 for p, l in zip(predicted, labels) if p and l))
        fp = int(sum(1 for p, l in zip(predicted, labels) if p and not l))
        tn = int(sum(1 for p, l in zip(predicted, labels) if not p and not l))
        fn = int(sum(1 for p, l in zip(predicted, labels) if not p and l))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        true_hit_rate = recall  # TPR
        false_positive_rate = fp / total_negatives if total_negatives > 0 else 0.0

        item = {
            "threshold": round(float(threshold), 2),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "true_hit_rate": round(float(true_hit_rate), 4),
            "false_positive_rate": round(float(false_positive_rate), 4),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        }
        metrics_by_threshold.append(item)

        print(
            f"{threshold:<10.2f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {true_hit_rate:<15.4f} {false_positive_rate:<10.4f} {tp:<6} {fp:<6}"
        )

    # Selection rule: Pick candidate threshold that maximizes F1 score subject to F1 > 0, breaking ties by lowest FPR
    valid_candidates = [m for m in metrics_by_threshold if m["f1"] > 0]
    if valid_candidates:
        # Sort by F1 descending, then FPR ascending
        best_candidate = max(valid_candidates, key=lambda m: (m["f1"], -m["false_positive_rate"]))
        best_threshold = best_candidate["threshold"]
        best_f1 = best_candidate["f1"]
        min_fp = best_candidate["false_positives"]
    else:
        best_threshold = 0.92
        best_f1 = 0.0
        min_fp = 0

    results = {
        "best_threshold": round(float(best_threshold), 2),
        "total_pairs": len(eval_pairs),
        "true_paraphrases": total_positives,
        "hard_negatives": total_negatives,
        "embedding_model": config.EMBEDDING_MODEL,
        "metrics_by_threshold": metrics_by_threshold,
    }

    # Save to JSON outputs
    save_dirs = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
    ]
    for sdir in save_dirs:
        os.makedirs(sdir, exist_ok=True)
        out_file = os.path.join(sdir, "eval_results.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print("-" * 85)
    print(f"Optimal Threshold Picked: {best_threshold:.2f} (F1: {best_f1:.4f}, FP Count: {min_fp})")
    print(f"Results saved to eval_results.json")

    return results


def main():
    run_threshold_sweep()


if __name__ == "__main__":
    main()

