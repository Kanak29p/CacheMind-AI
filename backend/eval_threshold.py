"""
eval_threshold.py

The single most interview-worthy artifact in this whole project: a small,
hand-labeled eval set of query pairs, and a script that measures how the
semantic cache behaves at different similarity thresholds.

Run it with:  python eval_threshold.py

This produces a table like:

  threshold | true_hit_rate | false_positive_rate
  0.85      | 0.91          | 0.30
  0.90      | 0.78          | 0.05
  0.95      | 0.52          | 0.00

...which is exactly the kind of number to quote in an interview:
"At threshold 0.90 I get a 78% true-match rate with only a 5% false-positive
rate on my eval set."

Extend EVAL_PAIRS with more examples from your own domain before your demo.
"""
from sentence_transformers import SentenceTransformer
import numpy as np
import config

# Each pair is (query_a, query_b, should_match: bool)
# "should_match" = True means these are semantically the same question and
# SHOULD be served from cache. False means they look similar on the surface
# but are meaningfully different questions -- a good cache should NOT merge
# these, even though naive similarity might.
EVAL_PAIRS = [
    ("How do I reset my password?", "I forgot my password, how do I change it?", True),
    ("What's your refund policy?", "Can I get my money back for this order?", True),
    ("How do I cancel my subscription?", "How do I pause my subscription?", False),
    ("What are your business hours?", "When are you open?", True),
    ("How do I upgrade my plan?", "How do I downgrade my plan?", False),
    ("Is there a free trial?", "Do you offer a trial period?", True),
    ("How do I delete my account?", "How do I deactivate my account temporarily?", False),
    ("What payment methods do you accept?", "Can I pay with PayPal?", False),
    ("How do I contact support?", "What's the best way to reach customer service?", True),
    ("Can I change my shipping address?", "Can I change my billing address?", False),
]

THRESHOLDS_TO_TEST = [0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.97]


def main():
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    a_texts = [p[0] for p in EVAL_PAIRS]
    b_texts = [p[1] for p in EVAL_PAIRS]
    labels = [p[2] for p in EVAL_PAIRS]

    emb_a = model.encode(a_texts, normalize_embeddings=True)
    emb_b = model.encode(b_texts, normalize_embeddings=True)
    similarities = np.sum(emb_a * emb_b, axis=1)  # cosine sim (vectors normalized)

    print(f"{'threshold':<10} {'true_hit_rate':<15} {'false_positive_rate':<20}")
    print("-" * 45)

    for threshold in THRESHOLDS_TO_TEST:
        predicted_match = similarities >= threshold

        true_positives = sum(1 for p, l in zip(predicted_match, labels) if p and l)
        should_match_count = sum(1 for l in labels if l)
        should_not_match_count = sum(1 for l in labels if not l)
        false_positives = sum(1 for p, l in zip(predicted_match, labels) if p and not l)

        true_hit_rate = true_positives / should_match_count if should_match_count else 0
        false_positive_rate = false_positives / should_not_match_count if should_not_match_count else 0

        print(f"{threshold:<10} {true_hit_rate:<15.2f} {false_positive_rate:<20.2f}")

    print("\nPer-pair similarity scores:")
    for (a, b, label), sim in zip(EVAL_PAIRS, similarities):
        print(f"  [{'MATCH' if label else 'NO-MATCH':>8}] {sim:.3f}  '{a}' <-> '{b}'")


if __name__ == "__main__":
    main()
