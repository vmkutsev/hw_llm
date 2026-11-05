"""Pair classification and query-level reranking with Amazon's linear ESCI gains."""

import gzip
import json
import math
from collections import defaultdict

LABELS = ("E", "S", "C", "I")
GAINS = (1.0, 0.1, 0.01, 0.0)


def predicted(probs):
    return LABELS[max(range(4), key=lambda i: probs[i])]


def score(probs):
    return sum(p * g for p, g in zip(probs, GAINS))


def query_stats(rows):
    # If model scores tie, order by a document ID independent of the relevance judgment.
    by_score = sorted(
        rows, key=lambda r: (-score(r["probs"]), r["product_id"], r["id"])
    )
    gains = [GAINS[LABELS.index(r["label"])] for r in by_score]

    def dcg(values):
        return sum(g / math.log2(i + 2) for i, g in enumerate(values[:10]))

    ideal = dcg(sorted(gains, reverse=True))
    matrix = [[0] * 4 for _ in LABELS]
    for r in rows:
        matrix[LABELS.index(r["label"])][LABELS.index(predicted(r["probs"]))] += 1
    return (dcg(gains) / ideal if ideal > 0 else None), matrix


def summarize(rows):
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        assert row["id"] not in seen
        seen.add(row["id"])
        p = row["probs"]
        assert len(p) == 4 and all(math.isfinite(v) and 0 <= v <= 1 for v in p)
        assert abs(sum(p) - 1) < 1e-5
        groups[row["query_id"]].append(row)
    vals, matrix = [], [[0] * 4 for _ in LABELS]
    for group in groups.values():
        v, m = query_stats(group)
        if v is not None:
            vals.append(v)
        for i in range(4):
            for j in range(4):
                matrix[i][j] += m[i][j]
    per_class = {}
    for i, k in enumerate(LABELS):
        tp, actual, called = matrix[i][i], sum(matrix[i]), sum(row[i] for row in matrix)
        per_class[k] = {
            "support": actual,
            "predicted": called,
            "tp": tp,
            "precision": tp / called if called else 0.0,
            "recall": tp / actual if actual else 0.0,
            "f1": 2 * tp / (actual + called) if actual + called else 0.0,
        }
    if not vals:
        raise ValueError("No query with positive ideal gain")
    return {
        "ndcg_at_10": sum(vals) / len(vals),
        "macro_f1": sum(x["f1"] for x in per_class.values()) / 4,
        "pairs": len(seen),
        "queries": len(groups),
        "queries_with_positive_idcg": len(vals),
        "per_class": per_class,
        "confusion_true_by_pred": matrix,
    }


def save_predictions(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_predictions(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
