"""Small hand-calculated checks for the two measures used in the report."""

import math
from pathlib import Path
from tempfile import TemporaryDirectory
from metrics import LABELS, load_predictions, save_predictions, summarize


def row(i, q, actual, pred, product=None):
    probs = [float(k == pred) for k in LABELS]
    return {
        "id": i,
        "query_id": q,
        "product_id": product or str(i),
        "label": actual,
        "probs": probs,
    }


def main():
    rows = [row(i, 1, k, k) for i, k in enumerate(LABELS)]
    assert summarize(rows)["macro_f1"] == summarize(rows)["ndcg_at_10"] == 1
    with TemporaryDirectory() as folder:
        path = Path(folder) / "predictions.jsonl.gz"
        save_predictions(path, rows)
        assert load_predictions(path) == rows
    # One row of each true class, all predicted E: F1(E)=2/5, other F1=0.
    assert math.isclose(
        summarize([row(i, 1, k, "E") for i, k in enumerate(LABELS)])["macro_f1"], 0.1
    )
    # One two-item list reversed (E at rank 2), and one perfectly ranked singleton.
    xs = [row(1, 10, "E", "I"), row(2, 10, "I", "E"), row(3, 20, "S", "S")]
    assert math.isclose(summarize(xs)["ndcg_at_10"], (1 / math.log2(3) + 1) / 2)
    # Linear, not exponential gains: reversed E/S.
    es = [row(1, 1, "E", "S"), row(2, 1, "S", "E")]
    assert math.isclose(
        summarize(es)["ndcg_at_10"], (0.1 + 1 / math.log2(3)) / (1 + 0.1 / math.log2(3))
    )
    tied = [row(1, 1, "E", "E", "z"), row(2, 1, "I", "E", "a")]
    assert math.isclose(summarize(tied[::-1])["ndcg_at_10"], 1 / math.log2(3))
    zero = xs + [row(4, 30, "I", "I")]
    assert summarize(zero)["queries_with_positive_idcg"] == 2
    print("Metric checks passed.")


if __name__ == "__main__":
    main()
