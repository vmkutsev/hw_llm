"""Audit prediction files, compute paired query bootstrap intervals and prepare review inputs."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from metrics import load_predictions, predicted, query_stats, score, summarize
from run_suite import REQUIRED_RUNS


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def macro(m):
    denom = m.sum(-1) + m.sum(-2)
    f = np.divide(
        2 * np.diagonal(m, axis1=-2, axis2=-1),
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=denom > 0,
    )
    return f.mean(-1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--draws", type=int, default=2000)
    args = p.parse_args()
    out = args.results
    protocol = json.loads((Path(__file__).parent / "protocol.json").read_text())
    names = [
        name for name in protocol["runs"] if (out / name / "summary.json").exists()
    ]
    missing = set(REQUIRED_RUNS).difference(names)
    if missing:
        raise ValueError(f"Missing required experiments: {sorted(missing)}")
    manifest = json.loads((out / "dataset_manifest.json").read_text())
    samples = {}
    summaries = {}
    identities = {}
    last_orders = set()
    table = []
    class_rows = []
    for name in names:
        s = json.loads((out / name / "summary.json").read_text())
        summaries[name] = s
        assert (
            s["config"] == protocol["runs"][name] and s["splits"] == manifest["splits"]
        )
        assert (s["model"], s["revision"]) == (protocol["model"], protocol["revision"])
        for k, v in s["common_config"].items():
            assert protocol[k] == v
        for part in ("validation", "test"):
            rows = load_predictions(out / name / f"{part}.jsonl.gz")
            obs = [(r["id"], r["query_id"], r["product_id"], r["label"]) for r in rows]
            if part not in identities:
                identities[part] = obs
            assert obs == identities[part]
            measured = summarize(rows)
            recorded = s["metrics"][part]
            assert measured.keys() == recorded.keys()
            for key, value in measured.items():
                if isinstance(value, float):
                    assert math.isclose(value, recorded[key], rel_tol=0, abs_tol=1e-12)
                else:
                    assert value == recorded[key]
            expected = manifest["splits"][part]
            assert (measured["pairs"], measured["queries"]) == (
                expected["pairs"],
                expected["queries"],
            )
            assert {
                k: v["support"] for k, v in measured["per_class"].items()
            } == expected["labels"]
            if part == "test":
                samples[name] = rows
        tr = s["training"]
        if tr["order_sha256"]:
            last_orders.add(tr["order_sha256"])
        cfg = s["config"]
        table.append(
            {
                "run": name,
                "rank": cfg.get("r"),
                "alpha": cfg.get("alpha"),
                "dropout": cfg.get("dropout"),
                "modules": ",".join(cfg["modules"]) if "modules" in cfg else None,
                "trainable_parameters": s["trainable_parameters"],
                "trainable_pct": 100
                * s["trainable_parameters"]
                / s["total_parameters"],
                "steps": tr["optimizer_steps"],
                "train_seconds": tr["seconds"],
                "peak_allocated_gib": tr["peak_allocated_gib"],
                "val_ndcg10": s["metrics"]["validation"]["ndcg_at_10"],
                "val_macro_f1": s["metrics"]["validation"]["macro_f1"],
                "test_ndcg10": s["metrics"]["test"]["ndcg_at_10"],
                "test_macro_f1": s["metrics"]["test"]["macro_f1"],
            }
        )
        for label, stats in s["metrics"]["test"]["per_class"].items():
            class_rows.append({"run": name, "label": label, **stats})
    assert len(last_orders) == 1
    write_csv(out / "comparison.csv", table)
    write_csv(out / "class_metrics.csv", class_rows)
    qids = sorted({r["query_id"] for r in samples["base"]})
    n = len(qids)
    rng = np.random.default_rng(20261105)
    weights = rng.multinomial(n, np.full(n, 1 / n), size=args.draws)
    draws = {}
    ranks = {}
    by_ids = {}
    for name in names:
        by_ids[name] = {r["id"]: r for r in samples[name]}
        groups = defaultdict(list)
        for r in samples[name]:
            groups[r["query_id"]].append(r)
        values = []
        matrices = []
        ranks[name] = {}
        for q in qids:
            v, m = query_stats(groups[q])
            values.append(np.nan if v is None else v)
            matrices.append(m)
            sorted_rows = sorted(
                groups[q], key=lambda r: (-score(r["probs"]), r["product_id"], r["id"])
            )
            ranks[name].update({r["id"]: i for i, r in enumerate(sorted_rows, 1)})
        v = np.asarray(values)
        good = np.isfinite(v)
        m = np.asarray(matrices).reshape(n, 16)
        draws[name] = {
            "ndcg_at_10": (weights[:, good] @ v[good]) / weights[:, good].sum(1),
            "macro_f1": macro((weights @ m).reshape(-1, 4, 4)),
        }
        assert np.isclose(
            macro(m.sum(0).reshape(4, 4)),
            summaries[name]["metrics"]["test"]["macro_f1"],
        )
    pairs = [("full", "base"), ("lora_ref", "base"), ("lora_ref", "full")]
    pairs += [
        (name, "lora_ref") for name in names if name not in ("base", "full", "lora_ref")
    ]
    pairs.extend(
        [("lora_r32_alpha64", "lora_r32"), ("lora_r32_alpha64", "lora_alpha64")]
    )
    best = max(
        (name for name in names if name.startswith("lora_")),
        key=lambda name: (
            summaries[name]["metrics"]["validation"]["ndcg_at_10"],
            summaries[name]["metrics"]["validation"]["macro_f1"],
        ),
    )
    for reference in ("base", "full"):
        if (best, reference) not in pairs:
            pairs.append((best, reference))
    intervals = []
    for a, b in pairs:
        for metric in ("ndcg_at_10", "macro_f1"):
            delta = (
                summaries[a]["metrics"]["test"][metric]
                - summaries[b]["metrics"]["test"][metric]
            )
            lo, hi = np.quantile(draws[a][metric] - draws[b][metric], [0.025, 0.975])
            intervals.append(
                {
                    "run": a,
                    "reference": b,
                    "metric": metric,
                    "difference": delta,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    write_csv(out / "paired_intervals.csv", intervals)
    manual = json.loads((out / "manual_inputs.json").read_text())
    assert [r["id"] for r in manual] == manifest["manual_ids"]
    checks = []
    for r in manual:
        record = {k: r[k] for k in ("id", "query_id", "query", "title", "label")}
        for name in ("base", "full", "lora_ref"):
            response = by_ids[name][r["id"]]
            assert (
                response["label"] == r["label"]
                and response["product_id"] == r["product_id"]
            )
            record[name] = predicted(response["probs"])
            record[name + "_rank"] = ranks[name][r["id"]]
        checks.append(record)
    write_csv(out / "manual_predictions.csv", checks)
    print(
        "Audited:",
        len(names),
        "configurations;",
        len(identities["test"]),
        "identical test pairs;",
        n,
        "queries",
    )
    print("Best LoRA by validation nDCG@10 (macro-F1 tie-break):", best)
    for row in table:
        print(
            row["run"],
            f"nDCG {row['test_ndcg10']:.5f}; macro-F1 {row['test_macro_f1']:.5f};",
            "seconds",
            row["train_seconds"],
            "GiB",
            row["peak_allocated_gib"],
        )


if __name__ == "__main__":
    main()
