"""Rebuild identical US ESCI candidate lists from the original Amazon Parquet files."""

import argparse
import hashlib
import json
import os
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import duckdb

REVISION = "7916cdf6ab75a462e77f20ab40428a10923998d5"
FILES = {
    "examples": (
        51286808,
        "4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a",
    ),
    "products": (
        1108857465,
        "25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265",
    ),
}
GROUPS = {"train": 600, "validation": 128, "test": 256}
SEED = 42


def sha(path):
    with Path(path).open("rb") as f:
        return (
            hashlib.file_digest(f, "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else stream_hash(f)
        )


def stream_hash(f):
    h = hashlib.sha256()
    for block in iter(lambda: f.read(8 << 20), b""):
        h.update(block)
    return h.hexdigest()


def hash_key(tag, key):
    return hashlib.sha256(f"{SEED}:{tag}:{key}".encode()).hexdigest()


def norm(s):
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def original(raw, name):
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / (name + ".parquet")
    part = path.with_suffix(".parquet.partial")
    size, digest = FILES[name]
    for p in (path, part):
        if p.is_file() and p.stat().st_size == size and sha(p) == digest:
            if p == part:
                os.replace(part, path)
            return path
    url = (
        f"https://media.githubusercontent.com/media/amazon-science/esci-data/{REVISION}/"
        f"shopping_queries_dataset/shopping_queries_dataset_{name}.parquet"
    )
    with urllib.request.urlopen(url, timeout=120) as src, part.open("wb") as out:
        for block in iter(lambda: src.read(4 << 20), b""):
            out.write(block)
    if part.stat().st_size != size or sha(part) != digest:
        raise ValueError(f"Unexpected original file: {name}")
    os.replace(part, path)
    return path


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path(".cache/source"))
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()
    source = {k: original(args.raw, k) for k in FILES}
    args.out.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect()
    db.execute("SET threads=4")
    db.read_parquet(str(source["examples"])).create_view("original_pairs")
    db.read_parquet(str(source["products"])).create_view("products")
    db.execute("""CREATE TEMP TABLE pairs AS SELECT example_id, query_id, query,
              product_id, product_locale, esci_label, split FROM original_pairs
              WHERE product_locale='us' AND small_version=1""")
    groups = db.execute("""SELECT split, query_id, query, count(*), count(DISTINCT product_id)
                           FROM pairs GROUP BY ALL ORDER BY query_id, split, query""").fetchall()
    families = defaultdict(list)
    for part, qid, query, count, unique in groups:
        families[norm(query)].append((part, qid, count, unique))
    eligible = defaultdict(list)
    excluded = Counter()
    for key, group in families.items():
        if not key or len(group) != 1:
            excluded["blank_or_repeated_normalized_query"] += 1
            continue
        part, qid, count, unique = group[0]
        if count != unique:
            excluded["repeated_product_in_query"] += 1
            continue
        eligible[part].append((key, qid, count))
    for part, values in eligible.items():
        values.sort(key=lambda x: (hash_key("query-" + part, x[0]), x[1]))
    selected = {
        "train": eligible["train"][: GROUPS["train"]],
        "validation": eligible["train"][
            GROUPS["train"] : GROUPS["train"] + GROUPS["validation"]
        ],
        "test": eligible["test"][: GROUPS["test"]],
    }
    assert all(len(selected[k]) == GROUPS[k] for k in GROUPS)
    assert len({q for rows in selected.values() for q, _, _ in rows}) == sum(
        GROUPS.values()
    )
    db.execute("CREATE TEMP TABLE sel(qid BIGINT PRIMARY KEY, part VARCHAR)")
    db.executemany(
        "INSERT INTO sel VALUES (?,?)",
        [(qid, part) for part, rows in selected.items() for _, qid, _ in rows],
    )
    joined = db.execute("""SELECT s.part, p.example_id, p.query_id, p.query, p.product_id,
                           t.product_title, p.esci_label, t.product_id IS NOT NULL FROM pairs p
                           JOIN sel s ON s.qid=p.query_id LEFT JOIN products t
                           ON p.product_id=t.product_id AND p.product_locale=t.product_locale
                           ORDER BY s.part, p.example_id""").fetchall()
    subsets = defaultdict(list)
    for split, eid, qid, query, product, title, label, matched in joined:
        if not matched or label not in ("E", "S", "C", "I"):
            raise ValueError(f"Invalid join or label at example {eid}")
        subsets[split].append(
            dict(
                id=eid,
                query_id=qid,
                query=query,
                product_id=product,
                title=" ".join((title or "").split()),
                label=label,
            )
        )
    manifest = {
        "dataset": "amazon-science/esci-data",
        "revision": REVISION,
        "locale": "us",
        "selection": "small_version=1; whole query groups; SHA256 sampling",
        "seed": SEED,
        "source_files": {k: {"bytes": v[0], "sha256": v[1]} for k, v in FILES.items()},
        "eligible_query_groups": {k: len(v) for k, v in eligible.items()},
        "exclusions_before_sampling": dict(excluded),
        "splits": {},
    }
    for part, rows in subsets.items():
        assert (
            len(rows)
            == len({r["id"] for r in rows})
            == sum(x[2] for x in selected[part])
        )
        path = args.out / (part + ".jsonl")
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
        manifest["splits"][part] = {
            "pairs": len(rows),
            "queries": len(selected[part]),
            "labels": dict(sorted(Counter(r["label"] for r in rows).items())),
            "blank_titles": sum(not r["title"] for r in rows),
            "sha256": sha(path),
        }
    # Eight examples fixed before any predictions: two per reference class, different queries.
    manual, seen = [], set()
    ordered = sorted(subsets["test"], key=lambda r: hash_key("manual", r["id"]))
    for label in ("E", "S", "C", "I"):
        count = 0
        for row in ordered:
            if row["label"] == label and row["query_id"] not in seen:
                manual.append(row["id"])
                seen.add(row["query_id"])
                count += 1
                if count == 2:
                    break
        assert count == 2
    manifest["manual_ids"] = manual
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
