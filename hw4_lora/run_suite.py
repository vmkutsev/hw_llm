"""Execute the frozen experiment matrix in fresh processes, sequentially on one GPU."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED_RUNS = (
    "base",
    "full",
    "lora_ref",
    "lora_r32",
    "lora_alpha64",
    "lora_drop15",
    "lora_all",
    "lora_r32_alpha64",
)


def save_review_inputs(data, out):
    """Include the frozen inputs needed for offline analysis of a new run."""
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data / "manifest.json").read_text())
    shutil.copyfile(data / "manifest.json", out / "dataset_manifest.json")
    selected = set(manifest["manual_ids"])
    found = {}
    for line in (data / "test.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["id"] in selected:
            found[row["id"]] = row
    train = [json.loads(s) for s in (data / "train.jsonl").read_text().splitlines()]
    sources = {
        "manual_inputs.json": [found[i] for i in manifest["manual_ids"]],
        "train_examples.json": [
            next(r for r in train if r["label"] == k) for k in ("E", "S", "C", "I")
        ],
    }
    for filename, rows in sources.items():
        (out / filename).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--cache", type=Path, default=Path(".cache/tokens"))
    p.add_argument("--checkpoint-dir", type=Path)
    args = p.parse_args()
    config = json.loads((ROOT / "protocol.json").read_text())
    names = args.runs or REQUIRED_RUNS
    if len(set(names)) != len(names) or not set(names) <= set(config["runs"]):
        raise ValueError("Unknown or duplicate run name")
    if any((args.out / name / "summary.json").exists() for name in names):
        raise FileExistsError("Choose an empty --out directory for a fresh run")
    save_review_inputs(args.data, args.out)
    summaries = []
    for name in names:
        command = [
            sys.executable,
            "-u",
            str(ROOT / "experiment.py"),
            "--run",
            name,
            "--data",
            str(args.data),
            "--out",
            str(args.out),
            "--cache",
            str(args.cache),
        ]
        if args.checkpoint_dir:
            command.extend(["--checkpoint-dir", str(args.checkpoint_dir)])
        subprocess.run(command, check=True)
        summaries.append(json.loads((args.out / name / "summary.json").read_text()))
    train_orders = {
        s["training"]["order_sha256"]
        for s in summaries
        if s["config"]["mode"] != "base"
    }
    assert len(train_orders) <= 1
    assert len({s["device"] for s in summaries}) == 1
    assert all(s["splits"] == summaries[0]["splits"] for s in summaries)
    print("DONE", ",".join(names), flush=True)


if __name__ == "__main__":
    main()
