"""Run the supplied data preparation verbatim and describe its output."""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    source = root / "starter/train.py"
    spec = importlib.util.spec_from_file_location("supplied_benchmark", source)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    benchmark.DATA = Path(sys.argv[1]).resolve()
    benchmark.DATA.mkdir(parents=True, exist_ok=True)
    benchmark.prepare_data()
    from datasets import load_from_disk

    dataset = load_from_disk(str(benchmark.DATA / "dataset"))
    article_sets = {name: set(split["article_id"]) for name, split in dataset.items()}
    assert not article_sets["train"] & article_sets["validation"]
    assert len(dataset["validation"]) >= 512
    statistics = {}
    for name, split in dataset.items():
        lengths = [len(ids) for ids in split["input_ids"]]
        assert min(lengths) >= 33 and max(lengths) <= 512
        statistics[name] = {
            "paragraphs": len(lengths), "articles_with_paragraphs": len(article_sets[name]),
            "minimum_length": min(lengths), "maximum_length": max(lengths),
            "mean_length": sum(lengths) / len(lengths),
            "target_tokens": sum(length - 1 for length in lengths),
        }
    manifest = {
        "starter_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "wikipedia_revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
        "tokenizer_revision": "a9307e696cd3c5b7f953ff4cb19d76a4d81821d5",
        "source_articles": 50500, "validation_source_articles": 500,
        "evaluation_paragraphs": 512, "article_overlap": 0, "splits": statistics,
        "files": {},
    }
    for path in sorted(benchmark.DATA.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            manifest["files"][str(path.relative_to(benchmark.DATA))] = {
                "bytes": path.stat().st_size,
                "sha256": checksum,
            }
    (benchmark.DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
    # The streaming reader can abort during interpreter finalization on this
    # container's native-library stack. All dataset files have been closed,
    # reloaded, checked and hashed above. Bypass only successful teardown;
    # exceptions in main() still fail the preparation process normally.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
