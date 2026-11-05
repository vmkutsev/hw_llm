"""Train or evaluate one fresh Qwen on the fixed ESCI classification/reranking task."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from metrics import LABELS, save_predictions, summarize

ROOT = Path(__file__).resolve().parent
SYSTEM = (
    "Judge a shopping search result using only the query and product title. "
    "E = Exact: the product matches the requested item and its stated requirements. "
    "S = Substitute: an alternative that serves the same purpose but differs in a requested detail. "
    "C = Complement: goes with the requested item but cannot replace it. "
    "I = Irrelevant: neither the requested item, a usable alternative, nor a complement. "
    "Answer with exactly one letter: E, S, C, or I."
)


def dump(path, obj):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def messages(query, title):
    fields = {
        "search_query": query.replace("<|", "< |"),
        "product_title": title.replace("<|", "< |"),
    }
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(fields, ensure_ascii=False)},
    ]


def encode(tok, row, limit):
    def ids(title):
        return tok.apply_chat_template(
            messages(row["query"], title), tokenize=True, add_generation_prompt=True
        )

    title = row["title"]
    tokens = ids(title)
    if len(tokens) <= limit:
        return tokens, False
    lo, hi = 0, len(title)
    if len(ids("")) > limit:
        raise ValueError("Query does not fit even without its product title")
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(ids(title[:mid])) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return ids(title[:lo]), True


def encoded_data(data, cache, tok, cfg):
    manifest = json.loads((data / "manifest.json").read_text())
    tag = hashlib.sha256(
        json.dumps(
            [manifest, cfg["revision"], cfg["max_tokens"], SYSTEM, "v1"], sort_keys=True
        ).encode()
    ).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (tag + ".json")
    for split, meta in manifest["splits"].items():
        if (
            hashlib.sha256((data / (split + ".jsonl")).read_bytes()).hexdigest()
            != meta["sha256"]
        ):
            raise ValueError("Dataset checksum mismatch: " + split)
    if path.exists():
        return json.loads(path.read_text()), manifest
    output = {}
    for split in ("train", "validation", "test"):
        output[split] = []
        for line in (data / (split + ".jsonl")).read_text().splitlines():
            row = json.loads(line)
            ids, cut = encode(tok, row, cfg["max_tokens"])
            if len(output[split]) == 0 and not cut:
                for label in LABELS:
                    full = tok.apply_chat_template(
                        messages(row["query"], row["title"])
                        + [{"role": "assistant", "content": label}],
                        tokenize=True,
                    )
                    expected = tok.encode(label, add_special_tokens=False)
                    assert len(expected) == 1 and full[: len(ids) + 1] == ids + expected
            output[split].append(
                {
                    **{k: row[k] for k in ("id", "query_id", "product_id", "label")},
                    "tokens": ids,
                    "title_truncated": cut,
                }
            )
        print("TOKENIZED", split, len(output[split]), flush=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(output, separators=(",", ":")))
    tmp.replace(path)
    return output, manifest


def batch_inputs(rows, pad):
    # Left padding makes sequence index -1 the first decision position for every example.
    width = 8 * math.ceil(max(len(r["tokens"]) for r in rows) / 8)
    ids, masks = [], []
    for row in rows:
        seq = row["tokens"]
        gap = width - len(seq)
        ids.append([pad] * gap + seq)
        masks.append([0] * gap + [1] * len(seq))
    ids = torch.tensor(ids, dtype=torch.long, device="cuda")
    mask = torch.tensor(masks, dtype=torch.long, device="cuda")
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "position_ids": (mask.cumsum(-1) - 1).clamp(min=0),
    }


def last_logits(model, rows, pad):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = model(**batch_inputs(rows, pad), use_cache=False, logits_to_keep=1)
    return result.logits[:, -1, :].float()


@torch.inference_mode()
def evaluate(model, rows, pad, label_ids, size):
    model.eval()
    predictions = []
    for start in range(0, len(rows), size):
        chunk = rows[start : start + size]
        probs = (
            last_logits(model, chunk, pad)
            .index_select(-1, label_ids)
            .softmax(-1)
            .cpu()
            .tolist()
        )
        for r, p in zip(chunk, probs):
            predictions.append(
                {
                    **{k: r[k] for k in ("id", "query_id", "product_id", "label")},
                    "probs": p,
                }
            )
    return predictions


def train(model, rows, pad, label_ids, cfg):
    parameters = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        parameters,
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True,
    )
    bs = cfg["batch_size"]
    total = cfg["epochs"] * math.ceil(len(rows) / bs)
    warm = max(1, math.ceil(total * cfg["warmup_fraction"]))
    order_hash = hashlib.sha256()
    history = []
    step = 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model.train()
    for epoch in range(cfg["epochs"]):
        indices = list(range(len(rows)))
        random.Random(cfg["seed"] + epoch).shuffle(indices)
        for ix in range(0, len(indices), bs):
            batch = [rows[i] for i in indices[ix : ix + bs]]
            order_hash.update((",".join(str(r["id"]) for r in batch) + ";").encode())
            step += 1
            scale = step / warm if step <= warm else (total - step + 1) / (total - warm)
            lr = cfg["learning_rate"] * scale
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            logits = last_logits(model, batch, pad)
            targets = label_ids[
                torch.tensor([LABELS.index(r["label"]) for r in batch], device="cuda")
            ]
            # Full-vocabulary next-token CE; the prompt is context, never a target.
            loss = torch.nn.functional.cross_entropy(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters, cfg["max_grad_norm"], error_if_nonfinite=True
            )
            opt.step()
            if step == 1 or step % 50 == 0 or step == total:
                rec = {
                    "step": step,
                    "epoch": epoch + 1,
                    "batch_ce": float(loss.detach()),
                    "lr": lr,
                }
                history.append(rec)
                print("TRAIN", json.dumps(rec), flush=True)
            del logits, loss
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return (
        {
            "optimizer_steps": step,
            "seen_pairs": len(rows) * cfg["epochs"],
            "seconds": seconds,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "order_sha256": order_hash.hexdigest(),
        },
        history,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--cache", type=Path, default=Path(".cache/tokens"))
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--checkpoint-dir", type=Path)
    args = p.parse_args()
    cfg = json.loads((ROOT / "protocol.json").read_text())
    setting = cfg["runs"][args.run]
    mode = setting["mode"]
    seed = cfg["seed"]
    dest = args.out / args.run
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "summary.json").exists():
        raise FileExistsError(f"Completed run already exists: {dest}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required")
    set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    snapshot = snapshot_download(
        cfg["model"],
        revision=cfg["revision"],
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
    )
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    letters = [tok.encode(k, add_special_tokens=False) for k in LABELS]
    if any(len(v) != 1 for v in letters) or len({v[0] for v in letters}) != 4:
        raise ValueError("Expected four distinct one-token labels")
    ids = torch.tensor([v[0] for v in letters], device="cuda", dtype=torch.long)
    splits, manifest = encoded_data(args.data, args.cache, tok, cfg)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, dtype=torch.float32, local_files_only=True, attn_implementation="sdpa"
    ).to("cuda")
    model.config.use_cache = False
    original_params = sum(p.numel() for p in model.parameters())
    if mode == "base":
        model.requires_grad_(False)
    elif mode == "lora":
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=setting["r"],
                lora_alpha=setting["alpha"],
                lora_dropout=setting["dropout"],
                target_modules=setting["modules"],
                bias="none",
                use_rslora=False,
            ),
        )
        train_names = [n for n, p in model.named_parameters() if p.requires_grad]
        assert train_names and all("lora_" in n for n in train_names)
        for suffix in setting["modules"]:
            assert any(("." + suffix + ".lora_") in n for n in train_names)
    elif mode == "full":
        assert all(p.requires_grad for p in model.parameters())
    else:
        raise ValueError(mode)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    # Reset stochastic state after adapter initialization; data permutations use a separate RNG.
    set_seed(seed)
    print("RUN", args.run, "trainable", trainable, flush=True)
    measures, history = (
        {
            "optimizer_steps": 0,
            "seen_pairs": 0,
            "seconds": None,
            "peak_allocated_gib": None,
            "peak_reserved_gib": None,
            "order_sha256": None,
        },
        [],
    )
    if mode != "base":
        measures, history = train(model, splits["train"], tok.pad_token_id, ids, cfg)
    scores = {}
    for part in ("validation", "test"):
        preds = evaluate(
            model, splits[part], tok.pad_token_id, ids, cfg["eval_batch_size"]
        )
        scores[part] = summarize(preds)
        save_predictions(dest / f"{part}.jsonl.gz", preds)
        print(
            "SCORE",
            args.run,
            part,
            scores[part]["ndcg_at_10"],
            scores[part]["macro_f1"],
            flush=True,
        )
    if args.checkpoint_dir and mode != "base":
        target = args.checkpoint_dir / args.run
        model.save_pretrained(target, safe_serialization=True)
        tok.save_pretrained(target)
    dump(dest / "loss.json", history)
    summary = {
        "name": args.run,
        "config": setting,
        "model": cfg["model"],
        "revision": cfg["revision"],
        "common_config": {
            k: v for k, v in cfg.items() if k not in ("runs", "revision", "model")
        },
        "device": torch.cuda.get_device_name(0),
        "original_parameters": original_params,
        "total_parameters": total_params,
        "trainable_parameters": trainable,
        "training": measures,
        "splits": manifest["splits"],
        "title_truncations": {
            k: sum(r["title_truncated"] for r in v) for k, v in splits.items()
        },
        "label_token_ids": [v[0] for v in letters],
        "metrics": scores,
        "versions": {
            k: importlib.metadata.version(k)
            for k in (
                "torch",
                "transformers",
                "peft",
                "accelerate",
                "huggingface-hub",
                "duckdb",
            )
        },
    }
    dump(dest / "summary.json", summary)


if __name__ == "__main__":
    main()
