"""Run independent fresh-model experiments sequentially on one allocated GPU."""

import argparse
import ast
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STARTER_SHA256 = "eb24509433f8cf1ed54e5b19356e9cb4431b54a842e18daebcb70db92d76eba3"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, required=True)
    args = parser.parse_args()
    export = args.export.resolve()
    export.mkdir(parents=True, exist_ok=True)
    source = ROOT / "starter/train.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == STARTER_SHA256
    runtime = json.loads(subprocess.check_output([sys.executable, "inspect_runtime.py"], cwd=ROOT, text=True))
    (export / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    shutil.copyfile(ROOT / "data/manifest.json", export / "data_manifest.json")
    (export / "experiments").mkdir(exist_ok=True)
    (ROOT / "results").symlink_to(export / "experiments", target_is_directory=True)
    (export / "logs").mkdir(exist_ok=True)
    (export / "configs").mkdir(exist_ok=True)
    records = []

    def run_one(config, original_baseline=False):
        name = config["name"]
        config_path = export / "configs" / f"{name}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        command = [sys.executable, "-u", "starter/train.py"] if original_baseline else [
            sys.executable, "-u", "run_experiment.py", str(config_path),
        ]
        env = dict(os.environ, HW2_GPU_UUID=runtime["gpu_uuid"], TOKENIZERS_PARALLELISM="false")
        env["TORCHINDUCTOR_CACHE_DIR"] = str(ROOT / ".cache" / name / "inductor")
        env["TRITON_CACHE_DIR"] = str(ROOT / ".cache" / name / "triton")
        record = {
            "name": name, "config": config, "gpu_uuid": runtime["gpu_uuid"],
            "source_sha256": STARTER_SHA256, "started_at_unix": time.time(),
            "cold_compile_cache": True, "original_baseline": original_baseline,
        }
        print(f"HW2 START {name}", flush=True)
        history = []
        current_step, current_elapsed = None, None
        timed_out = threading.Event()
        with (export / "logs" / f"{name}.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       bufsize=1, start_new_session=True)

            def cancel():
                timed_out.set()
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            timer = threading.Timer(max(1200, config["seconds"] + 600), cancel)
            timer.start()
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(f"[{name}] {line}", end="", flush=True)
                    try:
                        item = ast.literal_eval(line.strip())
                    except (ValueError, SyntaxError):
                        try:
                            item = json.loads(line.strip())
                        except (ValueError, TypeError):
                            continue
                    if isinstance(item, dict) and "elapsed_seconds" in item and "step" in item:
                        current_step, current_elapsed = item["step"], item["elapsed_seconds"]
                    if isinstance(item, dict) and "loss" in item:
                        history.append(dict(item, optimizer_step=current_step, elapsed_seconds=current_elapsed))
                returncode = process.wait()
            finally:
                timer.cancel()
        record.update(returncode=returncode, timed_out=timed_out.is_set(), finished_at_unix=time.time())
        summary_path = export / "experiments" / name / "summary.json"
        record["status"] = "completed" if returncode == 0 and summary_path.exists() else "failed"
        if summary_path.exists():
            record["summary"] = json.loads(summary_path.read_text())
        (export / "experiments" / name).mkdir(exist_ok=True)
        (export / "experiments" / name / "run_metadata.json").write_text(json.dumps(record, indent=2) + "\n")
        (export / "experiments" / name / "loss_history.json").write_text(json.dumps(history, indent=2) + "\n")
        records.append(record)
        (export / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"HW2 END {name}: {record['status']}", flush=True)
        return record

    def usable(record):
        summary = record.get("summary", {})
        return (record["status"] == "completed" and
                isinstance(summary.get("tokens_per_second"), (int, float)) and
                summary["tokens_per_second"] > 0 and
                math.isfinite(summary.get("validation_loss", math.nan)))

    baseline = run_one({"name": "baseline", "seconds": 300, "attention": "sdpa", "settings": {}}, True)
    if not usable(baseline):
        raise RuntimeError("The unmodified baseline did not produce usable metrics")
    for config_path in sorted((ROOT / "configs").glob("*.json")):
        run_one(json.loads(config_path.read_text()))
    successful = [record for record in records if usable(record)]
    if len(successful) < 6:
        raise RuntimeError(f"Only {len(successful)} usable configurations; need at least six")
    best = max(successful, key=lambda record: record["summary"]["tokens_per_second"])
    for batch in (64, 128):
        settings = dict(best["config"]["settings"], per_device_train_batch_size=batch, gradient_accumulation_steps=1)
        candidate = {"name": f"09_best_batch{batch}_acc1", "seconds": 90,
                     "attention": best["config"]["attention"], "settings": settings}
        if any(item["config"]["settings"] == settings and item["config"]["attention"] == candidate["attention"]
               for item in records):
            continue
        result = run_one(candidate)
        if not usable(result):
            break
    successful = [record for record in records if usable(record)]
    best = max(successful, key=lambda record: record["summary"]["tokens_per_second"])
    final_config = dict(best["config"], name="final", seconds=300)
    final = run_one(final_config)
    if not usable(final):
        raise RuntimeError("The final 300-second experiment did not produce usable metrics")
    final_gpu = json.loads(subprocess.check_output([sys.executable, "inspect_runtime.py"], cwd=ROOT, text=True))
    assert final_gpu["gpu_uuid"] == runtime["gpu_uuid"]
    report = {
        "status": "completed", "gpu_uuid": runtime["gpu_uuid"],
        "baseline": baseline["summary"], "selected_probe": best["name"],
        "final": final["summary"], "final_config": final_config,
        "speedup": final["summary"]["tokens_per_second"] / baseline["summary"]["tokens_per_second"],
        "successful_experiments": sum(usable(record) for record in records),
    }
    (export / "suite.json").write_text(json.dumps(report, indent=2) + "\n")
    print("HW2 RESULT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
