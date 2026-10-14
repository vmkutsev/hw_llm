"""Run the supplied benchmark with only the permitted settings overridden."""

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

from inspect_runtime import inspect_runtime


ALLOWED = {
    "per_device_train_batch_size", "gradient_accumulation_steps",
    "gradient_checkpointing", "torch_compile", "use_liger_kernel",
    "activation_offloading", "packing", "padding_free",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    experiment = json.loads(args.config.read_text())
    settings = experiment.get("settings", {})
    if set(settings) - ALLOWED:
        raise ValueError(f"Settings are not permitted: {set(settings) - ALLOWED}")
    root = Path(__file__).resolve().parent
    source = root / "starter/train.py"
    spec = importlib.util.spec_from_file_location("supplied_benchmark", source)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    benchmark.TRAINING_CONFIG.update(settings)
    benchmark.TRAINING_CONFIG["output_dir"] = str(root / "results" / experiment["name"])
    benchmark.TRAIN_SECONDS = experiment["seconds"]
    benchmark.ATTENTION_IMPLEMENTATION = experiment.get("attention", "sdpa")
    benchmark.DATA = root / "data"
    output = Path(benchmark.TRAINING_CONFIG["output_dir"])
    if output.exists():
        raise FileExistsError(output)
    gpu = inspect_runtime()
    if os.environ.get("HW2_GPU_UUID", gpu["gpu_uuid"]) != gpu["gpu_uuid"]:
        raise RuntimeError("The GPU changed during the benchmark suite")
    metadata = {
        "experiment": experiment,
        "training_config": benchmark.TRAINING_CONFIG,
        "model_config": benchmark.MODEL_CONFIG,
        "gpu": gpu,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "started_at_unix": time.time(),
        "status": "running",
    }
    print(json.dumps(metadata), flush=True)
    try:
        benchmark.main()
        metadata["status"] = "completed"
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        metadata["finished_at_unix"] = time.time()
        if output.is_dir():
            (output / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
