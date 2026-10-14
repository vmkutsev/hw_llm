"""Record the actual CUDA device and pinned training software."""

import csv
import importlib.metadata
import json
import platform
import subprocess


def inspect_runtime():
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible CUDA device, got {torch.cuda.device_count()}")
    device = torch.cuda.get_device_properties(0)
    if "A100" not in device.name:
        raise RuntimeError(f"Expected A100, got {device.name}")
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ], text=True)
    rows = [[value.strip() for value in row] for row in csv.reader(raw.splitlines())]
    cuda_uuid = str(getattr(device, "uuid", ""))
    matched = [row for row in rows if row[0].removeprefix("GPU-") == cuda_uuid.removeprefix("GPU-")]
    if not matched and len(rows) == 1:
        matched = rows
    if len(matched) != 1:
        raise RuntimeError("Could not identify the visible CUDA device in nvidia-smi")
    gpu_uuid, name, memory_mib, driver = matched[0]
    expected = {
        "torch": "2.6.0", "transformers": "4.56.2", "datasets": "3.6.0",
        "accelerate": "1.6.0", "trl": "0.23.1", "liger-kernel": "0.6.2",
        "flash-attn": "2.7.3",
    }
    versions = {name: importlib.metadata.version(name) for name in expected}
    for package, version in expected.items():
        if versions[package].split("+")[0] != version:
            raise RuntimeError(f"Wrong {package} version: {versions[package]}, expected {version}")
    return {
        "gpu_uuid": gpu_uuid, "gpu_name": name, "gpu_memory_mib": float(memory_mib),
        "cuda_visible_memory_bytes": device.total_memory, "driver": driver,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_runtime": torch.version.cuda, "python": platform.python_version(),
        "packages": versions,
    }


if __name__ == "__main__":
    print(json.dumps(inspect_runtime(), indent=2))
