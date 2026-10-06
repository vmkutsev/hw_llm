import argparse
import hashlib
import json
import re
from pathlib import Path

import your_solution as solution


parser = argparse.ArgumentParser()
parser.add_argument('config', type=Path)
args = parser.parse_args()
experiment = json.loads(args.config.read_text())
run_id = experiment['id']
assert re.fullmatch(r'[a-zA-Z0-9_-]+', run_id)
output = Path('output_dir/experiments') / run_id
if (output / 'events.jsonl').exists():
    raise FileExistsError(f'Run has already started: {output}')
allowed = {
    'eval_steps', 'save_steps', 'per_device_train_batch_size',
    'gradient_accumulation_steps', 'learning_rate', 'lr_scheduler_type',
    'torch_compile', 'optim', 'bf16', 'tf32', 'warmup_steps',
}
overrides = experiment['training_overrides']
assert set(overrides) <= allowed, set(overrides) - allowed
solution.TRAINING_CONFIG.update(overrides, output_dir=str(output))
output.mkdir(parents=True, exist_ok=True)
solution.write_json(output / 'experiment.json', {
    **experiment,
    'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'compilation_cache': 'separate empty cache per run',
    'initialized_from': 'random weights, seed 42; no checkpoint resume',
})
solution.train_model()
