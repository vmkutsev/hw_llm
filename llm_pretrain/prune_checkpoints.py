import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument('run_dir', type=Path)
args = parser.parse_args()
root = Path('output_dir').resolve()
run = args.run_dir.resolve()
assert run.is_relative_to(root) and run != root
summary = json.loads((run / 'run_summary.json').read_text())
assert (run / 'generations.json').exists()
assert (run / 'generations_sample.json').exists()
best = Path(summary['best_model_checkpoint']).resolve()
assert best.parent == run
weights = sorted(run.glob('model*.safetensors'))
assert weights
hashes = {p.name: digest(p) for p in weights}
assert all(digest(best / name) == value for name, value in hashes.items())
metadata = run / 'checkpoint_metadata'
metadata.mkdir(exist_ok=True)
removed = []
for checkpoint in run.iterdir():
    if re.fullmatch(r'checkpoint-\d+', checkpoint.name):
        assert checkpoint.is_dir() and not checkpoint.is_symlink()
        destination = metadata / checkpoint.name
        destination.mkdir(exist_ok=True)
        for item in checkpoint.glob('*.json'):
            shutil.copy2(item, destination / item.name)
        removed.append(checkpoint.name)
        shutil.rmtree(checkpoint)
(run / 'checkpoint_storage.json').write_text(json.dumps({
    'retained': 'Exported best model and tokenizer at run root; checkpoint JSON metadata retained.',
    'exported_model_sha256': hashes,
    'export_verified_identical_to_best_checkpoint': True,
    'removed_intermediate_checkpoints': removed,
    'note': 'Optimizer states removed after completion. Each experiment starts from scratch.',
}, indent=2) + '\n')
print({'pruned': removed, 'retained_model': str(run)})
