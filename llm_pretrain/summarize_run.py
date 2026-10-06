import argparse
import json
import math
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('run_dir', type=Path)
args = parser.parse_args()
p = args.run_dir
events = [json.loads(line) for line in (p / 'events.jsonl').read_text().splitlines()]
train = [row for row in events if 'loss' in row]
evaluations = [row for row in events if 'eval_loss' in row]
summary = json.loads((p / 'run_summary.json').read_text())
config = json.loads((p / 'training_args.json').read_text())
manifest = json.loads((p / 'dataset_manifest.json').read_text())
last = train[-1]
before_stop = [row for row in evaluations if row['wall_seconds'] < last['wall_seconds']]
eval_seconds = sum(row['eval_runtime'] for row in before_stop)
sequences = last['step'] * config['per_device_train_batch_size'] * config['gradient_accumulation_steps']
stop = re.search(r'Training stopped after ([\d.]+) seconds', (p / 'train_stdout.log').read_text())
facts = {
    'optimizer_steps': last['step'],
    'callback_stop_seconds': float(stop[1]),
    'last_optimizer_step_wall_seconds': last['wall_seconds'],
    'train_call_seconds_including_final_evaluation_and_save': summary['train_results']['train_runtime'],
    'train_sequences_seen': sequences,
    'token_positions_seen_including_padding': sequences * manifest['max_length'],
    'actual_sequences_per_second_until_stop': sequences / last['wall_seconds'],
    'actual_optimizer_steps_per_second_until_stop': last['step'] / last['wall_seconds'],
    'first_eval_step': evaluations[0]['step'],
    'first_eval_loss': evaluations[0]['eval_loss'],
    'last_weights_eval_loss': evaluations[-2]['eval_loss'],
    'selected_checkpoint': summary['best_model_checkpoint'],
    'selected_checkpoint_eval_loss': summary['eval_results']['eval_loss'],
    'exp_selected_eval_loss': math.exp(summary['eval_results']['eval_loss']),
    'completed_evaluations_before_stop': len(before_stop),
    'evaluation_seconds_before_stop': eval_seconds,
    'evaluation_fraction_of_time_until_stop': eval_seconds / last['wall_seconds'],
    'first_step_wall_seconds_including_initial_compilation': train[0]['wall_seconds'],
    'throughput_note': 'Rates use actual completed steps. Trainer train rates use the planned full epoch despite early stopping.',
}
(p / 'facts.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(facts, ensure_ascii=False, indent=2))
