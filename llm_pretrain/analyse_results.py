"""Rebuild figures and factual summaries from the saved experiment logs."""
import argparse
import ast
import hashlib
import html
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RUNS = ['baseline', '01_eval1000', '02_batch16', '03_compile_off',
        '04_lr3e4', '05_lr_extra', '06_inverse_sqrt', '07_combined']
LABELS = ['Baseline', 'Eval/save: 1000', 'Batch: 16', 'Compile: off',
          'LR: 3e-4', 'LR: 6e-4', 'Inverse sqrt', 'Batch 16 + LR 3e-4']
ARCH_KEYS = ['model_type', 'vocab_size', 'hidden_size', 'intermediate_size',
             'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads',
             'head_dim', 'hidden_act', 'bos_token_id', 'eos_token_id', 'pad_token_id']


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_runs(results):
    runs = []
    source_hash = sha256(ROOT / 'your_solution.py')
    prompts = read(ROOT / 'prompts.json')
    for name, label in zip(RUNS, LABELS):
        path = results / name
        config, facts = read(path / 'training_args.json'), read(path / 'facts.json')
        state, manifest = read(path / 'trainer_state.json'), read(path / 'dataset_manifest.json')
        environment = read(path / 'environment.json')
        events = [json.loads(line) for line in (path / 'events.jsonl').read_text().splitlines()]
        train = [event for event in events if 'loss' in event]
        evaluations = []
        for event in events:
            if 'train_runtime' in event:
                break
            if 'eval_loss' in event:
                evaluations.append(event)
        batch = config['per_device_train_batch_size'] * config['gradient_accumulation_steps']
        assert [event['step'] for event in train] == list(range(1, facts['optimizer_steps'] + 1))
        assert len(train) * batch == facts['train_sequences_seen']
        assert environment['source_sha256'] == source_hash
        assert environment['parameters'] == 960881664
        assert manifest['validation_rows'] == 5000 and manifest['max_length'] == 512
        assert manifest['tokenizer'] == 'ai-forever/rugpt3small_based_on_gpt2'
        assert config['seed'] == 42
        assert all(math.isfinite(event['loss']) for event in train)
        assert all(math.isfinite(event['eval_loss']) for event in evaluations)
        assert math.isclose(read(path / 'eval_results.json')['eval_loss'],
                            facts['selected_checkpoint_eval_loss'], abs_tol=1e-7)
        generations = {mode: read(path / filename) for mode, filename in
                       [('greedy', 'generations.json'), ('sampling', 'generations_sample.json')]}
        for content in generations.values():
            assert [item['prompt'] for item in content['generations']] == prompts
        runs.append(dict(name=name, label=label, config=config, facts=facts,
                         train=train, evaluations=evaluations, batch=batch,
                         manifest=manifest, architecture=read(path / 'model_config.json'),
                         best_step=state['best_global_step'], max_steps=state['max_steps'],
                         generations=generations))
    assert len({json.dumps({key: run['architecture'][key] for key in ARCH_KEYS}, sort_keys=True)
                for run in runs}) == 1
    assert len({(run['manifest']['validation_ids_sha256'], run['manifest']['train_rows'])
                for run in runs}) == 1
    for mode in ['greedy', 'sampling']:
        assert len({json.dumps(run['generations'][mode]['decoding'], sort_keys=True)
                    for run in runs}) == 1
    trees = [ast.parse((ROOT / path).read_text()) for path in
             ['starter/your_solution.py', 'your_solution.py']]
    for name in ['create_model', 'split_dataset', 'TimeoutCallback']:
        nodes = [next(node for node in tree.body if getattr(node, 'name', None) == name)
                 for tree in trees]
        assert ast.dump(nodes[0]) == ast.dump(nodes[1]), name
    return runs


def save_figure(fig, results, name):
    for axis in fig.axes:
        axis.grid(alpha=0.2)
        axis.spines[['top', 'right']].set_visible(False)
    fig.savefig(results / f'{name}.png', dpi=170)
    fig.savefig(results / f'{name}.svg')
    plt.close(fig)


def figures(runs, results):
    colors = plt.get_cmap('tab10').colors
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout='constrained')
    for run, color in zip(runs, colors):
        times = {event['step']: event['wall_seconds'] for event in run['train']}
        steps = [event['step'] for event in run['evaluations']]
        losses = [event['eval_loss'] for event in run['evaluations']]
        for axis, xs in zip(axes, [[times[step] / 60 for step in steps],
                                   [step * run['batch'] for step in steps]]):
            axis.plot(xs, losses, 'o-', color=color, markersize=3, label=run['label'])
            best_x = times[run['best_step']] / 60 if axis is axes[0] else run['best_step'] * run['batch']
            axis.scatter(best_x, run['facts']['selected_checkpoint_eval_loss'],
                         color=color, marker='*', s=95, zorder=5)
            axis.set_ylabel('Eval loss · первые 5000 статей')
    axes[0].axvline(15, color='gray', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Время до соответствующего шага обучения, мин')
    axes[1].set_xlabel('Обработано обучающих примеров')
    axes[0].legend(fontsize=8, loc='upper right')
    fig.suptitle('Оценка по чекпойнтам · звезда обозначает выбранный чекпойнт')
    save_figure(fig, results, 'evaluation')

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout='constrained')
    for run, color in zip(runs, colors):
        window = 1024 // run['batch']
        assert window * run['batch'] == 1024
        losses = np.convolve([event['loss'] for event in run['train']],
                             np.ones(window) / window, mode='valid')
        events = run['train'][window - 1:]
        axes[0].plot([event['wall_seconds'] / 60 for event in events], losses,
                     color=color, label=run['label'])
        axes[1].plot([event['step'] * run['batch'] for event in events], losses, color=color)
    axes[0].axvline(15, color='gray', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Время от начала обучения, мин')
    axes[1].set_xlabel('Обработано обучающих примеров')
    for axis in axes:
        axis.set_ylabel('Train loss · скользящее среднее')
    axes[0].legend(fontsize=8, loc='upper right')
    fig.suptitle('Train loss · окно сглаживания соответствует 1024 примерам')
    save_figure(fig, results, 'training_loss')

    fig, axis = plt.subplots(figsize=(9, 4.6), layout='constrained')
    for index in [1, 4, 5, 6, 7]:
        run = runs[index]
        axis.plot([event['step'] for event in run['train']],
                  [event['learning_rate'] * 1e4 for event in run['train']],
                  label=run['label'], color=colors[index])
    axis.set(xlabel='Шаг оптимизатора', ylabel='Learning rate × 10⁴',
             title='Фактический learning rate из логов · warmup 200 шагов')
    axis.legend(fontsize=9, loc='center right', bbox_to_anchor=(0.99, 0.75))
    save_figure(fig, results, 'learning_rates')


def summaries(runs, results):
    rows, generation_sections = [], ['# Генерации всех запусков\n',
        'Продолжения приведены полностью, без исправлений. Все режимы ограничены 128 новыми токенами.\n',
        'Sampling: temperature=0.8, top_p=0.95, top_k=0; seed=42 для каждого промпта.\n']
    for run in runs:
        facts, config = run['facts'], run['config']
        rows.append(dict(run=run['name'], batch=run['batch'],
                         learning_rate=config['learning_rate'], scheduler=config['lr_scheduler_type'],
                         torch_compile=config['torch_compile'], eval_save_steps=config['eval_steps'],
                         optimizer_steps=facts['optimizer_steps'], selected_step=run['best_step'],
                         examples=facts['train_sequences_seen'], stop_seconds=facts['callback_stop_seconds'],
                         eval_loss=facts['selected_checkpoint_eval_loss'],
                         perplexity_from_eval_loss=math.exp(facts['selected_checkpoint_eval_loss']),
                         eval_seconds_before_stop=facts['evaluation_seconds_before_stop'],
                         examples_per_second=facts['actual_sequences_per_second_until_stop'],
                         planned_scheduler_steps=run['max_steps'],
                         last_logged_learning_rate=run['train'][-1]['learning_rate']))
        generation_sections.append(f"\n## {run['name']}\n")
        for mode, content in run['generations'].items():
            for generation in content['generations']:
                title = html.escape(f"{mode} · {generation['prompt']}")
                generation_sections.append(f'<details>\n<summary>{title}</summary>\n\n'
                                           f"Промпт: **{generation['prompt']}**\n\n"
                                           '```text\n' + generation['continuation'] + '\n```\n\n</details>\n')
    summary = dict(runs=rows, matched_step_6000={}, checks=dict(
        identical_architecture_split_and_timeout_ast=True,
        identical_training_source_hash=True, same_seed_and_prompts=True,
        complete_training_logs=True, matched_generation_settings=True))
    for name in ['01_eval1000', '04_lr3e4', '05_lr_extra', '06_inverse_sqrt']:
        run = next(run for run in runs if run['name'] == name)
        event = next(event for event in run['evaluations'] if event['step'] == 6000)
        summary['matched_step_6000'][name] = dict(eval_loss=event['eval_loss'], examples=24000)
    summary['matched_example_evaluations'] = {}
    for comparison, examples, names in [
        ('eval_frequency', 4000, ['baseline', '01_eval1000']),
        ('compile', 20000, ['01_eval1000', '03_compile_off']),
        ('batch_size', 16000, ['01_eval1000', '02_batch16', '04_lr3e4', '07_combined']),
    ]:
        matched = {}
        for name in names:
            run = next(run for run in runs if run['name'] == name)
            assert examples % run['batch'] == 0
            step = examples // run['batch']
            event = next(event for event in run['evaluations'] if event['step'] == step)
            matched[name] = dict(optimizer_steps=step, eval_loss=event['eval_loss'])
        summary['matched_example_evaluations'][comparison] = dict(examples=examples, runs=matched)
    (results / 'analysis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (results / 'generations.md').write_text('\n'.join(generation_sections))
    table = ['| Запуск | Batch | LR | Scheduler | Compile | Eval/save, шаги | Шаги | Примеры | Стоп, с | Eval loss | exp(loss) |',
             '|---|---:|---:|---|:---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        table.append(f"| {row['run']} | {row['batch']} | {row['learning_rate']:.0e} | {row['scheduler']} | "
                     f"{'да' if row['torch_compile'] else 'нет'} | {row['eval_save_steps']} | "
                     f"{row['optimizer_steps']} | {row['examples']} | {row['stop_seconds']:.2f} | "
                     f"{row['eval_loss']:.4f} | {row['perplexity_from_eval_loss']:.2f} |")
    (results / 'metrics.md').write_text('\n'.join(table) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    parser.add_argument('--verify-weights', action='store_true')
    args = parser.parse_args()
    runs = load_runs(args.results)
    figures(runs, args.results)
    summary = summaries(runs, args.results)
    if args.verify_weights:
        for name in ['04_lr3e4', '07_combined']:
            expected = read(args.results / name / 'checkpoint_storage.json')['exported_model_sha256']
            for filename, checksum in expected.items():
                assert sha256(ROOT / 'saved_models' / name / filename) == checksum
        summary['checks']['two_local_models_sha256_verified'] = True
        (args.results / 'analysis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print((args.results / 'metrics.md').read_text())
    print('Verified:', summary['checks'])


if __name__ == '__main__':
    main()
