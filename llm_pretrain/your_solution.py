import os
import argparse
import hashlib
import json
import platform
from pathlib import Path
from datasets import Features, Sequence, Value, load_dataset
from transformers import (
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator,
    set_seed,
)
import torch
import time


# Don't change this parameter
MAX_TRAINING_TIME_SECONDS = 60 * 15
MAX_LENGTH = 512
INPUT_IDS = 'input_ids'
ATTENTION_MASK = 'attention_mask'
LABELS = 'labels'

# Don't change these parameters
TOKENIZER_NAME = "ai-forever/rugpt3small_based_on_gpt2"
OUTPUT_DIR = "./output_dir"
NUM_SHARDS = 32
VALIDATION_SIZE = 5000


# TODO: Configure training parameters
TRAINING_CONFIG = {
    'output_dir': f'{OUTPUT_DIR}/gpt2-1b-russian',
    'optim': 'adamw_torch_fused',
    'num_train_epochs': 1,
    'per_device_train_batch_size': 4,
    'save_steps': 100,
    'save_total_limit': 2,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'warmup_steps': 200,
    'logging_steps': 1,
    'eval_steps': 100,
    'eval_strategy': 'steps',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_loss',
    'bf16': True,
    'tf32': True,
    'gradient_checkpointing': False,
    'gradient_accumulation_steps': 1,
    'dataloader_num_workers': 4,
    'torch_compile': True,
    'report_to': 'none',
}


class TimeoutCallback(TrainerCallback):
    """Callback to stop training after a specified timeout."""
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                control.should_training_stop = True
                # Include the final weights in best-checkpoint selection.
                control.should_evaluate = True
                control.should_save = True
                print(f"Training stopped after {elapsed:.2f} seconds")
        return control


def prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_function(examples, tokenizer):
    batch = tokenizer(examples['text'], truncation=True,
                      padding='max_length', max_length=MAX_LENGTH)
    batch[LABELS] = [
        [token if active else -100 for token, active in zip(ids, mask)]
        for ids, mask in zip(batch[INPUT_IDS], batch[ATTENTION_MASK])
    ]
    return batch


def save_as_parquets(ds, output_dir=OUTPUT_DIR, num_shards=NUM_SHARDS):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for index in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=index, contiguous=True)
        shard.to_parquet(f'{output_dir}/{index:05d}.parquet')


def prepare_dataset():
    started = time.monotonic()
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train")
    tokenizer = prepare_tokenizer()
    features = Features({
        INPUT_IDS: Sequence(Value('int32'), length=MAX_LENGTH),
        ATTENTION_MASK: Sequence(Value('int8'), length=MAX_LENGTH),
        LABELS: Sequence(Value('int32'), length=MAX_LENGTH),
    })
    tokenized = dataset.map(
        tokenize_function, batched=True, batch_size=128,
        num_proc=min(8, os.cpu_count() or 1),
        fn_kwargs={'tokenizer': tokenizer},
        remove_columns=dataset.column_names, features=features,
        desc='Tokenizing Wikipedia',
    )
    save_as_parquets(tokenized)
    reloaded = load_tokenized_dataset()
    assert len(reloaded) == len(dataset)
    # Verify that serialization preserves the exact validation boundary.
    assert reloaded[:VALIDATION_SIZE + 1] == tokenized[:VALIDATION_SIZE + 1]
    validation_ids = dataset[:VALIDATION_SIZE]['id']
    manifest = {
        'dataset': 'wikimedia/wikipedia', 'config': '20231101.ru',
        'dataset_fingerprint': dataset._fingerprint,
        'tokenizer': TOKENIZER_NAME, 'vocab_size': tokenizer.vocab_size,
        'max_length': MAX_LENGTH, 'num_shards': NUM_SHARDS,
        'total_rows': len(dataset), 'train_rows': len(dataset) - VALIDATION_SIZE,
        'validation_rows': VALIDATION_SIZE, 'validation_ids': validation_ids,
        'first_train_id': dataset[VALIDATION_SIZE]['id'],
        'validation_ids_sha256': hashlib.sha256(
            json.dumps(validation_ids).encode()).hexdigest(),
        'padding_labels': -100, 'round_trip_verified': True,
        'prepare_seconds': time.monotonic() - started,
    }
    write_json(Path(OUTPUT_DIR) / 'dataset_manifest.json', manifest)
    print({key: value for key, value in manifest.items() if key != 'validation_ids'})



def load_tokenized_dataset(data_dir=OUTPUT_DIR):
    files = sorted(str(path) for path in Path(data_dir).glob('*.parquet'))
    if len(files) != NUM_SHARDS:
        raise ValueError(f'Expected {NUM_SHARDS} parquet files, found {len(files)}')
    return load_dataset('parquet', data_files=files, split='train')


def split_dataset(dataset, validation_size=VALIDATION_SIZE):
    dataset_size = len(dataset)
    train_dataset = dataset.select(range(validation_size, dataset_size))
    eval_dataset = dataset.select(range(validation_size))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def create_model(tokenizer):
    # Don't change this parameter
    MODEL_CONFIG = {
        'hidden_size': 2048,
        'num_hidden_layers': 12,
        'num_attention_heads': 16,
        'num_key_value_heads': 8,
        'intermediate_size': 8192,
        'head_dim': 128,
        'hidden_act': 'silu',
        'initializer_range': 0.02,
        'scale_attn_weights': True,
        'use_cache': True,
    }

    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **MODEL_CONFIG
    )
    
    model = Qwen3ForCausalLM._from_config(
        config,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16
    )
    
    print(f"Model pad token id: {model.config.pad_token_id}")
    
    with torch.no_grad():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params:,}")
    
    return model


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


class WallTimeLog(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.monotonic()
        self.path = Path(args.output_dir) / 'events.jsonl'
        self.path.write_text('')

    def on_log(self, args, state, control, logs=None, **kwargs):
        event = dict(logs or {}, step=state.global_step,
                     wall_seconds=time.monotonic() - self.started)
        with self.path.open('a') as stream:
            stream.write(json.dumps(event) + '\n')


def train_model():
    args = TrainingArguments(**TRAINING_CONFIG)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'trainer_state.json').exists():
        raise FileExistsError(f'Completed run already exists: {output}')
    set_seed(args.seed)
    tokenizer = prepare_tokenizer()
    dataset = load_tokenized_dataset()
    train_dataset, eval_dataset = split_dataset(dataset)
    model = create_model(tokenizer)
    write_json(output / 'training_args.json', json.loads(args.to_json_string()))
    write_json(output / 'model_config.json', model.config.to_dict())
    write_json(output / 'environment.json', {
        'python': platform.python_version(), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
        'triton_ptxas_path': os.environ.get('TRITON_PTXAS_PATH'),
        'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
        'parameters': sum(p.numel() for p in model.parameters()),
        'parameter_dtype': str(next(model.parameters()).dtype),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    timeout = TimeoutCallback(timeout_seconds=MAX_TRAINING_TIME_SECONDS)
    trainer = Trainer(
        model=model, args=args, train_dataset=train_dataset,
        eval_dataset=eval_dataset, processing_class=tokenizer,
        data_collator=default_data_collator,
        callbacks=[timeout, WallTimeLog()],
    )
    train_result = trainer.train()
    trainer.save_metrics('train', train_result.metrics)
    trainer.save_state()
    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")
    trainer.save_metrics('eval', eval_results)
    trainer.save_state()
    trainer.save_model()
    tokenizer.save_pretrained(output)
    write_json(output / 'run_summary.json', {
        'global_step': trainer.state.global_step,
        'best_model_checkpoint': trainer.state.best_model_checkpoint,
        'best_metric': trainer.state.best_metric,
        'eval_results': eval_results,
        'train_results': train_result.metrics,
        'evaluation_model': 'best checkpoint selected by the original config',
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'train'])
    command = parser.parse_args().command
    if command == 'prepare':
        prepare_dataset()
    else:
        train_model()
