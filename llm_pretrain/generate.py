import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM, set_seed

parser = argparse.ArgumentParser()
parser.add_argument('prompts', type=Path, help='JSON list of user-selected prompts')
parser.add_argument('--model', default='output_dir/gpt2-1b-russian')
parser.add_argument('--max-new-tokens', type=int, default=128)
parser.add_argument('--sample', action='store_true')
parser.add_argument('--temperature', type=float, default=0.8)
parser.add_argument('--top-p', type=float, default=0.95)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
output_path = args.output or Path(args.model) / 'generations.json'
if output_path.exists():
    raise FileExistsError(output_path)

prompts = json.loads(args.prompts.read_text())
assert prompts and all(isinstance(prompt, str) and prompt.strip() for prompt in prompts)
set_seed(args.seed)
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = Qwen3ForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2',
).to('cuda').eval()

generations = []
decoding = {'do_sample': args.sample, 'max_new_tokens': args.max_new_tokens}
if args.sample:
    decoding.update(temperature=args.temperature, top_p=args.top_p, top_k=0)
with torch.inference_mode():
    for prompt in prompts:
        set_seed(args.seed)
        inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
        output = model.generate(**inputs, **decoding,
                                pad_token_id=tokenizer.pad_token_id)
        continuation = tokenizer.decode(output[0, inputs.input_ids.shape[1]:],
                                        skip_special_tokens=True)
        generations.append({'prompt': prompt, 'continuation': continuation})
        print(prompt + continuation + '\n')

result = {'decoding': decoding, 'seed_per_prompt': args.seed,
          'model': args.model, 'generations': generations}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + '\n')
