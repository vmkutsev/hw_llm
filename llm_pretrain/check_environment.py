import os
import json
import importlib.metadata as metadata

# Keep preflight compilation separate from the timed experiment.
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/work/.cache/inductor-preflight'
os.environ['TRITON_CACHE_DIR'] = '/work/.cache/triton-preflight'

import torch
from flash_attn import flash_attn_func

print(json.dumps({
    'versions': {name: metadata.version(name) for name in
                 ['torch', 'transformers', 'datasets', 'accelerate', 'flash-attn', 'triton']},
    'cuda': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(),
    'bf16_supported': torch.cuda.is_bf16_supported(),
}, indent=2), flush=True)

q = torch.randn(2, 512, 16, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(2, 512, 8, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
v = torch.randn_like(k, requires_grad=True)
loss = flash_attn_func(q, k, v, causal=True).float().square().mean()
loss.backward()
assert torch.isfinite(q.grad).all()
print('PASS: Flash Attention 2 BF16 forward/backward', flush=True)

parameter = torch.nn.Parameter(torch.randn(64, 64, device='cuda', dtype=torch.bfloat16))
optimizer = torch.optim.AdamW([parameter], lr=5e-5, fused=True)
parameter.float().square().mean().backward()
optimizer.step()
assert torch.isfinite(parameter).all()
print('PASS: fused AdamW', flush=True)

compiled = torch.compile(lambda tensor: tensor.sin() * tensor)
actual = compiled(parameter)
torch.testing.assert_close(actual, parameter.sin() * parameter, atol=0.02, rtol=0.02)
actual.sum().backward()
torch.cuda.synchronize()
print('PASS: torch.compile forward/backward', flush=True)
