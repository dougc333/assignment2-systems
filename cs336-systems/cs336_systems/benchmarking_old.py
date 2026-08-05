from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
from transformers import HfArgumentParser
import torch
from contextlib import nullcontext
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import numpy as np
import time

from cs336_basics.nn_utils import cross_entropy, clip_gradient
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW

# parsing the benchmarking configuration
@dataclass
class BenchMarkingConfig:
    # treatment variables for scaling
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    # optional arguments
    benchmarking_iters: Optional[int] = field(default=5)
    warmup_iters: Optional[int] = field(default=1)
    wandb_run_name: Optional[str] = field(default='None')
    mixed_precision: Optional[bool] = field(default=False)
    enable_tf32: Optional[bool] = field(default=False)
    use_rms_norm: Optional[bool] = field(default=True)
    # fixed configs
    wandb_project: str = 'cs336-assignment2-systems'
    context_length: int = 128
    batch_size: int = 16
    vocab_size: int = 10000

    def __post_init__(self):
        self.wandb_logging = self.wandb_run_name != 'None'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


# parsing config
parser = HfArgumentParser(BenchMarkingConfig)
config = parser.parse_args_into_dataclasses()[0]

if config.enable_tf32:
    if config.device != 'cuda':
        raise ValueError('--enable_tf32 requires a CUDA device')
    torch.set_float32_matmul_precision('high')

if config.wandb_logging:
    import wandb
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=asdict(config),
    )
logging.info(f'Benchmarking with config: {asdict(config)}')

# generate random dataset for bench marking
x = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length))
x = x.to(config.device)
y = torch.randint(0, config.vocab_size, (config.batch_size, config.context_length))
y = y.to(config.device)

# Initialize the model with model arguments only. Benchmarking and logging
# options are not part of BasicsTransformerLM's constructor.
model = BasicsTransformerLM(
    vocab_size=config.vocab_size,
    context_length=config.context_length,
    d_model=config.d_model,
    num_layers=config.num_layers,
    num_heads=config.num_heads,
    d_ff=config.d_ff,
)
model = model.to(config.device)
model = torch.compile(model)
# loading the optimizer
optimizer = AdamW(model.parameters())


def training_context():
    """Return a fresh autocast context for each benchmark iteration."""
    if not config.mixed_precision:
        return nullcontext()

    if config.device == 'cuda':
        amp_dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
    else:
        amp_dtype = torch.bfloat16

    return torch.amp.autocast(
        device_type=config.device,
        dtype=amp_dtype,
    )


def synchronize():
    """Wait for queued GPU work without breaking CPU-only runs."""
    if config.device == 'cuda':
        torch.cuda.synchronize()

def forward_pass():
    synchronize()
    logits = model(x)
    loss = cross_entropy(logits, y)
    synchronize()
    return loss

def backward_pass():
    synchronize()
    optimizer.zero_grad()
    loss.backward()
    synchronize()

def timer(run: Callable):
    t1 = time.perf_counter()
    result = run()
    t2 = time.perf_counter()
    return t2-t1, result

# warm up
forward_times = np.zeros(config.benchmarking_iters)
backward_times = np.zeros(config.benchmarking_iters)
for _ in range(config.warmup_iters):
    with training_context():
        loss = forward_pass()
        backward_pass()
        clip_gradient(model.parameters(), 1.0)
        optimizer.step()

for i in range(config.benchmarking_iters):
    with training_context():
        forward_times[i], loss = timer(forward_pass)
        backward_times[i], _ = timer(backward_pass)
        clip_gradient(model.parameters(), 1.0)
        optimizer.step()


# benchmarking
forward_mean = float(np.mean(forward_times))
forward_std = float(np.std(forward_times))
backward_mean = float(np.mean(backward_times))
backward_std = float(np.std(backward_times))
total_mean = forward_mean + backward_mean
tokens_per_iteration = config.batch_size * config.context_length

metrics = {
    'forward_mean_seconds': forward_mean,
    'forward_std_seconds': forward_std,
    'backward_mean_seconds': backward_mean,
    'backward_std_seconds': backward_std,
    'total_mean_seconds': total_mean,
    'training_tokens_per_second': tokens_per_iteration / total_mean,
}

print(f'Forward pass time: {forward_mean}, std: {forward_std}')
print(f'Backward pass time: {backward_mean}, std: {backward_std}')
print(f'Total iteration time: {total_mean}')
print(f'Training throughput: {metrics["training_tokens_per_second"]} tokens/s')

if config.wandb_logging:
    wandb.log(metrics)
    wandb.finish()
