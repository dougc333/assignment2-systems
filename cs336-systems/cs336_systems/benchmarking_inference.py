from dataclasses import dataclass, asdict
import time

import numpy as np
import torch
from transformers import HfArgumentParser

from cs336_basics.model import BasicsTransformerLM


@dataclass
class InferenceConfig:
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int

    warmup_iters: int = 10
    benchmarking_iters: int = 100
    batch_size: int = 16
    context_length: int = 128
    vocab_size: int = 10_000

    mixed_precision: bool = False
    enable_tf32: bool = False
    compile_model: bool = True


config = HfArgumentParser(
    InferenceConfig
).parse_args_into_dataclasses()[0]

if not torch.cuda.is_available():
    raise RuntimeError("A CUDA GPU is required for this benchmark.")

device = torch.device("cuda")
device_name = torch.cuda.get_device_name(0)
compute_capability = torch.cuda.get_device_capability(0)

if config.enable_tf32:
    torch.set_float32_matmul_precision("high")

if config.mixed_precision:
    amp_dtype = (
        torch.bfloat16
        if compute_capability[0] >= 8
        else torch.float16
    )
else:
    amp_dtype = None

model = BasicsTransformerLM(
    vocab_size=config.vocab_size,
    context_length=config.context_length,
    d_model=config.d_model,
    num_layers=config.num_layers,
    num_heads=config.num_heads,
    d_ff=config.d_ff,
).to(device)

model.eval()

if config.compile_model:
    model = torch.compile(model)

x = torch.randint(
    low=0,
    high=config.vocab_size,
    size=(config.batch_size, config.context_length),
    device=device,
)


def autocast_context():
    return torch.amp.autocast(
        device_type="cuda",
        dtype=amp_dtype,
        enabled=config.mixed_precision,
    )


# Warm up torch.compile, kernels and memory allocations.
with torch.inference_mode():
    for _ in range(config.warmup_iters):
        with autocast_context():
            _ = model(x)

torch.cuda.synchronize()

latencies_ms = []

with torch.inference_mode():
    for _ in range(config.benchmarking_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with autocast_context():
            _ = model(x)

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

latencies = np.asarray(latencies_ms)
mean_latency_seconds = latencies.mean() / 1000

sequences_per_second = (
    config.batch_size / mean_latency_seconds
)

tokens_per_iteration = (
    config.batch_size * config.context_length
)
input_tokens_per_second = (
    tokens_per_iteration / mean_latency_seconds
)

results = {
    "config": asdict(config),
    "gpu": device_name,
    "compute_capability": compute_capability,
    "amp_dtype": str(amp_dtype),
    "latency_ms": {
        "mean": float(latencies.mean()),
        "std": float(latencies.std()),
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
    },
    "sequences_per_second": sequences_per_second,
    "input_tokens_per_second": input_tokens_per_second,
}

print(results)