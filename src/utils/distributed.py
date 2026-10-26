"""Process ownership and device setup for single-node distributed training."""

import os
from datetime import timedelta

import torch
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_primary() -> bool:
    return rank() == 0


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def initialize() -> torch.device:
    """Use torchrun's local GPU, or retain ordinary CUDA, MPS, and CPU execution."""
    if "RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            if local_rank >= torch.cuda.device_count():
                raise ValueError("torchrun requested more processes than available CUDA GPUs")
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device, backend = torch.device("cpu"), "gloo"
        dist.init_process_group(backend, timeout=timedelta(minutes=30))
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
