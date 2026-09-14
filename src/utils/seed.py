import random
import numpy as np
import torch

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.benchmark.deterministic.seed(seed_value)