# optimization/utils/seeding.py

import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """
    Set seed for reproducibility across random, numpy and torch.

    Parameters
    ----------
    seed : int
        Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
