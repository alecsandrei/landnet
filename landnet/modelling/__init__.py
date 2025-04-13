from __future__ import annotations

from functools import cache

import torch


@cache
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.get_default_device()


def torch_clear():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize(torch.cuda.current_device())
