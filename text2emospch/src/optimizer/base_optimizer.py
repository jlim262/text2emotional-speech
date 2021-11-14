
import torch
import torch.nn as nn
import torch.optim as optim

from built.registry import Registry


@Registry.register(category="optimizer")
class AdamWOptimizer(object):
    def __new__(cls, params, lr, total_steps):
        optimizer = optim.AdamW(params, lr=lr)
        return optimizer
