
import torch
import torch.nn as nn
import torch.optim as optim
from built.registry import Registry
import transformers

@Registry.register(category="scheduler")
class CosineScheduler(object):
    def __new__(cls, optimizer, num_warmup_steps, total_steps, **kwargs):
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps)
        
        return scheduler
