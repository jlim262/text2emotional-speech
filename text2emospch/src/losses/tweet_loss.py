
import torch.nn as nn

from built.loss import LossBase
from built.registry import Registry


@Registry.register(category="loss")
class TweetLoss(LossBase):
    def __call__(self, outputs, targets, data=None, is_train=False, device='cpu'):
        loss_fn = nn.CrossEntropyLoss()
        targets = targets.to(device)
        return loss_fn(input=outputs, target=targets)