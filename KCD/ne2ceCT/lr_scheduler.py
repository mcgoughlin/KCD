import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupExponentialDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, lr_max, decay_rate, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.lr_max = lr_max
        self.decay_rate = decay_rate
        self.gamma = decay_rate
        super(LinearWarmupExponentialDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            lr = max((self.lr_max / self.warmup_epochs) * self.last_epoch, self.base_lrs[0])
            return [lr for base_lr in self.base_lrs]
        else:
            decay_steps = self.last_epoch - self.warmup_epochs
            lr = self.lr_max * (self.decay_rate ** decay_steps)
            return [lr for base_lr in self.base_lrs]


    def _get_closed_form_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            lr = (self.lr_max / self.warmup_epochs) * self.last_epoch
            return [lr for base_lr in self.base_lrs]
        else:
            return [base_lr * self.decay_rate ** (self.last_epoch - self.warmup_epochs)
                    for base_lr in self.base_lrs]

