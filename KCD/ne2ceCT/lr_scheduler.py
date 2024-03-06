import numpy as np
import torch
from torch.optim.lr_scheduler import LRScheduler


class LinearWarmupExponentialDecayScheduler(LRScheduler):
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


# Example usage
model = ...  # your model
optimizer = torch.optim.Adam(model.parameters(), lr=0)  # start with lr=0
scheduler = LinearWarmupExponentialDecayScheduler(optimizer, warmup_epochs=20, total_epochs=100, lr_max=0.01, decay_rate=0.9999)

for epoch in range(100):
    # Train your model
    ...
    scheduler.step()  # Update the learning rate
