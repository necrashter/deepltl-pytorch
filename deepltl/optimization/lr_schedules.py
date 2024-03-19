import torch
from torch.optim.lr_scheduler import LambdaLR


class TransformerSchedule(LambdaLR):
    """
    Learning Rate Schedule proposed by Vaswani et al. (2017) that corresponds to a linear increase during the warmup phase followed by a decrease proportional to the inverse of the square root of the step number.
    """

    def __init__(self, optimizer, d_embedding, warmup_steps=4000):
        self.d_embedding = d_embedding
        self.warmup_steps = warmup_steps
        super(TransformerSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=-1
        )

    def lr_lambda(self, step):
        n = step + 1  # input step is 0-indexed
        return (self.d_embedding ** -0.5) * min(n**-0.5, n * self.warmup_steps**-1.5)
