import torch.nn as nn
from gate.models.core import reinit


class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()

    def init_weights(self):
        reinit(self)
