from typing import Dict

import torch

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, *inputs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
