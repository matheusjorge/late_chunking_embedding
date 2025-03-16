from typing import Dict, List, Protocol

import torch


class Model(Protocol):
    def encode(self, *args, **kwargs) -> Dict[str, torch.Tensor]: ...


class Tokenizer(Protocol):
    def tokenize(self, *args, **kwargs) -> Dict[str, torch.Tensor]: ...
