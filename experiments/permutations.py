import torch
import torch.nn as nn
import torchaudio


class CQTPermuter:
    def __init__(self, seed: int=0):
        """Sets up a seed for permutation consistency"""
        self.gen = torch.Generator()
        self.gen.manual_seed(seed)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Applies defined permutation to input"""
        return NotImplementedError


class CQTRandPerm(CQTPermuter):
    def __init__(
            self,
            seed: int=0
    ):
        super().__init__(seed)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Applies a random permutation across the frequency axis to input"""
        return x