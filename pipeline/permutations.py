from typing import Optional, List
import matplotlib.pyplot as plt
from einops import rearrange
import numpy as np

import torch
import torch.nn as nn



def plot_features(*args, k=0, cmap="magma", bipolar=False):
    """
    Plot one or more features for debugging and evaluation
    """
    features = list(args)
    if len(features) == 0:
        raise ValueError("No features provided")
    
    mats = []
    for idx, feat in enumerate(features):
        arr = feat.detach().cpu().numpy()

        if arr.ndim != 3:
            raise ValueError(f"Expected feature of shape [B, F, T], got {arr.shape}")
        
        K, F, T = arr.shape
        if not (0 <= k < K):
            raise IndexError(f"k out of range for feature {idx}: got {k}, expected 0 <= k < {K}")
        
        mat = arr[k]
        if bipolar:
            mat = np.abs(mat)
        mats.append(mat)

    n = len(mats)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 3))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        im = ax.imshow(mats[i].T, aspect="auto", origin="lower", cmap=cmap)
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Spectrogram [{k}] ({i})")
        plt.colorbar(im, ax=ax, label="Amplitude")

    plt.tight_layout()
    plt.show()



"""Adapters"""

class InputAdapter(nn.Module):
    """
    Einops based adapter
    
    Canonical shape: "b t f"
    Target shape MUST BE INVERTABLE
    """
    def __init__(
            self,
            target_shape: Optional[str] = "b t f",
            canonical_shape: Optional[str] = "b t f",
    ) -> None:
        self._to_shape   = f"{target_shape} -> {canonical_shape}"
        self._from_shape = f"{canonical_shape} -> {target_shape}"

    @torch.no_grad()
    def to_canonical(self, x: torch.Tensor) -> torch.Tensor:
        if self._to_shape is None:
            return x
        return rearrange(x, self._to_shape)
    
    @torch.no_grad()
    def from_canonical(self, x: torch.Tensor) -> torch.Tensor:
        if self._from_shape is None:
            return x
        return rearrange(x, self._from_shape)


"""Permutations"""

class Permutation(nn.Module):
    """
    Base class for frequency-axis permutations.
    """
    def __init__(
            self,
            name: str,
            adapter: Optional[nn.Module] = nn.Identity(),
            complex: Optional[bool] = False,
            seed: Optional[int] = 0,
    ):
        super().__init__()
        self.name    = name
        self.seed    = int(seed)
        self.adapter = adapter
        self.complex = complex
        
        self._gens   = {}

    def _get_generator(self, device: torch.device) -> torch.Tensor:
        if device not in self._gens:
            gen = torch.Generator(device=device)
            gen.manual_seed(self.seed)
            self._gens[device] = gen
        return self._gens[device]
    
    def permute(self, x: torch.Tensor) -> torch.Tensor:
        return NotImplementedError
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.complex:
            assert x.shape[1] == 2, f"Expected dim1 of input to be 2, got {x.shape[1]}"
            x, x_i = torch.unbind(x, dim=1)
        
        x = self.adapter.to_canonical(x)
        x = self.permute(x)
        x = self.adapter.from_canonical(x)

        if self.complex:
            x = torch.stack([x, x_i], dim=1)

        return x


class NoPermutation(Permutation):
    """
    Identity permutation
    """
    def __init__(
            self,
            name: str,
            complex: Optional[bool] = False,
            **kwargs
    ):
        super().__init__(name=name, complex=complex, **kwargs)
    
    def permute(self, x: torch.Tensor) -> torch.Tensor:
        return x


class RandomPermutation(Permutation):
    """
    Random per-frame bin swapping.

    p: probability of swapping a given bin with another bin
    """
    def __init__(
            self,
            name: str,
            adapter: nn.Module,
            p: float = 0.1,
            seed: Optional[int] = 0,
            complex: Optional[bool] = False,
            **kwargs
    ):
        super().__init__(name=name, adapter=adapter, complex=complex, seed=seed)
        
        self.p = p
    
    def permute(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        device = x.device
        gen = self._get_generator(device)

        base = torch.arange(F, device=device)

        swap_mask = torch.rand(B, T, F, device=device, generator=gen) < self.p

        random_keys = torch.rand(B, T, F, device=device, generator=gen)

        scores = torch.where(
            swap_mask,
            random_keys,
            base.view(1, 1, F).expand(B, T, F).float()
        )

        perm_idx = scores.argsort(dim=-1)

        return torch.gather(x, dim=-1, index=perm_idx)


class HighFreqPermutation(Permutation):
    """
    Permutes only high-frequency bins (>= start percentage),
    independently per frame.
    """
    def __init__(
            self,
            name: str,
            adapter: nn.Module,
            start: float = 0.0,
            seed: int = 0,
            complex: Optional[bool] = False,
            **kwargs
    ):
        super().__init__(name=name, adapter=adapter, seed=seed, complex=complex)
        
        assert 0.0 < start < 1.0
        self.start = start

    def permute(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        device = x.device
        gen = self._get_generator(device)

        start_bin = int(self.start * F)
        start_bin = min(max(start_bin, 0), F - 1)

        perm_idx = torch.arange(F, device=device).expand(B, T, F).clone()

        hf_len = F - start_bin
        hf_perm = torch.argsort(
            torch.rand(B, T, hf_len, device=device, generator=gen),
            dim=-1
        ) + start_bin

        perm_idx[..., start_bin:] = hf_perm

        return torch.gather(x, dim=-1, index=perm_idx)


class MicrotonalPermutation(Permutation):
    """
    Permutes bins within each semitone group.

    bins_per_semitone: number of CQT bins per semitone
    """
    def __init__(
            self, name: str,
            adapter: nn.Module,
            bins_per_semitone: int = 1,
            seed: int = 0,
            complex: Optional[bool] = False,
            **kwargs
    ):
        super().__init__(name=name, adapter=adapter, seed=seed, complex=complex)
        self.bins_per_semitone = bins_per_semitone

    @torch.no_grad()
    def permute(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        device = x.device
        gen = self._get_generator(device)

        bps = self.bins_per_semitone
        n_semitones = F // bps
        assert n_semitones * bps == F, "F must be divisible by bins_per_semitone"

        perm = torch.argsort(
            torch.rand(n_semitones, bps, device=device, generator=gen),
            dim=-1
        )

        perm_idx = (
            perm
            + torch.arange(n_semitones, device=device)[:, None] * bps
        ).reshape(-1)

        return x[..., perm_idx]



"""Maskings"""


class FeatureMasker(Permutation):
    """
    Base class for spectrogram masking
    """
    def __init__(
            self,
            name: str,
            adapter: nn.Module,
            seed: Optional[int] = 0,
            complex: Optional[bool] = False,
            **kwargs
    ):
        super().__init__(name=name, adapter=adapter, seed=seed, complex=complex)
        self.register_buffer("target", None)

    def load_target(self, target: torch.Tensor) -> None:
        self.target = target


class FundamentalMasking(FeatureMasker):
    """
    Fundamental note frequency masker
    """
    def __init__(
            self,
            name: str,
            adapter: nn.Module,
            seed: Optional[int] = 0,
            complex: Optional[bool] = False,
            **kwargs
    ):
        super().__init__(name=name, adapter=adapter, seed=seed, complex=complex)

    def permute(self, x: torch.Tensor) -> torch.Tensor:
        K_spec, T, N = x.shape
        K_roll, _, _ = self.target.shape

        target = self.target.to(device=x.device, dtype=x.dtype)

        # Match target to x
        pad = K_spec - K_roll
        assert pad >= 0

        pad = torch.zeros(pad, T, N, device=target.device, dtype=target.dtype)
        target = torch.cat([target, pad], dim=0)
        
        # Apply mask
        return x * (1.0 - target)


class HarmonicMasking(FeatureMasker):
    """
    Harmonic frequency masker
    """
    def __init__(
            self,
            name: str,
            adapter: nn.Module,
            seed: Optional[int] = 0,
            complex: Optional[bool] = False,
            **kwargs
    ):
        super().__init__(name=name, adapter=adapter, seed=seed, complex=complex)

    def permute(self, x: torch.Tensor) -> torch.Tensor:
        K_spec, T, N = x.shape
        K_roll, _, _ = self.target.shape

        target = self.target.to(device=x.device, dtype=x.dtype)

        # Match target to x
        pad = K_spec - K_roll
        assert pad >= 0

        pad = torch.zeros(pad, T, N, device=target.device, dtype=target.dtype)
        target = torch.cat([target, pad], dim=0)
        
        # Apply mask
        return x * target
    

class SoftHarmonicMasking(FeatureMasker):
    """
    Soft harmonic frequency masker
    """
    def __init__(
            self,
            name: str,
            a: float,
            adapter: nn.Module,
            seed: Optional[int] = 0,
            complex: Optional[bool] = False,
            **kwargs
    ):
        super().__init__(name, adapter, seed, complex=complex)
        self.a = a
        
    def permute(self, x: torch.Tensor) -> torch.Tensor:
        K_spec, T, N = x.shape
        K_roll, _, _ = self.target.shape

        target = self.target.to(device=x.device, dtype=x.dtype)

        # Match target to x
        pad = K_spec - K_roll
        assert pad >= 0

        pad = torch.zeros(pad, T, N, device=target.device, dtype=target.dtype)
        target = torch.cat([target, pad], dim=0)
        
        # Create softmask
        target = self.a + (1 - self.a) * target
        
        # Apply mask
        return x * target


class SoftFundamentalMasking(FeatureMasker):
    """
    Soft harmonic fundamental masker
    """
    def __init__(
            self,
            name: str,
            a: float,
            adapter: nn.Module,
            seed: Optional[int] = 0,
            complex: Optional[bool] = False,
            **kwargs
    ):
        super().__init__(name=name, adapter=adapter, seed=seed, complex=complex)
        self.a = a

    def permute(self, x: torch.Tensor) -> torch.Tensor:
        K_spec, T, N = x.shape
        K_roll, _, _ = self.target.shape

        target = self.target.to(device=x.device, dtype=x.dtype)

        # Match target to x
        pad = K_spec - K_roll
        assert pad >= 0

        pad = torch.zeros(pad, T, N, device=target.device, dtype=target.dtype)
        target = torch.cat([target, pad], dim=0)
        
        # Create softmask
        target = 1.0 - (1.0 - self.a) * target

        # Apply mask
        return x * target