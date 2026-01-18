from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi

import torch
import torch.nn as nn



"""Permutations"""

class Permutation(nn.Module):
    def __init__(self, name, seed: int = 0):
        """
        Base class for CQT frequency-axis permutations.
        """
        super().__init__()
        self.name = name
        self.seed = int(seed)
        self._gens = {}

    def _get_generator(self, device: torch.device) -> torch.Tensor:
        if device not in self._gens:
            gen = torch.Generator(device=device)
            gen.manual_seed(self.seed)
            self._gens[device] = gen
        return self._gens[device]
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, F]
        """
        return NotImplementedError


class NoPermutation(Permutation):
    def __init__(self, name: str, **kwargs):
        """
        Identity permutation
        """
        super().__init__(name)
    
    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x



class RandomPermutation(Permutation):
    def __init__(self, name: str, p: float=0.1, seed: int=0, **kwargs):
        """
        Random per-frame bin swapping.

        p: probability of swapping a given bin with another bin
        """
        super().__init__(name, seed)
        self.p = p

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, name: str, start: int, seed: int = 0):
        """
        Permutes only high-frequency bins (>= start percentage),
        independently per frame.
        """
        super().__init__(name, seed)
        assert 0.0 < start < 1.0
        self.start = start

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, name: str, bins_per_semitone: int, seed: int = 0):
        """
        Permutes bins within each semitone group.

        bins_per_semitone: number of CQT bins per semitone
        """
        super().__init__(name, seed)
        self.bins_per_semitone = bins_per_semitone

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
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

class FeatureMasker(nn.Module):
    def __init__(
            self,
            name: str,
            sr: int = 22050,
            n_overlapping_frames: int = 30,
            fft_hop: int = 256,
            audio_window_length: int = 2
    ):
        """
        Base class for spectrogram feature masking
        """
        super().__init__()
        self.name = name
        self.fft_hop = fft_hop
        self.fps = sr // fft_hop
        self.audio_n_samples = sr * audio_window_length - fft_hop
        self.n_overlapping_frames = n_overlapping_frames
        self.bins_per_octave = 3 * 12
        self.f_min = 27.7

    def load_label(self, midi_path: str):
        """
        Loads the MIDI as a binary piano roll and registers it as a buffer.
        Windowing is deferred to _align_label_to_input() where actual input dimensions are known.
        """
        midi_data = pretty_midi.PrettyMIDI(midi_path)

        duration = midi_data.get_end_time()

        piano_roll = midi_data.get_piano_roll(fs=self.fps)
        piano_roll = (piano_roll > 0).astype(np.float32)

        # make piano_roll match the audio duration in annotation frames
        n_frames_expected = int(np.ceil(duration * self.fps))
        if piano_roll.shape[1] < n_frames_expected:
            piano_roll = np.pad(piano_roll, ((0, 0), (0, n_frames_expected - piano_roll.shape[1])), mode="constant")
        elif piano_roll.shape[1] > n_frames_expected:
            piano_roll = piano_roll[:, :n_frames_expected]

        # Store raw piano roll - windowing happens in _align_label_to_input
        label_tensor = torch.from_numpy(piano_roll).float()  # [N, T_total]
        self.register_buffer("label_raw", label_tensor)

    def _align_label_to_input(self, B: int, T: int) -> torch.Tensor:
        """
        Window the raw piano roll to match input spectrogram dimensions.
        
        Args:
            B: Number of batches (windows) from the input spectrogram
            T: Number of time frames per window from the input spectrogram
            
        Returns:
            label: Tensor of shape [B, N, T] aligned with input
        """
        if not hasattr(self, "label_raw"):
            raise RuntimeError("No label_raw buffer found. Call load_label(...) first.")
        
        N, T_total = self.label_raw.shape
        
        # Calculate windowing parameters (same as basic_pitch inference)
        overlap_len_samples = self.n_overlapping_frames * self.fft_hop
        hop_size_samples = self.audio_n_samples - overlap_len_samples
        
        # Convert to annotation frames
        hop_size_frames = hop_size_samples // self.fft_hop
        overlap_frames = overlap_len_samples // self.fft_hop
        
        # Pad start to match basic_pitch's padding
        pad_start_frames = overlap_frames // 2
        piano_roll = torch.nn.functional.pad(self.label_raw, (pad_start_frames, 0))
        
        # Pad end if needed to accommodate all windows
        T_padded = piano_roll.shape[1]
        total_needed = (B - 1) * hop_size_frames + T
        if total_needed > T_padded:
            piano_roll = torch.nn.functional.pad(piano_roll, (0, total_needed - T_padded))
        
        # Extract windows to match input batches
        windows = []
        for i in range(B):
            start = i * hop_size_frames
            end = start + T
            windows.append(piano_roll[:, start:end])
        
        return torch.stack(windows, dim=0)  # [B, N, T]

    def note_to_bin(
        self,
        midi_notes: torch.Tensor,  # [N]
        F: int
    ) -> torch.Tensor:
        """
        Maps MIDI notes to CQT bin indices
        """
        freqs = 440.0 * (2.0 ** ((midi_notes - 69) / 12))
        bins = self.bins_per_octave * torch.log2(freqs / self.f_min)
        bins = bins.round().long()

        return bins.clamp(min=0, max=F - 1)

    @staticmethod
    def build_fundamental_mask(
        y: torch.Tensor,
        note_bins: torch.Tensor,
        F: int
    ) -> torch.Tensor:
        """
        Returns:
            fund_mask: BoolTensor [B, T, F]
        """
        B, N, T = y.shape
        device = y.device

        # [B, N, T] -> [B, T, N]
        y_bt_n = y.permute(0, 2, 1).bool()

        bins = note_bins.view(1, 1, N).expand(B, T, N)
        fund_mask = torch.zeros(B, T, F, device=device, dtype=torch.bool)

        fund_mask.scatter_(dim=2, index=bins, src=y_bt_n)

        return fund_mask

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def plot(
        self,
        x: torch.Tensor,
        batch: int = 0,
        figsize: Tuple[int, int] = (12, 5),
        cmap: str = "magma",
        label_cmap: str = "gray_r",
        show: bool = True,
    ):
        """
        Plot spectrogram x (B, T, F) and piano-roll label (B, N, T) for given batch index.
        Returns (fig, (ax_spec, ax_label)).
        """
        if x.dim() != 3:
            raise ValueError("x must be a 3D tensor with shape [B, T, F]")
        if not hasattr(self, "label_raw"):
            raise RuntimeError("No label_raw buffer found. Call load_label(...) before plotting.")

        B, T, F = x.shape
        # Align label to input dimensions for plotting
        label = self._align_label_to_input(B, T)  # [B, N, T]

        x_np = x[batch].detach().cpu().numpy()   # [T, F]
        label_np = label[batch].detach().cpu().numpy()  # [N, T]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        im1 = ax1.imshow(x_np.T, aspect="auto", origin="lower", cmap=cmap, interpolation="nearest")
        ax1.set_title("Spectrogram (freq bins x time frames)")
        ax1.set_xlabel("Time frames")
        ax1.set_ylabel("Frequency bins")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        im2 = ax2.imshow(label_np, aspect="auto", origin="lower", cmap=label_cmap, interpolation="nearest")
        ax2.set_title("Piano roll (MIDI pitch x time frames)")
        ax2.set_xlabel("Time frames")
        ax2.set_ylabel("MIDI pitch")
        fig.tight_layout()

        if show:
            plt.show()

        return fig, (ax1, ax2)


class FundamentalMasking(FeatureMasker):
    def __init__(
        self,
        spec_type: str = "cqt",
        bins_per_octave: int = 3 * 12,
        f_min: float = 27.7,
        f_max: Optional[float] = None,
    ):
        """
        Mask the fundamental frequencies of the active notes
        """
        super().__init__(spec_type, bins_per_octave, f_min, f_max)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        device = x.device

        # Align label to input dimensions
        label = self._align_label_to_input(B, T)  # [B, N, T]
        label = label.to(device)
        N = label.shape[1]

        midi = torch.arange(N, device=device)
        note_bins = self.note_to_bin(midi, F)

        fund_mask = self.build_fundamental_mask(label, note_bins, F)

        x_out = x.clone()
        x_out[fund_mask] = 0.0

        self.plot(x_out, 0)

        return x_out


class HarmonicMasking(FeatureMasker):
    def __init__(
        self,
        spec_type: str = "cqt",
        bins_per_octave: int = 3 * 12,
        f_min: float = 27.7,
        f_max: Optional[float] = None,
    ):
        """
        Keep only the fundamental frequencies of active notes
        """
        super().__init__(spec_type, bins_per_octave, f_min, f_max)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        device = x.device

        # Align label to input dimensions
        label = self._align_label_to_input(B, T)  # [B, N, T]
        label = label.to(device)
        N = label.shape[1]

        midi = torch.arange(N, device=device)
        note_bins = self.note_to_bin(midi, F)

        fund_mask = self.build_fundamental_mask(label, note_bins, F)

        x_out = torch.zeros_like(x)
        x_out[fund_mask] = x[fund_mask]

        return x_out


class SoftFundamentalMasking(FeatureMasker):
    def __init__(
        self,
        spec_type: str = "cqt",
        bins_per_octave: int = 3 * 12,
        f_min: float = 27.7,
        f_max: Optional[float] = None,
        temperature: float = 0.1,
        eps: float = 1e-8,
    ):
        """
        Softly emphasize fundamental frequencies using softmax attenuation
        """
        super().__init__(spec_type, bins_per_octave, f_min, f_max)
        self.temperature = temperature
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        device = x.device

        # Align label to input dimensions
        label = self._align_label_to_input(B, T)  # [B, N, T]
        label = label.to(device)
        N = label.shape[1]

        midi = torch.arange(N, device=device)
        note_bins = self.note_to_bin(midi, F)

        fund_mask = self.build_fundamental_mask(label, note_bins, F)

        # Build softmax scores
        scores = torch.full_like(x, float("-inf"))
        scores[fund_mask] = x[fund_mask] / self.temperature

        # Stable softmax over frequency
        weights = torch.softmax(scores, dim=-1)

        # Attenuate
        x_out = x * weights

        return x_out
