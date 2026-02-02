import sys
import pretty_midi
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from abc import abstractmethod
from scipy.ndimage import maximum_filter1d
import librosa

import torch
from torch import nn
import torchaudio


pretty_midi.pretty_midi.MAX_TICK = 1e10


class BaseModel():
    def __init__(
            self,
            name: str,
            weight_path: str,
            target_shape: str,
            complex: Optional[bool] = False
    ) -> None:
        self.name = name
        self.weight_path = weight_path
        self.target_shape = target_shape
        self.complex = complex

    @abstractmethod
    def load(self) -> None:
        return NotImplementedError
    
    @abstractmethod
    def load_hook(self, hook) -> None:
        return NotImplementedError

    @abstractmethod
    def create_midi_target(self, f_midi: str) -> torch.Tensor:
        return NotImplementedError

    @abstractmethod
    def predict(self, wav_path: str):
        return NotImplementedError
   
    def clear_hooks(self) -> None:
        if not hasattr(self, "model") or self.model is None:
            raise Exception("Hook clearing failed, model has not been loaded yet")
        for m in self.model.modules():
            for attr in ("_forward_pre_hooks", "_forward_hooks", "_backward_hooks",
                         "_forward_post_hooks", "_backward_pre_hooks"):
                if hasattr(m, attr):
                    try:
                        getattr(m, attr).clear()
                    except Exception:
                        try: 
                            setattr(m, attr, {})
                        except Exception:
                            pass
    
    def prepare_for_eval(self, midi_data: pretty_midi.PrettyMIDI) -> Tuple[np.array, np.array]:
        """
        Returns the intervals and pitches used for mir_eval transcription
        """
        intervals = []
        pitches = []

        for i in midi_data.instruments:
            if i.is_drum:
                continue
            for note in i.notes:
                intervals.append([note.start, note.end])
                pitches.append(note.pitch)
        
        return np.array(intervals), np.array(pitches)
    
    def write_midi(self, midi_data: pretty_midi.PrettyMIDI, file_path: str) -> None:
        midi_data.write(file_path)

    def _align_midi_F(
            self,
            roll: torch.Tensor,
            bins_per_semitone: int, 
            fmin: Optional[float] = 27.5,
            n_bins: int = None,
            center_bins: bool = True
    ) -> torch.Tensor:
        min_bin = int(librosa.hz_to_midi(fmin))

        roll = roll[min_bin:, :]
        roll = roll.repeat_interleave(bins_per_semitone, dim=0)
        
        if center_bins:
            shift = bins_per_semitone // 2
            diff = bins_per_semitone % 2
            roll = roll[shift:-(shift+diff), :]

        if n_bins < roll.shape[0]:
            roll = roll[:n_bins, :]
        elif n_bins > roll.shape[0]:
            roll = torch.nn.functional.pad(roll, (0, 0, 0, n_bins-roll.shape[0]))
            
        return roll

    def _align_midi_T(
            self,
            roll: torch.Tensor,
            sr: int,
            fft_hop: int,
            n_secs_chunk: float,
            n_frames_overlap: int,
            n_frames_offset: int = 0,
            lazy_chunking: bool = False,
    ) -> torch.Tensor:
        # Compute windowing parameters
        _, T = roll.shape
        
        n_samples_chunk   = sr * n_secs_chunk - fft_hop
        n_samples_overlap = n_frames_overlap * fft_hop
        n_samples_hop     = n_samples_chunk - 2 * n_samples_overlap

        n_frames_hop   = int(n_samples_hop / fft_hop)
        n_frames_chunk = int(n_secs_chunk * sr / fft_hop)
        n_chunks       = int(np.floor((T - n_frames_chunk) / n_frames_hop)) + 1
        
        # Pad 
        pad_len = n_chunks * n_frames_chunk - T
        roll = torch.nn.functional.pad(roll, (n_frames_offset, 0, 0, 0))
        roll = torch.nn.functional.pad(roll, (0, pad_len, 0, 0))
        roll = torch.nn.functional.pad(roll, (n_frames_overlap, 0, 0, 0))

        # Chunk roll
        if lazy_chunking:
            roll = roll.unfold(
                dimension=1,
                size=n_frames_chunk,
                step=n_frames_hop
            )
            roll = roll.permute(1, 2, 0).contiguous()
        else: 
            chunks = []
            for i in range(n_chunks):
                start_sample = i * n_samples_hop
                start_frame = int(round(start_sample / fft_hop))
                end_frame = start_frame + n_frames_chunk
                chunk = roll[:, start_frame: end_frame]
                chunks.append(chunk)
            
            roll = torch.stack(chunks, dim=0)
            roll = roll.permute(0, 2, 1)

        return roll


class BasicPitchWrapper(BaseModel):
    def __init__(self, name, weight_path, target_shape, **kwargs):
        super().__init__(name=name, weight_path=weight_path, target_shape=target_shape)

        self.sr                   = kwargs.get("sr", 22050)
        self.fft_hop              = kwargs.get("fft_hop", 256)
        self.n_overlapping_frames = kwargs.get("n_overlapping_frames", 30)
        self.n_frames_chunk       = kwargs.get("n_frames_chunk", 172)
        self.bins_per_semitone    = kwargs.get("bins_per_semitone", 3)
        
        self.inference_settings   = kwargs.get("inference_settings")

    def load(self):
        bp_path = Path(__file__).resolve().parents[1] / "models" / "basic_pitch"
        sys.path.insert(0, str(bp_path))

        try:
            from basic_pitch_torch.model import BasicPitchTorch
            from basic_pitch_torch import note_creation as infer
            from basic_pitch_torch.inference import unwrap_output, get_audio_input
        finally:
            try:
                sys.path.remove(str(bp_path))
            except ValueError:
                pass

        self.infer = infer
        self.unwrap_output = unwrap_output
        self.get_audio_input = get_audio_input

        self.model = BasicPitchTorch()
        self.model.load_state_dict(torch.load(str(self.weight_path), weights_only=True))
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

    def load_hook(self, hook: nn.Module):
        def pre_hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
            (x,) = inputs
            x_perm = hook(x)
            return (x_perm,)
        self.model.hs.register_forward_pre_hook(pre_hook)
    
    def create_midi_target(self, f_midi: str) -> torch.Tensor:
        sr                = self.sr
        fft_hop           = self.fft_hop
        bins_per_semitone = self.bins_per_semitone
        fs                = sr // fft_hop

        # Load MIDI 
        midi = pretty_midi.PrettyMIDI(f_midi)
        
        # Make roll
        roll = midi.get_piano_roll(fs=fs)
        roll = (roll > 0).astype(np.float32)
        roll = torch.from_numpy(roll)
        
        # Align F
        roll = self._align_midi_F(
            roll,
            bins_per_semitone=bins_per_semitone,
            fmin=27.5,
            n_bins=309,
            center_bins=True
        )
        
        # Align T + chunk
        roll = self._align_midi_T(
            roll,
            sr=sr,
            fft_hop=fft_hop,
            n_secs_chunk=2,
            n_frames_overlap=15,
            n_frames_offset=1,
            lazy_chunking=False
        )
        
        return roll

    def inference(self, wav_path: str) -> torch.Tensor:
        n_overlapping_frames = self.n_overlapping_frames
        overlap_len          = n_overlapping_frames * self.fft_hop
        audio_n_samples      = self.sr * 2 - self.fft_hop
        hop_size             = audio_n_samples - overlap_len

        # Window audio
        audio_windowed, _, audio_original_length = self.get_audio_input(wav_path, overlap_len, hop_size)
        audio_windowed = torch.from_numpy(audio_windowed.copy()).T
        if torch.cuda.is_available():
            audio_windowed = audio_windowed.cuda()

        # Infer
        output = self.model(audio_windowed)
        """ 
        Output: {
            'onset'  : [B, T, P],
            'frame'  : [B, T, P],
            'contour': [B, T, 3*P]
        }
        """
        
        # Unwrap audio
        unwrapped_output = {
            k: self.unwrap_output(output[k], audio_original_length, n_overlapping_frames)
            for k in output
        }
        return unwrapped_output
    
    def predict(self, wav_path: str) -> pretty_midi.PrettyMIDI:
        # Run inference
        model_output = self.inference(wav_path)

        # Convert to midi
        midi_data, _ = self.infer.model_output_to_notes(
            model_output,
            **self.inference_settings
        )
        return midi_data
    

class TimbreTrapWrapper(BaseModel):
    def __init__(self, name, weight_path, target_shape, complex, **kwargs):
        super().__init__(name=name, weight_path=weight_path, target_shape=target_shape, complex=complex)

        self.sr                = kwargs.get("sr", 22050)
        self.n_octaves         = kwargs.get("n_octaves", 9)
        self.bins_per_octave   = kwargs.get("bins_per_octave", 60)
        self.sample_rate       = kwargs.get("sample_rate", 22050)
        self.secs_per_block    = kwargs.get("secs_per_block", 3)
        self.max_window_length = kwargs.get("max_window_length", 1024)

        self.inference_settings = kwargs.get("inference_settings")

    def load(self):
        tt_path = Path(__file__).resolve().parents[1] / "models" / "timbre_trap"
        sys.path.insert(0, str(tt_path))

        try:
            from timbre_trap.framework.modules import TimbreTrap
        finally:
            try:
                sys.path.remove(str(tt_path))
            except ValueError:
                pass
            
        self.model = TimbreTrap(
            sample_rate=22050,
            n_octaves=9,
            bins_per_octave=60,
            secs_per_block=3,
            latent_size=128,
            model_complexity=2,
            skip_connections=False
        )
        self.model.load_state_dict(torch.load(str(self.weight_path), weights_only=True))
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

    def load_hook(self, hook: nn.Module):
        def pre_hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
            (x,) = inputs
            x_perm = hook(x)
            return (x_perm,)
        self.model.encoder.register_forward_pre_hook(pre_hook)

    def create_midi_target(self, f_midi: str) -> torch.Tensor:
        sr                = self.sr
        bins_per_semitone = int(self.bins_per_octave / 12)
        fft_hop           = 64.59
        fs                = sr // fft_hop

        # Load MIDI 
        midi = pretty_midi.PrettyMIDI(f_midi)
        
        # Make roll
        roll = midi.get_piano_roll(fs=fs)
        roll = (roll > 0).astype(np.float32)
        roll = torch.from_numpy(roll)
        
        # Align F
        roll = self._align_midi_F(
            roll,
            bins_per_semitone=bins_per_semitone,
            fmin=22,
            n_bins=540,
            center_bins=False
        )
        # Align T + chunk
        roll = self._align_midi_T(
            roll,
            sr=sr,
            fft_hop=fft_hop,
            n_secs_chunk=3,
            n_frames_overlap=256,
            n_frames_offset=256,
            lazy_chunking=False
        )

        return roll
    
    def inference(self, wav_path: str) -> torch.Tensor:
        wave = self._load_audio(wav_path).to(next(self.model.parameters()).device)
        device = wave.device

        block_length   = self.model.sliCQ.block_length
        n_frames_chunk = self.model.sliCQ.max_window_length
        hop_frames     = n_frames_chunk // 2

        hop_length = block_length // 2
        audio = torch.nn.functional.pad(wave, [hop_length] * 2)
        n_chunks = (audio.size(-1) - hop_length) // hop_length

        # build batched chunks
        chunks = [audio[..., i * hop_length : i * hop_length + block_length] for i in range(n_chunks)]
        audio_batch = torch.cat(chunks, dim=0)

        coeff_batch = self.model._inference(audio_batch, transcribe=True)  
        n_frames = self.model.sliCQ.get_expected_frames(audio.size(-1))

        window = torch.signal.windows.hann(n_frames_chunk, device=device).view(1, 1, 1, -1)
        coeff_win = coeff_batch * window  

        starts = (torch.arange(coeff_win.shape[0], device=device) * hop_frames).unsqueeze(1) 
        idxs = torch.arange(n_frames_chunk, device=device).unsqueeze(0)  
        positions = starts + idxs  

        index = positions.view(coeff_win.shape[0], 1, 1, n_frames_chunk).expand(-1, coeff_win.shape[1], coeff_win.shape[2], -1)

        chunks_acc = torch.zeros((coeff_win.shape[0], coeff_win.shape[1], coeff_win.shape[2], n_frames), device=device)
        chunks_acc.scatter_add_(-1, index, coeff_win)

        coefficients = chunks_acc.sum(dim=0, keepdim=True) 

        coefficients = coefficients[..., n_frames_chunk // 2 : -n_frames_chunk // 2]

        activations = self.model.to_activations(coefficients)
        return activations

    def predict(self, wav_path: str):
        # Run inference
        activations = self.inference(wav_path)
        activations = activations.squeeze(0).cpu().numpy()

        n_frames = activations.shape[-1]
        times = self.model.sliCQ.get_times(n_frames)
        midi_freqs = self.model.sliCQ.midi_freqs

        activations = self._filter_non_peaks(activations)
        activations = self._threshold(activations, self.inference_settings.get("frame_threshold"))

        midi_data = self._activations_to_midi(activations, times, midi_freqs)
        
        return midi_data


    def _load_audio(self, wav_path: str) -> torch.Tensor:
        wave, sr = torchaudio.load(wav_path)
        
        wave = torchaudio.functional.resample(wave, sr, 22050)
        
        wave = torch.mean(wave, axis=0, keepdim=True)
        
        if torch.cuda.is_available():
            wave = wave.cuda()

        wave /= wave.abs().max()

        wave = wave.unsqueeze(0)

        return wave

    def _filter_non_peaks(self, activations: np.ndarray) -> np.ndarray:
        max_filtered = maximum_filter1d(activations, size=3, axis=0, mode='constant')
        peaks = (activations == max_filtered) & (activations > 0)

        return activations * peaks
    
    def _threshold(self, activations: np.ndarray, t: float) -> np.ndarray:
        return (activations > t).astype(np.float32)

    def _activations_to_midi(
        self,
        activations: np.ndarray,
        times: np.ndarray, 
        midi_freqs: np.ndarray
    ) -> pretty_midi.PrettyMIDI:
        midi_data = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        
        min_note_length = self.inference_settings.get("min_note_length", 0.05)
        
        n_bins, n_frames = activations.shape
        
        for bin_idx in range(n_bins):
            pitch = int(round(midi_freqs[bin_idx]))
            
            if pitch < 0 or pitch > 127:
                continue
            
            active = activations[bin_idx, :] > 0
            
            onset_frames = np.where(np.diff(active.astype(int)) == 1)[0] + 1
            offset_frames = np.where(np.diff(active.astype(int)) == -1)[0] + 1
            
            if active[0]:
                onset_frames = np.concatenate([[0], onset_frames])
            if active[-1]:
                offset_frames = np.concatenate([offset_frames, [n_frames - 1]])
            
            for onset_idx, offset_idx in zip(onset_frames, offset_frames):
                onset_time = times[onset_idx]
                offset_time = times[min(offset_idx, n_frames - 1)]
                
                if offset_time - onset_time < min_note_length:
                    continue
                
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=pitch,
                    start=onset_time,
                    end=offset_time
                )
                instrument.notes.append(note)
        
        midi_data.instruments.append(instrument)
        return midi_data