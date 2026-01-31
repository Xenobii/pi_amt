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
        sr                   = self.sr
        fft_hop              = self.fft_hop
        n_frames_chunk       = self.n_frames_chunk
        bins_per_semitone    = self.bins_per_semitone
        n_overlapping_frames = self.n_overlapping_frames
        
        fs              = sr / fft_hop
        audio_n_samples = sr * 2 - fft_hop
        overlap_len     = n_overlapping_frames * fft_hop
        hop_size        = audio_n_samples - overlap_len

        # We have to cook bullshit algorithms instead of nice parallelized torch
        # because basic pitch uses a float step of 141.265625
        
        midi = pretty_midi.PrettyMIDI(f_midi)
        
        # Total chunks
        audio_duration = midi.get_end_time()
        total_audio_samples = int(audio_duration * sr)
        total_audio_samples_padded = total_audio_samples + overlap_len // 2
        
        n_chunks = int(np.ceil((total_audio_samples_padded - audio_n_samples) / hop_size)) + 1
        
        # Make roll
        roll = midi.get_piano_roll(fs=fs)
        roll = (roll > 0).astype(np.float32)
        roll = torch.from_numpy(roll)
        
        initial_pad_frames = overlap_len // 2 // fft_hop
        roll = torch.nn.functional.pad(roll, (initial_pad_frames, 0, 0, 0))
        
        # Edit F to match basic pitch (match centered bins)
        roll = roll[21:125, :]
        roll = roll.repeat_interleave(bins_per_semitone, dim=0)
        roll = roll[1:-2, :]
        
        # Match chunks
        chunks = []
        for i in range(n_chunks):
            start_sample = i * hop_size
            start_frame = int(round(start_sample / fft_hop))
            end_frame = start_frame + n_frames_chunk
            
            if end_frame <= roll.shape[1]:
                chunk = roll[:, start_frame:end_frame]
            else:
                chunk = torch.nn.functional.pad(
                    roll[:, start_frame:], 
                    (0, end_frame - roll.shape[1], 0, 0)
                )
            chunks.append(chunk)
        
        roll = torch.stack(chunks, dim=0)
        roll = roll.permute(0, 2, 1)
        
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

    def create_midi_target_(self, f_midi: str) -> torch.Tensor:
        sr = 22050
        secs_per_block = 3
        n_octaves = 9
        bins_per_octave = 60

        block_length = int(sr * secs_per_block)
        n_bins = n_octaves * bins_per_octave
        max_window_length = 1024
        hop_length_samples = block_length / max_window_length

        chunk_hop_samples = block_length // 2
        n_frames_chunk = max_window_length

        fmin_hz = (sr / 2) / (2 ** n_octaves)
        fmin_midi = librosa.hz_to_midi(fmin_hz)
        midi_freqs = fmin_midi + np.arange(n_bins) / (bins_per_octave / 12)

        midi = pretty_midi.PrettyMIDI(f_midi)
        
        audio_duration = midi.get_end_time()
        total_audio_samples = int(audio_duration * sr)
        total_audio_samples_padded = total_audio_samples + 2 * chunk_hop_samples
        
        n_chunks = (total_audio_samples_padded - chunk_hop_samples) // chunk_hop_samples
        
        fs = sr / hop_length_samples
        roll = midi.get_piano_roll(fs=fs)
        roll = (roll > 0).astype(np.float32)
        roll = torch.from_numpy(roll)
        
        initial_pad_frames = int(chunk_hop_samples / hop_length_samples)
        roll = torch.nn.functional.pad(roll, (initial_pad_frames, 0, 0, 0))
        
        roll_matched = torch.zeros((n_bins, roll.shape[1]))
        for i, midi_freq in enumerate(midi_freqs):
            closest_midi = int(np.round(midi_freq))
            if 0 <= closest_midi < roll.shape[0]:
                roll_matched[i, :] = roll[closest_midi, :]
        
        roll = roll_matched
        
        chunks = []
        for i in range(n_chunks):
            sample_start = i * chunk_hop_samples
            frame_start = int(sample_start / hop_length_samples)
            frame_stop = frame_start + n_frames_chunk
            
            if frame_stop <= roll.shape[1]:
                chunk = roll[:, frame_start:frame_stop]
            else:
                chunk = torch.nn.functional.pad(
                    roll[:, frame_start:],
                    (0, frame_stop - roll.shape[1], 0, 0)
                )
            
            if chunk.shape[1] < n_frames_chunk:
                chunk = torch.nn.functional.pad(
                    chunk,
                    (0, n_frames_chunk - chunk.shape[1], 0, 0)
                )
            elif chunk.shape[1] > n_frames_chunk:
                chunk = chunk[:, :n_frames_chunk]
            
            chunks.append(chunk)
        
        roll = torch.stack(chunks, dim=0)
        roll = roll.permute(0, 2, 1)
        return roll
    
    def create_midi_target(self, f_midi: str) -> torch.Tensor:
        sr                = self.sr
        secs_per_block    = self.secs_per_block
        n_octaves         = self.n_octaves
        bins_per_octave   = self.bins_per_octave
        max_window_length = self.max_window_length

        block_length       = int(sr * secs_per_block)
        n_bins             = n_octaves * bins_per_octave
        hop_length_samples = block_length / max_window_length

        fmin_hz = (sr / 2) / (2 ** n_octaves)
        fmin_midi = librosa.hz_to_midi(fmin_hz)
        midi_freqs = fmin_midi + np.arange(n_bins) / (bins_per_octave / 12)

        midi = pretty_midi.PrettyMIDI(f_midi)
        
        fs = sr / hop_length_samples
        roll = midi.get_piano_roll(fs=fs)
        roll = (roll > 0).astype(np.float32)
        roll = torch.from_numpy(roll)
        
        roll_matched = torch.zeros((n_bins, roll.shape[1]))
        for i, midi_freq in enumerate(midi_freqs):
            closest_midi = int(np.round(midi_freq))
            if 0 <= closest_midi < roll.shape[0]:
                roll_matched[i, :] = roll[closest_midi, :]
        
        roll = roll_matched.unsqueeze(0)  # [1, F, T]
        
        return roll.permute(0, 2, 1)

    def inference(self, wav_path: str) -> torch.Tensor:
        # Load audio
        wave = self._load_audio(wav_path)

        # Transcribe
        # activations = self.model.transcribe(wave)
        coeff = self.model.inference(wave, True)
        activations = self.model.to_activations(coeff)

        """
        Output: {[B, P, T]}
        """
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