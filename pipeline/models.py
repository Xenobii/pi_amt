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
        super().__init__(
            name=name, 
            weight_path=weight_path,
            target_shape=target_shape
        )

        self.inference_settings = kwargs.get("inference_settings")
        self.modules = kwargs.get("modules")

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
        AUDIO_SAMPLE_RATE = 22050
        FFT_HOP = 256
        AUDIO_N_SAMPLES = 22050 * 2 - 256
        n_overlapping_frames = 30
        overlap_len = n_overlapping_frames * FFT_HOP
        hop_size = AUDIO_N_SAMPLES - overlap_len
        
        n_frames_chunk = 172
        bins_per_semitone = 3
        fs = AUDIO_SAMPLE_RATE / FFT_HOP

        # We have to cook bullshit algorithms instead of nice parallelized torch
        # because basic pitch uses a float step of 141.265625
        
        midi = pretty_midi.PrettyMIDI(f_midi)
        
        # Total chunks
        audio_duration = midi.get_end_time()
        total_audio_samples = int(audio_duration * AUDIO_SAMPLE_RATE)
        total_audio_samples_padded = total_audio_samples + overlap_len // 2
        
        n_chunks = int(np.ceil((total_audio_samples_padded - AUDIO_N_SAMPLES) / hop_size)) + 1
        
        # Make roll
        roll = midi.get_piano_roll(fs=fs)
        roll = (roll > 0).astype(np.float32)
        roll = torch.from_numpy(roll)
        
        initial_pad_frames = overlap_len // 2 // FFT_HOP
        roll = torch.nn.functional.pad(roll, (initial_pad_frames, 0, 0, 0))
        
        # Edit F to match basic pitch (match centered bins)
        roll = roll[21:125, :]
        roll = roll.repeat_interleave(bins_per_semitone, dim=0)
        roll = roll[1:-2, :]
        
        # Match chunks
        chunks = []
        for i in range(n_chunks):
            start_sample = i * hop_size
            start_frame = int(round(start_sample / FFT_HOP))
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
        n_overlapping_frames = 30
        overlap_len = n_overlapping_frames * 256
        audio_n_samples = 22050 * 2 - 256
        hop_size = audio_n_samples - overlap_len

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
        super().__init__(
            name=name,
            weight_path=weight_path,
            target_shape=target_shape,
            complex=complex
        )

        self.inference_settings = kwargs.get("inference_settings")
        self.n_octaves = 9
        self.bins_per_octave = 60
        self.sample_rate = 22050 
        self.secs_per_block = 3 

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
        midi = pretty_midi.PrettyMIDI(f_midi)

        n_bins = self.n_octaves * self.bins_per_octave
        fmin = librosa.hz_to_midi((self.sample_rate / 2) / (2 ** self.n_octaves))
        midi_freqs = fmin + np.arange(n_bins) / (self.bins_per_octave / 12)
        
        audio_duration = midi.get_end_time()
        num_samples = int(audio_duration * self.sample_rate)
        
        block_length = int(self.secs_per_block * self.sample_rate)
        block_length = 2 ** int(np.ceil(np.log2(block_length)))
        max_window_length = block_length // 2
        hop_length = block_length / max_window_length
        
        n_frames = int(np.ceil((num_samples / block_length) * max_window_length))
        
        times = np.arange(n_frames) * hop_length / self.sample_rate
        
        multi_pitch = [np.empty(0)] * n_frames
        
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                pitch_hz = librosa.midi_to_hz(note.pitch)
                onset, offset = note.start, note.end
                
                for i in np.where((times >= onset) & (times < offset))[0]:
                    multi_pitch[i] = np.append(multi_pitch[i], pitch_hz)
        
        activations = np.zeros((n_bins, n_frames), dtype=np.float32)
        
        for frame_idx, pitches in enumerate(multi_pitch):
            for pitch_hz in pitches:
                pitch_midi = librosa.hz_to_midi(pitch_hz)
                if pitch_midi >= midi_freqs[0] and pitch_midi <= midi_freqs[-1]:
                    bin_idx = np.argmin(np.abs(midi_freqs - pitch_midi))
                    activations[bin_idx, frame_idx] = 1.0
        
        return torch.from_numpy(activations).unsqueeze(0)

    def inference(self, wav_path: str) -> torch.Tensor:
        # Load audio
        wave = self._load_audio(wav_path)

        # Transcribe
        activations = self.model.transcribe(wave)
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