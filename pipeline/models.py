import sys
import pretty_midi
import numpy as np
from pathlib import Path
from typing import Tuple
from abc import abstractmethod
import math

import torch
from torch import nn

# Basic Pitch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "basic_pitch"))
from models.basic_pitch.basic_pitch_torch.model import BasicPitchTorch
from models.basic_pitch.basic_pitch_torch import note_creation as infer
from models.basic_pitch.basic_pitch_torch.inference import unwrap_output, get_audio_input, predict
from models.basic_pitch.basic_pitch_torch.constants import AUDIO_N_SAMPLES, FFT_HOP

pretty_midi.pretty_midi.MAX_TICK = 1e10

# Onsets and Frames
# from models.of_alt.model.model import UnetTranscriptionModel


class BaseModel():
    def __init__(self, name: str, weight_path: str):
        self.name = name
        self.weight_path = weight_path

    @abstractmethod
    def load(self) -> None:
        return NotImplementedError
    
    @abstractmethod
    def load_hook(self, hook) -> None:
        return NotImplementedError

    @abstractmethod
    def create_midi_target(self, f_midi: str) -> torch.Tensor:
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
    
    @abstractmethod
    def predict(self, wav_path: str):
        return NotImplementedError
    
    @abstractmethod
    def prepare_for_eval(self, model_output):
        return NotImplementedError
    
    @abstractmethod
    def write_midi(self, model_output):
        return NotImplementedError


class BasicPitch(BaseModel):
    def __init__(self, name, weight_path, **kwargs):
        super().__init__(name, weight_path)

        self.inference_settings = kwargs.get("inference_settings")

    def load(self):
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

    def inference(self, wav_path: str):
        n_overlapping_frames = 30
        overlap_len = n_overlapping_frames * FFT_HOP
        hop_size = AUDIO_N_SAMPLES - overlap_len

        # Window audio
        audio_windowed, _, audio_original_length = get_audio_input(wav_path, overlap_len, hop_size)
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
            k: unwrap_output(output[k], audio_original_length, n_overlapping_frames)
            for k in output
        }
        return unwrapped_output
    
    def predict(self, wav_path: str):
        # Run inference
        model_output = self.inference(wav_path)

        # Convert to midi
        midi_data, _ = infer.model_output_to_notes(
            model_output,
            **self.inference_settings
        )
        return midi_data
    
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
    
    def write_midi(self, midi_data: pretty_midi.PrettyMIDI, file_path: str):
        midi_data.write(file_path)


class OnsetsAndFrames(BaseModel):
    def __init__(self, name, weight_path, **kwargs):
        super().__init__()

    # def load(self):
        # self.model = UnetTranscriptionModel(
        #     ds_ksize=(2, 2),
        #     ds_stride=(2, 2),
        #     mode='imagewise',
        # )