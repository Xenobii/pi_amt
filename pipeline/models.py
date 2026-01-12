import sys
import pretty_midi
import numpy as np
from pathlib import Path
from typing import Tuple
from abc import abstractmethod

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
    def load(self):
        return NotImplementedError
    
    @abstractmethod
    def load_hook(self, hook):
        return NotImplementedError

    def clear_hooks(self):
        if not hasattr(self, "model") or self.model is None:
            print("Hook clearing failed, model has not been loaded yet")
            return
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