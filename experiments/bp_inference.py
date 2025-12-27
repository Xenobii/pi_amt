import sys
import pathlib
from typing import Union, Optional, Dict, List, Tuple

import torch
from torch import nn
import numpy as np
import pretty_midi

# Resolve the path because someone didn't acount for using basic_pitch as a submodule
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "models" / "basic_pitch"))

from models.basic_pitch.basic_pitch_torch import note_creation as infer
from models.basic_pitch.basic_pitch_torch.model import BasicPitchTorch
from models.basic_pitch.basic_pitch_torch.inference import unwrap_output, get_audio_input
from models.basic_pitch.basic_pitch_torch.constants import (
    AUDIO_SAMPLE_RATE,
    AUDIO_N_SAMPLES,
    FFT_HOP
)


def custom_inference(
        audio_path: Union[pathlib.Path, str],
        model: nn.Module,
) -> Dict[str, np.array]:
    # Run model on the input audio path
    n_overlapping_frames = 30
    overlap_len = n_overlapping_frames * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len

    audio_windowed, _, audio_original_length = get_audio_input(audio_path, overlap_len, hop_size)
    audio_windowed = torch.from_numpy(audio_windowed).T
    if torch.cuda.is_available():
        audio_windowed = audio_windowed.cuda()

    # Add hook here
    output = model(audio_windowed)

    """ 
    Output: {
        'onset'  : [B, T, P],
        'frame'  : [B, T, P],
        'contour': [B, T, 3*P]
        }
    """
    unwrapped_output = {k: unwrap_output(output[k], audio_original_length, n_overlapping_frames) for k in output}

    return unwrapped_output



def custom_predict(
        audio_path: Union[pathlib.Path, str],
        model_path: Union[pathlib.Path, str],
        onset_threshold: float = 0.5, 
        frame_threshold: float = 0.3,
        minimum_note_length: float = 127.70,
        minimum_frequency: Optional[float] = None,
        maximum_frequency: Optional[float] = None,
        multiple_pitch_bends: bool = False,
        melodia_trick: bool = True,
        midi_tempo: float = 120,
) -> Tuple[Dict[str, np.array], pretty_midi.PrettyMIDI, List[Tuple[float, float, int, float, Optional[List[int]]]],]:
    # Run a single prediction

    model = BasicPitchTorch()
    model.load_state_dict(torch.load(str(model_path)))
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    print(f"Predicting MIDI fo {audio_path}...")

    model_output = custom_inference(audio_path, model)
    min_note_len = int(np.round(minimum_note_length / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
    midi_data, note_events = infer.model_output_to_notes(
        model_output,
        onset_thresh         = onset_threshold,
        frame_thresh         = frame_threshold,
        min_note_len         = min_note_len,
        min_freq             = minimum_frequency,
        max_freq             = maximum_frequency,
        multiple_pitch_bends = multiple_pitch_bends,
        melodia_trick        = melodia_trick,
        midi_tempo           = midi_tempo,
    )

    return model_output, midi_data, note_events



if __name__ == "__main__":
    
    wav_path    = "test_data/test_audio.wav"
    model_path  = "models/basic_pitch/assets/basic_pitch_pytorch_icassp_2022.pth"
    output_path = "test_data/test_output.mid"
    
    model_out, midi_data, note_events = custom_predict(wav_path, model_path=str(model_path))
    midi_data.write(str(output_path))