import sys
import pathlib
from typing import Union, Optional, Tuple, List, Dict

import torch
import torchaudio
import numpy as np

from models.of_alt.model.model import UnetTranscriptionModel


def custom_predict():
    model = UnetTranscriptionModel(
        ds_ksize=(2, 2),
        ds_stride=(2, 2),
        mode='imagewise',
    )

    weight_path = ""
    model.load_state_dict(torch.load(weight_path))

    model.eval()
    