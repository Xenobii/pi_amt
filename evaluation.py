import sys
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


def eval_model(
    model,
    dataset,
    permutation = None,
):
    return NotImplementedError


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def test_inference(cfg: DictConfig):
    wav_path = "test_data/maps_1.wav"
    midi_path = "test_data/maps_1.mid"
    
    # Load model
    model_cfg = cfg.models[0]
    model = instantiate(model_cfg)

    # Load permuter
    permutation_cfg = cfg.permutations[0]
    permutation = instantiate(permutation_cfg, bins_per_semitone=3)

    model.load()
    model.load_hook(permutation)
    model.predict(wav_path)



if __name__ == "__main__":
    test_inference()