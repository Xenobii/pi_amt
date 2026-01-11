import os
import numpy as np
import pretty_midi
from glob import glob
from abc import abstractmethod
from typing import Tuple, Dict, List

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.name = name

    @abstractmethod
    def __len__(self):
        return NotImplementedError
    
    @abstractmethod
    def __getitem__(self, idx):
        return NotImplementedError
    
    def create_eval_data(self, midi_path: str) -> Tuple[np.array, np.array]:
        """
        Returns the intervals and pitches used for mir_eval transcription
        """
        mid = pretty_midi.PrettyMIDI(midi_path)

        intervals = []
        pitches = []

        for i in mid.instruments:
            if i.is_drum:
                continue
            for note in i.notes:
                intervals.append([note.start, note.end])
                pitches.append(note.pitch)
        
        return np.array(intervals), np.array(pitches)
        
    

class MAPS(MyDataset):
    def __init__(self, name, data_folder):
        super().__init__(name)
        self.files = [
            y.replace(".wav", "")
            for x in os.walk(data_folder)
            for y in glob(os.path.join(x[0], "*.wav"))
        ]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx) -> Dict[str, str]:
        wav_file = os.path.join(self.files[idx] + ".wav")
        mid_file = os.path.join(self.files[idx] + ".mid")

        return {
            "wav_file": wav_file,
            "mid_file": mid_file,
        }

