import os
import numpy as np
from glob import glob
from abc import abstractmethod
from typing import Tuple, Dict

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.name = name

    @abstractmethod
    def __len__(self):
        return NotImplementedError
    
    @abstractmethod
    def __getitem__(self, index):
        return NotImplementedError
    

class MAPS(MyDataset):
    def __init__(self, name, data_folder):
        super().__init__(name)
        self.files = [
            y.replace('.wav', '')
            for x in os.walk(data_folder)
            for y in glob(os.path.join(x[0], '*.wav'))
        ]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx) -> Dict[str, str]:
        wav_file = os.path.join(self.files[idx] + ".wav")
        mid_file = os.path.join(self.files[idx] + ".mid")

        return {
            "wav_file": wav_file,
            "mid_file": mid_file
        }

