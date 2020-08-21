import os
from typing import Callable, Optional, Tuple
from warnings import warn

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import torch
from torch.utils.data import Dataset


class SurvivalDataset(Dataset):
    def __init__(self, alldata):
        [data, time, label], [mask1, mask2] = alldata
        self.data = data.astype(np.float32)
        self.time = time
        self.label = label
        self.mask1 = mask1.astype(int)
        self.mask2 = mask2.astype(int)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        """Get an input-target pair from the dataset.

        The images are assumed to be preprocessed and cached.

        Parameters
        ----------
        idx
            The index to retrieve (note: this is not the subject ID).

        Returns
        -------
        tuple of torch.Tensor and int
            The input-target pair.
        """
        return self.data[idx], self.label[idx], self.time[idx], self.mask1[idx], self.mask2[idx]

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)
