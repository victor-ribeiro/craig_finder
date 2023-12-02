"""
    Módulo que implementa as operações realizadas sobre os dados
"""
from dataclasses import dataclass
from typing import TypeVar, Sequence
from dotmap import DotMap
from pandas import DataFrame
from torch.utils.data import Dataset
from torch import Tensor

import numpy as np

_T = TypeVar("_T", Sequence, int)


@dataclass
class DataBatch:
    # testar essa implementação depois
    features: Tensor
    labels: Tensor


class PandasDataset(Dataset):
    def __init__(self, dataframe: DataFrame, target: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.targets = dataframe[target]
        self.dataframe = dataframe.drop(target, axis="columns")
        self.indices = np.arange(0, len(dataframe))

    def __getitem__(self, index: _T):
        features = self.dataframe.iloc[index]
        labels = self.targets.iloc[index]
        features, labels = Tensor(features.values).float(), Tensor([labels]).float()
        out = {"features": features, "targets": labels}
        out = DotMap(out)
        return out

    def __len__(self):
        return len(self.dataframe)
