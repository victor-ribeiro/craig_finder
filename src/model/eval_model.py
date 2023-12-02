from typing import Callable
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from pandas import DataFrame


def map_error(df: DataFrame, target_col: str, model: nn.Module):
    pred = predict(df, target_col, model)
    target = torch.Tensor(df[target_col].values).unsqueeze(1)
    error = F.binary_cross_entropy(pred, target, reduction="none").detach().numpy()
    return error


def predict(df: DataFrame, target_col: str, model: nn.Module):
    d = df.drop(target_col, axis="columns").values
    d = torch.Tensor(d)
    return model(d)


def eval_model(model: nn.Module, torch_dataset: Dataset, l_func: Callable):
    """Evaluate the torch module during the training loop

    Args:
        model (nn.Module): _description_
        torch_dataset (Dataset): _description_
        l_func (Callable): _description_

    Returns:
        _type_: _description_
    """
    model.eval()
    with torch.no_grad():
        batch_loss = 0
        for batch in torch_dataset:
            features, labels = batch.items()
            _, labels = labels
            _, features = features
            pred = model(features.to("cpu"))
            e_loss = l_func(pred, labels)
            batch_loss += e_loss.item() / len(torch_dataset)
    return batch_loss
