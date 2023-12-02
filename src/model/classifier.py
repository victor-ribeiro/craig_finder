"""
    Implementa os modelos e as operações de treinamento e avaliação
"""
import torch
from torch import nn
from torch.utils.data import Dataset

from DeepCore.deepcore.nets.nets_utils import recorder
from .eval_model import eval_model


def linear_block_factory(in_features: int, out_features: int, device: str = "cpu"):
    layer_set = [
        nn.Linear(
            in_features=in_features, out_features=out_features, device=device, bias=True
        ),
        nn.ReLU(),
    ]
    return nn.Sequential(*layer_set)


class EmployeeClassifier(nn.Module):
    device = "gpu" if torch.cuda.is_available() else "cpu"

    def __init__(self, in_features, out_features, dropout=0.5):
        super().__init__()
        self.input_layer = linear_block_factory(
            in_features=in_features, out_features=128
        )
        self.hidden = linear_block_factory(in_features=128, out_features=128)
        self.hidden1 = linear_block_factory(in_features=128, out_features=64)
        self.hidden2 = linear_block_factory(in_features=64, out_features=out_features)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.embedding_recorder = recorder.EmbeddingRecorder(record_embedding=True)

    def forward(self, x):
        x = self.dropout(self.input_layer(x))
        x = self.hidden(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.embedding_recorder(x)
        x = self.sigmoid(x)
        return x

    def get_last_layer(self):
        return self.hidden[0]


def train_torch_net(
    model: nn.Module,
    optmizer: nn.Module,
    loss_fn: nn.Module,
    data_train: Dataset,
    epochs: int,
    data_val: Dataset = None,
    data_test: Dataset = None,
):
    model.to("cpu")
    model.train(True)
    train_loss = []
    test_loss = []
    val_loss = []
    hist = dict()
    for _ in range(epochs):
        epoch_loss = 0
        for batch in data_train:
            features, labels = batch.items()
            labels = labels[1].float().to("cpu")
            features = (
                torch.autograd.Variable(features[1], requires_grad=True)
                .float()
                .to("cpu")
            )
            pred = model(features)
            loss = loss_fn(pred, labels)
            epoch_loss += loss.item()
            optmizer.zero_grad()
            loss.backward()
            optmizer.step()
        epoch_loss /= len(data_train)
        train_loss.append(epoch_loss)
        if data_test:
            t_loss = eval_model(model, data_test, l_func=loss_fn)
            test_loss.append(t_loss)
        if data_val:
            t_loss = eval_model(model, data_val, l_func=loss_fn)
            val_loss.append(t_loss)
        model.train(True)
    model.eval()
    hist["train_loss"] = train_loss
    if data_test:
        hist["test_loss"] = test_loss
    if data_val:
        hist["val_loss"] = val_loss
    return hist
