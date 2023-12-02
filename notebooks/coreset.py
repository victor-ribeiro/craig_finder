import sys

sys.path.append("..")
sys.path.append("../src")


from pathlib import Path
import pandas as pd
import numpy as np
import random
import torch
from torch.optim import Adam
from torch import nn
from torch.utils.data import DataLoader, Subset
from dotmap import DotMap
from sklearn.model_selection import train_test_split

from DeepCore.deepcore.methods.craig import Craig
import json
from sklearn.preprocessing import StandardScaler

import warnings
import tqdm


from data.dataset import PandasDataset as ClassifierDataset
from model.classifier import EmployeeClassifier as Classifier

warnings.filterwarnings("ignore")

data_path = Path("../data/employee/Employee.csv")
RANDON_SEED = 42


def init_weights(model):
    name = model.__class__.__name__
    if name == "Linear":
        nn.init.xavier_normal(model.weight)
        model.bias.data.fill_(10e-2)


def eval_model(model, ds_loader, loss_fn, device="cpu"):
    eval_loss = 0.0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for sset in ds_loader:
            attr, targets = sset.values()
            attr = torch.Tensor(attr).to(device)
            targets = torch.Tensor(targets).to(device)
            pred = model(attr)
            loss = loss_fn(pred, targets)
            eval_loss += loss.item() / len(ds_loader)
    return eval_loss


def train(
    model,
    loss_fn,
    optmizer,
    train_loader,
    val_loader,
    n_epochs,
    test_loader=None,
    device="cpu",
    init=False,
):
    # inicializa os pesos do modelo
    if init:
        model.apply(init_weights)

    model.to(device)
    model.train(True)
    train_loss = []
    val_loss = []
    test_loss = []
    # treinamento epocas
    for _ in tqdm.trange(n_epochs):
        loss_t = 0
        # loop de treinamento
        for sset in train_loader:
            attr, targets = sset.values()
            attr = torch.autograd.Variable(attr, requires_grad=True).float().to(device)
            targets = torch.Tensor(targets).to(device)

            pred = model(attr)
            loss = loss_fn(pred, targets)
            loss_t += loss.item() / len(train_loader)

            optmizer.zero_grad()
            loss.backward()
            optmizer.step()
        train_loss.append(loss_t)
        loss_v = eval_model(model=model, ds_loader=val_loader, loss_fn=loss_fn)
        val_loss.append(loss_v)
        if test_loader:
            loss_t = eval_model(model=model, ds_loader=test_loader, loss_fn=loss_fn)
            test_loss.append(loss_t)
        hist = {"val_loss": val_loss, "train_loss": train_loss}
        if test_loader:
            hist["test_loss"] = test_loss
        model.train(True)
    return hist


def preprocess(df, target_col, scaler):
    tgt = df[target_col].values
    transformed = df.drop(target_col, axis="columns")
    names = transformed.columns.values
    transformed = scaler.fit_transform(transformed.values)
    transformed = pd.DataFrame(data=transformed, columns=names)
    transformed[target_col] = tgt
    return transformed


def parse_slices(slice_finder):
    idx = [np.where(i) for i in slice_finder.top_slices_]
    names = [slice_finder.feature_names_in_[i] for i in idx]
    attr = [s[i] for s, i in zip(slice_finder.top_slices_, idx)]
    slices = []
    for name, val in zip(names, attr):
        assert len(name) == len(val), f"len(n){len(name)} != len({len(val)})"
        num = len(name)
        # aux = set((name[i], val[i]) for i in range(num))
        aux = [{"name": name[i], "value": val[i]} for i in range(num)]
        # print(name)
        slices.append(aux)
    return slices


torch.manual_seed(RANDON_SEED)
random.seed(RANDON_SEED)

if __name__ == "__main__":
    raw_data = pd.read_csv(data_path)
    raw_data.head()

    education = {"Bachelors": 0, "Masters": 1, "PHD": 2}
    oce_columns = ["City", "Gender", "EverBenched"]
    target_col = "LeaveOrNot"

    data = raw_data.copy()
    data["Education"] = data["Education"].map(education)
    data = pd.get_dummies(data=data, columns=oce_columns, drop_first=False, dtype=int)

    scaler = StandardScaler()

    features = data.copy()

    train_set, test_set = train_test_split(
        features, test_size=0.2, shuffle=True, random_state=RANDON_SEED
    )

    train_tgt = train_set[target_col].values
    train_ft = train_set.drop(target_col, axis="columns")
    feature_names = train_ft.columns.values
    train_ft = StandardScaler().fit_transform(train_ft)

    batch_size = 32
    shuffle = True
    n_epochs = 30
    in_features = len(feature_names)
    out_features = 1
    c = 0

    # device = torch.device(get_device())
    device = "cpu"

    train_set = preprocess(df=train_set, target_col=target_col, scaler=StandardScaler())
    craig_train_train = ClassifierDataset(dataframe=train_set, target=target_col)
    craig_train_train = Subset(
        dataset=craig_train_train, indices=craig_train_train.indices
    )

    for i in np.linspace(0.1, 1, 10, dtype=float):
        print(f"building corset for frac={i}")
        MLP_craig = Classifier(in_features=in_features, out_features=1, dropout=0.0)

        optmizer = Adam(lr=10e-6, params=MLP_craig.parameters())
        loss_fn = nn.BCELoss()

        # craig_train_train = ClassifierDataset(dataframe=train_set, target=target_col)

        args = dict(
            print_freq=10e3,
            num_classes=2,
            # num_classes=MLP_craig.get_last_layer().out_features,
            device="cpu",
            model_optimizer=optmizer,
            batch=256,
            train_batch=256,
            workers=1,
            selection_batch=128,
        )
        args = DotMap(args)

        selection_args = dict(
            # epochs=1,
            epochs=n_epochs,
            balance=True,
            greedy="LazyGreedy",
            fraction=i,
            fraction_pretrain=0.5,
        )

        craig = Craig(
            dst_train=craig_train_train,
            args=args,
            random_seed=RANDON_SEED,
            specific_model=MLP_craig,
            criterion=loss_fn,
            **selection_args,
        )
        coreset = craig.select()
        print("writing json file")
        with open(f"coreset_{i}.json", "w") as arq:
            coreset["indices"] = coreset["indices"].tolist()
            coreset["weights"] = coreset["weights"].tolist()
            out = json.dumps(coreset, indent=4)
            arq.writelines(out)
            arq.close()
