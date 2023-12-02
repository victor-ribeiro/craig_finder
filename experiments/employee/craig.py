from pathlib import Path
import json
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.nn import BCELoss
import sys

sys.path.append("../..")
sys.path.append("../../src")

from data.preprocessing import preprocess
from data.dataset import PandasDataset
from model.classifier import EmployeeClassifier, train_torch_net


# CONSTANTS
DATA_PATH = Path(
    "/Users/victor/Documents/doutorado/tese/craig_finder/data/employee/Employee.csv"
)
# preprocessing
CORESET_FOLDER = Path("craig")
TGT_COL = "LeaveOrNot"
EDU_DICT = {"Bachelors": 0, "Masters": 1, "PHD": 2}
OCE_COLS = ["City", "Gender", "EverBenched"]
DS_SPLIT = [0.8, 0.1, 0.1]
# training
BATCH_SIZE = [
    8,  # 0,1 - ok
    8,  # 0,2
    8,  # 0,3
    8,  # 0,4
    8,  # 0,5
    8,  # 0,6
    8,  # 0,7
]
N_EMPOCHS = 15
LR = [
    10e-7,  # 0,1
    10e-7,  # 0,2
    10e-7,  # 0,3
    10e-7,  # 0,4
    10e-7,  # 0,5
    10e-7,  # 0,6
    10e-7,  # 0,7
]
# END


def prepair_training_data(
    data_frame, map_col, map_dict, encode_cols, lengths, target_col, scaler, batch_size
):
    dataset = data_frame.copy()
    dataset[map_col] = dataset[map_col].map(map_dict)
    dataset = pd.get_dummies(
        data=dataset, columns=encode_cols, drop_first=False, dtype=int
    )
    dataset = preprocess(df=dataset, scaler=scaler, target_col=target_col)
    dataset = PandasDataset(dataframe=dataset, target=target_col)
    ds_train, ds_test, ds_val = random_split(dataset=dataset, lengths=lengths)
    ds_train = DataLoader(dataset=ds_train, batch_size=batch_size)
    ds_test = DataLoader(dataset=ds_test, batch_size=batch_size)
    ds_val = DataLoader(dataset=ds_val, batch_size=batch_size)
    return ds_train, ds_test, ds_val


import matplotlib.pyplot as plt

if __name__ == "__main__":
    raw_data = pd.read_csv(DATA_PATH)
    frac_pattern = re.compile(r"[0-9]+\.[0-9]+")

    fig = plt.figure()
    for json_file, lr, batch_size in zip(CORESET_FOLDER.glob("*.json"), LR, BATCH_SIZE):
        with json_file.open("r") as core_set_file:
            coreset_info = json.load(core_set_file)
        file_name = json_file.stem
        frac = float(frac_pattern.search(file_name)[0])

        core_index = coreset_info["indices"]
        core_weights = coreset_info["weights"]

        coreset = raw_data.iloc[core_index, ...].copy()
        core_train, core_test, core_val = prepair_training_data(
            data_frame=coreset,
            map_col="Education",
            map_dict=EDU_DICT,
            encode_cols=OCE_COLS,
            lengths=DS_SPLIT,
            target_col=TGT_COL,
            scaler=StandardScaler(),
            batch_size=batch_size,
        )
        fig = plt.figure()
        for i in range(10):
            core_model = EmployeeClassifier(in_features=12, out_features=1, dropout=0.0)
            loss_fn = BCELoss()
            optmizer = Adam(params=core_model.parameters(), lr=lr)

            hist = train_torch_net(
                model=core_model,
                data_train=core_train,
                data_test=core_test,
                data_val=core_val,
                epochs=N_EMPOCHS,
                optmizer=optmizer,
                loss_fn=loss_fn,
            )
            plt.plot(hist["train_loss"], label=i)
        plt.legend()
        plt.savefig(f"{frac}.png")
        plt.close(f"{frac}.png")
