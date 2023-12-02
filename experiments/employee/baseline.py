import sys

sys.path.append("../../src")
sys.path.append("../..")

from model.eval_model import map_error, predict
from model.classifier import EmployeeClassifier, train_torch_net
from slice_line.slices import (
    coverage,
    parse_predicates,
    parse_query,
    get_slice_indices,
    efect_size,
)
from data.dataset import PandasDataset

import numpy as np
import pandas as pd

import random
from sliceline.slicefinder import Slicefinder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


from torch import manual_seed, Generator
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.nn import BCELoss

from itertools import product
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


# CONSTANTS
## preprocessing
EDU_DICT = {"Bachelors": 0, "Masters": 1, "PHD": 2}
OCE_COLS = ["City", "Gender", "EverBenched"]
TGT_COL = "LeaveOrNot"
## trainning
RANDOM_SEED = 42
BATCH_SIZE = 32
DS_SPLIT = [0.8, 0.1, 0.1]
N_EPOCHS = 20
LR = 10e-5
# END
random.seed(RANDOM_SEED)
manual_seed(RANDOM_SEED)

# DATA LOADING AND PREPROCESSING
raw_data = pd.read_csv(
    "/Users/victor/Documents/doutorado/tese/craig_finder/data/employee/Employee.csv"
)

## classes balancing
true_raw = raw_data.loc[raw_data[TGT_COL] == 1]
n_true = len(true_raw)
false_raw = raw_data.loc[~(raw_data[TGT_COL] == 1)].sample(n_true)
raw_data = pd.concat([false_raw, true_raw]).sample(frac=1)
raw_data.index = np.arange(len(raw_data), dtype=int)

## PREPROCESSING
scaler = StandardScaler()

data = raw_data.copy()
data["Education"] = data["Education"].map(EDU_DICT)
data = pd.get_dummies(data=data, columns=OCE_COLS, drop_first=False, dtype=int)
target = data[TGT_COL].values
data = data.drop(TGT_COL, axis="columns")
columns = data.columns.values

data = scaler.fit_transform(data.values)
data = pd.DataFrame(data=data, columns=columns)
data[TGT_COL] = target

# END

# model_training
generator = Generator().manual_seed(RANDOM_SEED)

dataset = PandasDataset(dataframe=data, target=TGT_COL)
ds_train, ds_test, ds_val = random_split(
    dataset=dataset,
    lengths=DS_SPLIT,
    generator=generator,
)

ds_train = DataLoader(dataset=ds_train, batch_size=BATCH_SIZE)
ds_test = DataLoader(dataset=ds_test, batch_size=BATCH_SIZE)
ds_val = DataLoader(dataset=ds_val, batch_size=BATCH_SIZE)


in_features = data.shape[1] - 1
loss_fn = BCELoss()
base_model = EmployeeClassifier(in_features=in_features, out_features=1, dropout=0.0)
optmizer = Adam(lr=LR, params=base_model.parameters())
hist = train_torch_net(
    model=base_model,
    optmizer=optmizer,
    loss_fn=loss_fn,
    data_train=ds_train,
    data_test=ds_test,
    data_val=ds_val,
    epochs=N_EPOCHS,
)

base_pred = predict(data, target_col=TGT_COL, model=base_model)
base_pred = base_pred.squeeze(1).detach().numpy().round()

print("######################     base model report      ######################")
report = classification_report(y_true=data[TGT_COL].values, y_pred=base_pred)
print(report)


# slice evaluation
k_alpha = np.linspace(0.9, 1, 4).round(2)
k_min_sup = np.linspace(0.1, 0.7, 4).round(2)

error = map_error(df=data, target_col=TGT_COL, model=base_model)
slice_data = raw_data.drop(TGT_COL, axis="columns")

print("###################### EVALUATING: alpha, min_sup ######################")
base_result = []
for k in tqdm(range(1, 10)):
    for alpha, min_sup in product(k_alpha, k_min_sup):
        slice_finder = Slicefinder(alpha=alpha, min_sup=min_sup, verbose=False, k=k)
        slice_finder.fit(slice_data, errors=error)

        predicates = parse_predicates(
            in_names=slice_finder.feature_names_in_, slices=slice_finder.top_slices_
        )
        query_parser = parse_query(predicates)
        slice_indices = get_slice_indices(raw_data, query_parser)
        e_size = [
            efect_size(pred_error=error, slice_idx=i, idx=data.index)
            for i in slice_indices
        ]
        for i, top_slice in enumerate(slice_finder.top_slices_statistics_):
            result_k = {
                "k": k,
                "alpha": alpha,
                "min_sup": min_sup,
                "avg_loss": top_slice["slice_average_error"],
                "slice_size": top_slice["slice_size"],
                "efect_size": e_size[i],
            }
            base_result += [result_k]

print("[WRITING] baseline_metrics file", end=" - ")
base_result = pd.DataFrame.from_records(base_result)
base_result.to_csv(
    "/Users/victor/Documents/doutorado/tese/craig_finder/results/employee/baseline/baseline_metrics.csv",
    index=False,
)

print("OK")

pred_error = map_error(df=data, target_col=TGT_COL, model=base_model)
slice_finder = Slicefinder(alpha=1, min_sup=0.3, verbose=False, k=1)
slice_finder.fit(raw_data.drop(TGT_COL, axis="columns"), pred_error)

predicates = parse_predicates(
    in_names=slice_finder.feature_names_in_, slices=slice_finder.top_slices_
)

query_parser = parse_query(predicates_gen=predicates)
full_indices = get_slice_indices(dataset=raw_data, query_parser=query_parser)

print("#######################   EVALUATING: coverage   #######################")

n_sample = np.linspace(10e-3, 1, 20)
frac_cov = []
for i in tqdm(range(100)):
    for frac in n_sample:
        sample = raw_data.sample(frac=frac)
        sample_index = sample.index.values

        pred_error = map_error(
            df=data.iloc[sample_index], target_col=TGT_COL, model=base_model
        )
        slice_finder = Slicefinder(alpha=1, min_sup=0.3, verbose=False, k=1)
        slice_finder.fit(sample.drop(TGT_COL, axis="columns"), pred_error)

        predicates = parse_predicates(
            in_names=slice_finder.feature_names_in_, slices=slice_finder.top_slices_
        )

        query_parser = parse_query(predicates_gen=predicates)
        sample_slice_indices = get_slice_indices(
            dataset=sample, query_parser=query_parser
        )
        for sample_indices, base_indices in zip(sample_slice_indices, full_indices):
            result = {
                "acc": coverage(
                    sample_slice_indices=sample_indices, slice_base_indices=base_indices
                ),
                "frac": frac,
            }
            frac_cov += [result]
print("[WRITING] coverage file", end=" - ")
frac_cov = pd.DataFrame.from_records(frac_cov)
frac_cov.to_csv(
    "/Users/victor/Documents/doutorado/tese/craig_finder/results/employee/baseline/baseline_coverage.csv",
    index=False,
)
print("OK")
