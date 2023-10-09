# %%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, log_loss
from sliceline.slicefinder import Slicefinder


# %%
target_col = "LeaveOrNot"
attr_col = [
    "intercept",
    "Education",
    "JoiningYear",
    "City_0",
    "City_1",
    "PaymentTier",
    "Age",
    "ExperienceInCurrentDomain",
    "Gender_Female",
    "Gender_Male",
    "EverBenched_No",
    "EverBenched_Yes",
]

data = pd.read_csv("../data/employee/t_Employee.csv", index_col="Unnamed: 0")
d = pd.read_csv(
    "../data/employee/Employee.csv",
)
train_data, test_data = train_test_split(
    data, random_state=42, shuffle=True, train_size=0.7
)
# %%
#
# treinamento - Livre
#

y_data = train_data[target_col].values
x_data = train_data.drop(target_col, axis="columns").values

dt = DecisionTreeClassifier(max_depth=5)
dt.fit(x_data, y_data)

importance = dt.feature_importances_

plt.bar(attr_col, importance)
plt.xticks(range(0, len(attr_col)), attr_col, rotation=90)
plt.show()


y_data = test_data[target_col].values
x_data = test_data.drop(target_col, axis="columns").values

y_pred = dt.predict(x_data)
print(classification_report(y_true=y_data, y_pred=y_pred))

# %%
#
# Slicefinder Livre
y_data = test_data[target_col]
x_data = test_data.drop(target_col, axis="columns")


def logLoss(y, y_prob):
    return -1 * ((y * np.log(y_prob)) + (1 - y) * np.log(1 - y_prob))


y_prob = dt.predict_proba(x_data.values)
eps = 10e-15

prob_n = np.clip(y_prob[..., 1], eps, 1 - eps)
erro_n = logLoss(y=y_data.values, y_prob=prob_n)

slice_finder = Slicefinder(
    alpha=0.95, k=5, max_l=x_data.shape[1], min_sup=1, verbose=False
)

slice_finder.fit(X=x_data, errors=erro_n)
pd.DataFrame(
    slice_finder.top_slices_,
    columns=slice_finder.feature_names_in_,
    index=slice_finder.get_feature_names_out(),
)
############################################################
# %%
#
# treinamento - S/ Ano
#

attr_col = [
    "intercept",
    "Education",
    "City_0",
    "City_1",
    "PaymentTier",
    "Age",
    "ExperienceInCurrentDomain",
    "Gender_Female",
    "Gender_Male",
    "EverBenched_No",
    "EverBenched_Yes",
]


y_data = train_data[target_col].values
x_data = train_data.drop([target_col, "JoiningYear"], axis="columns").values

dt = DecisionTreeClassifier()
dt.fit(x_data, y_data)

importance = dt.feature_importances_

plt.bar(attr_col, importance)
plt.xticks(range(0, len(attr_col)), attr_col, rotation=90)
plt.show()


y_data = test_data[target_col].values
x_data = test_data.drop([target_col, "JoiningYear"], axis="columns").values

y_pred = dt.predict(x_data)
print(classification_report(y_true=y_data, y_pred=y_pred))


# %%

y_data = test_data[target_col]
x_data = test_data.drop([target_col, "JoiningYear"], axis="columns")


def logLoss(y, y_prob):
    return -1 * ((y * np.log(y_prob)) + (1 - y) * np.log(1 - y_prob))


y_prob = dt.predict_proba(x_data.values)
eps = 10e-15

prob_n = np.clip(y_prob[..., 0], eps, 1 - eps)
erro_n = logLoss(y=y_data.values, y_prob=prob_n)

slice_finder = Slicefinder(
    alpha=0.95, k=5, max_l=x_data.shape[1], min_sup=1, verbose=False
)

slice_finder.fit(X=x_data, errors=erro_n)
pd.DataFrame(
    slice_finder.top_slices_,
    columns=slice_finder.feature_names_in_,
    index=slice_finder.get_feature_names_out(),
)
# %%
d["PaymentTier"].plot(kind="hist")
plt.show()
# %%

d.loc[d["PaymentTier"] == 2].groupby("Gender").count()["LeaveOrNot"]
# %%
d.groupby("Gender").count()["LeaveOrNot"]
# %%
