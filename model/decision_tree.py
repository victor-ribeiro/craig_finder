# %%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.metrics import classification_report, log_loss
from sliceline.slicefinder import Slicefinder


# %%
target_col = "LeaveOrNot"
attr_col = [
    "Education",
    "JoiningYear",
    "City_Bangalore",
    "City_New",
    "Delhi,City_Pune",
    "PaymentTier",
    "Age",
    "ExperienceInCurrentDomain",
    "Gender_Female",
    "Gender_Male",
    "EverBenched_No",
    "EverBenched_Yes",
]

data = pd.read_csv("../data/employee/t_Employee.csv")
scaler = Normalizer()

# data = pd.DataFrame(data=scaler.fit_transform(data.values), columns=data.columns)

d = pd.read_csv(
    "../data/employee/Employee.csv",
)
train_data, test_data = train_test_split(
    data, random_state=42, shuffle=True, train_size=0.8
)
# %%
#
# treinamento - Livre
#

y_data = train_data[target_col].values
x_data = train_data.drop(target_col, axis="columns").values

dt = DecisionTreeClassifier()
dt.fit(scaler.fit_transform(x_data), y_data)

importance = dt.feature_importances_
print(len(attr_col), len(importance))
plt.bar(attr_col, importance)
plt.xticks(range(0, len(attr_col)), attr_col, rotation=90)
plt.show()


y_data = test_data[target_col].values
x_data = test_data.drop(target_col, axis="columns").values

y_pred = dt.predict(scaler.fit_transform(x_data))
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
    alpha=0.95, k=5, max_l=x_data.shape[1], min_sup=1, verbose=True
)

slice_finder.fit(X=d.iloc[x_data.index], errors=erro_n)
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
    "City_Bangalore",
    "City_New",
    "Delhi,City_Pune",
    "Education",
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
dt.fit(scaler.fit_transform(x_data), y_data)

importance = dt.feature_importances_

plt.bar(attr_col, importance)
plt.xticks(range(0, len(attr_col)), attr_col, rotation=90)
plt.show()


y_data = test_data[target_col].values
x_data = test_data.drop([target_col, "JoiningYear"], axis="columns").values

y_pred = dt.predict(scaler.fit_transform(x_data))
print(classification_report(y_true=y_data, y_pred=y_pred))


# %%

y_data = test_data[target_col]
x_data = test_data.drop([target_col, "JoiningYear"], axis="columns")


def logLoss(y, y_prob):
    return -1 * ((y * np.log(y_prob)) + (1 - y) * np.log(1 - y_prob))


y_prob = dt.predict_proba(x_data.values)
eps = 10e-15

prob_n = np.clip(y_prob[..., 1], eps, 1 - eps)
erro_n = logLoss(y=y_data.values, y_prob=prob_n)

slice_finder = Slicefinder(
    alpha=0.95, k=5, max_l=x_data.shape[1], min_sup=1, verbose=False
)

slice_finder.fit(X=d.iloc[x_data.index], errors=erro_n)
pd.DataFrame(
    slice_finder.top_slices_,
    columns=slice_finder.feature_names_in_,
    index=slice_finder.get_feature_names_out(),
)
# %%
d["PaymentTier"].plot(kind="hist")
plt.show()
# %%

d.loc[d["Education"] == "Bachelors"].groupby("Gender").count()["LeaveOrNot"]
# %%
d.groupby("Gender").count()["LeaveOrNot"]
# %%
