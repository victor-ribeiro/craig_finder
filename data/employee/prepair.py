import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from category_encoders.helmert import HelmertEncoder


education = {"Bachelors": 0, "Masters": 1, "PHD": 2}
data = pd.read_csv("Employee.csv")
education = {"Bachelors": 0, "Masters": 1, "PHD": 2}

ohenc = OneHotEncoder()
one_hot_cols = ["Gender", "EverBenched", "City"]

data["Education"] = data["Education"].map(education)
data["JoiningYear"] = data.JoiningYear.max() - data["JoiningYear"]
encoded = ohenc.fit_transform(data[one_hot_cols]).todense()

cols = ohenc.get_feature_names_out()
encoded = pd.DataFrame(data=encoded, columns=cols)

data = pd.concat([data, encoded], axis="columns")
data.drop(one_hot_cols, axis="columns", inplace=True)
data.head()

# city_encoder = HelmertEncoder(cols="City")
# data = city_encoder.fit_transform(data)
data.to_csv("t_Employee.csv", index=False)
