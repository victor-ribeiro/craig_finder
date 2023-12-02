from pandas import DataFrame
from sklearn.base import BaseEstimator


def preprocess(df: DataFrame, target_col: str, scaler: BaseEstimator):
    tgt = df[target_col].values
    transformed = df.drop(target_col, axis="columns")
    names = transformed.columns.values
    transformed = scaler.fit_transform(transformed.values)
    transformed = DataFrame(data=transformed, columns=names)
    transformed[target_col] = tgt
    return transformed
