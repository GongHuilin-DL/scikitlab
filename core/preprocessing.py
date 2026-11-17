"""根据预处理描述构建 ColumnTransformer。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


_NUMERIC_SCALERS = {
    "none": None,
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
}

_CATEGORICAL_ENCODERS = {
    "onehot": lambda: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "ordinal": lambda: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
}


@dataclass
class PreprocessingSpec:
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    numeric_scaler: str = "standard"          # none / standard / minmax
    categorical_encoder: str = "onehot"       # onehot / ordinal
    numeric_imputer: str = "median"            # mean / median / most_frequent
    categorical_imputer: str = "most_frequent"


def infer_spec_from_dataframe(
    df: pd.DataFrame, feature_cols: list[str]
) -> PreprocessingSpec:
    numeric: list[str] = []
    categorical: list[str] = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
        else:
            categorical.append(c)
    return PreprocessingSpec(
        numeric_features=numeric,
        categorical_features=categorical,
    )


def build_column_transformer(spec: PreprocessingSpec) -> ColumnTransformer:
    transformers: list[tuple] = []

    if spec.numeric_features:
        scaler_cls = _NUMERIC_SCALERS.get(spec.numeric_scaler)
        steps: list[tuple] = [
            ("impute", SimpleImputer(strategy=spec.numeric_imputer)),
        ]
        if scaler_cls is not None:
            steps.append(("scale", scaler_cls()))
        transformers.append(("num", Pipeline(steps), spec.numeric_features))

    if spec.categorical_features:
        steps_cat: list[tuple] = [
            ("impute", SimpleImputer(strategy=spec.categorical_imputer)),
            ("encode", _CATEGORICAL_ENCODERS[spec.categorical_encoder]()),
        ]
        transformers.append(("cat", Pipeline(steps_cat), spec.categorical_features))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)