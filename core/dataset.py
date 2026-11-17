"""数据契约：DatasetBundle + 数据加载/划分/任务推断。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.model_selection import train_test_split

from core.preprocessing import PreprocessingSpec
from core.registry import Task


TOY_DATASETS: dict[str, tuple[str, str]] = {
    "Iris（鸢尾花 · 分类）": ("iris", Task.CLASSIFICATION),
    "Wine（葡萄酒 · 分类）": ("wine", Task.CLASSIFICATION),
    "Breast Cancer（乳腺癌 · 分类）": ("breast_cancer", Task.CLASSIFICATION),
    "Digits（手写数字 · 分类）": ("digits", Task.CLASSIFICATION),
    "Diabetes（糖尿病进展 · 回归）": ("diabetes", Task.REGRESSION),
}


def load_toy(name_key: str) -> tuple[pd.DataFrame, str, str]:
    _, task_hint = TOY_DATASETS[name_key]
    loader = TOY_DATASETS[name_key][0]
    if loader == "iris":
        d = load_iris(as_frame=True)
    elif loader == "wine":
        d = load_wine(as_frame=True)
    elif loader == "breast_cancer":
        d = load_breast_cancer(as_frame=True)
    elif loader == "digits":
        d = load_digits(as_frame=True)
    elif loader == "diabetes":
        d = load_diabetes(as_frame=True)
    else:
        raise ValueError(f"未知玩具数据集：{loader}")
    df = d.frame.copy()
    if "target" in df.columns:
        return df, "target", task_hint
    last_col = df.columns[-1]
    return df, last_col, task_hint


@dataclass
class DatasetBundle:
    name: str
    feature_names: list[str]
    target_name: str | None
    task_hint: str

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray | None
    y_test: np.ndarray | None

    target_classes: list | None = None
    class_distribution: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    preprocessing_spec: Any = None

    feature_defaults: list[Any] = field(default_factory=list)
    feature_ranges: list[tuple[float, float] | None] = field(default_factory=list)

    split_info: dict = field(default_factory=dict)
    raw_shape: tuple[int, int] = (0, 0)

    def summary(self) -> str:
        feat_info = f"**特征数**：{len(self.feature_names)}"
        if self.categorical_features:
            feat_info += (
                f"（数值 {len(self.numeric_features)} / "
                f"类别 {len(self.categorical_features)}）"
            )
        lines = [
            f"**数据集**：{self.name}",
            f"**任务类型**：`{self.task_hint}`",
            f"**原始形状**：{self.raw_shape[0]} 行 × {self.raw_shape[1]} 列",
            feat_info,
            f"**训练集**：{self.X_train.shape[0]} 行",
            f"**测试集**：{self.X_test.shape[0]} 行",
        ]
        if self.target_name:
            lines.append(f"**目标列**：`{self.target_name}`")
        info = self.split_info
        if info:
            lines.append(
                f"**划分**：test_size={info.get('test_size')}，"
                f"random_state={info.get('random_state')}，"
                f"模式={info.get('mode')}"
            )
        if self.warnings:
            lines.append("\n**⚠ 警告**\n")
            for w in self.warnings:
                lines.append(f"- {w}")
        return "\n".join(lines)


def infer_task(series: pd.Series | None, n_rows: int) -> str:
    if series is None:
        return "unsupervised"
    if pd.api.types.is_bool_dtype(series):
        return Task.CLASSIFICATION
    if pd.api.types.is_integer_dtype(series):
        n_unique = series.nunique(dropna=True)
        threshold = max(20, int(n_rows * 0.05))
        return Task.CLASSIFICATION if n_unique <= threshold else Task.REGRESSION
    if pd.api.types.is_float_dtype(series):
        return Task.REGRESSION
    return Task.CLASSIFICATION


def read_tabular(file_path: str | Path) -> pd.DataFrame:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在：{p}")
    suffix = p.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(p, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(p, encoding="gbk")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(p)
    raise ValueError(f"暂不支持的文件类型：{suffix}（支持 .csv / .xlsx / .xls）")


def _class_distribution(
    y_train: np.ndarray, y_test: np.ndarray, classes: list
) -> dict[str, dict[str, int]]:
    train_counts = pd.Series(y_train).value_counts().reindex(classes, fill_value=0)
    test_counts = pd.Series(y_test).value_counts().reindex(classes, fill_value=0)
    return {
        "train": {str(c): int(train_counts.loc[c]) for c in classes},
        "test": {str(c): int(test_counts.loc[c]) for c in classes},
    }


def build_bundle_from_dataframe(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str | None,
    task_hint: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    split_mode: str = "stratified",
    name: str = "dataset",
    numeric_scaler: str = "standard",
    categorical_encoder: str = "onehot",
) -> DatasetBundle:
    if not feature_cols:
        raise ValueError("必须至少选择一个特征列")
    if not 0.01 <= test_size <= 0.9:
        raise ValueError("test_size 必须在 (0.01, 0.9) 之间")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"特征列不存在：{missing}")
    if target_col is not None and target_col not in df.columns:
        raise ValueError(f"目标列不存在：{target_col}")

    warnings: list[str] = []
    df = df.dropna(subset=([target_col] if target_col else [])).copy()

    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    if categorical_cols:
        preview = categorical_cols[:5]
        preview_str = ", ".join(preview) + ("…" if len(categorical_cols) > 5 else "")
        warnings.append(
            f"检测到 {len(categorical_cols)} 个类别特征（{preview_str}），"
            "将通过预处理管道编码后送入模型"
        )

    y_series: pd.Series | None = None
    target_classes: list | None = None
    y_encoded: pd.Series | None = None

    if target_col is None:
        task_hint = "unsupervised"
    else:
        y_series = df[target_col]
        if task_hint == Task.CLASSIFICATION:
            unique_vals = y_series.dropna().unique()
            try:
                target_classes = sorted(unique_vals)
            except TypeError:
                target_classes = list(unique_vals)
            mapping = {c: i for i, c in enumerate(target_classes)}
            y_encoded = y_series.map(mapping)
            if y_encoded.isna().any():
                raise ValueError("目标列包含无法映射的类别值")
        elif task_hint == Task.REGRESSION:
            if not pd.api.types.is_numeric_dtype(y_series):
                raise ValueError("回归任务的目标列必须是数值类型")
            y_encoded = y_series.astype(float)
        else:
            task_hint = "unsupervised"
            target_col = None

    stratify_arg: np.ndarray | None = None
    effective_mode = split_mode
    if y_encoded is not None and task_hint == Task.CLASSIFICATION:
        class_counts = y_encoded.value_counts()
        if split_mode == "stratified":
            if (class_counts < 2).any():
                warnings.append("部分类别样本不足 2 个，stratified 模式自动降级为 random")
                effective_mode = "random"
            else:
                stratify_arg = y_encoded.values
        elif split_mode == "sequential":
            effective_mode = "sequential"
    else:
        effective_mode = "random"

    indices = np.arange(len(df))

    def _split(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if effective_mode == "sequential":
            n_test = max(1, int(len(idx) * test_size))
            return idx[:-n_test], idx[-n_test:]
        tr, te = train_test_split(
            idx,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arg if stratify_arg is not None and len(stratify_arg) == len(idx) else None,
        )
        return tr, te

    train_idx, test_idx = _split(indices)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("划分后训练集或测试集为空，请调小 test_size 或检查数据量")

    X_train = df.iloc[train_idx][feature_cols].reset_index(drop=True)
    X_test = df.iloc[test_idx][feature_cols].reset_index(drop=True)
    y_train = y_encoded.iloc[train_idx].to_numpy() if y_encoded is not None else None
    y_test = y_encoded.iloc[test_idx].to_numpy() if y_encoded is not None else None

    class_dist: dict = {}
    if y_encoded is not None and target_classes is not None:
        class_dist = _class_distribution(y_train, y_test, target_classes)
        train_zero = [c for c, n in class_dist["train"].items() if n == 0]
        test_zero = [c for c, n in class_dist["test"].items() if n == 0]
        if train_zero:
            warnings.append(f"训练集缺失类别：{train_zero}")
        if test_zero:
            warnings.append(f"测试集缺失类别：{test_zero}")

    feature_defaults: list[Any] = []
    feature_ranges: list[tuple[float, float] | None] = []
    for c in feature_cols:
        if c in numeric_cols:
            col = df[c].astype(float)
            valid = col.dropna()
            if valid.empty:
                median, lo, hi = 0.0, 0.0, 1.0
            else:
                median = float(valid.median())
                lo = float(valid.min())
                hi = float(valid.max())
                if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                    lo = lo if np.isfinite(lo) else 0.0
                    hi = hi if np.isfinite(hi) else 1.0
            feature_defaults.append(median)
            feature_ranges.append((lo, hi))
        else:
            mode_series = df[c].dropna()
            default = str(mode_series.mode().iloc[0]) if not mode_series.empty else ""
            feature_defaults.append(default)
            feature_ranges.append(None)

    preprocessing_spec = PreprocessingSpec(
        numeric_features=numeric_cols,
        categorical_features=categorical_cols,
        numeric_scaler=numeric_scaler,
        categorical_encoder=categorical_encoder,
    )

    return DatasetBundle(
        name=name,
        feature_names=list(feature_cols),
        target_name=target_col,
        task_hint=task_hint,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        target_classes=target_classes,
        class_distribution=class_dist,
        warnings=warnings,
        numeric_features=numeric_cols,
        categorical_features=categorical_cols,
        preprocessing_spec=preprocessing_spec,
        feature_defaults=feature_defaults,
        feature_ranges=feature_ranges,
        split_info={
            "test_size": test_size,
            "random_state": random_state,
            "mode": effective_mode,
        },
        raw_shape=(int(df.shape[0]), int(df.shape[1])),
    )