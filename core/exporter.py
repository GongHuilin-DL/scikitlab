"""导出当前工作流为可独立运行的 Python 脚本。

设计原则：
- 完全自包含：数据以 CSV 快照形式与脚本同目录，不依赖内置数据集/训练时的原始文件
- 保真复现：类别目标反编码回原始类别名，预处理配置与 UI 完全一致
- 可读可改：分层注释，import 使用 sklearn 公开路径
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.registry import EstimatorSpec, Task


_EXPORTS_DIR = Path("artifacts/exports")


def _public_module(spec: EstimatorSpec) -> str:
    """提取公开 import 路径。sklearn 估计器常定义在私有子模块（如
    sklearn.linear_model._logistic），公开路径是 sklearn.linear_model。"""
    mod = spec.module
    if mod.startswith("sklearn."):
        parts = mod.split(".")
        public: list[str] = []
        for p in parts:
            if p.startswith("_"):
                break
            public.append(p)
        return ".".join(public) if public else mod
    return mod


def _import_lines(spec: EstimatorSpec) -> list[str]:
    lines = [
        "import numpy as np",
        "import pandas as pd",
        "import joblib",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.impute import SimpleImputer",
        "from sklearn.pipeline import Pipeline",
        "from sklearn.preprocessing import (",
        "    MinMaxScaler,",
        "    OneHotEncoder,",
        "    OrdinalEncoder,",
        "    StandardScaler,",
        ")",
    ]
    mod = _public_module(spec)
    lines.append(f"import {mod}")

    if spec.task == Task.CLASSIFICATION:
        lines.append(
            "from sklearn.metrics import ("
            "accuracy_score, classification_report, f1_score)"
        )
    elif spec.task == Task.REGRESSION:
        lines.append(
            "from sklearn.metrics import ("
            "mean_absolute_error, mean_squared_error, r2_score)"
        )
    return lines


def _preprocessing_code(dataset) -> list[str]:
    spec = dataset.preprocessing_spec
    num = list(getattr(spec, "numeric_features", []) or []) if spec else []
    cat = list(getattr(spec, "categorical_features", []) or []) if spec else []
    num_scaler = getattr(spec, "numeric_scaler", "standard") if spec else "standard"
    cat_encoder = getattr(spec, "categorical_encoder", "onehot") if spec else "onehot"
    num_imputer = getattr(spec, "numeric_imputer", "median") if spec else "median"
    cat_imputer = getattr(spec, "categorical_imputer", "most_frequent") if spec else "most_frequent"

    lines = [
        f"NUMERIC_FEATURES = {num!r}",
        f"CATEGORICAL_FEATURES = {cat!r}",
        "",
        "preprocessors = []",
    ]
    if num:
        scaler_map = {
            "none": None,
            "standard": "StandardScaler()",
            "minmax": "MinMaxScaler()",
        }
        scaler_expr = scaler_map.get(num_scaler)
        lines.append("if NUMERIC_FEATURES:")
        lines.append("    numeric_steps = [")
        lines.append(
            f'        ("impute", SimpleImputer(strategy={num_imputer!r})),'
        )
        if scaler_expr:
            lines.append(f'        ("scale", {scaler_expr}),')
        lines.append("    ]")
        lines.append(
            '    preprocessors.append(('
            '"num", Pipeline(numeric_steps), NUMERIC_FEATURES))'
        )
    if cat:
        encoder_expr = (
            'OneHotEncoder(handle_unknown="ignore", sparse_output=False)'
            if cat_encoder == "onehot"
            else 'OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)'
        )
        lines.append("if CATEGORICAL_FEATURES:")
        lines.append("    categorical_steps = [")
        lines.append(
            f'        ("impute", SimpleImputer(strategy={cat_imputer!r})),'
        )
        lines.append(f'        ("encode", {encoder_expr}),')
        lines.append("    ]")
        lines.append(
            '    preprocessors.append(('
            '"cat", Pipeline(categorical_steps), CATEGORICAL_FEATURES))'
        )
    lines.append("")
    lines.append(
        'preprocessor = ColumnTransformer(preprocessors, remainder="drop")'
    )
    return lines


def _evaluation_code(task: str) -> list[str]:
    if task == Task.CLASSIFICATION:
        return [
            "y_pred = pipeline.predict(X_test)",
            'print(f"accuracy: {accuracy_score(y_test, y_pred):.4f}")',
            'print(f"f1_macro: {f1_score(y_test, y_pred, '
            'average=\'macro\', zero_division=0):.4f}")',
            "print()",
            "print(classification_report(y_test, y_pred, zero_division=0))",
        ]
    if task == Task.REGRESSION:
        return [
            "y_pred = pipeline.predict(X_test)",
            'print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")',
            'print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")',
            'print(f"R^2: {r2_score(y_test, y_pred):.4f}")',
        ]
    return [
        "y_pred = pipeline.predict(X_test)",
        'print("predictions (first 10):", y_pred[:10])',
    ]


def _build_data_snapshot(dataset, snapshot_path: Path) -> None:
    X_train = dataset.X_train.reset_index(drop=True) if hasattr(
        dataset.X_train, "reset_index"
    ) else pd.DataFrame(dataset.X_train)
    X_test = dataset.X_test.reset_index(drop=True) if hasattr(
        dataset.X_test, "reset_index"
    ) else pd.DataFrame(dataset.X_test)

    target_col = dataset.target_name or "target"
    if dataset.y_train is not None:
        y_tr = np.asarray(dataset.y_train)
        y_te = np.asarray(dataset.y_test)
        # 分类任务反编码回原始类别名，便于脚本可读
        if dataset.task_hint == Task.CLASSIFICATION and dataset.target_classes:
            y_tr = np.array([dataset.target_classes[int(i)] for i in y_tr])
            y_te = np.array([dataset.target_classes[int(i)] for i in y_te])
        X_train[target_col] = y_tr
        X_test[target_col] = y_te

    X_train["_split"] = "train"
    X_test["_split"] = "test"
    merged = pd.concat([X_train, X_test], ignore_index=True)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(snapshot_path, index=False)


def export_training_script(
    dataset,
    spec: EstimatorSpec,
    params: dict[str, Any],
    *,
    output_dir: Path | str | None = None,
) -> Path:
    """生成完全自包含的训练脚本，返回脚本路径。"""
    out_dir = Path(output_dir) if output_dir else _EXPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    data_path = out_dir / f"data_{ts}.csv"
    _build_data_snapshot(dataset, data_path)

    algorithm = spec.name
    task = dataset.task_hint
    target_col = dataset.target_name or "target"
    feature_cols = list(dataset.feature_names)
    mod = _public_module(spec)

    lines: list[str] = []
    lines.append('"""')
    lines.append("ScikitLab 自动导出的训练脚本")
    lines.append(f"算法: {algorithm}  |  任务: {task}")
    lines.append(f"生成时间: {ts}")
    lines.append("")
    lines.append("使用方法：")
    lines.append(f"  python {algorithm}_train.py")
    lines.append("")
    lines.append(f"数据快照: {data_path.name}（与脚本同目录）")
    lines.append('"""')
    lines.append("")

    lines.extend(_import_lines(spec))
    lines.append("")

    lines.append("# ============================================================")
    lines.append("# 1. 载入数据")
    lines.append("# ============================================================")
    lines.append(f'df = pd.read_csv("{data_path.name}")')
    lines.append('train_df = df[df["_split"] == "train"].drop(columns=["_split"])')
    lines.append('test_df = df[df["_split"] == "test"].drop(columns=["_split"])')
    lines.append(f'TARGET_COL = "{target_col}"')
    lines.append(f"FEATURE_COLS = {feature_cols!r}")
    lines.append("")

    lines.append("# ============================================================")
    lines.append("# 2. 取出特征与标签")
    lines.append("# ============================================================")
    lines.append("X_train = train_df[FEATURE_COLS]")
    lines.append("y_train = train_df[TARGET_COL]")
    lines.append("X_test = test_df[FEATURE_COLS]")
    lines.append("y_test = test_df[TARGET_COL]")
    lines.append("")

    lines.append("# ============================================================")
    lines.append("# 3. 预处理管道")
    lines.append("# ============================================================")
    lines.extend(_preprocessing_code(dataset))
    lines.append("")

    lines.append("# ============================================================")
    lines.append("# 4. 模型与 Pipeline")
    lines.append("# ============================================================")
    lines.append(f"model = {mod}.{algorithm}(")
    for k, v in params.items():
        lines.append(f"    {k}={v!r},")
    lines.append(")")
    lines.append("")
    lines.append('pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])')
    lines.append("")

    lines.append("# ============================================================")
    lines.append("# 5. 训练")
    lines.append("# ============================================================")
    lines.append("pipeline.fit(X_train, y_train)")
    lines.append("")

    lines.append("# ============================================================")
    lines.append("# 6. 评估")
    lines.append("# ============================================================")
    lines.extend(_evaluation_code(task))
    lines.append("")

    lines.append("# ============================================================")
    lines.append("# 7. 保存模型")
    lines.append("# ============================================================")
    lines.append(f'joblib.dump(pipeline, "{algorithm}.joblib")')
    lines.append(f'print("model saved to {algorithm}.joblib")')

    script_path = out_dir / f"{algorithm}_{ts}.py"
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script_path