"""训练引擎：批量与增量两种模式，统一用生成器协议向 UI 推送进度。"""
from __future__ import annotations

import importlib
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Generator

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline

from core.registry import EstimatorSpec, Task, get_estimator


@dataclass
class TrainingSession:
    algorithm: str
    task: str
    params: dict[str, Any]
    X_train: pd.DataFrame | np.ndarray
    y_train: np.ndarray | None
    feature_names: list[str]
    target_name: str | None = None
    X_test: pd.DataFrame | np.ndarray | None = None
    y_test: np.ndarray | None = None
    preprocessing_spec: Any = None
    estimator: Any = None
    history: list[dict] = field(default_factory=list)
    status: str = "pending"
    error: str | None = None
    mode: str = "batch"
    started_at: float | None = None
    finished_at: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        self.cancel_event.set()

    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.started_at


def instantiate_estimator(spec: EstimatorSpec, params: dict[str, Any]) -> Any:
    module = importlib.import_module(spec.module)
    cls = getattr(module, spec.name)
    return cls(**params)


def resolve_spec(algorithm: str) -> EstimatorSpec:
    return get_estimator(algorithm)


def _evaluate(estimator: Any, X: Any, y: np.ndarray | None, task: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    try:
        if task == Task.CLASSIFICATION and y is not None:
            y_pred = estimator.predict(X)
            metrics["accuracy"] = float(accuracy_score(y, y_pred))
            metrics["f1_macro"] = float(f1_score(y, y_pred, average="macro", zero_division=0))
            if hasattr(estimator, "predict_proba"):
                try:
                    proba = estimator.predict_proba(X)
                    classes = getattr(estimator, "classes_", None)
                    if classes is None and isinstance(estimator, Pipeline):
                        classes = getattr(estimator.steps[-1][1], "classes_", None)
                    if classes is not None and proba.shape[1] == len(classes):
                        metrics["log_loss"] = float(log_loss(y, proba, labels=classes))
                except Exception:
                    pass
        elif task == Task.REGRESSION and y is not None:
            y_pred = estimator.predict(X)
            metrics["mse"] = float(mean_squared_error(y, y_pred))
            metrics["mae"] = float(mean_absolute_error(y, y_pred))
            metrics["r2"] = float(r2_score(y, y_pred))
        elif task in (Task.CLUSTERING, Task.BICLUSTERING):
            labels = getattr(estimator, "labels_", None)
            if labels is None and hasattr(estimator, "predict"):
                labels = estimator.predict(X)
            if labels is not None:
                n_unique = len(np.unique(labels))
                if 2 <= n_unique < len(labels):
                    metrics["silhouette"] = float(silhouette_score(np.asarray(X), labels))
            if hasattr(estimator, "inertia_"):
                metrics["inertia"] = float(estimator.inertia_)
        elif task in (Task.ANOMALY, Task.DENSITY):
            if hasattr(estimator, "decision_function"):
                scores = estimator.decision_function(X)
                metrics["score_mean"] = float(np.mean(scores))
                metrics["score_std"] = float(np.std(scores))
            elif hasattr(estimator, "score_samples"):
                scores = estimator.score_samples(X)
                metrics["score_mean"] = float(np.mean(scores))
                metrics["score_std"] = float(np.std(scores))
        elif task == Task.DIMRED:
            if hasattr(estimator, "explained_variance_ratio_"):
                evr = estimator.explained_variance_ratio_
                metrics["explained_variance"] = float(np.sum(evr))
                metrics["n_components"] = int(len(evr))
    except Exception:
        pass
    return metrics


def _intrinsic_history(estimator: Any) -> list[dict]:
    """提取内禀逐步分数。若 estimator 是 Pipeline，取最后一步。"""
    target = estimator
    if isinstance(estimator, Pipeline) and estimator.steps:
        target = estimator.steps[-1][1]
    points: list[dict] = []
    for attr in ("train_score_", "validation_scores_"):
        seq = getattr(target, attr, None)
        if seq is None:
            continue
        try:
            arr = np.asarray(seq, dtype=float).ravel()
        except Exception:
            continue
        for i, v in enumerate(arr, start=1):
            points.append({"step": i, "source": attr, "value": float(v)})
    return points


def run_training(
    session: TrainingSession,
    *,
    epochs: int = 20,
    batch_size: int | None = None,
) -> Generator[dict[str, Any], None, None]:
    session.status = "running"
    session.started_at = time.time()
    session.history = []

    try:
        spec = resolve_spec(session.algorithm)
        estimator = instantiate_estimator(spec, session.params)

        # 预处理管道：将 ColumnTransformer 与模型组装成 Pipeline
        has_preprocessing = (
            session.preprocessing_spec is not None
            and (
                getattr(session.preprocessing_spec, "numeric_features", [])
                or getattr(session.preprocessing_spec, "categorical_features", [])
            )
        )

        if has_preprocessing:
            from core.preprocessing import build_column_transformer
            preprocessor = build_column_transformer(session.preprocessing_spec)
            estimator = Pipeline([
                ("preprocess", preprocessor),
                ("model", estimator),
            ])
            # Pipeline 不支持 partial_fit，强制 batch 模式
            session.mode = "batch"
        else:
            session.mode = "incremental" if hasattr(estimator, "partial_fit") else "batch"

        session.estimator = estimator

        mode_note = session.mode
        if has_preprocessing:
            mode_note += "（含预处理管道）"
        yield _snapshot(session, log=f"初始化完成，模式：{mode_note}")

        if session.mode == "incremental":
            yield from _run_incremental(session, estimator, epochs=epochs, batch_size=batch_size)
        else:
            yield from _run_batch(session, estimator)

    except Exception as exc:
        session.status = "error"
        session.error = f"{type(exc).__name__}: {exc}"
        session.finished_at = time.time()
        yield _snapshot(session, log="训练失败：\n" + traceback.format_exc(limit=3))


def _run_incremental(
    session: TrainingSession,
    estimator: Any,
    *,
    epochs: int,
    batch_size: int | None,
) -> Generator[dict[str, Any], None, None]:
    X_train = session.X_train
    y_train = session.y_train
    n = len(X_train)
    bs = batch_size or max(1, n // 10)
    rng = np.random.RandomState(0)
    classes = np.unique(y_train) if session.task == Task.CLASSIFICATION and y_train is not None else None

    is_df = hasattr(X_train, "iloc")

    for epoch in range(1, epochs + 1):
        if session.cancel_event.is_set():
            session.status = "cancelled"
            session.finished_at = time.time()
            yield _snapshot(session, log="训练已取消")
            return

        order = rng.permutation(n)
        for start in range(0, n, bs):
            idx = order[start:start + bs]
            X_batch = X_train.iloc[idx] if is_df else X_train[idx]
            y_batch = y_train[idx] if y_train is not None else None
            try:
                if classes is not None:
                    estimator.partial_fit(X_batch, y_batch, classes=classes)
                else:
                    estimator.partial_fit(X_batch, y_batch)
            except TypeError:
                estimator.partial_fit(X_batch)

        train_metrics = _evaluate(estimator, X_train, y_train, session.task)
        val_metrics = (
            _evaluate(estimator, session.X_test, session.y_test, session.task)
            if session.X_test is not None and session.y_test is not None
            else {}
        )
        point = {
            "step": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "elapsed": session.elapsed(),
        }
        session.history.append(point)
        yield _snapshot(session, log=f"epoch {epoch}/{epochs}")

    session.status = "done"
    session.finished_at = time.time()
    yield _snapshot(session, log="训练完成（增量模式）")


def _run_batch(session: TrainingSession, estimator: Any) -> Generator[dict[str, Any], None, None]:
    yield _snapshot(session, log="批量算法：正在拟合，期间无法逐步推送指标…")

    estimator.fit(session.X_train, session.y_train)

    if session.cancel_event.is_set():
        session.status = "cancelled"
        session.finished_at = time.time()
        yield _snapshot(session, log="训练已取消（fit 已完成但结果被丢弃）")
        return

    train_metrics = _evaluate(estimator, session.X_train, session.y_train, session.task)
    val_metrics = (
        _evaluate(estimator, session.X_test, session.y_test, session.task)
        if session.X_test is not None and session.y_test is not None
        else {}
    )
    for p in _intrinsic_history(estimator):
        session.history.append({"step": p["step"], "source": p["source"], "value": p["value"]})
    session.history.append({
        "step": len(session.history) + 1,
        "train": train_metrics,
        "val": val_metrics,
        "elapsed": session.elapsed(),
    })
    session.status = "done"
    session.finished_at = time.time()
    yield _snapshot(session, log="训练完成（批量模式）")


def _snapshot(session: TrainingSession, *, log: str = "") -> dict[str, Any]:
    current = session.history[-1] if session.history else {}
    return {
        "status": session.status,
        "mode": session.mode,
        "algorithm": session.algorithm,
        "task": session.task,
        "elapsed": session.elapsed(),
        "log": log,
        "error": session.error,
        "current": current,
        "history": list(session.history),
        "estimator": session.estimator,
    }


def supports_partial_fit(spec: EstimatorSpec) -> bool:
    """判断估计器类是否定义 partial_fit 方法（不实例化，只查类属性）。"""
    try:
        module = importlib.import_module(spec.module)
        cls = getattr(module, spec.name)
        return callable(getattr(cls, "partial_fit", None))
    except Exception:
        return False