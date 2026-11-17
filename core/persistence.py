"""模型持久化：joblib 保存完整估计器对象，JSON 保存训练前配置与元数据。"""
from __future__ import annotations

import dataclasses
import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn

from core.trainer import TrainingSession


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_\-\u4e00-\u9fff]+")


def _sanitize(name: str) -> str:
    cleaned = _SAFE_NAME.sub("_", name.strip())
    return cleaned or "model"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


@dataclass
class ModelMetadata:
    name: str
    algorithm: str
    module: str
    task: str
    sklearn_version: str
    created_at: str
    mode: str
    params_before_fit: dict[str, Any]
    feature_names: list[str]
    target_name: str | None
    n_features: int
    n_classes: int | None
    training_metrics: dict[str, Any]
    training_history: list[dict]
    elapsed_seconds: float
    notes: str = ""
    # 新增字段：区分数值特征与类别特征。旧 JSON 缺失时使用空 dict，
    # 推理页会回退到全部按数值特征处理。
    feature_types: dict[str, list[str]] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(_jsonable(asdict(self)), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelMetadata":
        """仅保留 dataclass 已知字段，避免旧 JSON 缺字段或未来加字段时崩溃。"""
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in known}
        return cls(**filtered)


def save_model(
    session: TrainingSession,
    name: str,
    artifacts_dir: Path | str,
    *,
    notes: str = "",
) -> tuple[Path, Path]:
    if session.estimator is None:
        raise ValueError("会话尚未训练，无法保存")
    if session.status != "done":
        raise ValueError(f"会话状态为 {session.status}，只有 done 状态可保存")

    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = _sanitize(name)
    joblib_path = out_dir / f"{safe}.joblib"
    meta_path = out_dir / f"{safe}.json"

    joblib.dump(session.estimator, joblib_path, compress=3)

    est = session.estimator
    classes = getattr(est, "classes_", None)

    spec = getattr(session, "preprocessing_spec", None)
    feature_types = {
        "numeric": list(getattr(spec, "numeric_features", []) or []) if spec else [],
        "categorical": list(getattr(spec, "categorical_features", []) or []) if spec else [],
    }

    meta = ModelMetadata(
        name=safe,
        algorithm=session.algorithm,
        module=type(est).__module__,
        task=session.task,
        sklearn_version=sklearn.__version__,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        mode=session.mode,
        params_before_fit=session.params,
        feature_names=list(session.feature_names),
        target_name=session.target_name,
        n_features=len(session.feature_names),
        n_classes=int(len(classes)) if classes is not None else None,
        training_metrics=session.history[-1] if session.history else {},
        training_history=list(session.history),
        elapsed_seconds=session.elapsed(),
        notes=notes,
        feature_types=feature_types,
    )
    meta_path.write_text(meta.to_json(), encoding="utf-8")
    return joblib_path, meta_path


@dataclass
class LoadedModel:
    estimator: Any
    metadata: ModelMetadata
    warnings: list[str] = field(default_factory=list)


def load_model(name: str, artifacts_dir: Path | str) -> LoadedModel:
    base = Path(artifacts_dir)
    safe = _sanitize(name)
    joblib_path = base / f"{safe}.joblib"
    meta_path = base / f"{safe}.json"
    if not joblib_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"模型 {safe} 不完整：{joblib_path.name} / {meta_path.name}")

    estimator = joblib.load(joblib_path)
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata = ModelMetadata.from_dict(raw)

    warns: list[str] = []
    saved = metadata.sklearn_version
    current = sklearn.__version__
    if saved != current:
        saved_mm = ".".join(saved.split(".")[:2])
        current_mm = ".".join(current.split(".")[:2])
        level = "警告" if saved_mm != current_mm else "提示"
        warns.append(f"{level}：模型保存于 scikit-learn {saved}，当前环境为 {current}")
    return LoadedModel(estimator=estimator, metadata=metadata, warnings=warns)


def list_models(artifacts_dir: Path | str) -> list[ModelMetadata]:
    base = Path(artifacts_dir)
    if not base.exists():
        return []
    result: list[ModelMetadata] = []
    for meta_path in sorted(base.glob("*.json")):
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            result.append(ModelMetadata.from_dict(raw))
        except Exception:
            continue
    return sorted(result, key=lambda m: m.created_at, reverse=True)


def delete_model(name: str, artifacts_dir: Path | str) -> bool:
    base = Path(artifacts_dir)
    safe = _sanitize(name)
    jp = base / f"{safe}.joblib"
    mp = base / f"{safe}.json"
    removed = False
    for p in (jp, mp):
        if p.exists():
            p.unlink()
            removed = True
    return removed