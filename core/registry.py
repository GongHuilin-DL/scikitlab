"""算法注册表：运行时发现 scikit-learn 与可选第三方包的估计器并归类。"""
from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterator

import sklearn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 任务类型常量
# ---------------------------------------------------------------------------
class Task:
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY = "anomaly_detection"
    BICLUSTERING = "biclustering"
    DENSITY = "density_estimation"
    DIMRED = "dimensionality_reduction"
    FEATURE_SELECTION = "feature_selection"
    TRANSFORMATION = "transformation"
    META = "meta_estimator"
    COVARIANCE = "covariance_estimation"
    OTHER = "other"


TASK_LABELS = {
    Task.CLASSIFICATION: "分类",
    Task.REGRESSION: "回归",
    Task.CLUSTERING: "聚类",
    Task.ANOMALY: "异常检测",
    Task.BICLUSTERING: "双聚类",
    Task.DENSITY: "密度估计",
    Task.DIMRED: "降维",
    Task.FEATURE_SELECTION: "特征选择",
    Task.TRANSFORMATION: "特征变换",
    Task.META: "元估计器",
    Task.COVARIANCE: "协方差估计",
    Task.OTHER: "其它",
}


# ---------------------------------------------------------------------------
# 扫描配置
# ---------------------------------------------------------------------------
_SKIP_MODULE_PREFIXES = (
    "sklearn.tests",
    "sklearn.externals",
    "sklearn._build_utils",
    "sklearn._min_dependencies",
    "sklearn.utils.tests",
    "sklearn.utils._test_common",
    "sklearn.utils._mocking",
    "sklearn.utils._testing",
    "sklearn.utils.estimator_checks",
)

# 可选第三方包。未安装时自动跳过；安装后重启 app 即出现在注册表中。
_EXTERNAL_PACKAGES = ["xgboost", "lightgbm"]

_SKIP_CLASS_NAMES = {
    "BaseEstimator",
    "ClassifierMixin",
    "RegressorMixin",
    "ClusterMixin",
    "TransformerMixin",
    "BiclusterMixin",
    "OutlierMixin",
    "DensityMixin",
    "MetaEstimatorMixin",
    "OneToOneFeatureMixin",
    "ClassNamePrefixFeaturesOutMixin",
}

PARAM_CHOICES: dict[str, list[str]] = {
    "criterion": ["gini", "entropy", "log_loss", "squared_error", "friedman_mse",
                  "absolute_error", "poisson"],
    "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
    "solver": ["auto", "lbfgs", "liblinear", "newton-cg", "newton-cholesky",
               "sag", "saga", "sgd", "adam", "svd", "eigen", "cholesky", "cd"],
    "penalty": ["l1", "l2", "elasticnet", "none"],
    "loss": ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron",
             "squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
    "max_features": ["auto", "sqrt", "log2"],
    "activation": ["identity", "logistic", "tanh", "relu"],
    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    "init": ["k-means++", "random"],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "chebyshev", "minkowski", "cosine", "precomputed"],
    "linkage": ["ward", "complete", "average", "single"],
    "strategy": ["mean", "median", "most_frequent", "constant"],
    "booster": ["gbtree", "gblinear", "dart"],
    "tree_method": ["auto", "exact", "approx", "hist"],
    "objective": ["reg:squarederror", "reg:logistic", "binary:logistic",
                  "multi:softmax", "multi:softprob"],
}

_NESTED_ESTIMATOR_PARAM_NAMES = {
    "estimator", "base_estimator", "regressor", "classifier", "clf",
    "final_estimator", "estimators",
}

_FORCE_META_CLASSES: set[str] = {
    "TunedThresholdClassifierCV",
    "FixedThresholdClassifier",
}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class ParamSpec:
    name: str
    kind: str
    default: Any
    default_repr: str
    choices: list[str] | None = None
    required: bool = False


@dataclass
class EstimatorSpec:
    name: str
    qualname: str
    module: str
    scan_module: str
    task: str
    is_meta: bool
    doc: str
    params: list[ParamSpec] = field(default_factory=list)
    nested_estimator_params: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 包与模块遍历
# ---------------------------------------------------------------------------
def _available_packages() -> list[str]:
    pkgs = ["sklearn"]
    for p in _EXTERNAL_PACKAGES:
        try:
            importlib.import_module(p)
            pkgs.append(p)
        except ImportError:
            pass
    return pkgs


def _should_skip_module(name: str) -> bool:
    if any(name == p or name.startswith(p + ".") for p in _SKIP_MODULE_PREFIXES):
        return True
    parts = name.split(".")
    if any(part in {"tests", "testing", "externals", "_testing"} for part in parts):
        return True
    return False


def _iter_package_modules(package_name: str) -> Iterator[str]:
    try:
        pkg = importlib.import_module(package_name)
    except ImportError:
        return
    if not hasattr(pkg, "__path__"):
        return
    seen: set[str] = set()
    for info in pkgutil.walk_packages(pkg.__path__, prefix=f"{package_name}."):
        name = info.name
        if name in seen:
            continue
        seen.add(name)
        if _should_skip_module(name):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                importlib.import_module(name)
        except Exception as exc:
            logger.debug("跳过模块 %s: %s", name, exc)
            continue
        yield name


def _iter_sklearn_modules() -> Iterator[str]:
    """保留原名以兼容现有调用方。实际遍历所有被扫描的包（sklearn + 第三方）。"""
    for pkg in _available_packages():
        yield from _iter_package_modules(pkg)


# ---------------------------------------------------------------------------
# 候选判断与归类
# ---------------------------------------------------------------------------
def _is_candidate(obj: Any, module_name: str) -> bool:
    if not inspect.isclass(obj):
        return False
    if obj.__name__ in _SKIP_CLASS_NAMES:
        return False
    if obj.__name__.startswith("_") or obj.__name__.startswith("Base"):
        return False
    if inspect.isabstract(obj):
        return False
    if not callable(getattr(obj, "fit", None)):
        return False

    obj_mod = getattr(obj, "__module__", "") or ""
    pkgs = _available_packages()
    if not any(obj_mod.startswith(p) for p in pkgs):
        return False
    if any(obj_mod == p or obj_mod.startswith(p + ".") for p in _SKIP_MODULE_PREFIXES):
        return False

    doc = obj.__doc__ or ""
    if ".. deprecated::" in doc[:600]:
        return False
    return True


def _classify(obj: type, module_name: str) -> str | None:
    mro = {c.__name__ for c in obj.__mro__}
    est_type = getattr(obj, "_estimator_type", None)
    if not isinstance(est_type, str):
        est_type = None

    if "ClassifierMixin" in mro or est_type == "classifier":
        return Task.CLASSIFICATION
    if "RegressorMixin" in mro or est_type == "regressor":
        return Task.REGRESSION
    if "ClusterMixin" in mro or est_type == "clusterer":
        return Task.CLUSTERING
    if "OutlierMixin" in mro:
        return Task.ANOMALY
    if "BiclusterMixin" in mro:
        return Task.BICLUSTERING
    if "DensityMixin" in mro:
        return Task.DENSITY
    if "MetaEstimatorMixin" in mro:
        return Task.META
    if "TransformerMixin" in mro:
        if "decomposition" in module_name or "manifold" in module_name:
            return Task.DIMRED
        if "feature_selection" in module_name:
            return Task.FEATURE_SELECTION
        return Task.TRANSFORMATION
    if "covariance" in module_name:
        return Task.COVARIANCE

    # 第三方包兜底
    name = obj.__name__
    if name.endswith("Classifier") or name.endswith("Classification"):
        return Task.CLASSIFICATION
    if name.endswith("Regressor") or name.endswith("Regression"):
        return Task.REGRESSION
    if name.endswith("Ranker"):
        return Task.OTHER
    return None


# ---------------------------------------------------------------------------
# 参数提取
# ---------------------------------------------------------------------------
def _infer_kind(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if value is None:
        return "none"
    if isinstance(value, (list, tuple)):
        return "sequence"
    return "other"


def _safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__name__}>"


def _make_param_spec(name: str, default: Any, required: bool) -> ParamSpec:
    kind = _infer_kind(default)
    choices = PARAM_CHOICES.get(name)
    if choices and kind in ("str", "none"):
        kind = "choice"
    return ParamSpec(
        name=name,
        kind=kind,
        default=default,
        default_repr=_safe_repr(default),
        choices=choices,
        required=required,
    )


def _extract_params(obj: type) -> list[ParamSpec]:
    try:
        sig = inspect.signature(obj.__init__)
    except (ValueError, TypeError):
        return []
    specs: list[ParamSpec] = []
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        required = p.default is inspect.Parameter.empty
        default = None if required else p.default
        specs.append(_make_param_spec(name, default, required))
    return specs


def _extract_nested_estimator_params(obj: type) -> list[str]:
    found: list[str] = []
    try:
        sig = inspect.signature(obj.__init__)
    except (ValueError, TypeError):
        return found
    for name, p in sig.parameters.items():
        if name == "self" or p.kind in (inspect.Parameter.VAR_POSITIONAL,
                                        inspect.Parameter.VAR_KEYWORD):
            continue
        if name in _NESTED_ESTIMATOR_PARAM_NAMES:
            found.append(name)
    return found


# ---------------------------------------------------------------------------
# 构建与缓存
# ---------------------------------------------------------------------------
def _build_spec(obj: type, scan_module: str) -> EstimatorSpec | None:
    task = _classify(obj, scan_module)
    if task is None:
        return None
    params = _extract_params(obj)
    nested = _extract_nested_estimator_params(obj)
    required_nested = {p.name for p in params if p.name in nested and p.required}
    is_meta = bool(required_nested) or obj.__name__ in _FORCE_META_CLASSES
    return EstimatorSpec(
        name=obj.__name__,
        qualname=f"{scan_module}.{obj.__name__}",
        module=obj.__module__,
        scan_module=scan_module,
        task=task,
        is_meta=is_meta,
        doc=(obj.__doc__ or "").strip().split("\n\n")[0],
        params=params,
        nested_estimator_params=nested,
    )


_CACHE: dict[str, EstimatorSpec] | None = None


def _discover_all() -> dict[str, EstimatorSpec]:
    found: dict[str, tuple[int, EstimatorSpec]] = {}

    for module_name in _iter_sklearn_modules():
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        try:
            members = inspect.getmembers(module, inspect.isclass)
        except Exception:
            members = []

        extra_members: list[tuple[str, type]] = []
        all_names = getattr(module, "__all__", None) or []
        for attr_name in all_names:
            if not isinstance(attr_name, str) or attr_name.startswith("_"):
                continue
            try:
                obj = getattr(module, attr_name)
            except Exception:
                continue
            if inspect.isclass(obj):
                extra_members.append((attr_name, obj))

        seen_names: set[str] = set()
        combined: list[tuple[str, type]] = []
        for name, obj in list(members) + extra_members:
            if name in seen_names:
                continue
            seen_names.add(name)
            combined.append((name, obj))

        for name, obj in combined:
            if not _is_candidate(obj, module_name):
                continue
            spec = _build_spec(obj, module_name)
            if spec is None:
                continue
            prev = found.get(name)
            if prev is None or len(module_name) < prev[0]:
                found[name] = (len(module_name), spec)

    return {name: spec for name, (_, spec) in found.items()}


def get_registry(refresh: bool = False) -> dict[str, EstimatorSpec]:
    global _CACHE
    if _CACHE is None or refresh:
        _CACHE = _discover_all()
        logger.info("算法注册表构建完成：%d 个估计器", len(_CACHE))
    return dict(_CACHE)


def get_estimator(name: str) -> EstimatorSpec:
    registry = get_registry()
    if name not in registry:
        raise KeyError(f"未注册的估计器：{name}")
    return registry[name]


def list_by_task(task: str) -> list[EstimatorSpec]:
    return sorted(
        (s for s in get_registry().values() if s.task == task),
        key=lambda s: s.name,
    )