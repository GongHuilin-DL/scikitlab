# diagnose.py
import importlib
import inspect
import pkgutil
from pathlib import Path

import sklearn

from core import registry


def line(title):
    print("\n" + "=" * 10 + " " + title + " " + "=" * 10)


# 1) 文件系统
line("1. 文件系统")
root = Path(sklearn.__file__).parent
for t in ["linear_model/_logistic.py", "linear_model/_coordinate_descent.py", "linear_model/_bayes.py"]:
    print(f"  {t}: exists={(root / t).exists()}")

# 2) 导入 + __dict__
line("2. 导入与 __dict__")
for t, cn in [
    ("sklearn.linear_model._logistic", "LogisticRegression"),
    ("sklearn.linear_model._coordinate_descent", "LassoCV"),
]:
    try:
        mod = importlib.import_module(t)
        present = cn in mod.__dict__
        print(f"  {t}: import ok, {cn} in __dict__={present}")
        if present:
            cls = mod.__dict__[cn]
            print(f"    __module__ = {cls.__module__!r}")
    except Exception as e:
        print(f"  {t}: import failed: {type(e).__name__}: {e}")

# 3) walk_packages
line("3. pkgutil.walk_packages")
walked = {info.name for info in pkgutil.walk_packages(sklearn.__path__, prefix="sklearn.")}
for t in ["sklearn.linear_model._logistic", "sklearn.linear_model._coordinate_descent"]:
    print(f"  {t} in walk_packages: {t in walked}")

# 4) 我们的 _iter_sklearn_modules
line("4. _iter_sklearn_modules")
iterated = set(registry._iter_sklearn_modules())
for t in ["sklearn.linear_model._logistic", "sklearn.linear_model._coordinate_descent"]:
    print(f"  {t} yielded: {t in iterated}")

# 5) 直接测 _is_candidate 和 _build_spec
line("5. _is_candidate 与 _build_spec")
for t, cn in [
    ("sklearn.linear_model._logistic", "LogisticRegression"),
    ("sklearn.linear_model._coordinate_descent", "LassoCV"),
]:
    try:
        mod = importlib.import_module(t)
        cls = mod.__dict__[cn]
        print(f"  {t}.{cn}:")
        print(f"    _is_candidate = {registry._is_candidate(cls, t)}")
        spec = registry._build_spec(cls, t)
        print(f"    _build_spec = {spec.name if spec else None}")
    except Exception as e:
        print(f"  {t}: {type(e).__name__}: {e}")

# 6) 最终注册表
line("6. get_registry")
r = registry.get_registry(refresh=True)
for n in ["LogisticRegression", "LassoCV", "ElasticNetCV",
          "LogisticRegressionCV", "MultiTaskLassoCV", "MultiTaskElasticNetCV"]:
    print(f"  {n} in registry: {n in r}")