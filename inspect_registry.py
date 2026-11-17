"""打印注册表统计，用于快速验证算法发现覆盖率。"""
from collections import Counter

from core.registry import TASK_LABELS, get_registry

# 这些类必须出现在注册表中，否则说明发现路径有遗漏
MUST_HAVE = [
    "LogisticRegression", "LinearRegression", "Ridge", "Lasso", "ElasticNet",
    "LassoCV", "ElasticNetCV", "LogisticRegressionCV",
    "MultiTaskLassoCV", "MultiTaskElasticNetCV",
    "KMeans", "DBSCAN", "PCA", "SVC", "RandomForestClassifier",
    "GaussianNB", "DecisionTreeRegressor", "StandardScaler", "SelectKBest",
    "IsolationForest", "GaussianMixture", "TSNE",
]

# 这些内部/测试类不应出现在注册表中
MUST_NOT_HAVE = [
    "CheckingClassifier", "BaseEstimator", "ClassifierMixin",
]


def main() -> None:
    registry = get_registry(refresh=True)
    print(f"共发现 {len(registry)} 个估计器\n")

    counts = Counter(spec.task for spec in registry.values())
    for task, label in TASK_LABELS.items():
        if counts.get(task):
            print(f"  {label:<10} {counts[task]:>4}")

    missing = [n for n in MUST_HAVE if n not in registry]
    print("\n关键公开估计器检查：")
    if missing:
        print("  [缺失] " + ", ".join(missing))
    else:
        print("  全部存在")

    polluted = [n for n in MUST_NOT_HAVE if n in registry]
    print("\n内部类污染检查：")
    if polluted:
        print("  [误收] " + ", ".join(polluted))
    else:
        print("  未发现内部类")

    print("\n元估计器（需要嵌套估计器实例，UI 层将单独处理）：")
    metas = sorted(n for n, s in registry.items() if s.is_meta)
    print("  " + ", ".join(metas) if metas else "  无")

    print("\n全部条目：")
    for name in sorted(registry):
        spec = registry[name]
        flag = " [meta]" if spec.is_meta else ""
        print(f"  {name:<38} {TASK_LABELS[spec.task]:<10}{flag}  params={len(spec.params)}")


if __name__ == "__main__":
    main()