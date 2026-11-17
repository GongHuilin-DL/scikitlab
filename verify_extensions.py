# verify_extensions.py
import pandas as pd
from sklearn.datasets import load_iris
from core.dataset import build_bundle_from_dataframe
from core.trainer import TrainingSession, run_training
from core.registry import Task, get_registry, _available_packages

# 1) 类别特征流程
df = load_iris(as_frame=True).frame
df["cat_feature"] = (df["sepal length (cm)"] > 5.8).map({True: "high", False: "low"})
features = [c for c in df.columns if c != "target"]
bundle = build_bundle_from_dataframe(df, features, "target", Task.CLASSIFICATION)
print("categorical:", bundle.categorical_features)
print("numeric:", bundle.numeric_features)

sess = TrainingSession(
    algorithm="LogisticRegression",
    task=Task.CLASSIFICATION,
    params={"max_iter": 300},
    X_train=bundle.X_train, y_train=bundle.y_train,
    X_test=bundle.X_test, y_test=bundle.y_test,
    feature_names=bundle.feature_names,
    preprocessing_spec=bundle.preprocessing_spec,
)
for _ in run_training(sess):
    pass
print("training status:", sess.status, "| accuracy:", sess.history[-1]["val"]["accuracy"])

# 2) 元估计器
print("\nmeta available in registry:", [n for n, s in get_registry().items() if s.is_meta][:5])

# 3) 第三方包
print("\nscanned packages:", _available_packages())