"""评估页图表构建测试（不启动 Gradio 界面）。"""
import unittest

import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from ui.page_evaluate import (
    _build_classification_figure,
    _build_clustering_figure,
    _build_importance_figure,
    _build_regression_figure,
)


class ClassificationFigureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = load_iris()
        X_tr, X_te, y_tr, y_te = train_test_split(
            data.data, data.target, test_size=0.3,
            random_state=0, stratify=data.target,
        )
        cls.est = LogisticRegression(max_iter=300).fit(X_tr, y_tr)
        cls.X, cls.y = X_te, y_te
        cls.features = list(data.feature_names)
        cls.classes = list(np.unique(data.target))

    def test_confusion_and_roc(self):
        fig = _build_classification_figure(
            self.est, self.X, self.y, self.classes, "LogisticRegression"
        )
        titles = [ax.get_title() for ax in fig.axes]
        self.assertIn("Confusion Matrix", titles)
        self.assertIn("ROC Curve", titles)

    def test_importance_with_coef(self):
        fig = _build_importance_figure(self.est, self.features)
        self.assertGreaterEqual(len(fig.axes), 1)

    def test_importance_without_attr(self):
        svc = SVC(kernel="rbf").fit(self.X, self.y)
        fig = _build_importance_figure(svc, self.features)
        self.assertEqual(len(fig.axes), 1)
        self.assertFalse(fig.axes[0].axison)

    # def test_roc_without_proba_or_decision(self):
    #     class NoProba:
    #         def predict(self, X):
    #             return np.zeros(len(X), dtype=int)

    #     fig = _build_classification_figure(
    #         NoProba(), self.X, self.y, self.classes, "NoProba"
    #     )
    #     titles = [ax.get_title() for ax in fig.axes]
    #     self.assertIn("Confusion Matrix", titles)
    #     self.assertIn("ROC Curve", titles)

    def test_roc_without_proba_or_decision(self):
        class NoProba:
            def predict(self, X):
                return np.zeros(len(X), dtype=int)

        fig = _build_classification_figure(
            NoProba(), self.X, self.y, self.classes, "NoProba"
        )
        titles = [ax.get_title() for ax in fig.axes]
        self.assertIn("Confusion Matrix", titles)
        # 无 proba/decision 时 ROC 区域走 axis("off") 分支，无标题
        self.assertTrue(
            any(not ax.axison for ax in fig.axes),
            "预期 ROC 区域被关闭坐标轴",
        )

class RegressionFigureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = load_diabetes()
        X_tr, X_te, y_tr, y_te = train_test_split(
            data.data, data.target, test_size=0.3, random_state=0,
        )
        cls.est = LinearRegression().fit(X_tr, y_tr)
        cls.X, cls.y = X_te, y_te
        cls.features = list(data.feature_names)

    def test_regression_diagnostic(self):
        fig = _build_regression_figure(
            self.est, self.X, self.y, "LinearRegression"
        )
        self.assertEqual(len(fig.axes), 2)

    def test_regression_importance(self):
        fig = _build_importance_figure(self.est, self.features)
        self.assertGreaterEqual(len(fig.axes), 1)


class ClusteringFigureTest(unittest.TestCase):
    def test_labels_missing(self):
        class NoLabels:
            pass

        data = load_iris()
        fig = _build_clustering_figure(NoLabels(), data.data, "Stub")
        self.assertEqual(len(fig.axes), 1)
        self.assertFalse(fig.axes[0].axison)


class PipelineImportanceTest(unittest.TestCase):
    def test_aggregate_importance_back_to_original_features(self):
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from ui.page_evaluate import _build_importance_figure

        df = pd.DataFrame({
            "num_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "cat_b": ["x", "y", "x", "y", "x", "y"],
            "target": [0, 1, 0, 1, 0, 1],
        })
        preproc = ColumnTransformer([
            ("num", StandardScaler(), ["num_a"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["cat_b"]),
        ])
        pipeline = Pipeline([
            ("preprocess", preproc),
            ("model", RandomForestClassifier(n_estimators=5, random_state=0)),
        ])
        pipeline.fit(df[["num_a", "cat_b"]], df["target"])

        fig = _build_importance_figure(pipeline, ["num_a", "cat_b"])
        # 图表正常绘制（未因维度不匹配进入"不可用"分支）
        ax = fig.axes[0]
        self.assertTrue(ax.axison)
        self.assertGreater(len(ax.patches), 0)
        labels = [t.get_text() for t in ax.get_yticklabels()]
        # 聚合后应只剩原始特征名 num_a / cat_b
        self.assertTrue(set(labels).issubset({"num_a", "cat_b"}))


if __name__ == "__main__":
    unittest.main()