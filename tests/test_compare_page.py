"""对比页辅助函数测试。"""
import tempfile
import unittest
from pathlib import Path

from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split

from core.persistence import list_models, save_model
from core.registry import get_registry
from core.trainer import TrainingSession, run_training
from ui.page_compare import (
    _ALL_TASK,
    _build_compare_dataframe,
    _build_elapsed_figure,
    _build_metric_figure,
    _format_metrics,
)


def _train_and_save(
    tmpdir: Path, name: str, algo: str,
    X_tr, X_te, y_tr, y_te, features, task,
    params: dict | None = None,
) -> None:
    spec = get_registry()[algo]
    if params is None:
        params = {"max_iter": 300} if algo == "LogisticRegression" else {}
    session = TrainingSession(
        algorithm=algo, task=task, params=params,
        X_train=X_tr, y_train=y_tr,
        X_test=X_te, y_test=y_te,
        feature_names=list(features), target_name="target",
    )
    list(run_training(session))
    save_model(session, name, tmpdir)


class FormatTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_format_metrics(None), "-")
        self.assertEqual(_format_metrics({}), "-")

    def test_values(self):
        s = _format_metrics({"accuracy": 0.95, "f1_macro": 0.9})
        self.assertIn("accuracy=0.9500", s)
        self.assertIn("f1_macro=0.9000", s)


class BuildCompareDataframeTest(unittest.TestCase):
    def test_empty_models(self):
        df = _build_compare_dataframe([], _ALL_TASK)
        self.assertIn("提示", df.columns)

    def test_with_models(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            data = load_iris()
            X_tr, X_te, y_tr, y_te = train_test_split(
                data.data, data.target, test_size=0.3,
                random_state=0, stratify=data.target,
            )
            _train_and_save(
                tmpdir, "lr_iris", "LogisticRegression",
                X_tr, X_te, y_tr, y_te, data.feature_names, "classification",
            )
            _train_and_save(
                tmpdir, "rf_iris", "RandomForestClassifier",
                X_tr, X_te, y_tr, y_te, data.feature_names, "classification",
            )
            models = list_models(tmpdir)
            df = _build_compare_dataframe(models, _ALL_TASK)
            self.assertEqual(len(df), 2)
            self.assertIn("模型名", df.columns)
            self.assertIn("lr_iris", df["模型名"].values)
            self.assertIn("rf_iris", df["模型名"].values)

    def test_task_filter(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            iris = load_iris()
            X_tr, X_te, y_tr, y_te = train_test_split(
                iris.data, iris.target, test_size=0.3,
                random_state=0, stratify=iris.target,
            )
            _train_and_save(
                tmpdir, "lr_iris", "LogisticRegression",
                X_tr, X_te, y_tr, y_te, iris.feature_names, "classification",
            )
            dia = load_diabetes()
            X_tr, X_te, y_tr, y_te = train_test_split(
                dia.data, dia.target, test_size=0.3, random_state=0,
            )
            _train_and_save(
                tmpdir, "lin_dia", "LinearRegression",
                X_tr, X_te, y_tr, y_te, dia.feature_names, "regression",
            )

            models = list_models(tmpdir)
            df_all = _build_compare_dataframe(models, _ALL_TASK)
            self.assertEqual(len(df_all), 2)

            df_cls = _build_compare_dataframe(models, "classification")
            self.assertEqual(len(df_cls), 1)
            self.assertEqual(df_cls["模型名"].values[0], "lr_iris")

            df_reg = _build_compare_dataframe(models, "regression")
            self.assertEqual(len(df_reg), 1)
            self.assertEqual(df_reg["模型名"].values[0], "lin_dia")


class FigureTest(unittest.TestCase):
    def test_elapsed_empty(self):
        fig = _build_elapsed_figure([], _ALL_TASK)
        self.assertGreaterEqual(len(fig.axes), 1)

    def test_metric_empty(self):
        fig = _build_metric_figure([], _ALL_TASK, "accuracy")
        self.assertGreaterEqual(len(fig.axes), 1)

    def test_elapsed_with_data(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            data = load_iris()
            X_tr, X_te, y_tr, y_te = train_test_split(
                data.data, data.target, test_size=0.3,
                random_state=0, stratify=data.target,
            )
            _train_and_save(
                tmpdir, "m1", "LogisticRegression",
                X_tr, X_te, y_tr, y_te, data.feature_names, "classification",
            )
            _train_and_save(
                tmpdir, "m2", "SVC",
                X_tr, X_te, y_tr, y_te, data.feature_names, "classification",
            )
            models = list_models(tmpdir)
            fig = _build_elapsed_figure(models, _ALL_TASK)
            self.assertGreaterEqual(len(fig.axes), 1)
            self.assertGreater(len(fig.axes[0].patches), 0)


if __name__ == "__main__":
    unittest.main()