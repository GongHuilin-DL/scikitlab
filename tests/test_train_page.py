"""训练页辅助函数测试（不启动 Gradio 界面）。"""
import unittest

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from core.registry import Task, get_registry
from core.trainer import TrainingSession, run_training
from ui.page_train import (
    _build_figure,
    _format_metrics,
    _format_status,
    _predict_mode,
)


class FormatTest(unittest.TestCase):
    def test_format_status_empty(self):
        self.assertIn("训练准备中", _format_status(None))

    def test_format_status_running_batch(self):
        snap = {
            "status": "running", "algorithm": "SVC", "mode": "batch",
            "elapsed": 1.23, "error": None,
        }
        text = _format_status(snap)
        self.assertIn("running", text)
        self.assertIn("批量算法", text)

    def test_format_metrics_empty(self):
        df = _format_metrics(None)
        self.assertIn("提示", df.columns)

    def test_format_metrics_with_values(self):
        snap = {
            "current": {
                "train": {"accuracy": 0.9},
                "val": {"accuracy": 0.85, "f1_macro": 0.84},
            }
        }
        df = _format_metrics(snap)
        self.assertEqual(len(df), 3)
        self.assertIn("训练集", df["数据集"].values)
        self.assertIn("测试集", df["数据集"].values)


class FigureTest(unittest.TestCase):
    def test_empty_history_renders_placeholder(self):
        fig = _build_figure([], "LogisticRegression")
        self.assertEqual(len(fig.axes), 1)

    def test_batch_history_with_metrics(self):
        history = [{
            "step": 1,
            "train": {"accuracy": 0.9},
            "val": {"accuracy": 0.85},
        }]
        fig = _build_figure(history, "LogisticRegression")
        self.assertGreaterEqual(len(fig.axes), 1)

    def test_incremental_history_multiple_epochs(self):
        history = [
            {"step": 1, "train": {"accuracy": 0.6}, "val": {"accuracy": 0.55}},
            {"step": 2, "train": {"accuracy": 0.75}, "val": {"accuracy": 0.72}},
            {"step": 3, "train": {"accuracy": 0.9}, "val": {"accuracy": 0.85}},
        ]
        fig = _build_figure(history, "SGDClassifier")
        ax = fig.axes[0]
        line = ax.get_lines()[0]
        self.assertEqual(len(line.get_xdata()), 3)

    def test_intrinsic_history_plot(self):
        history = [
            {"step": 1, "source": "train_score_", "value": 0.5},
            {"step": 2, "source": "train_score_", "value": 0.7},
        ]
        fig = _build_figure(history, "GradientBoostingClassifier")
        self.assertGreaterEqual(len(fig.axes), 1)


class EndToEndFigureTest(unittest.TestCase):
    def test_incremental_training_figure(self):
        data = load_iris()
        X_tr, X_te, y_tr, y_te = train_test_split(
            data.data, data.target,
            test_size=0.3, random_state=0, stratify=data.target,
        )
        session = TrainingSession(
            algorithm="SGDClassifier",
            task=get_registry()["SGDClassifier"].task,
            params={"max_iter": 1, "random_state": 0},
            X_train=X_tr, y_train=y_tr,
            X_test=X_te, y_test=y_te,
            feature_names=list(data.feature_names),
            target_name="target",
        )
        list(run_training(session, epochs=3, batch_size=30))
        fig = _build_figure(session.history, "SGDClassifier")
        self.assertGreaterEqual(len(fig.axes), 1)


class PredictModeTest(unittest.TestCase):
    def test_no_dataset(self):
        mode, msg = _predict_mode(None, None)
        self.assertEqual(mode, "unknown")
        self.assertIn("尚未", msg)

    def test_batch_without_partial_fit(self):
        from core.dataset import build_bundle_from_dataframe

        data = load_iris(as_frame=True)
        df = data.frame
        features = [c for c in df.columns if c != "target"]
        dataset = build_bundle_from_dataframe(
            df, features, "target", Task.CLASSIFICATION,
            test_size=0.2, random_state=0,
        )
        spec = get_registry()["RandomForestClassifier"]
        mode, msg = _predict_mode(dataset, spec)
        self.assertEqual(mode, "batch")

    def test_incremental_no_preprocessing(self):
        from core.dataset import build_bundle_from_dataframe

        data = load_iris(as_frame=True)
        df = data.frame
        features = [c for c in df.columns if c != "target"]
        dataset = build_bundle_from_dataframe(
            df, features, "target", Task.CLASSIFICATION,
            test_size=0.2, random_state=0,
        )
        # iris 有数值特征 → 会构建 preprocessing_spec → 触发 Pipeline → batch
        spec = get_registry()["SGDClassifier"]
        mode, msg = _predict_mode(dataset, spec)
        self.assertEqual(mode, "batch")
        self.assertIn("Pipeline", msg)


if __name__ == "__main__":
    unittest.main()