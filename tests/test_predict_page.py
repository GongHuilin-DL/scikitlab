"""推理页辅助函数测试。"""
import tempfile
import unittest
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from core.persistence import save_model
from core.registry import get_registry
from core.trainer import TrainingSession, run_training
from ui.page_predict import _refresh_choices


def _train_and_save(tmpdir: Path, name: str = "test_lr") -> None:
    data = load_iris()
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.data, data.target, test_size=0.3,
        random_state=0, stratify=data.target,
    )
    spec = get_registry()["LogisticRegression"]
    session = TrainingSession(
        algorithm="LogisticRegression",
        task=spec.task,
        params={"max_iter": 200},
        X_train=X_tr, y_train=y_tr,
        X_test=X_te, y_test=y_te,
        feature_names=list(data.feature_names),
        target_name="target",
    )
    list(run_training(session))
    save_model(session, name, tmpdir)


class RefreshChoicesTest(unittest.TestCase):
    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as td:
            choices, value, msg = _refresh_choices(Path(td))
            self.assertEqual(choices, [])
            self.assertIsNone(value)
            self.assertIn("暂无", msg)

    def test_with_one_model(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            _train_and_save(tmpdir)
            choices, value, msg = _refresh_choices(tmpdir)
            self.assertEqual(choices, ["test_lr"])
            self.assertEqual(value, "test_lr")
            self.assertIn("1", msg)

    def test_with_multiple_models(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            _train_and_save(tmpdir, "model_a")
            _train_and_save(tmpdir, "model_b")
            choices, value, msg = _refresh_choices(tmpdir)
            self.assertEqual(sorted(choices), ["model_a", "model_b"])
            self.assertIn("2", msg)


if __name__ == "__main__":
    unittest.main()