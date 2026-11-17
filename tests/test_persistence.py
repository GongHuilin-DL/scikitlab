"""模型保存/载入/推理一致性测试。"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from core.persistence import delete_model, list_models, load_model, save_model
from core.registry import get_registry
from core.trainer import TrainingSession, run_training


def _train_small_model(tmpdir: Path) -> TrainingSession:
    data = load_iris()
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.data, data.target,
        test_size=0.3, random_state=0, stratify=data.target,
    )
    spec = get_registry()["LogisticRegression"]
    session = TrainingSession(
        algorithm="LogisticRegression",
        task=spec.task,
        params={"max_iter": 300},
        X_train=X_tr, y_train=y_tr,
        X_test=X_te, y_test=y_te,
        feature_names=list(data.feature_names),
        target_name="target",
    )
    list(run_training(session))
    save_model(session, "iris_lr", tmpdir, notes="smoke")
    return session


class PersistenceTest(unittest.TestCase):
    def test_save_load_predict_consistency(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            session = _train_small_model(tmpdir)

            self.assertTrue((tmpdir / "iris_lr.joblib").exists())
            self.assertTrue((tmpdir / "iris_lr.json").exists())

            loaded = load_model("iris_lr", tmpdir)
            X = session.X_test
            self.assertTrue(np.array_equal(session.estimator.predict(X),
                                           loaded.estimator.predict(X)))
            self.assertEqual(loaded.metadata.algorithm, "LogisticRegression")
            self.assertEqual(loaded.metadata.params_before_fit["max_iter"], 300)
            self.assertEqual(loaded.metadata.n_features, 4)

    def test_list_and_delete(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            _train_small_model(tmpdir)
            models = list_models(tmpdir)
            self.assertEqual(len(models), 1)
            self.assertTrue(delete_model("iris_lr", tmpdir))
            self.assertEqual(len(list_models(tmpdir)), 0)

    def test_missing_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                load_model("nope", Path(td))


if __name__ == "__main__":
    unittest.main()