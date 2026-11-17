"""训练引擎烟雾测试：批量 + 增量两种模式。"""
import unittest

import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split

from core.registry import Task, get_registry
from core.trainer import TrainingSession, instantiate_estimator, run_training


class TrainerBatchTest(unittest.TestCase):
    def test_logistic_regression_batch(self):
        data = load_iris()
        # iris 按类别排序，必须分层抽样，否则训练集与测试集类别不相交
        X_tr, X_te, y_tr, y_te = train_test_split(
            data.data, data.target,
            test_size=0.3, random_state=0, stratify=data.target,
        )
        spec = get_registry()["LogisticRegression"]
        session = TrainingSession(
            algorithm="LogisticRegression",
            task=spec.task,
            params={"max_iter": 200},
            X_train=X_tr,
            y_train=y_tr,
            X_test=X_te,
            y_test=y_te,
            feature_names=list(data.feature_names),
            target_name="target",
        )
        final = None
        for snap in run_training(session):
            final = snap
        self.assertIsNotNone(final)
        self.assertEqual(final["status"], "done", final.get("error"))
        self.assertEqual(final["mode"], "batch")
        metrics = final["current"]["val"]
        self.assertIn("accuracy", metrics)
        self.assertGreater(metrics["accuracy"], 0.8)


class TrainerIncrementalTest(unittest.TestCase):
    def test_sgd_incremental(self):
        data = load_diabetes()
        spec = get_registry()["SGDRegressor"]
        session = TrainingSession(
            algorithm="SGDRegressor",
            task=spec.task,
            params={"max_iter": 1, "random_state": 0},
            X_train=data.data[:300],
            y_train=data.target[:300],
            X_test=data.data[300:],
            y_test=data.target[300:],
            feature_names=list(data.feature_names),
            target_name="target",
        )
        snaps = list(run_training(session, epochs=5, batch_size=50))
        self.assertEqual(snaps[-1]["status"], "done")
        self.assertEqual(snaps[-1]["mode"], "incremental")
        self.assertEqual(len(session.history), 5)
        self.assertIn("mse", session.history[-1]["val"])


class EstimatorInstantiationTest(unittest.TestCase):
    def test_instantiate_from_spec(self):
        spec = get_registry()["KMeans"]
        est = instantiate_estimator(spec, {"n_clusters": 3, "random_state": 0})
        self.assertEqual(est.n_clusters, 3)


if __name__ == "__main__":
    unittest.main()