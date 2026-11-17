"""数据层单元测试：玩具数据集、推断、划分、类别分布、警告。"""
import unittest

import numpy as np
import pandas as pd

from core.dataset import (
    TOY_DATASETS,
    build_bundle_from_dataframe,
    infer_task,
    load_toy,
)
from core.registry import Task


class ToyDatasetTest(unittest.TestCase):
    def test_all_toy_datasets_load(self):
        for key in TOY_DATASETS:
            df, target, hint = load_toy(key)
            self.assertGreater(len(df), 0, key)
            self.assertIn(target, df.columns, key)
            self.assertIn(hint, (Task.CLASSIFICATION, Task.REGRESSION), key)


class TaskInferenceTest(unittest.TestCase):
    def test_binary_bool_like_int_is_classification(self):
        s = pd.Series([0, 1, 1, 0] * 50)
        self.assertEqual(infer_task(s, 200), Task.CLASSIFICATION)

    def test_continuous_float_is_regression(self):
        rng = np.random.RandomState(0)
        s = pd.Series(rng.randn(200))
        self.assertEqual(infer_task(s, 200), Task.REGRESSION)

    def test_object_is_classification(self):
        s = pd.Series(["a", "b", "c"] * 50)
        self.assertEqual(infer_task(s, 150), Task.CLASSIFICATION)

    def test_none_is_unsupervised(self):
        self.assertEqual(infer_task(None, 100), "unsupervised")


class BuildBundleTest(unittest.TestCase):
    def test_iris_stratified(self):
        df, target, hint = load_toy("Iris（鸢尾花 · 分类）")
        features = [c for c in df.columns if c != target]
        bundle = build_bundle_from_dataframe(
            df, features, target, hint,
            test_size=0.3, random_state=0, split_mode="stratified",
        )
        self.assertEqual(bundle.task_hint, Task.CLASSIFICATION)
        self.assertEqual(len(bundle.target_classes), 3)
        for c in bundle.target_classes:
            self.assertGreater(bundle.class_distribution["train"][str(c)], 0)
            self.assertGreater(bundle.class_distribution["test"][str(c)], 0)
        self.assertEqual(bundle.X_train.shape[1], len(features))

    def test_diabetes_regression(self):
        df, target, hint = load_toy("Diabetes（糖尿病进展 · 回归）")
        features = [c for c in df.columns if c != target]
        bundle = build_bundle_from_dataframe(
            df, features, target, hint,
            test_size=0.2, random_state=0, split_mode="random",
        )
        self.assertEqual(bundle.task_hint, Task.REGRESSION)
        self.assertIsNone(bundle.target_classes)
        self.assertEqual(bundle.y_train.dtype.kind, "f")

    def test_sequential_split_warns_on_sorted_classes(self):
        df, target, hint = load_toy("Iris（鸢尾花 · 分类）")
        features = [c for c in df.columns if c != target]
        bundle = build_bundle_from_dataframe(
            df, features, target, hint,
            test_size=0.2, random_state=0, split_mode="sequential",
        )
        self.assertTrue(any("缺失类别" in w for w in bundle.warnings))

    def test_unsupervised_without_target(self):
        df, _, _ = load_toy("Iris（鸢尾花 · 分类）")
        features = [c for c in df.columns if c != "target"]
        bundle = build_bundle_from_dataframe(
            df, features, None, "auto",
            test_size=0.2, random_state=0, split_mode="random",
        )
        self.assertEqual(bundle.task_hint, "unsupervised")
        self.assertIsNone(bundle.y_train)

    def test_non_numeric_feature_is_categorical(self):
        # 用 20 行数据确保 stratified split 时每类在测试集里都有样本
        df = pd.DataFrame({
            "a": list(range(20)),
            "b": ["x", "y", "z", "w"] * 5,
            "y": [0, 1] * 10,
        })
        bundle = build_bundle_from_dataframe(
            df, ["a", "b"], "y", Task.CLASSIFICATION,
            test_size=0.3, random_state=0, split_mode="random",
        )
        self.assertEqual(bundle.numeric_features, ["a"])
        self.assertEqual(bundle.categorical_features, ["b"])
        # 类别特征应触发警告，但不阻塞
        self.assertTrue(any("类别特征" in w for w in bundle.warnings))


if __name__ == "__main__":
    unittest.main()