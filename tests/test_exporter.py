"""导出器测试：生成的脚本可读、可解析、内容完整。"""
import ast
import tempfile
import unittest
from pathlib import Path

from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split

from core.dataset import build_bundle_from_dataframe
from core.exporter import export_training_script
from core.registry import Task, get_registry


def _make_dataset():
    data = load_iris(as_frame=True)
    df = data.frame
    features = [c for c in df.columns if c != "target"]
    return build_bundle_from_dataframe(
        df, features, "target", Task.CLASSIFICATION,
        test_size=0.3, random_state=0, split_mode="stratified",
    )


class ExporterTest(unittest.TestCase):
    def test_generates_valid_python(self):
        dataset = _make_dataset()
        spec = get_registry()["LogisticRegression"]
        with tempfile.TemporaryDirectory() as td:
            path = export_training_script(
                dataset, spec, {"max_iter": 200}, output_dir=Path(td)
            )
            self.assertTrue(path.exists())
            text = path.read_text(encoding="utf-8")
            # 必须能被 ast 解析（语法正确）
            ast.parse(text)
            # 关键结构存在
            self.assertIn("import sklearn.linear_model", text)
            self.assertIn("sklearn.linear_model.LogisticRegression(", text)
            self.assertIn("max_iter=200", text)
            self.assertIn("Pipeline", text)
            self.assertIn("ColumnTransformer", text)
            self.assertIn("pipeline.fit(X_train, y_train)", text)
            self.assertIn("accuracy_score", text)
            self.assertIn("joblib.dump", text)

    def test_data_snapshot_created(self):
        dataset = _make_dataset()
        spec = get_registry()["LogisticRegression"]
        with tempfile.TemporaryDirectory() as td:
            path = export_training_script(dataset, spec, {}, output_dir=Path(td))
            csv_files = list(Path(td).glob("data_*.csv"))
            self.assertEqual(len(csv_files), 1)
            # 脚本内引用的数据文件名与快照一致
            self.assertIn(csv_files[0].name, path.read_text(encoding="utf-8"))

    def test_regression_script_metrics(self):
        d = load_diabetes(as_frame=True)
        df = d.frame
        features = [c for c in df.columns if c != "target"]
        dataset = build_bundle_from_dataframe(
            df, features, "target", Task.REGRESSION,
            test_size=0.2, random_state=0, split_mode="random",
        )
        spec = get_registry()["LinearRegression"]
        with tempfile.TemporaryDirectory() as td:
            path = export_training_script(dataset, spec, {}, output_dir=Path(td))
            text = path.read_text(encoding="utf-8")
            self.assertIn("mean_squared_error", text)
            self.assertIn("r2_score", text)
            ast.parse(text)

    def test_snapshot_roundtrip_runs(self):
        """导出的 CSV 能被脚本逻辑读出正确的 train/test 划分。"""
        import pandas as pd
        dataset = _make_dataset()
        spec = get_registry()["LogisticRegression"]
        with tempfile.TemporaryDirectory() as td:
            path = export_training_script(
                dataset, spec, {"max_iter": 200}, output_dir=Path(td)
            )
            csv_path = next(Path(td).glob("data_*.csv"))
            df = pd.read_csv(csv_path)
            train = df[df["_split"] == "train"].drop(columns=["_split"])
            test = df[df["_split"] == "test"].drop(columns=["_split"])
            self.assertEqual(len(train), len(dataset.X_train))
            self.assertEqual(len(test), len(dataset.X_test))
            self.assertIn("target", train.columns)


if __name__ == "__main__":
    unittest.main()