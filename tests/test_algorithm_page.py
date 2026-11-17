"""算法页辅助逻辑测试（不启动 Gradio 界面）。"""
import unittest

from core.registry import Task, get_registry
from ui.page_algorithm import _algo_choices_for_task


class AlgorithmChoicesTest(unittest.TestCase):
    def test_classification_choices_include_meta(self):
        """新设计：分类下拉既包含分类算法，也包含 meta 估计器供在该任务下配置。"""
        choices = _algo_choices_for_task(Task.CLASSIFICATION)
        self.assertIn("LogisticRegression", choices)
        self.assertIn("SVC", choices)
        self.assertIn("RandomForestClassifier", choices)
        # 元估计器在新设计下也出现在下拉框中
        self.assertIn("GridSearchCV", choices)
        self.assertIn("VotingClassifier", choices)

        registry = get_registry()
        for name in choices:
            spec = registry[name]
            self.assertIn(
                spec.task, (Task.CLASSIFICATION, Task.META),
                f"{name} 的 task={spec.task}，不应出现在分类下拉框中",
            )

    def test_regression_choices(self):
        choices = _algo_choices_for_task(Task.REGRESSION)
        self.assertIn("LinearRegression", choices)
        self.assertIn("SVR", choices)
        self.assertNotIn("LogisticRegression", choices)

    def test_clustering_choices(self):
        choices = _algo_choices_for_task(Task.CLUSTERING)
        self.assertIn("KMeans", choices)
        self.assertIn("DBSCAN", choices)
        self.assertNotIn("PCA", choices)

    def test_dimred_choices(self):
        choices = _algo_choices_for_task(Task.DIMRED)
        self.assertIn("PCA", choices)
        self.assertIn("TSNE", choices)

    def test_empty_task(self):
        self.assertEqual(_algo_choices_for_task(""), [])
        self.assertEqual(_algo_choices_for_task(None), [])

    def test_choices_task_matches_or_meta(self):
        """下拉框中每个条目的 task 必须与当前选择的任务匹配，或为通用 meta。"""
        registry = get_registry()
        for task in (Task.CLASSIFICATION, Task.REGRESSION,
                     Task.CLUSTERING, Task.DIMRED,
                     Task.ANOMALY, Task.TRANSFORMATION):
            for name in _algo_choices_for_task(task):
                spec = registry[name]
                self.assertIn(
                    spec.task, (task, Task.META),
                    f"{task} 下拉框不应包含 {name}（task={spec.task}）",
                )


if __name__ == "__main__":
    unittest.main()