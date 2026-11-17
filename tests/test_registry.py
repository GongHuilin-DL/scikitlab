"""算法注册表与参数提取的回归测试。"""
import unittest

from core.registry import Task, get_registry


class RegistryDiscoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = get_registry(refresh=True)

    def test_registry_not_empty(self):
        self.assertGreater(
            len(self.registry), 120,
            f"只发现 {len(self.registry)} 个估计器，扫描逻辑可能有问题",
        )

    def test_expected_estimators_present(self):
        expected = [
            "LogisticRegression", "LinearRegression", "KMeans", "PCA",
            "RandomForestClassifier", "SVC", "DBSCAN", "IsolationForest",
            "GaussianNB", "DecisionTreeRegressor", "GradientBoostingClassifier",
            "Ridge", "Lasso", "MLPClassifier", "StandardScaler", "SelectKBest",
        ]
        for name in expected:
            self.assertIn(name, self.registry, f"注册表缺少 {name}")

    def test_task_classification(self):
        cases = {
            "LogisticRegression": Task.CLASSIFICATION,
            "LinearRegression": Task.REGRESSION,
            "KMeans": Task.CLUSTERING,
            "PCA": Task.DIMRED,
            "IsolationForest": Task.ANOMALY,
            "StandardScaler": Task.TRANSFORMATION,
            "SelectKBest": Task.FEATURE_SELECTION,
        }
        for name, task in cases.items():
            self.assertIn(name, self.registry)
            self.assertEqual(self.registry[name].task, task, name)

    def test_params_extraction(self):
        lr = self.registry["LogisticRegression"]
        names = {p.name for p in lr.params}
        for n in ("C", "penalty", "solver", "max_iter"):
            self.assertIn(n, names)
        c = next(p for p in lr.params if p.name == "C")
        self.assertEqual(c.kind, "float")
        self.assertAlmostEqual(c.default, 1.0)

    def test_every_spec_is_complete(self):
        for name, spec in self.registry.items():
            self.assertTrue(spec.task, name)
            self.assertTrue(spec.qualname, name)
            self.assertIsInstance(spec.params, list, name)

    def test_no_private_or_base_classes(self):
        for name in self.registry:
            self.assertFalse(name.startswith("_"), name)
            self.assertFalse(name.startswith("Base"), name)

    def test_no_internal_test_classes(self):
        for name in ("CheckingClassifier", "BaseEstimator", "ClassifierMixin"):
            self.assertNotIn(name, self.registry, f"注册表不应包含内部类 {name}")

    def test_param_level_deprecation_not_excluded(self):
        """docstring 中参数级 '.. deprecated::' 说明不应导致类被排除。"""
        for name in ("LogisticRegression", "LogisticRegressionCV",
                     "LassoCV", "ElasticNetCV",
                     "MultiTaskLassoCV", "MultiTaskElasticNetCV"):
            self.assertIn(name, self.registry,
                          f"{name} 因 docstring 中的参数级弃用说明被误杀")

    def test_estimator_with_default_wrapper_is_not_meta(self):
        """estimator 参数有默认值且默认值可独立使用的包装类不算 meta。"""
        for name in ("RandomForestClassifier", "RandomForestRegressor",
                     "ExtraTreesClassifier", "ExtraTreesRegressor",
                     "GradientBoostingClassifier", "GradientBoostingRegressor",
                     "AdaBoostRegressor", "BaggingClassifier", "BaggingRegressor",
                     "IsolationForest", "CalibratedClassifierCV",
                     "RANSACRegressor", "TransformedTargetRegressor"):
            self.assertIn(name, self.registry)
            self.assertFalse(self.registry[name].is_meta,
                             f"{name} 应可独立使用，不应标记为 meta")

    def test_required_estimator_wrapper_is_meta(self):
        """estimator 参数无默认值、或默认值 None 会导致 fit 失败的估计器必须为 meta。"""
        for name in ("GridSearchCV", "RandomizedSearchCV",
                     "HalvingGridSearchCV", "HalvingRandomSearchCV",
                     "VotingClassifier", "VotingRegressor",
                     "StackingClassifier", "StackingRegressor",
                     "RFE", "RFECV", "SelectFromModel",
                     "SequentialFeatureSelector",
                     "OneVsRestClassifier", "OneVsOneClassifier",
                     "MultiOutputClassifier", "MultiOutputRegressor",
                     "OutputCodeClassifier",
                     "TunedThresholdClassifierCV",   # estimator 无默认值
                     "FixedThresholdClassifier"):    # estimator=None 时 fit 必抛错
            self.assertIn(name, self.registry)
            self.assertTrue(self.registry[name].is_meta,
                            f"{name} 必须由用户提供嵌套估计器，应标记为 meta")

    def test_force_meta_classes_marked(self):
        """_FORCE_META_CLASSES 中的每个类都必须真实存在于注册表且被标记为 meta。"""
        from core.registry import _FORCE_META_CLASSES
        for name in _FORCE_META_CLASSES:
            self.assertIn(name, self.registry, f"_FORCE_META_CLASSES 中的 {name} 未注册")
            self.assertTrue(self.registry[name].is_meta,
                            f"_FORCE_META_CLASSES 中的 {name} 未被标记为 meta")

if __name__ == "__main__":
    unittest.main()