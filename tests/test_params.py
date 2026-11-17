"""参数组件与值转换测试。"""
import unittest

from core.params import build_component, coerce_value
from core.registry import ParamSpec


def _spec(name, kind, default, default_repr, choices=None):
    return ParamSpec(
        name=name, kind=kind, default=default,
        default_repr=default_repr, choices=choices, required=False,
    )


class BuildComponentTest(unittest.TestCase):
    def test_all_components_interactive(self):
        cases = [
            _spec("flag", "bool", False, "False"),
            _spec("n", "int", 10, "10"),
            _spec("rate", "float", 0.5, "0.5"),
            _spec("mode", "choice", "x", "'x'", choices=["x", "y"]),
            _spec("tag", "str", "z", "'z'"),
            _spec("limit", "none", None, "None"),
        ]
        for s in cases:
            comp = build_component(s)
            self.assertTrue(
                getattr(comp, "interactive", False),
                f"{s.name} 组件 interactive 应为 True",
            )

    def test_info_shows_default(self):
        comp = build_component(_spec("n", "int", 10, "10"))
        self.assertIn("默认值", comp.info)

    def test_none_kind_info_hint(self):
        comp = build_component(_spec("max_depth", "none", None, "None"))
        self.assertIn("None", comp.info)


class CoerceValueTest(unittest.TestCase):
    def test_none_string_returns_none(self):
        spec = _spec("max_depth", "none", None, "None")
        self.assertIsNone(coerce_value(spec, "None"))
        self.assertIsNone(coerce_value(spec, "none"))
        self.assertIsNone(coerce_value(spec, "NONE"))
        self.assertIsNone(coerce_value(spec, ""))
        self.assertIsNone(coerce_value(spec, None))

    # def test_none_kind_numeric_strings(self):
    #     spec = _spec("max_depth", "none", None, "None")
    #     self.assertEqual(coerce_value(spec, "5"), 5)
    #     self.assertEqual(coerce_value(spec, "1.5"), 1.5)
    #     self.assertEqual(coerce_value(spec, "true"), "true")  # 非 None 且非数值回退字符串

    def test_none_kind_numeric_strings(self):
        spec = _spec("max_depth", "none", None, "None")
        self.assertEqual(coerce_value(spec, "5"), 5)
        self.assertEqual(coerce_value(spec, "1.5"), 1.5)
        # json.loads 会把 JSON 字面量解析为对应 Python 对象
        self.assertIs(coerce_value(spec, "true"), True)
        self.assertIs(coerce_value(spec, "false"), False)

    def test_none_kind_plain_string_kept(self):
        spec = _spec("max_depth", "none", None, "None")
        # 非 JSON 字面量的字符串原样返回，由 sklearn 自行报错
        self.assertEqual(coerce_value(spec, "gini"), "gini")
        self.assertEqual(coerce_value(spec, "random_state"), "random_state")

    def test_none_kind_default_preserved(self):
        spec = _spec("max_depth", "none", None, "None")
        self.assertIsNone(coerce_value(spec, "None"))
        # 用户没有修改时传回 default_repr，应回退到 default
        spec2 = _spec("threshold", "none", 0.5, "0.5")
        self.assertEqual(coerce_value(spec2, "0.5"), 0.5)


if __name__ == "__main__":
    unittest.main()