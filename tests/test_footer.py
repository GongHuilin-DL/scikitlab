"""页脚烟雾测试：验证环境快照生成不抛错且包含关键字段。"""
import unittest

from app import _footer_markdown


class FooterTest(unittest.TestCase):
    def test_footer_contains_versions(self):
        text = _footer_markdown()
        self.assertIn("Python", text)
        self.assertIn("scikit-learn", text)
        self.assertIn("Gradio", text)

    def test_footer_contains_counts(self):
        text = _footer_markdown()
        self.assertIn("算法注册表", text)
        self.assertIn("可用算法", text)
        self.assertIn("已保存模型", text)

    def test_footer_reflects_registry_size(self):
        from core.registry import get_registry
        registry = get_registry()
        n_total = len(registry)
        n_meta = sum(1 for s in registry.values() if s.is_meta)
        text = _footer_markdown()
        self.assertIn(f"{n_total - n_meta} 个可用算法", text)
        self.assertIn(f"共 {n_total} 条", text)


if __name__ == "__main__":
    unittest.main()