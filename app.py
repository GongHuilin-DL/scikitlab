"""ScikitLab 入口：单页面 + 多 Tab，所有 State 在 Blocks 顶层共享。"""
import sys
from pathlib import Path

import gradio as gr
import sklearn

from core.persistence import list_models
from core.registry import get_registry
from ui import (
    page_algorithm,
    page_compare,
    page_data,
    page_evaluate,
    page_predict,
    page_train,
)

_ARTIFACTS_DIR = Path("artifacts/saved_models")


def _footer_markdown() -> str:
    try:
        registry = get_registry()
        n_total = len(registry)
        n_meta = sum(1 for s in registry.values() if s.is_meta)
        n_algo = n_total - n_meta
    except Exception as exc:
        return f"---\n⚠ 注册表不可用：`{type(exc).__name__}: {exc}`"

    try:
        n_models = len(list_models(_ARTIFACTS_DIR))
    except Exception:
        n_models = 0

    return (
        "---\n"
        f"**运行环境** · Python `{sys.version.split()[0]}` · "
        f"scikit-learn `{sklearn.__version__}` · "
        f"Gradio `{gr.__version__}`\n\n"
        f"**算法注册表** · {n_algo} 个可用算法 · "
        f"{n_meta} 个 meta 估计器（不在 UI 中展示） · "
        f"共 {n_total} 条\n\n"
        f"**已保存模型** · {n_models} 个（位于 `artifacts/saved_models/`）"
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="ScikitLab") as app:
        gr.Markdown(
            "# ScikitLab\n"
            "**传统机器学习 GUI 前端** · 数据 / 算法 / 训练 / 评估 / 推理 / 对比"
        )

        dataset_state = gr.State(value=None)
        spec_state = gr.State(value=None)
        params_state = gr.State(value={})
        session_state = gr.State(value=None)

        with gr.Tabs():
            with gr.Tab("📂 数据"):
                page_data.build(dataset_state)
            with gr.Tab("🧠 算法"):
                page_algorithm.build(dataset_state, spec_state, params_state)
            with gr.Tab("🏋 训练"):
                page_train.build(dataset_state, spec_state, params_state, session_state)
            with gr.Tab("📊 评估"):
                page_evaluate.build(session_state)
            with gr.Tab("🔮 推理"):
                page_predict.build(dataset_state)
            with gr.Tab("⚖️ 对比"):
                compare_api = page_compare.build()

        gr.Markdown(_footer_markdown())

        # 对比页初始加载
        app.load(
            compare_api["fn"],
            inputs=compare_api["inputs"],
            outputs=compare_api["outputs"],
        )

    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7860)