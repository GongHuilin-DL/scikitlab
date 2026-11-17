"""推理页：载入已保存模型，单条与批量预测。"""
from __future__ import annotations

import time
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from core.persistence import list_models, load_model


_ARTIFACTS_DIR = Path("artifacts/saved_models")
_UPLOADS_DIR = Path("artifacts/uploads")


def _refresh_choices(artifacts_dir: Path | None = None) -> tuple[list[str], str | None, str]:
    base = artifacts_dir or _ARTIFACTS_DIR
    models = list_models(base)
    if not models:
        return [], None, "（暂无已保存模型，请先在【训练】页保存）"
    choices = [m.name for m in models]
    return choices, choices[0], f"共 **{len(choices)}** 个已保存模型"


def _get_classes(estimator) -> list | None:
    """从估计器（可能是 Pipeline）提取 classes_。"""
    if hasattr(estimator, "classes_"):
        return list(estimator.classes_)
    if isinstance(estimator, Pipeline) and estimator.steps:
        last = estimator.steps[-1][1]
        if hasattr(last, "classes_"):
            return list(last.classes_)
    return None


def build(dataset_state: gr.State) -> None:
    gr.Markdown(
        "## 模型推理\n"
        "从 `artifacts/saved_models/` 载入模型进行单条或批量预测。"
        "模型文件与元数据由【训练】页的【保存当前模型】生成。"
    )

    with gr.Row():
        model_dd = gr.Dropdown(choices=[], label="已保存模型", scale=3)
        refresh_btn = gr.Button("🔄 刷新列表", scale=1)
        load_btn = gr.Button("📂 载入模型", variant="primary", scale=1)

    model_info_md = gr.Markdown("*尚未载入模型*")
    loaded_state = gr.State(value=None)

    with gr.Tabs():
        with gr.Tab("单条推理"):
            with gr.Column() as _single_container:
                @gr.render(inputs=[loaded_state])
                def render_single_form(loaded):
                    if loaded is None:
                        return [gr.Markdown("*请先在上方选择一个模型并点击【载入模型】*")]

                    metadata = loaded.metadata
                    estimator = loaded.estimator
                    feature_names = list(metadata.feature_names)

                    # 按 feature_types 为每列决定控件类型
                    feature_types = getattr(metadata, "feature_types", {}) or {}
                    categorical_set = set(feature_types.get("categorical", []))
                    # 旧模型没有 feature_types 时，全部按数值列处理（向后兼容）
                    all_numeric_fallback = not feature_types.get("numeric") and not categorical_set

                    components: list[gr.Component] = []
                    for name in feature_names:
                        if name in categorical_set and not all_numeric_fallback:
                            components.append(gr.Textbox(label=name, value=""))
                        else:
                            components.append(gr.Number(label=name, value=0.0))

                    predict_btn = gr.Button("🔮 预测", variant="primary")
                    result_md = gr.Markdown("")

                    def _predict(*values):
                        try:
                            # 用 dict → DataFrame，保留列顺序与 dtype
                            row = dict(zip(feature_names, values))
                            X = pd.DataFrame([row], columns=feature_names)
                            pred = estimator.predict(X)[0]
                            lines = [f"**预测结果**：`{pred}`"]

                            if hasattr(estimator, "predict_proba"):
                                try:
                                    proba = np.asarray(estimator.predict_proba(X))[0]
                                    classes = _get_classes(estimator)
                                    if classes is not None and len(classes) == len(proba):
                                        lines.append("\n**概率分布**：\n")
                                        for idx in np.argsort(proba)[::-1]:
                                            lines.append(
                                                f"- 类别 `{classes[idx]}`: "
                                                f"{proba[idx]:.4f}"
                                            )
                                except Exception:
                                    pass
                            return "\n".join(lines)
                        except Exception as e:
                            return f"❌ {type(e).__name__}: {e}"

                    predict_btn.click(_predict, inputs=components, outputs=[result_md])
                    return components + [predict_btn, result_md]

        with gr.Tab("批量推理"):
            gr.Markdown(
                "上传 CSV 文件，列名必须与模型训练时的 `feature_names` 一致。"
                "预测结果前 200 行显示在表格中，完整结果可在下方下载。"
            )
            with gr.Row():
                batch_file = gr.File(
                    label="上传 CSV",
                    file_types=[".csv"],
                    file_count="single",
                )
                batch_predict_btn = gr.Button("🔮 批量预测", variant="primary")
            batch_status_md = gr.Markdown("")
            batch_result_df = gr.Dataframe(
                label="预测结果（前 200 行）",
                interactive=False, wrap=True,
            )
            batch_download = gr.File(label="下载完整结果 CSV")

    # ------------------------------------------------------------------
    # 刷新模型列表
    # ------------------------------------------------------------------
    def _on_refresh():
        choices, value, msg = _refresh_choices()
        return gr.update(choices=choices, value=value), msg

    refresh_btn.click(_on_refresh, outputs=[model_dd, model_info_md])

    def _on_dataset_change(_bundle):
        choices, value, msg = _refresh_choices()
        return gr.update(choices=choices, value=value), msg

    dataset_state.change(
        _on_dataset_change, inputs=dataset_state,
        outputs=[model_dd, model_info_md],
    )

    # ------------------------------------------------------------------
    # 载入模型
    # ------------------------------------------------------------------
    def _on_load(name: str | None):
        if not name:
            return "❌ 请先选择一个模型", None
        try:
            loaded = load_model(name, _ARTIFACTS_DIR)
            lines = [
                f"**已载入**：`{loaded.metadata.algorithm}`",
                f"- 模型名：`{loaded.metadata.name}`",
                f"- 任务类型：`{loaded.metadata.task}`",
                f"- 保存时间：{loaded.metadata.created_at}",
                f"- sklearn 版本：{loaded.metadata.sklearn_version}",
                f"- 特征数：{loaded.metadata.n_features}",
            ]
            if loaded.metadata.n_classes:
                lines.append(f"- 类别数：{loaded.metadata.n_classes}")

            ft = getattr(loaded.metadata, "feature_types", {}) or {}
            n_numeric = len(ft.get("numeric", []))
            n_categorical = len(ft.get("categorical", []))
            if n_categorical > 0:
                lines.append(f"- 数值特征：{n_numeric} · 类别特征：{n_categorical}")
            elif n_numeric > 0:
                lines.append(f"- 数值特征：{n_numeric}")

            if loaded.metadata.notes:
                lines.append(f"- 备注：{loaded.metadata.notes}")
            if loaded.warnings:
                lines.append("\n**⚠ 警告**\n")
                for w in loaded.warnings:
                    lines.append(f"- {w}")
            return "\n".join(lines), loaded
        except Exception as e:
            return f"❌ {type(e).__name__}: {e}", None

    load_btn.click(
        _on_load, inputs=model_dd, outputs=[model_info_md, loaded_state],
    )

    # ------------------------------------------------------------------
    # 批量推理
    # ------------------------------------------------------------------
    def _on_batch_predict(loaded, file_path):
        if loaded is None:
            return "❌ 请先载入模型", pd.DataFrame(), None
        if file_path is None:
            return "❌ 请上传 CSV", pd.DataFrame(), None
        try:
            df = pd.read_csv(file_path)
            feature_names = list(loaded.metadata.feature_names)
            missing = [c for c in feature_names if c not in df.columns]
            if missing:
                return f"❌ CSV 缺少特征列：{missing}", pd.DataFrame(), None

            # 保留 DataFrame 形式，让 Pipeline 内的 ColumnTransformer 能按列名处理
            X = df[feature_names]
            preds = loaded.estimator.predict(X)
            result = df.copy()
            result["prediction"] = preds

            if hasattr(loaded.estimator, "predict_proba"):
                try:
                    proba = np.asarray(loaded.estimator.predict_proba(X))
                    classes = _get_classes(loaded.estimator)
                    if classes is not None and proba.shape[1] == len(classes):
                        for i, c in enumerate(classes):
                            result[f"proba_{c}"] = proba[:, i]
                except Exception:
                    pass

            _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = _UPLOADS_DIR / f"predictions_{int(time.time())}.csv"
            result.to_csv(out_path, index=False)

            return (
                f"✅ 已预测 **{len(result)}** 行",
                result.head(200),
                str(out_path),
            )
        except Exception as e:
            return f"❌ {type(e).__name__}: {e}", pd.DataFrame(), None

    batch_predict_btn.click(
        _on_batch_predict,
        inputs=[loaded_state, batch_file],
        outputs=[batch_status_md, batch_result_df, batch_download],
    )