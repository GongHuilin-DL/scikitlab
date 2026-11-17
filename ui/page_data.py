"""数据页：数据来源 → 预览 → 选列 → 预处理配置 → 划分。"""
from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd

from core.dataset import (
    TOY_DATASETS,
    build_bundle_from_dataframe,
    infer_task,
    load_toy,
    read_tabular,
)


_EMPTY_TARGET = "（无目标列 · 无监督）"
_TASK_OPTIONS = [
    ("自动推断", "auto"),
    ("分类", "classification"),
    ("回归", "regression"),
    ("无监督", "unsupervised"),
]
_SPLIT_OPTIONS = [
    ("分层抽样（stratified）", "stratified"),
    ("随机抽样（random）", "random"),
    ("顺序切片（sequential）", "sequential"),
]
_SCALER_OPTIONS = [
    ("不缩放", "none"),
    ("StandardScaler", "standard"),
    ("MinMaxScaler", "minmax"),
]
_ENCODER_OPTIONS = [
    ("OneHotEncoder（推荐）", "onehot"),
    ("OrdinalEncoder", "ordinal"),
]


def _class_dist_table(bundle) -> pd.DataFrame:
    if not bundle.class_distribution:
        return pd.DataFrame({"提示": ["当前为无监督任务，无类别分布"]})
    train = bundle.class_distribution.get("train", {})
    test = bundle.class_distribution.get("test", {})
    classes = list(train.keys())
    return pd.DataFrame({
        "类别": classes,
        "训练集": [train.get(c, 0) for c in classes],
        "测试集": [test.get(c, 0) for c in classes],
    })


def build(dataset_state: gr.State) -> None:
    gr.Markdown("## 1. 载入数据\n支持 CSV / Excel，也可使用内置玩具数据集。")

    with gr.Row():
        source_radio = gr.Radio(
            choices=[("上传文件", "upload"), ("玩具数据集", "toy")],
            value="upload",
            label="数据来源",
        )

    with gr.Group(visible=True) as upload_group:
        upload_file = gr.File(
            label="上传 CSV / Excel",
            file_types=[".csv", ".xlsx", ".xls"],
            file_count="single",
            type="filepath",
        )

    with gr.Group(visible=False) as toy_group:
        with gr.Row():
            toy_select = gr.Dropdown(
                choices=list(TOY_DATASETS.keys()),
                value=list(TOY_DATASETS.keys())[0],
                label="内置数据集",
            )
            toy_load_btn = gr.Button("加载", variant="primary")

    preview_df = gr.Dataframe(
        label="数据预览（前 50 行）",
        interactive=False,
        wrap=True,
    )

    pending_state = gr.State(value=None)

    gr.Markdown("## 2. 特征与目标")
    with gr.Row():
        feature_dd = gr.Dropdown(
            choices=[], label="特征列（可多选）", multiselect=True, interactive=True,
        )
        target_dd = gr.Dropdown(
            choices=[_EMPTY_TARGET],
            value=_EMPTY_TARGET,
            label="目标列",
            interactive=True,
            allow_custom_value=False,
        )

    task_radio = gr.Radio(
        choices=_TASK_OPTIONS, value="auto", label="任务类型",
        info="auto 时会依据目标列类型推断",
    )

    gr.Markdown(
        "## 3. 预处理配置\n"
        "数值列默认 StandardScaler 缩放；类别列（字符串 / 分类 dtype）自动 OneHotEncoder。"
        "如果数据含缺失值，将先用中位数/众数填充。"
    )
    with gr.Row():
        numeric_scaler_dd = gr.Dropdown(
            choices=_SCALER_OPTIONS, value="standard",
            label="数值特征缩放",
        )
        categorical_encoder_dd = gr.Dropdown(
            choices=_ENCODER_OPTIONS, value="onehot",
            label="类别特征编码",
        )

    gr.Markdown("## 4. 训练集 / 测试集划分")
    with gr.Row():
        test_size_slider = gr.Slider(
            minimum=0.05, maximum=0.5, value=0.2, step=0.05,
            label="测试集比例",
        )
        seed_number = gr.Number(value=42, precision=0, label="随机种子")
        split_mode_radio = gr.Radio(
            choices=_SPLIT_OPTIONS, value="stratified",
            label="划分方式",
            info="顺序切片等价于简单前后切分，用于复现旧脚本行为；分类任务推荐分层抽样",
        )

    prepare_btn = gr.Button("划分并准备训练", variant="primary")
    status_md = gr.Markdown("*尚未载入数据*")
    class_dist_df = gr.Dataframe(
        label="类别分布（训练集 vs 测试集）",
        interactive=False,
        visible=False,
    )

    def _toggle_source(choice: str):
        return (
            gr.update(visible=(choice == "upload")),
            gr.update(visible=(choice == "toy")),
        )

    source_radio.change(_toggle_source, inputs=source_radio,
                        outputs=[upload_group, toy_group])

    def _on_df_loaded(df: pd.DataFrame, name: str, task_hint: str, status_prefix: str):
        if df is None or df.empty:
            return (
                gr.update(), None, gr.update(), gr.update(), gr.update(),
                "❌ 载入的数据为空",
            )
        target_choices = [_EMPTY_TARGET] + list(df.columns)
        default_target = "target" if "target" in df.columns else df.columns[-1]
        default_features = [c for c in df.columns if c != default_target]

        inferred = infer_task(df[default_target], len(df))
        effective_task = task_hint if task_hint != "auto" else inferred

        status = (
            f"{status_prefix}：{df.shape[0]} 行 × {df.shape[1]} 列，"
            f"推断任务 `{effective_task}`。请确认特征列与目标列后点击【划分并准备训练】。"
        )

        return (
            df.head(50),
            df,
            gr.update(choices=list(df.columns), value=default_features),
            gr.update(choices=target_choices, value=default_target),
            gr.update(value=effective_task),
            status,
        )

    def _on_upload(file_path):
        try:
            if file_path is None:
                return (gr.update(), None, gr.update(), gr.update(),
                        gr.update(), "❌ 请先选择文件")
            df = read_tabular(file_path)
            return _on_df_loaded(df, Path(file_path).stem, "auto",
                                 f"已载入 `{Path(file_path).name}`")
        except Exception as e:
            return (gr.update(), None, gr.update(), gr.update(),
                    gr.update(), f"❌ {type(e).__name__}: {e}")

    def _on_toy_load(key: str):
        try:
            df, target_col, task_hint = load_toy(key)
            df = df.rename(columns={target_col: target_col})
            return _on_df_loaded(df, key, task_hint, f"已加载 `{key}`")
        except Exception as e:
            return (gr.update(), None, gr.update(), gr.update(),
                    gr.update(), f"❌ {type(e).__name__}: {e}")

    upload_file.change(
        _on_upload,
        inputs=upload_file,
        outputs=[preview_df, pending_state, feature_dd, target_dd, task_radio, status_md],
    )
    toy_load_btn.click(
        _on_toy_load,
        inputs=toy_select,
        outputs=[preview_df, pending_state, feature_dd, target_dd, task_radio, status_md],
    )

    def _prepare(df, feature_cols, target_choice, task_choice, test_size, seed,
                 split_mode, numeric_scaler, categorical_encoder):
        try:
            if df is None:
                return None, "❌ 尚未载入数据", gr.update(visible=False)
            if not feature_cols:
                return None, "❌ 请至少选择一个特征列", gr.update(visible=False)

            target_col = None if target_choice == _EMPTY_TARGET else target_choice

            if task_choice == "auto":
                if target_col is None:
                    task_hint = "unsupervised"
                else:
                    task_hint = infer_task(df[target_col], len(df))
            else:
                task_hint = task_choice

            if target_col is not None and target_col in feature_cols:
                return None, "❌ 目标列不能同时作为特征列", gr.update(visible=False)

            bundle = build_bundle_from_dataframe(
                df,
                feature_cols=list(feature_cols),
                target_col=target_col,
                task_hint=task_hint,
                test_size=float(test_size),
                random_state=int(seed),
                split_mode=split_mode,
                name="dataset",
                numeric_scaler=numeric_scaler,
                categorical_encoder=categorical_encoder,
            )
            dist_df = _class_dist_table(bundle) if bundle.target_classes else pd.DataFrame(
                {"提示": ["当前为无监督任务，无类别分布"]}
            )
            return bundle, bundle.summary(), gr.update(value=dist_df, visible=True)
        except Exception as e:
            return None, f"❌ {type(e).__name__}: {e}", gr.update(visible=False)

    prepare_btn.click(
        _prepare,
        inputs=[pending_state, feature_dd, target_dd, task_radio,
                test_size_slider, seed_number, split_mode_radio,
                numeric_scaler_dd, categorical_encoder_dd],
        outputs=[dataset_state, status_md, class_dist_df],
    )