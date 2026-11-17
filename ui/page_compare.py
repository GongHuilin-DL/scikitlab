"""对比页：已保存模型的横向对比与删除。"""
from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd
from matplotlib.figure import Figure

from core.persistence import ModelMetadata, delete_model, list_models


_ARTIFACTS_DIR = Path("artifacts/saved_models")

_TASK_LABELS = {
    "classification": "分类",
    "regression": "回归",
    "clustering": "聚类",
    "anomaly_detection": "异常检测",
    "biclustering": "双聚类",
    "density_estimation": "密度估计",
    "dimensionality_reduction": "降维",
    "feature_selection": "特征选择",
    "transformation": "特征变换",
    "meta_estimator": "元估计器",
    "covariance_estimation": "协方差估计",
    "other": "其它",
}

_TASK_COLORS = {
    "classification": "#4C72B0",
    "regression": "#DD8452",
    "clustering": "#55A868",
    "anomaly_detection": "#C44E52",
    "dimensionality_reduction": "#8172B3",
    "transformation": "#937860",
    "feature_selection": "#DA8BC3",
    "biclustering": "#8C8C8C",
    "density_estimation": "#CCB974",
    "covariance_estimation": "#64B5CD",
    "meta_estimator": "#D65F5F",
    "other": "#888888",
}

_ALL_TASK = "__all__"
_TASK_FILTER_OPTIONS = [("全部", _ALL_TASK)] + [
    (label, task) for task, label in _TASK_LABELS.items()
]

_CLASSIFICATION_METRICS = ["accuracy", "f1_macro", "log_loss"]
_REGRESSION_METRICS = ["mse", "mae", "r2"]


def _format_metrics(metrics: dict | None) -> str:
    if not metrics:
        return "-"
    parts: list[str] = []
    for k, v in metrics.items():
        if isinstance(v, bool):
            parts.append(f"{k}={v}")
        elif isinstance(v, (int, float)):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts) if parts else "-"


def _extract_train_val(meta: ModelMetadata) -> tuple[dict, dict]:
    current = meta.training_metrics or {}
    return current.get("train") or {}, current.get("val") or {}


def _build_compare_dataframe(
    models: list[ModelMetadata], task_filter: str
) -> pd.DataFrame:
    rows: list[dict] = []
    for m in models:
        if task_filter != _ALL_TASK and m.task != task_filter:
            continue
        train, val = _extract_train_val(m)
        rows.append({
            "模型名": m.name,
            "算法": m.algorithm,
            "任务": _TASK_LABELS.get(m.task, m.task),
            "训练模式": m.mode,
            "创建时间": m.created_at,
            "特征数": m.n_features,
            "类别数": m.n_classes if m.n_classes is not None else "-",
            "训练耗时(秒)": round(m.elapsed_seconds, 3),
            "训练集指标": _format_metrics(train),
            "测试集指标": _format_metrics(val),
            "备注": m.notes or "",
        })
    if not rows:
        return pd.DataFrame({"提示": ["当前筛选下无模型"]})
    return pd.DataFrame(rows)


def _build_elapsed_figure(
    models: list[ModelMetadata], task_filter: str
) -> Figure:
    fig = Figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    filtered = [
        m for m in models
        if task_filter == _ALL_TASK or m.task == task_filter
    ]
    if not filtered:
        ax.text(0.5, 0.5, "No models to compare",
                ha="center", va="center", fontsize=12, color="gray")
        ax.axis("off")
        return fig

    sorted_models = sorted(filtered, key=lambda m: m.elapsed_seconds)
    names = [m.name for m in sorted_models]
    values = [m.elapsed_seconds for m in sorted_models]
    colors = [_TASK_COLORS.get(m.task, "#888") for m in sorted_models]

    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Elapsed (seconds)")
    ax.set_title("Training Time Comparison")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


def _build_metric_figure(
    models: list[ModelMetadata], task_filter: str, metric_name: str
) -> Figure:
    fig = Figure(figsize=(10, 4))
    ax = fig.add_subplot(111)

    effective_task = (
        task_filter if task_filter != _ALL_TASK else "classification"
    )
    filtered = [m for m in models if m.task == effective_task]

    names: list[str] = []
    values: list[float] = []
    for m in filtered:
        _, val = _extract_train_val(m)
        v = val.get(metric_name)
        if v is not None and isinstance(v, (int, float)) and not isinstance(v, bool):
            names.append(m.name)
            values.append(float(v))

    if not values:
        ax.text(
            0.5, 0.5,
            f"No '{metric_name}' metric in saved models",
            ha="center", va="center", fontsize=11, color="gray",
        )
        ax.axis("off")
        return fig

    lower_better = metric_name in {"mse", "mae", "log_loss"}
    sorted_pairs = sorted(
        zip(names, values), key=lambda p: p[1], reverse=not lower_better,
    )
    sorted_names = [p[0] for p in sorted_pairs]
    sorted_values = [p[1] for p in sorted_pairs]

    ax.barh(range(len(sorted_names)), sorted_values, color="steelblue")
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel(metric_name)
    direction = "lower is better" if lower_better else "higher is better"
    ax.set_title(f"{metric_name} Comparison ({direction})")
    ax.grid(True, alpha=0.3, axis="x")
    for i, v in enumerate(sorted_values):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    return fig


def _refresh(
    task_filter: str, metric_name: str
) -> tuple[str, pd.DataFrame, Figure, Figure]:
    models = list_models(_ARTIFACTS_DIR)
    df = _build_compare_dataframe(models, task_filter)
    elapsed_fig = _build_elapsed_figure(models, task_filter)
    metric_fig = _build_metric_figure(models, task_filter, metric_name)

    if not models:
        summary = (
            "ℹ️ `artifacts/saved_models/` 为空，"
            "请先在【训练】页完成训练并保存模型。"
        )
    elif task_filter == _ALL_TASK:
        summary = f"共 **{len(models)}** 个已保存模型"
    else:
        n_filtered = sum(1 for m in models if m.task == task_filter)
        summary = (
            f"共 **{len(models)}** 个已保存模型，"
            f"当前筛选 `{_TASK_LABELS.get(task_filter, task_filter)}` 下 "
            f"**{n_filtered}** 个"
        )
    return summary, df, elapsed_fig, metric_fig


def _refresh_with_delete(task_filter: str, metric_name: str):
    summary, df, ef, mf = _refresh(task_filter, metric_name)
    models = list_models(_ARTIFACTS_DIR)
    return (
        summary, df, ef, mf,
        gr.update(choices=[m.name for m in models], value=None),
    )


def build() -> dict:
    gr.Markdown(
        "## 模型对比\n"
        "横向对比 `artifacts/saved_models/` 中的所有已保存模型。"
        "按任务类型筛选，查看训练耗时与关键指标的对比。"
    )

    with gr.Row():
        task_filter = gr.Dropdown(
            choices=_TASK_FILTER_OPTIONS,
            value=_ALL_TASK,
            label="任务类型筛选",
            scale=2,
        )
        metric_select = gr.Dropdown(
            choices=_CLASSIFICATION_METRICS + _REGRESSION_METRICS,
            value="accuracy",
            label="指标对比字段",
            allow_custom_value=True,
            scale=2,
        )
        refresh_btn = gr.Button("🔄 刷新", variant="primary", scale=1)

    summary_md = gr.Markdown("*点击【🔄 刷新】加载已保存模型列表*")
    compare_df = gr.Dataframe(
        label="模型对比表", interactive=False, wrap=True,
    )

    with gr.Row():
        elapsed_plot = gr.Plot(label="训练耗时对比")
        metric_plot = gr.Plot(label="指标对比")

    gr.Markdown("## 删除模型\n⚠ 删除不可恢复，`joblib` 与 `json` 会一并删除。")
    with gr.Row():
        delete_dd = gr.Dropdown(
            choices=[], label="选择要删除的模型", scale=3,
        )
        delete_btn = gr.Button("🗑 删除", variant="stop", scale=1)
    delete_status_md = gr.Markdown("")

    outputs = [summary_md, compare_df, elapsed_plot, metric_plot]
    inputs = [task_filter, metric_select]

    refresh_btn.click(
        _refresh_with_delete, inputs=inputs, outputs=outputs + [delete_dd],
    )
    task_filter.change(
        _refresh_with_delete, inputs=inputs, outputs=outputs + [delete_dd],
    )
    metric_select.change(
        _refresh_with_delete, inputs=inputs, outputs=outputs + [delete_dd],
    )

    def _on_delete(name: str | None):
        if not name:
            return "❌ 请先选择要删除的模型", gr.update()
        try:
            ok = delete_model(name, _ARTIFACTS_DIR)
            if ok:
                models = list_models(_ARTIFACTS_DIR)
                return (
                    f"✅ 已删除 `{name}`",
                    gr.update(choices=[m.name for m in models], value=None),
                )
            return f"❌ 模型 `{name}` 不存在", gr.update()
        except Exception as e:
            return f"❌ {type(e).__name__}: {e}", gr.update()

    delete_btn.click(
        _on_delete,
        inputs=delete_dd,
        outputs=[delete_status_md, delete_dd],
    ).then(
        _refresh, inputs=inputs, outputs=outputs,
    )

    return {"fn": _refresh, "inputs": inputs, "outputs": outputs}