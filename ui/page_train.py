"""训练页：会话生命周期、实时曲线、取消、保存模型、导出脚本。"""
from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd
from matplotlib.figure import Figure

from core.exporter import export_training_script
from core.persistence import save_model
from core.trainer import TrainingSession, run_training, supports_partial_fit

_ARTIFACTS_DIR = Path("artifacts/saved_models")

_STATUS_EMOJI = {
    "pending": "⏸",
    "running": "⏳",
    "done": "✅",
    "cancelled": "⛔",
    "error": "❌",
}


def _format_status(snap: dict | None) -> str:
    if not snap:
        return "⏳ **训练准备中…**"
    emoji = _STATUS_EMOJI.get(snap.get("status", ""), "·")
    lines = [
        f"{emoji} **{snap.get('status', '')}** · "
        f"算法 `{snap.get('algorithm', '')}` · "
        f"模式 `{snap.get('mode', '')}`",
        f"已用时：{snap.get('elapsed', 0):.2f} 秒",
    ]
    if snap.get("mode") == "batch" and snap.get("status") == "running":
        lines.append("当前为批量算法，期间无法逐步推送指标，请耐心等待 `fit` 返回。")
    if snap.get("error"):
        lines.append(f"\n**错误**：`{snap['error']}`")
    return "\n\n".join(lines)


def _format_metrics(snap: dict | None) -> pd.DataFrame:
    if not snap or not snap.get("current"):
        return pd.DataFrame({"提示": ["尚无指标"]})
    current = snap["current"]
    rows: list[dict] = []
    for split in ("train", "val"):
        for name, value in (current.get(split) or {}).items():
            rows.append({
                "数据集": "训练集" if split == "train" else "测试集",
                "指标": name,
                "数值": round(float(value), 6),
            })
    if not rows:
        return pd.DataFrame({"提示": ["当前算法尚未产生可展示的指标"]})
    return pd.DataFrame(rows)


def _build_figure(history: list[dict], algorithm: str) -> Figure:
    fig = Figure(figsize=(9, 5.5))
    if not history:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No training history yet",
                ha="center", va="center", fontsize=12, color="gray")
        ax.set_title(algorithm or "training")
        ax.axis("off")
        return fig

    metric_points = [h for h in history if "train" in h or "val" in h]
    intrinsic_points = [h for h in history if "source" in h]

    metric_names: set[str] = set()
    for h in metric_points:
        metric_names.update((h.get("train") or {}).keys())
        metric_names.update((h.get("val") or {}).keys())
    metric_names_sorted = sorted(metric_names)

    n_plots = len(metric_names_sorted) + (1 if intrinsic_points else 0)
    if n_plots == 0:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No metrics available",
                ha="center", va="center", fontsize=12, color="gray")
        ax.set_title(algorithm or "training")
        ax.axis("off")
        return fig

    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    for i, name in enumerate(metric_names_sorted):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        train_steps, train_vals = [], []
        val_pairs: list[tuple[int, float]] = []
        for h in metric_points:
            step = h.get("step")
            tr = (h.get("train") or {}).get(name)
            va = (h.get("val") or {}).get(name)
            if tr is not None:
                train_steps.append(step)
                train_vals.append(tr)
            if va is not None:
                val_pairs.append((step, va))
        if train_steps:
            ax.plot(train_steps, train_vals, marker="o", markersize=3, label="train")
        if val_pairs:
            xs, ys = zip(*val_pairs)
            ax.plot(xs, ys, marker="s", markersize=3, label="val")
        ax.set_title(name)
        ax.set_xlabel("step / epoch")
        ax.grid(True, alpha=0.3)
        if train_steps or val_pairs:
            ax.legend(fontsize=8)

    if intrinsic_points:
        ax = fig.add_subplot(n_rows, n_cols, len(metric_names_sorted) + 1)
        by_source: dict[str, list[tuple[int, float]]] = {}
        for h in intrinsic_points:
            by_source.setdefault(h["source"], []).append((h["step"], h["value"]))
        for source, pts in by_source.items():
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", markersize=3, label=source)
        ax.set_title("intrinsic score")
        ax.set_xlabel("step / epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def _predict_mode(dataset, spec) -> tuple[str, str]:
    """返回 (mode_key, message)。mode_key ∈ {'incremental','batch','unknown'}。"""
    if dataset is None or spec is None:
        return "unknown", "ℹ️ 尚未选择数据或算法。"
    supports = supports_partial_fit(spec)
    prep = getattr(dataset, "preprocessing_spec", None)
    has_preprocessing = prep is not None and (
        getattr(prep, "numeric_features", []) or getattr(prep, "categorical_features", [])
    )
    has_cat = bool(getattr(dataset, "categorical_features", []))

    if supports and has_preprocessing:
        return (
            "batch",
            "⚠️ **预计模式：批量（强制降级）** —— 当前算法支持 "
            "`partial_fit`，但数据将走预处理管道（`Pipeline`），"
            "`Pipeline` 无 `partial_fit`，实际会降级为批量模式。",
        )
    if supports and has_cat:
        return (
            "batch",
            "⚠️ **预计模式：批量** —— 数据含类别特征需预处理管道，"
            "无法进行逐 epoch 增量训练。",
        )
    if supports:
        return "incremental", "ℹ️ **预计模式：增量** —— 曲线将逐 epoch 实时更新。"
    return "batch", "ℹ️ **预计模式：批量** —— 训练期间无法逐步推送指标。"


def build(dataset_state: gr.State, spec_state: gr.State,
          params_state: gr.State, session_state: gr.State) -> None:
    gr.Markdown(
        "## 1. 训练配置\n"
        "`epochs` 与 `batch_size` **仅对支持增量学习（`partial_fit`）且无预处理管道的算法生效**；"
        "其余情况将走批量模式，`fit` 结束后一次性呈现完整指标与内禀曲线。"
    )

    mode_hint_md = gr.Markdown("*等待数据与算法就绪…*")

    with gr.Row():
        epochs_slider = gr.Slider(
            minimum=1, maximum=200, value=20, step=1,
            label="最大 epoch（增量算法）",
        )
        batch_size_number = gr.Number(
            value=None, precision=0,
            label="batch_size（留空自动）",
        )

    with gr.Row():
        train_btn = gr.Button("🚀 开始训练", variant="primary")
        cancel_btn = gr.Button("🛑 取消训练", variant="stop", interactive=False)

    gr.Markdown("## 2. 训练监控")
    status_md = gr.Markdown("*尚未开始训练*")
    with gr.Row():
        current_metrics_df = gr.Dataframe(
            label="当前指标", interactive=False, wrap=True,
        )
        history_plot = gr.Plot(label="训练历史曲线")
    log_text = gr.Textbox(
        label="训练日志", lines=6, interactive=False, max_lines=12,
    )

    gr.Markdown("## 3. 保存模型")
    with gr.Row():
        model_name_tb = gr.Textbox(
            label="模型名称",
            placeholder="例如 logistic_iris_v1（不要带扩展名）",
            scale=3,
        )
        save_btn = gr.Button("💾 保存当前模型", variant="secondary", scale=1)
    save_status_md = gr.Markdown("")

    gr.Markdown(
        "## 4. 导出训练脚本\n"
        "生成一份完全自包含的 `.py` 脚本（含数据快照 CSV），可直接 `python` 运行，"
        "用于二次开发或教学对照。"
    )
    with gr.Row():
        export_btn = gr.Button("📤 导出训练脚本（.py）", variant="secondary")
        export_status_md = gr.Markdown("")
    export_file = gr.File(label="下载导出脚本", interactive=False)

    # ------------------------------------------------------------------
    # 模式预估（数据或算法变化时更新）
    # ------------------------------------------------------------------
    def _update_mode_hint(dataset, spec):
        _, msg = _predict_mode(dataset, spec)
        return msg

    dataset_state.change(
        _update_mode_hint,
        inputs=[dataset_state, spec_state],
        outputs=mode_hint_md,
    )
    spec_state.change(
        _update_mode_hint,
        inputs=[dataset_state, spec_state],
        outputs=mode_hint_md,
    )

    # ------------------------------------------------------------------
    # 训练生成器
    # ------------------------------------------------------------------
    session: TrainingSession | None = None

    def _outputs(active, logs, train_interactive, cancel_interactive, save_status):
        return (
            _format_status(active),
            _format_metrics(active),
            _build_figure((active or {}).get("history", []),
                          (active or {}).get("algorithm", "")),
            "\n".join(logs[-100:]),
            gr.update(interactive=train_interactive),
            gr.update(interactive=cancel_interactive),
            session,
            save_status,
        )

    def _on_train(dataset, spec, params, epochs, batch_size):
        nonlocal session
        if dataset is None:
            yield (
                "❌ 尚未在【数据】页完成“划分并准备训练”",
                pd.DataFrame({"提示": ["无数据"]}),
                _build_figure([], ""),
                "",
                gr.update(interactive=True),
                gr.update(interactive=False),
                None,
                "",
            )
            return
        if spec is None or not params:
            yield (
                "❌ 尚未在【算法】页确认算法与参数",
                pd.DataFrame({"提示": ["无算法"]}),
                _build_figure([], ""),
                "",
                gr.update(interactive=True),
                gr.update(interactive=False),
                None,
                "",
            )
            return
        if spec.task == "classification" and dataset.y_train is None:
            yield (
                "❌ 分类任务缺少训练标签，请回到【数据】页重新准备",
                pd.DataFrame({"提示": ["无标签"]}),
                _build_figure([], ""),
                "",
                gr.update(interactive=True),
                gr.update(interactive=False),
                None,
                "",
            )
            return

        session = TrainingSession(
            algorithm=spec.name,
            task=spec.task,
            params=dict(params),
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=dataset.X_test,
            y_test=dataset.y_test,
            feature_names=list(dataset.feature_names),
            target_name=dataset.target_name,
            preprocessing_spec=dataset.preprocessing_spec,
        )

        logs: list[str] = []

        yield (
            f"⏳ 已创建训练会话：`{spec.name}`",
            pd.DataFrame({"提示": ["等待第一个 epoch…"]}),
            _build_figure([], spec.name),
            "",
            gr.update(interactive=False),
            gr.update(interactive=True),
            session,
            "",
        )

        last_snap = None
        try:
            bs = int(batch_size) if batch_size not in (None, "") else None
            for snap in run_training(session, epochs=int(epochs), batch_size=bs):
                last_snap = snap
                if snap.get("log"):
                    logs.append(snap["log"])
                yield _outputs(snap, logs,
                               train_interactive=False,
                               cancel_interactive=True,
                               save_status="")
            yield _outputs(last_snap, logs,
                           train_interactive=True,
                           cancel_interactive=False,
                           save_status="✅ 训练完成，可点击下方【保存当前模型】或【导出训练脚本】")
        except Exception as e:
            logs.append(f"训练异常：{type(e).__name__}: {e}")
            yield _outputs(None, logs,
                           train_interactive=True,
                           cancel_interactive=False,
                           save_status=f"❌ 训练异常：{type(e).__name__}: {e}")

    train_btn.click(
        _on_train,
        inputs=[dataset_state, spec_state, params_state,
                epochs_slider, batch_size_number],
        outputs=[status_md, current_metrics_df, history_plot, log_text,
                 train_btn, cancel_btn, session_state, save_status_md],
        concurrency_limit=1,
    )

    def _on_cancel(sess):
        if sess is None:
            return "⛔ 当前没有正在训练或已完成的会话"
        sess.cancel()
        return (
            "⛔ 已发送取消请求。"
            "增量算法将在当前 epoch 结束后停止；"
            "批量算法必须等 `fit` 返回后才生效。"
        )

    cancel_btn.click(
        _on_cancel, inputs=session_state, outputs=save_status_md, queue=False,
    )

    # ------------------------------------------------------------------
    # 保存模型
    # ------------------------------------------------------------------
    def _on_save(sess, name):
        if sess is None:
            return "❌ 没有可保存的会话"
        if getattr(sess, "status", None) != "done":
            return f"❌ 会话状态为 `{getattr(sess, 'status', 'unknown')}`，只有训练完成才能保存"
        if not name or not name.strip():
            return "❌ 请输入模型名称"
        try:
            jp, mp = save_model(sess, name.strip(), _ARTIFACTS_DIR)
            return (
                f"✅ 已保存到 `artifacts/saved_models/`：\n\n"
                f"- `{jp.name}`（完整估计器对象，含学习参数与预处理参数）\n"
                f"- `{mp.name}`（训练前参数快照、数据 schema、指标历史）"
            )
        except Exception as e:
            return f"❌ 保存失败：{type(e).__name__}: {e}"

    save_btn.click(
        _on_save, inputs=[session_state, model_name_tb], outputs=save_status_md,
    )

    # ------------------------------------------------------------------
    # 导出脚本
    # ------------------------------------------------------------------
    def _on_export(dataset, spec, params):
        if dataset is None:
            return "❌ 请先在【数据】页完成“划分并准备训练”", None
        if spec is None or not params:
            return "❌ 请先在【算法】页确认算法与参数", None
        try:
            path = export_training_script(dataset, spec, dict(params))
            return (
                f"✅ 已导出：`{path.name}`\n\n"
                f"数据快照保存在同目录（`{path.parent.name}/`），"
                f"与脚本一起下载后即可直接运行。",
                str(path),
            )
        except Exception as e:
            return f"❌ {type(e).__name__}: {e}", None

    export_btn.click(
        _on_export,
        inputs=[dataset_state, spec_state, params_state],
        outputs=[export_status_md, export_file],
    )