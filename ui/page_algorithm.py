"""算法页：任务类型 → 算法 → 动态参数面板（含元估计器简化配置）。"""
from __future__ import annotations

import dataclasses

import gradio as gr

from core.params import build_component, collect_kwargs
from core.registry import TASK_LABELS, Task, get_estimator, get_registry


_TASK_CHOICES = [
    (label, task) for task, label in TASK_LABELS.items()
    if task != Task.META
]

_DATASET_TASK_TO_UI_TASK = {
    Task.CLASSIFICATION: Task.CLASSIFICATION,
    Task.REGRESSION: Task.REGRESSION,
    "unsupervised": Task.CLUSTERING,
}


def _algo_choices_for_task(task: str | None) -> list[str]:
    if not task:
        return []
    return sorted(
        name for name, spec in get_registry().items()
        if spec.task == task or spec.task == Task.META
    )


def _sub_estimator_choices(task: str | None) -> list[str]:
    if not task or task == Task.META:
        task = Task.CLASSIFICATION
    return sorted(
        name for name, spec in get_registry().items()
        if spec.task == task and not spec.is_meta
    )


def _render_normal_params(spec, spec_state, params_state):
    if not spec.params:
        return [gr.Markdown("该算法无可配置参数，可直接在训练页开始训练。")]

    nested_names = set(spec.nested_estimator_params)
    components: list[gr.Component] = []
    nested_hints: list[str] = []

    for p in spec.params:
        if p.name in nested_names:
            components.append(gr.Textbox(
                value=p.default_repr,
                label=p.name,
                info="使用 sklearn 内置默认子估计器（当前版本不支持在 GUI 中指定）",
                interactive=False,
            ))
            nested_hints.append(p.name)
        else:
            components.append(build_component(p))

    hint_md = gr.Markdown(
        f"ℹ️ 参数 `{', '.join(nested_hints)}` "
        "使用 sklearn 内置默认子估计器，已锁定不可编辑。",
        visible=bool(nested_hints),
    )

    confirm_btn = gr.Button("✅ 确认参数并保存", variant="primary")
    status = gr.Markdown("*点击上方按钮保存参数*")

    def _confirm(*values):
        try:
            kwargs = collect_kwargs(spec.params, list(values))
            return (
                kwargs,
                spec,
                f"✅ 已保存 **{spec.name}** 的 {len(kwargs)} 个参数。"
                "请到【训练】页开始训练。",
            )
        except Exception as e:
            return (
                gr.update(), gr.update(),
                f"❌ 参数转换失败：{type(e).__name__}: {e}",
            )

    confirm_btn.click(
        _confirm,
        inputs=components,
        outputs=[params_state, spec_state, status],
    )

    return [hint_md] + components + [confirm_btn, status]


def _render_meta_params(spec, current_task, spec_state, params_state):
    """元估计器简化配置：子估计器选择 + 其余参数照常配置。"""
    nested_names = list(spec.nested_estimator_params)
    sub_choices = _sub_estimator_choices(current_task)

    sub_components: dict[str, gr.Component] = {}
    for nested_name in nested_names:
        if nested_name == "estimators":
            sub_components[nested_name] = gr.Dropdown(
                choices=sub_choices,
                label=f"子估计器列表（{nested_name}）",
                multiselect=True,
                value=[],
            )
        else:
            sub_components[nested_name] = gr.Dropdown(
                choices=sub_choices,
                label=f"子估计器（{nested_name}）",
                value=None,
            )

    other_params = [p for p in spec.params if p.name not in set(nested_names)]
    other_components = [build_component(p) for p in other_params]

    hint_md = gr.Markdown(
        "ℹ️ 元估计器模式：子估计器使用其 **默认参数** 实例化；"
        "如需对子估计器调参，可先在子估计器对应任务下手动确定参数，"
        "或通过 `param_grid` / `param_distributions` 传入搜索空间。"
    )

    confirm_btn = gr.Button("✅ 确认参数并保存", variant="primary")
    status = gr.Markdown("*点击上方按钮保存参数*")

    sub_keys = list(sub_components.keys())
    sub_values = list(sub_components.values())

    def _confirm(*values):
        sub_vals = values[:len(sub_values)]
        other_vals = values[len(sub_values):]
        try:
            from core.trainer import instantiate_estimator
            resolved: dict = {}
            for key, val in zip(sub_keys, sub_vals):
                if key == "estimators":
                    if not val:
                        return gr.update(), gr.update(), "❌ 请至少选择一个子估计器"
                    pairs = []
                    for name in val:
                        sub_spec = get_estimator(name)
                        pairs.append((name, instantiate_estimator(sub_spec, {})))
                    resolved[key] = pairs
                else:
                    if not val:
                        return gr.update(), gr.update(), "❌ 请选择子估计器"
                    sub_spec = get_estimator(val)
                    resolved[key] = instantiate_estimator(sub_spec, {})

            other_kwargs = collect_kwargs(other_params, list(other_vals))
            kwargs = {**other_kwargs, **resolved}

            effective_task = (
                current_task if current_task in (Task.CLASSIFICATION, Task.REGRESSION)
                else Task.CLASSIFICATION
            )
            effective_spec = dataclasses.replace(
                spec,
                task=effective_task if spec.task == Task.META else spec.task,
            )
            return (
                kwargs, effective_spec,
                f"✅ 已保存 **{spec.name}** 配置（子估计器已实例化）。请到【训练】页开始训练。",
            )
        except Exception as e:
            return gr.update(), gr.update(), f"❌ {type(e).__name__}: {e}"

    confirm_btn.click(
        _confirm,
        inputs=sub_values + other_components,
        outputs=[params_state, spec_state, status],
    )

    return [hint_md] + sub_values + other_components + [confirm_btn, status]


def build(dataset_state: gr.State, spec_state: gr.State, params_state: gr.State) -> None:
    gr.Markdown(
        "## 1. 选择任务类型与算法\n"
        "任务类型会根据【数据】页的准备结果自动预填，也可手动覆盖。\n\n"
        "元估计器（`GridSearchCV`、`VotingClassifier`、`RFE` 等）已在本页开放："
        "其子估计器参数使用默认值实例化，其余参数照常配置。"
    )

    with gr.Row():
        task_dd = gr.Dropdown(
            choices=_TASK_CHOICES,
            value=None,
            label="任务类型",
            interactive=True,
        )
        # 修复点 1：初始化时给一个默认非空 choices，防止初始状态为空导致缓存问题
        algo_dd = gr.Dropdown(
            choices=_algo_choices_for_task(Task.CLASSIFICATION),
            value=None,
            label="算法",
            interactive=True,
        )

    algo_info_md = gr.Markdown("")

    gr.Markdown("## 2. 参数配置")

    with gr.Column() as params_container:
        @gr.render(inputs=[algo_dd, task_dd])
        def render_params(algo_name: str | None, current_task: str | None):
            if not algo_name:
                return [gr.Markdown("*请先选择算法*")]
            try:
                spec = get_estimator(algo_name)
            except KeyError:
                return [gr.Markdown(f"❌ 未找到算法 `{algo_name}`")]
            if spec.is_meta:
                return _render_meta_params(spec, current_task, spec_state, params_state)
            return _render_normal_params(spec, spec_state, params_state)

    def _on_dataset_change(bundle):
        if bundle is None:
            return (
                gr.update(),
                gr.update(choices=_algo_choices_for_task(Task.CLASSIFICATION), value=None),
                "",
            )
        hint = getattr(bundle, "task_hint", None)
        ui_task = _DATASET_TASK_TO_UI_TASK.get(hint, Task.CLASSIFICATION)
        # 修复点 2：更新任务时，同时把候选列表赋给 algo_dd，避免中间状态为 []
        choices = _algo_choices_for_task(ui_task)
        return (
            gr.update(value=ui_task),
            gr.update(choices=choices, value=None),
            f"已从数据集推断任务类型：**{TASK_LABELS.get(ui_task, ui_task)}**",
        )

    dataset_state.change(
        _on_dataset_change,
        inputs=dataset_state,
        outputs=[task_dd, algo_dd, algo_info_md],
    )

    def _on_task_change(task: str | None):
        choices = _algo_choices_for_task(task)
        n_meta = sum(1 for n in choices if get_registry()[n].is_meta)
        return (
            gr.update(choices=choices, value=None),
            f"共 **{len(choices)}** 个可选算法"
            + (f"（其中 {n_meta} 个是元估计器）" if n_meta else ""),
        )

    task_dd.change(
        _on_task_change,
        inputs=task_dd,
        outputs=[algo_dd, algo_info_md],
    )

    def _on_algo_change(algo_name: str | None):
        if not algo_name:
            return None, {}, "*尚未选择算法*"
        spec = get_estimator(algo_name)
        tag = " · 元估计器" if spec.is_meta else ""
        desc = (
            f"**{spec.name}**{tag} · `{spec.qualname}`\n\n"
            f"{spec.doc or '（无文档摘要）'}"
        )
        return spec, {}, desc

    algo_dd.change(
        _on_algo_change,
        inputs=algo_dd,
        outputs=[spec_state, params_state, algo_info_md],
    )