"""参数元数据与 Gradio 组件之间的双向转换。

职责边界：
- 本模块只负责“单个参数的组件构建”与“组件值 -> Python 值”的转换。
- 组件的布局、动态渲染、事件绑定放在 ui/ 层，避免 core 依赖具体页面结构。
"""
from __future__ import annotations

import json
from typing import Any

import gradio as gr

from core.registry import ParamSpec


def _initial_value(spec: ParamSpec) -> Any:
    """给 Gradio 组件的初始值。"""
    if spec.kind == "bool":
        return bool(spec.default) if spec.default is not None else False
    if spec.kind == "int":
        try:
            return int(spec.default) if spec.default is not None else None
        except (TypeError, ValueError):
            return None
    if spec.kind == "float":
        try:
            return float(spec.default) if spec.default is not None else None
        except (TypeError, ValueError):
            return None
    if spec.kind == "choice":
        if isinstance(spec.default, str):
            return spec.default
        return None
    if spec.kind == "str":
        return str(spec.default) if spec.default is not None else ""
    # sequence / none / other 一律用可编辑文本展示
    return spec.default_repr


def build_component(spec: ParamSpec, **kwargs) -> gr.Component:
    default_info = "必填" if spec.required else f"默认值：{spec.default_repr}"
    common: dict[str, Any] = {
        "label": spec.name,
        "info": default_info,
        "interactive": True,
    }
    common.update(kwargs)
    value = _initial_value(spec)

    if spec.kind == "bool":
        return gr.Checkbox(value=bool(value), **common)
    if spec.kind == "int":
        return gr.Number(value=value, precision=0, **common)
    if spec.kind == "float":
        return gr.Number(value=value, precision=None, **common)
    if spec.kind == "choice":
        return gr.Dropdown(
            choices=spec.choices or [],
            value=value,
            allow_custom_value=True,
            **common,
        )
    if spec.kind == "str":
        return gr.Textbox(value=value or "", **common)
    if spec.kind == "none":
        common["info"] = f"{default_info}（可输入 None 或数值/字符串）"
    return gr.Textbox(value=value or "", **common)


def coerce_value(spec: ParamSpec, value: Any) -> Any:
    """把组件返回值转换成估计器初始化可以接受的 Python 对象。"""
    if spec.kind == "bool":
        return bool(value)

    if spec.kind == "int":
        if value in (None, ""):
            return spec.default if not spec.required else 0
        return int(float(value))

    if spec.kind == "float":
        if value in (None, ""):
            return spec.default if not spec.required else 0.0
        return float(value)

    if spec.kind == "choice":
        return value if value not in (None, "") else spec.default

    if spec.kind == "str":
        if value == "" and spec.default is not None:
            return spec.default
        return str(value)

    if spec.kind == "sequence":
        if isinstance(value, (list, tuple)):
            return list(value)
        if value in (None, ""):
            return spec.default
        if value == spec.default_repr:
            return spec.default
        try:
            return json.loads(value)
        except Exception:
            return [v.strip() for v in str(value).split(",") if v.strip()]

    # none / other
    if value in (None, ""):
        return spec.default
    if value == spec.default_repr:
        return spec.default
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() == "none":
            return None
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def build_parameter_panel(specs: list[ParamSpec]) -> list[gr.Component]:
    return [build_component(spec) for spec in specs]


def collect_kwargs(specs: list[ParamSpec], values: list[Any]) -> dict[str, Any]:
    if len(specs) != len(values):
        raise ValueError(
            f"参数规格数量（{len(specs)}）与组件值数量（{len(values)}）不一致"
        )
    return {spec.name: coerce_value(spec, v) for spec, v in zip(specs, values)}