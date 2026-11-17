"""评估页：基于当前会话的 estimator 生成诊断图表。"""
from __future__ import annotations

import gradio as gr
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

from core.registry import Task


# ---------------------------------------------------------------------------
# 分类
# ---------------------------------------------------------------------------
def _build_classification_figure(estimator, X, y, target_classes, algorithm):
    fig = Figure(figsize=(12, 5.5))

    y_pred = estimator.predict(X)
    cm = confusion_matrix(y, y_pred)

    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(cm, cmap="Blues")
    ax1.set_title("Confusion Matrix")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=9)
    fig.colorbar(im, ax=ax1, fraction=0.046)

    ax2 = fig.add_subplot(1, 2, 2)
    _plot_roc(ax2, estimator, X, y, target_classes)

    fig.tight_layout()
    return fig


def _plot_roc(ax, estimator, X, y, target_classes):
    n_classes = len(target_classes) if target_classes else len(np.unique(y))

    y_score: np.ndarray | None = None
    scorer: str | None = None
    if hasattr(estimator, "predict_proba"):
        try:
            y_score = np.asarray(estimator.predict_proba(X))
            scorer = "proba"
        except Exception:
            y_score = None
    elif hasattr(estimator, "decision_function"):
        try:
            y_score = np.asarray(estimator.decision_function(X))
            scorer = "decision"
        except Exception:
            y_score = None

    if y_score is None:
        ax.text(0.5, 0.5, "ROC unavailable\n(no proba / decision_function)",
                ha="center", va="center", fontsize=10, color="gray")
        ax.axis("off")
        return

    classes_ordered = list(target_classes) if target_classes else sorted(np.unique(y))

    if n_classes == 2:
        scores = y_score[:, 1] if scorer == "proba" else y_score.ravel()
        fpr, tpr, _ = roc_curve(y, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=1.5, label=f"AUC = {roc_auc:.3f}")
    else:
        y_bin = label_binarize(y, classes=classes_ordered)
        if y_bin.shape[1] == 1:
            y_bin = np.hstack([1 - y_bin, y_bin])
        for i, cls in enumerate(classes_ordered):
            if i >= y_bin.shape[1]:
                continue
            if scorer == "proba" and i < y_score.shape[1]:
                scores_i = y_score[:, i]
            elif scorer == "decision" and y_score.ndim == 2 and i < y_score.shape[1]:
                scores_i = y_score[:, i]
            elif scorer == "decision" and y_score.ndim == 1:
                scores_i = y_score
            else:
                continue
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, i], scores_i)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, linewidth=1,
                        label=f"class {cls} (AUC={roc_auc:.3f})")
            except Exception:
                continue

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curve")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# 回归
# ---------------------------------------------------------------------------
def _build_regression_figure(estimator, X, y, algorithm):
    fig = Figure(figsize=(12, 5.5))
    y_pred = estimator.predict(X)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(y, y_pred, alpha=0.5, s=12)
    lo = float(min(y.min(), y_pred.min()))
    hi = float(max(y.max(), y_pred.max()))
    ax1.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax1.set_xlabel("True")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Predicted vs True")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2)
    residuals = y - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5, s=12)
    ax2.axhline(0, color="r", linestyle="--", linewidth=1)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Residual")
    ax2.set_title("Residual Plot")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 特征重要性
# ---------------------------------------------------------------------------
def _extract_raw_importance(model):
    """从单个模型对象（不含 Pipeline）提取重要性；无则返回 None。"""
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        return np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    return None


def _aggregate_via_pipeline(estimator, original_features):
    """把 Pipeline 内部的特征重要性聚合回原始特征名。

    处理后的名称格式：``<transformer>__<original>_<value>``（OneHot）
    或 ``<transformer>__<original>``（缩放/序数编码）。返回 {原始特征: 重要性}
    或 None（无法聚合）。
    """
    if not isinstance(estimator, Pipeline) or len(estimator.steps) < 2:
        return None
    preproc = estimator.steps[0][1]
    model = estimator.steps[-1][1]
    raw = _extract_raw_importance(model)
    if raw is None:
        return None
    try:
        transformed = list(preproc.get_feature_names_out())
    except Exception:
        return None
    if len(transformed) != len(raw):
        return None

    # 按长度降序匹配，避免 `a` 与 `ab` 的前缀歧义
    ordered_features = sorted(original_features, key=len, reverse=True)
    aggregated = {name: 0.0 for name in original_features}
    matched_any = False
    for tname, imp in zip(transformed, raw):
        if "__" in tname:
            _, rest = tname.split("__", 1)
        else:
            rest = tname
        hit = None
        for orig in ordered_features:
            if rest == orig or rest.startswith(orig + "_"):
                hit = orig
                break
        if hit is None:
            continue
        aggregated[hit] += abs(float(imp))
        matched_any = True

    return aggregated if matched_any else None


def _build_importance_figure(estimator, feature_names):
    """支持 Pipeline：把 OneHot 展开后的哑变量重要性聚合回原始特征。"""
    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    aggregated = _aggregate_via_pipeline(estimator, feature_names)

    if aggregated is not None:
        names = list(aggregated.keys())
        values = np.array([aggregated[n] for n in names])
    else:
        target = estimator
        if isinstance(estimator, Pipeline) and estimator.steps:
            target = estimator.steps[-1][1]
        raw = _extract_raw_importance(target)
        if raw is None:
            ax.text(0.5, 0.5, "No feature_importances_ or coef_",
                    ha="center", va="center", fontsize=10, color="gray")
            ax.axis("off")
            return fig
        if len(raw) != len(feature_names):
            if isinstance(estimator, Pipeline):
                msg = ("Feature importance unavailable\n"
                       "(dimension changed after preprocessing)")
            else:
                msg = "Dimension mismatch"
            ax.text(0.5, 0.5, msg,
                    ha="center", va="center", fontsize=10, color="gray")
            ax.axis("off")
            return fig
        names = list(feature_names)
        values = raw

    order = np.argsort(values)[::-1]
    names_sorted = [names[i] for i in order]
    values_sorted = values[order]

    ax.barh(range(len(names_sorted)), values_sorted[::-1], color="steelblue")
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted[::-1], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# 聚类
# ---------------------------------------------------------------------------
def _build_clustering_figure(estimator, X, algorithm):
    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    # Pipeline 场景下取最后一步的 labels_
    target = estimator
    if isinstance(estimator, Pipeline) and estimator.steps:
        target = estimator.steps[-1][1]

    labels = getattr(target, "labels_", None)
    if labels is None:
        ax.text(0.5, 0.5, "No labels_ attribute",
                ha="center", va="center", fontsize=10, color="gray")
        ax.axis("off")
        return fig

    counts = pd.Series(labels).value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color="steelblue")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    ax.set_title(f"Cluster Sizes (n={len(counts)})")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------
def build(session_state: gr.State) -> None:
    gr.Markdown(
        "## 模型诊断\n"
        "基于【训练】页当前会话的 `estimator` 生成诊断图表。"
        "如未训练，请先在【训练】页完成训练。"
    )

    with gr.Row():
        split_radio = gr.Radio(
            choices=[("测试集", "test"), ("训练集", "train")],
            value="test",
            label="评估数据集",
        )
        evaluate_btn = gr.Button("🔍 生成诊断图", variant="primary")

    summary_md = gr.Markdown("*尚未评估*")
    diagnostic_plot = gr.Plot(label="诊断图表")
    report_df = gr.Dataframe(label="评估报告", interactive=False, wrap=True)
    importance_plot = gr.Plot(label="特征重要性")

    def _on_evaluate(session, split):
        if session is None:
            return ("❌ 当前无训练会话，请先在【训练】页完成训练",
                    None, pd.DataFrame({"提示": ["无会话"]}), None)
        estimator = getattr(session, "estimator", None)
        if estimator is None:
            return ("❌ 会话中无 estimator",
                    None, pd.DataFrame({"提示": ["无 estimator"]}), None)

        if split == "test":
            X, y = session.X_test, session.y_test
            label = "测试集"
        else:
            X, y = session.X_train, session.y_train
            label = "训练集"

        if X is None or len(X) == 0:
            return (f"❌ {label}为空", None,
                    pd.DataFrame({"提示": ["无样本"]}), None)

        task = session.task
        algorithm = session.algorithm
        summary = f"**算法**：`{algorithm}` · **任务**：`{task}` · **{label}**：{len(X)} 样本"

        try:
            if task == Task.CLASSIFICATION:
                if y is None:
                    return (summary + "\n\n❌ 分类任务缺少标签",
                            None, pd.DataFrame({"提示": ["无标签"]}), None)
                fig = _build_classification_figure(
                    estimator, X, y, session.target_classes, algorithm
                )
                y_pred = estimator.predict(X)
                report = classification_report(
                    y, y_pred, output_dict=True, zero_division=0
                )
                report_df_ = pd.DataFrame(report).transpose()
                importance_fig = _build_importance_figure(
                    estimator, session.feature_names
                )
                return summary, fig, report_df_, importance_fig

            if task == Task.REGRESSION:
                if y is None:
                    return (summary + "\n\n❌ 回归任务缺少标签",
                            None, pd.DataFrame({"提示": ["无标签"]}), None)
                fig = _build_regression_figure(estimator, X, y, algorithm)
                y_pred = estimator.predict(X)
                metrics = {
                    "MSE": mean_squared_error(y, y_pred),
                    "MAE": mean_absolute_error(y, y_pred),
                    "R²": r2_score(y, y_pred),
                }
                metrics_df = pd.DataFrame(
                    [{"指标": k, "数值": round(float(v), 6)}
                     for k, v in metrics.items()]
                )
                importance_fig = _build_importance_figure(
                    estimator, session.feature_names
                )
                return summary, fig, metrics_df, importance_fig

            if task in (Task.CLUSTERING, Task.BICLUSTERING):
                fig = _build_clustering_figure(estimator, X, algorithm)
                info_df = pd.DataFrame({"提示": ["聚类任务无标准标签评估指标"]})
                return summary, fig, info_df, None

            return (summary + f"\n\n当前任务 `{task}` 暂无可视化诊断图",
                    None, pd.DataFrame({"提示": ["暂不支持"]}), None)
        except Exception as e:
            return (summary + f"\n\n❌ {type(e).__name__}: {e}",
                    None, pd.DataFrame({"提示": [str(e)]}), None)

    evaluate_btn.click(
        _on_evaluate,
        inputs=[session_state, split_radio],
        outputs=[summary_md, diagnostic_plot, report_df, importance_plot],
    )