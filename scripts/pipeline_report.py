#!/usr/bin/env python3
"""Deterministic figures and Word/Markdown reports from measured Pipeline outputs."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#656565"


def style():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    fonts = [f for f in ("Arial", "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "DejaVu Sans")
             if f in installed]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": fonts,
                         "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.8, "legend.frameon": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "figure.facecolor": "white", "axes.facecolor": "white"})


# Figures stay in English (international publication); these replace vague panel text with standard terms.
FIGURE_TEXT = {
    "Development": "Training set", "Test": "Test set",
    "Validation gain over dummy baseline": "Improvement over baseline (validation folds)",
    "Training versus validation": "Training vs. validation score", "Paired baseline comparison": "Improvement over baseline",
    "Held-out performance and uncertainty": "Test-set performance (95% CI)", "Permutations": "Number of permutations",
    "Selected model": "Final model", "Dummy baseline": "Baseline",
    "Observed outcome": "Observed", "Predicted outcome": "Predicted", "Residual (observed - predicted)": "Residual (observed − predicted)",
    "Fraction at or below error": "Cumulative proportion", "Mean predicted outcome in bin": "Mean predicted value (bin)",
    "Prediction agreement": "Predicted vs. observed", "Bias and residual spread": "Residuals",
    "Error distribution": "Absolute error distribution", "Error across prediction levels": "Error by prediction range",
    "Class metric": "Score", "Where predictions fail": "Confusion matrix (row-normalised)", "Performance by class": "Per-class performance",
    "Observed positive fraction": "Observed positive rate", "Probability reliability": "Calibration curve",
    "Evidence supporting each bin": "Samples per bin", "Positive fraction": "Predicted positive rate",
    "Threshold trade-offs on held-out data": "Test-set performance by threshold",
    "Feature reliance and repeat variability": "Permutation importance", "Input feature": "Feature",
    "Training Spearman correlation": "Spearman correlation (training set)", "Correlated predictors can share reliance": "Feature correlations",
    "(Test mean - training mean) / SD": "Standardised mean difference (test − training)",
    "Feature distribution shift": "Mean difference", "Missingness shift": "Missing rate", "Missing fraction": "Missing rate",
}


def localize_figure(fig):
    """Clear English figure text: model display names, metric names, standard panel titles."""
    import re
    from matplotlib.text import Text
    from matplotlib.ticker import FixedLocator
    from report_explanations import model_name
    from report_narrative import metric_label

    def translate(text):
        prefix = ""
        match = re.match(r"^([a-d])  (.*)$", text, re.S)
        if match:
            prefix, text = match.group(1) + "  ", match.group(2)
        model = "Model" + text.replace("\n", "")
        if model_name(model, False) != model:
            name = model_name(model, False)
            return prefix + name[:1].upper() + name[1:]
        m = re.match(r"^(Held-out )?(\w+) \((lower|higher) is better\)$", text)
        if m:
            key = {"f1": "f1_macro", "precision": "precision_macro", "recall": "recall_macro"}.get(m.group(2), m.group(2))
            label = metric_label(key, False)
            label = label[:1].upper() + label[1:] if label.islower() else label
            if m.group(1) and not label[:2].isupper():
                label = label[:1].lower() + label[1:]
            return prefix + f"{'Test-set ' if m.group(1) else ''}{label} ({m.group(3)} is better)"
        m = re.match(r"^Decrease in (\w+) scorer", text)
        if m:
            return prefix + f"Change in {metric_label(m.group(1), False)} after permutation (larger = more important)"
        m = re.match(r"^Held-out performance by (time block|group)$", text)
        if m:
            return prefix + f"Test-set performance by {m.group(1)}"
        if text.endswith(")") and " (failed: " in text:
            name, failed = text[:-1].split(" (failed: ")
            label = model_name("Model" + name, False)
            return prefix + f"{label[:1].upper() + label[1:]} ({failed} failed)"
        return prefix + FIGURE_TEXT.get(text, text)

    fig.canvas.draw()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if not isinstance(axis.get_major_locator(), FixedLocator):
                continue  # numeric ticks are left to their formatter
            labels = [label.get_text() for label in axis.get_ticklabels()]
            updated = [translate(label) for label in labels]
            if updated != labels:
                axis.set_ticks(axis.get_ticklocs(), updated)
    for item in fig.findobj(Text):
        original = item.get_text()
        if original:
            updated = translate(original)
            if updated != original:
                item.set_text(updated)


def save_figure(fig, directory, name):
    fig.canvas.draw()
    # Optional local publication QA; ordinary installs need no external skills.
    import os
    import sys
    if os.environ.get("PSYTRAINER_FIGURE_QA_PATH"):
        sys.path.insert(0, os.environ["PSYTRAINER_FIGURE_QA_PATH"])
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig, json_out=str(directory / f"{name}.alignment.json"),
                                           tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(directory / f"{name}.png", dpi=600)
    fig.savefig(directory / f"{name}.svg")
    fig.savefig(directory / f"{name}.pdf")
    plt.close(fig)


def plots(directory, summary):
    style()
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    cv = pd.read_csv(directory / "cv-scores.csv")
    pred = pd.read_csv(directory / "test-predictions.csv", dtype={"sample_id": str, "observed": str, "predicted": str})
    zh = summary["language"] == "zh"
    n, folds = summary["test_n"], summary["cv_folds"]
    captions = []

    def add(fig, name, caption, source):
        localize_figure(fig)
        save_figure(fig, figures, name)
        captions.append({"name": name, "caption": caption, "source": source})

    if summary.get("search", {}).get("method", "none") != "none":
        trials = json.loads((directory / "search-results.json").read_text())
        # Paginate models to keep labels readable; all candidate values stay in JSON.
        names = list(dict.fromkeys(row["model"] for row in trials))
        for page in range(0, len(names), 3):
            fig, axes = plt.subplots(len(names[page:page + 3]), 1,
                figsize=(7.09, 2.0 * len(names[page:page + 3])), squeeze=False, layout="constrained")
            for ax, name in zip(axes.ravel(), names[page:page + 3]):
                valid = [row for row in trials if row["model"] == name and row["status"] == "completed"]
                ax.scatter([row["candidate"] for row in valid], [row["mean"] for row in valid], color=BLUE, s=20)
                failed = sum(row["model"] == name and row["status"] == "failed" for row in trials)
                ax.set(title=f"{name.removeprefix('Model')} (failed: {failed})",
                       xlabel="Candidate index", ylabel=summary["metric"])
            add(fig, f"parameter-search-{page // 3 + 1}",
                ("参数搜索：每点为一个候选参数配置的平均验证得分，候选编号对应源文件。搜索使用训练集，分数用于选择而非无偏性能估计；失败数单独列出。" if zh else
                 "Parameter search: each point is mean training CV score for a candidate indexed in the source file. Scores are for selection, not unbiased performance estimates; failed candidates are counted separately."),
                "search-results.json")

    if (directory / "baseline-cv.csv").exists():
        from pipeline_diagnostics_plots import diagnostic_plots
        diagnostic_plots(directory, summary, add)
        (figures / "figure-manifest.json").write_text(json.dumps({"width_mm": 180, "dpi": 600,
            "formats": ["png", "svg", "pdf"], "style": "Nature-inspired; no journal compliance certification",
            "figures": captions}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return captions

    fig, ax = plt.subplots(figsize=(7.09, 3.6), layout="constrained")
    for i, row in enumerate(summary["comparison"]):
        scores = cv.loc[cv.model == row["model"], "score"].to_numpy()
        ax.scatter(scores, np.full(len(scores), i) + np.linspace(-0.1, 0.1, len(scores)),
                   color=GRAY, alpha=0.6, s=18)
        ax.errorbar(row["mean"], i, xerr=row["sd"], fmt="o", color=BLUE, capsize=3)
    ax.set_yticks(range(len(summary["comparison"])),
                  [textwrap.fill(r["model"].removeprefix("Model"), 24) for r in summary["comparison"]])
    ax.invert_yaxis()
    ax.set_ylim(len(summary["comparison"]) - 0.5, -0.5)
    ax.set_xlabel(f"{summary['metric']} ({summary['direction']} is better)")
    add(fig, "cv-comparison",
        (f"模型选择：{folds} 折验证得分。灰点为各折，蓝点和误差线为均值及交叉验证标准差（非置信区间）。各模型使用相同划分。"
         if zh else f"Model selection across {folds} folds. Gray points: individual folds; blue: mean and sample SD, not a confidence interval. Splits are shared across models."),
        "cv-scores.csv")

    if summary["task"] == "regression":
        observed = pd.to_numeric(pred.observed)
        predicted = pd.to_numeric(pred.predicted)
        fig, ax = plt.subplots(figsize=(7.09, 4.0), layout="constrained")
        ax.scatter(observed, predicted, color=BLUE, s=16, alpha=0.65, rasterized=True)
        lower, upper = min(observed.min(), predicted.min()), max(observed.max(), predicted.max())
        ax.plot([lower, upper], [lower, upper], color=GRAY, linestyle="--", linewidth=1)
        ax.set(xlabel="Observed outcome", ylabel="Predicted outcome")
        add(fig, "observed-predicted",
            (f"独立测试集预测表现（n={n} 个样本）。每点为一个样本，虚线为理想预测线；未绘制推断区间。"
             if zh else f"Held-out predictions (n={n} samples). Each point is one sample; dashed line indicates ideal predictions. No inferential intervals."),
            "test-predictions.csv")
        fig, ax = plt.subplots(figsize=(7.09, 4.0), layout="constrained")
        ax.scatter(predicted, observed - predicted, color=ORANGE, s=16, alpha=0.65, rasterized=True)
        ax.axhline(0, color=GRAY, linestyle="--", linewidth=1)
        ax.set(xlabel="Predicted outcome", ylabel="Residual (observed - predicted)")
        add(fig, "residuals",
            (f"独立测试集残差诊断（n={n}）。零线表示无预测误差；散点用于检查系统偏差和误差随预测值变化的模式。"
             if zh else f"Held-out residual diagnostics (n={n}). Zero denotes no prediction error. Points show bias and variation across predicted outcomes."),
            "test-predictions.csv")
    else:
        classes = summary["classes"]
        matrix = confusion_matrix(pred.observed, pred.predicted, labels=classes)
        pd.DataFrame(matrix, index=classes, columns=classes).to_csv(directory / "confusion-matrix.csv", index_label="observed")
        fig, ax = plt.subplots(figsize=(7.09, 4.5), layout="constrained")
        ax.imshow(matrix, cmap="Blues", aspect="equal")
        # Numeric class keys keep long labels out of a fixed-size matrix.
        ax.set_xticks(range(len(classes)), [f"C{i}" for i in range(len(classes))])
        ax.set_yticks(range(len(classes)), [f"C{i}" for i in range(len(classes))])
        if len(classes) <= 12:
            for (i, j), value in np.ndenumerate(matrix):
                ax.text(j, i, str(value), ha="center", va="center",
                        color="white" if value > matrix.max() / 2 else "black", fontsize=8)
        ax.set(xlabel="Predicted class", ylabel="Observed class")
        mapping = "; ".join(f"C{i}={value}" for i, value in enumerate(classes))
        add(fig, "confusion-matrix",
            ((f"独立测试集混淆矩阵（n={n}），单元格为样本数。类别映射：" if zh else
              f"Held-out confusion matrix (n={n}); cells contain sample counts. Class key: ") + mapping),
            "confusion-matrix.csv; test-predictions.csv")
        if "positive_score" in pred and pred.observed.nunique() == 2:
            binary = pred.observed == classes[1]
            fpr, tpr, _ = roc_curve(binary, pred.positive_score)
            precision, recall, _ = precision_recall_curve(binary, pred.positive_score)
            pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr}).to_csv(directory / "roc.csv", index=False)
            pd.DataFrame({"recall": recall, "precision": precision}).to_csv(directory / "precision-recall.csv", index=False)
            for name, xx, yy, xlabel, ylabel, baseline in (
                ("roc", fpr, tpr, "False positive rate", "True positive rate", None),
                ("precision-recall", recall, precision, "Recall", "Precision", float(binary.mean())),
            ):
                fig, ax = plt.subplots(figsize=(7.09, 4.0), layout="constrained")
                if name == "roc":
                    ax.plot(xx, yy, color=BLUE, linewidth=1.4)
                else:
                    ax.step(xx, yy, where="post", color=BLUE, linewidth=1.4)
                if baseline is None:
                    ax.plot([0, 1], [0, 1], linestyle="--", color=GRAY, linewidth=1)
                else:
                    ax.axhline(baseline, linestyle="--", color=GRAY, linewidth=1)
                ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
                add(fig, name,
                    (f"独立测试集 {name} 曲线（n={n}，阳性类别={classes[1]}）。虚线为随机基线，曲线无置信区间。"
                     if zh else f"Held-out {name} curve (n={n}; positive class={classes[1]}). Dashed line: chance baseline. No confidence interval."),
                    f"{name}.csv; test-predictions.csv")

    if (directory / "importance.csv").exists():
        all_importance = pd.read_csv(directory / "importance.csv", dtype={"feature": str})
        imp = all_importance.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7.09, max(3.6, 0.3 * len(imp) + 1.2)), layout="constrained")
        ax.errorbar(imp["mean"], np.arange(len(imp)), xerr=imp.sd, fmt="o", color=BLUE, capsize=3)
        ax.set_yticks(range(len(imp)), [textwrap.fill(s, 25) for s in imp.feature])
        ax.axvline(0, color=GRAY, linestyle="--", linewidth=1)
        ax.set_xlabel(f"Decrease in {summary['metric']} scorer (higher = greater reliance)")
        add(fig, "permutation-importance",
            (f"测试集置换重要性（n={n}，{summary['importance_repeats']} 次置换）。展示前 {len(imp)}/{len(all_importance)} 个特征；全部数值保存在源数据。误差线为置换次数间总体标准差，非置信区间。使用越大越好的评分，损失指标取负值。重要性不表示因果效应。"
             if zh else f"Test permutation importance (n={n}; {summary['importance_repeats']} repeats). Top {len(imp)}/{len(all_importance)} features displayed; all values retained in source data. Bars: population SD across permutations, not confidence intervals. Scorers are oriented higher-is-better (losses negated). Importance is not causal."),
            "importance.csv; importance-repeats.csv")
    (figures / "figure-manifest.json").write_text(json.dumps({"width_mm": 180, "dpi": 600,
        "formats": ["png", "svg", "pdf"], "style": "Nature-inspired; no journal compliance certification",
        "figures": captions}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return captions


# Figure families grouped by the results subsection that discusses them.
SECTION_FIGURES = {
    "comparison": ("model-validation", "parameter-search", "cv-comparison"),
    "test": ("test-performance", "permutation-test"),
    "errors": ("regression-diagnostics", "classification-errors", "classification-discrimination",
               "probability-calibration", "threshold-tradeoffs", "observed-predicted", "residuals",
               "confusion-matrix", "roc", "precision-recall"),
    "predictors": ("permutation-importance", "feature-correlation"),
    "shift": ("feature-shift",),
    "strata": ("strata-performance",),
}


def figure_family(name):
    for families in SECTION_FIGURES.values():
        for family in families:
            if name == family or name.startswith(family + "-"):
                return family
    return name


def paper_caption(family, figure, s, n, zh):
    """Short journal-style caption; technical detail stays in figures/figure-manifest.json."""
    import re
    metric = n.mlabel()
    tech = figure.get("caption", "")
    key = re.search(r"(?:类别映射|映射|Class key|Key)[：:]\s*(.+?)[。.]?\s*$", tech)
    key = key.group(1).rstrip("。.") if key else ""
    classes = s.get("classes") or []
    shown = [str(c) for c in classes]
    if family == "classification-errors" and classes and len(classes) <= 6 and max(len(x) for x in shown) <= 12:
        key = ""  # figure already shows class names
    positive = n.cls(s["classes"][1]) if len(s.get("classes") or []) == 2 else ""
    unit = ("组" if s.get("split") == "group" else "时间块") if zh else ("group" if s.get("split") == "group" else "time block")
    ci = "误差线为 95% CI（测试集 bootstrap）。" if (n.f.get("interval") or {}).get("omitted_reason") is None else ""
    ci_en = "Error bars are 95% CIs (test-set bootstrap). " if ci else ""
    captions = {
        "model-validation": ((f"各候选模型的交叉验证表现：每折的训练得分（灰）与验证得分（蓝）。",
                              f"Cross-validated performance of candidate models: training (grey) and validation (blue) scores per fold.")
                             if s.get("analysis_level", "basic") == "basic" else
                             (f"各候选模型的交叉验证表现。(a) 每折的训练得分（灰）与验证得分（蓝）；(b) 每折中模型相对基线模型的提升，正值表示优于基线，绿色为均值 ± SD。",
                              f"Cross-validated performance of candidate models. (a) Training (grey) and validation (blue) scores per fold; (b) improvement over the baseline per fold (positive = better), green = mean ± SD.")),
        "test-performance": (f"最终模型与基线模型在测试集上的 {metric}。{ci}", f"Test-set {metric} of the final model and the baseline. {ci_en}"),
        "permutation-test": (f"置换检验。灰色直方图为将训练集标签随机打乱 {(s.get('permutation_test') or {}).get('valid')} 次、每次重新训练后在测试集上得到的 {metric}（即特征不含信息时的成绩分布）；蓝线为实际模型的成绩。",
                             f"Permutation test. Grey histogram: test-set {metric} after refitting on randomly shuffled training labels ({(s.get('permutation_test') or {}).get('valid')} permutations), i.e. performance when features carry no information; blue line: observed score."),
        "regression-diagnostics": ("测试集预测误差。(a) 预测值与实际值，虚线为 y = x；(b) 残差（实际 − 预测）与预测值，曲线为分箱均值；(c) 绝对误差的累积分布；(d) 不同预测区间的 MAE。",
                                   "Test-set prediction error. (a) Predicted vs. observed, dashed line y = x; (b) residuals (observed − predicted) vs. predicted, line = binned mean; (c) cumulative distribution of absolute error; (d) MAE by prediction range."),
        "classification-errors": (f"测试集混淆矩阵与各类别表现。(a) 按实际类别归一化的混淆矩阵；(b) 各类别的精确率、召回率与 F1。" + (f"类别编码：{key}。" if key else ""),
                                  f"Test-set confusion matrix and per-class performance. (a) Confusion matrix normalised by true class; (b) precision, recall and F1 per class." + (f" Class codes: {key}." if key else "")),
        "classification-discrimination": (f"测试集 ROC 曲线与精确率–召回率曲线（正类：{positive}）。虚线为随机水平。",
                                          f"Test-set ROC and precision–recall curves (positive class: {positive}). Dashed lines show chance level."),
        "probability-calibration": (f"校准曲线（正类：{positive}）。(a) 各概率区间的平均预测概率与实际正类比例，虚线为完美校准；(b) 各区间样本数。",
                                    f"Calibration curve (positive class: {positive}). (a) Mean predicted probability vs. observed positive rate per bin, dashed line = perfect calibration; (b) samples per bin."),
        "threshold-tradeoffs": ("不同决策阈值下的灵敏度、特异度与预测为正类的比例。虚线为默认阈值。",
                                "Sensitivity, specificity and fraction predicted positive across decision thresholds. Dashed line = default threshold."),
        "permutation-importance": (f"测试集置换重要性（Δ{metric}）。灰点为单次置换，蓝点与误差线为均值 ± SD。",
                                   f"Test-set permutation importance (Δ{metric}). Grey = single permutations; blue = mean ± SD."),
        "feature-correlation": ("训练集特征之间的 Spearman 相关。" + (f"编码：{key}。" if key else ""),
                                "Spearman correlations between features in the training set." + (f" Codes: {key}." if key else "")),
        "feature-shift": ("训练集与测试集的特征分布比较。左：标准化均值差（以训练集 SD 为单位）；右：缺失率。",
                          "Feature distributions in training vs. test set. Left: standardised mean difference (training-set SD units); right: missing rate."),
        "strata-performance": (f"测试集各{unit}的 {('MAE' if s['task'] == 'regression' else '准确率')}，括号内为样本量。",
                               f"Test-set {('MAE' if s['task'] == 'regression' else 'accuracy')} by {unit}; sample sizes in parentheses."),
        "parameter-search": ("参数搜索：各候选配置的交叉验证得分。", "Parameter search: cross-validated score of each candidate configuration."),
    }
    return captions.get(family, (tech, tech))[0 if zh else 1]


def cjk_font_name():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in ("Noto Serif CJK SC", "Noto Sans CJK SC", "SimSun", "Microsoft YaHei",
                             "Songti SC", "PingFang SC") if f in installed), "SimSun")


def save_context(directory, context):
    """Validate a context file and keep it with the run for later regenerations."""
    if context is None:
        return
    from report_facts import validate_context
    data = validate_context(json.loads(Path(context).read_text(encoding="utf-8")))
    (Path(directory) / "report-context.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


ANALYSIS_LEVELS = ("basic", "full")


def generate_report(directory, context=None, language=None, analysis=None):
    from report_document import ReportDocument
    from report_explanations import (FIGURES, METRICS, PER_MODEL_SPLIT_THRESHOLD, preprocessing,
                                     validation_methods)
    from report_facts import collect
    from report_narrative import STANDING, Narrator, fmt, interval, pct

    directory = Path(directory)
    save_context(directory, context)
    s = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if language:
        s["language"] = language
    if analysis:
        s["analysis_level"] = analysis
    s.setdefault("analysis_level", "basic")
    basic = s["analysis_level"] == "basic"
    from pipeline_analysis import analyze
    analysis = analyze(directory, s)
    captions = plots(directory, s)
    zh = s["language"] == "zh"
    facts = collect(directory, s, analysis)
    n = Narrator(s, facts, zh)
    T = n.t
    title = facts["context"].get("title") or T(f"{n.target}预测模型分析报告", f"Predicting {n.target}: model analysis report")
    doc = ReportDocument(title, zh)

    by_family = {}
    for figure in captions:
        by_family.setdefault(figure_family(figure["name"]), []).append(figure)
    used = []

    def figures(section):
        for family in SECTION_FIGURES[section]:
            for figure in by_family.get(family, []):
                source = figure["source"]
                if basic:
                    source = "; ".join(x for x in source.split("; ") if not x.startswith("baseline"))
                label = doc.figure(directory / "figures" / f"{figure['name']}.png",
                                   paper_caption(family, figure, s, n, zh), source)
                used.append((label, family))

    def labels_for(section):
        count = sum(len(by_family.get(f, [])) for f in SECTION_FIGURES[section])
        return [doc.next_figure_label(i) for i in range(1, count + 1)]

    def ref(section):
        names = labels_for(section)
        if not names:
            return ""
        return T(f"（{names[0]}" + (f"–{names[-1].split()[-1]}" if len(names) > 1 else "") + "）",
                 f" ({names[0]}" + (f"–{names[-1].split()[-1]}" if len(names) > 1 else "") + ")")

    # Abstract and key findings ------------------------------------------------
    doc.heading(T("摘要", "Abstract"))
    for label, text in n.abstract():
        doc.labelled(label, text)
    doc.heading(T("要点", "Key findings"))
    doc.bullets(n.key_points(), ordered=True)

    # Methods -----------------------------------------------------------------
    doc.heading(T("1 方法", "1 Methods"))
    doc.heading(T("1.1 数据", "1.1 Data"), 2)
    for text in n.data_description():
        doc.para(text)
    doc.heading(T("1.2 验证设计", "1.2 Validation design"), 2)
    doc.para(" ".join(validation_methods(s, zh)))
    doc.heading(T("1.3 预处理", "1.3 Preprocessing"), 2)
    doc.para(" ".join(preprocessing(s, zh)))
    doc.heading(T("1.4 候选模型与选择", "1.4 Candidate models and selection"), 2)
    doc.para(n.selection_methods())
    search = s.get("search") or {"method": "none"}
    if search.get("method", "none") != "none":
        doc.para(T(f"参数通过{'网格' if search['method'] == 'grid' else '随机'}搜索在训练集内选择，共评估 {search['evaluated_candidates']} 个配置（失败 {search['failed_candidates']} 个，每个模型上限 {search.get('max_candidates')} 个）；各配置的验证结果见 search-results.json。",
                   f"Parameters were chosen by {search['method']} search on training data: {search['evaluated_candidates']} configurations evaluated ({search['failed_candidates']} failed; budget {search.get('max_candidates')} per model); see search-results.json."))
    else:
        doc.para(T("各模型使用默认或显式指定的固定参数，未进行参数搜索。", "Models used default or explicitly configured fixed parameters; no parameter search was run."))
    if s["preprocessing"].get("selection") == "forward":
        doc.para(T("前向特征筛选只使用当前训练分区，其内部交叉验证沿用相同的划分设计。",
                   "Forward selection used only the current training partition, with inner cross-validation following the same design."))
    doc.heading(T("1.5 评价与统计方法", "1.5 Evaluation and statistics"), 2)
    for text in n.evaluation_methods(facts["interval"] or {}):
        doc.para(text)

    # Results -----------------------------------------------------------------
    doc.heading(T("2 结果", "2 Results"))
    doc.heading(T("2.1 模型比较", "2.1 Model comparison"), 2)
    comparison_text = n.comparison_results()
    if comparison_text:
        comparison_text[0] = comparison_text[0].rstrip("。.") + ref("comparison") + T("。", ".")
    for text in comparison_text:
        doc.para(text)
    metric = n.mlabel()
    headers = [T("模型", "Model"), T("类型", "Family"), T(f"CV {metric}（均值 ± SD）", f"CV {metric} (mean ± SD)"),
               T("各折范围", "Fold range"), T(f"训练集 {metric}", f"Train {metric}"), T("优于基线的折数", "Folds beating baseline"),
               T("结论", "Assessment")]
    if basic:
        headers.pop(5)
    rows = []
    for row in facts["comparison"]:
        rows.append([n.model(row["model"]) + (" *" if row["standing"] in ("selected", "sole") else ""), n.family(row["family"]),
                     f"{fmt(row['mean'])} ± {fmt(row.get('sd'))}",
                     f"{fmt(row.get('min'))}–{fmt(row.get('max'))}" if row.get("fold_scores") else "NA",
                     fmt(row.get("train_mean")),
                     f"{row['baseline_gain_positive_folds']}/{row['folds']}" if row.get("baseline_gain_positive_folds") is not None else "NA",
                     STANDING[row["standing"]][0 if zh else 1]])
        if basic:
            rows[-1].pop(5)
    doc.table(T("候选模型的交叉验证表现", "Cross-validated performance of candidate models"), headers, rows,
              T(f"* 最终模型。{metric}{n.better()}。CV 为交叉验证（验证折上的得分）；训练集{metric}为各折训练部分上的平均得分，若远好于 CV 得分则提示过拟合。“结论”按与最终模型的差距相对交叉验证标准差划分：≤1 倍为相当，1–3 倍为较弱，>3 倍为明显较弱。逐折结果见附录 A。",
                f"* Final model. {metric}: {n.better()}. CV = cross-validated score on validation folds; Train {metric} is the mean score on the training part of each fold, and a much better train score than CV suggests overfitting. Assessment compares the gap to the final model with the CV SD: ≤1× comparable, 1–3× weaker, >3× clearly weaker. Fold-level results in Appendix A."))
    figures("comparison")

    doc.heading(T("2.2 测试集表现", "2.2 Test performance"), 2)
    test_text = n.test_results()
    test_text[0] = test_text[0].rstrip("。.") + ref("test") + T("。", ".")
    for text in test_text:
        doc.para(text)
    test_rows = []
    order = list((s.get("test_metrics") or {}).keys()) + [k for k in facts["tests"] if k not in (s.get("test_metrics") or {})]
    for key in order:
        row = facts["tests"].get(key)
        if not row:
            continue
        gain = row.get("gain")
        gci = interval(row.get("gain_lower"), row.get("gain_upper")) or ""
        test_rows.append([n.mlabel(key) + (" *" if key == n.metric else ""), fmt(row.get("estimate")),
                          interval(row.get("lower"), row.get("upper")) or "NA",
                          fmt(row.get("baseline")), (fmt(gain) + (f" {gci}" if gci else "")) if gain is not None else "—"])
    if basic:
        test_rows = [row[:2] for row in test_rows if row[0].rstrip(" *") != n.mlabel("bias")]
        doc.table(T("最终模型在测试集上的表现", "Test-set performance of the final model"),
                  [T("指标", "Metric"), T("数值", "Value")], test_rows,
                  T("* 主要指标。", "* Primary metric."))
    else:
        doc.table(T("最终模型在测试集上的表现", "Test-set performance of the selected model"),
                  [T("指标", "Metric"), T("最终模型", "Final model"), T("95% CI", "95% CI"), T("基线模型", "Baseline"), T("相对基线的提升（95% CI）", "Improvement over baseline (95% CI)")],
                  test_rows, T(f"* 主要指标。基线模型：{n.baseline_rule()}。“相对基线的提升”为模型优于基线的幅度（误差类指标为基线减模型，得分类指标为模型减基线），正值表示模型更好；95% CI 由测试集 bootstrap 得到，不包含 0 说明提升稳定。NA/— 表示无法计算。",
                               f"* Primary metric. Baseline: {n.baseline_rule()}. Improvement = how much the model beats the baseline (baseline − model for errors, model − baseline for scores); positive favours the model, and a 95% CI (test-set bootstrap) excluding 0 indicates a reliable improvement. NA/— = not computable."))
    figures("test")

    error_text = n.error_sentence()
    calibration = n.calibration_sentence()
    if error_text or by_family.keys() & set(SECTION_FIGURES["errors"]):
        doc.heading(T("2.3 误差结构", "2.3 Error structure") if s["task"] == "regression" else T("2.3 分类错误与判别", "2.3 Classification errors and discrimination"), 2)
        if error_text:
            doc.para(error_text.rstrip("。.") + ref("errors") + T("。", "."))
        if calibration:
            doc.para(calibration)
        figures("errors")

    predictors = n.predictor_sentence()
    if predictors:
        doc.heading(T("2.4 特征重要性", "2.4 Feature importance"), 2)
        doc.para(predictors.rstrip("。.") + ref("predictors") + T("。", "."))
        ranked = (facts["importance"] or {}).get("ranked") or []
        if len(ranked) > 1:
            doc.table(T("测试集置换重要性", "Test-set permutation importance"),
                      [T("排名", "Rank"), T("特征", "Feature"), T("置换重要性（均值 ± SD）", "Permutation importance (mean ± SD)")],
                      [[str(i), n.feature(r["feature"]), f"{fmt(r['mean'])} ± {fmt(r['sd'])}"]
                       for i, r in enumerate(ranked[:15], 1)],
                      T(f"置换重要性 = {n.importance_phrase()}，重复 {s.get('importance_repeats')} 次。数值越大表示模型越依赖该特征，≤0 表示打乱后性能不受影响。{'仅列出前 15 项；' if len(ranked) > 15 else ''}完整结果见 importance.csv。",
                        f"Permutation importance = {n.importance_phrase()}, {s.get('importance_repeats')} repeats. Larger means stronger reliance; ≤0 means shuffling did not hurt. {'Top 15 shown; ' if len(ranked) > 15 else ''}Full results in importance.csv."))
        figures("predictors")

    shift = facts.get("shift") or {}
    if not basic and (by_family.get("feature-shift") or shift):
        doc.heading(T("2.5 训练集与测试集的可比性", "2.5 Comparability of training and test data"), 2)
        flags = [x for x in n.data_flags() if "分层" not in x and "Across test" not in x]
        if flags:
            doc.para(" ".join(flags).rstrip("。.") + ref("shift") + T("。", "."))
        elif shift.get("max_shift") is not None:
            doc.para(T(f"训练集与测试集在各特征上的均值差异均小于 {fmt(0.5)} 个训练集标准差（最大 {fmt(shift['max_shift'], 2)}），缺失率差异均小于 10 个百分点，未见明显的分布差异{ref('shift')}。",
                       f"All feature means differed by less than 0.5 training SD between partitions (maximum {fmt(shift['max_shift'], 2)}) and missingness by less than 10 percentage points{ref('shift')}."))
        figures("shift")

    if by_family.get("strata-performance"):
        doc.heading(T("2.6 分层稳定性", "2.6 Stability across strata"), 2)
        strata = facts.get("strata") or {}
        if strata:
            unit = T("组", "group") if s["split"] == "group" else T("时间块", "time block")
            worst = strata["worst"] if s["split"] == "group" else T(f"第 {strata['worst']} 个时间块", f"time block {strata['worst']}")
            doc.para(T(f"测试集按{unit}分为 {strata['n_strata']} 层，{n.mlabel(strata['measure'])}在 {fmt(strata['min'])} 到 {fmt(strata['max'])} 之间；表现最差的是{worst}（{fmt(strata['worst_value'])}，n = {strata['worst_n']}）{ref('strata')}。层间差异为描述性结果，可能受样本量和构成影响。",
                       f"The test set split into {strata['n_strata']} {unit}s; {n.mlabel(strata['measure'])} ranged from {fmt(strata['min'])} to {fmt(strata['max'])}, worst for {worst} ({fmt(strata['worst_value'])}, n = {strata['worst_n']}){ref('strata')}. Differences are descriptive and may reflect size and composition."))
        figures("strata")

    # Discussion --------------------------------------------------------------
    doc.heading(T("3 讨论", "3 Discussion"))
    for text in n.discussion():
        doc.para(text)
    for warning in s.get("warnings") or []:
        if "every training class" in warning:
            doc.para(T("注意：测试集未包含训练集中的全部类别，部分指标与曲线无法计算。", "Note: the test set lacks some training classes, so some metrics and curves are undefined."))
    steps = n.next_steps()
    if steps:
        doc.heading(T("建议的后续工作", "Suggested next steps"), 2)
        doc.bullets(steps)

    # Appendices --------------------------------------------------------------
    doc.heading(T("附录 A 各候选模型的详细结果", "Appendix A Detailed results by model"), page_break=True)
    split_models = len(facts["comparison"]) >= PER_MODEL_SPLIT_THRESHOLD
    fold_headers = [T("折", "Fold"), T("训练 n", "Train n"), T("验证 n", "Validation n"), T("训练得分", "Train score"),
                    T("验证得分", "Validation score"), T("基线（验证折）", "Baseline (validation fold)")][:5 if basic else 6]

    def fold_rows(row):
        return [[str(r["fold"]), str(r["train_n"] or "NA"), str(r["validation_n"] or "NA"), fmt(r["train_score"]),
                 fmt(r["score"]), fmt(r["baseline"])][:5 if basic else 6] for r in row.get("fold_table") or []]

    def model_heading(row):
        mark = T("（最终模型）", " (final model)") if row["standing"] in ("selected", "sole") else ""
        return f"{n.model(row['model'])}{mark}"

    search_params = {}
    search_path = directory / "search-results.json"
    if search_path.is_file():
        trials = [r for r in json.loads(search_path.read_text(encoding="utf-8")) if r.get("status") == "completed"]
        for trial in trials:
            best = search_params.get(trial["model"])
            better = best is None or (trial["mean"] < best["mean"] if s["direction"] == "lower" else trial["mean"] > best["mean"])
            if better:
                search_params[trial["model"]] = trial

    def model_blocks(target_doc, row, level):
        target_doc.heading(model_heading(row), level)
        target_doc.para(n.model_detail(row))
        trial = search_params.get(row["model"])
        if trial and trial.get("parameters") and s.get("search", {}).get("method", "none") != "none":
            target_doc.para(T("最佳搜索配置：", "Best searched configuration: ") +
                            json.dumps(trial["parameters"], ensure_ascii=False))
        if row.get("fold_table"):
            target_doc.table(T(f"{n.model(row['model'])}的逐折结果（{metric}）", f"Fold results for {n.model(row['model'])} ({metric})"),
                             fold_headers, fold_rows(row))

    if split_models:
        reports_dir = directory / "reports"
        reports_dir.mkdir(exist_ok=True)
        doc.para(T(f"共有 {len(facts['comparison'])} 个模型完成训练，每个模型的详细结果单独成文，索引如下。",
                   f"{len(facts['comparison'])} models completed training; each has its own detailed report, indexed below."))
        index = []
        for row in facts["comparison"]:
            tag = row["model"]
            sub = ReportDocument(f"{n.model(tag)}：{T('模型详细结果', 'detailed results')}" if zh else f"{n.model(tag)}: detailed results", zh)
            sub.para(T(f"本文件是“{title}”的附属报告，比较设定见总报告方法部分。", f"Companion to '{title}'; see the overview report for methods."))
            model_blocks(sub, row, 1)
            (reports_dir / f"{tag}.md").write_text(sub.markdown(directory), encoding="utf-8")
            sub.docx(reports_dir / f"{tag}.docx", cjk_font_name())
            index.append([model_heading(row), f"{fmt(row['mean'])} ± {fmt(row.get('sd'))}",
                          STANDING[row["standing"]][0 if zh else 1], f"[{tag}.md](reports/{tag}.md)"])
        doc.table(T("各模型详细报告索引", "Index of per-model reports"),
                  [T("模型", "Model"), T(f"CV {metric}（均值 ± SD）", f"CV {metric} (mean ± SD)"), T("结论", "Assessment"), T("文件", "File")], index)
    else:
        for row in facts["comparison"]:
            model_blocks(doc, row, 2)
    if s.get("failures"):
        doc.para(T("未完成训练的模型：", "Models that failed: ") + json.dumps(s["failures"], ensure_ascii=False))

    doc.heading(T("附录 B 指标释义", "Appendix B Metric definitions"))
    for key in order:
        spec = METRICS.get({"f1": "f1_macro", "precision": "precision_macro", "recall": "recall_macro"}.get(key, key))
        if spec:
            doc.labelled(spec[0] if zh else spec[2], spec[1] if zh else spec[3])

    guides = [(label, FIGURES[family]) for label, family in used if family in FIGURES]
    if guides:
        doc.heading(T("附录 C 图表阅读指南", "Appendix C How to read the figures"))
    for label, guide in guides:
        if guide:
            doc.labelled(f"{label} {guide[0]}" if zh else f"{label} {guide[3]}", (guide[1] + guide[2]) if zh else (guide[4] + " " + guide[5]))

    doc.heading(T(f"附录 {'D' if guides else 'C'} 复现信息", f"Appendix {'D' if guides else 'C'} Reproducibility"))
    versions = "; ".join(f"{k} {v}" for k, v in (s.get("versions") or {}).items() if v)
    doc.para(T(f"随机种子 {s['seed']}。完整配置、输入文件 SHA256 与软件版本记录在 summary.json；样本划分记录在 split-membership.csv 与 cv-membership.csv；pipeline.joblib 保存最终的预处理与模型，可直接用于预测。图的矢量版本（SVG/PDF）与源数据随报告保存在 figures/。",
               f"Random seed {s['seed']}. summary.json records configuration, input SHA256 hashes and software versions; split-membership.csv and cv-membership.csv list the splits; pipeline.joblib holds the fitted preprocessing and model for prediction. Vector figures (SVG/PDF) and source data are in figures/."))
    if versions:
        doc.para(T("软件版本：", "Software: ") + versions)
    if facts["context"]:
        doc.para(T("报告上下文（变量含义、单位等）由研究者提供，保存在 report-context.json。",
                   "Report context (variable meanings, units) was supplied by the researcher and saved in report-context.json."))

    doc.docx(directory / "report.docx", cjk_font_name())
    (directory / "report.md").write_text(doc.markdown(directory), encoding="utf-8")
