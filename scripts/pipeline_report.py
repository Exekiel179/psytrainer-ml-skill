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
                ("参数搜索：每点为一个候选参数配置的平均验证得分，候选编号对应源文件。搜索使用开发集，分数用于选择而非无偏性能估计；失败数单独列出。" if zh else
                 "Parameter search: each point is mean development CV score for a candidate indexed in the source file. Scores are for selection, not unbiased performance estimates; failed candidates are counted separately."),
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
        (f"模型选择：{folds} 折验证得分。灰点为各折，蓝点和误差线为均值及折间样本标准差（非置信区间）。各模型使用相同划分。"
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
    "test": ("test-performance",),
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


def generate_report(directory, context=None, language=None):
    from report_document import ReportDocument
    from report_explanations import (FIGURES, METRICS, PER_MODEL_SPLIT_THRESHOLD, preprocessing,
                                     uncertainty_description, validation_methods)
    from report_facts import collect
    from report_narrative import STANDING, Narrator, fmt, interval, pct

    directory = Path(directory)
    save_context(directory, context)
    s = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if language:
        s["language"] = language
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
                guide = FIGURES.get(family)
                label = doc.figure(directory / "figures" / f"{figure['name']}.png",
                                   (guide[0 if zh else 3] + T("。", ". ") if guide else "") + figure["caption"],
                                   figure["source"])
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
        doc.para(T(f"参数通过{'网格' if search['method'] == 'grid' else '随机'}搜索在开发集内选择，共评估 {search['evaluated_candidates']} 个配置（失败 {search['failed_candidates']} 个，每个模型上限 {search.get('max_candidates')} 个）；各配置的验证结果见 search-results.json。",
                   f"Parameters were chosen by {search['method']} search on development data: {search['evaluated_candidates']} configurations evaluated ({search['failed_candidates']} failed; budget {search.get('max_candidates')} per model); see search-results.json."))
    else:
        doc.para(T("各模型使用默认或显式指定的固定参数，未进行参数搜索。", "Models used default or explicitly configured fixed parameters; no parameter search was run."))
    if s["preprocessing"].get("selection") == "forward":
        doc.para(T("前向特征筛选只使用当前训练分区，其内部交叉验证沿用相同的划分设计。",
                   "Forward selection used only the current training partition, with inner cross-validation following the same design."))
    doc.heading(T("1.5 评价与统计方法", "1.5 Evaluation and statistics"), 2)
    for text in n.evaluation_methods(uncertainty_description(facts["interval"], zh) if facts["interval"] else []):
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
    headers = [T("模型", "Model"), T("类型", "Family"), T(f"验证{metric}（均值 ± SD）", f"Validation {metric} (mean ± SD)"),
               T("折范围", "Fold range"), T("优于基线的折", "Folds beating baseline"),
               T("训练−验证差", "Train−validation gap"), T("判断", "Assessment")]
    rows = []
    for row in facts["comparison"]:
        rows.append([n.model(row["model"]) + (" *" if row["standing"] in ("selected", "sole") else ""), n.family(row["family"]),
                     f"{fmt(row['mean'])} ± {fmt(row.get('sd'))}",
                     f"{fmt(row.get('min'))}–{fmt(row.get('max'))}" if row.get("fold_scores") else "NA",
                     f"{row['baseline_gain_positive_folds']}/{row['folds']}" if row.get("baseline_gain_positive_folds") is not None else "NA",
                     fmt(row.get("train_gap")), STANDING[row["standing"]][0 if zh else 1]])
    doc.table(T("候选模型的交叉验证表现", "Cross-validated performance of candidate models"), headers, rows,
              T(f"* 选定模型。{metric}{n.better()}；训练−验证差按正值表示训练表现更好计算。“判断”依据与选定模型的差距相对折间标准差的大小：≤1 为接近，1–3 为较弱，>3 为明显较弱。各模型的逐折结果见附录 A。",
                f"* Selected model. {metric}: {n.better()}; positive train−validation gaps favour training. Assessment uses the gap to the selected model in between-fold SD units: ≤1 close, 1–3 weaker, >3 clearly weaker. Fold-level results are in Appendix A."))
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
                          fmt(row.get("baseline")), (fmt(gain) + (f" {gci}" if gci else "")) if gain is not None else "NA"])
    doc.table(T("选定模型在测试集上的表现", "Test-set performance of the selected model"),
              [T("指标", "Metric"), T("模型", "Model"), T("95% CI", "95% CI"), T("基线", "Baseline"), T("改善（95% CI）", "Improvement (95% CI)")],
              test_rows, T("* 主要指标。改善按优劣方向计算，正值表示模型优于基线；NA 表示该设计或数据下无法计算。区间为固定模型下测试样本的 bootstrap 百分位区间。",
                           "* Primary metric. Improvement is oriented so positive favours the model; NA means not computable for this design or data. Intervals are bootstrap percentile intervals over test units for the fitted model."))
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
        doc.heading(T("2.4 预测变量的作用", "2.4 Predictor reliance"), 2)
        doc.para(predictors.rstrip("。.") + ref("predictors") + T("。", "."))
        ranked = (facts["importance"] or {}).get("ranked") or []
        if len(ranked) > 1:
            doc.table(T("测试集置换重要性", "Test-set permutation importance"),
                      [T("排名", "Rank"), T("特征", "Feature"), T("平均下降", "Mean decrease"), T("重复间 SD", "Repeat SD"), T("占正向重要性", "Share of positive")],
                      [[str(i), n.feature(r["feature"]), fmt(r["mean"]), fmt(r["sd"]), pct(r["share"]) if r["share"] else "—"]
                       for i, r in enumerate(ranked[:15], 1)],
                      T(f"平均下降为置换该特征后主要指标的变化（已统一为越大表示依赖越强）。{'仅列出前 15 项；' if len(ranked) > 15 else ''}完整结果见 importance.csv。",
                        f"Mean decrease is the change in the primary metric after permuting the feature (oriented so larger means more reliance). {'Top 15 shown; ' if len(ranked) > 15 else ''}Full results in importance.csv."))
        figures("predictors")

    shift = facts.get("shift") or {}
    if by_family.get("feature-shift") or shift:
        doc.heading(T("2.5 开发集与测试集的可比性", "2.5 Comparability of development and test data"), 2)
        flags = [x for x in n.data_flags() if "分层" not in x and "Across test" not in x]
        if flags:
            doc.para(" ".join(flags).rstrip("。.") + ref("shift") + T("。", "."))
        elif shift.get("max_shift") is not None:
            doc.para(T(f"开发集与测试集在各特征上的均值差异均小于 {fmt(0.5)} 个开发集标准差（最大 {fmt(shift['max_shift'], 2)}），缺失率差异均小于 10 个百分点，未见明显的分布差异{ref('shift')}。",
                       f"All feature means differed by less than 0.5 development SD between partitions (maximum {fmt(shift['max_shift'], 2)}) and missingness by less than 10 percentage points{ref('shift')}."))
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
        if "every development class" in warning:
            doc.para(T("注意：测试集未包含开发集中的全部类别，部分指标与曲线无法计算。", "Note: the test set lacks some development classes, so some metrics and curves are undefined."))
    steps = n.next_steps()
    if steps:
        doc.heading(T("建议的后续工作", "Suggested next steps"), 2)
        doc.bullets(steps)

    # Appendices --------------------------------------------------------------
    doc.heading(T("附录 A 各候选模型的详细结果", "Appendix A Detailed results by model"), page_break=True)
    split_models = len(facts["comparison"]) >= PER_MODEL_SPLIT_THRESHOLD
    fold_headers = [T("折", "Fold"), T("训练 n", "Train n"), T("验证 n", "Validation n"), T("训练得分", "Training"),
                    T("验证得分", "Validation"), T("同折基线", "Same-fold baseline")]

    def fold_rows(row):
        return [[str(r["fold"]), str(r["train_n"] or "NA"), str(r["validation_n"] or "NA"), fmt(r["train_score"]),
                 fmt(r["score"]), fmt(r["baseline"])] for r in row.get("fold_table") or []]

    def model_heading(row):
        mark = T("（选定）", " (selected)") if row["standing"] in ("selected", "sole") else ""
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
                  [T("模型", "Model"), T(f"验证{metric}", f"Validation {metric}"), T("判断", "Assessment"), T("文件", "File")], index)
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
