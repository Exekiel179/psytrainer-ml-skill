"""Evidence-oriented diagnostic panels drawn entirely from exported source data."""

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

BLUE, ORANGE, GRAY, GREEN = "#0072B2", "#D55E00", "#656565", "#009E73"


def diagnostic_plots(directory, summary, add):
    zh = summary["language"] == "zh"
    n = summary["test_n"]
    pred = pd.read_csv(directory / "test-predictions.csv", dtype={"sample_id": str, "observed": str, "predicted": str}, keep_default_na=False)
    cv = pd.read_csv(directory / "cv-scores.csv")
    base_cv = pd.read_csv(directory / "baseline-cv.csv").set_index("fold")
    metrics = pd.read_csv(directory / "metric-intervals.csv").set_index("metric")

    def panels(rows=1, cols=2, height=3.5):
        fig, axes = plt.subplots(rows, cols, figsize=(7.09, height), squeeze=False, layout="constrained")
        return fig, axes.ravel()

    def title(ax, letter, value):
        ax.set_title(f"{letter}  {value}", loc="left", fontsize=10, fontweight="bold", pad=10)

    def dots(ax, xx, yy, color=BLUE):
        if len(xx) > 1500:
            ax.hexbin(xx, yy, gridsize=35, mincnt=1, cmap="Blues", rasterized=True)
        else:
            ax.scatter(xx, yy, color=color, s=13, alpha=.6, linewidths=0, rasterized=True)

    names = [r["model"] for r in summary["comparison"]]
    fig, (ax, bx) = panels(height=max(3.5, .45 * len(names) + 1.8))
    direction = 1 if summary["direction"] == "higher" else -1
    for i, name in enumerate(names):
        rows = cv[cv.model == name].sort_values("fold")
        for _, row in rows.iterrows():
            ax.plot([row.train_score, row.score], [i - .12, i + .12], color=GRAY, alpha=.3, linewidth=.7)
        ax.scatter(rows.train_score, np.full(len(rows), i - .12), color=GRAY, s=16, label="Training" if i == 0 else None)
        ax.scatter(rows.score, np.full(len(rows), i + .12), color=BLUE, s=20, label="Validation" if i == 0 else None)
        gains = direction * (rows.score.to_numpy() - base_cv.loc[rows.fold, "score"].to_numpy())
        if np.isfinite(gains).all():
            bx.scatter(gains, np.full(len(gains), i) + np.linspace(-.1, .1, len(gains)), color=GRAY, s=15)
            bx.errorbar(gains.mean(), i, xerr=gains.std(ddof=1), fmt="o", color=GREEN, capsize=3)
        else:
            bx.text(.5, i, "Undefined for constant baseline", transform=bx.get_yaxis_transform(),
                    ha="center", fontsize=8)
    labels = [textwrap.fill(name.removeprefix("Model"), 19) for name in names]
    for axis in (ax, bx):
        axis.set_yticks(range(len(names)), labels if axis is ax else [])
        axis.set_ylim(len(names) - .5, -.5)
    ax.set_xlabel(f"{summary['metric']} ({summary['direction']} is better)")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.22), ncol=2)
    bx.axvline(0, color=GRAY, ls="--", lw=.8)
    bx.set_xlabel("Validation gain over dummy baseline")
    title(ax, "a", "Training versus validation")
    title(bx, "b", "Paired baseline comparison")
    add(fig, "model-validation", (f"模型验证（{summary['cv_folds']} 折，共享划分）。a，每折训练与验证得分的连线用于检查泛化差距。b，同一验证折内相对简单基线的有向增益，正值更好；点为折，绿色为均值±样本标准差，非置信区间。基线在对应训练折拟合，不参与候选模型选择。" if zh else
        f"Model validation ({summary['cv_folds']} shared folds). a, paired training/validation scores. b, oriented within-fold gain over a training-fold-fitted dummy baseline; positive is better. Gray: folds; green: mean and sample SD, not CI. The dummy does not participate in candidate selection."), "cv-scores.csv; baseline-cv.csv")

    primary = {"f1": "f1_macro", "precision": "precision_macro", "recall": "recall_macro"}.get(summary["metric"], summary["metric"])
    row = metrics.loc[primary]
    fig, ax = plt.subplots(figsize=(7.09, 2.7), layout="constrained")
    ax.scatter([row.estimate, row.baseline], [1, 0], c=[BLUE, GRAY], s=35)
    if pd.notna(row.lower):
        ax.hlines(1, row.lower, row.upper, color=BLUE, linewidth=1.4)
        ax.plot([row.lower, row.upper], [1, 1], "|", color=BLUE)
    ax.set_yticks([0, 1], ["Dummy baseline", "Selected model"])
    ax.set_ylim(-.6, 1.6)
    ax.set_xlabel(f"Held-out {primary} ({summary['direction']} is better)")
    title(ax, "a", "Held-out performance and uncertainty")
    add(fig, "test-performance", (f"独立测试主要指标（n={n}）。蓝色为所选模型，灰色为开发集拟合的基线。若可计算，线段为模型指标 95% 百分位 bootstrap 区间；随机设计按行、分组设计按整组重采样，时间设计不绘制区间。基线仅显示点估计；配对增益及其区间见源表。区间不覆盖训练和模型选择不确定性。" if zh else
        f"Held-out primary metric (n={n}). Blue: selected model; gray: development-fitted baseline point estimate. Where defined, model bars are 95% percentile bootstrap intervals (rows for random, whole groups for group designs; omitted for time). Paired gain intervals are in the source table. Training/selection uncertainty is excluded."), "metric-intervals.csv; analysis.json")

    if summary["task"] == "regression":
        observed, predicted = pd.to_numeric(pred.observed), pd.to_numeric(pred.predicted)
        residual = observed - predicted
        bins = pd.read_csv(directory / "regression-bins.csv")
        fig, axes = panels(2, 2, 6.4)
        ax, bx, cx, dx = axes
        dots(ax, observed, predicted)
        low, high = min(observed.min(), predicted.min()), max(observed.max(), predicted.max())
        pad = max((high - low) * .05, .1)
        ax.plot([low - pad, high + pad], [low - pad, high + pad], color=GRAY, ls="--", lw=.8)
        ax.set(xlabel="Observed outcome", ylabel="Predicted outcome", xlim=(low-pad, high+pad), ylim=(low-pad, high+pad))
        dots(bx, predicted, residual, ORANGE)
        bx.axhline(0, color=GRAY, ls="--", lw=.8)
        bx.plot(bins.predicted_mean, bins.bias, color=BLUE, marker="o", ms=3, lw=1)
        bx.fill_between(bins.predicted_mean, bins.residual_q10, bins.residual_q90, color=BLUE, alpha=.12)
        bx.set(xlabel="Predicted outcome", ylabel="Residual (observed - predicted)")
        absolute = np.sort(abs(residual))
        cx.step(absolute, np.arange(1, n + 1) / n, where="post", color=BLUE, lw=1.4)
        cx.set(xlabel="Absolute prediction error", ylabel="Fraction at or below error", ylim=(0, 1.02))
        dx.plot(bins.predicted_mean, bins.mae, "o-", color=ORANGE, ms=4, lw=1)
        dx.set(xlabel="Mean predicted outcome in bin", ylabel="Mean absolute error", ylim=(0, None))
        for axis, letter, label in zip(axes, "abcd", ["Prediction agreement", "Bias and residual spread", "Error distribution", "Error across prediction levels"]):
            title(axis, letter, label)
        add(fig, "regression-diagnostics", (f"独立测试误差结构（n={n}）。a，真实值与预测值及等值参考线。b，残差散点、分箱均值及箱内 10–90 分位带。c，绝对误差的经验累积分布。d，预测水平分箱后的 MAE。最多 5 个等频箱，重复边界合并，箱内人数见源表；分箱曲线为描述性摘要，不是拟合模型或预测区间。所有样本保留；超过 1500 点时散点改为六边形计数。" if zh else
            f"Held-out error structure (n={n}). a, agreement with identity reference. b, residuals with binned means and within-bin 10th–90th percentiles. c, empirical absolute-error CDF. d, MAE by prediction level. Up to five quantile bins, duplicate edges merged; counts in source data. Bin curves are descriptive, not prediction intervals. All samples retained; above 1500 points, hexbins replace scatter."), "test-predictions.csv; regression-bins.csv; error-cases.csv")
    else:
        classes = summary["classes"]
        matrix = confusion_matrix(pred.observed, pred.predicted, labels=classes)
        support = matrix.sum(axis=1)
        rates = np.divide(matrix, support[:, None], out=np.full(matrix.shape, np.nan, dtype=float), where=support[:, None] != 0)
        pd.DataFrame(matrix, index=classes, columns=classes).to_csv(directory / "confusion-matrix.csv", index_label="observed")
        fig, (ax, bx) = panels(height=max(3.8, .24 * len(classes) + 1.8))
        ax.imshow(rates, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        if len(classes) <= 8:
            for (i, j), value in np.ndenumerate(matrix):
                ax.text(j, i, f"{value}\n{rates[i, j]:.0%}" if support[i] else "NA", ha="center", va="center", fontsize=7,
                        color="white" if rates[i, j] > .5 else "black")
        keys = [f"C{i}" for i in range(len(classes))]
        ax.set_xticks(range(len(classes)), keys, rotation=90 if len(classes) > 12 else 0)
        ax.set_yticks(range(len(classes)), keys)
        ax.set(xlabel="Predicted class", ylabel="Observed class")
        table = pd.read_csv(directory / "class-metrics.csv")
        for column, offset, color in (("precision", -.15, GRAY), ("recall", 0, BLUE), ("f1", .15, GREEN)):
            bx.scatter(table[column], np.arange(len(classes)) + offset, c=color, s=20, label=column.capitalize())
        bx.set_yticks(range(len(classes)), [f"{key} (n={count})" for key, count in zip(keys, support)])
        bx.set(xlabel="Class metric", xlim=(-.03, 1.03), ylim=(len(classes)-.5, -.5))
        bx.legend(loc="upper center", bbox_to_anchor=(.5, -.22), ncol=3, fontsize=7)
        title(ax, "a", "Where predictions fail")
        title(bx, "b", "Performance by class")
        mapping = "; ".join(f"{key}={value}" for key, value in zip(keys, classes))
        add(fig, "classification-errors", (f"独立测试类别误差（n={n}）。a，颜色为按真实类别行归一化的比例，单元格为计数和行百分比；未出现的类别为 NA，超过 8 类省略格内文字。b，各类 precision、recall 和 F1 点估计及真实样本量，无区间。类别映射：{mapping}。" if zh else
            f"Held-out class errors (n={n}). a, row-normalized proportions; labels show counts/row percentages (omitted above eight classes). Absent rows: NA. b, precision, recall and F1 point estimates with observed support, without intervals. Class key: {mapping}."), "confusion-matrix.csv; class-metrics.csv")
        if len(classes) == 2 and "positive_score" in pred and pred.observed.nunique() == 2:
            binary = pred.observed == classes[1]
            fpr, tpr, _ = roc_curve(binary, pred.positive_score)
            precision, recall, _ = precision_recall_curve(binary, pred.positive_score)
            pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(directory / "roc.csv", index=False)
            pd.DataFrame({"recall": recall, "precision": precision}).to_csv(directory / "precision-recall.csv", index=False)
            fig, (ax, bx) = panels()
            ax.plot(fpr, tpr, color=BLUE, lw=1.4)
            ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=.8)
            bx.step(recall, precision, where="post", color=BLUE, lw=1.4)
            bx.axhline(binary.mean(), color=GRAY, ls="--", lw=.8)
            ax.set(xlabel="False positive rate", ylabel="True positive rate")
            bx.set(xlabel="Recall", ylabel="Precision")
            for axis in (ax, bx):
                axis.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02))
            title(ax, "a", f"Ranking: AUC={metrics.loc['roc_auc', 'estimate']:.3f}")
            title(bx, "b", f"Retrieval: AP={metrics.loc['average_precision', 'estimate']:.3f}")
            add(fig, "classification-discrimination", (f"独立测试判别能力（n={n}，阳性={classes[1]}，阳性比例={binary.mean():.3f}）。ROC 与 PR 分别展示排序和阳性检出表现；虚线为随机参考。标题为点估计，指标区间见源表，曲线不绘制置信带。" if zh else
                f"Held-out discrimination (n={n}; positive={classes[1]}, prevalence={binary.mean():.3f}). ROC and PR show ranking and positive retrieval; dashed lines indicate chance. Titles are point estimates; scalar intervals are in source data, without curve confidence bands."), "roc.csv; precision-recall.csv; metric-intervals.csv")
        if "positive_probability" in pred:
            calibration = pd.read_csv(directory / "calibration.csv")
            fig, (ax, bx) = panels()
            ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=.8)
            ax.plot(calibration.mean_probability, calibration.observed_fraction, "o-", color=BLUE, ms=4, lw=1.2)
            ax.set(xlabel="Mean predicted probability", ylabel="Observed positive fraction", xlim=(-.02, 1.02), ylim=(-.02, 1.02))
            bx.bar(calibration.left + .05, calibration.n, width=.085, color=BLUE, alpha=.8)
            bx.set(xlabel="Predicted probability bin", ylabel="Samples per bin", xlim=(0, 1))
            title(ax, "a", "Probability reliability")
            title(bx, "b", "Evidence supporting each bin")
            add(fig, "probability-calibration", (f"概率校准与样本支持（n={n}）。a，10 个等宽概率箱的平均预测概率与实际阳性率，虚线为理想校准；连线仅作视觉引导，无置信区间。b，每箱样本数。空箱省略，小样本箱不能支持稳定的校准结论。" if zh else
                f"Probability calibration and support (n={n}). a, mean probability versus positive fraction in ten equal-width bins, identity reference; connecting lines guide the eye, without confidence intervals. b, bin support. Empty bins omitted; sparse bins do not establish stable calibration."), "calibration.csv; metric-intervals.csv")
        if len(classes) == 2 and "positive_score" in pred:
            thresholds = pd.read_csv(directory / "thresholds.csv")
            fig, ax = plt.subplots(figsize=(7.09, 3.5), layout="constrained")
            for key, color in (("sensitivity", BLUE), ("specificity", GREEN), ("positive_fraction", GRAY)):
                ax.plot(thresholds.threshold, thresholds[key], label=key.replace("_", " ").capitalize(), color=color, lw=1.3)
            ax.axvline(.5 if "positive_probability" in pred else 0, color=ORANGE, ls="--", lw=.8)
            ax.set(xlabel="Probability threshold" if "positive_probability" in pred else "Decision score threshold", ylabel="Fraction", ylim=(-.02, 1.02))
            ax.legend(loc="upper center", bbox_to_anchor=(.5, -.22), ncol=3)
            title(ax, "a", "Threshold trade-offs on held-out data")
            add(fig, "threshold-tradeoffs", ("阈值敏感性为测试集描述性分析；虚线为常规参考阈值（概率 0.5 或决策分数 0），不一定等于厂商 predict 的规则。阳性预测比例反映待处理样本量。未自动选取最优阈值、未更改模型；任何阈值调整必须在开发集及实际误判成本下确定，并用新的独立数据验证。" if zh else
                "Descriptive test threshold sensitivity; dashed line is the conventional reference (probability 0.5 or decision score 0), not necessarily the vendor predict rule. Positive fraction describes workload. No optimal threshold is selected and the model is unchanged. Determine changes using development data and error costs, then validate on new independent data."), "thresholds.csv")

    if (directory / "importance.csv").exists():
        imp = pd.read_csv(directory / "importance.csv", dtype={"feature": str}, keep_default_na=False).head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7.09, max(3.5, .28 * len(imp) + 1.5)), layout="constrained")
        repeats = pd.read_csv(directory / "importance-repeats.csv", dtype={"feature": str}, keep_default_na=False).set_index("feature")
        for i, feature in enumerate(imp.feature):
            values = repeats.loc[feature].to_numpy(dtype=float)
            ax.scatter(values, np.full(len(values), i) + np.linspace(-.1, .1, len(values)), color=GRAY, alpha=.45, s=10)
        ax.errorbar(imp["mean"], np.arange(len(imp)), xerr=imp.sd, fmt="o", color=BLUE, capsize=3)
        ax.set_yticks(range(len(imp)), [textwrap.fill(f, 24) for f in imp.feature])
        ax.axvline(0, color=GRAY, ls="--", lw=.8)
        ax.set_xlabel(f"Decrease in {summary['metric']} scorer (greater = more reliance)")
        title(ax, "a", "Feature reliance and repeat variability")
        add(fig, "permutation-importance", (f"测试集置换重要性，展示前 {len(imp)}/{summary['feature_n']} 个特征、全部 {summary['importance_repeats']} 次重复。灰点为重复，蓝点和线为均值±置换总体标准差，非置信区间；评分统一为越大越好。负值代表置换后得分提高，不能直接据此删除特征。相关性及分组/时间依赖会影响解释，不代表因果。" if zh else
            f"Test permutation importance: top {len(imp)}/{summary['feature_n']} features, all {summary['importance_repeats']} repeats. Gray: repeats; blue: mean and population SD, not CI. Scorers oriented higher-is-better. Negative values mean improved scores after permutation, not grounds for test-driven feature removal. Correlation/group/time dependence limits interpretation; not causal."), "importance.csv; importance-repeats.csv")
        if (directory / "feature-correlations.csv").exists() and len(imp) > 1:
            corr = pd.read_csv(directory / "feature-correlations.csv", dtype={"feature": str}).set_index("feature")
            features = list(imp.feature.iloc[::-1])
            selected_corr = corr.loc[features, features]
            fig, ax = plt.subplots(figsize=(7.09, 4.5), layout="constrained")
            mesh = ax.imshow(selected_corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
            keys = [f"F{i+1}" for i in range(len(features))]
            ax.set_xticks(range(len(features)), keys)
            ax.set_yticks(range(len(features)), keys)
            ax.set(xlabel="Input feature", ylabel="Input feature")
            fig.colorbar(mesh, ax=ax, label="Development Spearman correlation", shrink=.85)
            title(ax, "a", "Correlated predictors can share reliance")
            mapping = "; ".join(f"{key}={feature}" for key, feature in zip(keys, features))
            add(fig, "feature-correlation", (f"开发集原始特征的 Spearman 相关，使用成对非缺失样本；常量或样本不足时为空。展示测试重要性前 {len(features)} 个特征，完整矩阵另存。不以此筛选模型。相关特征可能相互替代，因而单列置换重要性偏低；相关性不是因果证据。映射：{mapping}。" if zh else
                f"Development raw-feature Spearman correlations using pairwise complete observations; constant/insufficient pairs are blank. Top {len(features)} test-reliance features displayed; full matrix retained. No model selection uses this view. Correlated features can substitute, masking single-feature reliance; not causal evidence. Key: {mapping}."), "feature-correlations.csv; importance.csv")

    shift = pd.read_csv(directory / "feature-shift.csv", dtype={"feature": str})
    shift["magnitude"] = shift.standardized_mean_shift.abs()
    shown = shift.sort_values("magnitude", ascending=False, na_position="last").head(12).iloc[::-1]
    fig, (ax, bx) = panels(height=max(3.8, .28 * len(shown) + 1.6))
    yy = np.arange(len(shown))
    ax.scatter(shown.standardized_mean_shift, yy, color=ORANGE, s=22)
    ax.axvline(0, color=GRAY, ls="--", lw=.8)
    ax.set_yticks(yy, [textwrap.fill(f, 20) for f in shown.feature])
    ax.set_xlabel("(Test mean - development mean) / SD")
    bx.scatter(shown.development_missing, yy-.1, color=GRAY, label="Development", s=18)
    bx.scatter(shown.test_missing, yy+.1, color=BLUE, label="Test", s=22)
    bx.set_yticks(yy, [])
    bx.set(xlabel="Missing fraction", xlim=(-.03, 1.03))
    bx.legend(loc="upper center", bbox_to_anchor=(.5, -.22), ncol=2, fontsize=7)
    for axis in (ax, bx):
        axis.set_ylim(-.5, len(shown)-.5)
    title(ax, "a", "Feature distribution shift")
    title(bx, "b", "Missingness shift")
    add(fig, "feature-shift", (f"开发集与测试集输入差异，按绝对标准化均值差展示前 {len(shown)}/{len(shift)} 个特征。标准差仅来自开发集；常量/全缺失特征的位移不定义，保留空点。缺失率使用全部样本。完整表还提供越出开发集取值范围的比例。该图描述输入差异，不证明漂移导致误差。" if zh else
        f"Development/test input differences: top {len(shown)}/{len(shift)} by absolute standardized mean shift. SD uses development only; constant/all-missing features have undefined shifts and blank marks. Missingness uses all samples. Source also records fraction outside development range. Descriptive differences do not establish causes of error."), "feature-shift.csv")
    if (directory / "strata-metrics.csv").exists():
        strata = pd.read_csv(directory / "strata-metrics.csv", dtype={"stratum": str})
        pages = int(np.ceil(len(strata) / 20))
        for page in range(pages):
            block = strata.iloc[page*20:(page+1)*20]
            fig, ax = plt.subplots(figsize=(7.09, max(3.5, .25 * len(block) + 1.6)), layout="constrained")
            ax.scatter(block.value, np.arange(len(block)), color=BLUE, s=25)
            ax.set_yticks(range(len(block)), [textwrap.fill(f"{r.stratum} (n={r.n})", 28) for r in block.itertuples()])
            ax.set_ylim(len(block)-.5, -.5)
            ax.set_xlabel(block.metric.iloc[0])
            title(ax, "a", "Held-out performance by " + ("time block" if summary["split"] == "time" else "group"))
            add(fig, f"strata-performance-{page+1}", (f"测试集{'时间块' if summary['split'] == 'time' else '组别'}表现（第 {page+1}/{pages} 页），各层点估计及样本量，无区间。时间块按唯一时间值连续分成最多 8 块，等时间值不拆分；边界与样本归属见源表。所有层均保留，组间差异不等同于因果效应。" if zh else
                f"Held-out {summary['split']} performance (page {page+1}/{pages}): point estimates and support, no intervals. Time values are split into up to eight consecutive blocks, preserving ties; boundaries and membership in source. All strata retained; differences are not causal."), "strata-metrics.csv; strata-membership.csv")
