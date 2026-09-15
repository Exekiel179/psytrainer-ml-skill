"""Auditable diagnostics computed from saved predictions, without refitting models."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error,
                             precision_recall_fscore_support, precision_score, recall_score, r2_score, roc_auc_score)
from pipeline_options import pearson_score


def feature_shift(development, test):
    records = []
    for feature in development:
        a, b = development[feature].dropna(), test[feature].dropna()
        scale = float(a.std(ddof=0)) if len(a) else 0.0
        shift = float((b.mean() - a.mean()) / scale) if scale > 0 and len(b) else None
        records.append({"feature": feature, "development_mean": a.mean(), "test_mean": b.mean(),
                        "standardized_mean_shift": shift,
                        "development_missing": float(development[feature].isna().mean()),
                        "test_missing": float(test[feature].isna().mean()),
                        "outside_development_range": float(((b < a.min()) | (b > a.max())).mean())
                        if len(a) and len(b) else None,
                        "undefined_scale": scale == 0})
    return pd.DataFrame(records)


def metric_values(frame, task, classes):
    y, p = frame.observed.to_numpy(), frame.predicted.to_numpy()
    if task == "regression":
        y, p = y.astype(float), p.astype(float)
        correlation = pearson_score(y, p)
        return {"mae": float(mean_absolute_error(y, p)),
                "mse": float(mean_squared_error(y, p)),
                "pearson_r": correlation if np.isfinite(correlation) else None,
                "rmse": float(np.sqrt(mean_squared_error(y, p))),
                "r2": float(r2_score(y, p)) if len(y) > 1 and np.ptp(y) > 0 else None,
                "bias": float(np.mean(y - p))}
    matrix = confusion_matrix(y, p, labels=classes)
    support = matrix.sum(axis=1)
    result = {"accuracy": float(accuracy_score(y, p)),
              "balanced_accuracy": float(np.mean(matrix.diagonal() / support)) if np.all(support) else None,
              "f1_macro": float(f1_score(y, p, labels=classes, average="macro", zero_division=0)),
              "precision_macro": float(precision_score(y, p, labels=classes, average="macro", zero_division=0)),
              "recall_macro": float(recall_score(y, p, labels=classes, average="macro", zero_division=0))}
    if len(classes) == 2:
        binary = y == classes[1]
        if "positive_score" in frame:
            result.update(roc_auc=None, average_precision=None)
            if len(np.unique(y)) == 2:
                result.update(roc_auc=float(roc_auc_score(binary, frame.positive_score)),
                              average_precision=float(average_precision_score(binary, frame.positive_score)))
        if "positive_probability" in frame:
            result["brier"] = float(np.mean((binary.astype(float) - frame.positive_probability) ** 2))
    return result


def bootstrap_indices(n, rng, groups=None):
    if groups is None:
        return rng.integers(n, size=n)
    units = pd.unique(groups)
    return np.concatenate([np.flatnonzero(groups == unit) for unit in rng.choice(units, len(units))])


def uncertainty(pred, baseline, summary, membership):
    repeats = summary.get("bootstrap_repeats", 0)
    mode, groups = "row percentile bootstrap", None
    reason = None
    if not repeats:
        reason = "Bootstrap disabled or unavailable in this older run."
    elif summary["split"] == "time":
        mode = "not applied (time dependence)"
        reason = "Time dependence requires a justified block length; row bootstrap is omitted."
    elif summary["split"] == "group":
        groups = membership.set_index("sample_id").loc[pred.sample_id, "split_value"].to_numpy()
        mode = "cluster percentile bootstrap (resample whole test groups)"
        if len(pd.unique(groups)) < 5:
            reason = "Fewer than five test groups; cluster intervals are omitted."
    point = metric_values(pred, summary["task"], summary["classes"])
    base_point = metric_values(baseline, summary["task"], summary["classes"]) if baseline is not None else {}
    samples, gains = {k: [] for k in point}, {k: [] for k in point}
    rng = np.random.default_rng(summary["seed"])
    if reason is None:
        for _ in range(repeats):
            ix = bootstrap_indices(len(pred), rng, groups)
            values = metric_values(pred.iloc[ix], summary["task"], summary["classes"])
            base_values = metric_values(baseline.iloc[ix], summary["task"], summary["classes"]) if baseline is not None else {}
            for key, value in values.items():
                if value is not None and np.isfinite(value):
                    samples[key].append(value)
                    other = base_values.get(key)
                    if other is not None and key != "bias":
                        gains[key].append((other - value) if key in ("mae", "mse", "rmse", "brier") else value - other)
    rows = []
    for key, value in point.items():
        base = base_point.get(key)
        gain = None if base is None or value is None or key == "bias" else (
            base - value if key in ("mae", "mse", "rmse", "brier") else value - base)
        sufficient = len(samples[key]) >= max(100, .8 * repeats)
        paired_sufficient = len(gains[key]) >= max(100, .8 * repeats)
        lo, hi = np.quantile(samples[key], [.025, .975]) if sufficient else (None, None)
        glo, ghi = np.quantile(gains[key], [.025, .975]) if paired_sufficient else (None, None)
        rows.append({"metric": key, "estimate": value, "lower": lo, "upper": hi,
                     "baseline": base, "gain": gain, "gain_lower": glo, "gain_upper": ghi,
                     "valid_repeats": len(samples[key]), "valid_paired_repeats": len(gains[key])})
    return pd.DataFrame(rows), {"method": mode, "requested_repeats": repeats, "omitted_reason": reason,
        "scope": "95% percentile intervals conditional on this fitted model and held-out sample design; not training/selection uncertainty.",
        "units": len(pd.unique(groups)) if groups is not None else len(pred)}


def regression_bins(pred, bins=5):
    frame = pred.copy()
    frame["predicted"] = pd.to_numeric(frame.predicted)
    frame["observed"] = pd.to_numeric(frame.observed)
    frame["bin"] = pd.cut(frame.predicted, [-np.inf, np.inf]) if frame.predicted.nunique() == 1 else pd.qcut(
        frame.predicted, min(bins, frame.predicted.nunique()), duplicates="drop")
    rows = []
    for key, part in frame.groupby("bin", observed=True):
        residual = part.observed - part.predicted
        rows.append({"bin": str(key), "n": len(part), "predicted_mean": part.predicted.mean(),
                     "observed_mean": part.observed.mean(), "bias": residual.mean(),
                     "mae": residual.abs().mean(), "residual_q10": residual.quantile(.1),
                     "residual_q90": residual.quantile(.9)})
    return pd.DataFrame(rows)


def calibration_table(pred, positive):
    p = pred.positive_probability.to_numpy()
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite and between 0 and 1")
    ids = np.minimum((p * 10).astype(int), 9)
    y = (pred.observed == positive).to_numpy()
    return pd.DataFrame([{"bin": i, "left": i / 10, "right": (i + 1) / 10,
                          "n": int(np.sum(ids == i)), "mean_probability": float(p[ids == i].mean()),
                          "observed_fraction": float(y[ids == i].mean())}
                         for i in range(10) if np.any(ids == i)])


def threshold_table(pred, positive):
    score = pred.positive_score.to_numpy()
    y = (pred.observed == positive).to_numpy()
    thresholds = (np.linspace(0, 1, 101) if "positive_probability" in pred else
                  np.unique(np.r_[np.quantile(score, np.linspace(0, 1, 101)), 0]))
    rows = []
    for threshold in thresholds:
        selected = score >= threshold
        tp, fp = int(np.sum(selected & y)), int(np.sum(selected & ~y))
        fn, tn = int(np.sum(~selected & y)), int(np.sum(~selected & ~y))
        rows.append({"threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                     "sensitivity": tp / (tp + fn) if tp + fn else None,
                     "specificity": tn / (tn + fp) if tn + fp else None,
                     "precision": tp / (tp + fp) if tp + fp else None,
                     "positive_fraction": float(selected.mean())})
    return pd.DataFrame(rows)


def analyze(directory, summary):
    pred = pd.read_csv(directory / "test-predictions.csv", dtype={"sample_id": str, "observed": str, "predicted": str}, keep_default_na=False)
    membership = pd.read_csv(directory / "split-membership.csv", dtype=str, keep_default_na=False)
    baseline_path = directory / "baseline-predictions.csv"
    baseline = pd.read_csv(baseline_path, dtype={"sample_id": str, "observed": str, "predicted": str}, keep_default_na=False) if baseline_path.exists() else None
    if baseline is not None:
        baseline = baseline.set_index("sample_id").loc[pred.sample_id].reset_index()
        if not np.array_equal(pred.observed, baseline.observed):
            raise ValueError("baseline and model observations differ")
    metrics, interval = uncertainty(pred, baseline, summary, membership)
    metrics.to_csv(directory / "metric-intervals.csv", index=False)
    findings = []

    def finding(code, zh, en, source):
        findings.append({"code": code, "zh": zh, "en": en, "source": source})

    primary = {"f1": "f1_macro", "precision": "precision_macro", "recall": "recall_macro"}.get(summary["metric"], summary["metric"])
    result = metrics.set_index("metric").loc[primary]
    if pd.notna(result["gain"]):
        finding("baseline", f"测试集 {primary} 为 {result.estimate:.4g}，开发集拟合的基线为 {result.baseline:.4g}；按指标优劣方向计算的差值为 {result.gain:.4g}（正值表示改善）。" +
                (f"配对增益的 95% 区间为 [{result.gain_lower:.4g}, {result.gain_upper:.4g}]。" if pd.notna(result.gain_lower) else "当前设计未给出增益区间。"),
                f"Held-out {primary}={result.estimate:.4g}; development-fitted dummy baseline={result.baseline:.4g}; oriented gain={result.gain:.4g} (positive is better). " +
                (f"Paired 95% interval [{result.gain_lower:.4g}, {result.gain_upper:.4g}]. " if pd.notna(result.gain_lower) else "No supported gain interval for this design. "), "metric-intervals.csv; baseline-predictions.csv")
    cv = pd.read_csv(directory / "cv-scores.csv")
    if "train_score" in cv:
        selected = cv[cv.model == summary["selected_model"]]
        direction = 1 if summary["direction"] == "higher" else -1
        gap = float(direction * (selected.train_score - selected.score).mean())
        finding("generalization", f"所选模型训练与验证的平均差值为 {gap:.4g}（{summary['metric']}，按正值表示训练表现更好定向）。各折验证值介于 {selected.score.min():.4g} 与 {selected.score.max():.4g}。该差值描述模型在拟合样本与未参与拟合样本上的表现差异，折间范围同时反映划分间的变异。",
                f"The selected model's mean training-validation difference was {gap:.4g} ({summary['metric']}; oriented to favor training). Validation scores ranged from {selected.score.min():.4g} to {selected.score.max():.4g}. The difference describes transfer beyond the fitting samples; the range describes variation across folds.", "cv-scores.csv")
    if summary["task"] == "regression":
        bins = regression_bins(pred)
        bins.to_csv(directory / "regression-bins.csv", index=False)
        residual = pd.to_numeric(pred.observed) - pd.to_numeric(pred.predicted)
        pred["absolute_error"] = residual.abs()
        q50, q90 = residual.abs().quantile([.5, .9])
        finding("errors", f"测试集平均残差为 {residual.mean():.4g}，绝对误差的中位数为 {q50:.4g}，第 90 百分位数为 {q90:.4g}。残差定义为观测值减预测值，正值表示低估。中位数与高百分位数分别概括典型偏差及误差分布的上部范围。",
                f"Mean test residual was {residual.mean():.4g}; median absolute error was {q50:.4g} and its 90th percentile was {q90:.4g}. Residuals are observed minus predicted values, so positive values indicate underprediction. The median and upper percentile summarize typical error and the upper part of its distribution.", "regression-bins.csv; error-cases.csv")
        pred.sort_values("absolute_error", ascending=False).to_csv(directory / "error-cases.csv", index=False)
    else:
        precision, recall, f1, support = precision_recall_fscore_support(pred.observed, pred.predicted,
                labels=summary["classes"], zero_division=0)
        table = pd.DataFrame({"class": summary["classes"], "precision": precision, "recall": recall, "f1": f1, "n": support})
        table.loc[table.n == 0, ["recall", "f1"]] = np.nan
        table.to_csv(directory / "class-metrics.csv", index=False)
        present = table[table.n > 0]
        weakest = present.loc[present.recall.idxmin()]
        finding("class_errors", f"测试集中召回率最低的已出现类别为 {weakest['class']}（recall={weakest.recall:.3f}，n={int(weakest.n)}）。该类别中 {weakest.recall:.1%} 的记录被正确识别；混淆矩阵给出其余记录的预测去向。",
                f"The lowest recall among observed test classes was for {weakest['class']} (recall={weakest.recall:.3f}, n={int(weakest.n)}). The model correctly retrieved {weakest.recall:.1%} of this class; the confusion matrix identifies predictions for the remaining records.", "class-metrics.csv; confusion-matrix.csv")
        pred["error"] = pred.observed != pred.predicted
        pred.sort_values("error", ascending=False).to_csv(directory / "error-cases.csv", index=False)
        if len(summary["classes"]) == 2 and "positive_score" in pred:
            threshold_table(pred, summary["classes"][1]).to_csv(directory / "thresholds.csv", index=False)
            if "positive_probability" in pred:
                calibration = calibration_table(pred, summary["classes"][1])
                calibration.to_csv(directory / "calibration.csv", index=False)
                ece = float(np.average(abs(calibration.observed_fraction - calibration.mean_probability), weights=calibration.n))
                brier = float(metrics.set_index("metric").loc["brier", "estimate"])
                finding("calibration", f"测试集 Brier 分数为 {brier:.4g}，基于 10 个等宽分箱的预期校准误差（ECE）为 {ece:.4g}。Brier 分数是预测概率与实际二分类结果的均方差；ECE 是各箱预测概率与观测阳性比例绝对差的样本量加权平均。两者均以较低值表示较小误差，ECE 的数值同时依赖分箱与样本量。",
                        f"Test Brier score was {brier:.4g}; expected calibration error (ECE) across 10 equal-width bins was {ece:.4g}. Brier is the mean squared probability error. ECE is the count-weighted mean absolute difference between predicted and observed positive fractions across bins. Lower values indicate less error; ECE also depends on binning and sample size.", "calibration.csv; thresholds.csv; metric-intervals.csv")
            else:
                finding("decision_score", "模型提供决策分数而非概率，因此仅绘制阈值权衡，不计算 Brier 或概率校准。", "The model exposes decision scores, not probabilities; threshold trade-offs are available but probability calibration and Brier are omitted.", "test-predictions.csv")
    if summary["split"] in ("group", "time"):
        part = pred.merge(membership[membership.partition == "test"], on="sample_id", validate="one_to_one")
        if summary["split"] == "time":
            values = part.split_value
            numeric = pd.to_numeric(values, errors="coerce")
            order = numeric if numeric.notna().all() else pd.to_datetime(values, utc=True)
            part = part.iloc[np.argsort(order.to_numpy(), kind="stable")].copy()
            units = part.split_value.drop_duplicates().to_numpy()
            mapping = {value: i + 1 for i, block in enumerate(np.array_split(units, min(8, len(units)))) for value in block}
            part["stratum"] = part.split_value.map(mapping)
        else:
            part["stratum"] = part.split_value
        rows = []
        for key, block in part.groupby("stratum", sort=True):
            values = metric_values(block, summary["task"], summary["classes"])
            measure = "mae" if summary["task"] == "regression" else "accuracy"
            rows.append({"stratum": str(key), "n": len(block), "metric": measure, "value": values[measure],
                         "start": block.split_value.iloc[0], "end": block.split_value.iloc[-1]})
        strata = pd.DataFrame(rows)
        strata.to_csv(directory / "strata-metrics.csv", index=False)
        part[["sample_id", "stratum", "split_value"]].to_csv(directory / "strata-membership.csv", index=False)
        finding("stability", f"测试集按{'组别' if summary['split'] == 'group' else '有序时间块'}分解后，{measure} 范围为 [{strata.value.min():.4g}, {strata.value.max():.4g}]，各层 n={strata.n.min()}–{strata.n.max()}。该差异为描述性结果，可能受小样本和类别构成影响，不能直接归因为组别或时间效应。",
                f"Held-out {summary['split']} strata: {measure} range [{strata.value.min():.4g}, {strata.value.max():.4g}], n={strata.n.min()}–{strata.n.max()} per stratum. Differences are descriptive and may reflect sample size or class composition; they do not establish group/time effects.", "strata-metrics.csv; strata-membership.csv")
    if (directory / "feature-shift.csv").exists():
        shift = pd.read_csv(directory / "feature-shift.csv")
        defined = shift.dropna(subset=["standardized_mean_shift"])
        if len(defined):
            top = defined.loc[defined.standardized_mean_shift.abs().idxmax()]
            finding("shift", f"均值位移最大的可计算特征为 {top.feature}，测试/开发差异={top.standardized_mean_shift:.3g} 个开发集标准差。漂移图同时列出缺失率；均值差仅是分布诊断，不能证明性能下降的原因。开发集常量或全缺失特征的标准化位移不定义。",
                    f"Largest defined feature mean shift: {top.feature}, {top.standardized_mean_shift:.3g} development SDs. Missingness is shown separately. Mean shifts do not establish the cause of performance changes; standardized shifts are undefined for constant/all-missing development features.", "feature-shift.csv")
    if (directory / "importance.csv").exists():
        imp = pd.read_csv(directory / "importance.csv", dtype={"feature": str}, keep_default_na=False)
        leader = imp.iloc[0]
        finding("reliance", f"置换重要性最高的特征为 {leader.feature}，评分下降均值为 {leader['mean']:.4g}，重复间标准差为 {leader.sd:.4g}。{len(imp)} 个特征中有 {int((imp['mean'] <= 0).sum())} 个的平均重要性不大于零，表示在当前模型及置换条件下未观察到平均评分下降。",
                f"The highest permutation importance was for {leader.feature}, with mean score decrease {leader['mean']:.4g} and repeat SD {leader.sd:.4g}. Of {len(imp)} features, {int((imp['mean'] <= 0).sum())} had nonpositive mean importance: their permutations produced no average score decrease under the fitted model and current procedure.", "importance.csv; importance-repeats.csv; feature-correlations.csv")
    result = {"schema": "psytrainer-diagnostics/v1", "interval": interval, "findings": findings,
              "test_n": len(pred), "limitations": ["No causal claims or automatic test-driven tuning.",
              "No confidence bands for calibration, thresholds or strata; these are descriptive.",
              "OOF predictions are used for selection diagnostics and are not unbiased post-selection performance."]}
    (directory / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result
