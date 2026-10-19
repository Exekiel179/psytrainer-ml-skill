"""Findings-driven report narrative built from measured facts (report_facts.collect).

Design rules
- Lead with the answer to the research question, then the evidence for it.
- State each number once where it matters; definitions and reading guides live
  in appendices, and general limitations are consolidated in the discussion.
- Only mention data issues that this run actually triggered.
- Hedged readings (e.g. what a linear model winning suggests) are labelled as
  readings of the comparison, never as tests of mechanism.
"""

from __future__ import annotations

from report_explanations import model_name

LOSSES = ("mae", "mse", "rmse", "brier")

METRIC_SHORT = {
    "mae": ("MAE", "MAE"), "mse": ("MSE", "MSE"), "rmse": ("RMSE", "RMSE"), "r2": ("R²", "R²"),
    "pearson_r": ("Pearson r", "Pearson r"), "accuracy": ("准确率", "accuracy"),
    "balanced_accuracy": ("平衡准确率", "balanced accuracy"), "f1_macro": ("宏平均 F1", "macro F1"),
    "precision_macro": ("宏平均精确率", "macro precision"), "recall_macro": ("宏平均召回率", "macro recall"),
    "roc_auc": ("AUC", "AUC"), "average_precision": ("平均精确率", "average precision"),
    "brier": ("Brier 分数", "Brier score"), "bias": ("平均残差", "mean residual"),
}
FAMILY_NAMES = {
    "linear": ("线性模型", "linear"), "kernel": ("核方法", "kernel"), "neighbors": ("近邻方法", "nearest-neighbour"),
    "tree": ("单棵决策树", "single tree"), "bagging": ("袋装树集成", "bagged trees"),
    "boosting": ("提升树集成", "boosted trees"), "probabilistic": ("概率模型", "probabilistic"),
    "other": ("其他", "other"),
}
STANDING = {
    "selected": ("选定", "selected"), "sole": ("选定（唯一）", "selected (only)"),
    "near": ("与选定接近", "close to selected"), "weaker": ("较弱", "weaker"),
    "clearly_weaker": ("明显较弱", "clearly weaker"), "unranked": ("未排序", "unranked"),
}
BASELINES = {
    "mean": ("始终预测开发集结局均值", "always predicts the development outcome mean"),
    "median": ("始终预测开发集结局中位数", "always predicts the development outcome median"),
    "prior": ("按开发集类别比例给出概率、预测最多见类别", "uses development class frequencies and predicts the most frequent class"),
    "most_frequent": ("始终预测开发集最多见类别", "always predicts the most frequent development class"),
}


# ---------------------------------------------------------------- formatting

def fmt(value, digits=3):
    if value is None:
        return "NA"
    value = float(value)
    if abs(value) < 1e-9:
        return "0"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    text = f"{value:.{digits}g}"
    if "e" in text:
        text = f"{value:.2g}"
    return text


def pct(value):
    if value is None:
        return "NA"
    value = 100 * float(value)
    return f"{value:.0f}%" if abs(value) >= 10 else f"{value:.1f}%"


def metric_label(key, zh):
    return METRIC_SHORT.get(key, (key, key))[0 if zh else 1]


def join(items, zh, last=None):
    items = [str(i) for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if zh:
        return "、".join(items[:-1]) + (last or "和") + items[-1]
    return ", ".join(items[:-1]) + f" {last or 'and'} " + items[-1]


_CJK = "\u4e00-\u9fff\u3400-\u4dbf"


def tidy(text, zh):
    """Normalise spacing: CJK/Latin boundaries get one space in Chinese prose."""
    import re
    if not zh:
        text = re.sub(r" {2,}", " ", text).strip()
        return text[:1].upper() + text[1:]
    text = re.sub(rf"(?<=[{_CJK}])(?=[A-Za-z0-9(\[])", " ", text)
    text = re.sub(rf"(?<=[A-Za-z0-9%)\]²])(?=[{_CJK}])", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r" (?=[，。；：、）！？”])", "", text)
    text = re.sub(r"(?<=[（“。；：，、）]) ", "", text)
    return text.strip()


def interval(lo, hi):
    return f"[{fmt(lo)}, {fmt(hi)}]" if lo is not None and hi is not None else None


class Narrator:
    """Holds one run's facts and renders zh/en sentences from them."""

    def __init__(self, s, facts, zh):
        self.s, self.f, self.zh = s, facts, zh
        self.ctx = facts.get("context") or {}
        self.metric = {"f1": "f1_macro", "precision": "precision_macro", "recall": "recall_macro"}.get(s["metric"], s["metric"])

    # ------------------------------------------------------------ names
    def t(self, zh, en):
        return zh if self.zh else en

    @property
    def target(self):
        label = self.ctx.get("target_label")
        return label or self.s["target"]

    @property
    def unit(self):
        unit = self.ctx.get("target_unit")
        return f" {unit}" if unit and not self.zh else (unit or "")

    def feature(self, name):
        label = (self.ctx.get("feature_labels") or {}).get(name)
        if not label:
            return name
        return f"{label}（{name}）" if self.zh else f"{label} ({name})"

    def cls(self, name):
        label = (self.ctx.get("class_labels") or {}).get(str(name))
        if not label:
            return str(name)
        return f"{label}（{name}）" if self.zh else f"{label} ({name})"

    def model(self, tag):
        return model_name(tag, self.zh)

    def family(self, key):
        return FAMILY_NAMES.get(key, FAMILY_NAMES["other"])[0 if self.zh else 1]

    def mlabel(self, key=None):
        return metric_label(key or self.metric, self.zh)

    def better(self):
        return self.t("越低越好", "lower is better") if self.s["direction"] == "lower" else self.t("越高越好", "higher is better")

    # ------------------------------------------------------------ verdict
    def verdict(self):
        p = self.f["primary"]
        gain = p.get("gain")
        if gain is None or p.get("estimate") is None:
            return "unknown"
        if gain <= 0:
            return "not_supported"
        lo, hi = p.get("gain_lower"), p.get("gain_upper")
        if lo is None or hi is None:
            return "supported_point"
        return "supported" if lo > 0 else "uncertain"

    def leaders_text(self):
        imp = self.f.get("importance") or {}
        leaders = imp.get("leaders") or []
        if not leaders:
            return ""
        names = join([self.feature(x) for x in leaders], self.zh)
        share = imp.get("leader_share")
        if len(leaders) == 1:
            return self.t(f"预测主要依赖{names}", f"predictions relied mainly on {names}")
        return self.t(f"预测主要依赖{names}（合计约占正向置换重要性的 {pct(share)}）",
                      f"predictions relied mainly on {names} (about {pct(share)} of positive permutation importance)")

    def baseline_comparison(self, subject=None):
        """'MAE 0.163 vs 1.71 for the baseline (90% lower)' style clause."""
        p = self.f["primary"]
        est, base = p.get("estimate"), p.get("baseline")
        subject = self.t("测试集", "test ") if subject is None else subject
        if est is None:
            return self.t(f"{subject}{self.mlabel()}未能计算", f"{subject}{self.mlabel()} was undefined")
        ci = interval(p.get("lower"), p.get("upper"))
        text = self.t(f"{subject}{self.mlabel()}为 {fmt(est)}{self.unit}", (f"{subject}{self.mlabel()} was {fmt(est)}{self.unit}" if subject else f"{self.mlabel()} of {fmt(est)}{self.unit}"))
        if ci:
            text += f"（95% CI {ci}）" if self.zh else f" (95% CI {ci})"
        if base is not None:
            rel = p.get("relative")
            if rel is not None and rel > 0:
                text += self.t(f"，比基线（{fmt(base)}）低 {pct(rel)}", f", {pct(rel)} lower than the baseline ({fmt(base)})")
            elif rel is not None and rel < 0:
                text += self.t(f"，高于基线（{fmt(base)}）", f", higher than the baseline ({fmt(base)})")
            else:
                gain = p.get("gain")
                direction = (self.t("高于", "above") if (gain or 0) > 0 else self.t("低于", "below") if (gain or 0) < 0
                             else self.t("等于", "equal to"))
                text += self.t(f"，{direction}基线（{fmt(base)}）", f", {direction} the baseline ({fmt(base)})")
            gci = interval(p.get("gain_lower"), p.get("gain_upper"))
            if gci:
                text += self.t(f"；配对改善 {fmt(p.get('gain'))}（95% CI {gci}）",
                               f"; paired improvement {fmt(p.get('gain'))} (95% CI {gci})")
        return text

    def secondary_test_clause(self):
        tests = self.f["tests"]
        bits = []
        if self.s["task"] == "regression":
            r2 = tests.get("r2", {})
            if r2.get("estimate") is not None and self.metric != "r2":
                ci = interval(r2.get("lower"), r2.get("upper"))
                bits.append(f"R² = {fmt(r2['estimate'])}" + (f"（95% CI {ci}）" if ci and self.zh else f" (95% CI {ci})" if ci else ""))
            reg = self.f.get("regression") or {}
            if reg.get("within_tolerance") is not None:
                bits.append(self.t(f"{pct(reg['within_tolerance'])} 的测试记录误差不超过预设容许值 {fmt(reg['tolerance'])}{self.unit}",
                                   f"{pct(reg['within_tolerance'])} of test records were within the prespecified tolerance of {fmt(reg['tolerance'])}{self.unit}"))
        else:
            for key in ("roc_auc", "balanced_accuracy", "accuracy"):
                row = tests.get(key, {})
                if key != self.metric and row.get("estimate") is not None:
                    ci = interval(row.get("lower"), row.get("upper"))
                    bits.append(f"{self.mlabel(key)} = {fmt(row['estimate'])}" + (f"（95% CI {ci}）" if ci and self.zh else f" (95% CI {ci})" if ci else ""))
                    if len(bits) == 2:
                        break
        return self.t("；".join(bits), "; ".join(bits))

    def conclusion(self, leaders=True):
        v = self.verdict()
        target = self.target
        name = self.model(self.s["selected_model"])
        lead = self.leaders_text()
        if v == "supported":
            text = self.t(f"在本数据中，现有特征包含预测{target}的有效信息：{name}在测试集上优于不使用特征的基线，且配对改善的 95% 区间位于零以上",
                          f"In these data the features carry usable information about {target}: {name} outperformed the feature-free baseline on the test set, with a paired improvement interval above zero")
        elif v == "supported_point":
            text = self.t(f"在本数据中，{name}在测试集上优于不使用特征的基线，提示现有特征包含预测{target}的信息；本次未提供区间，优势的抽样不确定性尚未量化",
                          f"{name} outperformed the feature-free baseline on the test set, suggesting the features carry information about {target}; this design gives no interval, so sampling uncertainty of the advantage is unquantified")
        elif v == "uncertain":
            text = self.t(f"{name}的测试集点估计优于基线，但配对差值区间包含零，现有测试样本不足以确认特征对预测{target}带来可靠增益",
                          f"{name} beat the baseline on point estimate, but the paired interval includes zero; the test sample cannot confirm a reliable gain for predicting {target}")
        elif v == "not_supported":
            text = self.t(f"{name}在测试集上未优于不使用特征的基线；就本数据和本流程而言，结果不支持现有特征能够预测{target}",
                          f"{name} did not outperform the feature-free baseline on the test set; within these data and this pipeline, the results do not support that the features predict {target}")
        else:
            text = self.t(f"测试集主要指标或基线无法计算，无法判断现有特征能否预测{target}",
                          f"The primary test metric or baseline was undefined, so whether the features predict {target} cannot be judged")
        if leaders and lead and v in ("supported", "supported_point"):
            text += self.t(f"，{lead}", f"; {lead}")
        return text + self.t("。", ".")

    # ------------------------------------------------------------ abstract
    def abstract(self):
        s, f = self.s, self.f
        question = self.ctx.get("question") or s.get("question")
        design = {"random": ("随机", "random"), "group": ("分组", "grouped"), "time": ("时间前向", "forward-time")}[s["split"]][0 if self.zh else 1]
        source = (self.t("外部测试集", "external test set") if s["external_test"] else
                  self.t("内部留出测试集", "internal holdout test set"))
        names = join([self.model(r["model"]) for r in f["comparison"]], self.zh)
        objective = question or self.t(f"评估现有特征能否预测{self.target}，并比较候选模型。",
                                       f"To assess whether the available features predict {self.target} and to compare candidate models.")
        method = self.t(
            f"{s['development_n']} 条开发记录、{s['test_n']} 条{source}记录，{s['feature_n']} 项特征。"
            f"在开发集内以 {s['cv_folds']} 折{design}交叉验证比较 {len(f['comparison'])} 个模型（{names}），"
            f"按{self.mlabel()}（{self.better()}）选择最终模型，再在测试集上与不使用特征的基线比较。",
            f"{s['development_n']} development and {s['test_n']} {source} records with {s['feature_n']} features. "
            f"{len(f['comparison'])} models ({names}) were compared by {s['cv_folds']}-fold {design} cross-validation on "
            f"{self.mlabel()} ({self.better()}); the selected model was then compared with a feature-free baseline on the test set.")
        sel = f.get("selected") or {}
        result = self.t(f"{self.model(s['selected_model'])}被选定（验证{self.mlabel()} {fmt(sel.get('mean'))} ± {fmt(sel.get('sd'))}）。",
                        f"{self.model(s['selected_model'])} was selected (validation {self.mlabel()} {fmt(sel.get('mean'))} ± {fmt(sel.get('sd'))}). ")
        clause = self.baseline_comparison()
        result += (clause if self.zh else clause[:1].upper() + clause[1:]) + self.t("。", ". ")
        secondary = self.secondary_test_clause()
        if secondary:
            result += secondary + self.t("。", ". ")
        lead = self.leaders_text()
        if lead:
            result += (lead[0].upper() + lead[1:] if not self.zh else lead) + self.t("。", ". ")
        caveat = self.t("测试集为内部留出样本，结论需在独立来源数据上验证。" if not s["external_test"] else
                        "外部测试结果的推广范围取决于该测试数据的来源与采集方式。",
                        "The test set is an internal holdout; conclusions need validation on independent data." if not s["external_test"] else
                        "Transfer beyond the external test sample depends on its provenance.")
        flags = self.data_flags()
        if flags:
            caveat = flags[0] + self.t("", " ") + caveat
        return [("目的" if self.zh else "Objective", objective),
                ("方法" if self.zh else "Methods", method),
                ("结果" if self.zh else "Results", result.strip()),
                ("结论" if self.zh else "Conclusion", self.conclusion(leaders=False) + self.t("", " ") + caveat)]

    # ------------------------------------------------------------ key points
    def key_points(self):
        points = [self.conclusion(leaders=False)]
        comparison = self.model_comparison_sentence()
        if comparison:
            points.append(comparison)
        predictors = self.predictor_sentence(short=True)
        if predictors:
            points.append(predictors)
        errors = self.error_sentence(short=True)
        if errors:
            points.append(errors)
        points.extend(self.data_flags())
        cal = self.calibration_sentence()
        if cal:
            points.append(cal)
        return points

    def model_comparison_sentence(self, reading=True):
        f = self.f
        rows, sel = f["comparison"], f.get("selected")
        if not sel or len(rows) < 2:
            return ""
        others = sorted([r for r in rows if r is not sel], key=lambda r: r.get("margin") or 0)
        runner = others[0]
        text = self.t(f"在 {len(rows)} 个候选中，{self.model(sel['model'])}的验证{self.mlabel()}最好（{fmt(sel['mean'])}）；",
                      f"Among {len(rows)} candidates, {self.model(sel['model'])} had the best validation {self.mlabel()} ({fmt(sel['mean'])}); ")
        text += self.margin_clause(runner)
        short = self.family_reading(short=True) if reading else ""
        if short:
            text += self.t("。", ". ") + short
        return text + self.t("。", ".")

    def margin_clause(self, row):
        name = self.model(row["model"])
        rel, ratio = row.get("relative_margin"), row.get("margin_sd")
        text = self.t(f"次优的{name}为 {fmt(row['mean'])}", f"the runner-up {name} had {fmt(row['mean'])}")
        details = []
        if rel is not None and rel > 0:
            details.append(self.t(f"选定模型误差低 {pct(rel)}", f"{pct(rel)} lower error for the selected model"))
        if ratio is not None:
            details.append(self.t(f"差距约为 {fmt(ratio, 2)} 个折间标准差", f"a gap of about {fmt(ratio, 2)} between-fold SDs"))
        if details:
            text += self.t("（" + "，".join(details) + "）", " (" + "; ".join(details) + ")")
        if row["standing"] == "near":
            text += self.t("，两者差距在折间波动范围内", ", within between-fold variation")
        return text

    def family_reading(self, short=False):
        f = self.f
        reading = f.get("family_reading")
        if not reading:
            return ""
        rows = f["comparison"]
        linear = [r for r in rows if r["family"] == "linear"]
        nonlinear = [r for r in rows if r["family"] not in ("linear", "probabilistic", "other")]
        sel = f["selected"]
        if reading == "linear_best":
            best_nl = sorted(nonlinear, key=lambda r: r.get("margin") or 0)[0] if nonlinear else None
            near = best_nl is not None and best_nl["standing"] == "near"
            if near:
                return self.t("线性与非线性模型表现接近，更灵活的模型没有带来明显收益",
                              "Linear and non-linear models performed similarly; extra flexibility brought no clear gain")
            if short:
                return self.t("线性模型优于各非线性模型，提示结局与特征的关系可由线性结构较好近似",
                              "The linear model outperformed all non-linear models, suggesting an approximately linear relationship")
            return self.t("线性模型优于各非线性模型，提示在当前样本量下，结局与特征的关系可以由加性线性结构较好地近似，"
                          "更灵活的模型没有从额外复杂度中获益（也可能是样本量不足以支撑其更高的方差）。这是对模型比较的解读，并非对数据生成机制的检验。",
                          "The linear model outperformed all non-linear models, suggesting that at this sample size an additive linear "
                          "structure approximates the relationship well and flexible models did not benefit from extra complexity "
                          "(or lacked the data to support their variance). This is a reading of the comparison, not a test of mechanism.")
        best_lin = linear[0] if linear else None
        if best_lin is not None and best_lin["standing"] == "near":
            return self.t("非线性模型与线性模型表现接近，非线性结构带来的收益有限",
                          "Non-linear and linear models performed similarly; non-linear structure added little")
        if short:
            return self.t(f"{self.family(sel['family'])}优于线性模型，提示可能存在非线性关系或交互作用",
                          f"The {self.family(sel['family'])} model beat the linear model, suggesting non-linearity or interactions")
        return self.t(f"{self.family(sel['family'])}（{self.model(sel['model'])}）优于线性模型，提示特征与结局之间可能存在非线性关系或交互作用；"
                      "置换重要性只反映依赖程度，不能指明这种关系的具体形式。",
                      f"A {self.family(sel['family'])} model ({self.model(sel['model'])}) beat the linear model, suggesting non-linear relationships or "
                      "interactions; permutation importance shows reliance, not the form of these relationships.")

    def predictor_sentence(self, short=False):
        imp = self.f.get("importance") or {}
        ranked = imp.get("ranked") or []
        if not ranked:
            return ""
        top = [r for r in ranked if r["mean"] > 0 and r["share"] >= 0.05][:3] or [r for r in ranked if r["mean"] > 0][:1]
        if not top:
            return self.t("置换任何单个特征都未使测试表现平均下降，模型可能依赖相互替代的特征组合，或未从特征中获得稳定信息。",
                          "Permuting any single feature did not reduce test performance on average; the model may use substitutable features or carry little stable signal.")
        parts = [self.t(f"{self.feature(r['feature'])}（{fmt(r['mean'])}）", f"{self.feature(r['feature'])} ({fmt(r['mean'])})") for r in top]
        text = self.t(f"置换重要性最高的特征依次为{join(parts, True)}", f"The highest permutation importance was for {join(parts, False)}")
        lead_share = ranked[0].get("share")
        if lead_share and lead_share >= 0.5 and len(ranked) > 1:
            text += self.t(f"；{self.feature(ranked[0]['feature'])}单独约占正向重要性的 {pct(lead_share)}",
                           f"; {self.feature(ranked[0]['feature'])} alone accounted for about {pct(lead_share)} of positive importance")
        unstable = [r for r in top if not r["stable"]]
        if unstable and len(unstable) == len(top):
            text += self.t("。这些估计的重复间波动较大（均值不足两倍重复间标准差），特征之间的排序并不稳定",
                           ". These estimates varied considerably across permutations (mean below two repeat SDs), so the ranking is not stable")
        elif unstable:
            text += self.t(f"。其中{join([self.feature(r['feature']) for r in unstable], True)}的重复间波动较大，其排序不稳定",
                           f". Estimates for {join([self.feature(r['feature']) for r in unstable], False)} varied considerably across permutations")
        negligible = imp.get("negligible") or []
        if negligible:
            text += self.t(f"。{join([self.feature(x) for x in negligible], True)}被置换后测试表现几乎不变（平均下降为非正值，或不足正向重要性的 2% 且小于两倍重复间标准差），模型对其几乎没有依赖",
                           f". Permuting {join([self.feature(x) for x in negligible], False)} left test performance essentially unchanged (non-positive mean decrease, or under 2% of positive importance and within two repeat SDs), so the model barely relied on them")
        pairs = imp.get("correlated_pairs") or []
        if pairs and not short:
            pair = pairs[0]
            text += self.t(f"。{self.feature(pair['a'])}与{self.feature(pair['b'])}高度相关（Spearman ρ = {fmt(pair['rho'], 2)}），二者的单列重要性可能相互掩盖",
                           f". {self.feature(pair['a'])} and {self.feature(pair['b'])} were strongly correlated (Spearman ρ = {fmt(pair['rho'], 2)}), so their single-column importances may mask each other")
        return text + self.t("。", ".")

    def error_sentence(self, short=False):
        if self.s["task"] == "regression":
            reg = self.f.get("regression") or {}
            if not reg:
                return ""
            text = self.t(f"测试集绝对误差中位数为 {fmt(reg['median_ae'])}{self.unit}，90% 的记录误差不超过 {fmt(reg['p90_ae'])}{self.unit}，最大为 {fmt(reg['max_ae'])}{self.unit}",
                          f"Median absolute test error was {fmt(reg['median_ae'])}{self.unit}; 90% of records were within {fmt(reg['p90_ae'])}{self.unit}, maximum {fmt(reg['max_ae'])}{self.unit}")
            sd = reg.get("target_sd_test")
            if sd:
                text += self.t(f"（测试集结局标准差为 {fmt(sd)}）", f" (test outcome SD {fmt(sd)})")
            bias = self.f["tests"].get("bias", {})
            lo, hi = bias.get("lower"), bias.get("upper")
            if lo is not None and hi is not None and (lo > 0 or hi < 0):
                text += self.t(f"。平均残差为 {fmt(bias['estimate'])}（95% CI {interval(lo, hi)}），模型整体{'低估' if lo > 0 else '高估'}",
                               f". Mean residual was {fmt(bias['estimate'])} (95% CI {interval(lo, hi)}), indicating overall {'under' if lo > 0 else 'over'}prediction")
            edge = reg.get("edge_bias")
            worst = reg.get("worst_bin")
            if not short and edge and worst and worst["n"] >= 10 and reg.get("best_bin_mae") and worst["mae"] > 1.5 * reg["best_bin_mae"]:
                text += self.t(f"。误差在预测值 {worst['range']} 区间最大（MAE {fmt(worst['mae'])}，n = {worst['n']}）",
                               f". Error was largest for predictions in {worst['range']} (MAE {fmt(worst['mae'])}, n = {worst['n']})")
            if not short and edge and edge["low"] < 0 < edge["high"] and min(abs(edge["low"]), abs(edge["high"])) > 0.5 * (reg["median_ae"] or 0):
                text += self.t("；最低一组预测平均偏高、最高一组平均偏低，预测向均值收缩，极端结局可能被低估",
                               "; the lowest predictions were too high and the highest too low on average, i.e. predictions shrink toward the mean and extremes may be underestimated")
            return text + self.t("。", ".")
        cls = self.f.get("classification") or {}
        weakest = cls.get("weakest")
        if not weakest:
            return ""
        text = self.t(f"各类召回率在 {fmt(weakest['recall'])} 到 {fmt(cls['strongest']['recall'])} 之间，{self.cls(weakest['class'])}最难识别（召回率 {fmt(weakest['recall'])}，n = {weakest['n']}）",
                      f"Per-class recall ranged from {fmt(weakest['recall'])} to {fmt(cls['strongest']['recall'])}; {self.cls(weakest['class'])} was hardest to identify (recall {fmt(weakest['recall'])}, n = {weakest['n']})")
        confusion = cls.get("top_confusion")
        if confusion and not short:
            text += self.t(f"。最常见的错误是将{self.cls(confusion['observed'])}判为{self.cls(confusion['predicted'])}（{confusion['n']} 例，占全部错误的 {pct(confusion['share'])}）",
                           f". The most common error was predicting {self.cls(confusion['predicted'])} for {self.cls(confusion['observed'])} ({confusion['n']} cases, {pct(confusion['share'])} of errors)")
        return text + self.t("。", ".")

    def calibration_sentence(self):
        cls = self.f.get("classification") or {}
        ece = cls.get("ece")
        if ece is None:
            return ""
        brier = self.f["tests"].get("brier", {}).get("estimate")
        direction = cls.get("calibration_direction")
        text = self.t(f"预测概率的校准误差（ECE）为 {fmt(ece)}", f"Probability calibration error (ECE) was {fmt(ece)}")
        if brier is not None:
            text += self.t(f"，Brier 分数为 {fmt(brier)}", f", Brier score {fmt(brier)}")
        positive = self.cls(self.s["classes"][1]) if len(self.s.get("classes") or []) == 2 else ""
        if ece > 0.1 and direction in ("over", "under"):
            text += self.t(f"；整体上模型{'高估' if direction == 'over' else '低估'}了属于{positive}的概率，若直接使用概率需先校准",
                           f"; the model tended to {'over' if direction == 'over' else 'under'}estimate the probability of {positive}, so probabilities need recalibration before direct use")
        return text + self.t("。", ".")

    def data_flags(self):
        shift = self.f.get("shift") or {}
        flags = []
        for item in (shift.get("missing") or [])[:2]:
            flags.append(self.t(f"{self.feature(item['feature'])}在测试集中的缺失率（{pct(item['test'])}）与开发集（{pct(item['development'])}）明显不同，测试表现可能部分反映缺失值填补的影响。",
                                f"{self.feature(item['feature'])} was missing in {pct(item['test'])} of test records versus {pct(item['development'])} in development, so test results partly reflect imputation."))
        shifted = shift.get("mean_shift") or []
        if shifted:
            names = join([f"{self.feature(x['feature'])}（{fmt(x['shift'], 2)} SD）" if self.zh else
                          f"{self.feature(x['feature'])} ({fmt(x['shift'], 2)} SD)" for x in shifted[:3]], self.zh)
            flags.append(self.t(f"测试集与开发集在{names}上的均值相差至少半个开发集标准差，两组样本并非完全可比。",
                                f"Test and development means differed by at least half a development SD for {names}; the partitions are not fully comparable."))
        ranged = shift.get("out_of_range") or []
        if ranged:
            item = ranged[0]
            flags.append(self.t(f"{pct(item['fraction'])} 的测试记录在{self.feature(item['feature'])}上超出开发集取值范围，模型对这些记录是外推预测。",
                                f"{pct(item['fraction'])} of test records fell outside the development range of {self.feature(item['feature'])}, so they were extrapolated."))
        strata = self.f.get("strata") or {}
        if strata and strata.get("n_strata", 0) > 1:
            unit = self.t("组", "group") if self.s["split"] == "group" else self.t("时间块", "time block")
            sizes = str(strata["n_min"]) if strata["n_min"] == strata["n_max"] else f"{strata['n_min']}–{strata['n_max']}"
            flags.append(self.t(f"按{unit}分层，测试集{self.mlabel(strata['measure'])}在 {fmt(strata['min'])} 到 {fmt(strata['max'])} 之间变化（每层 n = {sizes}）。",
                                f"Across test {unit}s, {self.mlabel(strata['measure'])} ranged from {fmt(strata['min'])} to {fmt(strata['max'])} (n = {sizes} per stratum)."))
        return flags

    # ------------------------------------------------------------ methods
    def data_description(self):
        s, tgt = self.s, self.f.get("target") or {}
        out = []
        population = self.ctx.get("population")
        text = self.t(f"共 {s['development_n'] + s['test_n']} 条记录：开发集 {s['development_n']} 条，测试集 {s['test_n']} 条；预测变量 {s['feature_n']} 项，结局变量为{self.target}",
                      f"The analysis used {s['development_n'] + s['test_n']} records: {s['development_n']} for development and {s['test_n']} for testing, with {s['feature_n']} predictors and outcome {self.target}")
        if self.ctx.get("target_label"):
            text += f"（{s['target']}）" if self.zh else f" ({s['target']})"
        if self.ctx.get("target_unit"):
            text += self.t(f"，单位为{self.ctx['target_unit']}", f", measured in {self.ctx['target_unit']}")
        names = self.f.get("features") or []
        if names and len(names) <= 20:
            text += self.t(f"。预测变量为{join([self.feature(x) for x in names], True)}", f". Predictors: {join([self.feature(x) for x in names], False)}")
        out.append(text + self.t("。", ".") + (population if population else ""))
        if self.ctx.get("background"):
            out.insert(0, self.ctx["background"])
        dev, test = tgt.get("development") or {}, tgt.get("test") or {}
        if self.s["task"] == "regression" and dev.get("mean") is not None:
            out.append(self.t(
                f"开发集结局均值 {fmt(dev['mean'])}（SD {fmt(dev.get('sd'))}，范围 {fmt(dev['min'])} 至 {fmt(dev['max'])}）；"
                f"测试集均值 {fmt(test.get('mean'))}（SD {fmt(test.get('sd'))}，范围 {fmt(test.get('min'))} 至 {fmt(test.get('max'))}）。",
                f"Development outcome mean {fmt(dev['mean'])} (SD {fmt(dev.get('sd'))}, range {fmt(dev['min'])} to {fmt(dev['max'])}); "
                f"test mean {fmt(test.get('mean'))} (SD {fmt(test.get('sd'))}, range {fmt(test.get('min'))} to {fmt(test.get('max'))})."))
        elif dev.get("counts"):
            def counts(entry):
                total = sum(entry.values()) or 1
                return join([f"{self.cls(k)} {v}（{pct(v / total)}）" if self.zh else f"{self.cls(k)} {v} ({pct(v / total)})"
                             for k, v in entry.items()], self.zh)
            out.append(self.t(f"开发集类别构成：{counts(dev['counts'])}；测试集：{counts(test.get('counts') or {})}。",
                              f"Development classes: {counts(dev['counts'])}; test: {counts(test.get('counts') or {})}."))
        shift = self.f.get("shift") or {}
        if dev.get("complete_rows") is not None:
            if shift.get("any_missing"):
                out.append(self.t(f"开发集 {pct(dev['complete_rows'])}、测试集 {pct(test.get('complete_rows'))} 的记录特征完整。",
                                  f"{pct(dev['complete_rows'])} of development and {pct(test.get('complete_rows'))} of test records had complete predictors."))
            else:
                out.append(self.t("所有记录的特征均无缺失。", "No predictor values were missing."))
        return out

    def selection_methods(self):
        s, rows = self.s, self.f["comparison"]
        items = join([f"{self.model(r['model'])}（{self.family(r['family'])}）" if self.zh else
                      f"{self.model(r['model'])} ({self.family(r['family'])})" for r in rows], self.zh)
        text = self.t(f"比较的候选模型为{items}。各模型使用相同的交叉验证划分，以验证{self.mlabel()}均值最优者为最终模型，随后在完整开发集上重新拟合。",
                      f"Candidates were {items}. All shared the same cross-validation splits; the best mean validation {self.mlabel()} selected the final model, which was refitted on all development data.")
        if s.get("failures"):
            failed = join([self.model(x.get("model", str(x))) if isinstance(x, dict) else str(x) for x in s["failures"]], self.zh)
            text += self.t(f"{failed}未能完成训练，未参与比较（原因见 run.log 与 summary.json）。",
                           f" {failed} failed to train and were excluded (see run.log and summary.json).")
        return text

    def evaluation_methods(self, uncertainty_lines):
        s = self.s
        strategy = (s.get("baseline") or {}).get("strategy")
        base = BASELINES.get(strategy)
        base_text = self.t(f"基线模型{base[0]}，不使用任何特征；它在开发集上拟合，用于衡量特征带来的增益。" if base else
                           "基线模型不使用任何特征，只在开发集上学习简单规则，用于衡量特征带来的增益。",
                           f"The baseline {base[1]} without using any predictor; fitted on development data, it measures the gain contributed by the features." if base else
                           "The baseline learns a simple feature-free rule from development data and measures the gain contributed by the features.")
        primary = self.t(f"主要指标为{self.mlabel()}（{self.better()}），其余指标作为补充。", f"The primary metric was {self.mlabel()} ({self.better()}); other metrics are supplementary.")
        importance = self.t(f"特征作用以测试集置换重要性描述（每项特征重复置换 {s.get('importance_repeats')} 次，记录主要指标的平均下降）。",
                            f"Feature reliance was described by test-set permutation importance ({s.get('importance_repeats')} repeats per feature; mean decrease in the primary metric).")
        classes = s.get("classes") or []
        if s["task"] == "classification" and len(classes) == 2:
            importance = self.t(f"二分类中，AUC、平均精确率与概率校准均以{self.cls(classes[1])}作为“阳性”类别（按类别名排序后的第二类）。",
                                f"For binary outcomes, AUC, average precision and calibration treat {self.cls(classes[1])} as the positive class (second in sorted order). ") + importance
        none = self.t("本报告不进行假设检验，交叉验证的折间标准差和置换重复间标准差描述波动，而非置信区间。",
                      "No hypothesis tests are performed; between-fold and between-permutation SDs describe variability, not confidence intervals.")
        return [primary + self.t("", " ") + base_text, " ".join(uncertainty_lines) + importance + self.t("", " ") + none]

    # ------------------------------------------------------------ results prose
    def comparison_results(self):
        rows, sel = self.f["comparison"], self.f.get("selected")
        if not sel:
            return []
        out = []
        if len(rows) > 1:
            out.append(self.model_comparison_sentence(reading=False))
            reading = self.family_reading()
            if reading:
                out.append(reading if reading.endswith(("。", ".")) else reading + self.t("。", "."))
        positive = sel.get("baseline_gain_positive_folds")
        if positive is not None:
            text = self.t(f"选定模型在 {positive}/{sel['folds']} 个验证折中优于同折基线（平均改善 {fmt(sel.get('baseline_gain_mean'))}）",
                          f"The selected model beat the same-fold baseline in {positive}/{sel['folds']} folds (mean improvement {fmt(sel.get('baseline_gain_mean'))})")
            gap = sel.get("train_gap")
            if gap is not None:
                text += self.t(f"；训练与验证{self.mlabel()}平均相差 {fmt(gap)}", f"; training and validation {self.mlabel()} differed by {fmt(gap)} on average")
                if sel.get("mean") and abs(sel["mean"]) > 1e-12 and gap / abs(sel["mean"]) > 0.5:
                    text += self.t("，训练表现明显好于验证，存在过拟合迹象", ", a clear training advantage suggesting overfitting")
            out.append(text + self.t("。", "."))
        return out

    def test_results(self):
        name = self.model(self.s["selected_model"])
        text = self.t(f"在 {self.s['test_n']} 条测试记录上，选定的{name}{self.baseline_comparison(subject='')}。",
                      f"On {self.s['test_n']} test records, the selected {name} had {self.baseline_comparison(subject='')}.")
        secondary = self.secondary_test_clause()
        if secondary:
            text += self.t(f"补充指标：{secondary}。", f" Supplementary metrics: {secondary}.")
        return [text]

    # ------------------------------------------------------------ discussion
    def discussion(self):
        s, f = self.s, self.f
        paras = []
        main = self.conclusion()
        secondary = self.secondary_test_clause()
        if secondary:
            main += self.t(f"补充指标同样支持这一点：{secondary}。", f" Supplementary metrics agree: {secondary}.") if self.verdict() in ("supported", "supported_point") else \
                self.t(f"补充指标：{secondary}。", f" Supplementary metrics: {secondary}.")
        paras.append(main)
        reading = self.family_reading()
        if reading:
            paras.append(reading if reading.endswith(("。", ".")) else reading + self.t("。", "."))
        paras.append(self.strength_paragraph())
        flags = self.data_flags()
        if flags:
            paras.append(self.t("有几点数据特征会影响上述结论的解读：", "Several data characteristics qualify these conclusions: ") + self.t("", " ").join(flags))
        paras.append(self.limitations_paragraph())
        return paras

    def strength_paragraph(self):
        st = self.f["strength"]
        good, bad = [], []
        strengths = {
            "large_test": lambda v: self.t(f"测试样本量较大（n = {v}）", f"a reasonably large test sample (n = {v})"),
            "gain_interval": lambda v: self.t(f"相对基线改善的配对区间 {interval(*v)} 不包含零", f"a paired improvement interval ({interval(*v)}) excluding zero"),
            "clear_selection": lambda v: self.t("选定模型与其余候选的差距超过折间波动", "a selection margin exceeding between-fold variation"),
            "all_folds_beat_baseline": lambda v: self.t(f"选定模型在全部 {v} 个验证折中都优于基线", f"the selected model beating the baseline in all {v} folds"),
            "external_test": lambda v: self.t("测试数据来自另行提供的外部样本", "an externally supplied test sample"),
        }
        weaknesses = {
            "small_test": lambda v: self.t(f"测试样本仅 {v} 条，估计精度有限", f"only {v} test records, limiting precision"),
            "gain_interval": lambda v: self.t(f"相对基线改善的区间 {interval(*v)} 包含零或位于零以下", f"an improvement interval ({interval(*v)}) that does not exclude zero"),
            "wide_interval": lambda v: self.t(f"主要指标的 95% 区间宽度约为点估计的 {pct(v)}", f"a 95% interval about {pct(v)} as wide as the estimate"),
            "close_runner_up": lambda v: self.t(f"{join([self.model(m) for m in v], True)}与选定模型的差距在折间波动之内，模型排序并不稳定",
                                                f"a runner-up ({join([self.model(m) for m in v], False)}) within between-fold variation of the selected model, so the ranking is not stable"),
        }
        good = [strengths[code](value) for code, value in st["strengths"]]
        bad = [weaknesses[code](value) for code, value in st["weaknesses"]]
        text = ""
        if good:
            text += self.t(f"支持这一结论的证据包括：{'；'.join(good)}。", f"The conclusion is supported by {'; '.join(good)}. ")
        if bad:
            text += self.t(f"削弱其强度的因素包括：{'；'.join(bad)}。", f"Its strength is limited by {'; '.join(bad)}. ")
        text += self.t("此外，选定模型的交叉验证分数是在多个候选中取最优得到的，存在选择偏倚，因此以测试集结果为准。",
                       "In addition, the selected model's cross-validation score is the best of several candidates and therefore optimistic; the test result is the reference estimate.")
        return text

    def limitations_paragraph(self):
        s = self.s
        parts = []
        parts.append(self.t("测试集为内部留出样本，与开发集来自同一数据来源，结论能否推广到其他机构、人群或时间需要外部验证" if not s["external_test"] else
                            "测试数据另行提供，但推广到其他来源仍取决于其采集方式与人群构成",
                            "The test set is an internal holdout from the same source, so transfer to other sites, populations or periods requires external validation" if not s["external_test"] else
                            "The test data were supplied separately, but transfer to other sources still depends on how they were collected"))
        if s["split"] == "group":
            parts.append(self.t("分组划分避免了同组记录同时出现在训练与评价中，但组数较少时组间差异的估计不稳定",
                                "group-wise splitting prevents records of one group appearing in both training and evaluation, but estimates across few groups are unstable"))
        elif s["split"] == "time":
            parts.append(self.t("时间前向验证模拟了由过去预测未来，但相邻时间块并非独立，本报告未给出区间估计",
                                "forward-time validation mimics predicting the future from the past, but adjacent blocks are dependent and no intervals are given"))
        parts.append(self.t("置换重要性描述模型对特征的依赖，不代表因果效应；高度相关的特征可能相互替代",
                            "permutation importance describes the model's reliance, not causal effects, and correlated features can substitute for one another"))
        parts.append(self.t("流程内的预处理均在训练折内拟合，但无法纠正输入数据在进入本流程之前已经发生的信息泄漏",
                            "preprocessing is fitted within training folds, but this cannot undo leakage that occurred before the data entered the pipeline"))
        if self.s["task"] == "regression" and not self.ctx.get("tolerance"):
            parts.append(self.t("由于没有预先规定可接受误差，本报告不判断误差大小在实际应用中是否足够小",
                                "with no prespecified acceptable error, this report does not judge whether the error is small enough in practice"))
        return self.t("局限：" + "；".join(parts) + "。", "Limitations: " + "; ".join(parts) + ".")

    def next_steps(self):
        f, s = self.f, self.s
        steps = []
        if not s["external_test"]:
            steps.append(self.t("在独立来源（不同机构、批次或时间）的数据上评价已保存的 pipeline.joblib，以检验可推广性。",
                                "Evaluate the saved pipeline.joblib on data from an independent source (site, batch or period) to test transfer."))
        if any(code == "close_runner_up" for code, _ in f["strength"]["weaknesses"]):
            steps.append(self.t("如需在接近的候选间取舍，可在开发集内用更多折或重复交叉验证比较，不要依据本次测试结果重新排序。",
                                "To choose between close candidates, compare them on development data with more or repeated folds; do not re-rank on this test set."))
        sel = f.get("selected") or {}
        if sel.get("train_gap") is not None and sel.get("mean") and abs(sel["mean"]) > 1e-12 and sel["train_gap"] / abs(sel["mean"]) > 0.5:
            steps.append(self.t("选定模型存在过拟合迹象，可在开发集内比较正则化更强或更简单的配置（例如 --search grid）。",
                                "The selected model shows signs of overfitting; compare stronger regularization or simpler settings on development data (e.g. --search grid)."))
        imp = f.get("importance") or {}
        if imp.get("negligible"):
            steps.append(self.t(f"{join([self.feature(x) for x in imp['negligible']], True)}对预测几乎没有贡献；若考虑删除，应在开发集内重新比较，并用新的测试数据评价删减后的模型。",
                                f"{join([self.feature(x) for x in imp['negligible']], False)} contributed little; if removing them, re-compare on development data and evaluate on new test data."))
        shift = f.get("shift") or {}
        for item in (shift.get("missing") or [])[:1]:
            steps.append(self.t(f"核查{self.feature(item['feature'])}在测试集中缺失率变化的原因（采集流程、批次或设备）。",
                                f"Investigate why missingness of {self.feature(item['feature'])} changed in the test data (collection process, batch or device)."))
        cls = f.get("classification") or {}
        if cls.get("weakest") and cls["weakest"]["recall"] < 0.6:
            steps.append(self.t(f"{self.cls(cls['weakest']['class'])}的识别率偏低，可收集更多该类样本，或在开发集内比较类别重采样（--resample）。",
                                f"Recall for {self.cls(cls['weakest']['class'])} was low; collect more such cases or compare class resampling (--resample) on development data."))
        if cls.get("ece") is not None and cls["ece"] > 0.1:
            steps.append(self.t("若需要使用预测概率，应先在开发数据上进行概率校准，并在新数据上评价校准效果。",
                                "If predicted probabilities will be used, calibrate them on development data and evaluate calibration on new data."))
        if self.s["task"] == "regression" and not self.ctx.get("tolerance"):
            steps.append(self.t("结合领域知识确定可接受误差，用于判断模型在实际应用中是否足够准确。",
                                "Define an acceptable error from domain knowledge to judge practical adequacy."))
        return steps

    # ------------------------------------------------------------ per model
    def model_detail(self, row):
        s, sel = self.s, self.f.get("selected")
        name = self.model(row["model"])
        folds = row.get("fold_scores") or []
        rng = self.t(f"各折 {fmt(row.get('min'))}–{fmt(row.get('max'))}", f"folds {fmt(row.get('min'))}–{fmt(row.get('max'))}") if folds else ""
        text = self.t(f"验证{self.mlabel()}为 {fmt(row['mean'])} ± {fmt(row.get('sd'))}（{rng}，{row['folds']} 折）",
                      f"Validation {self.mlabel()} was {fmt(row['mean'])} ± {fmt(row.get('sd'))} ({rng}; {row['folds']} folds)")
        positive = row.get("baseline_gain_positive_folds")
        if positive is not None:
            text += self.t(f"，{positive}/{row['folds']} 折优于同折基线（平均改善 {fmt(row.get('baseline_gain_mean'))}）",
                           f"; it beat the same-fold baseline in {positive}/{row['folds']} folds (mean improvement {fmt(row.get('baseline_gain_mean'))})")
        if row.get("train_gap") is not None:
            text += self.t(f"，训练-验证差 {fmt(row['train_gap'])}", f"; training-validation difference {fmt(row['train_gap'])}")
        text += self.t("。", ". ")
        if row.get("train_gap") is not None and row.get("mean") and abs(row["mean"]) > 1e-12 and row["train_gap"] / abs(row["mean"]) > 0.5:
            text += self.t(f"该模型在训练折上的表现（平均 {fmt(row.get('train_mean'))}）远好于验证折，提示对训练数据过拟合。",
                           f"Its training-fold performance (mean {fmt(row.get('train_mean'))}) far exceeded validation, indicating overfitting. ")
        extra = {k: v for k, v in (row.get("secondary") or {}).items() if k != self.metric and k in METRIC_SHORT}
        if extra:
            text += self.t("其他验证指标均值：", "Other mean validation metrics: ") + \
                self.t("，", ", ").join(f"{self.mlabel(k)} {fmt(v)}" for k, v in extra.items()) + self.t("。", ". ")
        if row["standing"] in ("selected", "sole"):
            text += self.t("该模型被选为最终模型，其测试集表现见正文。", "This model was selected; its test performance is reported in the main text.")
        elif sel is not None:
            ratio = row.get("margin_sd")
            text += self.t(f"与选定的{self.model(sel['model'])}相比差 {fmt(row.get('margin'))}", f"It trailed the selected {self.model(sel['model'])} by {fmt(row.get('margin'))}")
            if ratio is not None:
                text += self.t(f"（约 {fmt(ratio, 2)} 个折间标准差）", f" (about {fmt(ratio, 2)} between-fold SDs)")
            text += self.t(f"，判断为“{STANDING[row['standing']][0]}”。", f", rated '{STANDING[row['standing']][1]}'. ")
            if row["standing"] == "near":
                text += self.t("差距处于折间波动范围内，不能认为该模型确实较差。", "The gap is within fold-to-fold variation, so it is not clearly worse.")
            text += self.t("按预设流程，候选模型只报告开发集验证结果，测试集仅用于评价选定模型。",
                           " By design only the selected model is evaluated on the test set.")
        return text.strip()
