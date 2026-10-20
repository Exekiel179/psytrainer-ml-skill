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
    "selected": ("最终模型", "final model"), "sole": ("最终模型（唯一）", "final model (only)"),
    "near": ("与最终模型相当", "comparable to final"), "weaker": ("较弱", "weaker"),
    "clearly_weaker": ("明显较弱", "clearly weaker"), "unranked": ("未排序", "unranked"),
}
BASELINES = {
    "mean": ("始终预测训练集目标变量均值", "always predicts the training outcome mean"),
    "median": ("始终预测训练集目标变量中位数", "always predicts the training outcome median"),
    "prior": ("按训练集类别比例给出概率、预测最多见类别", "uses training class frequencies and predicts the most frequent class"),
    "most_frequent": ("始终预测训练集最多见类别", "always predicts the most frequent training class"),
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
    if 0.995 <= abs(value) < 1:
        return f"{value:.3f}"
    if abs(value) >= 0.01:
        return f"{value:.2f}"
    if abs(value) >= 0.001:
        return f"{value:.3f}"
    return "0.00"


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


def apa_p(p):
    """APA p-value: no leading zero, three decimals, '< .001' floor."""
    if p is None:
        return "NA"
    return "p < .001" if p < .001 else "p = " + f"{p:.3f}".lstrip("0")


def stat(label, value, lo=None, hi=None):
    """APA-style 'MAE = 0.16, 95% CI [0.11, 0.21]'."""
    text = f"{label} = {fmt(value)}"
    ci = interval(lo, hi)
    return text + (f", 95% CI {ci}" if ci else "")


def interval(lo, hi):
    return f"[{fmt(lo)}, {fmt(hi)}]" if lo is not None and hi is not None else None


class Narrator:
    """Holds one run's facts and renders zh/en sentences from them."""

    def __init__(self, s, facts, zh):
        self.s, self.f, self.zh = s, facts, zh
        self.ctx = facts.get("context") or {}
        self.metric = {"f1": "f1_macro", "precision": "precision_macro", "recall": "recall_macro"}.get(s["metric"], s["metric"])
        self.basic = facts.get("level", "basic") == "basic"

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

    def baseline_rule(self):
        strategy = (self.s.get("baseline") or {}).get("strategy")
        rules = {
            "median": ("对每个样本都预测训练集目标变量的中位数", "always predicts the training-set median of the outcome"),
            "mean": ("对每个样本都预测训练集目标变量的均值", "always predicts the training-set mean of the outcome"),
            "prior": ("对每个样本都预测训练集中最多的类别", "always predicts the most frequent training-set class"),
            "most_frequent": ("对每个样本都预测训练集中最多的类别", "always predicts the most frequent training-set class"),
        }
        rule = rules.get(strategy, ("只使用训练集目标变量分布、不使用任何特征", "uses only the training outcome distribution, no features"))
        return rule[0 if self.zh else 1]

    def baseline_full(self):
        kind = "DummyRegressor" if self.s["task"] == "regression" else "DummyClassifier"
        return self.t(f"基线模型（{kind}，{self.baseline_rule()}）", f"the baseline ({kind}, which {self.baseline_rule()})")

    def importance_phrase(self):
        """What a permutation-importance value means for the primary metric."""
        if self.metric in LOSSES:
            return self.t(f"打乱该特征后测试集{self.mlabel()}的平均上升", f"mean increase in test {self.mlabel()} when the feature is shuffled")
        return self.t(f"打乱该特征后测试集{self.mlabel()}的平均下降", f"mean drop in test {self.mlabel()} when the feature is shuffled")

    def better(self):
        return self.t("越低越好", "lower is better") if self.s["direction"] == "lower" else self.t("越高越好", "higher is better")

    def overfit(self, row):
        """Large train-validation gap; not flagged for bagged trees, whose near-zero training error is by design."""
        gap, mean = row.get("train_gap"), row.get("mean")
        if gap is None or not mean or abs(mean) < 1e-12 or row.get("family") == "bagging":
            return False
        return gap / abs(mean) > 0.5

    # ------------------------------------------------------------ verdict
    def verdict(self):
        if self.basic:
            return "basic"
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
        return self.t(f"最重要的预测特征为{names}", f"the most important predictors were {names}")

    def primary_stat(self, key=None):
        row = self.f["tests"].get(key or self.metric) or (self.f["primary"] if key is None else {})
        return stat(self.mlabel(key), row.get("estimate"), row.get("lower"), row.get("upper"))

    def baseline_comparison(self, subject=None):
        """'MAE = 0.16, 95% CI [0.11, 0.21]; baseline MAE = 1.71' style clause."""
        p = self.f["primary"]
        if p.get("estimate") is None:
            return self.t(f"测试集{self.mlabel()}无法计算", f"test {self.mlabel()} was undefined")
        text = self.primary_stat()
        if p.get("baseline") is not None:
            text += self.t(f"；基线模型的{self.mlabel()} = {fmt(p['baseline'])}", f"; baseline {self.mlabel()} = {fmt(p['baseline'])}")
        return text

    def permutation_clause(self, short=False):
        test = self.s.get("permutation_test")
        if not test:
            return ""
        above = test["p_value"] < .05
        chance = f"M = {fmt(test['null_mean'])}, SD = {fmt(test['null_sd'])}"
        if short:
            return self.t(f"置换检验显示模型表现{'显著高于' if above else '未显著高于'}随机水平（{apa_p(test['p_value'])}）",
                          f"a permutation test showed performance {'significantly above' if above else 'not significantly above'} chance ({apa_p(test['p_value'])})")
        return self.t(f"置换检验显示，模型的测试集表现{'显著高于' if above else '未显著高于'}随机水平（{apa_p(test['p_value'])}，{test['valid']} 次置换；随机水平下的{self.mlabel()} {chance}）",
                      f"A permutation test showed that test performance was {'significantly above' if above else 'not significantly above'} chance ({apa_p(test['p_value'])}, {test['valid']} permutations; chance-level {self.mlabel()}: {chance})")

    def improvement_clause(self):
        p = self.f["primary"]
        if p.get("gain") is None:
            return ""
        diff = stat(self.t("差值", "difference"), p["gain"], p.get("gain_lower"), p.get("gain_upper"))
        rel = p.get("relative")
        if self.metric in LOSSES and rel is not None:
            return self.t(f"与基线模型相比，{self.mlabel()}降低了 {pct(rel)}（{diff}）", f"Compared with the baseline, {self.mlabel()} was {pct(rel)} lower ({diff})")
        ci = interval(p.get("gain_lower"), p.get("gain_upper"))
        tail = f", 95% CI {ci}" if ci else ""
        return self.t(f"与基线模型相比，{self.mlabel()}高出 {fmt(p['gain'])}{tail}", f"Compared with the baseline, {self.mlabel()} was {fmt(p['gain'])} higher{tail}")

    def secondary_test_clause(self):
        tests = self.f["tests"]
        bits = []
        if self.s["task"] == "regression":
            r2 = tests.get("r2", {})
            if r2.get("estimate") is not None and self.metric != "r2":
                bits.append(stat("R²", r2["estimate"], r2.get("lower"), r2.get("upper")))
            reg = self.f.get("regression") or {}
            if reg.get("within_tolerance") is not None:
                bits.append(self.t(f"绝对误差 ≤ {fmt(reg['tolerance'])}{self.unit}的测试样本占 {pct(reg['within_tolerance'])}",
                                   f"{pct(reg['within_tolerance'])} of test samples had absolute error ≤ {fmt(reg['tolerance'])}{self.unit}"))
        else:
            for key in ("roc_auc", "balanced_accuracy", "accuracy"):
                row = tests.get(key, {})
                if key != self.metric and row.get("estimate") is not None:
                    bits.append(stat(self.mlabel(key), row["estimate"], row.get("lower"), row.get("upper")))
                    if len(bits) == 2:
                        break
        return self.t("；".join(bits), "; ".join(bits))

    def difference_ci(self):
        p = self.f["primary"]
        return interval(p.get("gain_lower"), p.get("gain_upper"))

    def variance_clause(self):
        r2 = self.f["tests"].get("r2", {}).get("estimate")
        if self.s["task"] != "regression" or r2 is None or r2 <= 0:
            return ""
        return self.t(f"，可解释测试集中{self.target}约 {pct(r2)} 的变异", f", explaining about {pct(r2)} of the variance in {self.target}")

    def performance_clause(self):
        """Basic reports: test performance in plain numbers (primary metric, then R² or AUC)."""
        name = self.model(self.s["selected_model"])
        text = self.t(f"{name}在测试集上的{self.primary_stat()}", f"{name} achieved test {self.primary_stat()}")
        if self.s["task"] == "regression":
            r2 = self.f["tests"].get("r2", {}).get("estimate")
            if r2 is not None and self.metric != "r2":
                text += f", R² = {fmt(r2)}"
            text += self.variance_clause()
        else:
            for key in ("roc_auc", "accuracy"):
                value = self.f["tests"].get(key, {}).get("estimate")
                if value is not None and key != self.metric:
                    text += f", {self.mlabel(key)} = {fmt(value)}"
        return text

    def basic_answer(self):
        """Basic reports: answer the question from test performance against natural references."""
        target, name = self.target, self.model(self.s["selected_model"])
        test = self.s.get("permutation_test")
        sig = ""
        if test:
            sig = self.t(f"（置换检验 {apa_p(test['p_value'])}）", f" (permutation test {apa_p(test['p_value'])})")
            if test["p_value"] >= .05:
                return self.t(f"最终模型（{name}）的测试集表现未显著高于随机水平{sig}，现有特征对{target}的预测作用不明确",
                              f"Test performance of the final model ({name}) was not significantly above chance{sig}; the features' predictive value for {target} is unclear")
        if self.s["task"] == "regression":
            r2 = self.f["tests"].get("r2", {}).get("estimate")
            if r2 is None:
                return self.t(f"最终模型为{name}", f"The final model was {name}")
            if r2 <= 0:
                return self.t(f"现有特征未能预测{target}：最终模型（{name}）的测试集 R² = {fmt(r2)}，不如直接用平均值预测",
                              f"The features did not predict {target}: the final model ({name}) had test R² = {fmt(r2)}, no better than predicting the mean")
            return self.t(f"现有特征可以预测{target}{sig}：最终模型（{name}）可解释测试集中约 {pct(r2)} 的变异",
                          f"The features predicted {target}{sig}: the final model ({name}) explained about {pct(r2)} of the test-set variance")
        k = len(self.s.get("classes") or []) or 2
        value = self.f["primary"].get("estimate")
        if self.metric in ("balanced_accuracy", "recall_macro", "roc_auc") and value is not None:
            chance = 0.5 if self.metric == "roc_auc" else 1 / k
            relation = self.t("高于" if value > chance else "未高于", "above" if value > chance else "not above")
            return self.t(f"最终模型（{name}）的测试集{self.mlabel()} = {fmt(value)}，{relation}随机水平（{fmt(chance)}）{sig}",
                          f"The final model ({name}) had test {self.mlabel()} = {fmt(value)}, {relation} chance level ({fmt(chance)}){sig}")
        return self.t(f"最终模型（{name}）的测试集{self.primary_stat()}{sig}", f"The final model ({name}) achieved test {self.primary_stat()}{sig}")

    def summary_sentence(self):
        """Interpretive one-liner without statistics, for the abstract and discussion."""
        v, target, name = self.verdict(), self.target, self.model(self.s["selected_model"])
        if v == "basic":
            text = self.basic_answer()
            imp = self.f.get("importance") or {}
            if imp.get("leaders"):
                text += self.t(f"。预测主要依赖{join([self.feature(x) for x in imp['leaders']], True)}",
                               f". Predictions relied mainly on {join([self.feature(x) for x in imp['leaders']], False)}")
            if imp.get("negligible"):
                text += self.t(f"，{join([self.feature(x) for x in imp['negligible']], True)}几乎没有贡献",
                               f", while {join([self.feature(x) for x in imp['negligible']], False)} contributed little")
            return text + self.t("。", ".")
        text = {
            "supported": self.t(f"现有特征能够有效预测{target}，{name}在独立测试集上明显优于基线模型",
                                f"The available features predicted {target} well; {name} clearly outperformed the baseline on the held-out test set"),
            "supported_point": self.t(f"现有特征对{target}表现出一定的预测作用，但本次缺少区间估计",
                                      f"The features showed some predictive value for {target}, although no interval estimate was available"),
            "uncertain": self.t(f"现有特征对{target}的预测作用尚不明确，模型相对基线的优势不稳定",
                                f"The predictive value of the features for {target} remains unclear; the advantage over the baseline was not reliable"),
            "not_supported": self.t(f"现有特征未能有效预测{target}", f"The features did not predict {target} better than the baseline"),
        }.get(v, self.t(f"无法评估现有特征对{target}的预测作用", f"Predictive value for {target} could not be assessed"))
        if v in ("supported", "supported_point"):
            imp = self.f.get("importance") or {}
            leaders = imp.get("leaders") or []
            if leaders:
                text += self.t(f"，预测主要依赖{join([self.feature(x) for x in leaders], True)}",
                               f", relying mainly on {join([self.feature(x) for x in leaders], False)}")
            negligible = imp.get("negligible") or []
            if negligible:
                text += self.t(f"，{join([self.feature(x) for x in negligible], True)}几乎没有贡献",
                               f", while {join([self.feature(x) for x in negligible], False)} contributed little")
        return text + self.t("。", ".")

    def conclusion(self, leaders=True):
        v = self.verdict()
        target = self.target
        name = self.model(self.s["selected_model"])
        p = self.f["primary"]
        metric = self.mlabel()
        base = fmt(p.get("baseline"))
        diff = self.difference_ci()
        lead = self.leaders_text()
        if v == "basic":
            text = self.performance_clause()
            if self.permutation_clause():
                text += self.t("，", ", ") + self.permutation_clause(short=True)
        elif v == "supported":
            text = self.t(f"{name}对{target}具有较好的预测能力：测试集{self.primary_stat()}{self.variance_clause()}，优于基线模型（{metric} = {base}），且差值的 95% CI {diff} 不包含 0",
                          f"{name} predicted {target} on the held-out test set ({self.primary_stat()}){self.variance_clause()}, outperforming the baseline ({metric} = {base}); the 95% CI of the difference {diff} excluded 0")
        elif v == "supported_point":
            text = self.t(f"{name}在测试集上优于基线模型（{metric} = {fmt(p.get('estimate'))} vs. {base}），提示特征对{target}具有一定预测能力；本次未计算置信区间",
                          f"{name} outperformed the baseline on the test set ({metric} = {fmt(p.get('estimate'))} vs. {base}), suggesting some predictive ability for {target}; no confidence interval was computed")
        elif v == "uncertain":
            text = self.t(f"{name}在测试集上的{metric}（{fmt(p.get('estimate'))}）略优于基线模型（{base}），但差值的 95% CI {diff} 包含 0，尚不能认为模型对{target}具有稳定的预测能力",
                          f"{name} was slightly better than the baseline on test {metric} ({fmt(p.get('estimate'))} vs. {base}), but the 95% CI of the difference {diff} included 0, so reliable predictive ability for {target} was not established")
        elif v == "not_supported":
            text = self.t(f"{name}在测试集上未优于基线模型（{metric} = {fmt(p.get('estimate'))} vs. {base}），现有特征对{target}没有表现出预测能力",
                          f"{name} did not outperform the baseline on the test set ({metric} = {fmt(p.get('estimate'))} vs. {base}); the features showed no predictive ability for {target}")
        else:
            text = self.t(f"测试集主要指标或基线无法计算，无法评估模型对{target}的预测能力",
                          f"The primary test metric or baseline was undefined, so predictive ability for {target} cannot be assessed")
        if leaders and lead and v in ("supported", "supported_point", "basic"):
            text += self.t(f"；{lead}", f"; {lead}")
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
            f"训练集 {s['development_n']} 条、{source} {s['test_n']} 条，{s['feature_n']} 个特征。"
            f"在训练集上以 {s['cv_folds']} 折{design}交叉验证比较 {len(f['comparison'])} 个模型（{names}），"
            f"按{self.mlabel()}（{self.better()}）选择最终模型，" + ("再在测试集上评估。" if self.basic else f"再在测试集上与{self.baseline_full()}比较。"),
            f"{s['development_n']} training and {s['test_n']} {source} samples with {s['feature_n']} features. "
            f"{len(f['comparison'])} models ({names}) were compared by {s['cv_folds']}-fold {design} cross-validation on "
            f"{self.mlabel()} ({self.better()}); the selected model was then " + ("evaluated on the test set." if self.basic else f"compared on the test set with {self.baseline_full()}."))
        sel = f.get("selected") or {}
        result = self.t(f"{self.model(s['selected_model'])}的交叉验证表现最好（{self.mlabel()}: M = {fmt(sel.get('mean'))}, SD = {fmt(sel.get('sd'))}），被选为最终模型。",
                        f"{self.model(s['selected_model'])} performed best in cross-validation ({self.mlabel()}: M = {fmt(sel.get('mean'))}, SD = {fmt(sel.get('sd'))}) and was selected. ")
        if self.basic:
            result += self.performance_clause() + self.t("。", ". ")
        else:
            result += self.t(f"在测试集上，{self.baseline_comparison()}。", f"On the test set, {self.baseline_comparison()}. ")
        improvement = self.improvement_clause()
        if improvement:
            result += improvement + self.t("。", ". ")
        if self.permutation_clause():
            result += self.permutation_clause(short=True)[:1].upper() + self.permutation_clause(short=True)[1:] + self.t("。", ". ")
        secondary = "" if self.basic else self.secondary_test_clause()
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
                ("结论" if self.zh else "Conclusion", self.summary_sentence() + self.t("", " ") + caveat)]

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
        text = self.t(f"{self.s['cv_folds']} 折交叉验证结果显示，在 {len(rows)} 个候选模型中，{self.model(sel['model'])}表现最好（{self.mlabel()}: M = {fmt(sel['mean'])}, SD = {fmt(sel.get('sd'))}）；",
                      f"In {self.s['cv_folds']}-fold cross-validation, {self.model(sel['model'])} performed best among {len(rows)} candidates ({self.mlabel()}: M = {fmt(sel['mean'])}, SD = {fmt(sel.get('sd'))}); ")
        text += self.margin_clause(runner)
        short = self.family_reading(short=True) if reading else ""
        if short:
            text += self.t("。", ". ") + short
        return text + self.t("。", ".")

    def margin_clause(self, row):
        name = self.model(row["model"])
        text = self.t(f"其次为{name}（M = {fmt(row['mean'])}, SD = {fmt(row.get('sd'))}）",
                      f"the runner-up was {name} (M = {fmt(row['mean'])}, SD = {fmt(row.get('sd'))})")
        if row["standing"] == "near":
            text += self.t("，两者差距小于交叉验证标准差，可视为表现相当", "; the gap is smaller than the cross-validation SD, so they perform comparably")
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
                return self.t("线性模型优于各非线性模型，提示目标变量与特征的关系可由线性结构较好近似",
                              "The linear model outperformed all non-linear models, suggesting an approximately linear relationship")
            return self.t("线性模型优于各非线性模型，提示在当前样本量下，目标变量与特征的关系可以由加性线性结构较好地近似，"
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
        return self.t(f"{self.family(sel['family'])}（{self.model(sel['model'])}）优于线性模型，提示特征与目标变量之间可能存在非线性关系或交互作用；"
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
            return self.t("打乱任何单个特征都没有使测试表现变差，模型可能依赖可相互替代的特征，或未从特征中学到稳定信息。",
                          "Shuffling any single feature did not worsen test performance; the model may rely on interchangeable features or have learned little stable signal.")
        delta = f"Δ{self.mlabel()}"
        parts = [f"{self.feature(r['feature'])}（{delta} = {fmt(r['mean'])}, SD = {fmt(r['sd'])}）" if self.zh else
                 f"{self.feature(r['feature'])} ({delta} = {fmt(r['mean'])}, SD = {fmt(r['sd'])})" for r in top]
        if len(parts) == 1:
            text = self.t(f"最重要的预测特征为{parts[0]}", f"The most important predictor was {parts[0]}")
        else:
            text = self.t(f"最重要的预测特征为{parts[0]}，其次为{'、'.join(parts[1:])}",
                          f"The most important predictor was {parts[0]}, followed by {join(parts[1:], False)}")
        unstable = [r for r in top if not r["stable"]]
        if unstable and len(unstable) == len(top):
            text += self.t("。这些值的标准差与均值相当，特征排序不稳定",
                           ". Their SDs are comparable to the means, so the ranking is not stable")
        elif unstable:
            text += self.t(f"。其中{join([self.feature(r['feature']) for r in unstable], True)}的标准差较大，排序不稳定",
                           f". {join([self.feature(r['feature']) for r in unstable], False)} had large SDs, so their ranking is not stable")
        negligible = imp.get("negligible") or []
        if negligible:
            text += self.t(f"。{join([self.feature(x) for x in negligible], True)}的置换重要性接近 0，模型基本不依赖{'该特征' if len(negligible) == 1 else '这些特征'}",
                           f". {join([self.feature(x) for x in negligible], False)} had permutation importance near 0, so the model barely used {'it' if len(negligible) == 1 else 'them'}")
        pairs = imp.get("correlated_pairs") or []
        if pairs and not short:
            pair = pairs[0]
            text += self.t(f"。{self.feature(pair['a'])}与{self.feature(pair['b'])}高度相关（Spearman ρ = {fmt(pair['rho'], 2)}），二者可相互替代，单独打乱其中一个会低估它们的共同作用",
                           f". {self.feature(pair['a'])} and {self.feature(pair['b'])} were highly correlated (Spearman ρ = {fmt(pair['rho'], 2)}); shuffling either alone understates their joint contribution")
        return text + self.t("。", ".")

    def error_sentence(self, short=False):
        if self.s["task"] == "regression":
            reg = self.f.get("regression") or {}
            if not reg:
                return ""
            text = self.t(f"测试集绝对误差的中位数为 {fmt(reg['median_ae'])}{self.unit}，第 90 百分位数为 {fmt(reg['p90_ae'])}{self.unit}，最大值为 {fmt(reg['max_ae'])}{self.unit}",
                          f"The median absolute test error was {fmt(reg['median_ae'])}{self.unit} (90th percentile {fmt(reg['p90_ae'])}{self.unit}, maximum {fmt(reg['max_ae'])}{self.unit})")
            sd = reg.get("target_sd_test")
            if sd:
                text += self.t(f"（作为参照，测试集目标变量 SD = {fmt(sd)}）", f"; for reference, the test-set SD of {self.target} was {fmt(sd)}")
            bias = self.f["tests"].get("bias", {})
            lo, hi = bias.get("lower"), bias.get("upper")
            if lo is not None and hi is not None and (lo > 0 or hi < 0):
                text += self.t(f"。平均残差（实际值 − 预测值）= {fmt(bias['estimate'])}, 95% CI {interval(lo, hi)}，表明模型整体{'低估' if lo > 0 else '高估'}",
                               f". The mean residual (observed − predicted) was {fmt(bias['estimate'])}, 95% CI {interval(lo, hi)}, indicating systematic {'under' if lo > 0 else 'over'}prediction")
            edge = reg.get("edge_bias")
            worst = reg.get("worst_bin")
            if not short and edge and worst and worst["n"] >= 10 and reg.get("best_bin_mae") and worst["mae"] > 1.5 * reg["best_bin_mae"]:
                text += self.t(f"。误差在预测值 {worst['range']} 区间最大（MAE {fmt(worst['mae'])}，n = {worst['n']}）",
                               f". Error was largest for predictions in {worst['range']} (MAE {fmt(worst['mae'])}, n = {worst['n']})")
            if not short and edge and edge["low"] < 0 < edge["high"] and min(abs(edge["low"]), abs(edge["high"])) > 0.5 * (reg["median_ae"] or 0):
                text += self.t("；最低一组预测平均偏高、最高一组平均偏低，预测向均值收缩，极端值可能被低估",
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
        text = self.t(f"预测概率的校准情况：ECE = {fmt(ece)}", f"Calibration of predicted probabilities: ECE = {fmt(ece)}")
        if brier is not None:
            text += f", Brier = {fmt(brier)}"
        text += self.t("（两者越接近 0，预测概率越可信）", " (values closer to 0 mean more trustworthy probabilities)")
        classes = self.s.get("classes") or []
        if ece > 0.1 and direction in ("over", "under") and len(classes) == 2:
            pos, neg = self.cls(classes[1]), self.cls(classes[0])
            high, low = (pos, neg) if direction == "over" else (neg, pos)
            text += self.t(f"。模型给出的“属于{high}”的概率整体偏高（相应地，“属于{low}”的概率偏低）；若要直接使用预测概率，需要先做概率校准",
                           f". The predicted probability of {high} was too high overall (and that of {low} too low); recalibrate before using probabilities directly")
        return text + self.t("。", ".")

    def data_flags(self):
        if self.basic:
            return []
        shift = self.f.get("shift") or {}
        flags = []
        for item in (shift.get("missing") or [])[:2]:
            flags.append(self.t(f"{self.feature(item['feature'])}在测试集中的缺失率（{pct(item['test'])}）与训练集（{pct(item['development'])}）明显不同，测试表现可能部分反映缺失值填补的影响。",
                                f"{self.feature(item['feature'])} was missing in {pct(item['test'])} of test records versus {pct(item['development'])} in training, so test results partly reflect imputation."))
        shifted = shift.get("mean_shift") or []
        if shifted:
            names = join([f"{self.feature(x['feature'])}（{fmt(x['shift'], 2)} SD）" if self.zh else
                          f"{self.feature(x['feature'])} ({fmt(x['shift'], 2)} SD)" for x in shifted[:3]], self.zh)
            flags.append(self.t(f"测试集与训练集在{names}上的均值相差至少半个训练集标准差，两者分布并不完全一致。",
                                f"Test and training means differed by at least half a training SD for {names}; the partitions are not fully comparable."))
        ranged = shift.get("out_of_range") or []
        if ranged:
            item = ranged[0]
            flags.append(self.t(f"{pct(item['fraction'])} 的测试记录在{self.feature(item['feature'])}上超出训练集取值范围，模型对这些记录是外推预测。",
                                f"{pct(item['fraction'])} of test records fell outside the training range of {self.feature(item['feature'])}, so they were extrapolated."))
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
        text = self.t(f"共 {s['development_n'] + s['test_n']} 条记录：训练集 {s['development_n']} 条，测试集 {s['test_n']} 条；预测变量 {s['feature_n']} 项，目标变量为{self.target}",
                      f"The analysis used {s['development_n'] + s['test_n']} records: {s['development_n']} for training and {s['test_n']} for testing, with {s['feature_n']} predictors and outcome {self.target}")
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
                f"目标变量在训练集中 M = {fmt(dev['mean'])}, SD = {fmt(dev.get('sd'))}（范围 {fmt(dev['min'])} 至 {fmt(dev['max'])}），"
                f"在测试集中 M = {fmt(test.get('mean'))}, SD = {fmt(test.get('sd'))}（范围 {fmt(test.get('min'))} 至 {fmt(test.get('max'))}）。",
                f"{self.target} had M = {fmt(dev['mean'])}, SD = {fmt(dev.get('sd'))} (range {fmt(dev['min'])} to {fmt(dev['max'])}) in the training set "
                f"and M = {fmt(test.get('mean'))}, SD = {fmt(test.get('sd'))} (range {fmt(test.get('min'))} to {fmt(test.get('max'))}) in the test set."))
        elif dev.get("counts"):
            def counts(entry):
                total = sum(entry.values()) or 1
                return join([f"{self.cls(k)} {v}（{pct(v / total)}）" if self.zh else f"{self.cls(k)} {v} ({pct(v / total)})"
                             for k, v in entry.items()], self.zh)
            out.append(self.t(f"训练集类别构成：{counts(dev['counts'])}；测试集：{counts(test.get('counts') or {})}。",
                              f"Training classes: {counts(dev['counts'])}; test: {counts(test.get('counts') or {})}."))
        shift = self.f.get("shift") or {}
        if dev.get("complete_rows") is not None:
            if shift.get("any_missing"):
                out.append(self.t(f"训练集 {pct(dev['complete_rows'])}、测试集 {pct(test.get('complete_rows'))} 的记录特征完整。",
                                  f"{pct(dev['complete_rows'])} of training and {pct(test.get('complete_rows'))} of test records had complete predictors."))
            else:
                out.append(self.t("所有记录的特征均无缺失。", "No predictor values were missing."))
        return out

    def selection_methods(self):
        s, rows = self.s, self.f["comparison"]
        items = join([f"{self.model(r['model'])}（{self.family(r['family'])}）" if self.zh else
                      f"{self.model(r['model'])} ({self.family(r['family'])})" for r in rows], self.zh)
        text = self.t(f"比较的候选模型为{items}。各模型使用相同的交叉验证划分，以验证{self.mlabel()}均值最优者为最终模型，随后在完整训练集上重新拟合。",
                      f"Candidates were {items}. All shared the same cross-validation splits; the best mean validation {self.mlabel()} selected the final model, which was refitted on all training data.")
        if s.get("failures"):
            failed = join([self.model(x.get("model", str(x))) if isinstance(x, dict) else str(x) for x in s["failures"]], self.zh)
            text += self.t(f"{failed}未能完成训练，未参与比较（原因见 run.log 与 summary.json）。",
                           f" {failed} failed to train and were excluded (see run.log and summary.json).")
        return text

    def evaluation_methods(self, uncertainty_lines):
        s = self.s
        if self.basic:
            text = self.t(f"模型选择以 {s['cv_folds']} 折交叉验证的平均{self.mlabel()}为准（{self.better()}），结果写作 M ± SD。最终模型在测试集上评估一次。",
                          f"Models were selected by mean {s['cv_folds']}-fold cross-validated {self.mlabel()} ({self.better()}), reported as M ± SD. The final model was evaluated once on the test set.")
            out = [text]
            classes = s.get("classes") or []
            if s["task"] == "classification" and len(classes) == 2:
                out.append(self.t(f"二分类中以{self.cls(classes[1])}作为正类（AUC 等指标针对该类计算）。",
                                  f"{self.cls(classes[1])} was the positive class for AUC and related metrics."))
            if s.get("permutation_test"):
                test = s["permutation_test"]
                out.append(self.t(f"另以置换检验（{test['valid']} 次）检验测试集表现是否高于随机水平。",
                                  f"A permutation test ({test['valid']} permutations) assessed whether test performance exceeded chance."))
            return [" ".join(out) if not self.zh else "".join(out)]
        primary = self.t(f"模型选择以 {s['cv_folds']} 折交叉验证的平均{self.mlabel()}为准（{self.better()}），结果写作“均值 ± 标准差”（标准差为各折之间的波动）。其余指标作为补充。",
                         f"Models were selected by mean {s['cv_folds']}-fold cross-validated {self.mlabel()} ({self.better()}), reported as mean ± SD across folds. Other metrics are supplementary.")
        base_text = self.t(f"为判断特征是否有用，设置{self.baseline_full()}作为参照：它不读取任何特征，只用训练集目标变量本身做预测。模型只有明显优于它，才说明特征提供了预测信息。",
                           f"To judge whether the features help, {self.baseline_full()} serves as reference: it reads no features and predicts from the training outcome alone. Only a model that clearly beats it shows the features carry predictive information.")
        ci = self.t("测试集指标的 95% 置信区间（95% CI）用 bootstrap 估计：从测试集中有放回地重复抽样、每次重新计算指标，取第 2.5 和第 97.5 百分位数，反映测试样本有限带来的不确定性。"
                    "“与基线的差值”在同一批 bootstrap 样本上计算模型与基线的指标之差，其 95% CI 不包含 0 即说明模型稳定优于基线。",
                    "95% confidence intervals (95% CI) for test metrics come from bootstrap resampling of the test set (resample with replacement, recompute the metric, take the 2.5th and 97.5th percentiles), reflecting uncertainty from the finite test sample. "
                    "The difference from the baseline is computed on the same bootstrap samples; a 95% CI excluding 0 means the model reliably beats the baseline. ")
        importance = self.t(f"特征重要性采用测试集置换重要性（permutation importance）：将某一特征的取值随机打乱、其他特征不变，记录{self.mlabel()}的变化，重复 {s.get('importance_repeats')} 次取均值 ± 标准差。数值越大，模型越依赖该特征；接近 0 表示几乎不依赖。",
                            f"Feature importance is test-set permutation importance: shuffle one feature, keep the others, record the change in {self.mlabel()}, and repeat {s.get('importance_repeats')} times (mean ± SD). Larger values mean stronger reliance; values near 0 mean little reliance. ")
        classes = s.get("classes") or []
        extra = []
        if s["task"] == "classification" and len(classes) == 2:
            chosen = (s.get("arguments") or {}).get("positive_class")
            how = self.t("由研究者指定" if chosen else "默认取类别名排序后的第二类；可用 --positive-class 指定",
                         "specified by the researcher" if chosen else "by default the second class in sorted order; set with --positive-class")
            extra.append(self.t(f"二分类中，以{self.cls(classes[1])}作为正类（{how}），AUC、平均精确率、校准曲线和阈值分析均针对该类计算。",
                                f"For the binary outcome, {self.cls(classes[1])} was the positive class ({how}); AUC, average precision, calibration and threshold analyses refer to it."))
        test = s.get("permutation_test")
        if test:
            extra.append(self.t(
                f"为检验模型表现是否高于随机水平，进行了置换检验（permutation test）：将训练集的目标变量随机打乱，使特征与目标变量之间不再存在真实联系，"
                f"再用与最终模型相同的配置重新训练，并在同一测试集上计算{self.mlabel()}；重复 {test['valid']} 次得到随机水平下的成绩分布。"
                f"p 值为打乱后成绩不差于实际成绩的比例，即（次数 + 1）/（置换次数 + 1）。",
                f"To test whether performance exceeded chance, a permutation test was run: training labels were randomly shuffled (removing any real association with the features), "
                f"the final configuration was refitted, and {self.mlabel()} was computed on the same test set; {test['valid']} repetitions gave the chance distribution. "
                f"The p-value is the proportion of permuted scores at least as good as the observed score, (count + 1)/(permutations + 1)."))
        interval_info = uncertainty_lines if isinstance(uncertainty_lines, dict) else {}
        if interval_info.get("omitted_reason"):
            detail = self.t("本次未计算置信区间：" + {
                "Time dependence requires a justified block length; row bootstrap is omitted.": "时间序列样本之间不独立，不适合逐样本 bootstrap。",
                "Fewer than five test groups; cluster intervals are omitted.": "测试集的组数少于 5 个，按组 bootstrap 不可靠。",
            }.get(interval_info["omitted_reason"], "该运行未启用 bootstrap。"), "No confidence intervals were computed in this run: " + {
                "Time dependence requires a justified block length; row bootstrap is omitted.": "time-ordered samples are not independent, so sample-wise bootstrap is unsuitable.",
                "Fewer than five test groups; cluster intervals are omitted.": "the test set has fewer than five groups, so group bootstrap is unreliable.",
            }.get(interval_info["omitted_reason"], "bootstrap was disabled."))
        elif interval_info:
            unit = self.t("按组整体抽样" if "cluster" in str(interval_info.get("method")) else "按样本抽样", "resampling whole groups" if "cluster" in str(interval_info.get("method")) else "resampling samples")
            detail = self.t(f"本次{unit}，重复 {interval_info.get('requested_repeats')} 次。", f"This run used {unit}, {interval_info.get('requested_repeats')} repeats.")
        else:
            detail = ""
        if interval_info.get("omitted_reason") or not interval_info:
            ci = ""
        closing = "" if s.get("permutation_test") else self.t("报告不做显著性检验。", "No significance tests are performed.")
        return [primary + self.t("", " ") + base_text, ci + detail, importance + closing, *extra]

    # ------------------------------------------------------------ results prose
    def comparison_results(self):
        rows, sel = self.f["comparison"], self.f.get("selected")
        if not sel:
            return []
        out = []
        if len(rows) > 1:
            out.append(self.model_comparison_sentence(reading=False))
        bits = []
        positive = sel.get("baseline_gain_positive_folds")
        if positive is not None:
            bits.append(self.t(f"交叉验证中，最终模型在 {positive}/{sel['folds']} 折都优于基线模型",
                               f"In cross-validation the final model beat the baseline in {positive}/{sel['folds']} folds"))
        gap = sel.get("train_gap")
        if gap is not None and sel.get("train_mean") is not None:
            if sel.get("family") == "bagging" and gap / max(abs(sel.get("mean") or 0), 1e-12) > 0.5:
                bits.append(self.t(f"最终模型在训练折上几乎完全拟合（训练集{self.mlabel()} = {fmt(sel['train_mean'])}），这是树集成模型的正常特性；其泛化能力以交叉验证与测试集结果为准",
                                   f"the final model fits the training folds almost perfectly (train {self.mlabel()} = {fmt(sel['train_mean'])}), which is normal for tree ensembles; generalisation is judged from cross-validation and test results"))
            else:
                overfit = self.overfit(sel)
                bits.append(self.t(f"最终模型的训练集{self.mlabel()}（{fmt(sel['train_mean'])}）与 CV {self.mlabel()}（{fmt(sel['mean'])}）" +
                                   ("差距明显，存在过拟合迹象" if overfit else "接近，没有明显过拟合"),
                                   f"the final model's train {self.mlabel()} ({fmt(sel['train_mean'])}) and CV {self.mlabel()} ({fmt(sel['mean'])}) " +
                                   ("differ markedly, suggesting overfitting" if overfit else "are close, with no clear overfitting")))
        if bits:
            text = self.t("；", "; ").join(bits)
            out.append((text if self.zh else text[:1].upper() + text[1:]) + self.t("。", "."))
        return out

    def test_results(self):
        name = self.model(self.s["selected_model"])
        if self.basic:
            text = self.t(f"测试集（n = {self.s['test_n']}）结果：", f"Test set (n = {self.s['test_n']}): ") + \
                self.performance_clause().replace("在测试集上的", "的").replace("achieved test ", "achieved ") + self.t("。", ".")
            if self.permutation_clause():
                text += self.permutation_clause() + self.t("。", ".")
            return [text]
        text = self.t(f"在测试集（n = {self.s['test_n']}）上，{name}的{self.baseline_comparison()}。",
                      f"On the test set (n = {self.s['test_n']}), {name} achieved {self.baseline_comparison()}.")
        improvement = self.improvement_clause()
        if improvement:
            text += improvement + self.t("。", ". ")
        if self.permutation_clause():
            text += self.permutation_clause() + self.t("。", ". ")
        secondary = self.secondary_test_clause()
        if secondary:
            text += self.t(f"补充指标：{secondary}。", f" Supplementary metrics: {secondary}.")
        return [text]

    # ------------------------------------------------------------ discussion
    def discussion(self):
        s, f = self.s, self.f
        paras = []
        main = self.t("总体而言，", "Overall, ") + (self.summary_sentence() if self.zh else self.summary_sentence()[:1].lower() + self.summary_sentence()[1:])
        if self.basic:
            main = self.summary_sentence()
        secondary = "" if (self.variance_clause() or self.basic) else self.secondary_test_clause()
        reg = self.f.get("regression") or {}
        if self.variance_clause() and reg.get("within_tolerance") is not None:
            secondary = self.t(f"绝对误差 ≤ {fmt(reg['tolerance'])}{self.unit}的测试样本占 {pct(reg['within_tolerance'])}",
                               f"{pct(reg['within_tolerance'])} of test samples had absolute error ≤ {fmt(reg['tolerance'])}{self.unit}")
        if secondary:
            main += self.t(f"在测试集上，{secondary}。", f" On the test set, {secondary}.")
        paras.append(main)
        reading = self.family_reading()
        if reading:
            paras.append(reading if reading.endswith(("。", ".")) else reading + self.t("。", "."))
        paras.append(self.strength_paragraph())
        flags = self.data_flags()
        if flags:
            lead = (self.t("需要注意的是，", "Note that ") if len(flags) == 1 else
                    self.t("以下数据特征会影响上述结论的解读：", "Several data characteristics qualify these conclusions: "))
            body = self.t("", " ").join(flags)
            paras.append(lead + (body if self.zh or len(flags) > 1 else body[:1].lower() + body[1:]))
        paras.append(self.limitations_paragraph())
        return paras

    def strength_paragraph(self):
        st = self.f["strength"]
        good, bad = [], []
        strengths = {
            "large_test": lambda v: self.t(f"测试样本量较大（n = {v}）", f"a reasonably large test sample (n = {v})"),
            "gain_interval": lambda v: self.t(f"与基线差值的 95% CI {interval(*v)} 不包含 0", f"a 95% CI of the difference from baseline {interval(*v)} excluding 0"),
            "clear_selection": lambda v: self.t("最终模型领先其余模型的幅度大于交叉验证标准差", "a lead over other models larger than the cross-validation SD"),
            "all_folds_beat_baseline": lambda v: self.t(f"交叉验证的全部 {v} 折中，最终模型都优于基线", f"the selected model beating the baseline in all {v} cross-validation folds"),
            "external_test": lambda v: self.t("测试数据来自另行提供的外部样本", "an externally supplied test sample"),
            "permutation": lambda v: self.t(f"置换检验显示模型显著高于随机水平（{apa_p(v)}）", f"performance significantly above chance in a permutation test ({apa_p(v)})"),
        }
        weaknesses = {
            "small_test": lambda v: self.t(f"测试集只有 {v} 个样本，估计精度有限", f"only {v} test samples, limiting precision"),
            "gain_interval": lambda v: self.t(f"与基线差值的 95% CI {interval(*v)} 包含 0", f"a 95% CI of the difference from baseline {interval(*v)} that includes 0"),
            "wide_interval": lambda v: self.t(f"主要指标的 95% CI 较宽（宽度约为点估计的 {pct(v)}）", f"a wide 95% CI (about {pct(v)} of the estimate)"),
            "permutation": lambda v: self.t(f"置换检验未显示模型显著高于随机水平（{apa_p(v)}）", f"a permutation test not showing performance above chance ({apa_p(v)})"),
            "close_runner_up": lambda v: self.t(f"{join([self.model(m) for m in v], True)}与最终模型的差距小于交叉验证标准差，模型排序不稳定",
                                                f"a runner-up ({join([self.model(m) for m in v], False)}) within cross-validation variability of the selected model, so the ranking is not stable"),
        }
        good = [strengths[code](value) for code, value in st["strengths"]]
        bad = [weaknesses[code](value) for code, value in st["weaknesses"]]
        text = ""
        if self.basic:
            notes = bad
            if notes:
                text += self.t(f"解读时需要注意：{'；'.join(notes)}。", f"Note that: {'; '.join(notes)}. ")
            return text + self.t("最终模型是按交叉验证得分选出的，其交叉验证得分偏乐观，模型表现以测试集结果为准。",
                                 "The final model was chosen for its cross-validation score, which is therefore optimistic; the test-set result is the estimate to rely on.")
        if good:
            text += self.t(f"这一结论有较充分的依据：{'；'.join(good)}。", f"This conclusion is well supported: {'; '.join(good)}. ")
        if bad:
            text += self.t(f"但解读时需要注意：{'；'.join(bad)}。", f"However, {'; '.join(bad)}. ")
        text += self.t("另外，最终模型是从多个候选中按交叉验证得分选出的，其交叉验证得分会偏乐观，因此模型表现应以测试集结果为准。",
                       "Also, the final model was chosen for its cross-validation score among several candidates, so that score is optimistic; the test-set result is the estimate to rely on.")
        return text

    def limitations_paragraph(self):
        s = self.s
        parts = []
        parts.append(self.t("测试集为内部留出样本，与训练集来自同一数据来源，结论能否推广到其他机构、人群或时间需要外部验证" if not s["external_test"] else
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
            steps.append(self.t("如需在接近的候选间取舍，可在训练集内用更多折或重复交叉验证比较，不要依据本次测试结果重新排序。",
                                "To choose between close candidates, compare them on training data with more or repeated folds; do not re-rank on this test set."))
        sel = f.get("selected") or {}
        if self.overfit(sel):
            steps.append(self.t("最终模型存在过拟合迹象，可在训练集内比较正则化更强或更简单的配置（例如 --search grid）。",
                                "The selected model shows signs of overfitting; compare stronger regularization or simpler settings on training data (e.g. --search grid)."))
        imp = f.get("importance") or {}
        if imp.get("negligible"):
            steps.append(self.t(f"{join([self.feature(x) for x in imp['negligible']], True)}对预测几乎没有贡献；若考虑删除，应在训练集内重新比较，并用新的测试数据评价删减后的模型。",
                                f"{join([self.feature(x) for x in imp['negligible']], False)} contributed little; if removing them, re-compare on training data and evaluate on new test data."))
        shift = f.get("shift") or {}
        for item in (shift.get("missing") or [])[:1]:
            steps.append(self.t(f"核查{self.feature(item['feature'])}在测试集中缺失率变化的原因（采集流程、批次或设备）。",
                                f"Investigate why missingness of {self.feature(item['feature'])} changed in the test data (collection process, batch or device)."))
        cls = f.get("classification") or {}
        if cls.get("weakest") and cls["weakest"]["recall"] < 0.6:
            steps.append(self.t(f"{self.cls(cls['weakest']['class'])}的识别率偏低，可收集更多该类样本，或在训练集内比较类别重采样（--resample）。",
                                f"Recall for {self.cls(cls['weakest']['class'])} was low; collect more such cases or compare class resampling (--resample) on training data."))
        if cls.get("ece") is not None and cls["ece"] > 0.1:
            steps.append(self.t("若需要使用预测概率，应先在训练集上进行概率校准，并在新数据上评价校准效果。",
                                "If predicted probabilities will be used, calibrate them on training data and evaluate calibration on new data."))
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
        text = self.t(f"交叉验证{self.mlabel()}为 {fmt(row['mean'])} ± {fmt(row.get('sd'))}（{rng}，共 {row['folds']} 折）",
                      f"Cross-validated {self.mlabel()} was {fmt(row['mean'])} ± {fmt(row.get('sd'))} ({rng}; {row['folds']} folds)")
        positive = row.get("baseline_gain_positive_folds")
        if positive is not None:
            text += self.t(f"，{positive}/{row['folds']} 折优于基线（平均领先 {fmt(row.get('baseline_gain_mean'))}）",
                           f"; it beat the baseline in {positive}/{row['folds']} folds (mean lead {fmt(row.get('baseline_gain_mean'))})")
        if row.get("train_gap") is not None:
            text += self.t(f"，训练分与验证分平均相差 {fmt(row['train_gap'])}", f"; training and validation scores differed by {fmt(row['train_gap'])} on average")
        text += self.t("。", ". ")
        if self.overfit(row):
            text += self.t(f"该模型在训练折上的表现（平均 {fmt(row.get('train_mean'))}）远好于验证折，提示对训练数据过拟合。",
                           f"Its training-fold performance (mean {fmt(row.get('train_mean'))}) far exceeded validation, indicating overfitting. ")
        extra = {k: v for k, v in (row.get("secondary") or {}).items() if k != self.metric and k in METRIC_SHORT}
        if extra:
            text += self.t("其他指标的交叉验证均值：", "Other cross-validated metrics (mean): ") + \
                self.t("，", ", ").join(f"{self.mlabel(k)} {fmt(v)}" for k, v in extra.items()) + self.t("。", ". ")
        if row["standing"] in ("selected", "sole"):
            text += self.t("该模型被选为最终模型，其测试集表现见正文。", "This model was selected; its test performance is reported in the main text.")
        elif sel is not None:
            text += self.t(f"比选定的{self.model(sel['model'])}差 {fmt(row.get('margin'))}，判断为“{STANDING[row['standing']][0]}”。",
                           f"It trailed the selected {self.model(sel['model'])} by {fmt(row.get('margin'))} (rated '{STANDING[row['standing']][1]}'). ")
            if row["standing"] == "near":
                text += self.t("差距小于交叉验证标准差，两者表现相当。", "The gap is smaller than the cross-validation SD, so the two perform comparably.")
            text += self.t("按标准流程，非最终模型只报告交叉验证结果，测试集只用于评估最终模型。",
                           " By design only the selected model is evaluated on the test set.")
        return text.strip()
