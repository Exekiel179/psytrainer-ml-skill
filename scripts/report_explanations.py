"""Academic report prose and accessible definitions grounded in measured results."""

from __future__ import annotations

import json


METRICS = {
    "mae": ("平均绝对误差 MAE", "每个预测与真实值相差多少，取绝对值后求平均；越小越好，单位与目标相同。它不是每个样本的最大误差。",
            "Mean absolute error", "Average absolute distance from the observed outcome. Lower is better, in outcome units; it is not a maximum error."),
    "mse": ("均方误差 MSE", "先将每个误差平方再平均，因此特别重视大错误；越小越好，单位是目标单位的平方，不能直接读成平均差多少分。",
            "Mean squared error", "Average squared error, giving large mistakes extra weight. Lower is better; units are squared outcome units, not the original scale."),
    "rmse": ("均方根误差 RMSE", "对均方误差开平方，将结果还原为目标单位；越小越好。它比 MAE 更容易被少量大错误拉高。",
             "Root mean squared error", "Square root of MSE, returning to outcome units. Lower is better; a few large mistakes affect it more than MAE."),
    "r2": ("决定系数 R²", "比较模型误差与始终预测测试集平均值时的误差。1 表示完全吻合，0 表示在此标准下没有改善，负值表示更差；它不是预测正确率，也不是因果解释比例。这里的均值参照与图中只在开发集拟合的简单基线不同。",
           "R squared", "Compares squared errors with predicting the test outcome mean. One is perfect, zero gives no improvement under that reference, and negative is worse. It is not accuracy or causal explanation. This reference differs from the plotted baseline fitted only on development data."),
    "pearson_r": ("Pearson 相关系数", "衡量真实值与预测值是否一起升降，范围为 -1 到 1，越接近 1 表示正向线性一致性越强。即使相关很高，预测仍可能整体偏高或偏低，所以需同时看 MAE 和误差图。常量序列时无法计算。",
                  "Pearson correlation", "Measures linear co-movement from -1 to 1. A value near 1 can coexist with systematic over- or underprediction; also inspect absolute errors. Undefined for a constant series."),
    "accuracy": ("准确率", "所有测试样本中预测类别正确的比例；越高越好。某类别占绝大多数时，只猜该类别也能获得高准确率，因此必须同时看各类召回率。",
                 "Accuracy", "Fraction of test samples classified correctly. Higher is better, but a dominant class can make a trivial classifier look good; also inspect per-class recall."),
    "balanced_accuracy": ("平衡准确率", "先计算每个真实类别被正确找回的比例，再对类别等权平均；越高越好，不让大类别单凭人数主导结果。它不是总体正确人数比例；缺少某个目标类别时完整指标无法计算。",
                          "Balanced accuracy", "Equal-weight average of recall across classes. Higher is better and large classes do not dominate solely by count. It is not the overall fraction correct; a missing target class prevents the full metric."),
    "precision_macro": ("宏平均精确率", "对每个类别，先问被模型判为这一类的样本中有多少真的属于它，再对类别等权平均；越高越好。它强调误报，需与召回率一起看。",
                        "Macro precision", "For each class, asks how many predictions of that class are correct, then averages equally across classes. Higher is better; compare with recall to understand missed cases."),
    "recall_macro": ("宏平均召回率", "对每个类别，先问实际属于这一类的样本中有多少被找回来，再对类别等权平均；越高越好。它强调漏检；高召回不保证误报少。",
                     "Macro recall", "For each true class, asks how many members are retrieved, then averages equally across classes. Higher is better, but high recall need not mean few false alarms."),
    "f1_macro": ("宏平均 F1", "先对各类精确率与召回率求调和平均，再对类别等权平均。范围 0 到 1，越高越好；任一方面很低都会拉低 F1，但它仍不能代替具体误报和漏检成本。",
                 "Macro F1", "Computes each class's harmonic mean of precision and recall, then averages across classes. Higher is better on a 0–1 scale; it does not encode real error costs."),
    "roc_auc": ("ROC 曲线下面积 AUC", "衡量二分类模型把阳性排在阴性前面的能力，越高越好；0.5 是随机排序参照。它评估排序而非某个阈值下的准确率，也不能说明输出概率可信。",
                "ROC AUC", "Measures how well a binary model ranks positives ahead of negatives. Higher is better; 0.5 is a chance-ranking reference. It is neither threshold-specific accuracy nor probability reliability."),
    "average_precision": ("平均精确率 AP", "概括检出更多阳性时能维持怎样的精确率，越高越好。与测试集阳性比例一起读，不能直接与阳性比例不同的数据集比较。",
                          "Average precision", "Summarizes precision as recall increases. Higher is better; interpret against positive prevalence and avoid direct comparisons across differing prevalence."),
    "brier": ("Brier 概率误差", "将预测阳性概率与实际结果（阳性 1、阴性 0）的差平方后求平均；越小越好。它同时受到排序、校准和阳性比例影响，并非纯粹的校准分数。",
              "Brier score", "Mean squared distance between positive probabilities and binary outcomes. Lower is better; it reflects discrimination, calibration and prevalence, not calibration alone."),
    "bias": ("平均残差", "真实值减预测值后求平均。正值表示总体偏低估，负值表示总体偏高估；接近零也可能是正负大误差相互抵消。",
             "Mean residual", "Average observed minus predicted outcome. Positive indicates underprediction, negative overprediction; a near-zero average can hide cancelling large errors."),
}


def metric_explanation(key, value, zh):
    key = {"f1": "f1_macro", "precision": "precision_macro", "recall": "recall_macro"}.get(key, key)
    spec = METRICS.get(key)
    title, description = (spec[:2] if zh else spec[2:]) if spec else (key, "")
    result = ("本次未能计算；不能当作零或性能良好。" if zh else "Undefined in this run; do not treat it as zero or good performance.") if value is None else (
        f"本次测试值为 {value:.4g}。" if zh else f"The test value is {value:.4g}. ")
    if value is not None and key == "mae":
        result += (f"在该测试集中，预测值与观测值的平均绝对偏差为 {value:.4g} 个目标单位；其应用意义取决于预先规定的容许误差。" if zh else
                   f"Predictions differ from observations by {value:.4g} outcome units on average here; acceptability requires an application-specific tolerance chosen in advance.")
    if value is not None and key == "accuracy":
        result += (f"即该测试集中 {value:.1%} 的样本类别判断正确。" if zh else
                   f"This corresponds to {value:.1%} correct in this sample, not a guarantee for future batches.")
    return title, description + " " + result


def introduction(s, zh):
    if zh:
        return [
            f"分析以 {s['target']} 为结局变量，考察输入特征对该结局的预测能力。特征是模型可用的输入信息，结局是需要预测的观测结果；输入表每行对应一条样本记录。",
            ("回归任务估计连续数值，采用原始结局单位下的误差及预测与观测的一致性评价结果。" if s['task'] == 'regression' else
             f"分类任务估计样本的类别归属，类别编码为 {', '.join(map(str, s['classes']))}。评价同时考虑总体表现及各类别的误判情况；编码的领域含义沿用原始数据定义。"),
        ]
    return [
        f"The analysis evaluates whether input features predict the observed outcome {s['target']}. Features are information available to the model; the outcome is the quantity or category to be predicted. Each input row represents one sample record.",
        ("Regression estimates a continuous outcome. Evaluation considers errors in outcome units and agreement between predicted and observed values." if s['task'] == 'regression' else
         f"Classification estimates category membership, using labels {', '.join(map(str, s['classes']))}. Evaluation considers overall performance and class-specific errors; label meanings follow the source data."),
    ]


def model_name(tag, zh):
    names = {
        'ModelAdaBoostClassifier': ('AdaBoost 分类器', 'AdaBoost classifier'),
        'ModelAdaBoostRegressor': ('AdaBoost 回归器', 'AdaBoost regressor'),
        'ModelBaggingRegressor': ('Bagging 回归器', 'bagging regressor'),
        'ModelCatBoostRegressor': ('CatBoost 回归器', 'CatBoost regressor'),
        'ModelDecisionTreesClassifier': ('决策树分类器', 'decision tree classifier'),
        'ModelExtraTreeRegressor': ('Extra Trees 回归器', 'extra trees regressor'),
        'ModelGPRRegressor': ('高斯过程回归器', 'Gaussian process regressor'),
        'ModelGradientBoostingRegressor': ('梯度提升回归器', 'gradient boosting regressor'),
        'ModelKNeighborsClassifier': ('K 近邻分类器', 'k-nearest neighbors classifier'),
        'ModelKNeighborsRegressor': ('K 近邻回归器', 'k-nearest neighbors regressor'),
        'ModelLGBMClassifier': ('LightGBM 分类器', 'LightGBM classifier'),
        'ModelLGBMRegressor': ('LightGBM 回归器', 'LightGBM regressor'),
        'ModelLRClassifier': ('逻辑回归', 'logistic regression'),
        'ModelLRRegressor': ('线性回归', 'linear regression'),
        'ModelNaiveBayesClassifier': ('朴素贝叶斯分类器', 'naive Bayes classifier'),
        'ModelRandomForestClassifier': ('随机森林分类器', 'random forest classifier'),
        'ModelRandomForestRegressor': ('随机森林回归器', 'random forest regressor'),
        'ModelSVCClassifier': ('支持向量分类器', 'support vector classifier'),
        'ModelSVRRegressor': ('支持向量回归器', 'support vector regressor'),
        'ModelXGBoostClassifier': ('XGBoost 分类器', 'XGBoost classifier'),
        'ModelXGBoostRegressor': ('XGBoost 回归器', 'XGBoost regressor'),
    }
    return names.get(tag, (tag, tag))[0 if zh else 1]


# When at least this many models succeed, write one report file per model
# plus an overview technical report that indexes them.
PER_MODEL_SPLIT_THRESHOLD = 4

# Relative validation gap vs selected model (oriented so larger = worse).
_NEAR_GAP = 0.10
_CLEAR_GAP = 0.50


def primary_metric(s):
    return {'f1': 'f1_macro', 'precision': 'precision_macro', 'recall': 'recall_macro'}.get(s['metric'], s['metric'])


def selected_comparison_row(s):
    selected = s.get("selected_model")
    if not selected:
        return None
    for row in s.get("comparison") or []:
        if row["model"] == selected:
            return row
    return None


def relative_validation_gap(row_mean, selected_mean, direction):
    """How much worse ``row_mean`` is than the selected mean; 0 means tied or better."""
    scale = max(abs(selected_mean), 1e-12)
    raw = (row_mean - selected_mean) if direction == "lower" else (selected_mean - row_mean)
    return max(0.0, raw / scale)


def model_standing(row, s):
    """Return standing code: selected | near | weaker | clearly_weaker | sole."""
    comparison = s.get("comparison") or []
    if len(comparison) <= 1:
        return "sole" if row["model"] == s["selected_model"] else "weaker"
    if row["model"] == s["selected_model"]:
        return "selected"
    chosen = selected_comparison_row(s)
    if chosen is None:
        return "weaker"
    gap = relative_validation_gap(row["mean"], chosen["mean"], s["direction"])
    if gap <= _NEAR_GAP:
        return "near"
    if gap <= _CLEAR_GAP:
        return "weaker"
    return "clearly_weaker"


def model_role_short(row, s, zh):
    """Compact role label for index/appendix tables."""
    standing = model_standing(row, s)
    labels = {
        "selected": ("选定", "selected"),
        "sole": ("选定", "selected"),
        "near": ("接近选定", "near selected"),
        "weaker": ("弱于选定", "weaker"),
        "clearly_weaker": ("明显更差", "clearly weaker"),
    }
    return labels[standing][0 if zh else 1]


def selection_claim(s, zh):
    """One-sentence claim: which model is preferred in this run."""
    chosen = selected_comparison_row(s)
    if not chosen:
        return ""
    name = model_name(chosen["model"], zh)
    n = len(s.get("comparison") or [])
    if zh:
        if n <= 1:
            return f"主张：在本次可完成验证的配置中，{name} 是唯一可用模型，因而进入测试评价。"
        return f"主张：在共享开发集验证下，{name} 优于本次其余候选，故选定为最终模型。"
    if n <= 1:
        return f"Claim: {name} is the only configuration with usable validation and therefore proceeds to test evaluation."
    return f"Claim: under shared development validation, {name} outperforms the other candidates and is selected."


def selection_evidence(s, zh):
    """Evidence package for the selection claim (validation numbers only)."""
    comparison = s.get("comparison") or []
    chosen = selected_comparison_row(s)
    if not chosen:
        return ""
    name = model_name(chosen["model"], zh)
    metric = s["metric"]
    better = ("越低越好" if s["direction"] == "lower" else "越高越好") if zh else (
        "lower is better" if s["direction"] == "lower" else "higher is better")
    if len(comparison) == 1:
        return (f"证据：{name} 的开发集 {metric} 均值为 {chosen['mean']:.4g}，折间 SD 为 {chosen['sd']:.4g}"
                f"（指标方向：{better}）。" if zh else
                f"Evidence: {name} had development {metric} mean {chosen['mean']:.4g} and between-fold SD "
                f"{chosen['sd']:.4g} ({better}).")
    peers = [row for row in comparison if row["model"] != chosen["model"]]
    runner = min(peers, key=lambda row: relative_validation_gap(row["mean"], chosen["mean"], s["direction"]))
    gap = relative_validation_gap(runner["mean"], chosen["mean"], s["direction"])
    runner_name = model_name(runner["model"], zh)
    stability = ""
    if chosen["sd"] <= min(row["sd"] for row in peers) + 1e-12:
        stability = ("折间波动在候选中亦属较小一侧；" if zh else
                     "Between-fold SD was among the smaller values; ")
    elif chosen["sd"] > max(row["sd"] for row in peers):
        stability = ("折间波动高于部分候选；" if zh else
                     "Between-fold SD exceeded some candidates; ")
    if gap <= 1e-9:
        margin = ("与次优候选验证均值实质持平，按主指标均值规则选定。" if zh else
                  "validation means tied with the runner-up; the primary-metric mean rule selected it.")
    elif gap <= _NEAR_GAP:
        margin = (f"相对次优候选 {runner_name}（{metric}={runner['mean']:.4g}）优势有限（约 {gap:.1%}）。"
                  if zh else
                  f"the margin over runner-up {runner_name} ({metric}={runner['mean']:.4g}) is about {gap:.1%}.")
    else:
        margin = (f"相对次优候选 {runner_name}（{metric}={runner['mean']:.4g}）改善约 {gap:.1%}。"
                  if zh else
                  f"it improves on runner-up {runner_name} ({metric}={runner['mean']:.4g}) by about {gap:.1%}.")
    if zh:
        return (f"证据：主指标 {metric}（{better}）上，{name} 验证均值 {chosen['mean']:.4g}、"
                f"折间 SD {chosen['sd']:.4g}。{stability}{margin}")
    return (f"Evidence: on primary {metric} ({better}), {name} had mean {chosen['mean']:.4g} and "
            f"between-fold SD {chosen['sd']:.4g}. {stability}{margin}")


def selection_interpretation(s, zh):
    """What the selection evidence does and does not mean."""
    if zh:
        return ("阐释：该选择只回答「在本次共享划分与预指定主指标下何者更好」，"
                "不是测试集假设检验，也不声称跨样本或因果意义上的普遍最优。")
    return ("Interpretation: selection answers which candidate is better under this shared split and "
            "prespecified primary metric; it is not a held-out hypothesis test and does not claim "
            "universal or causal superiority.")


def selection_rationale(s, zh):
    """Claim + evidence + interpretation for model selection (joined for compact call sites)."""
    parts = [selection_claim(s, zh), selection_evidence(s, zh), selection_interpretation(s, zh)]
    return "".join(p for p in parts if p)


def model_result_paragraphs(row, s, fold_scores, search_params, zh):
    """Per-model section: claim → evidence → interpretation."""
    tag = row["model"]
    name = model_name(tag, zh)
    standing = model_standing(row, s)
    role = {
        "selected": ("最终选定模型", "selected model"),
        "sole": ("最终选定模型", "selected model"),
        "near": ("接近选定的候选", "near-selected candidate"),
        "weaker": ("弱于选定的候选", "weaker candidate"),
        "clearly_weaker": ("明显更差的候选", "clearly weaker candidate"),
    }[standing][0 if zh else 1]
    folds = ", ".join(f"{v:.4g}" for v in fold_scores) if fold_scores else ("无" if zh else "none")
    lines = []
    # Claim
    if zh:
        lines.append(f"主张：{name}（标识 {tag}）在本次比较中的角色是{role}。")
    else:
        lines.append(f"Claim: {name} (id {tag}) stands as the {role} in this comparison.")
    # Evidence
    if zh:
        lines.append(
            f"证据：在共享的 {row['folds']} 折开发集验证中，{s['metric']} 均值为 {row['mean']:.4g}，"
            f"折间样本标准差为 {row['sd']:.4g}；各折得分依次为 {folds}。"
        )
    else:
        lines.append(
            f"Evidence: across {row['folds']} shared development folds, mean {s['metric']} was {row['mean']:.4g} "
            f"with between-fold sample SD {row['sd']:.4g}; fold scores were {folds}."
        )
    if search_params:
        lines.append(("证据（续）：参数搜索采用配置 " if zh else "Evidence (continued): search configuration ") +
                     json.dumps(search_params, ensure_ascii=not zh) + ("。" if zh else "."))
    chosen = selected_comparison_row(s)
    # Interpretation (and comparative evidence for the selected model)
    if standing in ("selected", "sole"):
        if len(s.get("comparison") or []) > 1:
            lines.append(selection_evidence(s, zh))
        lines.append(selection_interpretation(s, zh))
        scores = "; ".join(
            f"{k}={v:.4g}" if v is not None else f"{k}=undefined"
            for k, v in s["test_metrics"].items())
        if zh:
            lines.append(
                f"阐释（测试）：仅该模型在留出测试集上完成正式评价（n={s['test_n']}）：{scores}。"
                "测试数字用于评价已选配置，不得用来在候选间二次挑选。"
            )
        else:
            lines.append(
                f"Interpretation (test): only this model received formal held-out evaluation "
                f"(n={s['test_n']}): {scores}. Test figures evaluate the chosen configuration and "
                "must not re-rank candidates."
            )
    elif chosen is not None:
        gap = relative_validation_gap(row["mean"], chosen["mean"], s["direction"])
        chosen_name = model_name(chosen["model"], zh)
        if zh:
            if standing == "near":
                lines.append(
                    f"阐释：相对选定模型 {chosen_name}（验证 {s['metric']}={chosen['mean']:.4g}），"
                    f"差距约 {gap:.1%}，尚属接近；接近不等于可互换，亦不得用本次测试集重新排序。"
                )
            elif standing == "clearly_weaker":
                lines.append(
                    f"阐释：相对选定模型 {chosen_name}（验证 {s['metric']}={chosen['mean']:.4g}），"
                    f"主指标差距约 {gap:.1%}，属于明显更差一侧；在当前验证设计下不宜作为首选替代。"
                )
            else:
                lines.append(
                    f"阐释：相对选定模型 {chosen_name}（验证 {s['metric']}={chosen['mean']:.4g}），"
                    f"主指标差距约 {gap:.1%}，弱于选定；分节保留验证数字供对照，不作测试集补评。"
                )
        else:
            if standing == "near":
                lines.append(
                    f"Interpretation: relative to selected {chosen_name} "
                    f"(CV {s['metric']}={chosen['mean']:.4g}), the gap is about {gap:.1%} (near). "
                    "Near does not mean interchangeable; do not re-rank on this test set."
                )
            elif standing == "clearly_weaker":
                lines.append(
                    f"Interpretation: relative to selected {chosen_name} "
                    f"(CV {s['metric']}={chosen['mean']:.4g}), the gap is about {gap:.1%}, "
                    "placing this candidate clearly weaker under the present design."
                )
            else:
                lines.append(
                    f"Interpretation: relative to selected {chosen_name} "
                    f"(CV {s['metric']}={chosen['mean']:.4g}), the gap is about {gap:.1%}. "
                    "Validation figures are retained for contrast without held-out re-evaluation."
                )
    else:
        lines.append(
            "阐释：该候选仅报告开发集验证；留出测试专用于最终选定模型。" if zh else
            "Interpretation: this candidate reports development validation only; the holdout evaluates the selected model."
        )
    return lines


def abstract(s, zh):
    """Abstract as claim → evidence → interpretation (problem/method folded into claim context)."""
    key = primary_metric(s)
    title = metric_explanation(key, None, zh)[0]
    value = s['test_metrics'].get(key)
    score = f'{value:.4g}' if value is not None else ('未能计算' if zh else 'undefined')
    source = ('外部测试集' if s['external_test'] else '内部留出测试集') if zh else (
        'external test set' if s['external_test'] else 'internal holdout')
    design = {'random': ('随机', 'random'), 'group': ('分组', 'grouped'), 'time': ('时间', 'temporal')}[s['split']][0 if zh else 1]
    name = model_name(s['selected_model'], zh)
    claim = selection_claim(s, zh)
    evidence = selection_evidence(s, zh)
    if zh:
        setup = (f"问题与方法：以 {s['target']} 为预测结局，纳入 {s['feature_n']} 项特征；"
                 f"开发集 {s['development_n']} 条，{design}设计 {s['cv_folds']} 折交叉验证，"
                 f"以{title}比较候选。")
        test = (f"测试证据：在 {s['test_n']} 条记录的{source}上，选定模型 {name} 的{title}为 {score}。")
        closing = ("阐释：下文结果说明该测试表现相对基线意味着什么；讨论综合何者更优、证据强度与不可外推边界。"
                   "不作假设检验推断或因果断言。")
        return setup + claim + evidence + test + closing
    setup = (f"Problem and method: predict {s['target']} from {s['feature_n']} features on "
             f"{s['development_n']} development records using {s['cv_folds']}-fold {design} CV with {title}.")
    test = (f"Test evidence: on the {source} of {s['test_n']} records, selected {name} had {title}={score}.")
    closing = ("Interpretation: results explain the baseline comparison; discussion weighs preference, "
               "evidence strength and non-extrapolation. No hypothesis-test or causal claims.")
    return " ".join(p for p in (setup, claim, evidence, test, closing) if p)


def validation_methods(s, zh):
    designs = {
        'random': ('开发集内部采用分层随机交叉验证。' if s['task'] == 'classification' else '开发集内部采用随机 K 折交叉验证。',
                   'Development cross-validation was stratified by class.' if s['task'] == 'classification' else 'Development data used shuffled K-fold cross-validation.'),
        'group': ('开发集与测试集的组别互不重叠，开发集内部采用 GroupKFold，验证折同样按组隔离。组标识仅用于划分，不作为预测特征。',
                  'Development and test groups are disjoint. GroupKFold also separates groups within development data; group identifiers are used only for splitting.'),
        'time': (f"样本按唯一时间值划分为有序块，采用前向交叉验证；训练记录早于验证记录，同一时间值不跨越划分边界。训练末端排除 {s['gap']} 个时间块作为间隔。",
                 f"Forward cross-validation uses ordered unique time blocks. Training precedes validation, equal timestamps remain together, and a gap of {s['gap']} blocks excludes the end of each training partition."),
    }
    source = ('另行提供的外部数据' if s['external_test'] else '模型选择前从输入数据划出的内部留出集') if zh else (
        'separately supplied external data' if s['external_test'] else 'an internal holdout reserved before model selection')
    lines = [f"测试集来源为{source}。" if zh else f"The test set comprises {source}.", designs[s['split']][0 if zh else 1]]
    lines.append((f"开发集的 {s['cv_folds']} 折验证用于比较候选模型，每折在训练部分拟合模型，在未参与该折拟合的记录上计算验证指标。"
                  "选定配置后，在完整开发集重新拟合，再评价测试集。" if zh else
                  f"The {s['cv_folds']} development folds compare candidates by fitting on each training partition and scoring records excluded from that fit. The selected configuration is refitted on all development records before test evaluation."))
    lines.append(f"随机过程使用种子 {s['seed']}。" if zh else f"Stochastic operations use seed {s['seed']}.")
    return lines


def result_summary(s, intervals, zh):
    """Primary test result as claim → evidence → interpretation (no significance threshold)."""
    key = primary_metric(s)
    label = metric_explanation(key, None, zh)[0]
    spec = METRICS.get(key)
    meaning = (spec[1] if zh else spec[3]) if spec else ""
    row = next((r for r in intervals if r['metric'] == key), None)
    name = model_name(s.get("selected_model", ""), zh) if s.get("selected_model") else (
        "选定模型" if zh else "the selected model")
    if row is None or row.get('estimate') is None:
        return (f"主张：测试集{label}未能计算，现有证据不足以评价主要终点。" if zh else
                f"Claim: test {label} is undefined, so the primary endpoint cannot be evaluated.")
    estimate, baseline, gain = row['estimate'], row.get('baseline'), row.get('gain')
    # Claim
    if baseline is not None and gain is not None:
        if zh:
            if gain > 0:
                claim = f"主张：在本测试样本上，{name} 相对简单基线表现出可观测改善（主指标 {label}）。"
            elif gain < 0:
                claim = f"主张：在本测试样本上，{name} 未表现出相对简单基线的改善（主指标 {label}）。"
            else:
                claim = f"主张：在本测试样本上，{name} 与简单基线在主指标 {label} 上持平。"
        else:
            if gain > 0:
                claim = f"Claim: on this test sample, {name} shows an observable improvement over the dummy baseline on {label}."
            elif gain < 0:
                claim = f"Claim: on this test sample, {name} does not improve on the dummy baseline on {label}."
            else:
                claim = f"Claim: on this test sample, {name} matches the dummy baseline on {label}."
    else:
        claim = (f"主张：选定模型在测试集上的主终点为 {label}={estimate:.4g}。" if zh else
                 f"Claim: the selected model's primary test endpoint is {label}={estimate:.4g}.")
    # Evidence
    evidence = (f"证据：测试集{label}点估计为 {estimate:.4g}" if zh else
                f"Evidence: test {label} point estimate was {estimate:.4g}")
    if row.get('lower') is not None and row.get('upper') is not None:
        evidence += f" (95% CI [{row['lower']:.4g}, {row['upper']:.4g}])"
    evidence += '。' if zh else '. '
    if baseline is not None and gain is not None:
        relation = ('高于' if gain > 0 else '低于' if gain < 0 else '等于') if s['direction'] == 'higher' else (
            '低于' if gain > 0 else '高于' if gain < 0 else '等于')
        evidence += (f"基线为 {baseline:.4g}，模型指标{relation}基线，按优劣方向的差值为 {gain:.4g}。" if zh else
                     f"Baseline was {baseline:.4g}; improvement-oriented difference was {gain:.4g}. ")
        lo, hi = row.get('gain_lower'), row.get('gain_upper')
        if lo is not None and hi is not None:
            evidence += (f"配对差值 95% CI 为 [{lo:.4g}, {hi:.4g}]，" if zh else
                         f"Paired 95% CI was [{lo:.4g}, {hi:.4g}], ")
            evidence += (("区间包含零，方向仍有不确定性。" if lo <= 0 <= hi else
                          "区间位于零以上。" if lo > 0 else "区间位于零以下。") if zh else
                         ("including zero (direction uncertain). " if lo <= 0 <= hi else
                          "lying above zero. " if lo > 0 else "lying below zero. "))
    elif key == 'pearson_r':
        evidence += ("常量预测基线的 Pearson 相关系数无法定义，故未计算该指标的基线差值。" if zh else
                     "Pearson correlation is undefined for the constant baseline, so no baseline difference is reported.")
    # Interpretation
    if zh:
        interp = (f"阐释：{label}的含义是：{meaning}"
                  "增益是否「够用」取决于预先规定的容许误差，本报告不作裁定；"
                  "点估计与区间只描述固定已选模型在本测试样本上的表现，不能外推为跨机构或前瞻保证，"
                  "也不构成因果或假设检验推断。")
    else:
        interp = (f"Interpretation: {label}: {meaning} "
                  "Whether the gain is 'enough' requires a prespecified tolerance; "
                  "estimates describe this fitted model on this test sample only—not cross-site guarantees, "
                  "causation, or hypothesis tests.")
    return claim + evidence + interp


def uncertainty_description(interval, zh):
    reason = interval.get('omitted_reason')
    if reason:
        reasons = {
            'Time dependence requires a justified block length; row bootstrap is omitted.':
                '时间记录存在依赖，区间估计需预先确定有依据的块长度；本次未进行将记录视为独立观测的 bootstrap 重采样。',
            'Bootstrap disabled or unavailable in this older run.':
                '该运行未启用 bootstrap，或其历史结果不包含所需记录。',
            'Fewer than five test groups; cluster intervals are omitted.':
                '测试组数少于五个，未计算整组重采样区间。',
        }
        return (["本次仅报告点估计，未计算置信区间。" + reasons.get(reason, reason)] if zh else
                ["This run reports point estimates without confidence intervals. " + reason])
    method = {'row percentile bootstrap': '按记录重采样的百分位 bootstrap',
              'cluster percentile bootstrap (resample whole test groups)': '按完整测试组重采样的百分位 bootstrap'}.get(interval['method'], interval['method'])
    if zh:
        return [
            "置信区间（confidence interval，CI）描述性能估计的抽样不确定性。Bootstrap 通过有放回地抽取测试单位并重复计算指标，近似描述估计的波动；区间越宽，估计精度越低。该区间与单个样本的预测区间不同。",
            f"采用{method}，请求重复 {interval['requested_repeats']} 次，重采样单位数为 {interval['units']}。报告固定模型下的 95% 百分位区间，不涵盖重新训练、模型选择及采集设计的不确定性。仅在有效重复至少为 100 次且达到请求次数的 80% 时报告区间。",
        ]
    return [
        "Confidence intervals (CIs) describe sampling uncertainty in performance estimates. Bootstrap draws test units with replacement and recomputes metrics; wider intervals indicate lower precision. These differ from individual prediction intervals.",
        f"The {interval['method']} requests {interval['requested_repeats']} repeats across {interval['units']} units. The 95% percentile intervals condition on the fitted model and exclude retraining, selection and sampling-design uncertainty. Intervals require at least 100 valid repeats and 80% of requested repeats.",
    ]


def discussion(s, analysis, zh):
    """Discussion: preference claim → supporting evidence → scope interpretation."""
    lines = []
    if s.get("selected_model") and s.get("comparison") is not None:
        claim = selection_claim(s, zh)
        if claim:
            lines.append(claim)
        evidence = selection_evidence(s, zh)
        if evidence:
            lines.append(evidence)
        comparison = s.get("comparison") or []
        if len(comparison) > 1:
            standings = [(model_role_short(row, s, zh), model_name(row["model"], zh))
                         for row in comparison if row["model"] != s["selected_model"]]
            near = [name for role, name in standings if role in ("接近选定", "near selected")]
            weak = [name for role, name in standings if role in ("明显更差", "clearly weaker")]
            if zh:
                bits = []
                if near:
                    bits.append("与 " + "、".join(near) + " 较为接近")
                if weak:
                    bits.append("相对 " + "、".join(weak) + " 优势更清楚")
                if bits:
                    lines.append("证据（候选对照）：" + "；".join(bits) +
                                 "。接近并不等于可互换，仍以预指定主指标与折间表现为准。")
                else:
                    lines.append("证据（候选对照）：其余候选弱于选定但未跨过大幅度阈值；"
                                 "分节保留验证数字，避免只读总表。")
            else:
                bits = []
                if near:
                    bits.append("near " + ", ".join(near))
                if weak:
                    bits.append("clearly ahead of " + ", ".join(weak))
                if bits:
                    lines.append("Evidence (candidates): selected model is " + " and ".join(bits) +
                                 ". Near does not mean interchangeable.")
                else:
                    lines.append("Evidence (candidates): others are weaker without a large-gap threshold; "
                                 "per-model sections retain validation figures.")
        if s.get("metric") is not None:
            key = primary_metric(s)
            test_value = (s.get("test_metrics") or {}).get(key)
            if test_value is not None:
                label = metric_explanation(key, None, zh)[0]
                if zh:
                    lines.append(
                        f"证据（测试）：选定模型{label}={test_value:.4g}（n={s['test_n']}），"
                        "只用于评价已选配置，不能二次挑选候选。"
                    )
                else:
                    lines.append(
                        f"Evidence (test): selected {label}={test_value:.4g} (n={s['test_n']}), "
                        "evaluating the chosen configuration only—not for re-ranking."
                    )
        interp = selection_interpretation(s, zh)
        if interp:
            lines.append(interp)
        if zh:
            lines.append(
                f"阐释（证据强度）：测试集含 {s['test_n']} 条；最优验证分数存在选择偏倚；"
                "证据强度受样本量、划分设计与验证边际共同约束，不能解读为普遍最优或因果效应。"
                "实际应用需预先明确容许误差或误判成本。"
            )
        else:
            lines.append(
                f"Interpretation (strength): test n={s['test_n']}; the winning validation score is "
                "selection-biased; strength is bounded by sample size, split design and margin—"
                "not universal optimality or causation. Practical use needs prespecified tolerances."
            )
    if s.get('external_test'):
        lines.append("边界：测试使用另行提供的数据，结果反映该测试样本；跨机构或前瞻适用性取决于来源与采集设计。" if zh else
                     "Scope: separately supplied test data describe that sample; transfer depends on provenance.")
    else:
        lines.append("边界：评价基于内部留出，结论限于本次数据来源与划分；跨机构或前瞻应用需外部验证。" if zh else
                     "Scope: an internal holdout limits conclusions to this source and split; external validation is required for transfer.")
    if s.get('split') == 'group':
        lines.append("边界（分组）：组隔离减少同组泄漏乐观，但组间表现仍受样本量与构成影响。" if zh else
                     "Scope (group): group separation reduces shared-group optimism; between-group performance still depends on support.")
    elif s.get('split') == 'time':
        lines.append("边界（时间）：前向验证评价由早到晚的预测；相邻块可能依赖，本流程不做将记录视为独立的 bootstrap 区间。" if zh else
                     "Scope (time): forward validation predicts later from earlier; adjacent blocks may depend, so row bootstrap is omitted.")
    lines.append("边界（机制）：置换重要性描述预测依赖而非因果；相关特征可相互替代。" if zh else
                 "Scope (mechanism): permutation importance is predictive reliance, not a causal effect; correlated inputs can substitute.")
    if s.get('split') and s['split'] != 'random':
        lines.append("边界（诊断）：逐行置换可能破坏组/时间依赖，重要性仅作描述性诊断。" if zh else
                     "Scope (diagnostics): row-wise permutations can disrupt group/time dependence; treat importance as descriptive.")
    lines.append("边界（数据）：折内预处理不能消除上游全样本或未来信息泄漏；解释依赖输入来源记录。" if zh else
                 "Scope (data): fold-local preprocessing cannot undo upstream leakage; interpretation depends on feature provenance.")
    lines.append("后续：可在开发集比较正则化、特征或类别处理并记录验证结果；若改动由本次测试诊断驱动，须用新的独立测试数据评价。" if zh else
                 "Next steps: compare regularization, features or class handling on development data; test-motivated changes need new independent test data.")
    return lines


def preprocessing(s, zh):
    p = s['preprocessing']
    lines = []
    def add(cn, en):
        lines.append(cn if zh else en)
    if p.get('impute') == 'median':
        add("缺失预测变量以训练样本中对应特征的中位数填补，使不完整记录能够进入后续分析。填充值仅从训练部分估计。", "Missing predictors are replaced by the corresponding training-feature median, allowing incomplete records to enter the analysis. Replacement values are estimated from training data only.")
    else:
        add("本次未进行缺失值填补，分析要求预测变量完整。", "No imputation is applied; complete predictors are required.")
    if p.get('scale'):
        add(("数值缩放采用最小值和最大值归一化，将训练范围映射到 0–1。新样本超出训练范围时可以超过这个区间。" if p.get('scaler') == 'minmax' else "数值标准化将每项特征减去训练均值并除以训练标准差，使不同计量单位更容易共同参与建模。"),
            ("Min-max scaling maps the training range to 0–1; new values can fall outside it." if p.get('scaler') == 'minmax' else "Standardization subtracts training means and divides by training standard deviations, bringing differing measurement scales into a comparable range."))
    else:
        add("本次不进行数值缩放，特征保留原始尺度。", "Features retain their original scales; no scaling is applied.")
    selection = p.get('selection')
    if selection == 'forward':
        add(f"逐步前向筛选从较少特征开始，利用训练部分的内部验证逐项尝试加入信息。本次选择数量设置为 {p.get('select_k') or '自动'}；它只寻找当前搜索路径上的改善，不保证找到了所有可能组合中的最优解。", f"Forward selection adds features using validation within training data; requested count is {p.get('select_k') or 'automatic'}. This searches one path, not every possible subset.")
    elif selection == 'f-threshold':
        add(f"特征筛选保留 F 统计量不低于 {p.get('f_threshold')} 的输入。该分数评价单个预测变量与结局的关联，可能遗漏仅在变量组合中体现的信息。", f"F-threshold selection retains inputs at or above {p.get('f_threshold')}. It assesses individual associations and can miss interactions.")
    elif p.get('select_k'):
        add(f"根据单变量关联分数保留 {p['select_k']} 个特征。各特征分别评分，因此筛选结果侧重其单独关联，可能遗漏交互信息。", f"The top {p['select_k']} individually scored features are retained. This prioritizes marginal associations and can miss interactions.")
    else:
        add("本次不做特征筛选，全部输入特征进入后续步骤。", "No feature selection is applied; all inputs enter subsequent steps.")
    if p.get('pca'):
        add(f"主成分分析（PCA）将处理后的特征压缩为 {p['pca']} 个综合成分。各成分是原特征的加权组合，以保留输入变异为目标；这一降维准则与最大化结局预测能力并不相同。", f"Principal component analysis (PCA) compresses processed inputs into {p['pca']} weighted components. It preserves input variation, an objective distinct from maximizing outcome prediction.")
    if p.get('resample', 'none') != 'none':
        add(f"训练时采用 {p['resample']} 调整类别样本构成，以改变不同类别对模型拟合的贡献。重采样仅作用于训练部分，验证与测试保持原始样本构成。", f"Training uses {p['resample']} to alter class representation and its contribution to model fitting. Resampling is confined to training partitions; validation and test retain their original composition.")
    add("上述处理构成统一的 Pipeline：每折仅用训练部分估计预处理参数，并将其应用于该折验证数据。最终在完整开发集拟合的预处理与模型一并保存，用于测试和后续预测。", "These steps form one Pipeline. Preprocessing parameters are estimated within each training fold and applied to its validation data. The final development-fitted preprocessing and estimator are saved together for test evaluation and subsequent predictions.")
    return lines


# Each entry explains the exact axes and marks of a generated diagnostic family.
FIGURES = {
    'model-validation': (
        '候选模型的训练与交叉验证表现',
        'a 面板每行是一种模型，横轴是所选指标；灰点是训练得分，蓝点是同一折的验证得分，连线连接一对结果。先按横轴的 higher/lower 判断好坏，再比较灰蓝差距。b 面板把验证表现减去同折简单基线的表现，并统一成正值更好；绿点是平均增益，线段是折间标准差，表示不同划分下的波动。',
        '折间标准差描述划分间的波动，区别于均值估计的置信区间。',
        'Training and cross-validation performance',
        'In panel a, each row is a model; gray and blue dots are paired training and validation scores. Check the higher/lower direction before comparing gaps. Panel b shows validation gain over the same-fold dummy baseline, oriented positive-is-better. Green marks show mean and between-fold SD.',
        'Between-fold SD describes split variability rather than confidence in an estimated mean.'),
    'test-performance': (
        '测试集性能及基线比较',
        '横轴是主要评价指标，纵轴分为最终模型和简单基线两行。蓝点表示模型，灰点表示仅在开发集学习简单规则的基线；根据指标方向判断哪个更好。如果出现蓝色横线，它表示固定模型下重复抽取测试样本得到的 95% 区间，不是单个预测的误差范围。',
        '图中基线为点估计；配对差值区间见结果表及源数据。区间的抽样单位和计算条件见估计不确定性部分。',
        'Test performance and baseline comparison',
        'The horizontal axis is the primary metric. Blue is the selected model; gray is the development-fitted dummy baseline. Read the metric direction. A blue interval, when shown, summarizes resampled test performance conditional on this fitted model, not individual prediction error.',
        'The baseline is a point estimate; paired-difference intervals are retained in the results and source data. Resampling units and conditions are specified under estimate uncertainty.'),
    'regression-diagnostics': (
        '预测一致性与残差分布',
        'a 面板横轴是真实值、纵轴是预测值，每点是一条测试记录；贴近对角虚线表示接近真实值，线上方是高估。b 面板横轴是预测值、纵轴是真实值减预测值；零线上方是低估，蓝线概括不同预测范围的平均偏差。c 面板横轴是绝对误差、纵轴是误差不超过这个数的样本比例，可从纵轴 0.9 读出约九成样本的误差上限。d 面板显示不同预测水平的平均绝对误差。',
        'b 的阴影是各箱内残差的 10–90 分位范围，不是未来个体的预测区间；分箱人数少时曲线容易波动。不能只看点云相关性而忽略整体偏移和少数大错误。',
        'Prediction agreement and residual distribution',
        'Panel a plots observed values horizontally and predictions vertically; points above the identity line are overpredictions. Panel b plots observed-minus-predicted residuals; above zero means underprediction. Panel c gives the fraction with absolute error at or below each horizontal value. Panel d shows MAE across prediction levels.',
        'The residual band contains within-bin 10th–90th percentiles, not future prediction intervals. Sparse bins are unstable; strong correlation can hide bias and large errors.'),
    'classification-errors': (
        '混淆矩阵与类别表现',
        'a 面板每行是真实类别，每列是预测类别；对角线是判对的样本，非对角线表示被混淆到哪里。颜色和百分比在每个真实类别内部计算，所以小类别也能单独阅读。b 面板对每类显示 precision（报出的是否真是）、recall（实际的找回多少）和 F1（两者的折中）；n 是实际样本数，C 编号与原标签的对应见图注。',
        '类别人数不同时不能只比较原始错误数量。不存在于测试集的类别不能据此声称已验证；小样本类别的高低分都可能很不稳定。',
        'Confusion matrix and class-specific performance',
        'Panel a uses true classes as rows and predicted classes as columns. Diagonal cells are correct; other cells show confusion destinations. Colors are normalized within true-class rows. Panel b gives precision, recall and F1 with sample counts; the caption maps C labels to original classes.',
        'Compare rates and support, not raw error counts alone. An absent class is not validated, and small classes can have unstable scores.'),
    'classification-discrimination': (
        'ROC 与精确率召回率曲线',
        '左侧 ROC 图横轴是假阳性率，即实际阴性中被误报的比例；纵轴是真阳性率，即阳性被检出的比例。曲线由改变阈值得到，靠近左上角通常更好。右侧 PR 图横轴是召回率、纵轴是精确率，用于观察检出更多阳性时误报是否增加；横虚线对应本批测试样本的阳性比例。',
        'AUC 或 AP 高并不保证概率准确，也不提供一个已验证的最佳阈值。“阳性”只是图注指定的类别，不能自动解释为高风险。',
        'ROC and precision recall curves',
        'ROC plots false-positive rate horizontally and true-positive rate vertically across thresholds; the upper left is favorable. PR plots recall horizontally and precision vertically; its horizontal reference is test prevalence. It shows the false-alarm cost of retrieving more positives.',
        'AUC and AP do not guarantee reliable probabilities or a validated threshold. Positive means the designated label, not automatically high risk.'),
    'probability-calibration': (
        '预测概率校准与分箱样本量',
        '左图把预测概率相近的样本放入同一箱，横轴为该箱平均预测概率，纵轴为实际阳性比例。例如一批样本平均预测为 0.8，理想情况下其中约八成是阳性；这是读图示例，不是本次测量结论。点在对角线上方表示实际阳性比预测更多，下方表示更少。右图是每箱人数，用来判断左图各点背后有多少证据。',
        '分箱样本量决定各点的估计精度。折线连接分箱摘要，稀疏分箱的偏离可能包含较大的抽样波动。',
        'Probability calibration and bin support',
        'The left plot bins similar probabilities. Horizontal values are mean predictions and vertical values observed positive fractions. For illustration, predictions averaging 0.8 ideally correspond to about 80% positives; this is not a measured result here. Above the identity line means more positives than predicted. The right plot gives supporting counts.',
        'Bin counts determine the precision of each summary. Connecting lines are descriptive; deviations in sparse bins may contain substantial sampling variation.'),
    'threshold-tradeoffs': (
        '决策阈值与分类表现的关系',
        '横轴是将样本判为阳性的分数或概率门槛；纵轴的三条线分别表示 sensitivity（检出真实阳性的比例）、specificity（正确排除阴性的比例）及 positive fraction（被判为阳性的全部样本比例）。提高门槛通常减少阳性预测和误报，却可能增加漏检；降低门槛反之。虚线只是常规参考。',
        '曲线描述当前测试数据中阈值与分类指标的对应关系。阈值选择还取决于误判成本；决策分数与概率的尺度需区分。',
        'Decision thresholds and classification performance',
        'The horizontal axis is the positive decision cutoff. Lines show sensitivity, specificity and the fraction predicted positive. Raising the cutoff usually reduces positive calls and false alarms but increases misses. The dashed line is a conventional reference only.',
        'The curves describe threshold-specific performance in this test sample. Threshold choice also depends on misclassification costs; decision scores and probabilities have different scales.'),
    'permutation-importance': (
        '特征置换重要性',
        '每行是一项原始特征。程序打乱该列在测试样本中的对应关系，其他列不变，观察评分下降多少；横轴越大表示打乱后损害越大，即当前模型更依赖它。灰点是多次打乱，蓝点是平均下降，横线表示重复间标准差。误差类指标先转为越大越好的评分，因此正的重要性仍表示性能受损。',
        '重复间标准差描述置换操作的波动。相关变量可能相互替代，使单列置换低估模型对这组信息的总体依赖。',
        'Permutation feature importance',
        'Each row is an original feature. Shuffling it across test samples breaks its alignment while leaving other columns unchanged. Larger score decreases imply greater reliance. Gray dots are repeats, blue is the mean and bars are repeat SD. Loss scores are oriented so positive importance still means harm from shuffling.',
        'Repeat SD describes variation across permutations. Correlated inputs can substitute, making single-column permutations understate reliance on their shared information.'),
    'feature-correlation': (
        '预测变量的相关结构',
        '横纵轴都是特征编号，对应名称见图注。每个方格表示开发样本中两个特征的 Spearman 排序相关，范围 -1 到 1：接近 1 表示经常一起升高，接近 -1 表示一个高时另一个常低，接近零表示单调关联较弱。对角线是特征与自身，不用于寻找额外信息。',
        '该矩阵描述开发集中的单调关联。低秩相关仍可能伴随非单调关系，常量变量的相关系数无法定义。',
        'Correlation structure of predictors',
        'Both axes are feature keys mapped in the caption. Cells show development Spearman rank correlations from -1 to 1: positive means rising together, negative opposing ranks, near zero weak monotonic association. Diagonal self-correlations add no new evidence.',
        'The matrix describes monotonic associations in development data. Low rank correlation can coexist with nonmonotonic dependence; correlations of constant variables are undefined.'),
    'feature-shift': (
        '开发集与测试集的特征分布差异',
        '左图每点是一项特征，横轴是测试均值减开发均值，再除以开发集标准差；零表示均值相同，正值表示测试均值更高，绝对值表示差异相当于多少个开发集标准差。右图灰蓝点分别是开发和测试缺失比例，0.2 就是两成记录缺失。这里只展示位移靠前的特征，完整结果在源表。',
        '均值接近不代表整个分布一致，均值不同也不自动证明模型失效。应结合采集方式、样本来源和误差分布调查；常量或全缺失输入的标准化差异无法定义。',
        'Feature distributions in development and test data',
        'Left: test-minus-development mean divided by development SD; zero is equal means, positive is a higher test mean. Right: development and test missing fractions; 0.2 means 20% missing. Displayed features have the largest defined mean shifts; the source table is complete.',
        'Equal means do not imply equal distributions, and shifted means do not prove model failure. Investigate collection and error patterns. Constant or all-missing inputs have undefined standardized shifts.'),
    'strata-performance': (
        '分组或时间分层的测试集表现',
        '每行代表一个测试组别或时间块，点的位置是该部分的指标，括号 n 是人数。回归通常读 MAE，越低越好；分类这里读 accuracy，越高越好。时间块按先后顺序组成，不能将相邻时间数据当成完全独立的重复证据。',
        '图中是点估计，没有置信区间。不同组可能有不同样本数、类别比例或任务难度，因此差异不能直接解释成不公平、组别原因或随时间退化。',
        'Test performance across groups or time strata',
        'Each row is a test group or ordered time block; n is its count. Regression uses MAE, lower-is-better; classification uses accuracy, higher-is-better. Adjacent time blocks are not independent replications.',
        'These are point estimates without intervals. Differences can reflect support, prevalence or task difficulty, not necessarily unfairness or temporal decline.'),
    'parameter-search': (
        '参数配置与交叉验证表现',
        '横轴的候选编号对应一组模型或预处理设置，纵轴是开发集交叉验证平均分。不同候选用相同划分比较，按指标的高低方向判断。编号只是标签，不是参数大小；精确设置见源表，标题同时记录失败数量。',
        '候选编号仅用于定位配置，横轴距离不表示参数差异的大小；模型选择依据同一验证设计下的平均得分。',
        'Parameter configurations and cross-validation performance',
        'Candidate indices identify model or preprocessing configurations; vertical values are mean development CV scores on shared splits. Follow the metric direction. Indices are labels, not parameter magnitudes; exact settings and failures are in source data.',
        'Candidate indices identify configurations; horizontal distances do not quantify parameter differences. Selection uses mean scores under the shared validation design.'),
}


def figure_guide(figure, s, analysis, zh):
    name = figure['name']
    key = next((key for key in FIGURES if name == key or name.startswith(key + '-')), None)
    if key is None:
        return name, [("如何阅读" if zh else "How to read", figure['caption'])]
    data = FIGURES[key]
    title, reading, limits = data[:3] if zh else data[3:]
    if key == 'strata-performance':
        measure = 'MAE' if s.get('task') == 'regression' else 'accuracy'
        reading = (f"每行代表一个测试{'时间块' if s.get('split') == 'time' else '组别'}，括号 n 为该层记录数，横轴为 {measure}。" +
                   ("MAE 表示平均绝对误差，较低值表示较小偏差。" if measure == 'MAE' else "accuracy 表示分类准确率，较高值表示正确分类比例较高。")) if zh else (
                   f"Each row is a test {'time block' if s.get('split') == 'time' else 'group'}; n denotes its record count. The horizontal axis shows {measure}, " +
                   ('mean absolute error, where lower values indicate smaller errors.' if measure == 'MAE' else 'the fraction classified correctly, where higher values indicate better performance.'))
    codes = {'model-validation': ['generalization'], 'test-performance': ['baseline'],
             'regression-diagnostics': ['errors'], 'classification-errors': ['class_errors'],
             'probability-calibration': ['calibration'], 'threshold-tradeoffs': ['decision_score'],
             'permutation-importance': ['reliance'], 'feature-shift': ['shift'],
             'strata-performance': ['stability']}.get(key, [])
    measured = [f['zh' if zh else 'en'] for f in analysis['findings'] if f['code'] in codes]
    if key == 'classification-discrimination':
        measured = [(f"{metric_explanation(k, None, zh)[0]} = {s['test_metrics'][k]:.4g}。" if zh else
                     f"{metric_explanation(k, None, zh)[0]} = {s['test_metrics'][k]:.4g}. ")
                    for k in ('roc_auc', 'average_precision') if s['test_metrics'].get(k) is not None]
    if key == 'parameter-search':
        search = s.get('search', {})
        measured = [(f"共比较 {search.get('evaluated_candidates', 0)} 个参数候选，失败 {search.get('failed_candidates', 0)} 个；最终选择{model_name(s['selected_model'], zh)}。" if zh else
                     f"The search evaluated {search.get('evaluated_candidates', 0)} configurations with {search.get('failed_candidates', 0)} failures, selecting {model_name(s['selected_model'], zh)}.")]
    paragraphs = []
    if measured:
        paragraphs.append(("结果" if zh else "Results", ' '.join(measured)))
    paragraphs.append(("图示说明" if zh else "Figure interpretation", reading))
    paragraphs.append(("方法说明" if zh else "Methodological note", limits))
    return title, paragraphs
