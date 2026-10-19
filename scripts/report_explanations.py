"""Academic report prose and accessible definitions grounded in measured results."""

from __future__ import annotations


METRICS = {
    "mae": ("平均绝对误差 MAE", "每个预测与真实值相差多少，取绝对值后求平均；越小越好，单位与目标相同。它不是每个样本的最大误差。",
            "Mean absolute error", "Average absolute distance from the observed outcome. Lower is better, in outcome units; it is not a maximum error."),
    "mse": ("均方误差 MSE", "先将每个误差平方再平均，因此特别重视大错误；越小越好，单位是目标单位的平方，不能直接读成平均差多少分。",
            "Mean squared error", "Average squared error, giving large mistakes extra weight. Lower is better; units are squared outcome units, not the original scale."),
    "rmse": ("均方根误差 RMSE", "对均方误差开平方，将结果还原为目标单位；越小越好。它比 MAE 更容易被少量大错误拉高。",
             "Root mean squared error", "Square root of MSE, returning to outcome units. Lower is better; a few large mistakes affect it more than MAE."),
    "r2": ("决定系数 R²", "比较模型误差与始终预测测试集平均值时的误差。1 表示完全吻合，0 表示在此标准下没有改善，负值表示更差；它不是预测正确率，也不是因果解释比例。这里的均值参照与图中只在训练集拟合的简单基线不同。",
           "R squared", "Compares squared errors with predicting the test outcome mean. One is perfect, zero gives no improvement under that reference, and negative is worse. It is not accuracy or causal explanation. This reference differs from the plotted baseline fitted only on training data."),
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


def validation_methods(s, zh):
    dev, test, k = s.get('development_n'), s.get('test_n'), s['cv_folds']
    share = round(100 * test / (dev + test)) if dev and test else None
    if s['external_test']:
        split = (f"训练集为全部输入数据（n = {dev}），测试集为另行提供的外部数据（n = {test}）。" if zh else
                 f"All input data formed the training set (n = {dev}); a separately supplied external set formed the test set (n = {test}).")
    elif s['split'] == 'group':
        split = (f"数据按组划分为训练集（n = {dev}）和测试集（n = {test}），同一组的样本只出现在其中一个集合中。" if zh else
                 f"Data were split by group into training (n = {dev}) and test (n = {test}) sets, so that each group appears in only one set.")
    elif s['split'] == 'time':
        split = (f"数据按时间顺序划分：较早的数据作为训练集（n = {dev}），最近的时间段作为测试集（n = {test}）。" if zh else
                 f"Data were split chronologically: earlier records formed the training set (n = {dev}) and the most recent period the test set (n = {test}).")
    else:
        ratio = f"按 {100 - share}:{share} " if share else ""
        ratio_en = f"{100 - share}:{share} " if share else ""
        split = (f"数据{ratio}随机划分为训练集（n = {dev}）和测试集（n = {test}）" + ("，按类别分层" if s['task'] == 'classification' else "") + "。" if zh else
                 f"Data were randomly split {ratio_en}into training (n = {dev}) and test (n = {test}) sets" + (", stratified by class" if s['task'] == 'classification' else "") + ".")
    hold = ("测试集在模型选择前预留，不参与任何训练、调参或预处理。" if zh else
            "The test set was held out before model selection and used for no training, tuning or preprocessing.")
    cv = {
        'random': (f"在训练集上进行 {k} 折交叉验证" + ("（分层）" if s['task'] == 'classification' else "") + "以比较候选模型。",
                   f"Candidate models were compared by {k}-fold cross-validation" + (" (stratified)" if s['task'] == 'classification' else "") + " on the training set."),
        'group': (f"在训练集上进行 {k} 折分组交叉验证（GroupKFold），验证折同样按组隔离；组标识只用于划分，不作为特征。",
                  f"Candidates were compared by {k}-fold grouped cross-validation (GroupKFold) on the training set; group identifiers were used only for splitting."),
        'time': (f"在训练集上进行 {k} 折前向时间序列交叉验证（训练数据总早于验证数据），两者之间间隔 {s['gap']} 个时间块。",
                 f"Candidates were compared by {k}-fold forward time-series cross-validation (training always precedes validation) with a gap of {s['gap']} time blocks."),
    }[s['split']][0 if zh else 1]
    refit = (f"最终模型在完整训练集上重新训练后，在测试集上评估一次。随机种子为 {s['seed']}。" if zh else
             f"The final model was refitted on the full training set and evaluated once on the test set. Random seed: {s['seed']}.")
    return [split, hold if not s['external_test'] else "", cv, refit]


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
        return [f"测试集指标的不确定性用{method}估计（{interval['requested_repeats']} 次重复，重采样单位 {interval['units']} 个），"
                "报告 95% 百分位区间；该区间以已拟合模型为条件，不包含重新训练与模型选择带来的不确定性，"
                "有效重复不足 100 次或不足请求次数的 80% 时不报告。"]
    return [f"Uncertainty in test metrics was estimated by {interval['method']} ({interval['requested_repeats']} repeats over "
            f"{interval['units']} units), reporting 95% percentile intervals conditional on the fitted model (excluding retraining and "
            "selection uncertainty); intervals are omitted with fewer than 100 or 80% valid repeats. "]


def preprocessing(s, zh):
    p = s['preprocessing']
    steps, notes = [], []

    def step(cn, en):
        steps.append(cn if zh else en)

    if p.get('impute') == 'median':
        step("缺失值以训练数据中该特征的中位数插补", "median imputation of missing values")
    if p.get('scale'):
        step("min-max 归一化（缩放到 0–1）" if p.get('scaler') == 'minmax' else "z 标准化",
             "min-max scaling to 0–1" if p.get('scaler') == 'minmax' else "z-standardisation")
    selection = p.get('selection')
    if selection == 'forward':
        step(f"前向逐步特征筛选（保留 {p.get('select_k') or '自动确定'} 个特征，使用训练折内部的交叉验证）",
             f"forward stepwise feature selection ({p.get('select_k') or 'automatic'} features, inner cross-validation)")
    elif selection == 'f-threshold':
        step(f"单变量 F 检验筛选（保留 F 值不低于 {p.get('f_threshold')} 的特征）", f"univariate F-test selection (F ≥ {p.get('f_threshold')})")
    elif p.get('select_k'):
        step(f"单变量筛选保留 {p['select_k']} 个特征", f"univariate selection of the top {p['select_k']} features")
    if p.get('pca'):
        step(f"主成分分析（PCA）降维至 {p['pca']} 个成分", f"principal component analysis (PCA) to {p['pca']} components")
    if p.get('resample', 'none') != 'none':
        step(f"对训练数据进行类别重采样（{p['resample']}），验证与测试数据不重采样",
             f"class resampling of training data ({p['resample']}); validation and test data were not resampled")
    if not steps:
        lead = ("未进行缺失值插补或缩放，" if zh else "No imputation or scaling was applied; ")
    else:
        lead = ("预处理包括：" + "、".join(steps) + "。" if zh else "Preprocessing comprised " + ", ".join(steps) + ". ")
    if p.get('impute') != 'median':
        notes.append("本次未进行缺失值插补，要求特征完整。" if zh else "No imputation was applied; complete features were required. ")
    if not selection or (selection == 'kbest' and not p.get('select_k')):
        notes.append("本次不做特征筛选，全部特征进入模型。" if zh else "No feature selection was applied. ")
    leak = ("所有步骤封装在 scikit-learn Pipeline 中，只在训练折上拟合，以避免数据泄漏。" if zh else
            "All steps were wrapped in a scikit-learn Pipeline fitted on training folds only, to prevent data leakage.")
    return [lead if steps else "", *notes, leak]


# Each entry explains the exact axes and marks of a generated diagnostic family.
FIGURES = {
    'model-validation': (
        '候选模型的训练与交叉验证表现',
        'a 面板每行是一种模型，横轴是所选指标；灰点是训练得分，蓝点是同一折的验证得分，连线连接一对结果。先按横轴的 higher/lower 判断好坏，再比较灰蓝差距。b 面板把验证表现减去同折简单基线的表现，并统一成正值更好；绿点是平均增益，线段是交叉验证标准差，表示不同划分下的波动。',
        '交叉验证标准差描述划分间的波动，区别于均值估计的置信区间。',
        'Training and cross-validation performance',
        'In panel a, each row is a model; gray and blue dots are paired training and validation scores. Check the higher/lower direction before comparing gaps. Panel b shows validation gain over the same-fold dummy baseline, oriented positive-is-better. Green marks show mean and cross-validation SD.',
        'Between-fold SD describes split variability rather than confidence in an estimated mean.'),
    'test-performance': (
        '测试集性能及基线比较',
        '横轴是主要评价指标，纵轴分为最终模型和简单基线两行。蓝点表示模型，灰点表示仅在训练集学习简单规则的基线；根据指标方向判断哪个更好。如果出现蓝色横线，它表示固定模型下重复抽取测试样本得到的 95% 区间，不是单个预测的误差范围。',
        '图中基线为点估计；相对基线的差值区间见结果表及源数据。区间的抽样单位和计算条件见方法 1.5 节。',
        'Test performance and baseline comparison',
        'The horizontal axis is the primary metric. Blue is the selected model; gray is the dummy baseline. Read the metric direction. A blue interval, when shown, summarizes resampled test performance conditional on this fitted model, not individual prediction error.',
        'The baseline is a point estimate; paired-difference intervals are retained in the results and source data. Resampling units and conditions are given in Methods 1.5.'),
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
        '横纵轴都是特征编号，对应名称见图注。每个方格表示训练集样本中两个特征的 Spearman 排序相关，范围 -1 到 1：接近 1 表示经常一起升高，接近 -1 表示一个高时另一个常低，接近零表示单调关联较弱。对角线是特征与自身，不用于寻找额外信息。',
        '该矩阵描述训练集中的单调关联。低秩相关仍可能伴随非单调关系，常量变量的相关系数无法定义。',
        'Correlation structure of predictors',
        'Both axes are feature keys mapped in the caption. Cells show training Spearman rank correlations from -1 to 1: positive means rising together, negative opposing ranks, near zero weak monotonic association. Diagonal self-correlations add no new evidence.',
        'The matrix describes monotonic associations in training data. Low rank correlation can coexist with nonmonotonic dependence; correlations of constant variables are undefined.'),
    'feature-shift': (
        '训练集与测试集的特征分布差异',
        '左图每点是一项特征，横轴是测试均值减训练集均值，再除以训练集标准差；零表示均值相同，正值表示测试均值更高，绝对值表示差异相当于多少个训练集标准差。右图灰蓝点分别是训练和测试缺失比例，0.2 就是两成记录缺失。这里只展示位移靠前的特征，完整结果在源表。',
        '均值接近不代表整个分布一致，均值不同也不自动证明模型失效。应结合采集方式、样本来源和误差分布调查；常量或全缺失输入的标准化差异无法定义。',
        'Feature distributions in training and test data',
        'Left: test-minus-development mean divided by training SD; zero is equal means, positive is a higher test mean. Right: training and test missing fractions; 0.2 means 20% missing. Displayed features have the largest defined mean shifts; the source table is complete.',
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
        '横轴的候选编号对应一组模型或预处理设置，纵轴是训练集交叉验证平均分。不同候选用相同划分比较，按指标的高低方向判断。编号只是标签，不是参数大小；精确设置见源表，标题同时记录失败数量。',
        '候选编号仅用于定位配置，横轴距离不表示参数差异的大小；模型选择依据同一验证设计下的平均得分。',
        'Parameter configurations and cross-validation performance',
        'Candidate indices identify model or preprocessing configurations; vertical values are mean training CV scores on shared splits. Follow the metric direction. Indices are labels, not parameter magnitudes; exact settings and failures are in source data.',
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
