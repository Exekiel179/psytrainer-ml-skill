"""Plain-language reading guides grounded in the current report's measurements."""

from __future__ import annotations


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
        result += (f"也就是说，在这批测试样本中，每次预测与真实结果平均相差约 {value:.4g} 个目标单位；是否可接受取决于业务或研究事先规定的容许误差。" if zh else
                   f"Predictions differ from observations by {value:.4g} outcome units on average here; acceptability requires an application-specific tolerance chosen in advance.")
    if value is not None and key == "accuracy":
        result += (f"对应这批样本约 {value:.1%} 的类别判断正确，而非未来每一批数据的保证。" if zh else
                   f"This corresponds to {value:.1%} correct in this sample, not a guarantee for future batches.")
    return title, description + " " + result


def introduction(s, zh):
    if zh:
        return [
            f"本报告研究能否用已有信息预测 {s['target']}。输入表的一行是一条样本记录，一列特征是一项用于预测的信息；目标是希望预测的真实结果。模型通过已知输入和结果学习规律，再对未参与学习的样本给出预测。这里的目标是评估预测能力，不是证明某个因素导致了结果。",
            ("本次是回归任务：预测一个数值，重点看预测偏差有多大、哪些范围容易出错。" if s['task'] == 'regression' else
             f"本次是分类任务：预测属于哪个类别。实际类别为 {', '.join(map(str, s['classes']))}；这些名称是数据标签，不自动代表疾病、风险或其他领域含义。重点看误判和漏检发生在哪些类别。"),
            f"数据分为两部分：{s['development_n']} 条开发样本用于学习和比较模型，{s['test_n']} 条测试样本留到最后评价。开发集内部又轮流留出一部分作为验证集，共做 {s['cv_folds']} 折交叉验证。每折相当于一次模拟测验；最终测试集相当于模型选择完成后才打开的考卷。因此，挑选模型的验证分数与最终测试分数承担不同作用。",
            "阅读时先看独立测试结果及其相对简单基线的改善，再看错误是否集中在某类样本或取值范围，最后看哪些输入被模型依赖。单张漂亮的图或单个高分不能证明模型已经适合应用。报告中 NA 或 undefined 表示当前数据或设计无法支持该数值，不表示零误差。",
        ]
    return [
        f"This report asks whether available information can predict {s['target']}. A row is a sample, a feature column is an input, and the target is the observed outcome to predict. A model learns from examples and is evaluated on samples withheld from learning. Predictive performance does not establish causation.",
        ("This is regression: predicting a number. Read errors in outcome units and examine where predictions fail." if s['task'] == 'regression' else
         f"This is classification: predicting a category. Labels are {', '.join(map(str, s['classes']))}; no clinical or risk meaning is inferred from their names. Examine both false alarms and missed cases."),
        f"The {s['development_n']} development samples support learning and model comparison. Within them, {s['cv_folds']}-fold cross-validation rotates held-back validation samples as practice assessments. The {s['test_n']} test samples are reserved for a final assessment after selection. Validation scores select models; test scores evaluate that selection.",
        "Read held-out performance against a simple baseline first, then error patterns and feature reliance. A single high score is insufficient for deployment. NA or undefined means the data or design does not support an estimate, not zero error.",
    ]


def preprocessing(s, zh):
    p = s['preprocessing']
    lines = []
    def add(cn, en):
        lines.append(cn if zh else en)
    if p.get('impute') == 'median':
        add("缺失值用训练样本中该特征的中位数填补。这让不完整记录可以进入模型，但不能恢复真实缺失值；测试样本不参与计算填充值。", "Missing inputs use training-feature medians. This permits prediction but does not recover the missing truth; test samples do not set replacement values.")
    else:
        add("本次未填补缺失值；输入必须完整。程序不会悄悄删掉缺失样本来提高分数。", "No imputation is used; inputs must be complete. Missing samples are not silently removed to improve scores.")
    if p.get('scale'):
        add(("数值缩放采用最小值和最大值归一化，将训练范围映射到 0–1。新样本超出训练范围时可以超过这个区间。" if p.get('scaler') == 'minmax' else "数值标准化将每项特征减去训练均值并除以训练标准差，使不同计量单位更容易共同参与建模。"),
            ("Min-max scaling maps the training range to 0–1; new values can fall outside it." if p.get('scaler') == 'minmax' else "Standardization subtracts training means and divides by training standard deviations, bringing differing measurement scales into a comparable range."))
    else:
        add("本次不进行数值缩放，特征保留原始尺度。", "Features retain their original scales; no scaling is applied.")
    selection = p.get('selection')
    if selection == 'forward':
        add(f"逐步前向筛选从较少特征开始，利用训练部分的内部验证逐项尝试加入信息。本次选择数量设置为 {p.get('select_k') or '自动'}；它只寻找当前搜索路径上的改善，不保证找到了所有可能组合中的最优解。", f"Forward selection adds features using validation within training data; requested count is {p.get('select_k') or 'automatic'}. This searches one path, not every possible subset.")
    elif selection == 'f-threshold':
        add(f"特征筛选保留 F 统计量超过 {p.get('f_threshold')} 的输入。这个分数描述单个输入与结果的统计关联，不代表因果，也可能遗漏必须组合才有效的信息。", f"F-threshold selection retains inputs above {p.get('f_threshold')}. It assesses individual statistical associations, not causation, and can miss interactions.")
    elif p.get('select_k'):
        add(f"本次按单变量关联分数保留 {p['select_k']} 个特征。每个特征先单独评分，因此互相配合才有用的信息可能被遗漏。", f"The top {p['select_k']} individually scored features are retained. Inputs useful only in combination may be overlooked.")
    else:
        add("本次不做特征筛选，全部输入特征进入后续步骤。", "No feature selection is applied; all inputs enter subsequent steps.")
    if p.get('pca'):
        add(f"PCA 将已处理的特征压缩为 {p['pca']} 个综合成分。成分是原特征的加权组合，可减少维度，但不再一一对应原变量，也不保证保留了最有预测价值的信息。", f"PCA compresses processed inputs into {p['pca']} weighted components. It reduces dimension, but components are not individual original variables and need not preserve the most predictive information.")
    if p.get('resample', 'none') != 'none':
        add(f"训练时使用 {p['resample']} 调整类别样本构成，帮助模型学习较少见的类别。这不创造新的独立观测证据；验证和测试数据保持原样。", f"Training uses {p['resample']} to adjust class representation. This creates no new independent evidence; validation and test data remain unchanged.")
    add("以上所有需要从数据学习的步骤都只在当前训练部分计算，再原样用于验证和测试。这是避免答案提前进入学习过程的措施；如果输入表在进入程序前已经使用全部样本或未来信息处理过，本流程无法消除那种信息泄漏。", "Every learned step fits only the current training partition and is reused unchanged for validation and test. This prevents looking at held-back answers, but cannot undo leakage already embedded in the input table.")
    return lines


# Each entry explains the exact axes and marks of a generated diagnostic family.
FIGURES = {
    'model-validation': (
        '模型能否把训练中学到的规律用到没参与学习的样本上',
        'a 面板每行是一种模型，横轴是所选指标；灰点是训练得分，蓝点是同一折的验证得分，连线连接一对结果。先按横轴的 higher/lower 判断好坏，再比较灰蓝差距。b 面板把验证表现减去同折简单基线的表现，并统一成正值更好；绿点是平均增益，线段是折间标准差，表示不同划分下的波动。',
        '训练好而验证弱可能提示模型记住了训练细节，也可能反映样本难度差异。不要把标准差线段当作置信区间，也不能只按线段是否重叠判断统计显著性。',
        'Does learning transfer to unseen samples',
        'In panel a, each row is a model; gray and blue dots are paired training and validation scores. Check the higher/lower direction before comparing gaps. Panel b shows validation gain over the same-fold dummy baseline, oriented positive-is-better. Green marks show mean and between-fold SD.',
        'A training advantage can reflect overfitting or split difficulty. SD bars are not confidence intervals, and overlap is not a significance test.'),
    'test-performance': (
        '最终模型在保留到最后的测试数据上是否有用',
        '横轴是主要评价指标，纵轴分为最终模型和简单基线两行。蓝点表示模型，灰点表示仅在开发集学习简单规则的基线；根据指标方向判断哪个更好。如果出现蓝色横线，它表示固定模型下重复抽取测试样本得到的 95% 区间，不是单个预测的误差范围。',
        '超过基线只说明优于这个简单参照，不代表达到实际使用标准。没有区间不表示结果确定无疑；需查看区间省略原因。',
        'How useful is the final held-out performance',
        'The horizontal axis is the primary metric. Blue is the selected model; gray is the development-fitted dummy baseline. Read the metric direction. A blue interval, when shown, summarizes resampled test performance conditional on this fitted model, not individual prediction error.',
        'Beating a dummy does not establish practical utility. An omitted interval is not evidence of certainty.'),
    'regression-diagnostics': (
        '数值预测错在哪里以及误差有多大',
        'a 面板横轴是真实值、纵轴是预测值，每点是一条测试记录；贴近对角虚线表示接近真实值，线上方是高估。b 面板横轴是预测值、纵轴是真实值减预测值；零线上方是低估，蓝线概括不同预测范围的平均偏差。c 面板横轴是绝对误差、纵轴是误差不超过这个数的样本比例，可从纵轴 0.9 读出约九成样本的误差上限。d 面板显示不同预测水平的平均绝对误差。',
        'b 的阴影是各箱内残差的 10–90 分位范围，不是未来个体的预测区间；分箱人数少时曲线容易波动。不能只看点云相关性而忽略整体偏移和少数大错误。',
        'Where numerical predictions fail',
        'Panel a plots observed values horizontally and predictions vertically; points above the identity line are overpredictions. Panel b plots observed-minus-predicted residuals; above zero means underprediction. Panel c gives the fraction with absolute error at or below each horizontal value. Panel d shows MAE across prediction levels.',
        'The residual band contains within-bin 10th–90th percentiles, not future prediction intervals. Sparse bins are unstable; strong correlation can hide bias and large errors.'),
    'classification-errors': (
        '哪些类别容易漏检或被误判',
        'a 面板每行是真实类别，每列是预测类别；对角线是判对的样本，非对角线表示被混淆到哪里。颜色和百分比在每个真实类别内部计算，所以小类别也能单独阅读。b 面板对每类显示 precision（报出的是否真是）、recall（实际的找回多少）和 F1（两者的折中）；n 是实际样本数，C 编号与原标签的对应见图注。',
        '类别人数不同时不能只比较原始错误数量。不存在于测试集的类别不能据此声称已验证；小样本类别的高低分都可能很不稳定。',
        'Which classes are missed or confused',
        'Panel a uses true classes as rows and predicted classes as columns. Diagonal cells are correct; other cells show confusion destinations. Colors are normalized within true-class rows. Panel b gives precision, recall and F1 with sample counts; the caption maps C labels to original classes.',
        'Compare rates and support, not raw error counts alone. An absent class is not validated, and small classes can have unstable scores.'),
    'classification-discrimination': (
        '模型能否将阳性样本排在前面',
        '左侧 ROC 图横轴是假阳性率，即实际阴性中被误报的比例；纵轴是真阳性率，即阳性被检出的比例。曲线由改变阈值得到，靠近左上角通常更好。右侧 PR 图横轴是召回率、纵轴是精确率，用于观察检出更多阳性时误报是否增加；横虚线对应本批测试样本的阳性比例。',
        'AUC 或 AP 高并不保证概率准确，也不提供一个已验证的最佳阈值。“阳性”只是图注指定的类别，不能自动解释为高风险。',
        'Can the model rank positive samples first',
        'ROC plots false-positive rate horizontally and true-positive rate vertically across thresholds; the upper left is favorable. PR plots recall horizontally and precision vertically; its horizontal reference is test prevalence. It shows the false-alarm cost of retrieving more positives.',
        'AUC and AP do not guarantee reliable probabilities or a validated threshold. Positive means the designated label, not automatically high risk.'),
    'probability-calibration': (
        '模型说出的概率是否与实际频率相符',
        '左图把预测概率相近的样本放入同一箱，横轴为该箱平均预测概率，纵轴为实际阳性比例。例如一批样本平均预测为 0.8，理想情况下其中约八成是阳性；这是读图示例，不是本次测量结论。点在对角线上方表示实际阳性比预测更多，下方表示更少。右图是每箱人数，用来判断左图各点背后有多少证据。',
        '少数样本的箱不能支持稳定结论；折线只是连接摘要点。概率校准要在开发集内完成，不能看完此测试图再修改概率并继续把同一批数据称为独立测试。',
        'Do predicted probabilities match observed frequencies',
        'The left plot bins similar probabilities. Horizontal values are mean predictions and vertical values observed positive fractions. For illustration, predictions averaging 0.8 ideally correspond to about 80% positives; this is not a measured result here. Above the identity line means more positives than predicted. The right plot gives supporting counts.',
        'Sparse bins do not establish reliability. Connecting lines are descriptive. Fit any calibration on development data and evaluate changes on new test data.'),
    'threshold-tradeoffs': (
        '判定阈值改变时漏检误报和工作量如何变化',
        '横轴是将样本判为阳性的分数或概率门槛；纵轴的三条线分别表示 sensitivity（检出真实阳性的比例）、specificity（正确排除阴性的比例）及 positive fraction（被判为阳性的全部样本比例）。提高门槛通常减少阳性预测和误报，却可能增加漏检；降低门槛反之。虚线只是常规参考。',
        '不能从这张测试图直接宣布某个阈值最优。需要先明确漏检与误报的实际成本，在开发集选择，再用新的独立数据检查；决策分数不是概率。',
        'How thresholds trade missed cases for false alarms and workload',
        'The horizontal axis is the positive decision cutoff. Lines show sensitivity, specificity and the fraction predicted positive. Raising the cutoff usually reduces positive calls and false alarms but increases misses. The dashed line is a conventional reference only.',
        'This test plot cannot select an optimal threshold. Specify error costs, choose on development data, and evaluate on a new test set. Decision scores are not probabilities.'),
    'permutation-importance': (
        '当前模型在预测时依赖哪些输入',
        '每行是一项原始特征。程序打乱该列在测试样本中的对应关系，其他列不变，观察评分下降多少；横轴越大表示打乱后损害越大，即当前模型更依赖它。灰点是多次打乱，蓝点是平均下降，横线表示重复间标准差。误差类指标先转为越大越好的评分，因此正的重要性仍表示性能受损。',
        '这不是特征导致结果改变的大小，也不说明正负效应方向。多个特征高度相关时可互相替代；重要性接近零或为负不意味着可以凭这批测试结果直接删掉它。',
        'Which inputs does this fitted model rely on',
        'Each row is an original feature. Shuffling it across test samples breaks its alignment while leaving other columns unchanged. Larger score decreases imply greater reliance. Gray dots are repeats, blue is the mean and bars are repeat SD. Loss scores are oriented so positive importance still means harm from shuffling.',
        'Importance gives neither a causal effect nor its direction. Correlated inputs can substitute, and zero or negative importance is not permission for test-driven feature removal.'),
    'feature-correlation': (
        '哪些输入可能携带重复信息',
        '横纵轴都是特征编号，对应名称见图注。每个方格表示开发样本中两个特征的 Spearman 排序相关，范围 -1 到 1：接近 1 表示经常一起升高，接近 -1 表示一个高时另一个常低，接近零表示单调关联较弱。对角线是特征与自身，不用于寻找额外信息。',
        '高度相关的输入可使单列重要性看起来较低；低相关也不代表没有复杂关系。这张图不证明因果，也不能代替在开发集重新训练后的删减比较。',
        'Which inputs may carry overlapping information',
        'Both axes are feature keys mapped in the caption. Cells show development Spearman rank correlations from -1 to 1: positive means rising together, negative opposing ranks, near zero weak monotonic association. Diagonal self-correlations add no new evidence.',
        'Correlated inputs can mask single-feature reliance. Low rank correlation does not exclude complex dependence. Neither causality nor safe feature removal follows from this view.'),
    'feature-shift': (
        '测试数据与学习时见到的数据是否相似',
        '左图每点是一项特征，横轴是测试均值减开发均值，再除以开发集标准差；零表示均值相同，正值表示测试均值更高，绝对值表示差异相当于多少个开发集标准差。右图灰蓝点分别是开发和测试缺失比例，0.2 就是两成记录缺失。这里只展示位移靠前的特征，完整结果在源表。',
        '均值接近不代表整个分布一致，均值不同也不自动证明模型失效。应结合采集方式、样本来源和误差分布调查；常量或全缺失输入的标准化差异无法定义。',
        'Do test inputs resemble the learning data',
        'Left: test-minus-development mean divided by development SD; zero is equal means, positive is a higher test mean. Right: development and test missing fractions; 0.2 means 20% missing. Displayed features have the largest defined mean shifts; the source table is complete.',
        'Equal means do not imply equal distributions, and shifted means do not prove model failure. Investigate collection and error patterns. Constant or all-missing inputs have undefined standardized shifts.'),
    'strata-performance': (
        '总体分数是否掩盖了不同组别或时间段的差异',
        '每行代表一个测试组别或时间块，点的位置是该部分的指标，括号 n 是人数。回归通常读 MAE，越低越好；分类这里读 accuracy，越高越好。时间块按先后顺序组成，不能将相邻时间数据当成完全独立的重复证据。',
        '图中是点估计，没有置信区间。不同组可能有不同样本数、类别比例或任务难度，因此差异不能直接解释成不公平、组别原因或随时间退化。',
        'Does an overall score conceal group or time differences',
        'Each row is a test group or ordered time block; n is its count. Regression uses MAE, lower-is-better; classification uses accuracy, higher-is-better. Adjacent time blocks are not independent replications.',
        'These are point estimates without intervals. Differences can reflect support, prevalence or task difficulty, not necessarily unfairness or temporal decline.'),
    'parameter-search': (
        '为什么选择这一组模型设置',
        '横轴的候选编号对应一组模型或预处理设置，纵轴是开发集交叉验证平均分。不同候选用相同划分比较，按指标的高低方向判断。编号只是标签，不是参数大小；精确设置见源表，标题同时记录失败数量。',
        '候选越多，挑中的验证分数越可能偏乐观。这张图用于解释开发阶段的选择过程，不能替代最终测试，也不能用测试结果反过来选择候选。',
        'Why was this configuration selected',
        'Candidate indices identify model or preprocessing configurations; vertical values are mean development CV scores on shared splits. Follow the metric direction. Indices are labels, not parameter magnitudes; exact settings and failures are in source data.',
        'Searching more candidates can make the winning validation score optimistic. This explains selection, not final performance, and test results must not select the candidate.'),
}


def figure_guide(figure, s, analysis, zh):
    name = figure['name']
    key = next((key for key in FIGURES if name == key or name.startswith(key + '-')), None)
    if key is None:
        return name, [("如何阅读" if zh else "How to read", figure['caption'])]
    data = FIGURES[key]
    title, reading, limits = data[:3] if zh else data[3:]
    codes = {'model-validation': ['generalization'], 'test-performance': ['baseline'],
             'regression-diagnostics': ['errors'], 'classification-errors': ['class_errors'],
             'probability-calibration': ['calibration'], 'threshold-tradeoffs': ['decision_score'],
             'permutation-importance': ['reliance'], 'feature-shift': ['shift'],
             'strata-performance': ['stability']}.get(key, [])
    measured = [f['zh' if zh else 'en'] for f in analysis['findings'] if f['code'] in codes]
    if key == 'classification-discrimination':
        measured = [metric_explanation(k, s['test_metrics'].get(k), zh)[1]
                    for k in ('roc_auc', 'average_precision') if k in s['test_metrics']]
    if key == 'parameter-search':
        search = s.get('search', {})
        measured = [(f"本次比较了 {search.get('evaluated_candidates', 0)} 个参数候选，失败 {search.get('failed_candidates', 0)} 个；最终选择 {s['selected_model']}。" if zh else
                     f"This run evaluated {search.get('evaluated_candidates', 0)} configurations with {search.get('failed_candidates', 0)} failures; selected {s['selected_model']}.")]
    paragraphs = [("读图方法" if zh else "Reading the figure", reading)]
    if measured:
        paragraphs.append(("本次结果" if zh else "This run", ' '.join(measured)))
    paragraphs.append(("解释边界" if zh else "Interpretation limits", limits))
    return title, paragraphs
