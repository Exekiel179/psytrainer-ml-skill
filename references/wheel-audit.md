# PsyTrainer 0.2.0 功能迁移审计

审计对象为仓库内原始 `vendor/PsyTrainer-0.2.0-cp314-none-any.whl`，
逐项核对包内 79 个文件所属的公开模型、特征处理、重采样、训练、评分、
领域入口及辅助模块。以下是功能迁移，不承诺复现旧版存在泄漏或错误的数值结果。
默认 Pipeline 不导入 wheel；旧 INI 和历史模型保留兼容入口。

## 功能对照与验证

| 原能力/源码 | 本地实现与决定 | 验证 |
|---|---|---|
| `model/model_factory.py`：9 分类、12 回归模型 | 全部保留名称与算法映射，`model_registry.py`；XGBoost 分类统一编码并恢复原标签 | `test_model_registry` 全模型拟合、预测、序列化及可选原工厂对比；`test_migration` 字符串标签验证 |
| `ModelBase.training`：参数网格与最佳参数 | `--search grid`，条件网格 JSON，整条 Pipeline 每折重新拟合；新增有种子的 `random` 搜索与预算 | 搜索候选实际拟合、无效候选记录、测试标签改动不改变选择、候选预算与复现测试 |
| `params/train/general_train_params.py`：通用默认网格 | 用 `compact_grid` 的 21 模型小网格替代，仍支持自定义原来有效的参数和值 | 每个模型从本地网格抽取配置实际拟合预测；并非穷举所有参数组合 |
| `params/train/{audio,face,gait,text}`：领域参数预设 | `--preset` 保留各领域模型集合，搜索使用维护的小网格；预处理显式选择 | 领域集合从 wheel AST 提取模型名核对；每个领域执行真实模型比较 |
| `FFByNor` | `--scaler minmax`，在训练折拟合，保存缩放器并在预测时复用 | 测试训练数据极值、留出隔离、序列化预测一致 |
| `FFByPca` | `--pca N`，折内拟合并随模型保存 | `smoke_pipeline` 随机/分组/时间真实运行及保存预测 |
| `FFByFregression` | `--selection f-threshold --f-threshold 5`；回归 F 检验，分类扩展为 ANOVA F 检验 | 最后一列强信号必须保留；空选择明确失败；折内拟合与保存预测 |
| `FFByStepForward` | `--selection forward`，使用 sklearn SequentialFeatureSelector；内部 CV 保持随机/组/时间设计，内部预处理和重采样也折内拟合 | 三种内部划分边界、原始缺失值、内部填补样本数、序列化预测；时间设计真实运行 |
| `RandomOverSampler` | `--resample random-over` | 七种方法分别拟合、预测、保存重载；预测不重采样 |
| `SMOTE` | `--resample smote` | 同上；真实分组 CV；邻居样本不足明确失败 |
| `SMOTETomek` | `--resample smote-tomek` | 同上 |
| `SMOTEENN` | `--resample smote-enn` | 同上 |
| `ClusterCentroids` | `--resample cluster-centroids` | 同上 |
| `RandomUnderSampler` | `--resample random-under` | 同上 |
| `NearMiss` | `--resample near-miss` | 同上 |
| `Trainer.run`：遍历模型/筛选/重采样组合 | `--preset all` 或多次 `--model`；`preprocess__*` 条件搜索比较预处理/重采样组合 | 条件空间包含不同 scaler/筛选/采样方案，实际训练并检查获选配置 |
| `Trainer.save_result`：得分排序、最佳配置、每折指标 | `comparison`、`search-results.json`、`cv-scores.csv`，按正确指标方向排序 | R2 优先较高得分；损失方向、无效候选、选择隔离测试 |
| `done_tag.csv`：断点续跑 | `--resume` 按候选/折存 JSON，原子写入；检查输入、参数、源码、Python 和依赖版本一致 | 在第二折中断后只补未完成折和最终拟合；拒绝输入/参数变更 |
| `ModelBase.save`：保存多个模型 | 默认保存获选完整 Pipeline；`--save-candidates` 额外保存每个成功模型的最佳配置，均只拟合开发集 | 两个分类模型分别从保存文件预测原标签 |
| 保存特征列表/PCA | `selected-features.csv`，预处理完整保存在 `pipeline.joblib`；PCA 输出是分量名 | 选择/缩放/PCA 保存后预测验证 |
| MAE/MSE/RMSE/R2/Pearson r | 全部提供；常量相关明确为 undefined，R2 越大越好 | 指标方向测试、常量相关、真实 MSE/Pearson 运行与报告 |
| accuracy/precision/recall/F1/AUC | 全部提供；precision/recall/F1 明确为宏平均，AUC 使用概率或 decision score | 原标签分类、precision 主指标报告、AUC 缺类边界测试 |
| `base_interface` 与 `apps/general_interface`：快捷组合入口 | 统一 CLI + 模型预设 + 搜索 JSON；不逐个复制重复 Python 函数签名 | CLI、预设和组合训练测试 |
| `apps/audio/face/gait/text`：领域接口 | 保留模型集合及可配置处理能力；输入仍是数值特征表 | AST 清单与领域预设执行验证 |
| 多目标/批量预测（Skill 外部包装能力） | 每目标独立运行、输出不同目录；`predict` 对整个 CSV 批量预测，`--probabilities` 导出原类别概率；原 INI 多目标入口保留 | 旧 CLI 样本对齐/预测测试，新模型批量预测及概率和为 1 的测试 |
| `util/data_utils`、JSON 保存、demo、DEBUG | 数据对齐/结果提取由本地脚本实现；演示由 fixtures/smoke 替代，不保留全局 DEBUG 改写训练参数 | CLI 数据验证、结果读取、smoke 工作流 |

## 明确弃用的行为及原因

1. **全数据先筛选/缩放/重采样再 CV**：会泄漏验证信息，全部改为训练折内执行。
2. **原 F 检验跳过最后一列**：源码循环 `range(0, len(columns)-1)` 是错误，已修正。
3. **归一化只保存特征名单**：无法还原训练时缩放，改为保存完整变换器。
4. **原巨大网格作为默认值**：含 `min_samples_split=1`、`oob_score=True` 配合
   `bootstrap=False`、不兼容的 LogisticRegression 参数组合、可能要求非负目标的
   Poisson loss 等；部分网格有数百万组合。改为小网格和显式条件 JSON，不默默套用。
5. **固定 30 维 PCA/逐步筛选前隐式 PCA**：小样本容易失败，也改变特征含义。
   PCA 维数由用户明确配置，不再根据隐藏常量自动改变研究问题。
6. **把常量预测的 Pearson r 写成 0**：相关系数未定义，保留空值并说明原因。
   选择 Pearson r 时常量候选失败；常量 dummy 无相关指标或相关增益。
7. **用 Pearson p 值调参**：p 值依赖样本量及独立性假设，不是泛化误差，搜索后
   也不能当作有效显著性检验。保留 r 与误差指标，不提供 p 值搜索。
8. **负向 R2、用离散预测标签计算 AUC、隐含二分类 precision/recall/F1**：分别
   改正方向、使用连续分数、明确宏平均及类别范围。旧分数不可直接作数值对照。
9. **基于名字直接续跑、自动清空输出目录**：可能混用不同数据/配置或删除结果。
   改为身份校验续跑与新目录；续跑只重用已完成 CV 折，最终拟合和报告重新生成。
10. **复制领域接口的重复/失效函数**：包内部分入口使用失效绝对导入、缺少输出
    路径或同名函数覆盖。统一配置入口保留有效能力，不保留这些错误函数签名。
11. **宣称音频/人脸/步态/文本原始特征提取**：wheel 的领域入口仅接受 X/Y
    特征与标签；包内并无对应原始媒体提取引擎。这不是迁移缺失。

## 验证边界

运行 `python -m unittest discover -s tests -v` 与 `python tests/smoke_pipeline.py`。
迁移专项在 `tests/test_migration.py`。CI 配置覆盖三系统、Python 3.12/3.13/3.14；
本次本机验证记录如下，不将 CI 配置等同于本次已运行结果。
数值行为按新验证设计验证，不要求重现旧版有泄漏的分数。随机种子不保证跨库版本
或操作系统逐位一致。重采样可能因少数类样本不足失败，失败会记录而不会自动换方法。
选优 CV（包括搜索后的 OOF）有选择偏差，报告的最终评估来自留出集。

新增源码为本地实现，基于算法库 API；未复制厂商训练或参数网格源码。
wheel 本体及许可说明保持原样。

## 本次本机验证记录

2026-09-14，macOS arm64：

- 新建独立 CPython 3.12.13 虚拟环境，仅从 requirements.txt 安装 30 个依赖包，
  确认 `ccpl_training_models` 不存在，依赖一致性检查通过。
- 完整测试发现 60 项：59 项通过，1 项原厂工厂对比因没有安装 wheel 跳过。
  在已有 CPython 3.14 环境单独执行模型注册表 4 项测试，包含原厂对比，全部通过。
- 合成数据真实随机回归：8 个网格候选、MinMax、筛选与 PCA、MSE 主指标，
  完成预测、7 张图（各 PDF/SVG/PNG）及中文 Word 报告。
- 合成数据真实分组分类：4 个随机搜索候选、SMOTE、宏平均 precision 主指标，
  完成预测、11 张图及中文 Word 报告。
- 合成数据真实时间回归：逐步前向筛选、时间 gap、Pearson r 主指标，
  完成预测、7 张图及中文 Word 报告；时间设计按约定不生成 bootstrap 区间。
- Skill 元数据校验及 `git diff --check` 通过。

合成数据只验证运行和统计边界，不能作为真实研究的效果证据。本次没有在
Windows/Linux 或 Python 3.13 上本地实跑；相关 CI 仍需在推送后执行。
