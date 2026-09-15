# psytrainer-ml

[English](README.md) | [简体中文](README.zh-CN.md)

用于表格数据分类、回归和批量预测的 PsyClaw / Claude Code / Codex Skill。大语言模型负责理解任务和调用工具，本地 Python 脚本负责训练、验证、分析、绘图及生成 Word 报告。

## 能做什么

- 使用 Pipeline 在每个训练折内拟合预处理，支持缺失值填补、标准化、特征筛选、PCA 和分类重采样。
- 支持随机、分组、时间交叉验证，以及内部留出测试集或单独提供的外部测试集。
- 比较候选模型，分析训练与验证差距、相对基线的收益、误差结构、校准、特征依赖、输入分布变化和分组/时间稳定性。
- 输出 PDF、可编辑 SVG、600 dpi PNG 图表，以及中文或英文的 `report.docx` 和 `report.md`。
- 报告面向无机器学习背景的读者，解释任务、数据划分、预处理和指标，并逐图说明读法、本次实测结果及解释边界。
- 保存包含预处理的模型、预测结果、数据划分、指标和来源记录，便于复核。
- 支持有预算的网格/随机搜索、逐步前向与 F 阈值筛选、七种分类重采样，以及校验数据和配置后续跑。

支持 9 个分类模型和 12 个回归模型，提供领域模型集合与自定义参数配置。

### 本地数据处理能力

以下处理由本项目脚本直接调用算法库实现。

| 处理 | 参数与行为 |
|---|---|
| 缺失值 | 默认拒绝；`--impute median` 在训练折拟合中位数填补 |
| 标准化与归一化 | 默认 StandardScaler；`--scaler minmax` 使用 MinMaxScaler；`--no-scale` 关闭 |
| 单变量特征筛选 | `--select-k N` 保留前 N 个特征；`--selection f-threshold --f-threshold 5` 按 F 值筛选 |
| 逐步前向筛选 | `--selection forward --select-k N`，内部交叉验证保留分组或时间边界 |
| PCA 降维 | `--pca N`，与填补、缩放和筛选一起保存在模型中，预测时复用 |
| 分类重采样 | `--resample` 支持 `random-over`、`smote`、`smote-tomek`、`smote-enn`、`cluster-centroids`、`random-under`、`near-miss`，仅训练时执行 |
| 参数和预处理组合搜索 | `--search grid` / `random`，`--search-space` 配置条件组合，`--max-candidates` 限制每个模型的候选数 |
| 领域模型集合 | `--preset audio` / `face` / `gait` / `text` 使用对应的回归模型集合；输入是已提取的数值特征，不是原始媒体 |
| 续跑与模型保存 | `--resume` 校验输入、配置和版本后复用 CV 折；`--save-candidates` 保存各模型的最佳配置 |

所有学习型预处理均在对应训练折拟合；独立测试集不参与筛选和搜索。图表与 Word 报告记录实际采用的处理、搜索结果、验证边界和失败项。

图表和报告的设计参考已融入本项目，使用者无需另装报告绘图类 Skill。

Skill 发布包包含运行所需的指令、脚本和配置。首次配置只创建虚拟环境，第三方依赖在执行任务时按需补装。Python 解释器和系统共享库需在本机准备。

## 让大语言模型帮助安装

可以向具有本地终端和联网能力的编码助手发送：

```text
安装 https://github.com/Exekiel179/psytrainer-ml-skill 的最新 Release Skill 包，按 references/setup.md 配置。
```

Claude Code 可直接用这条，避免安装到其他宿主目录：

```text
将 https://github.com/Exekiel179/psytrainer-ml-skill 的最新 Release Skill 包安装到 ~/.claude/skills/psytrainer-ml，执行 scripts/install_runtime.py。
```

智能体使用[短安装指南](references/setup.md)，执行任务时读取 `SKILL.md`，其他参考按需读取。详细 Word 解读由本地脚本生成，无需反复调用模型或将整份报告读入上下文。

安装只需将 Skill 放入宿主技能目录并运行一次环境配置脚本，无需提前下载全部算法库，也不运行样例训练。每次使用前，任务入口自动检查所需依赖，缺少或版本不兼容时才补装。

## 安装

推荐流程：**下载 Release Skill 包，解压到技能目录，运行环境配置脚本。**

在 [GitHub Releases](https://github.com/Exekiel179/psytrainer-ml-skill/releases/latest) 下载 **`psytrainer-ml-skill.zip`**。此包不携带开发测试和样例数据；Git 克隆适用于需要开发源码的用户，会包含测试文件。

### 1. 准备 Python 并选择目录

支持 **CPython 3.12、3.13、3.14**。可从 [Python 官网](https://www.python.org/downloads/)安装；已安装 `uv` 的用户也可以执行 `uv python install 3.12`。运行环境配置脚本会查找已安装的受支持解释器。

| 使用方式 | Skill 的最终目录 |
|---|---|
| Claude Code，当前用户的所有项目 | `~/.claude/skills/psytrainer-ml` |
| Claude Code，仅当前项目 | `<项目>/.claude/skills/psytrainer-ml` |
| Codex，当前用户的所有项目 | `~/.agents/skills/psytrainer-ml` |
| Codex，仅当前项目 | `<项目>/.agents/skills/psytrainer-ml` |
| PsyClaw，当前用户 | `~/.psyclaw/skills/psytrainer-ml` |

Codex 目录规则见[官方文档](https://learn.chatgpt.com/docs/build-skills)。Windows 的 `~` 指用户目录；其他宿主请使用其配置的技能目录。将完整 Skill 放入对应目录后，按宿主的发现或启用机制加载它。

Claude Code 使用 `.claude/skills`，不能照搬 Codex 的 `.agents/skills`。仅将本仓库下载到普通项目目录，或运行 Python 环境脚本，并不会让 Claude Code 发现它。目录规则见 [Claude Code 官方说明](https://code.claude.com/docs/en/skills#choose-where-skills-load)。

### 2. 放置 Skill 并配置运行环境

将 ZIP 中的完整 `psytrainer-ml` 文件夹解压到选定目录。下面以 Claude Code 用户技能目录为例；Codex 换成 `$HOME/.agents/skills/psytrainer-ml`，PsyClaw 换成 `$HOME/.psyclaw/skills/psytrainer-ml`。无需 Git。

macOS / Linux：

```bash
cd "$HOME/.claude/skills/psytrainer-ml"
python3 scripts/install_runtime.py
```

Windows PowerShell：

```powershell
Set-Location "$HOME/.claude/skills/psytrainer-ml"
powershell -ExecutionPolicy Bypass -File scripts\install_runtime.ps1
```

Windows CMD 用户在 Skill 目录内执行 `scripts\install_runtime.cmd`。

确保路径是 `psytrainer-ml/SKILL.md`，不要多嵌套一层同名目录，也不要只下载 `SKILL.md`。

运行环境配置脚本默认只创建 `.venv` 并记录解释器路径，不下载模型依赖。`runtime.json` 的 `ready: true` 表示环境已创建；每次执行任务都会重新检查本次需要的依赖及版本，不以旧标记代替检查。

默认训练只准备基础建模、绘图和 Word 报告依赖；选用 LightGBM、XGBoost、CatBoost 时才安装相应库，使用重采样时才安装 imbalanced-learn。预测按所加载模型补齐依赖，重新生成报告不安装无关模型库。已有包满足要求时不联网下载。主动预装某个包可用 `--packages xgboost`；只有明确需要完整预装或发布测试时才用 `--all`。

macOS 如遇 LightGBM 的 OpenMP 动态库错误，需要安装系统库 `brew install libomp` 后重试。安装过程中会显示原始错误。

### 3. 加载 Skill

Claude Code 中输入 `/psytrainer-ml`；Codex 中用 `$psytrainer-ml`；PsyClaw 中用 `/skill:psytrainer-ml`。Claude Code 也可根据“CSV 分类回归、批量预测、Word 分析报告”等任务描述自动调用。

Claude Code 未识别时，确认 `~/.claude/skills/psytrainer-ml/SKILL.md` 存在，打开新会话后输入 `/psytrainer-ml`。项目级安装只在对应项目范围内生效。若目录正确仍不显示，检查 Claude Code 的技能禁用配置或同名技能；依赖重装不能解决技能发现问题。

### 网络慢或中断

在 Skill 目录执行，国内网络可指定[清华 PyPI 镜像](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)：

```bash
python3 scripts/install_runtime.py --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_runtime.ps1 -IndexUrl https://pypi.tuna.tsinghua.edu.cn/simple
```

- 默认使用 pip，保留其缓存和源配置；指定的镜像、安装工具和超时参数会记入当前 Skill 的 `runtime.json`，后续按需补装沿用，不修改系统全局配置。
- 已安装 uv 时可加 `--installer uv`（PowerShell：`-Installer uv`），使用其并发下载和缓存。uv 有独立的源配置与缓存，不读取 pip 配置；需要镜像时同时传入 `--index-url`。
- 默认网络超时 20 秒、重试 2 次，单次依赖安装最多 600 秒；可用 `--timeout`、`--retries`、`--max-seconds` 调整。PowerShell 对应 `-Timeout`、`-Retries`、`-MaxSeconds`。
- 中断后重跑相同命令，复用已安装依赖和已完成的缓存下载；未完成的大文件可能需要重新下载。无需 `--recreate`。
- 只检查指定包：`python3 scripts/install_runtime.py --check --packages xgboost`，不联网、不下载；缺失或冲突会列出补装命令。单独 `--check` 只检查环境，`--check --all` 检查全部依赖。PowerShell 用 `-Check -Packages xgboost` 或 `-Check -All`。
- 已有匹配平台和 Python 版本的依赖目录时，用 `--wheelhouse PATH`（PowerShell：`-Wheelhouse PATH`）离线安装。

安装只接受预编译包，避免弱网下进入耗时的源码构建；没有匹配包时会明确失败。完整训练、预测和报告测试由维护者在发布前执行。

**下载失败必须给出补装路径。** 任务会停止并列出所需包、目标解释器和完整安装命令。用户可以自己执行，也可以让编码助手代为执行，例如：

```bash
python3 scripts/install_runtime.py --packages xgboost --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

PowerShell 使用 `-Packages xgboost -IndexUrl https://pypi.tuna.tsinghua.edu.cn/simple`。也可用 `--wheelhouse PATH` 从本地目录安装。完成后重跑原任务；助手不能只报告网络失败而省略这些操作说明，也不能跳过依赖继续声称成功。

## 开始分析

安装后可直接告诉助手：

```text
使用 psytrainer-ml，读取我的 features.csv 和 labels.csv，预测 score，做回归分析。
输出独立留出评估、诊断图表和中文 Word 报告。
```

每个 CSV 的第一列必须是唯一的样本 ID；特征和标签的 ID 必须一致，标签会按 ID 对齐。特征必须是数值，目标不能缺失。一次运行一个目标。分组或时间元数据单独提供，不能误当作预测特征。

也可直接运行命令。以下命令从 Skill 目录执行，`data/` 为你自己的数据路径，跨项目使用时建议填写绝对路径；输出目录应使用新目录。

```bash
.venv/bin/python scripts/pipeline_train.py train --features data/features.csv --labels data/labels.csv --task regression --target score --output-dir outputs/run-01
.venv/bin/python scripts/pipeline_train.py predict --features data/new.csv --model outputs/run-01/pipeline.joblib --output outputs/predictions.csv
```

Windows PowerShell 将命令开头的 `.venv/bin/python` 换成 `& .\.venv\Scripts\python.exe`，其他参数相同。

默认先留出 20% 测试数据，再在开发集上做 5 折交叉验证、比较两个候选模型，最后输出中文报告。常用选项：

| 需求 | 参数 |
|---|---|
| 分类 | `--task classification --target diagnosis` |
| 按受试者或中心分组 | `--split group --metadata data/meta.csv --group-column subject` |
| 按时间验证 | `--split time --metadata data/meta.csv --time-column time` |
| 外部测试集 | `--test-features data/test-X.csv --test-labels data/test-Y.csv` |
| 显式填补缺失特征 | `--impute median` |
| 指定研究问题 | `--question "基线指标能否预测后续评分？"` |
| 英文报告 | `--language en` |
| 配置算法参数 | `--model-params config/models.json`，内容按模型名称映射到参数对象 |
| 搜索参数或比较预处理组合 | `--search grid` 或 `--search random --max-candidates 12`；自定义 `--search-space config/search.json` |
| 逐步前向筛选 | `--selection forward --select-k 3`，内部验证保持分组/时间约束 |
| F 值筛选 / 归一化 | `--selection f-threshold --f-threshold 5` / `--scaler minmax` |
| 领域模型 / 全部模型 | `--preset audio`、`face`、`gait`、`text` 或 `all` |
| 继续相同的中断任务 | 原命令加 `--resume`，复用已完成的 CV 折 |
| 保存所有成功候选模型 | `--save-candidates`，每个模型保存其最佳配置 |

分组/时间设计的外部测试集还需要相应元数据。完整参数、数据约束和统计边界见 [Pipeline 参考](references/pipeline.md)。

### 结果文件

| 文件 | 内容 |
|---|---|
| `report.docx` / `report.md` | 方法、实测结果、图表、局限和来源记录 |
| `figures/` | PDF、SVG、PNG 图及图注/数据来源清单 |
| `summary.json` / `analysis.json` | 指标、模型比较、数值证据和诊断结论 |
| `pipeline.joblib` | 拟合后的预处理与模型，可用于新数据预测 |
| `test-predictions.csv` / `oof-predictions.csv` | 测试集预测和开发集折外预测 |
| `split-membership.csv` / `cv-membership.csv` | 样本划分及交叉验证归属 |
| `run.log` | 详细运行日志与警告 |

报告可以用 `scripts/pipeline_train.py report outputs/run-01` 重新生成，无需重训。该命令同样使用 `.venv` 中的 Python。

图表包括验证差距、基线比较、误差或分类校准、特征依赖、输入分布变化及分组/时间稳定性，按任务和有效数据生成。不能根据测试集诊断反复调参并继续把同一测试集称为未见数据；置信区间仅在支持的条件下生成，时间设计默认不生成区间。模型重要性不代表因果关系。

## 升级与常见安装问题

更新 Skill 文件后，在原目录重跑环境配置脚本即可，依赖满足时不会再次下载。Git 安装可先运行 `git pull --ff-only`；ZIP 安装使用新版本的完整内容，并保留 `.venv`、自己的数据和结果。

| 现象 | 处理方式 |
|---|---|
| 宿主找不到 Skill | 检查宿主技能目录与 `psytrainer-ml/SKILL.md` 层级，并按宿主要求重新加载或启用技能 |
| 找不到支持的 Python | 安装 CPython 3.12、3.13 或 3.14，或用 `--python /path/to/python` 指定 |
| 希望切换 `.venv` 的 Python 版本 | 用 `--python PATH --recreate`；PowerShell 使用 `-Python PATH -Recreate` |
| 移动目录后解释器路径失效 | 在新位置用 `--recreate` 重建环境和 `runtime.json`，不要复制其他机器的 `.venv` |

`--recreate` 会删除并重建虚拟环境，不要把数据或结果保存在其中。普通重复安装无需这个选项。

## 开发验证

`scripts/ml.py capabilities` 可查询支持的模型与处理选项。

以下命令只在源码仓库执行，发布包不携带 `tests/` 和 `fixtures/`（Windows 同样替换解释器路径）：

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tests/smoke_pipeline.py
```

维护者发布版本时，按[发布流程](references/releasing.md)执行自动构建、跨平台验证和 Release 发布。

## 许可与模型文件

本 Skill 使用 [MIT 许可](LICENSE)，第三方说明见 [NOTICE.md](NOTICE.md)。预测会加载 joblib/pickle 模型文件，只加载可信来源的模型。
