# psytrainer-ml

[English](README.md) | [简体中文](README.zh-CN.md)

用于表格数据分类、回归和批量预测的 PsyClaw / Claude code /Codex Skill。大语言模型负责理解任务和调用工具，本地 Python 脚本负责训练、验证、分析、绘图及生成 Word 报告。

## 能做什么

- 使用 Pipeline 在每个训练折内拟合预处理，支持缺失值填补、标准化、特征筛选、PCA 和分类重采样。
- 支持随机、分组、时间交叉验证，以及内部留出测试集或单独提供的外部测试集。
- 比较候选模型，分析训练与验证差距、相对基线的收益、误差结构、校准、特征依赖、输入分布变化和分组/时间稳定性。
- 输出 PDF、可编辑 SVG、600 dpi PNG 图表，以及中文或英文的 `report.docx` 和 `report.md`。
- 保存包含预处理的模型、预测结果、数据划分、指标和来源记录，便于复核。
- 支持有预算的网格/随机搜索、逐步前向与 F 阈值筛选、七种分类重采样，以及校验数据和配置后续跑。

旧 wheel 的功能已逐项审计，迁移入口、测试与弃用原因见[功能迁移审计](references/wheel-audit.md)。原有模型与领域模型集合保留，无效默认网格和可能泄漏验证信息的处理方式已替换。

图表和报告的设计参考已融入本项目，使用者无需另装报告绘图类 Skill。

**完整下载 Skill 不等于安装了运行环境。** Skill 包包含全部项目文件，首次安装仍需联网下载第三方 Python 依赖。包内不包含 Python 解释器和系统共享库。

## 让大语言模型帮助安装

可以向具有本地终端和联网能力的编码助手发送：

```text
请从 https://github.com/Exekiel179/psytrainer-ml-skill 安装完整的 psytrainer-ml Skill，
放入当前宿主能发现的技能目录。保留 SKILL.md、scripts、references、requirements.txt 和 vendor 等完整内容。
阅读 README.zh-CN.md，检查 CPython 3.12、3.13 或 3.14，执行 scripts/install_runtime.py 或 Windows 包装脚本。
不要停在下载说明文件这一步。使用 runtime.json 中记录的 Python 运行 tests/smoke_pipeline.py，
验证真实训练、预测、图表和 Word 报告。完成后告诉我安装位置、实际测试结果和未解决的问题。
```

助手按这个流程下载整个仓库或完整发布包时，会一起取得 `vendor/` 内的 wheel。宿主的“安装 Skill”功能是否会执行 Python 安装器，取决于宿主实现；本项目没有自动执行安装器的宿主安装钩子。

## 安装

推荐流程：**将完整目录放入技能目录，运行一次安装器，再完成真实运行验证。**

在 [GitHub Releases](https://github.com/Exekiel179/psytrainer-ml-skill/releases/latest) 下载 **`psytrainer-ml-skill.zip`**，完整解压后运行安装器即可。使用 Git 克隆仓库时，无需另下载发布附件。

### 1. 准备 Python 并选择目录

新 Pipeline 支持 **CPython 3.12、3.13、3.14**。可从 [Python 官网](https://www.python.org/downloads/)安装；已安装 `uv` 的用户也可以执行 `uv python install 3.12`。安装器会查找支持的解释器，但不会自动下载 Python。只有显式安装原 INI 兼容引擎时，内置 wheel 才要求 CPython 3.14。

| 使用方式 | Skill 的最终目录 |
|---|---|
| Codex，当前用户的所有项目 | `~/.agents/skills/psytrainer-ml` |
| Codex，仅当前项目 | `<项目>/.agents/skills/psytrainer-ml` |
| PsyClaw，当前用户 | `~/.psyclaw/skills/psytrainer-ml` |

Codex 目录规则见[官方文档](https://learn.chatgpt.com/docs/build-skills)。Windows 的 `~` 指用户目录；其他宿主请使用其配置的技能目录。安装器只负责 Python 环境，不会替你注册或启用宿主中的 Skill。

### 2. 完整下载并安装依赖

下面以 Codex 用户技能目录为例，首次安装需要 Git。使用其他宿主时替换目标路径。

macOS / Linux：

```bash
mkdir -p "$HOME/.agents/skills"
git clone https://github.com/Exekiel179/psytrainer-ml-skill "$HOME/.agents/skills/psytrainer-ml"
cd "$HOME/.agents/skills/psytrainer-ml"
python3 scripts/install_runtime.py
```

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force "$HOME/.agents/skills" | Out-Null
git clone https://github.com/Exekiel179/psytrainer-ml-skill "$HOME/.agents/skills/psytrainer-ml"
Set-Location "$HOME/.agents/skills/psytrainer-ml"
powershell -ExecutionPolicy Bypass -File scripts\install_runtime.ps1
```

Windows CMD 用户在 Skill 目录内执行 `scripts\install_runtime.cmd`。

不使用 Git：在 [GitHub Releases](https://github.com/Exekiel179/psytrainer-ml-skill/releases/latest) 下载 `psytrainer-ml-skill.zip`，将完整的 `psytrainer-ml` 文件夹解压到上述位置，再进入该目录执行安装器，跳过 `git clone`。确保路径是 `psytrainer-ml/SKILL.md`，不要多嵌套一层同名目录，也不要只下载 `SKILL.md`。

默认安装器创建 `.venv`，一次性解析并安装所有 Pipeline 依赖，包括全部 21 个模型使用的算法库。它运行 `pip check`，导入必要模块并构造全部估计器，成功后才写入 `ready: true`、`capabilities.pipeline: true` 的 `runtime.json`。默认不会读取或安装 PsyTrainer wheel；任何必需依赖失败都不会留下成功标记。

macOS 如遇 LightGBM 的 OpenMP 动态库错误，需要安装系统库 `brew install libomp` 后重试。安装过程中会显示原始错误。

### 3. 验证训练、预测和报告

在安装后的 Skill 目录执行：

```bash
# macOS / Linux
.venv/bin/python tests/smoke_pipeline.py
```

```powershell
# Windows PowerShell
& .\.venv\Scripts\python.exe tests\smoke_pipeline.py
```

测试会在临时目录生成合成数据，执行随机、分组和时间验证，检查真实预测、诊断图表及 Word 报告。测试结束后临时结果自动清理。`ready: true` 代表依赖与导入检查通过；这个测试进一步验证完整工作流。`--dry-run` 只检查数据形状，不能代替真实运行。

Codex 会自动发现技能；若未出现，重启 Codex 后再检查。可用 `$psytrainer-ml` 指定本 Skill；PsyClaw 中使用 `/skill:psytrainer-ml`。

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

更新完整 Skill 文件后，在原目录重跑安装器，即可补齐新增依赖。Git 安装可先运行 `git pull --ff-only`；ZIP 安装应使用新版本的完整内容，并保留自己的数据和结果。

| 现象 | 处理方式 |
|---|---|
| 宿主找不到 Skill | 检查宿主技能目录与 `psytrainer-ml/SKILL.md` 层级；安装器不负责技能发现 |
| 兼容模式的 wheel 缺失或校验失败 | 重新取得完整包并保留 `vendor/`；默认 Pipeline 不需要 wheel |
| 找不到支持的 Python | 安装 CPython 3.12、3.13 或 3.14，或用 `--python /path/to/python` 指定 |
| 希望切换 `.venv` 的 Python 版本 | 用 `--python PATH --recreate`；PowerShell 使用 `-Python PATH -Recreate` |
| 移动目录后解释器路径失效 | 在新位置用 `--recreate` 重建环境和 `runtime.json`，不要复制其他机器的 `.venv` |

`--recreate` 会删除并重建当前模式的虚拟环境，不要把数据或结果保存在其中。普通重复安装无需这个选项。原来的 Python 3.12/3.13 环境可以直接使用。`--wheel` / `PSYTRAINER_WHEEL` 用于替换兼容引擎的 wheel，指定它们会启用兼容安装模式。

## 旧版工作流与开发验证

已有 INI 项目可以继续使用 `scripts/ml.py`、`scripts/PsyTrainer.py` 和 `scripts/batch_predict.py`。先安装可选的原引擎：

```bash
python3 scripts/install_runtime.py --legacy
```

Windows CMD 使用 `scripts\install_runtime.cmd --legacy`；PowerShell 包装脚本加 `-Legacy`。此模式校验并安装内置原始 wheel，要求 CPython 3.14，独立创建 `.venv-legacy` 与 `runtime-legacy.json`。主环境 `.venv` 和 `runtime.json` 保留。兼容命令使用 `runtime-legacy.json` 中的 Python；例如 macOS / Linux 使用 `.venv-legacy/bin/python scripts/ml.py train --config config/run.ini`。

原 INI 配置、模型目录和预处理文件继续由原入口处理，不会被自动转换或改变训练语义。安装前已有 PsyTrainer 的环境仍可继续使用，安装器不会自动卸载它。迁移对应关系、JSON 参数示例和历史模型兼容边界见[迁移说明](references/migration.md)，原命令见[命令参考](references/commands.md)与[配置参考](references/configuration.md)。

`ml.py capabilities` 查询独立 Pipeline 的 21 个模型；`ml.py capabilities --legacy` 查询原引擎模型、筛选器、重采样器和指标。新分析推荐 `pipeline_train.py`，两套入口的预处理和验证行为不同。

开发者完整验证命令（Windows 同样替换解释器路径）：

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tests/smoke_pipeline.py
# 可选原引擎安装后验证
.venv-legacy/bin/python tests/smoke_runtime.py
.venv-legacy/bin/python tests/smoke_agent.py
```

## 许可与模型文件

本 Skill 使用 [MIT 许可](LICENSE)；随包附带的原始 PsyTrainer wheel 使用其内嵌的 Apache 2.0 许可，来源、校验值和说明见 [NOTICE.md](NOTICE.md)。预测会加载 joblib/pickle 模型文件，只加载可信来源的模型。
