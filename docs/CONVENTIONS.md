# 工程规范

> 共享仓库的底线约定，不是课程评分标准；组内可以按需要补充。
> 目标：让四个人改同一份代码时不出乱子，而不是增加流程负担。

## 1. 语言

- 文档、注释：中文。
- 代码标识符、分支名、commit message：英文。
- 报告：中文（除非课程要求英文）。

## 2. 目录

```text
src/reid/       可复用逻辑：manifest、mAP/CMC、日志解析
scripts/        一次性入口：数据校验、评测、拉取上游
configs/        数据集与实验配置
tests/          单元测试
third_party/    上游基线代码（不入库，脚本拉取）
patches/        上游兼容补丁
data/raw/       原始数据（只读，不入库）
data/processed/ 生成物（可重建，不入库）
outputs/        实验产物（不入库）
reports/        报告、图、表
```

- `src/` 放可复用逻辑；`scripts/` 只做参数解析和编排；`notebooks/` 只做探索。
- 路径用 `pathlib.Path`；配置放 `configs/`，不要硬编码超参。

## 3. Python

- 遵循 PEP 8；提交前跑 `ruff format .` 和 `ruff check --fix .`。
- 公共函数写类型标注；复杂逻辑写简短 docstring。
- 异常不要静默吞掉；日志用 `logging`，入口摘要可以用 `print`。
- 可复用逻辑必须有测试；bug 修复先写失败测试。

## 4. Git

- `main` 保持可运行；不直接在 `main` 上改。
- 每人开自己的分支：
  - `feat/<slug>`：新功能
  - `exp/<slug>`：实验
  - `fix/<slug>`：修 bug
  - `docs/<slug>`：文档
- Commit 格式：`<type>(<scope>): <summary>`，英文、祈使句、≤72 字符。
  - 示例：`feat(data): add market1501 manifest`
  - 一个 commit 只做一件事。
- 合并前：能跑、无冲突、至少 1 人看过。
- 不提交：数据、权重、`checkpoints/`、`outputs/`、`.venv/`、日志、密钥。
- 远程仓库确定后，不 force push；有问题开新 commit。

## 5. 数据

- `data/raw/` 只读，任何转换输出到 `data/processed/`。
- 使用官方 query/gallery 划分，不另造 split。
- 数据来源、下载日期、大小、校验和记录在 `data/README.md`。
- 数据集、压缩包、抽帧图片一律不入库。

## 6. 实验

- 超参来自 `configs/`；代码里不出现魔法数字。
- 每次实验固定随机种子，记录环境（Python/PyTorch/CUDA/GPU）。
- 每次运行一个目录：`outputs/YYYYMMDD-HHMM-<name>/`。
- 至少保留：`config.yaml`、`test_log.txt`、`metrics.json`、`env.txt`、`git.txt`。
- 失败实验保留并标 `failed`，不删除；结论写进 `docs/EXPERIMENTS.md`。

## 7. 报告

- 所有数字来自 `outputs/*/metrics.json`，由脚本生成到 `reports/tables/`。
- 图由脚本生成到 `reports/figures/`，文件名含实验名和指标。
- 每张表/图注明来源 run id；不手改生成的表格。
- 提交前核对：数字与台账一致、图能由命令重建。
