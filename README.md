# 行人重识别（ReID）课程项目

给定 query 行人图像，在不同摄像头、时间、视角的 gallery 中按身份相似度排序。
4 人小组，交付代码 + 报告。主评测集 Market1501，课程验证集 Move，换衣加分集 CCVID。

本仓库是小组的共享框架；具体实现由各成员按分工推进，文档可以按组内约定调整。

## 当前状态

- M0 项目初始化：完成
- M1 数据与评测就绪：进行中（manifest/指标接口已定义，A 负责实现；等待数据集与权重）
- 基线：TransReID（主）、CLIP-ReID（对比），已固定 commit 并提供兼容补丁
- 环境：项目 `.venv` 已建（复用系统 torch 2.5.1+cu121；torchvision 0.20.1、
  timm 0.9.16、yacs 已装）

## 快速开始

1. 读 [`docs/PROJECT.md`](docs/PROJECT.md)，确认目标、指标与建议分工
2. 按 [`docs/SETUP.md`](docs/SETUP.md) 搭环境
3. 按 [`docs/DATA.md`](docs/DATA.md) 准备 Market1501，生成 manifest
4. 按 [`docs/BASELINES.md`](docs/BASELINES.md) 跑 TransReID / CLIP-ReID
5. 在 GitHub Issues 或 [`docs/ROADMAP.md`](docs/ROADMAP.md) 认领任务

## 常用命令

```bash
# 拉取上游基线（固定 commit + PyTorch 2.x 补丁）
bash scripts/setup_third_party.sh

# 数据校验 + 生成 manifest（A 实现后可用）
py -3 -m scripts.check_dataset --config configs/datasets/market1501.yaml

# 从距离矩阵计算 mAP / CMC（A 实现后可用）
py -3 -m scripts.evaluate \
  --manifest data/processed/manifests/market1501.json \
  --distmat outputs/<run>/distmat.npy \
  --output outputs/<run>/metrics.json \
  --run-id <run>

# 从上游 test_log.txt 生成 metrics.json（B 实现后可用）
py -3 -m scripts.log_to_metrics \
  --log outputs/<run>/test_log.txt \
  --output outputs/<run>/metrics.json \
  --run-id <run>

# 测试与格式
py -3 -m pytest -q
ruff format . && ruff check --fix .
```

## 目录

```text
src/reid/        manifest、mAP/CMC、日志解析
scripts/         数据校验、评测、上游拉取
configs/         数据集配置（实验配置随后续任务加入）
third_party/     上游基线代码（不入库，脚本拉取）
patches/         上游兼容补丁
data/raw/        原始数据集（只读，不入库）
outputs/         实验产物（不入库）
reports/         报告、图表、表格
docs/            项目文档
```

## 分工

| 成员 | 工作流 | 主要交付 |
|---|---|---|
| A | 数据与评测 | 数据集、manifest、指标对拍 |
| B | TransReID 基线 | 环境、权重、Market1501/Move 结果 |
| C | CLIP-ReID + 加分 | 第二基线、CCVID、遮挡/换衣实验 |
| D | 集成与报告 | 实验入口、复现、图表、报告 |

## 文档

- [项目定义](docs/PROJECT.md)
- [工程规范](docs/CONVENTIONS.md)
- [数据说明](docs/DATA.md)
- [基线部署](docs/BASELINES.md)
- [环境搭建](docs/SETUP.md)
- [决策日志](docs/DECISIONS.md)
- [路线图](docs/ROADMAP.md)
- [实验台账](docs/EXPERIMENTS.md)
