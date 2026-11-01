# 行人重识别（ReID）课程项目

给定 query 行人图像，在不同摄像头、时间、视角的 gallery 中按身份相似度排序。
4 人小组，交付代码 + 报告。主评测集 Market1501，课程验证集 Move，换衣加分集 CCVID。

本仓库是小组的共享框架；具体实现由各成员按分工推进，文档可以按组内约定调整。

## 当前状态

- M0 项目初始化：完成
- M1 数据与评测：A 的 manifest、统一指标和归档评测已实现；真实模型矩阵验收待权重与运行产物
- 基线：TransReID（主）、CLIP-ReID（对比），已固定 commit 并提供兼容补丁
- 环境：搭建步骤与版本见 `docs/SETUP.md`（torch 2.5.x + torchvision 0.20.1、
  timm/yacs）；已在一台机器上验证，各成员各自在本机搭建

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

# 生成 CCVID 官方划分清单（原始数据只读）
conda run -n rl_v2 python scripts/check_dataset.py --dataset ccvid \
  --root data/raw/ccvid --manifest data/processed/manifests/ccvid.json

# 对 C/B 的同一次成功归档复算，结果写入新目录
conda run -n rl_v2 python scripts/evaluate.py --run-dir outputs/<run> \
  --protocol ccvid_general --distance native \
  --output-dir outputs/<run>/evaluation/ccvid_general/native

# 同时发现原有测试和工作区根目录 test/ 中的 A 测试
conda run -n rl_v2 python -m pytest -c pyproject.toml tests ../test -ra
```

A 的接口、三种 CCVID 协议、距离统一、测试依赖及 B/D 交接见
[`docs/A_EVALUATION.md`](docs/A_EVALUATION.md)。`../test/` 位于本 Git 仓库外，
提交或拷贝项目时须独立携带，不另复制一套。

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
| B | TransReID 基线 | 权重、Market1501/Move 结果 |
| C | CLIP-ReID + 加分 | 第二基线、CCVID、遮挡/换衣实验 |
| D | 集成与报告 | 实验入口、复现、图表、报告 |

## 文档

- [项目定义](docs/PROJECT.md)
- [工程规范](docs/CONVENTIONS.md)
- [数据说明](docs/DATA.md)
- [基线部署](docs/BASELINES.md)
- [环境搭建](docs/SETUP.md)
- [路线图](docs/ROADMAP.md)
- [实验台账](docs/EXPERIMENTS.md)
