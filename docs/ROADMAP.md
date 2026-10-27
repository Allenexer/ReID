# 路线图与任务板

> 周目标由小组在 GitHub Issues 里拆；本文件只记录里程碑和关键路径。
> 分工建议：A 数据与评测 / B TransReID / C CLIP-ReID + 加分 / D 集成与报告。
> 角色可以互换；具体任务以组内实际安排为准。

## 里程碑

| ID | 里程碑 | 目标 | 建议截止 | 状态 |
|---|---|---|---|---|
| M0 | 项目初始化 | 仓库、环境、任务定义 | 2026-10-27 | 完成 |
| M1 | 数据与评测就绪 | Market1501 可用；manifest/指标接口冻结 | 2026-11-02 | 进行中 |
| M2 | 基线跑通 | TransReID + CLIP-ReID 在 Market1501 可复现 | 2026-11-09 | 未开始 |
| M3 | 三数据集覆盖 | Move、CCVID 适配完成，3×2 组合有结果 | 2026-11-16 | 未开始 |
| M4 | 调优与加分 | 训练/调参、遮挡/换衣改进有对照结论 | 2026-11-23 | 未开始 |
| M5 | 报告与交付 | 报告、图表、复现说明、最终打包 | TBD | 未开始 |

## 当前迭代（W1：2026-10-27 → 2026-11-02）

**目标**：数据接口与指标实现冻结；Market1501 跑通至少一个基线。

| 任务 | 建议 owner | 状态 | 验收标准 |
|---|---|---|---|
| 环境与依赖锁定 | B | 进行中 | `.venv` + `pip freeze`；torch CUDA 可用 |
| Market1501 下载、校验、manifest | A | 待开始 | 3368 query / 750 query ID；`check_dataset` 通过；行序与上游 glob 一致 |
| mAP/CMC 实现与上游对拍 | A | 进行中 | 单测通过；同一 distmat 与上游误差 < 0.1% |
| 跨基线统一评测路径 | A | 待开始 | B/C 数字出自同一套评测（保存 distmat 重算，或记录日志对拍） |
| 数据台账 `data/README.md` | A | 待开始 | 来源、大小、校验和、下载步骤齐全 |
| TransReID Market1501 基线 | B | 待开始 | 有 `metrics.json`；与上游报告值差 ≤1pp |
| CLIP-ReID Market1501 基线 | C | 待开始 | 有 `metrics.json`；与上游报告值差 ≤1pp |
| 统一实验入口 `scripts/run_baseline.py` | D | 待开始 | 一条命令生成 run 目录、日志、`metrics.json` |

## 后续关键路径

1. **Market1501 全链路**：A 的 manifest → B/C 跑基线并保存 distmat/日志 → A 统一出数 → D 汇总。
2. **Move 适配（作业硬性要求）**：修改 test.py 让两个基线在课程测试集上运行；
   等课程数据；A+B 先确定目录结构和 test-only 方案。
3. **CCVID（加分）**：C 先做单帧 baseline，再做多帧聚合。
4. **训练/调参（加分，B 主线）**：Market1501 训练集，优先 Colab；
   先 smoke test / overfit 单 batch；断点续训防免费会话超时。
5. **线性分类器小实验（作业推荐方法，D）**：query + 3 候选的相似度与匹配概率，
   复用 A 的评测，产出报告小节。

## 风险与阻塞

| 风险 | 影响 | 应对 | Owner |
|---|---|---|---|
| Move 数据未发布 | 三数据集覆盖延期 | 先用 Market1501 跑通全链路，适配器接口预留 | A |
| CCVID 视频格式复杂 | 加分项延期 | 先做单帧 baseline，再做多帧聚合 | C |
| 4GB 显存 OOM | 推理/训练中断 | batch 从 8 起；训练改用 Colab | B/C |
| 共用一台 GPU | B/C 排队 | 错峰跑基线；Move/CCVID 用 Colab 或各自机器 | B/C |
| 上游代码旧 | 无法导入/报错 | 固定 commit + `patches/`；锁 torch 2.5.x | B |
| 权重下载失败 | 基线无法复现 | 提前下载到 `checkpoints/`；记录来源与校验和 | B/C |
| 指标实现不一致 | 报告数字错误 | A 的本地实现与上游对拍；D 只读 `metrics.json` | A/D |

状态：待开始 / 进行中 / 待 review / 完成 / 阻塞
