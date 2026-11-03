# 实验台账

> 由负责实验记录的人维护。数字必须来自 `outputs/<run>/metrics.json`，禁止手抄。
> 每次实验一行；失败也登记。

## 评测协议

- 主指标：mAP；辅助：CMC Rank-1 / 5 / 10。
- 特征 L2 归一化，余弦距离；对每个 query 剔除同 ID 同摄像头的 gallery 样本。
- 三个数据集都使用官方 query/gallery 划分；Move/CCVID 的适配方式记在 `docs/DATA.md`。
- 报告保留 1 位小数；`metrics.json` 保留 6 位。

## 台账

| Run ID | 日期 | 分支/commit | 配置 | 数据版本 | 主指标 | 对比基线 | 结论 |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — |
| transreid-market | 2026-11-03 | feat/transreid-baseline @ `973c6ef` | `configs/transreid/market1501.yml`，batch 8 | Market1501 官方划分，manifest 见 `outputs/transreid-market/config.yaml` | [metrics.json](../outputs/transreid-market/metrics.json)：mAP 0.871810；Rank-1 0.946853；Rank-5 0.982482；Rank-10 0.991983 | TransReID Baseline (ViT) 官方 Market 权重 | 与官方 87.1/94.6% 相差 +0.081/+0.085 个百分点，达到 baseline 验收；T4 推理 178.865 秒 |
| 20261101-1625-clipreid-ccvid-multiframe-25588451 | 2026-11-01 | exp/c-ccvid-multiframe，版本见 git.txt | configs/clipreid/ccvid_multiframe.yml，--frames 1 | CCVID 官方划分（General） | [metrics.json](../outputs/20261101-1625-clipreid-ccvid-multiframe-25588451/metrics.json) | 20261101-1555-clipreid-ccvid-single-frame-33182134 | 特征、距离矩阵和指标与基线完全一致 |
| 20261101-1628-clipreid-ccvid-multiframe-29605161 | 2026-11-01 | exp/c-ccvid-multiframe，版本见 git.txt | configs/clipreid/ccvid_multiframe.yml，--frames 4 | CCVID 官方划分（General） | [metrics.json](../outputs/20261101-1628-clipreid-ccvid-multiframe-29605161/metrics.json) | 20261101-1625-clipreid-ccvid-multiframe-25588451 | 仅改变采样帧数，mAP、Rank-1/5/10 均提升，耗时增加 |
| 20261101-1629-clipreid-ccvid-multiframe-41578689 | 2026-11-01 | exp/c-ccvid-multiframe，版本见 git.txt | configs/clipreid/ccvid_multiframe.yml，--frames 8 | CCVID 官方划分（General） | [metrics.json](../outputs/20261101-1629-clipreid-ccvid-multiframe-41578689/metrics.json) | 20261101-1628-clipreid-ccvid-multiframe-29605161 | 仅改变采样帧数，mAP 略低，Rank-1 相同，耗时增加 |

## 约定

- Run ID = `YYYYMMDD-HHMM-<name>`，与 `outputs/` 目录同名。
- 主指标定义见 `docs/PROJECT.md`。
- 结论只写可验证的一句话；详细分析放 `reports/` 或 `docs/notes/`。
- 失败实验标记 `failed`，写明失败原因（OOM/NaN/中断）。
- 每次运行至少保留：`config.yaml`、`test_log.txt`、`metrics.json`、`env.txt`、`git.txt`。
