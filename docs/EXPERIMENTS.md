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

## 约定

- Run ID = `YYYYMMDD-HHMM-<name>`，与 `outputs/` 目录同名。
- 主指标定义见 `docs/PROJECT.md`。
- 结论只写可验证的一句话；详细分析放 `reports/` 或 `docs/notes/`。
- 失败实验标记 `failed`，写明失败原因（OOM/NaN/中断）。
- 每次运行至少保留：`config.yaml`、`test_log.txt`、`metrics.json`、`env.txt`、`git.txt`。
