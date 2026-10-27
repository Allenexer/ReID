# 数据说明

> 本目录内容默认不入库（README 除外）。`raw/` 只读；`processed/` 可重建。

## 数据集登记

| 数据集 | 来源/链接 | 角色 | 规模 | 下载日期 | 校验和 | 本地路径 | 状态 |
|---|---|---|---|---|---|---|---|
| Market1501 | <https://www.kaggle.com/datasets/pengcw1/market-1501> | 主评测，可训练 | train 751 ID/12936；query 750 ID/3368 | TBD | TBD | `data/raw/market1501/` | 待下载 |
| Move | 课程提供 | 验证，仅测试 | 200 ID/800 query，含遮挡 | TBD | TBD | `data/raw/move/` | 待发布 |
| CCVID | <https://github.com/ziyuanzhangtony/GaitNet-CVPR2019> | 换衣加分 | 以官方为准 | TBD | TBD | `data/raw/ccvid/` | 待下载 |

## 重建步骤

```bash
# Market1501：下载解压到 data/raw/ 后
py -3 -m scripts.check_dataset --config configs/datasets/market1501.yaml
```

详细说明见 `docs/DATA.md`。
