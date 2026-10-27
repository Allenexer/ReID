# 源码

当前模块（`src/reid/`）：

- `manifest.py` — 数据集 manifest：`(path, pid, camid)` 的 query/gallery/train 清单；
  已支持 Market1501 官方目录，Move/CCVID 适配器后续加入。
- `metrics.py` — mAP / CMC 的独立实现，输入 query×gallery 距离矩阵。
- `logparse.py` — 从 TransReID/CLIP-ReID 日志解析 mAP、Rank-1/5/10。
- 规划中：`adapters/`（Move/CCVID → manifest）、`features.py`（特征提取/距离矩阵）。

规则：

- 入口脚本只做编排；核心逻辑写成可测试函数。
- 指标、数据解析、日志解析必须有单测。
- 不直接改 `third_party/`；上游兼容改动放 `patches/`。
