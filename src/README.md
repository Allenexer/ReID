# 源码

`src/reid/` 是各成员实现的 Python 包；当前只保留职责占位：

- `manifest.py` — 数据集 manifest 生成与校验（A）
- `metrics.py` — mAP/CMC 评测（A）
- `logparse.py` — 上游日志解析（B）

接口、数据结构、命令行参数与实现均由对应负责人设计；跨成员使用时由双方协商，
并在 `docs/DECISIONS.md` 记录。规划中：`adapters/`（Move/CCVID → manifest）、
`features.py`（特征提取/距离矩阵）。

规则：

- 入口脚本只做编排；核心逻辑写成可测试函数。
- 指标、数据解析、日志解析由实现者补充单测。
- 不直接改 `third_party/`；上游兼容改动放 `patches/`。
