# 源码

当前模块（`src/reid/`）：

- `manifest.py` — manifest 的 JSON schema 与 `Sample` / `DatasetManifest` 容器已定义；
  Market1501 解析与 `verify_manifest` 由 A 实现（TODO）。
- `metrics.py` — `RankMetrics` 输出结构与 `evaluate_rank` 接口已定义；
  mAP/CMC 算法由 A 实现（TODO）。
- `logparse.py` — 日志解析与 payload 构建接口已定义；由 B 实现（TODO）。
- 规划中：`adapters/`（Move/CCVID → manifest）、`features.py`（特征提取/距离矩阵）。

规则：

- 入口脚本只做编排；核心逻辑写成可测试函数。
- 指标、数据解析、日志解析必须有单测。
- 不直接改 `third_party/`；上游兼容改动放 `patches/`。
