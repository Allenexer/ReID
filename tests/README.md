# 测试

- 运行：`py -3 -m pytest -q`（或 `pytest -q`）
- 当前状态：
  - `test_manifest.py` — 共享 JSON schema 的 save/load 往返测试通过；
    Market1501 解析测试为 skipped 占位（A 实现后启用）。
  - `test_metrics.py` — skipped 占位（A 实现 mAP/CMC 后启用）。
  - `test_logparse.py` — skipped 占位（B 实现日志解析后启用）。
- 必测：transform、指标、后处理、划分、配置解析。
- 模型：2 样本 smoke test；训练：先 overfit 单 batch。
- 测试不依赖网络与大数据。

先写失败测试，再修 bug；成员实现对应功能后，把 skipped 占位替换成真实用例。
