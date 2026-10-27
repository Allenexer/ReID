# 测试

- 运行：`py -3 -m pytest -q`（或 `pytest -q`）
- 当前覆盖：
  - `test_metrics.py` — mAP/CMC 的手算样例、同摄像头剔除、异常输入；
  - `test_manifest.py` — Market1501 文件名解析、junk/distractor 过滤、序列化；
  - `test_logparse.py` — 上游日志解析。
- 必测：transform、指标、后处理、划分、配置解析。
- 模型：2 样本 smoke test；训练：先 overfit 单 batch。
- 测试不依赖网络与大数据。

先写失败测试，再修 bug。
