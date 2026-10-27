# 测试

- 运行：`py -3 -m pytest -q`（或 `pytest -q`）
- 当前只有 `test_package.py`：确认 `src.reid` 可以导入。
- 各成员实现自己的模块后，在 `tests/` 新增对应测试；测试不依赖网络与大数据。
- 必测：transform、指标、后处理、划分、配置解析。
- 模型：2 样本 smoke test；训练：先 overfit 单 batch。

先写失败测试，再修 bug。
