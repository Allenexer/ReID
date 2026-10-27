# 脚本

- `setup_third_party.sh` — 按固定 commit 克隆 TransReID/CLIP-ReID 并打补丁（可用）。
- `check_dataset.py` — 接口骨架；manifest 生成与校验由 A 实现。
- `evaluate.py` — 接口骨架；由距离矩阵计算 mAP/CMC 由 A 实现。
- `log_to_metrics.py` — 接口骨架；上游日志解析由 B 实现。
- 规划中：
  - `run_baseline.py` — 统一实验入口：生成配置、跑上游 test、tee 日志、解析指标；
  - `prepare_move.py` — Move → manifest / Market1501 兼容目录；
  - `prepare_ccvid.py` — CCVID 抽帧与 tracklet 清单。

每个脚本在顶部 docstring 写用途、输入、输出、运行示例；可复用逻辑移到 `src/`。
成员实现对应功能后，把脚本中的 TODO 替换为实际调用，并补充测试。
