# 脚本

- `setup_third_party.sh` — 按固定 commit 克隆 TransReID/CLIP-ReID 并打补丁。
- `check_dataset.py` — 校验数据集并生成 manifest。
- `evaluate.py` — 从距离矩阵计算 mAP/CMC，写 `metrics.json`。
- `log_to_metrics.py` — 从上游 `test_log.txt` 解析 mAP/CMC，写同格式 `metrics.json`；
  上游没有保存距离矩阵时用这个。
- 规划中：
  - `run_baseline.py` — 统一实验入口：生成配置、跑上游 test、tee 日志、解析指标；
  - `prepare_move.py` — Move → manifest / Market1501 兼容目录；
  - `prepare_ccvid.py` — CCVID 抽帧与 tracklet 清单。

每个脚本在顶部 docstring 写用途、输入、输出、运行示例；可复用逻辑移到 `src/`。
