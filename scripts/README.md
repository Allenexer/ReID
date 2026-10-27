# 脚本

- `setup_third_party.sh` — 按固定 commit 克隆 TransReID/CLIP-ReID 并打补丁（可用）。
- `check_dataset.py` — 占位；数据集校验与 manifest 生成（A）。
- `evaluate.py` — 占位；由距离矩阵计算 mAP/CMC（A）。
- `log_to_metrics.py` — 占位；上游日志转 `metrics.json`（B）。
- 规划中：
  - `run_baseline.py` — 统一实验入口（D）；
  - `prepare_move.py` — Move → manifest / Market1501 兼容目录（A）；
  - `prepare_ccvid.py` — CCVID 抽帧与 tracklet 清单（C）。

各脚本的命令行、输入输出与实现由负责人设计；完成后在此补充一句用法。
