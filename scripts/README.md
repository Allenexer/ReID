# 脚本

- `setup_third_party.sh` — 按固定 commit 克隆 TransReID/CLIP-ReID 并打补丁（可用）。
- `run_clipreid.py` — C 的 CLIP-ReID 入口：在根目录运行
  `python scripts/run_clipreid.py`，保存配置、环境、上游指标及原始特征/距离矩阵；
  CCVID 单帧使用 `--config configs/clipreid/ccvid_single_frame.yml`；
  可用 `--dry-run` 检查路径，详见 [`docs/CLIPREID.md`](../docs/CLIPREID.md)。
- `check_dataset.py` — 占位；数据集校验与 manifest 生成（A）。
- `evaluate.py` — 占位；由距离矩阵计算 mAP/CMC（A）。
- `log_to_metrics.py` — 占位；上游日志转 `metrics.json`（B）。
统一实验入口（D）、Move 适配（A）、CCVID 适配（C）等脚本由对应负责人按需设计，
文件名、命令行与接口自定；完成后在此补充一句用法。
