# 脚本

- `setup_third_party.sh` — 按固定 commit 克隆 TransReID/CLIP-ReID 并打补丁（可用）。
- `check_dataset.py` — 占位；数据集校验与 manifest 生成（A）。
- `evaluate.py` — 占位；由距离矩阵计算 mAP/CMC（A）。
- `log_to_metrics.py` — 占位；上游日志转 `metrics.json`（B）。
统一实验入口（D）、Move 适配（A）、CCVID 适配（C）等脚本由对应负责人按需设计，
文件名、命令行与接口自定；完成后在此补充一句用法。
