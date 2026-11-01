# 脚本

- `setup_third_party.sh` — 按固定 commit 克隆 TransReID/CLIP-ReID 并打补丁（可用）。
- `run_clipreid.py` — C 的 CLIP-ReID 入口：在根目录运行
  `python scripts/run_clipreid.py`，保存配置、环境、上游指标及原始特征/距离矩阵；
  CCVID 单帧使用 `--config configs/clipreid/ccvid_single_frame.yml`；
  多帧使用 `--config configs/clipreid/ccvid_multiframe.yml --frames 8`；
  可用 `--dry-run` 检查路径，详见 [`docs/CLIPREID.md`](../docs/CLIPREID.md)。
- `check_dataset.py` — A 的数据校验与 manifest 生成，输出 JSON 索引、split CSV 和
  同名 `.validation.json`；`--verify-images` 全量解码，已有目标拒绝覆盖。
- `evaluate.py` — A 的基线归档评测，仅接收 `status=success` 的同次归档，
  选择 native 或从特征重算 cosine；写入新目录，保留原指标和数组。
- `log_to_metrics.py` — 占位；上游日志转 `metrics.json`（B）。
统一实验入口由 D 接入。Move 标准表可登记，正式相机过滤协议待课程确认。

在 `ReID/` 执行，路径参数均相对当前调用目录：

```bash
conda run -n rl_v2 python scripts/check_dataset.py --dataset market1501 \
  --root data/raw --manifest data/processed/manifests/market1501.json
conda run -n rl_v2 python scripts/check_dataset.py --dataset ccvid \
  --root data/raw/ccvid --manifest data/processed/manifests/ccvid.json --verify-images
conda run -n rl_v2 python scripts/check_dataset.py --dataset move \
  --root data/raw/move --labels data/processed/move_labels.csv \
  --manifest data/processed/manifests/move.json
conda run -n rl_v2 python scripts/evaluate.py --run-dir outputs/<run> \
  --protocol ccvid_cc --distance cosine \
  --output-dir outputs/<run>/evaluation/ccvid_cc/cosine
```

协议可选 `market1501,ccvid_general,ccvid_cc,ccvid_sc`。
完整契约及验收命令见 [`A_EVALUATION.md`](../docs/A_EVALUATION.md)。
