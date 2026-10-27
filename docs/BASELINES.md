# 基线部署

> 上游代码放在 `third_party/`（不入库），由脚本按固定 commit 克隆。
> 任何兼容性修改都放 `patches/`，不要直接手改上游后不记录。

## 1. 固定版本

| 基线 | 仓库 | Commit |
|---|---|---|
| TransReID | <https://github.com/heshuting555/TransReID> | `dec55046fcdfadee14e2c28e2df89305d8f7557a` |
| CLIP-ReID | <https://github.com/Syliz517/CLIP-ReID> | `eb1898b72c882875f478bebfc6d41644eece0a5d` |

拉取并打补丁：

```bash
bash scripts/setup_third_party.sh
```

脚本会：克隆两个仓库 → checkout 到上表 commit → 应用：

- `patches/transreid-py312.patch`：修复 PyTorch 2.x 已删除的 `torch._six`；
- `patches/transreid-timm.patch`：把 `RandomErasing(max_count=1)` 改成
  `count=1`，兼容新版 timm。

## 2. 环境

- 系统 Python 3.12.10，已装 torch 2.5.1+cu121（`torch.cuda.is_available() == True`）。
- 项目仍建议使用 `.venv`；建环境见 `docs/SETUP.md`。**不要升级到 torch 2.6+**：
  上游 `torch.load` 未传 `weights_only=False`，2.6 起默认值变化会导致加载失败。
- 两个上游还需要 `timm`、`yacs`、`opencv-python`。建议先装：
  `pip install timm==0.9.16 yacs==0.1.8 opencv-python`；跑通后写入
  `requirements.lock`。CLIP-ReID 首次运行会下载 CLIP 权重。

## 3. TransReID

配置文件：`third_party/TransReID/configs/Market/vit_base.yml`（普通 ViT baseline）。
需要覆盖的项：

| 配置 | 值 | 原因 |
|---|---|---|
| `DATASETS.ROOT_DIR` | `../../data/raw` | 上游在它下面找 `market1501/` |
| `MODEL.DEVICE_ID` | `('0')` | 默认 `('7')` 本机不存在 |
| `MODEL.PRETRAIN_CHOICE` | `'none'` | 推理时跳过 ImageNet 预训练；否则会去找配置里写死的 `/home/heshuting/...` 路径 |
| `TEST.IMS_PER_BATCH` | 先 `8`，再试 `16` | 默认 256，4GB 显存会 OOM |
| `DATALOADER.NUM_WORKERS` | `0` | Windows 多进程容易卡住 |
| `TEST.WEIGHT` | `../../checkpoints/transreid_market_vit_base.pth` | 必填 |
| `OUTPUT_DIR` | `../../outputs/<run>` | 日志目录 |

**权重**：打开 TransReID README 的 “Trained Models and logs (Size 256)”，
选 **Baseline(ViT)** 行的 **Market** 列（Google Drive）：
<https://drive.google.com/file/d/1crYsKRrW4eUq6abT4KK8_atMLFsbq56W/view?usp=sharing>
下载后重命名为 `checkpoints/transreid_market_vit_base.pth`。
这一行对应报告值 **mAP 87.1 / Rank-1 94.6**，适合先跑通。
如果想复现论文里的 TransReID 完整版（89.0 / 95.1），改用
`configs/Market/vit_transreid.yml` + 同表 **TransReID (ViT)** 行 Market 列的权重。

命令模板（**先 `cd third_party/TransReID`**，路径都相对于这个目录）：

```bash
python test.py \
  --config_file configs/Market/vit_base.yml \
  DATASETS.ROOT_DIR "../../data/raw" \
  MODEL.DEVICE_ID "('0')" \
  MODEL.PRETRAIN_CHOICE "'none'" \
  TEST.IMS_PER_BATCH 8 \
  DATALOADER.NUM_WORKERS 0 \
  TEST.WEIGHT "../../checkpoints/transreid_market_vit_base.pth" \
  OUTPUT_DIR "../../outputs/20261027-transreid-market"
```

注意 `MODEL.PRETRAIN_CHOICE "'none'"` 的引号：外层双引号是 shell 的，内层单引号是
yacs 的，表示字符串 `none`。跑通后再把 batch 加到 16。

上游把日志写到 `OUTPUT_DIR/test_log.txt`（注意不是 `test.log`），里面会有
`mAP:`、`Rank-1/5/10`。这是唯一允许写进报告的指标来源，不要手抄日志里的数字。

`scripts/log_to_metrics.py` 的命令行由 B 设计；B 实现后在此补充实际命令。实现前
保留原始 `test_log.txt`，不要把日志里的数字手工填进报告。

## 4. CLIP-ReID

配置：`third_party/CLIP-ReID/configs/person/vit_clipreid.yml`（或 `vit_base.yml`）。
入口是 `test_clipreid.py`（CLIP-ReID 模型）或 `test.py`（普通 ViT baseline）。
需要覆盖 `DATASETS.NAMES ('market1501')`、`DATASETS.ROOT_DIR`、`TEST.WEIGHT`、
`MODEL.DEVICE_ID ('0')`、`TEST.IMS_PER_BATCH 16`、`DATALOADER.NUM_WORKERS 0`。

权重：CLIP-ReID README 的表格（Market1501 列，`ViT-CLIP-ReID` 行），
放到 `checkpoints/clipreid_market_vit_b16.pth`。

命令模板：

```bash
python test_clipreid.py \
  --config_file configs/person/vit_clipreid.yml \
  DATASETS.NAMES "('market1501')" \
  DATASETS.ROOT_DIR "/d/计算机视觉/课程项目/data/raw" \
  MODEL.DEVICE_ID "('0')" \
  TEST.IMS_PER_BATCH 16 \
  DATALOADER.NUM_WORKERS 0 \
  TEST.WEIGHT "/d/计算机视觉/课程项目/checkpoints/clipreid_market_vit_b16.pth" \
  OUTPUT_DIR "/d/计算机视觉/课程项目/outputs/<run>"
```

## 5. 4GB 显存建议

- 先 `TEST.IMS_PER_BATCH 8` 验证能跑通，再逐步加到 16/32；OOM 就减半。
- 推理用 `torch.no_grad()`，上游已经做了；不要开梯度。
- 训练 ViT-B 按上游 batch 64 不可行：用 Colab，或改用 ResNet-50 / 更小分辨率，
  并在报告中说明与官方设置的差异。
- 跑之前先看显存：`nvidia-smi --query-gpu=memory.used,memory.total --format=csv`。

## 6. 测试集没有训练集时（Move）

上游 `make_dataloader` 一定会读 `dataset.train`，`make_model` 用
`num_train_pids` 决定分类头大小，`load_param` 直接按 key `copy_`。Move 只有
query/gallery，直接套用会失败。候选方案（按优先级）：

1. **自定义 dataset 类**：在 `src/reid/adapters/` 注册一个只含 query/gallery 的
   dataset，`train` 用 gallery 中每 ID 的几张图占位，`num_train_pids` 固定为
   checkpoint 的分类数（Market1501 为 751），避免分类头尺寸不匹配；
2. **Market1501 兼容目录**：把 Move 转成 `query/` + `bounding_box_test/` 命名，
   让上游原样读取；
3. 修改上游 `make_dataloader`，跳过训练集构造（补丁放 `patches/`）。

方案选定后更新本节。在 Move 数据到手前，先用 Market1501 跑通全链路。

## 7. 输出契约

每次基线运行至少产出：

```text
outputs/<run-id>/
  config.yaml     本次实际使用的配置（或上游配置副本 + 覆盖项）
  test_log.txt    上游完整日志
  metrics.json    由解析/评测脚本生成，禁止手写
  env.txt         torch/CUDA/GPU 信息
  git.txt         本项目 commit + 上游 commit
```

`metrics.json` 的字段由 A 在评测实现时定稿并写入文档；报告只引用它。
