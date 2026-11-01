# CLIP-ReID 接口与集成说明

本文定义 CLIP-ReID 运行入口、产物格式与集成验证要求，供公共评测和报告汇总使用。
环境、权重与完整运行步骤见 [CLIPREID.md](CLIPREID.md)，实验记录见
[EXPERIMENTS.md](EXPERIMENTS.md)。接口变更时同步更新本文及相关测试。

## 运行入口

入口为 [`scripts/run_clipreid.py`](../scripts/run_clipreid.py)。在仓库根目录激活
`.venv` 后运行：

```bash
python scripts/run_clipreid.py --config configs/clipreid/market1501.yml
python scripts/run_clipreid.py --config configs/clipreid/ccvid_single_frame.yml
python scripts/run_clipreid.py --config configs/clipreid/ccvid_multiframe.yml --frames 4
```

每次运行写入独立的 `outputs/<run_id>/`。`--dry-run` 检查配置和路径；实际推理要求 CUDA。

## 产物格式

| 文件 | 定义 |
|---|---|
| `distmat.npy` | float32，形状为 Q × G，平方欧氏距离，越小越相似 |
| `query.csv` / `gallery.csv` | query / gallery 样本清单，行序对应距离矩阵及特征 |
| `query_features.npy` / `gallery_features.npy` | float32，形状为 Q × D / G × D，L2 归一化特征 |
| `metrics.json` | 指标及评测元数据，字段见下文 |
| `config.yaml` / `env.txt` / `git.txt` | 实际配置、运行环境和代码版本 |
| `status.json` | 运行状态及起止时间；状态为 running、success 或 failed |

Q、G、D 分别为 query 数、gallery 数和特征维度。`distmat[i, j]` 对应 query 清单的
第 i 行与 gallery 清单的第 j 行，均不计 CSV 表头。仅汇总 `status=success` 的运行。

### 样本清单

| 字段 | 定义 |
|---|---|
| `index` | 在各自清单内从 0 开始的行号 |
| `path` | 相对实际 `DATASETS.ROOT_DIR` 的图像路径 |
| `pid` / `camid` / `viewid` | 整数身份、摄像头或协议编号、视角标签 |
| `sequence` | CCVID 序列相对路径 |
| `clothes_label` | CCVID 衣着标签，与 pid 共同标识衣着类别 |
| `num_frames` | CCVID 原序列帧数 |
| `sampled_frames` | CCVID 实际采样帧数 |
| `frame_paths` | CCVID 采样帧相对路径列表，以 JSON 字符串存入 CSV |

前五个字段为基础字段，其余字段仅用于 CCVID。CCVID 的 `path` 为首个采样帧，
每行仍代表整个序列；单帧与多帧输出采用相同清单结构。读取清单时须保留原行序。

### 指标与元数据

- `schema_version=1`，`model=clipreid`，`evaluation_source=clipreid_upstream`。
- `dataset` 为 `market1501` 或 `ccvid`；`protocol` 为 `market1501` 或 `ccvid_general`。
- `mAP,rank1,rank5,rank10` 为 0～1；展示为百分数时乘 100。gallery 过小时不可用的 Rank 为 null。
- `num_query,num_gallery` 与矩阵形状一致。
- `distance_metric=squared_euclidean`，`feature_normalization=l2`，`re_ranking=false`。
- CCVID 另含 `sample_unit=sequence`、`frames_per_sequence`、`frame_sampling`、`aggregation`。
  `frames_per_sequence` 为请求帧数，实际帧数以 CSV 的 `sampled_frames` 为准；
  抽帧方式为 index 或 uniform，聚合方式为 mean_then_l2。
- `inference_seconds` 为数据读取、权重加载、特征提取及评测耗时，单位为秒，
  不含环境记录与结果导出；历史运行可能没有此字段。

## 数据与评测协议

| 数据集 | 检索单位 | Q × G | 编号与过滤规则 |
|---|---|---|---|
| Market1501 | 图像 | 3368 × 15913 | 过滤 pid=-1，保留 pid=0；camid 为 0～5 |
| CCVID | 视频序列 | 834 × 1074 | 按官方划分文件行序；camid 取序列后缀，session3 加 12 |

以上规模对应当前完整划分。CCVID 的 camid 为评测协议编号，并非物理摄像头编号。
两套当前协议均剔除同 pid、同 camid 的 gallery 样本。CCVID General 允许同衣和换衣匹配；
CC/SC 协议尚未接入，不能将 General 指标标为换衣专用指标。

多帧测试按文件名排序后均匀抽帧，平均原始特征后再做 L2 归一化；短序列使用全部帧。
T=1 使用 `FRAME_INDEX`，T>1 包含序列首尾。CCVID 配置使用 Market1501 预训练权重，
保持 `CHECKPOINT_CLASSES=751`，关闭 SIE，不使用 CCVID 训练集。

## 集成与验证

### 公共评测

读取同一次运行的 `distmat.npy,query.csv,gallery.csv`，按对应协议复算 mAP 和 CMC。
与原 `metrics.json` 比较时使用同一矩阵、相同过滤规则，指标误差应小于 0.1 个百分点。

统一使用余弦距离时，可从归一化特征重新计算。平方欧氏距离与余弦距离排序等价，
数值不同，应保留实际距离类型。同矩阵对拍与距离统一分别验证。

### 入口与报告汇总

统一入口可调用上述命令；以进程退出码和 `status.json` 判断运行结果。
报告汇总读取 `metrics.json`，按模型、数据集、协议、配置及帧数区分实验；
复现记录读取 `config.yaml,env.txt,git.txt`。耗时比较同时注明硬件及 batch size。

实验产物不随 Git 分发。指标复算按需共享矩阵及两份清单，或在本机重新生成；
报告汇总按需共享指标 JSON 和复现记录。

### 待对齐事项

- 公共 manifest 字段与评测函数的调用方式。
- CC/SC 协议定义、实现分工及验证样例。
- 统一入口及报告汇总所需的字段映射。
