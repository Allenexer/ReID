# CLIP-ReID 专用运行流程

目标：复用已经跑通的 Market1501 模型，将一次推理所需的配置、环境、版本、
上游指标和可复算结果保存到独立运行目录。入口放 `scripts/`，可复用逻辑放
`src/reid/clipreid.py`；不修改上游文件，不实现 A 的公共评测或 B 的日志解析器。

## 使用命令

在仓库根目录激活 `.venv` 后运行：

```bash
python scripts/run_clipreid.py --dry-run
python scripts/run_clipreid.py
python scripts/run_clipreid.py --batch-size 4
```

默认配置：`configs/clipreid/market1501.yml`。`--config` 和 `--output-dir` 的相对
路径以项目根目录为基准；为兼容已有配置，YAML 内的数据与权重相对路径仍以
`third_party/CLIP-ReID/` 为基准，运行时转换为绝对路径。

`--dry-run` 校验配置、数据目录和权重路径，打印实际配置，不运行 GPU、不创建
运行目录。真正运行要求 CUDA；输出目录必须不存在，避免覆盖已有实验。

## 产物约定

| 文件 | 内容 |
|---|---|
| `config.yaml` | 合并默认值及命令行参数后的实际配置，包含随机种子 |
| `env.txt` | Python、PyTorch、CUDA、GPU 与 Python 包版本 |
| `git.txt` | 项目及上游 commit、工作区状态 |
| `test_log.txt` | 推理及指标日志 |
| `metrics.json` | 上游评测器输出的原始精度指标，取值为 0～1 |
| `query.csv` / `gallery.csv` | 实际推理顺序，字段为 index、path、pid、camid、viewid |
| `query_features.npy` / `gallery_features.npy` | L2 归一化特征，行顺序对应 CSV |
| `distmat.npy` | query × gallery 的平方欧氏距离，越小越相似 |
| `status.json` | running / success / failed 及时间；失败保留 `error.txt` |

CSV 中 `path` 相对于实际 `DATASETS.ROOT_DIR`，`camid` 使用上游的 0 起始编号。
`distmat[i, j]` 对应 query.csv 的第 i 个样本与 gallery.csv 的第 j 个样本。
样本数采用上游过滤 junk 后的数量，gallery 中 pid=0 的 distractor 保留。

当前只支持 Market1501、特征归一化、无重排序，复用上游的 `R1_mAP_eval`。
`metrics.json` 标记 `evaluation_source=clipreid_upstream`，格式是 C 的初版产物，
需要与 A 的公共评测接口对齐。上游平方欧氏距离与项目约定的余弦距离数值不同；
对 L2 归一化特征，两者理论上相差 2 倍，但不能把前者误标为余弦距离。
A 可从导出的特征计算统一余弦距离，或先使用原始矩阵进行上游指标对拍。

以前只保存了日志的实验无法恢复特征，需要用新入口重新推理。旧目录保留。
失败实验也保留；未提交代码的运行会在 `git.txt` 中留下工作区状态，正式复现
建议在提交后再执行。

## 验收与验证

- 正常运行产生上述全部文件，矩阵、特征与 CSV 行数和顺序一致。
- 保存真实上游指标，不从已四舍五入的日志恢复精度。
- 输入错误在推理前报告；已存在的输出目录拒绝覆盖；运行失败保留现场。
- 新增文件遵循中文说明、英文标识符；使用 pathlib、类型标注及 pytest。
- 测试不依赖 GPU、下载或真实大数据；用小数组检查导出对应关系与错误处理。

```bash
python -m pytest -q
ruff format .
ruff check .
```
