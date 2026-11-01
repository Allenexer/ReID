# A：数据、统一评测与 B/C/D 交接

实现遵循工作区 `../prompt/plan.md`。源码在 `zzj_a_part`，新增测试在仓库外的
`../test/`，两者交付时都需要携带。原始数据及基线归档只读，输出拒绝覆盖。

## 公共 Python 接口

| 模块 | 接口 | 说明 |
|---|---|---|
| `src.reid.manifest` | `Sample/Manifest/ValidationReport` | 清单行、内存清单及验证报告 |
| 同上 | `build_manifest(dataset, root, labels=None)` | 解析数据，完整规模由后续校验检查 |
| 同上 | `validate_manifest(manifest, verify_images=False)` | 返回报告，调用 `require_valid()` 使错误失败 |
| 同上 | `save_manifest(manifest, path, verify_images=False)` | 再校验后发布 CSV、验证报告及索引 |
| 同上 | `load_manifest(path)` | 检查 CSV 指纹并保留行序，离线读取不依赖原图 |
| `src.reid.metrics` | `evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, protocol, q_clothes=None, g_clothes=None, max_rank=50)` | 只依赖 NumPy，不读文件或加载模型 |
| `src.reid.features` | `cosine_distance(query_features, gallery_features)` | 检查非零有限特征，重做 L2 并分块计算 |
| `src.reid.artifacts` | `load_run(run_dir)` | 返回只读 mmap 数组、内存清单、原元数据与指纹 |
| 同上 | `compare_manifest(run_manifest, source)` | 按显式根目录、完整路径、行序、标签及序列实际采样核对 |
| 同上 | `evaluate_run(run_dir, protocol, distance, output_dir)` | 先校验、复算、对拍，再向新目录发布 |

`EvaluationResult` 含完整 `cmc,mAP,num_valid_query,num_skipped_query`；`as_metrics()`
在序列化边界提取 Rank-1/5/10 并保留 6 位，内部计算不舍入。
新增 A 模块使用 Python 3.8 兼容语法；实际测试环境为 Linux/Python 3.10 的 `rl_v2`。
导入 A 不引入 torch、torchvision 或 yacs；读取归档配置时才导入项目已有依赖 PyYAML，
全量解码时才导入 Pillow。

## 数据清单

命令均在 `ReID/` 执行，路径相对当前调用目录：

```bash
conda run -n rl_v2 python scripts/check_dataset.py --dataset market1501 \
  --root data/raw --manifest data/processed/manifests/market1501.json --verify-images
conda run -n rl_v2 python scripts/check_dataset.py --dataset ccvid \
  --root data/raw/ccvid --manifest data/processed/manifests/ccvid.json --verify-images
```

输出为 `<dataset>.json`、`<dataset>/train.csv,query.csv,gallery.csv` 和
`<dataset>.validation.json`；Move 只有 query/gallery。索引含 schema_version=1、
dataset/protocol/sample_unit/data_root、排序规则、来源、预期和实际规模及各 CSV SHA256。
data_root 相对索引目录，CSV path 相对 data_root，使用 POSIX 路径。

基础 CSV 字段是 `index,path,pid,camid,viewid`，CCVID 追加
`sequence,clothes_label,num_frames,sampled_frames,frame_paths`，最后一项是 JSON 列表字符串。
pid 保留原始值，训练分类标签映射由 B/C 单独生成。衣服类别必须用 `(pid,clothes_label)`。
Market1501 过滤 -1、保留 0，camid 减 1，JPG 路径字典序；官方目录与小写目录均可解析，
两份数据不同时失败。C 当前只接受官方目录；B 后续应显式承接根目录或清单。
CCVID 复用 C 的读取器，保留 txt 原行序、session3 相机偏移和序列边界。
默认数据清单为 T=1、FRAME_INDEX=0，不替代运行的多帧清单。

Move 标准表为 `split,path,pid,camid`，camid 可为空，不制造 train 或相机。
标签表映射和课程过滤协议未发布，manifest 的协议为 pending，正式指标暂不启用。
校验覆盖非空划分、越界、重复、非法标签、缺文件、规模、训练身份泄漏及查询身份集合；
`--verify-images` 全量解码，CCVID 检查全部原始帧。失败时不删样本继续。

## 接收 B/C 的归档

同一次成功运行必须包含：

```text
status.json, metrics.json, query.csv, gallery.csv
distmat.npy, query_features.npy, gallery_features.npy
config.yaml, env.txt, git.txt
```

沿用 C 的 schema_version=1、0～1 标度、请求与实际帧数字段。
CSV 行序必须对应数组，不接受单独拼接的 CSV/NPY，也不按 basename 匹配。
C 的数组为 float32、L2 特征、平方欧氏距离、无重排序；B 导出真实的距离、归一化及
重排序元数据。配置至少保留实际 DATASETS.NAMES/ROOT_DIR 与 TEST.FEAT_NORM/RE_RANKING；
CCVID 另有 DATASETS.NUM_FRAMES/FRAME_INDEX。ROOT_DIR 记录实际绝对路径，离线可不存在。
Market1501 的 viewid 按生产者检查：CLIP-ReID 为 0，TransReID 为 1，指标不使用 viewid。

读取时检查状态、维度、标签、连续索引、重复、有限值、L2 声明和配置一致性；
无重排序时分块核对矩阵与特征所声明的距离。原始矩阵不改写，不额外构造完整 Q×G 索引矩阵。

需要对照原始 manifest 时，可直接调用 `compare_manifest`；归档配置还可显式记录
`DATASETS.MANIFEST` 的绝对路径，使 `load_run` 自动执行此校验。
该文件应代表实际采样，不能把默认单帧清单用于多帧运行。A 不回写 C 的已有配置。
未提供时会记录 `source_verification.status=not_requested`，只代表归档内部契约通过。
C 的历史产物没有签名的行序指纹，离线无法证明任意重新编号且自洽的 CSV 未曾改变。

## 协议和结果

| protocol | 同身份 gallery 的剔除规则 |
|---|---|
| `market1501` / `ccvid_general` | 同 camid |
| `ccvid_cc` | 同 camid，或同衣服 |
| `ccvid_sc` | 同 camid，或不同衣服 |

其他身份的 gallery 始终作为负样本保留。无正样本 query 跳过并计数，全无效则报错。
排序为 `np.argsort(kind="quicksort")`，AP 使用完整过滤后列表；CMC 长度固定为
`min(max_rank,G)`，不足时补末值；超长 Rank 输出 null。

```bash
conda run -n rl_v2 python scripts/evaluate.py --run-dir outputs/<run> \
  --protocol ccvid_general --distance native \
  --output-dir outputs/<run>/evaluation/ccvid_general/native
conda run -n rl_v2 python scripts/evaluate.py --run-dir outputs/<run> \
  --protocol ccvid_cc --distance cosine \
  --output-dir outputs/<run>/evaluation/ccvid_cc/cosine
```

native 原值复算；cosine 重做 L2 后计算 `1-q·g`，不通过修改距离名代替重算。
新目录中有 `metrics.json,status.json,verification.json`，cosine 另有派生 `distmat.npy`。
写完临时结果后才发布，只有完整结果具有 success；原数据根目录禁止作为输出位置。

metrics 保留原字段，并增加 `evaluation_source=reid_unified`、distance、source_run、
source_hashes、num_valid_query/num_skipped_query、完整 CMC 及来源校验状态。
verification 保存上游 git/environment 原文、双方未舍入的指标及差值、排序/环境、
输入与派生指纹；cosine 额外记录与 native 的排序及指标差异。
同协议 native 与原完整精度指标的绝对误差必须严格小于 0.001，否则不发布 success；
跨协议标为 not_comparable。通用算法同矩阵对拍目标为 1e-6。

D 只读取显式的 `evaluation/<protocol>/<distance>/metrics.json` 且 status 为 success，
按 model/dataset/protocol/distance/帧数分组，使用 source_run 追溯原配置、环境和版本。
原始与派生结果属于同一实验，不能重复计数；报告百分数保留 1 位。
B 的导出与 D 的汇总仍为待接入契约，不代表双方已完成实现。

## 验证和实际边界

```bash
# 普通测试，不依赖数据、权重或上游
conda run -n rl_v2 python -m pytest -c pyproject.toml tests ../test \
  -m 'not local_data and not upstream' -ra

# 本机完整数据规模和采样验收
conda run -n rl_v2 python -m pytest -c pyproject.toml ../test/test_data_local.py \
  -m local_data --data-root data/raw/ccvid --market-root data/raw -ra

# 固定上游手算、随机矩阵及真实数据加载顺序
conda run -n rl_v2 python -m pytest -c pyproject.toml ../test/test_upstream_parity.py \
  -m upstream --upstream-root third_party --market-root data/raw \
  -k 'not real_run' --junitxml=outputs/a-parity.xml -ra

# 获得真实基线归档后全矩阵 native/cosine 对拍
conda run -n rl_v2 python -m pytest -c pyproject.toml ../test/test_upstream_parity.py \
  -m upstream --run-dir outputs/<run> --upstream-root third_party -k real_run -ra
```

上游分别在独立进程导入，按固定源码 SHA256 检查版本；CC/SC 衣服按联合键编码。
上游缺失或未提供外部路径时明确 skip，指定路径无效则失败。固定 CCVID 源码使用
NumPy 2 已移除的 `np.in1d`，其独立子进程需 NumPy 1.26 环境；可通过
`--upstream-pythonpath <目录>` 只向子进程提供依赖，不改上游函数或伪造 API。

本次 `rl_v2` 原环境缺 PyYAML/yacs/ruff，将项目已有依赖装到 `/tmp/reid-a-deps`，
测试命令前加 `PYTHONPATH=/tmp/reid-a-deps`。固定上游源码只读下载至
`/tmp/reid-a-upstream`，未伪装成完整 checkout；CCVID 子进程用
`--upstream-pythonpath /tmp/reid-a-upstream-deps` 的 NumPy 1.26.4。
常规 A 评测仍在 NumPy 2.2.6 下运行；无真实权重/GPU 推理的测试不能作为模型性能结果。
当前缺少真实运行矩阵，真实 run 两项对拍保留显式 skip；Move 等待课程数据和协议。

2026-11-01 本机最终验收为 **135 passed，2 skipped**，两项 skip 仅对应未提供真实
run 的全矩阵对拍。固定上游手算/随机矩阵的最大 CMC 绝对误差为
`1.9868214962137642e-08`，mAP 最大误差为 `1.3877787807814457e-17`。
两套真实 Market1501 读取器的原始身份模式逐行对照共 32217 行均一致，分别保留
viewid 0/1；实际训练分类重编号仍由 B/C 负责。
Market1501 清单图片 32217 张、CCVID 全部 347833 帧全量解码通过。
机器可读证据与源码/测试指纹见
[`reports/a_validation_20261101.json`](../reports/a_validation_20261101.json)，
原始测试结果见 [JUnit XML](../reports/a_validation_20261101.xml)。

来源为方案固定的
[CLIP-ReID 评测器](https://github.com/Syliz517/CLIP-ReID/blob/eb1898b72c882875f478bebfc6d41644eece0a5d/utils/metrics.py)、
[TransReID 评测器](https://github.com/damo-cv/TransReID/blob/dec55046fcdfadee14e2c28e2df89305d8f7557a/utils/metrics.py) 和
[CCVID 官方评测](https://github.com/guxinqian/Simple-CCReID/blob/f773d013508a7697f09ff0ca5f1622dfc46aa531/tools/eval_metrics.py)。
