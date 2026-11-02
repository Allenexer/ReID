# 数据说明

> `data/raw/` 只读；数据集与帧文件不入 git。所有命令在仓库根目录执行。

## 1. 数据集状态

| 数据集 | 角色 | 状态 | 本地路径 |
|---|---|---|---|
| Market1501 | 主评测，可训练 | 本机已下载、CRC/清单与全量解码校验通过（2026-11-01） | `data/raw/Market-1501-v15.09.15/` |
| Move | 课程验证集，仅测试 | 待课程发布 | `data/raw/move/` |
| CCVID | 换衣探索/加分 | 本机已下载并校验（2026-11-01，见 `data/README.md`） | `data/raw/ccvid/` |

### 1.1 数据共享与一致性

原始数据不入 git，四位成员各自保存在本机，通过以下方式同步：

- Market1501：按课程提供的 Kaggle 链接各自下载；下载困难时由 A 上传到组内网盘
  （OneDrive / Google Drive / 阿里云盘等），在群内共享链接。
- Move：课程平台或群文件发布，按课程要求使用，不公开上传。
- CCVID：官方仓库或组内网盘，由 C 负责。
- 统一目录：所有人解压到 `data/raw/<dataset>/`，目录层级按本文件。
- 一致性校验：A 在 `data/README.md` 登记来源、日期、大小与 SHA256；成员下载后
  先核对哈希，再运行数据校验。
- manifest 不入库：通常包含本机路径信息，换机器后需用本地数据重新生成。
  如确需共享，放组内网盘，不放入 git。

## 2. Market1501

来源（课程给出）：<https://www.kaggle.com/datasets/pengcw1/market-1501>

下载方式一（浏览器）：登录 Kaggle，下载并解压到 `data/raw/`，得到
`Market-1501-v15.09.15/`。

下载方式二（Kaggle CLI，需先配置 `~/.kaggle/kaggle.json`）：

```bash
kaggle datasets download -d pengcw1/market-1501 -p data/raw --unzip
```

建议把解压后的目录整理成下面任一形式：

```text
data/raw/market1501/
  bounding_box_train/   12936 张，751 ID
  query/                 3368 张，750 ID
  bounding_box_test/    19732 张（含 junk/distractor）
```

或：

```text
data/raw/Market-1501-v15.09.15/
  bounding_box_train/
  query/
  bounding_box_test/
```

生成清单和验证报告：

```bash
conda run -n rl_v2 python scripts/check_dataset.py --dataset market1501 \
  --root data/raw --manifest data/processed/manifests/market1501.json
```

预期规模：query 3368 张 / 750 ID，train 12936 张 / 751 ID，过滤 junk 后 gallery
15913 张。原始 JPG 数不能直接当作评测行数。规模不符时脚本失败，不删样本继续。
两种目录内容不同时报歧义；内容相同优先官方目录。C 的现有入口仅接受官方目录，
仅有小写目录时需由 C 接入路径解析；禁止在 `raw/` 中复制或建立别名来适配。

索引 JSON 不重复保存样本行，引用同级 `market1501/train.csv,query.csv,gallery.csv`，
包含来源、排序规则、规模及 CSV SHA256。`data_root` 相对索引目录，CSV `path`
相对该根目录；各 split 的 `index` 从 0 连续递增。原始 pid 保留，camid 转为 0～5，
按 JPG 路径字典序排列。加 `--verify-images` 可全量解码；字段及编程接口见
[`A_EVALUATION.md`](A_EVALUATION.md)。

评测协议：

- 文件名格式 `{pid}_c{cam}s{seq}_{frame}_{det}.jpg`；
- `pid == -1` 是 junk，解析时剔除；
- `pid == 0` 是 distractor，保留在 gallery（与 TransReID 上游一致，作为不匹配项参与排序）；
- 评测时对每个 query 剔除同 ID 同摄像头的 gallery 样本。

## 3. Move（课程数据）

- 名称在课程材料中可能写作 `Move` 或 `M0ve`，以实际发布为准。
- 规模：200 ID、800 query，含遮挡场景；仅测试，不提供训练集。
- 获取方式：等课程发布。收到后放到 `data/raw/move/`，先记录目录结构，再写适配器。
- A 已支持标准表 `split,path,pid,camid`，split 只能是 query/gallery，无相机时
  camid 留空。课程原标签到标准表的映射待真实数据发布后补齐。清单协议记为
  `pending`，正式指标入口在相机与过滤规则核实前拒绝 Move。
- 已知挑战：没有 `bounding_box_train`，而上游 `make_dataloader` 默认会构造训练集；
  适配时需要绕过训练集/分类头尺寸依赖（见 `docs/BASELINES.md` 的 test-only 方案）。

## 4. CCVID（加分）

原始 FVG 来源：<https://github.com/ziyuanzhangtony/GaitNet-CVPR2019>；
整理版 CCVID：<https://github.com/guxinqian/Simple-CCReID#ccvid-dataset>。

- CCVID 从 FVG 原视频检测裁剪得到，增加衣服标签和 ReID 划分；整理版已包含 JPG 帧。
- 共 226 ID / 2856 序列 / 347833 帧；train 为 75 ID / 948 序列，
  query 为 151 ID / 834 序列，gallery 为 151 ID / 1074 序列。
- 数据根目录为 `data/raw/ccvid/`，直接包含 `train.txt`、`query.txt`、`gallery.txt`、
  `readme.txt` 和 `session1/`、`session2/`、`session3/`。
- 本机已按此结构保存；原始 `CCVID.zip` 与 SHA256、下载和校验记录保存在
  `data/archives/ccvid/`，完整登记见 [`data/README.md`](../data/README.md)。
- 划分文件每行：`sequence_path pid clothes_label`。保留原有行序、身份及衣服标签。
- C 的单帧适配默认取每段序列的首帧，复用 Market1501 权重，先计算 General 指标；
  详见 [`CLIPREID.md`](CLIPREID.md)。随后比较多帧聚合，并与 A 对齐 CC 换衣评测。

```bash
conda run -n rl_v2 python scripts/check_dataset.py --dataset ccvid \
  --root data/raw/ccvid --manifest data/processed/manifests/ccvid.json --verify-images
```

A 直接复用 C 的 `load_ccvid_split`，数据清单默认单帧、保留划分文件行序；
多帧评测使用 C 归档的实际采样 CSV，不以这个默认清单覆盖。
camid 为序列后缀，session3 加 12。全量解码包括所有原始帧，而非仅采样帧。
数据清单、目录及 `.validation.json` 已存在时拒绝覆盖，重验应指定新索引名。

## 5. 版本与校验

在 `data/README.md` 登记：来源 URL、下载日期、压缩包大小、SHA256、解压后目录名。
数据一旦变更（重新划分、补帧、清洗），更新登记。

```bash
# Windows Git Bash 计算校验和
sha256sum data/raw/<archive>
```

## 6. 目录约定

```text
data/
  raw/        原始数据，只读，不入库
  processed/  转换后的数据与 manifest，不入库，可重建
```
