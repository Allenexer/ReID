# 数据说明

> `data/raw/` 只读；数据集与帧文件不入 git。所有命令在仓库根目录执行。

## 1. 数据集状态

| 数据集 | 角色 | 状态 | 本地路径 |
|---|---|---|---|
| Market1501 | 主评测，可训练 | 待下载 | `data/raw/market1501/` |
| Move | 课程验证集，仅测试 | 待课程发布 | `data/raw/move/` |
| CCVID | 换衣探索/加分 | 待下载 | `data/raw/ccvid/` |

## 2. Market1501

来源（课程给出）：<https://www.kaggle.com/datasets/pengcw1/market-1501>

下载方式一（浏览器）：登录 Kaggle，下载并解压到 `data/raw/`，得到
`Market-1501-v15.09.15/`。

下载方式二（Kaggle CLI，需先配置 `~/.kaggle/kaggle.json`）：

```bash
kaggle datasets download -d pengcw1/market-1501 -p data/raw --unzip
```

把解压后的目录整理成下面任一形式（解析器两种都认）：

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

生成并校验 manifest：

```bash
py -3 -m scripts.check_dataset --config configs/datasets/market1501.yaml
```

预期输出：query 3368 张 / 750 ID，train 12936 张 / 751 ID。若数量不符，
先检查目录层级和文件名，不要继续跑实验。

评测协议：

- 文件名格式 `{pid}_c{cam}s{seq}_{frame}_{det}.jpg`；
- `pid == -1` 是 junk，解析时剔除；
- `pid == 0` 是 distractor，保留在 gallery（与 TransReID 上游一致，作为不匹配项参与排序）；
- 评测时对每个 query 剔除同 ID 同摄像头的 gallery 样本。

## 3. Move（课程数据）

- 名称在课程材料中可能写作 `Move` 或 `M0ve`，以实际发布为准。
- 规模：200 ID、800 query，含遮挡场景；仅测试，不提供训练集。
- 获取方式：等课程发布。收到后放到 `data/raw/move/`，先记录目录结构，再写适配器。
- 已知挑战：没有 `bounding_box_train`，而上游 `make_dataloader` 默认会构造训练集；
  适配时需要绕过训练集/分类头尺寸依赖（见 `docs/BASELINES.md` 的 test-only 方案）。

## 4. CCVID（加分）

来源：<https://github.com/ziyuanzhangtony/GaitNet-CVPR2019>

- 视频/序列数据，换衣场景；具体划分以官方仓库和课程说明为准。
- 使用前需要抽帧，并把 tracklet 聚合成一个检索单元（多帧特征平均或投票）。
- 先做单帧 baseline 跑通，再比较多帧聚合；不要一上来做复杂时序模型。

## 5. 版本与校验

在 `data/README.md` 登记：来源 URL、下载日期、压缩包大小、SHA256、解压后目录名。
数据一旦变更（重新划分、补帧、清洗），更新登记并在 `DECISIONS.md` 记一条。

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
