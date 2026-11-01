# 数据说明

> 本目录内容默认不入库（README 除外）。`raw/` 只读；`processed/` 可重建。

## 数据集登记

| 数据集 | 来源/链接 | 角色 | 规模 | 下载日期 | 校验和 | 本地路径 | 状态 |
|---|---|---|---|---|---|---|---|
| Market1501 | <https://www.kaggle.com/datasets/pengcw1/market-1501> | 主评测，可训练 | train 751 ID/12936；query 750 ID/3368；gallery 15913 | 2026-11-01 | SHA256 见下 | `data/raw/Market-1501-v15.09.15/` | 本机已下载、解压并全量解码校验 |
| Move | 课程提供 | 验证，仅测试 | 200 ID/800 query，含遮挡 | TBD | TBD | `data/raw/move/` | 待发布 |
| CCVID | [Google Drive 压缩包](https://drive.google.com/file/d/1vkZxm5v-aBXa_JEi23MMeW4DgisGtS4W/view)，[整理版项目](https://github.com/guxinqian/Simple-CCReID#ccvid-dataset) | 换衣加分 | 226 ID / 2856 序列 / 347833 帧 | 2026-11-01 | SHA256 见下 | `data/raw/ccvid/` | 本机已下载、解压并校验 |

## Market1501 本机下载与 A 清单验收

来源为上表指定 Kaggle 数据集，使用公开 GET 下载接口，未修改数据文件名和划分。
北京时间 2026-11-01 21:43:18 开始、21:43:42 下载完成；下载耗时 24.27 秒，
含解压和全 ZIP CRC 检查共 36.22 秒。压缩包为 153,011,595 字节。

```text
data/archives/market1501/
  market-1501.zip
  market-1501.zip.sha256
  download_metadata.json
  validation_report.json
data/raw/Market-1501-v15.09.15/
  bounding_box_train/
  query/
  bounding_box_test/
data/processed/manifests/
  market1501.json
  market1501.validation.json
  market1501/train.csv
  market1501/query.csv
  market1501/gallery.csv
```

本次压缩包 SHA256（本地指纹，未取得来源提供的官方 SHA256）：

```text
5d3ff58679b1867e3256643b44e1f06a5df1d3a32e367960bba46457f86875ca
```

正式 `check_dataset.py --verify-images` 已通过：train 12936 张 / 751 ID，
query 3368 张 / 750 ID，gallery 过滤 pid=-1 后 15913 张 / 751 ID（含 pid=0）。
共全量解码 32217 张评测/训练清单图片，身份划分无泄漏，CSV 指纹保存在 JSON 索引。
来源中少量 `.jpg.jpg` 路径原样保留，按固定上游的标签前缀解析，不重命名。

```bash
cd data/archives/market1501
sha256sum -c market-1501.zip.sha256
```

## CCVID 本机下载记录

数据按现有配置放在 `data/raw/ccvid/`，不需要修改 CCVID 单帧或多帧配置。
原始压缩包单独保留在 `data/archives/ccvid/`，便于重新解压、核对与组内同步。
以下路径均相对于 `ReID/` 根目录；本机绝对根目录为
`/home/zjzheng/Lesson/CV/PJ/ReID/`。

```text
data/
  archives/ccvid/
    CCVID.zip                 原始下载包，1,562,574,671 字节
    CCVID.zip.sha256           本地 SHA256 指纹
    download_metadata.json    下载来源、时间与速度
    validation_report.json    解压及全量数据校验结果
  raw/ccvid/
    train.txt
    query.txt
    gallery.txt
    readme.txt                数据包自带说明
    session1/
    session2/
    session3/
  downloads/ccvid/
    download_ccvid.py          本次分段下载脚本
    prepare_ccvid.py           本次解压与规模校验脚本
    download.log              下载进度日志
    prepare.log               解压与校验日志
    download_status.json
    prepare_status.json
```

使用 8 路并发分段下载，完成后清理临时分段。北京时间 2026-11-01
21:01:30 开始，21:04:00 传输完成，21:04:02 合并及哈希完成，
21:04:31 解压及数据校验完成。

- 网络传输：149.66 秒（约 2 分 30 秒），平均 9.96 MiB/s。
- 下载、合并与 SHA256：151.81 秒（约 2 分 32 秒）。
- 解压及全量校验：29.04 秒。
- 从正式开始下载到数据就绪：约 3 分 01 秒，不含前期项目检查与网络排查。
- 解压后的文件内容共 1,549,018,313 字节；实际磁盘占用会因小文件分配而增加。

压缩包 SHA256：

```text
050a7d8d6aa3f68962e4cf943eacc1cdb963e06933f23f33e3e0c4b3f5aade02
```

此值为本次下载生成的本地指纹，来源链接未提供可供比对的官方 SHA256。
解压时已验证全部 ZIP 条目的 CRC，另核对所有划分记录、序列目录、帧数量、
身份数量，以及训练与评测身份不重叠、query/gallery 身份集合一致。

| 划分 | ID 数 | 序列数 | JPG 帧数 |
|---|---:|---:|---:|
| train | 75 | 948 | 118613 |
| query | 151 | 834 | 116799 |
| gallery | 151 | 1074 | 112421 |

全部数据共有 226 个不同身份、2856 个序列、347833 张 JPG。
数据、压缩包、下载脚本和日志均被现有 `.gitignore` 排除；仅本登记文档入库。

A 本次还通过正式 `check_dataset.py --verify-images` 全量解码全部 347833 帧；
索引及 CSV 位于 `data/processed/manifests/ccvid.json` 和同名 `ccvid/` 目录，
验证结果为 `ccvid.validation.json`。复用 C 的单帧读取器、原始行序与 session3
相机偏移；运行的 T=4/8 实际采样以 C 归档 CSV 为准。

复查压缩包指纹：

```bash
cd data/archives/ccvid
sha256sum -c CCVID.zip.sha256
```

## 重建步骤

Market1501 下载解压到 `data/raw/` 后，运行 A 提供的 manifest 生成命令；该命令由 A
在实现 `scripts/check_dataset.py` 时确定，并补充到此处。详细说明见 `docs/DATA.md`。
