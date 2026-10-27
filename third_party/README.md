# 上游基线代码

本目录**不入 git**（只保留本 README）。第一次使用运行：

```bash
bash scripts/setup_third_party.sh
```

脚本会按固定 commit 克隆：

| 目录 | 仓库 | Commit |
|---|---|---|
| `TransReID/` | <https://github.com/heshuting555/TransReID> | `dec55046fcdfadee14e2c28e2df89305d8f7557a` |
| `CLIP-ReID/` | <https://github.com/Syliz517/CLIP-ReID> | `eb1898b72c882875f478bebfc6d41644eece0a5d` |

并应用 `patches/` 中的兼容补丁。

规则：

- 不直接修改本目录下的文件；需要改动时写 `patches/*.patch`，由脚本应用。
- 不把上游代码、权重、日志提交进本项目。
- 升级上游版本时，更新 `scripts/setup_third_party.sh`、本文件和 `docs/BASELINES.md`。
