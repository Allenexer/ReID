# 配置

- `datasets/` — 数据集卡片：路径、manifest 输出、预期规模。
- 实验配置：在 `docs/BASELINES.md` 的命令模板基础上追加覆盖项；
  每个 run 的实际配置保存到 `outputs/<run>/config.yaml`。
- 上游仓库自带的配置留在 `third_party/`，不要复制后忘记同步。
- 禁止在代码中硬编码超参；配置随代码提交，运行产物放 `outputs/`。
