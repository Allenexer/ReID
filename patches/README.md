# 上游兼容补丁

上游代码固定在 `third_party/` 的特定 commit；所有修改以 patch 形式保存，
由 `scripts/setup_third_party.sh` 幂等应用。

## 当前补丁

| 文件 | 目标 | 作用 |
|---|---|---|
| `transreid-py312.patch` | TransReID | 把已删除的 `torch._six.container_abcs` 换成 `collections.abc.Iterable`，兼容 PyTorch 2.x / Python 3.12 |
| `transreid-timm.patch` | TransReID | 把 `RandomErasing(max_count=1)` 改成 `count=1`，兼容新版 timm |

## 新增补丁

```bash
# 1. 在 third_party/<repo> 中修改并测试
# 2. 生成补丁（在 third_party/<repo> 下）
git diff -- <path> > ../../patches/<repo>-<purpose>.patch

# 3. 把补丁加入 scripts/setup_third_party.sh 的 apply_patch 调用
# 4. 在 docs/BASELINES.md 和 docs/DECISIONS.md 记录原因
```

补丁必须能用 `git apply` 干净应用；不要包含无关的格式化改动。
