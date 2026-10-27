# 环境搭建

> 目标：任何组员在一台新机器上按本文件可复现环境。
> 本机基线（2026-10-27 记录）：Windows 11、Python 3.12.10、torch 2.5.1+cu121、
> RTX 3050 Ti Laptop（4GB）、驱动 616.92、CUDA 可用。

## 1. 前置检查

- Python 3.10–3.12
- git
- NVIDIA 驱动：运行 `nvidia-smi`，确认 GPU 和显存
- `py -3 --version` 正常输出（本机 `python` 指向 Microsoft Store 别名，不要用）

## 2. 虚拟环境

```bash
# Windows（Git Bash）
py -3 -m venv .venv
source .venv/Scripts/activate

# cmd / PowerShell: .venv\Scripts\activate
# macOS / Linux:   python3 -m venv .venv && source .venv/bin/activate

python -m pip install -U pip
```

如果只是临时验证、不想重复下载 PyTorch，可建带系统包的 venv：

```bash
py -3 -m venv --system-site-packages .venv
```

正式实验仍建议独立安装，保证 `pip freeze` 可复现。

## 3. 安装 PyTorch

本机驱动支持 CUDA 12.1 的 wheel。锁定 2.5.x，**不要升级到 2.6+**（上游
`torch.load` 未传 `weights_only=False`，默认值变化会导致权重加载失败）。

```bash
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121

# 仅 CPU
# pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

> 已知问题（2026-10-27 检查）：本机系统 Python 里是 torch 2.5.1+cu121 配
> torchvision 0.26.0，两者版本不匹配，`import torchvision` 会报
> `RuntimeError: operator torchvision::nms does not exist`。不要用系统 Python
> 直接跑基线；在 `.venv` 里安装配对的 torchvision 0.20.1。如果 venv 用了
> `--system-site-packages`，venv 内安装的 0.20.1 会优先于系统版本，不必重新
> 下载 torch。

## 4. 安装其余依赖

```bash
pip install -r requirements.txt
pip install pytest ruff
pip install timm==0.9.16 yacs==0.1.8 opencv-python
pip freeze > requirements.lock
```

`timm` 先锁 0.9.16；如果与你的 Python/驱动不兼容，再试 0.6.x，并把实际版本
写进 `requirements.lock`。装上后先跑：

```bash
python -c "import timm, yacs, cv2; print(timm.__version__, cv2.__version__)"
```

## 5. 拉取上游基线

```bash
bash scripts/setup_third_party.sh
```

脚本会按固定 commit 克隆 TransReID / CLIP-ReID，并应用两个补丁：
`transreid-py312.patch`（修复 `torch._six`）和 `transreid-timm.patch`
（兼容新版 timm 的 `RandomErasing` 参数）。

## 6. 验证

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest -q
ruff check .
```

预期：`2.5.1+cu121 True`；测试全部通过。

## 7. 记录

| 项 | 值 |
|---|---|
| 记录日期 | 2026-10-27 |
| Python | 3.12.10（`py -3`） |
| PyTorch（系统） | 2.5.1+cu121 |
| CUDA / 驱动 | wheel cu121 / 驱动 616.92，CUDA UMD 13.4 |
| GPU | RTX 3050 Ti Laptop，4096 MiB |
| 项目 `.venv` | 2026-10-27 创建（`--system-site-packages`）；torchvision 0.20.1+cu121、timm 0.9.16、yacs 0.1.8、pytest、ruff；`requirements.lock` 待干净环境生成 |
| 基线依赖 | timm 0.9.16 / yacs 0.1.8 / cv2（系统）；TransReID 上游导入冒烟通过 |
