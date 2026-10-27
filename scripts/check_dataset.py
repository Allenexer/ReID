#!/usr/bin/env python
"""Build and validate a dataset manifest.

Examples (from the repo root)::

    python -m scripts.check_dataset --config configs/datasets/market1501.yaml
    python scripts/check_dataset.py --dataset market1501 --root data/raw

The manifest is written to ``data/processed/manifests/<dataset>.json`` and
is the input for ``scripts/evaluate.py`` and the baseline adapters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.reid.manifest import (  # noqa: E402
    build_market1501_manifest,
    verify_manifest,
)

try:
    import yaml
except ImportError:  # pragma: no cover - requirements.txt includes PyYAML
    yaml = None


def _load_config(path: Path) -> dict:
    if yaml is None:
        raise SystemExit("PyYAML is required for --config; install requirements.txt")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a YAML mapping")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="dataset YAML config")
    parser.add_argument(
        "--dataset",
        default="market1501",
        choices=["market1501"],
        help="dataset adapter (more adapters land with the course data)",
    )
    parser.add_argument("--root", type=Path, help="dataset root")
    parser.add_argument("--out", type=Path, help="manifest output path")
    args = parser.parse_args()

    cfg = _load_config(args.config) if args.config else {}
    dataset = str(cfg.get("name", args.dataset))
    root = args.root or Path(cfg.get("root", "data/raw"))
    out = args.out or Path(
        cfg.get("manifest", f"data/processed/manifests/{dataset}.json")
    )
    expected = cfg.get("expected", {}) or {}

    if dataset != "market1501":
        raise SystemExit(f"unsupported dataset: {dataset}")

    manifest = build_market1501_manifest(root, name=dataset)
    stats = manifest.stats()
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    checks = [
        ("query_images", stats["query"]["images"]),
        ("query_ids", stats["query"]["pids"]),
        ("train_images", stats["train"]["images"]),
        ("train_ids", stats["train"]["pids"]),
    ]
    for key, actual in checks:
        if key in expected and actual != expected[key]:
            print(f"WARNING: {key} = {actual}, expected {expected[key]}")

    issues = verify_manifest(manifest)
    for issue in issues:
        print(f"WARNING: {issue}")

    manifest.save(out)
    print(f"manifest written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
