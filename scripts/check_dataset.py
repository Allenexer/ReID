#!/usr/bin/env python
"""Build and validate a dataset manifest.

This file is the shared CLI entry point. The dataset parser and the
validation checks are TODO for member A; see ``src/reid/manifest.py``.

Intended usage (from the repo root)::

    python -m scripts.check_dataset --config configs/datasets/market1501.yaml

The manifest should be written to
``data/processed/manifests/<dataset>.json`` and is the input for
``scripts/evaluate.py`` and the baseline adapters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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

    print(f"dataset={dataset} root={root} out={out}")
    # TODO(A): build the manifest with src.reid.manifest, validate it,
    # print the stats/issues, and save it to `out`.
    raise SystemExit(
        "TODO(A): manifest generation is not implemented yet; "
        "see src/reid/manifest.py"
    )


if __name__ == "__main__":
    raise SystemExit(main())
