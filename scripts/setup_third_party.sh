#!/usr/bin/env bash
# Clone the upstream baselines at pinned commits and apply local patches.
#
# Usage (from the repo root):
#   bash scripts/setup_third_party.sh
#
# third_party/ is gitignored: each teammate runs this script once instead of
# committing someone else's repository. Keep the pins below in sync with
# docs/BASELINES.md whenever a baseline is upgraded.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="$ROOT/third_party"

clone_at() {
  local url="$1"
  local dir="$2"
  local commit="$3"

  if [ -d "$dir/.git" ]; then
    echo "[skip] $dir already exists"
  else
    echo "[clone] $url"
    git clone "$url" "$dir"
  fi

  git -C "$dir" checkout --detach "$commit"
  echo "[pin] $(basename "$dir") -> $commit"
}

apply_patch() {
  local repo="$1"
  local patch_file="$2"

  if git -C "$repo" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    echo "[skip] already applied: $(basename "$patch_file")"
  elif git -C "$repo" apply --check "$patch_file" >/dev/null 2>&1; then
    git -C "$repo" apply "$patch_file"
    echo "[apply] $(basename "$patch_file")"
  else
    echo "[error] patch does not apply cleanly: $patch_file" >&2
    exit 1
  fi
}

mkdir -p "$TP"

clone_at "https://github.com/heshuting555/TransReID.git" \
  "$TP/TransReID" "dec55046fcdfadee14e2c28e2df89305d8f7557a"
clone_at "https://github.com/Syliz517/CLIP-ReID.git" \
  "$TP/CLIP-ReID" "eb1898b72c882875f478bebfc6d41644eece0a5d"

apply_patch "$TP/TransReID" "$ROOT/patches/transreid-py312.patch"
apply_patch "$TP/TransReID" "$ROOT/patches/transreid-timm.patch"

echo "third_party ready."
