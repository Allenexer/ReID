"""Minimal shared smoke test: the package can be imported."""

import src.reid


def test_package_imports():
    assert src.reid.__name__ == "src.reid"
