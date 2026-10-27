import pytest

from src.reid.manifest import DatasetManifest, Sample


def test_manifest_roundtrip(tmp_path):
    """The shared JSON schema must survive save/load."""
    manifest = DatasetManifest(
        name="demo",
        root="/tmp/demo",
        query=[Sample("query/0001_c1s1_000001_00.jpg", 1, 1)],
        gallery=[Sample("bounding_box_test/0001_c2s1_000002_00.jpg", 1, 2)],
        protocol="demo",
    )

    out = tmp_path / "manifests" / "demo.json"
    manifest.save(out)
    loaded = DatasetManifest.load(out)

    assert loaded == manifest


@pytest.mark.skip(reason="A: implement the Market1501 parser first")
def test_market1501_parser_todo():
    """A should cover: filename parsing, junk (pid -1) removal,
    distractor (pid 0) kept in gallery, nested-root detection, and
    ``verify_manifest`` warnings."""
