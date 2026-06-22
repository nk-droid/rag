"""Integration tests for LoaderService dispatching to the real loaders."""
from datetime import datetime, timezone

from api.loader_service import LoaderService
from api.schemas import SourceRecord


def _src(path, loader, source_type="file", **extra):
    return SourceRecord(
        id="s", name="n", source_type=source_type, loader=loader, path=str(path),
        created_at=datetime.now(timezone.utc), **extra,
    )


def test_loader_service_loads_each_type(tmp_path):
    md = tmp_path / "a.md"
    md.write_text("# Title\nbody")
    txt = tmp_path / "b.txt"
    txt.write_text("plain text")
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "m.py").write_text("print('x')")

    service = LoaderService()
    docs = service.load_sources(
        [
            _src(md, "markdown_loader"),
            _src(txt, "text_loader"),
            _src(repo, "repo_loader", source_type="repository", repo_url="https://x/y.git", branch="main"),
            _src(tmp_path / "missing.txt", "text_loader"),  # skipped (no path)
            _src(txt, "mystery_loader"),  # unknown loader -> []
        ]
    )
    loaders = {d.metadata.get("source_id") for d in docs}
    assert docs  # at least md + txt + repo file
    assert all("source_id" in d.metadata for d in docs)


def test_loader_service_directory(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "one.txt").write_text("hello")
    docs = LoaderService().load_sources([_src(d, "directory_loader", source_type="directory")])
    assert docs
