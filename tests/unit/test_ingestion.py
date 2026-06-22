"""Unit tests for repo cloning (git mocked), file filtering, and code/repo loaders."""
import subprocess
from pathlib import Path

import pytest

from components.ingestion import repo_cloner as rc
from components.ingestion.code_loader import CodeLoader, CodeLoaderSettings
from components.ingestion.repo_cloner import RepoCloneError, RepoCloner, RepoClonerSettings
from components.ingestion.repo_file_filter import RepoFileFilter, RepoFileFilterSettings
from components.ingestion.repo_loader import RepoLoader, RepoLoaderSettings


# --------------------------------------------------------------------------- #
# RepoCloner — pure helpers
# --------------------------------------------------------------------------- #
def test_source_id_from_url():
    assert RepoCloner._source_id_from_url("https://github.com/a/MyRepo.git").startswith("myrepo-")
    assert RepoCloner._source_id_from_url("https://x/").startswith("x-")
    # tail of only special chars falls back to the "repo" slug
    assert RepoCloner._source_id_from_url("https://h/===.git").startswith("repo-")


def test_with_token():
    assert RepoCloner._with_token("https://github.com/a/b", None) == "https://github.com/a/b"
    assert RepoCloner._with_token("git@github.com:a/b", "tok") == "git@github.com:a/b"
    assert RepoCloner._with_token("https://github.com/a/b", "tok") == "https://x-access-token:tok@github.com/a/b"


class _Completed:
    def __init__(self, stdout=""):
        self.stdout = stdout


def _git_factory(checkout_marker_text="print('x')"):
    def fake_run(cmd, cwd=None, check=True, capture_output=True, text=True, timeout=120):
        sub = cmd[1]
        if sub == "clone":
            dest = Path(cmd[-1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "main.py").write_text(checkout_marker_text)
            return _Completed("")
        if sub == "rev-parse":
            return _Completed("deadbeef\n")
        if sub == "branch":
            return _Completed("main\n")
        return _Completed("")

    return fake_run


def test_clone_or_update_fresh_clone(tmp_path, monkeypatch):
    monkeypatch.setattr(rc.subprocess, "run", _git_factory())
    cloner = RepoCloner(RepoClonerSettings(root_dir=str(tmp_path)))
    checkout = cloner.clone_or_update("https://github.com/a/b", branch="main", source_id="b")
    assert checkout.commit_sha == "deadbeef"
    assert checkout.working_tree.exists()
    assert checkout.manifest_path.exists()


def test_clone_or_update_existing_does_fetch(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, cwd=None, check=True, capture_output=True, text=True, timeout=120):
        calls.append(cmd[1])
        if cmd[1] == "rev-parse":
            return _Completed("sha123\n")
        if cmd[1] == "branch":
            return _Completed("main\n")
        return _Completed("")

    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    cloner = RepoCloner(RepoClonerSettings(root_dir=str(tmp_path)))
    (tmp_path / "b" / "working-tree").mkdir(parents=True)
    checkout = cloner.clone_or_update("https://github.com/a/b", source_id="b")
    assert checkout.commit_sha == "sha123"
    assert "fetch" in calls and "reset" in calls


def test_clone_or_update_empty_url_raises(tmp_path):
    with pytest.raises(RepoCloneError):
        RepoCloner(RepoClonerSettings(root_dir=str(tmp_path))).clone_or_update("  ")


def test_run_git_error_redacts_token(tmp_path, monkeypatch):
    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr="failed with secret-token")

    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    cloner = RepoCloner(RepoClonerSettings(root_dir=str(tmp_path)))
    with pytest.raises(RepoCloneError) as exc:
        cloner._run_git(["rev-parse"], cwd=None, redacted_token="secret-token")
    assert "***" in str(exc.value) and "secret-token" not in str(exc.value)


def test_run_git_timeout(tmp_path, monkeypatch):
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 120)

    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    cloner = RepoCloner(RepoClonerSettings(root_dir=str(tmp_path)))
    with pytest.raises(RepoCloneError):
        cloner._run_git(["fetch"], cwd=None)


def test_current_branch_handles_error(tmp_path, monkeypatch):
    cloner = RepoCloner(RepoClonerSettings(root_dir=str(tmp_path)))
    monkeypatch.setattr(cloner, "_run_git", lambda *a, **k: "feature\n")
    assert cloner._current_branch(tmp_path) == "feature"

    def _raise(*a, **k):
        raise RepoCloneError("no")

    monkeypatch.setattr(cloner, "_run_git", _raise)
    assert cloner._current_branch(tmp_path) is None


def test_enforce_repo_size(tmp_path):
    cloner = RepoCloner(RepoClonerSettings(root_dir=str(tmp_path), max_repo_size_mb=1))
    small = tmp_path / "small"
    small.mkdir()
    (small / "f.txt").write_text("hi")
    cloner._enforce_repo_size(small)  # under limit, no raise
    big = tmp_path / "big"
    big.mkdir()
    (big / "f.bin").write_bytes(b"x" * (1024 * 1024 + 10))
    with pytest.raises(RepoCloneError):
        cloner._enforce_repo_size(big)


# --------------------------------------------------------------------------- #
# RepoFileFilter
# --------------------------------------------------------------------------- #
def _make_repo(tmp_path):
    # include globs (e.g. **/*.py) require a path separator, so nest under src/.
    src = tmp_path / "src"
    src.mkdir()
    (src / "keep.py").write_text("print('ok')")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("nope")
    (src / "big.py").write_text("x" * (600 * 1024))  # over 512KB default
    (src / "img.png").write_bytes(b"\x00\x01")
    return tmp_path


def test_file_filter_iter_files(tmp_path):
    _make_repo(tmp_path)
    files = RepoFileFilter(RepoFileFilterSettings()).iter_files(tmp_path)
    names = {p.name for p in files}
    assert "keep.py" in names
    assert "x.js" not in names and "big.py" not in names and "img.png" not in names


def test_file_filter_missing_root_and_outside_path(tmp_path):
    rf = RepoFileFilter(RepoFileFilterSettings())
    assert rf.iter_files(tmp_path / "missing") == []
    assert rf.should_include(Path("/etc/hosts"), tmp_path) is False


def test_file_filter_matches_and_binary(tmp_path):
    assert RepoFileFilter._matches("a/b.py", ["**/*.py"]) is True
    assert RepoFileFilter._matches("Dockerfile", ["Dockerfile"]) is True
    binfile = tmp_path / "b.py"
    binfile.write_bytes(b"abc\x00def")
    assert RepoFileFilter._looks_binary(binfile) is True
    txt = tmp_path / "t.py"
    txt.write_text("clean")
    assert RepoFileFilter._looks_binary(txt) is False


def test_file_filter_max_files(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        (src / f"f{i}.py").write_text("x")
    files = RepoFileFilter(RepoFileFilterSettings(max_files=2)).iter_files(tmp_path)
    assert len(files) == 2


# --------------------------------------------------------------------------- #
# CodeLoader
# --------------------------------------------------------------------------- #
def test_code_loader_python(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("print('hi')")
    docs = CodeLoader(CodeLoaderSettings()).load(str(f))
    assert docs[0].metadata["language"] == "python"
    assert docs[0].metadata["file_type"] == "source_code"


def test_code_loader_dockerfile_md_and_unknown(tmp_path):
    docker = tmp_path / "Dockerfile"
    docker.write_text("FROM python")
    assert CodeLoader(CodeLoaderSettings()).load(str(docker))[0].metadata["language"] == "dockerfile"
    md = tmp_path / "r.md"
    md.write_text("# hi")
    assert CodeLoader(CodeLoaderSettings()).load(str(md))[0].metadata["file_type"] == "documentation"
    other = tmp_path / "x.bin"
    other.write_text("data")
    meta = CodeLoader(CodeLoaderSettings()).load(str(other))[0].metadata
    assert meta["language"] == "text" and meta["file_type"] == "text"


def test_code_loader_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        CodeLoader(CodeLoaderSettings()).load(str(tmp_path / "nope.py"))


# --------------------------------------------------------------------------- #
# RepoLoader
# --------------------------------------------------------------------------- #
def test_repo_loader_loads_with_relative_paths(tmp_path):
    # files must be nested for the default include globs to match
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("print(1)")
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "b.py").write_text("print(2)")
    docs = RepoLoader(RepoLoaderSettings()).load(str(tmp_path), metadata={"source_id": "repo"})
    rels = {d.metadata["relative_path"] for d in docs}
    assert rels == {"src/a.py", "pkg/b.py"}
    assert all(d.metadata["source_type"] == "repo_code" for d in docs)
    assert all(d.metadata["source_id"] == "repo" for d in docs)
