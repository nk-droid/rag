import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from api.schemas import SourceRecord
from components.ingestion.repo_cloner import RepoCheckout

REPO_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = REPO_ROOT / "data" / "uploads"
MANIFEST_PATH = UPLOAD_DIR / "sources_manifest.json"

TEXT_SUFFIXES = {".txt", ".log"}
MARKDOWN_SUFFIXES = {".md", ".markdown"}
PUBLIC_REPO_SCHEMES = {"http", "https", "git", "ssh"}

class SourceStore:
    def __init__(self, manifest_path: Path = MANIFEST_PATH) -> None:
        self.manifest_path = manifest_path
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def list_sources(self) -> list[SourceRecord]:
        payload = self._read_manifest()
        sources: list[SourceRecord] = []
        for item in payload.get("sources", []):
            if not isinstance(item, dict):
                continue
            try:
                sources.append(_validate_source_record(item))
            except Exception:
                continue

        return sorted(sources, key=lambda source: source.created_at, reverse=True)

    def get_sources_by_ids(self, source_ids: list[str]) -> list[SourceRecord]:
        source_map = {source.id: source for source in self.list_sources()}
        return [source_map[source_id] for source_id in source_ids if source_id in source_map]

    def add_source(
        self,
        *,
        name: str,
        source_type: str,
        loader: str,
        path: str,
        size_bytes: int | None,
        repo_url: str | None = None,
        branch: str | None = None,
        commit_sha: str | None = None,
    ) -> SourceRecord:
        source = SourceRecord(
            id=f"src_{uuid4().hex[:12]}",
            name=name,
            source_type=source_type,
            loader=loader,
            path=path,
            size_bytes=size_bytes,
            repo_url=repo_url,
            branch=branch,
            commit_sha=commit_sha,
            created_at=datetime.now(timezone.utc),
        )

        payload = self._read_manifest()
        raw_sources = payload.get("sources")
        if not isinstance(raw_sources, list):
            raw_sources = []

        raw_sources.append(_dump_model_json(source))
        payload["sources"] = raw_sources
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write_manifest(payload)

        return source

    def add_repository_source(self, checkout: RepoCheckout) -> SourceRecord:
        repo_name = _repo_name_from_url(checkout.repo_url)
        return self.add_source(
            name=f"{repo_name}@{checkout.branch}",
            source_type="repository",
            loader="repo_loader",
            path=str(checkout.working_tree),
            size_bytes=_directory_size(checkout.working_tree),
            repo_url=checkout.repo_url,
            branch=checkout.branch,
            commit_sha=checkout.commit_sha,
        )

    def resolve_loader_for_path(self, path: Path) -> tuple[str, str]:
        if path.is_dir():
            return "directory", "directory_loader"

        suffix = path.suffix.lower()
        if suffix in MARKDOWN_SUFFIXES:
            return "file", "markdown_loader"

        if suffix in TEXT_SUFFIXES:
            return "file", "text_loader"

        return "file", "document_loader"

    def persist_uploaded_file(self, *, filename: str, contents: bytes) -> SourceRecord:
        safe_name = Path(filename or "upload.txt").name
        suffix = Path(safe_name).suffix.lower()

        if suffix in MARKDOWN_SUFFIXES:
            loader = "markdown_loader"
        elif suffix in TEXT_SUFFIXES:
            loader = "text_loader"
        else:
            raise ValueError(
                f"Unsupported file extension '{suffix or 'unknown'}'. "
                "Supported: .md, .markdown, .txt, .log"
            )

        source_dir = UPLOAD_DIR / "files"
        source_dir.mkdir(parents=True, exist_ok=True)

        file_id = f"upload_{uuid4().hex[:12]}"
        stored_path = source_dir / f"{file_id}_{safe_name}"
        stored_path.write_bytes(contents)

        return self.add_source(
            name=safe_name,
            source_type="file",
            loader=loader,
            path=str(stored_path),
            size_bytes=len(contents),
        )

    def _read_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"sources": [], "updated_at": datetime.now(timezone.utc).isoformat()}

        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"sources": [], "updated_at": datetime.now(timezone.utc).isoformat()}

        return payload if isinstance(payload, dict) else {"sources": []}

    def _write_manifest(self, payload: dict[str, Any]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def _validate_source_record(payload: dict[str, Any]) -> SourceRecord:
    validator = getattr(SourceRecord, "model_validate", None)
    if callable(validator):
        return validator(payload)
    parser = getattr(SourceRecord, "parse_obj")
    return parser(payload)

def _dump_model_json(model: SourceRecord) -> dict[str, Any]:
    dumper = getattr(model, "model_dump", None)
    if callable(dumper):
        return dumper(mode="json")
    return model.dict()

def validate_public_repo_url(repo_url: str) -> str:
    value = repo_url.strip()
    if not value:
        raise ValueError("Repository URL cannot be empty.")

    parsed = urlparse(value)
    if parsed.scheme in {"http", "https", "git"} and parsed.netloc:
        return value

    if parsed.scheme == "ssh" and parsed.netloc:
        return value

    if re.match(r"^git@[^:]+:[^/].+", value):
        return value

    raise ValueError("Use a public repository URL such as https://github.com/owner/repo.git.")

def _repo_name_from_url(repo_url: str) -> str:
    tail = repo_url.rstrip("/").split("/")[-1]
    if ":" in tail:
        tail = tail.split(":")[-1]
    return tail[:-4] if tail.endswith(".git") else tail or "repository"

def _directory_size(root: Path) -> int | None:
    if not root.exists() or not root.is_dir():
        return None
    total = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total
