import re
import json
import hashlib
import subprocess
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone

from components._base import ComponentSettings

class RepoCloneError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class RepoCheckout:
    source_id: str
    repo_url: str
    branch: str
    working_tree: Path
    commit_sha: str
    manifest_path: Path

class RepoClonerSettings(ComponentSettings):
    _CONFIG_PATH = "ingestion.repo_cloner"

    root_dir: str = "data/repos"
    default_branch: str = "main"
    clone_depth: int = 1
    max_repo_size_mb: int = 500

class RepoCloner:
    def __init__(self, settings: RepoClonerSettings) -> None:
        self.settings = settings
        self.root_dir = Path(settings.root_dir)

    def clone_or_update(
        self,
        repo_url: str,
        branch: str | None = None,
        source_id: str | None = None,
        access_token: str | None = None
    ) -> RepoCheckout:
        repo_url = repo_url.strip()
        if not repo_url:
            raise RepoCloneError("repo_url cannot be empty")

        resolved_source_id = source_id or self._source_id_from_url(repo_url)
        checkout_dir = self.root_dir / resolved_source_id / "working-tree"
        manifest_path = self.root_dir / resolved_source_id / "manifest.json"

        self.root_dir.mkdir(parents=True, exist_ok=True)

        clone_url = self._with_token(repo_url, access_token)
        if not checkout_dir.exists():
            checkout_dir.parent.mkdir(parents=True, exist_ok=True)
            clone_args = ["clone", "--depth", str(max(1, int(self.settings.clone_depth)))]
            if branch:
                clone_args.extend(["--branch", branch])

            clone_args.extend([clone_url, str(checkout_dir)])

            self._run_git(
                clone_args,
                cwd=None,
                redacted_token=access_token
            )
        else:
            resolved_branch = branch or self._current_branch(checkout_dir) or self.settings.default_branch
            self._run_git(
                ["fetch", "origin", resolved_branch, "--depth", str(max(1, int(self.settings.clone_depth)))],
                cwd=str(checkout_dir),
                redacted_token=access_token
            )

            self._run_git(
                ["checkout", "-B", resolved_branch, f"origin/{resolved_branch}"],
                cwd=str(checkout_dir),
                redacted_token=access_token
            )
            
            self._run_git(
                ["reset", "--hard", f"origin/{resolved_branch}"],
                cwd=str(checkout_dir),
                redacted_token=access_token
            )

        resolved_branch = branch or self._current_branch(checkout_dir) or self.settings.default_branch
        commit_sha = self._run_git(
            ["rev-parse", "HEAD"],
            cwd=str(checkout_dir),
            redacted_token=access_token
        ).strip()
        self._enforce_repo_size(checkout_dir)

        manifest = {
            "source_id": resolved_source_id,
            "repo_url": repo_url,
            "branch": resolved_branch,
            "commit_sha": commit_sha,
            "working_tree": str(checkout_dir),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        manifest_path.write_text(
            json.dumps(manifest, indent=4, ensure_ascii=False),
            encoding="utf-8"
        )

        return RepoCheckout(
            source_id=resolved_source_id,
            repo_url=repo_url,
            branch=resolved_branch,
            working_tree=checkout_dir,
            commit_sha=commit_sha,
            manifest_path=manifest_path
        )

    def _run_git(
        self,
        args: list[str],
        cwd: Path | None,
        redacted_token: str | None = None
    ) -> str:
        cmd = ["git", *args]
        try:
            result = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                check=True,
                capture_output=True,
                text=True,
                timeout=120
            )

            return result.stdout
        except subprocess.CalledProcessError as e:
            stderr = e.stderr or ""
            if redacted_token:
                stderr = stderr.replace(redacted_token, "***")

            raise RepoCloneError(f"Git command failed: git {' '.join(args)}\n{stderr}") from e
        except subprocess.TimeoutExpired as e:
            raise RepoCloneError(f"Git command timeout: git {' '.join(args)}") from e

    def _enforce_repo_size(self, root: Path) -> None:
        max_bytes = max(1, int(self.settings.max_repo_size_mb)) * 1024 * 1024
        total = 0
        for path in root.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
            
            if total > max_bytes:
                raise RepoCloneError(f"Repo size exceeds limit({self.settings.max_repo_size_mb}MB): {root}")

    def _current_branch(self, checkout_dir: Path) -> str | None:
        try:
            branch = self._run_git(
                ["branch", "--show-current"],
                cwd=str(checkout_dir),
            ).strip()
        except RepoCloneError:
            return None
        return branch or None

    @staticmethod
    def _source_id_from_url(repo_url: str) -> str:
        tail = repo_url.rstrip('/').split('/')[-1]
        tail = tail[:-4] if tail.endswith(".git") else tail
        slug = re.sub(r"[^a-zA-Z0-9\-]+", "-", tail).strip("-").lower() or "repo"
        digest = hashlib.sha1(repo_url.encode('utf-8')).hexdigest()[:8]
        return f"{slug}-{digest}"

    @staticmethod
    def _with_token(repo_url: str, access_token: str | None) -> str:
        if not access_token:
            return repo_url
        
        if repo_url.startswith("https://"):
            return repo_url.replace("https://", f"https://x-access-token:{access_token}@", 1)
        
        return repo_url
