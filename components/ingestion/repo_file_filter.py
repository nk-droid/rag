from fnmatch import fnmatch
from pathlib import Path

from components._base import ComponentSettings

DEFAULT_INCLUDE_PATTERNS = [
    "**/*.py",
    "**/*.md",
    "**/*.markdown",
    "**/*.txt",
    "**/*.yaml",
    "**/*.yml",
    "**/*.json",
    "**/*.toml",
    "**/*.js",
    "**/*.ts",
    "**/*.tsx",
    "Dockerfile",
    "**/Dockerfile",
    "Makefile",
    "**/Makefile",
    "requirements.txt",
    "**/requirements.txt",
    "pyproject.toml",
    "**/pyproject.toml",
    "package.json",
    "**/package.json",
]

DEFAULT_EXCLUDE_PATTERNS = [
    ".git/**",
    "**/.git/**",
    "node_modules/**",
    "**/node_modules/**",
    "venv/**",
    "**/venv/**",
    ".venv/**",
    "**/.venv/**",
    "__pycache__/**",
    "**/__pycache__/**",
    "dist/**",
    "**/dist/**",
    "build/**",
    "**/build/**",
    "coverage/**",
    "**/coverage/**",
    ".mypy_cache/**",
    "**/.mypy_cache/**",
    ".pytest_cache/**",
    "**/.pytest_cache/**",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.webp",
    "*.pdf",
    "*.pyc",
    "*.lock",
    "*.sqlite",
    "*.db",
    "*.log",
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa",
    "credentials*",
    "secrets*",
    "service-account*.json",
]

class RepoFileFilterSettings(ComponentSettings):
    _CONFIG_PATH = "ingestion.repo_file_filter"

    include_patterns: list[str] = DEFAULT_INCLUDE_PATTERNS
    exclude_patterns: list[str] = DEFAULT_EXCLUDE_PATTERNS
    max_file_size_kb: int = 512
    max_files: int = 5000

class RepoFileFilter:
    def __init__(self, settings: RepoFileFilterSettings) -> None:
        self.settings = settings

    def iter_files(self, root: Path) -> list[Path]:
        repo_root = Path(root).resolve()
        
        if not repo_root.exists() or not repo_root.is_dir():
            return []
        
        files: list[Path] = []
        for path in sorted(repo_root.rglob("*")):
            if not path.is_file():
                continue

            if not self.should_include(path, repo_root):
                continue

            files.append(path)
            if len(files) >= self.settings.max_files:
                break
            
        return files

    def should_include(self, path: Path, root: Path) -> bool:
        try:
            rel_path = path.relative_to(root).as_posix()
        except ValueError:
            return False
        
        if self._matches(rel_path, self.settings.exclude_patterns):
            return False

        if not self._matches(rel_path, self.settings.include_patterns):
            return False
        
        max_bytes = max(1, int(self.settings.max_file_size_kb)) * 1024
        try:
            if path.stat().st_size > max_bytes:
                return False
        except OSError:
            return False
        
        return not self._looks_binary(path)

    @staticmethod
    def _matches(path: str, patterns: list[str]) -> bool:
        name = Path(path).name
        return any(fnmatch(path, pattern) or fnmatch(name, pattern) for pattern in patterns)

    @staticmethod
    def _looks_binary(path: Path) -> bool:
        try:
            sample = path.read_bytes()[:2048]
        except OSError:
            return True

        return b"\x00" in sample or b"\xff" in sample