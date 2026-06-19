from pathlib import Path

from components._base import ComponentSettings
from components.ingestion.base_loader import BaseLoader
from components.ingestion.ingestion_schema import SourceDocument

LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescriptreact",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
}

class CodeLoaderSettings(ComponentSettings):
    _CONFIG_PATH = "ingestion.code"

    encoding: str = "utf-8"

class CodeLoader(BaseLoader):
    def __init__(self, settings: CodeLoaderSettings) -> None:
        self.settings = settings

    def load(self, source: str) -> list[SourceDocument]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        
        try:
            text = path.read_text(encoding=self.settings.encoding)
        except UnicodeDecodeError:
            text = path.read_text(encoding=self.settings.encoding, errors="ignore")

        suffix = path.suffix.lower()
        return [
            SourceDocument(
                text=text,
                source=str(path),
                metadata={
                    "loader": self.__class__.__name__,
                    "suffix": suffix,
                    "language": self._language_for(path),
                    "file_type": self._file_type_for(path),
                    "path": str(path)
                }
            )
        ]

    @staticmethod
    def _language_for(path: Path) -> str:
        if path.name in {"Makefile", "Dockerfile"}:
            return path.name.lower()
        return LANGUAGE_BY_SUFFIX.get(path.suffix.lower(), "text")

    @staticmethod
    def _file_type_for(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".py", ".js", ".ts", ".tsx"}:
            return "source_code"
        
        if suffix in {".yaml", ".yml", ".json", ".toml"} or path.name in {"Dockerfile", "Makefile"}:
            return "config"
        
        if suffix in {".md", ".markdown", ".txt"}:
            return "documentation"
        
        return "text"