import ast
import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

from components._base import ComponentSettings
from components.shared_types import Chunk

@dataclass
class _Span:
    title: str
    text: str
    chunk_type: str
    symbol: str | None
    start_line: int
    end_line: int

class CodeAwareChunkerSettings(ComponentSettings):
    _CONFIG_PATH = "chunking.code_aware"

    chunk_size: int = 900
    chunk_overlap: int = 80
    include_import_chunk: bool = True

class CodeAwareChunker:
    def __init__(self, settings: CodeAwareChunkerSettings) -> None:
        self.settings = settings

    def chunk(self, text: str) -> list[Chunk]:
        spans = self._python_spans(text)
        if not spans:
            spans = self._markdown_spans(text)

        if not spans:
            spans = list(self._window_spans(text))


        chunks: list[Chunk] = []
        for idx, span in enumerate(spans):
            chunks.append(
                Chunk(
                    text=span.text,
                    index=idx,
                    metadata={
                        "chunk_id": self._chunk_id(span.text, idx),
                        "chunk_type": span.chunk_type,
                        "symbol": span.symbol,
                        "start_line": span.start_line,
                        "end_line": span.end_line,
                        "title": span.title
                    }
                )
            )

        return chunks

    def _python_spans(self, text: str) -> list[_Span]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []

        lines = text.splitlines()
        spans: list[_Span] = []

        if self.settings.include_import_chunk:
            import_lines = [
                node.lineno
                for node in tree.body
                if isinstance(node, (ast.Import, ast.ImportFrom)) and hasattr(node, "lineno")
            ]
            if import_lines:
                start, end = min(import_lines), max(import_lines)
                spans.append(
                    _Span(
                        title="imports",
                        text="\n".join(lines[start - 1 : end]),
                        chunk_type="imports",
                        symbol=None,
                        start_line=start,
                        end_line=end,
                    )
                )

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                spans.append(self._span_for_node(node, lines, "class", node.name))
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        spans.append(self._span_for_node(child, lines, "method", f"{node.name}.{child.name}"))

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                spans.append(self._span_for_node(node, lines, "function", node.name))

        return [span for span in spans if span.text.strip()]

    @staticmethod
    def _span_for_node(node: ast.AST, lines: list[str], chunk_type: str, symbol: str) -> _Span:
        start = int(getattr(node, "lineno", 1))
        end = int(getattr(node, "end_lineno", start))
        return _Span(
            title=symbol,
            text="\n".join(lines[start - 1 : end]),
            chunk_type=chunk_type,
            symbol=symbol,
            start_line=start,
            end_line=end,
        )

    def _markdown_spans(self, text: str) -> list[_Span]:
        if "#" not in text[:5000]:
            return []

        lines = text.splitlines()
        headings: list[tuple[int, str]] = []
        for idx, line in enumerate(lines, start=1):
            if re.match(r"^#{1,6}\s+", line):
                headings.append((idx, line.strip("# ").strip()))

        if not headings:
            return []

        spans: list[_Span] = []
        for pos, (start, title) in enumerate(headings):
            end = headings[pos + 1][0] - 1 if pos + 1 < len(headings) else len(lines)
            body = "\n".join(lines[start - 1 : end])
            spans.append(
                _Span(
                    title=title,
                    text=body,
                    chunk_type="section",
                    symbol=title,
                    start_line=start,
                    end_line=end,
                )
            )
        return [span for span in spans if span.text.strip()]

    def _window_spans(self, text: str) -> Iterable[_Span]:
        size = max(100, int(self.settings.chunk_size))
        overlap = max(0, min(int(self.settings.chunk_overlap), size // 2))
        start = 0
        index = 0
        while start < len(text):
            end = min(len(text), start + size)
            chunk_text = text[start:end]
            start_line = text[:start].count("\n") + 1
            end_line = text[:end].count("\n") + 1
            yield _Span(
                title=f"chunk-{index}",
                text=chunk_text,
                chunk_type="text",
                symbol=None,
                start_line=start_line,
                end_line=end_line,
            )
            if end == len(text):
                break
            
            start = max(end - overlap, start + 1)
            index += 1

    @staticmethod
    def _chunk_id(text: str, index: int) -> str:
        digest = hashlib.sha1(f"{index}:{text}".encode("utf-8")).hexdigest()[:16]
        return f"chunk:{digest}"