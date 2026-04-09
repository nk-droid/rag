from dataclasses import dataclass, field
from typing import Any

Metadata = dict[str, Any]

@dataclass(slots=True)
class SourceDocument:
    text: str
    source: str
    metadata: Metadata = field(default_factory=dict)

@dataclass(slots=True)
class IngestionJob:
    sources: list[str]
    recursive: bool = True
    metadata: Metadata = field(default_factory=dict)
