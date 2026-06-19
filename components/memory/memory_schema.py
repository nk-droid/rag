from dataclasses import dataclass, field
from typing import Any

@dataclass(slots=True)
class MemoryQuery:
    text: str
    top_k: int = 5

@dataclass(slots=True)
class MemoryRecordModel:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
