from dataclasses import dataclass, field
from typing import Any, Iterator

Metadata = dict[str, Any]
RetrievalPlan = dict[str, Any]

@dataclass(slots=True)
class RetrievedChunk:
    id: str
    text: str
    score: float = 0.0
    metadata: Metadata = field(default_factory=dict)

@dataclass(slots=True)
class Chunk:
    text: str
    index: int
    metadata: Metadata = field(default_factory=dict)

@dataclass(slots=True)
class MemoryRecord:
    id: str
    content: str
    metadata: Metadata = field(default_factory=dict)

@dataclass(slots=True)
class EvaluationResult:
    metric: str
    score: float
    details: Metadata = field(default_factory=dict)

StreamingText = Iterator[str]
