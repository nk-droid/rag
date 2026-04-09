from dataclasses import dataclass, field
from typing import Any

Metadata = dict[str, Any]

@dataclass(slots=True)
class IndexRecord:
    id: str
    text: str
    embedding: list[float]
    metadata: Metadata = field(default_factory=dict)
