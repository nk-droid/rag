from pathlib import Path

class LocalVectorDB:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[dict[str, object]]:
        raise NotImplementedError

    def add_records(self, records) -> int:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError
