from typing import Any
from abc import ABC, abstractmethod

class BaseCache(ABC):
    @abstractmethod
    def get(self, key: str) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl_sec: int | None = None) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def delete(self, key: str) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def has(self, key: str) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def clear(self, prefix: str | None = None) -> int:
        raise NotImplementedError