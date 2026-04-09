from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseLLM(ABC):
    @abstractmethod
    def invoke(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    def stream(self, prompt: str, **kwargs):
        raise NotImplementedError
