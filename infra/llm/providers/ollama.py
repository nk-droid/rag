from langchain_ollama import ChatOllama
from infra.llm.llm_wrapper import BaseLLM

class OllamaLLM(BaseLLM):
    def __init__(self, model):
        self.llm = ChatOllama(model=model)

    def invoke(self, prompt, **kwargs):
        return self.llm.invoke(prompt, **kwargs).content
    
    def stream(self, prompt, **kwargs):
        for chunk in self.llm.stream(prompt, **kwargs):
            yield chunk.content if hasattr(chunk, "content") else chunk
