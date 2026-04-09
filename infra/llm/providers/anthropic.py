from langchain_anthropic import ChatAnthropic
from infra.llm.llm_wrapper import BaseLLM

class AnthropicLLM(BaseLLM):
    def __init__(self, model):
        self.llm = ChatAnthropic(model=model)

    def invoke(self, prompt, **kwargs):
        return self.llm.invoke(prompt, **kwargs).content

    def stream(self, prompt, **kwargs):
        for chunk in self.llm.stream(prompt, **kwargs):
            yield chunk.content if hasattr(chunk, "content") else chunk
