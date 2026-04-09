from infra.llm.providers.openai import OpenAILLM
from infra.llm.providers.anthropic import AnthropicLLM
from infra.llm.providers.huggingface import HuggingFaceLLM
from infra.llm.providers.ollama import OllamaLLM

def get_llm(config):
    # TODO
    provider = config["models"]["llm"]["provider"]

    if provider == "openai":
        return OpenAILLM(model=config["models"]["llm"]["model_name"])

    elif provider == "anthropic":
        return AnthropicLLM(model=config["models"]["llm"]["model_name"])

    elif provider == "huggingface":
        return HuggingFaceLLM(repo_id=config["models"]["llm"]["model_name"])

    elif provider == "ollama":
        return OllamaLLM(model=config["models"]["llm"]["model_name"])

    else:
        raise ValueError(f"Unsupported provider: {provider}")