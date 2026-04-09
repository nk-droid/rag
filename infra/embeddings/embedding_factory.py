from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

def _model_name(config):
    return config.get("model") or config.get("model_name")

def get_embeddings(config):
    provider = config["provider"]
    model_name = _model_name(config)

    if provider == "openai":
        return OpenAIEmbeddings(model=model_name)

    elif provider == "huggingface":
        return HuggingFaceEmbeddings(model_name=model_name)

    elif provider == "ollama":
        return OllamaEmbeddings(model=model_name)

    else:
        raise ValueError("Unsupported embedding provider")
