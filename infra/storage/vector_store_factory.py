from infra.embeddings.embedding_factory import get_embeddings

def get_vector_store(config: dict):
    vector_store_config = config.get("vector_store", {})
    provider = vector_store_config.get("provider", "faiss")
    embedding_config = config.get("models", {}).get("embedding", {})
    embeddings = get_embeddings(embedding_config)

    if provider == "faiss":
        from infra.storage.faiss_store import LangChainFAISSStore

        persist_path = vector_store_config.get("path", "data/embeddings/faiss_index")
        return LangChainFAISSStore(embeddings=embeddings, persist_path=persist_path)

    if provider == "pinecone":
        from langchain_pinecone import PineconeVectorStore

        index_name = vector_store_config.get("index_name")
        namespace = vector_store_config.get("namespace")
        if not index_name:
            raise ValueError("Pinecone vector store requires 'vector_store.index_name' in config.")
        return PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            namespace=namespace,
        )

    raise ValueError(f"Unsupported vector store provider: {provider}")
