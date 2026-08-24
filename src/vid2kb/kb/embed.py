from __future__ import annotations


def embed_texts(texts: list[str]) -> list[list[float]]:
    from llama_index.embeddings.ollama import OllamaEmbedding

    from vid2kb.config import settings

    model = OllamaEmbedding(
        model_name=settings.embed_model,
        base_url=settings.ollama_base_url,
    )
    return model.get_text_embedding_batch(texts)
