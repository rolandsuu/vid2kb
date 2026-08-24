from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llama_index.core.schema import Node
    from llama_index.vector_stores.postgres import PGVectorStore


def _embed_model():
    from llama_index.embeddings.ollama import OllamaEmbedding

    from vid2kb.config import settings

    return OllamaEmbedding(
        model_name=settings.embed_model,
        base_url=settings.ollama_base_url,
    )


def _async_connection_string(sync_url: str) -> str:
    return sync_url.replace('+psycopg', '+asyncpg')


def ensure_store(table_name: str = 'vid2kb_nodes') -> PGVectorStore:
    from vid2kb.config import settings

    if settings.vector_store != 'pgvector':
        raise NotImplementedError(
            f'vector_store={settings.vector_store} not wired yet; only pgvector'
        )

    from llama_index.vector_stores.postgres import PGVectorStore

    return PGVectorStore(
        connection_string=settings.pgvector_database_url,
        async_connection_string=_async_connection_string(settings.pgvector_database_url),
        table_name=table_name,
        embed_dim=settings.embed_dims,
    )


def _split_markdown(md_text: str) -> list[Node]:
    try:
        from llama_index.core.node_parser import MarkdownNodeParser
        from llama_index.core.schema import Document

        return MarkdownNodeParser().get_nodes_from_documents([Document(text=md_text)])
    except Exception:
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.schema import TextNode

        chunks = SentenceSplitter(chunk_size=800, chunk_overlap=100).split_text(md_text)
        return [TextNode(text=chunk) for chunk in chunks]


def ingest_document(
    md_text: str,
    metadata: dict,
    doc_id: str,
    table_name: str = 'vid2kb_nodes',
) -> int:
    from llama_index.core import StorageContext, VectorStoreIndex

    nodes = _split_markdown(md_text)
    for node in nodes:
        node.metadata.update(metadata)
        node.metadata['doc_id'] = doc_id

    storage_context = StorageContext.from_defaults(vector_store=ensure_store(table_name))
    VectorStoreIndex(nodes=nodes, embed_model=_embed_model(), storage_context=storage_context)
    return len(nodes)


def query_documents(
    question: str,
    top_k: int = 5,
    table_name: str = 'vid2kb_nodes',
) -> list[dict]:
    from llama_index.core import VectorStoreIndex

    index = VectorStoreIndex.from_vector_store(ensure_store(table_name), embed_model=_embed_model())
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(question)
    return [
        {
            'node_id': node.node_id,
            'score': float(node.score) if node.score else 0.0,
            'text': node.node.get_content(),
            'metadata': dict(node.node.metadata),
        }
        for node in nodes
    ]
