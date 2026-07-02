"""v1 hybrid retrieval (docs/10 T3.2).

`HybridRetriever` implements the stable `Retriever` interface: BM25 (lexical) + dense
(embeddings in a vector store) + cross-encoder rerank, always consent-scoped. Model
backends are behind ports; deterministic fakes drive tests, real adapters (Qdrant,
MedCPT/BGE, bge-reranker) load lazily.
"""

from .bm25 import BM25
from .embedder import HashEmbedder, HfEmbedder
from .eval import EvalQuery, EvalResult, evaluate, ndcg_at_k, recall_at_k, reciprocal_rank
from .hybrid import RETRIEVER_VERSION, HybridRetriever
from .ports import Embedder, Reranker, VectorStore
from .reranker import HfReranker, LexicalReranker
from .vector_store import InMemoryVectorStore, QdrantVectorStore

__all__ = [
    "RETRIEVER_VERSION",
    "BM25",
    "Embedder",
    "EvalQuery",
    "EvalResult",
    "HashEmbedder",
    "HfEmbedder",
    "HfReranker",
    "HybridRetriever",
    "InMemoryVectorStore",
    "LexicalReranker",
    "QdrantVectorStore",
    "Reranker",
    "VectorStore",
    "evaluate",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
