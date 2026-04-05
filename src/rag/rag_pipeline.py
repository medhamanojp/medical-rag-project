"""
RAG Pipeline — Medical knowledge retrieval using MedQuAD from HuggingFace.

Dataset: "lavita/medical-qa-datasets" (subset: "all-processed")
         ~16k real medical Q&A pairs from NLM/NIH sources (MedQuAD).

Pipeline:
  1. Load dataset from HuggingFace (cached after first download)
  2. Embed questions+answers with sentence-transformers (all-MiniLM-L6-v2)
  3. Build a FAISS index for fast nearest-neighbour retrieval
  4. At query time, embed the query and return top-k most relevant passages

The index is cached to disk so it only needs to be built once.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).parent.parent.parent / ".rag_cache"
_INDEX_PATH = _CACHE_DIR / "faiss.index"
_DOCS_PATH = _CACHE_DIR / "docs.pkl"

_CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_DATASET_NAME = "lavita/medical-qa-datasets"
HF_DATASET_CONFIG = "all-processed"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_DOCS = 8000   # cap to keep index fast; dataset has ~16k entries


# ---------------------------------------------------------------------------
# RAG pipeline class
# ---------------------------------------------------------------------------


class MedicalRAGPipeline:
    """
    Retrieval-Augmented Generation pipeline over MedQuAD.

    Builds / loads a FAISS index of medical Q&A embeddings.
    At query time returns the top-k most semantically similar documents.
    """

    def __init__(self):
        self._index = None
        self._docs: list[dict[str, str]] = []
        self._embedder = None
        self._load_or_build()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 3) -> list[dict[str, str]]:
        """
        Retrieve top-k documents most relevant to the query.

        Returns:
            List of dicts with keys: question, answer, source (optional).
        """
        if self._index is None or not self._docs:
            logger.warning("RAG index not available — returning empty results")
            return []

        q_vec = self._embed([query])          # (1, dim)
        distances, indices = self._index.search(q_vec, k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self._docs):
                results.append(self._docs[idx])
        return results

    # ------------------------------------------------------------------
    # Index build / load
    # ------------------------------------------------------------------

    def _load_or_build(self) -> None:
        if _INDEX_PATH.exists() and _DOCS_PATH.exists():
            logger.info("Loading cached FAISS index from %s", _CACHE_DIR)
            self._load_index()
        else:
            logger.info("Building FAISS index from HuggingFace dataset...")
            self._build_index()

    def _load_index(self) -> None:
        import faiss
        self._index = faiss.read_index(str(_INDEX_PATH))
        with open(_DOCS_PATH, "rb") as f:
            self._docs = pickle.load(f)
        logger.info("Loaded %d documents from cache", len(self._docs))

    def _build_index(self) -> None:
        import faiss
        from datasets import load_dataset

        logger.info("Downloading %s / %s from HuggingFace...", HF_DATASET_NAME, HF_DATASET_CONFIG)
        try:
            ds = load_dataset(
                HF_DATASET_NAME,
                HF_DATASET_CONFIG,
                split="train",
                trust_remote_code=True,
            )
        except Exception as e:
            logger.error("Failed to load HuggingFace dataset: %s", e)
            logger.info("Falling back to empty RAG index")
            return

        # Build document list — keep entries with both question and answer
        docs: list[dict[str, str]] = []
        for row in ds:
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            if q and a:
                docs.append({
                    "question": q,
                    "answer": a,
                    "source": row.get("source", "MedQuAD"),
                })
            if len(docs) >= MAX_DOCS:
                break

        if not docs:
            logger.warning("No usable documents found in dataset")
            return

        logger.info("Embedding %d documents with %s...", len(docs), EMBED_MODEL)
        texts = [f"{d['question']} {d['answer'][:300]}" for d in docs]
        embeddings = self._embed(texts)   # (N, dim)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)   # inner-product = cosine on L2-normalised vecs
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        # Persist
        faiss.write_index(index, str(_INDEX_PATH))
        with open(_DOCS_PATH, "wb") as f:
            pickle.dump(docs, f)

        self._index = index
        self._docs = docs
        logger.info("FAISS index built and saved (%d vectors, dim=%d)", len(docs), dim)

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised sentence embeddings as float32 numpy array."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(EMBED_MODEL)
            logger.info("Loaded sentence-transformer: %s", EMBED_MODEL)

        vecs = self._embedder.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 200,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_rag_instance: MedicalRAGPipeline | None = None


def get_rag_pipeline() -> MedicalRAGPipeline:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = MedicalRAGPipeline()
    return _rag_instance
