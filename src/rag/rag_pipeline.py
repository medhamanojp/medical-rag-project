"""
RAG pipeline — retrieves relevant medical knowledge for agent queries.

Knowledge base: MedQuAD (NIH/NLM medical Q&A dataset, ~16k pairs)
Embeddings:     sentence-transformers/all-MiniLM-L6-v2
Vector store:   FAISS (IndexFlatIP, cosine similarity)
Cache:          .rag_cache/ — index built once, reused on subsequent runs

Usage:
    retriever = get_retriever()
    results   = retriever.query("what causes migraines", top_k=3)
"""

import os
import pickle
import numpy as np
from pathlib import Path

CACHE_DIR   = Path(".rag_cache")
INDEX_PATH  = CACHE_DIR / "faiss.index"
DOCS_PATH   = CACHE_DIR / "docs.pkl"
DATASET_ID  = "keivalya/MedQuad-MedicalQnADataset"


# ---------------------------------------------------------------------------
# Build index
# ---------------------------------------------------------------------------

def _build_index():
    """
    Load MedQuAD from HuggingFace, embed all Q&A pairs, build a FAISS
    index, and cache everything to .rag_cache/.
    """
    import faiss
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    print("Building RAG index from MedQuAD dataset...")
    CACHE_DIR.mkdir(exist_ok=True)

    print("  Loading dataset...")
    ds  = load_dataset(DATASET_ID, split="train")
    df  = ds.to_pandas()

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    q_col = _find_col(df, ["question", "q", "query"])
    a_col = _find_col(df, ["answer", "a", "response"])

    df     = df[[q_col, a_col]].dropna().reset_index(drop=True)
    docs   = [
        f"Q: {row[q_col]}\nA: {row[a_col]}"
        for _, row in df.iterrows()
    ]
    print(f"  {len(docs)} Q&A pairs loaded.")

    print("  Embedding documents (this takes a minute on first run)...")
    model      = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(docs, batch_size=64, show_progress_bar=True,
                              normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    print("  Building FAISS index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])   # inner product = cosine (normalised)
    index.add(embeddings)

    # Save
    faiss.write_index(index, str(INDEX_PATH))
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(docs, f)

    print(f"  RAG index cached to {CACHE_DIR}/")
    return index, docs, model


def _find_col(df, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in {df.columns.tolist()}")


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------

class MedicalRetriever:
    """
    Loads the FAISS index from cache (or builds it on first run) and
    provides semantic similarity search over MedQuAD Q&A pairs.
    """

    def __init__(self):
        import faiss
        from sentence_transformers import SentenceTransformer

        if INDEX_PATH.exists() and DOCS_PATH.exists():
            print("Loading RAG index from cache...")
            self.index = faiss.read_index(str(INDEX_PATH))
            with open(DOCS_PATH, "rb") as f:
                self.docs = pickle.load(f)
        else:
            self.index, self.docs, _ = _build_index()

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def query(self, text: str, top_k: int = 3) -> list[str]:
        """
        Return the top_k most relevant Q&A pairs for the given query.

        Args:
            text:  natural language clinical query
            top_k: number of results to return

        Returns:
            list of Q&A strings ranked by relevance
        """
        vec = self.model.encode([text], normalize_embeddings=True)
        vec = np.array(vec, dtype="float32")
        _, indices = self.index.search(vec, top_k)
        return [self.docs[i] for i in indices[0] if i < len(self.docs)]


# Singleton
_retriever: MedicalRetriever | None = None


def get_retriever() -> MedicalRetriever:
    global _retriever
    if _retriever is None:
        _retriever = MedicalRetriever()
    return _retriever
