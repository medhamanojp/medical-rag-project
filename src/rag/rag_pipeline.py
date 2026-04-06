"""
RAG pipeline — retrieves relevant medical knowledge to ground agent responses.

Knowledge bases (combined):
  1. MedQuAD  — ~16,000 NIH/NLM medical Q&A pairs
  2. PubMedQA — ~1,000 research-backed clinical Q&A from PubMed abstracts

Embedding model:
  neuml/pubmedbert-base-embeddings
    - Based on PubMedBERT (Microsoft), pretrained on 14M+ PubMed abstracts
    - Fine-tuned for semantic similarity on medical text
    - Understands clinical synonyms: "dyspnea" = "shortness of breath", etc.
    - Far more accurate than general models (e.g. all-MiniLM-L6-v2) for
      medical terminology retrieval

Vector store:
  FAISS IndexFlatIP (inner product = cosine similarity on normalised vectors)

Cache:
  .rag_cache/ — index built once on first run, reused forever after
  Rebuild by deleting the .rag_cache/ folder and restarting.
"""

import pickle
import numpy as np
from pathlib import Path

CACHE_DIR  = Path(".rag_cache")
INDEX_PATH = CACHE_DIR / "faiss.index"
DOCS_PATH  = CACHE_DIR / "docs.pkl"

EMBED_MODEL = "neuml/pubmedbert-base-embeddings"


# ---------------------------------------------------------------------------
# Document loaders
# ---------------------------------------------------------------------------

def _load_medquad() -> list[str]:
    """
    Load MedQuAD — ~16,000 medical Q&A pairs from NIH/NLM.
    Returns a list of "Q: ...\nA: ..." strings.
    """
    from datasets import load_dataset

    print("  Loading MedQuAD...")
    ds  = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    df  = ds.to_pandas()
    df.columns = [c.strip().lower() for c in df.columns]

    q_col = _find_col(df, ["question", "q", "query"])
    a_col = _find_col(df, ["answer", "a", "response"])
    df    = df[[q_col, a_col]].dropna()

    docs = [
        f"Q: {row[q_col]}\nA: {row[a_col]}"
        for _, row in df.iterrows()
    ]
    print(f"  MedQuAD: {len(docs):,} Q&A pairs loaded.")
    return docs


def _load_pubmedqa() -> list[str]:
    """
    Load PubMedQA — research-backed clinical Q&A derived from PubMed abstracts.
    Uses the 'pqa_labeled' subset (~1,000 expert-labelled entries).

    Each entry has:
      question    — clinical research question
      context     — dict with 'contexts' (list of abstract sentences)
      long_answer — detailed answer synthesised from the abstract

    Returns a list of "Q: ...\nContext: ...\nA: ..." strings.
    """
    from datasets import load_dataset

    print("  Loading PubMedQA (pqa_labeled)...")
    ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    df = ds.to_pandas()

    docs = []
    for _, row in df.iterrows():
        question = row.get("question", "")
        answer   = row.get("long_answer", "")

        # Context is a dict with key 'contexts' (list of sentences)
        ctx_field = row.get("context", {})
        if isinstance(ctx_field, dict):
            sentences = ctx_field.get("contexts", [])
            context   = " ".join(sentences[:3])   # first 3 sentences
        else:
            context = ""

        if question and answer:
            docs.append(
                f"Q: {question}\n"
                f"Context: {context}\n"
                f"A: {answer}"
            )

    print(f"  PubMedQA: {len(docs):,} Q&A pairs loaded.")
    return docs


def _find_col(df, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in {df.columns.tolist()}")


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def _build_index():
    """
    Load both knowledge bases, embed with PubMedBERT, build FAISS index,
    and cache everything to .rag_cache/.
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    print("Building RAG index...")
    CACHE_DIR.mkdir(exist_ok=True)

    # Load and combine both knowledge bases
    medquad_docs  = _load_medquad()
    pubmedqa_docs = _load_pubmedqa()
    all_docs      = medquad_docs + pubmedqa_docs

    print(f"  Total documents: {len(all_docs):,} "
          f"(MedQuAD: {len(medquad_docs):,} + PubMedQA: {len(pubmedqa_docs):,})")

    # Embed with PubMedBERT
    print(f"  Embedding with {EMBED_MODEL}...")
    print("  (First run downloads the model ~400MB — subsequent runs use cache)")
    model      = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        all_docs,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,   # required for cosine similarity via inner product
    )
    embeddings = np.array(embeddings, dtype="float32")

    # Build FAISS index
    print("  Building FAISS index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # Save to cache
    faiss.write_index(index, str(INDEX_PATH))
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(all_docs, f)

    print(f"  RAG index cached to {CACHE_DIR}/")
    print(f"  Index size: {index.ntotal:,} vectors, dimension {embeddings.shape[1]}")
    return index, all_docs, model


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class MedicalRetriever:
    """
    Loads FAISS index from cache (or builds it on first run).
    Provides semantic similarity search over MedQuAD + PubMedQA.

    Uses PubMedBERT embeddings — understands clinical terminology
    and medical synonyms much better than general-purpose models.
    """

    def __init__(self):
        import faiss
        from sentence_transformers import SentenceTransformer

        if INDEX_PATH.exists() and DOCS_PATH.exists():
            print("Loading RAG index from cache...")
            self.index = faiss.read_index(str(INDEX_PATH))
            with open(DOCS_PATH, "rb") as f:
                self.docs = pickle.load(f)
            print(f"  {self.index.ntotal:,} vectors loaded.")
        else:
            self.index, self.docs, _ = _build_index()

        print(f"  Embedding model: {EMBED_MODEL}")
        self.model = SentenceTransformer(EMBED_MODEL)

    def query(self, text: str, top_k: int = 3) -> list[str]:
        """
        Return the top_k most relevant Q&A pairs for the given query.

        Uses PubMedBERT to embed the query — clinical terms are understood
        semantically, not just as keyword matches.

        Args:
            text:  clinical question or topic
            top_k: number of results (default 3)

        Returns:
            list of Q&A strings ranked by relevance
        """
        vec = self.model.encode(
            [text],
            normalize_embeddings=True,
        )
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
