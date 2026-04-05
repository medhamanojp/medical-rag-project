"""
CrewAI tool: MedicalRAGTool
Retrieves relevant medical Q&A passages from the MedQuAD knowledge base
(loaded via HuggingFace) using semantic similarity search (FAISS).

The DiagnosticianAgent and ExplainerAgent use this tool to ground their
answers in real medical literature rather than relying solely on the LLM.
"""

import json
import logging

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src.rag.rag_pipeline import get_rag_pipeline

logger = logging.getLogger(__name__)


class RAGToolInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "A clinical query string, e.g. "
            "'pneumonia symptoms and treatment in elderly patients' "
            "or 'COVID-19 loss of taste diagnostic criteria'."
        ),
    )
    k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of evidence passages to retrieve (default 3).",
    )


class MedicalRAGTool(BaseTool):
    name: str = "MedicalRAGTool"
    description: str = (
        "Retrieves relevant medical evidence passages from the MedQuAD "
        "knowledge base (real NIH/NLM medical Q&A data). "
        "Use this tool to find established clinical information about "
        "diseases, symptoms, diagnostics, or treatments to support and "
        "validate the diagnostic reasoning. "
        "Input: a natural language medical query string."
    )
    args_schema: type[BaseModel] = RAGToolInput

    def _run(self, query: str, k: int = 3) -> str:
        logger.info("MedicalRAGTool query: %s (k=%d)", query, k)
        try:
            rag = get_rag_pipeline()
            docs = rag.retrieve(query, k=k)
        except Exception as e:
            logger.exception("RAG retrieval failed")
            return json.dumps({"error": "RAG_FAILED", "detail": str(e)})

        if not docs:
            return json.dumps({"message": "No relevant documents found.", "results": []})

        results = []
        for i, doc in enumerate(docs, 1):
            results.append({
                "rank": i,
                "question": doc["question"],
                "answer": doc["answer"][:600],   # truncate for LLM context
                "source": doc.get("source", "MedQuAD"),
            })

        return json.dumps({"query": query, "results": results}, indent=2)
