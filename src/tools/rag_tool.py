"""
MedicalRAGTool — CrewAI tool that retrieves relevant medical Q&A
from the MedQuAD knowledge base. Used by agents to ground their
responses in real medical literature rather than pure LLM reasoning.
"""

import json
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from src.rag.rag_pipeline import get_retriever


class RAGInput(BaseModel):
    query: str = Field(
        ...,
        description="A clinical question or topic to look up, e.g. "
                    "'what are the causes of migraine' or 'flu treatment'."
    )
    top_k: int = Field(
        default=3,
        description="Number of relevant results to return (default 3)."
    )


class MedicalRAGTool(BaseTool):
    name: str = "MedicalRAGTool"
    description: str = (
        "Searches the MedQuAD medical knowledge base (NIH/NLM) for relevant "
        "Q&A pairs related to a clinical query. Use this to ground your "
        "responses in real medical evidence rather than relying solely on "
        "your own knowledge."
    )
    args_schema: type[BaseModel] = RAGInput

    def _run(self, query: str, top_k: int = 3) -> str:
        retriever = get_retriever()
        results   = retriever.query(query, top_k=top_k)
        return json.dumps({"results": results}, indent=2)
