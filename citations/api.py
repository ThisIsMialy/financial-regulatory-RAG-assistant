from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from rag_engine import HybridRAGEngine

app = FastAPI(title="FinPolicy Hybrid RAG API", version="1.0.0")
engine = HybridRAGEngine()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="User compliance question")
    top_k: int = Field(4, ge=1, le=10)


class QueryResponse(BaseModel):
    answer: str
    best_similarity: float
    sources: List[Dict[str, Any]]


@app.get("/health")
def health_check():
    return {"status": "ok", "indexed_chunks": len(engine.docs)}


@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    try:
        result = engine.answer(query=req.query, top_k=req.top_k)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))