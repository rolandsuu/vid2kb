from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from vid2kb.kb.query import query_knowledge

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@router.post('/query')
def query(body: QueryRequest) -> dict:
    return {'results': query_knowledge(body.question, body.top_k)}
