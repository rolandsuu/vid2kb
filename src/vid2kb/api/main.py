from __future__ import annotations

from fastapi import FastAPI

from vid2kb.api.query import router as query_router
from vid2kb.api.runs import router as runs_router

app = FastAPI(title='vid2kb', version='0.1.0')
app.include_router(runs_router)
app.include_router(query_router)


@app.get('/health')
def health() -> dict:
    return {'status': 'ok'}
