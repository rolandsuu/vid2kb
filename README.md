# vid2kb — Video Knowledge Agent

An agent-native pipeline that takes a natural-language prompt and a video, then autonomously turns the video into a structured document and ingests it into a queryable knowledge base (PostgreSQL + pgvector).

Give the agent one prompt — e.g. *"turn this video into a LangChain tutorial and file it under LangChain"* — and it decides its own plan: download → transcribe → understand visuals → write a schema-constrained document → render Markdown/PDF → embed into the KB → report. The LLM plans the steps; the code doesn't hardcode the order.

## Architecture

The pipeline is a [LangGraph](https://langchain-ai.github.io/langgraph/) agent. Every stage is a typed tool the planner node calls:

```
user prompt + video
      │
      ▼
┌────────────┐   chooses next tool via LLM JSON   ┌──────────────┐
│  planner   │ ─────────────────────────────────► │  ingest      │
└────────────┘                                    │  transcribe  │
      ▲                                           │  visual      │
      └──────────── loop (max 12) ────────────────│  compose     │
                                                 │  render      │
                                                 │  ingest_kb   │
                                                 │  report      │
                                                 └──────────────┘
```

The planner can skip stages (e.g. no visual analysis for a podcast), retry failed stages with a fallback engine, and stops early when the input is insufficient. Runs are durable via a SQLite checkpointer (LangGraph `SqliteSaver`) and each stage writes an audit trace (`audit.jsonl`) with duration and token counts.

## Tech stack

- Python 3.13, uv, FastAPI
- LangGraph (agent orchestration) + LangChain OpenAI-compatible clients
- LlamaIndex (chunking, embedding, vector store)
- PostgreSQL + pgvector (primary KB store) — qdrant as dev fallback
- FunASR SenseVoiceSmall (Chinese ASR), faster-whisper fallback
- Qwen-VL-Max (DashScope) for visual timeline understanding
- DeepSeek-V3 for document composition
- qwen3-embedding:0.6b via ollama (1024-dim)
- FFmpeg, weasyprint (PDF render)

## Status

| Phase | Scope | Status |
|-------|-------|--------|
| 0 | Scaffold | done |
| 1 | Media pipeline + ASR | done |
| 2 | Schemas + Qwen-VL vision | done |
| 3 | Docgen (plan/compose/validate/render) | done |
| 4 | Knowledge base (pgvector) | done |
| 5 | Agent graph (planner + tools + checkpointer) | done |
| 6 | FastAPI + end-to-end | done — real E2E run: all 7 stages ok, Markdown + PDF produced, KB retrieval hit (score 0.73) |
| 7 | Eval harness (per-stage success + cost) | in progress |
| 8 | Temporal, MinIO, CosyVoice edit, MCP server | planned |

46 tests.

## Quick start

```bash
uv sync --extra dev
cp .env.example .env   # fill DEEPSEEK_API_KEY, DASHSCOPE_API_KEY
uv run uvicorn vid2kb.api.main:app --port 8000
```

Prereqs: PostgreSQL 17 + pgvector (db `vid2kb`), ollama with `qwen3-embedding:0.6b`, ffmpeg.

```bash
# submit a run (URL source)
curl -X POST http://localhost:8000/runs -H 'Content-Type: application/json' \
  -d '{"prompt": "把视频整理成一篇 LangChain 入门教程文档，并入库", "source_url": "https://..."}'

# poll status
curl http://localhost:8000/runs/<run_id>

# query the knowledge base
curl -X POST http://localhost:8000/query -d '{"question": "视频里讲了什么是 Agent?"}'
```

## Tests

```bash
uv run pytest -v
```

## Notes / known quirks

- Qwen-VL-Max ignores strict JSON schema; frame fields are coerced from strings.
- FunASR transcripts contain `<|zh|><|NEUTRAL|>` special tokens — stripped in the ASR engine.
- files.pythonhosted.org is unreachable from mainland China; use a PyPI mirror (`UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/`).

## License

MIT
