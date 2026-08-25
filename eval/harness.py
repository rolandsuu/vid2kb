from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from uuid import uuid4

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompts import PROBE_QUESTIONS  # noqa: E402

from vid2kb.agent.checkpointer import build_checkpointer  # noqa: E402
from vid2kb.agent.graph import build_graph  # noqa: E402
from vid2kb.config import settings  # noqa: E402
from vid2kb.kb.query import query_knowledge  # noqa: E402
from vid2kb.schemas import KnowledgeDocument  # noqa: E402

# Approximate public prices as of 2025, in CNY per 1M tokens. These are rough
# list prices used only to estimate eval spend; actual billing may differ.
PRICING = {
    'deepseek': {'input_per_m': 2.0, 'output_per_m': 8.0},
    'qwen_vl_max': {'input_per_m': 20.0, 'output_per_m': 20.0},
}

# Map audit-record vendors (see vid2kb.llm.record_usage) to pricing keys.
_VENDOR_PRICING = {
    'deepseek': 'deepseek',
    'dashscope': 'qwen_vl_max',
}

STAGES = ['ingest', 'transcribe', 'visual', 'compose', 'render', 'ingest_kb', 'report']

_DEFAULT_PROMPT = '请总结这个视频内容并生成知识文档。'


def _pg_conn() -> psycopg.Connection:
    return psycopg.connect(settings.pgvector_database_url.replace('+psycopg', ''))


def check_services() -> list[str]:
    """Return a list of blocker messages for unavailable required services."""
    blockers: list[str] = []
    try:
        with _pg_conn() as conn:
            conn.execute('SELECT 1')
    except Exception as e:
        blockers.append(f'pgvector unavailable: {e}')
    try:
        with urllib.request.urlopen(settings.ollama_base_url + '/api/tags', timeout=3):
            pass
    except Exception as e:
        blockers.append(f'ollama embedding unavailable: {e}')
    return blockers


def _parse_audit(path: Path) -> tuple[dict[str, bool], dict[str, dict], dict[str, float]]:
    """Parse audit.jsonl into per-stage success, token usage and duration.

    The final record for a stage is authoritative: ``status == 'done'`` means
    the stage succeeded. ``tokens`` (when present) is a vendor-keyed dict of
    input/output token counts written by the tools after each LLM call.
    """
    stage_ok: dict[str, bool] = {}
    tokens: dict[str, dict] = {}
    durations: dict[str, float] = {}
    if not path.exists():
        return stage_ok, tokens, durations

    last: dict[str, dict] = {}
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            last[record.get('stage')] = record

    for stage in STAGES:
        record = last.get(stage)
        if record is None:
            stage_ok[stage] = False
            continue
        stage_ok[stage] = record.get('status') == 'done'
        durations[stage] = float(record.get('seconds') or 0.0)
        if record.get('tokens'):
            tokens[stage] = record['tokens']

    return stage_ok, tokens, durations


def _stage_cost(tokens: dict) -> float | None:
    """Compute CNY cost for a stage from its vendor-keyed token counts."""
    if not tokens:
        return None
    total = 0.0
    found = False
    for vendor, counts in tokens.items():
        key = _VENDOR_PRICING.get(vendor)
        if key is None:
            continue
        price = PRICING[key]
        inp = counts.get('input') or 0
        out = counts.get('output') or 0
        total += inp / 1_000_000 * price['input_per_m']
        total += out / 1_000_000 * price['output_per_m']
        found = True
    return total if found else None


def _count_kb_nodes(run_id: str) -> int | None:
    try:
        with _pg_conn() as conn:
            cur = conn.execute(
                "SELECT count(*) FROM data_vid2kb_nodes WHERE metadata_->>'run_id' = %s",
                (run_id,),
            )
            return int(cur.fetchone()[0])
    except Exception:
        return None


def _retrieval_probe() -> tuple[int, int, float]:
    hits = 0
    total = len(PROBE_QUESTIONS)
    for probe in PROBE_QUESTIONS:
        try:
            results = query_knowledge(probe['question'], top_k=3)
        except Exception:
            continue
        text = '\n'.join(r.get('text', '') for r in results)
        lowered = text.lower()
        for kw in probe['keywords']:
            if kw and kw.lower() in lowered:
                hits += 1
                break
    rate = hits / total if total else 0.0
    return hits, total, rate


def run_one(sample: Path) -> dict:
    run_id = uuid4().hex[:12]
    graph = build_graph(checkpointer=build_checkpointer())
    initial = {
        'run_id': run_id,
        'user_prompt': _DEFAULT_PROMPT,
        'source': str(sample),
        'errors': [],
        'steps': [],
        'iterations': 0,
    }
    try:
        result = graph.invoke(initial, config={'configurable': {'thread_id': run_id}})
    except Exception as e:
        result = {'error': str(e)}

    audit_path = Path(settings.data_dir) / 'runs' / run_id / 'audit.jsonl'
    stage_ok, tokens, durations = _parse_audit(audit_path)

    doc_valid = False
    if result.get('document'):
        try:
            KnowledgeDocument.model_validate(result['document'])
            doc_valid = True
        except Exception:
            doc_valid = False

    out_dir = Path(settings.data_dir) / 'runs' / run_id / 'out'
    render_ok = (out_dir / 'document.md').exists() and (out_dir / 'document.pdf').exists()

    kb_nodes = _count_kb_nodes(run_id)

    hits, total, rate = _retrieval_probe()

    stage_cost: dict[str, float | None] = {s: _stage_cost(tokens.get(s, {})) for s in STAGES}
    cost_total = sum(c for c in stage_cost.values() if c is not None)

    return {
        'video': str(sample),
        'run_id': run_id,
        'stage_ok': stage_ok,
        'durations': durations,
        'tokens': tokens,
        'stage_cost': stage_cost,
        'cost_total': cost_total,
        'doc_valid': doc_valid,
        'render_ok': render_ok,
        'kb_nodes': kb_nodes,
        'retrieval_hits': hits,
        'retrieval_total': total,
        'retrieval_rate': rate,
        'error': result.get('error'),
    }


def _fmt_ok(ok: bool) -> str:
    return 'ok' if ok else 'FAIL'


def _fmt_cost(cost: float | None) -> str:
    return 'n/a' if cost is None else f'{cost:.4f}'


def _print_table(rows: list[dict]) -> None:
    header = ['video', *STAGES, 'cost', 'retrieval']
    widths = {'video': 34}
    for s in STAGES:
        widths[s] = max(len(s), 4)
    widths['cost'] = 10
    widths['retrieval'] = 12

    def _pad(text: str, width: int) -> str:
        # CJK glyphs render double-width; pad by visual width.
        vis = sum(2 if ord(c) > 0x2E80 else 1 for c in text)
        return text + ' ' * max(0, width - vis)

    print(_pad(header[0], widths['video']), end='')
    for s in STAGES:
        print(' ' + _pad(s, widths[s]), end='')
    print(' ' + _pad('cost', widths['cost']) + ' ' + _pad('retrieval', widths['retrieval']))

    for row in rows:
        print(_pad(row['video'], widths['video']), end='')
        for s in STAGES:
            print(' ' + _pad(_fmt_ok(row['stage_ok'][s]), widths[s]), end='')
        print(' ' + _pad(_fmt_cost(row['cost_total']), widths['cost']), end='')
        print(
            ' '
            + _pad(
                f"{row['retrieval_hits']}/{row['retrieval_total']} ({row['retrieval_rate']:.2f})",
                widths['retrieval'],
            )
        )

    print()
    for row in rows:
        print(f"run {row['run_id']}: {row['video']}")
        if row.get('error'):
            print(f"  error: {row['error']}")
        for s in STAGES:
            dur = row['durations'].get(s)
            cost = row['stage_cost'].get(s)
            dur_s = f'{dur:.1f}s' if dur is not None else '-'
            print(
                f"  {s:10s} {_fmt_ok(row['stage_ok'][s]):4s} "
                f"{dur_s:8s} cost={_fmt_cost(cost)} CNY"
            )
        print(
            f"  doc_valid={row['doc_valid']} render_ok={row['render_ok']} "
            f"kb_nodes={row['kb_nodes']}"
        )
        print()


def _gate_and_summary(rows: list[dict]) -> int:
    n = len(rows)
    avg_stage = sum(sum(r['stage_ok'].values()) / len(STAGES) for r in rows) / n * 100
    avg_retrieval = sum(r['retrieval_rate'] for r in rows) / n
    total_cost = sum(r['cost_total'] for r in rows)

    print('summary:')
    print(f'  avg stage success: {avg_stage:.1f}%')
    print(f'  avg retrieval hit: {avg_retrieval:.2f}')
    print(f'  total cost: {total_cost:.4f} CNY')

    if avg_stage >= 80.0 and avg_retrieval >= 0.5:
        print('PASS')
        return 0
    print('FAIL (gate: >=80% stage success and >=0.5 avg retrieval hit)')
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='vid2kb eval harness (real LLM end-to-end)')
    parser.add_argument(
        '--samples',
        nargs='*',
        help='local video files to evaluate (default: eval/samples/*.mp4)',
    )
    args = parser.parse_args(argv)

    if args.samples:
        samples = [Path(s) for s in args.samples]
    else:
        samples = sorted((Path(__file__).resolve().parent / 'samples').glob('*.mp4'))
    samples = [s for s in samples if s.is_file()]

    if not samples:
        print('no sample videos found; pass --samples or run eval/samples/download.sh')
        return 1

    blockers = check_services()
    if blockers:
        for message in blockers:
            print(f'BLOCKER: {message}')
        return 1

    print(f'eval: {len(samples)} sample(s)')
    rows = [run_one(sample) for sample in samples]
    _print_table(rows)
    return _gate_and_summary(rows)


if __name__ == '__main__':
    sys.exit(main())
